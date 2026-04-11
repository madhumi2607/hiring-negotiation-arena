"""
HiringNegotiationArena — core environment.

State machine that implements the OpenEnv step()/reset()/state() interface.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

# Dual-import pattern: relative works in-repo, bare works in Docker
try:
    from ..models import (
        HiringAction, HiringObservation, HiringReward,
        HiringState, StepResult, PartyResponse,
    )
except ImportError:
    from models import (  # type: ignore
        HiringAction, HiringObservation, HiringReward,
        HiringState, StepResult, PartyResponse,
    )

try:
    from .parties import CandidateParty, TeamLeadParty, BudgetSystem
    from .role_grader import score_role_fit, score_negotiation
    from .bias_detector import BiasDetector
    from .task_configs import TASKS
    from .stochastic import apply_stochastic
except ImportError:
    from server.parties import CandidateParty, TeamLeadParty, BudgetSystem  # type: ignore
    from server.role_grader import score_role_fit, score_negotiation  # type: ignore
    from server.bias_detector import BiasDetector  # type: ignore
    from server.task_configs import TASKS  # type: ignore
    from server.stochastic import apply_stochastic  # type: ignore


VALID_ACTIONS = {
    "probe_candidate", "probe_team_lead", "check_budget",
    "make_offer", "reject_candidate", "extend_deadline",
}


class HiringEnvironment:
    def __init__(self, task_name: str = "task1_easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
        self.task_name = task_name
        self._config = TASKS[task_name]
        self._state: Optional[_EpisodeState] = None

    def reset(self) -> HiringObservation:
        cfg = self._config
        role = cfg["role"]
        is_marathon = "candidates" in cfg

        candidate_cfg = cfg["candidates"][0] if is_marathon else cfg["candidate"]

        candidate_hidden = candidate_cfg["hidden"]
        team_lead_hidden = cfg["team_lead"]["hidden"]
        budget_hidden = cfg["budget"]["hidden"]

        # Apply stochastic noise — different every episode
        candidate_hidden, team_lead_hidden, budget_hidden = apply_stochastic(
            candidate_hidden, team_lead_hidden, budget_hidden
        )

        self._state = _EpisodeState(
            task_name=self.task_name,
            max_steps=cfg["max_steps"],
            role=role,
            candidate_cfg={**{k: v for k, v in candidate_cfg.items() if k != "hidden"}, "hidden": candidate_hidden},
            team_lead_hidden=team_lead_hidden,
            budget_hidden=budget_hidden,
            is_marathon=is_marathon,
            all_candidates=cfg.get("candidates", []),
        )
        return self._state.to_observation()

    def step(self, action: HiringAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode already done. Call reset() to start a new episode.")

        obs, reward, done = self._state.apply_action(action)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={"task": self.task_name, "step": self._state.step},
        )

    def state(self) -> HiringState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state.to_full_state()


class _EpisodeState:
    def __init__(
        self,
        task_name: str,
        max_steps: int,
        role: Dict[str, Any],
        candidate_cfg: Dict[str, Any],
        team_lead_hidden: Dict[str, Any],
        budget_hidden: Dict[str, Any],
        is_marathon: bool = False,
        all_candidates: List[Dict[str, Any]] = None,
    ):
        self.task_name = task_name
        self.max_steps = max_steps
        self.role = role
        self.step = 0
        self.done = False
        self.outcome: Optional[str] = None
        self.offers_made: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.rewards_per_step: List[float] = []
        self.cumulative_reward = 0.0
        self.last_responses: List[PartyResponse] = []
        self.last_error: Optional[str] = None

        self.is_marathon = is_marathon
        self.all_candidates = all_candidates or []
        self.marathon_candidate_idx = 0
        self.marathon_scores: List[float] = []
        self.discovered_budget_cap: Optional[float] = None
        self.discovered_must_have_skills: Optional[List[str]] = None

        self._init_candidate(candidate_cfg)
        self.team_lead = TeamLeadParty(team_lead_hidden)
        self.budget = BudgetSystem(budget_hidden)
        self.bias_detector = BiasDetector(
            candidate_college_tier=self.candidate.profile["college_tier"],
            candidate_current_salary=self.candidate.profile["current_salary"],
            experience_level=role["experience_level"],
        )

    def _init_candidate(self, candidate_cfg: Dict[str, Any]):
        profile = {k: v for k, v in candidate_cfg.items() if k != "hidden"}
        self.candidate = CandidateParty(profile, candidate_cfg["hidden"])

    def apply_action(self, action: HiringAction) -> tuple:
        self.step += 1
        self.last_responses = []
        self.last_error = None

        if action.action_type not in VALID_ACTIONS:
            self.last_error = f"Invalid action_type '{action.action_type}'. Valid: {sorted(VALID_ACTIONS)}"
            reward = -0.05
            self.rewards_per_step.append(reward)
            self.cumulative_reward += reward
            obs = self.to_observation()
            return obs, reward, self._check_done()

        self.candidate.tick()
        self.bias_detector.on_action(action.action_type, action.action_data, context={"role": self.role})

        reward = 0.0

        if action.action_type == "probe_candidate":
            result = self.candidate.respond_to_probe(action.action_data.get("question", ""))
            self.last_responses.append(
                PartyResponse(party="candidate", message=result["message"], revealed_info=result["revealed_info"])
            )
            reward = 0.05

        elif action.action_type == "probe_team_lead":
            result = self.team_lead.respond_to_probe(
                action.action_data.get("topic", ""), self.candidate.profile["skills"]
            )
            if result["revealed_info"].get("college_bias_expressed"):
                self.bias_detector.on_team_lead_bias_expressed()
            self.last_responses.append(
                PartyResponse(party="team_lead", message=result["message"], revealed_info=result["revealed_info"])
            )
            reward = 0.05

        elif action.action_type == "check_budget":
            proposed = action.action_data.get("proposed_salary", 0.0)
            justification = action.action_data.get("justification", "")
            result = self.budget.check_salary(proposed, justification)
            if result["revealed_info"].get("approved"):
                self.discovered_budget_cap = proposed
            self.last_responses.append(
                PartyResponse(party="budget", message=result["message"], revealed_info=result["revealed_info"])
            )
            reward = 0.03

        elif action.action_type == "extend_deadline":
            self.last_responses.append(
                PartyResponse(
                    party="candidate",
                    message="I can give you a little more time, but not much.",
                    revealed_info={"deadline_extended": True},
                )
            )
            self.candidate.interest = min(1.0, self.candidate.interest + 0.05)
            reward = 0.02

        elif action.action_type == "make_offer":
            reward = self._handle_offer(action.action_data)

        elif action.action_type == "reject_candidate":
            self.outcome = "rejected"
            self.done = True
            self.last_responses.append(
                PartyResponse(party="system", message="Candidate rejected. Episode ended.", revealed_info={})
            )
            reward = self._compute_final_reward()

        self.action_history.append({
            "step": self.step,
            "action_type": action.action_type,
            "action_data": action.action_data,
            "reward": reward,
        })
        self.rewards_per_step.append(reward)
        self.cumulative_reward += reward

        done = self._check_done()
        if done and self.outcome is None:
            self.outcome = "timeout"
            reward += self._compute_final_reward()
            self.rewards_per_step[-1] = reward

        obs = self.to_observation()
        return obs, reward, done

    def _handle_offer(self, data: Dict[str, Any]) -> float:
        salary = data.get("salary", 0.0)
        title = data.get("title", self.role["title"])
        start_date = data.get("start_date", "TBD")

        offer = {"salary": salary, "title": title, "start_date": start_date, "step": self.step}
        self.offers_made.append(offer)

        tl_result = self.team_lead.evaluate_candidate(
            self.candidate.profile["skills"], self.candidate.profile["college_tier"]
        )
        self.last_responses.append(
            PartyResponse(party="team_lead", message=tl_result["message"], revealed_info=tl_result["revealed_info"])
        )

        budget_result = self.budget.check_salary(salary, justification="formal offer")
        self.last_responses.append(
            PartyResponse(party="budget", message=budget_result["message"], revealed_info=budget_result["revealed_info"])
        )

        candidate_result = self.candidate.respond_to_offer(salary, title)
        self.last_responses.append(
            PartyResponse(party="candidate", message=candidate_result["message"], revealed_info=candidate_result["revealed_info"])
        )

        if candidate_result.get("accepted"):
            self.outcome = "accepted"
            self.done = True
            return self._compute_final_reward()
        elif candidate_result["revealed_info"].get("withdrew"):
            self.outcome = "withdrew"
            self.done = True

        return 0.10

    def _check_done(self) -> bool:
        if self.done:
            return True
        if self.step >= self.max_steps:
            self.done = True
            return True
        if self.candidate.interest <= 0.0:
            self.outcome = "withdrew"
            self.done = True
            return True
        return False

    def _compute_final_reward(self) -> float:
        try:
            from .role_grader import score_role_fit, score_negotiation
        except ImportError:
            from server.role_grader import score_role_fit, score_negotiation  # type: ignore

        bias_result = self.bias_detector.finalize()

        neg_scores = score_negotiation(
            candidate_accepted=(self.outcome == "accepted"),
            team_lead_approved=self.team_lead.approval_given,
            budget_approved=self.budget.approved,
            candidate_interest_at_close=self.candidate.interest,
            steps_used=self.step,
            max_steps=self.max_steps,
            team_lead_consulted=self.team_lead.consulted,
            budget_checked=self.budget.checked,
        )

        last_offer = self.offers_made[-1] if self.offers_made else None
        fit_scores = score_role_fit(
            required_skills=self.role["required_skills"],
            preferred_skills=self.role["preferred_skills"],
            experience_level=self.role["experience_level"],
            candidate_skills=self.candidate.profile["skills"],
            candidate_experience_years=self.candidate.profile["experience_years"],
            offered_salary=last_offer["salary"] if last_offer else None,
        )

        bias_penalty = 1.0 - bias_result["bias_score"]
        raw = neg_scores["negotiation_score"] + fit_scores["role_fit_score"]
        total = max(0.0, min(1.0, raw - bias_penalty * 0.3))
        return round(total, 4)

    def to_observation(self) -> HiringObservation:
        bias_result = self.bias_detector.finalize() if self.done else {
            "bias_score": self.bias_detector.bias_score,
            "bias_flags": list(self.bias_detector.flags),
        }
        return HiringObservation(
            step=self.step,
            max_steps=self.max_steps,
            task_name=self.task_name,
            role_title=self.role["title"],
            required_skills=self.role["required_skills"],
            preferred_skills=self.role["preferred_skills"],
            experience_level=self.role["experience_level"],
            salary_budget_visible=self.role["salary_budget_visible"],
            candidate_name=self.candidate.profile["name"],
            candidate_skills=self.candidate.profile["skills"],
            candidate_experience_years=self.candidate.profile["experience_years"],
            candidate_college_tier=self.candidate.profile["college_tier"],
            candidate_current_salary=self.candidate.profile["current_salary"],
            last_responses=list(self.last_responses),
            offers_made=list(self.offers_made),
            team_lead_approval=self.team_lead.approval_given,
            budget_approved=self.budget.approved,
            candidate_interest=round(self.candidate.interest, 3),
            bias_flags=bias_result["bias_flags"],
            bias_score=bias_result["bias_score"],
            episode_done=self.done,
            outcome=self.outcome,
            last_action_error=self.last_error,
        )

    def to_full_state(self) -> HiringState:
        return HiringState(
            task_name=self.task_name,
            step=self.step,
            max_steps=self.max_steps,
            done=self.done,
            candidate_hidden=self.candidate.hidden,
            team_lead_hidden=self.team_lead.hidden,
            budget_hidden=self.budget.hidden,
            action_history=list(self.action_history),
            cumulative_reward=round(self.cumulative_reward, 4),
            rewards_per_step=list(self.rewards_per_step),
        )
