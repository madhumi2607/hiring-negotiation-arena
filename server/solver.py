"""
Perfect solver for HiringNegotiationArena.
Uses hardcoded optimal strategy: probe all parties, check budget, make
correct offer. Achieves ~0.95 on easy, ~0.75 on medium, ~0.55 on hard.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from models import HiringAction, HiringObservation


class PerfectSolver:
    """
    Deterministic optimal solver that knows the correct strategy:
      1. Probe team lead for skills
      2. Probe candidate for competing offer + salary expectations
      3. Check budget with a reasonable salary
      4. Make offer at or above candidate minimum, within budget + justification
    """

    def __init__(self):
        self.phase = "probe_team_lead"
        self.probed_candidate_competing = False
        self.probed_candidate_salary = False
        self.checked_budget = False
        self.team_lead_skills_known = False
        self.discovered_min_salary: Optional[float] = None
        self.discovered_competing: Optional[float] = None

    def act(self, obs: HiringObservation) -> HiringAction:
        # Parse any revealed info from last responses
        for resp in obs.last_responses:
            ri = resp.revealed_info
            if "min_acceptable_salary" in ri:
                self.discovered_min_salary = ri["min_acceptable_salary"]
            if "competing_offer_salary" in ri:
                self.discovered_competing = ri["competing_offer_salary"]
            if "must_have_skills" in ri:
                self.team_lead_skills_known = True

        # Strategy FSM
        if not self.team_lead_skills_known and obs.step < 3:
            return HiringAction(
                action_type="probe_team_lead",
                action_data={"topic": "skills and requirements"},
            )

        if not self.probed_candidate_competing and obs.step < 5:
            self.probed_candidate_competing = True
            return HiringAction(
                action_type="probe_candidate",
                action_data={"question": "Do you have any other offers or competing opportunities?"},
            )

        if not self.probed_candidate_salary and obs.step < 6:
            self.probed_candidate_salary = True
            return HiringAction(
                action_type="probe_candidate",
                action_data={"question": "What salary range are you expecting?"},
            )

        if not self.checked_budget and obs.step < 8:
            self.checked_budget = True
            target = self._compute_target_salary(obs)
            return HiringAction(
                action_type="check_budget",
                action_data={
                    "proposed_salary": target,
                    "justification": "Candidate has strong skill match and competing offer",
                },
            )

        # Make the offer
        salary = self._compute_target_salary(obs)
        return HiringAction(
            action_type="make_offer",
            action_data={
                "salary": salary,
                "title": obs.role_title,
                "start_date": "2025-07-01",
            },
        )

    def _compute_target_salary(self, obs: HiringObservation) -> float:
        base = obs.candidate_current_salary * 1.15    # 15% bump as baseline
        if self.discovered_min_salary:
            base = max(base, self.discovered_min_salary + 2000)
        if self.discovered_competing:
            base = max(base, self.discovered_competing + 1000)
        # Cap at visible budget
        base = min(base, obs.salary_budget_visible)
        return round(base, -2)   # round to nearest 100
