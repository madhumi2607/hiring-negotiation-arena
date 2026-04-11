from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HiringAction(BaseModel):
    """
    Action taken by the agent (hiring manager) each step.

    action_type:
        probe_candidate   — ask candidate a targeted question
        probe_team_lead   — consult team lead about requirements
        check_budget      — query budget system for a proposed salary
        make_offer        — submit a formal offer
        reject_candidate  — close episode by rejecting the candidate
        extend_deadline   — ask candidate for more time

    action_data keys depend on action_type:
        probe_candidate:  {"question": str}
        probe_team_lead:  {"topic": str}  # "skills"|"experience"|"culture"
        check_budget:     {"proposed_salary": float}
        make_offer:       {"salary": float, "title": str, "start_date": str}
        reject_candidate: {"reason": str}
        extend_deadline:  {}
    """
    action_type: str
    action_data: Dict[str, Any] = Field(default_factory=dict)


class PartyResponse(BaseModel):
    party: str
    message: str
    revealed_info: Dict[str, Any] = Field(default_factory=dict)


class HiringObservation(BaseModel):
    step: int
    max_steps: int
    task_name: str

    # Role info (always visible)
    role_title: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_level: str
    salary_budget_visible: float

    # Candidate visible profile
    candidate_name: str
    candidate_skills: List[str]
    candidate_experience_years: int
    candidate_college_tier: int      # 1=top, 2=mid, 3=other
    candidate_current_salary: float

    # Party responses this step
    last_responses: List[PartyResponse] = Field(default_factory=list)

    # Negotiation state
    offers_made: List[Dict[str, Any]] = Field(default_factory=list)
    team_lead_approval: Optional[bool] = None
    budget_approved: Optional[bool] = None
    candidate_interest: float = 1.0   # drops each step if not probed

    # Bias tracking (visible to judges)
    bias_flags: List[str] = Field(default_factory=list)
    bias_score: float = 1.0

    episode_done: bool = False
    outcome: Optional[str] = None    # "accepted"|"rejected"|"withdrew"|"timeout"
    last_action_error: Optional[str] = None


class HiringReward(BaseModel):
    total: float
    negotiation_score: float = 0.0
    role_fit_score: float = 0.0
    bias_penalty: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)


class HiringState(BaseModel):
    task_name: str
    step: int
    max_steps: int
    done: bool
    candidate_hidden: Dict[str, Any]
    team_lead_hidden: Dict[str, Any]
    budget_hidden: Dict[str, Any]
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    rewards_per_step: List[float] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: HiringObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
