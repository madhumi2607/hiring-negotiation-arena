from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HiringAction(BaseModel):
    action_type: str
    salary_offer: Optional[float] = None
    message: Optional[str] = None
    benefits: Optional[Dict[str, Any]] = None
    skill_to_verify: Optional[str] = None
    # Legacy flat dict support
    action_data: Dict[str, Any] = Field(default_factory=dict)


class PartyResponse(BaseModel):
    party: str
    message: str
    revealed_info: Dict[str, Any] = Field(default_factory=dict)


class HiringObservation(BaseModel):
    step: int
    max_steps: int
    task_name: str
    role_title: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_level: str
    salary_budget_visible: float
    candidate_name: str
    candidate_skills: List[str]
    candidate_experience_years: int
    candidate_college_tier: int
    candidate_current_salary: float
    last_responses: List[PartyResponse] = Field(default_factory=list)
    offers_made: List[Dict[str, Any]] = Field(default_factory=list)
    team_lead_approval: Optional[bool] = None
    budget_approved: Optional[bool] = None
    candidate_interest: float = 1.0
    bias_flags: List[str] = Field(default_factory=list)
    bias_score: float = 1.0
    episode_done: bool = False
    outcome: Optional[str] = None
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
