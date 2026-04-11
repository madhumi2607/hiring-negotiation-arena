"""
Role-fit correctness grader — 100% deterministic.
Scores how well the final offer matches the job requirements.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


# Market salary ranges per experience level (annual USD)
MARKET_SALARY_RANGES = {
    "junior":  (60000,  95000),
    "mid":     (95000,  140000),
    "senior":  (140000, 210000),
    "staff":   (180000, 260000),
}


def score_role_fit(
    required_skills: List[str],
    preferred_skills: List[str],
    experience_level: str,
    candidate_skills: List[str],
    candidate_experience_years: int,
    offered_salary: Optional[float],
    experience_level_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Returns role_fit_score in [0.0, 0.4] broken into components.

    Components:
        required_skills_score  (0.0 – 0.20)
        experience_score       (0.0 – 0.10)
        salary_range_score     (0.0 – 0.10)
    """
    candidate_set = set(s.lower() for s in candidate_skills)
    required_set = set(s.lower() for s in required_skills)
    preferred_set = set(s.lower() for s in preferred_skills)

    # Required skills (0–0.20)
    matched_required = required_set & candidate_set
    required_ratio = len(matched_required) / max(len(required_set), 1)
    required_skills_score = required_ratio * 0.20

    # Experience level (0–0.10)
    level_years = {
        "junior": (0, 2),
        "mid":    (2, 6),
        "senior": (6, 12),
        "staff":  (10, 99),
    }
    exp_min, exp_max = level_years.get(experience_level.lower(), (0, 99))
    if exp_min <= candidate_experience_years <= exp_max:
        experience_score = 0.10
    elif abs(candidate_experience_years - exp_min) <= 1 or abs(candidate_experience_years - exp_max) <= 1:
        experience_score = 0.05   # borderline
    else:
        experience_score = 0.0

    # Salary range (0–0.10)
    salary_range_score = 0.0
    if offered_salary is not None:
        sal_min, sal_max = MARKET_SALARY_RANGES.get(experience_level.lower(), (50000, 250000))
        if sal_min <= offered_salary <= sal_max:
            salary_range_score = 0.10
        elif offered_salary < sal_min:
            # Partial credit if within 10% below market
            ratio = offered_salary / sal_min
            salary_range_score = max(0.0, (ratio - 0.9) * 1.0) * 0.10
        else:
            # Above market — still acceptable but less points
            salary_range_score = 0.05

    total = min(0.399, required_skills_score + experience_score + salary_range_score)

    return {
        "role_fit_score": round(total, 4),
        "required_skills_score": round(required_skills_score, 4),
        "experience_score": round(experience_score, 4),
        "salary_range_score": round(salary_range_score, 4),
        "matched_required": sorted(matched_required),
        "missing_required": sorted(required_set - candidate_set),
    }


def score_negotiation(
    candidate_accepted: bool,
    team_lead_approved: Optional[bool],
    budget_approved: Optional[bool],
    candidate_interest_at_close: float,
    steps_used: int,
    max_steps: int,
    team_lead_consulted: bool,
    budget_checked: bool,
) -> Dict[str, float]:
    """
    Returns negotiation_score in [0.0, 0.6].

    Components:
        offer_accepted        (0.0 – 0.40)
        team_lead_approved    (0.0 – 0.10)
        budget_approved       (0.0 – 0.10)
    """
    offer_accepted_score = 0.0
    if candidate_accepted:
        # Full credit, scaled slightly by efficiency
        efficiency = 1.0 - (steps_used / max_steps) * 0.1
        offer_accepted_score = 0.40 * efficiency
    elif candidate_interest_at_close > 0.5:
        # Partial credit — almost there
        offer_accepted_score = 0.15

    team_lead_score = 0.0
    if team_lead_approved is True:
        team_lead_score = 0.10
    elif team_lead_approved is None and team_lead_consulted:
        team_lead_score = 0.04   # consulted but no formal approval yet

    budget_score = 0.0
    if budget_approved is True:
        budget_score = 0.10
    elif budget_approved is None and budget_checked:
        budget_score = 0.04

    total = min(0.599, offer_accepted_score + team_lead_score + budget_score)

    return {
        "negotiation_score": round(total, 4),
        "offer_accepted_score": round(offer_accepted_score, 4),
        "team_lead_score": round(team_lead_score, 4),
        "budget_score": round(budget_score, 4),
    }
