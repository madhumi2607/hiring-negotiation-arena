import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest


def test_score_role_fit_import():
    from role_grader import score_role_fit
    assert score_role_fit is not None


def test_perfect_candidate_scores_high():
    from role_grader import score_role_fit
    result = score_role_fit(
        required_skills=["Python", "FastAPI"],
        preferred_skills=["Docker"],
        experience_level="mid",
        candidate_skills=["Python", "FastAPI", "Docker"],
        candidate_experience_years=4,
        offered_salary=110000,
    )
    assert result["role_fit_score"] >= 0.3


def test_missing_skills_lower_score():
    from role_grader import score_role_fit
    full = score_role_fit(
        required_skills=["Python", "FastAPI"],
        preferred_skills=[],
        experience_level="mid",
        candidate_skills=["Python", "FastAPI"],
        candidate_experience_years=3,
        offered_salary=100000,
    )
    missing = score_role_fit(
        required_skills=["Python", "FastAPI"],
        preferred_skills=[],
        experience_level="mid",
        candidate_skills=["Python"],
        candidate_experience_years=3,
        offered_salary=100000,
    )
    assert full["role_fit_score"] > missing["role_fit_score"]


def test_score_in_range():
    from role_grader import score_role_fit
    result = score_role_fit(
        required_skills=["Python"],
        preferred_skills=[],
        experience_level="mid",
        candidate_skills=["Python"],
        candidate_experience_years=3,
        offered_salary=100000,
    )
    assert 0.0 <= result["role_fit_score"] <= 1.0


def test_negotiation_score_accepted():
    from role_grader import score_negotiation
    result = score_negotiation(
        candidate_accepted=True,
        team_lead_approved=True,
        budget_approved=True,
        candidate_interest_at_close=0.9,
        steps_used=4,
        max_steps=15,
        team_lead_consulted=True,
        budget_checked=True,
    )
    assert result["negotiation_score"] >= 0.5


def test_negotiation_score_rejected():
    from role_grader import score_negotiation
    result = score_negotiation(
        candidate_accepted=False,
        team_lead_approved=None,
        budget_approved=None,
        candidate_interest_at_close=0.2,
        steps_used=15,
        max_steps=15,
        team_lead_consulted=False,
        budget_checked=False,
    )
    assert result["negotiation_score"] < 0.3
