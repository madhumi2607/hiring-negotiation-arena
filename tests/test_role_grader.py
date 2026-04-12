import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


def test_role_grader_import():
    from role_grader import RoleGrader
    assert RoleGrader is not None


def test_perfect_candidate_scores_high():
    from role_grader import RoleGrader
    grader = RoleGrader()
    score = grader.score(
        required_skills=["Python", "FastAPI"],
        candidate_skills=["Python", "FastAPI", "Docker"],
        experience_level="mid",
        experience_years=4,
        salary_offered=110000,
        salary_budget=120000,
        market_min=95000,
        market_max=140000,
    )
    assert score >= 0.7


def test_missing_skills_lower_score():
    from role_grader import RoleGrader
    grader = RoleGrader()
    score_full = grader.score(
        required_skills=["Python", "FastAPI"],
        candidate_skills=["Python", "FastAPI"],
        experience_level="mid", experience_years=3,
        salary_offered=100000, salary_budget=120000,
        market_min=95000, market_max=140000,
    )
    score_missing = grader.score(
        required_skills=["Python", "FastAPI"],
        candidate_skills=["Python"],
        experience_level="mid", experience_years=3,
        salary_offered=100000, salary_budget=120000,
        market_min=95000, market_max=140000,
    )
    assert score_full > score_missing


def test_score_in_range():
    from role_grader import RoleGrader
    grader = RoleGrader()
    score = grader.score(
        required_skills=["Python"],
        candidate_skills=["Python"],
        experience_level="mid", experience_years=3,
        salary_offered=100000, salary_budget=120000,
        market_min=95000, market_max=140000,
    )
    assert 0.0 <= score <= 1.0
