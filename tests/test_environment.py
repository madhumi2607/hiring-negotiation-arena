import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from models import HiringAction, HiringObservation


def test_hiring_action_valid():
    a = HiringAction(action_type="probe_candidate", action_data={"question": "test"})
    assert a.action_type == "probe_candidate"


def test_hiring_action_defaults():
    a = HiringAction(action_type="make_offer")
    assert a.action_data == {}


def test_observation_fields():
    obs = HiringObservation(
        step=1, max_steps=10, task_name="task1_easy",
        role_title="Engineer", required_skills=["Python"],
        preferred_skills=[], experience_level="mid",
        salary_budget_visible=120000.0, candidate_name="Test",
        candidate_skills=["Python"], candidate_experience_years=3,
        candidate_college_tier=2, candidate_current_salary=90000.0,
    )
    assert obs.bias_score == 1.0
    assert obs.candidate_interest == 1.0
    assert obs.episode_done == False


def test_observation_bias_defaults():
    obs = HiringObservation(
        step=1, max_steps=10, task_name="task1_easy",
        role_title="Engineer", required_skills=["Python"],
        preferred_skills=[], experience_level="mid",
        salary_budget_visible=120000.0, candidate_name="Test",
        candidate_skills=["Python"], candidate_experience_years=3,
        candidate_college_tier=2, candidate_current_salary=90000.0,
    )
    assert obs.bias_flags == []
    assert obs.offers_made == []
