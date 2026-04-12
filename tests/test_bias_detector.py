import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

import pytest


def test_bias_detector_import():
    from bias_detector import BiasDetector
    assert BiasDetector is not None


def test_no_bias_full_score():
    from bias_detector import BiasDetector
    bd = BiasDetector()
    score = bd.compute_bias_score(
        action_history=[
            {"action_type": "probe_team_lead", "action_data": {"topic": "skills"}},
            {"action_type": "probe_candidate", "action_data": {"question": "salary?"}},
            {"action_type": "make_offer", "action_data": {"salary": 110000}},
        ],
        candidate_college_tier=3,
        market_min=95000,
    )
    assert score >= 0.8


def test_college_bias_penalised():
    from bias_detector import BiasDetector
    bd = BiasDetector()
    score = bd.compute_bias_score(
        action_history=[
            {"action_type": "reject_candidate",
             "action_data": {"reason": "candidate did not attend top-tier university"}},
        ],
        candidate_college_tier=3,
        market_min=95000,
    )
    assert score < 0.8


def test_below_market_penalised():
    from bias_detector import BiasDetector
    bd = BiasDetector()
    score = bd.compute_bias_score(
        action_history=[
            {"action_type": "make_offer", "action_data": {"salary": 50000}},
        ],
        candidate_college_tier=2,
        market_min=95000,
    )
    assert score < 1.0


def test_score_bounded():
    from bias_detector import BiasDetector
    bd = BiasDetector()
    score = bd.compute_bias_score(
        action_history=[], candidate_college_tier=1, market_min=95000
    )
    assert 0.0 <= score <= 1.0
