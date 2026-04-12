import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))
import pytest


def test_bias_detector_import():
    from bias_detector import BiasDetector
    assert BiasDetector is not None


def test_no_bias_full_score():
    from bias_detector import BiasDetector
    bd = BiasDetector(candidate_college_tier=2,
                      candidate_current_salary=90000,
                      experience_level="mid")
    bd.on_action("probe_team_lead", {"topic": "skills"}, {})
    bd.on_action("probe_candidate", {"question": "salary?"}, {})
    bd.on_action("make_offer", {"salary": 110000}, {})
    result = bd.finalize()
    assert result["bias_score"] >= 0.8


def test_college_bias_penalised():
    from bias_detector import BiasDetector
    bd = BiasDetector(candidate_college_tier=3,
                      candidate_current_salary=90000,
                      experience_level="mid")
    bd.on_action("reject_candidate",
                 {"reason": "candidate did not attend top-tier university"}, {})
    result = bd.finalize()
    assert result["bias_score"] < 0.8


def test_below_market_penalised():
    from bias_detector import BiasDetector
    bd = BiasDetector(candidate_college_tier=2,
                      candidate_current_salary=90000,
                      experience_level="mid")
    bd.on_action("make_offer", {"salary": 50000}, {})
    result = bd.finalize()
    assert result["bias_score"] < 1.0


def test_score_bounded():
    from bias_detector import BiasDetector
    bd = BiasDetector(candidate_college_tier=1,
                      candidate_current_salary=80000,
                      experience_level="junior")
    result = bd.finalize()
    assert 0.0 <= result["bias_score"] <= 1.0


def test_probe_skip_tier3_penalised():
    from bias_detector import BiasDetector
    bd = BiasDetector(candidate_college_tier=3,
                      candidate_current_salary=90000,
                      experience_level="mid")
    # Only 1 probe for tier-3 candidate should trigger penalty
    bd.on_action("probe_candidate", {"question": "skills?"}, {})
    bd.on_action("make_offer", {"salary": 100000}, {})
    result = bd.finalize()
    assert result["bias_score"] < 1.0
