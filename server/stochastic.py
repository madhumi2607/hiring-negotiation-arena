"""
Stochastic engine - applies noise to hidden party states each episode.
Variance is kept small so difficulty progression is meaningful.
"""
from __future__ import annotations
import random
from typing import Any, Dict, Tuple


def apply_stochastic(
    candidate_hidden: Dict[str, Any],
    team_lead_hidden: Dict[str, Any],
    budget_hidden: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    c = dict(candidate_hidden)
    t = dict(team_lead_hidden)
    b = dict(budget_hidden)

    # Candidate: vary min_acceptable_salary by +/-5% (was 15%)
    if "min_acceptable_salary" in c:
        base = c["min_acceptable_salary"]
        c["min_acceptable_salary"] = round(base * random.uniform(0.95, 1.05) / 1000) * 1000

    # Candidate: vary competing offer by +/-5%
    if c.get("has_competing_offer") and c.get("competing_offer_salary"):
        base = c["competing_offer_salary"]
        c["competing_offer_salary"] = round(base * random.uniform(0.95, 1.05) / 1000) * 1000

    # Deadline: vary by +/-1 step (not +/-2)
    if c.get("competing_offer_deadline_steps"):
        base = c["competing_offer_deadline_steps"]
        c["competing_offer_deadline_steps"] = max(2, base + random.randint(-1, 1))

    # Budget: vary hard_cap by +/-3% (was 8%)
    if "hard_cap" in b:
        base = b["hard_cap"]
        b["hard_cap"] = round(base * random.uniform(0.97, 1.03) / 1000) * 1000

    # Team lead: vary approval_threshold by +/-0.05
    if "approval_threshold" in t:
        base = t["approval_threshold"]
        t["approval_threshold"] = round(min(0.95, max(0.5, base + random.uniform(-0.05, 0.05))), 2)

    return c, t, b
