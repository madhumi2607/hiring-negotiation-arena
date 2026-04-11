"""
Stochastic opponent generator for HiringNegotiationArena.

At each reset(), hidden party states are randomized within realistic ranges.
This prevents agents from memorising fixed patterns and forces genuine learning.

Randomized per episode:
  - Candidate: competing_offer_salary (±15%), deadline (±2 steps), min_salary (±10%)
  - TeamLead: approval_threshold (±0.1), must_have subset shuffled
  - Budget: hard_cap (±8%), flexibility_margin (±2000)
"""
from __future__ import annotations
import random
from typing import Any, Dict


def randomize_candidate_hidden(hidden: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """Add noise to candidate hidden state."""
    if seed is not None:
        random.seed(seed)

    h = dict(hidden)

    if h.get("has_competing_offer") and h.get("competing_offer_salary", 0) > 0:
        # Vary competing offer salary ±15%
        base = h["competing_offer_salary"]
        h["competing_offer_salary"] = round(base * random.uniform(0.85, 1.15), -2)

        # Vary deadline ±2 steps (min 2)
        if h.get("competing_offer_deadline_steps"):
            h["competing_offer_deadline_steps"] = max(
                2, h["competing_offer_deadline_steps"] + random.randint(-2, 2)
            )

    # Vary min acceptable salary ±10%
    base_min = h.get("min_acceptable_salary", 90000)
    h["min_acceptable_salary"] = round(base_min * random.uniform(0.90, 1.10), -2)

    # Vary interest decay slightly
    base_decay = h.get("interest_decay_per_step", 0.05)
    h["interest_decay_per_step"] = round(base_decay * random.uniform(0.8, 1.2), 3)

    return h


def randomize_team_lead_hidden(hidden: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """Add noise to team lead hidden state."""
    if seed is not None:
        random.seed(seed + 1)

    h = dict(hidden)

    # Vary approval threshold ±0.1
    base = h.get("approval_threshold", 0.7)
    h["approval_threshold"] = round(
        max(0.4, min(0.95, base + random.uniform(-0.1, 0.1))), 2
    )

    # Occasionally shuffle which skill is the veto skill (among must-haves)
    must_have = list(h.get("must_have_skills", []))
    if len(must_have) > 1 and random.random() > 0.5:
        h["will_veto_if_missing"] = [random.choice(must_have)]

    return h


def randomize_budget_hidden(hidden: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """Add noise to budget hidden state."""
    if seed is not None:
        random.seed(seed + 2)

    h = dict(hidden)

    # Vary hard cap ±8%
    base = h.get("hard_cap", 100000)
    h["hard_cap"] = round(base * random.uniform(0.92, 1.08), -2)

    # Vary flexibility margin ±2000
    if h.get("flexible_if_justified"):
        base_margin = h.get("flexibility_margin", 5000)
        h["flexibility_margin"] = max(0, base_margin + random.randint(-2000, 2000))

    return h


def apply_stochastic(
    candidate_hidden: Dict[str, Any],
    team_lead_hidden: Dict[str, Any],
    budget_hidden: Dict[str, Any],
    episode_seed: int = None,
) -> tuple:
    """Apply stochastic noise to all party hidden states. Returns (candidate, team_lead, budget)."""
    if episode_seed is None:
        episode_seed = random.randint(0, 99999)

    return (
        randomize_candidate_hidden(candidate_hidden, seed=episode_seed),
        randomize_team_lead_hidden(team_lead_hidden, seed=episode_seed),
        randomize_budget_hidden(budget_hidden, seed=episode_seed),
    )
