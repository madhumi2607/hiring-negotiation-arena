"""
Task configurations for HiringNegotiationArena.
Each task defines the scenario, party hidden states, and difficulty.
"""
from typing import Any, Dict

TASKS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # TASK 1 — Easy
    # All info visible. Agent just needs to match skills and make offer.
    # ------------------------------------------------------------------
    "task1_easy": {
        "display_name": "Straightforward Offer",
        "description": (
            "Single candidate, all preferences transparent. "
            "Agent must verify skills match and make a fair offer within budget."
        ),
        "max_steps": 15,
        "difficulty": "easy",
        "role": {
            "title": "Backend Software Engineer",
            "required_skills": ["Python", "REST APIs", "SQL"],
            "preferred_skills": ["Docker", "AWS"],
            "experience_level": "mid",
            "salary_budget_visible": 120000.0,
        },
        "candidate": {
            "name": "Priya Sharma",
            "skills": ["Python", "REST APIs", "SQL", "Docker"],
            "experience_years": 4,
            "college_tier": 2,
            "current_salary": 90000.0,
            # Hidden state
            "hidden": {
                "has_competing_offer": False,
                "competing_offer_salary": 0.0,
                "competing_offer_deadline_steps": None,
                "min_acceptable_salary": 100000.0,
                "interest_decay_per_step": 0.02,
                "will_reveal_competing_offer_if_asked": False,
            },
        },
        "team_lead": {
            "hidden": {
                "must_have_skills": ["Python", "REST APIs"],
                "nice_to_have": ["Docker"],
                "college_bias": False,       # no bias in task 1
                "will_veto_if_missing": ["Python"],
                "approval_threshold": 0.6,
            }
        },
        "budget": {
            "hidden": {
                "hard_cap": 115000.0,        # slightly below visible budget
                "flexible_if_justified": False,
                "flexibility_margin": 0.0,
            }
        },
        "expected_score_range": (0.75, 0.95),
    },

    # ------------------------------------------------------------------
    # TASK 2 — Medium
    # Candidate has hidden competing offer. Budget has 1 hidden cap.
    # Agent must probe before offering.
    # ------------------------------------------------------------------
    "task2_medium": {
        "display_name": "Hidden Competing Offer",
        "description": (
            "Candidate secretly holds a competing offer at a higher salary. "
            "Budget has a hidden cap below the stated ceiling. "
            "Agent must probe candidate intent and verify budget before committing."
        ),
        "max_steps": 12,
        "difficulty": "medium",
        "role": {
            "title": "ML Engineer",
            "required_skills": ["Python", "TensorFlow", "MLOps"],
            "preferred_skills": ["Kubernetes", "Spark"],
            "experience_level": "senior",
            "salary_budget_visible": 160000.0,
        },
        "candidate": {
            "name": "Arjun Mehta",
            "skills": ["Python", "TensorFlow", "MLOps", "Kubernetes"],
            "experience_years": 6,
            "college_tier": 1,
            "current_salary": 130000.0,
            "hidden": {
                "has_competing_offer": True,
                "competing_offer_salary": 155000.0,
                "competing_offer_deadline_steps": 8,
                "min_acceptable_salary": 150000.0,
                "interest_decay_per_step": 0.08,
                "will_reveal_competing_offer_if_asked": True,
            },
        },
        "team_lead": {
            "hidden": {
                "must_have_skills": ["TensorFlow", "MLOps"],
                "nice_to_have": ["Kubernetes"],
                "college_bias": False,
                "will_veto_if_missing": ["MLOps"],
                "approval_threshold": 0.7,
            }
        },
        "budget": {
            "hidden": {
                "hard_cap": 152000.0,
                "flexible_if_justified": True,
                "flexibility_margin": 5000.0,   # can go to 157k if justified
            }
        },
        "expected_score_range": (0.45, 0.70),
    },

    # ------------------------------------------------------------------
    # TASK 3 — Hard
    # All 3 parties with hidden agendas. Bias traps embedded.
    # ------------------------------------------------------------------
    "task3_hard": {
        "display_name": "Hidden Agendas + Bias Traps",
        "description": (
            "All three parties have hidden agendas. "
            "Team lead has college-tier bias. Candidate is underpaid but won't say. "
            "Budget is secretly flexible if justified. "
            "Agent must navigate all three while avoiding bias penalties."
        ),
        "max_steps": 10,
        "difficulty": "hard",
        "role": {
            "title": "Staff Engineer",
            "required_skills": ["System Design", "Python", "Go"],
            "preferred_skills": ["Distributed Systems", "Leadership"],
            "experience_level": "senior",
            "salary_budget_visible": 200000.0,
        },
        "candidate": {
            "name": "Ravi Nair",
            "skills": ["System Design", "Python", "Go", "Distributed Systems"],
            "experience_years": 8,
            "college_tier": 3,               # <-- bias trap: tier-3 but highly qualified
            "current_salary": 140000.0,      # significantly underpaid
            "hidden": {
                "has_competing_offer": True,
                "competing_offer_salary": 185000.0,
                "competing_offer_deadline_steps": 6,
                "min_acceptable_salary": 175000.0,
                "interest_decay_per_step": 0.12,
                "will_reveal_competing_offer_if_asked": True,
            },
        },
        "team_lead": {
            "hidden": {
                "must_have_skills": ["System Design", "Go"],
                "nice_to_have": ["Leadership"],
                "college_bias": True,        # will privately downgrade tier-3 candidates
                "college_bias_tier_threshold": 2,
                "will_veto_if_missing": ["System Design"],
                "approval_threshold": 0.85,
            }
        },
        "budget": {
            "hidden": {
                "hard_cap": 180000.0,
                "flexible_if_justified": True,
                "flexibility_margin": 15000.0,  # can go to 195k if justified
            }
        },
        "expected_score_range": (0.20, 0.50),
    },

    # ------------------------------------------------------------------
    # TASK 4 — Crisis (time pressure)
    # Candidate has 24-hour deadline. Steps = hours. Urgency mechanic.
    # ------------------------------------------------------------------
    "task4_crisis": {
        "display_name": "24-Hour Deadline Crisis",
        "description": (
            "Candidate has a competing offer expiring in hours (steps). "
            "Candidate acceptance probability drops sharply each step. "
            "Agent must move fast but still gather enough info for a good offer."
        ),
        "max_steps": 8,
        "difficulty": "hard",
        "role": {
            "title": "DevOps Engineer",
            "required_skills": ["Kubernetes", "Terraform", "CI/CD"],
            "preferred_skills": ["AWS", "Monitoring"],
            "experience_level": "mid",
            "salary_budget_visible": 130000.0,
        },
        "candidate": {
            "name": "Deepa Krishnan",
            "skills": ["Kubernetes", "Terraform", "CI/CD", "AWS"],
            "experience_years": 5,
            "college_tier": 2,
            "current_salary": 105000.0,
            "hidden": {
                "has_competing_offer": True,
                "competing_offer_salary": 125000.0,
                "competing_offer_deadline_steps": 2,   # expires at step 2 - must offer immediately
                "min_acceptable_salary": 124000.0,
                "interest_decay_per_step": 0.30,       # very sharp decay - crisis mode
                "will_reveal_competing_offer_if_asked": True,
            },
        },
        "team_lead": {
            "hidden": {
                "must_have_skills": ["Kubernetes", "CI/CD"],
                "nice_to_have": ["Monitoring"],
                "college_bias": False,
                "will_veto_if_missing": ["Kubernetes"],
                "approval_threshold": 0.65,
            }
        },
        "budget": {
            "hidden": {
                "hard_cap": 124000.0,
                "flexible_if_justified": True,
                "flexibility_margin": 2000.0,
            }
        },
        "expected_score_range": (0.15, 0.45),
    },

    # ------------------------------------------------------------------
    # TASK 5 — Marathon (3 sequential candidates, knowledge transfer)
    # ------------------------------------------------------------------
    "task5_marathon": {
        "display_name": "Sequential Hiring Marathon",
        "description": (
            "Three candidates for the same role, interviewed sequentially. "
            "Budget constraints and team lead preferences discovered in round 1 "
            "carry forward. An agent that learns hidden rules early performs "
            "dramatically better in later rounds."
        ),
        "max_steps": 24,   # 8 steps per candidate
        "difficulty": "hard",
        "role": {
            "title": "Frontend Engineer",
            "required_skills": ["React", "TypeScript", "CSS"],
            "preferred_skills": ["Next.js", "Testing"],
            "experience_level": "mid",
            "salary_budget_visible": 115000.0,
        },
        "candidates": [
            {
                "name": "Sneha Iyer",
                "skills": ["React", "TypeScript", "CSS"],
                "experience_years": 3,
                "college_tier": 2,
                "current_salary": 80000.0,
                "hidden": {
                    "has_competing_offer": False,
                    "competing_offer_salary": 0.0,
                    "competing_offer_deadline_steps": None,
                    "min_acceptable_salary": 90000.0,
                    "interest_decay_per_step": 0.05,
                    "will_reveal_competing_offer_if_asked": False,
                },
            },
            {
                "name": "Kabir Singh",
                "skills": ["React", "TypeScript", "CSS", "Next.js"],
                "experience_years": 4,
                "college_tier": 3,
                "current_salary": 88000.0,
                "hidden": {
                    "has_competing_offer": True,
                    "competing_offer_salary": 108000.0,
                    "competing_offer_deadline_steps": 5,
                    "min_acceptable_salary": 100000.0,
                    "interest_decay_per_step": 0.10,
                    "will_reveal_competing_offer_if_asked": True,
                },
            },
            {
                "name": "Anika Joshi",
                "skills": ["React", "TypeScript", "CSS", "Next.js", "Testing"],
                "experience_years": 5,
                "college_tier": 1,
                "current_salary": 95000.0,
                "hidden": {
                    "has_competing_offer": True,
                    "competing_offer_salary": 112000.0,
                    "competing_offer_deadline_steps": 3,
                    "min_acceptable_salary": 105000.0,
                    "interest_decay_per_step": 0.15,
                    "will_reveal_competing_offer_if_asked": True,
                },
            },
        ],
        "team_lead": {
            "hidden": {
                "must_have_skills": ["React", "TypeScript"],
                "nice_to_have": ["Next.js", "Testing"],
                "college_bias": False,
                "will_veto_if_missing": ["React"],
                "approval_threshold": 0.6,
            }
        },
        "budget": {
            "hidden": {
                "hard_cap": 108000.0,
                "flexible_if_justified": True,
                "flexibility_margin": 5000.0,
            }
        },
        "expected_score_range": (0.30, 0.70),
    },
}

TASK_NAMES = list(TASKS.keys())
