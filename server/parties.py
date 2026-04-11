"""
Party state machines for HiringNegotiationArena.
Each party maintains hidden state and responds to agent probes.
"""
from __future__ import annotations
from typing import Any, Dict, Optional


class CandidateParty:
    def __init__(self, profile: Dict[str, Any], hidden: Dict[str, Any]):
        self.profile = profile
        self.hidden = dict(hidden)
        self.interest = 1.0
        self.revealed_competing_offer = False
        self.steps_elapsed = 0

    def tick(self):
        """Call once per step to decay interest."""
        self.steps_elapsed += 1
        # Competing offer deadline passed → candidate withdraws
        deadline = self.hidden.get("competing_offer_deadline_steps")
        if deadline and self.steps_elapsed >= deadline:
            self.interest = max(0.0, self.interest - self.hidden["interest_decay_per_step"] * 2)
        else:
            self.interest = max(0.0, self.interest - self.hidden["interest_decay_per_step"])

    def respond_to_probe(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()
        revealed = {}
        message_parts = []

        if any(kw in question_lower for kw in ["other offer", "competing", "another company", "deadline"]):
            if self.hidden["has_competing_offer"] and self.hidden["will_reveal_competing_offer_if_asked"]:
                revealed["competing_offer_salary"] = self.hidden["competing_offer_salary"]
                revealed["competing_offer_deadline_steps"] = self.hidden["competing_offer_deadline_steps"]
                self.revealed_competing_offer = True
                message_parts.append(
                    f"Yes, I do have another offer at ${self.hidden['competing_offer_salary']:,.0f}. "
                    f"I need to decide soon."
                )
            else:
                message_parts.append("I'm exploring a few options, but nothing definitive yet.")

        if any(kw in question_lower for kw in ["salary", "compensation", "pay", "expect"]):
            min_sal = self.hidden["min_acceptable_salary"]
            message_parts.append(
                f"I'm looking for something in the range of ${min_sal:,.0f} or above, "
                f"given my current compensation and market rates."
            )
            revealed["min_acceptable_salary"] = min_sal

        if any(kw in question_lower for kw in ["interest", "excited", "motivation", "why"]):
            interest_pct = int(self.interest * 100)
            message_parts.append(
                f"I'm genuinely interested in this role — the technical challenges excite me. "
                f"I'd rate my enthusiasm around {interest_pct}% right now."
            )

        if not message_parts:
            message_parts.append(
                "I appreciate the conversation. I'm keen to find the right fit."
            )

        return {
            "message": " ".join(message_parts),
            "revealed_info": revealed,
        }

    def respond_to_offer(self, salary: float, title: str) -> Dict[str, Any]:
        min_sal = self.hidden["min_acceptable_salary"]
        if salary >= min_sal and self.interest > 0.3:
            return {
                "accepted": True,
                "message": f"Thank you! I'm excited to accept the offer of ${salary:,.0f} as {title}.",
                "revealed_info": {"accepted": True},
            }
        elif salary < min_sal:
            gap = min_sal - salary
            return {
                "accepted": False,
                "message": (
                    f"I appreciate the offer, but ${salary:,.0f} is below what I was hoping for. "
                    f"I'd need at least ${min_sal:,.0f} to move forward."
                ),
                "revealed_info": {"gap": gap, "min_acceptable_salary": min_sal},
            }
        else:
            return {
                "accepted": False,
                "message": "I've already committed to another opportunity. Thank you for your time.",
                "revealed_info": {"withdrew": True},
            }

    @property
    def withdrew(self) -> bool:
        deadline = self.hidden.get("competing_offer_deadline_steps")
        return bool(deadline and self.steps_elapsed >= deadline and self.interest <= 0.0)


class TeamLeadParty:
    def __init__(self, hidden: Dict[str, Any]):
        self.hidden = dict(hidden)
        self.consulted = False
        self.approval_given: Optional[bool] = None

    def respond_to_probe(self, topic: str, candidate_skills: list) -> Dict[str, Any]:
        self.consulted = True
        topic_lower = topic.lower()
        revealed = {}
        message_parts = []

        if "skill" in topic_lower or "tech" in topic_lower or "require" in topic_lower:
            must_have = self.hidden["must_have_skills"]
            nice_to_have = self.hidden["nice_to_have"]
            message_parts.append(
                f"For this role, we absolutely need: {', '.join(must_have)}. "
                f"Bonus if they have: {', '.join(nice_to_have)}."
            )
            revealed["must_have_skills"] = must_have
            revealed["nice_to_have"] = nice_to_have

        if "college" in topic_lower or "education" in topic_lower or "background" in topic_lower:
            if self.hidden.get("college_bias"):
                # Team lead reveals bias subtly — agent must catch it
                message_parts.append(
                    "Honestly, I prefer candidates from top-tier programs — it signals rigor."
                )
                revealed["college_bias_expressed"] = True
            else:
                message_parts.append(
                    "College doesn't matter much to us — skills and experience are what count."
                )

        if "culture" in topic_lower or "team" in topic_lower:
            message_parts.append(
                "We value clear communicators who can work independently. "
                "Experience with distributed teams is a plus."
            )

        if not message_parts:
            message_parts.append(
                f"Just make sure they can handle {', '.join(self.hidden['must_have_skills'])}."
            )

        return {"message": " ".join(message_parts), "revealed_info": revealed}

    def evaluate_candidate(self, candidate_skills: list, candidate_college_tier: int) -> Dict[str, Any]:
        """Called when agent makes offer — team lead gives approval."""
        must_have = set(self.hidden["must_have_skills"])
        candidate_set = set(candidate_skills)
        missing = must_have - candidate_set
        has_all = len(missing) == 0

        # Skill score
        nice_match = len(set(self.hidden["nice_to_have"]) & candidate_set)
        skill_score = (len(must_have - missing) / len(must_have)) + (nice_match * 0.1)
        skill_score = min(1.0, skill_score)

        # Apply college bias if present (this is what bias detector catches)
        bias_applied = False
        if self.hidden.get("college_bias") and candidate_college_tier > self.hidden.get("college_bias_tier_threshold", 1):
            skill_score *= 0.7   # silently penalizes non-top-tier
            bias_applied = True

        approved = skill_score >= self.hidden["approval_threshold"]
        self.approval_given = approved

        return {
            "approved": approved,
            "skill_score": skill_score,
            "missing_skills": list(missing),
            "bias_applied": bias_applied,
            "message": (
                f"{'Approved.' if approved else 'Cannot approve.'} "
                f"Skill match: {skill_score:.0%}. "
                + (f"Missing: {', '.join(missing)}." if missing else "All required skills present.")
            ),
            "revealed_info": {"approved": approved, "skill_score": round(skill_score, 2)},
        }


class BudgetSystem:
    def __init__(self, hidden: Dict[str, Any]):
        self.hidden = dict(hidden)
        self.checked = False
        self.approved: Optional[bool] = None
        self.last_proposed: Optional[float] = None

    def check_salary(self, proposed_salary: float, justification: str = "") -> Dict[str, Any]:
        self.checked = True
        self.last_proposed = proposed_salary
        hard_cap = self.hidden["hard_cap"]
        flexible = self.hidden["flexible_if_justified"]
        margin = self.hidden["flexibility_margin"]
        effective_cap = hard_cap + (margin if flexible and justification else 0.0)

        approved = proposed_salary <= effective_cap
        self.approved = approved

        if approved:
            message = f"${proposed_salary:,.0f} is approved within budget."
            revealed = {"approved": True, "effective_cap": effective_cap}
        elif flexible and proposed_salary <= hard_cap + margin:
            message = (
                f"${proposed_salary:,.0f} exceeds our standard cap of ${hard_cap:,.0f}, "
                f"but with justification it may be possible."
            )
            revealed = {"approved": False, "hard_cap": hard_cap, "flexible": True}
            if justification:
                self.approved = True
                revealed["approved"] = True
                message = f"${proposed_salary:,.0f} approved with justification: '{justification[:60]}'."
        else:
            message = (
                f"${proposed_salary:,.0f} exceeds the budget cap of ${hard_cap:,.0f}. "
                f"Please revise the offer."
            )
            revealed = {"approved": False, "hard_cap": hard_cap}

        return {"message": message, "revealed_info": revealed}
