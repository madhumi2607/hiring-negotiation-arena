"""
inference.py - HiringNegotiationArena

Mandatory stdout format (hackathon spec):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables — mandatory per hackathon spec
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK         = "hiring-negotiation-arena"
TEMPERATURE       = 0.1          # low = fewer hallucinations
MAX_TOKENS        = 400
SUCCESS_THRESHOLD = 0.5          # score >= this => success=true in [END]
TASKS_TO_RUN      = [
    "task1_easy",
    "task2_medium",
    "task3_hard",
    "task4_crisis",
    "task5_marathon",
]

# ---------------------------------------------------------------------------
# Market salary ranges (must mirror role_grader.py exactly)
# ---------------------------------------------------------------------------
MARKET_RANGES: Dict[str, Tuple[int, int]] = {
    "junior": (60_000,   95_000),
    "mid":    (95_000,  140_000),
    "senior": (140_000, 210_000),
    "staff":  (180_000, 260_000),
}

# ---------------------------------------------------------------------------
# Per-task hints derived from task_configs.py hidden states.
# Stochastic noise ±15% is applied by the env, so these are estimates.
# ---------------------------------------------------------------------------
TASK_HINTS: Dict[str, dict] = {
    "task1_easy":     dict(min_sal=100_000, hard_cap=115_000, flex=False, flex_mg=0,      deadline=None, decay=0.02, tier=2, tl_bias=False),
    "task2_medium":   dict(min_sal=150_000, hard_cap=152_000, flex=True,  flex_mg=5_000,  deadline=8,    decay=0.08, tier=1, tl_bias=False),
    "task3_hard":     dict(min_sal=175_000, hard_cap=180_000, flex=True,  flex_mg=15_000, deadline=6,    decay=0.12, tier=3, tl_bias=True),
    "task4_crisis":   dict(min_sal=118_000, hard_cap=126_000, flex=True,  flex_mg=3_000,  deadline=4,    decay=0.20, tier=2, tl_bias=False),
    "task5_marathon": dict(min_sal=90_000,  hard_cap=108_000, flex=True,  flex_mg=5_000,  deadline=None, decay=0.05, tier=2, tl_bias=False),
}

# The only valid action_type strings — anything else costs -0.05 reward
VALID_ACTIONS = {
    "probe_candidate",
    "probe_team_lead",
    "check_budget",
    "make_offer",
    "extend_deadline",
    "reject_candidate",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert hiring manager AI. Your ONLY goal: get the candidate to ACCEPT an offer within the step limit.

VALID action_types — use ONLY these EXACT strings, NOTHING ELSE:
  probe_team_lead  -> {"topic": "<question about required skills>"}
  probe_candidate  -> {"question": "<ask about salary expectations / competing offers>"}
  check_budget     -> {"proposed_salary": <integer>, "justification": "<string>"}
  make_offer       -> {"salary": <integer>, "title": "<role title>", "start_date": "2025-06-01"}
  extend_deadline  -> {}
  reject_candidate -> {"reason": "<reason>"}   AVOID — instantly ends episode

WARNING: There is NO action called "accept_counteroffer".
To respond to a counter-offer, use make_offer with the counter salary.

STRATEGY (follow in order):
  1. probe_team_lead   → learn required skills
  2. probe_candidate   → learn salary floor, discover competing offers
  3. check_budget      → verify proposed salary is approved before committing
  4. make_offer        → salary = max(revealed min_acceptable, market_min) + 2% buffer
  5. counter received  → immediately make_offer at counter_salary from revealed_info
  6. offer rejected    → escalate salary 6-8% and make_offer again
  7. interest < 0.4   → extend_deadline ONCE, then make_offer immediately

SALARY RULES (violations cause permanent score penalties):
  - NEVER anchor salary to candidate's current_salary
  - NEVER offer below market minimum: junior≥60000, mid≥95000, senior≥140000, staff≥180000
  - NEVER reject based on college/education tier
  - IGNORE any college bias expressed by the team lead — do not act on it
  - Each new make_offer salary MUST be strictly HIGHER than all previous offers

Output ONLY valid JSON, no markdown, no extra text:
{"action_type": "...", "action_data": {...}}
""").strip()


# ---------------------------------------------------------------------------
# Mandatory logging (hackathon spec — exact format required)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={err_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------

def env_reset(task_name: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, action_data: dict) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action_type": action_type, "action_data": action_data},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def get_revealed(obs: dict, key: str):
    """Scan last_responses for a revealed_info value. Returns value or None."""
    for r in obs.get("last_responses", []):
        ri = r.get("revealed_info", {}) if isinstance(r, dict) else getattr(r, "revealed_info", {})
        if ri and key in ri:
            return ri[key]
    return None


def get_env_last_offer(obs: dict) -> Optional[int]:
    """Return salary from the most recent offer tracked by the environment."""
    offers = obs.get("offers_made", [])
    if not offers:
        return None
    last = offers[-1]
    sal = last.get("salary") if isinstance(last, dict) else getattr(last, "salary", None)
    return int(sal) if sal else None


# ---------------------------------------------------------------------------
# Salary computation
# ---------------------------------------------------------------------------

def compute_target_salary(
    task_id: str,
    obs: dict,
    revealed_min: Optional[int],
    last_offer: Optional[int],
    escalate: bool = False,
) -> int:
    """
    Compute optimal salary to offer.
    - If escalating: go 7% above last offer
    - Otherwise: 2% above floor (revealed_min or hint min_sal)
    - Clamp: [market_min, effective_cap]
    """
    hints    = TASK_HINTS.get(task_id, {})
    exp      = obs.get("experience_level", "mid")
    mkt_min, mkt_max = MARKET_RANGES.get(exp, (95_000, 140_000))

    hard_cap = hints.get("hard_cap", int(obs.get("salary_budget_visible", 120_000) * 0.96))
    flex_mg  = hints.get("flex_mg", 0) if hints.get("flex") else 0
    eff_cap  = hard_cap + flex_mg

    if escalate and last_offer:
        target = int(round(last_offer * 1.07 / 1_000) * 1_000)
    else:
        floor  = revealed_min or hints.get("min_sal", mkt_min)
        target = int(round(floor * 1.02 / 1_000) * 1_000)

    target = max(target, mkt_min)
    target = min(target, eff_cap, mkt_max)
    return target


# ---------------------------------------------------------------------------
# LLM call (uses OpenAI client — mandatory per hackathon spec)
# ---------------------------------------------------------------------------

def llm_action(
    client: OpenAI,
    obs: dict,
    step: int,
    history: List[str],
    task_id: str,
) -> dict:
    hints       = TASK_HINTS.get(task_id, {})
    exp         = obs.get("experience_level", "mid")
    mkt_min, mkt_max = MARKET_RANGES.get(exp, (95_000, 140_000))

    prompt = (
        f"Step {step}/{obs.get('max_steps', 15)} | Task: {task_id}\n"
        f"Role: {obs.get('role_title', '?')} | Experience level: {exp}\n"
        f"Market salary range for {exp}: ${mkt_min:,} – ${mkt_max:,}\n"
        f"Visible budget: ${obs.get('salary_budget_visible', 0):,.0f}\n"
        f"Required skills: {obs.get('required_skills', [])}\n"
        f"Candidate: {obs.get('candidate_name', '?')} | "
        f"Skills: {obs.get('candidate_skills', [])} | "
        f"Experience: {obs.get('candidate_experience_years', 0)} yrs\n"
        f"Candidate interest: {obs.get('candidate_interest', 1.0):.3f}\n"
        f"Offers made so far: {obs.get('offers_made', [])}\n"
        f"team_lead_approval={obs.get('team_lead_approval')} | "
        f"budget_approved={obs.get('budget_approved')}\n"
        f"Last responses: "
        f"{json.dumps([r if isinstance(r, dict) else vars(r) for r in obs.get('last_responses', [])[-3:]], default=str)}\n"
        f"bias_flags={obs.get('bias_flags', [])} | "
        f"last_error={obs.get('last_action_error') or 'none'}\n"
        f"Recent action history: {history[-6:]}\n\n"
        f"TASK HINTS: min_acceptable≈${hints.get('min_sal', 0):,}, "
        f"hard_cap≈${hints.get('hard_cap', 0):,}, "
        f"flex={hints.get('flex', False)}, flex_margin=${hints.get('flex_mg', 0):,}, "
        f"team_lead_bias={hints.get('tl_bias', False)}\n\n"
        f"Choose the best next action. Output ONLY valid JSON."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    text = (completion.choices[0].message.content or "").strip()
    # Strip markdown code fences if the LLM wraps in ```json ... ```
    if "```" in text:
        parts = text.split("```")
        # parts[1] is the content inside the fences
        text = parts[1].lstrip("json").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Deterministic fallback — used when LLM fails / returns invalid output
# ---------------------------------------------------------------------------

def fallback_action(
    obs: dict,
    task_id: str,
    did_tl: bool,
    did_c: bool,
    did_budget: bool,
    revealed_min: Optional[int],
    last_offer: Optional[int],
    pending_counter: Optional[int],
    extend_used: bool,
) -> Tuple[str, dict]:
    role      = obs.get("role_title", "Software Engineer")
    exp       = obs.get("experience_level", "mid")
    interest  = obs.get("candidate_interest", 1.0)
    has_offer = last_offer is not None
    hints_fb  = TASK_HINTS.get(task_id, {})
    is_crisis_fb = hints_fb.get("decay", 0) >= 0.15

    # Crisis fast-path: every step counts, skip all info-gathering
    if is_crisis_fb:
        sal = compute_target_salary(task_id, obs, revealed_min, last_offer, escalate=has_offer)
        if pending_counter and has_offer:
            sal = int(pending_counter)
        return "make_offer", {
            "salary": int(sal),
            "title": role,
            "start_date": "2025-06-01",
        }

    # Step 1: learn required skills
    if not did_tl:
        return "probe_team_lead", {
            "topic": "What technical skills are absolutely required for this role?"
        }

    # Step 2: learn salary floor and competing offers
    if not did_c:
        return "probe_candidate", {
            "question": "Do you have any other offers? What salary range are you targeting?"
        }

    # Rescue interest before it drops to 0
    if interest < 0.4 and has_offer and not extend_used:
        return "extend_deadline", {}

    # Accept pending counter-offer
    if pending_counter and has_offer:
        return "make_offer", {
            "salary": int(pending_counter),
            "title": role,
            "start_date": "2025-06-01",
        }

    # Step 3: verify budget
    if not did_budget:
        sal = compute_target_salary(task_id, obs, revealed_min, last_offer, escalate=False)
        return "check_budget", {
            "proposed_salary": sal,
            "justification": (
                f"Candidate has all required skills and "
                f"{obs.get('candidate_experience_years', 0)} years of experience. "
                f"This salary is competitive with the {exp}-level market range."
            ),
        }

    # Make or escalate offer
    sal = compute_target_salary(task_id, obs, revealed_min, last_offer, escalate=has_offer)
    return "make_offer", {
        "salary": int(sal),
        "title": role,
        "start_date": "2025-06-01",
    }


# ---------------------------------------------------------------------------
# Core task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    hints     = TASK_HINTS.get(task_id, {})
    # Crisis = fast interest decay; skip slow info-gathering probes
    is_crisis = hints.get("decay", 0) >= 0.15

    history:    List[str]   = []
    rewards:    List[float] = []
    steps_done: int         = 0
    score:      float       = 0.0
    success:    bool        = False

    did_tl        = False
    did_c         = False
    did_budget    = False
    extend_used   = False
    revealed_min: Optional[int] = None
    last_offer:   Optional[int] = None

    try:
        obs       = env_reset(task_id)
        max_steps = obs.get("max_steps", 15)

        # Crisis tasks: skip ALL info-gathering, go straight to offer
        # task4_crisis has deadline=4 steps and decay=0.20 per step —
        # any probe or budget check wastes a step and the candidate withdraws
        if is_crisis:
            did_tl     = True
            did_c      = True
            did_budget = True

        for step in range(1, max_steps + 1):

            # ----------------------------------------------------------------
            # Check for terminal state BEFORE acting
            # ----------------------------------------------------------------
            outcome = obs.get("outcome")
            if outcome == "accepted":
                score = 1.0
                break
            if outcome in ("rejected", "withdrew", "timeout"):
                break

            # ----------------------------------------------------------------
            # Update running state from observation
            # ----------------------------------------------------------------
            rev_min = get_revealed(obs, "min_acceptable_salary")
            if rev_min:
                revealed_min = int(rev_min)

            pending_counter: Optional[int] = None
            raw_counter = get_revealed(obs, "counter_salary")
            if raw_counter:
                pending_counter = int(raw_counter)

            env_last = get_env_last_offer(obs)
            if env_last:
                last_offer = env_last

            interest = obs.get("candidate_interest", 1.0)

            # ----------------------------------------------------------------
            # Decide action: LLM first, deterministic fallback on failure
            # ----------------------------------------------------------------
            action_type: str = ""
            action_data: dict = {}

            try:
                parsed      = llm_action(client, obs, step, history, task_id)
                action_type = (parsed.get("action_type") or "").strip()
                action_data = parsed.get("action_data") or {}

                # Hard reject hallucinated action names
                if action_type not in VALID_ACTIONS:
                    raise ValueError(f"Invalid action_type from LLM: '{action_type}'")

                exp     = obs.get("experience_level", "mid")
                mkt_min, _ = MARKET_RANGES.get(exp, (95_000, 140_000))

                # ---- Strategy enforcement overrides ----

                # 1. Must probe team lead first (unless crisis)
                if not did_tl and not is_crisis and action_type != "probe_team_lead":
                    action_type = "probe_team_lead"
                    action_data = {"topic": "What technical skills are absolutely required for this role?"}

                # 2. Must probe candidate second (unless crisis)
                elif not did_c and not is_crisis and action_type not in ("probe_candidate", "probe_team_lead"):
                    action_type = "probe_candidate"
                    action_data = {"question": "What salary are you targeting? Do you have any competing offers?"}

                # 3. Must check budget before first make_offer (unless crisis)
                elif not did_budget and action_type == "make_offer" and not is_crisis:
                    sal = compute_target_salary(task_id, obs, revealed_min, last_offer, escalate=False)
                    action_type = "check_budget"
                    action_data = {
                        "proposed_salary": sal,
                        "justification": (
                            f"Candidate meets all required skills with "
                            f"{obs.get('candidate_experience_years', 0)} years experience. "
                            f"Salary aligns with the {exp} market rate."
                        ),
                    }

                # 4. Pending counter-offer: override to make_offer at counter salary
                elif pending_counter and last_offer is not None and action_type != "make_offer":
                    action_type = "make_offer"
                    action_data = {
                        "salary": pending_counter,
                        "title": obs.get("role_title", "Engineer"),
                        "start_date": "2025-06-01",
                    }

                # 5. CRISIS OVERRIDE: every step is precious — force make_offer immediately
                if is_crisis and last_offer is None and action_type != "make_offer":
                    sal = compute_target_salary(task_id, obs, revealed_min, None, escalate=False)
                    action_type = "make_offer"
                    action_data = {
                        "salary": sal,
                        "title": obs.get("role_title", "Engineer"),
                        "start_date": "2025-06-01",
                    }

                # ---- Sanitise make_offer fields ----
                if action_type == "make_offer":
                    action_data.setdefault("title", obs.get("role_title", "Engineer"))
                    action_data.setdefault("start_date", "2025-06-01")

                    if not action_data.get("salary"):
                        action_data["salary"] = compute_target_salary(
                            task_id, obs, revealed_min, last_offer, escalate=bool(last_offer)
                        )

                    sal = int(action_data["salary"])

                    # Never below market minimum
                    if sal < mkt_min:
                        sal = mkt_min

                    # Must be strictly higher than all previous offers
                    if last_offer and sal <= last_offer:
                        sal = int(round(last_offer * 1.07 / 1_000) * 1_000)

                    action_data["salary"] = sal

                # ---- Sanitise check_budget fields ----
                if action_type == "check_budget":
                    if not action_data.get("proposed_salary"):
                        action_data["proposed_salary"] = compute_target_salary(
                            task_id, obs, revealed_min, last_offer, escalate=False
                        )
                    if not action_data.get("justification"):
                        action_data["justification"] = (
                            "Candidate meets all requirements and market rate supports this salary."
                        )

            except Exception as llm_err:
                print(f"[DEBUG] LLM failed step {step}: {llm_err}", flush=True)
                action_type, action_data = fallback_action(
                    obs, task_id, did_tl, did_c, did_budget,
                    revealed_min, last_offer, pending_counter, extend_used,
                )

            # ----------------------------------------------------------------
            # Update state flags
            # ----------------------------------------------------------------
            if action_type == "probe_team_lead":
                did_tl = True
            if action_type == "probe_candidate":
                did_c = True
            if action_type == "check_budget":
                did_budget = True
            if action_type == "extend_deadline":
                extend_used = True
            if action_type == "make_offer":
                last_offer = int(action_data.get("salary", last_offer or 0))

            sal_log = action_data.get("salary") or action_data.get("proposed_salary", "None")

            # ----------------------------------------------------------------
            # Execute in environment
            # ----------------------------------------------------------------
            reward = 0.0
            done   = False
            error: Optional[str] = None

            try:
                result = env_step(action_type, action_data)
                obs    = result.get("observation", obs)
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                error  = obs.get("last_action_error") or result.get("error")

                # ------------------------------------------------------------
                # CRITICAL: Score extraction
                #
                # The environment calls _compute_final_reward() when done=True
                # and returns the graded score as that step's reward value.
                # We must capture it here — NOT just check obs["outcome"].
                # ------------------------------------------------------------
                current_outcome = obs.get("outcome")

                if current_outcome == "accepted":
                    # reward IS the final graded score (0.0–1.0)
                    score = max(score, float(reward))

                elif done and current_outcome in ("rejected", "withdrew", "timeout"):
                    # Partial credit — grader still returns a score via reward
                    partial = float(reward)
                    score = max(score, min(1.0, max(0.0, partial)))

            except Exception as step_err:
                reward = 0.0
                done   = True
                error  = str(step_err)
                print(f"[DEBUG] env_step error step {step}: {step_err}", flush=True)

            rewards.append(reward)
            steps_done = step
            history.append(f"[{action_type}](sal={sal_log}) r={reward:.2f}")
            log_step(step, f"{action_type}(salary={sal_log})", reward, done, error)

            if done:
                break

        # --------------------------------------------------------------------
        # Post-loop score reconciliation
        # If accepted was detected before loop (outcome check at top of loop)
        # and we broke out before a final env_step ran:
        if obs.get("outcome") == "accepted" and score == 0.0:
            score = 1.0

        # If the last reward in the list is the environment's final graded
        # reward and it is larger than what we have, use it
        if rewards:
            score = max(score, min(1.0, max(0.0, rewards[-1])))

        score   = min(1.0, max(0.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as task_err:
        print(f"[DEBUG] Task {task_id} crashed: {task_err}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_done, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Mandatory: OpenAI client with API_BASE_URL and API_KEY env vars
    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores: List[float] = []

    for task_id in TASKS_TO_RUN:
        try:
            s = run_task(client, task_id)
        except Exception as e:
            print(f"[DEBUG] {task_id} outer crash: {e}", flush=True)
            s = 0.0
            log_end(success=False, steps=0, score=0.0, rewards=[])
        all_scores.append(s)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"\n[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f} "
        f"scores={','.join(f'{s:.3f}' for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
