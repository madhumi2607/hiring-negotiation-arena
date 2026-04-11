
"""
inference.py — HiringNegotiationArena
======================================
Mandatory stdout format:
    [START] task=<task_name> env=hiring-negotiation-arena model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
Environment variables:
    API_BASE_URL   LLM endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       API key
    ENV_BASE_URL   Server URL (default: http://localhost:7860)
"""
from __future__ import annotations
import json, os, textwrap
from typing import List, Optional
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "hiring-negotiation-arena"
MAX_STEPS    = 15
TEMPERATURE  = 0.3
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.4

TASKS_TO_RUN = [
    "task1_easy",
    "task2_medium",
    "task3_hard",
    "task4_crisis",
    "task5_marathon",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ---------------------------------------------------------------------------
# Environment client — correct API schema
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action: dict) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_score() -> float:
    try:
        return float(requests.get(f"{ENV_BASE_URL}/score", timeout=10).json().get("score", 0.0))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert hiring manager AI negotiating a job offer.
Goal: hire the best candidate — get them to accept, get team lead approval, stay in budget, avoid bias.

STRATEGY:
- Step 1: consult_team_lead to discover required skills
- Step 2: probe_candidate to discover competing offers and salary expectations
- Step 3: check_budget with a business case (unlocks flexible budgets)
- Step 4+: make_offer at market rate — DO NOT keep probing

RULES:
- NEVER reject based on college, school, or university
- NEVER anchor salary to current salary — use market range
- Make offer by step 4 at the latest unless you learn something critical

Respond ONLY with valid JSON:
{
  "action_type": "<probe_candidate|consult_team_lead|check_budget|request_skills_test|make_offer|accept_counteroffer|extend_deadline>",
  "salary_offer": <number or null>,
  "message": "<string>",
  "benefits": {"remote_days": 3, "signing_bonus": 10000, "equity_percent": 0.05, "pto_days": 25} or null,
  "skill_to_verify": "<string or null>"
}
""").strip()


def build_prompt(obs: dict, step: int, history: List[str]) -> str:
    return (
        f"Step {step}/{obs.get('steps_remaining', 0) + step} | "
        f"Role: {obs.get('role_title','?')} | "
        f"Market: ${obs.get('market_salary_min',0):,.0f}-${obs.get('market_salary_max',0):,.0f}\n"
        f"Required skills: {', '.join(obs.get('required_skills', []))}\n"
        f"Candidate: {obs.get('candidate_name','?')} | "
        f"Status: {obs.get('candidate_status','?')} | "
        f"Sentiment: {obs.get('candidate_sentiment',0):.2f}\n"
        f"Skills: {', '.join(obs.get('candidate_skills', []))}\n"
        f"Current salary: ${obs.get('candidate_current_salary') or 0:,.0f}\n"
        f"Candidate says: \"{obs.get('candidate_message','')[:150]}\"\n"
        f"Team lead [{obs.get('team_lead_status','?')}]: \"{obs.get('team_lead_message','')[:100]}\"\n"
        f"Budget: {'APPROVED' if obs.get('budget_approved') else 'not checked'} | "
        f"{obs.get('budget_message','')[:80]}\n"
        f"Bias score: {obs.get('bias_score',0):.2f} | "
        f"Offers made: {len(obs.get('offers_made',[]))}\n"
        f"History: {' | '.join(history[-4:]) or 'none'}\n\nChoose action:"
    )


def force_action(obs: dict, step: int) -> dict:
    """Deterministic fallback — guarantees offer by step 4."""
    mid = (obs.get("market_salary_min", 100000) + obs.get("market_salary_max", 150000)) / 2
    salary = round(mid / 1000) * 1000

    if step == 1:
        return {"action_type": "consult_team_lead",
                "message": "What skills are essential for this role?",
                "salary_offer": None, "benefits": None, "skill_to_verify": None}
    elif step == 2:
        return {"action_type": "probe_candidate",
                "message": "Do you have other offers? What salary are you expecting?",
                "salary_offer": None, "benefits": None, "skill_to_verify": None}
    elif step == 3:
        return {"action_type": "check_budget",
                "salary_offer": salary,
                "message": "This hire will improve team velocity and ROI.",
                "benefits": None, "skill_to_verify": None}
    else:
        return {"action_type": "make_offer",
                "salary_offer": salary,
                "message": "We'd love to have you join the team.",
                "benefits": {"remote_days": 3, "signing_bonus": 10000,
                             "equity_percent": 0.05, "pto_days": 25},
                "skill_to_verify": None}


def get_action(client: OpenAI, obs: dict, step: int, history: List[str]) -> dict:
    # Force offer after step 6 no matter what
    if step > 6:
        return force_action(obs, 4)

    # If we have all info and LLM keeps stalling, force offer
    has_probed   = any("probe_candidate" in h or "consult_team_lead" in h for h in history)
    has_budget   = any("check_budget" in h for h in history)
    ready        = has_probed and has_budget

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, step, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json")
        parsed = json.loads(text.strip())

        action_type = parsed.get("action_type", "")

        # LLM still probing when it already has all info
        if ready and step >= 4 and action_type in ["probe_candidate", "consult_team_lead"]:
            return force_action(obs, 4)

        return {
            "action_type":   action_type,
            "salary_offer":  parsed.get("salary_offer"),
            "message":       parsed.get("message"),
            "benefits":      parsed.get("benefits"),
            "skill_to_verify": parsed.get("skill_to_verify"),
        }
    except Exception:
        return force_action(obs, step)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env_reset(task_id)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.get("candidate_status") in ("hired", "rejected", "accepted_competitor"):
                break

            action = get_action(client, obs, step, history)
            action_type = action.get("action_type", "probe_candidate")

            try:
                result  = env_step(action)
                obs     = result["observation"]
                reward  = float(result.get("reward", 0.0))
                done    = bool(result.get("done", False))
                error   = obs.get("last_action_error")
            except Exception as e:
                reward, done, error = 0.0, True, str(e)

            rewards.append(reward)
            steps_taken = step
            history.append(f"[{action_type}] r={reward:.2f}")

            log_step(
                step=step,
                action=f"{action_type}(salary={action.get('salary_offer')})",
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score   = env_score()
        score   = min(1.0, max(0.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task failed: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []

    for task_id in TASKS_TO_RUN:
        try:
            score = run_task(client, task_id)
            all_scores.append(score)
        except Exception as e:
            print(f"[DEBUG] {task_id} failed: {e}", flush=True)
            all_scores.append(0.0)
            log_end(False, 0, 0.0, [])

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f} "
          f"scores={','.join(f'{s:.3f}' for s in all_scores)}", flush=True)


if __name__ == "__main__":
    main()
