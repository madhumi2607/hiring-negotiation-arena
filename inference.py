"""
inference.py — HiringNegotiationArena baseline inference script.

Mandatory stdout format:
    [START] task=<task_name> env=hiring-negotiation-arena model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL   LLM endpoint (default: HF router)
    MODEL_NAME     Model identifier
    HF_TOKEN       API key
    HIRING_TASK    Task name (default: task1_easy)
    ENV_BASE_URL   HiringNegotiationArena server URL (default: http://localhost:7860)
"""
from __future__ import annotations
import json
import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "hiring-negotiation-arena"
MAX_STEPS = 15
TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.4

TASKS_TO_RUN = [
    "task1_easy",
    "task2_medium",
    "task3_hard",
    "task4_crisis",
    "task5_marathon",
]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_safe = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task_name: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_name": task_name}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, action_data: dict) -> dict:
    payload = {"action_type": action_type, "action_data": action_data}
    resp = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert hiring manager AI. Hire the best candidate by negotiating a job offer.

STRICT STEP STRATEGY — FOLLOW EXACTLY:
- Step 1: probe_team_lead with topic "required skills"
- Step 2: probe_candidate with question about competing offers and salary expectations
- Step 3: check_budget with proposed_salary = candidate_current_salary * 1.20
- Step 4+: make_offer with salary = max(candidate_current_salary * 1.20, budget_visible * 0.90)

DO NOT probe more than twice. After step 3, ALWAYS make_offer. Never probe again after step 2.

OFFER FORMULA:
  salary = candidate_current_salary * 1.20
  If salary > salary_budget_visible: salary = salary_budget_visible * 0.92
  Round to nearest 1000.

RULES:
- NEVER reject based on college tier
- NEVER probe more than 2 times total
- ALWAYS make_offer by step 4 at the latest

Respond with ONLY a JSON object:
{
  "action_type": "<probe_candidate|probe_team_lead|check_budget|make_offer>",
  "action_data": {<fields>},
  "reasoning": "<one line>"
}

make_offer fields: {"salary": <number>, "title": "<role_title>", "start_date": "2025-07-01"}
check_budget fields: {"proposed_salary": <number>, "justification": "market rate"}
probe_candidate fields: {"question": "<string>"}
probe_team_lead fields: {"topic": "<string>"}
""").strip()


def build_user_prompt(obs: dict, step: int, history: List[str]) -> str:
    last_responses = obs.get("last_responses", [])
    responses_text = "\n".join(
        f"  [{r['party'].upper()}]: {r['message']}" for r in last_responses
    ) or "  (none yet)"

    history_text = "\n".join(history[-6:]) or "  (none)"

    return textwrap.dedent(f"""
    === CURRENT STATE (Step {step}/{obs['max_steps']}) ===
    Role: {obs['role_title']} ({obs['experience_level']})
    Required Skills: {', '.join(obs['required_skills'])}
    Budget (visible): ${obs['salary_budget_visible']:,.0f}

    Candidate: {obs['candidate_name']}
    Skills: {', '.join(obs['candidate_skills'])}
    Experience: {obs['candidate_experience_years']} years | College Tier: {obs['candidate_college_tier']}
    Current Salary: ${obs['candidate_current_salary']:,.0f}
    Interest Level: {obs['candidate_interest']:.0%}

    Offers Made: {len(obs.get('offers_made', []))}
    Team Lead Approved: {obs.get('team_lead_approval')}
    Budget Approved: {obs.get('budget_approved')}

    Last Responses:
    {responses_text}

    Bias Score: {obs.get('bias_score', 1.0):.2f} (1.0 = no bias, lower = penalized)
    Bias Flags: {obs.get('bias_flags', [])}

    === ACTION HISTORY ===
    {history_text}

    Choose your next action:
    """).strip()


def force_action(obs: dict, step: int) -> dict:
    """Deterministic fallback strategy — guarantees an offer is made."""
    current_salary = obs.get("candidate_current_salary", 90000)
    budget = obs.get("salary_budget_visible", 120000)
    title = obs.get("role_title", "Software Engineer")
    salary = round(min(current_salary * 1.20, budget * 0.92) / 1000) * 1000

    if step == 1:
        return {"action_type": "probe_team_lead", "action_data": {"topic": "required skills"}, "reasoning": "gather requirements"}
    elif step == 2:
        return {"action_type": "probe_candidate", "action_data": {"question": "Do you have other offers? What salary are you expecting?"}, "reasoning": "gauge interest"}
    elif step == 3:
        return {"action_type": "check_budget", "action_data": {"proposed_salary": salary, "justification": "market rate"}, "reasoning": "verify budget"}
    else:
        return {"action_type": "make_offer", "action_data": {"salary": salary, "title": title, "start_date": "2025-07-01"}, "reasoning": "make offer"}


def get_agent_action(client: OpenAI, obs: dict, step: int, history: List[str]) -> dict:
    # Force deterministic offer by step 4
    if step >= 4:
        return force_action(obs, step)

    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        # Override if LLM still probing after step 3
        if step >= 3 and parsed.get("action_type") in ["probe_candidate", "probe_team_lead"]:
            return force_action(obs, step)
        return parsed
    except Exception as e:
        return force_action(obs, step)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    obs = env_reset(task_name)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.get("episode_done"):
                break

            parsed = get_agent_action(client, obs, step, history)
            action_type = parsed.get("action_type", "probe_candidate")
            action_data = parsed.get("action_data", {})
            reasoning = parsed.get("reasoning", "")

            result = env_step(action_type, action_data)
            obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = obs.get("last_action_error")

            rewards.append(reward)
            steps_taken = step
            history.append(f"Step {step}: [{action_type}] {json.dumps(action_data)[:80]} -> reward={reward:.2f}")

            log_step(
                step=step,
                action=f"{action_type}({json.dumps(action_data)[:60]})",
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Score = cumulative reward clamped to [0, 1]
        score = min(0.999, max(0.001, sum(rewards)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Run failed: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []

    for task_name in TASKS_TO_RUN:
        try:
            score = run_task(client, task_name)
            all_scores.append(score)
        except Exception as e:
            print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)
            all_scores.append(0.0)
            log_end(success=False, steps=0, score=0.0, rewards=[])

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f} scores={','.join(f'{s:.3f}' for s in all_scores)}", flush=True)


if __name__ == "__main__":
    main()

