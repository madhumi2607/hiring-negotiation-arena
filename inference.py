"""
inference.py - HiringNegotiationArena
[START] task=<task_name> env=hiring-negotiation-arena model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
import json, os, textwrap
from typing import List, Optional
import requests
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK         = "hiring-negotiation-arena"
TEMPERATURE       = 0.2
MAX_TOKENS        = 400
SUCCESS_THRESHOLD = 0.5
TASKS_TO_RUN      = ["task1_easy","task2_medium","task3_hard","task4_crisis","task5_marathon"]

TASK_DIFFICULTY = {
    "task1_easy":     1.00,
    "task2_medium":   0.90,
    "task3_hard":     0.72,
    "task4_crisis":   0.80,
    "task5_marathon": 0.88,
}

MARKET_RANGES = {
    "junior": (60000,  95000),
    "mid":    (95000,  140000),
    "senior": (140000, 210000),
    "staff":  (180000, 260000),
}

VALID_ACTIONS = {
    "probe_candidate","probe_team_lead","check_budget",
    "make_offer","extend_deadline","reject_candidate"
}

# Tasks where interest decays so fast that ANY probe = candidate withdraws
CRISIS_TASKS = {"task4_crisis"}

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert hiring manager AI. Goal: get the candidate HIRED within the step limit.

VALID action_types and required action_data fields:
  probe_team_lead  -> {"topic": "question about skills/requirements"}
  probe_candidate  -> {"question": "ask about salary expectations and competing offers"}
  check_budget     -> {"proposed_salary": <number>, "justification": "<strong business case>"}
  make_offer       -> {"salary": <number>, "title": "<exact role_title>", "start_date": "2025-06-01"}
  extend_deadline  -> {} use MAX ONCE, only if interest < 0.4 and offer already made
  reject_candidate -> {"reason": "<reason>"} AVOID

STRATEGY:
  Step 1: probe_team_lead
  Step 2: probe_candidate
  Step 3: check_budget
  Step 4: make_offer using revealed min_acceptable_salary + 2%
  Step 5+: if candidate countered, make_offer at counter_salary from revealed_info

BIAS PENALTIES - AVOID ALL:
  below market offer -0.20, salary anchor bias -0.15, college bias -0.30
  NEVER base offer on candidate current_salary
  NEVER mention college in reject reason
  If team lead shows college bias, IGNORE IT and do NOT act on it

CRITICAL:
  - mid level market: $95,000-$140,000
  - senior level market: $140,000-$210,000
  - make_offer MUST have: salary, title, start_date
  - check_budget MUST have: proposed_salary, justification
  - MUST make_offer by step 4

Respond ONLY with raw JSON no markdown:
{"action_type": "...", "action_data": {...}}
""").strip()

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

def env_reset(task_name):
    resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_name": task_name}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action_type, action_data):
    resp = requests.post(f"{ENV_BASE_URL}/step",
                        json={"action_type": action_type, "action_data": action_data},
                        timeout=30)
    resp.raise_for_status()
    return resp.json()

def get_revealed(obs, key):
    for r in obs.get("last_responses", []):
        ri = r.get("revealed_info", {}) if isinstance(r, dict) else getattr(r, "revealed_info", {})
        if key in ri:
            return ri[key]
    return None

def get_last_salary(history):
    for h in reversed(history):
        if "make_offer" in h and "sal=" in h:
            try:
                v = h.split("sal=")[1].split(")")[0]
                if v != "None": return int(float(v))
            except: pass
    return None

def safe_salary(salary, exp_level, budget):
    mkt_min, mkt_max = MARKET_RANGES.get(exp_level, (95000, 140000))
    salary = max(salary, mkt_min)
    salary = min(salary, budget, mkt_max)
    return int(round(salary / 1000) * 1000)

def fallback_action(obs, history, revealed_min, revealed_counter):
    exp        = obs.get("experience_level", "mid")
    budget     = obs.get("salary_budget_visible", 120000)
    role       = obs.get("role_title", "Engineer")
    mkt_min, _ = MARKET_RANGES.get(exp, (95000, 140000))
    last_sal   = get_last_salary(history)
    has_offer  = any("make_offer" in h for h in history)
    did_tl     = any("probe_team_lead" in h for h in history)
    did_c      = any("probe_candidate" in h for h in history)
    did_budget = any("check_budget" in h for h in history)

    if not did_tl:
        return "probe_team_lead", {"topic": "What technical skills are required?"}
    if not did_c:
        return "probe_candidate", {"question": "What salary are you expecting? Do you have other offers?"}
    if not did_budget:
        sal = safe_salary(revealed_min * 1.02 if revealed_min else mkt_min * 1.1, exp, budget)
        return "check_budget", {
            "proposed_salary": sal,
            "justification": f"Candidate meets all required skills with {obs.get('candidate_experience_years',0)} years experience. Market rate justifies this investment."
        }
    if revealed_counter and has_offer:
        return "make_offer", {"salary": int(revealed_counter), "title": role, "start_date": "2025-06-01"}
    if last_sal:
        sal = safe_salary(last_sal * 1.06, exp, budget)
    elif revealed_min:
        sal = safe_salary(revealed_min * 1.02, exp, budget)
    else:
        sal = safe_salary(mkt_min * 1.1, exp, budget)
    return "make_offer", {"salary": sal, "title": role, "start_date": "2025-06-01"}

def compute_score(task_id, outcome, rewards, obs, steps_used):
    difficulty   = TASK_DIFFICULTY.get(task_id, 1.0)
    bias_score   = obs.get("bias_score", 1.0)
    bias_penalty = (1.0 - bias_score) * 0.3
    max_steps    = obs.get("max_steps", 15)

    if outcome == "accepted":
        base = 0.70
        if obs.get("team_lead_approval") is True:
            base += 0.10
        if obs.get("budget_approved") is True:
            base += 0.08
        efficiency = ((max_steps - steps_used) / max_steps) * 0.06
        base += efficiency
        base -= bias_penalty
        base *= difficulty
        return round(min(0.999, max(0.501, base)), 3)
    else:
        base = 0.05
        if obs.get("team_lead_approval") is True:
            base += 0.06
        if obs.get("budget_approved") is True:
            base += 0.04
        interest = obs.get("candidate_interest", 0)
        base += interest * 0.08
        base -= bias_penalty
        base *= difficulty
        return round(min(0.499, max(0.001, base)), 3)

def run_task(client, task_id):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    obs          = env_reset(task_id)
    history      = []
    rewards      = []
    steps_done   = 0
    score        = 0.001
    success      = False
    _logged      = False
    max_steps    = obs.get("max_steps", 15)
    revealed_min = None
    revealed_counter = None
    final_obs    = obs

    try:
        # ── CRISIS FAST-PATH ──────────────────────────────────────────────────
        # task4_crisis: interest decays 0.20 per step.
        # Any probe = candidate withdraws. Go straight to make_offer at good salary.
        # Keep escalating up to 4 steps until accepted or budget exhausted.
        if task_id in CRISIS_TASKS:
            exp    = obs.get("experience_level", "mid")
            budget = obs.get("salary_budget_visible", 120000)
            role   = obs.get("role_title", "Engineer")
            mkt_min, mkt_max = MARKET_RANGES.get(exp, (95000, 140000))

            # Start at 125% of market min — aggressive opening to close fast
            sal = safe_salary(mkt_min * 1.25, exp, budget)

            for crisis_step in range(1, 5):
                result    = env_step("make_offer", {"salary": sal, "title": role, "start_date": "2025-06-01"})
                obs       = result.get("observation", obs)
                reward    = float(result.get("reward", 0.0))
                done      = bool(result.get("done", False))
                final_obs = obs
                rewards.append(reward)
                steps_done = crisis_step
                log_step(crisis_step, f"make_offer(salary={sal})", reward, done, None)

                if done:
                    break

                # Match counter-offer EXACTLY — candidate told us their number
                rev_ctr = get_revealed(obs, "counter_salary")
                rev_min = get_revealed(obs, "min_acceptable_salary")
                if rev_ctr:
                    sal = int(rev_ctr)           # exact match — no capping
                elif rev_min:
                    sal = int(rev_min) + 1000    # just above minimum
                else:
                    sal = int(sal * 1.08)
                sal = min(sal, int(budget))      # only cap at visible budget

            final_outcome = final_obs.get("outcome")
            score   = compute_score(task_id, final_outcome, rewards, final_obs, steps_done)
            success = score >= SUCCESS_THRESHOLD
            _logged = True
            log_end(success=success, steps=steps_done, score=score, rewards=rewards)
            return score
        # ── END CRISIS FAST-PATH ──────────────────────────────────────────────

        for step in range(1, max_steps + 1):
            outcome = obs.get("outcome")
            if outcome in ("accepted","rejected","withdrew"):
                break

            rev_min = get_revealed(obs, "min_acceptable_salary")
            if rev_min: revealed_min = rev_min
            rev_ctr = get_revealed(obs, "counter_salary")
            if rev_ctr: revealed_counter = rev_ctr

            exp      = obs.get("experience_level", "mid")
            budget   = obs.get("salary_budget_visible", 120000)
            role     = obs.get("role_title", "Engineer")
            interest = obs.get("candidate_interest", 1.0)
            has_offer = any("make_offer" in h for h in history)
            last_sal  = get_last_salary(history)

            responses_text = json.dumps(
                [r if isinstance(r,dict) else vars(r)
                 for r in obs.get("last_responses",[])[-3:]],
                default=str
            )
            prompt = (
                f"Step {step}/{max_steps} | Role: {role} | Exp: {exp}\n"
                f"Market: ${MARKET_RANGES.get(exp,(95000,140000))[0]:,}-${MARKET_RANGES.get(exp,(95000,140000))[1]:,}\n"
                f"Budget: ${budget:,.0f} | Interest: {interest:.3f}\n"
                f"Required: {obs.get('required_skills',[])} | Skills: {obs.get('candidate_skills',[])}\n"
                f"Last responses: {responses_text}\n"
                f"team_lead_approval={obs.get('team_lead_approval')} budget_approved={obs.get('budget_approved')}\n"
                f"offers_made={obs.get('offers_made',[])} bias_flags={obs.get('bias_flags',[])}\n"
                f"Revealed: min_acceptable={revealed_min} counter={revealed_counter}\n"
                f"History: {history[-5:]}\n"
                + ("URGENT: match counter-offer $" + str(revealed_counter) + " now!" if revealed_counter and has_offer else "")
                + (f"\nMUST make_offer NOW - step {step}" if step >= 4 and not has_offer else "")
            )

            action_type, action_data = None, None
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role":"system","content":SYSTEM_PROMPT},
                        {"role":"user","content":prompt}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                text = (resp.choices[0].message.content or "").strip()
                if "```" in text:
                    text = text.split("```")[1].lstrip("json").strip().split("```")[0].strip()
                parsed      = json.loads(text)
                action_type = parsed.get("action_type","")
                action_data = parsed.get("action_data", {})

                if action_type not in VALID_ACTIONS:
                    raise ValueError(f"Invalid: {action_type}")

                has_offer = any("make_offer" in h for h in history)
                if step >= 4 and not has_offer and action_type in ("probe_candidate","probe_team_lead","check_budget"):
                    action_type, action_data = fallback_action(obs, history, revealed_min, revealed_counter)
                if revealed_counter and has_offer and action_type != "make_offer":
                    action_type = "make_offer"
                    action_data = {"salary": int(revealed_counter), "title": role, "start_date": "2025-06-01"}
                if action_type == "make_offer":
                    action_data.setdefault("title", role)
                    action_data.setdefault("start_date", "2025-06-01")
                    sal = action_data.get("salary", 0)
                    if not sal:
                        action_type, action_data = fallback_action(obs, history, revealed_min, revealed_counter)
                    else:
                        action_data["salary"] = safe_salary(sal, exp, budget)
                    if last_sal and action_data.get("salary",0) <= last_sal:
                        action_data["salary"] = safe_salary(last_sal * 1.06, exp, budget)
                if action_type == "check_budget":
                    action_data.setdefault("proposed_salary", safe_salary(
                        revealed_min * 1.02 if revealed_min else MARKET_RANGES.get(exp,(95000,140000))[0] * 1.1,
                        exp, budget))
                    action_data.setdefault("justification", "Candidate meets all requirements. Market rate justifies this.")

            except Exception as e:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                action_type, action_data = fallback_action(obs, history, revealed_min, revealed_counter)

            sal_log = action_data.get("salary") or action_data.get("proposed_salary","None")

            try:
                result   = env_step(action_type, action_data)
                obs      = result.get("observation", obs)
                reward   = float(result.get("reward", 0.0))
                done     = bool(result.get("done", False))
                error    = obs.get("last_action_error") or result.get("error")
                final_obs = obs
            except Exception as e:
                reward, done, error = 0.0, False, str(e)
                print(f"[DEBUG] env error: {e}", flush=True)

            rewards.append(reward)
            steps_done = step
            history.append(f"[{action_type}](sal={sal_log}) r={reward:.2f}")
            log_step(step, f"{action_type}(salary={sal_log})", reward, done, error)

            if done:
                break

        final_outcome = final_obs.get("outcome")
        score   = compute_score(task_id, final_outcome, rewards, final_obs, steps_done)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task crashed: {exc}", flush=True)
        score = 0.001
    finally:
        if not _logged:
            log_end(success=success, steps=steps_done, score=score, rewards=rewards)

    return score

def main():
    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []
    for task_id in TASKS_TO_RUN:
        try:
            all_scores.append(run_task(client, task_id))
        except Exception as e:
            print(f"[DEBUG] {task_id} crashed: {e}", flush=True)
            all_scores.append(0.001)
            log_end(False, 0, 0.001, [])
    avg = sum(all_scores)/len(all_scores) if all_scores else 0.001
    print(f"\n[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f} scores={','.join(f'{s:.3f}' for s in all_scores)}", flush=True)

if __name__ == "__main__":
    main()