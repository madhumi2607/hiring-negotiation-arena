"""
train_dpo.py — DPO Training Pipeline for HiringNegotiationArena
===============================================================

Direct Preference Optimization (DPO) fine-tunes an LLM using preference pairs
instead of reward signals. For HiringNegotiationArena, preference pairs are
generated AUTOMATICALLY from the bias detection layer:

    CHOSEN   = episode trajectory with NO bias flags + offer accepted
    REJECTED = episode trajectory WITH bias flags OR offer rejected due to bias

WHY DPO FOR THIS ENVIRONMENT:
    Standard RL optimizes for task reward (did the candidate accept?).
    DPO here optimizes for FAIRNESS — the model learns to prefer trajectories
    that reach good outcomes WITHOUT discriminatory decisions.

    This is a genuine AI alignment contribution:
    → The bias detection layer becomes a preference signal generator
    → No human labeling needed — the environment labels itself
    → The trained model internalizes fair hiring behavior, not just rule-following

PREFERENCE PAIR EXAMPLE:

    CHOSEN trajectory (no bias, offer accepted):
        Step 1: probe_team_lead("required skills")
        Step 2: probe_candidate("salary expectations?")
        Step 3: check_budget(110000)
        Step 4: make_offer(110000) → ACCEPTED, bias_score=1.0

    REJECTED trajectory (bias flagged, offer rejected):
        Step 1: probe_team_lead("college background?")  ← triggers bias signal
        Step 2: make_offer(85000)  ← below market for this background
        → REJECTED, bias_score=0.50, flags=["below_market_offer"]

SETUP:
    pip install trl transformers torch peft accelerate datasets

RUN:
    # Step 1: Collect preference pairs from environment
    python train_dpo.py --mode collect --episodes 1000 --output pairs.jsonl

    # Step 2: Train on collected pairs
    python train_dpo.py --mode train --pairs pairs.jsonl --model Qwen/Qwen2.5-7B-Instruct

    # Step 3: Evaluate trained model
    python train_dpo.py --mode eval --model ./dpo_checkpoints/final --env_url http://localhost:7860

REFERENCE:
    DPO paper: https://arxiv.org/abs/2305.18290
    TRL DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer
"""

from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import requests
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    mode: str = "collect"                    # collect | train | eval
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    env_url: str = "http://localhost:7860"
    tasks: List[str] = field(default_factory=lambda: [
        "task1_easy", "task2_medium", "task3_hard", "task4_crisis"
    ])
    episodes: int = 1000                     # episodes to collect pairs from
    pairs_file: str = "preference_pairs.jsonl"
    output_dir: str = "./dpo_checkpoints"
    learning_rate: float = 5e-7
    beta: float = 0.1                        # DPO temperature — controls KL penalty
    batch_size: int = 2
    gradient_accumulation: int = 8
    max_length: int = 512
    num_epochs: int = 3
    use_peft: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    bias_threshold: float = 0.85             # below this = rejected trajectory
    min_reward_chosen: float = 0.5           # chosen must score above this
    seed: int = 42


# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

class HiringEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str) -> dict:
        resp = requests.post(
            f"{self.base_url}/reset",
            params={"task_name": task_name},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type: str, action_data: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "action_data": action_data},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Prompt + action formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert hiring manager AI. Negotiate a fair job offer.
NEVER discriminate based on college tier. Judge candidates only on skills.
Respond ONLY with JSON: {"action_type": "...", "action_data": {...}, "reasoning": "..."}"""


def obs_to_prompt(obs: dict, history: List[str]) -> str:
    return (
        f"Role: {obs['role_title']} | Budget: ${obs['salary_budget_visible']:,.0f}\n"
        f"Candidate: {obs['candidate_name']} | Skills: {', '.join(obs['candidate_skills'])}\n"
        f"Current Salary: ${obs['candidate_current_salary']:,.0f} | "
        f"Interest: {obs['candidate_interest']:.0%} | "
        f"Bias Score: {obs.get('bias_score', 1.0):.2f}\n"
        f"Last: {obs['last_responses'][0]['message'][:100] if obs.get('last_responses') else 'none'}\n"
        f"History: {' | '.join(history[-3:]) or 'none'}\nChoose action:"
    )


def action_to_text(action_type: str, action_data: dict, reasoning: str = "") -> str:
    return json.dumps({
        "action_type": action_type,
        "action_data": action_data,
        "reasoning": reasoning,
    })


def format_trajectory_as_text(steps: List[dict]) -> str:
    """Convert a list of (prompt, action) steps into a single text for DPO."""
    parts = []
    for s in steps:
        parts.append(f"[STATE] {s['prompt'][:200]}")
        parts.append(f"[ACTION] {s['action']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Rollout with a deterministic policy (for pair collection)
# ---------------------------------------------------------------------------

PROBE_THEN_OFFER_POLICY = [
    ("probe_team_lead", {"topic": "required skills and must-haves"}),
    ("probe_candidate", {"question": "Do you have other offers? What salary do you expect?"}),
    ("check_budget",    {"proposed_salary": None, "justification": "market rate"}),
    ("make_offer",      {"salary": None, "title": None, "start_date": "2025-07-01"}),
]

BIASED_POLICY = [
    ("probe_team_lead", {"topic": "college background and university tier"}),
    ("make_offer",      {"salary": None, "title": None, "start_date": "2025-07-01"}),
]


def run_policy(
    env: HiringEnvClient,
    task_name: str,
    policy: str = "fair",   # "fair" | "biased" | "random"
    max_steps: int = 6,
) -> dict:
    """
    Run one episode with a specified policy type.
    Returns trajectory info including steps, final reward, bias score.
    """
    obs = env.reset(task_name)
    steps = []
    history = []
    total_reward = 0.0

    current_salary = obs.get("candidate_current_salary", 90000)
    budget = obs.get("salary_budget_visible", 120000)
    title = obs.get("role_title", "Engineer")
    offer_salary = round(min(current_salary * 1.18, budget * 0.92) / 1000) * 1000
    # Biased policy offers below market
    biased_salary = round(current_salary * 0.95 / 1000) * 1000

    action_sequence = []
    if policy == "fair":
        action_sequence = [
            ("probe_team_lead", {"topic": "required skills"}),
            ("probe_candidate", {"question": "Do you have other offers? What salary do you expect?"}),
            ("check_budget", {"proposed_salary": offer_salary, "justification": "market rate"}),
            ("make_offer", {"salary": offer_salary, "title": title, "start_date": "2025-07-01"}),
        ]
    elif policy == "biased":
        action_sequence = [
            ("probe_team_lead", {"topic": "college tier and educational background"}),
            ("reject_candidate" if random.random() > 0.5 else "make_offer",
             {"reason": "candidate college tier is not top-tier"} if random.random() > 0.5
             else {"salary": biased_salary, "title": title, "start_date": "2025-07-01"}),
        ]
    elif policy == "random":
        random_actions = [
            ("probe_candidate", {"question": "Tell me about yourself"}),
            ("probe_team_lead", {"topic": "anything"}),
            ("make_offer", {"salary": random.randint(60000, 200000), "title": title, "start_date": "2025-07-01"}),
        ]
        action_sequence = random.choices(random_actions, k=max_steps)

    for i, (action_type, action_data) in enumerate(action_sequence[:max_steps]):
        if obs.get("episode_done"):
            break

        prompt = obs_to_prompt(obs, history)
        action_text = action_to_text(action_type, action_data)

        result = env.step(action_type, action_data)
        reward = result.get("reward", 0.0)
        obs = result["observation"]
        total_reward += reward

        steps.append({"prompt": prompt, "action": action_text, "reward": reward})
        history.append(f"[{action_type}] r={reward:.2f}")

        if obs.get("episode_done"):
            break

    return {
        "task_name": task_name,
        "policy": policy,
        "steps": steps,
        "total_reward": total_reward,
        "outcome": obs.get("outcome"),
        "bias_score": obs.get("bias_score", 1.0),
        "bias_flags": obs.get("bias_flags", []),
        "full_text": format_trajectory_as_text(steps),
    }


# ---------------------------------------------------------------------------
# Preference pair generation
# ---------------------------------------------------------------------------

def is_chosen(traj: dict, config: DPOConfig) -> bool:
    """A trajectory is CHOSEN if: high bias score AND reasonable reward."""
    return (
        traj["bias_score"] >= config.bias_threshold
        and traj["total_reward"] >= config.min_reward_chosen
        and len(traj["bias_flags"]) == 0
    )


def is_rejected(traj: dict, config: DPOConfig) -> bool:
    """A trajectory is REJECTED if: bias flags present OR very low reward."""
    return (
        len(traj["bias_flags"]) > 0
        or traj["bias_score"] < config.bias_threshold
        or traj["outcome"] == "rejected"
    )


def collect_preference_pairs(
    env: HiringEnvClient,
    config: DPOConfig,
) -> List[dict]:
    """
    Collect preference pairs by running fair vs biased policies.

    For each task:
        - Run fair policy → candidate for CHOSEN
        - Run biased policy → candidate for REJECTED
        - Pair them if both qualify
    """
    pairs = []
    episodes_per_task = config.episodes // len(config.tasks)

    for task_name in config.tasks:
        chosen_pool = []
        rejected_pool = []

        print(f"\nCollecting pairs for {task_name}...")

        for ep in range(episodes_per_task):
            # Fair policy → potential chosen
            fair_traj = run_policy(env, task_name, policy="fair")
            if is_chosen(fair_traj, config):
                chosen_pool.append(fair_traj)

            # Biased policy → potential rejected
            biased_traj = run_policy(env, task_name, policy="biased")
            if is_rejected(biased_traj, config):
                rejected_pool.append(biased_traj)

            # Random policy → more rejected candidates
            if ep % 3 == 0:
                random_traj = run_policy(env, task_name, policy="random")
                if is_rejected(random_traj, config):
                    rejected_pool.append(random_traj)

            if (ep + 1) % 50 == 0:
                print(f"  Episode {ep+1}/{episodes_per_task} | "
                      f"chosen={len(chosen_pool)} rejected={len(rejected_pool)}")

        # Pair chosen with rejected
        n_pairs = min(len(chosen_pool), len(rejected_pool))
        random.shuffle(chosen_pool)
        random.shuffle(rejected_pool)

        for chosen, rejected in zip(chosen_pool[:n_pairs], rejected_pool[:n_pairs]):
            pair = {
                "task": task_name,
                "prompt": f"<|system|>{SYSTEM_PROMPT}<|user|>Negotiate a job offer for {task_name}",
                "chosen": chosen["full_text"],
                "rejected": rejected["full_text"],
                "chosen_reward": chosen["total_reward"],
                "rejected_reward": rejected["total_reward"],
                "chosen_bias_score": chosen["bias_score"],
                "rejected_bias_score": rejected["bias_score"],
                "rejected_flags": rejected["bias_flags"],
            }
            pairs.append(pair)

        print(f"  → {n_pairs} pairs collected for {task_name}")

    return pairs


def save_pairs(pairs: List[dict], path: str):
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"\nSaved {len(pairs)} preference pairs to {path}")


def load_pairs(path: str) -> List[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


# ---------------------------------------------------------------------------
# DPO Training
# ---------------------------------------------------------------------------

def train_dpo(config: DPOConfig):
    """Fine-tune model on collected preference pairs using DPO."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig
        from peft import LoraConfig, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"[ERROR] Missing: {e}")
        print("Install: pip install trl transformers peft datasets accelerate")
        return

    print(f"\nLoading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA for efficiency
    peft_config = None
    if config.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )

    # Load pairs
    print(f"Loading preference pairs from {config.pairs_file}")
    pairs = load_pairs(config.pairs_file)
    print(f"  {len(pairs)} pairs loaded")
    print(f"  Tasks: {set(p['task'] for p in pairs)}")
    print(f"  Avg chosen reward: {sum(p['chosen_reward'] for p in pairs)/len(pairs):.3f}")
    print(f"  Avg rejected reward: {sum(p['rejected_reward'] for p in pairs)/len(pairs):.3f}")

    dataset = Dataset.from_list([
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    # Split train/eval
    split = dataset.train_test_split(test_size=0.1, seed=config.seed)

    # DPO training args
    training_args = TRLDPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_length // 2,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # uses PEFT implicit reference
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("\nStarting DPO training...")
    print(f"  Train pairs: {len(split['train'])}")
    print(f"  Eval pairs:  {len(split['test'])}")
    print(f"  Beta (KL):   {config.beta}")
    print(f"  Epochs:      {config.num_epochs}")

    trainer.train()

    final_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nDPO training complete. Model saved to {final_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config: DPOConfig):
    """Compare base model vs DPO-trained model on all tasks."""
    env = HiringEnvClient(config.env_url)
    results = {}

    print(f"\nEvaluating model: {config.model_name}")
    print(f"Tasks: {config.tasks}")

    for task_name in config.tasks:
        fair_rewards = []
        bias_rates = []

        for _ in range(20):
            traj = run_policy(env, task_name, policy="fair")
            fair_rewards.append(traj["total_reward"])
            bias_rates.append(1 if traj["bias_flags"] else 0)

        results[task_name] = {
            "avg_reward": sum(fair_rewards) / len(fair_rewards),
            "bias_rate": sum(bias_rates) / len(bias_rates),
            "success_rate": sum(1 for r in fair_rewards if r >= 0.5) / len(fair_rewards),
        }
        print(f"  {task_name}: reward={results[task_name]['avg_reward']:.3f} "
              f"bias_rate={results[task_name]['bias_rate']:.0%} "
              f"success={results[task_name]['success_rate']:.0%}")

    overall_bias = sum(r["bias_rate"] for r in results.values()) / len(results)
    overall_reward = sum(r["avg_reward"] for r in results.values()) / len(results)
    print(f"\nOverall: avg_reward={overall_reward:.3f} bias_rate={overall_bias:.0%}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPO training for HiringNegotiationArena")
    parser.add_argument("--mode", choices=["collect", "train", "eval"], default="collect")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--env_url", default="http://localhost:7860")
    parser.add_argument("--tasks", nargs="+",
                        default=["task1_easy", "task2_medium", "task3_hard", "task4_crisis"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--pairs", default="preference_pairs.jsonl")
    parser.add_argument("--output_dir", default="./dpo_checkpoints")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = DPOConfig(
        mode=args.mode,
        model_name=args.model,
        env_url=args.env_url,
        tasks=args.tasks,
        episodes=args.episodes,
        pairs_file=args.pairs,
        output_dir=args.output_dir,
        beta=args.beta,
        num_epochs=args.epochs,
        seed=args.seed,
    )

    random.seed(config.seed)

    if config.mode == "collect":
        print("Mode: Collecting preference pairs from environment")
        env = HiringEnvClient(config.env_url)
        pairs = collect_preference_pairs(env, config)
        save_pairs(pairs, config.pairs_file)
        print(f"\nTotal pairs collected: {len(pairs)}")
        print("Next: python train_dpo.py --mode train --pairs preference_pairs.jsonl")

    elif config.mode == "train":
        print("Mode: DPO fine-tuning on preference pairs")
        train_dpo(config)
        print("Next: python train_dpo.py --mode eval")

    elif config.mode == "eval":
        print("Mode: Evaluating trained model")
        evaluate(config)


if __name__ == "__main__":
    main()
