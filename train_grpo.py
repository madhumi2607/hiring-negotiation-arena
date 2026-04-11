"""
train_grpo.py — GRPO Training Pipeline for HiringNegotiationArena
=================================================================

Group Relative Policy Optimization (GRPO) fine-tunes an LLM to become
a better hiring manager by learning from environment rewards.

HOW IT WORKS:
    1. For each training step, the agent generates G candidate action sequences
       (a "group") for the same negotiation state
    2. Each sequence is evaluated by the environment — reward comes from
       actual negotiation outcomes (offer accepted, bias score, role fit)
    3. GRPO normalizes rewards within the group and uses them as advantages
       to update the policy — no separate value network needed
    4. Over thousands of episodes, the agent learns:
         - When to probe vs when to offer
         - How to detect and avoid bias traps
         - How to read hidden competing offers from partial signals

WHY GRPO FOR THIS ENVIRONMENT:
    - Reward is sparse but meaningful (final negotiation score)
    - Partial rewards (probe rewards) provide dense signal throughout
    - The bias penalty creates a natural alignment signal
    - Hidden state requires exploration — GRPO's group sampling helps

SETUP:
    pip install trl transformers torch accelerate peft

RUN:
    python train_grpo.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --env_url http://localhost:7860 \
        --tasks task1_easy task2_medium task3_hard \
        --episodes 500 \
        --group_size 8 \
        --output_dir ./grpo_checkpoints

REFERENCE:
    DeepSeekMath GRPO paper: https://arxiv.org/abs/2402.03300
    TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer
"""

from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import requests
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    env_url: str = "http://localhost:7860"
    tasks: List[str] = field(default_factory=lambda: ["task1_easy", "task2_medium", "task3_hard"])
    episodes_per_task: int = 200
    group_size: int = 8           # G — number of rollouts per state for GRPO
    max_steps: int = 8            # max steps per episode
    learning_rate: float = 1e-6
    kl_coeff: float = 0.01        # KL penalty vs reference model
    output_dir: str = "./grpo_checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_peft: bool = True         # LoRA for memory efficiency
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size: int = 4
    gradient_accumulation: int = 4
    save_every: int = 50          # save checkpoint every N episodes
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

    def state(self) -> dict:
        resp = requests.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert hiring manager AI negotiating a job offer.
Your goal: get the candidate to accept an offer that satisfies the team lead and stays within budget.
Avoid any bias based on college tier.

Respond with ONLY a JSON object:
{
  "action_type": "<probe_candidate|probe_team_lead|check_budget|make_offer>",
  "action_data": {<action-specific fields>},
  "reasoning": "<one sentence>"
}"""


def build_prompt(obs: dict, history: List[str]) -> str:
    return f"""
Role: {obs['role_title']} | Budget: ${obs['salary_budget_visible']:,.0f}
Candidate: {obs['candidate_name']} | Skills: {', '.join(obs['candidate_skills'])}
Current Salary: ${obs['candidate_current_salary']:,.0f} | Interest: {obs['candidate_interest']:.0%}
Bias Score: {obs.get('bias_score', 1.0):.2f}

Last responses:
{chr(10).join(f"  [{r['party']}]: {r['message']}" for r in obs.get('last_responses', [])) or '  (none)'}

History: {' | '.join(history[-3:]) or 'none'}

Choose action:""".strip()


# ---------------------------------------------------------------------------
# Rollout — one episode trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One complete episode trajectory for GRPO training."""
    task_name: str
    prompts: List[str]
    responses: List[str]
    rewards: List[float]
    total_reward: float
    outcome: Optional[str]
    bias_flags: List[str]


def rollout(
    env: HiringEnvClient,
    model,
    tokenizer,
    task_name: str,
    max_steps: int = 8,
    temperature: float = 0.8,
) -> Trajectory:
    """Run one episode and collect (prompt, response, reward) triples."""
    obs = env.reset(task_name)
    prompts, responses, rewards = [], [], []
    history = []

    for step in range(1, max_steps + 1):
        if obs.get("episode_done"):
            break

        prompt = build_prompt(obs, history)
        full_prompt = f"<|system|>{SYSTEM_PROMPT}<|user|>{prompt}<|assistant|>"

        # Generate action from model
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse action
        try:
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1].lstrip("json")
            parsed = json.loads(response_text)
            action_type = parsed.get("action_type", "probe_candidate")
            action_data = parsed.get("action_data", {})
        except Exception:
            action_type = "probe_candidate"
            action_data = {"question": "What salary are you expecting?"}

        # Step environment
        result = env.step(action_type, action_data)
        reward = result.get("reward", 0.0)
        obs = result["observation"]

        prompts.append(full_prompt)
        responses.append(response_text)
        rewards.append(reward)
        history.append(f"[{action_type}] r={reward:.2f}")

    total = sum(rewards)
    return Trajectory(
        task_name=task_name,
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        total_reward=total,
        outcome=obs.get("outcome"),
        bias_flags=obs.get("bias_flags", []),
    )


# ---------------------------------------------------------------------------
# GRPO advantage computation
# ---------------------------------------------------------------------------

def compute_grpo_advantages(group_rewards: List[float]) -> List[float]:
    """
    GRPO advantage = (reward - mean) / (std + eps)
    Normalizes within the group so relative performance drives learning.
    """
    import statistics
    if len(group_rewards) < 2:
        return [0.0] * len(group_rewards)
    mean = statistics.mean(group_rewards)
    std = statistics.stdev(group_rewards) + 1e-8
    return [(r - mean) / std for r in group_rewards]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: GRPOConfig):
    print(f"GRPO Training — HiringNegotiationArena")
    print(f"Model: {config.model_name}")
    print(f"Tasks: {config.tasks}")
    print(f"Device: {config.device}")
    print(f"Group size G={config.group_size}")
    print("-" * 60)

    # Load model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if config.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("Install: pip install transformers peft accelerate trl")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate
    )

    env = HiringEnvClient(config.env_url)
    os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0
    episode_rewards = []

    for episode in range(config.episodes_per_task * len(config.tasks)):
        task_name = config.tasks[episode % len(config.tasks)]

        # ── GRPO: collect G rollouts for same task ──
        group: List[Trajectory] = []
        for _ in range(config.group_size):
            traj = rollout(env, model, tokenizer, task_name, config.max_steps)
            group.append(traj)

        group_total_rewards = [t.total_reward for t in group]
        advantages = compute_grpo_advantages(group_total_rewards)

        # ── Policy gradient update ──
        total_loss = torch.tensor(0.0, requires_grad=True)
        for traj, advantage in zip(group, advantages):
            if advantage <= 0:
                continue   # only learn from above-average trajectories
            for prompt, response in zip(traj.prompts, traj.responses):
                full = prompt + response
                inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=512).to(config.device)
                labels = inputs["input_ids"].clone()
                # Mask prompt tokens — only train on response
                prompt_len = len(tokenizer(prompt)["input_ids"])
                labels[0, :prompt_len] = -100

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss * advantage
                total_loss = total_loss + loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── Logging ──
        avg_reward = sum(group_total_rewards) / len(group_total_rewards)
        best_reward = max(group_total_rewards)
        bias_episodes = sum(1 for t in group if t.bias_flags)
        episode_rewards.append(avg_reward)
        global_step += 1

        print(
            f"[Episode {episode+1:4d}] task={task_name:<16} "
            f"avg_reward={avg_reward:.3f} best={best_reward:.3f} "
            f"bias_episodes={bias_episodes}/{config.group_size} "
            f"loss={total_loss.item():.4f}"
        )

        # ── Checkpoint ──
        if (episode + 1) % config.save_every == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint_{episode+1}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  → Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Average reward over training: {sum(episode_rewards)/len(episode_rewards):.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> GRPOConfig:
    parser = argparse.ArgumentParser(description="GRPO training for HiringNegotiationArena")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--env_url", default="http://localhost:7860")
    parser.add_argument("--tasks", nargs="+", default=["task1_easy", "task2_medium", "task3_hard"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--output_dir", default="./grpo_checkpoints")
    parser.add_argument("--no_peft", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return GRPOConfig(
        model_name=args.model,
        env_url=args.env_url,
        tasks=args.tasks,
        episodes_per_task=args.episodes,
        group_size=args.group_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        use_peft=not args.no_peft,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train(config)
