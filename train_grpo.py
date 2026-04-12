"""
train_grpo.py — GRPO Training Pipeline for HiringNegotiationArena
Confirmed API schema:
  POST /reset → params={"task_name": "task1_easy"}   (Query param)
  POST /step  → json={"action_type": "...", "action_data": {...}}
"""

from __future__ import annotations
import argparse, json, os, random
from dataclasses import dataclass, field
from typing import List, Optional
import requests, torch


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    env_url: str = "http://localhost:7860"
    tasks: List[str] = field(default_factory=lambda: ["task1_easy", "task2_medium", "task3_hard"])
    episodes_per_task: int = 30
    group_size: int = 2
    max_steps: int = 8
    learning_rate: float = 1e-6
    output_dir: str = "./grpo_checkpoints_full"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_peft: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    save_every: int = 10
    seed: int = 42


class HiringEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except:
            return False

    def reset(self, task_name: str) -> dict:
        r = requests.post(
            f"{self.base_url}/reset",
            params={"task_name": task_name},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, action_data: dict) -> dict:
        r = requests.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "action_data": action_data},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


SYSTEM_PROMPT = (
    "You are an expert hiring manager AI negotiating a job offer.\n"
    "NEVER discriminate based on college tier.\n"
    "Respond ONLY with valid JSON, one of:\n"
    '  {"action_type": "probe_candidate",  "action_data": {"question": "..."}}\n'
    '  {"action_type": "probe_team_lead",  "action_data": {"topic": "skills"}}\n'
    '  {"action_type": "check_budget",     "action_data": {"proposed_salary": 95000}}\n'
    '  {"action_type": "make_offer",       "action_data": {"salary": 95000, "title": "Engineer", "start_date": "2025-09-01"}}\n'
)


def build_prompt(obs: dict, history: List[str]) -> str:
    last = obs.get("last_responses", [])
    last_msg = last[0]["message"][:120] if last else "(none)"
    return (
        f"Role: {obs.get('role_title','?')} | Budget: ${obs.get('salary_budget_visible',0):,.0f}\n"
        f"Required: {', '.join(obs.get('required_skills',[]))}\n"
        f"Candidate: {obs.get('candidate_name','?')} | "
        f"Skills: {', '.join(obs.get('candidate_skills',[]))}\n"
        f"Exp: {obs.get('candidate_experience_years',0)} yrs | "
        f"Cur salary: ${obs.get('candidate_current_salary',0):,.0f} | "
        f"Interest: {obs.get('candidate_interest',1.0):.0%}\n"
        f"Says: \"{last_msg}\"\n"
        f"Step {obs.get('step',0)}/{obs.get('max_steps',15)} | "
        f"History: {' | '.join(history[-3:]) or 'none'}\nChoose action:"
    ).strip()


@dataclass
class Trajectory:
    task_name: str
    prompts: List[str]
    responses: List[str]
    rewards: List[float]
    total_reward: float
    outcome: Optional[str]
    bias_flags: List[str]


def rollout(env, model, tokenizer, task_name, max_steps=8, temperature=0.9) -> Trajectory:
    try:
        obs = env.reset(task_name)
    except Exception as e:
        print(f"    [env.reset error] {e}")
        return Trajectory(task_name, [], [], [0.0], 0.0, None, [])

    prompts, responses, rewards, history = [], [], [], []

    for _ in range(max_steps):
        if obs.get("episode_done") or obs.get("done"):
            break

        prompt = build_prompt(obs, history)
        full_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(full_prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=100, temperature=temperature,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        action_type = "probe_candidate"
        action_data = {"question": "What are your salary expectations?"}
        try:
            text = response_text
            if text.startswith("```"):
                text = text.split("```")[1].lstrip("json").strip()
            parsed = json.loads(text)
            action_type = parsed.get("action_type", "probe_candidate")
            action_data = parsed.get("action_data", {"question": "Tell me more."})
        except:
            pass

        reward = 0.0
        try:
            result = env.step(action_type, action_data)
            reward = result.get("reward", 0.0)
            obs = result.get("observation", obs)
        except Exception as e:
            print(f"    [env.step error] {e}")

        prompts.append(full_prompt)
        responses.append(response_text)
        rewards.append(reward)
        history.append(f"[{action_type}] r={reward:.2f}")

    return Trajectory(
        task_name=task_name, prompts=prompts, responses=responses,
        rewards=rewards, total_reward=float(sum(rewards)),
        outcome=obs.get("outcome"), bias_flags=obs.get("bias_flags", []),
    )


def compute_grpo_advantages(group_rewards: List[float]) -> List[float]:
    if len(group_rewards) < 2:
        return [0.0] * len(group_rewards)
    mean = sum(group_rewards) / len(group_rewards)
    std = (sum((r - mean)**2 for r in group_rewards) / len(group_rewards))**0.5 + 1e-8
    return [(r - mean) / std for r in group_rewards]


def load_model(config: GRPOConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[INFO] Loading model in bfloat16 (no bitsandbytes)")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    if config.use_peft:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=config.lora_rank,
                lora_alpha=config.lora_alpha, target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
            ))
            model.print_trainable_parameters()
        except ImportError:
            print("[WARN] peft not installed")
    return model, tokenizer


def train(config: GRPOConfig):
    print("=" * 60)
    print("GRPO Training — HiringNegotiationArena")
    print(f"  Model:      {config.model_name}")
    print(f"  Tasks:      {config.tasks}")
    print(f"  Episodes:   {config.episodes_per_task * len(config.tasks)}")
    print(f"  Group size: {config.group_size}")
    print(f"  Device:     {config.device}")
    print("=" * 60)

    env = HiringEnvClient(config.env_url)
    if not env.health():
        raise RuntimeError(f"Server not reachable at {config.env_url}")
    print(f"[OK] Environment at {config.env_url} is healthy.")

    model, tokenizer = load_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    os.makedirs(config.output_dir, exist_ok=True)

    best_reward = -float("inf")
    episode_rewards = []
    total_episodes = config.episodes_per_task * len(config.tasks)

    for episode in range(total_episodes):
        task_name = config.tasks[episode % len(config.tasks)]
        print(f"\n[Episode {episode+1}/{total_episodes}] task={task_name}")

        group = []
        for i in range(config.group_size):
            traj = rollout(env, model, tokenizer, task_name, config.max_steps)
            group.append(traj)
            print(f"  Rollout {i+1}: reward={traj.total_reward:.3f} outcome={traj.outcome}")

        group_rewards = [t.total_reward for t in group]
        advantages = compute_grpo_advantages(group_rewards)
        avg_reward = sum(group_rewards) / len(group_rewards)
        print(f"  Scores: {[f'{r:.3f}' for r in group_rewards]} | Avg: {avg_reward:.3f}")
        episode_rewards.append(avg_reward)

        # GRPO update — accumulate into list (FIX: no double advantage application)
        loss_list = []
        model.train()
        for traj, advantage in zip(group, advantages):
            if advantage <= 0 or not traj.prompts:
                continue
            for prompt, response in zip(traj.prompts, traj.responses):
                inputs = tokenizer(prompt + response, return_tensors="pt",
                                   truncation=True, max_length=512).to(config.device)
                labels = inputs["input_ids"].clone()
                plen = tokenizer(prompt, return_tensors="pt",
                                 truncation=True, max_length=512)["input_ids"].shape[1]
                labels[0, :plen] = -100
                # FIX: compute loss once, multiply by advantage once
                loss_list.append(model(**inputs, labels=labels).loss * advantage)

        if loss_list:
            total_loss = torch.stack(loss_list).sum()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f"  Loss: {total_loss.item():.5f}")
        else:
            print("  Loss: 0.00000")
        model.eval()

        if (episode + 1) % config.save_every == 0:
            ckpt = os.path.join(config.output_dir, f"checkpoint_{episode+1}")
            model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
            print(f"  → Saved: {ckpt}")

        # Save best checkpoint
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_path = os.path.join(config.output_dir, "best")
            model.save_pretrained(best_path); tokenizer.save_pretrained(best_path)

    final_path = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_path); tokenizer.save_pretrained(final_path)
    print(f"\n✓ Done. Best: {os.path.join(config.output_dir,'best')} | Final: {final_path}")
    print(f"  Avg reward: {sum(episode_rewards)/len(episode_rewards):.3f}")


def parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--env_url",    default="http://localhost:7860")
    p.add_argument("--tasks",      nargs="+", default=["task1_easy","task2_medium","task3_hard"])
    p.add_argument("--episodes",   type=int,   default=30)
    p.add_argument("--group_size", type=int,   default=2)
    p.add_argument("--max_steps",  type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-6)
    p.add_argument("--output_dir", default="./grpo_checkpoints_full")
    p.add_argument("--no_peft",    action="store_true")
    p.add_argument("--seed",       type=int,   default=42)
    a = p.parse_args()
    return GRPOConfig(
        model_name=a.model, env_url=a.env_url, tasks=a.tasks,
        episodes_per_task=a.episodes, group_size=a.group_size,
        max_steps=a.max_steps, learning_rate=a.lr,
        output_dir=a.output_dir, use_peft=not a.no_peft, seed=a.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    random.seed(config.seed); torch.manual_seed(config.seed)
    train(config)
