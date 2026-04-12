"""
train_dpo.py — DPO Training Pipeline for HiringNegotiationArena
Confirmed API schema:
  POST /reset → params={"task_name": "task1_easy"}   (Query param)
  POST /step  → json={"action_type": "...", "action_data": {...}}

Fixed for trl>=0.11: uses DPOConfig + processing_class instead of
TrainingArguments + tokenizer + beta as top-level kwarg.
"""

import argparse, json, os, torch
from dataclasses import dataclass
from typing import List


@dataclass
class DPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    pairs_file: str = "preference_pairs.jsonl"
    output_dir: str = "./dpo_checkpoints"
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16


def train_dpo(config):
    import trl
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer
    from peft import LoraConfig, get_peft_model, TaskType

    print("=" * 50)
    print("DPO Training - HiringNegotiationArena")
    print(f"  Model: {config.model_name}")
    print(f"  Pairs: {config.pairs_file}")
    print(f"  trl version: {trl.__version__}")
    print("=" * 50)

    # Load pairs
    rows = []
    with open(config.pairs_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({
                "prompt":   d["prompt"],
                "chosen":   d["chosen"],
                "rejected": d["rejected"],
            })
    print(f"Loaded {len(rows)} preference pairs")
    if not rows:
        print("No pairs found — exiting")
        return

    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=max(1, int(len(dataset) * 0.1)), seed=42)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[INFO] Loading model in bfloat16")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    os.makedirs(config.output_dir, exist_ok=True)

    # trl version-aware API
    trl_ver = tuple(int(x) for x in trl.__version__.split(".")[:2])

    if trl_ver >= (0, 11):
        # trl >= 0.11: beta + max_length go inside DPOConfig, use processing_class
        from trl import DPOConfig as TRLDPOConfig
        training_args = TRLDPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=config.learning_rate,
            beta=config.beta,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            remove_unused_columns=False,
            logging_steps=5,
            save_steps=50,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_available(),
            report_to="none",
            dataloader_num_workers=0,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            processing_class=tokenizer,   # trl >= 0.11 uses this instead of tokenizer=
            peft_config=lora_cfg,
        )
    else:
        # trl < 0.11: old API
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=config.learning_rate,
            remove_unused_columns=False,
            logging_steps=5,
            save_steps=50,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_available(),
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            beta=config.beta,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
        )

    print("Starting DPO training...")
    trainer.train()

    final_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Done. Saved to {final_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       default="train")
    p.add_argument("--model",      default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--pairs",      default="preference_pairs.jsonl")
    p.add_argument("--output_dir", default="./dpo_checkpoints")
    p.add_argument("--epochs",     type=int,   default=1)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--beta",       type=float, default=0.1)
    args = p.parse_args()
    return DPOConfig(
        model_name=args.model, pairs_file=args.pairs,
        output_dir=args.output_dir, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        beta=args.beta,
    )


if __name__ == "__main__":
    config = parse_args()
    if config.pairs_file and os.path.exists(config.pairs_file):
        train_dpo(config)
    else:
        print(f"Pairs file not found: {config.pairs_file}")
