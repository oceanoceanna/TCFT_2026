#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA SFT for QA critique generation from leakage prompt + critique data."
    )
    parser.add_argument("--model-id", required=True, help="Base model ID or local path.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_DATA_DIR / "valid.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--response-prefix", default="\n\nCritique:\n")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-dir", type=Path, default=None)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--progress-bar-width", type=int, default=30)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_input(record: dict[str, Any]) -> str:
    return (
        f"Event: {record['event']}\n"
        f"True event date: {record['true_date']}\n"
        f"Cutoff date: {record['ask_date']}\n\n"
        f"Original prompt:\n{record['leakage_prompt']}\n\n"
        f"Model answer:\n{record['leakage_answer_raw']}"
    )


def tokenize_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    max_length: int,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]


class CritiqueDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        response_prefix: str,
        use_chat_template: bool,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        eos = tokenizer.eos_token or ""

        for row in rows:
            prompt = f"{format_input(row).rstrip()}{response_prefix}"
            label = str(row["critique"]).strip()

            if use_chat_template:
                prompt_ids = tokenize_chat_messages(
                    tokenizer,
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    max_length=max_length,
                )
                full_ids = tokenize_chat_messages(
                    tokenizer,
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": label},
                    ],
                    add_generation_prompt=False,
                    max_length=max_length,
                )
            else:
                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
                full_ids = tokenizer(
                    prompt + label + eos,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]

            if len(full_ids) <= len(prompt_ids):
                continue

            labels = full_ids.copy()
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100

            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

        if not self.examples:
            raise ValueError("No usable examples after tokenization/truncation.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


@dataclass
class Collator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class StepProgressCallback(TrainerCallback):
    def __init__(self, width: int = 30) -> None:
        self.width = max(width, 10)

    def _format_bar(self, step: int, total: int) -> str:
        if total <= 0:
            return "[?]"
        ratio = min(max(step / total, 0.0), 1.0)
        filled = int(round(ratio * self.width))
        return "[" + "#" * filled + "." * (self.width - filled) + f"] {ratio * 100:5.1f}%"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        logs = logs or {}
        step = int(state.global_step)
        total = int(state.max_steps)
        parts = [f"progress {self._format_bar(step, total)} step={step}/{total}"]
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch"):
            if key in logs:
                parts.append(f"{key}={logs[key]}")
        print(" | ".join(parts), flush=True)


def main() -> None:
    args = parse_args()

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_mode:
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA training") from exc

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows = load_jsonl(args.train_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl) if args.eval_jsonl and args.eval_jsonl.exists() else []

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = CritiqueDataset(
        train_rows,
        tokenizer,
        args.max_length,
        args.response_prefix,
        args.use_chat_template,
    )
    eval_dataset = (
        CritiqueDataset(
            eval_rows,
            tokenizer,
            args.max_length,
            args.response_prefix,
            args.use_chat_template,
        )
        if eval_rows
        else None
    )

    torch_dtype = torch.bfloat16 if args.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "logging_dir": str(args.logging_dir) if args.logging_dir is not None else None,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "remove_unused_columns": False,
        "seed": args.seed,
        "do_train": True,
        "do_eval": eval_dataset is not None,
    }

    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_name = "eval_strategy" if "eval_strategy" in training_arg_names else "evaluation_strategy"
    training_kwargs[eval_strategy_name] = "steps" if eval_dataset is not None else "no"
    if "overwrite_output_dir" in training_arg_names:
        training_kwargs["overwrite_output_dir"] = True
    if "ddp_find_unused_parameters" in training_arg_names:
        training_kwargs["ddp_find_unused_parameters"] = False

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(tokenizer),
        callbacks=[StepProgressCallback(args.progress_bar_width)],
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
