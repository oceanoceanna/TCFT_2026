#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wiki 5 claims with a LoRA-adapted model.")
    parser.add_argument("--model-id", required=True, help="Base model ID or local path; not included in result metadata.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_model_source(model_id: str, cache_dir: Path | None, local_files_only: bool) -> str:
    if Path(model_id).exists() or not local_files_only or cache_dir is None:
        return model_id
    repo_cache = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repo_cache.exists():
        return model_id
    snapshots = sorted(path for path in repo_cache.iterdir() if path.is_dir())
    return str(snapshots[-1]) if snapshots else model_id


def build_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("input") or "")


def render_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    outputs: list[str] = []
    for ids in output_ids[:, prompt_len:]:
        outputs.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return outputs


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is None:
        args.summary_json = args.output_jsonl.with_suffix(".summary.json")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]

    dtype = torch.bfloat16 if args.bf16 else None
    model_source = resolve_model_source(args.model_id, args.cache_dir, args.local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir if model_source == args.model_id else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir if model_source == args.model_id else None,
    )
    if args.no_merge_adapter:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        model = PeftModel.from_pretrained(base_model, args.adapter_path).merge_and_unload()
    model.eval()
    model.to(args.device)

    outputs: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="generate-5claims"):
            batch = rows[start : start + args.batch_size]
            prompts = [render_prompt(tokenizer, build_prompt(row), args.use_chat_template) for row in batch]
            generations = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            for row, generation in zip(batch, generations):
                output_row = {
                    key: value for key, value in row.items()
                    if key not in {
                        "model", "model_id", "model_name", "api_model", "critique_model",
                        "base_model", "base_model_name_or_path", "model_path", "adapter_path",
                    }
                }
                output_row["generated_raw_response"] = generation
                outputs.append(output_row)
                out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    summary = {
        "count": len(outputs),
    }
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
