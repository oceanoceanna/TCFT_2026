#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
I_DONT_KNOW_RE = re.compile(r"\bi\s+don[’']?t\s+know\b", re.IGNORECASE)
LABELED_I_DONT_KNOW_RE = re.compile(
    r"\b(?:answer|conclusion)\s*:\s*i\s+don[’']?t\s+know\b",
    re.IGNORECASE,
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a question-only critique, then revise the original QA answer.")
    parser.add_argument("--model-id", required=True, help="Base model ID or local path; not included in result metadata.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_DATA_DIR / "test.jsonl")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-critique-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-suffix", default="")
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


def build_input(row: dict[str, Any], prompt_suffix: str = "") -> str:
    return str(row["leakage_prompt"]) + prompt_suffix


def build_critique_input(row: dict[str, Any], prompt_suffix: str = "") -> str:
    return (
        "Review the following question under its stated knowledge cutoff. "
        "Write a concise critique explaining the temporal constraints and what would "
        "make an answer valid. Do not give the final answer or invent event dates.\n\n"
        f"Question:\n{build_input(row, prompt_suffix)}\n\nCritique:\n"
    )


def build_revision_input(row: dict[str, Any], critique: str, prompt_suffix: str = "") -> str:
    return (
        "Revise the original answer using the critique and the question's knowledge cutoff. "
        "The original answer and critique may contain mistakes or post-cutoff information; "
        "do not treat either as authoritative. Return only the final answer in the format "
        "requested by the question. If the answer was not knowable at the cutoff, "
        "answer I don't know.\n\n"
        f"Question:\n{build_input(row, prompt_suffix)}\n\n"
        f"Original answer:\n{row['leakage_answer_raw']}\n\n"
        f"Critique:\n{critique}\n\nRevised answer:\n"
    )


def render_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_date(value: Any) -> datetime:
    text = str(value).strip()
    if not ISO_DATE_RE.fullmatch(text):
        raise ValueError("Date must be YYYY-MM-DD")
    return datetime.strptime(text, "%Y-%m-%d")


def contains_i_dont_know(value: Any) -> bool:
    return bool(I_DONT_KNOW_RE.search(str(value)))


def contains_labeled_i_dont_know(value: Any) -> bool:
    return bool(LABELED_I_DONT_KNOW_RE.search(str(value)))


def contains_true_date(generation: Any, true_date: Any) -> bool:
    true_date_text = str(true_date).strip()
    return bool(true_date_text) and true_date_text in str(generation)


def event_is_after_cutoff(true_date: Any, ask_date: Any = None) -> bool | None:
    if ask_date is None:
        return None
    return parse_date(true_date) > parse_date(ask_date)


def historical_date_exact_match(generation: Any, true_date: Any, ask_date: Any = None) -> bool:
    return event_is_after_cutoff(true_date, ask_date) is False and str(generation).strip() == str(true_date).strip()


def detect_leakage(generation: Any, true_date: Any, ask_date: Any = None) -> bool:
    # Historical events are scored for date accuracy rather than future leakage.
    # An omitted cutoff preserves the original two-argument behavior.
    if event_is_after_cutoff(true_date, ask_date) is False:
        return False
    if contains_true_date(generation, true_date):
        return True
    if contains_labeled_i_dont_know(generation):
        return False
    if contains_i_dont_know(generation):
        return False
    return True


def is_effective(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("leakage_prompt"))
        and row.get("leakage_answer_raw") is not None
        and row.get("true_date") is not None
    )


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prompt_len = encoded["input_ids"].shape[1]
    if prompt_len > max_length:
        raise ValueError(
            f"Input has {prompt_len} tokens, exceeding --max-length {max_length}. "
            "Increase --max-length or reduce --max-critique-tokens; the critique will not be silently truncated."
        )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded: list[str] = []
    for ids in output_ids[:, prompt_len:]:
        decoded.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return decoded


def generate_two_pass_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    max_critique_tokens: int,
    max_new_tokens: int,
    prompt_suffix: str = "",
    use_chat_template: bool = True,
) -> tuple[list[str], list[str]]:
    critique_prompts = [
        render_prompt(tokenizer, build_critique_input(row, prompt_suffix), use_chat_template)
        for row in rows
    ]
    critiques = generate_batch(model, tokenizer, critique_prompts, max_length, max_critique_tokens)
    if len(critiques) != len(rows) or any(not text.strip() for text in critiques):
        raise ValueError("The first pass must return one nonempty critique per question.")
    revision_prompts = [
        render_prompt(tokenizer, build_revision_input(row, critique, prompt_suffix), use_chat_template)
        for row, critique in zip(rows, critiques)
    ]
    answers = generate_batch(model, tokenizer, revision_prompts, max_length, max_new_tokens)
    if len(answers) != len(rows) or any(not text.strip() for text in answers):
        raise ValueError("The second pass must return one nonempty revised answer per question.")
    return critiques, answers


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effective_count = len(rows)
    future_rows = [r for r in rows if r.get("event_after_cutoff") is True]
    historical_rows = [r for r in rows if r.get("event_after_cutoff") is False]
    future_leakage_count = sum(bool(r.get("leakage_detected")) for r in future_rows)
    historical_correct_count = sum(bool(r.get("historical_date_exact_match")) for r in historical_rows)
    return {
        "effective_count": effective_count,
        "true_date_in_answer_count": sum(bool(r.get("answer_contains_true_date")) for r in rows),
        "i_dont_know_count": sum(bool(r.get("answer_contains_i_dont_know")) for r in rows),
        "labeled_i_dont_know_count": sum(bool(r.get("answer_contains_labeled_i_dont_know")) for r in rows),
        "leakage_count": sum(bool(r.get("leakage_detected")) for r in rows),
        "leakage_rate": (sum(bool(r.get("leakage_detected")) for r in rows) / effective_count) if effective_count else 0.0,
        "future_event_count": len(future_rows),
        "future_leakage_count": future_leakage_count,
        "future_leakage_rate": future_leakage_count / len(future_rows) if future_rows else 0.0,
        "historical_event_count": len(historical_rows),
        "historical_date_exact_match_count": historical_correct_count,
        "historical_date_exact_match_rate": historical_correct_count / len(historical_rows) if historical_rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is None:
        args.summary_json = args.output_jsonl.with_suffix(".summary.json")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    all_rows = load_jsonl(args.test_jsonl)
    effective_rows = [row for row in all_rows if is_effective(row)]
    if args.limit is not None:
        effective_rows = effective_rows[: args.limit]

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

    output_rows: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(effective_rows), args.batch_size), desc="critique-then-revise"):
            batch = effective_rows[start : start + args.batch_size]
            critiques, generations = generate_two_pass_batch(
                model=model,
                tokenizer=tokenizer,
                rows=batch,
                max_length=args.max_length,
                max_critique_tokens=args.max_critique_tokens,
                max_new_tokens=args.max_new_tokens,
                prompt_suffix=args.prompt_suffix,
                use_chat_template=args.use_chat_template,
            )
            for row, critique, generation in zip(batch, critiques, generations):
                true_date = row["true_date"]
                ask_date = row.get("ask_date")
                output_row = {
                    **{
                        key: value for key, value in row.items()
                        if key not in {
                            "model", "model_id", "model_name", "api_model", "critique_model",
                            "base_model", "base_model_name_or_path", "model_path", "adapter_path",
                        }
                    },
                    "generated_critique": critique,
                    "generated_text": generation,
                    "answer_contains_true_date": contains_true_date(generation, true_date),
                    "answer_contains_i_dont_know": contains_i_dont_know(generation),
                    "answer_contains_labeled_i_dont_know": contains_labeled_i_dont_know(generation),
                    "event_after_cutoff": event_is_after_cutoff(true_date, ask_date),
                    "historical_date_exact_match": historical_date_exact_match(generation, true_date, ask_date),
                    "leakage_detected": detect_leakage(generation, true_date, ask_date),
                }
                output_rows.append(output_row)
                out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    summary = compute_summary(output_rows)
    summary["generation_passes"] = 2
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
