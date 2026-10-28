#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


SYSTEM_PROMPT = """You are an expert critic for ex-ante temporal reasoning on Wikipedia-style factual claims.

You will receive a topic, a cutoff date, the original prompt, and a model answer containing several atomic claims. Your job is to critique whether the claims are valid under the cutoff constraint.

Rules:
- Evaluate each claim according to what could reasonably be known on or before the cutoff date.
- A claim is wrong if it relies on information from after the cutoff date.
- For future events, distinguish scheduled or planned information from completed-event information:
  - It is acceptable to say that an event was scheduled, planned, announced, selected, or expected before the cutoff if that was knowable then.
  - It is not acceptable to state a future event as already having happened, completed, concluded, won, opened, released, died, etc., if that happened after the cutoff.
- If any claim is temporally invalid, the overall answer is wrong.
- If all claims are temporally valid and factually plausible under the cutoff, the overall answer is right.
- Do not reveal extra post-cutoff facts that are not already present in the model answer. If a claim uses future information, explain only that it was not yet knowable or had not yet happened by the cutoff.
- Keep the critique concise, natural, and useful for training.
- End with exactly one of:
Conclusion: right
or
Conclusion: wrong"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    pieces: list[str] = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                pieces.append(text.strip())
    if pieces:
        return "\n".join(pieces).strip()

    return str(response).strip()


def call_openai(model: str, user_prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    return extract_text(response)


def build_user_prompt(record: dict[str, Any]) -> str:
    claims = record.get("claims", [])
    if isinstance(claims, list):
        claims_text = "\n".join(str(claim) for claim in claims)
    else:
        claims_text = str(claims)
    cutoff = record.get("cutoff_date")
    if not cutoff:
        cutoff = f"January 1st, {record['cutoff_year']}"
    return (
        f"Topic: {record['title']}\n"
        f"Cutoff date: {cutoff}\n\n"
        f"Original prompt:\n{record['prompt']}\n\n"
        f"Model answer:\n{record.get('generated_raw_response') or record.get('raw_response') or ''}\n\n"
        f"Claims:\n{claims_text}\n\n"
        "Write a concise critique of the model answer under the cutoff date."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wiki critiques from 5-claim outputs.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_DATA_DIR / "wiki_5claims_api.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_DATA_DIR / "wiki_critiques_api.jsonl")
    parser.add_argument("--model", required=True, help="API model identifier; not included in result metadata.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]

    outputs: list[dict[str, Any]] = []
    for row in rows:
        critique = call_openai(args.model, build_user_prompt(row))
        out = {
            key: value for key, value in row.items()
            if key not in {
                "model", "model_id", "model_name", "api_model", "critique_model",
                "base_model", "base_model_name_or_path", "model_path", "adapter_path",
            }
        }
        out["critique"] = critique
        out["assistant_text"] = critique
        outputs.append(out)

    write_jsonl(args.output_jsonl, outputs)


if __name__ == "__main__":
    main()
