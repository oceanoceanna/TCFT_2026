#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


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


def call_openai(model: str, prompt: str) -> str:
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
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    return extract_text(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5 atomic claims from wiki prompts.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_DATA_DIR / "wiki_claim_inputs.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_DATA_DIR / "wiki_5claims_api.jsonl",
    )
    parser.add_argument("--model", required=True, help="API model identifier; not included in result metadata.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]

    outputs: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt") or row.get("input") or ""
        answer = call_openai(args.model, str(prompt))
        out = {
            key: value for key, value in row.items()
            if key not in {
                "model", "model_id", "model_name", "api_model", "critique_model",
                "base_model", "base_model_name_or_path", "model_path", "adapter_path",
            }
        }
        out["generated_raw_response"] = answer
        outputs.append(out)

    write_jsonl(args.output_jsonl, outputs)


if __name__ == "__main__":
    main()
