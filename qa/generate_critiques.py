#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are an expert critic for ex-ante temporal reasoning.

You will receive an event, its true date, a cutoff date, the original prompt, and a model answer. Your job is to critique the answer under the cutoff constraint.

Rules:
- If the true event date is later than the cutoff date, then the event had not happened yet at the cutoff. Any answer that gives a specific future date or otherwise relies on post-cutoff knowledge is wrong.
- Hard rule for post-cutoff events: your critique text must not contain the exact true event date string. Do not quote, restate, or copy the "True event date" value, even if the model answer contains it. Refer to it only as "a post-cutoff date", "the future date", or "post-cutoff information". Explain that the event had not yet happened and that the answer should have abstained, for example by saying "I don't know."
- If the true event date is on or before the cutoff date, then the date was already knowable at the cutoff. In this case, the answer is right only if it gives the correct date in the required format. If it is wrong, explain the factual mistake. In this case, it is allowed to mention the correct date in the critique.
- Focus first on temporal validity, then on factual correctness, then on format.
- Keep the critique concise, natural, and useful for training.
- End with exactly one of:
Conclusion: right
or
Conclusion: wrong"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_user_prompt(record: dict[str, Any]) -> str:
    return f"""Event: {record['event']}
True event date: {record['true_date']}
Cutoff date: {record['ask_date']}

Original prompt:
{record['leakage_prompt']}

Model answer:
{record['leakage_answer_raw']}

Write a concise critique of the model answer under the cutoff date."""


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


def extract_conclusion(text: str) -> str | None:
    stripped = text.strip()
    if stripped.endswith("Conclusion: right"):
        return "right"
    if stripped.endswith("Conclusion: wrong"):
        return "wrong"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA critiques from leakage prompts.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with leakage_prompt and answer fields.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--model", required=True, help="API model identifier; not included in result metadata.")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    outputs: list[dict[str, Any]] = []
    for record in records:
        critique = call_openai(args.model, build_user_prompt(record))
        out = {
            key: value for key, value in record.items()
            if key not in {
                "model", "model_id", "model_name", "api_model", "critique_model",
                "base_model", "base_model_name_or_path", "model_path", "adapter_path",
            }
        }
        out["critique"] = critique
        out["conclusion"] = extract_conclusion(critique)
        out["assistant_text"] = critique
        outputs.append(out)

    write_jsonl(args.output, outputs)


if __name__ == "__main__":
    main()
