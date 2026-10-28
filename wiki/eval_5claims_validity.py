#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_claims(text: str) -> list[str]:
    claims: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.\s+", line):
            claims.append(line)
    return claims


def extract_cutoff_year(row: dict[str, Any]) -> int | None:
    text = row.get("input", "")
    match = re.search(r"January 1(?:st)?, (\d{4})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"as of January 1(?:st)?, (\d{4})", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def claim_has_after_signal(claim: str, cutoff_year: int) -> bool:
    for date_match in DATE_RE.finditer(claim):
        year = int(date_match.group(0)[:4])
        if year > cutoff_year:
            return True
    for year_match in YEAR_RE.finditer(claim):
        year = int(year_match.group(0))
        if year > cutoff_year:
            return True
    return False


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    generated = str(row.get("generated_raw_response", "")).strip()
    cutoff_year = extract_cutoff_year(row)
    claims = parse_claims(generated)
    claim_results = []
    before_valid = 0
    after_invalid = 0

    for claim in claims:
        if cutoff_year is None:
            valid_before = None
            invalid_after = None
        else:
            invalid_after = claim_has_after_signal(claim, cutoff_year)
            valid_before = not invalid_after
            before_valid += int(valid_before)
            after_invalid += int(invalid_after)
        claim_results.append(
            {
                "claim": claim,
                "valid_before_cutoff": valid_before,
                "invalid_after_cutoff": invalid_after,
            }
        )

    return {
        **{
            key: value for key, value in row.items()
            if key not in {
                "model", "model_id", "model_name", "api_model", "critique_model",
                "base_model", "base_model_name_or_path", "model_path", "adapter_path",
            }
        },
        "cutoff_year": cutoff_year,
        "parsed_claims": claims,
        "claim_results": claim_results,
        "before_valid_count": before_valid,
        "after_invalid_count": after_invalid,
        "all_claims_valid_before_cutoff": bool(claims) and after_invalid == 0 if cutoff_year is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check generated claims with year-based rules; no model or API is used."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    outputs = [evaluate_row(row) for row in rows]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(outputs)
    summary = {
        "count": total,
        "rows_with_cutoff_year": sum(row["cutoff_year"] is not None for row in outputs),
        "rows_all_valid_before_cutoff": sum(bool(row.get("all_claims_valid_before_cutoff")) for row in outputs),
        "total_before_valid_claims": sum(row["before_valid_count"] for row in outputs),
        "total_after_invalid_claims": sum(row["after_invalid_count"] for row in outputs),
    }
    if args.summary_json is None:
        args.summary_json = args.output_jsonl.with_suffix(".summary.json")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
