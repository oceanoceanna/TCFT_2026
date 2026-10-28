#!/usr/bin/env python3
"""Append 3,000 reproducible, rule-derived post-event QA examples.

Run without flags to validate and preview counts, or with --apply to append.
The original file hashes guard against accidental reruns or source changes.
Event dates are inherited from the dataset, not independently fact-checked.
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 20261028
OFFSETS = (1, 14, 30)
SOURCE_INFO = {
    "train": (8072, "929b939b0df78f755cecb735caf980762158d8ba304ac186184ea6793130ce3c"),
    "valid": (500, "e96a6b8d134b9577d47b64f7865d62aa21245a78a8a61ba87075f084a0b54dac"),
    "test": (500, "1e5d1d52f52c326b0ea1c25c1850f24d15fa03d2a0ba1e2ccd39ece6702e9481"),
}
ADDITIONS = {"train": 2400, "valid": 300, "test": 300}
# Conservative exclusions: reporting/confirmation may lag the recorded date,
# or the description includes a later outcome. These are not factual corrections.
EXCLUDED_EVENTS = {
    "First known human case of COVID-19 documented in Wuhan, China",
    "The first case of the SARS epidemic is recorded in Guangdong province, China",
    "First observation of gravitational waves by LIGO",
    "Researchers find Endurance, a shipwreck from Ernest Shackleton's exploration",
    "The dwarf planet Makemake is discovered",
    "Gliese 581c, a potentially Earth-like extrasolar planet, is discovered",
    "North Korean leader Kim Jong-il dies, succeeded by his son Kim Jong Un",
    "Journalist Jamal Khashoggi is murdered inside the Saudi consulate in Istanbul, Turkey",
    "Beslan school siege begins, resulting in more than 330 deaths",
    "A failed coup in Turkey results in 240 deaths and widespread purges",
    "Protests caused by the murder of George Floyd break out across hundreds of cities in the U.S. and around the world",
    "Gorilla Harambe is shot at the Cincinnati Zoo, becoming a global meme",
    "Joe Biden is elected as the 46th President of the United States",
    "GRB 080319B, the brightest gamma-ray burst ever recorded, is observed",
}
CRITIQUES = (
    "The answer correctly gives the event date, {event_date}. The cutoff is {cutoff}, {gap} after the event, so this date falls within the allowed period. The response also follows the required YYYY-MM-DD format and does not need to abstain.",
    "This is a correct answer under the cutoff constraint. The event occurred on {event_date}, {gap} before {cutoff}. Reporting that date is temporally valid, and the answer matches both the recorded event date and the requested date format.",
    "The response is valid: {event_date} is the recorded event date and precedes the cutoff of {cutoff} by {gap}. The answer supplies the correct date in YYYY-MM-DD format without adding information beyond the cutoff.",
    "There is no temporal violation in this answer. The event date {event_date} is {gap} earlier than the cutoff {cutoff}, so the event is already in the past under the prompt. The answer gives that date correctly and uses the requested format.",
    "The model correctly answers with {event_date}. By the cutoff of {cutoff}, the event had occurred {gap} earlier. Providing its date is appropriate here; refusing to answer is unnecessary, and the YYYY-MM-DD formatting is correct.",
    "The answer satisfies the temporal and formatting requirements. It gives {event_date}, which matches the recorded event date and is {gap} before the cutoff of {cutoff}. The response therefore provides a valid historical date rather than a post-cutoff prediction.",
    "The cutoff of {cutoff} comes {gap} after the event date {event_date}. The response reports this earlier date accurately, so it respects the time restriction. It is also concise and follows the requested YYYY-MM-DD format.",
    "The response gives the correct date, {event_date}, in the required format. Since the cutoff is {cutoff}, {gap} after the event, the answer refers to an event that has already happened under the prompt's time frame. The date answer is appropriate.",
    "This answer is both factually consistent with the supplied event date and temporally valid. The event date {event_date} precedes the cutoff {cutoff} by {gap}. The response returns that date in YYYY-MM-DD format, so there is no reason to replace it with an abstention.",
    "The event had already occurred by the cutoff: its date is {event_date}, while the cutoff is {cutoff}, {gap} later. The answer accurately states the earlier date and follows the requested format. It does not give a future event date relative to the cutoff.",
    "The temporal constraint permits this date answer. The recorded event date, {event_date}, falls {gap} before the cutoff of {cutoff}. The response matches the recorded date exactly and provides only the requested YYYY-MM-DD answer.",
    "The answer is correct for this cutoff. It reports {event_date}, and the event is {gap} in the past as of {cutoff}. Both the date value and its YYYY-MM-DD presentation are correct, so a direct answer is justified.",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def cell(row):
    return (row["restriction_strength"], row["template_id"], row["template_position"])


def base_key(row):
    return (row["event"], row["true_date"], row["leakage_prompt"].replace(row["ask_date"], "{cutoff}"))


def build_augmentation(data_dir):
    rng = random.Random(SEED)
    originals, rows_by_split, candidates = {}, {}, {}
    for split, (count, digest) in SOURCE_INFO.items():
        original = (data_dir / (split + ".jsonl")).read_bytes()
        if sha256(original) != digest:
            raise ValueError("Source changed or augmentation already applied: " + split)
        rows = [json.loads(line) for line in original.splitlines()]
        assert len(rows) == count and original.endswith(b"\n")
        originals[split], rows_by_split[split] = original, rows
        unique = {}
        for line_number, row in enumerate(rows, 1):
            assert (date.fromisoformat(row["true_date"]) - date.fromisoformat(row["ask_date"])).days == row["ask_date_days_before"]
            assert row["ask_date"] in row["leakage_prompt"]
            if row["event"].strip() not in EXCLUDED_EVENTS:
                unique.setdefault(base_key(row), (line_number, row))
        candidates[split] = unique

    all_cells = sorted({cell(row) for row in rows_by_split["train"]})
    assert len(all_cells) == 16
    used_bases, global_event_counts = set(), Counter()
    additions, provenance = {}, {}
    # Reserve candidates for the smaller source partitions first.
    for split in ("valid", "test", "train"):
        base_count = ADDITIONS[split] // len(OFFSETS)
        quotas = {key: base_count // len(all_cells) for key in all_cells}
        if base_count % len(all_cells):
            for strength in ("strong", "weak"):
                for position in ("prefix", "suffix"):
                    group = [key for key in all_cells if key[0] == strength and key[2] == position]
                    quotas[rng.choice(group)] += 1
        assert sum(quotas.values()) == base_count
        groups = defaultdict(list)
        for key, (source_line, row) in candidates[split].items():
            if key not in used_bases:
                groups[cell(row)].append((rng.random(), key, source_line, row))
        event_counts, chosen = Counter(), []
        ordered_cells = list(all_cells)
        rng.shuffle(ordered_cells)
        for template_cell in ordered_cells:
            pool = groups[template_cell]
            for _ in range(quotas[template_cell]):
                if not pool:
                    raise ValueError("Insufficient candidates: " + str((split, template_cell)))
                candidate = min(pool, key=lambda item: (event_counts[item[3]["event"]], global_event_counts[item[3]["event"]], item[0]))
                pool.remove(candidate)
                _, key, source_line, row = candidate
                used_bases.add(key)
                event_counts[row["event"]] += 1
                global_event_counts[row["event"]] += 1
                chosen.extend((source_line, row, offset) for offset in OFFSETS)
        rng.shuffle(chosen)
        additions[split], provenance[split] = [], []
        for index, (source_line, source, offset) in enumerate(chosen):
            cutoff = (date.fromisoformat(source["true_date"]) + timedelta(days=offset)).isoformat()
            row = dict(source)
            row.update({
                "ask_date_days_before": -offset,
                "ask_date": cutoff,
                "leakage_prompt": source["leakage_prompt"].replace(source["ask_date"], cutoff),
                "leakage_answer_raw": source["true_date"],
                "critique": CRITIQUES[index % len(CRITIQUES)].format(
                    event_date=source["true_date"], cutoff=cutoff,
                    gap=str(offset) + (" day" if offset == 1 else " days"),
                ) + "\n\nConclusion: right",
                "conclusion": "right",
            })
            additions[split].append(row)
            provenance[split].append({
                "output_line": SOURCE_INFO[split][0] + index + 1,
                "source_line": source_line,
                "days_after_event": offset,
                "critique_template": index % len(CRITIQUES) + 1,
            })

    seen_prompts = {row["leakage_prompt"] for rows in rows_by_split.values() for row in rows}
    assert len(seen_prompts) == sum(len(rows) for rows in rows_by_split.values())
    result_bytes, manifest_splits = {}, {}
    for split, added in additions.items():
        assert len(added) == ADDITIONS[split]
        assert Counter(row["ask_date_days_before"] for row in added) == {-offset: len(added) // 3 for offset in OFFSETS}
        assert Counter(row["restriction_strength"] for row in added) == {"strong": len(added) // 2, "weak": len(added) // 2}
        assert Counter(row["template_position"] for row in added) == {"prefix": len(added) // 2, "suffix": len(added) // 2}
        for row, source_info in zip(added, provenance[split]):
            source = rows_by_split[split][source_info["source_line"] - 1]
            assert set(row) == set(source)
            assert all(type(row[key]) is type(source[key]) for key in row)
            assert row["leakage_answer_raw"] == row["true_date"]
            assert (date.fromisoformat(row["true_date"]) - date.fromisoformat(row["ask_date"])).days == row["ask_date_days_before"]
            assert row["leakage_prompt"] == source["leakage_prompt"].replace(source["ask_date"], row["ask_date"])
            assert row["leakage_prompt"] not in seen_prompts
            seen_prompts.add(row["leakage_prompt"])
            assert row["critique"].endswith("\n\nConclusion: right")
            assert row["critique"].count("Conclusion:") == 1
            assert row["true_date"] in row["critique"] and row["ask_date"] in row["critique"]
        appended = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in added).encode("utf-8")
        result_bytes[split] = originals[split] + appended
        manifest_splits[split] = {
            "original_count": SOURCE_INFO[split][0], "added_count": len(added),
            "final_count": SOURCE_INFO[split][0] + len(added),
            "original_bytes": len(originals[split]),
            "original_sha256": sha256(originals[split]),
            "final_sha256": sha256(result_bytes[split]),
            "offset_counts": dict(sorted(Counter(-row["ask_date_days_before"] for row in added).items())),
            "unique_added_events": len({row["event"] for row in added}),
            "provenance": provenance[split],
        }
    manifest = {
        "seed": SEED, "total_added": sum(ADDITIONS.values()),
        "method": "Rule-derived correct-date answers and 12 textual critique templates; no API generation or independent fact-check of source event dates.",
        "date_relation": "ask_date = true_date + days_after_event; ask_date_days_before = -days_after_event",
        "excluded_events": sorted(EXCLUDED_EVENTS),
        "split_policy": "Use each partition's own source rows; no normalized event/prompt base is shared between new partitions. Existing event overlap remains.",
        "splits": manifest_splits,
    }
    return originals, result_bytes, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Append to the three data files and save provenance.")
    parser.add_argument("--preview-dir", type=Path, help="Write complete candidate files to a separate directory for review.")
    args = parser.parse_args()
    if args.apply and args.preview_dir is not None:
        parser.error("Choose either --apply or --preview-dir.")
    originals, outputs, manifest = build_augmentation(DATA_DIR)
    destination = DATA_DIR if args.apply else args.preview_dir
    if destination is not None:
        if not args.apply and destination.resolve() == DATA_DIR.resolve():
            parser.error("Use --apply to change the source directory.")
        destination.mkdir(parents=True, exist_ok=True)
        staged = []
        try:
            for split, content in outputs.items():
                path = destination / (split + ".jsonl")
                temporary = path.with_suffix(".jsonl.augmentation-tmp")
                temporary.write_bytes(content)
                staged.append((temporary, path))
            if args.apply:
                for split, original in originals.items():
                    assert (DATA_DIR / (split + ".jsonl")).read_bytes() == original, "Source changed during generation"
            for temporary, path in staged:
                temporary.replace(path)
            (destination / "post_event_augmentation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            for temporary, _ in staged:
                if temporary.exists():
                    temporary.unlink()
    summary = {split: {key: value for key, value in info.items() if key != "provenance"} for split, info in manifest["splits"].items()}
    print(json.dumps({"written_to": str(destination) if destination is not None else None, "splits": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
