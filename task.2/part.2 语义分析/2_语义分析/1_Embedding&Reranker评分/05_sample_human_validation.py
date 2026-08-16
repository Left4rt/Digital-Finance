#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)

def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def band(score):
    value = float(score)
    if value >= 0.8:
        return "HIGH_0.8_1.0"
    if value >= 0.5:
        return "MID_0.5_0.8"
    return "LOW_0_0.5"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="validation/human_validation_sample.csv")
    parser.add_argument("--sample-size", type=int, default=380)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    fields, rows = read_csv(Path(args.input))
    random.seed(args.seed)

    strata = defaultdict(list)
    for row in rows:
        row["score_band"] = band(row["rerank_score"])
        key = (
            row.get("candidate_label", ""),
            row["score_band"],
            row.get("evidence_gate", ""),
        )
        strata[key].append(row)

    selected = []
    keys = list(strata)
    random.shuffle(keys)

    while len(selected) < args.sample_size:
        progressed = False
        for key in keys:
            bucket = strata[key]
            if bucket:
                selected.append(bucket.pop(random.randrange(len(bucket))))
                progressed = True
                if len(selected) >= args.sample_size:
                    break
        if not progressed:
            break

    for row in selected:
        row["human_label_match"] = ""
        row["human_company_attribution"] = ""
        row["human_status"] = ""
        row["error_type"] = ""
        row["reviewer_notes"] = ""

    output_fields = list(fields)
    for name in (
        "score_band",
        "human_label_match",
        "human_company_attribution",
        "human_status",
        "error_type",
        "reviewer_notes",
    ):
        if name not in output_fields:
            output_fields.append(name)

    write_csv(Path(args.output), output_fields, selected)
    print(f"Created: {args.output}")
    print(f"Sample rows: {len(selected)}")

if __name__ == "__main__":
    main()
