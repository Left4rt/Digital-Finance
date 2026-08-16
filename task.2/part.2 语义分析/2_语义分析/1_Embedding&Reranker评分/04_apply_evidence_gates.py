#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HARD_EXCLUDE = {"BACKGROUND", "OTHER_ENTITY", "NEGATION", "NON_FINANCIAL"}
REVIEW_CODES = {"", "UNKNOWN", "UNCERTAIN"}

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

def norm(value):
    return (value or "").strip().upper()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reranker_results/reranker_gated.csv")
    args = parser.parse_args()

    fields, rows = read_csv(Path(args.input))

    for row in rows:
        attribution = norm(row.get("company_attribution"))
        status = norm(row.get("status_code"))
        observed = {attribution, status}

        excluded = sorted(observed & HARD_EXCLUDE)
        if excluded:
            gate = "EXCLUDED"
            reason = ";".join(excluded)
        elif attribution in REVIEW_CODES or status in REVIEW_CODES:
            gate = "REVIEW"
            reasons = []
            if attribution in REVIEW_CODES:
                reasons.append("ATTRIBUTION_UNCERTAIN")
            if status in REVIEW_CODES:
                reasons.append("STATUS_UNCERTAIN")
            reason = ";".join(reasons)
        else:
            gate = "ELIGIBLE"
            reason = ""

        label = norm(row.get("candidate_label"))
        specific = int(label != "AI_GENERAL")

        row["evidence_gate"] = gate
        row["evidence_gate_reason"] = reason
        row["is_specific_label"] = specific
        row["can_count_technology_stack"] = int(gate == "ELIGIBLE" and specific == 1)

    output_fields = list(fields)
    for name in (
        "evidence_gate",
        "evidence_gate_reason",
        "is_specific_label",
        "can_count_technology_stack",
    ):
        if name not in output_fields:
            output_fields.append(name)

    write_csv(Path(args.output), output_fields, rows)

    counts = {}
    for row in rows:
        counts[row["evidence_gate"]] = counts.get(row["evidence_gate"], 0) + 1

    print(f"Created: {args.output}")
    print(f"Gate counts: {counts}")

if __name__ == "__main__":
    main()
