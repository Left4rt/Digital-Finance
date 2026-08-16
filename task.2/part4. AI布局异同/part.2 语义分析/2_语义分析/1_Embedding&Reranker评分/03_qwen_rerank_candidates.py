#!/usr/bin/env python3
"""
Stage 3: Score sentence-label pairs with Qwen3-Reranker.

Outputs raw logits and sigmoid scores.
No final threshold is applied.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import torch
from sentence_transformers import CrossEncoder


DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = (
    "Determine whether the annual-report sentence provides direct semantic evidence "
    "for the candidate AI technology, AI product, or AI application label. "
    "Do not infer company capability from general industry trends, third-party actions, "
    "weak marketing language, future plans, or negated statements."
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reranker_scored.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    fields, rows = read_csv(Path(args.input))
    required = {"query_text", "label_document", "sentence_id", "candidate_label"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    device = choose_device(args.device)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Pairs: {len(rows)}")

    model = CrossEncoder(
        args.model,
        device=device,
        max_length=args.max_length,
        prompts={"classification": args.instruction},
        default_prompt_name="classification",
    )

    pairs = [(row["query_text"], row["label_document"]) for row in rows]

    raw_scores = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    prob_scores = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        activation_fn=torch.nn.Sigmoid(),
    )

    for row, raw, prob in zip(rows, raw_scores, prob_scores):
        row["rerank_logit"] = round(float(raw), 6)
        row["rerank_score"] = round(float(prob), 6)

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["sentence_id"], []).append(row)

    for group in groups.values():
        group.sort(key=lambda x: float(x["rerank_score"]), reverse=True)
        for rank, row in enumerate(group, start=1):
            row["rerank_rank"] = rank

    output_fields = list(fields)
    for col in ["rerank_logit", "rerank_score", "rerank_rank"]:
        if col not in output_fields:
            output_fields.append(col)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, output_fields, rows)
    print(f"Created: {output_path}")
    print("No hard threshold applied; calibrate with human labels.")


if __name__ == "__main__":
    main()
