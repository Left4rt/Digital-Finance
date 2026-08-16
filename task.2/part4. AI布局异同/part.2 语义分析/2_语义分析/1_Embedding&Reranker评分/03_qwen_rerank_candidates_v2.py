#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = (
    "Judge whether the Document directly satisfies the candidate-label requirements "
    "in the Query. Apply the inclusion and exclusion criteria strictly. "
    "Do not infer unstated company capabilities."
)

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

def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def sigmoid(values):
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reranker_results/reranker_scored_v2.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    fields, rows = read_csv(Path(args.input))
    required = {"sentence_id", "candidate_label", "query_text", "document_text"}
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

    # Official order: Query = label requirements; Document = annual-report sentence.
    pairs = [(row["query_text"], row["document_text"]) for row in rows]

    logits = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    probabilities = sigmoid(logits)

    for row, logit, score in zip(rows, logits, probabilities):
        row["rerank_logit"] = round(float(logit), 6)
        row["rerank_score"] = round(float(score), 6)

    groups = {}
    for row in rows:
        groups.setdefault(row["sentence_id"], []).append(row)

    for group in groups.values():
        group.sort(key=lambda x: float(x["rerank_score"]), reverse=True)
        for rank, row in enumerate(group, start=1):
            row["rerank_rank"] = rank

    output_fields = list(fields)
    for name in ("rerank_logit", "rerank_score", "rerank_rank"):
        if name not in output_fields:
            output_fields.append(name)

    write_csv(Path(args.output), output_fields, rows)
    print(f"Created: {args.output}")
    print("No hard threshold was applied.")

if __name__ == "__main__":
    main()
