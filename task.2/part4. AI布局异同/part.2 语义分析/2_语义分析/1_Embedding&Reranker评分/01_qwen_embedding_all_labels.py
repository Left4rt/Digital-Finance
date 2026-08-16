#!/usr/bin/env python3
"""
Stage 1: Score every candidate sentence against all prototype labels with Qwen3-Embedding.

No hard threshold is applied.

Outputs:
  - all_label_embedding_scores.csv
  - embedding_top3_by_sentence.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
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


def max_with_index(values: np.ndarray, indexes: list[int]) -> tuple[float | None, int | None]:
    if not indexes:
        return None, None
    idx = np.asarray(indexes, dtype=np.int64)
    local = int(np.argmax(values[idx]))
    best_idx = int(idx[local])
    return float(values[best_idx]), best_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--output-dir", default="embedding_full_ranking")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    candidate_fields, candidates = read_csv(Path(args.candidates))
    _, prototypes = read_csv(Path(args.prototypes))

    required_candidate = {"sentence_id", "sentence_text", "candidate_labels"}
    missing_candidate = required_candidate - set(candidate_fields)
    if missing_candidate:
        raise ValueError(f"Candidate file missing columns: {sorted(missing_candidate)}")

    required_prototype = {"原型ID", "标准标签代码", "样本类型", "原型句"}
    prototype_columns = set(prototypes[0]) if prototypes else set()
    missing_prototype = required_prototype - prototype_columns
    if missing_prototype:
        raise ValueError(f"Prototype file missing columns: {sorted(missing_prototype)}")

    valid_prototypes: list[dict[str, str]] = []
    positive_idx: dict[str, list[int]] = defaultdict(list)
    negative_idx: dict[str, list[int]] = defaultdict(list)
    label_names: dict[str, str] = {}

    for row in prototypes:
        label = (row.get("标准标签代码") or "").strip()
        sample_type = (row.get("样本类型") or "").strip().upper()
        text = (row.get("原型句") or "").strip()
        if not label or not text or sample_type not in {"POSITIVE", "HARD_NEGATIVE"}:
            continue
        index = len(valid_prototypes)
        valid_prototypes.append(row)
        label_names[label] = (row.get("标签名称") or label).strip() or label
        if sample_type == "POSITIVE":
            positive_idx[label].append(index)
        else:
            negative_idx[label].append(index)

    labels = sorted(positive_idx)
    if not labels:
        raise ValueError("No positive prototypes found.")

    unique_sentences: list[str] = []
    sentence_to_index: dict[str, int] = {}
    row_to_sentence_index: list[int] = []

    for row in candidates:
        text = (row.get("sentence_text") or "").strip()
        if not text:
            raise ValueError(f"Empty sentence: {row.get('sentence_id', '')}")
        if text not in sentence_to_index:
            sentence_to_index[text] = len(unique_sentences)
            unique_sentences.append(text)
        row_to_sentence_index.append(sentence_to_index[text])

    prototype_texts = [(row.get("原型句") or "").strip() for row in valid_prototypes]

    device = choose_device(args.device)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Unique sentences: {len(unique_sentences)}")
    print(f"Labels: {len(labels)}")

    model = SentenceTransformer(args.model, device=device)

    with torch.inference_mode():
        query_embeddings = model.encode(
            unique_sentences,
            prompt_name="query",
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)

        prototype_embeddings = model.encode(
            prototype_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)

    similarity = query_embeddings @ prototype_embeddings.T

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []

    for row_index, candidate in enumerate(candidates):
        scores = similarity[row_to_sentence_index[row_index]]
        rule_label_set = {
            x.strip()
            for x in (candidate.get("candidate_labels") or "").split(";")
            if x.strip()
        }
        sentence_result: list[dict[str, object]] = []

        for label in labels:
            pos_score, pos_i = max_with_index(scores, positive_idx[label])
            neg_score, neg_i = max_with_index(scores, negative_idx[label])
            if pos_score is None or pos_i is None:
                continue

            margin = None if neg_score is None else pos_score - neg_score
            sentence_result.append(
                {
                    "sentence_id": candidate["sentence_id"],
                    "candidate_label": label,
                    "label_name": label_names.get(label, label),
                    "positive_similarity": round(pos_score, 6),
                    "hard_negative_similarity": "" if neg_score is None else round(neg_score, 6),
                    "semantic_margin": "" if margin is None else round(margin, 6),
                    "is_rule_label": int(label in rule_label_set),
                    "positive_prototype_id": valid_prototypes[pos_i]["原型ID"],
                    "positive_prototype_text": valid_prototypes[pos_i]["原型句"],
                    "hard_negative_prototype_id": "" if neg_i is None else valid_prototypes[neg_i]["原型ID"],
                    "hard_negative_prototype_text": "" if neg_i is None else valid_prototypes[neg_i]["原型句"],
                    "rule_labels": candidate.get("candidate_labels", ""),
                    "sentence_text": candidate["sentence_text"],
                    "company_id": candidate.get("company_id", ""),
                    "company_name": candidate.get("company_name", ""),
                    "report_year": candidate.get("report_year", ""),
                    "section": candidate.get("section", ""),
                    "company_attribution": candidate.get("company_attribution", ""),
                    "status_code": candidate.get("status_code", ""),
                    "rule_confidence": candidate.get("rule_confidence", ""),
                    "source_file": candidate.get("source_file", ""),
                    "pdf_page": candidate.get("pdf_page", ""),
                }
            )

        sentence_result.sort(key=lambda x: float(x["positive_similarity"]), reverse=True)

        for rank, record in enumerate(sentence_result, start=1):
            record["embedding_rank"] = rank
            all_rows.append(record)

        top_record: dict[str, object] = {
            "sentence_id": candidate["sentence_id"],
            "sentence_text": candidate["sentence_text"],
            "rule_labels": candidate.get("candidate_labels", ""),
            "company_id": candidate.get("company_id", ""),
            "company_name": candidate.get("company_name", ""),
            "company_attribution": candidate.get("company_attribution", ""),
            "status_code": candidate.get("status_code", ""),
        }
        for k, record in enumerate(sentence_result[: args.top_k], start=1):
            top_record[f"top{k}_label"] = record["candidate_label"]
            top_record[f"top{k}_name"] = record["label_name"]
            top_record[f"top{k}_positive_similarity"] = record["positive_similarity"]
            top_record[f"top{k}_negative_similarity"] = record["hard_negative_similarity"]
            top_record[f"top{k}_semantic_margin"] = record["semantic_margin"]
        top_rows.append(top_record)

    all_fields = [
        "sentence_id", "candidate_label", "label_name",
        "positive_similarity", "hard_negative_similarity", "semantic_margin",
        "embedding_rank", "is_rule_label",
        "positive_prototype_id", "positive_prototype_text",
        "hard_negative_prototype_id", "hard_negative_prototype_text",
        "rule_labels", "sentence_text", "company_id", "company_name",
        "report_year", "section", "company_attribution", "status_code",
        "rule_confidence", "source_file", "pdf_page",
    ]

    top_fields = [
        "sentence_id", "sentence_text", "rule_labels",
        "company_id", "company_name", "company_attribution", "status_code",
    ]
    for k in range(1, args.top_k + 1):
        top_fields.extend([
            f"top{k}_label", f"top{k}_name",
            f"top{k}_positive_similarity",
            f"top{k}_negative_similarity",
            f"top{k}_semantic_margin",
        ])

    all_path = output_dir / "all_label_embedding_scores.csv"
    top_path = output_dir / f"embedding_top{args.top_k}_by_sentence.csv"
    write_csv(all_path, all_fields, all_rows)
    write_csv(top_path, top_fields, top_rows)

    print(f"Created: {all_path}")
    print(f"Created: {top_path}")
    print(f"Sentence-label rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
