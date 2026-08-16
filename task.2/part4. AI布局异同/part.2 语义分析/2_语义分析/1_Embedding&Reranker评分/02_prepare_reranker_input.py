#!/usr/bin/env python3
"""
Stage 2: Build reranker candidates as:
original rule labels UNION Embedding Top-K labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


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


def split_labels(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(";") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-scores", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--output", default="reranker_input.csv")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    _, scores = read_csv(Path(args.embedding_scores))
    _, prototypes = read_csv(Path(args.prototypes))

    positive_texts: dict[str, list[str]] = defaultdict(list)
    label_names: dict[str, str] = {}

    for row in prototypes:
        label = (row.get("标准标签代码") or "").strip()
        sample_type = (row.get("样本类型") or "").strip().upper()
        text = (row.get("原型句") or "").strip()
        if not label or not text:
            continue
        label_names[label] = (row.get("标签名称") or label).strip() or label
        if sample_type == "POSITIVE":
            positive_texts[label].append(text)

    by_sentence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scores:
        by_sentence[row["sentence_id"]].append(row)

    output_rows: list[dict[str, object]] = []

    for sentence_id, rows in by_sentence.items():
        rows.sort(key=lambda x: int(x["embedding_rank"]))
        first = rows[0]

        rule_labels = set(split_labels(first.get("rule_labels", "")))
        top_labels = {
            row["candidate_label"]
            for row in rows
            if int(row["embedding_rank"]) <= args.top_k
        }
        union_labels = rule_labels | top_labels
        score_by_label = {row["candidate_label"]: row for row in rows}

        for label in sorted(
            union_labels,
            key=lambda x: int(score_by_label[x]["embedding_rank"])
            if x in score_by_label else 10**9,
        ):
            score_row = score_by_label.get(label)
            if score_row is None:
                continue

            examples = positive_texts.get(label, [])
            label_document = (
                f"标签代码：{label}\n"
                f"标签名称：{label_names.get(label, label)}\n"
                "正向语义示例：\n- " + "\n- ".join(examples)
            )

            source = []
            if label in rule_labels:
                source.append("RULE")
            if label in top_labels:
                source.append(f"EMBEDDING_TOP{args.top_k}")

            output_rows.append({
                "sentence_id": sentence_id,
                "candidate_label": label,
                "label_name": label_names.get(label, label),
                "candidate_source": "+".join(source),
                "is_rule_label": int(label in rule_labels),
                "is_embedding_topk": int(label in top_labels),
                "positive_similarity": score_row["positive_similarity"],
                "hard_negative_similarity": score_row["hard_negative_similarity"],
                "semantic_margin": score_row["semantic_margin"],
                "embedding_rank": score_row["embedding_rank"],
                "query_text": first["sentence_text"],
                "label_document": label_document,
                "company_id": first.get("company_id", ""),
                "company_name": first.get("company_name", ""),
                "company_attribution": first.get("company_attribution", ""),
                "status_code": first.get("status_code", ""),
                "rule_confidence": first.get("rule_confidence", ""),
                "section": first.get("section", ""),
                "source_file": first.get("source_file", ""),
                "pdf_page": first.get("pdf_page", ""),
            })

    fields = [
        "sentence_id", "candidate_label", "label_name", "candidate_source",
        "is_rule_label", "is_embedding_topk",
        "positive_similarity", "hard_negative_similarity",
        "semantic_margin", "embedding_rank",
        "query_text", "label_document",
        "company_id", "company_name", "company_attribution",
        "status_code", "rule_confidence", "section",
        "source_file", "pdf_page",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, fields, output_rows)
    print(f"Created: {output_path}")
    print(f"Reranker pairs: {len(output_rows)}")


if __name__ == "__main__":
    main()
