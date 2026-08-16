#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PRIORITY_LABELS = {
    "AI_GENERAL",
    "PRODUCT_PAYMENT",
    "PRODUCT_FINANCE",
    "AI_ENGINEERING",
}

def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)

def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def split_labels(raw: str):
    return [x.strip() for x in (raw or "").split(";") if x.strip()]

def build_label_document(label, label_name, positives, negatives):
    pos = "\n".join(f"- {x}" for x in positives) or "- 无"
    neg = "\n".join(f"- {x}" for x in negatives) or "- 无"
    extra = ""
    if label == "AI_GENERAL":
        extra = "\n特殊边界：该标签仅表示总体涉及AI，不能据此认定具体技术栈或产品线。"
    return (
        "判断目标：判断一条公司年报句子是否直接支持下面的标签。\n"
        f"标签代码：{label}\n"
        f"标签名称：{label_name}\n"
        "纳入条件：句子应明确表达与该标签一致的技术、产品或应用语义；"
        "不要仅因出现平台、模型、数据、智能、金融机构等通用词就判为相关。\n"
        "正向语义示例：\n"
        f"{pos}\n"
        "排除条件：行业趋势、第三方行为、普通数字化、弱宣传、未来关注、"
        "否定表达，以及仅与下列难负例相似的内容，不应判为该标签。\n"
        "难负例：\n"
        f"{neg}"
        f"{extra}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-scores", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--output", default="reranker_results/reranker_input.csv")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    _, score_rows = read_csv(Path(args.embedding_scores))
    _, prototype_rows = read_csv(Path(args.prototypes))

    positives = defaultdict(list)
    negatives = defaultdict(list)
    label_names = {}

    for row in prototype_rows:
        label = (row.get("标准标签代码") or "").strip()
        name = (row.get("标签名称") or label).strip() or label
        kind = (row.get("样本类型") or "").strip().upper()
        text = (row.get("原型句") or "").strip()
        if not label or not text:
            continue
        label_names[label] = name
        if kind == "POSITIVE":
            positives[label].append(text)
        elif kind == "HARD_NEGATIVE":
            negatives[label].append(text)

    by_sentence = defaultdict(list)
    for row in score_rows:
        by_sentence[row["sentence_id"]].append(row)

    output_rows = []

    for sentence_id, rows in by_sentence.items():
        rows.sort(key=lambda r: int(r["embedding_rank"]))
        first = rows[0]
        rule_labels = set(split_labels(first.get("rule_labels", "")))
        top_labels = {
            r["candidate_label"]
            for r in rows
            if int(r["embedding_rank"]) <= args.top_k
        }
        union_labels = rule_labels | top_labels
        score_by_label = {r["candidate_label"]: r for r in rows}

        for label in sorted(
            union_labels,
            key=lambda x: int(score_by_label[x]["embedding_rank"])
            if x in score_by_label else 10**9,
        ):
            score = score_by_label.get(label)
            if score is None:
                continue

            is_rule = int(label in rule_labels)
            is_topk = int(label in top_labels)

            sources = []
            if is_rule:
                sources.append("RULE")
            if is_topk:
                sources.append(f"EMBEDDING_TOP{args.top_k}")

            reasons = []
            if label in PRIORITY_LABELS:
                reasons.append("PRIORITY_LABEL")
            if first.get("company_attribution", "") == "UNKNOWN":
                reasons.append("UNKNOWN_ATTRIBUTION")
            if first.get("rule_confidence", "").upper() in {"LOW", "WEAK"}:
                reasons.append("WEAK_RULE")

            output_rows.append({
                "sentence_id": sentence_id,
                "candidate_label": label,
                "label_name": label_names.get(label, label),
                "candidate_source": "+".join(sources),
                "is_rule_label": is_rule,
                "is_embedding_topk": is_topk,
                "positive_similarity": score["positive_similarity"],
                "hard_negative_similarity": score["hard_negative_similarity"],
                "semantic_margin": score["semantic_margin"],
                "embedding_rank": score["embedding_rank"],
                "query_text": first["sentence_text"],
                "label_document": build_label_document(
                    label,
                    label_names.get(label, label),
                    positives.get(label, []),
                    negatives.get(label, []),
                ),
                "priority_review": int(bool(reasons)),
                "priority_reason": ";".join(reasons),
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
        "priority_review", "priority_reason",
        "company_id", "company_name", "company_attribution",
        "status_code", "rule_confidence", "section",
        "source_file", "pdf_page",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, fields, output_rows)

    print(f"Created: {output_path}")
    print(f"Sentences: {len({r['sentence_id'] for r in output_rows})}")
    print(f"Reranker pairs: {len(output_rows)}")
    print(f"Priority-review pairs: {sum(int(r['priority_review']) for r in output_rows)}")

if __name__ == "__main__":
    main()
