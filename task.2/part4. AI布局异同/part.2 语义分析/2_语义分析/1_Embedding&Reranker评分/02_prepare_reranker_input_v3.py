#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SPECIAL_POLICIES = {
    "TECH_RL": {
        "include": "仅在句子明确涉及强化学习、奖励信号、策略学习、环境反馈、策略优化或多智能体强化学习时纳入。",
        "exclude": "仅出现智能体、Agent、任务编排、自主执行、智能助手、工作流或大模型Agent，但没有奖励、策略或环境反馈证据时排除。",
    },
    "TECH_ML": {
        "include": "纳入明确的机器学习、监督学习、无监督学习、分类、回归、聚类、预测、异常检测或传统深度学习建模。",
        "exclude": "仅出现大模型、基础模型、生成式AI或LLM产品时排除TECH_ML；此时优先判断TECH_LLM，避免重复计数。",
    },
    "AI_ENGINEERING": {
        "include": "仅纳入MLOps、模型训练平台、部署、推理服务、模型监控、特征平台、AI中台、模型管理或训练推理基础设施。",
        "exclude": "普通AI平台、产品矩阵、技术体系、数据平台、业务平台，若没有研发部署基础设施证据则排除。",
    },
    "TECH_PRIVACY": {
        "include": "仅纳入隐私计算、联邦学习、多方安全计算、可信执行环境、隐私保护联合建模或同态加密。",
        "exclude": "普通数据安全、传输加密、权限管理、数据治理、机构合作、反欺诈或普通模型训练不属于该标签。",
    },
    "AI_GENERAL": {
        "include": "仅判断句子是否总体直接涉及人工智能披露。",
        "exclude": "该标签不能作为具体技术栈或产品线标签。",
    },
}

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

def split_labels(value: str):
    return [x.strip() for x in (value or "").split(";") if x.strip()]

def make_query(label, name, positives, negatives):
    policy = SPECIAL_POLICIES.get(label, {})
    include = policy.get("include", "仅在句子直接表达该标签对应的技术、产品或应用语义时纳入。")
    exclude = policy.get("exclude", "行业背景、第三方行为、普通数字化、弱宣传、未来关注和否定表达均排除。")
    pos = "\n".join(f"- {x}" for x in positives) or "- 无"
    neg = "\n".join(f"- {x}" for x in negatives) or "- 无"
    return (
        f"候选标签：{label}（{name}）\n"
        f"纳入条件：{include}\n"
        f"排除条件：{exclude}\n"
        f"正向语义示例：\n{pos}\n"
        f"难负例：\n{neg}\n"
        "只判断年报句子是否直接支持该标签，不要推断句中未明确陈述的能力。"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-scores", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--output", default="reranker_results/reranker_input_v3.csv")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    _, scores = read_csv(Path(args.embedding_scores))
    _, prototypes = read_csv(Path(args.prototypes))

    positives = defaultdict(list)
    negatives = defaultdict(list)
    label_names = {}

    for row in prototypes:
        label = (row.get("标准标签代码") or "").strip()
        text = (row.get("原型句") or "").strip()
        kind = (row.get("样本类型") or "").strip().upper()
        name = (row.get("标签名称") or label).strip() or label
        if not label or not text:
            continue
        label_names[label] = name
        if kind == "POSITIVE":
            positives[label].append(text)
        elif kind == "HARD_NEGATIVE":
            negatives[label].append(text)

    grouped = defaultdict(list)
    for row in scores:
        grouped[row["sentence_id"]].append(row)

    output = []

    for sentence_id, rows in grouped.items():
        rows.sort(key=lambda x: int(x["embedding_rank"]))
        first = rows[0]
        rule_labels = set(split_labels(first.get("rule_labels", "")))
        top_labels = {x["candidate_label"] for x in rows if int(x["embedding_rank"]) <= args.top_k}
        union = rule_labels | top_labels
        by_label = {x["candidate_label"]: x for x in rows}

        for label in sorted(union, key=lambda x: int(by_label[x]["embedding_rank"])):
            score = by_label[label]
            sources = []
            if label in rule_labels:
                sources.append("RULE")
            if label in top_labels:
                sources.append(f"EMBEDDING_TOP{args.top_k}")

            output.append({
                "sentence_id": sentence_id,
                "candidate_label": label,
                "label_name": label_names.get(label, label),
                "candidate_source": "+".join(sources),
                "is_rule_label": int(label in rule_labels),
                "is_embedding_topk": int(label in top_labels),
                "positive_similarity": score["positive_similarity"],
                "hard_negative_similarity": score["hard_negative_similarity"],
                "semantic_margin": score["semantic_margin"],
                "embedding_rank": score["embedding_rank"],
                "query_text": make_query(
                    label,
                    label_names.get(label, label),
                    positives.get(label, []),
                    negatives.get(label, []),
                ),
                "document_text": first["sentence_text"],
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
        "is_rule_label", "is_embedding_topk", "positive_similarity",
        "hard_negative_similarity", "semantic_margin", "embedding_rank",
        "query_text", "document_text", "company_id", "company_name",
        "company_attribution", "status_code", "rule_confidence", "section",
        "source_file", "pdf_page",
    ]
    write_csv(Path(args.output), fields, output)
    print(f"Created: {args.output}")
    print(f"Sentences: {len({x['sentence_id'] for x in output})}")
    print(f"Reranker pairs: {len(output)}")

if __name__ == "__main__":
    main()
