# -*- coding: utf-8 -*-
"""
task4 / 01_job_classification.py
------------------------------------------------------------------
阶段 1-2：数据准备 + 岗位名称标准化与岗位分类

输入：第三部分结构化招聘数据 company_jobs_merged.csv (GB18030)
输出：
    data/01_jobs_classified.csv          每条招聘记录 + 标准岗位名 + 岗位大类
    data/01_title_normalize_map.csv      原始职位名 -> 标准职位名 映射表（中间产物）
    data/01_category_summary.csv         岗位大类规模统计
    data/01_classification_log.json      清洗与分类过程日志（可复现证据）

方法：
    规则命中（有序关键词词典） + TF-IDF 字符 n-gram 余弦相似度兜底
    —— 无外网环境下，用字符级 n-gram 向量作为"轻量语义嵌入"的替代实现，
       对未被规则命中的长尾职位名，与各大类原型文本比对，取最相似大类。
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from config.taxonomy import (JOB_CATEGORY_RULES, FALLBACK_CATEGORY, CATEGORY_ORDER,
                             TITLE_NORMALIZE_MAP, TITLE_PREFIX_TOKENS,
                             TITLE_SUFFIX_TOKENS, JD_STRICT_KEYWORDS)

BASE = Path(__file__).parent

# 原始数据路径：默认相对本脚本所在目录，可用环境变量 JOBS_RAW_CSV 覆盖。
# 不再使用任何绝对路径，保证换机器可直接运行。
import os

RAW = Path(os.environ.get("JOBS_RAW_CSV", BASE / "raw_data" / "company_jobs_merged.csv"))
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)

if not RAW.exists():
    raise FileNotFoundError(
        f"未找到原始招聘数据：{RAW}\n"
        f"请将第三部分产出的 company_jobs_merged.csv 放到 {BASE / 'raw_data'} 目录下，"
        f"或设置环境变量 JOBS_RAW_CSV 指向该文件。"
    )

LOG = {}


# ---------------------------------------------------------------- 1. 数据准备
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW, encoding="gb18030")
    LOG["raw_rows"] = len(df)
    LOG["raw_companies"] = int(df["official_name"].nunique())

    # 1) 去除完全重复行 & 同一职位链接重复抓取
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["job_link"], keep="first")
    LOG["after_link_dedup"] = len(df)

    # 2) 业务去重：同公司 + 同职位名 + 同城市 + 同薪资，视为同一岗位的多次挂出
    df["city"] = df["location"].astype(str).str.split("·").str[0].str.strip()
    df = df.drop_duplicates(subset=["company_name", "title", "city", "salary"], keep="first")
    LOG["after_business_dedup"] = len(df)

    # 3) 剔除劳务派遣岗（口径说明：派遣岗多为外包运维，会污染岗位画像）
    n_dispatch = int((df["is_dispatch"] == "是").sum())
    df = df[df["is_dispatch"] != "是"].copy()
    LOG["dropped_dispatch"] = n_dispatch

    # 4) 文本字段兜底填充
    for col in ["skills", "welfare", "jd", "experience", "degree"]:
        df[col] = df[col].fillna("")

    # 5) 构造分析用文本：职位名 + 技能标签 + JD（第三部分抽取字段的原料）
    df["text_all"] = (df["title"].astype(str) + "\n" + df["skills"].astype(str)
                      + "\n" + df["jd"].astype(str)).str.lower()
    LOG["final_rows"] = len(df)
    LOG["final_companies"] = int(df["official_name"].nunique())
    return df.reset_index(drop=True)


# ------------------------------------------------------- 2. 岗位名称标准化
def normalize_title(raw: str) -> str:
    """把 2000+ 个五花八门的原始职位名收敛成标准岗位名。"""
    s = str(raw).lower().strip()
    # 去掉括号内的补充说明、编号、地点后缀
    s = re.sub(r"[（(\[【].*?[)）\]】]", "", s)
    s = re.sub(r"[（(\[【].*$", "", s)          # 未闭合括号
    s = re.sub(r"[a-z]{1,3}\d{4,}", "", s)      # 岗位编号 MJ002455
    s = re.sub(r"[-—_/、,，|]+.*$", "", s) if len(s) > 12 else s
    s = re.sub(r"\s+", "", s)
    # 剥离职级/招聘修饰词：前缀只切词首，后缀只切词尾，保护"金融实习生"这类词
    changed = True
    while changed:
        changed = False
        for tok in TITLE_PREFIX_TOKENS:
            if s.startswith(tok) and len(s) - len(tok) >= 2:
                s, changed = s[len(tok):], True
        for tok in TITLE_SUFFIX_TOKENS:
            if s.endswith(tok) and len(s) - len(tok) >= 2:
                s, changed = s[:-len(tok)], True
    s = s.strip("·-/ ")
    if not s:
        s = str(raw).lower().strip()
    # 同义词归并
    s = TITLE_NORMALIZE_MAP.get(s, s)
    return s


# ------------------------------------------------------------- 3. 岗位分类
def rule_classify(std_title: str, text_all: str):
    """有序规则命中：返回 (大类, 命中依据, 命中层级)。"""
    for cat, kws, excludes in JOB_CATEGORY_RULES:
        if any(ex in std_title for ex in excludes):
            continue
        for kw in kws:
            if kw in std_title:
                return cat, kw, "title"
    # 职位名未命中 -> 用高精度词典在 JD 前 400 字内兜底（避免整段 JD 噪声）
    snippet = text_all[:400]
    for cat, _, _ in JOB_CATEGORY_RULES:
        for kw in JD_STRICT_KEYWORDS.get(cat, []):
            if kw in snippet:
                return cat, kw, "jd"
    return None, None, None


def build_prototypes(df: pd.DataFrame) -> dict:
    """用已被 title 规则高置信命中的样本，为每个大类构造原型文本。"""
    proto = {}
    for cat in CATEGORY_ORDER:
        sub = df[(df["category_rule"] == cat) & (df["match_level"] == "title")]
        if len(sub) == 0:
            continue
        proto[cat] = " ".join(sub["std_title"].tolist()[:400])
    return proto


def similarity_backfill(df: pd.DataFrame, prototypes: dict) -> pd.DataFrame:
    """对规则未命中的长尾职位，用字符 n-gram TF-IDF 余弦相似度归类。"""
    mask = df["category_rule"].isna()
    LOG["rule_unmatched"] = int(mask.sum())
    if mask.sum() == 0:
        df["category"] = df["category_rule"]
        df["assign_method"] = "rule"
        return df

    cats = list(prototypes.keys())
    corpus = [prototypes[c] for c in cats] + df.loc[mask, "std_title"].tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(corpus)
    sim = cosine_similarity(X[len(cats):], X[:len(cats)])

    best_idx = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)
    SIM_THRESHOLD = 0.12
    assigned = [cats[i] if s >= SIM_THRESHOLD else FALLBACK_CATEGORY
                for i, s in zip(best_idx, best_sim)]

    df["category"] = df["category_rule"]
    df["assign_method"] = "rule"
    df.loc[mask, "category"] = assigned
    df.loc[mask, "assign_method"] = ["embedding_sim" if s >= SIM_THRESHOLD else "fallback"
                                     for s in best_sim]
    df.loc[mask, "sim_score"] = best_sim.round(4)
    LOG["sim_assigned"] = int((pd.Series(assigned) != FALLBACK_CATEGORY).sum())
    LOG["fallback_assigned"] = int((pd.Series(assigned) == FALLBACK_CATEGORY).sum())
    return df


# ------------------------------------------------------------------- main
def main():
    print("[1/4] 读取并清洗第三部分招聘数据 ...")
    df = load_and_clean()
    print(f"      {LOG['raw_rows']} -> {LOG['final_rows']} 条，"
          f"{LOG['final_companies']} 家上市公司")

    print("[2/4] 岗位名称标准化 ...")
    df["std_title"] = df["title"].map(normalize_title)
    LOG["unique_title_before"] = int(df["title"].nunique())
    LOG["unique_title_after"] = int(df["std_title"].nunique())
    print(f"      唯一职位名 {LOG['unique_title_before']} -> {LOG['unique_title_after']}")

    print("[3/4] 规则分类 + 相似度兜底 ...")
    res = df.apply(lambda r: rule_classify(r["std_title"], r["text_all"]),
                   axis=1, result_type="expand")
    df[["category_rule", "match_keyword", "match_level"]] = res
    LOG["rule_matched"] = int(df["category_rule"].notna().sum())
    df["sim_score"] = pd.NA
    prototypes = build_prototypes(df)
    df = similarity_backfill(df, prototypes)
    print(f"      规则命中 {LOG['rule_matched']} 条，"
          f"相似度归类 {LOG.get('sim_assigned', 0)} 条，"
          f"兜底 {LOG.get('fallback_assigned', 0)} 条")

    print("[4/4] 输出中间产物 ...")
    keep = ["ts_code", "official_name", "company_name", "company_industry",
            "company_scale", "title", "std_title", "category", "assign_method",
            "match_keyword", "match_level", "sim_score", "salary", "location",
            "city", "experience", "degree", "skills", "welfare", "jd", "job_link"]
    df[keep].to_csv(OUT / "01_jobs_classified.csv", index=False, encoding="utf-8-sig")

    # 名称标准化映射表（抽样可核查）
    mp = (df.groupby(["title", "std_title", "category"])
            .size().reset_index(name="n").sort_values("n", ascending=False))
    mp.to_csv(OUT / "01_title_normalize_map.csv", index=False, encoding="utf-8-sig")

    summary = (df.groupby("category")
                 .agg(岗位数=("title", "size"),
                      公司数=("official_name", "nunique"),
                      唯一职位名数=("std_title", "nunique"))
                 .reindex(CATEGORY_ORDER).dropna(how="all"))
    summary["占比%"] = (summary["岗位数"] / summary["岗位数"].sum() * 100).round(2)
    summary = summary.sort_values("岗位数", ascending=False)
    summary.to_csv(OUT / "01_category_summary.csv", encoding="utf-8-sig")

    with open(OUT / "01_classification_log.json", "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=2)

    print("\n=== 岗位大类分布 ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
