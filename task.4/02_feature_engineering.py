# -*- coding: utf-8 -*-
"""
task4 / 02_feature_engineering.py
------------------------------------------------------------------
阶段 3：构建岗位特征指标（岗位特征矩阵）

输入：data/01_jobs_classified.csv
输出：
    data/02_job_level_features.csv       岗位级特征（每条招聘记录的五维证据，中间产物）
    data/02_category_feature_matrix.csv  岗位大类级特征矩阵（模块 3 的直接输入）
    data/02_skill_frequency.csv          高频硬技能 / 软技能 / AI 技术栈词频（中间产物）
    data/02_feature_log.json             特征构建过程日志

核心思想：让五维评分"尽可能数据化"
    R 任务重复性     <- JD 中重复性动作词覆盖率、创造性词覆盖率（反向）、学历/经验门槛
    S 规则化程度     <- JD 中制度/流程/准则类词覆盖率
    D 数据数字化程度 <- hard_skills 中 Python/SQL/Excel/数据库/BI 等出现比例
    A AI 技术成熟度  <- 该类岗位中出现 AI 技术要求的岗位占比（AI 技术渗透度）
    H 人类不可替代   <- soft_skills 中沟通/谈判/领导/协调/客户关系/决策等出现比例
                        + 高经验门槛占比（经验壁垒）
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config.taxonomy import (AI_SKILL_KEYWORDS, DIGITAL_SKILL_KEYWORDS,
                             HUMAN_SKILL_KEYWORDS, REPETITIVE_KEYWORDS,
                             CREATIVE_KEYWORDS, RULE_BASED_KEYWORDS,
                             CALIBRATION_RANGE, CATEGORY_ORDER)

BASE = Path(__file__).parent
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)
LOG = {}

DEGREE_LEVEL = {"学历不限": 1, "中专/中技": 1, "高中": 1, "大专": 2,
                "本科": 3, "硕士": 4, "博士": 5}
EXP_LEVEL = {"经验不限": 1, "在校/应届": 1, "1年以内": 1, "1-3年": 2,
             "3-5年": 3, "5-10年": 4, "10年以上": 5}


# ------------------------------------------------------------ 工具函数
def parse_salary(s: str):
    """把 '10-15K·13薪' / '300-500元/天' 统一折算为月薪中位数（千元）。"""
    s = str(s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[kK]", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        months = 12.0
        mm = re.search(r"·\s*(\d+)\s*薪", s)
        if mm:
            months = float(mm.group(1))
        return (lo + hi) / 2 * months / 12
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*元/天", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 * 21.75 / 1000
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*元/月", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 / 1000
    return np.nan


def hit_keywords(text: str, kws) -> list:
    """返回文本命中的关键词列表（用于保留可核查证据）。"""
    t = str(text).lower()
    return [k for k in kws if k in t]


def calibrate(series: pd.Series, lo=CALIBRATION_RANGE[0], hi=CALIBRATION_RANGE[1]) -> pd.Series:
    """组内相对标定：把原始比例值 min-max 映射到 [lo, hi]。

    说明：原始覆盖率天然偏低（例如仅 15% 的岗位明确写 Python），
    直接 ×100 会让所有维度都挤在低分区、失去区分度。
    这里做的是"岗位大类之间的相对位置"标定，符合替代风险本身
    只在岗位之间横向可比的性质。
    """
    v = series.astype(float)
    if v.max() - v.min() < 1e-9:
        return pd.Series(np.full(len(v), (lo + hi) / 2), index=v.index)
    return lo + (v - v.min()) / (v.max() - v.min()) * (hi - lo)


# ------------------------------------------------------------ 岗位级特征
def build_job_features(df: pd.DataFrame) -> pd.DataFrame:
    # 技能字段：优先用平台结构化 skills 标签，JD 作为补充语料
    df["skills_text"] = df["skills"].fillna("").astype(str).str.lower()
    df["jd_text"] = df["jd"].fillna("").astype(str).str.lower()
    df["full_text"] = (df["title"].astype(str).str.lower() + " "
                       + df["skills_text"] + " " + df["jd_text"])

    dims = {
        "ai": AI_SKILL_KEYWORDS,
        "digital": DIGITAL_SKILL_KEYWORDS,
        "human": HUMAN_SKILL_KEYWORDS,
        "repeat": REPETITIVE_KEYWORDS,
        "creative": CREATIVE_KEYWORDS,
        "rule": RULE_BASED_KEYWORDS,
    }
    for name, kws in dims.items():
        hits = df["full_text"].map(lambda t, k=kws: hit_keywords(t, k))
        df[f"{name}_hits"] = hits.map(lambda x: "|".join(x))
        df[f"n_{name}"] = hits.map(len)
        df[f"has_{name}"] = (df[f"n_{name}"] > 0).astype(int)

    # 结构性门槛特征
    df["degree_level"] = df["degree"].map(DEGREE_LEVEL).fillna(3)
    df["exp_level"] = df["experience"].map(EXP_LEVEL).fillna(2)
    df["low_barrier"] = ((df["degree_level"] <= 2) | (df["exp_level"] <= 1)).astype(int)
    df["senior_barrier"] = ((df["exp_level"] >= 4) | (df["degree_level"] >= 4)).astype(int)
    df["salary_k"] = df["salary"].map(parse_salary)

    # 岗位级"AI 技术栈"字段（对齐第三部分的 ai_skills 口径）
    df["ai_skills"] = df["ai_hits"]
    df["hard_skills_digital"] = df["digital_hits"]
    df["soft_skills_human"] = df["human_hits"]
    return df


# ---------------------------------------------------- 岗位大类级特征矩阵
def build_category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("category")
    m = pd.DataFrame({
        "岗位数": g.size(),
        "公司数": g["official_name"].nunique(),
        "薪资中位数K": g["salary_k"].median().round(2),
        "平均学历层级": g["degree_level"].mean().round(2),
        "平均经验层级": g["exp_level"].mean().round(2),
        # 覆盖率：该类岗位中出现该类关键词的岗位占比
        "AI渗透度": g["has_ai"].mean(),
        "数字化技能覆盖率": g["has_digital"].mean(),
        "人际能力覆盖率": g["has_human"].mean(),
        "重复性任务覆盖率": g["has_repeat"].mean(),
        "创造性任务覆盖率": g["has_creative"].mean(),
        "规则化表述覆盖率": g["has_rule"].mean(),
        # 强度：平均命中关键词数（覆盖率之外的第二重证据）
        "AI关键词强度": g["n_ai"].mean(),
        "数字化关键词强度": g["n_digital"].mean(),
        "人际关键词强度": g["n_human"].mean(),
        "重复性关键词强度": g["n_repeat"].mean(),
        "规则化关键词强度": g["n_rule"].mean(),
        "低门槛岗位占比": g["low_barrier"].mean(),
        "高经验壁垒占比": g["senior_barrier"].mean(),
    })

    # ---------- 五维"数据证据分"原始值（0-1 区间的合成指标）----------
    def z(col):  # 强度指标先做 0-1 归一，避免量纲差异
        v = m[col]
        return (v - v.min()) / (v.max() - v.min() + 1e-9)

    m["R_raw"] = (0.40 * m["重复性任务覆盖率"]
                  + 0.20 * z("重复性关键词强度")
                  + 0.20 * (1 - m["创造性任务覆盖率"])
                  + 0.20 * m["低门槛岗位占比"])
    m["S_raw"] = 0.70 * m["规则化表述覆盖率"] + 0.30 * z("规则化关键词强度")
    m["D_raw"] = 0.70 * m["数字化技能覆盖率"] + 0.30 * z("数字化关键词强度")
    m["A_raw"] = 0.70 * m["AI渗透度"] + 0.30 * z("AI关键词强度")
    m["H_raw"] = (0.55 * m["人际能力覆盖率"]
                  + 0.20 * z("人际关键词强度")
                  + 0.25 * m["高经验壁垒占比"])

    # ---------- 组内相对标定到 [10, 90] ----------
    for d in ["R", "S", "D", "A", "H"]:
        m[f"{d}_data"] = calibrate(m[f"{d}_raw"]).round(2)

    m = m.reindex([c for c in CATEGORY_ORDER if c in m.index])
    return m


def main():
    print("[1/3] 读取岗位分类结果 ...")
    df = pd.read_csv(OUT / "01_jobs_classified.csv")
    LOG["n_jobs"] = len(df)

    print("[2/3] 抽取岗位级五维证据特征 ...")
    df = build_job_features(df)
    LOG["ai_job_ratio_overall"] = round(float(df["has_ai"].mean()), 4)
    LOG["digital_job_ratio_overall"] = round(float(df["has_digital"].mean()), 4)
    LOG["human_job_ratio_overall"] = round(float(df["has_human"].mean()), 4)
    LOG["salary_parsed_ratio"] = round(float(df["salary_k"].notna().mean()), 4)

    cols = ["ts_code", "official_name", "title", "std_title", "category", "city",
            "salary", "salary_k", "degree", "degree_level", "experience", "exp_level",
            "low_barrier", "senior_barrier",
            "n_ai", "has_ai", "ai_skills",
            "n_digital", "has_digital", "hard_skills_digital",
            "n_human", "has_human", "soft_skills_human",
            "n_repeat", "has_repeat", "repeat_hits",
            "n_creative", "has_creative", "creative_hits",
            "n_rule", "has_rule", "rule_hits"]
    df[cols].to_csv(OUT / "02_job_level_features.csv", index=False, encoding="utf-8-sig")

    print("[3/3] 汇总岗位大类特征矩阵 ...")
    m = build_category_matrix(df)
    m.round(4).to_csv(OUT / "02_category_feature_matrix.csv", encoding="utf-8-sig")

    # 技能词频（用于报告里的证据展示）
    rows = []
    for dim, col in [("AI技术栈", "ai_skills"), ("硬技能-数字化", "hard_skills_digital"),
                     ("软技能-人际", "soft_skills_human")]:
        cnt = (df[col].fillna("").str.split("|").explode()
               .replace("", np.nan).dropna().value_counts())
        for k, v in cnt.head(30).items():
            rows.append({"维度": dim, "关键词": k, "出现岗位数": int(v),
                         "占全部岗位%": round(v / len(df) * 100, 2)})
    pd.DataFrame(rows).to_csv(OUT / "02_skill_frequency.csv", index=False, encoding="utf-8-sig")

    with open(OUT / "02_feature_log.json", "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=2)

    print("\n=== 岗位大类特征矩阵（关键列）===")
    show = m[["岗位数", "AI渗透度", "数字化技能覆盖率", "人际能力覆盖率",
              "重复性任务覆盖率", "规则化表述覆盖率",
              "R_data", "S_data", "D_data", "A_data", "H_data"]]
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
