# -*- coding: utf-8 -*-
"""
task4 / 03_risk_scoring.py
------------------------------------------------------------------
阶段 4-5：AI 替代风险指标体系 -> 计算风险得分 -> 风险分层

输入：data/02_category_feature_matrix.csv, data/02_job_level_features.csv
输出：
    data/03_risk_scores_category.csv   岗位大类风险总表（五维 + Risk + 分层）
    data/03_risk_scores_subjob.csv     细分岗位风险表（典型岗位排名图的数据源）
    data/03_dimension_detail.csv       五维"数据分/先验分/融合分"三列对照（中间产物）
    data/03_sensitivity.csv            权重敏感性分析（稳健性检验）
    data/03_cluster_assignment.csv     KMeans 聚类分层（与阈值分层交叉验证）
    data/03_scoring_log.json           评分过程日志

模型
----
Risk_j = 0.25*R_j + 0.20*S_j + 0.20*D_j + 0.20*A_j + 0.15*(100 - H_j)

其中每个维度分由"招聘数据证据分"与"专家先验分"融合：
    X_j = w_data(n_j) * X_data_j + (1 - w_data(n_j)) * X_expert_j

样本量收缩（shrinkage）：
    w_data(n) = w_base * n / (n + K),  K = 30
    —— 岗位数越少，文本证据越不稳定，权重越向专家先验收缩。
       例如信贷与授信审批仅 8 条样本，其数据分权重被压到基准的 21%。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from config.taxonomy import (EXPERT_PRIOR, EXPERT_PRIOR_RATIONALE, RISK_WEIGHTS,
                             BLEND_WEIGHTS, RISK_TIERS, CATEGORY_ORDER,
                             FINANCE_CORE_CATEGORIES, CALIBRATION_RANGE,
                             TYPICAL_FINANCE_ROLES)

BASE = Path(__file__).parent
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)
LOG = {}
DIMS = ["R", "S", "D", "A", "H"]
SHRINK_K = 30          # 收缩强度：样本量等于 K 时，数据权重减半
MIN_SUBJOB_N = 5       # 细分岗位（std_title 口径）进入排名的最小样本量
MIN_ROLE_N = 5         # 典型职能岗进入排名的最小样本量


# ------------------------------------------------------------- 评分核心
def risk_formula(row) -> float:
    return (RISK_WEIGHTS["R"] * row["R"] + RISK_WEIGHTS["S"] * row["S"]
            + RISK_WEIGHTS["D"] * row["D"] + RISK_WEIGHTS["A"] * row["A"]
            + RISK_WEIGHTS["H"] * (100 - row["H"]))


def tier_of(score: float) -> str:
    for lo, hi, name in RISK_TIERS:
        if lo <= score < hi or (hi == 100 and score >= 100):
            return name
    return RISK_TIERS[0][2] if score >= 70 else RISK_TIERS[-1][2]


def blend(m: pd.DataFrame) -> pd.DataFrame:
    """数据证据分 + 专家先验分 融合（含样本量收缩）。"""
    detail = []
    for d in DIMS:
        base_w = BLEND_WEIGHTS[d]["data"]
        n = m["岗位数"].astype(float)
        w_data = base_w * n / (n + SHRINK_K)          # 收缩后的数据权重
        expert = m.index.map(lambda c: EXPERT_PRIOR[c][d]).astype(float)
        m[f"{d}_expert"] = expert
        m[f"{d}_weight_data"] = w_data.round(3)
        m[d] = (w_data * m[f"{d}_data"] + (1 - w_data) * expert).round(2)
        for c in m.index:
            detail.append({"岗位大类": c, "维度": d,
                           "数据证据分": m.loc[c, f"{d}_data"],
                           "专家先验分": m.loc[c, f"{d}_expert"],
                           "数据权重": m.loc[c, f"{d}_weight_data"],
                           "融合分": m.loc[c, d]})
    return m, pd.DataFrame(detail)


# --------------------------------------------------- 细分岗位（std_title）评分
def score_subjobs(jobs: pd.DataFrame, cat_scores: pd.DataFrame) -> pd.DataFrame:
    """对高频细分岗位打分：以所属大类分为基线，用岗位自身文本证据做偏移。

    偏移量 = (该细分岗位证据分 - 所属大类平均证据分) * 调整系数，
    并按样本量收缩，保证小样本细分岗位不会因个别 JD 用词而剧烈跳动。
    """
    grp = jobs.groupby(["category", "std_title"])
    agg = grp.agg(
        岗位数=("title", "size"),
        公司数=("official_name", "nunique"),
        薪资中位数K=("salary_k", "median"),
        AI渗透度=("has_ai", "mean"),
        数字化覆盖率=("has_digital", "mean"),
        人际覆盖率=("has_human", "mean"),
        重复性覆盖率=("has_repeat", "mean"),
        创造性覆盖率=("has_creative", "mean"),
        规则化覆盖率=("has_rule", "mean"),
        低门槛占比=("low_barrier", "mean"),
        高经验壁垒占比=("senior_barrier", "mean"),
    ).reset_index()
    agg = agg[agg["岗位数"] >= MIN_SUBJOB_N].copy()

    # 大类内部均值，用于计算相对偏移
    src_map = {
        "R": ("重复性覆盖率", +1), "S": ("规则化覆盖率", +1),
        "D": ("数字化覆盖率", +1), "A": ("AI渗透度", +1), "H": ("人际覆盖率", +1),
    }
    cat_mean = jobs.groupby("category").agg(
        重复性覆盖率=("has_repeat", "mean"), 规则化覆盖率=("has_rule", "mean"),
        数字化覆盖率=("has_digital", "mean"), AI渗透度=("has_ai", "mean"),
        人际覆盖率=("has_human", "mean"), 创造性覆盖率=("has_creative", "mean"),
        低门槛占比=("low_barrier", "mean"),
    )

    ADJ = 35.0   # 偏移幅度上限系数（覆盖率相差 1.0 时最多偏移 35 分）
    for d in DIMS:
        col, sign = src_map[d]
        base = agg["category"].map(cat_scores[d])
        diff = agg[col].values - agg["category"].map(cat_mean[col]).values
        w = agg["岗位数"] / (agg["岗位数"] + 15)     # 细分岗位样本量收缩
        agg[d] = (base + sign * diff * ADJ * w).clip(5, 95).round(2)

    # 重复性额外考虑低门槛占比（录入类岗位的重要特征）
    diff_bar = agg["低门槛占比"].values - agg["category"].map(cat_mean["低门槛占比"]).values
    agg["R"] = (agg["R"] + diff_bar * 12).clip(5, 95).round(2)

    agg["Risk"] = agg.apply(risk_formula, axis=1).round(1)
    agg["风险等级"] = agg["Risk"].map(tier_of)
    agg["是否金融核心岗"] = agg["category"].isin(FINANCE_CORE_CATEGORIES)
    return agg.sort_values("Risk", ascending=False).reset_index(drop=True)


# ------------------------------------------------- 典型金融职能岗评分
def score_typical_roles(jobs: pd.DataFrame, cat_scores: pd.DataFrame):
    """按"任务导向的职能岗"口径评分（细分岗位排名图的数据源）。

    每个职能岗以其锚定大类的五维融合分为基线，再用该职能岗自身
    招聘文本证据相对锚定大类均值的偏移做调整；偏移按样本量收缩。
    """
    t = jobs["title"].astype(str).str.lower()
    used = pd.Series(False, index=jobs.index)
    src_map = {"R": "has_repeat", "S": "has_rule", "D": "has_digital",
               "A": "has_ai", "H": "has_human"}
    cat_mean = jobs.groupby("category")[list(src_map.values())
                                        + ["low_barrier"]].mean()
    ADJ, dropped, rows = 35.0, [], []

    for name, pat, exc, anchor in TYPICAL_FINANCE_ROLES:
        m = t.str.contains(pat, regex=True, na=False) & (~used)
        if exc:
            m &= ~t.str.contains(exc, regex=True, na=False)
        used |= m
        sub = jobs[m]
        if len(sub) < MIN_ROLE_N:
            dropped.append({"职能岗": name, "样本量": int(len(sub)),
                            "处理": f"样本量不足{MIN_ROLE_N}，不纳入排名"})
            continue
        rec = {"职能岗": name, "锚定大类": anchor, "岗位数": int(len(sub)),
               "公司数": int(sub["official_name"].nunique()),
               "薪资中位数K": round(float(sub["salary_k"].median()), 2)
               if sub["salary_k"].notna().any() else np.nan,
               "样本占比%": round(len(sub) / len(jobs) * 100, 2)}
        w = len(sub) / (len(sub) + 15)          # 样本量收缩
        for d, col in src_map.items():
            base = cat_scores.loc[anchor, d]
            diff = float(sub[col].mean()) - float(cat_mean.loc[anchor, col])
            rec[d] = round(float(np.clip(base + diff * ADJ * w, 5, 95)), 2)
            rec[f"{d}_证据覆盖率"] = round(float(sub[col].mean()), 3)
        diff_bar = float(sub["low_barrier"].mean()) - float(cat_mean.loc[anchor, "low_barrier"])
        rec["R"] = round(float(np.clip(rec["R"] + diff_bar * 12 * w, 5, 95)), 2)
        rec["Risk"] = round(risk_formula(rec), 1)
        rec["风险等级"] = tier_of(rec["Risk"])
        # 证据强度：提示读者该分数由数据驱动还是由专家先验驱动
        rec["数据权重"] = round(float(w), 3)
        rec["证据强度"] = ("强（数据驱动）" if len(sub) >= 50 else
                       "中" if len(sub) >= 20 else "弱（以专家先验为主）")
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("Risk", ascending=False).reset_index(drop=True)
    out.insert(0, "排名", range(1, len(out) + 1))
    return out, pd.DataFrame(dropped)


# ------------------------------------------------------------ 敏感性分析
def sensitivity(m: pd.DataFrame) -> pd.DataFrame:
    """三类稳健性检验：等权重、纯专家分、纯数据分，与基准排名比较。"""
    from scipy.stats import spearmanr
    rows = []
    base_rank = m["Risk"].rank(ascending=False)

    scenarios = {}
    # 1) 五维等权
    eq = (m[DIMS[:4]].sum(axis=1) + (100 - m["H"])) / 5
    scenarios["等权重(各0.20)"] = eq
    # 2) 纯专家先验
    exp_df = pd.DataFrame({d: m[f"{d}_expert"] for d in DIMS}, index=m.index)
    scenarios["纯专家先验"] = exp_df.apply(risk_formula, axis=1)
    # 3) 纯数据证据
    dat_df = pd.DataFrame({d: m[f"{d}_data"] for d in DIMS}, index=m.index)
    scenarios["纯数据证据"] = dat_df.apply(risk_formula, axis=1)
    # 4) 逐维权重 ±0.05 扰动（补偿到其余维度）
    for d in DIMS:
        w = dict(RISK_WEIGHTS)
        w[d] = w[d] + 0.05
        rest = [x for x in DIMS if x != d]
        for r in rest:
            w[r] -= 0.05 / len(rest)
        s = (w["R"] * m["R"] + w["S"] * m["S"] + w["D"] * m["D"]
             + w["A"] * m["A"] + w["H"] * (100 - m["H"]))
        scenarios[f"{d}权重+0.05"] = s

    for name, s in scenarios.items():
        rho = spearmanr(base_rank, s.rank(ascending=False)).statistic
        rows.append({"情景": name, "与基准排名Spearman相关": round(float(rho), 4),
                     "最高风险岗位": s.idxmax(), "最低风险岗位": s.idxmin(),
                     "得分极差": round(float(s.max() - s.min()), 1)})
    return pd.DataFrame(rows)


def main():
    print("[1/5] 载入岗位特征矩阵 ...")
    m = pd.read_csv(OUT / "02_category_feature_matrix.csv", index_col=0)
    jobs = pd.read_csv(OUT / "02_job_level_features.csv")
    LOG["n_categories"] = len(m)
    LOG["n_jobs"] = len(jobs)

    print("[2/5] 数据证据分与专家先验分融合（含样本量收缩）...")
    m, detail = blend(m)
    detail.to_csv(OUT / "03_dimension_detail.csv", index=False, encoding="utf-8-sig")

    print("[3/5] 计算五维加权风险指数 ...")
    m["Risk"] = m.apply(risk_formula, axis=1).round(1)
    m["风险等级"] = m["Risk"].map(tier_of)
    m["风险排名"] = m["Risk"].rank(ascending=False).astype(int)
    m["是否金融核心岗"] = m.index.isin(FINANCE_CORE_CATEGORIES)
    m["专家先验依据"] = m.index.map(EXPERT_PRIOR_RATIONALE)

    print("[4/5] KMeans 聚类分层（与阈值分层交叉验证）...")
    X = m[DIMS].values
    km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X)
    m["cluster"] = km.labels_
    # 按簇内平均 Risk 排序命名
    order = m.groupby("cluster")["Risk"].mean().sort_values(ascending=False).index
    name_map = {c: n for c, n in zip(order, ["聚类-高替代", "聚类-中替代", "聚类-低替代"])}
    m["聚类分层"] = m["cluster"].map(name_map)
    m[["Risk", "风险等级", "聚类分层"]].to_csv(
        OUT / "03_cluster_assignment.csv", encoding="utf-8-sig")
    agree = (((m["风险等级"] == "高替代风险") & (m["聚类分层"] == "聚类-高替代"))
             | ((m["风险等级"] == "中等替代风险") & (m["聚类分层"] == "聚类-中替代"))
             | ((m["风险等级"] == "低替代风险") & (m["聚类分层"] == "聚类-低替代"))).mean()
    LOG["tier_cluster_agreement"] = round(float(agree), 3)

    print("[5/5] 细分岗位 / 典型职能岗评分 + 敏感性分析 ...")
    sub = score_subjobs(jobs, m)
    sub.to_csv(OUT / "03_risk_scores_subjob.csv", index=False, encoding="utf-8-sig")
    roles, dropped = score_typical_roles(jobs, m)
    roles.to_csv(OUT / "03_risk_scores_typical_roles.csv", index=False, encoding="utf-8-sig")
    if len(dropped):
        dropped.to_csv(OUT / "03_roles_dropped.csv", index=False, encoding="utf-8-sig")
    LOG["n_typical_roles"] = len(roles)
    LOG["typical_roles_coverage"] = round(float(roles["岗位数"].sum() / len(jobs)), 3)
    LOG["roles_dropped"] = dropped.to_dict("records") if len(dropped) else []
    sens = sensitivity(m)
    sens.to_csv(OUT / "03_sensitivity.csv", index=False, encoding="utf-8-sig")

    out_cols = (["岗位数", "公司数", "薪资中位数K"]
                + [f"{d}_data" for d in DIMS] + [f"{d}_expert" for d in DIMS]
                + DIMS + ["Risk", "风险等级", "风险排名", "聚类分层",
                          "是否金融核心岗", "专家先验依据"])
    m_out = m[out_cols].sort_values("Risk", ascending=False)
    m_out.to_csv(OUT / "03_risk_scores_category.csv", encoding="utf-8-sig")

    LOG["risk_max"] = float(m["Risk"].max())
    LOG["risk_min"] = float(m["Risk"].min())
    LOG["tier_counts"] = m["风险等级"].value_counts().to_dict()
    LOG["n_subjobs_scored"] = len(sub)
    with open(OUT / "03_scoring_log.json", "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=2)

    print("\n=== 岗位大类 AI 替代风险指数 ===")
    print(m_out[DIMS + ["Risk", "风险等级", "聚类分层"]].to_string())
    print("\n=== 典型金融职能岗 AI 替代风险排名 ===")
    print(roles[["排名", "职能岗", "岗位数", "R", "S", "D", "A", "H",
                 "Risk", "风险等级"]].to_string(index=False))
    print("\n=== 细分岗位风险 Top15 ===")
    print(sub.head(15)[["std_title", "category", "岗位数", "Risk", "风险等级"]].to_string(index=False))
    print("\n=== 细分岗位风险 Bottom10 ===")
    print(sub.tail(10)[["std_title", "category", "岗位数", "Risk", "风险等级"]].to_string(index=False))
    print("\n=== 敏感性分析 ===")
    print(sens.to_string(index=False))
    print(f"\n阈值分层与聚类分层一致率: {LOG['tier_cluster_agreement']:.1%}")


if __name__ == "__main__":
    main()
