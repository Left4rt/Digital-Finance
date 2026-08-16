# -*- coding: utf-8 -*-
"""
Step 4 — 任务 5：不同公司（资产 / 盈利等差异度）在 AI 技术布局上的异同点。

输入：
  out/financial_panel_2025.csv  财务面板（2025）
  out/ai_layout_2025.csv        2025 年报研发投入章节的 AI 布局向量
  company_tech_stack.csv        2024 年报三章节的 AI 技术栈（step5 产物）
  company_tech_matrix.csv       2024 年 10 方向布局矩阵
输出（全部写入 out/）：
  merged_panel.csv              合并后的分析主表
  group_compare_*.csv           五套分组对比表
  correlation_spearman.csv      秩相关
  hypothesis_tests.csv          组间检验
  quadrant_assignment.csv       研发强度 × AI 布局 四象限
  distance_mantel.csv           财务距离 vs 布局距离（Mantel 检验）
  cross_year_consistency.csv    2024 vs 2025 一致性
"""
import os, itertools
import numpy as np
import pandas as pd
from scipy import stats

OUT = "out"
pd.set_option("display.width", 200)

fin = pd.read_csv(f"{OUT}/financial_panel_2025.csv")
ai = pd.read_csv(f"{OUT}/ai_layout_2025.csv")
stack = pd.read_csv("company_tech_stack.csv")
stack.columns = [c.lstrip("\ufeff") for c in stack.columns]
mat = pd.read_csv("company_tech_matrix.csv")
mat.columns = [c.lstrip("\ufeff") for c in mat.columns]

df = fin.merge(ai.drop(columns=["company_name"]), on="company_id", how="left")
df = df.merge(
    stack[["company_id", "stack_breadth", "stack_depth", "hhi",
           "primary_cn", "evidence_sentences", "product_line_count"]]
    .rename(columns={"stack_breadth": "breadth_2024", "stack_depth": "depth_2024",
                     "hhi": "hhi_2024", "primary_cn": "primary_2024",
                     "evidence_sentences": "evidence_2024",
                     "product_line_count": "products_2024"}),
    on="company_id", how="left")

for c in ["ai_kw_hits_2025", "ai_breadth_2025", "ai_depth_2025",
          "ai_product_lines_2025", "ai_density_2025"]:
    df[c] = df[c].fillna(0)

# ---------------------------------------------------------------- 分档
def tercile(s, labels):
    ok = s.notna()
    out = pd.Series(index=s.index, dtype=object)
    if ok.sum() >= 3:
        out[ok] = pd.qcut(s[ok], 3, labels=labels, duplicates="drop")
    return out

df["revenue_tier"] = tercile(df.revenue_2025_yi, ["小规模", "中规模", "大规模"])
df["rd_tier"] = tercile(df.rd_intensity_pct, ["低研发强度", "中研发强度", "高研发强度"])


def profit_tier(r):
    if pd.isna(r.net_profit_2025_yi):
        return np.nan
    if r.net_profit_2025_yi < 0:
        return "亏损"
    if pd.isna(r.net_margin_pct):
        return np.nan          # 有利润但无营收口径，不参与盈利分档
    if r.net_margin_pct >= 20:
        return "高盈利(净利率≥20%)"
    return "微利/一般盈利"


df["profit_tier"] = df.apply(profit_tier, axis=1)
df.to_csv(f"{OUT}/merged_panel.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 分组对比
METRICS = ["ai_depth_2025", "ai_breadth_2025", "ai_density_2025",
           "ai_product_lines_2025", "rd_intensity_pct", "rd_amount_2025_yi",
           "depth_2024", "breadth_2024"]


def group_table(by, fname):
    g = df.groupby(by, dropna=True)
    t = g.agg(n=("company_id", "count"),
              研发章节披露率=("rd_disclosed", lambda x: round(np.mean(x) * 100, 1)),
              有AI披露公司占比=("ai_kw_hits_2025", lambda x: round(np.mean(x > 0) * 100, 1)),
              **{m: (m, "mean") for m in METRICS})
    t = t.round(3)
    top = g.apply(lambda x: (x.loc[x.ai_depth_2025.idxmax(), "primary_direction_2025"]
                             if x.ai_depth_2025.max() > 0 else "—"),
                  include_groups=False)
    t["组内最强公司主方向"] = top
    t.to_csv(f"{OUT}/{fname}", encoding="utf-8-sig")
    return t


t_asset = group_table("asset_intensity", "group_compare_asset_intensity.csv")
t_seg = group_table("segment", "group_compare_segment_2025.csv")
t_rev = group_table("revenue_tier", "group_compare_revenue_tier.csv")
t_prof = group_table("profit_tier", "group_compare_profit_tier.csv")
t_rd = group_table("rd_tier", "group_compare_rd_tier.csv")

print("=== 按资产密集度 ===\n", t_asset.to_string(), "\n")
print("=== 按营收规模三分档 ===\n", t_rev.to_string(), "\n")
print("=== 按盈利分档 ===\n", t_prof.to_string(), "\n")
print("=== 按研发强度三分档 ===\n", t_rd.to_string(), "\n")

# ---------------- 稳健性：仅用「叙述型」披露的子样本（文本可比） ----------------
narr = df[df.rd_text_type == "叙述型(含研发项目描述)"].copy()
narr_tbl = narr.groupby("asset_intensity").agg(
    n=("company_id", "count"),
    ai_depth=("ai_depth_2025", "mean"), ai_breadth=("ai_breadth_2025", "mean"),
    ai_density=("ai_density_2025", "mean"), rd_intensity=("rd_intensity_pct", "mean")
).round(3)
narr_tbl.to_csv(f"{OUT}/robustness_narrative_subsample.csv", encoding="utf-8-sig")
df.groupby(["board", "rd_text_type"]).size().unstack(fill_value=0) \
    .to_csv(f"{OUT}/disclosure_format_by_board.csv", encoding="utf-8-sig")
print("=== 稳健性：叙述型子样本 (n=%d) ===\n" % len(narr), narr_tbl.to_string(), "\n")
print("=== 披露版式 × 上市板块 ===\n",
      df.groupby(["board", "rd_text_type"]).size().unstack(fill_value=0).to_string(), "\n")

# ---------------------------------------------------------------- 秩相关
pairs = [("revenue_2025_yi", "ai_depth_2025"), ("revenue_2025_yi", "ai_breadth_2025"),
         ("revenue_2025_yi", "ai_density_2025"), ("revenue_2025_yi", "rd_intensity_pct"),
         ("rd_amount_2025_yi", "ai_depth_2025"), ("rd_amount_2025_yi", "ai_breadth_2025"),
         ("rd_intensity_pct", "ai_depth_2025"), ("rd_intensity_pct", "ai_breadth_2025"),
         ("rd_intensity_pct", "ai_density_2025"),
         ("net_profit_2025_yi", "ai_depth_2025"), ("net_margin_pct", "ai_depth_2025"),
         ("net_margin_pct", "rd_intensity_pct"), ("net_margin_pct", "ai_breadth_2025"),
         ("total_assets_2025_yi", "ai_depth_2025"),
         ("revenue_2025_yi", "depth_2024"), ("rd_intensity_pct", "depth_2024"),
         ("net_margin_pct", "depth_2024")]
rows = []
for a, b in pairs:
    sub = df[[a, b]].dropna()
    if len(sub) >= 5:
        rho, p = stats.spearmanr(sub[a], sub[b])
        rows.append(dict(变量X=a, 变量Y=b, N=len(sub), spearman_rho=round(rho, 3),
                         p_value=round(p, 4),
                         显著性=("**p<0.01" if p < .01 else
                               "*p<0.05" if p < .05 else
                               "†p<0.1" if p < .1 else "不显著")))
for a, b in [("rd_intensity_pct", "ai_depth_2025"), ("revenue_2025_yi", "ai_depth_2025"),
             ("net_margin_pct", "ai_depth_2025")]:
    sub = narr[[a, b]].dropna()
    if len(sub) >= 5:
        rho, p = stats.spearmanr(sub[a], sub[b])
        rows.append(dict(变量X=a + "[叙述型子样本]", 变量Y=b, N=len(sub),
                         spearman_rho=round(rho, 3), p_value=round(p, 4),
                         显著性=("**p<0.01" if p < .01 else "*p<0.05" if p < .05
                               else "†p<0.1" if p < .1 else "不显著")))
corr = pd.DataFrame(rows).sort_values("p_value")
corr.to_csv(f"{OUT}/correlation_spearman.csv", index=False, encoding="utf-8-sig")
print("=== Spearman 秩相关 ===\n", corr.to_string(index=False), "\n")

# ---------------------------------------------------------------- 组间检验
tests = []
a = df[df.asset_intensity == "资产密集型金融机构"]
b = df[df.asset_intensity == "轻资产科技服务商"]
for m in ["ai_depth_2025", "ai_breadth_2025", "ai_density_2025", "rd_intensity_pct"]:
    x, y = a[m].dropna(), b[m].dropna()
    if len(x) >= 3 and len(y) >= 3:
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        tests.append(dict(检验="Mann-Whitney U", 分组="资产密集 vs 轻资产", 指标=m,
                          n1=len(x), n2=len(y), 均值1=round(x.mean(), 3),
                          均值2=round(y.mean(), 3), 统计量=round(u, 1),
                          p_value=round(p, 4)))
# 研发披露率的卡方
ct = pd.crosstab(df.asset_intensity, df.rd_disclosed)
chi2, p, dof, _ = stats.chi2_contingency(ct)
tests.append(dict(检验="卡方(独立性)", 分组="资产密集 vs 轻资产", 指标="是否披露研发投入表",
                  n1=int(ct.iloc[0].sum()), n2=int(ct.iloc[1].sum()),
                  均值1=round(ct.iloc[0, 1] / ct.iloc[0].sum() * 100, 1),
                  均值2=round(ct.iloc[1, 1] / ct.iloc[1].sum() * 100, 1),
                  统计量=round(chi2, 3), p_value=round(p, 5)))
# 亏损 vs 盈利（二分）
for m in ["ai_depth_2025", "ai_breadth_2025", "rd_intensity_pct", "ai_density_2025"]:
    x = df[df.is_loss == 1][m].dropna()
    y = df[df.is_loss == 0][m].dropna()
    if len(x) >= 3 and len(y) >= 3:
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        tests.append(dict(检验="Mann-Whitney U", 分组="亏损 vs 盈利", 指标=m,
                          n1=len(x), n2=len(y), 均值1=round(x.mean(), 3),
                          均值2=round(y.mean(), 3), 统计量=round(u, 1),
                          p_value=round(p, 4)))

# 跨 segment 的 Kruskal-Wallis
for m in ["ai_depth_2025", "ai_breadth_2025", "rd_intensity_pct"]:
    groups = [g[m].dropna().values for _, g in df.groupby("segment")
              if g[m].dropna().shape[0] >= 3]
    if len(groups) >= 3:
        h, p = stats.kruskal(*groups)
        tests.append(dict(检验="Kruskal-Wallis", 分组="业务属性(≥3家的组)", 指标=m,
                          n1=len(groups), n2=sum(len(g) for g in groups),
                          均值1=None, 均值2=None, 统计量=round(h, 3),
                          p_value=round(p, 4)))
# 盈利分档的 Kruskal-Wallis
for m in ["ai_depth_2025", "rd_intensity_pct"]:
    groups = [g[m].dropna().values for _, g in df.groupby("profit_tier")
              if g[m].dropna().shape[0] >= 3]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        tests.append(dict(检验="Kruskal-Wallis", 分组="盈利分档", 指标=m,
                          n1=len(groups), n2=sum(len(g) for g in groups),
                          均值1=None, 均值2=None, 统计量=round(h, 3),
                          p_value=round(p, 4)))
tst = pd.DataFrame(tests)
tst.to_csv(f"{OUT}/hypothesis_tests.csv", index=False, encoding="utf-8-sig")
print("=== 组间检验 ===\n", tst.to_string(index=False), "\n")

# ---------------------------------------------------------------- 四象限
q = df[df.rd_intensity_pct.notna()].copy()
rd_med, ai_med = q.rd_intensity_pct.median(), q.ai_depth_2025.median()


def quad(r):
    hi_rd = r.rd_intensity_pct >= rd_med
    hi_ai = r.ai_depth_2025 >= ai_med
    return ("Ⅰ 技术驱动型（高投入·高AI布局）" if hi_rd and hi_ai else
            "Ⅱ 投入未转化（高投入·低AI布局）" if hi_rd else
            "Ⅲ 轻投入高声量（低投入·高AI布局）" if hi_ai else
            "Ⅳ 双低（低投入·低AI布局）")


q["quadrant"] = q.apply(quad, axis=1)
q[["company_id", "company_name", "segment", "asset_intensity", "revenue_2025_yi",
   "net_profit_2025_yi", "net_margin_pct", "rd_amount_2025_yi", "rd_intensity_pct",
   "ai_depth_2025", "ai_breadth_2025", "primary_direction_2025", "quadrant"]] \
    .sort_values(["quadrant", "ai_depth_2025"], ascending=[True, False]) \
    .to_csv(f"{OUT}/quadrant_assignment.csv", index=False, encoding="utf-8-sig")
print("象限阈值： 研发强度中位数 %.2f%%  AI布局强度中位数 %.2f" % (rd_med, ai_med))
print(q.quadrant.value_counts().to_string(), "\n")

# ---------------------------------------------------------------- Mantel 检验
sub = df[df.revenue_2025_yi.notna() & (df.ai_depth_2025 > 0)].copy()
dirs = [c for c in ai.columns if c.startswith("ai_")]
# 布局向量用 2024 的 10 方向矩阵（口径统一、非零覆盖高），与财务距离比较
m2 = mat.set_index("company_id")
common = [c for c in sub.company_id if c in m2.index]
if len(common) >= 8:
    V = m2.loc[common].values
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    D_ai = 1 - Vn @ Vn.T                                   # 余弦距离
    f = sub.set_index("company_id").loc[common]
    fv = np.log10(f.revenue_2025_yi.values.astype(float) + 1e-3)
    D_fin = np.abs(fv[:, None] - fv[None, :])              # log 营收距离
    iu = np.triu_indices(len(common), 1)
    rho, p = stats.spearmanr(D_ai[iu], D_fin[iu])
    perm = []
    rng = np.random.default_rng(42)
    for _ in range(4999):
        idx = rng.permutation(len(common))
        perm.append(stats.spearmanr(D_ai[np.ix_(idx, idx)][iu], D_fin[iu]).statistic)
    p_perm = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)
    pd.DataFrame([dict(样本数=len(common), 配对数=len(iu[0]),
                       spearman_rho=round(rho, 3), 渐近p=round(p, 4),
                       置换检验p=round(p_perm, 4),
                       说明="AI布局余弦距离 vs log10(营业收入)距离")]) \
        .to_csv(f"{OUT}/distance_mantel.csv", index=False, encoding="utf-8-sig")
    print("=== Mantel 检验（布局距离 ~ 财务距离）rho=%.3f, 置换p=%.4f, n=%d ===\n"
          % (rho, p_perm, len(common)))

# ---------------------------------------------------------------- 跨年一致性
cy = df[df.depth_2024.notna()].copy()
r1, p1 = stats.spearmanr(cy.depth_2024, cy.ai_depth_2025)
r2, p2 = stats.spearmanr(cy.breadth_2024, cy.ai_breadth_2025)
same = (cy.primary_2024 == cy.primary_direction_2025).mean()
pd.DataFrame([
    dict(对比="2024布局强度 vs 2025布局强度", N=len(cy), rho=round(r1, 3), p=round(p1, 4)),
    dict(对比="2024布局广度 vs 2025布局广度", N=len(cy), rho=round(r2, 3), p=round(p2, 4)),
    dict(对比="核心技术方向跨年一致率(%)", N=len(cy), rho=round(same * 100, 1), p=None),
]).to_csv(f"{OUT}/cross_year_consistency.csv", index=False, encoding="utf-8-sig")
print("跨年一致性： 强度 rho=%.3f(p=%.3f) 广度 rho=%.3f(p=%.3f) 主方向一致率=%.1f%%"
      % (r1, p1, r2, p2, same * 100))
