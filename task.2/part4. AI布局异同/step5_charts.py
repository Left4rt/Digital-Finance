# -*- coding: utf-8 -*-
"""Step 5 — 可视化：出 5 张图，全部存到 out/fig_*.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
INK, GRID = "#1f2937", "#e5e7eb"

OUT = "out"
df = pd.read_csv(f"{OUT}/merged_panel.csv")
mat = pd.read_csv("company_tech_matrix.csv")
mat.columns = [c.lstrip("\ufeff") for c in mat.columns]

# ---------- 图1 资产密集度对比 ----------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
pairs = [("ai_depth_2025", "AI布局强度（2025研发投入章节）"),
         ("ai_density_2025", "AI披露密度（次/千字）"),
         ("rd_intensity_pct", "研发投入强度（%）")]
groups = ["资产密集型金融机构", "轻资产科技服务商"]
for ax, (col, title) in zip(axes, pairs):
    data = [df[df.asset_intensity == g][col].dropna().values for g in groups]
    bp = ax.boxplot(data, patch_artist=True, widths=.55,
                    medianprops=dict(color=INK, lw=1.6))
    for patch, c in zip(bp["boxes"], ["#94a3b8", "#2563eb"]):
        patch.set_facecolor(c); patch.set_alpha(.55); patch.set_edgecolor(INK)
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.normal(i, .06, len(d)), d, s=14, c=INK, alpha=.5, zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["资产密集型\n金融机构", "轻资产\n科技服务商"], fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.suptitle("图1  资产密集度不同的公司，AI 技术布局差异（Mann-Whitney U 检验均 p<0.001）",
             fontsize=11, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_asset_intensity.png", bbox_inches="tight")

# ---------- 图2 研发强度 vs 2024 布局深度 ----------
d2 = df[df.rd_intensity_pct.notna() & df.depth_2024.notna()]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
colors = np.where(d2.asset_intensity == "资产密集型金融机构", "#94a3b8", "#2563eb")
ax.scatter(d2.rd_intensity_pct, d2.depth_2024, s=60, c=colors, alpha=.85,
           edgecolor=INK, lw=.6)
for _, r in d2.iterrows():
    ax.annotate(r.company_name, (r.rd_intensity_pct, r.depth_2024),
                fontsize=7, xytext=(4, 3), textcoords="offset points", color=INK)
z = np.polyfit(d2.rd_intensity_pct, d2.depth_2024, 1)
xs = np.linspace(d2.rd_intensity_pct.min(), d2.rd_intensity_pct.max(), 50)
ax.plot(xs, np.polyval(z, xs), ls="--", c="#dc2626", lw=1.3)
ax.set_xlabel("2025 年研发投入强度（研发投入/营业收入，%）")
ax.set_ylabel("2024 年 AI 技术布局强度（证据加权）")
ax.set_title("图2  研发投入强度可以预测 AI 布局强度（Spearman ρ=0.521, p=0.016, N=21）",
             fontsize=10.5)
ax.grid(color=GRID); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_rd_vs_layout.png", bbox_inches="tight")

# ---------- 图3 四象限 ----------
q = pd.read_csv(f"{OUT}/quadrant_assignment.csv").merge(
    df[["company_id", "rd_text_type"]], on="company_id", how="left")
fig, ax = plt.subplots(figsize=(7.6, 5.2))
rd_med, ai_med = q.rd_intensity_pct.median(), q.ai_depth_2025.median()
size = q.revenue_2025_yi.fillna(3).clip(1, 200) ** .55 * 14
col = np.where(q.net_profit_2025_yi.isna(), "#cbd5e1",
               np.where(q.net_profit_2025_yi < 0, "#dc2626", "#16a34a"))
narrmask = q.rd_text_type == "叙述型(含研发项目描述)"
ax.scatter(q.rd_intensity_pct[narrmask], q.ai_depth_2025[narrmask],
           s=size[narrmask], c=col[narrmask], alpha=.78, edgecolor=INK, lw=.6,
           label="叙述型披露（AI布局可测）")
ax.scatter(q.rd_intensity_pct[~narrmask], q.ai_depth_2025[~narrmask],
           s=size[~narrmask], facecolors="none", edgecolor="#6b7280", lw=1.2,
           marker="s", label="纯表格型披露（章节内无文本，AI布局不可测）")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
for _, r in q.iterrows():
    ax.annotate(r.company_name, (r.rd_intensity_pct, r.ai_depth_2025), fontsize=7,
                xytext=(5, 4), textcoords="offset points", color=INK)
ax.axvline(rd_med, color=INK, ls=":", lw=1); ax.axhline(ai_med, color=INK, ls=":", lw=1)
ax.set_xlabel("研发投入强度（%）"); ax.set_ylabel("2025 年 AI 布局强度")
ax.set_title("图3  研发投入强度 × AI 布局强度 四象限\n"
             "（气泡=营业收入；红=亏损，绿=盈利，灰=未取到利润；方块=章节仅有表格，零值为披露版式所致）",
             fontsize=10)
ax.grid(color=GRID); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_quadrant.png", bbox_inches="tight")

# ---------- 图4 分档对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (key, order, title) in zip(axes, [
        ("revenue_tier", ["小规模", "中规模", "大规模"], "按营业收入三分档"),
        ("profit_tier", ["亏损", "微利/一般盈利", "高盈利(净利率≥20%)"], "按盈利能力分档")]):
    g = df.groupby(key)[["ai_depth_2025", "ai_breadth_2025"]].mean().reindex(order)
    x = np.arange(len(order))
    ax.bar(x - .19, g.ai_depth_2025, .38, label="AI布局强度", color="#2563eb", alpha=.85)
    ax.bar(x + .19, g.ai_breadth_2025, .38, label="AI布局广度", color="#f59e0b", alpha=.9)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8.5)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.suptitle("图4  规模档与盈利档下的 AI 布局：规模不单调、亏损组反而最激进", fontsize=11, y=1.04)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_tiers.png", bbox_inches="tight")

# ---------- 图5 技术方向热力图 ----------
CN = {"TECH_LLM": "大模型", "TECH_ML": "机器学习", "TECH_NLP": "NLP", "TECH_CV": "计算机视觉",
      "TECH_SPEECH": "智能语音", "TECH_KG": "知识图谱", "TECH_RL": "强化学习",
      "TECH_PRIVACY": "隐私计算", "TECH_DECISION": "智能决策", "AI_ENGINEERING": "AI工程化"}
m = mat.set_index("company_id")
j = df.set_index("company_id").loc[[i for i in m.index if i in set(df.company_id)]]
mm = m.loc[j.index]
grp = mm.groupby(j.asset_intensity).mean()
grp2 = mm.groupby(j.segment).mean()
H = pd.concat([grp, grp2]).rename(columns=CN)
H = H.loc[H.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(H) + 1.8))
im = ax.imshow(H.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=40, ha="right", fontsize=8.5)
ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index, fontsize=8.5)
for i in range(H.shape[0]):
    for k in range(H.shape[1]):
        v = H.values[i, k]
        if v > 0.01:
            ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > H.values.max() * .55 else INK)
ax.set_title("图5  各类公司在 10 个 AI 技术方向上的平均布局强度（2024 年报口径）", fontsize=10.5, pad=10)
fig.colorbar(im, ax=ax, shrink=.7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_direction_heatmap.png", bbox_inches="tight")
print("图已输出：fig1~fig5")
