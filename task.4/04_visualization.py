# -*- coding: utf-8 -*-
"""
task4 / 04_visualization.py
------------------------------------------------------------------
阶段 6：风险结果可视化

输入：data/03_*.csv
输出（figures/）：
    fig1_risk_ranking.png     典型金融职能岗 AI 替代风险排名（横向条形图）
    fig2_risk_matrix.png      AI 替代风险 × 人类不可替代性 二维四象限矩阵
    fig3_radar.png            代表性岗位五维风险雷达图
    fig4_heatmap.png          岗位大类 × 五维指标 热力图
    fig5_data_vs_expert.png   数据证据分 vs 专家先验分 对照（方法论透明度）
    fig6_evidence.png         招聘数据证据：AI/数字化/人际技能覆盖率对比
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from config.taxonomy import RISK_WEIGHTS

BASE = Path(__file__).parent
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ---- 中文字体（容器内 Noto CJK） ----
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

TIER_COLOR = {"高替代风险": "#C0392B", "中等替代风险": "#E29B26", "低替代风险": "#2E86AB"}
DIMS = ["R", "S", "D", "A", "H"]
DIM_LABEL = {"R": "R 任务重复性", "S": "S 规则化程度", "D": "D 任务数据数字化程度",
             "A": "A AI技术成熟度", "H": "H 人类不可替代性"}
# 说明：D、A 两维的"数据证据分"来自招聘文本中的技能需求（数字化技能需求强度、
# AI 技能需求强度），它们是维度本身的代理变量而非维度本身，故在图注中单独标明。
DATA_PROXY_NOTE = ("注：D、A 两维的数据证据分为招聘文本中的"
                   "数字化技能需求强度 / AI 技能需求强度，属代理变量而非维度本身。")


def fig1_ranking(roles: pd.DataFrame):
    d = roles.sort_values("Risk")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors = [TIER_COLOR[t] for t in d["风险等级"]]
    hatches = ["//" if "弱" in e else "" for e in d["证据强度"]]
    bars = ax.barh(d["职能岗"], d["Risk"], color=colors, edgecolor="white", height=0.72)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for b, v, n in zip(bars, d["Risk"], d["岗位数"]):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=10, fontweight="bold", color="#333")
        ax.text(1.5, b.get_y() + b.get_height() / 2, f"n={n}",
                va="center", fontsize=8, color="white")
    for x, lab in [(40, "中风险线"), (70, "高风险线")]:
        ax.axvline(x, color="#666", ls="--", lw=1, alpha=0.7)
        ax.text(x, len(d) - 0.3, lab, fontsize=8, color="#666", ha="center")
    ax.set_xlim(0, 85)
    ax.set_xlabel("AI 替代风险指数 Risk（0–100，越高越易被替代）", fontsize=11)
    ax.set_title("图1　典型金融财务职能岗 AI 替代风险排名\n"
                 "数据来源：40 家 A 股金融/财务类上市公司 3254 条招聘岗位",
                 fontsize=13, fontweight="bold", pad=14)
    handles = [Patch(facecolor=c, label=k) for k, c in TIER_COLOR.items()]
    handles.append(Patch(facecolor="#999", hatch="//", label="斜纹=样本量<20，\n以专家先验为主"))
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG / "fig1_risk_ranking.png")
    plt.close(fig)


def fig2_matrix(roles: pd.DataFrame):
    """横轴 AI 替代风险，纵轴人类不可替代性 H，气泡大小 = 岗位供给规模。"""
    fig, ax = plt.subplots(figsize=(10.5, 8))
    x, y = roles["Risk"], roles["H"]
    size = 90 + roles["岗位数"] ** 0.62 * 26
    ax.scatter(x, y, s=size, c=[TIER_COLOR[t] for t in roles["风险等级"]],
               alpha=0.7, edgecolors="white", linewidths=1.6, zorder=3)

    xm, ym = 50, 50
    ax.axvline(xm, color="#444", lw=1.2, ls="--", zorder=2)
    ax.axhline(ym, color="#444", lw=1.2, ls="--", zorder=2)
    ax.axvspan(xm, 100, ymin=0, ymax=0.5, color="#C0392B", alpha=0.05, zorder=1)
    ax.axvspan(0, xm, ymin=0.5, ymax=1, color="#2E86AB", alpha=0.05, zorder=1)

    quad = [(0.965, 0.965, "Ⅰ 人机协同区\n高不可替代 · 高替代压力\n→ 任务重构，AI 做前段", "right", "top"),
            (0.035, 0.965, "Ⅱ 安全区\n高不可替代 · 低替代压力\n→ 岗位稳固，需补数字化能力", "left", "top"),
            (0.035, 0.035, "Ⅲ 观望区\n低不可替代 · 低替代压力\n→ 价值有限，易被边缘化", "left", "bottom"),
            (0.965, 0.035, "Ⅳ 高危区\n低不可替代 · 高替代压力\n→ 优先自动化，转型最紧迫", "right", "bottom")]
    for fx, fy, txt, ha, va in quad:
        ax.text(fx, fy, txt, transform=ax.transAxes, fontsize=9, ha=ha, va=va,
                color="#444", bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                        ec="#bbb", alpha=0.85))

    # --- 标签贪心避让：候选偏移中选第一个不与已放置标签重叠的位置 ---
    XR, YR = 85 - 20, 88 - 5          # 坐标轴数据跨度
    CAND = [(0, 2.4), (0, -3.0), (0, 4.6), (0, -5.2),
            (7.5, 1.2), (-7.5, 1.2), (7.5, -1.8), (-7.5, -1.8),
            (0, 6.8), (0, -7.4), (10.5, 3.0), (-10.5, 3.0)]
    placed = []

    def overlap(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for _, r in roles.sort_values("岗位数", ascending=False).iterrows():
        txt = r["职能岗"]
        w = len(txt) * 0.0155 * XR      # 中文字宽的经验估计（数据坐标）
        h = 0.032 * YR
        for dx, dy in CAND:
            cx, cy = r["Risk"] + dx, r["H"] + dy
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if not any(overlap(box, p) for p in placed):
                placed.append(box)
                ax.annotate(txt, (r["Risk"], r["H"]), xytext=(cx, cy),
                            textcoords="data", ha="center", va="center",
                            fontsize=8.5, color="#222", zorder=5,
                            arrowprops=dict(arrowstyle="-", color="#999",
                                            lw=0.7, shrinkA=0, shrinkB=6)
                            if abs(dx) + abs(dy) > 4 else None)
                break

    ax.set_xlim(20, 85)
    ax.set_ylim(5, 88)
    ax.set_xlabel("AI 替代风险指数 Risk →", fontsize=11)
    ax.set_ylabel("人类不可替代性 H →", fontsize=11)
    ax.set_title("图2　AI 替代风险 × 人类不可替代性 二维矩阵\n"
                 "气泡大小 = 该职能岗招聘需求量", fontsize=13, fontweight="bold", pad=14)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig2_risk_matrix.png")
    plt.close(fig)


def fig3_radar(roles: pd.DataFrame):
    pick = ["会计核算与记账", "信贷审批与贷后管理", "风控策略与风险监控",
            "财税咨询顾问", "证券分析与投资研究", "客户经理与客户维护"]
    d = roles.set_index("职能岗").reindex([p for p in pick if p in roles["职能岗"].values])
    labels = [DIM_LABEL[x] for x in DIMS]
    ang = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist()
    ang += ang[:1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5),
                             subplot_kw=dict(polar=True))
    for ax, (name, row) in zip(axes.flat, d.iterrows()):
        vals = [row[x] for x in DIMS]
        vals += vals[:1]
        c = TIER_COLOR[row["风险等级"]]
        ax.plot(ang, vals, color=c, lw=2)
        ax.fill(ang, vals, color=c, alpha=0.25)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75])
        ax.set_yticklabels(["25", "50", "75"], fontsize=7, color="#999")
        ax.set_title(f"{name}\nRisk = {row['Risk']:.1f}（{row['风险等级']}）",
                     fontsize=11, fontweight="bold", pad=18, color=c)
    for ax in axes.flat[len(d):]:
        ax.axis("off")
    fig.suptitle("图3　代表性金融岗位五维风险画像\n"
                 "R 重复性 / S 规则化 / D 数字化 / A AI成熟度 / H 人类不可替代性",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_radar.png")
    plt.close(fig)


def fig4_heatmap(cat: pd.DataFrame):
    d = cat.sort_values("Risk", ascending=False)
    mat = d[DIMS].values
    fig, ax = plt.subplots(figsize=(9.5, 7))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=10, vmax=95, aspect="auto")
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([DIM_LABEL[x] for x in DIMS], fontsize=9.5)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{i}  (Risk {r:.1f})" for i, r in zip(d.index, d["Risk"])],
                       fontsize=9.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if (v > 72 or v < 28) else "#222")
    plt.colorbar(im, ax=ax, shrink=0.75, label="维度得分（0–100）")
    ax.set_title("图4　13 类岗位五维风险指标热力图（按综合风险降序）\n"
                 "注：H 越高风险越低，公式中以 (100−H) 计入",
                 fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, -0.03, DATA_PROXY_NOTE, ha="center", fontsize=8.5, color="#555")
    fig.savefig(FIG / "fig4_heatmap.png")
    plt.close(fig)


def fig5_data_vs_expert(cat: pd.DataFrame):
    """展示"数据证据分"与"专家先验分"的差异及融合结果，体现评分可追溯。"""
    d = cat.sort_values("Risk", ascending=False)
    fig, axes = plt.subplots(1, 5, figsize=(17, 6.2), sharey=True)
    yp = np.arange(len(d))
    for ax, dim in zip(axes, DIMS):
        ax.barh(yp + 0.2, d[f"{dim}_data"], height=0.35, color="#4C9F70",
                label="数据证据分（招聘文本代理）", alpha=0.9)
        ax.barh(yp - 0.2, d[f"{dim}_expert"], height=0.35, color="#8E7CC3",
                label="专家先验分", alpha=0.9)
        ax.scatter(d[dim], yp, color="#C0392B", s=32, zorder=5,
                   marker="D", label="融合分")
        ax.set_title(DIM_LABEL[dim], fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(yp)
    axes[0].set_yticklabels(d.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("图5　五维指标：招聘数据证据分 vs 专家先验分 vs 融合分\n"
                 "融合权重随样本量收缩——样本越少越向专家先验靠拢（可追溯性检验）",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, DATA_PROXY_NOTE, ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_data_vs_expert.png")
    plt.close(fig)


def fig6_evidence(cat: pd.DataFrame):
    """直接呈现招聘数据的原始证据（覆盖率），避免"评分凭空产生"的质疑。"""
    src = DATA / "02_category_feature_matrix.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"缺少 {src.name}：图6 直接绘制招聘文本的原始覆盖率，"
            f"必须先运行 01_job_classification.py 与 02_feature_engineering.py 生成该文件。"
        )
    m = pd.read_csv(src, index_col=0)
    d = m.reindex(cat.sort_values("Risk", ascending=False).index)
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    x = np.arange(len(d))
    w = 0.27
    ax.bar(x - w, d["AI渗透度"] * 100, w, label="AI 技术要求覆盖率", color="#C0392B")
    ax.bar(x, d["数字化技能覆盖率"] * 100, w, label="数字化硬技能覆盖率", color="#E29B26")
    ax.bar(x + w, d["人际能力覆盖率"] * 100, w, label="高阶人际软技能覆盖率", color="#2E86AB")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=32, ha="right", fontsize=9)
    ax.set_ylabel("该类岗位中出现相应要求的岗位占比 (%)", fontsize=10.5)
    ax.set_title("图6　招聘文本原始证据：三类技能要求在各岗位大类中的覆盖率\n"
                 "（A、D、H 三个维度数据证据分的直接来源）",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig6_evidence.png")
    plt.close(fig)


def fig9_risk_vs_salary(cat: pd.DataFrame):
    """替代风险 × 薪资水平：识别"低替代风险 ≠ 高职业价值"的结构性陷阱。"""
    d = cat.dropna(subset=["薪资中位数K"])
    med = 11.5   # 全样本薪资中位数（见 02 特征日志）
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.scatter(d["Risk"], d["薪资中位数K"], s=90 + d["岗位数"] ** 0.62 * 22,
               c=[TIER_COLOR[t] for t in d["风险等级"]], alpha=0.72,
               edgecolors="white", linewidths=1.6, zorder=3)
    ax.axhline(med, color="#444", ls="--", lw=1.1)
    ax.axvline(50, color="#444", ls="--", lw=1.1)
    ax.text(30, med + 0.35, f"全样本薪资中位数 {med}K", fontsize=8.5, color="#444")

    notes = [(0.03, 0.96, "高薪 · 低替代\n→ 最优职业区间", "left", "top"),
             (0.97, 0.96, "高薪 · 高替代\n→ 需主动转向人机协同角色", "right", "top"),
             (0.03, 0.04, "低薪 · 低替代\n→「低风险陷阱」：不被替代，\n但议价能力弱、增长空间有限", "left", "bottom"),
             (0.97, 0.04, "低薪 · 高替代\n→ 最应尽快转型", "right", "bottom")]
    for fx, fy, t, ha, va in notes:
        ax.text(fx, fy, t, transform=ax.transAxes, fontsize=9, ha=ha, va=va,
                color="#444", bbox=dict(boxstyle="round,pad=0.38", fc="white",
                                        ec="#bbb", alpha=0.88))
    ax.set_xlim(28, 80)
    ax.set_ylim(5.5, 23)
    XR, YR = 80 - 28, 23 - 5.5
    CAND = [(0, 0.052), (0, -0.062), (0, 0.098), (0, -0.108),
            (0.115, 0.022), (-0.115, 0.022), (0.115, -0.035), (-0.115, -0.035),
            (0, 0.145), (0, -0.155)]
    placed = []

    def ov(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for i, r in d.sort_values("岗位数", ascending=False).iterrows():
        w, h = len(i) * 0.0155 * XR, 0.034 * YR
        for fdx, fdy in CAND:
            cx, cy = r["Risk"] + fdx * XR, r["薪资中位数K"] + fdy * YR
            bx = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if not any(ov(bx, q) for q in placed):
                placed.append(bx)
                ax.annotate(i, (r["Risk"], r["薪资中位数K"]), xytext=(cx, cy),
                            textcoords="data", ha="center", va="center", fontsize=8.5,
                            zorder=5,
                            arrowprops=dict(arrowstyle="-", color="#999", lw=0.7,
                                            shrinkA=0, shrinkB=6)
                            if abs(fdx) + abs(fdy) > 0.06 else None)
                break
    ax.set_xlabel("AI 替代风险指数 Risk →", fontsize=11)
    ax.set_ylabel("岗位薪资中位数（K/月，折算 12 薪）→", fontsize=11)
    ax.set_title("图9　AI 替代风险 × 薪资水平：低替代风险并不等于高职业价值\n"
                 "气泡大小 = 招聘需求量", fontsize=13, fontweight="bold", pad=14)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig9_risk_vs_salary.png")
    plt.close(fig)


def main():
    cat = pd.read_csv(DATA / "03_risk_scores_category.csv", index_col=0)
    roles = pd.read_csv(DATA / "03_risk_scores_typical_roles.csv")

    print("[1/7] 风险排名图 ...");      fig1_ranking(roles)
    print("[2/7] 二维风险矩阵 ...");    fig2_matrix(roles)
    print("[3/7] 五维雷达图 ...");      fig3_radar(roles)
    print("[4/7] 五维热力图 ...");      fig4_heatmap(cat)
    print("[5/7] 数据分/先验分对照 ..."); fig5_data_vs_expert(cat)
    print("[6/7] 原始证据覆盖率图 ...")
    try:
        fig6_evidence(cat)
    except FileNotFoundError as e:
        print(f"      [跳过] {e}")
    print("[7/7] 风险-薪资散点图 ..."); fig9_risk_vs_salary(cat)

    for f in sorted(FIG.glob("*.png")):
        print(f"  ✓ {f.name}  ({f.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
