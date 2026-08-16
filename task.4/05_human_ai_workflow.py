# -*- coding: utf-8 -*-
"""
task4 / 05_human_ai_workflow.py
------------------------------------------------------------------
阶段 7：典型岗位任务拆解 + "人类 + AI" 协同工作流可视化

选定岗位：信贷审批与贷后管理（Risk 71.9，高替代风险，位于二维矩阵 Ⅳ 高危区）
    —— 该岗位同时具备"高规则化 + 高数字化 + 低人类不可替代性"，
       是人机职责边界最清晰、最适合做 Human-in-the-loop 设计的岗位。

输出（figures/）：
    fig7_workflow.png        人机协同工作流（含置信度决策节点与反馈闭环）
    fig8_task_boundary.png   任务级人机职责边界矩阵
输出（data/）：
    05_task_decomposition.csv  信贷审批岗任务拆解表（中间产物）
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

BASE = Path(__file__).parent
FIG = BASE / "figures"
DATA = BASE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

C_AI, C_HUMAN, C_BOTH, C_DATA = "#2E86AB", "#C0392B", "#7B5EA7", "#5F6B73"

# ---------------------------------------------------------------- 任务拆解表
# 责任分配：A=AI主导 / H=人类主导 / C=协同
TASKS = [
    # 任务, 环节, 耗时占比%, 重复性, 规则化, 数字化, AI可承担度, 责任归属, 人类角色
    ("客户资料收集与影像归档", "受理", 12, 95, 95, 90, 95, "A", "异常件补录、原件核验"),
    ("证件与财报要素提取（OCR）", "受理", 10, 95, 90, 95, 95, "A", "识别置信度低的字段复核"),
    ("征信报告解析与数据入库", "尽调", 8, 90, 95, 95, 95, "A", "报告缺失/异议处理"),
    ("反欺诈规则与名单筛查", "尽调", 6, 90, 95, 95, 90, "A", "命中名单的申诉复核"),
    ("财务指标计算与比率分析", "尽调", 8, 88, 90, 92, 92, "A", "抽查勾稽关系"),
    ("经营真实性交叉验证", "尽调", 10, 45, 40, 55, 45, "C", "实地走访、上下游访谈"),
    ("信用评分卡打分", "评估", 6, 85, 92, 95, 92, "A", "模型分与经验判断的偏离审查"),
    ("风险摘要与授信报告生成", "评估", 10, 60, 55, 80, 78, "C", "结论改写、风险点补充"),
    ("授信额度与定价方案设计", "决策", 8, 40, 55, 70, 55, "C", "风险偏好与客户关系权衡"),
    ("异常与非标案件判断", "决策", 8, 15, 20, 35, 25, "H", "行业周期、软信息、动机识别"),
    ("最终审批签字与问责", "决策", 4, 10, 25, 30, 10, "H", "承担合规与法律责任"),
    ("贷后监测与预警触发", "贷后", 6, 90, 90, 92, 92, "A", "预警分级与处置决策"),
    ("逾期客户沟通与重组谈判", "贷后", 4, 25, 25, 30, 25, "H", "情绪安抚、还款方案协商"),
]
COLS = ["任务", "环节", "耗时占比%", "重复性", "规则化", "数字化",
        "AI可承担度", "责任归属", "人类角色"]


def export_tasks() -> pd.DataFrame:
    df = pd.DataFrame(TASKS, columns=COLS)
    label = {"A": "AI主导·人监督", "C": "人机协同", "H": "人类主导·AI辅助"}
    df["责任归属说明"] = df["责任归属"].map(label)
    df.to_csv(DATA / "05_task_decomposition.csv", index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------- 工作流图
def box(ax, x, y, w, h, text, fc, fontsize=9, tc="white", style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                fc=fc, ec="white", lw=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=tc, zorder=4, linespacing=1.45)


def diamond(ax, x, y, w, h, text, fc):
    ax.add_patch(plt.Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2],
                              [x - w / 2, y]], fc=fc, ec="white", lw=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color="white",
            zorder=4, linespacing=1.4)


def arrow(ax, p1, p2, label="", color="#666", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.3, zorder=2, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=8, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
                zorder=5)


def fig7_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 10.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- 主链路（AI 前段）----
    box(ax, 20, 94, 30, 6.5, "① 原始客户资料\n（申请表 / 证件 / 财报 / 流水）", C_DATA)
    box(ax, 20, 84, 30, 6.5, "② OCR + NLP 信息抽取\nAI 主导 · 字段级置信度输出", C_AI)
    box(ax, 20, 74, 30, 6.5, "③ 数据清洗与特征工程\n征信 / 工商 / 税务多源合并", C_AI)
    box(ax, 20, 64, 30, 6.5, "④ 机器学习风险评分\n评分卡 + XGBoost 违约概率 PD", C_AI)
    box(ax, 20, 54, 30, 7.5, "⑤ 大语言模型生成风险摘要\n关键风险点 · 佐证片段 · 建议额度\n（要求逐条附数据出处）", C_AI)

    for y1, y2 in [(90.75, 87.25), (80.75, 77.25), (70.75, 67.25), (60.75, 57.75)]:
        arrow(ax, (20, y1), (20, y2))

    # ---- 决策节点 ----
    diamond(ax, 20, 41, 26, 12,
            "⑥ 置信度 / 规则双重判断\nOCR 置信度 ≥0.95 ？\nPD 落在明确区间 ？\n无红线规则命中 ？", "#E29B26")
    arrow(ax, (20, 50.25), (20, 47))

    # ---- 分支 ----
    box(ax, 6, 24, 24, 8.5, "⑦A 高置信标准件\nAI 出具建议结论\n人工抽样复核（≥10%）",
        C_AI, fontsize=8.6)
    box(ax, 40, 24, 30, 8.5, "⑦B 低置信 / 异常 / 非标件\n人工重点复核\nAI 提供证据包与相似案例",
        C_HUMAN, fontsize=8.6)
    arrow(ax, (11, 38.5), (6, 28.25), "高置信\n约 70% 案件", rad=-0.12)
    arrow(ax, (29, 38.5), (40, 28.25), "低置信 / 命中异常规则\n约 30% 案件", rad=0.12)

    # ---- 人类决策 ----
    box(ax, 23, 11, 40, 7.5, "⑧ 人工最终审批与签字\n承担合规与法律责任 · 记录否决理由", C_HUMAN)
    arrow(ax, (6, 19.75), (18, 14.75), rad=0.1)
    arrow(ax, (40, 19.75), (30, 14.75), rad=-0.1)

    # ---- 右侧闭环 ----
    box(ax, 76, 74, 32, 7.5, "⑨ 贷后表现回流\n逾期 / 违约 / 提前还款标签", C_DATA)
    box(ax, 76, 60, 32, 8.5, "⑩ 模型监控与再训练\nPSI 漂移检测 · 分群 KS\n人工否决样本作为高价值负例", C_BOTH)
    box(ax, 76, 45, 32, 8.5, "⑪ 策略与阈值调优\n人类设定风险偏好\nAI 给出参数—损失曲线", C_BOTH)
    box(ax, 76, 30, 32, 8.5, "⑫ 模型可解释性与合规审计\nSHAP 归因 · 公平性检验\n全链路留痕可回溯", C_HUMAN)

    arrow(ax, (43, 11), (76, 26), "决策结果与理由入库", rad=-0.25, color="#8A8A8A")
    arrow(ax, (76, 34.25), (76, 40.75), color="#8A8A8A")
    arrow(ax, (76, 49.25), (76, 55.75), color="#8A8A8A")
    arrow(ax, (76, 64.25), (76, 70.25), color="#8A8A8A")
    arrow(ax, (60, 74), (35, 64), "模型迭代回流", rad=0.2, color="#8A8A8A", ls="--")

    # ---- 图例与标题 ----
    handles = [Patch(fc=C_AI, label="AI 主导（人类监督/抽查）"),
               Patch(fc=C_HUMAN, label="人类主导（AI 提供证据）"),
               Patch(fc=C_BOTH, label="人机协同（共同决策）"),
               Patch(fc="#E29B26", label="决策分流节点"),
               Patch(fc=C_DATA, label="数据与事实层")]
    ax.legend(handles=handles, loc="upper right", fontsize=9.5, frameon=True,
              bbox_to_anchor=(1.0, 0.99))
    ax.set_title("图7　信贷审批岗「人类 + AI」协同工作流（Human-in-the-Loop）\n"
                 "AI 承担识别—计算—初判，人类承担判断—问责—纠偏；贷后结果回流形成闭环",
                 fontsize=13.5, fontweight="bold", pad=16)
    fig.savefig(FIG / "fig7_workflow.png")
    plt.close(fig)


# ------------------------------------------------------- 职责边界矩阵图
def fig8_boundary(df: pd.DataFrame):
    d = df.sort_values("AI可承担度", ascending=True).reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                   gridspec_kw={"width_ratios": [2.1, 1]})

    # 左：AI 可承担度 + 责任归属
    cmap = {"A": C_AI, "C": C_BOTH, "H": C_HUMAN}
    colors = [cmap[x] for x in d["责任归属"]]
    bars = ax1.barh(d["任务"], d["AI可承担度"], color=colors, height=0.68,
                    edgecolor="white")
    for b, v, r in zip(bars, d["AI可承担度"], d["耗时占比%"]):
        ax1.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v}",
                 va="center", fontsize=9, color="#333", fontweight="bold")
        ax1.text(1.5, b.get_y() + b.get_height() / 2, f"耗时{r}%",
                 va="center", fontsize=7.6, color="white")
    ax1.axvline(70, color="#666", ls="--", lw=1)
    ax1.text(70, len(d) - 0.2, "AI 可承担度 70 分界线", fontsize=8, color="#666", ha="center")
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("AI 可承担度（0–100）", fontsize=10.5)
    ax1.set_title("信贷审批岗 13 项任务的人机职责边界", fontsize=12, fontweight="bold")
    ax1.legend(handles=[Patch(fc=C_AI, label="AI 主导 · 人监督"),
                        Patch(fc=C_BOTH, label="人机协同"),
                        Patch(fc=C_HUMAN, label="人类主导 · AI 辅助")],
               loc="lower right", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.2)

    # 右：工时结构重构（AI 化前 vs 后）
    grp = df.groupby("责任归属")["耗时占比%"].sum().reindex(["A", "C", "H"]).fillna(0)
    labels = ["AI 主导\n(可自动化)", "人机协同", "人类主导"]
    before = grp.values
    # 假设：AI 主导任务人工工时压缩至 15%，协同任务压缩至 55%，人类任务不变
    after = np.array([grp["A"] * 0.15, grp["C"] * 0.55, grp["H"]])
    after = after / after.sum() * 100

    x = np.arange(3)
    ax2.bar(x - 0.2, before, 0.4, label="AI 化之前的人工工时结构",
            color=["#9FC5D8", "#C3B4DC", "#E0A9A2"], edgecolor="white")
    ax2.bar(x + 0.2, after, 0.4, label="AI 化之后的人工工时结构",
            color=[C_AI, C_BOTH, C_HUMAN], edgecolor="white")
    for i, (b, a) in enumerate(zip(before, after)):
        ax2.text(i - 0.2, b + 1, f"{b:.0f}%", ha="center", fontsize=9)
        ax2.text(i + 0.2, a + 1, f"{a:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("占人工总工时比重 (%)", fontsize=10)
    ax2.set_title("协同后人力从「执行」转向「判断」", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 68)
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("图8　任务级人机职责边界与工时结构重构", fontsize=13.5,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_task_boundary.png")
    plt.close(fig)


def main():
    print("[1/3] 导出任务拆解表 ...")
    df = export_tasks()
    print(f"      共 {len(df)} 项任务，"
          f"AI 主导 {(df['责任归属']=='A').sum()} 项 / "
          f"协同 {(df['责任归属']=='C').sum()} 项 / "
          f"人类主导 {(df['责任归属']=='H').sum()} 项")
    ai_time = df.loc[df["责任归属"] == "A", "耗时占比%"].sum()
    print(f"      AI 可主导任务占原工时 {ai_time}%")

    print("[2/3] 绘制人机协同工作流 ...")
    fig7_workflow()
    print("[3/3] 绘制职责边界矩阵 ...")
    fig8_boundary(df)
    for f in ["fig7_workflow.png", "fig8_task_boundary.png"]:
        print(f"  ✓ {f}  ({(FIG/f).stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
