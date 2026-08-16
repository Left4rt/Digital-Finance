# -*- coding: utf-8 -*-
"""切片器离线自测：不联网、不调 AI，只验证确定性规则。

    python tests/test_slicer.py

三份合成年报覆盖了实际会踩的坑：
  A 新版式（2021 后）+ 研发投入有详细内容 + 正文里有"加大研发投入"这类干扰句
  B 新版式 + 研发投入勾选「不适用」
  C 旧版式（2019）：第三节 公司业务概要 / 第四节 经营情况讨论与分析
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.slicer import detect_not_applicable, slice_report, squeeze  # noqa: E402

HEADER = "示例科技股份有限公司 2025 年年度报告"


def page(body: str, n: int) -> str:
    """模拟 PDF 每页都有页眉页脚。"""
    return f"{HEADER}\n{body}\n第 {n} 页 共 200 页\n"


def filler(topic: str, times: int = 6) -> str:
    return "\n".join(
        f"报告期内，公司{topic}保持稳健，各项经营指标符合预期，"
        f"经营活动现金流量净额同比变动，具体情况详见财务报表附注。" for _ in range(times))


TOC = """目录
第一节 释义 ....................................... 4
第二节 公司简介和主要财务指标 ..................... 6
第三节 管理层讨论与分析 ........................... 12
第四节 公司治理 ................................... 60
第五节 环境和社会责任 ............................. 88
第六节 重要事项 ................................... 95
"""


def build_new_layout(rd_block: str) -> str:
    parts = [
        page("示例科技股份有限公司\n2025 年年度报告\n2026 年 4 月", 1),
        page("第一节 重要提示、目录和释义\n公司董事会及全体董事保证本报告内容不存在虚假记载。", 2),
        page(TOC, 3),
        page("第一节 释义\n本报告中，除非另有说明，下列词语具有如下含义。\n"
             "公司、本公司：指示例科技股份有限公司。\n" + filler("释义相关表述"), 4),
        page("第二节 公司简介和主要财务指标\n一、公司简介\n股票简称：示例科技\n"
             "二、主要会计数据和财务指标\n营业收入 1,234,567,890.00 元。\n"
             + filler("主要财务指标"), 6),
        page("第三节 管理层讨论与分析\n"
             "一、报告期内公司所处行业情况\n"
             "公司所处行业为软件和信息技术服务业。行业整体保持较快增长，"
             "下游需求旺盛，行业集中度持续提升。\n" + filler("所处行业"), 12),
        page("二、报告期内公司从事的主要业务\n"
             "公司主要从事企业级软件的研发、销售与技术服务，主要产品包括数据中台、"
             "智能风控平台。公司采用直销与渠道相结合的销售模式。\n"
             "报告期内，公司持续加大研发投入力度，研发投入占营业收入比例逐年提升。\n"
             + filler("主要业务"), 14), 
        page("三、核心竞争力分析\n"
             "（一）技术优势\n公司在分布式计算领域积累了多项核心专利。\n"
             "（二）人才优势\n公司核心技术团队稳定。\n" + filler("核心竞争力"), 20),
        page("四、主营业务分析\n"
             "（一）概述\n参见本节「一、报告期内公司所处行业情况」相关内容。\n"
             + filler("主营业务概述"), 26),
        page("（二）收入与成本\n1、营业收入构成\n分行业情况如下表所示。\n"
             + filler("收入构成"), 30),
        page("（三）费用\n销售费用同比增长 12.30%。\n" + filler("费用"), 34),
        page(rd_block, 38),
        page("（五）现金流\n经营活动产生的现金流量净额为正。\n" + filler("现金流"), 44),
        page("五、非主营业务分析\n主要为投资收益。\n" + filler("非主营业务"), 48),
        page("六、资产及负债状况分析\n货币资金占总资产比例为 22.15%。\n"
             "有关公司治理情况详见第四节 公司治理。\n" + filler("资产负债"), 52),
        page("七、公司未来发展的展望\n公司将继续聚焦主业。\n" + filler("未来展望"), 56),
        page("第四节 公司治理\n一、公司治理的基本状况\n公司严格按照《公司法》规范运作。\n"
             + filler("公司治理"), 60),
        page("第五节 环境和社会责任\n一、重大环保问题\n不适用。\n" + filler("环境责任"), 88),
        page("第六节 重要事项\n一、承诺事项履行情况\n" + filler("重要事项"), 95),
    ]
    return "\n".join(parts)


RD_DETAIL = """（四）研发投入
1、研发投入情况
报告期内公司研发投入总额为 156,780,000.00 元，占营业收入的比例为 12.70%，
较上年同期增长 18.65%。研发投入资本化金额为 0.00 元，资本化比例为 0.00%。
2、研发人员情况
公司研发人员数量为 1,286 人，占公司总人数的比例为 38.60%，
其中硕士及以上学历 412 人。
3、主要研发项目
新一代数据中台 V5.0，项目进度 80%，拟达到的目标为提升实时计算性能。
智能风控平台二期，项目进度 60%。
""" + filler("研发项目推进情况", 8)

RD_NA = """（四）研发投入
□适用 √不适用
"""

RD_NA_TEXT = """（四）研发投入
不适用
"""


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))
    return cond


def run_case(title: str, raw: str, expect_rd_na: bool) -> bool:
    print(f"\n===== {title} =====")
    r = slice_report(raw)
    ok = True
    ch = r["chapters"]
    print("  识别到的节：" + " | ".join(f"第{c['ordinal']}节 {c['title']}({c['chars']}字)"
                                        for c in ch))
    s = r["sections"]

    m = s["mdna"]
    ok &= check("管理层讨论与分析已切出", m["found"], m["how"])
    if m["found"]:
        body = m["text"]
        ok &= check("整节起点是节标题", body.lstrip().startswith("第三节"),
                    repr(body[:20]))
        ok &= check("整节包含最后一条条目（七、公司未来发展的展望）",
                    "七、公司未来发展的展望" in body)
        ok &= check("整节未越界到下一节（第四节 公司治理）",
                    "第四节 公司治理\n一、公司治理的基本状况" not in body)
        ok &= check("整节长度合理", m["chars"] > 3000, f"{m['chars']} 字")

    b = s["business"]
    ok &= check("业务概况从条目切出", b["found"], b["how"])
    if b["found"]:
        ok &= check("业务概况含行业情况与主要业务",
                    "所处行业" in b["text"] and "主要业务" in b["text"])
        ok &= check("业务概况未吃掉核心竞争力",
                    "三、核心竞争力分析" not in b["text"])

    c = s["core"]
    ok &= check("核心竞争力已切出", c["found"], c["how"])
    if c["found"]:
        ok &= check("核心竞争力未吃掉主营业务分析",
                    "四、主营业务分析" not in c["text"])

    d = s["rd"]
    ok &= check("研发投入已定位", d["found"], d["how"])
    if expect_rd_na:
        ok &= check("研发投入被判定为「不适用」", d["na"], d["na_reason"])
    else:
        ok &= check("研发投入不是「不适用」", not d["na"])
        if d["found"]:
            ok &= check("研发投入含金额与人员两块内容",
                        "研发投入总额" in d["text"] and "研发人员数量" in d["text"])
            ok &= check("研发投入未越界到（五）现金流",
                        "（五）现金流" not in d["text"])
            ok &= check("研发投入没有被正文干扰句抢走起点",
                        "公司持续加大研发投入力度" not in d["text"][:200])
        ok &= check("研发投入挂在主营业务分析下", "主营业务分析" in (d["parent_title"] or ""),
                    d["parent_title"])
    for n in r["notes"]:
        print("  · " + n)
    return bool(ok)


def build_old_layout() -> str:
    toc = """目录
第一节 释义 ....................................... 4
第二节 公司简介 ................................... 6
第三节 公司业务概要 ............................... 10
第四节 经营情况讨论与分析 ......................... 18
第五节 重要事项 ................................... 60
"""
    return "\n".join([
        page("示例科技股份有限公司\n2019 年年度报告", 1),
        page(toc, 3),
        page("第一节 释义\n" + filler("释义"), 4),
        page("第二节 公司简介\n" + filler("公司简介"), 6),
        page("第三节 公司业务概要\n一、报告期内公司从事的主要业务\n"
             "公司主要从事企业级软件的研发与销售。\n" + filler("业务概要"), 10),
        page("第四节 经营情况讨论与分析\n一、概述\n" + filler("经营概述"), 18),
        page("二、主营业务分析\n（一）概述\n" + filler("主营业务"), 24),
        page("（四）研发投入\n报告期内研发投入 8,900 万元，占营业收入 9.80%。\n"
             "研发人员数量 620 人。\n" + filler("研发", 8), 30),
        page("三、核心竞争力分析\n公司具备较强的技术积累。\n" + filler("核心竞争力"), 40),
        page("第五节 重要事项\n一、承诺事项\n" + filler("重要事项"), 60),
    ])


RD_ABSENT = """（四）政府补助
报告期内公司收到与日常经营活动相关的政府补助 320 万元。
公司持续加大研发投入力度，研发投入占营业收入比例逐年提升，但本节不单列研发投入小节。
""" + filler("政府补助", 8)



def build_non_chapter_mdna(title: str, next_title: str = "三、公司治理") -> str:
    return "\n".join([
        page("目录\n二、" + title + " ........ 15\n三、公司治理 ........ 60", 2),
        page("一、公司简介\n" + filler("公司简介", 8), 5),
        page("二、" + title + "\n（一）报告期内公司所处行业情况\n" + filler("行业", 12), 15),
        page("（二）报告期内公司从事的主要业务\n" + filler("业务", 12), 25),
        page("（三）主营业务分析\n" + filler("主营业务", 12), 35),
        page("（四）资产及负债状况分析\n" + filler("资产负债", 12), 45),
        page("（五）公司未来发展的展望\n" + filler("未来展望", 12), 55),
        page(next_title + "\n（一）治理基本情况\n" + filler("治理", 8), 60),
    ])


def run_mdna_variant_cases() -> bool:
    ok = True
    print("\n===== F MD&A 任意层级与标题别名 =====")
    for title in ("经营层讨论与分析", "管理层讨论及分析"):
        r = slice_report(build_non_chapter_mdna(title))
        m = r["sections"]["mdna"]
        ok &= check(f"命中非第X节标题：{title}", m["found"], m["how"])
        ok &= check("包含未来展望", "公司未来发展的展望" in m.get("text", ""))
        ok &= check("未越界到公司治理", "治理基本情况" not in m.get("text", ""))

    raw = "\n".join([
        page("第一部分 公司简介\n" + filler("简介", 8), 3),
        page("第五部分 管理层讨论与分析\n一、报告期内公司所处行业情况\n" + filler("行业", 15), 10),
        page("二、主营业务分析\n" + filler("主营", 15), 20),
        page("三、资产及负债状况分析\n" + filler("资产", 15), 30),
        page("四、公司未来发展的展望\n" + filler("展望", 15), 40),
        page("第六部分 公司治理\n一、治理情况\n" + filler("治理", 8), 50),
    ])
    m = slice_report(raw)["sections"]["mdna"]
    ok &= check("支持第X部分", m["found"], m["how"])
    ok &= check("第X部分按下一部分结束", "第六部分公司治理" not in squeeze(m.get("text", "")))
    return bool(ok)

def main() -> int:
    all_ok = True
    all_ok &= run_case("A 新版式 · 研发投入有详细内容", build_new_layout(RD_DETAIL), False)
    all_ok &= run_case("B 新版式 · 研发投入勾选不适用", build_new_layout(RD_NA), True)
    all_ok &= run_case("B2 新版式 · 研发投入正文写不适用", build_new_layout(RD_NA_TEXT), True)
    all_ok &= run_mdna_variant_cases()

    print("\n===== E 研发投入小节整体缺失（正文里有干扰句） =====")
    r = slice_report(build_new_layout(RD_ABSENT))
    rd = r["sections"]["rd"]
    all_ok &= check("没有把正文句子「加大研发投入力度」当成小节",
                    (not rd["found"]) or ("加大研发投入力度" not in rd["text"][:120]),
                    rd["how"])
    all_ok &= check("缺失情况被记进 notes 供汇报",
                    any("研发投入" in n for n in r["notes"]) or rd["found"],
                    "；".join(r["notes"]))
    print("  研发投入：", rd["how"])

    print("\n===== C 旧版式（2019） =====")
    r = slice_report(build_old_layout())
    print("  识别到的节：" + " | ".join(f"第{c['ordinal']}节 {c['title']}({c['chars']}字)"
                                        for c in r["chapters"]))
    s = r["sections"]
    all_ok &= check("管理层讨论与分析命中「经营情况讨论与分析」", s["mdna"]["found"],
                    s["mdna"]["how"])
    all_ok &= check("业务概况命中独立的「第三节 公司业务概要」",
                    s["business"]["found"] and "第3节" in s["business"]["how"].replace("第三节", "第3节"),
                    s["business"]["how"])
    all_ok &= check("研发投入在经营情况讨论与分析内命中", s["rd"]["found"], s["rd"]["how"])

    print("\n===== D 「不适用」判定单测 =====")
    cases = [
        ("（四）研发投入\n不适用", True),
        ("（四）研发投入\n□适用 √不适用", True),
        ("（四）研发投入\n√适用 □不适用\n报告期内研发投入 1.2 亿元。", False),
        ("（四）研发投入\n是否适用 □是 √否", True),
        ("（四）研发投入\n报告期内研发投入 8,900 万元，本项目不适用于合并范围外主体。", False),
    ]
    for txt, want in cases:
        got, why = detect_not_applicable(txt.split("\n", 1)[1], txt.split("\n", 1)[0])
        all_ok &= check(f"{txt.splitlines()[1][:24]} → {'不适用' if want else '适用'}",
                        got == want, why)

    print("\n" + ("全部通过" if all_ok else "存在失败项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
