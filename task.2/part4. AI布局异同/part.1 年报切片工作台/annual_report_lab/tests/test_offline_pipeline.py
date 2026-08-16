# -*- coding: utf-8 -*-
"""不联网的端到端冒烟测试：切片 → 后验（AI 关闭）→ 落盘 → 导出结果表。

    python tests/test_offline_pipeline.py

用来确认改动后的落盘字段、Excel 列、_meta.json 结构都还是通的。
真实的 AI 概括与后验需要 DeepSeek Key，这里只验证"AI 关掉时能正常降级"。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_assist import DeepSeekClient, render_summary            # noqa: E402
from core.config import SECTION_KEYS, SECTION_NAMES, SS              # noqa: E402
from core.pipeline import _business_origin_label, _rd_status_label   # noqa: E402
from core.slicer import slice_report                                 # noqa: E402
from core.store import Store                                         # noqa: E402
from core.verify import summarize_verdicts, verify_sections          # noqa: E402
from test_slicer import RD_DETAIL, RD_NA, build_new_layout           # noqa: E402


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="arlab_")
    ok = True
    try:
        for label, raw in (("有研发投入", build_new_layout(RD_DETAIL)),
                           ("研发不适用", build_new_layout(RD_NA))):
            print(f"\n===== {label} =====")
            sliced = slice_report(raw)
            sections = sliced["sections"]

            client = DeepSeekClient("")            # 没 Key → 全部降级
            sections, verdicts = verify_sections(sliced["normalized"], sections, client)
            worst, problems, detail = summarize_verdicts(verdicts)
            print("  后验：", worst, "|", detail)
            assert worst == "skip" and not problems, "AI 关闭时后验应全部 skip"

            store = Store(os.path.join(tmp, label))
            paths = store.save_sections("000001.SZ", "示例科技", 2025, sections,
                                        verdicts,
                                        {"name": "示例科技", "ts_code": "000001.SZ",
                                         "title": "2025 年年度报告",
                                         "notes": sliced["notes"]})
            for k in SECTION_KEYS:
                mark = "√" if k in paths else "×"
                print(f"  {mark} {SECTION_NAMES[k]}：{sections[k].get('how')}")
            d = store.section_dir("000001.SZ", "示例科技", 2025)
            meta = json.load(open(os.path.join(d, "_meta.json"), encoding="utf-8"))
            assert "verify" in meta and "slice_notes" in meta

            rd = sections["rd"]
            print("  研发投入状态：", _rd_status_label(rd))
            print("  业务概况来源：", _business_origin_label(sections["business"]))
            if label == "研发不适用":
                ok &= rd.get("status") == SS.NOT_APPLICABLE
                txt = open(paths["rd"], encoding="utf-8-sig").read()
                assert "不适用" in txt, "不适用切片文件应写明依据"
                print("  研发投入文件头：", txt.splitlines()[1])

            rec = {"ts_code": "000001.SZ", "name": "示例科技", "year": 2025,
                   "status": "PARTIAL", "sections":
                       {k: {kk: vv for kk, vv in sections[k].items() if kk != "text"}
                        for k in SECTION_KEYS},
                   "rd_status": _rd_status_label(rd),
                   "business_origin": _business_origin_label(sections["business"]),
                   "verify": verdicts, "verify_worst": worst,
                   "verify_detail": detail, "verify_problems": problems,
                   "slice_notes": sliced["notes"], "message": ""}
            exports = store.export_table([rec])
            print("  导出：", {k: os.path.basename(v) for k, v in exports.items()})
            head = open(exports["csv"], encoding="utf-8-sig").readline().strip()
            print("  结果表列：", head[:200], "…")
            assert "研发投入是否不适用" in head and "业务概况来源" in head
            assert "后验结论" in head

        print("\n===== 概括渲染（离线用假 JSON） =====")
        demo = {"一句话定位": "一家企业级软件公司",
                "主营业务与主要产品服务": ["数据中台", "智能风控平台"],
                "经营模式": {"销售模式": "直销与渠道结合", "研发模式": "年报未披露"},
                "年报未披露的条目": ["主要客户名称"]}
        print(render_summary(demo))
        print("\n全部通过" if ok else "\n存在失败项")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
