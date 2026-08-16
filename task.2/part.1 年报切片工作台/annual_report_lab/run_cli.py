# -*- coding: utf-8 -*-
"""命令行版本（不开页面，适合 72 家 × 3 年这种长时间批跑）。

    python run_cli.py --csv company_list.csv --out D:\\reports --years 2025 2024 2023
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import (ABNORMAL, DEEPSEEK_ADVANCED_MODEL, DEEPSEEK_API_KEY,
                         DEEPSEEK_MODEL, DEFAULT_OUTPUT, STATUS_LABEL,
                         TUSHARE_TOKEN)
from core.pipeline import STATE, run_job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2024, 2023])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefer", default="auto", choices=["auto", "tushare", "cninfo"])
    ap.add_argument("--token", default=TUSHARE_TOKEN)
    ap.add_argument("--overwrite", action="store_true",
                    help="忽略已有完成状态，重新下载并重做所选任务")
    ap.add_argument("--no-resume", action="store_true",
                    help="不读取 manifest/检查点/已有章节成果")
    ap.add_argument("--no-fulltext", action="store_true")
    ap.add_argument("--deepseek-key", default=DEEPSEEK_API_KEY,
                    help="填了就启用 AI 辅助定位 / 业务概况概括 / 切片后验")
    ap.add_argument("--deepseek-model", default=DEEPSEEK_MODEL, help="定位用的常规模型")
    ap.add_argument("--advanced-model", default=DEEPSEEK_ADVANCED_MODEL,
                    help="业务概况概括与切片后验用的高级模型")
    ap.add_argument("--no-ai-summary", action="store_true",
                    help="不生成业务概况的 AI 概括")
    ap.add_argument("--no-verify", action="store_true", help="不做切片后验")
    a = ap.parse_args()

    cfg = {"csv_path": a.csv, "output": a.out, "years": a.years,
           "priority_year": max(a.years), "workers": a.workers,
           "prefer": a.prefer, "token": a.token, "overwrite": a.overwrite,
           "resume": not a.no_resume,
           "save_fulltext": not a.no_fulltext,
           "ai_enabled": bool((a.deepseek_key or "").strip()),
           "deepseek_key": a.deepseek_key,
           "deepseek_model": a.deepseek_model,
           "deepseek_advanced_model": a.advanced_model,
           "ai_summary": not a.no_ai_summary,
           "ai_verify": not a.no_verify}

    t = threading.Thread(target=run_job, args=(cfg,), daemon=True)
    t.start()

    shown = 0
    while t.is_alive() or shown < len(STATE.logs):
        logs = STATE.logs[shown:]
        shown += len(logs)
        for l in logs:
            print(f"[{l['t']}] {l['msg']}")
        time.sleep(0.4)

    print("\n===== 汇总 =====")
    counts = {}
    for r in STATE.records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {STATUS_LABEL.get(k, k):<16} {v}")
    print("  需人工复核：",
          sum(v for k, v in counts.items() if k in ABNORMAL))
    rd_na = [r for r in STATE.records
             if (r.get("sections") or {}).get("rd", {}).get("na")]
    rd_miss = [r for r in STATE.records
               if str(r.get("rd_status", "")).startswith("小节缺失")]
    biz_ai = [r for r in STATE.records if r.get("business_origin") == "AI 概括生成"]
    vbad = [r for r in STATE.records
            if r.get("verify_worst") in ("fail", "warn", "error")]
    print(f"  研发投入·公司填报不适用：{len(rd_na)}")
    for r in rd_na[:30]:
        print(f"      {r['ts_code']} {r['name']} {r['year']}")
    print(f"  研发投入·小节缺失：{len(rd_miss)}")
    for r in rd_miss[:30]:
        print(f"      {r['ts_code']} {r['name']} {r['year']}")
    print(f"  业务概况·AI 概括生成：{len(biz_ai)}（非年报原文）")
    print(f"  切片后验·存疑或失败：{len(vbad)}")
    for k, p in (STATE.exports or {}).items():
        print(f"  结果表({k})：{p}")


if __name__ == "__main__":
    main()
