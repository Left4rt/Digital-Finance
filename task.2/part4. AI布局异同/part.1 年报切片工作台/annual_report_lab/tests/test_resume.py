# -*- coding: utf-8 -*-
"""断点续跑离线测试。

    python tests/test_resume.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import S, SECTION_KEYS, SS  # noqa: E402
from core.pipeline import RunState, _record, _restore_task, run_job  # noqa: E402
from core.store import Store, write_text  # noqa: E402


def make_sections():
    return {
        key: {
            "found": True,
            "how": "测试切片",
            "chars": 600,
            "status": SS.ORIGINAL,
            "text": (f"{key} 测试正文\n" * 80),
            "loose": False,
            "ai": False,
        }
        for key in SECTION_KEYS
    }


def make_record(store: Store, status=S.OK):
    sections = make_sections()
    files = store.save_sections(
        "000001.SZ", "示例科技", 2025, sections, {},
        {"name": "示例科技", "ts_code": "000001.SZ", "title": "2025 年年度报告"})
    return {
        **_record("000001.SZ", "示例科技", 2025),
        "status": status,
        "message": "四个章节全部就绪",
        "sections": {
            k: {kk: vv for kk, vv in sections[k].items() if kk != "text"}
            for k in SECTION_KEYS
        },
        "section_files": files,
        "rd_status": "已切出（有内容）",
        "business_origin": "年报原文切出",
    }


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="arlab_resume_")
    try:
        print("===== 1. manifest 与完成成果恢复 =====")
        store = Store(tmp)
        rec = make_record(store)
        store.save_manifest([rec], {"run_id": "old"})
        history, info = store.load_resume_records()
        got = {r["key"]: r for r in history}["000001.SZ|2025"]
        assert store.is_reusable_record(got)
        restored, skipped = _restore_task(_record("000001.SZ", "示例科技", 2025),
                                          got, store)
        assert skipped and restored["status"] == S.OK and restored["resumed"]
        print("  [PASS] 已完成任务读取后直接复用")

        print("\n===== 2. RUNNING 检查点会继续执行 =====")
        running = dict(rec)
        running["status"] = S.RUNNING
        running["message"] = "处理中"
        store.save_task_checkpoint(running, {"run_id": "crashed"})
        history, _ = store.load_resume_records()
        got = {r["key"]: r for r in history}["000001.SZ|2025"]
        assert got["status"] == S.RUNNING
        restored, skipped = _restore_task(_record("000001.SZ", "示例科技", 2025),
                                          got, store)
        assert not skipped and restored["status"] == S.PENDING
        assert restored["previous_status"] == S.RUNNING
        print("  [PASS] 崩溃时正在执行的任务不会误判为完成")

        print("\n===== 3. 单任务完成检查点覆盖中断状态 =====")
        store.save_task_checkpoint(rec, {"run_id": "resumed"})
        history, info = store.load_resume_records()
        got = {r["key"]: r for r in history}["000001.SZ|2025"]
        assert got["status"] == S.OK and store.is_reusable_record(got)
        assert info["checkpoints"] >= 1
        print("  [PASS] 每任务检查点可独立恢复")

        print("\n===== 4. 主 manifest 损坏时读取备份 =====")
        rec2 = dict(rec)
        rec2["message"] = "第二版"
        store.save_manifest([rec2], {"run_id": "new"})  # 生成 .bak
        with open(store.manifest_path, "w", encoding="utf-8") as f:
            f.write("{broken")
        payload, source = store.load_manifest()
        assert payload and source.endswith(".bak")
        print("  [PASS] 自动回退 manifest.json.bak")

        print("\n===== 5. 兼容旧版本：仅凭 _meta.json 恢复 =====")
        tmp2 = tempfile.mkdtemp(prefix="arlab_resume_meta_")
        try:
            store2 = Store(tmp2)
            make_record(store2)  # 只留下 sections/**/_meta.json
            history, info = store2.load_resume_records()
            got = {r["key"]: r for r in history}["000001.SZ|2025"]
            assert got["status"] == S.OK
            assert got.get("_resume_source") == "章节元数据"
            assert store2.is_reusable_record(got)
            print("  [PASS] 没有旧 manifest 也能从既有章节成果恢复")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("\n===== 6. 公司简称变化仍能找到旧 PDF/全文 =====")
        old_pdf = store.pdf_path("000002.SZ", "旧简称", 2024)
        os.makedirs(os.path.dirname(old_pdf), exist_ok=True)
        with open(old_pdf, "wb") as f:
            f.write(b"x" * 12000)
        old_text = store.text_path("000002.SZ", "旧简称", 2024)
        write_text(old_text, "已有全文\n" * 200)
        assert store.find_existing_pdf("000002.SZ", "新简称", 2024) == old_pdf
        assert store.find_existing_text("000002.SZ", "新简称", 2024) == old_text
        print("  [PASS] 通过股票代码和年度发现旧文件")

        print("\n===== 7. 整轮启动自动跳过已有成果（无需联网） =====")
        tmp3 = tempfile.mkdtemp(prefix="arlab_resume_job_")
        try:
            out = os.path.join(tmp3, "out")
            store3 = Store(out)
            rec3 = make_record(store3)
            store3.save_task_checkpoint(rec3, {"run_id": "previous"})
            csv_path = os.path.join(tmp3, "companies.csv")
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                f.write("ts_code,name\n000001.SZ,示例科技\n")
            state = RunState()
            run_job({
                "csv_path": csv_path,
                "output": out,
                "years": [2025],
                "priority_year": 2025,
                "workers": 1,
                "resume": True,
                "overwrite": False,
                "save_fulltext": True,
                "ai_enabled": False,
            }, state)
            assert state.total == 1 and state.done == 1 and state.resumed == 1
            assert state.records[0]["status"] == S.OK
            assert state.records[0]["resume_skipped"]
            assert os.path.isfile(state.exports["csv"])
            assert not state.error
            print("  [PASS] 全部已有时不连接外部服务，直接恢复并导出")
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)

        print("\n全部通过")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
