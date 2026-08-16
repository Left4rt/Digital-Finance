# -*- coding: utf-8 -*-
"""落盘与结果导出。

目录结构：
    <输出目录>/
        raw/2025/002657.SZ_中科金财_2025_年度报告.pdf
        fulltext/2025/002657.SZ_中科金财_2025.txt
        sections/2025/002657.SZ_中科金财_2025/管理层讨论与分析.txt
                                              业务概要.txt
                                              核心竞争力.txt
                                              研发投入.txt
                                              _meta.json
        manifest.json
        report.xlsx / report.csv
        logs/run_YYYYmmdd_HHMMSS.log

所有文本一律 UTF-8-SIG 写出，Windows 记事本 / Excel 直接打开不乱码。
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import shutil
import threading
from typing import Dict, List

from .config import (ABNORMAL, DIR_LOG, DIR_RAW, DIR_SECTION, DIR_TEXT,
                     MANIFEST, PRIMARY_SECTION_KEYS, REPORT_CSV, REPORT_XLSX, SECTION_KEYS,
                     SECTION_NAMES, SS, SS_LABEL, STATUS_LABEL, TEXT_ENCODING,
                     VERIFY_LABEL)
from .pdf_slice import save_pdf_section

_ILLEGAL = re.compile(r'[\\/:*?"<>|\r\n\t]+')
_STATE_DIR = ".state"
_TASK_STATE_DIR = "tasks"
_STATE_VERSION = 2


def safe_name(s: str, limit: int = 80) -> str:
    s = _ILLEGAL.sub("_", (s or "").strip())
    s = s.replace("*", "").strip(" .")
    return (s or "unnamed")[:limit]


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=TEXT_ENCODING, newline="\n") as f:
        f.write(content or "")


class Store:
    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self._lock = threading.RLock()
        for d in (DIR_RAW, DIR_TEXT, DIR_SECTION, DIR_LOG):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        os.makedirs(self.task_state_dir, exist_ok=True)

    # ---- 路径 ----
    def stem(self, ts_code: str, name: str, year: int) -> str:
        return f"{ts_code}_{safe_name(name, 20)}_{year}"

    def pdf_path(self, ts_code, name, year) -> str:
        return os.path.join(self.root, DIR_RAW, str(year),
                            self.stem(ts_code, name, year) + "_年度报告.pdf")

    def text_path(self, ts_code, name, year) -> str:
        return os.path.join(self.root, DIR_TEXT, str(year),
                            self.stem(ts_code, name, year) + ".txt")

    def section_dir(self, ts_code, name, year) -> str:
        return os.path.join(self.root, DIR_SECTION, str(year),
                            self.stem(ts_code, name, year))

    def log_path(self, run_id: str) -> str:
        return os.path.join(self.root, DIR_LOG, f"run_{run_id}.log")

    @property
    def state_dir(self) -> str:
        return os.path.join(self.root, _STATE_DIR)

    @property
    def task_state_dir(self) -> str:
        return os.path.join(self.state_dir, _TASK_STATE_DIR)

    @property
    def run_state_path(self) -> str:
        return os.path.join(self.state_dir, "run.json")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, MANIFEST)

    def task_checkpoint_path(self, ts_code: str, year: int) -> str:
        return os.path.join(self.task_state_dir, f"{safe_name(ts_code, 32)}_{int(year)}.json")

    # ---- 原子 JSON / 断点状态 ----
    def _atomic_json(self, path: str, payload: Dict, backup: bool = False) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with self._lock:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            if backup and os.path.isfile(path) and self._read_json(path):
                try:
                    shutil.copy2(path, path + ".bak")
                except OSError:
                    pass
            os.replace(tmp, path)
        return path

    @staticmethod
    def _read_json(path: str) -> Dict | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def save_run_state(self, meta: Dict) -> str:
        payload = {
            "version": _STATE_VERSION,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "meta": meta,
        }
        return self._atomic_json(self.run_state_path, payload)

    def save_task_checkpoint(self, record: Dict, run_meta: Dict | None = None) -> str:
        """每个任务独立落盘。

        单任务文件比频繁重写整份 manifest 更抗崩溃；进程被强制结束时，
        最多只会丢失当前尚未完成的那一个任务阶段。
        """
        payload = {
            "version": _STATE_VERSION,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "run": run_meta or {},
            "record": record,
        }
        return self._atomic_json(
            self.task_checkpoint_path(record.get("ts_code", ""), int(record.get("year", 0))),
            payload,
        )

    def load_manifest(self) -> tuple[Dict | None, str]:
        """读取主清单；主文件损坏时自动尝试备份和未完成的临时文件。"""
        for path in (self.manifest_path, self.manifest_path + ".bak",
                     self.manifest_path + ".tmp"):
            data = self._read_json(path)
            if data and isinstance(data.get("records"), list):
                return data, path
        return None, ""

    def load_task_checkpoints(self) -> tuple[List[Dict], int]:
        records: List[Dict] = []
        broken = 0
        for path in glob.glob(os.path.join(self.task_state_dir, "*.json")):
            payload = self._read_json(path)
            rec = (payload or {}).get("record")
            if not isinstance(rec, dict) or not rec.get("ts_code") or not rec.get("year"):
                broken += 1
                continue
            rec = dict(rec)
            rec["_resume_source"] = "任务检查点"
            rec["_resume_updated_at"] = (payload or {}).get("updated_at", "")
            try:
                rec["_resume_mtime"] = os.path.getmtime(path)
            except OSError:
                rec["_resume_mtime"] = 0
            records.append(rec)
        return records, broken

    def find_existing_pdf(self, ts_code: str, name: str, year: int) -> str:
        canonical = self.pdf_path(ts_code, name, year)
        candidates = [canonical]
        pattern = os.path.join(
            self.root, DIR_RAW, str(year), f"{glob.escape(ts_code)}_*_{year}_年度报告.pdf")
        candidates.extend(glob.glob(pattern))
        valid = [p for p in dict.fromkeys(candidates)
                 if os.path.isfile(p) and os.path.getsize(p) > 10240]
        return max(valid, key=os.path.getmtime) if valid else ""

    def find_existing_text(self, ts_code: str, name: str, year: int) -> str:
        canonical = self.text_path(ts_code, name, year)
        candidates = [canonical]
        pattern = os.path.join(
            self.root, DIR_TEXT, str(year), f"{glob.escape(ts_code)}_*_{year}.txt")
        candidates.extend(glob.glob(pattern))
        valid = [p for p in dict.fromkeys(candidates)
                 if os.path.isfile(p) and os.path.getsize(p) > 200]
        return max(valid, key=os.path.getmtime) if valid else ""

    def resolve_section_file(self, record: Dict, key: str) -> str:
        direct = (record.get("section_files") or {}).get(key, "")
        if direct and os.path.isfile(direct):
            return direct
        ts_code = str(record.get("ts_code", ""))
        year = int(record.get("year", 0) or 0)
        name = str(record.get("name", ""))
        d = self.section_dir(ts_code, name, year)
        stem = self.stem(ts_code, name, year)
        candidates = []
        if key == "mdna":
            candidates.append(os.path.join(d, f"{stem}_{SECTION_NAMES[key]}.pdf"))
        candidates += [
            os.path.join(d, f"{stem}_{SECTION_NAMES[key]}.txt"),
            os.path.join(d, f"{SECTION_NAMES[key]}.txt"),
        ]
        pattern_root = os.path.join(self.root, DIR_SECTION, str(year),
                                    f"{glob.escape(ts_code)}_*_{year}")
        if key == "mdna":
            candidates += glob.glob(os.path.join(pattern_root, f"*_{SECTION_NAMES[key]}.pdf"))
        candidates += glob.glob(os.path.join(pattern_root, f"*_{SECTION_NAMES[key]}.txt"))
        valid = [p for p in candidates if os.path.isfile(p)]
        return max(valid, key=os.path.getmtime) if valid else ""

    def is_reusable_record(self, record: Dict) -> bool:
        """只有已产生可读切片的 OK/PARTIAL 任务才直接跳过。

        失败、运行中和中止任务会自动重试；这样既避免重复消耗，又不会把失败状态
        永久冻结。
        """
        if record.get("status") not in ("OK", "PARTIAL"):
            return False
        secs = record.get("sections") or {}
        found = [k for k in PRIMARY_SECTION_KEYS if (secs.get(k) or {}).get("found")]
        if len(found) != len(PRIMARY_SECTION_KEYS):
            return False
        return all(bool(self.resolve_section_file(record, k)) for k in PRIMARY_SECTION_KEYS)

    @staticmethod
    def _record_time(record: Dict) -> float:
        try:
            return float(record.get("_resume_mtime") or 0)
        except (TypeError, ValueError):
            return 0.0

    def discover_section_records(self) -> List[Dict]:
        """兼容旧版本：即使没有任务检查点，也能从 sections/**/_meta.json 恢复成果。"""
        out: List[Dict] = []
        pattern = os.path.join(self.root, DIR_SECTION, "*", "*", "_meta.json")
        rank = {"skip": 0, "pass": 1, "fixed": 1, "warn": 2, "fail": 3, "error": 4}
        for meta_path in glob.glob(pattern):
            meta = self._read_json(meta_path)
            if not meta:
                continue
            ts_code = str(meta.get("ts_code") or "")
            year = int(meta.get("year") or 0)
            if not ts_code or not year:
                continue
            secs = meta.get("sections") or {}
            found = [k for k in SECTION_KEYS if (secs.get(k) or {}).get("found")]
            if not found:
                continue
            verify = meta.get("verify") or {}
            verdicts = {k: (verify.get(k) or {}).get("verdict", "skip") for k in SECTION_KEYS}
            worst = max(verdicts.values(), key=lambda x: rank.get(x, 0), default="skip")
            missing = [k for k in PRIMARY_SECTION_KEYS if not (secs.get(k) or {}).get("found")]
            loose = [k for k in PRIMARY_SECTION_KEYS if (secs.get(k) or {}).get("loose")]
            primary_bad = any((verify.get(k) or {}).get("verdict") in ("warn", "fail", "error")
                              for k in PRIMARY_SECTION_KEYS)
            status = "PARTIAL" if missing or loose or primary_bad else "OK"
            d = os.path.dirname(meta_path)
            files = {}
            fake_record = {"ts_code": ts_code, "name": str(meta.get("name") or ts_code),
                           "year": year, "section_files": {}}
            for k in found:
                p = self.resolve_section_file(fake_record, k)
                if p:
                    files[k] = p
            rd = secs.get("rd") or {}
            biz = secs.get("business") or {}
            rd_status = ("公司填报不适用" if rd.get("na") else
                         "已切出（有内容）" if rd.get("found") else
                         "小节缺失（年报中未出现）")
            business_origin = ("AI 概括生成" if biz.get("status") == SS.AI_SUMMARY else
                               "年报原文切出" if biz.get("found") else "未生成")
            problems = []
            detail_parts = []
            for k in SECTION_KEYS:
                v = verify.get(k) or {}
                if v:
                    detail_parts.append(f"{SECTION_NAMES[k]}={v.get('verdict', 'skip')}")
                    if v.get("verdict") in ("warn", "fail", "error") and v.get("reason"):
                        problems.append(f"{SECTION_NAMES[k]}：{v['reason']}")
            rec = {
                "key": f"{ts_code}|{year}",
                "ts_code": ts_code,
                "name": str(meta.get("name") or ts_code),
                "year": year,
                "status": status,
                "message": "从已有章节文件恢复",
                "title": meta.get("title", ""),
                "ann_date": meta.get("ann_date", ""),
                "source": meta.get("source", ""),
                "url": "",
                "pdf_path": meta.get("pdf", ""),
                "text_path": self.find_existing_text(ts_code, str(meta.get("name") or ""), year),
                "pages": 0,
                "engine": "已有成果",
                "sections": secs,
                "section_files": files,
                "rd_status": rd_status,
                "business_origin": business_origin,
                "verify": verify,
                "verify_worst": worst,
                "verify_detail": "；".join(detail_parts),
                "verify_problems": problems,
                "slice_notes": meta.get("slice_notes") or [],
                "resumed": True,
                "_resume_source": "章节元数据",
                "_resume_mtime": os.path.getmtime(meta_path),
            }
            if self.is_reusable_record(rec):
                out.append(rec)
        return out

    def load_resume_records(self) -> tuple[List[Dict], Dict]:
        """合并 manifest、单任务检查点和旧版章节元数据，较新的记录优先。"""
        merged: Dict[str, Dict] = {}
        info = {"manifest": 0, "checkpoints": 0, "discovered": 0, "broken": 0,
                "manifest_source": ""}

        manifest, manifest_source = self.load_manifest()
        if manifest:
            info["manifest_source"] = manifest_source
            try:
                mtime = os.path.getmtime(manifest_source)
            except OSError:
                mtime = 0
            for raw in manifest.get("records") or []:
                if not isinstance(raw, dict) or not raw.get("ts_code") or not raw.get("year"):
                    continue
                rec = dict(raw)
                rec["_resume_source"] = "manifest"
                rec["_resume_mtime"] = mtime
                merged[rec.get("key") or f"{rec['ts_code']}|{rec['year']}"] = rec
                info["manifest"] += 1

        discovered = self.discover_section_records()
        info["discovered"] = len(discovered)
        for rec in discovered:
            key = rec["key"]
            if key not in merged or self._record_time(rec) > self._record_time(merged[key]):
                merged[key] = rec

        checkpoints, broken = self.load_task_checkpoints()
        info["checkpoints"] = len(checkpoints)
        info["broken"] = broken
        for rec in checkpoints:
            key = rec.get("key") or f"{rec['ts_code']}|{rec['year']}"
            rec["key"] = key
            if key not in merged or self._record_time(rec) >= self._record_time(merged[key]):
                merged[key] = rec

        return list(merged.values()), info

    # ---- 结果导出 ----
    def save_manifest(self, records: List[Dict], meta: Dict) -> str:
        payload = {
            "version": _STATE_VERSION,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "meta": meta,
            "records": records,
        }
        return self._atomic_json(self.manifest_path, payload, backup=True)

    def export_table(self, records: List[Dict]) -> Dict[str, str]:
        rows = []
        for r in records:
            secs = r.get("sections") or {}
            ai_hit = [SECTION_NAMES[k] for k in SECTION_KEYS
                      if secs.get(k, {}).get("status") == SS.AI_LOCATED]
            row = {
                "股票代码": r.get("ts_code", ""),
                "公司简称": r.get("name", ""),
                "报告年度": r.get("year", ""),
                "状态": STATUS_LABEL.get(r.get("status"), r.get("status", "")),
                "是否异常": "是" if r.get("status") in ABNORMAL else "否",
                "研发投入情况": r.get("rd_status", ""),
                "研发投入是否不适用": "是" if secs.get("rd", {}).get("na") else "否",
                "不适用依据": secs.get("rd", {}).get("na_reason", ""),
                "业务概况来源": r.get("business_origin", ""),
                "后验结论": VERIFY_LABEL.get(r.get("verify_worst", "skip"),
                                             r.get("verify_worst", "")),
                "后验逐章节": r.get("verify_detail", ""),
                "后验问题": "；".join(r.get("verify_problems") or []),
                "AI辅助定位章节": "、".join(ai_hit),
                "公告标题": r.get("title", ""),
                "公告日期": r.get("ann_date", ""),
                "来源": r.get("source", ""),
                "PDF页数": r.get("pages", ""),
                "解析引擎": r.get("engine", ""),
                "PDF路径": r.get("pdf_path", ""),
                "管理层讨论与分析·结构化切片": (r.get("section_files") or {}).get("mdna", ""),
                "切片提示": "；".join(r.get("slice_notes") or []),
                "备注": r.get("message", ""),
            }
            for k in SECTION_KEYS:
                sec = secs.get(k) or {}
                row[f"{SECTION_NAMES[k]}·状态"] = SS_LABEL.get(
                    sec.get("status", SS.MISSING), sec.get("status", ""))
                row[f"{SECTION_NAMES[k]}·字数"] = sec.get("chars", 0)
                row[f"{SECTION_NAMES[k]}·识别方式"] = sec.get("how", "未识别")
                row[f"{SECTION_NAMES[k]}·后验"] = VERIFY_LABEL.get(
                    (r.get("verify") or {}).get(k, {}).get("verdict", "skip"), "")
            rows.append(row)

        out = {}
        csv_path = os.path.join(self.root, REPORT_CSV)
        try:
            import csv as _csv
            with open(csv_path, "w", encoding=TEXT_ENCODING, newline="") as f:
                if rows:
                    w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            out["csv"] = csv_path
        except Exception:                                  # noqa: BLE001
            pass

        try:
            import pandas as pd
            xlsx = os.path.join(self.root, REPORT_XLSX)
            df = pd.DataFrame(rows)
            empty = df.iloc[0:0]
            with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name="全部结果")
                bad = df[df["是否异常"] == "是"] if "是否异常" in df else empty
                bad.to_excel(w, index=False, sheet_name="异常清单")
                # 研发投入需要单独汇报：不适用 + 小节缺失各成一张表
                na = df[df["研发投入是否不适用"] == "是"] if "研发投入是否不适用" in df else empty
                na.to_excel(w, index=False, sheet_name="研发投入-不适用")
                miss = (df[df["研发投入情况"].astype(str).str.startswith("小节缺失")]
                        if "研发投入情况" in df else empty)
                miss.to_excel(w, index=False, sheet_name="研发投入-缺失")
                aibiz = (df[df["业务概况来源"] == "AI 概括生成"]
                         if "业务概况来源" in df else empty)
                aibiz.to_excel(w, index=False, sheet_name="业务概况-AI概括")
                vb = (df[df["后验结论"].isin(["存疑", "不通过", "后验失败"])]
                      if "后验结论" in df else empty)
                vb.to_excel(w, index=False, sheet_name="后验存疑")
            out["xlsx"] = xlsx
        except Exception:                                  # noqa: BLE001
            pass
        return out

    def save_sections(self, ts_code, name, year, sections: Dict,
                      verdicts: Dict | None = None, header: Dict | None = None) -> Dict:
        """写出章节文本，并把 MD&A 从原 PDF 裁成结构化 PDF。

        结构化 PDF 是主产物；同名 TXT 作为检索/AI/审计用的辅助副本保留。
        所有文件名都带 ``股票代码_公司简称_年度_`` 前缀，即使已经位于公司文件夹内。
        """
        header = header or {}
        verdicts = verdicts or {}
        d = self.section_dir(ts_code, name, year)
        os.makedirs(d, exist_ok=True)
        paths: Dict[str, str] = {}
        structural: Dict[str, Dict] = {}
        base = self.stem(ts_code, name, year)
        for key in SECTION_KEYS:
            sec = sections.get(key) or {}
            if not sec.get("found"):
                continue
            body = sec.get("text") or ""
            status = sec.get("status", SS.ORIGINAL)
            if status == SS.NOT_APPLICABLE and not body.strip():
                body = "（公司在本小节填报「不适用」，年报中无实质内容）"
            if not body.strip():
                continue

            v = verdicts.get(key) or {}
            lines = [
                f"# {header.get('name', '')}（{header.get('ts_code', '')}）"
                f" {year} 年年度报告 — {SECTION_NAMES[key]}",
                f"# 内容性质：{SS_LABEL.get(status, status)}",
            ]
            if status == SS.AI_SUMMARY:
                lines += [
                    "# ⚠ 本文件为 AI 依据「管理层讨论与分析」概括生成，"
                    "**不是年报原文**，不可直接作为原文引用；",
                    f"# ⚠ 生成模型：{sec.get('summary_model', '')}；引用前请回原文核对。",
                ]
            if status == SS.NOT_APPLICABLE:
                lines.append(f"# 不适用依据：{sec.get('na_reason', '')}")
            lines += [
                f"# 识别方式：{sec.get('how', '')}",
                f"# 所属上级条目：{sec.get('parent_title', '') or '—'}",
                f"# 后验结论：{VERIFY_LABEL.get(v.get('verdict', 'skip'), '未后验')}"
                + (f"｜{v.get('reason', '')}" if v.get("reason") else ""),
                f"# 公告标题：{header.get('title', '')}",
                f"# 字数：{sec.get('chars', 0)}",
                "-" * 60, "",
            ]
            txt_path = os.path.join(d, f"{base}_{SECTION_NAMES[key]}.txt")
            write_text(txt_path, "\n".join(lines) + "\n" + body)
            paths[key] = txt_path

            if key == "mdna" and header.get("pdf_path"):
                pdf_path = os.path.join(d, f"{base}_{SECTION_NAMES[key]}.pdf")
                info = save_pdf_section(header.get("pdf_path", ""), pdf_path, sec)
                structural[key] = info
                sec["structure_mode"] = info.get("mode", "")
                sec["structure_message"] = info.get("message", "")
                if info.get("ok"):
                    paths["mdna_text"] = txt_path
                    paths["mdna"] = pdf_path
                    sec["structure_file"] = pdf_path
                else:
                    sec["structure_file"] = ""

        meta = {
            "ts_code": ts_code, "name": name, "year": year,
            "title": header.get("title", ""), "ann_date": header.get("ann_date", ""),
            "source": header.get("source", ""), "pdf": header.get("pdf_path", ""),
            "slice_notes": header.get("notes", []),
            "files": paths,
            "structural": structural,
            "sections": {k: {kk: vv for kk, vv in (sections.get(k) or {}).items()
                             if kk not in ("text", "_source_excerpt")}
                         for k in SECTION_KEYS},
            "verify": {k: (verdicts.get(k) or {}) for k in SECTION_KEYS},
        }
        self._atomic_json(os.path.join(d, "_meta.json"), meta)
        return paths

