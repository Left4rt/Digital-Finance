# -*- coding: utf-8 -*-
"""采集流水线。

一个任务 = (公司, 年度)。状态机：

    检索公告 → 下载 PDF → 抽取文本 → 规则切片 → AI 补定位 → 业务概况 AI 概括
             → AI 后验（质检 / 保守修正）→ 落盘

任何一步失败都只影响该任务，并写入明确的状态码，绝不中断整轮运行。
AI 相关的三步全部是"可选增强"：没开、没 Key、连不上，都只降级不报错。
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .ai_assist import DeepSeekClient, enhance_with_ai, summarize_business
from .config import (ABNORMAL, AI_VERIFY_AUTOFIX, DEEPSEEK_ADVANCED_MODEL,
                     DEEPSEEK_MODEL, DEFAULT_OUTPUT, PRIMARY_SECTION_KEYS, S, SECTION_KEYS,
                     SECTION_NAMES, SS, SS_LABEL, STATUS_LABEL)
from .csv_loader import load_company_list
from .pdf_text import extract_text
from .slicer import slice_report
from .sources import (AnnouncementFinder, CninfoClient, TushareClient,
                      download, _session)
from .store import Store, write_text
from .verify import summarize_verdicts, verify_sections


class RunState:
    """线程安全的运行状态，供 Web 层轮询。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.running = False
        self.stop_flag = False
        self.run_id = ""
        self.started_at = ""
        self.finished_at = ""
        self.total = 0
        self.done = 0
        self.records: List[Dict] = []
        self.logs: List[Dict] = []
        self.output = ""
        self.exports: Dict[str, str] = {}
        self.error = ""
        self.resumed = 0
        self.resume_found = 0
        self.connection = {"ok": None, "msg": "未检测"}
        self.ai_connection = {"ok": None, "msg": "未启用"}

    def log(self, msg: str, level: str = "info"):
        with self.lock:
            self.logs.append({
                "t": dt.datetime.now().strftime("%H:%M:%S"),
                "level": level, "msg": str(msg)[:500],
            })
            if len(self.logs) > 4000:
                del self.logs[:1500]

    def snapshot(self, log_offset: int = 0) -> Dict:
        with self.lock:
            counts = {}
            for r in self.records:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            abnormal = sum(v for k, v in counts.items() if k in ABNORMAL)
            rd_na = sum(1 for r in self.records
                        if (r.get("sections") or {}).get("rd", {}).get("na"))
            verify_bad = sum(1 for r in self.records
                             if r.get("verify_worst") in ("fail", "warn", "error"))
            return {
                "running": self.running,
                "run_id": self.run_id,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "total": self.total,
                "done": self.done,
                "abnormal": abnormal,
                "rd_na": rd_na,
                "verify_bad": verify_bad,
                "counts": {STATUS_LABEL.get(k, k): v for k, v in counts.items()},
                "records": self.records,
                "logs": self.logs[log_offset:],
                "log_total": len(self.logs),
                "output": self.output,
                "exports": self.exports,
                "error": self.error,
                "resumed": self.resumed,
                "resume_found": self.resume_found,
                "connection": self.connection,
                "ai_connection": self.ai_connection,
            }


STATE = RunState()


# --------------------------------------------------------------------------
def _record(ts_code, name, year) -> Dict:
    return {
        "key": f"{ts_code}|{year}", "ts_code": ts_code, "name": name,
        "year": year, "status": S.PENDING, "message": "",
        "title": "", "ann_date": "", "source": "", "url": "",
        "pdf_path": "", "text_path": "", "pages": 0, "engine": "",
        "sections": {k: {"found": False, "how": "未识别", "chars": 0,
                         "status": SS.MISSING} for k in SECTION_KEYS},
        "section_files": {},
        "rd_status": "", "business_origin": "",
        "verify": {}, "verify_worst": "skip", "verify_detail": "",
        "verify_problems": [], "slice_notes": [],
        "resumed": False, "resume_source": "", "resume_skipped": False,
        "previous_status": "", "task_started_at": "", "task_finished_at": "",
    }


def _rd_status_label(sec: Dict) -> str:
    if sec.get("na"):
        return "公司填报不适用"
    if sec.get("found"):
        return "已切出（有内容）" + ("｜待复核" if sec.get("loose") else "")
    return "小节缺失（年报中未出现）"


def _business_origin_label(sec: Dict) -> str:
    st = sec.get("status")
    if st == SS.AI_SUMMARY:
        return "AI 概括生成"
    if st in (SS.ORIGINAL, SS.AI_LOCATED) and sec.get("found"):
        return "年报原文切出"
    return "未生成"


def process_one(rec: Dict, finder: AnnouncementFinder, store: Store,
                sess, state: RunState, overwrite: bool, save_fulltext: bool,
                ai_client: "DeepSeekClient | None" = None,
                ai_summary: bool = True, ai_verify: bool = True,
                reuse_existing: bool = True):
    ts_code, name, year = rec["ts_code"], rec["name"], rec["year"]
    tag = f"{ts_code} {name} {year}"

    def slog(msg, level="info"):
        state.log(f"[{tag}] {msg}", level)

    try:
        # ---------- 1. 检索年报公告 ----------
        canonical_pdf = store.pdf_path(ts_code, name, year)
        existing_pdf = (store.find_existing_pdf(ts_code, name, year)
                        if reuse_existing and not overwrite else "")
        pdf_path = existing_pdf or canonical_pdf
        need_download = overwrite or not existing_pdf
        if need_download:
            pdf_path = canonical_pdf
            hit = finder.find(ts_code, year)
            if not hit:
                rec["status"] = S.NO_ANN
                rec["message"] = f"{year} 年度年报公告未检索到（可能尚未披露或已退市）"
                slog("未找到年报", "warn")
                return
            rec.update({"title": hit["title"], "ann_date": hit["ann_date"],
                        "source": hit["source"], "url": hit["url"]})
            slog(f"命中公告：{hit['title']}（{hit['source']}）")

            # ---------- 2. 下载 ----------
            dl = download(hit["url"], pdf_path, sess)
            if not dl["ok"]:
                rec["status"] = S.DOWNLOAD_FAIL
                rec["message"] = dl["msg"]
                slog(f"下载失败：{dl['msg']}", "error")
                return
            slog(f"下载完成 {dl['size'] // 1024} KB")
        else:
            rec["message"] = "断点续跑：复用本地已有 PDF"
            slog(f"断点续跑：复用本地 PDF（{os.path.basename(pdf_path)}）")
        rec["pdf_path"] = pdf_path

        # ---------- 3. 抽取文本 / 复用已保存全文 ----------
        ext = None
        cached_text = (store.find_existing_text(ts_code, name, year)
                       if reuse_existing and not overwrite else "")
        if cached_text:
            try:
                with open(cached_text, "r", encoding="utf-8-sig", errors="replace") as f:
                    cached_body = f.read()
                if len(cached_body.strip()) > 200:
                    ext = {
                        "ok": True,
                        "text": cached_body,
                        "pages": int(rec.get("pages") or 0),
                        "engine": "cached-fulltext",
                        "msg": "复用已保存全文",
                    }
                    rec["text_path"] = cached_text
                    slog(f"断点续跑：复用已保存全文 {len(cached_body):,} 字")
            except OSError as e:
                slog(f"读取已有全文失败，将重新解析 PDF：{e}", "warn")

        if ext is None:
            ext = extract_text(pdf_path)
            rec["pages"] = ext["pages"]
            rec["engine"] = ext["engine"]
            if not ext["ok"]:
                rec["status"] = S.NO_TEXT if ext["pages"] else S.PDF_BROKEN
                rec["message"] = ext["msg"]
                slog(ext["msg"], "error")
                return

            if save_fulltext:
                tp = store.text_path(ts_code, name, year)
                write_text(tp, ext["text"])
                rec["text_path"] = tp
        else:
            rec["pages"] = ext["pages"]
            rec["engine"] = ext["engine"]

        # ---------- 4. 规则切片（确定性，标题行锚定） ----------
        sliced = slice_report(ext["text"])
        text = sliced["normalized"]
        sections = sliced["sections"]
        rec["slice_notes"] = list(sliced.get("notes") or [])
        for n in rec["slice_notes"]:
            slog("切片提示：" + n)
        mdna = sections.get("mdna") or {}
        if mdna.get("found"):
            slog(f"管理层讨论与分析整节切出 {mdna['chars']} 字｜{mdna['how']}")

        ai_on = ai_client is not None and ai_client.enabled

        # ---------- 4.5 AI 补定位（只补规则没切到 / 只算疑似的章节） ----------
        ai_stats = {"calls": 0, "resolved": [], "tried": False}
        if ai_on:
            try:
                sections, ai_stats = enhance_with_ai(
                    text, {**sliced, "sections": sections}, ai_client,
                    log=lambda m, lv="info": slog(m, lv))
            except Exception as e:                        # noqa: BLE001
                slog(f"AI 辅助定位异常（已忽略，按规则结果继续）：{e}", "warn")

        # ---------- 4.6 业务概况：年报没有这一板块时，由高级模型概括 ----------
        biz = sections.get("business") or {}
        if ai_on and ai_summary and not biz.get("found"):
            mdna_text = (sections.get("mdna") or {}).get("text") or ""
            if mdna_text:
                try:
                    core_text = (sections.get("core") or {}).get("text") or ""
                    res = summarize_business(
                        mdna_text, ai_client, company=name, year=str(year),
                        extra_text=core_text[:6000],
                        log=lambda m, lv="info": slog(m, lv))
                except Exception as e:                    # noqa: BLE001
                    res = None
                    slog(f"业务概况概括异常（已忽略）：{e}", "warn")
                if res:
                    sections["business"] = {
                        **biz, "found": True, "status": SS.AI_SUMMARY,
                        "ai": True, "loose": False,
                        "how": f"AI 概括生成（{res['model']}，依据管理层讨论与分析"
                               + ("，原文过长已头尾取样" if res["truncated"] else "") + "）",
                        "chars": len(res["text"]), "text": res["text"],
                        "start": None, "end": None, "origin": "AI概括",
                        "summary_model": res["model"],
                        "_source_excerpt": mdna_text[:16000],
                    }
                    slog(f"业务概况由 {res['model']} 概括生成 {len(res['text'])} 字"
                         f"（非年报原文，已标注）", "ok")
            else:
                slog("管理层讨论与分析未切出，无法概括业务概况", "warn")
        elif not biz.get("found") and not ai_on:
            slog("业务概况未在年报中找到独立板块；未启用 AI，无法概括", "warn")

        # ---------- 4.7 AI 后验：检验切片是否正确、准确、完整 ----------
        verdicts: Dict[str, Dict] = {}
        if ai_on and ai_verify:
            try:
                sections, verdicts = verify_sections(
                    text, sections, ai_client, company=name, year=str(year),
                    log=lambda m, lv="info": slog("后验：" + m, lv),
                    autofix=AI_VERIFY_AUTOFIX)
            except Exception as e:                        # noqa: BLE001
                slog(f"AI 后验异常（已忽略，保留规则切片）：{e}", "warn")
        worst, problems, detail = summarize_verdicts(verdicts) if verdicts \
            else ("skip", [], "未后验")
        rec["verify"] = {k: {kk: vv for kk, vv in (verdicts.get(k) or {}).items()}
                         for k in SECTION_KEYS}
        rec["verify_worst"] = worst
        rec["verify_problems"] = problems
        rec["verify_detail"] = detail
        if problems:
            slog("后验存疑：" + "；".join(problems[:3]), "warn")
        elif verdicts:
            slog("后验通过：" + detail, "ok")

        # ---------- 5. 汇总与落盘 ----------
        for sec in sections.values():
            sec.pop("_source_excerpt", None)

        rec["chapters"] = sliced["chapters"]
        rec["ai_stats"] = ai_stats
        rec["rd_status"] = _rd_status_label(sections.get("rd") or {})
        rec["business_origin"] = _business_origin_label(sections.get("business") or {})

        # 研发投入单独汇报（不适用 / 缺失都要留痕，不能混进"未识别"一笔带过）
        rd = sections.get("rd") or {}
        if rd.get("na"):
            slog(f"研发投入：公司填报「不适用」（{rd.get('na_reason', '')}），已单独记录", "warn")
        elif not rd.get("found"):
            slog("研发投入：年报中未出现该小节，已单独记录", "warn")

        primary_found = [k for k in PRIMARY_SECTION_KEYS if sections[k].get("found")]
        if len(primary_found) != len(PRIMARY_SECTION_KEYS):
            rec["status"] = S.NO_SECTION
            rec["message"] = (f"全文 {ext['pages']} 页可读，但未能可靠定位管理层讨论与分析；"
                              f"已识别 {sliced.get('heading_count', 0)} 个标题候选")
            slog("管理层讨论与分析未识别", "error")
            return

        rec["section_files"] = store.save_sections(
            ts_code, name, year, sections, verdicts,
            {"name": name, "ts_code": ts_code, "title": rec["title"],
             "ann_date": rec["ann_date"], "source": rec["source"],
             "pdf_path": pdf_path, "notes": rec["slice_notes"]})
        rec["sections"] = {k: {kk: vv for kk, vv in sections[k].items() if kk != "text"}
                           for k in SECTION_KEYS}
        mdna_file = rec["section_files"].get("mdna", "")
        if mdna_file.lower().endswith(".pdf"):
            slog(f"结构化 MD&A 已保存：{os.path.basename(mdna_file)}", "ok")
        else:
            slog("结构化 PDF 生成失败，已保留文本副本并标记复核", "warn")

        loose = [SECTION_NAMES[k] for k in PRIMARY_SECTION_KEYS if sections[k].get("loose")]
        primary_verify_bad = [SECTION_NAMES[k] for k in PRIMARY_SECTION_KEYS
                              if (verdicts.get(k) or {}).get("verdict") in ("warn", "fail", "error")]
        parts = []
        if loose:
            parts.append("边界使用兜底规则，建议抽查")
        if primary_verify_bad:
            parts.append("后验存疑：" + "、".join(primary_verify_bad))
        if not mdna_file.lower().endswith(".pdf"):
            parts.append("结构化 PDF 未生成")

        if parts:
            rec["status"] = S.PARTIAL
            rec["message"] = "；".join(parts)
            slog("MD&A 已切出，但 " + rec["message"], "warn")
        else:
            rec["status"] = S.OK
            rec["message"] = "管理层讨论与分析已完整切出（结构化 PDF）"
            slog("管理层讨论与分析结构化切片就绪", "ok")

    except Exception as e:                                  # noqa: BLE001
        rec["status"] = S.ERROR
        rec["message"] = f"{type(e).__name__}: {e}"
        state.log(f"[{tag}] 未预期异常：{e}\n{traceback.format_exc(limit=3)}", "error")


# --------------------------------------------------------------------------
def _restore_task(fresh: Dict, previous: Dict | None, store: Store) -> tuple[Dict, bool]:
    """把历史记录合并进本轮任务；返回（任务，是否直接复用）。"""
    if not previous:
        return fresh, False

    source = previous.get("_resume_source") or previous.get("resume_source") or "历史记录"
    if store.is_reusable_record(previous):
        restored = {**fresh, **previous}
        restored["key"] = fresh["key"]
        restored["ts_code"] = fresh["ts_code"]
        restored["year"] = fresh["year"]
        # CSV 中有明确名称时优先采用；文件位置仍由 section_files 保存。
        if fresh.get("name") and fresh["name"] != fresh["ts_code"]:
            restored["name"] = fresh["name"]
        restored["resumed"] = True
        restored["resume_source"] = source
        restored["resume_skipped"] = True
        restored["previous_status"] = previous.get("status", "")
        old_msg = str(previous.get("message") or "").strip()
        marker = f"断点续跑：复用已有成果（{source}）"
        restored["message"] = f"{old_msg}；{marker}" if old_msg else marker
        return restored, True

    # 未完成/失败任务要重试，但保留可复用的下载和解析信息。
    for field in ("title", "ann_date", "source", "url", "pdf_path", "text_path",
                  "pages", "engine"):
        if previous.get(field):
            fresh[field] = previous[field]
    fresh["resume_source"] = source
    fresh["previous_status"] = previous.get("status", "")
    return fresh, False


def _manifest_meta(cfg: Dict, state: RunState, years: List[int], status: str) -> Dict:
    return {
        "run_id": state.run_id,
        "status": status,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "csv": cfg.get("csv_path", ""),
        "years": years,
        "output": state.output,
        "resume": bool(cfg.get("resume", True)),
        "resumed_tasks": state.resumed,
        "total": state.total,
        "done": state.done,
        "error": state.error,
    }


def _save_final_outputs(store: Store, cfg: Dict, state: RunState,
                        years: List[int], status: str) -> None:
    if status != "running" and not state.finished_at:
        state.finished_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state.exports = store.export_table(state.records)
    store.save_manifest(state.records, _manifest_meta(cfg, state, years, status))


def run_job(cfg: Dict, state: RunState = STATE):
    """运行批处理。

    新增断点能力：
    - 启动时读取 manifest、每任务检查点和旧版 sections/**/_meta.json；
    - OK/PARTIAL 且切片文件完整的任务直接复用；
    - RUNNING/STOPPED/失败任务自动继续，优先复用 PDF 与全文；
    - 每个任务开始和结束时都独立原子落盘。
    """
    with state.lock:
        if state.running:
            return
        state.running = True
        state.stop_flag = False
        state.run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        state.started_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state.finished_at = ""
        state.records = []
        state.logs = []
        state.done = 0
        state.total = 0
        state.error = ""
        state.exports = {}
        state.resumed = 0
        state.resume_found = 0

    store: Store | None = None
    years: List[int] = []
    finalized = False

    try:
        output = cfg.get("output") or DEFAULT_OUTPUT
        store = Store(output)
        state.output = store.root
        state.log(f"输出目录：{store.root}")

        loaded = load_company_list(cfg["csv_path"])
        for w in loaded["warnings"]:
            state.log("清单：" + w, "warn")
        state.log(f"清单编码识别为 {loaded['encoding']}，有效公司 {len(loaded['rows'])} 家")
        if not loaded["rows"]:
            state.error = "公司清单为空或无法解析"
            return

        years = sorted({int(y) for y in cfg.get("years", [])}, reverse=True)
        priority = cfg.get("priority_year")
        if priority and int(priority) in years:
            years.remove(int(priority))
            years.insert(0, int(priority))

        retry_keys = set(cfg.get("retry_keys") or [])
        overwrite = bool(cfg.get("overwrite"))
        resume_enabled = bool(cfg.get("resume", True)) and not overwrite and not retry_keys

        previous_map: Dict[str, Dict] = {}
        if resume_enabled:
            history, info = store.load_resume_records()
            previous_map = {
                (r.get("key") or f"{r.get('ts_code')}|{r.get('year')}"): r
                for r in history
            }
            state.resume_found = len(previous_map)
            state.log(
                "已读取历史成果："
                f"manifest {info['manifest']} 条，任务检查点 {info['checkpoints']} 条，"
                f"章节目录恢复 {info['discovered']} 条"
                + (f"，损坏检查点 {info['broken']} 条已忽略" if info["broken"] else "")
            )
        elif overwrite:
            state.log("已启用强制重新下载：本轮忽略历史完成状态", "warn")
        elif retry_keys:
            state.log("异常重跑模式：指定任务将强制重新执行，不按完成状态跳过", "warn")
        else:
            state.log("断点续跑已关闭：本轮重新执行所选任务", "warn")

        tasks: List[Dict] = []
        reused = 0
        for row in loaded["rows"]:
            base_name = row["name"] or row["ts_code"]
            for y in years:
                key = f'{row["ts_code"]}|{y}'
                if retry_keys and key not in retry_keys:
                    continue
                fresh = _record(row["ts_code"], base_name, y)
                task, skipped = _restore_task(fresh, previous_map.get(key), store)
                tasks.append(task)
                reused += int(skipped)

        with state.lock:
            state.records = tasks
            state.total = len(tasks)
            state.done = reused
            state.resumed = reused

        if retry_keys:
            state.log(f"重跑模式：仅处理上一轮的 {len(tasks)} 个异常任务", "warn")
        state.log(f"共 {len(loaded['rows'])} 家公司 × {len(years)} 个年度 = {len(tasks)} 个任务")
        if reused:
            state.log(f"断点续跑：{reused} 个已有成果通过完整性检查，将直接跳过", "ok")

        pending = [r for r in tasks if not r.get("resume_skipped")]
        store.save_run_state(_manifest_meta(cfg, state, years, "running"))

        if not pending:
            state.log("所选任务均已有完整成果，无需联网或重复处理", "ok")
            _save_final_outputs(store, cfg, state, years, "finished")
            finalized = True
            return

        # 只有存在待执行任务时才检测外部服务。
        ts = TushareClient(cfg.get("token") or None) if cfg.get("token") else TushareClient()
        ping = ts.ping()
        state.connection = {"ok": ping["ok"], "msg": ping["msg"]}
        state.log(f"Tushare 网关：{ping['msg']}"
                  + (f"（在册股票 {ping['count']} 只）" if ping["ok"] else ""),
                  "ok" if ping["ok"] else "warn")

        name_map = {}
        if ping["ok"]:
            try:
                name_map = ts.stock_name_map()
            except Exception:                               # noqa: BLE001
                name_map = {}
        for rec in pending:
            if (not rec.get("name") or rec["name"] == rec["ts_code"]) and \
                    name_map.get(rec["ts_code"]):
                rec["name"] = name_map[rec["ts_code"]]

        cn = CninfoClient()
        finder = AnnouncementFinder(ts, cn, cfg.get("prefer", "auto"),
                                    log=lambda m: state.log(m, "warn"))

        ai_client = None
        ai_summary = bool(cfg.get("ai_summary", True))
        ai_verify = bool(cfg.get("ai_verify", True))
        if cfg.get("ai_enabled"):
            ai_client = DeepSeekClient(
                cfg.get("deepseek_key") or "",
                model=cfg.get("deepseek_model") or DEEPSEEK_MODEL,
                advanced_model=cfg.get("deepseek_advanced_model")
                or DEEPSEEK_ADVANCED_MODEL)
            if ai_client.enabled:
                ai_ping = ai_client.ping()
                state.ai_connection = {"ok": ai_ping["ok"], "msg": ai_ping["msg"]}
                state.log(f"DeepSeek：{ai_ping['msg']}", "ok" if ai_ping["ok"] else "warn")
                if ai_ping["ok"]:
                    adv = ai_client.ping(model=ai_client.advanced_model)
                    state.log(f"高级模型（概括/后验）：{adv['msg']}",
                              "ok" if adv["ok"] else "warn")
                    if not adv["ok"]:
                        state.log("高级模型不可用，业务概况概括与后验将使用可用的降级模型",
                                  "warn")
                    state.log(f"业务概况 AI 概括：{'开启' if ai_summary else '关闭'}；"
                              f"切片 AI 后验：{'开启' if ai_verify else '关闭'}")
                else:
                    state.log("DeepSeek 连接失败，本轮只用规则切片，不影响其余流程", "warn")
            else:
                state.log("未填写 DeepSeek API Key，跳过全部 AI 环节", "warn")
                ai_client = None
        else:
            state.ai_connection = {"ok": None, "msg": "未启用"}
            state.log("未启用 AI：业务概况将无法概括生成，切片也不会做后验", "warn")

        sess = _session()
        workers = max(1, min(int(cfg.get("workers", 4)), 12))
        save_fulltext = cfg.get("save_fulltext", True)

        for y in years:
            batch = [t for t in pending if t["year"] == y]
            if not batch:
                continue
            state.log(f"—— 开始处理 {y} 年度（待执行 {len(batch)} 个任务）——")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = []
                for rec in batch:
                    if state.stop_flag:
                        break
                    futures.append(pool.submit(
                        _wrap, rec, finder, store, sess, state, overwrite,
                        save_fulltext, ai_client, ai_summary, ai_verify, not overwrite))
                for f in futures:
                    try:
                        f.result()
                    except Exception as e:                  # noqa: BLE001
                        state.log(f"任务线程异常：{e}", "error")
            if state.stop_flag:
                state.log("已收到停止指令，剩余任务标记为已中止", "warn")
                for t in tasks:
                    if t["status"] in (S.PENDING, S.RUNNING):
                        t["status"] = S.STOPPED
                        t["message"] = "用户中止；下次启动会从此任务继续"
                        t["task_finished_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        try:
                            store.save_task_checkpoint(t, {"run_id": state.run_id})
                        except Exception as e:              # noqa: BLE001
                            state.log(f"保存中止检查点失败：{e}", "warn")
                break

        with state.lock:
            state.done = sum(1 for r in state.records
                             if r.get("status") not in (S.PENDING, S.RUNNING))

        _save_final_outputs(store, cfg, state, years,
                            "stopped" if state.stop_flag else "finished")
        finalized = True

        ok = sum(1 for r in state.records if r["status"] == S.OK)
        bad = sum(1 for r in state.records if r["status"] in ABNORMAL)
        rd_na = [f'{r["ts_code"]} {r["name"]} {r["year"]}' for r in state.records
                 if (r.get("sections") or {}).get("rd", {}).get("na")]
        rd_missing = [f'{r["ts_code"]} {r["name"]} {r["year"]}' for r in state.records
                      if r.get("rd_status", "").startswith("小节缺失")]
        biz_ai = sum(1 for r in state.records if r.get("business_origin") == "AI 概括生成")
        vbad = [r for r in state.records
                if r.get("verify_worst") in ("fail", "warn", "error")]

        state.log(f"全部结束：完成 {ok}，需复核 {bad}，断点复用 {state.resumed}", "ok")
        state.log(f"研发投入汇报：填报「不适用」{len(rd_na)} 篇，小节缺失 {len(rd_missing)} 篇"
                  + ("｜不适用清单：" + "，".join(rd_na[:20]) if rd_na else ""),
                  "warn" if (rd_na or rd_missing) else "ok")
        state.log(f"业务概况：{biz_ai} 篇由 AI 概括生成（非年报原文，请勿直接当原文引用）")
        state.log(f"切片后验：{len(vbad)} 篇存在存疑或失败结论，详见结果表「后验」列",
                  "warn" if vbad else "ok")
        state.log(f"结果表已导出到 {store.root}", "ok")
        if ai_client is not None and ai_client.model_notes:
            for n in ai_client.model_notes:
                state.log("模型降级：" + n, "warn")

    except Exception as e:                                  # noqa: BLE001
        state.error = f"{type(e).__name__}: {e}"
        state.log("运行失败：" + state.error, "error")
        state.log(traceback.format_exc(limit=5), "error")
    finally:
        with state.lock:
            state.running = False
            state.finished_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state.done = sum(1 for r in state.records
                             if r.get("status") not in (S.PENDING, S.RUNNING))

        if store is not None:
            if not finalized:
                try:
                    _save_final_outputs(store, cfg, state, years,
                                        "failed" if state.error else
                                        ("stopped" if state.stop_flag else "finished"))
                except Exception as e:                      # noqa: BLE001
                    state.log(f"最终进度保存失败：{e}", "error")
            try:
                store.save_run_state(_manifest_meta(
                    cfg, state, years,
                    "failed" if state.error else
                    ("stopped" if state.stop_flag else "finished")))
            except Exception as e:                          # noqa: BLE001
                state.log(f"运行状态保存失败：{e}", "warn")
            try:
                with open(store.log_path(state.run_id), "w",
                          encoding="utf-8-sig", newline="\n") as f:
                    for line in state.logs:
                        f.write(f"[{line['t']}][{line['level']}] {line['msg']}\n")
            except Exception:                              # noqa: BLE001
                pass


def _wrap(rec, finder, store, sess, state, overwrite, save_fulltext,
          ai_client=None, ai_summary=True, ai_verify=True, reuse_existing=True):
    if state.stop_flag:
        rec["status"] = S.STOPPED
        rec["message"] = "用户中止；下次启动会继续"
        rec["task_finished_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            store.save_task_checkpoint(rec, {"run_id": state.run_id})
        finally:
            with state.lock:
                state.done += 1
        return

    rec["status"] = S.RUNNING
    rec["task_started_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        store.save_task_checkpoint(rec, {"run_id": state.run_id})
    except Exception as e:                                  # noqa: BLE001
        state.log(f"[{rec['key']}] 保存启动检查点失败：{e}", "warn")

    try:
        process_one(rec, finder, store, sess, state, overwrite, save_fulltext,
                    ai_client, ai_summary, ai_verify, reuse_existing)
    finally:
        rec["task_finished_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            store.save_task_checkpoint(rec, {"run_id": state.run_id})
        except Exception as e:                              # noqa: BLE001
            state.log(f"[{rec['key']}] 保存完成检查点失败：{e}", "error")
        with state.lock:
            state.done += 1
