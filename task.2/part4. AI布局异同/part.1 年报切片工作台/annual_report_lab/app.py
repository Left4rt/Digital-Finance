# -*- coding: utf-8 -*-
"""年报采集与章节切片工作台 —— Web 服务入口。

启动：
    python app.py                # 默认 http://127.0.0.1:8777
    python app.py --port 9000 --no-browser
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import subprocess
import sys
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ai_assist import DeepSeekClient
from core.config import (ABNORMAL, AI_VERIFY_ENABLED_DEFAULT,
                         DEEPSEEK_ADVANCED_MODEL, DEEPSEEK_API_KEY,
                         DEEPSEEK_MODEL, DEFAULT_OUTPUT, SECTION_KEYS,
                         SECTION_NAMES, STATUS_LABEL, TUSHARE_TOKEN)
from core.csv_loader import load_company_list
from core.pipeline import STATE, run_job
from core.sources import TushareClient
from core.store import Store

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
try:
    app.json.ensure_ascii = False          # Flask >= 2.3
except Exception:                          # noqa: BLE001
    app.config["JSON_AS_ASCII"] = False

_worker: threading.Thread | None = None


@app.after_request
def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_output=DEFAULT_OUTPUT,
        default_token=TUSHARE_TOKEN,
        default_deepseek_key=DEEPSEEK_API_KEY,
        default_deepseek_model=DEEPSEEK_MODEL,
        default_deepseek_advanced_model=DEEPSEEK_ADVANCED_MODEL,
        default_ai_verify=AI_VERIFY_ENABLED_DEFAULT,
        section_keys=SECTION_KEYS,
        section_names=SECTION_NAMES,
        status_label=STATUS_LABEL,
    )


@app.post("/api/validate")
def api_validate():
    path = (request.json or {}).get("csv_path", "").strip().strip('"')
    res = load_company_list(path)
    return jsonify({
        "ok": bool(res["rows"]),
        "encoding": res["encoding"],
        "count": len(res["rows"]),
        "warnings": res["warnings"][:20],
        "preview": res["rows"][:12],
    })


@app.post("/api/ping")
def api_ping():
    token = (request.json or {}).get("token") or TUSHARE_TOKEN
    res = TushareClient(token).ping()
    STATE.connection = {"ok": res["ok"], "msg": res["msg"]}
    return jsonify(res)


@app.post("/api/ping_ai")
def api_ping_ai():
    """两个模型都测：定位用的常规模型 + 概括/后验用的高级模型。"""
    cfg = request.json or {}
    key = (cfg.get("deepseek_key") or "").strip()
    model = (cfg.get("deepseek_model") or "").strip() or DEEPSEEK_MODEL
    adv = (cfg.get("deepseek_advanced_model") or "").strip() or DEEPSEEK_ADVANCED_MODEL
    client = DeepSeekClient(key, model=model, advanced_model=adv)
    base = client.ping()
    if not base["ok"]:
        STATE.ai_connection = {"ok": False, "msg": base["msg"]}
        return jsonify(base)
    hi = client.ping(model=adv)
    msg = f"定位模型 {model} 可用；高级模型 {adv} " + ("可用" if hi["ok"] else "不可用")
    if not hi["ok"]:
        msg += f"（{hi['msg']}），概括与后验将自动降级到可用模型"
    res = {"ok": True, "msg": msg, "advanced_ok": hi["ok"]}
    STATE.ai_connection = {"ok": True, "msg": msg}
    return jsonify(res)


@app.post("/api/start")
def api_start():
    global _worker
    if STATE.running:
        return jsonify({"ok": False, "msg": "已有任务在运行"}), 409
    cfg = request.json or {}
    csv_path = (cfg.get("csv_path") or "").strip().strip('"')
    if not os.path.isfile(csv_path):
        return jsonify({"ok": False, "msg": f"找不到清单文件：{csv_path}"}), 400
    years = [int(y) for y in (cfg.get("years") or [])]
    if not years:
        return jsonify({"ok": False, "msg": "请至少选择一个报告年度"}), 400

    job = {
        "csv_path": csv_path,
        "output": (cfg.get("output") or DEFAULT_OUTPUT).strip(),
        "years": years,
        "priority_year": cfg.get("priority_year") or max(years),
        "workers": int(cfg.get("workers") or 4),
        "prefer": cfg.get("prefer") or "auto",
        "overwrite": bool(cfg.get("overwrite")),
        "resume": bool(cfg.get("resume", True)),
        "save_fulltext": bool(cfg.get("save_fulltext", True)),
        "token": (cfg.get("token") or "").strip() or TUSHARE_TOKEN,
        "retry_keys": cfg.get("retry_keys") or [],
        "ai_enabled": bool(cfg.get("ai_enabled")),
        "deepseek_key": (cfg.get("deepseek_key") or "").strip() or DEEPSEEK_API_KEY,
        "deepseek_model": (cfg.get("deepseek_model") or "").strip() or DEEPSEEK_MODEL,
        "deepseek_advanced_model": ((cfg.get("deepseek_advanced_model") or "").strip()
                                    or DEEPSEEK_ADVANCED_MODEL),
        "ai_summary": bool(cfg.get("ai_summary", True)),
        "ai_verify": bool(cfg.get("ai_verify", AI_VERIFY_ENABLED_DEFAULT)),
    }
    _worker = threading.Thread(target=run_job, args=(job,), daemon=True)
    _worker.start()
    return jsonify({"ok": True})


@app.post("/api/retry")
def api_retry():
    """只重跑上一轮标记为异常的任务。"""
    if STATE.running:
        return jsonify({"ok": False, "msg": "任务运行中"}), 409
    keys = [r["key"] for r in STATE.records
            if r.get("status") in ABNORMAL]
    if not keys:
        return jsonify({"ok": False, "msg": "没有需要重跑的异常任务"}), 400
    return jsonify({"ok": True, "keys": keys, "count": len(keys)})


@app.post("/api/stop")
def api_stop():
    STATE.stop_flag = True
    STATE.log("收到停止指令，正在等待在途任务结束…", "warn")
    return jsonify({"ok": True})


@app.get("/api/status")
def api_status():
    try:
        offset = int(request.args.get("log_from", 0))
    except ValueError:
        offset = 0
    return jsonify(STATE.snapshot(offset))


@app.get("/api/preview")
def api_preview():
    ts_code = request.args.get("ts_code", "")
    year = request.args.get("year", "")
    key = request.args.get("key", "")
    name = request.args.get("name", "")
    if key not in SECTION_KEYS:
        return jsonify({"ok": False, "msg": "未知章节"}), 400
    store = Store(STATE.output or DEFAULT_OUTPUT)
    record = next((r for r in STATE.records
                   if r.get("ts_code") == ts_code and str(r.get("year")) == str(year)), None)
    path = store.resolve_section_file(
        record or {"ts_code": ts_code, "name": name, "year": int(year)}, key)
    if not path or not os.path.isfile(path):
        return jsonify({"ok": False, "msg": "该章节未切出或文件已被移动"}), 404
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    return jsonify({"ok": True, "path": path, "chars": len(text),
                    "text": text[:60000],
                    "truncated": len(text) > 60000})


@app.get("/api/download")
def api_download():
    kind = request.args.get("kind", "xlsx")
    path = STATE.exports.get(kind)
    if not path or not os.path.isfile(path):
        return jsonify({"ok": False, "msg": "结果文件尚未生成"}), 404
    return send_file(path, as_attachment=True)


@app.post("/api/reveal")
def api_reveal():
    """在系统文件管理器中打开输出目录。"""
    path = STATE.output or DEFAULT_OUTPUT
    if not os.path.isdir(path):
        return jsonify({"ok": False, "msg": "目录尚不存在"}), 404
    try:
        if platform.system() == "Windows":
            os.startfile(path)                              # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return jsonify({"ok": True, "path": path})
    except Exception as e:                                  # noqa: BLE001
        return jsonify({"ok": False, "msg": str(e)}), 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  年报采集与章节切片工作台 已启动 → {url}\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
