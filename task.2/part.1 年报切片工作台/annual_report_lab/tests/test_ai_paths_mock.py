# -*- coding: utf-8 -*-
"""用假的 DeepSeek 响应跑通「概括」和「后验」两条 AI 路径（不联网）。

    python tests/test_ai_paths_mock.py

重点验证防幻觉的两道闸：
  * 后验给的引文能在原文里精确找到 → 允许修正边界；
  * 引文找不到（模型现编） → 拒绝修正，只留警告。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ai_assist import DeepSeekClient, summarize_business   # noqa: E402
from core.config import SS                                      # noqa: E402
from core.slicer import slice_report                            # noqa: E402
from core.verify import summarize_verdicts, verify_sections     # noqa: E402
from test_slicer import RD_DETAIL, RD_NA, build_new_layout, check  # noqa: E402


def fake_client(responder):
    c = DeepSeekClient("sk-fake-key-for-offline-test")
    c.chat_json = lambda system, user, **kw: responder(system, user, kw)  # type: ignore
    return c


def main() -> int:
    ok = True
    raw = build_new_layout(RD_DETAIL)
    sliced = slice_report(raw)
    text = sliced["normalized"]

    # ---------- 1. 业务概况概括 ----------
    print("\n===== 1. 业务概况：高级模型概括 =====")
    summary_json = {
        "一句话定位": "一家企业级软件公司",
        "所处行业与行业趋势": ["软件和信息技术服务业，行业集中度提升"],
        "主营业务与主要产品服务": ["数据中台", "智能风控平台"],
        "经营模式": {"销售模式": "直销与渠道相结合", "采购模式": "年报未披露"},
        "年报未披露的条目": ["主要客户名称"],
    }
    calls = {"model": None}

    def resp_summary(system, user, kw):
        calls["model"] = kw.get("model")
        return summary_json

    cli = fake_client(resp_summary)
    res = summarize_business(sliced["sections"]["mdna"]["text"], cli,
                             company="示例科技", year="2025")
    ok &= check("概括成功返回", bool(res))
    ok &= check("用的是高级模型", calls["model"] == cli.advanced_model, str(calls["model"]))
    ok &= check("渲染文本含条目编号", res and "1. 一句话定位" in res["text"])
    ok &= check("保留「年报未披露」标记", res and "年报未披露" in res["text"])
    print("  概括预览：", (res["text"][:60] + "…") if res else "—")

    # ---------- 2. 后验：引文可核对 → 允许修正 ----------
    print("\n===== 2. 后验：引文能在原文里找到，允许修正边界 =====")
    sec = dict(sliced["sections"]["rd"])
    orig_start = sec["start"]
    # 故意把起点往后挪 40 个字，模拟"切早/切晚了"
    sec["start"] = orig_start + 40
    sec["text"] = text[sec["start"]:sec["end"]]
    sec["chars"] = sec["end"] - sec["start"]
    true_quote = text[orig_start:orig_start + 14]

    def resp_fix(system, user, kw):
        if "质检员" in system and "不适用" in system:
            return {"is_na": True, "verdict": "pass", "reason": "确为不适用"}
        return {"start_ok": False, "end_ok": True, "content_match": True,
                "complete": True, "verdict": "warn", "reason": "起点晚了",
                "issues": ["漏掉了小节标题"],
                "suggest_start_quote": true_quote, "suggest_end_quote": ""}

    secs = {**sliced["sections"], "rd": sec}
    secs, verdicts = verify_sections(text, secs, fake_client(resp_fix))
    ok &= check("研发投入起点被修回标题处", secs["rd"]["start"] == orig_start,
                f"{secs['rd']['start']} vs {orig_start}")
    ok &= check("后验结论标为 fixed", verdicts["rd"]["verdict"] == "fixed",
                verdicts["rd"]["verdict"])
    ok &= check("识别方式里留了修正痕迹", "后验修正边界" in secs["rd"]["how"])

    # ---------- 3. 后验：引文核对不上 → 拒绝修正 ----------
    print("\n===== 3. 后验：引文是编的，必须拒绝修正 =====")
    sec2 = dict(sliced["sections"]["rd"])
    before = (sec2["start"], sec2["end"])

    def resp_halluc(system, user, kw):
        return {"start_ok": False, "end_ok": False, "content_match": True,
                "complete": False, "verdict": "fail", "reason": "边界不对",
                "issues": [],
                "suggest_start_quote": "本节由模型凭空编造的一句话，原文里没有",
                "suggest_end_quote": "同样是编的另一句话"}

    secs2 = {**sliced["sections"], "rd": sec2}
    secs2, v2 = verify_sections(text, secs2, fake_client(resp_halluc))
    ok &= check("区间没有被改动", (secs2["rd"]["start"], secs2["rd"]["end"]) == before)
    ok &= check("结论保持 fail 未变 fixed", v2["rd"]["verdict"] == "fail",
                v2["rd"]["verdict"])
    worst, problems, detail = summarize_verdicts(v2)
    ok &= check("汇总结论取最差项", worst == "fail", f"{worst}｜{detail}")
    ok &= check("问题清单非空", bool(problems), str(problems[:1]))

    # ---------- 4. 后验推翻「不适用」误判 ----------
    print("\n===== 4. 后验推翻「不适用」误判 =====")
    s_na = slice_report(build_new_layout(RD_NA))
    na_sec = s_na["sections"]["rd"]
    ok &= check("规则先判为不适用", na_sec["status"] == SS.NOT_APPLICABLE)

    def resp_overturn(system, user, kw):
        if "不适用" in system:
            return {"is_na": False, "verdict": "fail", "reason": "正文其实有内容",
                    "issues": ["勾选框识别有误"]}
        return {"start_ok": True, "end_ok": True, "content_match": True,
                "complete": True, "verdict": "pass", "reason": "ok"}

    secs3, v3 = verify_sections(s_na["normalized"], dict(s_na["sections"]),
                                fake_client(resp_overturn))
    ok &= check("不适用判定被推翻", secs3["rd"].get("na") is False)
    ok &= check("状态改回普通切片", secs3["rd"]["status"] == SS.ORIGINAL)
    ok &= check("留下人工确认提示", "请人工确认" in secs3["rd"]["how"])

    # ---------- 5. 业务概况概括的幻觉核查 ----------
    print("\n===== 5. 业务概况幻觉核查 =====")
    biz = {**sliced["sections"]["business"], "status": SS.AI_SUMMARY,
           "_source_excerpt": sliced["sections"]["mdna"]["text"][:8000],
           "text": "1. 主要客户\n   - 某某银行、某某保险（原文未出现）"}

    def resp_unsup(system, user, kw):
        if "事实核查员" in system:
            return {"unsupported": ["某某银行、某某保险"], "verdict": "warn",
                    "reason": "客户名称原文没有"}
        return {"start_ok": True, "end_ok": True, "content_match": True,
                "complete": True, "verdict": "pass", "reason": "ok"}

    secs4, v4 = verify_sections(text, {**sliced["sections"], "business": biz},
                                fake_client(resp_unsup))
    ok &= check("发现未支持表述 → 结论 fail", v4["business"]["verdict"] == "fail",
                v4["business"]["verdict"])
    ok &= check("问题写进 issues", bool(v4["business"]["issues"]))

    print("\n" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
