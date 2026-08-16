# -*- coding: utf-8 -*-
"""切片后验（AI 质检）。

程序切完之后，再让高级模型回头检查每一段切片：**起点对不对、终点对不对、
内容和章节名是否相符、有没有被截断**。这一步只做三件事：

1. 出结论（pass / warn / fail），写进结果表和 _meta.json，异常项会把任务状态降为"需复核"；
2. 在模型给出**可核对的原文引文**时，做保守的边界修正（引文必须能在原文里精确找到，
   修正后的长度不得超过原切片的 AI_VERIFY_MAX_EXPAND_RATIO 倍，否则放弃修正只记警告）；
3. 对两类特殊切片换一套检法：
   - 「不适用」切片 → 复核公司是不是真的填了不适用（防止把"本项目不适用于…"误判）；
   - AI 概括生成的业务概况 → 复核概括内容是否全部有原文支撑（查幻觉），不查边界。

设计上刻意让 AI **不能凭空写正文**：它能做的只有"给一段原文引文当新边界"，
引文核对不上就整条作废。最坏情况是"没修成"，不会是"内容被改写"。
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from .ai_assist import DeepSeekClient
from .config import (AI_VERIFY_CONTEXT_CHARS, AI_VERIFY_HEAD_CHARS,
                     AI_VERIFY_MAX_CALLS_PER_REPORT, AI_VERIFY_MAX_EXPAND_RATIO,
                     AI_VERIFY_TAIL_CHARS, SECTION_DESC, SECTION_KEYS,
                     SECTION_NAMES, SS)

_VERDICTS = ("pass", "warn", "fail")


def _blank_verdict(verdict: str = "skip", reason: str = "") -> Dict:
    return {"verdict": verdict, "reason": reason, "issues": [],
            "start_ok": None, "end_ok": None, "content_match": None,
            "complete": None, "repaired": False, "model": ""}


def _norm_verdict(v: str) -> str:
    v = (v or "").strip().lower()
    return v if v in _VERDICTS else "warn"


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
_SYS_BOUNDARY = (
    "你是年报切片的质检员。用户会给你：章节名、该章节应有的内容说明、"
    "程序切出来的这一段的【开头】和【结尾】，以及切片【前面一点】和【后面一点】的原文。\n"
    "请判断这一段切得对不对，逐项回答：\n"
    "- start_ok：切片是不是正好从该章节的标题行开始（既没有把上一节的尾巴带进来，"
    "也没有漏掉标题和开头几段）；\n"
    "- end_ok：切片是不是正好停在该章节结束的位置（下一个同级或更高级标题之前）；\n"
    "- content_match：内容是不是确实属于这个章节；\n"
    "- complete：有没有被从中间截断（结尾是不是一句话说到一半、表格断在中间）。\n"
    "如果 start_ok 为 false，请在 suggest_start_quote 里给出**正确起点所在那一行的原文**"
    "（从我给你的文本里逐字复制 8~30 个字，不得改写）；\n"
    "如果 end_ok 为 false，请在 suggest_end_quote 里给出**该章节结束后紧接着的那一行原文**"
    "（同样逐字复制）。不确定就留空字符串，不要猜。\n"
    "严格返回 JSON，不要解释：\n"
    '{"start_ok":true,"end_ok":true,"content_match":true,"complete":true,'
    '"verdict":"pass","reason":"一句话结论","issues":["具体问题"],'
    '"suggest_start_quote":"","suggest_end_quote":""}\n'
    "verdict 取值：pass（可直接使用）/ warn（可用但需人工看一眼）/ fail（切错了）。")

_SYS_NA = (
    "你是年报切片的质检员。程序判定某个小节被公司填报为「不适用」。"
    "请根据给出的原文判断这个结论对不对：真正的「不适用」是指该小节正文只有"
    "「不适用」「无」或勾选了「□适用 √不适用」；如果正文其实有实质内容，"
    "或者「不适用」三个字只是出现在一句话中间（如「本项目不适用于合并范围外主体」），"
    "那么这个判定是错的。严格返回 JSON：\n"
    '{"is_na":true,"verdict":"pass","reason":"一句话结论","issues":[]}')

_SYS_SUMMARY = (
    "你是学术研究的事实核查员。用户给你一份由 AI 生成的公司「业务概况」，"
    "以及它所依据的年报原文节选。请核查这份概况有没有**原文不支持的内容**："
    "编造的数字、原文没有的客户或产品名称、超出原文的推断。\n"
    "注意：概况里写「年报未披露」的条目属于正确做法，不算问题。\n"
    "严格返回 JSON，不要解释：\n"
    '{"unsupported":["有问题的表述原文照抄"],"verdict":"pass",'
    '"reason":"一句话结论","issues":[]}')


# --------------------------------------------------------------------------
def _payload_boundary(text: str, sec: Dict, name: str) -> str:
    s, e = sec["start"], sec["end"]
    ctx = AI_VERIFY_CONTEXT_CHARS
    before = text[max(0, s - ctx):s]
    head = text[s:s + AI_VERIFY_HEAD_CHARS]
    tail = text[max(s, e - AI_VERIFY_TAIL_CHARS):e]
    after = text[e:e + ctx]
    body_note = ("（切片较短，开头和结尾有重叠）"
                 if e - s <= AI_VERIFY_HEAD_CHARS + AI_VERIFY_TAIL_CHARS else "")
    return (
        f"章节名：{name}\n"
        f"该章节应有的内容：{SECTION_DESC.get(sec.get('_key', ''), name)}\n"
        f"程序的识别方式：{sec.get('how', '')}\n"
        f"切片总长度：{e - s} 字{body_note}\n\n"
        f"【切片前面的原文】\n{before}\n\n"
        f"【切片开头】\n{head}\n\n"
        f"【切片结尾】\n{tail}\n\n"
        f"【切片后面的原文】\n{after}\n")


def _locate_quote(text: str, quote: str, lo: int, hi: int) -> Optional[int]:
    """引文必须能在指定窗口里精确找到，否则作废（防幻觉）。"""
    q = (quote or "").strip()
    if len(q) < 6:
        return None
    lo, hi = max(0, lo), min(len(text), hi)
    pos = text.find(q, lo, hi)
    if pos >= 0:
        return pos
    # 模型可能吞掉了空格/换行：退一步在压缩串上找，再映射回原文
    win = text[lo:hi]
    idx = [i for i, ch in enumerate(win) if not ch.isspace()]
    compact = "".join(win[i] for i in idx)
    cq = re.sub(r"\s+", "", q)
    p = compact.find(cq)
    if p >= 0:
        return lo + idx[p]
    return None


def _try_repair(text: str, sec: Dict, res: Dict, log: Callable) -> bool:
    """按后验意见保守重切；任何一项校验不过就整体放弃。"""
    s, e = sec["start"], sec["end"]
    ctx = AI_VERIFY_CONTEXT_CHARS
    new_s, new_e = s, e

    if res.get("start_ok") is False:
        p = _locate_quote(text, res.get("suggest_start_quote", ""),
                          max(0, s - ctx), s + AI_VERIFY_HEAD_CHARS)
        if p is not None:
            new_s = p
    if res.get("end_ok") is False:
        p = _locate_quote(text, res.get("suggest_end_quote", ""),
                          max(s, e - AI_VERIFY_TAIL_CHARS), e + ctx)
        if p is not None:
            new_e = p

    if (new_s, new_e) == (s, e):
        return False
    if new_e - new_s < 100:
        log("后验修正被否决：修正后长度过短", "warn")
        return False
    cap = max((e - s) * AI_VERIFY_MAX_EXPAND_RATIO, 3000)
    if new_e - new_s > cap:
        log("后验修正被否决：修正后长度超出安全上限", "warn")
        return False

    sec["start"], sec["end"] = new_s, new_e
    sec["text"] = text[new_s:new_e].strip()
    sec["chars"] = new_e - new_s
    sec["how"] = (sec.get("how", "") + "｜后验修正边界"
                  + (f"（起点 {s}→{new_s}）" if new_s != s else "")
                  + (f"（终点 {e}→{new_e}）" if new_e != e else ""))
    sec["loose"] = False
    return True


# --------------------------------------------------------------------------
def verify_sections(text: str, sections: Dict, client: DeepSeekClient,
                    company: str = "", year: str = "",
                    log: Callable = lambda *a: None,
                    max_calls: int = AI_VERIFY_MAX_CALLS_PER_REPORT,
                    autofix: bool = True) -> Tuple[Dict, Dict]:
    """对已切好的各章节做后验。返回 (更新后的 sections, {key: verdict})。"""
    verdicts: Dict[str, Dict] = {}
    if not client.enabled:
        for k in SECTION_KEYS:
            verdicts[k] = _blank_verdict("skip", "未启用 AI 后验")
        return sections, verdicts

    calls = 0
    model = client.advanced_model
    for key in SECTION_KEYS:
        sec = sections.get(key) or {}
        name = SECTION_NAMES.get(key, key)
        if not sec.get("found"):
            verdicts[key] = _blank_verdict("skip", "该章节未切出，无可后验内容")
            continue
        if calls >= max_calls:
            verdicts[key] = _blank_verdict("skip", "已达本篇后验调用上限")
            continue

        sec["_key"] = key
        try:
            if sec.get("status") == SS.NOT_APPLICABLE:
                calls += 1
                snippet = text[max(0, sec["start"] - 200):sec["end"] + 200]
                res = client.chat_json(
                    _SYS_NA,
                    f"小节名：{name}\n程序判定依据：{sec.get('na_reason', '')}\n\n"
                    f"原文：\n{snippet}",
                    model=model, max_tokens=500)
                if not res:
                    verdicts[key] = _blank_verdict("error", client.last_error)
                    continue
                v = _norm_verdict(res.get("verdict"))
                if res.get("is_na") is False:
                    v = "fail"
                    sec["na"] = False
                    sec["status"] = SS.ORIGINAL
                    sec["how"] += "｜后验推翻「不适用」判定，请人工确认"
                    log(f"后验推翻「{name}」的不适用判定，已改回普通切片并标记复核", "warn")
                verdicts[key] = {**_blank_verdict(v, res.get("reason", "")),
                                 "issues": list(res.get("issues") or []),
                                 "model": client._resolve(model)}

            elif sec.get("status") == SS.AI_SUMMARY:
                calls += 1
                src = sec.get("_source_excerpt") or ""
                res = client.chat_json(
                    _SYS_SUMMARY,
                    f"公司：{company}　年度：{year}\n\n【AI 生成的业务概况】\n"
                    f"{sec.get('text', '')[:6000]}\n\n【年报原文节选】\n{src[:14000]}",
                    model=model, max_tokens=900, timeout=150)
                if not res:
                    verdicts[key] = _blank_verdict("error", client.last_error)
                    continue
                unsup = [u for u in (res.get("unsupported") or []) if str(u).strip()]
                v = "fail" if unsup else _norm_verdict(res.get("verdict"))
                if unsup:
                    log(f"业务概况后验发现 {len(unsup)} 处原文未支持的表述，已标记复核", "warn")
                verdicts[key] = {**_blank_verdict(v, res.get("reason", "")),
                                 "issues": unsup + list(res.get("issues") or []),
                                 "model": client._resolve(model)}

            else:
                calls += 1
                res = client.chat_json(_SYS_BOUNDARY,
                                       _payload_boundary(text, sec, name),
                                       model=model, max_tokens=900, timeout=150)
                if not res:
                    verdicts[key] = _blank_verdict("error", client.last_error)
                    continue
                v = _norm_verdict(res.get("verdict"))
                issues = [str(x) for x in (res.get("issues") or []) if str(x).strip()]
                repaired = False
                # MD&A 的结构化 PDF 边界来自确定性标题树；AI 只质检，不自动改写其边界。
                if autofix and key != "mdna" and v != "pass":
                    repaired = _try_repair(text, sec, res,
                                           lambda m, lv="warn": log(f"「{name}」{m}", lv))
                    if repaired:
                        log(f"「{name}」按后验意见重切完成（{sec['chars']} 字）", "ok")
                verdicts[key] = {
                    "verdict": "fixed" if repaired else v,
                    "reason": str(res.get("reason", ""))[:300],
                    "issues": issues,
                    "start_ok": res.get("start_ok"), "end_ok": res.get("end_ok"),
                    "content_match": res.get("content_match"),
                    "complete": res.get("complete"),
                    "repaired": repaired, "model": client._resolve(model),
                }
                if v != "pass" and not repaired:
                    log(f"「{name}」后验结论 {v}：{res.get('reason', '')}", "warn")
        except Exception as e:                            # noqa: BLE001
            verdicts[key] = _blank_verdict("error", f"{type(e).__name__}: {e}")
        finally:
            sec.pop("_key", None)

    return sections, verdicts


def summarize_verdicts(verdicts: Dict) -> Tuple[str, List[str], str]:
    """把各章节的后验结论压成一行，写进结果表。

    返回 (最差结论, 问题清单, 逐章节结论串)。
    """
    order = {"fail": 0, "warn": 1, "error": 2, "fixed": 3, "pass": 4, "skip": 5}
    worst, parts, problems = "skip", [], []
    for k in SECTION_KEYS:
        v = verdicts.get(k) or {}
        vv = v.get("verdict", "skip")
        if order.get(vv, 9) < order.get(worst, 9):
            worst = vv
        parts.append(f"{SECTION_NAMES.get(k, k)}:{vv}")
        if vv in ("fail", "warn", "error"):
            detail = v.get("reason") or "；".join(v.get("issues") or [])
            problems.append(f"{SECTION_NAMES.get(k, k)}—{detail[:120]}")
    return worst, problems, "、".join(parts)
