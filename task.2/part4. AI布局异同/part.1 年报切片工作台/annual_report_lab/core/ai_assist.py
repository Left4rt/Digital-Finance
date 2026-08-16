# -*- coding: utf-8 -*-
"""DeepSeek 客户端 + 两类 AI 能力。

一、辅助定位（`enhance_with_ai`）
    LLM 只回答"某章节从哪里开始"，形式是**从原文逐字复制的一小段锚句**。程序拿到锚句
    后用精确子串匹配核对；核对不上就当没找到。终点仍由 core/slicer.py 的确定性规则算。
    → AI 出错的后果被限制在"起点选偏了"，不会出现"整段是模型现编的"。

二、业务概况概括（`summarize_business`）
    年报通常没有独立的"业务概况"板块，正文里切不出来时，用**高级模型**
    （DEEPSEEK_ADVANCED_MODEL，默认 deepseek-v4-pro）对已经切出的"管理层讨论与分析"
    做条目化概括。这一路产出的是**AI 生成文本，不是原文**，因此：
      - 单独标记 status = SS.AI_SUMMARY；
      - 落盘文件头部显著标注"AI 概括生成，非年报原文"；
      - 提示词强制要求"只依据给定原文，原文没写的一律写「年报未披露」"，
        并要求逐条给出原文依据关键词，便于人工回查。
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

from .config import (AI_CHUNK_CHARS, AI_CHUNK_OVERLAP, AI_MAX_CALLS_PER_REPORT,
                     AI_MAX_SECTION_CHARS, AI_SUMMARY_MAX_INPUT_CHARS,
                     AI_SUMMARY_MAX_TOKENS, DEEPSEEK_ADVANCED_MODEL,
                     DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
                     DEEPSEEK_MODEL_FALLBACKS, DEEPSEEK_RATE_PER_MIN,
                     DEEPSEEK_TIMEOUT, SECTION_DESC, SECTION_NAMES, USER_AGENT)
from .sources import RateLimiter

_CHAPTER_HEAD = re.compile(r"第[一二三四五六七八九十]{1,3}[节章]")
_ITEM_HEAD = re.compile(
    r"(?:[一二三四五六七八九十]{1,3}[、.．]|[(（][一二三四五六七八九十]{1,3}[)）]|"
    r"\d{1,2}[、.．]|[(（]\d{1,2}[)）])"
)

_MODEL_ERR = re.compile(r"(model.{0,20}(not|no).{0,20}(exist|found|support)|"
                        r"invalid.{0,15}model|unknown model|模型.{0,6}(不存在|不支持))",
                        re.I)


class DeepSeekClient:
    """一个客户端同时管两个模型：常规模型做定位，高级模型做概括与后验。"""

    def __init__(self, api_key: str, base_url: str = DEEPSEEK_BASE_URL,
                 model: str = DEEPSEEK_MODEL,
                 advanced_model: str = DEEPSEEK_ADVANCED_MODEL):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or DEEPSEEK_MODEL
        self.advanced_model = advanced_model or DEEPSEEK_ADVANCED_MODEL
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        self.limiter = RateLimiter(DEEPSEEK_RATE_PER_MIN)
        self.last_error = ""
        self._bad_models: set = set()      # 服务端不认的模型名，记下来不再重试
        self.model_notes: List[str] = []

    # ---- 基础 ----
    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _resolve(self, requested: Optional[str]) -> str:
        want = requested or self.model
        if want not in self._bad_models:
            return want
        for m in [self.model, self.advanced_model] + DEEPSEEK_MODEL_FALLBACKS:
            if m and m not in self._bad_models:
                return m
        return want

    def ping(self, model: Optional[str] = None) -> Dict:
        if not self.enabled:
            return {"ok": False, "msg": "未填写 DeepSeek API Key"}
        res = self.chat_json("你只回答 JSON。", '返回 {"pong": true}',
                             model=model, retries=1, timeout=20, max_tokens=64)
        if res is None:
            return {"ok": False, "msg": self.last_error or "DeepSeek 无响应"}
        used = self._resolve(model)
        return {"ok": True, "msg": f"DeepSeek 连通（模型 {used}）"}

    def chat_json(self, system: str, user: str, model: Optional[str] = None,
                  retries: int = 2, timeout: int = DEEPSEEK_TIMEOUT,
                  max_tokens: int = 900) -> Optional[Dict]:
        """要求模型返回 JSON；模型名无效时按 DEEPSEEK_MODEL_FALLBACKS 依次降级。"""
        if not self.enabled:
            self.last_error = "未配置 API Key"
            return None
        url = f"{self.base_url}/chat/completions"
        tried: List[str] = []
        for attempt in range(retries + 1):
            use = self._resolve(model)
            tried.append(use)
            payload = {
                "model": use,
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }
            self.limiter.acquire()
            try:
                r = self.sess.post(url, json=payload, timeout=timeout)
                if r.status_code == 401:
                    self.last_error = "API Key 无效或未授权"
                    return None
                if r.status_code in (400, 404) and _MODEL_ERR.search(r.text or ""):
                    self._bad_models.add(use)
                    nxt = self._resolve(model)
                    note = f"模型「{use}」不可用，改用「{nxt}」"
                    if note not in self.model_notes:
                        self.model_notes.append(note)
                    self.last_error = note
                    if nxt == use:
                        return None
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                return _loads_loose(content)
            except Exception as e:                        # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(1.2 * (attempt + 1))
        return None


def _loads_loose(content: str) -> Optional[Dict]:
    """模型偶尔会在 JSON 外面包 ```json 代码块，这里做宽松解析。"""
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    try:
        return json.loads(content)
    except Exception:                                     # noqa: BLE001
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:                             # noqa: BLE001
                return None
        return None


# ==========================================================================
# 一、辅助定位
# ==========================================================================
def _chunks(region: str, size: int, overlap: int) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    i, n = 0, len(region)
    if n <= size:
        return [(0, region)]
    while i < n:
        out.append((i, region[i:i + size]))
        if i + size >= n:
            break
        i += size - overlap
    return out


def _find_end(text: str, start: int, hard_cap: int) -> int:
    """起点已定，终点交给确定性规则：下一个'第X节' / 下一个条目标记 / 长度上限。"""
    limit = min(len(text), start + hard_cap)
    window = text[start:limit]
    search_from = min(40, len(window))
    candidates = []
    m = _CHAPTER_HEAD.search(window, search_from)
    if m:
        candidates.append(m.start())
    m = _ITEM_HEAD.search(window, search_from)
    if m:
        candidates.append(m.start())
    end_rel = min(candidates) if candidates else len(window)
    return start + end_rel


def enhance_with_ai(normalized_text: str, sliced: Dict, client: DeepSeekClient,
                    log: Callable = lambda *a: None,
                    max_calls: int = AI_MAX_CALLS_PER_REPORT
                    ) -> Tuple[Dict, Dict]:
    """对未命中 / 疑似命中的章节尝试用 AI 定位起点。

    「业务概况」不在这里处理 —— 它没有独立板块，走 `summarize_business`。
    「研发投入」若已判定为"不适用"，也不再尝试定位。
    """
    from .config import SS

    sections = {k: dict(v) for k, v in sliced["sections"].items()}
    stats = {"calls": 0, "resolved": [], "tried": False}
    if not client.enabled:
        return sections, stats

    unresolved = [k for k, v in sections.items()
                  if (not v.get("found") or v.get("loose"))
                  and v.get("status") != SS.NOT_APPLICABLE
                  and k not in ("business", "mdna")]
    if not unresolved:
        return sections, stats
    stats["tried"] = True

    mdna_span = sliced.get("mdna_span")
    if mdna_span and "mdna" not in unresolved:
        region_start, region_end = mdna_span
    else:
        region_start, region_end = 0, len(normalized_text)
    region = normalized_text[region_start:region_end]

    for w_offset, w_text in _chunks(region, AI_CHUNK_CHARS, AI_CHUNK_OVERLAP):
        if not unresolved or stats["calls"] >= max_calls:
            break
        if not w_text.strip():
            continue

        targets = "\n".join(f"- {k}：{SECTION_DESC.get(k, SECTION_NAMES.get(k, k))}"
                            for k in unresolved)
        system = (
            "你是年报章节定位助手。用户给你一段年报正文和若干「待定位章节」说明。"
            "你只需判断这段正文里是否出现了某个待定位章节的**标题行**（章节的开头）。"
            "注意：正文句子里出现同样的词（例如「公司持续加大研发投入」）**不算**标题，"
            "只有独立成行的小标题才算。若确认是标题，请把该标题行**逐字复制** 10~30 个字"
            "作为 quote（不得改写、补序号或翻译）。严格返回 JSON，不要解释：\n"
            '{"found":[{"key":"章节key","quote":"逐字复制的标题行"}]}\n'
            '若一个都没有，返回 {"found":[]}。')
        user = f"待定位章节：\n{targets}\n\n年报正文片段：\n{w_text}"

        stats["calls"] += 1
        res = client.chat_json(system, user)
        if not res:
            log(f"DeepSeek 定位调用失败：{client.last_error}", "warn")
            continue

        for item in (res.get("found") or []):
            key = item.get("key")
            quote = (item.get("quote") or "").strip()
            if key not in unresolved or not quote or len(quote) < 4:
                continue
            pos = w_text.find(quote)
            if pos < 0:
                log(f"AI 定位「{SECTION_NAMES.get(key, key)}」作废：引文与原文核对不上", "warn")
                continue
            start = region_start + w_offset + pos
            end = _find_end(normalized_text, start, AI_MAX_SECTION_CHARS)
            sections[key] = {
                **sections.get(key, {}),
                "found": True, "loose": False, "ai": True, "status": SS.AI_LOCATED,
                "how": f"AI 定位（{client._resolve(None)}）：{quote[:24]}",
                "chars": end - start, "origin": "原文",
                "text": normalized_text[start:end].strip(),
                "start": start, "end": end,
            }
            unresolved.remove(key)
            stats["resolved"].append(key)
            log(f"AI 定位命中「{SECTION_NAMES.get(key, key)}」：{quote[:20]}…", "ok")

    return sections, stats


# ==========================================================================
# 二、业务概况：对管理层讨论与分析做条目化概括
# ==========================================================================
_SUMMARY_SCHEMA = {
    "一句话定位": "一句话说明这家公司是做什么的",
    "所处行业与行业趋势": ["条目"],
    "主营业务与主要产品服务": ["条目"],
    "经营模式": {"研发模式": "", "采购模式": "", "生产模式": "",
                 "销售模式": "", "盈利模式": ""},
    "主要客户与销售区域": ["条目"],
    "行业地位与竞争格局": ["条目"],
    "报告期业务进展": ["条目"],
    "核心竞争力": ["条目"],
    "主要风险": ["条目"],
    "关键经营数据": ["指标：数值（须为原文出现的数字）"],
    "年报未披露的条目": ["列出上面哪些项目原文没写"],
}

_SUMMARY_SYSTEM = (
    "你是资深行业研究员，正在为学术研究整理上市公司年报。用户会给你某公司年报中"
    "「管理层讨论与分析」的原文。请据此写一份**业务概况**。\n"
    "硬性要求：\n"
    "1. 只依据给定原文，严禁引入任何外部知识、严禁推测、严禁编造数字或客户名称；\n"
    "2. 原文没有写的项目，该字段直接写「年报未披露」，并在「年报未披露的条目」里列出；\n"
    "3. 条目化表述，每条一句话，尽量保留原文的专有名词、产品名、行业术语；\n"
    "4. 「关键经营数据」里的每个数字都必须能在原文中原样找到，找不到就不要写；\n"
    "5. 全部使用简体中文；\n"
    "6. 严格按给定 JSON 结构输出，不要输出 JSON 以外的任何内容。")


def _shrink(text: str, budget: int) -> Tuple[str, bool]:
    """太长时头尾取样：业务与行业在开头，展望与风险在结尾，中间多是财务明细表。"""
    if len(text) <= budget:
        return text, False
    head = int(budget * 0.7)
    tail = budget - head
    return (text[:head] + "\n\n〔……中间部分（多为财务明细表格）已略去……〕\n\n"
            + text[-tail:]), True


def summarize_business(mdna_text: str, client: DeepSeekClient,
                       company: str = "", year: str = "",
                       extra_text: str = "",
                       log: Callable = lambda *a: None) -> Optional[Dict]:
    """用高级模型把管理层讨论与分析概括成业务概况。

    返回 {'text': 可读文本, 'data': 原始 JSON, 'model': 实际使用的模型,
          'truncated': 是否做了头尾取样}；失败返回 None。
    """
    if not client.enabled:
        return None
    src = (mdna_text or "").strip()
    if extra_text:
        src += "\n\n〔核心竞争力章节原文〕\n" + extra_text.strip()
    if len(src) < 400:
        log("管理层讨论与分析正文过短，跳过业务概况概括", "warn")
        return None

    src, truncated = _shrink(src, AI_SUMMARY_MAX_INPUT_CHARS)
    model = client.advanced_model
    user = (f"公司：{company or '（未提供）'}\n报告年度：{year or '（未提供）'}\n\n"
            f"请按下面的 JSON 结构输出（键名照抄，值按实际内容填写）：\n"
            f"{json.dumps(_SUMMARY_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
            f"以下是「管理层讨论与分析」原文：\n{src}")

    data = client.chat_json(_SUMMARY_SYSTEM, user, model=model,
                            max_tokens=AI_SUMMARY_MAX_TOKENS, timeout=180)
    if not data:
        log(f"业务概况概括失败：{client.last_error}", "warn")
        return None
    return {"text": render_summary(data), "data": data,
            "model": client._resolve(model), "truncated": truncated}


def render_summary(data: Dict) -> str:
    """把模型返回的 JSON 渲染成便于阅读与检索的纯文本。"""
    lines: List[str] = []
    order = list(_SUMMARY_SCHEMA.keys())
    keys = order + [k for k in data if k not in order]
    n = 0
    for k in keys:
        if k not in data:
            continue
        v = data[k]
        n += 1
        lines.append(f"{n}. {k}")
        if isinstance(v, dict):
            for kk, vv in v.items():
                lines.append(f"   - {kk}：{_flat(vv)}")
        elif isinstance(v, list):
            for item in v:
                lines.append(f"   - {_flat(item)}")
        else:
            lines.append(f"   {_flat(v)}")
        lines.append("")
    return "\n".join(lines).strip()


def _flat(v) -> str:
    if isinstance(v, (list, tuple)):
        return "；".join(_flat(x) for x in v)
    if isinstance(v, dict):
        return "；".join(f"{k}：{_flat(x)}" for k, x in v.items())
    return str(v).strip()
