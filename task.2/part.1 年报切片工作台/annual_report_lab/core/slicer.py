# -*- coding: utf-8 -*-
"""年报章节切片（标题行锚定版）。

和初版的根本区别
----------------
初版是"在全文里搜关键词"，谁先出现算谁的，所以"公司持续加大**研发投入**力度"这种
正文句子会被当成小节标题。本版改成**先把全文解析成标题树，再在标题上做匹配**：

    整篇正文 → 逐行判断"这一行是不是标题" → 标题带层级（第X节 / 一、/（一）/ 1、/（1））
             → 目录区整体剔除 → 在标题上按规则打分匹配 → 切片区间由相邻标题决定

由此得到四条硬性质：

1. **管理层讨论与分析按整节切**：命中的是"第X节"级标题，终点就是**下一个"第X节"标题**，
   中间无论有多少条目都完整保留（`slice_chapter`）。
2. **研发投入按小节切**：终点是下一个同级或更高级的条目标题（"（五）研发投入"的下一个
   是"（六）…"或"五、…"或"第四节"），不会切到一半，也不会吃掉相邻小节。
3. **关键词只在标题行上匹配**，正文句子里出现同样的词不会命中；匹配还要打分，
   分数不够只记为"疑似"（loose），交给 AI 后验，而不是直接当成正确结果。
4. **公司填「不适用」能被识别**：小节存在但正文只有"不适用"/"□适用 √不适用"，
   记为 NA 状态单独汇报，而不是混在"未识别"里。

PDF 抽文的三个老问题仍然照顾到：
  * 字间插空格（"第 三 节  管 理 层"）→ 每行先压缩空白再判断；
  * 标题跨行（"第三节" 独占一行，标题在下一行）→ 相邻行合并后再判断；
  * 页眉页脚每页重复 / 目录点线 → 高频短行剔除 + 目录区识别后整体屏蔽。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import NA_MAX_BODY_CHARS, NA_TICKS, SECTION_RULES, SS

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
LEVEL_CHAPTER = 0        # 第X节 / 第X章 / 第X部分 / 第X篇
LEVEL_CN = 1             # 一、二、三、
LEVEL_CN_PAREN = 2       # （一）（二）
LEVEL_AR = 3             # 1、2、3、
LEVEL_AR_PAREN = 4       # （1）（2）
LEVEL_WEAK = 9           # 没有序号的短行，只能当起点，不能当终点

STRONG_SCORE = 55        # 达到这个分数才算"精确切出"
LOOSE_SCORE = 38         # 低于 STRONG 但达到这个分数，记为"疑似"，交给 AI 后验

_FULL2HALF = {ord(c): ord(c) - 0xFEE0 for c in
              "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
              "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"}

_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
           "八": 8, "九": 9, "十": 10}


def cn2int(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + _CN_NUM.get(s[1:2], 0)
    if "十" in s:
        a, _, b = s.partition("十")
        return _CN_NUM.get(a, 0) * 10 + (_CN_NUM.get(b, 0) if b else 0)
    return _CN_NUM.get(s, 0)


def squeeze(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


# --------------------------------------------------------------------------
# 1. 文本规整
# --------------------------------------------------------------------------
_LEADER_RE = re.compile(r"[.．·…‥]{3,}\s*\d{0,4}\s*$")
_PAGENO_RE = re.compile(r"^[-—－\s第]*\d{1,4}[\s页/]*(共?\s?\d{0,4}\s?页?)?$")
_CHAPTER_LINE_RE = re.compile(r"^第[一二三四五六七八九十百0-9]{1,4}(?:节|章|篇|部分)")


def normalize(raw: str) -> Tuple[str, List[str], List[int], List[bool], List[int]]:
    """去页眉页脚、页码、目录点线，并保留“行属于哪一页”的映射。

    PDF 抽取器会在页面之间插入 ``\f``。该标记不进入规整正文，只用于
    给结构化 PDF 裁切提供页码提示；旧缓存没有分页标记时，页码统一为 0，
    后续仍可通过 PDF 标题搜索完成定位。
    """
    text = (raw or "").translate(_FULL2HALF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    src_lines = text.split("\n")

    freq: Dict[str, int] = {}
    for ln in src_lines:
        if ln.strip() == "\f":
            continue
        k = squeeze(ln)
        if 4 <= len(k) <= 40:
            freq[k] = freq.get(k, 0) + 1
    threshold = max(15, len(src_lines) // 400)
    noisy = {k for k, v in freq.items() if v >= threshold}

    out_lines: List[str] = []
    leaders: List[bool] = []
    page_nos: List[int] = []
    page_no = 0
    for ln in src_lines:
        if ln.strip() == "\f":
            page_no += 1
            continue
        k = squeeze(ln)
        if not k:
            if out_lines and out_lines[-1] == "":
                continue
            out_lines.append("")
            leaders.append(False)
            page_nos.append(page_no)
            continue
        if k in noisy and not _CHAPTER_LINE_RE.match(k):
            continue
        if _PAGENO_RE.match(k):
            continue
        had_leader = bool(_LEADER_RE.search(ln))
        ln = _LEADER_RE.sub("", ln)
        out_lines.append(ln.rstrip())
        leaders.append(had_leader)
        page_nos.append(page_no)

    offsets: List[int] = []
    cur = 0
    for ln in out_lines:
        offsets.append(cur)
        cur += len(ln) + 1
    return "\n".join(out_lines), out_lines, offsets, leaders, page_nos


# --------------------------------------------------------------------------
# 2. 标题识别
# --------------------------------------------------------------------------
_RE_CHAPTER = re.compile(r"^第([一二三四五六七八九十百0-9]{1,4})(?:节|章|篇|部分)\s*(.*)$")
_RE_L1 = re.compile(r"^([一二三四五六七八九十]{1,3})\s*[、.．]\s*(.*)$")
_RE_L2 = re.compile(r"^[(（]\s*([一二三四五六七八九十]{1,3})\s*[)）]\s*(.*)$")
_RE_L3 = re.compile(r"^(\d{1,2})\s*[、.．]\s*(?!\d)(.*)$")
_RE_L4 = re.compile(r"^[(（]\s*(\d{1,2})\s*[)）]\s*(.*)$")

_MARKER_ONLY = re.compile(
    r"^(第[一二三四五六七八九十百0-9]{1,4}(?:节|章|篇|部分)|[一二三四五六七八九十]{1,3}[、.．]|"
    r"[(（][一二三四五六七八九十]{1,3}[)）]|\d{1,2}[、.．]|[(（]\d{1,2}[)）])$")

# 一行结尾出现这些，说明它是句子的一部分，不是标题
_NOT_TITLE_TAIL = re.compile(r"[，,；;：:、和及与或的]$")
# 交叉引用："详见第四节 财务报告"
_XREF = re.compile(r"(详见|参见|见本报告|请见|如前|同上|参阅)")


@dataclass
class Heading:
    line_no: int
    pos: int                 # 行首在正文中的字符偏移
    line_end: int            # 行尾偏移（不含换行）
    key: str                 # 压缩掉空白的整行
    level: int
    marker: str = ""         # 序号本身，如 "第三节"、"（五）"
    title: str = ""          # 去掉序号后的标题正文（可能含粘连的正文）
    ordinal: int = 0         # 第X节 的 X
    in_toc: bool = False
    merged_body: bool = False  # 标题和正文粘在同一行
    from_fallback: bool = False
    page_no: int = -1


def _classify(key: str) -> Optional[Tuple[int, str, str, int]]:
    """把压缩后的行文本判成 (level, marker, title, ordinal)；不是标题返回 None。"""
    if not key:
        return None
    m = _RE_CHAPTER.match(key)
    if m:
        n = cn2int(m.group(1))
        if 1 <= n <= 30:
            return LEVEL_CHAPTER, key[:m.start(2)], m.group(2).strip(), n
    for rx, lv in ((_RE_L2, LEVEL_CN_PAREN), (_RE_L4, LEVEL_AR_PAREN),
                   (_RE_L1, LEVEL_CN), (_RE_L3, LEVEL_AR)):
        m = rx.match(key)
        if m:
            title = m.group(2).strip()
            if not title:
                return None
            return lv, key[:m.start(2)], title, cn2int(m.group(1))
    return None


def _is_weak_title(key: str, next_key: str) -> bool:
    """没有序号的短行，可能是加粗小标题（"研发投入情况"）。判定从严。"""
    if not (2 <= len(key) <= 26):
        return False
    if _NOT_TITLE_TAIL.search(key):
        return False
    if key.endswith("。") or key.endswith("."):
        return False
    if re.search(r"\d{4}年|%|％|元|股|万|亿", key):
        return False
    if not next_key:                       # 后面没内容，八成是表格碎片
        return False
    return True


def build_headings(lines: Sequence[str], offsets: Sequence[int], page_nos: Sequence[int] | None = None) -> List[Heading]:
    """逐行判断标题；序号独占一行的情况与下一行合并。"""
    keys = [squeeze(ln) for ln in lines]
    page_nos = list(page_nos or [0] * len(lines))
    n = len(lines)
    heads: List[Heading] = []
    i = 0
    while i < n:
        key = keys[i]
        if not key:
            i += 1
            continue

        merged_to = -1
        probe = key
        # "第三节" 独占一行 → 把下一非空行拼上来当标题
        if _MARKER_ONLY.match(key):
            j = i + 1
            while j < n and not keys[j]:
                j += 1
            if j < n and len(keys[j]) <= 40:
                probe = key + keys[j]
                merged_to = j

        info = _classify(probe)
        if info:
            level, marker, title, ordinal = info
            # 章节标题必须短；太长说明是正文里带序号的句子
            if level == LEVEL_CHAPTER and len(title) > 60:
                title, merged = title[:40], True
            else:
                merged = len(probe) > 80
            if level == LEVEL_CHAPTER and _XREF.search(probe[:12]):
                i += 1
                continue
            heads.append(Heading(
                line_no=i, pos=offsets[i], line_end=offsets[i] + len(lines[i]),
                key=probe, level=level, marker=marker, title=title,
                ordinal=ordinal, merged_body=merged, page_no=page_nos[i]))
        else:
            nxt = ""
            for j in range(i + 1, min(i + 3, n)):
                if keys[j]:
                    nxt = keys[j]
                    break
            if _is_weak_title(key, nxt):
                heads.append(Heading(
                    line_no=i, pos=offsets[i], line_end=offsets[i] + len(lines[i]),
                    key=key, level=LEVEL_WEAK, marker="", title=key, ordinal=0, page_no=page_nos[i]))
        i = merged_to + 1 if merged_to > i else i + 1
    return heads


def fallback_chapter_scan(text: str) -> List[Heading]:
    """整页被抽成一行时的兜底：在全文里扫"第X节"，但要求前面不是汉字。

    "详见第四节财务报告" 里 "第" 前面是 "见"（汉字）→ 排除；
    正文标题前面通常是换行、空格或标点 → 保留。
    """
    heads: List[Heading] = []
    for m in re.finditer(r"第\s*([一二三四五六七八九十百]{1,4})\s*[节章]", text):
        s = m.start()
        prev = text[s - 1] if s else "\n"
        if "\u4e00" <= prev <= "\u9fff":
            continue
        n = cn2int(re.sub(r"\s", "", m.group(1)))
        if not (1 <= n <= 30):
            continue
        tail = squeeze(text[m.end():m.end() + 60])
        heads.append(Heading(line_no=-1, pos=s, line_end=m.end(),
                             key=squeeze(text[s:m.end()]) + tail[:30],
                             level=LEVEL_CHAPTER, marker=m.group(0),
                             title=tail[:24], ordinal=n, from_fallback=True))
    return heads


# --------------------------------------------------------------------------
# 3. 目录区识别
# --------------------------------------------------------------------------
def detect_toc_ranges(text: str, lines: Sequence[str], offsets: Sequence[int],
                      leaders: Sequence[bool], heads: Sequence[Heading]
                      ) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    total = max(len(text), 1)

    # (a) 目录点线成簇
    idxs = [i for i, f in enumerate(leaders) if f]
    grp: List[int] = []
    for i in idxs:
        if grp and i - grp[-1] > 4:
            if len(grp) >= 3:
                ranges.append((offsets[grp[0]], offsets[grp[-1]] + len(lines[grp[-1]])))
            grp = []
        grp.append(i)
    if len(grp) >= 3:
        ranges.append((offsets[grp[0]], offsets[grp[-1]] + len(lines[grp[-1]])))

    # (b) "目录"锚点：目录里的节标题是"序号连续递增且彼此紧挨着"的一串，
    #     正文里的第一节会让序号回落（或间距突然变大），以此判定目录结束在哪。
    #     不用"后面有没有大段正文"来判断 —— 释义、公司简介这些短节会被误伤。
    chapters = [h for h in heads if h.level == LEVEL_CHAPTER]
    for i, ln in enumerate(lines):
        k = squeeze(ln)
        if k in ("目录", "目次", "释义目录", "本报告目录") and offsets[i] < total * 0.3:
            seq = [h for h in chapters if h.pos > offsets[i]]
            last_ord, last_pos, end_pos = 0, offsets[i], None
            for h in seq:
                if h.ordinal <= last_ord or h.pos - last_pos > 600:
                    break
                last_ord, last_pos, end_pos = h.ordinal, h.pos, h.line_end
            if end_pos and last_ord >= 3:        # 至少连着三条才算目录
                ranges.append((offsets[i], end_pos))
            break

    # (c) 兜底：没有点线也没有"目录"两个字时，才用"节标题密集出现"来判目录。
    #     这一条最容易误伤（短年报的正文节本来就短），所以只在前两条都没结果时启用，
    #     并且要求连着 4 个以上节标题彼此只隔几十个字。
    if not ranges:
        run: List[Heading] = []
        for a, h in enumerate(chapters):
            if run and h.pos - run[-1].pos >= 300:
                if len(run) >= 4 and run[0].pos < total * 0.35:
                    ranges.append((run[0].pos, run[-1].line_end))
                run = []
            run.append(h)
        if len(run) >= 4 and run[0].pos < total * 0.35:
            ranges.append((run[0].pos, run[-1].line_end))

    if not ranges:
        return []
    ranges.sort()
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s <= merged[-1][1] + 200:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def mark_toc(heads: List[Heading], ranges: Sequence[Tuple[int, int]]) -> None:
    for h in heads:
        for s, e in ranges:
            if s <= h.pos < e:
                h.in_toc = True
                break


# --------------------------------------------------------------------------
# 4. 节（第X节）序列
# --------------------------------------------------------------------------
def select_body_chapters(heads: Sequence[Heading], text_len: int) -> List[Dict]:
    """挑出正文里真正的"第X节"序列。

    规则：序号从小到大，每个序号取**第一次**出现且位置在上一节之后的那次。
    页眉里重复出现的同一节标题因此被自动忽略；序号缺失（某节没识别到）也不影响，
    它只会让上一节的区间延伸到下一个识别到的节 —— 这正是"到下个标题之前"的语义。
    """
    cands = [h for h in heads if h.level == LEVEL_CHAPTER and not h.in_toc]
    if not cands:
        return []
    by_ord: Dict[int, List[Heading]] = {}
    for h in cands:
        by_ord.setdefault(h.ordinal, []).append(h)
    for v in by_ord.values():
        v.sort(key=lambda x: x.pos)

    picked: List[Heading] = []
    last_pos = -1
    for n in sorted(by_ord):
        for h in by_ord[n]:
            if h.pos > last_pos:
                picked.append(h)
                last_pos = h.pos
                break

    out: List[Dict] = []
    for i, h in enumerate(picked):
        end = picked[i + 1].pos if i + 1 < len(picked) else text_len
        out.append({"ordinal": h.ordinal, "title": h.title or h.key,
                    "start": h.pos, "end": end, "chars": end - h.pos,
                    "heading": h})
    return out


# --------------------------------------------------------------------------
# 5. 标题打分匹配（本版避免误识别的核心）
# --------------------------------------------------------------------------
def score_title(title: str, patterns: Sequence[str]) -> Tuple[int, str]:
    """在**标题**上打分。正文句子进不到这里，因为它压根不会被判成标题。"""
    best, hit = 0, ""
    t = squeeze(title)
    if not t:
        return 0, ""
    for i, pat in enumerate(patterns):
        prio = max(0, 12 - i * 2)          # 规则表里越靠前优先级越高
        m = re.match(pat, t)
        if not m:
            continue
        if re.fullmatch(pat, t):
            sc = 100
        elif len(t) <= len(m.group(0)) + 6:
            sc = 82                        # 标题后面只多了几个字（如"研发投入情况"）
        elif len(t) <= 30:
            sc = 66                        # 标题略长
        else:
            sc = 44                        # 标题和正文粘在一行，可信度打折
        sc += prio
        if sc > best:
            best, hit = sc, m.group(0)
    return best, hit


def _end_of_item(heads: Sequence[Heading], h: Heading, hi: int) -> int:
    """条目终点 = 下一个同级或更高级标题 / 下一个"第X节" / 父区间末尾。"""
    for other in heads:
        if other.pos <= h.pos or other.pos >= hi or other.in_toc:
            continue
        if other.level == LEVEL_CHAPTER:
            return other.pos
        if h.level == LEVEL_WEAK:
            if other.level <= LEVEL_AR_PAREN:
                return other.pos
        elif other.level != LEVEL_WEAK and other.level <= h.level:
            return other.pos
    return hi


def find_item(heads: Sequence[Heading], lo: int, hi: int,
              patterns: Sequence[str]) -> Optional[Dict]:
    """在 [lo,hi) 的标题里找最匹配的一条，返回带区间与分数的候选。"""
    best: Optional[Dict] = None
    for h in heads:
        if h.in_toc or h.pos < lo or h.pos >= hi:
            continue
        if h.level == LEVEL_CHAPTER:
            continue
        sc, hit = score_title(h.title, patterns)
        if sc <= 0:
            continue
        if h.level == LEVEL_WEAK:
            sc -= 18                       # 无序号标题降权
        if h.merged_body:
            sc -= 12
        if sc < LOOSE_SCORE:
            continue
        end = _end_of_item(heads, h, hi)
        cand = {"start": h.pos, "end": end, "score": sc, "heading": h,
                "title": (h.marker + h.title)[:48], "hit": hit,
                "loose": sc < STRONG_SCORE}
        if best is None or cand["score"] > best["score"] or (
                cand["score"] == best["score"] and cand["start"] < best["start"]):
            best = cand
    return best


def find_items_merged(heads: Sequence[Heading], lo: int, hi: int,
                      patterns: Sequence[str]) -> Optional[Dict]:
    """业务概况这类"由相邻两三条组成"的情况：把连着的条目合成一段。"""
    hits: List[Dict] = []
    for pat in patterns:
        c = find_item(heads, lo, hi, [pat])
        if c and not c["loose"]:
            hits.append(c)
    if not hits:
        return find_item(heads, lo, hi, patterns)
    uniq: Dict[int, Dict] = {}
    for c in hits:
        cur = uniq.get(c["start"])
        if cur is None or c["score"] > cur["score"]:
            uniq[c["start"]] = c
    ordered = sorted(uniq.values(), key=lambda x: x["start"])
    merged = dict(ordered[0])
    titles = [merged["title"]]
    for c in ordered[1:]:
        if c["start"] <= merged["end"] + 80:
            merged["end"] = max(merged["end"], c["end"])
            titles.append(c["title"])
        else:
            break
    merged["title"] = " + ".join(titles[:3])
    return merged


# --------------------------------------------------------------------------
# 6.「不适用」判定
# --------------------------------------------------------------------------
_NA_TICK_YES = re.compile(rf"[{NA_TICKS}]\s*适用")
_NA_TICK_NO = re.compile(rf"[{NA_TICKS}]\s*不适用")
_BOX = "□☐▢○"


def detect_not_applicable(section_text: str, heading_line: str = "") -> Tuple[bool, str]:
    """判断这一小节是不是公司填的「不适用」。返回 (是否不适用, 依据)。"""
    body = section_text or ""
    # 去掉首行标题本身
    if heading_line:
        first_nl = body.find("\n")
        if first_nl != -1 and squeeze(body[:first_nl]) == squeeze(heading_line):
            body = body[first_nl + 1:]
    k = squeeze(body)
    head_k = squeeze(heading_line)
    probe = (head_k + k)[:120]

    # 勾选框优先：勾在"适用"前面就是适用，别误判
    if _NA_TICK_YES.search(probe) and not _NA_TICK_NO.search(probe):
        return False, ""
    if _NA_TICK_NO.search(probe):
        return True, "勾选「不适用」"
    if re.search(rf"是否适用[{_BOX}]?是[{NA_TICKS}]否", probe):
        return True, "勾选「是否适用：否」"

    if not k:
        return False, ""
    if len(k) <= NA_MAX_BODY_CHARS:
        if re.fullmatch(r"(不适用|不适用。|无|无。|—+|-+|/|不涉及|不适用[.。]?)", k):
            return True, "正文仅填「不适用」"
        if k.startswith("不适用") and len(k) <= 20:
            return True, "正文以「不适用」开头且无实质内容"
    return False, ""


# --------------------------------------------------------------------------
# 7. 管理层讨论与分析：语义标题 + 层级边界 + 内容簇兜底
# --------------------------------------------------------------------------
_MDna_ALIAS_RE = re.compile(
    r"^(?:本节)?(?:管理层讨论(?:与|及|和)分析|管理层分析与讨论|"
    r"经营层讨论(?:与|及|和)分析|经营情况讨论(?:与|及|和)分析|"
    r"经营管理层讨论(?:与|及|和)分析|董事会报告)(?:情况|概述)?$",
    re.I,
)
_MDna_STRONG_RE = re.compile(
    r"(?:管理层讨论(?:与|及|和)分析|经营层讨论(?:与|及|和)分析|"
    r"经营情况讨论(?:与|及|和)分析|经营管理层讨论(?:与|及|和)分析|"
    r"管理层分析与讨论)", re.I,
)
_MDna_INTERNAL = [
    re.compile(p) for p in (
        r"报告期内公司所处行业", r"报告期内公司从事的主要业务", r"主营业务分析",
        r"非主营业务分析", r"资产及负债状况", r"投资状况分析", r"重大资产和股权出售",
        r"主要控股参股公司分析", r"公司未来发展的展望", r"未来发展展望", r"核心竞争力分析",
    )
]
_MAJOR_AFTER_MDNA = re.compile(
    r"^(?:公司治理|董事会工作报告|监事会报告|重要事项|股份变动及股东情况|"
    r"优先股相关情况|债券相关情况|财务报告|备查文件目录|环境和社会责任|社会责任|"
    r"员工情况|公司简介|释义)(?:情况|报告)?$"
)
_COMMON_MARKER = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]{1,4}(?:节|章|篇|部分)|"
    r"[一二三四五六七八九十]{1,3}[、.．]|[(（][一二三四五六七八九十]{1,3}[)）]|"
    r"\d{1,2}[、.．]|[(（]\d{1,2}[)）])"
)


def _semantic_title(raw: str) -> str:
    t = squeeze(raw).strip("：:。．")
    t = _COMMON_MARKER.sub("", t)
    return t.strip("：:。．")


def _is_mdna_title(raw: str) -> bool:
    t = _semantic_title(raw)
    if not t or _XREF.search(t[:12]):
        return False
    return bool(_MDna_ALIAS_RE.fullmatch(t))


def _mdna_title_score(h: Heading) -> int:
    t = _semantic_title(h.title or h.key)
    if not t:
        return 0
    if _MDna_ALIAS_RE.fullmatch(t):
        score = 170
        if t == "董事会报告":
            score = 118
        elif "经营层" in t:
            score += 8
        elif t == "管理层讨论与分析":
            score += 12
    elif _MDna_STRONG_RE.search(t) and len(t) <= 32:
        score = 105
    else:
        return 0
    if h.level == LEVEL_WEAK:
        score -= 4
    if h.from_fallback:
        score -= 8
    return score


def _extra_mdna_heads(lines: Sequence[str], offsets: Sequence[int],
                      page_nos: Sequence[int]) -> List[Heading]:
    """不依赖通用标题分类器，直接捞取任何层级的 MD&A 标题行。"""
    keys = [squeeze(x) for x in lines]
    out: List[Heading] = []
    for i, key in enumerate(keys):
        if not key:
            continue
        probes = [(key, i)]
        j = i + 1
        while j < len(keys) and not keys[j]:
            j += 1
        if j < len(keys) and len(key) <= 16 and len(keys[j]) <= 32:
            probes.append((key + keys[j], j))
        for probe, end_i in probes:
            if not _is_mdna_title(probe):
                continue
            info = _classify(probe)
            if info:
                level, marker, title, ordinal = info
            else:
                level, marker, title, ordinal = LEVEL_WEAK, "", probe, 0
            out.append(Heading(
                line_no=i, pos=offsets[i], line_end=offsets[end_i] + len(lines[end_i]),
                key=probe, level=level, marker=marker, title=title, ordinal=ordinal,
                page_no=page_nos[i]))
            break
    return out


def _mdna_boundary(heads: Sequence[Heading], target: Heading, text_len: int
                   ) -> Tuple[int, Optional[Heading]]:
    """按目标标题的实际层级找下一个兄弟/祖先标题，而不是假设它一定是“第X节”。"""
    for h in heads:
        if h.in_toc or h.pos <= target.pos:
            continue
        # 每页重复的“第三节 管理层讨论与分析”是页眉，不是终点。
        if _is_mdna_title(h.title or h.key):
            continue
        if target.level != LEVEL_WEAK:
            if h.level != LEVEL_WEAK and h.level <= target.level:
                return h.pos, h
        else:
            # 无序号大标题下的一、（一）等都是内部条目，只在下一个真正大标题处结束。
            if h.level == LEVEL_CHAPTER:
                return h.pos, h
            if h.level == LEVEL_WEAK and _MAJOR_AFTER_MDNA.fullmatch(
                    _semantic_title(h.title or h.key)):
                return h.pos, h
    return text_len, None


def _content_cluster_fallback(heads: Sequence[Heading], text_len: int
                             ) -> Optional[Tuple[Heading, int, Optional[Heading], int]]:
    """标题文字损坏时，用 MD&A 特有的内部条目序列反推整段。至少命中三类。"""
    hits: List[Heading] = []
    seen = set()
    for h in heads:
        if h.in_toc or h.level == LEVEL_CHAPTER:
            continue
        t = _semantic_title(h.title or h.key)
        for idx, pat in enumerate(_MDna_INTERNAL):
            if pat.search(t):
                if idx not in seen:
                    seen.add(idx)
                    hits.append(h)
                break
    if len(hits) < 3:
        return None
    hits.sort(key=lambda x: x.pos)
    # 只保留第一组相对紧凑的条目簇，避免把目录/附录中的零散引用拼起来。
    cluster = [hits[0]]
    for h in hits[1:]:
        if h.pos - cluster[-1].pos <= 60000:
            cluster.append(h)
        elif len(cluster) >= 3:
            break
        else:
            cluster = [h]
    if len(cluster) < 3:
        return None
    first, last = cluster[0], cluster[-1]
    start_h = first
    # 回溯到最近的更高层级标题；它可能因 OCR 损坏而未命中标题别名。
    for h in heads:
        if h.in_toc or h.pos >= first.pos:
            break
        if first.pos - h.pos <= 5000 and h.level < first.level:
            start_h = h
    end, end_h = _mdna_boundary(heads, start_h, text_len)
    if end <= last.pos:
        end, end_h = text_len, None
    return start_h, end, end_h, 62


def locate_mdna(heads: Sequence[Heading], text_len: int) -> Optional[Dict]:
    candidates: List[Tuple[int, Heading, int, Optional[Heading]]] = []
    for h in heads:
        if h.in_toc:
            continue
        base = _mdna_title_score(h)
        if not base:
            continue
        end, end_h = _mdna_boundary(heads, h, text_len)
        length = end - h.pos
        score = base
        if length >= 1800:
            score += 18
        elif text_len > 50000 and length < 700:
            score -= 80
        if 0.03 * text_len <= h.pos <= 0.9 * text_len:
            score += 6
        score += min(length // 12000, 10)
        candidates.append((score, h, end, end_h))
    if candidates:
        score, h, end, end_h = max(candidates, key=lambda x: (x[0], x[2]-x[1].pos, -x[1].pos))
        return {"heading": h, "start": h.pos, "end": end, "end_heading": end_h,
                "score": score, "inferred": False}
    fb = _content_cluster_fallback(heads, text_len)
    if fb:
        h, end, end_h, score = fb
        return {"heading": h, "start": h.pos, "end": end, "end_heading": end_h,
                "score": score, "inferred": True}
    return None


# --------------------------------------------------------------------------
# 7. 主入口
# --------------------------------------------------------------------------
def _blank_section() -> Dict:
    return {"found": False, "status": SS.MISSING, "how": "未识别", "chars": 0,
            "loose": False, "text": "", "start": None, "end": None,
            "ai": False, "na": False, "na_reason": "", "score": 0,
            "origin": "", "parent_title": ""}


def slice_report(raw_text: str) -> Dict:
    """返回切片结果。

    result = {
      'normalized': 规整后的全文（后续所有偏移量都相对它）,
      'chapters':  [{'ordinal','title','start','end','chars'}...],
      'sections':  {key: {...}},
      'mdna_span': (start,end) | None,
      'notes':     [识别过程中的提示],
    }
    """
    text, lines, offsets, leaders, page_nos = normalize(raw_text)
    notes: List[str] = []
    result: Dict = {"normalized": text, "chapters": [], "sections": {},
                    "mdna_span": None, "notes": notes,
                    "toc_ranges": [], "heading_count": 0}
    if len(squeeze(text)) < 2000:
        notes.append("正文过短（可能是扫描件或抽取失败），跳过切片")
        for rule in SECTION_RULES:
            result["sections"][rule["key"]] = _blank_section()
        return result

    heads = build_headings(lines, offsets, page_nos)
    chap_heads = [h for h in heads if h.level == LEVEL_CHAPTER]
    if len(chap_heads) < 3:
        fb = fallback_chapter_scan(text)
        if len(fb) > len(chap_heads):
            notes.append(f"按行识别只找到 {len(chap_heads)} 个节标题，"
                         f"改用全文扫描兜底（找到 {len(fb)} 个），建议人工抽查")
            heads = [h for h in heads if h.level != LEVEL_CHAPTER] + fb
            heads.sort(key=lambda x: x.pos)

    # 通用标题树之外，再做一次专门的 MD&A 标题扫描，覆盖“第X部分/一、/无序号/跨行”等版式。
    by_pos = {h.pos: h for h in heads}
    for h in _extra_mdna_heads(lines, offsets, page_nos):
        old = by_pos.get(h.pos)
        if old is None or _mdna_title_score(h) > _mdna_title_score(old):
            by_pos[h.pos] = h
    heads = sorted(by_pos.values(), key=lambda x: x.pos)

    toc = detect_toc_ranges(text, lines, offsets, leaders, heads)
    mark_toc(heads, toc)
    result["toc_ranges"] = toc
    result["heading_count"] = len(heads)
    if toc:
        notes.append(f"识别到目录区 {len(toc)} 段，已从标题候选中剔除")

    chapters = select_body_chapters(heads, len(text))
    result["chapters"] = [{"ordinal": c["ordinal"], "title": c["title"],
                           "chars": c["chars"], "start": c["start"], "end": c["end"]}
                          for c in chapters]
    if not chapters:
        notes.append("未识别到任何「第X节」标题，年报版式异常")

    def cut(a: int, b: int) -> str:
        return text[max(0, a):max(0, b)].strip()

    sections: Dict[str, Dict] = {}

    # ---------- 7.1 管理层讨论与分析：任意层级语义定位 ----------
    mdna_hit = locate_mdna(heads, len(text))
    mdna_span: Optional[Tuple[int, int]] = None
    if mdna_hit:
        h = mdna_hit["heading"]
        start, end = mdna_hit["start"], mdna_hit["end"]
        end_h = mdna_hit.get("end_heading")
        mdna_span = (start, end)
        title_text = (h.marker + h.title) or h.key
        end_title = ((end_h.marker + end_h.title) if end_h else "")
        end_reason = f"至「{end_title[:40]}」之前" if end_h else "至全文结束"
        inferred = bool(mdna_hit.get("inferred"))
        how = ("依据 MD&A 内部条目簇反推边界" if inferred else
               f"语义标题命中：{title_text[:48]}")
        how += f"｜按标题层级完整切分，{end_reason}｜得分{mdna_hit['score']}"
        loose = inferred or (len(text) > 50000 and end - start < 1800)
        if loose:
            notes.append("管理层讨论与分析使用内容簇/短区间兜底，建议抽查边界")
        sections["mdna"] = {
            **_blank_section(), "found": True, "status": SS.ORIGINAL,
            "how": how, "chars": end - start, "text": cut(start, end),
            "start": start, "end": end, "score": mdna_hit["score"],
            "origin": "原文", "loose": loose,
            "start_heading": title_text, "end_heading": end_title,
            "start_page": h.page_no,
            "end_page": (end_h.page_no if end_h else (max(page_nos) if page_nos else -1)),
            "boundary_level": h.level,
            "boundary_inferred": inferred,
        }
    else:
        notes.append("未定位到管理层讨论与分析：标题别名与内部条目簇均未达到阈值")
        sections["mdna"] = _blank_section()

    result["mdna_span"] = mdna_span
    lo, hi = mdna_span if mdna_span else (0, len(text))

    # ---------- 7.2 其余章节 ----------
    for rule in SECTION_RULES:
        key = rule["key"]
        if key == "mdna":
            continue
        sec = _blank_section()

        # (a) 先按"第X节"找（旧版式里业务概要、核心竞争力是独立节）
        if rule["level"] in ("chapter", "chapter_then_item") and rule.get("chapter_patterns"):
            ch = _pick_chapter(chapters, rule)
            if ch:
                sec = {**sec, "found": True, "status": SS.ORIGINAL,
                       "how": f"第{ch['ordinal']}节 {ch['title']}（整节切出）",
                       "chars": ch["end"] - ch["start"],
                       "text": cut(ch["start"], ch["end"]),
                       "start": ch["start"], "end": ch["end"],
                       "score": 100, "origin": "原文"}

        # (b) 再按条目层级找（新版式并入管理层讨论与分析）
        if not sec["found"] and rule.get("item_patterns"):
            pats = rule["item_patterns"]
            finder = find_items_merged if rule.get("merge_adjacent") else find_item
            cand = finder(heads, lo, hi, pats)
            scope = "管理层讨论与分析内"
            if cand is None and mdna_span:
                cand = finder(heads, 0, len(text), pats)
                if cand and cand["loose"]:
                    cand = None                # 全文范围内只接受高分命中
                scope = "全文范围"
            if cand:
                start, end = cand["start"], cand["end"]
                body = cut(start, end)
                sec = {**sec, "found": True,
                       "status": SS.ORIGINAL,
                       "how": (("条目匹配：" if not cand["loose"] else "疑似条目（待复核）：")
                               + cand["title"] + f"｜{scope}｜得分{cand['score']}"),
                       "chars": end - start, "text": body,
                       "start": start, "end": end,
                       "loose": cand["loose"], "score": cand["score"],
                       "origin": "原文",
                       "parent_title": _parent_title(heads, cand["start"])}

                # (c)「不适用」判定
                if rule.get("check_na"):
                    is_na, why = detect_not_applicable(
                        body, cand["heading"].key if cand.get("heading") else "")
                    if is_na:
                        sec.update({"status": SS.NOT_APPLICABLE, "na": True,
                                    "na_reason": why, "loose": False,
                                    "how": f"{cand['title']}：公司填报「不适用」（{why}）"})
                        notes.append(f"「{rule['name']}」在本篇年报中为「不适用」（{why}）")
                elif end - start < rule.get("min_chars", 0):
                    sec["loose"] = True
                    sec["how"] += f"｜正文仅 {end - start} 字，偏短，待复核"

        if not sec["found"] and rule.get("check_na"):
            # 小节整体缺失，也要单独记录，别混进"未识别"里一笔带过
            notes.append(f"「{rule['name']}」小节在本篇年报中未出现（既非不适用，也无内容）")
        sections[key] = sec

    result["sections"] = sections
    result["headings"] = [{"pos": h.pos, "level": h.level, "in_toc": h.in_toc,
                           "page": h.page_no,
                           "title": (h.marker + h.title)[:60]} for h in heads]
    return result


def _pick_chapter(chapters: Sequence[Dict], rule: Dict) -> Optional[Dict]:
    """在已识别的节序列里挑最像目标章节的一节。"""
    pats = rule.get("chapter_patterns") or []
    expect = set(rule.get("expect_ordinals") or [])
    best, best_sc = None, 0
    for c in chapters:
        sc, _hit = score_title(c["title"], pats)
        if sc <= 0:
            continue
        if c["ordinal"] in expect:
            sc += 15
        sc += min(c["chars"] / 20000.0, 8)       # 正文越长越像正章而非引用
        if sc > best_sc:
            best, best_sc = c, sc
    return best


def _parent_title(heads: Sequence[Heading], pos: int) -> str:
    """往前找最近的一级条目标题（"四、主营业务分析"），便于人工核对位置。"""
    cur = ""
    for h in heads:
        if h.pos >= pos:
            break
        if h.level in (LEVEL_CN, LEVEL_CHAPTER) and not h.in_toc:
            cur = (h.marker + h.title)[:40]
    return cur


def next_heading_pos(headings: Sequence[Dict], pos: int,
                     max_level: int = LEVEL_AR_PAREN) -> Optional[int]:
    """给 verify 模块用：从 pos 往后找下一个标题行的偏移。

    headings 是 slice_report 返回的可序列化标题表（不是 Heading 对象）。
    """
    for h in headings:
        if h["pos"] > pos and h["level"] <= max_level and not h["in_toc"]:
            return h["pos"]
    return None
