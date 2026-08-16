# -*- coding: utf-8 -*-
"""从原始年报 PDF 中裁出结构化章节。

优先使用 PyMuPDF 的 ``show_pdf_page`` 复制原页矢量内容，并在首尾页按标题 Y 坐标裁切；
因此段落排版、表格、图片、字体和线条都会保留。若无法精确找到标题坐标，则降级为
整页复制，仍保留版式，只可能在首尾页多带少量相邻内容。
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

_FULL2HALF = {ord(c): ord(c) - 0xFEE0 for c in
              "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
              "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"}
_MARKER = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]{1,4}(?:节|章|篇|部分)|"
    r"[一二三四五六七八九十]{1,3}[、.．]|[(（][一二三四五六七八九十]{1,3}[)）]|"
    r"\d{1,2}[、.．]|[(（]\d{1,2}[)）])"
)


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").translate(_FULL2HALF)).strip("：:。．")


def _variants(s: str) -> List[str]:
    raw = _compact(s)
    vals = [raw]
    stripped = _MARKER.sub("", raw).strip("：:。．")
    if stripped and stripped != raw:
        vals.append(stripped)
    # 最稳定的语义核心，适用于序号与标题被拆成两行的 PDF。
    for token in ("管理层讨论与分析", "管理层讨论及分析", "管理层讨论和分析",
                  "经营层讨论与分析", "经营情况讨论与分析", "董事会报告",
                  "公司治理", "重要事项", "财务报告", "环境和社会责任"):
        if token in raw:
            vals.append(token)
    return list(dict.fromkeys(x for x in vals if len(x) >= 4))


def _page_lines(page) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    out = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip():
                out.append((text, tuple(line.get("bbox", block.get("bbox")))))
    return out


def _candidate_pages(page_count: int, hint: int, lo: int = 0) -> Iterable[int]:
    if 0 <= hint < page_count:
        nearby = [p for d in range(0, 4) for p in (hint - d, hint + d)
                  if lo <= p < page_count]
        yielded = set()
        for p in nearby:
            if p not in yielded:
                yielded.add(p); yield p
        for p in range(lo, page_count):
            if p not in yielded:
                yield p
    else:
        yield from range(lo, page_count)


def _find_heading(doc, heading: str, hint: int = -1, lo: int = 0
                 ) -> Optional[Tuple[int, float, str, int]]:
    variants = _variants(heading)
    if not variants:
        return None
    hits = []
    for pno in _candidate_pages(doc.page_count, hint, lo):
        lines = _page_lines(doc[pno])
        probes = list(lines)
        # 标题序号与标题文字跨行时，合并相邻两行再比较；bbox 取第一行顶部到第二行底部。
        for i in range(len(lines) - 1):
            t1, b1 = lines[i]; t2, b2 = lines[i + 1]
            if abs(b2[1] - b1[3]) < 24:
                probes.append((t1 + t2, (min(b1[0], b2[0]), b1[1], max(b1[2], b2[2]), b2[3])))
        for text, bbox in probes:
            c = _compact(text)
            for v in variants:
                if c == v:
                    score = 120
                elif v in c and len(c) <= len(v) + 12:
                    score = 95
                elif v in c and len(c) <= len(v) + 28:
                    score = 72
                else:
                    continue
                # 目录行通常带长点线与页码，降低优先级；离文本定位页越近越好。
                if re.search(r"[.．·…‥]{3,}\d*$", c):
                    score -= 45
                if hint >= 0:
                    score -= min(abs(pno - hint) * 8, 40)
                else:
                    score -= min(pno, 8)  # 无提示时取最早的正文精确标题
                hits.append((score, pno, float(bbox[1]), text, len(c)))
        if hits and hint >= 0 and pno == hint:
            break
    if not hits:
        return None
    score, pno, y, text, _ = max(hits, key=lambda x: (x[0], -x[1], -x[2]))
    return pno, y, text, score


def _copy_with_fitz(source_pdf: str, output_pdf: str, start_heading: str,
                    end_heading: str, start_page: int, end_page: int) -> Dict:
    import fitz  # type: ignore
    src = fitz.open(source_pdf)
    try:
        sh = _find_heading(src, start_heading, start_page, 0)
        if sh:
            sp, sy, _, ss = sh
        else:
            sp = start_page if 0 <= start_page < src.page_count else 0
            sy, ss = 0.0, 0
        eh = _find_heading(src, end_heading, end_page, sp) if end_heading else None
        if eh:
            ep, ey, _, es = eh
        else:
            ep = end_page if sp <= end_page < src.page_count else src.page_count - 1
            ey, es = float(src[ep].rect.height), 0
        if ep < sp or (ep == sp and ey <= sy + 10):
            ep = max(sp, end_page if end_page >= sp else sp)
            ep = min(ep, src.page_count - 1)
            ey = float(src[ep].rect.height)
        out = fitz.open()
        copied = 0
        for pno in range(sp, ep + 1):
            page = src[pno]
            rect = page.rect
            top = max(0.0, sy - 4.0) if pno == sp else 0.0
            bottom = min(float(rect.height), ey - 2.0) if pno == ep else float(rect.height)
            if bottom - top < 12:
                continue
            clip = fitz.Rect(0, top, float(rect.width), bottom)
            newp = out.new_page(width=clip.width, height=clip.height)
            newp.show_pdf_page(newp.rect, src, pno, clip=clip)
            copied += 1
        if copied == 0:
            raise RuntimeError("结构化裁切未产生页面")
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        out.set_metadata({"title": os.path.basename(output_pdf),
                          "subject": "管理层讨论与分析结构化切片"})
        out.save(output_pdf, garbage=3, deflate=True)
        out.close()
        exact = bool(sh and (eh or not end_heading))
        return {"ok": True, "path": output_pdf,
                "mode": "精确坐标裁切" if exact else "整页边界降级",
                "start_page": sp, "end_page": ep,
                "start_y": round(sy, 2), "end_y": round(ey, 2),
                "start_score": ss, "end_score": es,
                "message": "保留原 PDF 的段落、表格、图片与矢量版式"}
    finally:
        src.close()


def _copy_with_pypdf(source_pdf: str, output_pdf: str,
                     start_page: int, end_page: int) -> Dict:
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(source_pdf)
    n = len(reader.pages)
    sp = start_page if 0 <= start_page < n else 0
    ep = end_page if sp <= end_page < n else n - 1
    writer = PdfWriter()
    for pno in range(sp, ep + 1):
        writer.add_page(reader.pages[pno])
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with open(output_pdf, "wb") as f:
        writer.write(f)
    return {"ok": True, "path": output_pdf, "mode": "整页复制降级",
            "start_page": sp, "end_page": ep,
            "message": "已保留原 PDF 结构；首尾页可能带少量相邻内容"}


def _int_value(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def save_pdf_section(source_pdf: str, output_pdf: str, section: Dict) -> Dict:
    if not source_pdf or not os.path.isfile(source_pdf):
        return {"ok": False, "path": "", "mode": "", "message": "原始 PDF 不存在"}
    try:
        return _copy_with_fitz(
            source_pdf, output_pdf,
            str(section.get("start_heading") or ""),
            str(section.get("end_heading") or ""),
            _int_value(section.get("start_page", -1)),
            _int_value(section.get("end_page", -1)),
        )
    except Exception as fitz_error:  # noqa: BLE001
        try:
            out = _copy_with_pypdf(
                source_pdf, output_pdf,
                _int_value(section.get("start_page", -1)),
                _int_value(section.get("end_page", -1)),
            )
            out["message"] += f"；PyMuPDF 精确裁切失败：{type(fitz_error).__name__}"
            return out
        except Exception as pypdf_error:  # noqa: BLE001
            return {"ok": False, "path": "", "mode": "",
                    "message": f"结构化 PDF 生成失败：{fitz_error}；{pypdf_error}"}
