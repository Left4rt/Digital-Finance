# -*- coding: utf-8 -*-
"""PDF → 带分页映射的文本。

策略（按速度排序，逐级降级）：
  1. PyMuPDF (fitz)：最快，若环境里有就用；
  2. pypdf：够快，覆盖绝大多数年报；
  3. pdfplumber：最慢但版面还原最好，用于前两者抽不出字的情况。

抽不出字 = 扫描件，标记 NO_TEXT，提示走 OCR，而不是当作空白年报静默通过。
"""
from __future__ import annotations

import re
from typing import Dict

_CJK = re.compile(r"[\u4e00-\u9fff]")


def _clean(text: str) -> str:
    if not text:
        return ""
    # 统一换行、去掉不可见控制字符与软连字符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "").replace("\x00", "")
    text = re.sub(r"[ \t\u3000]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _try_fitz(path: str):
    try:
        import fitz                                       # type: ignore
    except Exception:                                     # noqa: BLE001
        return None
    doc = fitz.open(path)
    pages = [p.get_text("text") or "" for p in doc]
    n = doc.page_count
    doc.close()
    return pages, n


def _try_pypdf(path: str):
    try:
        from pypdf import PdfReader
    except Exception:                                     # noqa: BLE001
        return None
    reader = PdfReader(path)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:                                 # noqa: BLE001
            pass
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:                                 # noqa: BLE001
            pages.append("")
    return pages, len(reader.pages)


def _try_pdfplumber(path: str, max_pages: int = 0):
    try:
        import pdfplumber
    except Exception:                                     # noqa: BLE001
        return None
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        limit = total if max_pages <= 0 else min(total, max_pages)
        for i in range(limit):
            try:
                pages.append(pdf.pages[i].extract_text() or "")
            except Exception:                             # noqa: BLE001
                pages.append("")
    return pages, total


def extract_text(path: str) -> Dict:
    """返回 {'ok','text','pages','engine','cjk_per_page','msg','has_page_map'}；页面间以 \f 分隔"""
    out = {"ok": False, "text": "", "pages": 0, "engine": "",
           "cjk_per_page": 0.0, "msg": "", "has_page_map": False}
    errors = []

    for name, fn in (("pymupdf", _try_fitz), ("pypdf", _try_pypdf),
                     ("pdfplumber", _try_pdfplumber)):
        try:
            res = fn(path)
        except Exception as e:                            # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if res is None:
            continue
        pages, total = res
        text = _clean("\n\f\n".join(pages))
        cjk = len(_CJK.findall(text))
        per_page = cjk / max(1, total)
        out.update({"text": text, "pages": total, "engine": name,
                    "cjk_per_page": round(per_page, 1),
                    "has_page_map": "\f" in text})
        # 每页平均中文字符 < 80 基本可以断定是扫描图片
        if per_page >= 80:
            out["ok"] = True
            return out
        errors.append(f"{name}: 每页仅 {per_page:.0f} 个中文字符")

    if out["pages"] == 0:
        out["msg"] = "PDF 无法打开或页面为 0；" + "；".join(errors[-2:])
    else:
        out["msg"] = "未检测到文本层（疑似扫描件，需 OCR）；" + "；".join(errors[-2:])
    return out
