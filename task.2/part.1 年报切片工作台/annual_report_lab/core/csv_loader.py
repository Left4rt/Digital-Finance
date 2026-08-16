# -*- coding: utf-8 -*-
"""读取公司清单 CSV。

真实世界的清单文件编码非常杂（GBK / GB18030 / UTF-8 / UTF-8-BOM / UTF-16），
列名也不统一。这里做三件事：
1. 逐个候选编码试解码，失败再用 chardet 猜，最后兜底 latin-1（保证不抛异常）。
2. 自动识别分隔符（逗号 / 制表符 / 分号）。
3. 把各种列名归一到 ts_code / name，并把代码补齐成 6 位 + 交易所后缀。
"""
from __future__ import annotations

import csv
import io
import os
import re
from typing import List, Dict, Tuple

CANDIDATE_ENCODINGS = [
    "utf-8-sig", "utf-8", "gb18030", "gbk", "big5",
    "utf-16", "utf-16-le", "utf-16-be", "cp936", "latin-1",
]

CODE_ALIASES = {"ts_code", "code", "股票代码", "证券代码", "代码", "symbol",
                "stock_code", "secid", "ticker", "证券编码"}
NAME_ALIASES = {"name", "公司简称", "证券简称", "股票简称", "公司名称",
                "简称", "名称", "sec_name", "company"}


def sniff_decode(path: str) -> Tuple[str, str]:
    """返回 (文本, 实际使用的编码)。永不抛编码异常。"""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        return "", "utf-8"

    for enc in CANDIDATE_ENCODINGS[:-1]:
        try:
            text = raw.decode(enc)
            # utf-16 误判保护：解出大量 NUL 说明猜错了
            if "\x00" in text:
                continue
            return text, enc
        except (UnicodeDecodeError, LookupError):
            continue

    # chardet 兜底
    try:
        import chardet
        guess = chardet.detect(raw)
        if guess and guess.get("encoding"):
            return raw.decode(guess["encoding"], errors="replace"), guess["encoding"]
    except Exception:
        pass
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _sniff_dialect(sample: str) -> str:
    counts = {d: sample.count(d) for d in [",", "\t", ";", "|"]}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def normalize_code(code: str) -> str:
    """000001 / 000001.SZ / SZ000001 / sz.000001 → 000001.SZ"""
    if not code:
        return ""
    c = str(code).strip().upper().replace(" ", "")
    c = c.replace("SH.", "").replace("SZ.", "").replace("BJ.", "")
    m = re.search(r"(\d{6})", c)
    if not m:
        return ""
    digits = m.group(1)
    if ".SH" in c or c.startswith("SH"):
        suffix = "SH"
    elif ".SZ" in c or c.startswith("SZ"):
        suffix = "SZ"
    elif ".BJ" in c or c.startswith("BJ"):
        suffix = "BJ"
    else:
        if digits[0] == "6":
            suffix = "SH"
        elif digits[0] in ("0", "2", "3"):
            suffix = "SZ"
        elif digits[0] in ("4", "8", "9"):
            suffix = "BJ"
        else:
            suffix = "SZ"
    return f"{digits}.{suffix}"


def load_company_list(path: str) -> Dict:
    """返回 {'rows': [{'ts_code','name','raw_code'}], 'encoding':..., 'warnings':[...]}"""
    result = {"rows": [], "encoding": "", "warnings": [], "path": path}
    if not path or not os.path.isfile(path):
        result["warnings"].append(f"文件不存在：{path}")
        return result

    text, enc = sniff_decode(path)
    result["encoding"] = enc
    if not text.strip():
        result["warnings"].append("文件为空")
        return result

    delim = _sniff_dialect(text[:4000])
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = [r for r in reader if any((c or "").strip() for c in r)]
    if not rows:
        result["warnings"].append("没有解析到任何数据行")
        return result

    header = [(c or "").strip().lstrip("\ufeff").lower() for c in rows[0]]
    code_idx = name_idx = -1
    for i, h in enumerate(header):
        if code_idx < 0 and h in CODE_ALIASES:
            code_idx = i
        if name_idx < 0 and h in NAME_ALIASES:
            name_idx = i

    body = rows[1:]
    if code_idx < 0:
        # 没有表头：找第一列像 6 位代码的
        body = rows
        for i in range(len(rows[0])):
            if normalize_code(rows[0][i]):
                code_idx = i
                break
        if code_idx < 0:
            result["warnings"].append("未找到股票代码列（支持 ts_code / code / 股票代码 等）")
            return result
        name_idx = 1 if len(rows[0]) > 1 and code_idx != 1 else -1
        result["warnings"].append("未检测到表头，已按位置推断代码列")

    seen = set()
    for ln, r in enumerate(body, start=2):
        if code_idx >= len(r):
            continue
        raw_code = (r[code_idx] or "").strip()
        ts_code = normalize_code(raw_code)
        name = ""
        if 0 <= name_idx < len(r):
            name = (r[name_idx] or "").strip()
        if not ts_code:
            if raw_code:
                result["warnings"].append(f"第{ln}行代码无法识别：{raw_code!r}")
            continue
        if ts_code in seen:
            continue
        seen.add(ts_code)
        result["rows"].append({"ts_code": ts_code, "name": name, "raw_code": raw_code})

    if not result["rows"]:
        result["warnings"].append("没有解析出有效的股票代码")
    return result
