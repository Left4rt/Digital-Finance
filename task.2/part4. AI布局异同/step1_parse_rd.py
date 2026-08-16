# -*- coding: utf-8 -*-
"""
Step 1 — 解析 58 家公司 2025 年报「研发投入」章节 txt，抽取结构化研发/财务字段。

两种版式：
  A 深市版：研发投入金额（元） / 研发投入占营业总收入比例 / 研发人员数量（人）（本年+上年两列）
  B 沪市版：本期费用化研发投入 / 本期资本化研发投入 / 研发投入合计 /
            研发投入总额占营业收入比例（%） / 公司研发人员的数量（仅本年一列）

输出：
  out/rd2025_parsed.csv        结构化表
  out/rd2025_parse_audit.csv   每字段命中的原始行（人工复核用）
"""
import os, re, glob
import pandas as pd

SRC, OUT = "rd2025", "out"
os.makedirs(OUT, exist_ok=True)
NUM = r"[-+]?\d[\d,]*\.?\d*"


def split_head(text):
    parts = text.split("-" * 30, 1)
    head, body = parts[0], (parts[1] if len(parts) > 1 else "")
    meta = {}
    for ln in head.splitlines():
        ln = ln.strip().lstrip("\ufeff")
        if ln.startswith("#") and "：" in ln:
            k, v = ln.lstrip("# ").split("：", 1)
            meta[k.strip()] = v.strip()
    m = re.search(r"#\s*(.+?)（(\d{6}\.[A-Z]{2})）", head)
    if m:
        meta["公司简称"], meta["company_id"] = m.group(1).strip(), m.group(2)
    return meta, body


def to_f(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def next_tokens(lines, i, want, kind, maxlook=8):
    """向下收集 want 个数值 token。kind='num' 跳过百分号行（那是变动比例）。"""
    got, snips = [], []
    for j in range(i + 1, min(i + 1 + maxlook, len(lines))):
        ln = lines[j].strip()
        if ln in ("", "-", "—", "--", "不适用", "/", "√适用", "□适用"):
            snips.append(ln or "空")
            continue
        if re.fullmatch(r"例[（(]%[)）]|单位：.*|本年度|上年度|变化幅度[（(]%[)）]", ln):
            snips.append("[标签续行]" + ln)
            continue
        if re.fullmatch(NUM + r"%", ln):
            if kind == "num":
                snips.append("[跳过]" + ln)
                continue
            got.append(ln); snips.append(ln)
        elif re.fullmatch(NUM, ln):
            got.append(ln); snips.append(ln)
        else:
            break
        if len(got) >= want:
            break
    return got, snips


PAT = {
    "rd_amount":     (r"^研发投入金额", "num", 2),
    "rd_total_B":    (r"^研发投入合计", "num", 1),
    "rd_expensed_B": (r"^(本期)?费用化研发投入", "num", 1),
    "rd_cap_B":      (r"^(本期)?资本化研发投入", "num", 1),
    "rd_ratio":      (r"^研发投入占营业(总)?收入(的)?比例|^研发投入总额占营业收入比", "pct", 2),
    "rd_cap_amt":    (r"^研发投入资本化的?金额", "num", 2),
    "rd_cap_ratio":  (r"^资本化研发投入占研发投入(的)?比例|^研发投入资本化的比重", "pct", 2),
    "rd_staff":      (r"^研发人员数量[（(]?人?[)）]?$|^公司研发人员的数量", "num", 2),
    "rd_staff_pct":  (r"^研发人员数量占比|^研发人员数量占公司总人数的比例", "pct", 2),
}


def parse_block(body):
    lines = [l.strip() for l in body.splitlines()]
    res, audit = {}, {}
    for key, (pat, kind, want) in PAT.items():
        for idx, s in enumerate(lines):
            if re.search(pat, s):
                vals, snips = next_tokens(lines, idx, want, kind)
                res[key + "_2025"] = to_f(vals[0]) if len(vals) > 0 else None
                if want > 1:
                    res[key + "_2024"] = to_f(vals[1]) if len(vals) > 1 else None
                res[key + "_unit"] = "万元" if "万元" in s else "元"
                audit[key] = s + " ||| " + " / ".join(snips)
                break
    return res, audit


def main():
    rows, audits = [], []
    for fp in sorted(glob.glob(os.path.join(SRC, "*.txt"))):
        text = open(fp, encoding="utf-8", errors="replace").read()
        meta, body = split_head(text)
        cid = meta.get("company_id") or os.path.basename(fp).split("_")[0]
        v, audit = parse_block(body)
        nature = meta.get("内容性质", "")

        fmt = "A_深市版" if v.get("rd_amount_2025") is not None else (
            "B_沪市版" if (v.get("rd_total_B_2025") is not None or
                          v.get("rd_expensed_B_2025") is not None) else "无投入表")

        amt25 = v.get("rd_amount_2025")
        if amt25 is None:
            amt25 = v.get("rd_total_B_2025")
            if amt25 is None and v.get("rd_expensed_B_2025") is not None:
                amt25 = (v.get("rd_expensed_B_2025") or 0) + (v.get("rd_cap_B_2025") or 0)
        wan = ("单位：万元" in body) or (v.get("rd_amount_unit") == "万元")
        if wan and amt25:
            amt25 *= 1e4
        amt24 = v.get("rd_amount_2024")
        if wan and amt24:
            amt24 *= 1e4

        na = ("不适用" in nature)
        rows.append(dict(
            company_id=cid, company_name=meta.get("公司简称", ""),
            section_nature=nature, rd_format=fmt, rd_disclosed=not na,
            rd_section_words=to_f(meta.get("字数", "")),
            rd_amount_2025=amt25, rd_amount_2024=amt24,
            rd_ratio_2025=v.get("rd_ratio_2025"), rd_ratio_2024=v.get("rd_ratio_2024"),
            rd_cap_amt_2025=(v.get("rd_cap_amt_2025") if v.get("rd_cap_amt_2025") is not None
                             else v.get("rd_cap_B_2025")),
            rd_cap_ratio_2025=v.get("rd_cap_ratio_2025"),
            rd_staff_2025=v.get("rd_staff_2025"), rd_staff_2024=v.get("rd_staff_2024"),
            rd_staff_pct_2025=v.get("rd_staff_pct_2025"),
            body_chars=len(body),
        ))
        a = {"company_id": cid, "company_name": meta.get("公司简称", "")}
        a.update(audit)
        audits.append(a)

    df = pd.DataFrame(rows)
    df["revenue_implied_2025"] = df.apply(
        lambda r: r.rd_amount_2025 / (r.rd_ratio_2025 / 100)
        if r.rd_amount_2025 and r.rd_ratio_2025 else None, axis=1)
    df["revenue_implied_2024"] = df.apply(
        lambda r: r.rd_amount_2024 / (r.rd_ratio_2024 / 100)
        if r.rd_amount_2024 and r.rd_ratio_2024 else None, axis=1)
    df["rd_growth_pct"] = df.apply(
        lambda r: (r.rd_amount_2025 / r.rd_amount_2024 - 1) * 100
        if r.rd_amount_2025 and r.rd_amount_2024 else None, axis=1)
    df["staff_total_implied_2025"] = df.apply(
        lambda r: r.rd_staff_2025 / (r.rd_staff_pct_2025 / 100)
        if r.rd_staff_2025 and r.rd_staff_pct_2025 else None, axis=1)

    # 合理性校验：上市公司营收落在 [0.5亿, 5000亿] 之外视为解析存疑
    df["parse_flag"] = df.revenue_implied_2025.apply(
        lambda x: "" if (x is None or pd.isna(x)) else
        ("" if 5e7 <= x <= 5e11 else "营收反推存疑"))
    df = df.sort_values("company_id").reset_index(drop=True)
    df.to_csv(f"{OUT}/rd2025_parsed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(f"{OUT}/rd2025_parse_audit.csv", index=False,
                                encoding="utf-8-sig")
    print(df[["company_id", "company_name", "rd_format", "rd_amount_2025",
              "rd_ratio_2025", "revenue_implied_2025", "rd_staff_2025"]].to_string())
    print("\n版式分布:\n", df.rd_format.value_counts().to_string())
    print("研发金额可得:", df.rd_amount_2025.notna().sum(),
          "| 强度可得:", df.rd_ratio_2025.notna().sum(),
          "| 反推营收可得:", df.revenue_implied_2025.notna().sum())


if __name__ == "__main__":
    main()
