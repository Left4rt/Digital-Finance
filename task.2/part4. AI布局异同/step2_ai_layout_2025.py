# -*- coding: utf-8 -*-
"""
Step 2 — 用第 3-4 步建成的 AI 金融技术词库（263 词 / 18 方向）回标 2025 年报「研发投入」
章节全文，得到每家公司 2025 年的 AI 技术布局向量。

与 step 5（2024 年报三章节）口径保持一致：
  权重 tier: CORE=1.0, EXT=0.5
  ai_depth   = Σ(命中次数 × tier权重)         —— 布局强度
  ai_breadth = 命中的技术方向数（仅 TECH 层）  —— 布局广度
  hhi        = Σ(方向份额²)                    —— 布局集中度
输出：out/ai_layout_2025.csv, out/ai_hits_2025_detail.csv
"""
import os, re, glob
import pandas as pd

OUT = "out"
os.makedirs(OUT, exist_ok=True)

lex = pd.read_csv("ai_fintech_lexicon.csv")
lex.columns = [c.strip().lstrip("\ufeff") for c in lex.columns]
lex["w"] = lex["tier"].map({"CORE": 1.0, "EXT": 0.5}).fillna(0.5)


def split_head(text):
    parts = text.split("-" * 30, 1)
    head, body = parts[0], (parts[1] if len(parts) > 1 else "")
    m = re.search(r"#\s*(.+?)（(\d{6}\.[A-Z]{2})）", head)
    return (m.group(1).strip() if m else ""), (m.group(2) if m else ""), body


rows, detail = [], []
for fp in sorted(glob.glob("rd2025/*.txt")):
    name, cid, body = split_head(open(fp, encoding="utf-8", errors="replace").read())
    if not cid:
        cid = os.path.basename(fp).split("_")[0]
    txt = body.replace("\n", "")          # 跨行断词还原（PDF 抽出的文本按版面折行）
    per_dir, total_hits = {}, 0
    for _, r in lex.iterrows():
        kw = str(r["keyword"])
        n = txt.count(kw)
        if n:
            total_hits += n
            key = (r["layer"], r["label"], r["direction_cn"])
            per_dir[key] = per_dir.get(key, 0) + n * r["w"]
            detail.append(dict(company_id=cid, company_name=name, keyword=kw,
                               label=r["label"], direction_cn=r["direction_cn"],
                               layer=r["layer"], tier=r["tier"], hits=n,
                               weighted=n * r["w"]))
    tech = {k: v for k, v in per_dir.items() if k[0] == "TECH"}
    prod = {k: v for k, v in per_dir.items() if k[0] == "PRODUCT"}
    depth = sum(tech.values())
    hhi = sum((v / depth) ** 2 for v in tech.values()) if depth else None
    top = sorted(tech.items(), key=lambda x: -x[1])[:3]
    rows.append(dict(
        company_id=cid, company_name=name,
        ai_kw_hits_2025=total_hits,
        ai_breadth_2025=len(tech),
        ai_depth_2025=round(depth, 3),
        ai_hhi_2025=round(hhi, 3) if hhi else None,
        ai_product_lines_2025=len(prod),
        primary_direction_2025=top[0][0][2] if top else "",
        top3_2025=" > ".join(f"{k[2]}({v:.1f})" for k, v in top),
        text_len=len(txt),
        ai_density_2025=round(total_hits / max(len(txt), 1) * 1000, 3),   # 每千字命中
        rd_text_type=("叙述型(含研发项目描述)" if ("主要研发项目" in txt or "项目目的" in txt)
                      else ("纯表格型(仅投入金额表)" if ("研发投入" in txt and len(txt) > 60)
                            else "未披露/极简")),
    ))

df = pd.DataFrame(rows).sort_values("ai_depth_2025", ascending=False)
df.to_csv(f"{OUT}/ai_layout_2025.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(detail).to_csv(f"{OUT}/ai_hits_2025_detail.csv", index=False,
                            encoding="utf-8-sig")
print(df.head(20).to_string(index=False))
print("\n有 AI 命中的公司:", (df.ai_kw_hits_2025 > 0).sum(), "/", len(df))
