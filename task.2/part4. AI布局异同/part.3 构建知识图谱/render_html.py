# -*- coding: utf-8 -*-
"""把 build_kg.py 的产物注入 template.html, 生成可离线打开的交互式图谱"""
import json, pandas as pd
OUT = "/mnt/user-data/outputs"
d = json.load(open(f"{OUT}/kg_data.json", encoding="utf-8"))
lex = pd.read_csv(f"{OUT}/ai_fintech_lexicon.csv", encoding="utf-8-sig")
d["lexicon"] = lex[["keyword", "label", "direction_cn", "layer", "tier",
                    "hit_companies", "hit_sentences"]].to_dict("records")
tpl = open("template.html", encoding="utf-8").read()
html = tpl.replace("/*__DATA__*/", json.dumps(d, ensure_ascii=False, separators=(",", ":")))
open(f"{OUT}/ai_product_knowledge_graph.html", "w", encoding="utf-8").write(html)
print("wrote", len(html), "bytes")
