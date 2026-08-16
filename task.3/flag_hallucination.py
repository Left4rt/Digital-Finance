# -*- coding: utf-8 -*-
"""
幻觉两阶段判定 —— 阶段1：自动初筛。
对模型输出的每一个"信息单元"（列表元素 + 每个非空标量），
在归一化后的招聘原文中做证据检索；检索不到的进入人工裁定队列。
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import parser as P
from normalize import norm, norm_text, evidence_in

samples = {r["sample_id"]: r for r in json.load(open("data/sample20.json", encoding="utf-8"))}
SRC = {k: norm_text(v["input_text"]) for k, v in samples.items()}

flagged = {}
for ver in ["v1", "v2", "v3"]:
    raw = json.load(open("outputs/run1_%s_raw.json" % ver, encoding="utf-8"))
    for sid, txt in raw.items():
        ji, meta = P.parse(txt)
        d = ji.to_dict()
        units = []
        for f in ["职位名称", "最低薪资", "最高薪资", "薪资周期", "工作地点", "学历要求", "经验要求"]:
            if d[f]:
                units.append((f, d[f]))
        for f in ["硬技能", "软技能", "AI相关技术栈要求"]:
            for x in d[f]:
                units.append((f, x))
        for f, u in units:
            if not evidence_in(u, SRC[sid], f):
                flagged.setdefault(sid, {}).setdefault(u, []).append("%s/%s" % (ver, f))

json.dump(flagged, open("eval/flagged_units.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
tot = sum(len(v) for v in flagged.values())
print("待人工裁定的信息单元数:", tot)
for sid in sorted(flagged):
    print("\n==", sid)
    for u, w in flagged[sid].items():
        print("   %-38s %s" % (u, ",".join(w)))
