# -*- coding: utf-8 -*-
"""重复运行稳定性：输出一致率 + 字段级波动定位"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser as P
from normalize import norm, norm_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SUB = ["J01", "J03", "J04", "J06", "J09", "J13", "J17", "J19"]
RUNS = ["run1", "run2", "run3"]
SCALARS = ["职位名称", "最低薪资", "最高薪资", "薪资周期", "工作地点", "学历要求", "经验要求"]
LISTS = ["硬技能", "软技能", "AI相关技术栈要求"]
FIELDS = SCALARS + LISTS


def sig(val, is_list):
    return tuple(sorted(norm_set(val))) if is_list else norm(val)


out = {}
print("%-6s %12s %12s %s" % ("版本", "一致字段", "输出一致率", "波动字段"))
print("-" * 78)
for ver in ["v1", "v2", "v3"]:
    parsed = {}
    for r in RUNS:
        raw = json.load(open("outputs/%s_%s_raw.json" % (r, ver), encoding="utf-8"))
        for sid in SUB:
            parsed[(r, sid)] = P.parse(raw[sid])[0].to_dict()
    same = 0
    drift = []
    perfield = {f: 0 for f in FIELDS}
    for sid in SUB:
        for f in FIELDS:
            vals = {sig(parsed[(r, sid)][f], f in LISTS) for r in RUNS}
            if len(vals) == 1:
                same += 1
                perfield[f] += 1
            else:
                drift.append("%s.%s" % (sid, f))
    total = len(SUB) * len(FIELDS)
    out[ver] = {"一致字段数": same, "总字段数": total, "输出一致率": same / total,
                "波动字段": drift, "各字段一致数": perfield}
    print("%-6s %6d/%-5d %11.1f%%  %s" % (ver, same, total, same / total * 100,
                                          ", ".join(drift) if drift else "—"))

print("\n各字段一致率（%d条子样本，3次重复）:" % len(SUB))
print("%-18s %6s %6s %6s" % ("字段", "v1", "v2", "v3"))
for f in FIELDS:
    print("%-18s %5.0f%% %5.0f%% %5.0f%%" % (
        f, out["v1"]["各字段一致数"][f] / len(SUB) * 100,
        out["v2"]["各字段一致数"][f] / len(SUB) * 100,
        out["v3"]["各字段一致数"][f] / len(SUB) * 100))

json.dump(out, open("eval/consistency.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
