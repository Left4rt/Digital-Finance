# -*- coding: utf-8 -*-
"""
主评测：字段级准确率 / 列表字段 P-R-F1 / 幻觉率 / JSON有效率 / Token与耗时估算
用法: python3 eval/evaluate.py run1
"""
import json, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser as P
from normalize import norm, norm_set, norm_text, evidence_in

RUN = sys.argv[1] if len(sys.argv) > 1 else "run1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

GT = json.load(open("data/ground_truth.json", encoding="utf-8"))
SAMPLES = {r["sample_id"]: r for r in json.load(open("data/sample20.json", encoding="utf-8"))}
ACCEPT = json.load(open("eval/accept_sets.json", encoding="utf-8"))
SUPPORT = json.load(open("eval/semantic_support.json", encoding="utf-8"))
PROMPT_TXT = {v: open("prompts/%s.txt" % f, encoding="utf-8").read() for v, f in
              [("v1", "v1_zeroshot"), ("v2", "v2_role_schema"), ("v3", "v3_fewshot_selfcheck")]}

SCALARS = ["职位名称", "最低薪资", "最高薪资", "薪资周期", "工作地点", "学历要求", "经验要求"]
LISTS = ["硬技能", "软技能", "AI相关技术栈要求"]
FIELDS = SCALARS + LISTS          # 10 个原子字段


def gt_flat(sid):
    g = GT[sid]
    return {
        "职位名称": g["职位名称"],
        "最低薪资": g["薪资范围"]["最低薪资"],
        "最高薪资": g["薪资范围"]["最高薪资"],
        "薪资周期": g["薪资范围"]["薪资周期"],
        "工作地点": g["工作地点"],
        "学历要求": g["学历要求"],
        "经验要求": g["经验要求"],
        "硬技能": g["技能要求"]["硬技能"],
        "软技能": g["技能要求"]["软技能"],
        "AI相关技术栈要求": g["AI相关技术栈要求"],
    }


def scalar_ok(sid, f, pred, gold):
    if gold is None:
        return pred is None
    if pred is None:
        return False
    acc = {norm(gold)} | {norm(x) for x in ACCEPT.get(sid, {}).get(f, [])}
    return norm(pred) in acc


def prf(pred, gold):
    p, g = norm_set(pred), norm_set(gold)
    tp = len(p & g); fp = len(p - g); fn = len(g - p)
    if not p and not g:
        return 1.0, 1.0, 1.0, 0, 0, 0
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    return pr, rc, f1, tp, fp, fn


def tokens(s):
    """粗略 token 估算：中文字符 ≈1 token，其余按 4 字符 ≈1 token（OpenAI/Claude 中文分词的常用近似）"""
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", s))
    other = len(s) - cjk
    return int(cjk + other / 4)


# 推理耗时估算模型（需在真实 API 上标定；此处参数取自公开的中等规模模型典型值）
T_OVERHEAD = 0.35      # 网络往返 + 排队，秒
PREFILL_TPS = 3000.0   # 输入 token 处理速率
DECODE_TPS = 42.0      # 输出 token 生成速率


def est_runtime(t_in, t_out):
    return T_OVERHEAD + t_in / PREFILL_TPS + t_out / DECODE_TPS


def evaluate(ver):
    raw = json.load(open("outputs/%s_%s_raw.json" % (RUN, ver), encoding="utf-8"))
    res = {"version": ver, "run": RUN, "per_sample": {}}
    n_strict = n_repaired = n_text = n_schema = 0
    fld_hit = {f: 0 for f in FIELDS}
    fld_hit_lenient = {f: 0 for f in FIELDS}
    agg = {f: [0, 0, 0] for f in LISTS}     # tp, fp, fn
    macro = {f: [] for f in LISTS}
    hal_units = 0; gen_units = 0; hal_samples = 0
    tin = tout = 0.0; rt = 0.0
    hal_detail = []

    for sid in sorted(raw):
        txt = raw[sid]
        src = norm_text(SAMPLES[sid]["input_text"])
        obj, lvl = P.try_json(txt)
        schema_ok = False
        if obj is not None:
            schema_ok, _ = P.validate_schema(obj)
        if lvl == "strict": n_strict += 1
        elif lvl == "repaired": n_repaired += 1
        else: n_text += 1
        if schema_ok: n_schema += 1

        ji, meta = P.parse(txt)
        pred = ji.to_dict()
        gold = gt_flat(sid)

        s_detail = {}
        for f in SCALARS:
            ok = scalar_ok(sid, f, pred[f], gold[f])
            fld_hit[f] += ok; fld_hit_lenient[f] += ok
            s_detail[f] = {"pred": pred[f], "gold": gold[f], "ok": ok}
        for f in LISTS:
            pr, rc, f1, tp, fp, fn = prf(pred[f], gold[f])
            agg[f][0] += tp; agg[f][1] += fp; agg[f][2] += fn
            macro[f].append(f1)
            exact = (norm_set(pred[f]) == norm_set(gold[f]))
            fld_hit[f] += exact
            fld_hit_lenient[f] += (f1 >= 0.8)
            s_detail[f] = {"P": round(pr, 4), "R": round(rc, 4), "F1": round(f1, 4),
                           "exact": exact, "FP项": sorted(norm_set(pred[f]) - norm_set(gold[f])),
                           "FN项": sorted(norm_set(gold[f]) - norm_set(pred[f]))}

        # ---- 幻觉 ----
        units = [(f, pred[f]) for f in SCALARS if pred[f]]
        for f in LISTS:
            units += [(f, x) for x in pred[f]]
        gen_units += len(units)
        sample_hal = 0
        for f, u in units:
            if evidence_in(u, src, f):
                continue
            v = SUPPORT.get(sid, {}).get(u)
            if v and v[0] == "SUPPORTED":
                continue
            sample_hal += 1
            hal_detail.append({"sample": sid, "field": f, "unit": u,
                               "reason": v[1] if v else "自动初筛未命中且无人工支持记录"})
        hal_units += sample_hal
        hal_samples += (sample_hal > 0)
        s_detail["_幻觉数"] = sample_hal
        s_detail["_解析层级"] = lvl or "text"
        s_detail["_schema合法"] = schema_ok
        res["per_sample"][sid] = s_detail

        ti = tokens(PROMPT_TXT[ver].replace("{job_description}", SAMPLES[sid]["input_text"]))
        to = tokens(txt)
        tin += ti; tout += to; rt += est_runtime(ti, to)

    n = len(raw)
    res["JSON有效率_严格"] = n_strict / n
    res["JSON有效率_修复后"] = (n_strict + n_repaired) / n
    res["Schema通过率"] = n_schema / n
    res["非JSON输出数"] = n_text
    res["字段级准确率_严格"] = sum(fld_hit.values()) / (n * len(FIELDS))
    res["字段级准确率_宽松"] = sum(fld_hit_lenient.values()) / (n * len(FIELDS))
    res["各字段准确率"] = {f: fld_hit[f] / n for f in FIELDS}
    res["列表字段指标"] = {}
    for f in LISTS:
        tp, fp, fn = agg[f]
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        res["列表字段指标"][f] = {"micro_P": round(pr, 4), "micro_R": round(rc, 4),
                                  "micro_F1": round(f1, 4),
                                  "macro_F1": round(sum(macro[f]) / len(macro[f]), 4),
                                  "TP": tp, "FP": fp, "FN": fn}
    res["幻觉率_单元级"] = hal_units / gen_units
    res["幻觉率_样本级"] = hal_samples / n
    res["生成信息单元总数"] = gen_units
    res["幻觉单元数"] = hal_units
    res["幻觉明细"] = hal_detail
    res["平均输入Token"] = tin / n
    res["平均输出Token"] = tout / n
    res["平均运行时间_估算s"] = rt / n
    return res


if __name__ == "__main__":
    allres = {}
    for v in ["v1", "v2", "v3"]:
        allres[v] = evaluate(v)
    json.dump(allres, open("eval/metrics_%s.json" % RUN, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("=" * 92)
    print("%-6s %10s %10s %10s %10s %10s %10s %10s" %
          ("版本", "JSON严格", "Schema", "准确率严格", "准确率宽松", "幻觉率单元", "幻觉率样本", "耗时估算s"))
    print("-" * 92)
    for v in ["v1", "v2", "v3"]:
        r = allres[v]
        print("%-6s %9.1f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%% %10.2f" % (
            v, r["JSON有效率_严格"] * 100, r["Schema通过率"] * 100,
            r["字段级准确率_严格"] * 100, r["字段级准确率_宽松"] * 100,
            r["幻觉率_单元级"] * 100, r["幻觉率_样本级"] * 100,
            r["平均运行时间_估算s"]))
    print("=" * 92)
    for v in ["v1", "v2", "v3"]:
        print("\n[%s] 列表字段 micro 指标" % v)
        for f, m in allres[v]["列表字段指标"].items():
            print("   %-14s P=%.3f R=%.3f F1=%.3f (macroF1=%.3f)  TP=%d FP=%d FN=%d"
                  % (f, m["micro_P"], m["micro_R"], m["micro_F1"], m["macro_F1"],
                     m["TP"], m["FP"], m["FN"]))
    print("\n各字段准确率：")
    print("%-18s %8s %8s %8s" % ("字段", "v1", "v2", "v3"))
    for f in FIELDS:
        print("%-18s %7.0f%% %7.0f%% %7.0f%%" % (
            f, allres["v1"]["各字段准确率"][f] * 100,
            allres["v2"]["各字段准确率"][f] * 100,
            allres["v3"]["各字段准确率"][f] * 100))
    print("\nToken 与耗时：")
    for v in ["v1", "v2", "v3"]:
        r = allres[v]
        print("  %s  输入%.0f tok  输出%.0f tok  估算%.2f s" %
              (v, r["平均输入Token"], r["平均输出Token"], r["平均运行时间_估算s"]))
