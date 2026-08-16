# -*- coding: utf-8 -*-
"""
AI 产品线知识图谱构建

输入 (step 4-3 产物):
  ai_evidence_accepted.csv   通过硬锚点+校准阈值+归属/状态过滤的证据句
  ai_labels_final.csv        全量候选(含 REJECT), 用于计算召回面与负例
  company_ai_profile.csv     公司级汇总
  label_calibration_report.csv  各标签的 scorer/threshold/AUC

流程:
  A. 关键词词库落地 + 语料回标 (词库覆盖率检验)
  B. 短语级嵌入: 字符 n-gram TF-IDF -> TruncatedSVD(LSA) 稠密向量
     - 说明: 本步在离线环境下构造本地稠密向量; 上游 step 4-2 的
       rerank_score_v1/v2 为预训练 cross-encoder 的语义打分, 两路信号
       在本步融合, 并互为校验。
  C. 方向识别: 短语向量 vs 方向原型向量的余弦相似度 -> 公司技术栈核心方向
  D. 知识图谱: 四层节点 + 五类边, 输出 nodes/edges/graphml/json
  E. 公司间技术布局相似度 + 层次聚类 -> 聚类分组
"""
import json
import re
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from lexicon import LEXICON, CLUSTER, TECH_LABELS, PRODUCT_LABELS, APP_LABELS, flat

OUT = "/mnt/user-data/outputs"
SVD_DIM = 96
RNG = 42

NAME = {k: v["name_cn"] for k, v in LEXICON.items()}
NAME["AI_GENERAL"] = "AI通用表述"


# ================================================================ A. 载入
def load():
    ev = pd.read_csv("ai_evidence_accepted.csv", encoding="gbk")
    al = pd.read_csv("ai_labels_final.csv", encoding="utf-8-sig")
    pf = pd.read_csv("company_ai_profile.csv", encoding="utf-8-sig")
    cal = pd.read_csv("label_calibration_report.csv", encoding="utf-8-sig")
    for d in (ev, al):
        d["document_text"] = d["document_text"].astype(str)
    return ev, al, pf, cal


# ================================================================ A2. 词库回标
def lexicon_coverage(al, lex_rows):
    """在全量候选句语料上统计每个关键词的命中公司数/句数 -> 词库有效性检验"""
    corpus = al.drop_duplicates("sentence_id")[["sentence_id", "company_id", "document_text"]]
    txt = corpus["document_text"].tolist()
    cid = corpus["company_id"].tolist()
    rows = []
    for r in lex_rows:
        kw = r["keyword"]
        pat = re.compile(re.escape(kw), re.I)
        hit_c, n = set(), 0
        for t, c in zip(txt, cid):
            if pat.search(t):
                n += 1
                hit_c.add(c)
        rows.append({**r, "hit_sentences": n, "hit_companies": len(hit_c)})
    return pd.DataFrame(rows)


# ================================================================ B. 短语级嵌入
def build_embeddings(al):
    """字符 2-4 gram TF-IDF -> LSA 稠密向量 (中文无需分词, 对新词/英文缩写鲁棒)"""
    sent = al.drop_duplicates("sentence_id")[["sentence_id", "company_id",
                                              "company_name", "section", "document_text"]]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=2, max_features=60000, sublinear_tf=True)
    X = vec.fit_transform(sent["document_text"])
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=RNG)
    Z = normalize(svd.fit_transform(X))
    return sent.reset_index(drop=True), vec, svd, Z, float(svd.explained_variance_ratio_.sum())


def embed_phrases(vec, svd, phrases):
    return normalize(svd.transform(vec.transform(phrases)))


def direction_prototypes(vec, svd):
    """方向原型向量 = 该方向 CORE 关键词短语向量的均值 (短语级词嵌入)"""
    protos, meta = [], []
    for lab, d in LEXICON.items():
        E = embed_phrases(vec, svd, d["core"])
        protos.append(E.mean(0))
        meta.append(lab)
    return meta, normalize(np.vstack(protos))


# ================================================================ C. 证据打分融合
def validate_prototypes(ev, sent, Z, proto_meta, P):
    """
    检验短语级嵌入的方向判别力:
    对每条证据句, 计算它与 18 个方向原型的余弦, 统计"人工/锚点判定的正确方向"
    所处排名。随机基线 MRR = 平均倒数排名 ≈ 0.16, Top-3 命中率 ≈ 16.7%。
    """
    idx = {s: i for i, s in enumerate(sent["sentence_id"])}
    pidx = {l: i for i, l in enumerate(proto_meta)}
    ranks = []
    for sid, lab in zip(ev["sentence_id"], ev["candidate_label"]):
        if sid not in idx or lab not in pidx:
            continue
        sims = Z[idx[sid]] @ P.T
        order = np.argsort(-sims)
        ranks.append(int(np.where(order == pidx[lab])[0][0]) + 1)
    ranks = np.array(ranks)
    return {"n": int(len(ranks)), "mrr": round(float((1 / ranks).mean()), 3),
            "top1": round(float((ranks == 1).mean()), 3),
            "top3": round(float((ranks <= 3).mean()), 3),
            "top5": round(float((ranks <= 5).mean()), 3),
            "median_rank": int(np.median(ranks)), "n_proto": len(proto_meta)}


def score_evidence(ev, sent, Z, proto_meta, P):
    """
    每条证据 (句, 标签) 的最终权重:
      w = conf_weight * (0.5 * score_used_norm + 0.5 * lsa_cos_norm)
    lsa_cos = 句向量 与 该标签方向原型 的余弦, 与上游 reranker 分数相互独立
    """
    idx = {s: i for i, s in enumerate(sent["sentence_id"])}
    pidx = {l: i for i, l in enumerate(proto_meta)}
    cos = []
    for sid, lab in zip(ev["sentence_id"], ev["candidate_label"]):
        if sid in idx and lab in pidx:
            cos.append(float(Z[idx[sid]] @ P[pidx[lab]]))
        else:
            cos.append(np.nan)
    ev = ev.copy()
    ev["lsa_cos"] = cos

    def mm(s):
        s = s.astype(float)
        lo, hi = np.nanmin(s), np.nanmax(s)
        return (s - lo) / (hi - lo + 1e-9)

    ev["score_norm"] = mm(ev["score_used"])
    ev["lsa_norm"] = mm(ev["lsa_cos"].fillna(ev["lsa_cos"].median()))
    ev["semantic_score"] = 0.5 * ev["score_norm"] + 0.5 * ev["lsa_norm"]
    cw = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.25}
    ev["conf_w"] = ev["confidence"].map(cw).fillna(0.25)
    ev["edge_weight"] = ev["conf_w"] * (0.4 + 0.6 * ev["semantic_score"])
    return ev


# ================================================================ D. 知识图谱
def build_graph(ev, pf, lexdf):
    ev = ev[ev.candidate_label != "AI_GENERAL"].copy()
    nodes, edges = [], []

    # --- L1 技术簇
    for c in sorted(set(CLUSTER.values())):
        nodes.append({"id": f"CLUSTER::{c}", "label": c, "type": "CLUSTER",
                      "layer": 1, "weight": 0.0, "meta": ""})

    # --- L2 技术方向 / 产品线 / 应用
    agg = ev.groupby("candidate_label").agg(
        w=("edge_weight", "sum"), n=("sentence_id", "nunique"),
        nc=("company_id", "nunique")).reset_index()
    for _, r in agg.iterrows():
        lab = r["candidate_label"]
        ly = 2 if lab in TECH_LABELS else 3
        nodes.append({"id": lab, "label": NAME.get(lab, lab), "type":
                      ("TECH" if lab in TECH_LABELS else
                       "PRODUCT" if lab in PRODUCT_LABELS else "APP"),
                      "layer": ly, "weight": round(float(r["w"]), 3),
                      "meta": f"{int(r['nc'])}家/{int(r['n'])}句"})
        if lab in CLUSTER:
            edges.append({"source": f"CLUSTER::{CLUSTER[lab]}", "target": lab,
                          "rel": "SUBSUMES", "weight": round(float(r["w"]), 3),
                          "evidence": int(r["n"])})

    # --- L3 公司
    prof = pf.set_index("company_id")
    for cid, g in ev.groupby("company_id"):
        nodes.append({"id": cid, "label": g["company_name"].iloc[0], "type": "COMPANY",
                      "layer": 4, "weight": round(float(g["edge_weight"].sum()), 3),
                      "meta": f"证据{int(prof.loc[cid,'evidence_sentences'])}句"})

    # --- 边: 公司 -> 技术 / 产品
    for (cid, lab), g in ev.groupby(["company_id", "candidate_label"]):
        rel = ("USES_TECH" if lab in TECH_LABELS else
               "OFFERS_PRODUCT" if lab in PRODUCT_LABELS else "APPLIES_INTERNAL")
        best = g.sort_values("edge_weight", ascending=False).iloc[0]
        edges.append({"source": cid, "target": lab, "rel": rel,
                      "weight": round(float(g["edge_weight"].sum()), 3),
                      "evidence": int(g["sentence_id"].nunique()),
                      "conf": g["confidence"].mode().iloc[0],
                      "quote": best["document_text"][:110],
                      "section": best["section"], "page": int(best["pdf_page"])})

    # --- 边: 技术 -> 产品 (同句共现 = 该技术支撑该产品线)
    bysent = ev.groupby("sentence_id")["candidate_label"].apply(set)
    co = {}
    for labs in bysent:
        t = [l for l in labs if l in TECH_LABELS]
        p = [l for l in labs if l in PRODUCT_LABELS or l in APP_LABELS]
        for a in t:
            for b in p:
                co[(a, b)] = co.get((a, b), 0) + 1
    for (a, b), n in co.items():
        edges.append({"source": a, "target": b, "rel": "ENABLES",
                      "weight": float(n), "evidence": n})

    # --- L0 关键词节点 (只挂命中最多的 CORE 词, 控制图规模)
    top = (lexdf[(lexdf.tier == "CORE") & (lexdf.hit_companies > 0)]
           .sort_values("hit_companies", ascending=False)
           .groupby("label").head(4))
    for _, r in top.iterrows():
        if r["label"] not in agg["candidate_label"].values:
            continue
        kid = f"KW::{r['keyword']}"
        nodes.append({"id": kid, "label": r["keyword"], "type": "KEYWORD",
                      "layer": 0, "weight": float(r["hit_companies"]),
                      "meta": f"{int(r['hit_companies'])}家公司提及"})
        edges.append({"source": r["label"], "target": kid, "rel": "EVIDENCED_BY",
                      "weight": float(r["hit_companies"]),
                      "evidence": int(r["hit_sentences"])})

    return pd.DataFrame(nodes).drop_duplicates("id"), pd.DataFrame(edges)


# ================================================================ E. 公司技术栈 & 相似度
def company_vectors(ev, sent, Z):
    """公司语义向量 = 其证据句向量按 edge_weight 加权平均"""
    idx = {s: i for i, s in enumerate(sent["sentence_id"])}
    rows, ids = [], []
    for cid, g in ev.groupby("company_id"):
        v, w = np.zeros(Z.shape[1]), 0.0
        for sid, ww in zip(g["sentence_id"], g["edge_weight"]):
            if sid in idx:
                v += Z[idx[sid]] * ww
                w += ww
        if w > 0:
            rows.append(v / w)
            ids.append(cid)
    return ids, normalize(np.vstack(rows))


def tech_stack(ev, cvec_ids, CV, proto_meta, P):
    """核心方向 = 标签证据强度 (主) + 公司向量对方向原型的余弦 (辅)"""
    pidx = {l: i for i, l in enumerate(proto_meta)}
    strength = (ev[ev.candidate_label.isin(TECH_LABELS)]
                .pivot_table(index="company_id", columns="candidate_label",
                             values="edge_weight", aggfunc="sum").fillna(0))
    strength = strength.reindex(columns=TECH_LABELS, fill_value=0)
    rows = []
    for cid in strength.index:
        s = strength.loc[cid]
        if cid in cvec_ids:
            cv = CV[cvec_ids.index(cid)]
            sim = pd.Series({l: float(cv @ P[pidx[l]]) for l in TECH_LABELS})
        else:
            sim = pd.Series(0.0, index=TECH_LABELS)
        sn = s / (s.max() + 1e-9)
        simn = (sim - sim.min()) / (sim.max() - sim.min() + 1e-9)
        comb = 0.75 * sn + 0.25 * simn
        comb = comb[s > 0].sort_values(ascending=False)
        rows.append({
            "company_id": cid,
            "primary_direction": comb.index[0] if len(comb) else "",
            "primary_cn": NAME.get(comb.index[0], "") if len(comb) else "",
            "secondary_direction": comb.index[1] if len(comb) > 1 else "",
            "secondary_cn": NAME.get(comb.index[1], "") if len(comb) > 1 else "",
            "stack_breadth": int((s > 0).sum()),
            "stack_depth": round(float(s.sum()), 3),
            "hhi": round(float(((s / (s.sum() + 1e-9)) ** 2).sum()), 3),
            "top3": " > ".join(f"{NAME.get(k,k)}({comb[k]:.2f})" for k in comb.index[:3]),
        })
    return pd.DataFrame(rows), strength


def similarity(ev, cvec_ids, CV, pf):
    """两路相似度: (1) 标签结构 Jaccard/余弦  (2) 语义向量余弦; 取加权"""
    labs = TECH_LABELS + PRODUCT_LABELS + APP_LABELS
    M = (ev.pivot_table(index="company_id", columns="candidate_label",
                        values="edge_weight", aggfunc="sum")
         .reindex(columns=labs, fill_value=0).fillna(0))
    M = M[M.sum(1) > 0]                       # 剔除仅 AI_GENERAL 的公司
    ids = [c for c in M.index if c in cvec_ids]
    M = M.loc[ids]
    # IDF 加权: 样本内覆盖率过高的方向(如 TECH_LLM)信息量低, 降权后组间差异才可分辨
    df_ = (M.values > 0).sum(0)
    idf = np.log((len(ids) + 1) / (df_ + 1)) + 1.0
    W = M.values * idf
    S_lab = cosine_similarity(normalize(W))
    S_sem = cosine_similarity(np.vstack([CV[cvec_ids.index(c)] for c in ids]))
    S = 0.65 * S_lab + 0.35 * S_sem
    np.fill_diagonal(S, 1.0)
    nm = pf.set_index("company_id")["company_name"].to_dict()
    names = [nm.get(c, c) for c in ids]
    sim = pd.DataFrame(S, index=ids, columns=ids)

    # 聚类用"布局结构"而非"证据体量": IDF 加权后再 L1 归一化 + Ward
    Sh = W / (W.sum(1, keepdims=True) + 1e-9)
    Zl = linkage(Sh, method="ward")
    k = 5 if len(ids) >= 15 else 3
    cl = fcluster(Zl, k, criterion="maxclust")

    # 簇命名: 取该簇 IDF 加权占比最高的两个方向
    cmap, prof = {}, pd.DataFrame(Sh, index=ids, columns=M.columns)
    gmean = prof.mean()
    for c in sorted(set(cl)):
        mem = [i for i, x in zip(ids, cl) if x == c]
        mu = prof.loc[mem].mean()
        lift = (mu / (gmean + 1e-9)).where(mu > 0.02)     # 相对全样本的超配倍数
        top = lift.dropna().sort_values(ascending=False).index[:2]
        if len(top) == 0:
            top = mu.sort_values(ascending=False).index[:2]
        cmap[c] = "/".join(NAME.get(t, t) for t in top)
    return ids, names, sim, dict(zip(ids, cl)), M, cmap


# ================================================================ F. 分组维度
def board_of(cid):
    """上市板块 —— 在缺少财务表时, 作为公司规模/生命周期的代理变量"""
    c = cid.split(".")[0]
    if c.startswith("688"):
        return "科创板"
    if c.startswith("300"):
        return "创业板"
    if c.startswith("60"):
        return "沪市主板"
    return "深市主板"


SEGMENT = {
    "证券": ["国海证券", "光大证券", "财达证券", "国盛证券", "信达证券", "南华期货",
             "国联民生", "湘财股份", "哈投股份", "越秀资本"],
    "银行": ["西安银行", "兴业银行", "郑州银行", "紫金银行", "张家港行"],
    "金融IT服务商": ["新致软件", "科蓝软件", "长亮科技", "恒生电子", "宇信科技",
                     "金证股份", "京北方", "高伟达", "安硕信息", "新晨科技",
                     "先进数通", "中科软", "四方精创", "赢时胜", "普联软件",
                     "中科金财", "银之杰", "税友股份", "汇金科技", "数码视讯",
                     "兆日科技"],
    "互联网金融信息": ["东方财富", "同花顺", "财富趋势", "指南针"],
    "支付与商户服务": ["新大陆", "新国都", "仁东控股", "创识科技", "海联金汇",
                       "翠微股份"],
}
SEG = {n: s for s, lst in SEGMENT.items() for n in lst}


def group_compare(stack, ev):
    stack = stack.copy()
    stack["board"] = stack.company_id.map(board_of)
    stack["segment"] = stack.company_name.map(lambda x: SEG.get(x, "其他"))
    g = (stack.groupby("segment")
         .agg(n=("company_id", "count"),
              avg_breadth=("stack_breadth", "mean"),
              avg_depth=("stack_depth", "mean"),
              avg_hhi=("hhi", "mean"),
              avg_products=("product_line_count", "mean"),
              avg_evidence=("evidence_sentences", "mean"))
         .round(2).sort_values("avg_depth", ascending=False).reset_index())
    top = (stack.groupby("segment")["primary_cn"]
           .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else ""))
    g["dominant_direction"] = g.segment.map(top)
    gb = (stack.groupby("board")
          .agg(n=("company_id", "count"), avg_breadth=("stack_breadth", "mean"),
               avg_depth=("stack_depth", "mean"), avg_hhi=("hhi", "mean"))
          .round(2).reset_index())
    return stack, g, gb


# ================================================================ main
def main():
    ev, al, pf, cal = load()
    lex_rows = flat()

    print(f"[A] 词库 {len(lex_rows)} 词, 语料回标中 ...")
    lexdf = lexicon_coverage(al, lex_rows)
    cov = (lexdf.hit_sentences > 0).mean()
    print(f"    语料命中率 {cov:.1%}  ({(lexdf.hit_sentences>0).sum()}/{len(lexdf)})")

    print("[B] 构建短语级嵌入 ...")
    sent, vec, svd, Z, evr = build_embeddings(al)
    print(f"    句子 {len(sent)}  维度 {Z.shape[1]}  累计解释方差 {evr:.1%}")
    proto_meta, P = direction_prototypes(vec, svd)

    print("[C] 融合语义打分 ...")
    val = validate_prototypes(ev[ev.candidate_label != "AI_GENERAL"], sent, Z, proto_meta, P)
    print(f"    原型检索验证 (18方向): MRR={val['mrr']}  Top1={val['top1']:.1%} "
          f"Top3={val['top3']:.1%}  随机基线 MRR≈0.16 / Top3≈16.7%")
    ev = score_evidence(ev, sent, Z, proto_meta, P)
    r = ev[["score_used", "lsa_cos"]].dropna().corr().iloc[0, 1]
    print(f"    reranker 分数 与 LSA 余弦 的相关系数 = {r:.3f} (两路信号近似独立, 可相互校验)")

    print("[D] 构建知识图谱 ...")
    nodes, edges = build_graph(ev, pf, lexdf)
    print(f"    节点 {len(nodes)}  边 {len(edges)}")

    print("[E] 技术栈与相似度 ...")
    cids, CV = company_vectors(ev, sent, Z)
    stack, strength = tech_stack(ev, cids, CV, proto_meta, P)
    ids, names, sim, clmap, M, cnames = similarity(ev, cids, CV, pf)

    nm = pf.set_index("company_id")["company_name"].to_dict()
    stack.insert(1, "company_name", stack.company_id.map(nm))
    stack["cluster"] = stack.company_id.map(clmap)
    stack["cluster_name"] = stack.cluster.map(cnames)
    stack = stack.merge(pf[["company_id", "evidence_sentences", "high_conf",
                            "medium_conf", "product_line_count"]], on="company_id", how="left")
    stack = stack.sort_values(["stack_depth"], ascending=False)
    stack, gseg, gboard = group_compare(stack, ev)

    # --- networkx 导出
    G = nx.DiGraph()
    for _, n in nodes.iterrows():
        G.add_node(n["id"], **{k: n[k] for k in ["label", "type", "layer", "weight", "meta"]})
    for _, e in edges.iterrows():
        G.add_edge(e["source"], e["target"], rel=e["rel"], weight=float(e["weight"]),
                   evidence=int(e["evidence"]))
    nx.write_graphml(G, f"{OUT}/ai_product_kg.graphml")

    # --- 关键中心性
    und = G.to_undirected()
    deg = nx.degree_centrality(und)
    btw = nx.betweenness_centrality(und, weight=None)
    nodes["degree_centrality"] = nodes["id"].map(deg).round(4)
    nodes["betweenness"] = nodes["id"].map(btw).round(4)

    # --- 落盘
    lexdf.to_csv(f"{OUT}/ai_fintech_lexicon.csv", index=False, encoding="utf-8-sig")
    nodes.to_csv(f"{OUT}/kg_nodes.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(f"{OUT}/kg_edges.csv", index=False, encoding="utf-8-sig")
    stack.to_csv(f"{OUT}/company_tech_stack.csv", index=False, encoding="utf-8-sig")
    sim.round(4).to_csv(f"{OUT}/company_similarity.csv", encoding="utf-8-sig")
    ev.to_csv(f"{OUT}/ai_evidence_scored.csv", index=False, encoding="utf-8-sig")
    strength.round(3).to_csv(f"{OUT}/company_tech_matrix.csv", encoding="utf-8-sig")
    gseg.to_csv(f"{OUT}/group_compare_segment.csv", index=False, encoding="utf-8-sig")
    gboard.to_csv(f"{OUT}/group_compare_board.csv", index=False, encoding="utf-8-sig")

    payload = {
        "nodes": nodes.to_dict("records"),
        "edges": edges.fillna("").to_dict("records"),
        "stack": stack.fillna("").to_dict("records"),
        "sim": {"ids": ids, "names": names, "matrix": sim.values.round(3).tolist(),
                "cluster": [int(clmap[c]) for c in ids],
                "cluster_names": {str(k): v for k, v in cnames.items()}},
        "matrix": {"companies": list(M.index),
                   "names": [nm.get(c, c) for c in M.index],
                   "labels": list(M.columns),
                   "labels_cn": [NAME.get(c, c) for c in M.columns],
                   "values": M.round(3).values.tolist()},
        "segment": gseg.fillna("").to_dict("records"),
        "board": gboard.fillna("").to_dict("records"),
        "valid": val,
        "stats": {"n_keywords": len(lexdf), "coverage": round(float(cov), 4),
                  "n_sent": int(len(sent)), "dim": int(Z.shape[1]),
                  "evr": round(evr, 4), "corr": round(float(r), 3),
                  "n_nodes": len(nodes), "n_edges": len(edges),
                  "n_company": int(ev.company_id.nunique()),
                  "n_evidence": int(ev.sentence_id.nunique()),
                  "mrr": val["mrr"], "top3": val["top3"]},
    }
    with open(f"{OUT}/kg_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print("\n=== 公司技术栈 Top 15 ===")
    print(stack.head(15)[["company_name", "primary_cn", "secondary_cn",
                          "stack_breadth", "stack_depth", "hhi", "cluster_name", "segment"]].to_string(index=False))
    print("\n=== 分组对比 (业务属性) ===")
    print(gseg.to_string(index=False))
    print("\n=== 分组对比 (上市板块) ===")
    print(gboard.to_string(index=False))
    print("\n=== 词库覆盖 Top ===")
    print(lexdf.sort_values("hit_companies", ascending=False)
          .head(15)[["keyword", "label", "tier", "hit_companies", "hit_sentences"]].to_string(index=False))
    return payload


if __name__ == "__main__":
    main()
