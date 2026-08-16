# -*- coding: utf-8 -*-
"""
embed_rank.py — 用原型句 embedding 对候选句表做语义确认与打分。
实现词库「评分与验证」工作表的口径：
  · 候选召回： score = max(与该标签 POSITIVE 原型的余弦)；阈值 0.72（高误报词 0.78）
  · 类别判定： 正向原型相似度 − 难负例相似度 ≥ 0.08（margin）才保留该标签
  · 句内聚合： label_score = max(rule_confidence, 语义相似度)，按 sentence_id+label 去重
  · 证据权重： 状态权重 × 章节权重（否定/背景/他者 = 0）
  · AI_GENERAL： 只做召回锚点，技术得分 = 0，不自动映射到 TECH_ML

用法：
  1) 本地模型（推荐中文）：
     pip install sentence-transformers pandas numpy
     python embed_rank.py --backend st --model BAAI/bge-large-zh-v1.5 \
            --candidates candidate_sentences_v2.csv --prototypes lexicon_prototypes.csv \
            --out candidate_sentences_scored.csv
  2) OpenAI 接口：
     pip install openai pandas numpy ; export OPENAI_API_KEY=...
     python embed_rank.py --backend openai --model text-embedding-3-large ...
  3) 逻辑自测（无需网络/模型，向量为假）：
     python embed_rank.py --backend fake ...
"""
import argparse, csv, json, hashlib
import numpy as np

# ---- 权重（来自「评分与验证」）----
STATUS_WEIGHT = {'NEGATION':0.0,'BACKGROUND':0.0,'OTHER_ENTITY':0.0,
                 'ATTENTION':0.1,'PLANNED':0.2,'R_AND_D':0.5,'PILOT':0.7,
                 'APPLIED':1.0,'COMMERCIAL':1.2,'':0.5,'WEAK_GENERIC':0.4,'NON_FINANCIAL':0.3}
SECTION_WEIGHT = {'主要产品':1.0,'主要业务':1.0,'业务概要':1.0,'核心竞争力':1.0,
                  '研发投入':0.9,'在研项目':0.9,'经营情况讨论与分析':0.7,'经营讨论':0.7,'行业':0.3}
BASE_THRESHOLD = 0.72
HIGH_FP_THRESHOLD = 0.78
HIGH_FP_LABELS = {'AI_GENERAL'}          # 弱/高误报标签用更高阈值
MARGIN = 0.08                            # 正向 − 难负例 的最小差值

# ---------------------------------------------------------------- encoders
class STEncoder:
    def __init__(self, model): 
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)
    def encode(self, texts):
        v = self.m.encode(texts, normalize_embeddings=True, batch_size=64,
                           show_progress_bar=True)
        return np.asarray(v, dtype=np.float32)

class OpenAIEncoder:
    def __init__(self, model):
        from openai import OpenAI
        self.c = OpenAI(); self.model = model
    def encode(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            chunk = texts[i:i+256]
            r = self.c.embeddings.create(model=self.model, input=chunk)
            out += [d.embedding for d in r.data]
        v = np.asarray(out, dtype=np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v

class FakeEncoder:
    """确定性假向量，仅用于跑通流程与自测，不代表真实语义。"""
    D = 64
    def encode(self, texts):
        v = np.zeros((len(texts), self.D), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in set(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[i, h % self.D] += 1.0
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v

def get_encoder(backend, model):
    if backend == 'st':     return STEncoder(model)
    if backend == 'openai': return OpenAIEncoder(model)
    return FakeEncoder()

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='fake', choices=['st','openai','fake'])
    ap.add_argument('--model', default='BAAI/bge-large-zh-v1.5')
    ap.add_argument('--candidates', default='candidate_sentences_v2.csv')
    ap.add_argument('--prototypes', default='lexicon_prototypes.csv')
    ap.add_argument('--out', default='candidate_sentences_scored.csv')
    ap.add_argument('--semantic_recall', action='store_true',
                    help='对所有标签做纯语义召回，捕捉规则漏掉的候选（更高召回）')
    args = ap.parse_args()

    enc = get_encoder(args.backend, args.model)

    # ---- 原型库 ----
    protos = list(csv.DictReader(open(args.prototypes, encoding='utf-8-sig')))
    pos_by, neg_by = {}, {}
    for p in protos:
        (pos_by if p['sample_type']=='POSITIVE' else neg_by).setdefault(p['label'], []).append(p['prototype_text'])
    labels = sorted(pos_by)
    proto_texts, index = [], {}
    for lab in labels:
        for t in pos_by.get(lab, []): index.setdefault(('P',lab),[]).append(len(proto_texts)); proto_texts.append(t)
        for t in neg_by.get(lab, []): index.setdefault(('N',lab),[]).append(len(proto_texts)); proto_texts.append(t)
    P = enc.encode(proto_texts)

    # ---- 候选句 ----
    rows = list(csv.DictReader(open(args.candidates, encoding='utf-8-sig')))
    S = enc.encode([r['sentence_text'] for r in rows])

    def thr(lab): return HIGH_FP_THRESHOLD if lab in HIGH_FP_LABELS else BASE_THRESHOLD
    def sim_bank(svec, key):
        idx = index.get(key)
        if not idx: return 0.0
        return float(np.max(P[idx] @ svec))

    out_rows = []
    for i, r in enumerate(rows):
        svec = S[i]
        rule_labels = [l for l in r['candidate_labels'].split('; ') if l]
        scan = set(rule_labels) | (set(labels) if args.semantic_recall else set())

        confirmed, per_label = [], {}
        max_pos = 0.0
        for lab in scan:
            pos = sim_bank(svec, ('P',lab)); neg = sim_bank(svec, ('N',lab))
            per_label[lab] = {'pos':round(pos,4),'neg':round(neg,4),'margin':round(pos-neg,4)}
            max_pos = max(max_pos, pos)
            if pos >= thr(lab) and (pos-neg) >= MARGIN:
                confirmed.append(lab)

        # 句内聚合：label_score = max(rule_confidence, 语义相似度)
        rule_conf = float(r['rule_confidence'] or 0)
        tech_labels = [l for l in confirmed if l.startswith('TECH_') or l.startswith('PRODUCT_')
                       or l in ('AI_ENGINEERING','APP_INTERNAL')]  # AI_GENERAL 技术得分=0
        label_score = max([rule_conf] + [per_label[l]['pos'] for l in tech_labels]) if tech_labels else rule_conf

        sw = STATUS_WEIGHT.get(r['status_code'], 0.5)
        cw = SECTION_WEIGHT.get(r['section'], 0.7)
        tech_evidence = round(label_score * sw * cw, 4) if tech_labels else 0.0

        r['max_embedding_similarity'] = round(max_pos, 4)
        r['embedding_confirmed_labels'] = '; '.join(sorted(confirmed))
        r['embedding_scores'] = json.dumps(per_label, ensure_ascii=False)
        r['evidence_weight'] = round(sw*cw, 3)
        r['tech_evidence_score'] = tech_evidence
        # 语义确认后的候选：规则标签里至少一个通过 embedding，或纯语义召回命中
        r['is_candidate_semantic'] = 1 if confirmed else 0
        out_rows.append(r)

    fields = list(rows[0].keys())
    with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)

    conf = sum(1 for r in out_rows if r['is_candidate_semantic'])
    print(f'backend={args.backend} model={args.model}')
    print(f'candidates={len(out_rows)} | embedding-confirmed={conf}')
    print(f'wrote -> {args.out}')

if __name__ == '__main__':
    main()
