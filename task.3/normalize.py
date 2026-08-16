# -*- coding: utf-8 -*-
"""归一化：消除大小写/全半角/分隔符/虚词差异，再套用同义别名类。"""
import re, unicodedata

# 需要在比较时抹掉的连接虚词与标点
_DROP = re.compile(r"[\s\-_·・.。,，、；;:：/／\\|｜()（）\[\]【】“”\"'’‘*#>+~!！?？]")
_STOP = re.compile(r"[的与和及等]")

# 语义等价类：key 为归一化后的写法，value 为统一代表
ALIAS = {
    "js": "javascript",
    "oc": "objectivec",
    "rn": "reactnative",
    "reactnative": "reactnative",
    "nlp": "自然语言处理",
    "自然语言处理算法": "自然语言处理",
    "cv": "计算机视觉",
    "计算机视觉cv": "计算机视觉",
    "mvc开发": "mvc",
    "算法工程化经验": "算法工程化",
    "银行信贷": "银行信贷业务",
    "信贷业务": "银行信贷业务",
    "开户": "开户业务",
    "开户流程": "开户业务",
    "渠道合作": "外部渠道合作",
    "高净值客户销售": "对高净值客户销售",
    "个人客户销售": "对普通个人客户销售",
    "对个人客户销售": "对普通个人客户销售",
    "机构销售": "对机构销售",
    "webfront": "web前端开发",
    "web前端": "web前端开发",
    "前端工程化模块化": "前端工程化",
    "数仓建模": "数据仓库建模",
    "数据仓库数仓建模": "数据仓库建模",
    "特征工程": "特征分析",
    "英文文献阅读": "中英文文献阅读能力",
    "中英文文献阅读": "中英文文献阅读能力",
    "独立完成任务能力": "独立完成任务能力",
    "分析解决问题能力": "分析解决问题能力",
    "问题分析解决能力": "问题发现分析解决能力",
    "问题发现分析解决能力": "问题发现分析解决能力",
    "跨部门沟通协作": "跨部门沟通协作",
    "严谨细致": "工作严谨负责",
    "工作严谨负责": "工作严谨负责",
    "sqlserver": "sqlserver",
    "websphere": "websphere",
    "vugdraggable": "vuedraggable",
    "本科以上": "本科及以上",
    "本科及以上学历": "本科及以上",
    "本科含以上": "本科及以上",
    "2年以上": "两年以上",
    "两年以上": "两年以上",
}


def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = _DROP.sub("", s)
    s = _STOP.sub("", s)
    return ALIAS.get(s, s)


def norm_set(items):
    return {norm(x) for x in (items or []) if norm(x)}


def norm_text(t: str) -> str:
    """用于幻觉证据检索的原文归一化（与 norm 同规则，保留连续性）"""
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = _DROP.sub("", t)
    t = _STOP.sub("", t)
    return t


def norm_raw(s: str) -> str:
    """不套用别名类的纯字面归一化——专用于'原文证据检索'，
    避免把模型输出映射成原文中并不存在的代表词。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = _DROP.sub("", s)
    s = _STOP.sub("", s)
    return s


def evidence_in(unit: str, src_norm: str, field: str = "") -> bool:
    """字面证据检索：命中任一形式即视为有原文依据。"""
    cands = {norm_raw(unit), norm(unit)}
    if field in ("最低薪资", "最高薪资"):
        d = re.sub(r"[^0-9.]", "", str(unit))      # 薪资只比对数值部分
        if d:
            cands.add(d)
    if field == "薪资周期":
        cands |= {norm_raw(str(unit) + "薪"), norm_raw("元/" + str(unit)), norm_raw(str(unit) + "/")}
    return any(c and c in src_norm for c in cands)
