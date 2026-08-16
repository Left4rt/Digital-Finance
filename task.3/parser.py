# -*- coding: utf-8 -*-
"""
解析层：LLM 原始输出 -> 统一 JobInfo 结构

三级解析策略：
  L0 严格 json.loads          -> 记为 strict_valid
  L1 修复后解析（去代码围栏、截取首尾花括号、去尾逗号） -> 记为 repaired_valid
  L2 自由文本启发式解析（专为 v1.0 这类无格式约束的输出兜底） -> 记为 text_parsed
全部失败 -> parse_failed
"""
import json, re
from dataclasses import dataclass, field, asdict
from typing import Optional, List

NULL_TOKENS = {
    "", "无", "未提及", "未提供", "未说明", "未明确", "未明确说明", "未明确提及",
    "招聘信息中未提及", "招聘信息中未提供", "招聘信息未提及", "n/a", "na", "none",
    "null", "-", "—", "不详", "未知", "无明确要求", "未提及（一般要求本科及以上）",
}


@dataclass
class JobInfo:
    """结构化 Schema（等价于 Pydantic BaseModel，避免引入外部依赖）"""
    职位名称: Optional[str] = None
    最低薪资: Optional[str] = None
    最高薪资: Optional[str] = None
    薪资周期: Optional[str] = None
    工作地点: Optional[str] = None
    硬技能: List[str] = field(default_factory=list)
    软技能: List[str] = field(default_factory=list)
    学历要求: Optional[str] = None
    经验要求: Optional[str] = None
    AI相关技术栈要求: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


SCALAR_FIELDS = ["职位名称", "最低薪资", "最高薪资", "薪资周期",
                 "工作地点", "学历要求", "经验要求"]
LIST_FIELDS = ["硬技能", "软技能", "AI相关技术栈要求"]
ALL_FIELDS = SCALAR_FIELDS[:1] + SCALAR_FIELDS[1:4] + ["工作地点", "硬技能", "软技能",
                                                       "学历要求", "经验要求",
                                                       "AI相关技术栈要求"]


def _clean_scalar(v):
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        v = str(v)
    s = str(v).strip().strip("。;；").strip()
    if s.lower() in NULL_TOKENS or s in NULL_TOKENS:
        return None
    # 去掉"（一般要求…）"这类模型自加的括注前的 null 判定
    if re.match(r"^(未提及|未提供|未说明|无)", s):
        return None
    return s


def _clean_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        s = _clean_scalar(v)
        if s is None:
            return []
        parts = re.split(r"[、,，;；/|]\s*|\s{2,}", s)
    elif isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, str):
                parts.append(x)
            else:
                parts.append(str(x))
    else:
        parts = [str(v)]
    out, seen = [], set()
    for p in parts:
        p = p.strip().strip("。.;；·-—*").strip()
        p = re.sub(r"^\d+[\.、]\s*", "", p)
        if not p or p.lower() in NULL_TOKENS:
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


# ---------------- L0 / L1 ----------------
def try_json(raw: str):
    """返回 (obj, level)  level in {'strict','repaired',None}"""
    try:
        return json.loads(raw), "strict"
    except Exception:
        pass
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        s2 = s[i:j + 1]
        s2 = re.sub(r",(\s*[}\]])", r"\1", s2)          # 尾逗号
        s2 = s2.replace("“", '"').replace("”", '"')      # 中文引号
        try:
            return json.loads(s2), "repaired"
        except Exception:
            pass
    return None, None


def from_obj(obj) -> JobInfo:
    """把（可能字段名/嵌套不规范的）dict 映射到 Schema"""
    g = lambda *ks: next((obj[k] for k in ks if isinstance(obj, dict) and k in obj), None)
    sal = g("薪资范围", "薪资")
    lo = hi = pc = None
    if isinstance(sal, dict):
        lo, hi, pc = sal.get("最低薪资"), sal.get("最高薪资"), sal.get("薪资周期")
    elif isinstance(sal, str):
        lo, hi, pc = parse_salary_text(sal)
    skills = g("技能要求", "技能")
    hard = soft = None
    if isinstance(skills, dict):
        hard, soft = skills.get("硬技能"), skills.get("软技能")
    else:
        hard, soft = g("硬技能"), g("软技能")
    return JobInfo(
        职位名称=_clean_scalar(g("职位名称", "岗位名称")),
        最低薪资=_clean_scalar(lo), 最高薪资=_clean_scalar(hi), 薪资周期=_clean_scalar(pc),
        工作地点=_clean_scalar(g("工作地点", "地点")),
        硬技能=_clean_list(hard), 软技能=_clean_list(soft),
        学历要求=_clean_scalar(g("学历要求", "学历")),
        经验要求=_clean_scalar(g("经验要求", "经验")),
        AI相关技术栈要求=_clean_list(g("AI相关技术栈要求", "AI相关技术栈", "AI技术栈")),
    )


SAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(K|k|千|万|元)?\s*[-~－—到至]\s*(\d+(?:\.\d+)?)\s*(K|k|千|万|元)?")
PERIOD_RE = re.compile(r"(1[0-9]薪|/\s*月|每月|元/月|月薪|/\s*年|年薪|/\s*天|元/天|日薪)")


def parse_salary_text(s):
    if s is None:
        return None, None, None
    s = str(s)
    if _clean_scalar(s) is None:
        return None, None, None
    m = SAL_RE.search(s)
    lo = hi = None
    if m:
        u = m.group(4) or m.group(2) or ""
        lo = m.group(1) + u
        hi = m.group(3) + u
    p = PERIOD_RE.search(s)
    period = None
    if p:
        t = p.group(1)
        period = ("月" if "月" in t else "年" if "年" in t else
                  "日" if ("天" in t or "日" in t) else t)
    return lo, hi, period


# ---------------- L2 自由文本解析 ----------------
LABELS = {
    "职位名称": r"职位名称|岗位名称",
    "薪资范围": r"薪资范围|薪资待遇|薪资",
    "工作地点": r"工作地点|地点",
    "硬技能": r"硬技能",
    "软技能": r"软技能",
    "学历要求": r"学历要求|学历",
    "经验要求": r"经验要求|工作经验|经验",
    "AI相关技术栈要求": r"AI\s*相关技术栈要求|AI\s*相关技术栈|AI\s*技术栈",
}


def _strip_md(line):
    line = re.sub(r"^\s*[\|\-\*\+•>]+\s*", "", line)
    line = re.sub(r"^\s*\d+[\.、\)]\s*", "", line)
    line = line.replace("**", "").replace("`", "")
    return line.strip()


def from_text(raw: str) -> JobInfo:
    vals = {k: None for k in LABELS}
    for line in raw.split("\n"):
        ln = _strip_md(line)
        if not ln:
            continue
        # markdown 表格行： | 字段 | 值 |
        if line.count("|") >= 2:
            cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                for key, pat in LABELS.items():
                    if re.fullmatch(r"\s*(?:%s)\s*" % pat, cells[0]):
                        if vals[key] is None:
                            vals[key] = cells[1]
                continue
        for key, pat in LABELS.items():
            m = re.match(r"^\s*(?:%s)\s*[:：]\s*(.*)$" % pat, ln)
            if m and vals[key] is None:
                vals[key] = m.group(1).strip()
                break
    lo, hi, pc = parse_salary_text(vals["薪资范围"])
    return JobInfo(
        职位名称=_clean_scalar(vals["职位名称"]),
        最低薪资=lo, 最高薪资=hi, 薪资周期=pc,
        工作地点=_clean_scalar(vals["工作地点"]),
        硬技能=_clean_list(vals["硬技能"]),
        软技能=_clean_list(vals["软技能"]),
        学历要求=_clean_scalar(vals["学历要求"]),
        经验要求=_clean_scalar(vals["经验要求"]),
        AI相关技术栈要求=_clean_list(vals["AI相关技术栈要求"]),
    )


def validate_schema(obj) -> (bool, list):
    """Schema 合法性：顶层7字段齐备、薪资/技能为嵌套对象、列表字段确为 list"""
    errs = []
    if not isinstance(obj, dict):
        return False, ["顶层不是对象"]
    for k in ["职位名称", "薪资范围", "工作地点", "技能要求", "学历要求", "经验要求", "AI相关技术栈要求"]:
        if k not in obj:
            errs.append("缺字段:" + k)
    s = obj.get("薪资范围")
    if not isinstance(s, dict):
        errs.append("薪资范围非对象")
    else:
        for k in ["最低薪资", "最高薪资", "薪资周期"]:
            if k not in s:
                errs.append("薪资范围缺:" + k)
    sk = obj.get("技能要求")
    if not isinstance(sk, dict):
        errs.append("技能要求非对象")
    else:
        for k in ["硬技能", "软技能"]:
            if not isinstance(sk.get(k), list):
                errs.append(k + "非数组")
    if not isinstance(obj.get("AI相关技术栈要求"), list):
        errs.append("AI相关技术栈要求非数组")
    return len(errs) == 0, errs


def parse(raw: str):
    """返回 (JobInfo, meta)"""
    obj, lvl = try_json(raw)
    if obj is not None:
        ok, errs = validate_schema(obj)
        return from_obj(obj), {"parse_level": lvl, "schema_ok": ok, "schema_errors": errs}
    ji = from_text(raw)
    return ji, {"parse_level": "text", "schema_ok": False, "schema_errors": ["非JSON输出"]}
