from __future__ import annotations

"""
金融科技主营业务分类程序 v3。

在 v2 的“主营业务关键词预筛 + 大模型判断”基础上，新增：
1. 识别全部金融机构及金融相关机构；
2. 不将其自动认定为金融科技公司；
3. 单独导出年报复核名单，提示进一步核验科技、数字化和AI业务。
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 参数配置
# ============================================================

INPUT_FILE = Path("data/stock_pool_main_business_20251231.csv")
OUTPUT_DIR = Path("data")

# 公司级分类结果
COMPANY_OUTPUT_FILE = OUTPUT_DIR / "deepseek_company_classification_v3.csv"

# 将公司级标签回写到原始主营明细
ROW_OUTPUT_FILE = OUTPUT_DIR / "stock_pool_main_business_with_ai_labels_v3.csv"

# 预筛结果，便于检查被送入/未送入模型的公司
PREFILTER_OUTPUT_FILE = OUTPUT_DIR / "deepseek_prefilter_results_v3.csv"

# 金融机构及金融相关机构名单：统一提示进入年报进一步核验金融科技业务
FINANCIAL_INSTITUTION_REVIEW_FILE = (
    OUTPUT_DIR / "financial_institution_annual_report_review_v3.csv"
)

# 断点续跑文件：每成功处理一家公司就追加一条JSON
CHECKPOINT_FILE = OUTPUT_DIR / "deepseek_results_v3.jsonl"

# 本次运行失败记录
FAILED_FILE = OUTPUT_DIR / "deepseek_failed_v3.csv"

# API Key只允许从环境变量读取，避免密钥写入代码或被提交到版本库
DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY", "sk-2b24159721004658816ce8608137a77c").strip()
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
).strip()

# 适合大批量初筛；需要更高质量时可改为 deepseek-v4-pro
MODEL_NAME = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-v4-flash",
).strip()

# 首次测试建议设为20或50；正式运行改为None
MAX_COMPANIES: int | None = None

# True：先进行宽松规则预筛，只调用可能相关的公司
# False：全市场每家公司都调用模型
USE_KEYWORD_PREFILTER = True

# 如果为True，优先只用P（按产品）和I（按行业）口径
PREFER_PRODUCT_AND_INDUSTRY = True

# 每个主营分类口径最多提供多少条业务项目
MAX_ITEMS_PER_TYPE = 20

MAX_RETRIES = 5
REQUEST_INTERVAL = 0.25
REQUEST_TIMEOUT_SECONDS = 120.0

# 模型输出最大token
MAX_OUTPUT_TOKENS = 1000


# ============================================================
# 2. 预筛词库
# ============================================================

# 命中任意一个强关键词即可进入模型判断。
# 这些词尽量描述“金融场景中的软件、系统、数据或科技产品”。
STRONG_FINTECH_KEYWORDS = [
    "金融科技服务",
    "金融科技解决方案",
    "银行核心系统",
    "核心银行系统",
    "核心业务系统",
    "银行IT",
    "银行软件",
    "银行信息化",
    "开放银行",
    "信贷系统",
    "信贷管理系统",
    "证券IT",
    "证券软件",
    "证券交易系统",
    "资管系统",
    "资产管理系统",
    "保险IT",
    "保险软件",
    "保险核心系统",
    "智能风控",
    "风险决策系统",
    "反欺诈",
    "信用评分",
    "监管报送",
    "监管科技",
    "反洗钱系统",
    "智能投顾",
    "财富管理系统",
    "智能投研",
    "金融数据终端",
    "金融数据库",
    "行情数据",
    "投研数据",
    "征信数据",
    "财务云",
    "财税SaaS",
    "财税软件",
    "电子发票",
    "费用管理系统",
    "费控平台",
    "司库系统",
    "业财一体化",
    "支付系统",
    "聚合支付",
    "收单系统",
    "清算系统",
    "数字人民币",
]

# 弱匹配必须同时命中：
# “金融/财税场景词” + “软件/系统/平台等技术产品词”
FINANCIAL_CONTEXT_KEYWORDS = [
    "银行",
    "证券",
    "券商",
    "保险",
    "基金",
    "信托",
    "资管",
    "资产管理",
    "财富管理",
    "金融机构",
    "金融行业",
    "金融业务",
    "信贷",
    "授信",
    "征信",
    "反洗钱",
    "支付",
    "清算",
    "收单",
    "财务",
    "财税",
    "税务",
    "发票",
    "费控",
    "司库",
    "会计",
]

TECH_PRODUCT_KEYWORDS = [
    "软件",
    "系统",
    "平台",
    "解决方案",
    "SaaS",
    "数据服务",
    "技术服务",
    "云服务",
    "数字化",
    "信息系统",
    "信息技术",
    "应用开发",
    "系统集成",
    "数据平台",
    "智能化",
]

# 模糊词：只有这些宽泛描述时，不应给出过高置信度
AMBIGUOUS_TERMS = [
    "软件服务",
    "软件开发",
    "信息技术服务",
    "信息技术",
    "数字化服务",
    "数字化解决方案",
    "数据服务",
    "系统集成",
    "技术服务",
    "云服务",
    "金融",
    "其他业务",
]

# 这些基础行业中的公司即使被模型排除，只要主营描述模糊，
# 仍应降低置信度并建议进一步查看年报。
TECH_INDUSTRY_KEYWORDS = [
    "软件",
    "互联网",
    "IT",
    "信息技术",
    "计算机",
    "数据",
    "通信",
    "数字",
]


# ============================================================
# 2.1 金融机构及金融相关机构识别词库
# ============================================================

# 这部分不用于直接认定“金融科技公司”，而是用于建立年报复核名单。
# 原因是银行、证券、保险等机构的金融科技能力通常不会单独写入主营业务项目。
#
# 识别顺序：
# 1. 优先依据基础行业识别传统金融机构；
# 2. 行业字段不充分时，再通过明确的金融牌照/机构业务描述补充识别；
# 3. 对支付、征信、金融信息服务等金融相关机构单独标记。
FINANCIAL_INSTITUTION_INDUSTRY_KEYWORDS = [
    "银行",
    "证券",
    "券商",
    "保险",
    "非银金融",
    "多元金融",
    "信托",
    "基金",
    "期货",
    "金融控股",
    "金融租赁",
    "融资租赁",
    "消费金融",
    "小额贷款",
    "小贷",
    "融资担保",
    "担保",
    "典当",
]

FINANCIAL_INSTITUTION_BUSINESS_KEYWORDS = [
    "商业银行业务",
    "银行业务",
    "存贷款业务",
    "公司银行业务",
    "个人银行业务",
    "证券经纪业务",
    "证券承销业务",
    "证券自营业务",
    "投资银行业务",
    "期货经纪业务",
    "保险业务",
    "人寿保险业务",
    "财产保险业务",
    "再保险业务",
    "信托业务",
    "基金管理业务",
    "公募基金管理",
    "私募基金管理",
    "资产管理业务",
    "金融租赁业务",
    "融资租赁业务",
    "消费金融业务",
    "小额贷款业务",
    "融资担保业务",
    "商业保理业务",
    "典当业务",
]

FINANCIAL_RELATED_INDUSTRY_KEYWORDS = [
    "金融服务",
    "金融信息服务",
    "金融数据",
    "互联网金融",
    "金融科技",
    "支付",
    "支付服务",
    "第三方支付",
    "征信",
    "征信服务",
    "信用服务",
    "财富管理",
    "资产管理",
    "投资管理",
    "投资咨询",
    "基金销售",
    "商业保理",
    "财税服务",
    "会计服务",
]


# ============================================================
# 3. 分类枚举
# ============================================================

ALLOWED_TRACKS = {
    "银行IT",
    "证券IT",
    "保险IT",
    "智能风控",
    "智能投顾",
    "财富管理科技",
    "财务SaaS",
    "财税科技",
    "监管科技",
    "金融数据",
    "支付科技",
    "征信科技",
    "金融信息安全",
    "AI金融基础设施",
    "其他金融科技",
}

AI_EVIDENCE_LEVELS = {"explicit", "indirect", "none"}
BUSINESS_MATERIALITY_LEVELS = {"core", "important", "weak", "none"}


# ============================================================
# 4. 提示词
# ============================================================

SYSTEM_PROMPT = """
你是一名谨慎的A股金融科技行业分类研究员。

用户会提供一家上市公司的基础行业信息和主营业务构成。你必须只依据
用户提供的数据判断，不得使用你对公司的外部知识、历史记忆、新闻资料、
证券市场概念标签或公司名称联想。

任务：
1. 判断公司是否实质从事金融科技业务；
2. 判断可能属于哪些金融科技赛道；
3. 判断输入数据中是否存在明确的人工智能证据；
4. 判断是否需要查看年度报告进一步验证；
5. 返回严格合法的json对象。

金融科技赛道只能从以下值中选择：
- 银行IT
- 证券IT
- 保险IT
- 智能风控
- 智能投顾
- 财富管理科技
- 财务SaaS
- 财税科技
- 监管科技
- 金融数据
- 支付科技
- 征信科技
- 金融信息安全
- AI金融基础设施
- 其他金融科技

核心判断原则：
- 银行、券商、保险、信托、基金等传统金融机构自身，不因为使用软件或AI
  就自动属于金融科技供应商。
- 仅仅拥有金融客户，不等于公司从事金融科技。
- 通用硬件、服务器、网络设备、普通系统集成、通用网络安全、一般数据服务，
  不能直接认定为金融科技。
- 金融科技应体现为面向金融、财务、财税、支付等场景的软件、系统、平台、
  SaaS、数据产品、技术解决方案或科技服务。
- 服务银行不一定等于银行IT；服务证券公司不一定等于证券IT。
  必须存在专用产品或解决方案证据。
- 不能仅因“金融行业收入”“金融客户”而归入“金融数据”。
  只有明确出现金融数据库、行情数据、金融终端、投研数据、征信数据、
  金融信息服务等产品时，才可归为金融数据。
- 如果没有明确出现人工智能、AI、大模型、机器学习、深度学习、知识图谱、
  自然语言处理、智能决策、智能风控、智能投顾等证据，不得凭空认定AI能力。
- “智能化”单独出现通常只能视为间接证据；必须结合具体产品或金融场景判断。
- 主营业务构成没有AI信息时，应标记需要年度报告验证，而不是直接否定公司
  未来可能具有AI能力。

收入约束：
- 不得自行计算任何主营收入占比。
- 按产品P和按行业I属于不同披露口径，不能相加，也不能互相作为分母。
- 除非用户明确提供收入比例字段，否则不得声称某业务占比为多少。
- 业务重要性只能依据业务项目的明确程度、排序和披露金额做定性判断，
  不得编造精确比例。

证据约束：
- evidence最多返回5条。
- evidence必须逐字来自用户输入的主营业务项目，不得改写成输入中不存在的产品。
- 不得编造客户、订单、收入占比、产品名称或技术能力。

置信度规则：
- 90—100：输入中存在非常明确、直接、充分的正面证据或排除证据；
- 75—89：证据总体明确，但主营项目描述仍不完整；
- 60—74：信息有限，存在多种合理解释；
- 40—59：只能初步推断，强烈需要年报验证；
- 0—39：无法依据输入可靠判断。
- 如果主营项目主要是“软件服务”“软件开发”“信息技术服务”“金融”
  “数据服务”“数字化服务”等宽泛名称，confidence不得高于75。
- 如果判断为非金融科技，但公司属于软件、互联网、数据、信息技术等行业，
  且主营描述不够具体，confidence不得高于80。

必须返回如下格式的合法json对象，不要返回Markdown，不要使用代码块：

{
  "fintech_related": true,
  "ai_evidence_level": "none",
  "tracks": ["银行IT"],
  "business_materiality": "important",
  "confidence": 80,
  "evidence": ["银行数字化解决方案"],
  "reason": "主营业务明确包含银行专用数字化解决方案，但输入没有AI技术证据。",
  "needs_annual_report_validation": true
}

字段要求：
- fintech_related：布尔值；
- ai_evidence_level：只能是 explicit、indirect、none；
- tracks：数组，不相关时返回空数组；
- business_materiality：只能是 core、important、weak、none；
- confidence：0到100之间整数；
- evidence：字符串数组，最多5条；
- reason：简洁说明依据和局限；
- needs_annual_report_validation：布尔值。
""".strip()


# ============================================================
# 5. 基础工具
# ============================================================

def validate_config() -> None:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未检测到DEEPSEEK_API_KEY环境变量。\n"
            "Windows PowerShell示例：\n"
            '$env:DEEPSEEK_API_KEY="你的API Key"'
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"输入文件不存在：{INPUT_FILE}")

    if MAX_COMPANIES is not None and MAX_COMPANIES <= 0:
        raise ValueError("MAX_COMPANIES必须为正整数或None。")


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def first_nonempty(series: pd.Series) -> str:
    for value in series:
        text = safe_text(value)
        if text:
            return text
    return ""


def format_number(value: Any) -> str:
    try:
        number = float(value)
        if pd.isna(number):
            return "未披露"
        return f"{number:,.2f}"
    except (TypeError, ValueError):
        return "未披露"


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}

    return False


# ============================================================
# 6. 预筛
# ============================================================

def evaluate_prefilter(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)

    strong_hits = [
        keyword
        for keyword in STRONG_FINTECH_KEYWORDS
        if normalize_text(keyword) in normalized
    ]

    financial_hits = [
        keyword
        for keyword in FINANCIAL_CONTEXT_KEYWORDS
        if normalize_text(keyword) in normalized
    ]

    tech_hits = [
        keyword
        for keyword in TECH_PRODUCT_KEYWORDS
        if normalize_text(keyword) in normalized
    ]

    passed = bool(strong_hits) or (
        bool(financial_hits) and bool(tech_hits)
    )

    if strong_hits:
        reason = "命中明确金融科技产品/服务关键词"
    elif financial_hits and tech_hits:
        reason = "同时命中金融场景词与技术产品词"
    else:
        reason = "未同时满足金融场景与技术产品条件"

    return {
        "prefilter_passed": passed,
        "prefilter_reason": reason,
        "prefilter_strong_hits": "、".join(sorted(set(strong_hits))),
        "prefilter_financial_hits": "、".join(sorted(set(financial_hits))),
        "prefilter_tech_hits": "、".join(sorted(set(tech_hits))),
    }


def evaluate_financial_institution_candidate(
    industry: str,
    business_items: list[str],
) -> dict[str, Any]:
    """
    识别传统金融机构及金融相关机构，建立独立的年报复核名单。

    该函数不会直接把公司认定为金融科技公司，也不会改变原有模型预筛条件。
    它仅表示：这类机构很可能在年报中披露科技投入、数字化平台、AI应用、
    风控系统或内部金融科技子公司，值得进一步分析年度报告。
    """
    normalized_industry = normalize_text(industry)
    normalized_business = normalize_text(" ".join(business_items))

    institution_industry_hits = [
        keyword
        for keyword in FINANCIAL_INSTITUTION_INDUSTRY_KEYWORDS
        if normalize_text(keyword) in normalized_industry
    ]

    institution_business_hits = [
        keyword
        for keyword in FINANCIAL_INSTITUTION_BUSINESS_KEYWORDS
        if normalize_text(keyword) in normalized_business
    ]

    related_industry_hits = [
        keyword
        for keyword in FINANCIAL_RELATED_INDUSTRY_KEYWORDS
        if normalize_text(keyword) in normalized_industry
    ]

    is_financial_institution = bool(
        institution_industry_hits
        or institution_business_hits
    )

    is_financial_related_institution = bool(
        related_industry_hits
    )

    is_review_candidate = (
        is_financial_institution
        or is_financial_related_institution
    )

    if is_financial_institution:
        category = "金融机构"
        reason = (
            "公司属于银行、证券、保险、信托、期货、租赁等金融机构。"
            "金融科技能力通常不会单独列入主营业务项目，建议通过年度报告进一步核验。"
        )
    elif is_financial_related_institution:
        category = "金融相关机构"
        reason = (
            "公司属于支付、征信、金融信息服务、财富管理等金融相关机构。"
            "建议通过年度报告进一步核验其金融科技、数字化或人工智能业务。"
        )
    else:
        category = ""
        reason = ""

    return {
        "financial_institution_review_candidate": is_review_candidate,
        "financial_institution_category": category,
        "financial_institution_industry_hits": "、".join(
            sorted(set(institution_industry_hits))
        ),
        "financial_institution_business_hits": "、".join(
            sorted(set(institution_business_hits))
        ),
        "financial_related_industry_hits": "、".join(
            sorted(set(related_industry_hits))
        ),
        "financial_institution_review_reason": reason,
    }


# ============================================================
# 7. 数据读取与清洗
# ============================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "ts_code": str,
            "symbol": str,
        },
        low_memory=False,
    )

    required_columns = {"ts_code", "bz_item"}
    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"输入文件缺少必要字段：{sorted(missing)}\n"
            f"当前字段：{df.columns.tolist()}"
        )

    optional_columns = [
        "name",
        "industry",
        "market",
        "exchange",
        "query_type",
        "bz_sales",
        "bz_profit",
        "bz_cost",
        "end_date",
    ]

    for column in optional_columns:
        if column not in df.columns:
            df[column] = None

    df["ts_code"] = df["ts_code"].fillna("").astype(str).str.strip()
    df = df[df["ts_code"] != ""].copy()

    df["bz_item"] = (
        df["bz_item"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["query_type"] = (
        df["query_type"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    for column in ["bz_sales", "bz_profit", "bz_cost"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if PREFER_PRODUCT_AND_INDUSTRY:
        pi_df = df[df["query_type"].isin(["P", "I"])].copy()

        # 只有确实存在P/I数据时才使用，避免把没有P/I的公司全部丢掉
        codes_with_pi = set(pi_df["ts_code"].unique())
        other_df = df[~df["ts_code"].isin(codes_with_pi)].copy()
        df = pd.concat([pi_df, other_df], ignore_index=True)

    return df.reset_index(drop=True)


# ============================================================
# 8. 公司级聚合
# ============================================================

def build_business_description(company_df: pd.DataFrame) -> str:
    sections: list[str] = []

    query_type_order = [
        ("P", "按产品P"),
        ("I", "按行业I"),
        ("D", "按地区D"),
        ("UNKNOWN", "其他口径"),
    ]

    existing_types = set(company_df["query_type"].unique())

    for query_type, label in query_type_order:
        if query_type not in existing_types:
            continue

        part = company_df[
            company_df["query_type"] == query_type
        ].copy()

        part = (
            part
            .sort_values(
                "bz_sales",
                ascending=False,
                na_position="last",
            )
            .drop_duplicates(subset=["bz_item"], keep="first")
            .head(MAX_ITEMS_PER_TYPE)
        )

        lines: list[str] = []

        for row in part.itertuples(index=False):
            business_item = safe_text(row.bz_item)

            if not business_item:
                continue

            lines.append(
                f"- 业务项目：{business_item}"
                f"；该口径披露收入：{format_number(row.bz_sales)}"
                f"；该口径披露利润：{format_number(row.bz_profit)}"
            )

        if lines:
            sections.append(
                f"{label}（不同口径不得合并计算占比）：\n"
                + "\n".join(lines)
            )

    # 兼容其他未预定义query_type
    known_types = {item[0] for item in query_type_order}
    unknown_types = sorted(existing_types - known_types)

    for query_type in unknown_types:
        part = company_df[
            company_df["query_type"] == query_type
        ].copy()

        part = (
            part
            .sort_values(
                "bz_sales",
                ascending=False,
                na_position="last",
            )
            .drop_duplicates(subset=["bz_item"], keep="first")
            .head(MAX_ITEMS_PER_TYPE)
        )

        lines = []

        for row in part.itertuples(index=False):
            business_item = safe_text(row.bz_item)
            if business_item:
                lines.append(
                    f"- 业务项目：{business_item}"
                    f"；该口径披露收入：{format_number(row.bz_sales)}"
                    f"；该口径披露利润：{format_number(row.bz_profit)}"
                )

        if lines:
            sections.append(
                f"口径{query_type}（不得与其他口径合并计算占比）：\n"
                + "\n".join(lines)
            )

    return "\n\n".join(sections)


def collect_business_items(company_df: pd.DataFrame) -> list[str]:
    items = [
        safe_text(item)
        for item in company_df["bz_item"].tolist()
        if safe_text(item)
    ]

    # 保持原顺序去重
    return list(dict.fromkeys(items))


def build_company_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for ts_code, company_df in df.groupby("ts_code", sort=True):
        name = first_nonempty(company_df["name"])
        industry = first_nonempty(company_df["industry"])
        market = first_nonempty(company_df["market"])
        exchange = first_nonempty(company_df["exchange"])
        report_period = first_nonempty(company_df["end_date"])

        business_description = build_business_description(company_df)
        business_items = collect_business_items(company_df)

        # 明确不使用公司名称进行预筛，避免“金融街”等名称误命中
        prefilter_text = " ".join(
            [
                industry,
                " ".join(business_items),
            ]
        )

        prefilter_result = evaluate_prefilter(prefilter_text)

        financial_institution_result = (
            evaluate_financial_institution_candidate(
                industry=industry,
                business_items=business_items,
            )
        )

        record = {
            "ts_code": ts_code,
            "name": name,
            "industry": industry,
            "market": market,
            "exchange": exchange,
            "report_period": report_period,
            "business_description": business_description,
            "business_items": business_items,
            **prefilter_result,
            **financial_institution_result,
        }

        records.append(record)

    return records


# ============================================================
# 9. 断点续跑
# ============================================================

def load_completed_results() -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}

    if not CHECKPOINT_FILE.exists():
        return completed

    with CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts_code = safe_text(result.get("ts_code"))
            if ts_code:
                completed[ts_code] = result

    return completed


def append_checkpoint(result: dict[str, Any]) -> None:
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


# ============================================================
# 10. 模型输入
# ============================================================

def build_user_prompt(record: dict[str, Any]) -> str:
    return f"""
请根据以下数据分类，并只返回规定格式的json对象。

股票代码：{record["ts_code"]}
公司名称：{record["name"]}
基础行业：{record["industry"]}
市场板块：{record["market"]}
交易所：{record["exchange"]}
报告期：{record["report_period"]}

注意：
- 公司名称只用于识别，不得根据名称联想业务。
- 按产品P和按行业I是不同口径，严禁相加或自行计算占比。
- 只能引用下列主营业务项目作为evidence。

主营业务构成：

{record["business_description"] or "未提供有效主营业务构成"}
""".strip()


# ============================================================
# 11. 模型结果验证与本地校准
# ============================================================

def validate_raw_result(result: dict[str, Any]) -> None:
    required_fields = {
        "fintech_related",
        "ai_evidence_level",
        "tracks",
        "business_materiality",
        "confidence",
        "evidence",
        "reason",
        "needs_annual_report_validation",
    }

    missing = required_fields - set(result)

    if missing:
        raise ValueError(f"模型结果缺少字段：{sorted(missing)}")

    if not isinstance(result["fintech_related"], bool):
        raise ValueError("fintech_related必须是布尔值。")

    if result["ai_evidence_level"] not in AI_EVIDENCE_LEVELS:
        raise ValueError("ai_evidence_level值不合法。")

    if result["business_materiality"] not in BUSINESS_MATERIALITY_LEVELS:
        raise ValueError("business_materiality值不合法。")

    if not isinstance(result["tracks"], list):
        raise ValueError("tracks必须是数组。")

    if not isinstance(result["evidence"], list):
        raise ValueError("evidence必须是数组。")

    if not isinstance(result["needs_annual_report_validation"], bool):
        raise ValueError(
            "needs_annual_report_validation必须是布尔值。"
        )

    try:
        confidence = int(result["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence必须是整数。") from exc

    if not 0 <= confidence <= 100:
        raise ValueError("confidence必须在0到100之间。")

    result["confidence"] = confidence


def evidence_exists_in_input(
    evidence: str,
    business_items: list[str],
) -> bool:
    normalized_evidence = normalize_text(evidence)

    if not normalized_evidence:
        return False

    for item in business_items:
        normalized_item = normalize_text(item)

        # 允许模型完整引用原项目，或引用原项目中的明确短语
        if (
            normalized_evidence == normalized_item
            or normalized_evidence in normalized_item
        ):
            return True

    return False


def calibrate_result(
    result: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    result = dict(result)

    result["fintech_related"] = is_true(
        result.get("fintech_related")
    )
    result["needs_annual_report_validation"] = is_true(
        result.get("needs_annual_report_validation")
    )

    # 清洗赛道，只保留允许值并去重
    tracks = [
        safe_text(track)
        for track in result.get("tracks", [])
        if safe_text(track) in ALLOWED_TRACKS
    ]
    tracks = list(dict.fromkeys(tracks))

    if not result["fintech_related"]:
        tracks = []

    result["tracks"] = tracks

    # 证据必须可追溯到输入业务项目
    valid_evidence = [
        safe_text(item)
        for item in result.get("evidence", [])
        if evidence_exists_in_input(
            safe_text(item),
            record["business_items"],
        )
    ]
    result["evidence"] = list(dict.fromkeys(valid_evidence))[:5]

    business_text = normalize_text(
        " ".join(record["business_items"])
    )
    industry_text = normalize_text(record["industry"])

    ambiguous_hit = any(
        normalize_text(term) in business_text
        for term in AMBIGUOUS_TERMS
    )

    tech_industry_hit = any(
        normalize_text(term) in industry_text
        for term in TECH_INDUSTRY_KEYWORDS
    )

    confidence = int(result["confidence"])

    # 模糊主营描述不得过度自信
    if ambiguous_hit:
        confidence = min(confidence, 75)

    # 软件、互联网、信息技术等行业被排除，但信息模糊时仍需复核
    if (
        tech_industry_hit
        and not result["fintech_related"]
        and ambiguous_hit
    ):
        confidence = min(confidence, 80)
        result["needs_annual_report_validation"] = True

    # 金融科技相关但主营构成未提供AI证据，必须查看年报
    if (
        result["fintech_related"]
        and result["ai_evidence_level"] == "none"
    ):
        result["needs_annual_report_validation"] = True

    # 金融科技相关但模型没有提供有效引用，也应降低置信度
    if result["fintech_related"] and not result["evidence"]:
        confidence = min(confidence, 65)
        result["needs_annual_report_validation"] = True

    # 非金融科技时，业务重要性应为none或weak
    if (
        not result["fintech_related"]
        and result["business_materiality"] in {"core", "important"}
    ):
        result["business_materiality"] = "none"

    # 金融科技相关但没有赛道，归入其他金融科技并降低置信度
    if result["fintech_related"] and not result["tracks"]:
        result["tracks"] = ["其他金融科技"]
        confidence = min(confidence, 70)

    result["confidence"] = confidence

    # 这两个字段由程序统一派生，避免模型概念混淆
    result["fintech_candidate_for_annual_report"] = (
        result["fintech_related"]
        and result["business_materiality"] in {"core", "important"}
    )

    result["confirmed_ai_from_main_business"] = (
        result["ai_evidence_level"] in {"explicit", "indirect"}
    )

    result["ai_fintech_candidate"] = (
        result["fintech_candidate_for_annual_report"]
        and result["confirmed_ai_from_main_business"]
    )

    # 金融机构及金融相关机构单独进入年报复核名单。
    # 这里不改变fintech_related的模型判断，避免把传统金融机构直接视作
    # 金融科技供应商。
    result["financial_institution_review_candidate"] = bool(
        record.get("financial_institution_review_candidate")
    )
    result["financial_institution_category"] = safe_text(
        record.get("financial_institution_category")
    )
    result["financial_institution_review_reason"] = safe_text(
        record.get("financial_institution_review_reason")
    )

    # 统一的年报金融科技复核候选：
    # 1. 主营业务已显示重要金融科技业务；或
    # 2. 公司属于金融机构/金融相关机构，需要从年报继续确认。
    result["annual_report_fintech_review_candidate"] = (
        result["fintech_candidate_for_annual_report"]
        or result["financial_institution_review_candidate"]
    )

    if result["financial_institution_review_candidate"]:
        result["needs_annual_report_validation"] = True

    return result


# ============================================================
# 12. DeepSeek调用
# ============================================================

def classify_company(
    client: OpenAI,
    record: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(record),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=False,
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                },
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("模型返回空content。")

            raw_result = json.loads(content)
            validate_raw_result(raw_result)

            result = calibrate_result(
                result=raw_result,
                record=record,
            )

            usage = response.usage

            result.update(
                {
                    "ts_code": record["ts_code"],
                    "name": record["name"],
                    "industry": record["industry"],
                    "market": record["market"],
                    "exchange": record["exchange"],
                    "report_period": record["report_period"],
                    "prefilter_passed": record["prefilter_passed"],
                    "prefilter_reason": record["prefilter_reason"],
                    "prefilter_strong_hits": (
                        record["prefilter_strong_hits"]
                    ),
                    "prefilter_financial_hits": (
                        record["prefilter_financial_hits"]
                    ),
                    "prefilter_tech_hits": (
                        record["prefilter_tech_hits"]
                    ),
                    "financial_institution_industry_hits": (
                        record["financial_institution_industry_hits"]
                    ),
                    "financial_institution_business_hits": (
                        record["financial_institution_business_hits"]
                    ),
                    "financial_related_industry_hits": (
                        record["financial_related_industry_hits"]
                    ),
                    "model": MODEL_NAME,
                    "prompt_tokens": getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    ),
                    "completion_tokens": getattr(
                        usage,
                        "completion_tokens",
                        None,
                    ),
                    "total_tokens": getattr(
                        usage,
                        "total_tokens",
                        None,
                    ),
                }
            )

            return result

        except Exception as exc:
            last_error = exc

            if attempt >= MAX_RETRIES:
                break

            wait_seconds = min(2 ** attempt, 30) + random.random()

            print(
                f"  第{attempt}/{MAX_RETRIES}次调用失败：{exc}"
            )
            print(f"  {wait_seconds:.1f}秒后重试。")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"DeepSeek连续调用失败：{last_error}"
    ) from last_error


# ============================================================
# 13. 导出
# ============================================================

def load_checkpoint_as_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not CHECKPOINT_FILE.exists():
        return pd.DataFrame()

    with CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        return result_df

    result_df = (
        result_df
        .drop_duplicates(subset=["ts_code"], keep="last")
        .reset_index(drop=True)
    )

    for column in ["tracks", "evidence"]:
        if column in result_df.columns:
            result_df[column] = result_df[column].apply(
                lambda value: "、".join(value)
                if isinstance(value, list)
                else value
            )

    return result_df


def export_prefilter_results(
    company_records: list[dict[str, Any]],
) -> None:
    rows = []

    for record in company_records:
        rows.append(
            {
                "ts_code": record["ts_code"],
                "name": record["name"],
                "industry": record["industry"],
                "report_period": record["report_period"],
                "prefilter_passed": record["prefilter_passed"],
                "prefilter_reason": record["prefilter_reason"],
                "prefilter_strong_hits": (
                    record["prefilter_strong_hits"]
                ),
                "prefilter_financial_hits": (
                    record["prefilter_financial_hits"]
                ),
                "prefilter_tech_hits": (
                    record["prefilter_tech_hits"]
                ),
                "financial_institution_review_candidate": (
                    record["financial_institution_review_candidate"]
                ),
                "financial_institution_category": (
                    record["financial_institution_category"]
                ),
                "financial_institution_industry_hits": (
                    record["financial_institution_industry_hits"]
                ),
                "financial_institution_business_hits": (
                    record["financial_institution_business_hits"]
                ),
                "financial_related_industry_hits": (
                    record["financial_related_industry_hits"]
                ),
                "financial_institution_review_reason": (
                    record["financial_institution_review_reason"]
                ),
                "business_item_count": len(record["business_items"]),
            }
        )

    pd.DataFrame(rows).to_csv(
        PREFILTER_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def export_financial_institution_review_results(
    company_records: list[dict[str, Any]],
    result_df: pd.DataFrame,
) -> None:
    """
    导出全部金融机构及金融相关机构。

    该名单独立于主营业务金融科技模型筛选：
    即使公司没有通过关键词预筛、没有调用模型，仍会进入此名单，
    并统一提示通过年度报告进一步确认金融科技、数字化和AI业务。
    """
    rows: list[dict[str, Any]] = []

    result_by_code: dict[str, dict[str, Any]] = {}

    if not result_df.empty and "ts_code" in result_df.columns:
        result_by_code = (
            result_df
            .drop_duplicates(subset=["ts_code"], keep="last")
            .set_index("ts_code")
            .to_dict(orient="index")
        )

    for record in company_records:
        if not record["financial_institution_review_candidate"]:
            continue

        model_result = result_by_code.get(record["ts_code"], {})

        rows.append(
            {
                "ts_code": record["ts_code"],
                "name": record["name"],
                "industry": record["industry"],
                "market": record["market"],
                "exchange": record["exchange"],
                "report_period": record["report_period"],
                "financial_institution_category": (
                    record["financial_institution_category"]
                ),
                "financial_institution_industry_hits": (
                    record["financial_institution_industry_hits"]
                ),
                "financial_institution_business_hits": (
                    record["financial_institution_business_hits"]
                ),
                "financial_related_industry_hits": (
                    record["financial_related_industry_hits"]
                ),
                "prefilter_passed": record["prefilter_passed"],
                "prefilter_reason": record["prefilter_reason"],
                "model_result_available": bool(model_result),
                "fintech_related_from_main_business": model_result.get(
                    "fintech_related"
                ),
                "ai_evidence_level_from_main_business": model_result.get(
                    "ai_evidence_level"
                ),
                "tracks_from_main_business": model_result.get("tracks"),
                "business_materiality_from_main_business": model_result.get(
                    "business_materiality"
                ),
                "confidence_from_main_business": model_result.get(
                    "confidence"
                ),
                "annual_report_fintech_review_candidate": True,
                "annual_report_review_reason": (
                    record["financial_institution_review_reason"]
                ),
                "recommended_next_step": (
                    "查看年度报告中的科技投入、信息科技、数字化转型、"
                    "人工智能、大模型、智能风控、线上平台、研发投入及"
                    "金融科技子公司等相关章节。"
                ),
            }
        )

    review_df = pd.DataFrame(rows)

    if review_df.empty:
        review_df = pd.DataFrame(
            columns=[
                "ts_code",
                "name",
                "industry",
                "financial_institution_category",
                "annual_report_fintech_review_candidate",
                "annual_report_review_reason",
                "recommended_next_step",
            ]
        )
    else:
        review_df = review_df.sort_values(
            [
                "financial_institution_category",
                "industry",
                "ts_code",
            ],
            na_position="last",
        )

    review_df.to_csv(
        FINANCIAL_INSTITUTION_REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "  金融机构年报复核名单："
        f"{FINANCIAL_INSTITUTION_REVIEW_FILE}"
    )
    print(
        "  金融机构及金融相关机构数量："
        f"{len(review_df):,}"
    )


def export_results(
    original_df: pd.DataFrame,
    company_records: list[dict[str, Any]],
) -> None:
    result_df = load_checkpoint_as_dataframe()

    # 无论是否已有模型成功结果，都先导出金融机构年报复核名单。
    export_financial_institution_review_results(
        company_records=company_records,
        result_df=result_df,
    )

    if result_df.empty:
        print("没有可导出的DeepSeek成功结果。")
        return

    sort_columns = [
        column
        for column in [
            "fintech_candidate_for_annual_report",
            "confirmed_ai_from_main_business",
            "fintech_related",
            "confidence",
        ]
        if column in result_df.columns
    ]

    ascending_map = {
        "fintech_candidate_for_annual_report": False,
        "confirmed_ai_from_main_business": False,
        "fintech_related": False,
        "confidence": False,
    }

    if sort_columns:
        result_df = result_df.sort_values(
            sort_columns,
            ascending=[
                ascending_map[column]
                for column in sort_columns
            ],
        )

    result_df.to_csv(
        COMPANY_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    label_columns = [
        "ts_code",
        "fintech_related",
        "fintech_candidate_for_annual_report",
        "confirmed_ai_from_main_business",
        "ai_fintech_candidate",
        "financial_institution_review_candidate",
        "financial_institution_category",
        "annual_report_fintech_review_candidate",
        "financial_institution_review_reason",
        "financial_institution_industry_hits",
        "financial_institution_business_hits",
        "financial_related_industry_hits",
        "ai_evidence_level",
        "tracks",
        "business_materiality",
        "confidence",
        "evidence",
        "reason",
        "needs_annual_report_validation",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    available_label_columns = [
        column
        for column in label_columns
        if column in result_df.columns
        and column not in {
            "financial_institution_review_candidate",
            "financial_institution_category",
            "annual_report_fintech_review_candidate",
            "financial_institution_review_reason",
        }
    ]

    # 先把公司级金融机构标记回写到全部主营明细。
    # 这一步与模型是否被调用无关，因此未通过预筛的银行、证券、保险等
    # 公司仍会得到明确的年报复核提示。
    company_review_df = pd.DataFrame(
        [
            {
                "ts_code": record["ts_code"],
                "financial_institution_review_candidate": (
                    record["financial_institution_review_candidate"]
                ),
                "financial_institution_category": (
                    record["financial_institution_category"]
                ),
                "financial_institution_review_reason": (
                    record["financial_institution_review_reason"]
                ),
                "financial_institution_industry_hits": (
                    record["financial_institution_industry_hits"]
                ),
                "financial_institution_business_hits": (
                    record["financial_institution_business_hits"]
                ),
                "financial_related_industry_hits": (
                    record["financial_related_industry_hits"]
                ),
            }
            for record in company_records
        ]
    )

    enriched_df = original_df.merge(
        company_review_df,
        on="ts_code",
        how="left",
    )

    enriched_df = enriched_df.merge(
        result_df[available_label_columns],
        on="ts_code",
        how="left",
    )

    model_fintech_candidate = (
        enriched_df["fintech_candidate_for_annual_report"]
        if "fintech_candidate_for_annual_report" in enriched_df.columns
        else pd.Series(False, index=enriched_df.index)
    )

    institution_candidate = (
        enriched_df["financial_institution_review_candidate"]
        .fillna(False)
        .astype(bool)
    )

    enriched_df["annual_report_fintech_review_candidate"] = (
        model_fintech_candidate.fillna(False).astype(bool)
        | institution_candidate
    )

    if "needs_annual_report_validation" in enriched_df.columns:
        enriched_df["needs_annual_report_validation"] = (
            enriched_df["needs_annual_report_validation"]
            .fillna(False)
            .astype(bool)
            | institution_candidate
        )
    else:
        enriched_df["needs_annual_report_validation"] = (
            institution_candidate
        )

    enriched_df.to_csv(
        ROW_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n结果已导出：")
    print(f"  公司级分类：{COMPANY_OUTPUT_FILE}")
    print(f"  明细回写：  {ROW_OUTPUT_FILE}")
    print(f"  预筛明细：  {PREFILTER_OUTPUT_FILE}")
    print(
        "  金融机构年报复核名单："
        f"{FINANCIAL_INSTITUTION_REVIEW_FILE}"
    )

    print("\n公司级分类统计：")

    for column in [
        "fintech_related",
        "fintech_candidate_for_annual_report",
        "financial_institution_review_candidate",
        "annual_report_fintech_review_candidate",
        "confirmed_ai_from_main_business",
        "ai_fintech_candidate",
        "needs_annual_report_validation",
    ]:
        if column in result_df.columns:
            print(f"\n{column}:")
            print(result_df[column].value_counts(dropna=False))

    if "tracks" in result_df.columns:
        track_series = (
            result_df["tracks"]
            .fillna("")
            .str.split("、")
            .explode()
        )
        track_series = track_series[track_series != ""]

        print("\n赛道分布：")
        print(track_series.value_counts().head(30))

    if "total_tokens" in result_df.columns:
        token_series = pd.to_numeric(
            result_df["total_tokens"],
            errors="coerce",
        )

        print("\nToken统计：")
        print(f"  总Token：{token_series.sum():,.0f}")
        print(f"  平均每家公司：{token_series.mean():,.1f}")


# ============================================================
# 14. 主程序
# ============================================================

def main() -> None:
    validate_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    original_df = load_data()
    company_records = build_company_records(original_df)

    export_prefilter_results(company_records)

    if USE_KEYWORD_PREFILTER:
        target_records = [
            record
            for record in company_records
            if record["prefilter_passed"]
        ]
    else:
        target_records = company_records

    if MAX_COMPANIES is not None:
        target_records = target_records[:MAX_COMPANIES]

    completed = load_completed_results()
    failed_records: list[dict[str, Any]] = []

    print(f"模型：{MODEL_NAME}")
    print(
        "原始上市公司数量："
        f"{original_df['ts_code'].nunique():,}"
    )
    print(f"预筛通过公司数量：{sum(r['prefilter_passed'] for r in company_records):,}")
    print(
        "金融机构及金融相关机构数量："
        f"{sum(r['financial_institution_review_candidate']
               for r in company_records):,}"
    )
    print(f"本轮准备调用模型：{len(target_records):,}")
    print(f"断点文件已有结果：{len(completed):,}")

    total = len(target_records)

    for index, record in enumerate(target_records, start=1):
        ts_code = record["ts_code"]
        name = record["name"]

        if ts_code in completed:
            print(
                f"[{index}/{total}] 跳过已完成："
                f"{ts_code} {name}"
            )
            continue

        print(
            f"[{index}/{total}] 判断："
            f"{ts_code} {name}"
        )

        try:
            result = classify_company(
                client=client,
                record=record,
            )

            append_checkpoint(result)
            completed[ts_code] = result

            print(
                "  金融科技="
                f"{result['fintech_related']}，"
                "年报候选="
                f"{result['fintech_candidate_for_annual_report']}，"
                "主营确认AI="
                f"{result['confirmed_ai_from_main_business']}，"
                "置信度="
                f"{result['confidence']}"
            )

        except Exception as exc:
            print(f"  处理失败：{exc}")

            failed_records.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "industry": record["industry"],
                    "error": str(exc),
                }
            )

        time.sleep(REQUEST_INTERVAL)

    if failed_records:
        pd.DataFrame(failed_records).to_csv(
            FAILED_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\n失败记录已导出：{FAILED_FILE}")

    export_results(
        original_df=original_df,
        company_records=company_records,
    )
    print("\n全部任务完成。")


if __name__ == "__main__":
    main()
