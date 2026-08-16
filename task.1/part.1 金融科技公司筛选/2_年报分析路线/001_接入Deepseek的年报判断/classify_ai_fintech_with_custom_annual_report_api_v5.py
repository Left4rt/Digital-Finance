from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


# 运行依赖：
# pip install -U pandas openai requests pymupdf tushare
# tushare仅在需要自动下载年报时使用；本地PDF模式无需Tushare权限。


# ============================================================
# 1. 参数配置
# ============================================================

INPUT_FILE = Path("data/stock_pool_main_business_20251231.csv")
OUTPUT_DIR = Path("data")

# 公司级分类结果
COMPANY_OUTPUT_FILE = OUTPUT_DIR / "deepseek_company_classification_v4.csv"

# 将公司级标签回写到原始主营明细
ROW_OUTPUT_FILE = OUTPUT_DIR / "stock_pool_main_business_with_ai_labels_v4.csv"

# 预筛结果，便于检查被送入/未送入模型的公司
PREFILTER_OUTPUT_FILE = OUTPUT_DIR / "deepseek_prefilter_results_v4.csv"

# 金融机构及金融相关主体池
FINANCIAL_ENTITY_OUTPUT_FILE = OUTPUT_DIR / "financial_entities_v4.csv"

# 断点续跑文件：每成功处理一家公司就追加一条JSON
CHECKPOINT_FILE = OUTPUT_DIR / "deepseek_results_v4.jsonl"

# 本次运行失败记录
FAILED_FILE = OUTPUT_DIR / "deepseek_failed_v4.csv"

# 年报分析输出与断点
ANNUAL_REPORT_OUTPUT_FILE = OUTPUT_DIR / "annual_report_fintech_analysis_v4.csv"
ANNUAL_REPORT_CHECKPOINT_FILE = OUTPUT_DIR / "annual_report_fintech_results_v4.jsonl"
ANNUAL_REPORT_FAILED_FILE = OUTPUT_DIR / "annual_report_fintech_failed_v4.csv"
ANNUAL_REPORT_DIR = OUTPUT_DIR / "annual_reports"
ANNUAL_REPORT_TEXT_DIR = OUTPUT_DIR / "annual_report_text"

# 安全要求：API Key只从环境变量读取，不在源码中保留默认值
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2b24159721004658816ce8608137a77c").strip()
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
).strip()

# Tushare兼容接口配置：
# - TUSHARE_TOKEN：只通过环境变量传入，不要写入源码。
# - TUSHARE_HTTP_URL：第三方兼容网关或自建网关地址；留空则使用SDK默认地址。
# 当前购买方提供的示例地址为：https://tx.xiaodefa.top/
# 没有Token时，程序仍会分析ANNUAL_REPORT_DIR内已存在的本地PDF。
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "828df0fd34bce93075c85fd21f691936ac43818c6f1c491c9ca3da67").strip()
TUSHARE_HTTP_URL = os.getenv("TUSHARE_HTTP_URL", "https://tx.xiaodefa.top/").strip()

# True：对识别出的金融机构/金融相关主体自动执行年报检测
ENABLE_ANNUAL_REPORT_ANALYSIS = True

# local_or_tushare：先找本地PDF，找不到再使用Tushare下载
# local_only：只分析本地PDF
ANNUAL_REPORT_SOURCE = os.getenv(
    "ANNUAL_REPORT_SOURCE",
    "local_or_tushare",
).strip().lower()

# 输入数据没有有效报告期时使用的兜底年度
DEFAULT_ANNUAL_REPORT_YEAR = int(
    os.getenv("ANNUAL_REPORT_YEAR", "2025")
)

# 首次测试可设置为10或20；正式运行改为None
MAX_ANNUAL_REPORT_COMPANIES: int | None = None

# 每份年报最多发送给模型的证据片段数
MAX_ANNUAL_REPORT_EVIDENCE = 24
MAX_ANNUAL_REPORT_OUTPUT_EVIDENCE = 8
ANNUAL_REPORT_CONTEXT_CHARS = 180
MIN_PDF_TEXT_CHARS = 1500

# True：上次未找到年报或PDF需OCR时，本次仍重新尝试
RETRY_INCOMPLETE_ANNUAL_REPORTS = True

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
# 2.1 金融机构识别与年报关键词
# ============================================================

FINANCIAL_ENTITY_CATEGORY_KEYWORDS = {
    "银行": ["银行", "农商行", "城商行", "农村商业银行"],
    "证券": ["证券", "券商", "证券经纪", "证券投资银行"],
    "保险": ["保险", "人寿保险", "财产保险", "再保险"],
    "信托": ["信托"],
    "基金": ["基金管理", "公募基金", "基金销售"],
    "期货": ["期货", "期货经纪"],
    "金融租赁": ["金融租赁", "融资租赁"],
    "消费金融": ["消费金融"],
    "金融控股": ["金融控股", "金控"],
    "资产管理": ["资产管理", "财富管理", "投资管理"],
    "支付清算": ["支付机构", "第三方支付", "银行卡收单", "清算机构"],
    "征信评级": ["征信", "信用评级"],
    "担保保理小贷": ["融资担保", "商业保理", "小额贷款", "小贷"],
    "其他金融相关": ["金融信息服务", "金融数据服务", "投资咨询", "财务公司"],
}

# 公司名称只采用较强模式，避免“金融街”等普通名称误判。
FINANCIAL_ENTITY_NAME_PATTERNS = {
    "银行": [r"银行(?:股份有限公司)?$", r"农商行$"],
    "证券": [r"证券(?:股份有限公司)?$"],
    "保险": [r"保险(?:股份有限公司)?$", r"人寿$", r"财险$"],
    "信托": [r"信托(?:股份有限公司)?$"],
    "期货": [r"期货(?:股份有限公司)?$"],
    "金融租赁": [r"金融租赁(?:股份有限公司)?$"],
    "消费金融": [r"消费金融(?:股份有限公司)?$"],
}

REGULATED_FINANCIAL_CATEGORIES = {
    "银行", "证券", "保险", "信托", "基金", "期货",
    "金融租赁", "消费金融", "金融控股",
}

FINTECH_SERVICE_ENTITY_KEYWORDS = sorted(set(
    STRONG_FINTECH_KEYWORDS
    + [
        "银行数字化", "证券数字化", "保险数字化", "金融软件",
        "金融系统", "金融云", "金融信息化", "金融数据平台",
    ]
))

ANNUAL_REPORT_KEYWORD_GROUPS = {
    "金融科技直接表述": [
        "金融科技", "fintech", "数字金融", "数字银行", "开放银行",
        "保险科技", "证券科技", "监管科技",
    ],
    "人工智能与大模型": [
        "人工智能", "生成式人工智能", "大模型", "基础模型", "行业模型",
        "机器学习", "深度学习", "自然语言处理", "知识图谱", "智能问答",
        "智能客服", "智能营销", "智能决策", "算法模型",
    ],
    "智能风控与合规": [
        "智能风控", "智能反欺诈", "反欺诈", "风险模型", "信用评分",
        "智能授信", "智能审贷", "反洗钱系统", "监管报送", "合规科技",
    ],
    "数字化渠道与业务": [
        "数字化转型", "数字化经营", "线上化", "手机银行", "网上银行",
        "远程银行", "数字化渠道", "数字化运营", "线上服务平台",
    ],
    "数据平台与数据治理": [
        "数据中台", "数据平台", "大数据平台", "数据仓库", "数据湖",
        "数据治理", "数据资产", "数据要素", "实时计算", "数据分析平台",
    ],
    "云计算与技术基础设施": [
        "云计算", "金融云", "私有云", "混合云", "分布式架构",
        "核心系统", "信息系统建设", "科技基础设施", "微服务",
    ],
    "支付清算与数字人民币": [
        "数字人民币", "支付系统", "清算系统", "聚合支付", "移动支付",
        "收单系统", "跨境支付", "支付清算",
    ],
    "投顾投研与财富科技": [
        "智能投顾", "智能投研", "智能选股", "财富管理平台",
        "资产配置系统", "投研平台", "量化交易", "程序化交易",
    ],
    "科技投入与组织建设": [
        "科技投入", "信息科技投入", "科技人员", "信息科技人员",
        "金融科技部", "信息科技部", "数字金融部", "数据管理部",
        "科技子公司", "研发投入", "研发人员",
    ],
}

ANNUAL_REPORT_KEYWORD_WEIGHTS = {
    "金融科技直接表述": 6,
    "人工智能与大模型": 5,
    "智能风控与合规": 4,
    "支付清算与数字人民币": 4,
    "投顾投研与财富科技": 4,
    "数字化渠道与业务": 3,
    "数据平台与数据治理": 3,
    "云计算与技术基础设施": 3,
    "科技投入与组织建设": 2,
}

ANNUAL_REPORT_TITLE_EXCLUDE_TERMS = [
    "摘要", "英文版", "英文", "问询函", "回复公告", "更正公告",
    "董事会", "监事会", "审计委员会", "业绩说明会", "社会责任报告",
    "环境、社会及管治", "ESG报告", "可持续发展报告", "取消公告",
]

ANNUAL_REPORT_ALLOWED_TRACKS = {
    "人工智能与大模型",
    "智能风控与反欺诈",
    "数字化渠道与开放银行",
    "数据平台与数据治理",
    "云计算与技术基础设施",
    "智能投顾与财富管理",
    "支付清算与数字人民币",
    "监管科技与合规",
    "保险科技",
    "证券科技",
    "科技投入与组织建设",
    "其他金融科技",
}

ANNUAL_REPORT_STRENGTH_LEVELS = {
    "substantial", "moderate", "mentioned", "none"
}

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


ANNUAL_REPORT_SYSTEM_PROMPT = """
你是一名谨慎的金融机构数字化与金融科技研究员。

用户会提供一家金融机构或金融相关公司的年度报告关键词证据片段，每条片段都有
唯一证据编号和页码。你必须只依据这些片段判断该机构是否存在实质性的金融科技
活动，不得使用外部知识，不得补充输入中不存在的事实。

这里判断的是“金融机构内部的金融科技建设或应用”，并不等同于判断该公司是
金融科技供应商。

请重点区分：
1. “金融科技、数字金融、人工智能、智能风控、数据平台、云计算、数字人民币、
   智能投顾、科技投入和科技组织”等具体建设或应用，属于有效信号；
2. “科技金融”如果只是向科技企业提供贷款或金融支持，不等于金融科技；
3. 仅出现行业趋势、监管政策、风险提示、客户案例或泛泛宣传，不能认定为公司
   已经实际开展相关建设；
4. 必须有公司自身的系统、平台、产品、投入、人员、组织、项目或应用证据，才能
   给出较高置信度；
5. evidence_ids只能选择用户提供的证据编号，最多8个，不得编造编号。

tracks只能从以下值中选择：
- 人工智能与大模型
- 智能风控与反欺诈
- 数字化渠道与开放银行
- 数据平台与数据治理
- 云计算与技术基础设施
- 智能投顾与财富管理
- 支付清算与数字人民币
- 监管科技与合规
- 保险科技
- 证券科技
- 科技投入与组织建设
- 其他金融科技

必须返回合法json对象，不要返回Markdown：
{
  "annual_report_fintech_activity": true,
  "activity_strength": "substantial",
  "ai_evidence_level": "explicit",
  "tracks": ["人工智能与大模型", "智能风控与反欺诈"],
  "confidence": 86,
  "evidence_ids": ["E1", "E4"],
  "reason": "年报披露了公司自身的大模型应用和智能风控系统建设。",
  "needs_job_analysis": true
}

字段约束：
- annual_report_fintech_activity：布尔值；
- activity_strength：只能是substantial、moderate、mentioned、none；
- ai_evidence_level：只能是explicit、indirect、none；
- tracks：数组；
- confidence：0到100整数；
- evidence_ids：证据编号数组，最多8个；
- reason：简洁说明有效证据和局限；
- needs_job_analysis：是否建议继续通过招聘岗位、组织架构或官网核验。
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

    if (
        MAX_ANNUAL_REPORT_COMPANIES is not None
        and MAX_ANNUAL_REPORT_COMPANIES <= 0
    ):
        raise ValueError(
            "MAX_ANNUAL_REPORT_COMPANIES必须为正整数或None。"
        )

    if ANNUAL_REPORT_SOURCE not in {"local_only", "local_or_tushare"}:
        raise ValueError(
            "ANNUAL_REPORT_SOURCE只能是local_only或local_or_tushare。"
        )

    if TUSHARE_HTTP_URL and not re.match(
        r"^https?://",
        TUSHARE_HTTP_URL,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            "TUSHARE_HTTP_URL必须以http://或https://开头。"
        )


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




def sanitize_filename(value: str, max_length: int = 160) -> str:
    value = re.sub(r'[\\/:*?"<>|]+', "_", safe_text(value))
    value = re.sub(r"\s+", " ", value).strip(" ._")
    return (value or "unnamed")[:max_length]


def join_unique(values: list[str], separator: str = "、") -> str:
    return separator.join(dict.fromkeys(value for value in values if value))


def parse_report_year(record: dict[str, Any]) -> int:
    period = safe_text(record.get("report_period"))
    match = re.search(r"(20\d{2})", period)
    if match:
        return int(match.group(1))
    return DEFAULT_ANNUAL_REPORT_YEAR


def evaluate_financial_entity(
    name: str,
    industry: str,
    business_items: list[str],
) -> dict[str, Any]:
    normalized_industry = normalize_text(industry)
    normalized_business = normalize_text(" ".join(business_items))
    normalized_name = safe_text(name).strip()

    industry_categories: dict[str, list[str]] = {}
    business_related_categories: dict[str, list[str]] = {}

    for category, keywords in FINANCIAL_ENTITY_CATEGORY_KEYWORDS.items():
        industry_hits = [
            keyword
            for keyword in keywords
            if normalize_text(keyword) in normalized_industry
        ]
        if industry_hits:
            industry_categories[category] = sorted(set(industry_hits))

        # 业务文本可以识别金融相关主体，但不能仅因“银行核心系统”
        # 就把软件供应商认定为银行等持牌金融机构。
        if category not in REGULATED_FINANCIAL_CATEGORIES:
            business_hits = [
                keyword
                for keyword in keywords
                if normalize_text(keyword) in normalized_business
            ]
            if business_hits:
                business_related_categories[category] = sorted(set(business_hits))

    name_categories: dict[str, list[str]] = {}
    for category, patterns in FINANCIAL_ENTITY_NAME_PATTERNS.items():
        matched = [
            pattern
            for pattern in patterns
            if re.search(pattern, normalized_name)
        ]
        if matched:
            name_categories[category] = matched

    regulated_categories = list(dict.fromkeys([
        *[
            category
            for category in industry_categories
            if category in REGULATED_FINANCIAL_CATEGORIES
        ],
        *name_categories.keys(),
    ]))

    related_categories = list(dict.fromkeys([
        *[
            category
            for category in industry_categories
            if category not in REGULATED_FINANCIAL_CATEGORIES
        ],
        *business_related_categories.keys(),
    ]))

    fintech_service_hits = [
        keyword
        for keyword in FINTECH_SERVICE_ENTITY_KEYWORDS
        if normalize_text(keyword) in normalized_business
    ]
    if fintech_service_hits:
        related_categories.append("金融科技服务商")

    related_categories = list(dict.fromkeys(related_categories))
    categories = list(dict.fromkeys([
        *regulated_categories,
        *related_categories,
    ]))
    related = bool(categories)

    if regulated_categories:
        entity_type = "金融机构"
        reason = "公司行业或强名称模式命中传统/持牌金融机构规则"
    elif related_categories:
        entity_type = "金融相关主体"
        reason = "行业或主营业务命中金融相关业务/金融科技服务规则"
    else:
        entity_type = "非金融主体"
        reason = "行业、主营业务和强公司名称模式均未命中金融主体规则"

    keyword_hits = [
        keyword
        for hits in [
            *industry_categories.values(),
            *business_related_categories.values(),
        ]
        for keyword in hits
    ]
    keyword_hits.extend(fintech_service_hits)

    return {
        "financial_entity_related": related,
        "financial_entity_type": entity_type,
        "financial_entity_categories": "、".join(categories),
        "financial_entity_keyword_hits": join_unique(keyword_hits),
        "financial_entity_name_rule_hits": "、".join(name_categories.keys()),
        "financial_entity_reason": reason,
        "annual_report_analysis_candidate": related,
        "annual_report_or_job_analysis_recommended": related,
        "further_research_suggestion": (
            "建议结合年度报告的信息科技投入、数字化转型、人工智能/大模型、"
            "数据与风控系统，以及招聘岗位和组织架构进一步核验。"
            if related else ""
        ),
    }


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
        financial_entity_result = evaluate_financial_entity(
            name=name,
            industry=industry,
            business_items=business_items,
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
            **financial_entity_result,
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
# 13. 金融机构年报自动检测
# ============================================================

def annual_report_checkpoint_key(ts_code: str, report_year: int) -> str:
    return f"{ts_code}|{report_year}"


def load_annual_report_completed() -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not ANNUAL_REPORT_CHECKPOINT_FILE.exists():
        return completed

    with ANNUAL_REPORT_CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = safe_text(row.get("checkpoint_key"))
            if key:
                completed[key] = row

    return completed


def append_annual_report_checkpoint(result: dict[str, Any]) -> None:
    with ANNUAL_REPORT_CHECKPOINT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def should_skip_annual_report_result(result: dict[str, Any]) -> bool:
    if not RETRY_INCOMPLETE_ANNUAL_REPORTS:
        return True

    status = safe_text(result.get("annual_report_analysis_status"))
    return status in {"completed", "completed_rule_no_signal"}


def annual_report_title_score(title: str, report_year: int) -> int:
    title = safe_text(title)
    normalized = normalize_text(title)

    if "年度报告" not in title:
        return -10_000

    if any(normalize_text(term) in normalized for term in ANNUAL_REPORT_TITLE_EXCLUDE_TERMS):
        return -10_000

    score = 10
    if f"{report_year}年年度报告" in title:
        score += 60
    if "年度报告全文" in title:
        score += 10
    if any(term in title for term in ["修订版", "修订稿", "更新后", "更正版"]):
        score += 15

    return score


def find_local_annual_report(
    record: dict[str, Any],
    report_year: int,
) -> Path | None:
    ts_code = safe_text(record["ts_code"])
    symbol = ts_code.split(".")[0]
    name = safe_text(record.get("name"))

    if not ANNUAL_REPORT_DIR.exists():
        return None

    candidates: list[tuple[int, float, Path]] = []
    for path in ANNUAL_REPORT_DIR.rglob("*.pdf"):
        filename = path.name
        normalized = normalize_text(filename)

        code_hit = normalize_text(ts_code) in normalized or symbol in normalized
        name_hit = bool(name) and normalize_text(name) in normalized
        year_hit = str(report_year) in filename

        # 优先严格匹配代码+年度；公司独立子目录中允许仅年度匹配。
        parent_hit = symbol in normalize_text(str(path.parent))
        if not ((code_hit and year_hit) or (parent_hit and year_hit) or (name_hit and year_hit)):
            continue

        if any(normalize_text(term) in normalized for term in ANNUAL_REPORT_TITLE_EXCLUDE_TERMS):
            continue

        score = 0
        if code_hit:
            score += 40
        if name_hit:
            score += 20
        if year_hit:
            score += 30
        if "年度报告" in filename:
            score += 20
        if any(term in filename for term in ["修订版", "修订稿", "更新后", "更正版"]):
            score += 10

        candidates.append((score, path.stat().st_mtime, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def get_tushare_client() -> Any:
    """创建Tushare客户端，并按需切换到兼容HTTP网关。"""
    if not TUSHARE_TOKEN:
        raise RuntimeError(
            "未设置TUSHARE_TOKEN，无法自动下载年报；可将PDF放入"
            f"{ANNUAL_REPORT_DIR}后使用local_only模式。"
        )

    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError(
            "未安装tushare，请执行：pip install -U tushare"
        ) from exc

    pro = ts.pro_api(TUSHARE_TOKEN)

    if TUSHARE_HTTP_URL:
        # Tushare SDK当前没有公开的base_url构造参数。
        # 兼容网关通常通过DataApi内部HTTP地址切换。
        http_url = TUSHARE_HTTP_URL.rstrip("/") + "/"

        if not hasattr(pro, "_DataApi__http_url"):
            raise RuntimeError(
                "当前tushare版本不支持_DataApi__http_url。"
                "请先执行：pip install -U tushare"
            )

        pro._DataApi__http_url = http_url

    return pro


def query_tushare_annual_report(
    pro: Any,
    record: dict[str, Any],
    report_year: int,
) -> dict[str, Any] | None:
    today = date.today().strftime("%Y%m%d")
    start_date = f"{report_year}0101"
    planned_end = f"{report_year + 1}1231"
    end_date = min(today, planned_end)

    if end_date < start_date:
        return None

    announcements = pro.anns_d(
        ts_code=record["ts_code"],
        start_date=start_date,
        end_date=end_date,
    )

    if announcements is None or announcements.empty:
        return None

    rows: list[dict[str, Any]] = []
    for row in announcements.to_dict("records"):
        score = annual_report_title_score(
            safe_text(row.get("title")),
            report_year,
        )
        if score <= -10_000:
            continue
        row = dict(row)
        row["_title_score"] = score
        rows.append(row)

    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            int(row.get("_title_score", 0)),
            safe_text(row.get("ann_date")),
            safe_text(row.get("rec_time")),
        ),
        reverse=True,
    )
    return rows[0]


def download_annual_report(
    url: str,
    output_path: Path,
) -> None:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "未安装requests，请执行：pip install -U requests"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".pdf.part")

    response = requests.get(
        url,
        timeout=120,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    content = response.content
    if not content.startswith(b"%PDF"):
        raise RuntimeError("公告URL返回内容不是PDF文件。")

    temp_path.write_bytes(content)
    temp_path.replace(output_path)


def resolve_annual_report(
    record: dict[str, Any],
    report_year: int,
    tushare_client_holder: dict[str, Any],
) -> dict[str, Any]:
    local_path = find_local_annual_report(record, report_year)
    if local_path is not None:
        return {
            "annual_report_found": True,
            "annual_report_source": "local",
            "annual_report_title": local_path.stem,
            "annual_report_ann_date": "",
            "annual_report_url": "",
            "annual_report_local_path": str(local_path),
        }

    if ANNUAL_REPORT_SOURCE == "local_only" or not TUSHARE_TOKEN:
        return {
            "annual_report_found": False,
            "annual_report_source": (
                "local" if ANNUAL_REPORT_SOURCE == "local_only"
                else "local_no_tushare_token"
            ),
            "annual_report_title": "",
            "annual_report_ann_date": "",
            "annual_report_url": "",
            "annual_report_local_path": "",
        }

    if "client" not in tushare_client_holder:
        tushare_client_holder["client"] = get_tushare_client()

    announcement = query_tushare_annual_report(
        pro=tushare_client_holder["client"],
        record=record,
        report_year=report_year,
    )
    if announcement is None:
        return {
            "annual_report_found": False,
            "annual_report_source": "tushare",
            "annual_report_title": "",
            "annual_report_ann_date": "",
            "annual_report_url": "",
            "annual_report_local_path": "",
        }

    title = safe_text(announcement.get("title"))
    url = safe_text(announcement.get("url"))
    ann_date = safe_text(announcement.get("ann_date"))
    if not url:
        raise RuntimeError("Tushare公告记录未返回PDF URL。")

    filename = sanitize_filename(
        f"{record['ts_code']}_{report_year}_{ann_date}_{title}"
    ) + ".pdf"
    output_path = ANNUAL_REPORT_DIR / record["ts_code"] / filename

    if not output_path.exists():
        download_annual_report(url, output_path)

    return {
        "annual_report_found": True,
        "annual_report_source": "tushare",
        "annual_report_title": title,
        "annual_report_ann_date": ann_date,
        "annual_report_url": url,
        "annual_report_local_path": str(output_path),
    }


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "未安装PyMuPDF，请执行：pip install -U pymupdf"
        ) from exc

    pages: list[dict[str, Any]] = []
    total_chars = 0

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            try:
                text = page.get_text("text", sort=True)
            except Exception:
                text = ""

            text = clean_pdf_text(text)
            total_chars += len(text)
            pages.append({
                "page": page_index + 1,
                "text": text,
            })

    return pages, total_chars


def save_extracted_text(
    record: dict[str, Any],
    report_year: int,
    pages: list[dict[str, Any]],
) -> Path:
    ANNUAL_REPORT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANNUAL_REPORT_TEXT_DIR / (
        f"{sanitize_filename(record['ts_code'])}_{report_year}.txt"
    )

    content = "\n\n".join(
        f"===== PAGE {page['page']} =====\n{page['text']}"
        for page in pages
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def normalize_snippet_text(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


def collect_annual_report_evidence(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for page in pages:
        page_number = int(page["page"])
        text = safe_text(page["text"])
        if not text:
            continue

        lower_text = text.lower()
        for group, keywords in ANNUAL_REPORT_KEYWORD_GROUPS.items():
            for keyword in keywords:
                search_keyword = keyword.lower()
                start = 0
                occurrences = 0

                while occurrences < 4:
                    index = lower_text.find(search_keyword, start)
                    if index < 0:
                        break

                    left = max(0, index - ANNUAL_REPORT_CONTEXT_CHARS)
                    right = min(
                        len(text),
                        index + len(keyword) + ANNUAL_REPORT_CONTEXT_CHARS,
                    )
                    snippet = normalize_snippet_text(text[left:right])
                    dedupe_key = (page_number, group, normalize_text(snippet))

                    if snippet and dedupe_key not in seen:
                        seen.add(dedupe_key)
                        evidence.append({
                            "page": page_number,
                            "group": group,
                            "keyword": keyword,
                            "snippet": snippet,
                            "score": ANNUAL_REPORT_KEYWORD_WEIGHTS.get(group, 1),
                        })

                    occurrences += 1
                    start = index + max(len(keyword), 1)

    evidence.sort(
        key=lambda item: (-int(item["score"]), int(item["page"]), item["keyword"])
    )

    # 避免同页高度相似片段占满输入。
    selected: list[dict[str, Any]] = []
    selected_normalized: list[tuple[str, str]] = []
    for item in evidence:
        normalized = normalize_text(item["snippet"])
        if any(
            item["group"] == previous_group
            and (normalized in previous or previous in normalized)
            for previous_group, previous in selected_normalized
        ):
            continue
        selected.append(item)
        selected_normalized.append((item["group"], normalized))
        if len(selected) >= MAX_ANNUAL_REPORT_EVIDENCE:
            break

    for index, item in enumerate(selected, start=1):
        item["evidence_id"] = f"E{index}"

    return selected


def build_annual_report_rule_summary(
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = list(dict.fromkeys(item["group"] for item in evidence))
    keywords = list(dict.fromkeys(item["keyword"] for item in evidence))
    pages = sorted(set(int(item["page"]) for item in evidence))

    direct_count = sum(
        item["group"] == "金融科技直接表述"
        for item in evidence
    )
    high_value_groups = {
        "人工智能与大模型",
        "智能风控与合规",
        "支付清算与数字人民币",
        "投顾投研与财富科技",
    }
    high_value_count = sum(item["group"] in high_value_groups for item in evidence)

    rule_signal = bool(
        direct_count > 0
        or high_value_count > 0
        or len(groups) >= 2
    )

    return {
        "annual_report_rule_signal": rule_signal,
        "annual_report_keyword_hit_count": len(evidence),
        "annual_report_keyword_groups": "、".join(groups),
        "annual_report_keyword_hits": "、".join(keywords),
        "annual_report_keyword_pages": "、".join(str(page) for page in pages),
    }


def build_annual_report_user_prompt(
    record: dict[str, Any],
    report_meta: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    evidence_text = "\n\n".join(
        (
            f"[{item['evidence_id']}] 页码：{item['page']}；"
            f"关键词组：{item['group']}；命中词：{item['keyword']}\n"
            f"原文片段：{item['snippet']}"
        )
        for item in evidence
    )

    return f"""
请只依据以下年度报告证据片段返回规定格式的json对象。

股票代码：{record['ts_code']}
公司名称：{record['name']}
基础行业：{record['industry']}
金融主体类型：{record['financial_entity_type']}
金融主体类别：{record['financial_entity_categories']}
年度报告年度：{parse_report_year(record)}
年度报告标题：{report_meta.get('annual_report_title', '')}

判断目标：
- 判断公司自身是否在年报中披露了实质性的金融科技建设或应用；
- 不要把传统金融机构自动认定为金融科技供应商；
- “科技金融”若仅指支持科技企业融资，不属于金融科技活动；
- evidence_ids只能从下列编号中选择。

证据片段：

{evidence_text or '没有提取到关键词证据片段'}
""".strip()


def validate_annual_report_raw_result(result: dict[str, Any]) -> None:
    required = {
        "annual_report_fintech_activity",
        "activity_strength",
        "ai_evidence_level",
        "tracks",
        "confidence",
        "evidence_ids",
        "reason",
        "needs_job_analysis",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"年报模型结果缺少字段：{sorted(missing)}")

    if not isinstance(result["annual_report_fintech_activity"], bool):
        raise ValueError("annual_report_fintech_activity必须是布尔值。")
    if result["activity_strength"] not in ANNUAL_REPORT_STRENGTH_LEVELS:
        raise ValueError("activity_strength值不合法。")
    if result["ai_evidence_level"] not in AI_EVIDENCE_LEVELS:
        raise ValueError("ai_evidence_level值不合法。")
    if not isinstance(result["tracks"], list):
        raise ValueError("tracks必须是数组。")
    if not isinstance(result["evidence_ids"], list):
        raise ValueError("evidence_ids必须是数组。")
    if not isinstance(result["needs_job_analysis"], bool):
        raise ValueError("needs_job_analysis必须是布尔值。")

    confidence = int(result["confidence"])
    if not 0 <= confidence <= 100:
        raise ValueError("confidence必须在0到100之间。")
    result["confidence"] = confidence


def calibrate_annual_report_result(
    raw_result: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    result = dict(raw_result)
    evidence_by_id = {
        item["evidence_id"]: item
        for item in evidence
    }

    valid_ids = [
        safe_text(item)
        for item in result.get("evidence_ids", [])
        if safe_text(item) in evidence_by_id
    ]
    valid_ids = list(dict.fromkeys(valid_ids))[:MAX_ANNUAL_REPORT_OUTPUT_EVIDENCE]

    tracks = [
        safe_text(track)
        for track in result.get("tracks", [])
        if safe_text(track) in ANNUAL_REPORT_ALLOWED_TRACKS
    ]
    tracks = list(dict.fromkeys(tracks))

    activity = is_true(result.get("annual_report_fintech_activity"))
    confidence = int(result["confidence"])

    if activity and not valid_ids:
        activity = False
        result["activity_strength"] = "none"
        tracks = []
        confidence = min(confidence, 40)
        result["reason"] = (
            safe_text(result.get("reason"))
            + " 模型未提供可追溯的有效证据编号，程序已撤销正面结论。"
        ).strip()

    if not activity:
        tracks = []
        if result["activity_strength"] != "mentioned":
            result["activity_strength"] = "none"

    selected_evidence = [evidence_by_id[item] for item in valid_ids]

    if result["ai_evidence_level"] == "explicit":
        has_explicit_ai = any(
            item["group"] == "人工智能与大模型"
            for item in selected_evidence
        )
        if not has_explicit_ai:
            result["ai_evidence_level"] = "none"
            confidence = min(confidence, 65)

    result["annual_report_fintech_activity"] = activity
    result["tracks"] = tracks
    result["confidence"] = confidence
    result["evidence_ids"] = valid_ids
    result["annual_report_evidence_pages"] = "、".join(
        str(item["page"]) for item in selected_evidence
    )
    result["annual_report_evidence_keywords"] = "、".join(
        item["keyword"] for item in selected_evidence
    )
    result["annual_report_evidence"] = " || ".join(
        f"P{item['page']} [{item['keyword']}] {item['snippet']}"
        for item in selected_evidence
    )
    result["needs_job_analysis"] = is_true(result.get("needs_job_analysis"))

    return result


def classify_annual_report(
    client: OpenAI,
    record: dict[str, Any],
    report_meta: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": ANNUAL_REPORT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_annual_report_user_prompt(
                            record=record,
                            report_meta=report_meta,
                            evidence=evidence,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("年报模型返回空content。")

            raw_result = json.loads(content)
            validate_annual_report_raw_result(raw_result)
            result = calibrate_annual_report_result(raw_result, evidence)

            usage = response.usage
            result.update({
                "annual_report_model": MODEL_NAME,
                "annual_report_prompt_tokens": getattr(usage, "prompt_tokens", None),
                "annual_report_completion_tokens": getattr(usage, "completion_tokens", None),
                "annual_report_total_tokens": getattr(usage, "total_tokens", None),
            })
            return result

        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            wait_seconds = min(2 ** attempt, 30) + random.random()
            print(f"  年报模型第{attempt}/{MAX_RETRIES}次失败：{exc}")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"DeepSeek年报分析连续调用失败：{last_error}"
    ) from last_error


def build_base_annual_report_result(
    record: dict[str, Any],
    report_year: int,
) -> dict[str, Any]:
    return {
        "checkpoint_key": annual_report_checkpoint_key(
            record["ts_code"], report_year
        ),
        "ts_code": record["ts_code"],
        "name": record["name"],
        "industry": record["industry"],
        "report_year": report_year,
        "financial_entity_related": record["financial_entity_related"],
        "financial_entity_type": record["financial_entity_type"],
        "financial_entity_categories": record["financial_entity_categories"],
        "annual_report_fintech_activity": False,
        "activity_strength": "none",
        "ai_evidence_level": "none",
        "tracks": [],
        "confidence": 0,
        "evidence_ids": [],
        "annual_report_evidence_pages": "",
        "annual_report_evidence_keywords": "",
        "annual_report_evidence": "",
        "reason": "",
        "needs_job_analysis": True,
    }


def analyze_one_annual_report(
    client: OpenAI,
    record: dict[str, Any],
    tushare_client_holder: dict[str, Any],
) -> dict[str, Any]:
    report_year = parse_report_year(record)
    result = build_base_annual_report_result(record, report_year)

    report_meta = resolve_annual_report(
        record=record,
        report_year=report_year,
        tushare_client_holder=tushare_client_holder,
    )
    result.update(report_meta)

    if not report_meta["annual_report_found"]:
        result.update({
            "annual_report_analysis_status": "no_report",
            "reason": (
                "未找到对应年度报告PDF。可配置TUSHARE_TOKEN，或将PDF放入"
                f"{ANNUAL_REPORT_DIR}。"
            ),
        })
        return result

    pdf_path = Path(report_meta["annual_report_local_path"])
    pages, total_chars = extract_pdf_pages(pdf_path)
    text_path = save_extracted_text(record, report_year, pages)

    result["annual_report_pdf_page_count"] = len(pages)
    result["annual_report_extracted_chars"] = total_chars
    result["annual_report_text_path"] = str(text_path)

    if total_chars < MIN_PDF_TEXT_CHARS:
        result.update({
            "annual_report_analysis_status": "needs_ocr",
            "reason": (
                "PDF可提取文本过少，可能是扫描版或字体编码异常，建议先OCR后重试。"
            ),
        })
        return result

    evidence = collect_annual_report_evidence(pages)
    rule_summary = build_annual_report_rule_summary(evidence)
    result.update(rule_summary)

    if not evidence:
        result.update({
            "annual_report_analysis_status": "completed_rule_no_signal",
            "confidence": 55,
            "reason": "年报可提取文本中未发现预设金融科技关键词证据。",
            "needs_job_analysis": True,
        })
        return result

    model_result = classify_annual_report(
        client=client,
        record=record,
        report_meta=report_meta,
        evidence=evidence,
    )
    result.update(model_result)
    result["annual_report_analysis_status"] = "completed"
    return result


def load_annual_report_results_as_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not ANNUAL_REPORT_CHECKPOINT_FILE.exists():
        return pd.DataFrame()

    with ANNUAL_REPORT_CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
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

    result_df = result_df.drop_duplicates(
        subset=["checkpoint_key"],
        keep="last",
    ).reset_index(drop=True)

    for column in ["tracks", "evidence_ids"]:
        if column in result_df.columns:
            result_df[column] = result_df[column].apply(
                lambda value: "、".join(value)
                if isinstance(value, list)
                else value
            )

    return result_df


def export_financial_entities(
    company_records: list[dict[str, Any]],
) -> None:
    rows = []
    for record in company_records:
        if not record["financial_entity_related"]:
            continue
        rows.append({
            "ts_code": record["ts_code"],
            "name": record["name"],
            "industry": record["industry"],
            "market": record["market"],
            "exchange": record["exchange"],
            "report_period": record["report_period"],
            "financial_entity_related": record["financial_entity_related"],
            "financial_entity_type": record["financial_entity_type"],
            "financial_entity_categories": record["financial_entity_categories"],
            "financial_entity_keyword_hits": record["financial_entity_keyword_hits"],
            "financial_entity_name_rule_hits": record["financial_entity_name_rule_hits"],
            "financial_entity_reason": record["financial_entity_reason"],
            "annual_report_analysis_candidate": record["annual_report_analysis_candidate"],
            "annual_report_or_job_analysis_recommended": record[
                "annual_report_or_job_analysis_recommended"
            ],
            "further_research_suggestion": record["further_research_suggestion"],
            "prefilter_passed": record["prefilter_passed"],
            "business_items": "、".join(record["business_items"]),
        })

    pd.DataFrame(rows).to_csv(
        FINANCIAL_ENTITY_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def export_annual_report_results() -> None:
    result_df = load_annual_report_results_as_dataframe()
    if result_df.empty:
        print("没有可导出的年报分析结果。")
        return

    sort_columns = [
        column
        for column in [
            "annual_report_fintech_activity",
            "confidence",
            "financial_entity_type",
        ]
        if column in result_df.columns
    ]
    if sort_columns:
        ascending = [False if column != "financial_entity_type" else True for column in sort_columns]
        result_df = result_df.sort_values(sort_columns, ascending=ascending)

    result_df.to_csv(
        ANNUAL_REPORT_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  年报金融科技检测：{ANNUAL_REPORT_OUTPUT_FILE}")


def run_annual_report_analysis(
    client: OpenAI,
    company_records: list[dict[str, Any]],
) -> None:
    if not ENABLE_ANNUAL_REPORT_ANALYSIS:
        print("\n年报自动检测已关闭。")
        return

    target_records = [
        record
        for record in company_records
        if record["annual_report_analysis_candidate"]
    ]

    if MAX_ANNUAL_REPORT_COMPANIES is not None:
        target_records = target_records[:MAX_ANNUAL_REPORT_COMPANIES]

    completed = load_annual_report_completed()
    failed_rows: list[dict[str, Any]] = []
    tushare_client_holder: dict[str, Any] = {}

    print("\n开始金融机构年报自动检测：")
    print(f"  年报候选公司：{len(target_records):,}")
    print(f"  年报来源模式：{ANNUAL_REPORT_SOURCE}")
    print(
        "  Tushare接入地址："
        f"{TUSHARE_HTTP_URL or 'SDK默认地址'}"
    )
    print(f"  本地年报目录：{ANNUAL_REPORT_DIR}")

    total = len(target_records)
    for index, record in enumerate(target_records, start=1):
        report_year = parse_report_year(record)
        key = annual_report_checkpoint_key(record["ts_code"], report_year)

        existing = completed.get(key)
        if existing and should_skip_annual_report_result(existing):
            print(
                f"[{index}/{total}] 跳过已完成年报："
                f"{record['ts_code']} {record['name']}"
            )
            continue

        print(
            f"[{index}/{total}] 分析年报："
            f"{record['ts_code']} {record['name']} {report_year}"
        )

        try:
            result = analyze_one_annual_report(
                client=client,
                record=record,
                tushare_client_holder=tushare_client_holder,
            )
            append_annual_report_checkpoint(result)
            completed[key] = result
            print(
                "  状态="
                f"{result['annual_report_analysis_status']}，"
                "金融科技活动="
                f"{result['annual_report_fintech_activity']}，"
                "AI证据="
                f"{result['ai_evidence_level']}，"
                "置信度="
                f"{result['confidence']}"
            )

        except Exception as exc:
            print(f"  年报处理失败：{exc}")
            failure_result = build_base_annual_report_result(record, report_year)
            failure_result.update({
                "annual_report_analysis_status": "failed",
                "reason": str(exc),
            })
            append_annual_report_checkpoint(failure_result)
            completed[key] = failure_result
            failed_rows.append({
                "ts_code": record["ts_code"],
                "name": record["name"],
                "report_year": report_year,
                "error": str(exc),
            })

        time.sleep(REQUEST_INTERVAL)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(
            ANNUAL_REPORT_FAILED_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"  年报失败记录：{ANNUAL_REPORT_FAILED_FILE}")

    export_annual_report_results()


# ============================================================
# 14. 导出
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
                "business_item_count": len(record["business_items"]),
                "financial_entity_related": record["financial_entity_related"],
                "financial_entity_type": record["financial_entity_type"],
                "financial_entity_categories": record["financial_entity_categories"],
                "annual_report_analysis_candidate": record[
                    "annual_report_analysis_candidate"
                ],
            }
        )

    pd.DataFrame(rows).to_csv(
        PREFILTER_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def export_results(original_df: pd.DataFrame) -> None:
    result_df = load_checkpoint_as_dataframe()

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
    ]

    enriched_df = original_df.merge(
        result_df[available_label_columns],
        on="ts_code",
        how="left",
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

    print("\n公司级分类统计：")

    for column in [
        "fintech_related",
        "fintech_candidate_for_annual_report",
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
# 15. 主程序
# ============================================================

def main() -> None:
    validate_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNUAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ANNUAL_REPORT_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    original_df = load_data()
    company_records = build_company_records(original_df)

    export_prefilter_results(company_records)
    export_financial_entities(company_records)

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

    export_results(original_df)
    print(f"  金融机构主体池：{FINANCIAL_ENTITY_OUTPUT_FILE}")

    run_annual_report_analysis(
        client=client,
        company_records=company_records,
    )

    print("\n全部任务完成。")


if __name__ == "__main__":
    main()
