# -*- coding: utf-8 -*-
"""全局配置：Token、路径、章节切片规则、状态码。

所有文本文件统一以 UTF-8-SIG 写出（带 BOM），保证 Excel / 记事本 直接双击不乱码。

本版相对初版的主要改动
----------------------
1. 「管理层讨论与分析」按**整节**处理：从该节标题开始，到**下一个节标题之前**结束。
2. 「研发投入」按**小节**处理，并识别公司填写「不适用」的情形（单独记录、单独汇报）。
3. 「业务概况」年报中通常没有独立板块，正文切不出来时改由 DeepSeek 高级模型
   （`DEEPSEEK_ADVANCED_MODEL`）对管理层讨论与分析做条目化概括。
4. 切片完成后由 AI 做**后验**（起点 / 终点 / 内容相符 / 完整性）。
5. 所有正文定位改为「标题行锚定 + 评分」，不再单纯相信关键词出现。
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Tushare（第三方代理网关）
# --------------------------------------------------------------------------
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "ba76ba20cb9d445e8d81539eab11a065")
TUSHARE_HTTP_URL = os.environ.get("TUSHARE_URL", "https://ts.gyzcloud.top/api")
TUSHARE_RATE_PER_MIN = 140          # 官方限额 150/min，留出余量
TUSHARE_TIMEOUT = 30

# --------------------------------------------------------------------------
# 巨潮资讯网（年报 PDF 的权威来源，作为 Tushare 公告接口的兜底）
# --------------------------------------------------------------------------
CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
CNINFO_TIMEOUT = 30

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# --------------------------------------------------------------------------
# DeepSeek（AI 辅助定位 / 概括 / 后验）
# --------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 定位用的常规模型（便宜、快）
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
# 概括「业务概况」与切片后验用的高级模型
DEEPSEEK_ADVANCED_MODEL = os.environ.get("DEEPSEEK_ADVANCED_MODEL", "deepseek-v4-pro")
# 高级模型名不被服务端接受时，依次降级尝试（避免整轮任务因模型名失效而全废）
DEEPSEEK_MODEL_FALLBACKS = ["deepseek-v4-pro", "deepseek-reasoner",
                            "deepseek-v4-flash", "deepseek-chat"]
DEEPSEEK_TIMEOUT = 90
DEEPSEEK_RATE_PER_MIN = 60          # 保守限速，避免触发对端限流

# 定位：AI 只给"起点锚句"，终点仍由确定性规则算，幻觉影响被限制在"起点准不准"
AI_MAX_CALLS_PER_REPORT = 6         # 单篇年报「定位」最多调用几次
AI_CHUNK_CHARS = 9000               # 每次喂给模型的正文窗口大小（字符数）
AI_CHUNK_OVERLAP = 400              # 相邻窗口重叠，避免章节标题正好被切断
AI_MAX_SECTION_CHARS = 15000        # AI 定位到起点后，若找不到明确终点，兜底截取的最大长度

# 概括（业务概况）
AI_SUMMARY_MAX_INPUT_CHARS = 26000  # 喂给高级模型的管理层讨论与分析上限，超出则头尾取样
AI_SUMMARY_MAX_TOKENS = 4000

# 后验（切片质检）
AI_VERIFY_ENABLED_DEFAULT = True
AI_VERIFY_MAX_CALLS_PER_REPORT = 6  # 每篇年报最多验几个切片
AI_VERIFY_HEAD_CHARS = 1400         # 送检的切片开头长度
AI_VERIFY_TAIL_CHARS = 1200         # 送检的切片结尾长度
AI_VERIFY_CONTEXT_CHARS = 500       # 切片前后各带多少上下文，用来判断边界对不对
AI_VERIFY_AUTOFIX = True            # 后验发现边界不对且给出可核对的原文引文时，自动重切
AI_VERIFY_MAX_EXPAND_RATIO = 3.0    # 自动重切的长度不得超过原切片的几倍，防跑飞

# --------------------------------------------------------------------------
# 输出目录结构
# --------------------------------------------------------------------------
DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "annual_reports_out")
DIR_RAW = "raw"            # 原始 PDF
DIR_TEXT = "fulltext"      # 全文纯文本
DIR_SECTION = "sections"   # 章节切片
DIR_LOG = "logs"
MANIFEST = "manifest.json"
REPORT_XLSX = "report.xlsx"
REPORT_CSV = "report.csv"

TEXT_ENCODING = "utf-8-sig"

# --------------------------------------------------------------------------
# 任务状态码
# --------------------------------------------------------------------------
class S:
    PENDING = "PENDING"              # 待处理
    RUNNING = "RUNNING"              # 处理中
    OK = "OK"                        # 全部成功
    PARTIAL = "PARTIAL"              # 年报拿到，但部分章节未识别 / 后验存疑
    NO_ANN = "NO_ANN"                # 未检索到该年度年报公告
    DOWNLOAD_FAIL = "DOWNLOAD_FAIL"  # 下载失败
    PDF_BROKEN = "PDF_BROKEN"        # PDF 损坏 / 无法解析
    NO_TEXT = "NO_TEXT"              # 扫描件，无文本层，需 OCR
    NO_SECTION = "NO_SECTION"        # 全文可读但一个目标章节都没切出来
    SKIPPED = "SKIPPED"              # 断点续跑：已存在，跳过
    STOPPED = "STOPPED"              # 用户中止
    ERROR = "ERROR"                  # 其它异常

STATUS_LABEL = {
    S.PENDING: "待处理",
    S.RUNNING: "处理中",
    S.OK: "完成",
    S.PARTIAL: "需复核",
    S.NO_ANN: "未找到年报",
    S.DOWNLOAD_FAIL: "下载失败",
    S.PDF_BROKEN: "文件损坏",
    S.NO_TEXT: "无文本层(需OCR)",
    S.NO_SECTION: "无法识别章节",
    S.SKIPPED: "已存在",
    S.STOPPED: "已中止",
    S.ERROR: "异常",
}

# 视为“需要人工复核”的状态
ABNORMAL = {S.NO_ANN, S.DOWNLOAD_FAIL, S.PDF_BROKEN, S.NO_TEXT,
            S.NO_SECTION, S.PARTIAL, S.ERROR}


# --------------------------------------------------------------------------
# 单个章节的结果状态（与任务状态区分开）
# --------------------------------------------------------------------------
class SS:
    ORIGINAL = "ORIGINAL"            # 按标题精确切出的原文
    AI_LOCATED = "AI_LOCATED"        # AI 帮忙定位起点，正文仍是原文
    AI_SUMMARY = "AI_SUMMARY"        # AI 概括生成（**不是**原文）
    NOT_APPLICABLE = "NA"            # 公司在年报中明确填写「不适用」
    MISSING = "MISSING"              # 没识别到

SS_LABEL = {
    SS.ORIGINAL: "原文切出",
    SS.AI_LOCATED: "AI定位·原文",
    SS.AI_SUMMARY: "AI概括生成",
    SS.NOT_APPLICABLE: "公司填报不适用",
    SS.MISSING: "未识别",
}

# 后验结论
VERIFY_LABEL = {
    "pass": "通过",
    "warn": "存疑",
    "fail": "不通过",
    "fixed": "已按后验修正",
    "skip": "未后验",
    "error": "后验失败",
}


# --------------------------------------------------------------------------
# 「不适用」判定
# --------------------------------------------------------------------------
# 年报里"不适用"有三种写法：
#   1) 小节正文只有"不适用"三个字；
#   2) 勾选框："□适用 √不适用"（勾在"不适用"前面）；
#   3) "是否适用 □是 √否"。
# 注意第 2 种：勾在"适用"前面时（"√适用 □不适用"）是**适用**，不能误判。
NA_TICKS = "√✓☑■●⊠×"
NA_MAX_BODY_CHARS = 60              # 正文短于这个长度才考虑判为"不适用"

# --------------------------------------------------------------------------
# 章节切片规则
# --------------------------------------------------------------------------
# level 取值：
#   "chapter"           只在"第X节"层级定位，命中后整节切出（到下一节标题之前）
#   "chapter_then_item" 先找"第X节"，找不到再到父章节内部按条目层级找
#   "item"              只在父章节（默认 mdna）内部按条目层级找
#
# chapter_patterns / item_patterns 都是**匹配标题行**的正则（在去掉序号后的标题上跑），
# 不是在全文里搜关键词 —— 这是本版避免误识别的关键。列表顺序即优先级。
SECTION_RULES = [
    {
        "key": "mdna",
        "name": "管理层讨论与分析",
        "level": "semantic_section",
        "required": True,
        # 不再依赖“第三节/第四节”。匹配的是标题语义，标题可处在任意层级：
        # 第X节、第X部分、一、（一）、数字序号或无序号加粗标题都可以。
        "chapter_patterns": [
            r"^管理层讨论与分析$",
            r"^管理层讨论及分析$",
            r"^管理层讨论和分析$",
            r"^经营层讨论与分析$",
            r"^经营层讨论及分析$",
            r"^经营情况讨论与分析$",
            r"^经营情况讨论及分析$",
            r"^经营管理层讨论与分析$",
            r"^管理层分析与讨论$",
            r"^董事会报告$",
        ],
        "expect_ordinals": [],       # 章节序号不参与硬判断
        "item_patterns": [],
        "min_chars": 1800,
    },
    {
        "key": "business",
        "name": "业务概况",
        "level": "chapter_then_item",
        "required": False,
        # 旧版式里是独立一节
        "chapter_patterns": [
            r"^公司业务概要$",
            r"^业务概要$",
            r"^公司业务概况$",
            r"^业务概况$",
        ],
        # 新版式里并入管理层讨论与分析，通常是开头的一、二两条
        "item_patterns": [
            r"^报告期内公司(所处|从事)的?(行业|主要业务)(情况)?",
            r"^报告期内公司从事的主要业务",
            r"^报告期内公司所处行业情况",
            r"^公司所处行业情况",
            r"^公司(主要业务|主营业务)(情况|简介)?$",
            r"^主要业务(情况|简介)?$",
            r"^行业情况说明$",
        ],
        "merge_adjacent": True,      # 相邻的两条合并成一段
        "ai_summary_fallback": True, # 正文里切不出来 → 由高级模型概括管理层讨论与分析
        "min_chars": 300,
    },
    {
        "key": "core",
        "name": "核心竞争力",
        "level": "chapter_then_item",
        "required": False,
        "chapter_patterns": [r"^核心竞争力分析$", r"^公司核心竞争力分析$"],
        "item_patterns": [
            r"^核心竞争力分析$",
            r"^公司核心竞争力(分析)?$",
            r"^核心竞争力",
        ],
        "min_chars": 200,
    },
    {
        "key": "rd",
        "name": "研发投入",
        "level": "item",
        "required": False,
        "chapter_patterns": [],
        # 典型形态："四、主营业务分析" → "(五)研发投入" / "5、研发投入"
        "item_patterns": [
            r"^研发投入$",
            r"^研发投入情况$",
            r"^研发投入及专利情况$",
            r"^研发情况$",
            r"^研发支出$",
            r"^研发投入(情况)?(表|明细)?$",
            r"^研发投入",
            r"^公司研发投入",
        ],
        "check_na": True,            # 需要识别公司填「不适用」
        "min_chars": 0,              # 允许极短（"不适用"）
        "parent": "mdna",
    },
]

SECTION_KEYS = [r["key"] for r in SECTION_RULES]
# 任务是否成功只由主目标决定；其余三项保留为兼容性/辅助产物，不再拖累主任务状态。
PRIMARY_SECTION_KEYS = ["mdna"]
SECTION_NAMES = {r["key"]: r["name"] for r in SECTION_RULES}
SECTION_RULE_MAP = {r["key"]: r for r in SECTION_RULES}

# 供 AI 提示词使用的章节说明
SECTION_DESC = {
    "mdna": "管理层讨论与分析（部分年报叫「经营情况讨论与分析」或「董事会报告」），"
            "是年报中篇幅最大的一节，包含行业情况、主营业务分析、财务数据变动、"
            "投资状况、未来展望等内容",
    "business": "公司业务概况：主营业务、主要产品与服务、所处行业、经营模式、市场地位",
    "core": "核心竞争力分析：公司相对同行的竞争优势（技术、渠道、品牌、成本、人才等）",
    "rd": "研发投入：研发人员数量与占比、研发投入金额、研发投入占营业收入比例、"
          "资本化情况、重点研发项目等",
}
