from __future__ import annotations

"""
============================================================
金融科技识别结果 · 二次复核脚本（v6）
============================================================

本脚本用于对第一轮识别流程（classify_ai_fintech_with_annual_report_v4.py /
classify_ai_fintech_with_custom_annual_report_api_v5.py）产出的结果做第二轮更严格的复核，
重点解决两个问题：

1. 第一轮把"金融相关主体"（即非持牌金融机构，只是业务或名称中沾边金融字样、或被识别为
   金融科技服务商的公司）年报/主营业务中出现的通用"智慧化""智能化"表述
   （如智慧园区、智能制造、智慧文旅、智慧物业、OA/ERP智能化、生产线AI质检、
   笼统的"研发投入""研发人员"等）误判为金融科技的问题；
2. 第一轮置信度低于90分的记录，由于初版文本切分较粗（按固定字符窗口截取），
   可能存在证据不完整、上下文被截断的问题，需要用更细颗粒度（按句子切分）重新抽取证据。

设计要点：
- 直接复用第一轮脚本（下称"base模块"）的数据管道与关键词库/枚举常量，避免逻辑不一致；
- 使用比第一轮更强的模型（默认 deepseek-v4-pro，而非 deepseek-v4-flash），
  并保留完整推理（thinking），追求准确率而非速度；
- 年报证据改为"句子级"切分（前一句+命中句+后一句），比第一轮固定字符窗口更完整、更不容易截断语义；
- 新增"金融业务锚定词"与"泛化智能化噪音词"两套词库，对每条证据同时标注，
  并把两类证据分别喂给模型，要求模型只依据真正带有金融锚定的证据做判断；
- 复核范围：
  a) 年报层面：financial_entity_type == "金融相关主体" 的全部记录，
     以及 confidence < 90 的全部记录（两者取并集）；
  b) 主营业务层面：同样按 financial_entity_type == "金融相关主体"（按base规则重新计算，
     与年报层面独立）或 confidence < 90 取并集；
- 复核结果单独落盘，不覆盖第一轮结果，便于人工比对第一轮/复核结论的差异。

运行前置条件：
1. 已经完整运行过第一轮脚本，且至少生成了：
   - data/deepseek_company_classification_v4.csv （主营业务层面结果）
   - data/annual_report_fintech_analysis_v4.csv  （年报层面结果）
   - data/annual_report_text/*.txt               （年报纯文本缓存，年报复核会直接读取，不重新下载/解析PDF）
   - data/stock_pool_main_business_20251231.csv  （原始输入数据，主营业务复核需要重建业务描述）
2. 本脚本需要与第一轮脚本放在同一目录下（本脚本会 import 第一轮脚本模块以复用其函数与常量）；
3. 环境变量 DEEPSEEK_API_KEY 必须设置（出于安全考虑，本脚本不在源码中保留任何默认Key）；
4. 可选环境变量：
   - DEEPSEEK_BASE_URL          ：默认 https://api.deepseek.com
   - DEEPSEEK_RECHECK_MODEL     ：默认 deepseek-v4-pro（更高质量模型，而非flash）
   - DEEPSEEK_RECHECK_THINKING  ：enabled / disabled / omit，默认 enabled
   - BASE_PIPELINE_MODULE       ：手动指定要复用的第一轮脚本模块名

运行方式：
    python recheck_ai_fintech_v6.py

依赖：pandas, openai
"""

import contextlib
import importlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


# ============================================================
# 1. 复核任务总体配置
# ============================================================

# 是否执行年报层面复核 / 主营业务层面复核
RUN_ANNUAL_REPORT_RECHECK = True
RUN_COMPANY_LEVEL_RECHECK = True

# 置信度复核阈值：低于该分数的记录一律纳入复核范围（不论金融主体类型）
CONFIDENCE_THRESHOLD = 90

# 测试阶段可设置较小数值（如 10 / 20），正式跑全量改为 None
RECHECK_MAX_ANNUAL_REPORT_TARGETS: int | None = None
RECHECK_MAX_COMPANY_TARGETS: int | None = None

# 年报证据抽取参数（比第一轮更细、更多）
RECHECK_MAX_ANNUAL_EVIDENCE = 50          # 金融科技候选证据条数上限（第一轮为24）
RECHECK_MAX_NOISE_EVIDENCE = 25           # 泛化智能化噪音证据条数上限
RECHECK_MAX_OCCURRENCES_PER_KEYWORD = 8   # 单个关键词最多命中次数（第一轮为4）
RECHECK_MAX_OUTPUT_EVIDENCE_IDS = 15      # 模型最多可引用的证据编号数（第一轮为8）

# 模型调用参数
RECHECK_MODEL_NAME = os.getenv("DEEPSEEK_RECHECK_MODEL", "deepseek-v4-pro").strip()
# 注意：实测发现部分网关下 deepseek-v4-pro 等推理档模型若开启thinking，容易在思考阶段
# 耗尽max_tokens导致最终content为空（"复核模型返回空content"）。因此这里把默认值改为
# disabled（与第一轮脚本一致，已被大规模验证稳定），如需开启推理请自行设置环境变量
# DEEPSEEK_RECHECK_THINKING=enabled，并相应调大 RECHECK_MAX_OUTPUT_TOKENS。
RECHECK_ENABLE_THINKING = os.getenv("DEEPSEEK_RECHECK_THINKING", "disabled").strip().lower()
RECHECK_MAX_OUTPUT_TOKENS = 4096
RECHECK_MAX_RETRIES = 5
RECHECK_REQUEST_INTERVAL = 0.3
RECHECK_REQUEST_TIMEOUT_SECONDS = 180.0


# ============================================================
# 2. 导入并复用第一轮识别脚本（数据管道 / 关键词库 / 枚举常量）
# ============================================================

def _import_base_module():
    """
    尝试导入第一轮识别脚本模块，优先使用环境变量 BASE_PIPELINE_MODULE 指定的模块名，
    否则依次尝试自定义年报API版(v5)与标准版(v4)。
    """
    explicit_name = os.getenv("BASE_PIPELINE_MODULE", "").strip()
    candidates = [explicit_name] if explicit_name else []
    candidates += [
        "classify_ai_fintech_with_custom_annual_report_api_v5",
        "classify_ai_fintech_with_annual_report_v4",
    ]

    last_error: Exception | None = None
    for module_name in candidates:
        if not module_name:
            continue
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "未能导入第一轮分类脚本模块。请确认 "
        "classify_ai_fintech_with_annual_report_v4.py 或 "
        "classify_ai_fintech_with_custom_annual_report_api_v5.py "
        "与本脚本（recheck_ai_fintech_v6.py）位于同一目录下，"
        "或通过环境变量 BASE_PIPELINE_MODULE 指定要导入的模块名。\n"
        f"原始错误：{last_error}"
    )


try:
    base = _import_base_module()
except ImportError as exc:
    print(f"[致命错误] {exc}")
    sys.exit(1)


# 复用第一轮脚本的输出路径常量，保证与第一轮完全一致，避免路径漂移
COMPANY_LEVEL_INPUT_FILE = base.COMPANY_OUTPUT_FILE
ANNUAL_REPORT_INPUT_FILE = base.ANNUAL_REPORT_OUTPUT_FILE
FINANCIAL_ENTITY_INPUT_FILE = base.FINANCIAL_ENTITY_OUTPUT_FILE  # 仅用于展示参考，不参与主流程判断

RECHECK_OUTPUT_DIR = base.OUTPUT_DIR / "recheck_v6"
RECHECK_ANNUAL_CHECKPOINT = RECHECK_OUTPUT_DIR / "recheck_annual_report_v6.jsonl"
RECHECK_COMPANY_CHECKPOINT = RECHECK_OUTPUT_DIR / "recheck_company_level_v6.jsonl"
RECHECK_ANNUAL_OUTPUT_CSV = RECHECK_OUTPUT_DIR / "recheck_annual_report_v6.csv"
RECHECK_COMPANY_OUTPUT_CSV = RECHECK_OUTPUT_DIR / "recheck_company_level_v6.csv"
RECHECK_FINAL_MERGED_CSV = RECHECK_OUTPUT_DIR / "recheck_final_merged_v6.csv"


# ============================================================
# 2.1 CSV/编码健壮性工具
# ============================================================
#
# 背景：第一轮脚本以 encoding="utf-8-sig" 写出CSV，但在Windows上如果该文件之后被
# Excel/WPS重新打开并保存过，会被默认转存为本地ANSI（简体中文环境下通常是GBK/GB2312），
# 此时用UTF-8读取会抛出 UnicodeDecodeError（如 "can't decode byte 0xb3 ..."）。
# 下面的工具函数会依次尝试常见编码，自动兼容这种情况；同时提供一个临时的
# pandas.read_csv 补丁上下文管理器，用于让第一轮脚本(base模块)的 load_data() 在
# 未显式指定encoding时也具备同样的健壮性。

_CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk"]


def read_csv_robust(path: Path, **kwargs: Any) -> pd.DataFrame:
    """健壮读取CSV：依次尝试 utf-8-sig / utf-8 / gb18030 / gbk，直到成功为止。"""
    last_error: Exception | None = None
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"无法用以下任一编码读取文件 {path}：{_CANDIDATE_ENCODINGS}。\n"
        f"最后一次错误：{last_error}\n"
        "常见原因：该CSV在Windows上被Excel/WPS重新打开并保存过，被转成了本地ANSI(GBK)编码。\n"
        "建议：用 VSCode/Notepad++ 打开该文件，选择「另存为」并指定编码为 UTF-8 后重试；"
        "或直接删除该文件，重新运行第一轮脚本重新生成。"
    )


@contextlib.contextmanager
def robust_csv_encoding_patch():
    """
    临时给 pandas.read_csv 打补丁：调用方若未显式传入 encoding 参数，则自动依次尝试
    utf-8-sig / utf-8 / gb18030 / gbk，直到成功为止。仅在上下文内生效，退出后自动恢复，
    用于兼容 base 模块（第一轮脚本）内部未做编码兜底的 pd.read_csv 调用（如 load_data()）。
    """
    original_read_csv = pd.read_csv

    def _patched_read_csv(*args: Any, **kwargs: Any):
        if kwargs.get("encoding"):
            return original_read_csv(*args, **kwargs)
        last_error: Exception | None = None
        for encoding in _CANDIDATE_ENCODINGS:
            try:
                return original_read_csv(*args, encoding=encoding, **kwargs)
            except (UnicodeDecodeError, UnicodeError) as exc:
                last_error = exc
                continue
        raise last_error  # type: ignore[misc]

    pd.read_csv = _patched_read_csv
    try:
        yield
    finally:
        pd.read_csv = original_read_csv


# ============================================================
# 3. 复核专用关键词库：金融业务锚定词 与 泛化智能化噪音词
# ============================================================
#
# 设计说明：
# 第一轮的 ANNUAL_REPORT_KEYWORD_GROUPS 中包含不少"低信噪比词"，例如"研发投入""研发人员"
# "数据治理""云计算"等——这些词几乎任何行业的公司都可能出现（不限于金融科技），
# 单独命中不能作为金融科技证据。本复核脚本额外维护两套词库：
#   1) FINANCE_ANCHOR_KEYWORDS：真正锚定"金融业务/金融机构/金融监管"场景的词，
#      要求候选证据片段中同时出现锚定词，才算具备较强的金融科技可信度；
#   2) GENERIC_SMART_NOISE_KEYWORDS：明显与金融无关的通用"智慧/智能化"业务场景词，
#      命中这些词但没有金融锚定词同现，大概率是第一轮误判的来源。
#
# 同时覆盖简体/繁体常见写法，因为部分H股年报公告为繁体中文。

FINANCE_ANCHOR_KEYWORDS = sorted(set([
    # 金融机构/牌照类
    "银行", "銀行", "证券", "證券", "券商", "保险", "保險", "信托", "信託", "基金",
    "期货", "期貨", "金融租赁", "金融租賃", "消费金融", "消費金融", "金融控股", "金控",
    "财务公司", "財務公司", "融资租赁", "融資租賃", "小额贷款", "小額貸款", "小贷", "小貸",
    "融资担保", "融資擔保", "商业保理", "商業保理", "征信", "徵信", "信用评级", "信用評級",
    # 金融业务/场景类
    "信贷", "信貸", "授信", "贷款", "貸款", "存款", "理财", "理財", "资管", "資管",
    "财富管理", "財富管理", "投顾", "投研", "投资银行", "投資銀行", "承销", "承銷",
    "做市", "量化交易", "程序化交易", "支付", "清算", "收单", "收單", "结算", "結算",
    "跨境支付", "移动支付", "数字人民币", "數字人民幣", "反洗钱", "反洗錢",
    "监管报送", "監管報送", "合规风控", "合規風控", "风险管理", "風險管理",
    "反欺诈", "反欺詐", "信用评分", "信用評分", "供应链金融", "供應鏈金融",
    "消费信贷", "消費信貸", "资产证券化", "資產證券化", "行情数据", "行情數據",
    "金融数据", "金融數據", "金融终端", "金融終端", "金融信息服务", "金融信息服務",
    "核心银行系统", "核心銀行系統", "核心交易系统", "核心交易系統", "核心系统", "核心系統",
    "信贷系统", "信貸系統", "风控系统", "風控系統", "智能风控", "智能風控",
    "智能投顾", "智能投研", "财税", "財稅", "发票", "發票", "司库", "司庫",
    "客户经理", "客戶經理", "贷后", "貸後", "催收", "授信审批", "授信審批",
    "证券交易系统", "證券交易系統", "监管科技", "監管科技", "金融科技",
]))

GENERIC_SMART_NOISE_KEYWORDS = sorted(set([
    "智慧园区", "智慧園區", "智慧文旅", "智慧农业", "智慧農業", "智慧物业", "智慧物業",
    "智慧社区", "智慧社區", "智慧医疗", "智慧醫療", "智慧教育", "智慧交通", "智慧政务",
    "智慧政務", "智能制造", "智能製造", "智能工厂", "智能工廠", "生产线", "生產線",
    "智能质检", "智能質檢", "智能仓储", "智能倉儲", "智慧景区", "智慧景區",
    "智慧酒店", "智慧停车", "智慧停車", "智慧楼宇", "智慧樓宇", "智慧养老", "智慧養老",
    "智能硬件", "智能家居", "智慧城市", "智慧零售门店", "智慧零售門店",
    "智能选品", "智能设计", "智能設計", "智能穿戴", "智能装备", "智能裝備",
    "人力资源管理系统", "人力資源管理系統", "OA办公系统", "OA辦公系統",
    "ERP系统", "ERP系統", "行政管理数字化", "行政管理數字化", "生产经营数字化",
    "生產經營數字化", "数字化转型（泛化）", "數字化轉型（泛化）",
    "智能客服（非金融）", "研发投入", "研發投入", "研发人员", "研發人員",
    "云计算（基础设施）", "雲計算（基礎設施）", "大数据平台（泛化）", "大數據平台（泛化）",
]))


# ============================================================
# 4. 复核系统提示词（更严格版本）
# ============================================================

RECHECK_ANNUAL_SYSTEM_PROMPT = """
你是一名极其严格、擅长甄别"伪金融科技"的金融科技尽职调查专家。你的任务是对第一轮AI初筛结果
进行第二轮独立复核。

第一轮结果可能存在系统性误判：把公司业务/管理/生产环节中的通用"智慧化""智能化""数字化转型"
表述（如智慧园区、智能制造、智慧文旅、智慧物业、企业内部OA/ERP系统升级、生产线AI质检、
笼统的"研发投入""研发人员""云计算""数据治理"等）误判为"金融科技"；也可能把"公司只是
金融机构的普通客户"（如获得银行贷款、发行债券）误判为"公司自身从事金融科技"。

你必须严格区分：

一、真正的金融科技（应认定为相关，activity=true）：
1）该公司本身是持牌金融机构（银行/证券/保险/信托/基金/期货等），且证据显示其在支付、信贷、
   证券交易、财富管理、保险、征信、反洗钱、监管报送等金融核心业务环节，采用了人工智能/大模型/
   大数据/云计算等技术，形成了具体的、可追溯的系统、平台、产品或量化建设成果
   （如"XX大模型正式上线"、"智能风控平台覆盖XX个风控场景"、"金融科技投入XX亿元"、
   "金融科技人员XX人"、"AI大模型应用超XX个场景"等，需要有具体的名称/数字/场景，而不是空洞口号）；
2）或该公司作为技术/软件供应商，其产品/服务本身明确面向金融机构、金融场景
   （银行IT、证券IT、保险IT、支付清算、征信评分、量化交易系统等），且有具体产品名称/客户/
   收入规模等可验证信息。

二、伪金融科技（必须排除，即使原文出现"人工智能""大模型""智能化""数字化转型"等词，
   activity=false）：
1）智慧园区、智慧文旅、智慧农业、智慧物业、智慧社区、智慧医疗、智慧教育、智慧交通、智慧政务；
2）智能制造/智能工厂/生产线AI质检、智能仓储物流（非"供应链金融"场景）、智能硬件生产、
   智慧酒店/智慧景区/智慧停车/智慧楼宇；
3）公司内部管理智能化但与对外金融业务无关，如OA办公系统智能化、ERP系统升级、
   人力资源数字化、行政管理数字化、一般性生产经营数字化转型等泛化表述；
4）仅有"积极拥抱人工智能""探索AI应用""关注前沿科技发展趋势"等无具体落地内容的口号式表述，
   没有具体系统/产品/数字支撑；
5）笼统提及"研发投入XX万元""研发人员XX人""持续加大科技投入"，但未说明该研发投入/人员是否
   投向金融相关产品或系统的；
6）公司只是金融机构的普通客户/被服务对象（如企业获得银行贷款、发行债券、被托管），
   不构成公司自身从事金融科技；
7）传统金融机构自身，仅因使用了通用的云计算/大数据/AI技术进行内部管理或客户服务，
   但未与金融核心业务场景（信贷、风控、投研、支付、合规等）产生具体关联的，
   证据强度应大幅降低。

三、判断方法：
- 你会收到两组证据："候选金融科技相关证据"（已按第一轮关键词库抽取，并标注了是否命中
  金融业务锚定词、是否命中泛化智能化噪音词）与"疑似泛化智能化噪音证据"（供你对照参考，
  帮助识别第一轮可能被误导的来源，不能直接作为认定依据）；
- 只有当证据片段中同时具备"具体的金融业务/金融机构场景描述"时，才能作为真正的金融科技证据；
- 若候选证据片段仅命中关键词表面字符，但整体语境明显在讲非金融的通用业务
  （如"智慧文旅""智能制造"等），必须排除，并在 excluded_generic_smart_business_terms /
  noise_evidence_ids_confused_by_round1 中列出；
- 若第一轮结论与你的复核结论不一致，请如实反映，不要迁就第一轮结论；
- 置信度应比第一轮更保守：证据越具体、越可追溯，置信度才可以更高；证据模糊或存在泛化噪音干扰
  时，应主动降低置信度并建议人工复核。

必须返回如下格式的合法json对象，不要返回Markdown，不要使用代码块：

{
  "revised_annual_report_fintech_activity": false,
  "revised_activity_strength": "none",
  "revised_ai_evidence_level": "none",
  "revised_tracks": [],
  "revised_confidence": 30,
  "genuine_finance_evidence_ids": [],
  "noise_evidence_ids_confused_by_round1": ["F3", "F7"],
  "is_pseudo_fintech_false_positive": true,
  "reason": "简洁说明复核依据，以及与第一轮结论的差异",
  "recommend_manual_review": false
}

字段约束：
- revised_annual_report_fintech_activity：布尔值；
- revised_activity_strength：只能是 substantial、moderate、mentioned、none；
- revised_ai_evidence_level：只能是 explicit、indirect、none；
- revised_tracks：数组，只能从以下值中选择：人工智能与大模型、智能风控与反欺诈、
  数字化渠道与开放银行、数据平台与数据治理、云计算与技术基础设施、智能投顾与财富管理、
  支付清算与数字人民币、监管科技与合规、保险科技、证券科技、科技投入与组织建设、其他金融科技；
  不相关时返回空数组；
- revised_confidence：0到100之间整数；
- genuine_finance_evidence_ids：字符串数组，只能从"候选金融科技相关证据"的编号（F开头）中选择，
  最多15个，必须是你认定为真正金融科技证据（而非泛化智能化噪音）的编号；
- noise_evidence_ids_confused_by_round1：字符串数组，列出"候选金融科技相关证据"中你认为
  第一轮可能误当作金融科技、实际属于泛化智能化表述的证据编号（F开头）；
- is_pseudo_fintech_false_positive：若第一轮判断为金融科技活动相关，但你复核后认为实际是
  被泛化智能化表述误导导致的误判，必须设为true；
- reason：简洁说明依据，明确指出与第一轮结论的差异（如有）；
- recommend_manual_review：证据模糊、正反信号都存在、难以完全靠规则判断清楚时设为true。
""".strip()


RECHECK_COMPANY_SYSTEM_PROMPT = """
你是一名极其严格的金融科技尽职调查专家，负责对"公司主营业务是否属于金融科技"的第一轮AI初筛
结论进行第二轮独立复核。

第一轮可能存在的系统性误判：
1）把公司主营业务描述中出现的通用"智能化""数字化""智慧化"字样（例如智慧物业、智能制造、
   智慧文旅、智能选品、智能仓储、智慧零售门店、AI辅助设计等，均与金融业务无关）误认为
   "金融科技"；
2）把"金融主体类型"规则命中的类别（如"担保保理小贷""其他金融相关"等）不加甄别地当作
   公司确实从事相关金融业务的证据——但规则命中有可能只是业务描述中偶然出现相关字样
   （例如"提供购物担保服务"中的"担保"二字），与真实的金融科技业务无关。

判断金融科技必须同时满足：
1）业务描述中出现具体的、面向金融/财务/财税/支付/征信/风控等金融场景的产品、系统、平台或
   技术解决方案（而不是泛泛的"智能化转型"口号，也不是仅仅服务于零售/文旅/物业/农业/制造业
   等非金融行业客户的智能化产品）；
2）技术手段为人工智能、大数据、云计算、区块链等，且明确服务于金融机构或金融业务环节
   （如银行、证券、保险、支付机构等），或公司本身即持牌金融机构/类金融机构。

以下情形一律不得认定为金融科技，即使原文含有"智能""数字化""AI"字样：
- 智慧园区、智慧文旅、智慧农业、智慧物业、智慧社区、智慧医疗、智慧教育、智慧交通、智慧政务；
- 智能制造、智能工厂、生产线AI质检、智能仓储物流（非金融供应链金融场景）；
- 面向零售/文旅/物业/农业/制造业等非金融行业客户的一般性智能化产品或服务；
- 公司只是获得银行贷款、发行债券等金融服务的普通客户，不构成公司自身从事金融科技；
- 业务描述中的"担保""保理""信托""基金"等字样若并非指公司真实持有相关金融牌照或
  真实开展相关金融业务（例如"购物保障金""退款保证""员工持股信托计划"等日常用语），
  不能仅凭字面匹配认定为金融相关。

请严格依据用户提供的主营业务构成原文与金融主体类型判定依据重新判断，不得凭空推测，
也不得受第一轮结论影响，输出比第一轮更谨慎保守的结论。

必须返回如下格式的合法json对象，不要返回Markdown，不要使用代码块：

{
  "revised_fintech_related": false,
  "revised_ai_evidence_level": "none",
  "revised_tracks": [],
  "revised_business_materiality": "none",
  "revised_confidence": 30,
  "genuine_finance_evidence": [],
  "excluded_generic_smart_business_terms": ["智能选品系统", "智慧门店管理平台"],
  "is_pseudo_fintech_false_positive": true,
  "entity_type_seems_misclassified": true,
  "entity_type_reason": "简要说明金融主体类型规则判定是否合理",
  "reason": "简洁说明复核依据，以及与第一轮结论的差异",
  "recommend_manual_review": false
}

字段约束：
- revised_fintech_related：布尔值；
- revised_ai_evidence_level：只能是 explicit、indirect、none；
- revised_tracks：数组，赛道枚举与第一轮一致（银行IT、证券IT、保险IT、智能风控、智能投顾、
  财富管理科技、财务SaaS、财税科技、监管科技、金融数据、支付科技、征信科技、金融信息安全、
  AI金融基础设施、其他金融科技），不相关时返回空数组；
- revised_business_materiality：只能是 core、important、weak、none；
- revised_confidence：0到100之间整数；
- genuine_finance_evidence：字符串数组，必须逐字来自用户输入的主营业务构成原文，最多5条；
- excluded_generic_smart_business_terms：字符串数组，标注被识别并排除的泛化智能化表述，最多5条；
- is_pseudo_fintech_false_positive：若第一轮判断为金融科技相关，但复核认为实际是被泛化智能化
  表述误导，应设为true；
- entity_type_seems_misclassified：布尔值，若通过规则命中的金融主体类别明显是因业务描述中
  偶然出现相关字样、而实际业务与金融无关导致的误判，应设为true；
- entity_type_reason：简要说明entity_type_seems_misclassified的判断依据；
- reason：简洁说明依据，明确指出与第一轮结论的差异（如有）；
- recommend_manual_review：证据模糊、正反信号都存在时设为true，提示需人工复核。
""".strip()


# ============================================================
# 5. 年报层面复核：更细粒度证据抽取（句子级切分 + 金融锚定/噪音标注）
# ============================================================

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；\n])")


def split_into_sentences(text: str) -> list[str]:
    text = base.safe_text(text)
    if not text:
        return []
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def load_cached_annual_report_pages(text_path: Any) -> list[dict]:
    """直接读取第一轮已缓存的年报纯文本（不重新下载/解析PDF），按PAGE标记切分为分页文本。"""
    path_str = base.safe_text(text_path)
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        return []

    content: str | None = None
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return []
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            content = raw_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if content is None:
        # 所有候选编码均解码失败，退化为忽略非法字节，保证流程不中断
        content = raw_bytes.decode("utf-8", errors="ignore")

    segments = re.split(r"={5}\s*PAGE\s+(\d+)\s*={5}", content)
    pages: list[dict] = []
    if len(segments) <= 1:
        cleaned = content.strip()
        if cleaned:
            pages.append({"page": 1, "text": cleaned})
        return pages

    it = iter(segments[1:])
    for page_no_str, page_text in zip(it, it):
        try:
            page_num = int(page_no_str)
        except ValueError:
            continue
        cleaned = page_text.strip()
        if cleaned:
            pages.append({"page": page_num, "text": cleaned})
    return pages


def collect_refined_evidence(
    pages: list[dict],
    max_evidence: int = RECHECK_MAX_ANNUAL_EVIDENCE,
    max_noise_evidence: int = RECHECK_MAX_NOISE_EVIDENCE,
    max_occurrences_per_keyword: int = RECHECK_MAX_OCCURRENCES_PER_KEYWORD,
) -> tuple[list[dict], list[dict]]:
    """
    按句子级切分重新抽取证据：命中句 + 前一句 + 后一句 作为语义完整的证据片段，
    并对每条候选证据同时标注"是否命中金融业务锚定词""是否命中泛化智能化噪音词"，
    同时独立采集"泛化智能化噪音证据"供模型对照参考。
    """
    finance_evidence: list[dict] = []
    noise_evidence: list[dict] = []
    seen_finance: set[tuple] = set()
    seen_noise: set[tuple] = set()
    finance_kw_counter: dict[tuple, int] = {}
    noise_kw_counter: dict[str, int] = {}

    for page in pages:
        page_number = int(page.get("page", 0) or 0)
        sentences = split_into_sentences(page.get("text", ""))
        if not sentences:
            continue
        lower_sentences = [s.lower() for s in sentences]

        for idx, lower_sentence in enumerate(lower_sentences):
            window = sentences[max(0, idx - 1): idx + 2]
            snippet = "".join(window).strip()
            if not snippet:
                continue
            lower_snippet = snippet.lower()

            # ---- 候选金融科技证据（沿用第一轮关键词分组）----
            for group, keywords in base.ANNUAL_REPORT_KEYWORD_GROUPS.items():
                for keyword in keywords:
                    if keyword.lower() not in lower_sentence:
                        continue
                    counter_key = (group, keyword)
                    if finance_kw_counter.get(counter_key, 0) >= max_occurrences_per_keyword:
                        continue
                    dedupe_key = (page_number, group, base.normalize_text(snippet))
                    if dedupe_key in seen_finance:
                        continue
                    seen_finance.add(dedupe_key)
                    finance_kw_counter[counter_key] = finance_kw_counter.get(counter_key, 0) + 1

                    anchor_hits = [t for t in FINANCE_ANCHOR_KEYWORDS if t.lower() in lower_snippet]
                    noise_hits = [t for t in GENERIC_SMART_NOISE_KEYWORDS if t.lower() in lower_snippet]

                    finance_evidence.append({
                        "page": page_number,
                        "group": group,
                        "keyword": keyword,
                        "snippet": snippet,
                        "score": base.ANNUAL_REPORT_KEYWORD_WEIGHTS.get(group, 1),
                        "finance_anchor_hit": bool(anchor_hits),
                        "matched_finance_anchor_terms": anchor_hits[:6],
                        "noise_hit": bool(noise_hits),
                        "matched_noise_terms": noise_hits[:6],
                    })

            # ---- 独立采集：疑似泛化智能化噪音证据（供模型对照）----
            for noise_kw in GENERIC_SMART_NOISE_KEYWORDS:
                if noise_kw.lower() not in lower_sentence:
                    continue
                if noise_kw_counter.get(noise_kw, 0) >= max_occurrences_per_keyword:
                    continue
                dedupe_key = (page_number, noise_kw, base.normalize_text(snippet))
                if dedupe_key in seen_noise:
                    continue
                seen_noise.add(dedupe_key)
                noise_kw_counter[noise_kw] = noise_kw_counter.get(noise_kw, 0) + 1

                anchor_hits = [t for t in FINANCE_ANCHOR_KEYWORDS if t.lower() in lower_snippet]
                noise_evidence.append({
                    "page": page_number,
                    "keyword": noise_kw,
                    "snippet": snippet,
                    "finance_anchor_hit": bool(anchor_hits),
                    "matched_finance_anchor_terms": anchor_hits[:6],
                })

    # 优先展示具备金融锚定的证据，其次按第一轮权重排序
    finance_evidence.sort(key=lambda e: (-int(e["finance_anchor_hit"]), -int(e["score"]), e["page"]))
    noise_evidence.sort(key=lambda e: (int(e["finance_anchor_hit"]), e["page"]))

    finance_evidence = finance_evidence[:max_evidence]
    noise_evidence = noise_evidence[:max_noise_evidence]

    for i, item in enumerate(finance_evidence, start=1):
        item["evidence_id"] = f"F{i}"
    for i, item in enumerate(noise_evidence, start=1):
        item["evidence_id"] = f"N{i}"

    return finance_evidence, noise_evidence


def build_recheck_annual_report_prompt(
    row: dict,
    finance_evidence: list[dict],
    noise_evidence: list[dict],
) -> str:
    finance_evidence_text = "\n\n".join(
        f"[{item['evidence_id']}] 页码：{item['page']}；关键词组：{item['group']}；"
        f"命中词：{item['keyword']}；"
        f"金融锚定命中：{'是' if item['finance_anchor_hit'] else '否'}"
        f"（{'、'.join(item['matched_finance_anchor_terms']) if item['matched_finance_anchor_terms'] else '无'}）；"
        f"疑似泛化智能化噪音：{'是' if item['noise_hit'] else '否'}"
        f"（{'、'.join(item['matched_noise_terms']) if item['matched_noise_terms'] else '无'}）\n"
        f"原文片段：{item['snippet']}"
        for item in finance_evidence
    )
    noise_evidence_text = "\n\n".join(
        f"[{item['evidence_id']}] 页码：{item['page']}；命中泛化智能化词：{item['keyword']}；"
        f"金融锚定命中：{'是' if item['finance_anchor_hit'] else '否'}"
        f"（{'、'.join(item['matched_finance_anchor_terms']) if item['matched_finance_anchor_terms'] else '无'}）\n"
        f"原文片段：{item['snippet']}"
        for item in noise_evidence
    )

    return f"""
请对以下公司的金融科技年报证据进行第二轮严格复核。第一轮结论仅供参考背景信息，
你必须独立判断，不得直接沿用或迁就第一轮结论。

股票代码：{row.get('ts_code', '')}
公司名称：{row.get('name', '')}
基础行业：{row.get('industry', '')}
金融主体类型：{row.get('financial_entity_type', '')}
（注意：若为"金融相关主体"而非"金融机构"，说明该公司并非持牌金融机构，
需格外警惕将其年报中的泛化智能化描述误判为金融科技）
金融主体类别：{row.get('financial_entity_categories', '')}
年度报告年度：{row.get('report_year', '')}

第一轮结论（仅供参考背景，不得直接采信）：
- 金融科技活动：{row.get('annual_report_fintech_activity', '')}
- 活跃度：{row.get('activity_strength', '')}
- AI证据级别：{row.get('ai_evidence_level', '')}
- 置信度：{row.get('confidence', '')}
- 第一轮理由：{row.get('reason', '')}

一、候选金融科技相关证据（按关键词组抽取，已标注金融锚定/泛化噪音命中情况）：

{finance_evidence_text or '未提取到相关证据片段'}

二、疑似泛化智能化噪音证据（可能是第一轮误判的来源，仅供对照参考，不能作为认定依据）：

{noise_evidence_text or '未发现明显的泛化智能化噪音表述'}

请严格依据"一"中真正具备金融锚定、且描述具体（有明确系统/产品/数字支撑）的证据重新判断，
忽略"二"中与金融业务无关的泛化智能化表述，并识别"一"中可能被第一轮误判为金融科技、
实际属于泛化智能化的证据编号。
""".strip()


# ============================================================
# 6. 复核模型调用（更强模型）
# ============================================================

def build_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    if not api_key:
        raise RuntimeError(
            "未检测到DEEPSEEK_API_KEY环境变量。\n"
            "Windows PowerShell示例：\n"
            '$env:DEEPSEEK_API_KEY="你的API Key"'
        )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=RECHECK_REQUEST_TIMEOUT_SECONDS)


def _build_thinking_extra_body() -> dict | None:
    if RECHECK_ENABLE_THINKING == "enabled":
        return {"thinking": {"type": "enabled"}}
    if RECHECK_ENABLE_THINKING == "disabled":
        return {"thinking": {"type": "disabled"}}
    return None  # "omit"：不传该参数，交由模型/网关默认处理


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> dict:
    """
    宽松提取JSON对象：兼容模型偶尔在json_object模式下仍返回被```json代码块包裹、
    或前后带有多余说明文字的情况。
    """
    text = text.strip()
    fence_match = _JSON_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1)
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace: last_brace + 1]
    return json.loads(text)


def call_recheck_model(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict, Any]:
    """
    调用复核模型，并采用分级降级重试策略应对"复核模型返回空content"这类问题：

    - 第1次：按配置调用（默认 thinking=disabled）；
    - 第2次起：显式强制关闭thinking（而不是仅仅省略该参数——某些推理档模型省略参数时
      仍会按默认行为继续"思考"），并把max_tokens提高到至少6000，避免思考阶段耗尽
      token导致最终没有内容输出；
    - 第3次起：进一步放弃强制 response_format=json_object 约束，改为完全依赖提示词中的
      格式说明，并对模型输出做宽松JSON提取（兼容```json代码块包裹等情况），
      同时把max_tokens提高到至少8000。

    每次失败都会打印 finish_reason / reasoning_content长度 / completion_tokens 等诊断信息，
    便于判断是"思考耗尽token"还是其他网关兼容性问题。
    """
    last_error: Exception | None = None
    base_max_tokens = RECHECK_MAX_OUTPUT_TOKENS

    for attempt in range(1, RECHECK_MAX_RETRIES + 1):
        if attempt == 1:
            extra_body = _build_thinking_extra_body()
            use_json_mode = True
            max_tokens = base_max_tokens
        elif attempt == 2:
            extra_body = {"thinking": {"type": "disabled"}}
            use_json_mode = True
            max_tokens = max(base_max_tokens, 6000)
        else:
            extra_body = {"thinking": {"type": "disabled"}}
            use_json_mode = False
            max_tokens = max(base_max_tokens, 8000)

        try:
            kwargs: dict[str, Any] = dict(
                model=RECHECK_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                stream=False,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None)

            if not content:
                finish_reason = getattr(choice, "finish_reason", "未知")
                reasoning_content = getattr(message, "reasoning_content", None) or ""
                completion_tokens = getattr(response.usage, "completion_tokens", "未知")
                raise ValueError(
                    "复核模型返回空content"
                    f"（finish_reason={finish_reason}，"
                    f"reasoning_content长度={len(reasoning_content)}，"
                    f"completion_tokens={completion_tokens}，"
                    f"本次max_tokens={max_tokens}，"
                    f"本次thinking参数={extra_body}，"
                    f"本次是否强制json_object={use_json_mode}）"
                )

            raw_result = _extract_json_object(content)
            return raw_result, response.usage

        except Exception as exc:
            last_error = exc

            if attempt >= RECHECK_MAX_RETRIES:
                break

            wait_seconds = min(2 ** attempt, 40) + random.random()
            hint = ""
            if attempt == 1:
                hint = "（下次将强制关闭thinking并提高max_tokens至6000）"
            elif attempt == 2:
                hint = "（下次将放宽JSON格式约束、提高max_tokens至8000并宽松解析输出）"
            print(f"    第{attempt}/{RECHECK_MAX_RETRIES}次调用失败：{exc}")
            print(f"    {wait_seconds:.1f}秒后重试{hint}。")
            time.sleep(wait_seconds)

    raise RuntimeError(f"复核模型连续调用失败：{last_error}") from last_error


# ============================================================
# 7. 年报层面复核结果校验与校准
# ============================================================

RECHECK_REQUIRED_FIELDS_ANNUAL = {
    "revised_annual_report_fintech_activity",
    "revised_activity_strength",
    "revised_ai_evidence_level",
    "revised_tracks",
    "revised_confidence",
    "genuine_finance_evidence_ids",
    "noise_evidence_ids_confused_by_round1",
    "is_pseudo_fintech_false_positive",
    "reason",
    "recommend_manual_review",
}


def validate_recheck_annual_result(result: dict) -> None:
    missing = RECHECK_REQUIRED_FIELDS_ANNUAL - set(result)
    if missing:
        raise ValueError(f"复核模型结果缺少字段：{sorted(missing)}")

    if not isinstance(result["revised_annual_report_fintech_activity"], bool):
        raise ValueError("revised_annual_report_fintech_activity必须是布尔值。")
    if result["revised_activity_strength"] not in base.ANNUAL_REPORT_STRENGTH_LEVELS:
        raise ValueError("revised_activity_strength值不合法。")
    if result["revised_ai_evidence_level"] not in base.AI_EVIDENCE_LEVELS:
        raise ValueError("revised_ai_evidence_level值不合法。")
    if not isinstance(result["revised_tracks"], list):
        raise ValueError("revised_tracks必须是数组。")
    if not isinstance(result["genuine_finance_evidence_ids"], list):
        raise ValueError("genuine_finance_evidence_ids必须是数组。")
    if not isinstance(result["noise_evidence_ids_confused_by_round1"], list):
        raise ValueError("noise_evidence_ids_confused_by_round1必须是数组。")
    if not isinstance(result["is_pseudo_fintech_false_positive"], bool):
        raise ValueError("is_pseudo_fintech_false_positive必须是布尔值。")
    if not isinstance(result["recommend_manual_review"], bool):
        raise ValueError("recommend_manual_review必须是布尔值。")

    confidence = int(result["revised_confidence"])
    if not 0 <= confidence <= 100:
        raise ValueError("revised_confidence必须在0到100之间。")
    result["revised_confidence"] = confidence


def calibrate_recheck_annual_result(raw_result: dict, finance_evidence: list[dict]) -> dict:
    result = dict(raw_result)
    evidence_by_id = {item["evidence_id"]: item for item in finance_evidence}

    valid_genuine_ids = [
        base.safe_text(x) for x in result.get("genuine_finance_evidence_ids", [])
        if base.safe_text(x) in evidence_by_id
    ]
    valid_genuine_ids = list(dict.fromkeys(valid_genuine_ids))[:RECHECK_MAX_OUTPUT_EVIDENCE_IDS]

    activity = base.is_true(result.get("revised_annual_report_fintech_activity"))
    confidence = int(result["revised_confidence"])

    genuine_evidence_objs = [evidence_by_id[i] for i in valid_genuine_ids]
    anchor_backed = [e for e in genuine_evidence_objs if e.get("finance_anchor_hit")]

    if activity and not valid_genuine_ids:
        # 认定为金融科技活动，但没有任何可追溯证据支撑，强制推翻
        activity = False
        result["revised_activity_strength"] = "none"
        confidence = min(confidence, 35)
        result["is_pseudo_fintech_false_positive"] = True
        result["reason"] = (
            base.safe_text(result.get("reason"))
            + " 【程序复核强制校准】模型未提供可追溯的有效金融科技证据编号，"
              "复核结论已强制推翻为不相关。"
        ).strip()
    elif activity and not anchor_backed:
        # 有证据编号但均未命中金融业务锚定词，高度疑似被泛化智能化表述误导
        confidence = min(confidence, 55)
        result["recommend_manual_review"] = True
        result["reason"] = (
            base.safe_text(result.get("reason"))
            + " 【程序复核提示】所引用证据均未命中明确的金融业务锚定词，"
              "存在被泛化智能化表述误导的风险，建议人工复核。"
        ).strip()

    tracks = [
        base.safe_text(t) for t in result.get("revised_tracks", [])
        if base.safe_text(t) in base.ANNUAL_REPORT_ALLOWED_TRACKS
    ]
    tracks = list(dict.fromkeys(tracks)) if activity else []

    noise_ids = [
        base.safe_text(x) for x in result.get("noise_evidence_ids_confused_by_round1", [])
        if base.safe_text(x) in evidence_by_id
    ]

    result["revised_annual_report_fintech_activity"] = activity
    result["revised_tracks"] = tracks
    result["revised_confidence"] = confidence
    result["genuine_finance_evidence_ids"] = valid_genuine_ids
    result["noise_evidence_ids_confused_by_round1"] = list(dict.fromkeys(noise_ids))
    result["genuine_evidence_pages"] = "、".join(str(e["page"]) for e in genuine_evidence_objs)
    result["genuine_evidence_keywords"] = "、".join(
        dict.fromkeys(e["keyword"] for e in genuine_evidence_objs)
    )
    result["genuine_evidence_snippets"] = " || ".join(
        f"P{e['page']} [{e['keyword']}] {e['snippet']}" for e in genuine_evidence_objs
    )
    return result


# ============================================================
# 8. 主营业务层面复核结果校验与校准
# ============================================================

RECHECK_REQUIRED_FIELDS_COMPANY = {
    "revised_fintech_related",
    "revised_ai_evidence_level",
    "revised_tracks",
    "revised_business_materiality",
    "revised_confidence",
    "genuine_finance_evidence",
    "excluded_generic_smart_business_terms",
    "is_pseudo_fintech_false_positive",
    "entity_type_seems_misclassified",
    "entity_type_reason",
    "reason",
    "recommend_manual_review",
}


def validate_recheck_company_result(result: dict) -> None:
    missing = RECHECK_REQUIRED_FIELDS_COMPANY - set(result)
    if missing:
        raise ValueError(f"复核模型结果缺少字段：{sorted(missing)}")

    if not isinstance(result["revised_fintech_related"], bool):
        raise ValueError("revised_fintech_related必须是布尔值。")
    if result["revised_ai_evidence_level"] not in base.AI_EVIDENCE_LEVELS:
        raise ValueError("revised_ai_evidence_level值不合法。")
    if result["revised_business_materiality"] not in base.BUSINESS_MATERIALITY_LEVELS:
        raise ValueError("revised_business_materiality值不合法。")
    if not isinstance(result["revised_tracks"], list):
        raise ValueError("revised_tracks必须是数组。")
    if not isinstance(result["genuine_finance_evidence"], list):
        raise ValueError("genuine_finance_evidence必须是数组。")
    if not isinstance(result["excluded_generic_smart_business_terms"], list):
        raise ValueError("excluded_generic_smart_business_terms必须是数组。")
    if not isinstance(result["is_pseudo_fintech_false_positive"], bool):
        raise ValueError("is_pseudo_fintech_false_positive必须是布尔值。")
    if not isinstance(result["entity_type_seems_misclassified"], bool):
        raise ValueError("entity_type_seems_misclassified必须是布尔值。")
    if not isinstance(result["recommend_manual_review"], bool):
        raise ValueError("recommend_manual_review必须是布尔值。")

    confidence = int(result["revised_confidence"])
    if not 0 <= confidence <= 100:
        raise ValueError("revised_confidence必须在0到100之间。")
    result["revised_confidence"] = confidence


def calibrate_recheck_company_result(raw_result: dict, record: dict) -> dict:
    result = dict(raw_result)
    result["revised_fintech_related"] = base.is_true(result.get("revised_fintech_related"))
    result["entity_type_seems_misclassified"] = base.is_true(
        result.get("entity_type_seems_misclassified")
    )

    valid_evidence = [
        base.safe_text(e) for e in result.get("genuine_finance_evidence", [])
        if base.evidence_exists_in_input(base.safe_text(e), record.get("business_items", []))
    ]
    result["genuine_finance_evidence"] = list(dict.fromkeys(valid_evidence))[:5]

    excluded_terms = [
        base.safe_text(t) for t in result.get("excluded_generic_smart_business_terms", [])
    ]
    result["excluded_generic_smart_business_terms"] = list(dict.fromkeys(excluded_terms))[:5]

    tracks = [
        base.safe_text(t) for t in result.get("revised_tracks", [])
        if base.safe_text(t) in base.ALLOWED_TRACKS
    ]
    tracks = list(dict.fromkeys(tracks))

    confidence = int(result["revised_confidence"])

    if not result["revised_fintech_related"]:
        tracks = []
        if result.get("revised_business_materiality") in {"core", "important"}:
            result["revised_business_materiality"] = "none"
    else:
        if not result["genuine_finance_evidence"]:
            result["revised_fintech_related"] = False
            confidence = min(confidence, 35)
            result["is_pseudo_fintech_false_positive"] = True
            tracks = []
            result["revised_business_materiality"] = "none"
            result["reason"] = (
                base.safe_text(result.get("reason"))
                + " 【程序复核强制校准】未提供可追溯至主营业务原文的有效证据，"
                  "复核结论已强制推翻为不相关。"
            ).strip()
        elif not tracks:
            tracks = ["其他金融科技"]
            confidence = min(confidence, 70)

    result["revised_tracks"] = tracks
    result["revised_confidence"] = confidence
    return result


# ============================================================
# 9. 断点续跑
# ============================================================

def load_recheck_checkpoint(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = base.safe_text(row.get("recheck_key"))
            if key:
                completed[key] = row
    return completed


def append_recheck_checkpoint(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def coerce_report_year(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return getattr(base, "DEFAULT_ANNUAL_REPORT_YEAR", 0)


def annual_report_recheck_key(ts_code: str, report_year: Any) -> str:
    return f"{ts_code}|{coerce_report_year(report_year)}"


# ============================================================
# 10. 年报层面复核主流程
# ============================================================

def load_annual_report_results() -> pd.DataFrame:
    if not ANNUAL_REPORT_INPUT_FILE.exists():
        return pd.DataFrame()
    return read_csv_robust(ANNUAL_REPORT_INPUT_FILE, dtype={"ts_code": str}, low_memory=False)


def select_annual_report_recheck_targets(df_annual: pd.DataFrame) -> pd.DataFrame:
    if df_annual.empty:
        return df_annual

    df = df_annual.copy()
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(0)

    entity_type = df.get("financial_entity_type", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    text_path = df.get("annual_report_text_path", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    status = df.get("annual_report_analysis_status", pd.Series(dtype=str)).fillna("").astype(str).str.strip()

    is_non_regulated = entity_type == "金融相关主体"
    is_low_confidence = df["confidence"] < CONFIDENCE_THRESHOLD
    has_report_text = text_path != ""
    status_ok = status.isin(["completed", "completed_rule_no_signal"])

    target_mask = (is_non_regulated | is_low_confidence) & has_report_text & status_ok
    targets = df[target_mask].reset_index(drop=True)

    # 每个ts_code只保留一条最新（若同一公司有多个report_year结果，全部保留分别复核，
    # 因为不同年度年报独立评估）
    return targets


def run_annual_report_recheck(client: OpenAI) -> None:
    df_annual = load_annual_report_results()
    if df_annual.empty:
        print(f"\n[年报层面复核] 未找到第一轮年报结果文件：{ANNUAL_REPORT_INPUT_FILE}，跳过。")
        return

    targets = select_annual_report_recheck_targets(df_annual)
    if RECHECK_MAX_ANNUAL_REPORT_TARGETS is not None:
        targets = targets.head(RECHECK_MAX_ANNUAL_REPORT_TARGETS)

    completed = load_recheck_checkpoint(RECHECK_ANNUAL_CHECKPOINT)
    total = len(targets)

    print("\n" + "=" * 70)
    print("[年报层面复核]")
    print(f"  第一轮结果文件：{ANNUAL_REPORT_INPUT_FILE}")
    print(f"  触发条件：financial_entity_type == '金融相关主体' 或 confidence < {CONFIDENCE_THRESHOLD}")
    print(f"  待复核记录：{total} 条（断点续跑，已完成 {len(completed)} 条）")
    print(f"  复核模型：{RECHECK_MODEL_NAME}（thinking={RECHECK_ENABLE_THINKING}）")
    print("=" * 70)

    for idx, row in enumerate(targets.to_dict("records"), start=1):
        ts_code = base.safe_text(row.get("ts_code"))
        report_year = coerce_report_year(row.get("report_year"))
        key = annual_report_recheck_key(ts_code, report_year)

        if key in completed and base.safe_text(completed[key].get("recheck_status")) == "completed":
            print(f"[{idx}/{total}] 跳过已复核：{ts_code} {row.get('name')} {report_year}")
            continue

        trigger = (
            "非持牌金融相关主体"
            if base.safe_text(row.get("financial_entity_type")) == "金融相关主体"
            else "低置信度"
        )
        print(
            f"[{idx}/{total}] 复核年报：{ts_code} {row.get('name')} {report_year}"
            f"（触发原因：{trigger}；第一轮confidence={row.get('confidence')}）"
        )

        try:
            pages = load_cached_annual_report_pages(row.get("annual_report_text_path"))
            if not pages:
                raise RuntimeError(
                    f"未能读取缓存年报文本：{row.get('annual_report_text_path')}，"
                    "请确认第一轮已成功生成该文件。"
                )

            finance_evidence, noise_evidence = collect_refined_evidence(pages)
            if not finance_evidence:
                print("    句子级重新抽取未发现任何候选金融科技证据，直接判定为不相关。")
                result = {
                    "revised_annual_report_fintech_activity": False,
                    "revised_activity_strength": "none",
                    "revised_ai_evidence_level": "none",
                    "revised_tracks": [],
                    "revised_confidence": 20,
                    "genuine_finance_evidence_ids": [],
                    "noise_evidence_ids_confused_by_round1": [],
                    "is_pseudo_fintech_false_positive": bool(
                        base.is_true(row.get("annual_report_fintech_activity"))
                    ),
                    "reason": "【程序复核】句子级重新抽取未发现任何候选金融科技相关证据。",
                    "recommend_manual_review": False,
                    "recheck_model": "rule_only_no_evidence",
                    "recheck_prompt_tokens": None,
                    "recheck_completion_tokens": None,
                    "recheck_total_tokens": None,
                }
            else:
                user_prompt = build_recheck_annual_report_prompt(row, finance_evidence, noise_evidence)
                raw_result, usage = call_recheck_model(
                    client, RECHECK_ANNUAL_SYSTEM_PROMPT, user_prompt
                )
                validate_recheck_annual_result(raw_result)
                result = calibrate_recheck_annual_result(raw_result, finance_evidence)
                result.update({
                    "recheck_model": RECHECK_MODEL_NAME,
                    "recheck_prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "recheck_completion_tokens": getattr(usage, "completion_tokens", None),
                    "recheck_total_tokens": getattr(usage, "total_tokens", None),
                })

            result.update({
                "recheck_key": key,
                "ts_code": ts_code,
                "name": row.get("name"),
                "industry": row.get("industry"),
                "report_year": report_year,
                "round1_financial_entity_type": row.get("financial_entity_type"),
                "round1_financial_entity_categories": row.get("financial_entity_categories"),
                "round1_annual_report_fintech_activity": row.get("annual_report_fintech_activity"),
                "round1_activity_strength": row.get("activity_strength"),
                "round1_confidence": row.get("confidence"),
                "round1_tracks": row.get("tracks"),
                "round1_reason": row.get("reason"),
                "recheck_trigger": trigger,
                "finance_evidence_count": len(finance_evidence),
                "noise_evidence_count": len(noise_evidence),
                "finance_anchor_evidence_count": sum(
                    1 for e in finance_evidence if e.get("finance_anchor_hit")
                ),
                "recheck_status": "completed",
            })

            append_recheck_checkpoint(RECHECK_ANNUAL_CHECKPOINT, result)
            completed[key] = result

            verdict_changed = (
                base.is_true(row.get("annual_report_fintech_activity"))
                != result["revised_annual_report_fintech_activity"]
            )
            print(
                "    复核结论="
                f"{result['revised_annual_report_fintech_activity']}"
                f"（第一轮={row.get('annual_report_fintech_activity')}，"
                f"{'★结论发生变化' if verdict_changed else '结论一致'}），"
                f"复核置信度={result['revised_confidence']}，"
                f"疑似伪金融科技={result['is_pseudo_fintech_false_positive']}，"
                f"建议人工复核={result.get('recommend_manual_review')}"
            )

        except Exception as exc:
            print(f"    复核失败：{exc}")
            failure_result = {
                "recheck_key": key,
                "ts_code": ts_code,
                "name": row.get("name"),
                "report_year": report_year,
                "round1_financial_entity_type": row.get("financial_entity_type"),
                "round1_confidence": row.get("confidence"),
                "recheck_trigger": trigger,
                "recheck_status": "failed",
                "recheck_error": str(exc),
            }
            append_recheck_checkpoint(RECHECK_ANNUAL_CHECKPOINT, failure_result)
            completed[key] = failure_result

        time.sleep(RECHECK_REQUEST_INTERVAL)

    export_annual_report_recheck_csv(completed)


def export_annual_report_recheck_csv(completed: dict[str, dict]) -> None:
    if not completed:
        print("没有可导出的年报层面复核结果。")
        return

    rows = list(completed.values())
    df = pd.DataFrame(rows)
    for col in ["revised_tracks", "round1_tracks", "genuine_finance_evidence_ids",
                "noise_evidence_ids_confused_by_round1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: "、".join(v) if isinstance(v, list) else v)

    RECHECK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RECHECK_ANNUAL_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  年报层面复核结果已导出：{RECHECK_ANNUAL_OUTPUT_CSV}")

    if "recheck_status" in df.columns:
        completed_df = df[df["recheck_status"] == "completed"]
        if not completed_df.empty:
            if "is_pseudo_fintech_false_positive" in completed_df.columns:
                flagged = completed_df[completed_df["is_pseudo_fintech_false_positive"] == True]  # noqa: E712
                print(f"  疑似伪金融科技（年报层面）：{len(flagged)} / {len(completed_df)}")
            if {"revised_annual_report_fintech_activity", "round1_annual_report_fintech_activity"} <= set(completed_df.columns):
                changed = completed_df[
                    completed_df["revised_annual_report_fintech_activity"].astype(str)
                    != completed_df["round1_annual_report_fintech_activity"].astype(str)
                ]
                print(f"  结论发生变化（年报层面）：{len(changed)} / {len(completed_df)}")
            if "recommend_manual_review" in completed_df.columns:
                need_review = completed_df[completed_df["recommend_manual_review"] == True]  # noqa: E712
                print(f"  建议人工复核（年报层面）：{len(need_review)} / {len(completed_df)}")


# ============================================================
# 11. 主营业务层面复核主流程
# ============================================================

def load_company_level_results() -> pd.DataFrame:
    if not COMPANY_LEVEL_INPUT_FILE.exists():
        return pd.DataFrame()
    return read_csv_robust(COMPANY_LEVEL_INPUT_FILE, dtype={"ts_code": str}, low_memory=False)


def build_recheck_company_prompt(
    record: dict,
    round1_row: dict,
) -> str:
    anchor_hits = [
        kw for kw in FINANCE_ANCHOR_KEYWORDS
        if kw.lower() in base.normalize_text(record.get("business_description", ""))
    ]
    noise_hits = [
        kw for kw in GENERIC_SMART_NOISE_KEYWORDS
        if kw.lower() in base.normalize_text(record.get("business_description", ""))
    ]

    return f"""
请对以下公司的主营业务金融科技归属进行第二轮严格复核。第一轮结论仅供参考背景信息，
不得直接采信或迁就。

股票代码：{record.get('ts_code', '')}
公司名称：{record.get('name', '')}
基础行业：{record.get('industry', '')}
金融主体类型（按当前规则重新计算）：{record.get('financial_entity_type', '')}
金融主体类别：{record.get('financial_entity_categories', '')}
金融主体规则命中的关键词：{record.get('financial_entity_keyword_hits', '')}
金融主体规则命中的公司名称模式：{record.get('financial_entity_name_rule_hits', '')}
金融主体规则判定依据：{record.get('financial_entity_reason', '')}

第一轮AI结论（仅供参考背景，不得直接采信）：
- 金融科技相关：{round1_row.get('fintech_related', '')}
- 赛道：{round1_row.get('tracks', '')}
- 置信度：{round1_row.get('confidence', '')}
- 第一轮理由：{round1_row.get('reason', '')}

程序初步规则扫描结果（仅供参考，不代表最终判断）：
- 命中金融业务锚定词：{'、'.join(anchor_hits) if anchor_hits else '无'}
- 命中泛化智能化噪音词：{'、'.join(noise_hits) if noise_hits else '无'}

注意：
- 公司名称只用于识别，不得根据名称联想业务；
- 按产品P和按行业I是不同披露口径，严禁相加或自行计算占比；
- genuine_finance_evidence 必须逐字来自下列主营业务构成原文。

主营业务构成原文：

{record.get('business_description', '') or '未提供有效主营业务构成'}
""".strip()


def select_company_level_recheck_targets(
    df_company: pd.DataFrame,
    record_by_code: dict[str, dict],
) -> pd.DataFrame:
    if df_company.empty:
        return df_company

    df = df_company.copy()
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(0)
    df["financial_entity_type"] = df["ts_code"].map(
        lambda code: record_by_code.get(code, {}).get("financial_entity_type", "未知")
    )

    is_non_regulated = df["financial_entity_type"] == "金融相关主体"
    is_low_confidence = df["confidence"] < CONFIDENCE_THRESHOLD
    targets = df[is_non_regulated | is_low_confidence].reset_index(drop=True)
    return targets


def run_company_level_recheck(client: OpenAI) -> None:
    df_company = load_company_level_results()
    if df_company.empty:
        print(f"\n[主营业务层面复核] 未找到第一轮主营业务结果文件：{COMPANY_LEVEL_INPUT_FILE}，跳过。")
        return

    print("\n" + "=" * 70)
    print("[主营业务层面复核]")
    print("  正在按当前规则重建全市场公司主营业务描述与金融主体类型（复用第一轮数据管道）……")

    try:
        # base.load_data() 内部的 pd.read_csv 未做编码兜底，这里临时打补丁以兼容
        # 原始输入文件可能被Excel/WPS重新保存为GBK等本地编码的情况
        with robust_csv_encoding_patch():
            original_df = base.load_data()
        all_records = base.build_company_records(original_df)
    except Exception as exc:
        print(f"  重建主营业务数据失败，跳过主营业务层面复核：{exc}")
        return

    record_by_code = {r["ts_code"]: r for r in all_records}

    targets_meta = select_company_level_recheck_targets(df_company, record_by_code)
    if RECHECK_MAX_COMPANY_TARGETS is not None:
        targets_meta = targets_meta.head(RECHECK_MAX_COMPANY_TARGETS)

    completed = load_recheck_checkpoint(RECHECK_COMPANY_CHECKPOINT)
    total = len(targets_meta)

    print(f"  触发条件：financial_entity_type == '金融相关主体' 或 confidence < {CONFIDENCE_THRESHOLD}")
    print(f"  待复核公司：{total} 家（断点续跑，已完成 {len(completed)} 家）")
    print(f"  复核模型：{RECHECK_MODEL_NAME}（thinking={RECHECK_ENABLE_THINKING}）")
    print("=" * 70)

    for idx, meta_row in enumerate(targets_meta.to_dict("records"), start=1):
        ts_code = base.safe_text(meta_row.get("ts_code"))
        name = meta_row.get("name")

        if ts_code in completed and base.safe_text(completed[ts_code].get("recheck_status")) == "completed":
            print(f"[{idx}/{total}] 跳过已复核：{ts_code} {name}")
            continue

        record = record_by_code.get(ts_code)
        if record is None:
            print(f"[{idx}/{total}] 未在原始数据中找到 {ts_code}，跳过。")
            continue

        trigger = (
            "非持牌金融相关主体"
            if record.get("financial_entity_type") == "金融相关主体"
            else "低置信度"
        )
        print(
            f"[{idx}/{total}] 复核主营业务：{ts_code} {name}"
            f"（触发原因：{trigger}；第一轮confidence={meta_row.get('confidence')}）"
        )

        try:
            user_prompt = build_recheck_company_prompt(record, meta_row)
            raw_result, usage = call_recheck_model(
                client, RECHECK_COMPANY_SYSTEM_PROMPT, user_prompt
            )
            validate_recheck_company_result(raw_result)
            result = calibrate_recheck_company_result(raw_result, record)

            result.update({
                "recheck_key": ts_code,
                "ts_code": ts_code,
                "name": name,
                "industry": record.get("industry"),
                "round1_financial_entity_type": record.get("financial_entity_type"),
                "round1_financial_entity_categories": record.get("financial_entity_categories"),
                "round1_fintech_related": meta_row.get("fintech_related"),
                "round1_confidence": meta_row.get("confidence"),
                "round1_tracks": meta_row.get("tracks"),
                "round1_reason": meta_row.get("reason"),
                "recheck_trigger": trigger,
                "recheck_model": RECHECK_MODEL_NAME,
                "recheck_prompt_tokens": getattr(usage, "prompt_tokens", None),
                "recheck_completion_tokens": getattr(usage, "completion_tokens", None),
                "recheck_total_tokens": getattr(usage, "total_tokens", None),
                "recheck_status": "completed",
            })

            append_recheck_checkpoint(RECHECK_COMPANY_CHECKPOINT, result)
            completed[ts_code] = result

            verdict_changed = (
                base.is_true(meta_row.get("fintech_related"))
                != result["revised_fintech_related"]
            )
            print(
                "    复核结论="
                f"{result['revised_fintech_related']}（第一轮={meta_row.get('fintech_related')}，"
                f"{'★结论发生变化' if verdict_changed else '结论一致'}），"
                f"复核置信度={result['revised_confidence']}，"
                f"疑似伪金融科技={result['is_pseudo_fintech_false_positive']}，"
                f"金融主体类型疑似误判={result['entity_type_seems_misclassified']}"
            )

        except Exception as exc:
            print(f"    复核失败：{exc}")
            failure_result = {
                "recheck_key": ts_code,
                "ts_code": ts_code,
                "name": name,
                "round1_financial_entity_type": record.get("financial_entity_type"),
                "round1_confidence": meta_row.get("confidence"),
                "recheck_trigger": trigger,
                "recheck_status": "failed",
                "recheck_error": str(exc),
            }
            append_recheck_checkpoint(RECHECK_COMPANY_CHECKPOINT, failure_result)
            completed[ts_code] = failure_result

        time.sleep(RECHECK_REQUEST_INTERVAL)

    export_company_recheck_csv(completed)


def export_company_recheck_csv(completed: dict[str, dict]) -> None:
    if not completed:
        print("没有可导出的主营业务层面复核结果。")
        return

    rows = list(completed.values())
    df = pd.DataFrame(rows)
    for col in ["revised_tracks", "round1_tracks", "genuine_finance_evidence",
                "excluded_generic_smart_business_terms"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: "、".join(v) if isinstance(v, list) else v)

    RECHECK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RECHECK_COMPANY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  主营业务层面复核结果已导出：{RECHECK_COMPANY_OUTPUT_CSV}")

    if "recheck_status" in df.columns:
        completed_df = df[df["recheck_status"] == "completed"]
        if not completed_df.empty:
            if "is_pseudo_fintech_false_positive" in completed_df.columns:
                flagged = completed_df[completed_df["is_pseudo_fintech_false_positive"] == True]  # noqa: E712
                print(f"  疑似伪金融科技（主营业务层面）：{len(flagged)} / {len(completed_df)}")
            if "entity_type_seems_misclassified" in completed_df.columns:
                mis_typed = completed_df[completed_df["entity_type_seems_misclassified"] == True]  # noqa: E712
                print(f"  金融主体类型疑似误判：{len(mis_typed)} / {len(completed_df)}")
            if {"revised_fintech_related", "round1_fintech_related"} <= set(completed_df.columns):
                changed = completed_df[
                    completed_df["revised_fintech_related"].astype(str)
                    != completed_df["round1_fintech_related"].astype(str)
                ]
                print(f"  结论发生变化（主营业务层面）：{len(changed)} / {len(completed_df)}")


# ============================================================
# 12. 导出与汇总
# ============================================================

def export_final_merged_summary() -> None:
    """
    汇总年报层面与主营业务层面的复核结果：
    - 若某公司同时有年报层面复核结果，优先采用年报层面结论（证据更具体、更权威）；
    - 否则采用主营业务层面复核结论。
    """
    frames: list[pd.DataFrame] = []
    covered_codes: set[str] = set()

    if RECHECK_ANNUAL_OUTPUT_CSV.exists():
        df_a = read_csv_robust(RECHECK_ANNUAL_OUTPUT_CSV, dtype={"ts_code": str})
        if "recheck_status" in df_a.columns:
            df_a = df_a[df_a["recheck_status"] == "completed"]
        if not df_a.empty and "revised_confidence" in df_a.columns:
            df_a = df_a.sort_values(["ts_code", "revised_confidence"], ascending=[True, False])
            df_a = df_a.drop_duplicates(subset=["ts_code"], keep="first")
            df_a = df_a.rename(columns={
                "revised_annual_report_fintech_activity": "final_fintech_related",
                "revised_confidence": "final_confidence",
                "revised_tracks": "final_tracks",
            })
            df_a["recheck_source"] = "annual_report_level"
            keep_cols = [
                "ts_code", "name", "final_fintech_related", "final_confidence",
                "final_tracks", "is_pseudo_fintech_false_positive",
                "recommend_manual_review", "recheck_trigger", "recheck_source",
            ]
            keep_cols = [c for c in keep_cols if c in df_a.columns]
            frames.append(df_a[keep_cols])
            covered_codes = set(df_a["ts_code"])

    if RECHECK_COMPANY_OUTPUT_CSV.exists():
        df_c = read_csv_robust(RECHECK_COMPANY_OUTPUT_CSV, dtype={"ts_code": str})
        if "recheck_status" in df_c.columns:
            df_c = df_c[df_c["recheck_status"] == "completed"]
        if not df_c.empty:
            df_c = df_c[~df_c["ts_code"].isin(covered_codes)]
        if not df_c.empty:
            df_c = df_c.rename(columns={
                "revised_fintech_related": "final_fintech_related",
                "revised_confidence": "final_confidence",
                "revised_tracks": "final_tracks",
            })
            df_c["recheck_source"] = "company_level_only"
            keep_cols = [
                "ts_code", "name", "final_fintech_related", "final_confidence",
                "final_tracks", "is_pseudo_fintech_false_positive",
                "recommend_manual_review", "recheck_trigger", "recheck_source",
            ]
            keep_cols = [c for c in keep_cols if c in df_c.columns]
            frames.append(df_c[keep_cols])

    if not frames:
        print("\n暂无可汇总的复核结果（可能两个环节都尚未产出成功记录）。")
        return

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["is_pseudo_fintech_false_positive", "recommend_manual_review", "final_confidence"]
                 if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(
            sort_cols,
            ascending=[False] * (len(sort_cols) - 1) + [True] if sort_cols else None,
        )

    RECHECK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RECHECK_FINAL_MERGED_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"复核汇总已导出：{RECHECK_FINAL_MERGED_CSV}")
    print(f"  共复核 {len(merged)} 家公司/记录")
    if "is_pseudo_fintech_false_positive" in merged.columns:
        print(f"  疑似伪金融科技（建议从名单中剔除）：{int(merged['is_pseudo_fintech_false_positive'].sum())} 家")
    if "recommend_manual_review" in merged.columns:
        print(f"  建议人工复核：{int(merged['recommend_manual_review'].sum())} 家")
    print("=" * 70)


# ============================================================
# 13. 主程序
# ============================================================

def main() -> None:
    RECHECK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("金融科技识别结果 · 二次复核（更严格模型 + 句子级细粒度证据切分）")
    print(f"  复用第一轮脚本模块：{base.__name__}")
    print(f"  复核模型：{RECHECK_MODEL_NAME}（第一轮通常使用 deepseek-v4-flash，本轮改用更高级模型）")
    print(f"  推理模式（thinking）：{RECHECK_ENABLE_THINKING}")
    print(f"  置信度复核阈值：< {CONFIDENCE_THRESHOLD}")
    print("  复核范围：financial_entity_type == '金融相关主体'（全部） 或 confidence < 阈值（全部）")
    print("=" * 70)

    client = build_client()

    if RUN_ANNUAL_REPORT_RECHECK:
        run_annual_report_recheck(client)
    else:
        print("\n[年报层面复核] 已在配置中关闭，跳过。")

    if RUN_COMPANY_LEVEL_RECHECK:
        run_company_level_recheck(client)
    else:
        print("\n[主营业务层面复核] 已在配置中关闭，跳过。")

    export_final_merged_summary()

    print("\n输出文件一览：")
    print(f"  年报层面复核明细：{RECHECK_ANNUAL_OUTPUT_CSV}")
    print(f"  主营业务层面复核明细：{RECHECK_COMPANY_OUTPUT_CSV}")
    print(f"  复核汇总（重点看 is_pseudo_fintech_false_positive / recommend_manual_review 两列）：")
    print(f"    {RECHECK_FINAL_MERGED_CSV}")
    print("\n二次复核全部完成。")


if __name__ == "__main__":
    main()
