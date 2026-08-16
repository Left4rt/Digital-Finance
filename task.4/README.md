# 第四部分：人类与机器的协同思考 —— 代码与数据说明

基于 40 家 A 股金融/财务相关上市公司的 3471 条原始招聘记录（清洗后 3254 条有效岗位），
构建金融财务岗位 AI 替代风险量化框架，回答课程第四部分的问题 9、10、11。

正文报告见 **`Part4_Report.md`**，人机协同流程与职业建议详稿见 **`05_human_ai_workflow.md`**。

---

## 1. 目录结构

```
task4/
├── README.md                            本文件
├── requirements.txt                     依赖版本
├── Part4_Report.md                      第四部分正文报告
├── 05_human_ai_workflow.md              协同流程设计与职业建议（问题 10、11 详稿）
├── config/
│   ├── __init__.py
│   └── taxonomy.py                      分类体系 / 五维词典 / 专家先验 / 模型超参
├── raw_data/
│   └── company_jobs_merged.csv          原始招聘数据（★ 运行前需自行放入，见第 2 节）
├── 01_job_classification.py             模块 1：清洗、岗位名标准化、岗位分类
├── 02_feature_engineering.py            模块 2：五维证据特征抽取与聚合
├── 03_risk_scoring.py                   模块 3：风险评分、分层、敏感性分析
├── 04_visualization.py                  模块 4：风险结果可视化（fig1–fig6、fig9）
├── 05_human_ai_workflow.py              模块 5：任务拆解与协同流程图（fig7–fig8）
├── data/                                运行时自动生成的全部中间产物与结果
└── figures/                             运行时自动生成的 9 张图表
```

`data/` 与 `figures/` 由脚本自动创建，无需手工建立。

---

## 2. 运行前准备

### 2.1 放入原始数据

把第三部分产出的 `company_jobs_merged.csv`（GB18030 编码）放到 `raw_data/` 目录下：

```
task4/raw_data/company_jobs_merged.csv
```

也可以不移动文件，改用环境变量指定路径：

```bash
export JOBS_RAW_CSV=/your/path/company_jobs_merged.csv     # macOS / Linux
set JOBS_RAW_CSV=D:\your\path\company_jobs_merged.csv      # Windows CMD
```

脚本中**不包含任何绝对路径**，所有路径均以脚本所在目录（`Path(__file__).parent`）为基准解析，换机器可直接运行。

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

### 2.3 中文字体（仅影响图表）

绘图脚本按以下顺序寻找中文字体：`Noto Sans CJK JP` → `Noto Sans CJK SC` → `DejaVu Sans`。
若图中中文显示为方框，请安装任一 Noto CJK 字体，或在 `04_visualization.py` /
`05_human_ai_workflow.py` 顶部的 `plt.rcParams["font.sans-serif"]` 中改为本机已有的中文字体
（Windows 可用 `Microsoft YaHei`，macOS 可用 `PingFang SC`）。

---

## 3. 运行顺序

必须按编号顺序执行，后一步依赖前一步的产物：

```bash
cd task4
python 01_job_classification.py     # -> data/01_*.csv, data/01_classification_log.json
python 02_feature_engineering.py    # -> data/02_*.csv, data/02_feature_log.json
python 03_risk_scoring.py           # -> data/03_*.csv, data/03_scoring_log.json
python 04_visualization.py          # -> figures/fig1–fig6, fig9
python 05_human_ai_workflow.py      # -> figures/fig7–fig8, data/05_task_decomposition.csv
```

模块 5 不依赖招聘数据（任务拆解表内置于脚本中），可独立运行。

### 依赖关系

| 脚本 | 输入 | 输出 |
|---|---|---|
| `01` | `raw_data/company_jobs_merged.csv` | `data/01_*` |
| `02` | `data/01_jobs_classified.csv` | `data/02_*` |
| `03` | `data/02_category_feature_matrix.csv`、`data/02_job_level_features.csv` | `data/03_*` |
| `04` | `data/03_risk_scores_category.csv`、`data/03_risk_scores_typical_roles.csv`、`data/02_category_feature_matrix.csv`（仅 fig6 需要） | `figures/fig1–6, 9` |
| `05` | 无（自包含） | `figures/fig7–8`、`data/05_task_decomposition.csv` |

> 若 `data/02_category_feature_matrix.csv` 缺失，`04` 会跳过 fig6 并打印提示，其余 6 张图正常输出。
> 完整 9 张图需要从 `01` 开始跑完整条链路。

---

## 4. 结果口径说明（阅读报告与图表前请先看这一节）

本研究混合使用了三类不同性质的数字，报告中已逐处标注，此处集中说明：

| 性质 | 含义 | 例子 |
|---|---|---|
| **招聘数据实测** | 直接由 3254 条招聘记录统计得到 | 各类岗位数、薪资中位数、技能覆盖率（39.4% / 14.5% / 74.1%）、词频 |
| **模型评分** | 数据证据分与专家先验分融合后的结果 | 五维得分 R/S/D/A/H、Risk 指数、风险分层 |
| **情景参数** | 研究者在方案设计中设定的假设值，**非实测** | 工时压缩率 15% / 55% / 100% 及由此推演的 `21:39:40`；分流比例 70% / 30%；OCR 置信度阈值 0.95；PD 灰区 2%–15%；人工抽查率 ≥10% |

另有两点方法上的口径约定：

1. **代理变量 ≠ 维度本身**。`D_data` 是"数字化技能需求强度"（招聘方是否要求 Python/SQL/BI），
   而维度 `D` 是"任务数据数字化程度"；`A_data` 是"AI 技能需求强度"，而维度 `A` 是"AI 技术成熟度"。
   两者相关但不等价（如柜面岗几乎不要求 SQL，但业务流程已高度数字化），
   因此模型用专家先验对维度本身打分，并用样本量收缩限制代理变量的权重。
2. **小样本岗位的排名应谨慎解释**。典型职能岗表中标注"证据强度=弱"者样本量 < 20
   （资产盘点 n=10、清结算 n=7、信贷审批 n=12），其得分以专家先验为主，
   稳健的结论是"这一组任务整体处于高替代压力区间"，而非具体名次。

---

## 5. 本压缩包当前状态：已完成端到端复现

本包内的**全部 CSV / JSON / 图表，均由 `01`→`02`→`03`→`04`→`05` 从原始数据
`company_jobs_merged.csv` 一次性重新生成**，非历史遗留产物。

| 项目 | 状态 |
|---|---|
| 目录结构（`config/` / `data/` / `figures/` / `raw_data/`） | ✅ 与报告第八节交付清单一致 |
| 模块导入（`from config.taxonomy import ...`） | ✅ 已加 `config/__init__.py` |
| 绝对路径 | ✅ 已全部清除，改为相对路径 + 环境变量覆盖 |
| 五个脚本顺序执行 | ✅ 全部通过，无报错 |
| `data/` 中间产物 | ✅ **17 个文件全部生成** |
| `figures/` 图表 | ✅ **9 张全部生成**（含此前缺失的 `fig6_evidence.png`） |
| 报告数字与重跑结果一致性 | ✅ 已逐项核对，见下表 |

### 5.1 关键数字复核（报告值 vs 重跑值）

| 指标 | 报告值 | 重跑值 | 结果 |
|---|---|---|---|
| 原始记录 / 清洗后有效岗位 | 3471 / 3254 | 3471 / 3254 | ✅ |
| 唯一职位名收敛 | 2008 → 1315 | 2008 → 1315 | ✅ |
| 规则命中 / 相似度 / 兜底 | 2891 / 73 / 290 | 2891 / 73 / 290 | ✅ |
| 薪资解析成功率 | 99.91% | 99.91% | ✅ |
| AI / 数字化 / 人际技能覆盖率 | 14.5% / 39.4% / 74.1% | 14.51% / 39.37% / 74.06% | ✅ |
| 信贷与授信审批 Risk | 73.8 | 73.8 | ✅ |
| 财务会计核算 Risk | 72.8 | 72.8 | ✅ |
| 客户营销与销售 Risk | 32.9 | 32.9 | ✅ |
| 全样本薪资中位数 | 11.5K | 11.5K | ✅ |
| 财富管理 / 算法岗薪资中位数 | 8.5K / 20.25K | 8.5K / 20.25K | ✅ |
| 财富+销售数字化技能覆盖率 | 4.1% | 4.06% | ✅ |
| 敏感性 Spearman（等权/纯专家/纯数据） | 0.9615 / 0.9725 / 0.7747 | 完全一致 | ✅ |
| 聚类与阈值分层一致率 | 61.5% | 61.5% | ✅ |
| 典型职能岗覆盖岗位数 | ~~1227（37.7%）~~ | **1225（37.6%）** | ⚠️ 已订正报告 |

唯一需要订正的是最后一项：典型职能岗实际覆盖 **1225 条**（37.6%），报告原写 1227 条，
已在 `Part4_Report.md` 第 4.4 节改正，并补注了因样本量不足 5 而被剔除的 2 个职能岗
（财务分析与预算管理 n=2、反欺诈与反洗钱 n=0，记录见 `data/03_roles_dropped.csv`）。
其余全部指标与报告完全一致。

---

## 6. 环境

开发与验证环境：Python 3.10+，pandas 2.x、scikit-learn 1.x、matplotlib 3.x、numpy 1.24+。
全流程离线运行，不需要联网。
