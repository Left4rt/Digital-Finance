# data 目录说明

本目录存放全流程的中间产物与结果，**全部由脚本运行时自动生成**，可逐层核查，
不存在无法追溯的"黑箱得分"。当前包内 17 个文件均由完整流程一次性重新生成。

| 文件 | 生成脚本 | 内容 |
|---|---|---|
| `01_jobs_classified.csv` | `01` | 3254 条岗位 + 标准职位名 + 岗位大类 + 归类方式 |
| `01_title_normalize_map.csv` | `01` | 原始职位名 → 标准职位名映射（2008 → 1315，可逐条抽查） |
| `01_category_summary.csv` | `01` | 13 个岗位大类规模统计 |
| `01_classification_log.json` | `01` | 清洗与分类过程日志 |
| `02_job_level_features.csv` | `02` | 岗位级五维证据（含每条 JD 命中的关键词，可核查） |
| `02_category_feature_matrix.csv` | `02` | 大类级特征矩阵（图 6 的唯一数据源） |
| `02_skill_frequency.csv` | `02` | 三类技能词频 Top30 |
| `02_feature_log.json` | `02` | 特征构建日志（全样本覆盖率） |
| `03_risk_scores_category.csv` | `03` | 13 大类风险总表（数据分/先验分/融合分 + Risk + 分层 + 先验依据） |
| `03_risk_scores_typical_roles.csv` | `03` | 16 个典型职能岗风险表（含证据强度标注） |
| `03_risk_scores_subjob.csv` | `03` | 71 个细分职位名风险表 |
| `03_dimension_detail.csv` | `03` | 五维数据分/先验分/融合分三列对照 |
| `03_sensitivity.csv` | `03` | 8 组权重敏感性检验 |
| `03_cluster_assignment.csv` | `03` | KMeans 聚类分层（探索性分析） |
| `03_roles_dropped.csv` | `03` | 样本量不足 5 而被剔除的职能岗（2 个） |
| `03_scoring_log.json` | `03` | 评分过程日志 |
| `05_task_decomposition.csv` | `05` | 信贷审批岗 13 项任务拆解 |

> 若删除本目录后重跑，需从 `01` 开始按顺序执行；`02_category_feature_matrix.csv`
> 缺失时 `04_visualization.py` 会跳过 fig6 并打印提示，其余图表正常输出。
