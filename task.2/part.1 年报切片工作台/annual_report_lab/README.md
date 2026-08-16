# 年报“管理层讨论与分析”结构化切片工具 v3

本版本把任务主目标收敛为：**从上市公司年度报告中完整识别并裁出“管理层讨论与分析”对应章节**。
业务概况、核心竞争力和研发投入仍保留为兼容性辅助结果，但它们不再决定任务是否成功。

## v3 的核心变化

1. **不再绑定章节序号**：标题可以是“第三节”“第四部分”“一、”、数字序号，甚至无序号。
2. **支持标题别名**：包括“管理层讨论与分析”“管理层讨论及/和分析”“经营层讨论与分析”
   “经营情况讨论与分析”“董事会报告”等。
3. **完整边界按标题层级计算**：起点命中后，终点取下一个同级或更高级标题；不会因为内部小标题而提前截断。
4. **内容簇兜底**：标题被 OCR/抽取破坏时，利用“所处行业、主要业务、主营业务分析、
   资产负债状况、投资状况、未来展望”等 MD&A 特有条目序列反推整段。
5. **结构化 PDF 作为主产物**：直接从原始 PDF 裁切，保留段落、表格、图片、字体与矢量线条；
   TXT 仅作为检索和审计副本。
6. **统一命名**：例如：
   `sections/2025/000750.SZ_国海证券_2025/000750.SZ_国海证券_2025_管理层讨论与分析.pdf`

## 运行

```bash
pip install -r requirements.txt
python run_cli.py --csv company_list.csv --out ./annual_reports_out --years 2025
```

Web 方式仍可运行：

```bash
python app.py
```

## 输出

```text
annual_reports_out/
├─ raw/2025/000750.SZ_国海证券_2025_年度报告.pdf
├─ fulltext/2025/000750.SZ_国海证券_2025.txt
├─ sections/2025/000750.SZ_国海证券_2025/
│  ├─ 000750.SZ_国海证券_2025_管理层讨论与分析.pdf   # 主产物，保留版式
│  ├─ 000750.SZ_国海证券_2025_管理层讨论与分析.txt   # 辅助文本
│  └─ _meta.json
├─ report.xlsx
└─ report.csv
```

结构化裁切优先使用 PyMuPDF 精确定位首尾标题的纵坐标；若坐标搜索失败，会降级为整页复制，
仍保留原 PDF 结构，并在 `_meta.json` 的 `structural` 字段中记录裁切模式。

## 自测

```bash
python tests/test_slicer.py
python tests/test_offline_pipeline.py
python tests/test_structured_pdf.py
```
