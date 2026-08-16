# 原始数据放置目录

请把第三部分产出的招聘数据文件放在本目录下，文件名保持为：

```
company_jobs_merged.csv
```

编码为 GB18030。`01_job_classification.py` 会从这里读取；若文件缺失，脚本会直接抛出
`FileNotFoundError` 并给出提示，不会静默失败。

如果不想移动文件，也可以用环境变量指定路径：

```bash
export JOBS_RAW_CSV=/your/path/company_jobs_merged.csv
```

本目录**已随包附带** `company_jobs_merged.csv`，因此解压后可直接按 `README.md` 第 3 节
顺序运行全流程，无需额外准备数据。若替换为其它数据源，保持列结构与文件名一致即可。
