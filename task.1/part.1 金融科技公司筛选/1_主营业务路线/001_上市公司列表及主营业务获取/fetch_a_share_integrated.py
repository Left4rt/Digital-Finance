# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:51:35 2026

@author: 15774
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import tushare as ts


# ============================================================
# 1. 参数配置
# ============================================================

# 建议通过环境变量设置：
# macOS/Linux: export TUSHARE_TOKEN="xxx"
# Windows:     $env:TUSHARE_TOKEN="xxx"
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()

# 自定义 Tushare API 地址。
# 可通过环境变量覆盖：
# macOS/Linux: export TUSHARE_API_URL="https://ts.gyzcloud.top/api"
# Windows:     $env:TUSHARE_API_URL="https://ts.gyzcloud.top/api"
TUSHARE_API_URL = os.getenv(
    "TUSHARE_API_URL",
    "https://ts.gyzcloud.top/api",
).strip()

# 当前日期为2026年时，可将20251231作为最新完整年度报告期。
# 后续更新时修改为对应年度，例如20261231。
REPORT_PERIOD = os.getenv("REPORT_PERIOD", "20251231").strip()

# 是否包含北京证券交易所
INCLUDE_BSE = True

# 是否排除ST、*ST公司
EXCLUDE_ST = False

# 主营业务构成类型：
# P = 按产品
# I = 按行业
# D = 按地区
BUSINESS_TYPES = ("P", "I")

# 是否使用fina_mainbz_vip。
# 普通接口通常需要2000积分，逐只股票获取；
# VIP接口需要更高权限，可按报告期获取全市场。
USE_VIP_API = True

# 每次接口调用后的等待时间。
# 频率受账户积分和Tushare权限影响，可按实际情况调整。
REQUEST_INTERVAL = 0.45

# API失败后的最大重试次数
MAX_RETRIES = 5

# 测试时可通过环境变量限制股票数量：
# macOS/Linux: MAX_STOCKS=20 python fetch_a_share_main_business.py
# 设为0表示获取全部股票。
MAX_STOCKS = int(os.getenv("MAX_STOCKS", "0"))

# 是否强制重新抓取已经成功获取的数据
FORCE_REFRESH = False

# 输出目录
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "a_share_fintech.db"

STOCK_BASIC_FIELDS = (
    "ts_code,symbol,name,area,industry,fullname,market,"
    "exchange,curr_type,list_status,list_date,delist_date,"
    "is_hs,act_name,act_ent_type"
)

MAIN_BUSINESS_FIELDS = (
    "ts_code,end_date,bz_item,bz_code,bz_sales,"
    "bz_profit,bz_cost,curr_type,update_flag"
)


# ============================================================
# 2. 基础工具
# ============================================================

def validate_config() -> None:
    """检查必要配置。"""

    if not TUSHARE_TOKEN:
        raise RuntimeError(
            "未检测到TUSHARE_TOKEN。\n"
            "请先设置环境变量，例如：\n"
            'export TUSHARE_TOKEN="你的Token"'
        )

    if not TUSHARE_API_URL.startswith(("http://", "https://")):
        raise ValueError(
            f"TUSHARE_API_URL格式错误：{TUSHARE_API_URL}，"
            "必须以http://或https://开头。"
        )

    if not re.fullmatch(r"\d{8}", REPORT_PERIOD):
        raise ValueError(
            f"REPORT_PERIOD格式错误：{REPORT_PERIOD}，"
            "应为YYYYMMDD，例如20251231。"
        )

    valid_types = {"P", "I", "D"}
    invalid_types = set(BUSINESS_TYPES) - valid_types

    if invalid_types:
        raise ValueError(
            f"BUSINESS_TYPES存在无效值：{invalid_types}"
        )


def create_pro_api():
    """初始化Tushare Pro客户端，并接入自定义API地址。"""

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    pro._DataApi__http_url = TUSHARE_API_URL

    return pro


def api_call_with_retry(
    api_func: Callable,
    **kwargs,
) -> pd.DataFrame:
    """
    调用Tushare接口，出现异常时自动重试。

    Parameters
    ----------
    api_func:
        Tushare接口函数。
    kwargs:
        传给接口的参数。
    """

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = api_func(**kwargs)

            if result is None:
                return pd.DataFrame()

            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"接口返回类型不是DataFrame，而是：{type(result)}"
                )

            return result

        except Exception as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            # 指数退避：2、4、8、16秒……
            wait_seconds = min(2 ** attempt, 30)

            print(
                f"接口调用失败，第{attempt}/{MAX_RETRIES}次："
                f"{exc}"
            )
            print(f"{wait_seconds}秒后重试。")

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"接口调用连续失败{MAX_RETRIES}次：{last_error}"
    ) from last_error


# ============================================================
# 3. 数据库
# ============================================================

def initialize_database(conn: sqlite3.Connection) -> None:
    """初始化主营业务构成和抓取日志表。"""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS main_business (
            ts_code       TEXT NOT NULL,
            end_date      TEXT NOT NULL,
            bz_item       TEXT,
            bz_code       TEXT,
            bz_sales      REAL,
            bz_profit     REAL,
            bz_cost       REAL,
            curr_type     TEXT,
            update_flag   TEXT,
            query_type    TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_main_business_code_period
        ON main_business(ts_code, end_date)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_main_business_item
        ON main_business(bz_item)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_log (
            ts_code       TEXT NOT NULL,
            period        TEXT NOT NULL,
            query_type    TEXT NOT NULL,
            status        TEXT NOT NULL,
            row_count     INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY(ts_code, period, query_type)
        )
        """
    )

    conn.commit()


def save_stock_pool(
    conn: sqlite3.Connection,
    stocks: pd.DataFrame,
) -> None:
    """将股票池保存到SQLite。"""

    stocks.to_sql(
        "stock_basic",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stock_basic_ts_code
        ON stock_basic(ts_code)
        """
    )

    conn.commit()


def update_fetch_log(
    conn: sqlite3.Connection,
    ts_code: str,
    period: str,
    query_type: str,
    status: str,
    row_count: int = 0,
    error_message: str | None = None,
) -> None:
    """新增或更新抓取日志。"""

    conn.execute(
        """
        INSERT INTO fetch_log (
            ts_code,
            period,
            query_type,
            status,
            row_count,
            error_message,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ON CONFLICT(ts_code, period, query_type)
        DO UPDATE SET
            status = excluded.status,
            row_count = excluded.row_count,
            error_message = excluded.error_message,
            updated_at = excluded.updated_at
        """,
        (
            ts_code,
            period,
            query_type,
            status,
            row_count,
            error_message,
        ),
    )

    conn.commit()


def get_completed_tasks(
    conn: sqlite3.Connection,
    period: str,
) -> set[tuple[str, str]]:
    """
    读取已经完成的任务。

    返回：
        {(ts_code, query_type), ...}
    """

    query = """
        SELECT ts_code, query_type
        FROM fetch_log
        WHERE period = ?
          AND status IN ('success', 'no_data')
    """

    rows = conn.execute(query, (period,)).fetchall()

    return {(row[0], row[1]) for row in rows}


# ============================================================
# 4. 获取股票基础池
# ============================================================

def fetch_stock_pool(pro) -> pd.DataFrame:
    """获取当前正常上市A股。"""

    print("开始获取A股基础池。")

    stocks = api_call_with_retry(
        pro.stock_basic,
        exchange="",
        list_status="L",
        fields=STOCK_BASIC_FIELDS,
    )

    if stocks.empty:
        raise RuntimeError("stock_basic接口返回空数据。")

    # 默认只保留沪深交易所
    allowed_exchanges = ["SSE", "SZSE"]

    if INCLUDE_BSE:
        allowed_exchanges.append("BSE")

    stocks = stocks[
        stocks["exchange"].isin(allowed_exchanges)
    ].copy()

    # 排除ST和*ST
    if EXCLUDE_ST:
        is_st = stocks["name"].fillna("").str.contains(
            r"\*?ST",
            case=False,
            regex=True,
        )
        stocks = stocks[~is_st].copy()

    # 基本清洗
    stocks = (
        stocks
        .drop_duplicates(subset=["ts_code"], keep="last")
        .sort_values(["exchange", "ts_code"])
        .reset_index(drop=True)
    )

    if MAX_STOCKS > 0:
        stocks = stocks.head(MAX_STOCKS).copy()
        print(f"测试模式：只处理前{MAX_STOCKS}只股票。")

    print(f"股票基础池数量：{len(stocks):,}")

    return stocks


# ============================================================
# 5. 主营业务构成数据清洗与保存
# ============================================================

def clean_main_business(
    df: pd.DataFrame,
    period: str,
    query_type: str,
) -> pd.DataFrame:
    """统一主营业务构成字段和数据类型。"""

    expected_columns = [
        "ts_code",
        "end_date",
        "bz_item",
        "bz_code",
        "bz_sales",
        "bz_profit",
        "bz_cost",
        "curr_type",
        "update_flag",
    ]

    df = df.copy()

    for column in expected_columns:
        if column not in df.columns:
            df[column] = None

    df["end_date"] = df["end_date"].fillna(period).astype(str)
    df["query_type"] = query_type

    numeric_columns = [
        "bz_sales",
        "bz_profit",
        "bz_cost",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Tushare可能返回更新前、更新后的多条记录。
    # 优先保留update_flag较高的一条。
    df["_update_sort"] = pd.to_numeric(
        df["update_flag"],
        errors="coerce",
    ).fillna(-1)

    df = (
        df
        .sort_values("_update_sort")
        .drop_duplicates(
            subset=[
                "ts_code",
                "end_date",
                "query_type",
                "bz_code",
                "bz_item",
            ],
            keep="last",
        )
        .drop(columns="_update_sort")
        .reset_index(drop=True)
    )

    final_columns = expected_columns + ["query_type"]

    return df[final_columns]


def insert_main_business(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
) -> None:
    """批量写入主营业务构成表。"""

    if df.empty:
        return

    columns = [
        "ts_code",
        "end_date",
        "bz_item",
        "bz_code",
        "bz_sales",
        "bz_profit",
        "bz_cost",
        "curr_type",
        "update_flag",
        "query_type",
    ]

    records = []

    for row in df[columns].itertuples(index=False, name=None):
        cleaned_row = tuple(
            None if pd.isna(value) else value
            for value in row
        )
        records.append(cleaned_row)

    conn.executemany(
        """
        INSERT INTO main_business (
            ts_code,
            end_date,
            bz_item,
            bz_code,
            bz_sales,
            bz_profit,
            bz_cost,
            curr_type,
            update_flag,
            query_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    conn.commit()


# ============================================================
# 6. 普通接口：逐只股票获取
# ============================================================

def fetch_main_business_standard(
    pro,
    conn: sqlite3.Connection,
    stocks: pd.DataFrame,
    period: str,
) -> None:
    """使用普通fina_mainbz接口逐只股票获取数据。"""

    completed_tasks = (
        set()
        if FORCE_REFRESH
        else get_completed_tasks(conn, period)
    )

    total_tasks = len(stocks) * len(BUSINESS_TYPES)
    current_task = 0

    print(
        f"开始获取主营业务构成，报告期：{period}，"
        f"任务数量：{total_tasks:,}"
    )

    for stock in stocks.itertuples(index=False):
        ts_code = stock.ts_code
        company_name = stock.name

        for query_type in BUSINESS_TYPES:
            current_task += 1
            task_key = (ts_code, query_type)

            if task_key in completed_tasks:
                continue

            print(
                f"[{current_task:,}/{total_tasks:,}] "
                f"{ts_code} {company_name} "
                f"类型={query_type}"
            )

            try:
                df = api_call_with_retry(
                    pro.fina_mainbz,
                    ts_code=ts_code,
                    period=period,
                    type=query_type,
                    fields=MAIN_BUSINESS_FIELDS,
                )

                # 删除该股票、该报告期、该口径的旧数据，
                # 防止程序重复运行后数据重复。
                conn.execute(
                    """
                    DELETE FROM main_business
                    WHERE ts_code = ?
                      AND end_date = ?
                      AND query_type = ?
                    """,
                    (ts_code, period, query_type),
                )
                conn.commit()

                if df.empty:
                    update_fetch_log(
                        conn=conn,
                        ts_code=ts_code,
                        period=period,
                        query_type=query_type,
                        status="no_data",
                        row_count=0,
                    )

                    print("  未返回主营业务数据。")
                    time.sleep(REQUEST_INTERVAL)
                    continue

                df = clean_main_business(
                    df=df,
                    period=period,
                    query_type=query_type,
                )

                insert_main_business(conn, df)

                update_fetch_log(
                    conn=conn,
                    ts_code=ts_code,
                    period=period,
                    query_type=query_type,
                    status="success",
                    row_count=len(df),
                )

                print(f"  成功获取{len(df)}条。")

            except Exception as exc:
                error_message = str(exc)

                update_fetch_log(
                    conn=conn,
                    ts_code=ts_code,
                    period=period,
                    query_type=query_type,
                    status="failed",
                    row_count=0,
                    error_message=error_message,
                )

                print(f"  获取失败：{error_message}")

            time.sleep(REQUEST_INTERVAL)


# ============================================================
# 7. VIP接口：按报告期获取全市场
# ============================================================

def fetch_main_business_vip(
    pro,
    conn: sqlite3.Connection,
    period: str,
) -> None:
    """使用fina_mainbz_vip获取全市场数据。"""

    print(f"使用VIP接口获取报告期{period}的全市场数据。")

    for query_type in BUSINESS_TYPES:
        print(f"获取主营业务口径：{query_type}")

        df = api_call_with_retry(
            pro.fina_mainbz_vip,
            period=period,
            type=query_type,
            fields=MAIN_BUSINESS_FIELDS,
        )

        # 覆盖该报告期、该口径的原有数据
        conn.execute(
            """
            DELETE FROM main_business
            WHERE end_date = ?
              AND query_type = ?
            """,
            (period, query_type),
        )
        conn.commit()

        if df.empty:
            print(f"口径{query_type}未返回数据。")
            continue

        df = clean_main_business(
            df=df,
            period=period,
            query_type=query_type,
        )

        insert_main_business(conn, df)

        print(
            f"口径{query_type}成功获取："
            f"{len(df):,}条。"
        )

        time.sleep(REQUEST_INTERVAL)


# ============================================================
# 8. 导出CSV
# ============================================================

def export_results(
    conn: sqlite3.Connection,
    stocks: pd.DataFrame,
    period: str,
) -> None:
    """导出股票池、主营构成和合并结果。"""

    stock_file = DATA_DIR / "stock_pool.csv"
    raw_file = DATA_DIR / f"main_business_raw_{period}.csv"
    merged_file = DATA_DIR / f"stock_pool_main_business_{period}.csv"
    summary_file = DATA_DIR / f"main_business_summary_{period}.csv"
    failed_file = DATA_DIR / f"failed_tasks_{period}.csv"

    stocks.to_csv(
        stock_file,
        index=False,
        encoding="utf-8-sig",
    )

    main_business = pd.read_sql_query(
        """
        SELECT *
        FROM main_business
        WHERE end_date = ?
        ORDER BY ts_code, query_type, bz_sales DESC
        """,
        conn,
        params=(period,),
    )

    main_business.to_csv(
        raw_file,
        index=False,
        encoding="utf-8-sig",
    )

    # 左连接：即使某家公司没有主营构成数据，
    # 也会保留在最终股票池中。
    merged = stocks.merge(
        main_business,
        on="ts_code",
        how="left",
        suffixes=("", "_mainbz"),
    )

    merged.to_csv(
        merged_file,
        index=False,
        encoding="utf-8-sig",
    )

    # 生成公司级汇总
    if not main_business.empty:
        summary = (
            main_business
            .groupby(
                ["ts_code", "query_type"],
                as_index=False,
            )
            .agg(
                main_business_count=("bz_item", "count"),
                disclosed_sales_sum=("bz_sales", "sum"),
                disclosed_profit_sum=("bz_profit", "sum"),
                disclosed_cost_sum=("bz_cost", "sum"),
            )
        )

        summary = stocks.merge(
            summary,
            on="ts_code",
            how="left",
        )

    else:
        summary = stocks.copy()
        summary["query_type"] = None
        summary["main_business_count"] = 0
        summary["disclosed_sales_sum"] = None
        summary["disclosed_profit_sum"] = None
        summary["disclosed_cost_sum"] = None

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    failed_tasks = pd.read_sql_query(
        """
        SELECT *
        FROM fetch_log
        WHERE period = ?
          AND status = 'failed'
        ORDER BY ts_code, query_type
        """,
        conn,
        params=(period,),
    )

    failed_tasks.to_csv(
        failed_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n数据导出完成：")
    print(f"股票基础池：       {stock_file}")
    print(f"主营构成原始表：   {raw_file}")
    print(f"股票与主营合并表： {merged_file}")
    print(f"公司级汇总表：     {summary_file}")
    print(f"失败任务表：       {failed_file}")
    print(f"SQLite数据库：     {DB_PATH}")


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    validate_config()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Tushare版本：{ts.__version__}")
    print(f"API地址：{TUSHARE_API_URL}")
    print(f"目标报告期：{REPORT_PERIOD}")
    print(f"包含北交所：{INCLUDE_BSE}")
    print(f"排除ST公司：{EXCLUDE_ST}")
    print(f"主营构成口径：{BUSINESS_TYPES}")
    print(f"使用VIP接口：{USE_VIP_API}")

    pro = create_pro_api()

    with sqlite3.connect(DB_PATH) as conn:
        initialize_database(conn)

        stocks = fetch_stock_pool(pro)
        save_stock_pool(conn, stocks)

        if USE_VIP_API:
            fetch_main_business_vip(
                pro=pro,
                conn=conn,
                period=REPORT_PERIOD,
            )
        else:
            fetch_main_business_standard(
                pro=pro,
                conn=conn,
                stocks=stocks,
                period=REPORT_PERIOD,
            )

        export_results(
            conn=conn,
            stocks=stocks,
            period=REPORT_PERIOD,
        )

    print("\n全部任务执行完成。")


if __name__ == "__main__":
    main()
