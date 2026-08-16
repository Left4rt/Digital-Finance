# -*- coding: utf-8 -*-
"""
Step 3 — 构建 2025 年财务面板（资产 / 盈利 / 规模 / 研发投入），并标注每个数据的来源。

三个来源，全部在 fin_source 字段中显式标注，可追溯、可替换：
  ① 年报反推   : 由 step1 解析的「研发投入金额 ÷ 研发投入占营业收入比例」反推营业收入。
                 与已核实公司对照，误差 < 0.1%（见 out/revenue_crosscheck.csv）。
  ② 年报检索   : 2025 年年度报告 / 年报摘要公开披露值（联网检索核对，见 SOURCES 表）。
  ③ 结构分类   : 资产密集度按行业属性归类（券商/银行/信托/期货=资产密集，
                 总资产量级 10^11～10^12；科技服务商=轻资产，量级 10^9）。
                 这是序数代理变量，不是估算数字，不参与需要具体金额的计算。

输出：out/financial_panel_2025.csv, out/revenue_crosscheck.csv
"""
import os
import pandas as pd

OUT = "out"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 检索核实值
# 单位：亿元；None = 未取到。ROE 为加权平均净资产收益率(%)。
VERIFIED = {
    # id        : (营业收入, 归母净利润, 总资产, ROE)
    "600570.SH": (57.83, 12.31, 159.07, 13.33),    # 恒生电子 2025年报摘要
    "300663.SZ": (5.91, -3.95, 17.66, -64.06),     # 科蓝软件 2025年报摘要
    "688590.SH": (21.07, -1.27, 32.10, None),      # 新致软件 2025年报摘要
    "002657.SZ": (8.56, -1.10, None, None),        # 中科金财 2025年报全文
    "300348.SZ": (19.58, 0.21, 29.51, 0.97),       # 长亮科技 2025年度财务决算报告
    "300674.SZ": (36.23, 4.32, None, None),        # 宇信科技 2025年报
    "603927.SH": (60.28, 2.24, None, None),        # 中科软 2025年报摘要
    "002987.SZ": (48.39, 3.27, None, None),        # 京北方 2025年报
    "603171.SH": (20.54, 1.36, None, None),        # 税友股份 2025年报
    "688318.SH": (3.68, 3.15, 40.55, None),        # 财富趋势 2025年报
    "300085.SZ": (7.58, -1.33, 9.13, -27.22),      # 银之杰 2025年报摘要
    "300033.SZ": (60.29, 32.05, None, None),       # 同花顺 2025年报
    "300059.SZ": (160.68, 120.85, None, None),     # 东方财富 2025年报
    "601788.SH": (108.52, 37.29, None, None),      # 光大证券 2025年报
    "000997.SZ": (87.58, 10.11, None, 13.91),      # 新大陆 2025年报摘要+界面新闻
    "300130.SZ": (31.80, 4.69, None, None),        # 新国都 2025年报(证券之星年报简析)
    "300377.SZ": (13.70, -0.0526, None, None),     # 赢时胜 2025年报(投资者关系活动记录表)
    "600446.SH": (24.19, -1.68, None, None),       # 金证股份 2025年报
    "300541.SZ": (35.84, 1.19, None, 7.03),        # 先进数通 2025年报
    "300996.SZ": (8.25, 0.7398, None, None),       # 普联软件 2025年报
    "300468.SZ": (6.31, 0.7430, None, None),       # 四方精创 2025年报
    "300803.SZ": (21.46, 2.28, None, None),        # 指南针 2025年报
    "600928.SH": (None, 26.50, 5381.67, None),     # 西安银行 2025年报摘要；总资产由2026Q1报告倒推
}

# 资产密集度分类（按行业属性，序数代理）
ASSET_HEAVY_SEGMENTS = {"证券", "银行", "信托", "期货", "其他金融"}

# step5 的 company_tech_stack.csv 只覆盖 32 家，其余 26 家在此补齐行业归类
SEGMENT_FIX = {
    "000567.SZ": "其他金融", "000686.SZ": "证券", "000750.SZ": "证券",
    "000776.SZ": "证券", "001236.SZ": "期货", "002537.SZ": "其他",
    "002797.SZ": "证券", "002926.SZ": "证券", "300079.SZ": "其他",
    "300333.SZ": "金融IT服务商", "300380.SZ": "金融IT服务商",
    "300803.SZ": "互联网金融信息", "300941.SZ": "支付与商户服务",
    "600095.SH": "证券", "600624.SH": "其他", "600816.SH": "信托",
    "600864.SH": "证券", "600909.SH": "证券", "601059.SH": "证券",
    "601099.SH": "证券", "601136.SH": "证券", "601198.SH": "证券",
    "601375.SH": "证券", "601456.SH": "证券", "601519.SH": "互联网金融信息",
    "603123.SH": "其他",
}


def main():
    rd = pd.read_csv(f"{OUT}/rd2025_parsed.csv")
    stack = pd.read_csv("company_tech_stack.csv")
    stack.columns = [c.lstrip("\ufeff") for c in stack.columns]
    seg = stack.set_index("company_id")[["segment", "board"]].to_dict("index")

    rows = []
    for _, r in rd.iterrows():
        cid = r.company_id
        v = VERIFIED.get(cid, (None, None, None, None))
        rev_web, np_web, ta_web, roe_web = v
        rev_imp = (r.revenue_implied_2025 / 1e8
                   if pd.notna(r.revenue_implied_2025) else None)

        if rev_web is not None:
            revenue, src = rev_web, "年报检索"
        elif rev_imp is not None and r.get("parse_flag", "") != "营收反推存疑":
            revenue, src = round(rev_imp, 2), "年报反推"
        else:
            revenue, src = None, "缺失"

        s = seg.get(cid, {}).get("segment") or SEGMENT_FIX.get(cid, "未分类")
        rows.append(dict(
            company_id=cid, company_name=r.company_name,
            segment=s, board=seg.get(cid, {}).get("board", ""),
            in_kg_sample=cid in seg,
            revenue_2025_yi=revenue, revenue_source=src,
            revenue_implied_yi=round(rev_imp, 2) if rev_imp else None,
            net_profit_2025_yi=np_web,
            total_assets_2025_yi=ta_web,
            roe_2025_pct=roe_web,
            net_margin_pct=(round(np_web / revenue * 100, 2)
                            if (np_web is not None and revenue) else None),
            is_loss=(None if np_web is None else int(np_web < 0)),
            rd_amount_2025_yi=(round(r.rd_amount_2025 / 1e8, 4)
                               if pd.notna(r.rd_amount_2025) else None),
            rd_intensity_pct=r.rd_ratio_2025 if pd.notna(r.rd_ratio_2025) else None,
            rd_growth_pct=(round(r.rd_growth_pct, 2)
                           if pd.notna(r.rd_growth_pct) else None),
            rd_cap_ratio_pct=(r.rd_cap_ratio_2025
                              if pd.notna(r.rd_cap_ratio_2025) else None),
            rd_staff_2025=(r.rd_staff_2025 if pd.notna(r.rd_staff_2025) else None),
            rd_disclosed=bool(r.rd_disclosed) and r.rd_format != "无投入表",
            rd_format=r.rd_format,
            asset_intensity=("资产密集型金融机构"
                             if s in ASSET_HEAVY_SEGMENTS else "轻资产科技服务商"),
        ))

    fin = pd.DataFrame(rows)
    fin.to_csv(f"{OUT}/financial_panel_2025.csv", index=False, encoding="utf-8-sig")

    # 反推法校验：同时有检索值和反推值的公司
    cc = fin[fin.revenue_implied_yi.notna() &
             fin.company_id.isin(VERIFIED.keys())].copy()
    cc["revenue_web_yi"] = cc.company_id.map(lambda x: VERIFIED[x][0])
    cc["abs_err_pct"] = ((cc.revenue_implied_yi - cc.revenue_web_yi).abs()
                         / cc.revenue_web_yi * 100).round(3)
    cc[["company_id", "company_name", "revenue_implied_yi", "revenue_web_yi",
        "abs_err_pct"]].to_csv(f"{OUT}/revenue_crosscheck.csv", index=False,
                               encoding="utf-8-sig")

    print(cc[["company_name", "revenue_implied_yi", "revenue_web_yi",
              "abs_err_pct"]].to_string(index=False))
    print("\n平均绝对误差 %.3f%%  中位数 %.3f%%" %
          (cc.abs_err_pct.mean(), cc.abs_err_pct.median()))
    print("\n覆盖度： 营收", fin.revenue_2025_yi.notna().sum(),
          "| 净利", fin.net_profit_2025_yi.notna().sum(),
          "| 总资产", fin.total_assets_2025_yi.notna().sum(),
          "| 研发投入", fin.rd_amount_2025_yi.notna().sum(), "/", len(fin))
    print("KG(2024)样本内公司数:", fin.in_kg_sample.sum())


if __name__ == "__main__":
    main()
