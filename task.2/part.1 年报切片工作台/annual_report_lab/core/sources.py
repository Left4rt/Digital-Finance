# -*- coding: utf-8 -*-
"""年报来源层。

两条腿走路，任何一条通了就能拿到 PDF：
  A. Tushare 网关（用户提供的 token）—— 公司基础信息 + 公告接口 anns_d / anns。
  B. 巨潮资讯网 cninfo —— A 股公告的权威披露源，免登录，作为兜底与交叉校验。

对外只暴露 AnnouncementFinder.find(ts_code, name, year) → dict | None
"""
from __future__ import annotations

import datetime as dt
import io
import re
import threading
import time
from typing import Dict, List, Optional

import requests

from .config import (CNINFO_QUERY_URL, CNINFO_SEARCH_URL, CNINFO_STATIC,
                     CNINFO_TIMEOUT, TUSHARE_HTTP_URL, TUSHARE_RATE_PER_MIN,
                     TUSHARE_TIMEOUT, TUSHARE_TOKEN, USER_AGENT)


# ==========================================================================
# 限流器：令牌桶，保证不超过网关的 150 次/分钟
# ==========================================================================
class RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / max(1, per_minute)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = max(now, self._next) + self.interval


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return s


# ==========================================================================
# Tushare 网关客户端（直接走 HTTP，不依赖 tushare SDK 版本）
# ==========================================================================
class TushareClient:
    def __init__(self, token: str = TUSHARE_TOKEN, url: str = TUSHARE_HTTP_URL):
        self.token = token
        self.url = url
        self.sess = _session()
        self.limiter = RateLimiter(TUSHARE_RATE_PER_MIN)
        self.last_error = ""

    def call(self, api_name: str, params: Dict = None, fields: str = "",
             retries: int = 3) -> Optional[List[Dict]]:
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": params or {},
            "fields": fields,
        }
        for attempt in range(retries):
            self.limiter.acquire()
            try:
                r = self.sess.post(self.url, json=payload, timeout=TUSHARE_TIMEOUT)
                r.raise_for_status()
                js = r.json()
            except Exception as e:                       # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if js.get("code") != 0:
                self.last_error = str(js.get("msg", ""))
                # 权限 / 积分类错误重试无意义
                if any(k in self.last_error for k in ("积分", "权限", "不存在", "没有")):
                    return None
                time.sleep(1.5 * (attempt + 1))
                continue

            data = js.get("data") or {}
            cols = data.get("fields") or []
            items = data.get("items") or []
            return [dict(zip(cols, row)) for row in items]
        return None

    def ping(self) -> Dict:
        """连通性自检，用于页面上的状态灯。"""
        rows = self.call("stock_basic", {"list_status": "L"}, "ts_code,name")
        if rows is None:
            return {"ok": False, "msg": self.last_error or "网关无响应", "count": 0}
        return {"ok": True, "msg": "网关连通", "count": len(rows)}

    def stock_name_map(self) -> Dict[str, str]:
        rows = self.call("stock_basic", {"list_status": "L"}, "ts_code,name") or []
        rows += self.call("stock_basic", {"list_status": "D"}, "ts_code,name") or []
        rows += self.call("stock_basic", {"list_status": "P"}, "ts_code,name") or []
        return {r["ts_code"]: r["name"] for r in rows if r.get("ts_code")}

    # ---- 公告接口：不同网关开放程度不同，逐个尝试 ----
    def announcements(self, ts_code: str, start: str, end: str) -> List[Dict]:
        out: List[Dict] = []
        for api, fields in (("anns_d", "ts_code,name,title,url,rec_time,ann_date"),
                            ("anns", "ts_code,ann_date,title,url")):
            rows = self.call(api, {"ts_code": ts_code, "start_date": start,
                                   "end_date": end}, fields)
            if rows:
                for r in rows:
                    out.append({
                        "title": (r.get("title") or "").strip(),
                        "url": (r.get("url") or "").strip(),
                        "ann_date": str(r.get("ann_date") or r.get("rec_time") or "")[:10],
                        "source": f"tushare:{api}",
                    })
                break
        return out


# ==========================================================================
# 巨潮资讯网客户端
# ==========================================================================
class CninfoClient:
    def __init__(self):
        self.sess = _session()
        self.sess.headers.update({
            "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        })
        self._orgid_cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        # 巨潮没有公开配额，保守限速，避免被临时封禁
        self.limiter = RateLimiter(90)

    def _post(self, url: str, data: Dict, retries: int = 3):
        for attempt in range(retries):
            self.limiter.acquire()
            try:
                r = self.sess.post(url, data=data, timeout=CNINFO_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception:                             # noqa: BLE001
                time.sleep(1.2 * (attempt + 1))
        return None

    def org_id(self, code6: str) -> str:
        with self._lock:
            if code6 in self._orgid_cache:
                return self._orgid_cache[code6]
        org = ""
        js = self._post(CNINFO_SEARCH_URL, {"keyWord": code6, "maxNum": "10"})
        for item in (js or []):
            if str(item.get("code", "")).strip() == code6:
                org = item.get("orgId") or ""
                break
        with self._lock:
            self._orgid_cache[code6] = org
        return org

    def announcements(self, ts_code: str, start: str, end: str) -> List[Dict]:
        """start/end 为 YYYYMMDD。返回年报候选公告。"""
        code6, _, mkt = ts_code.partition(".")
        column = {"SH": "sse", "SZ": "szse", "BJ": "bse"}.get(mkt, "szse")
        org = self.org_id(code6)
        se = f"{start[:4]}-{start[4:6]}-{start[6:]}~{end[:4]}-{end[4:6]}-{end[6:]}"
        out: List[Dict] = []
        for col in (column, "szse", "sse", "bse"):
            data = {
                "pageNum": "1", "pageSize": "50", "column": col,
                "tabName": "fulltext", "plate": "",
                "stock": f"{code6},{org}" if org else code6,
                "searchkey": "", "secid": "",
                "category": "category_ndbg_szsh;", "trade": "",
                "seDate": se, "sortName": "", "sortType": "", "isHLtitle": "true",
            }
            js = self._post(CNINFO_QUERY_URL, data)
            if not js:
                continue
            for it in (js.get("announcements") or []):
                adj = it.get("adjunctUrl") or ""
                out.append({
                    "title": re.sub(r"<[^>]+>", "", it.get("announcementTitle") or "").strip(),
                    "url": CNINFO_STATIC + adj if adj else "",
                    "ann_date": dt.datetime.fromtimestamp(
                        (it.get("announcementTime") or 0) / 1000).strftime("%Y-%m-%d")
                    if it.get("announcementTime") else "",
                    "source": "cninfo",
                })
            if out:
                break
        return out


# ==========================================================================
# 年报公告筛选
# ==========================================================================
# 排除掉“摘要 / 英文版 / 审计报告 / 社会责任报告 / 更正公告”等噪声
_EXCLUDE = re.compile(
    r"摘要|英文|English|ENGLISH|H股|内部控制|社会责任|可持续发展|ESG报告|"
    r"审计报告|审核报告|专项说明|核查意见|募集资金|问询|回复|监事会|独立董事|"
    r"债券|存续期|受托管理|提示性公告|取消|催告|更正公告|补充公告|"
    r"半年度|第一季度|第三季度|季度报告"
)
_YEAR_RE = re.compile(r"(20\d{2})\s*(?:年年度报告|年度报告|年报)")


def pick_annual_report(cands: List[Dict], year: int) -> Optional[Dict]:
    """从公告列表里挑出指定年度的年报正文。

    规则：标题含“{year}年年度报告”且不含摘要/英文等噪声词；
    若有多份（原始版 + 修订版），取公告日期最新的一份。
    """
    hits = []
    for c in cands:
        title = c.get("title") or ""
        if not c.get("url"):
            continue
        if "年度报告" not in title and "年报" not in title:
            continue
        if _EXCLUDE.search(title):
            continue
        m = _YEAR_RE.search(title.replace(" ", ""))
        if not m or int(m.group(1)) != year:
            continue
        score = 0
        if re.search(r"修订|更新|更正后|重述", title):
            score += 1
        hits.append((c.get("ann_date") or "", score, c))
    if not hits:
        return None
    hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return hits[0][2]


class AnnouncementFinder:
    """统一入口：先 Tushare，拿不到再走巨潮。"""

    def __init__(self, tushare: TushareClient, cninfo: CninfoClient,
                 prefer: str = "auto", log=lambda *a: None):
        self.ts = tushare
        self.cn = cninfo
        self.prefer = prefer          # auto | tushare | cninfo
        self.log = log

    def find(self, ts_code: str, year: int) -> Optional[Dict]:
        start = f"{year + 1}0101"
        today = dt.date.today()
        end_date = min(dt.date(year + 2, 12, 31), today)
        end = end_date.strftime("%Y%m%d")
        if end < start:                        # 该年度年报尚未到披露期
            return None

        order = {"tushare": ["tushare"], "cninfo": ["cninfo"]}.get(
            self.prefer, ["tushare", "cninfo"])
        for src in order:
            try:
                cands = (self.ts.announcements(ts_code, start, end) if src == "tushare"
                         else self.cn.announcements(ts_code, start, end))
            except Exception as e:                        # noqa: BLE001
                self.log(f"[{ts_code}] {src} 公告检索异常：{e}")
                cands = []
            hit = pick_annual_report(cands, year)
            if hit:
                return hit
        return None


def download(url: str, dest: str, sess: requests.Session = None,
             retries: int = 3, timeout: int = 90) -> Dict:
    """带重试的下载，返回 {'ok':bool,'msg':str,'size':int}"""
    import os
    sess = sess or _session()
    last = ""
    for attempt in range(retries):
        try:
            with sess.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                size = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk)
                            size += len(chunk)
            if size < 10240:
                last = f"文件过小({size}B)，疑似下载不完整"
                os.remove(tmp)
                time.sleep(1.5 * (attempt + 1))
                continue
            with open(tmp, "rb") as f:
                head = f.read(5)
            if not head.startswith(b"%PDF"):
                last = "返回内容不是 PDF"
                os.remove(tmp)
                time.sleep(1.5 * (attempt + 1))
                continue
            os.replace(tmp, dest)
            return {"ok": True, "msg": "", "size": size}
        except Exception as e:                            # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return {"ok": False, "msg": last, "size": 0}
