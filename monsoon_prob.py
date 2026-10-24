#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国夏季东南季风（南海夏季风）概率预测工具
============================================

目标量：南海夏季风指数 SCSMI 的 JJAS（6-9 月）标准化序列（李建平团队，
基于动力学标准化季节变率 DNS 的统一季风指数框架，Li & Zeng 2002/2003）。
按标准化值分为三档：偏强 / 正常 / 偏弱（阈值 ±0.5σ）。

预报因子（起报时已公开，无未来泄漏）：
  * 起报前 6 个月的 ONI 轨迹（NOAA CPC，逐月更新）
  * 起报前 6 个月的 SOI 轨迹（NOAA CPC 标准化序列，逐月更新）
  * 前一年的 SCSMI JJAS 值/状态（南海夏季风自身的持续性）

方法沿用姊妹项目 enso-probability 的架构：
  1. 状态转移模型：P(今年季风态 | 去年季风态, 起报时 ENSO 态)，Dirichlet 先验
  2. 历史相似型模型：多维特征（6 月 ONI + 6 月 SOI + 6 月 DMI 轨迹 + 前一年 SCSMI，经滚动标准化）欧氏距离 + 高斯核加权
  3. 起报月分层 + 经验贝叶斯收缩：p = λ·p_同月 + (1-λ)·p_全历史，λ = n/(n+24)
  4. 组合权重按样本外三分类 Brier 技能分分配，负技能剔除，双负退化等权
  5. 90% 区间来自 Dirichlet 蒙特卡洛抽样（固定种子）
  6. 相似年份透明化 + 夏季逐月强度（6/7/8/9 月与 JJAS）加权点预测
  7. 逐月三档概率（相似型计票 + Dirichlet 后验区间）与偏强月数分布
  8. 技能面板：起报月（1-12 月）× 其唯一提前期（1-12 个月）的 Brier 技能
  9. 概率可靠性校准曲线

起报月约定：月 m ∈ 1..12；若 m ≤ 5 目标为同年 JJAS，若 m ≥ 6 目标为次年 JJAS。
提前期 L = 6-m（m≤5）或 18-m（m≥6），即距目标夏季（6 月）的月数。

限制：纯统计外推，非官方季节预报；SCSMI 每年夏季结束后由源站点更新，
存在约一年滞后；9 个月以上提前期技能接近气候态。

纯 Python 标准库实现，无第三方依赖。Python 3.9+。
"""

import argparse
import datetime
import html
import json
import math
import os
import random
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- 常量 ----

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
SOI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
# 李建平个人主页 -> 南海夏季风指数 SCSMI（附件 ASCII:876，1948-至今）
SCSMI_URL = "http://lijianping.cn/dct/attach/Y2xiOmNsYjpBU0NJSTo4NzY="

ONI_CACHE = os.path.join(DATA_DIR, "oni_cache.csv")
SOI_CACHE = os.path.join(DATA_DIR, "soi_cache.txt")
SCSMI_CACHE = os.path.join(DATA_DIR, "scsmi_cache.txt")

# 印度洋偶极子 DMI（西印度洋-东印度洋海温距平差）：NOAA PSL 月指数文本
# （基于 HadISST，1870-至今），作为季风的第三类外强迫因子（IOD 影响南海季风爆发与强度）。
DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
DMI_CACHE = os.path.join(DATA_DIR, "dmi_monthly_cache.txt")

# 南海季风爆发日期：NCEP/NCAR R1 逐日 850hPa 纬向风（PSL THREDDS OPeNDAP 文本子集），
# 区域 (10°-20°N, 110°-120°E)（南海中北部，主流文献与业务监测口径）候平均，
# 按王斌等 (2004) 标准判定爆发候。
# 注意：R1 已于 2026 年 3 月停产，完整爆发窗口（4 月初-7 月底）的最新可用年为 2025。
U850_BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure"
U850_CACHE = os.path.join(DATA_DIR, "u850_scs_cache.txt")
U850_MAX_YEAR = 2025     # R1 停产：不再更新以后年份的爆发窗口
U850_LON_IDX = (44, 48)  # 110°E..120°E（2.5° 网格索引 44..48）
U850_LAT_IDX = (28, 32)  # 20°N..10°N（索引 28..32，纬度自北向南；南海中北部业务监测区）
U850_LEVEL_IDX = 2       # 850 hPa 层索引
ONSET_WIN_DAYS = 123     # 抓取窗口：4 月 1 日 - 7 月 31 日
ONSET_START_PENTAD = 24  # 王斌等 (2004)：自第 24 候（4/26-30）起找爆发
ONSET_PERSIST = 3        # 爆发候定义：候平均 U850>0 且其后连续 2 候亦为正（共 3 候）
ONSET_VERIFY_START = 1979  # 爆发日期留一验证的最早目标年（卫星时代后样本更可靠）
ONSET_REGION_LABEL = "(10°-20°N, 110°-120°E)"

STATE_STRONG = "偏强"
STATE_NORMAL = "正常"
STATE_WEAK = "偏弱"
STATES = [STATE_STRONG, STATE_NORMAL, STATE_WEAK]
THRESHOLD = 0.5            # 状态阈值（标准化 σ 单位）

WINDOW = 6                 # 起报前轨迹窗口月数
PRIOR_STRENGTH = 4.0       # Dirichlet 先验强度（倍数于历史边际频率）
SHRINK_K = 24.0            # 经验贝叶斯收缩：λ = n / (n + K)
BW_FLOOR = 0.35            # 高斯核带宽下限（标准化距离单位）
MIN_CANDIDATES = 10        # 相似型最少候选样本，不足时逐级降维
MIN_AXES = 4               # 距离计算最少公共特征轴数
MC_DRAWS = 1500            # 蒙特卡洛抽样次数
MC_SEED = 20240101         # 固定随机种子（可复现）
TOP_N_ANALOGS = 5          # 报告展示的最相似年份数
VERIFY_START = 1952        # 最早验证目标年

MONTH_CN = ["", "1月", "2月", "3月", "4月", "5月", "6月",
            "7月", "8月", "9月", "10月", "11月", "12月"]

# ONI 文件里的重叠三月季按"结束月份"归档：DJF 结束于 2 月 …… NDJ 结束于 1 月
SEASON_LABELS = ["", "NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ",
                 "MJJ", "JJA", "JAS", "ASO", "SON", "OND"]


def pad(text, width):
    s = str(text)
    return s + " " * max(0, width - len(s))


# ------------------------------------------------------------ 基础工具 ----

def mon_norm(y, m):
    """把 (年, 月) 归一化到 m ∈ 1..12。"""
    while m < 1:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    return y, m


def prev_months(y, m, k):
    """返回以 (y, m) 为终点、往前共 k 个月的 (年, 月) 列表（时间正序）。"""
    out = []
    yy, mm = mon_norm(y, m)
    for _ in range(k):
        out.append((yy, mm))
        mm -= 1
        if mm == 0:
            mm = 12
            yy -= 1
    out.reverse()
    return out


def lead_months(m):
    """起报月 m 到目标夏季（6 月）的提前期月数。"""
    if 1 <= m <= 5:
        return 6 - m
    return 18 - m


def state_of(v):
    """标准化季风强度值 -> 状态。"""
    if v is None:
        return None
    if v >= THRESHOLD:
        return STATE_STRONG
    if v <= -THRESHOLD:
        return STATE_WEAK
    return STATE_NORMAL


def iqr(xs):
    xs = sorted(xs)
    n = len(xs)

    def q(p):
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

    return q(0.75) - q(0.25)


def weighted_quantile(pairs, q):
    """pairs: [(值, 权重)...]，返回加权 q 分位：按权重累计的阶梯分位；
    当目标恰好落在两个值之间的累计边界上时取中点（偶样本中位数约定）。"""
    ps = sorted((t for t in pairs if t[1] > 0), key=lambda t: t[0])
    if not ps:
        return None
    total = sum(w for _, w in ps)
    if total <= 0:
        return None
    target = q * total
    acc = 0.0
    for i, (v, w) in enumerate(ps):
        acc += w
        if acc >= target:
            if abs(acc - target) <= 1e-12 * max(1.0, total) and i < len(ps) - 1:
                return 0.5 * (v + ps[i + 1][0])
            return v
    return ps[-1][0]


def silverman_bw(distances, floor_bw=0.1, min_n=3):
    """一维距离上的 Silverman 经验带宽。"""
    d = [x for x in distances if x == x]  # 去掉 NaN
    n = len(d)
    if n < min_n:
        return floor_bw
    sig = statistics_stdev(d)
    if sig <= 1e-9:
        return max(floor_bw, 0.3)
    iq = iqr(d)
    h = 0.9 * min(sig, iq / 1.34) * (n ** -0.2)
    return max(floor_bw, h)


def statistics_stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


# ------------------------------------------------------------ 数据抓取 ----

def fetch_text(url, cache_path, refresh=False, allow_network=True, timeout=40):
    """抓取文本；成功则写入缓存（带时间戳注释行）；失败回退本地缓存。"""
    if refresh and allow_network:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (statistical research tool)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            text = raw.decode("ascii", errors="replace")
            header = "# fetched=%sZ source=%s\n" % (
        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), url)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="ascii", errors="replace") as f:
                f.write(header + text)
            return text, False
        except Exception as exc:  # 网络失败 -> 回退缓存
            print("  [warn] 抓取失败(%s)，回退本地缓存 %s" % (exc.__class__.__name__, os.path.basename(cache_path)))
            return _load_cache(cache_path), True
    if allow_network and not os.path.exists(cache_path):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (statistical research tool)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            text = raw.decode("ascii", errors="replace")
            header = "# fetched=%sZ source=%s\n" % (
        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), url)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="ascii", errors="replace") as f:
                f.write(header + text)
            return text, False
        except Exception as exc:
            print("  [warn] 抓取失败(%s)" % exc.__class__.__name__)
            return "", True
    return _load_cache(cache_path), True


def _load_cache(cache_path):
    """读取缓存并去掉 # 注释行。"""
    try:
        with open(cache_path, "r", encoding="ascii", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    keep = [ln for ln in lines if not ln.startswith("#")]
    return "".join(keep)


# ------------------------------------------------------------ 数据解析 ----

def parse_oni(text):
    """解析 NOAA ONI 文本 -> {(年, 结束月): 距平}，-99.90 视为缺失。"""
    out = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 4:
            continue
        season, year_s, _, anom_s = parts[0], parts[1], parts[2], parts[3]
        if not year_s.isdigit() or season not in SEASON_LABELS:
            continue
        try:
            year = int(year_s)
            anom = float(anom_s)
        except ValueError:
            continue
        m = SEASON_LABELS.index(season)
        if m < 1 or anom <= -99.0:
            continue
        out[(year, m)] = anom
    return out


def parse_soi(text):
    """解析 CPC SOI 文本（含两段：ANOMALY 与 STANDARDIZED），取标准化段。
    返回 {(年, 月): 值}，-999.9 视为缺失。"""
    sec_marker = re.compile(r"STANDARDIZED", re.IGNORECASE)
    out = {}
    current_series = 1
    for ln in text.splitlines():
        if sec_marker.search(ln):
            current_series = 2
            continue
        m = re.match(r"^\s*(\d{4})", ln)
        if not m:
            continue
        try:
            year = int(m.group(1))
        except ValueError:
            continue
        if year < 1900 or year > 2100:
            continue
        rest = ln[m.end():]
        vals = re.findall(r"[-+]?\d+\.\d+", rest)
        if len(vals) < 12:
            continue
        row = {}
        for i in range(12):
            v = float(vals[i])
            row[i + 1] = None if v <= -999.0 else v
        if current_series == 2:
            out[year] = row  # 标准化段按年份覆盖
        else:
            out[year] = row
    # 逐月字典
    res = {}
    for year, row in out.items():
        for i in range(1, 13):
            res[(year, i)] = row.get(i)
    return res


def parse_scsmi(text):
    """解析 SCSMI 文本 -> {年: {"raw": [Jun..Sep, JJAS], "norm": [Jun..Sep, JJAS]}}。
    99.99 视为缺失。"""
    out = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*(\d{4})\b", ln)
        if not m:
            continue
        try:
            year = int(m.group(1))
        except ValueError:
            continue
        if year < 1900:
            continue
        vals = re.findall(r"[-+]?\d+\.\d+", ln[m.end():])
        if len(vals) < 10:
            continue
        def f(s):
            v = float(s)
            return None if abs(v - 99.99) < 0.01 else v
        raw = [f(s) for s in vals[0:5]]
        norm = [f(s) for s in vals[5:10]]
        out[year] = {"raw": raw, "norm": norm}
    return out


def parse_dmi(text):
    """解析 PSL DMI 月指数文本 -> {(年, 月): 值}，-9999 视为缺失。
    文件含表头（起止年）与页脚（Timeseries output...），按行年号过滤。"""
    out = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*(\d{4})(?:\s+|$)", ln)
        if not m:
            continue
        try:
            year = int(m.group(1))
        except ValueError:
            continue
        if year < 1870 or year > 2100:
            continue
        vals = re.findall(r"[-+]?\d+\.\d+", ln[m.end():])
        if len(vals) < 12:
            continue
        for i in range(12):
            v = float(vals[i])
            out[(year, i + 1)] = None if v <= -9990.0 else v
    return out


# ------------------------------------------------------------ 数据装载 ----

def load_oni(allow_network=True):
    text, _ = fetch_text(ONI_URL, ONI_CACHE, refresh=False, allow_network=allow_network)
    return parse_oni(text)


def load_soi(allow_network=True):
    text, _ = fetch_text(SOI_URL, SOI_CACHE, refresh=False, allow_network=allow_network)
    return parse_soi(text)


def load_scsmi(allow_network=True):
    text, _ = fetch_text(SCSMI_URL, SCSMI_CACHE, refresh=False, allow_network=allow_network)
    return parse_scsmi(text)


def load_dmi(allow_network=True):
    text, _ = fetch_text(DMI_URL, DMI_CACHE, refresh=False, allow_network=allow_network)
    return parse_dmi(text)


# ------------------------------------------------------------ 爆发日期 ----
# 南海季风爆发日期模块：区域 (5°-15°N, 110°-120°E) 850 hPa 纬向风由负转正的候，
# 即"东亚夏季风第一次北跳/南海季风建立"的日期（王斌等 2004 标准）。

def _days_in_year(year):
    return 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365


def _apr1_index(year):
    """4 月 1 日在 NCEP dailyavgs time 轴上的 0 基索引（1 月 1 日=0）。"""
    return 90 + (1 if _days_in_year(year) == 366 else 0)


def _doy_to_md(year, doy):
    """1 基数 day-of-year -> (月, 日)。"""
    dims = [31, 29 if _days_in_year(year) == 366 else 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31]
    for i, d in enumerate(dims, 1):
        if doy <= d:
            return i, doy
        doy -= d
    return 12, 31


def pentad_of_md(month, day):
    """(月, 日) -> WMO 候号（1..73，1 月 1 日为第 1 候；只对非闰年口径安全）。"""
    dims = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = sum(dims[:month - 1]) + day
    return (doy - 1) // 5 + 1


def pentad_center_md(year, p):
    """候号 -> 中心日期 (月, 日) 字符串，如 (5, 18)。"""
    doy = (p - 1) * 5 + 3
    return _doy_to_md(year, doy)


def parse_u850_ascii(text, n_days=ONSET_WIN_DAYS):
    """解析 OPeNDAP .ascii 响应 -> n_days 个区域平均日值列表（失败返回 None）。
    数据段位于 'uwnd.uwnd[...]' 行之后，每行 5 个格点（经度方向），
    连续 5 行（纬度方向）构成一天。"""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("uwnd.uwnd"):
        i += 1
    if i >= len(lines):
        return None
    cells = []
    for ln in lines[i + 1:]:
        if not ln.startswith("["):
            continue
        parts = ln.split(",")
        if len(parts) < 6:
            continue
        try:
            row = [float(x) for x in parts[1:6]]
        except ValueError:
            continue
        cells.append(sum(row) / len(row))  # 一行 = 一个 (天, 纬度) 的 5 个经度值平均
    if len(cells) < n_days * 5:
        return None
    out = []
    for t in range(n_days):
        block = cells[t * 5:(t + 1) * 5]
        out.append(sum(block) / len(block) if len(block) == 5 else None)
    return out


def fetch_u850_year(year, timeout=90):
    """抓取一年的 4/1-7/31 逐日 U850 区域平均序列；失败返回 None。"""
    i0 = _apr1_index(year)
    url = ("%s/uwnd.%d.nc.ascii?uwnd%%5B%d:%d%%5D%%5B%d%%5D%%5B%d:%d%%5D%%5B%d:%d%%5D"
           % (U850_BASE, year, i0, i0 + ONSET_WIN_DAYS - 1, U850_LEVEL_IDX,
              U850_LAT_IDX[0], U850_LAT_IDX[1], U850_LON_IDX[0], U850_LON_IDX[1]))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (statistical research tool)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return parse_u850_ascii(raw.decode("ascii", errors="replace"))
    except Exception as exc:
        print("  [warn] U850 抓取失败 %d (%s)" % (year, exc.__class__.__name__))
        return None


def _load_u850_cache():
    """读取 U850 缓存 -> {年: [123 个 4/1-7/31 区域平均日值]}，999.99 视为缺测。"""
    out = {}
    try:
        with open(U850_CACHE, "r", encoding="ascii", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return out
    for ln in lines:
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        vals = []
        for x in parts[1:]:
            try:
                v = float(x)
            except ValueError:
                continue
            vals.append(None if abs(v - 999.99) < 0.01 else v)
        if len(vals) == ONSET_WIN_DAYS:
            out[int(parts[0])] = vals
    return out


def _save_u850_cache(data):
    header = ("# fetched=%sZ source=%s/datasets 850hPa daily u-wind "
              "area-mean(10-20N,110-120E) Apr1-Jul31, R1 stopped 2026-03\n"
              % (datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), U850_BASE))
    with open(U850_CACHE, "w", encoding="ascii", errors="replace") as f:
        f.write(header)
        for y in sorted(data):
            row = data[y]
            f.write("%d" % y + "".join(
                " %.2f" % (v if v is not None else 999.99) for v in row) + "\n")


def load_u850(allow_network=True):
    """装载 U850 缓存；允许联网时并发补齐缺失年份（1948..U850_MAX_YEAR）。"""
    data = _load_u850_cache()
    if allow_network:
        missing = [y for y in range(1948, U850_MAX_YEAR + 1) if y not in data]
        if missing:
            print("U850 爆发窗口缓存缺少 %d 个年份，联网补齐…" % len(missing))
            with ThreadPoolExecutor(max_workers=4) as ex:
                results = list(ex.map(fetch_u850_year, missing))
            for y, vals in zip(missing, results):
                if vals is not None:
                    data[y] = vals
            if data:
                _save_u850_cache(data)
    return data


def daily_to_pentads(daily):
    """123 天（4 月 1 日起）-> {候号: 候均值}，覆盖第 19..43 候；
    最末候（第 43 候，7/30-8/3）窗口内只有 2 天，用可用天平均。"""
    out = {}
    for p in range(19, 44):
        block = daily[(p - 19) * 5:(p - 19) * 5 + 5]
        real = [v for v in block if v is not None]
        out[p] = (sum(real) / len(real)) if real else None
    return out


def pentad_label(pv):
    """候号（可小数）-> '第N候(约MM月DD日)'。"""
    if pv is None:
        return "NA"
    base = int(math.floor(pv))
    frac = pv - base
    day_in = min(max(int(round(frac * 5)) + 3, 1), 5)  # 候内第 3 天为候中心
    doy = (base - 1) * 5 + day_in
    mo, dy = _doy_to_md(2001, doy)  # 用非闰年换算日期即可
    return "第%d候(%02d月%02d日)" % (base, mo, dy)


def detect_onset(pents, year, start=ONSET_START_PENTAD, persist=ONSET_PERSIST):
    """王斌等 (2004) 判定：自第 start 候（第 24 候 = 4/26-30）起，找第一个候 p，
    使 p..p+persist-1 候平均 U850 连续为正。返回 (候号, 近似日期) 或 (None, None)。"""
    for p in sorted(pents):
        if p < start:
            continue
        ok = True
        for k in range(persist):
            v = pents.get(p + k)
            if v is None or v <= 0.0:
                ok = False
                break
        if ok:
            mo, dy = pentad_center_md(year, p)
            return p, "%d月%d日" % (mo, dy)
    return None, None


def onset_series(u850):
    """{年: 日值序列} -> {年: 爆发候}（数据缺失或未检出爆发的年份不记录）。"""
    out = {}
    for y in sorted(u850):
        daily = u850.get(y)
        if not daily or len(daily) < ONSET_WIN_DAYS:
            continue
        p, _ = detect_onset(daily_to_pentads(daily), y)
        if p is not None:
            out[y] = p
    return out


def _pearson(a, b):
    """皮尔逊相关系数；样本 < 3 或零方差时返回 None。"""
    n = len(a)
    if n < 3:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    sxx = sum((x - ma) ** 2 for x in a)
    syy = sum((y - mb) ** 2 for y in b)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return sxy / math.sqrt(sxx * syy)


def _ols(X, y):
    """普通最小二乘（正规方程 + 高斯消元）-> [截距, 斜率...]；退化时返回 None。"""
    n = len(y)
    k = len(X[0]) if n else 0
    if n < k + 2:
        return None
    cols = k + 1
    A = [[0.0] * cols for _ in range(cols)]
    b = [0.0] * cols
    for i in range(n):
        row = [1.0] + list(X[i])
        for r in range(cols):
            b[r] += row[r] * y[i]
            for c in range(cols):
                A[r][c] += row[r] * row[c]
    for col in range(cols):  # 列主元高斯消元
        piv = max(range(col, cols), key=lambda r2: abs(A[r2][col]))
        if abs(A[piv][col]) < 1e-12:
            return None
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, cols):
            f = A[r][col] / A[col][col]
            for c in range(col, cols):
                A[r][c] -= f * A[col][c]
            b[r] -= f * b[col]
    coef = [0.0] * cols
    for r in range(cols - 1, -1, -1):
        s = b[r] - sum(A[r][c] * coef[c] for c in range(r + 1, cols))
        coef[r] = s / A[r][r]
    return coef


def _clim_before(seq, y_t):
    """目标年前所有爆发候的均值（气候态基准）；样本不足返回 None。"""
    vs = [seq[y] for y in sorted(seq) if y < y_t]
    return (sum(vs) / len(vs)) if len(vs) >= 3 else None


def onset_stats(seq, oni=None, soi=None):
    """爆发候序列统计：均值/中位/最早/最晚/标准差/线性趋势（候每 10 年）/
    滞后 1 年自相关，以及（可选）与 DJF ONI、4 月 SOI 的同期相关。"""
    years = sorted(seq)
    vals = [seq[y] for y in years]
    n = len(vals)
    if n < 3:
        return None
    mean = sum(vals) / n
    med = sorted(vals)[n // 2]
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    e_y, l_y = min(years, key=lambda y: seq[y]), max(years, key=lambda y: seq[y])
    coef = _ols([[float(y)] for y in years], vals)
    trend = (coef[1] * 10.0) if coef else None
    lag1 = None
    if n >= 4:
        lag1 = _pearson(vals[:-1], vals[1:])
    out = {
        "n": n, "mean": mean, "median": med, "sd": sd,
        "earliest": (e_y, seq[e_y]), "latest": (l_y, seq[l_y]),
        "trend_per_decade": trend, "lag1_r": lag1,
        "first_year": years[0], "last_year": years[-1],
    }
    if oni is not None:
        xs, ys = [], []
        for y in years:
            o = oni.get((y, 2))  # DJF（结束月 2）
            if o is not None:
                xs.append(o)
                ys.append(seq[y])
        out["corr_oni_djf"] = _pearson(xs, ys) if len(xs) >= 3 else None
    if soi is not None:
        xs, ys = [], []
        for y in years:
            s = soi.get((y, 4))  # 4 月 SOI（爆发起报前最新）
            if s is not None:
                xs.append(s)
                ys.append(seq[y])
        out["corr_soi_apr"] = _pearson(xs, ys) if len(xs) >= 3 else None
    return out


def onset_analog_forecast(seq, years, weights):
    """相似型近邻对爆发候的加权点预测 + 90% 分位区间；近邻足够返回 dict。"""
    pairs = [(seq[y], w) for y, w in zip(years, weights) if y in seq]
    pairs = [(v, w) for v, w in pairs if v is not None and w > 0]
    if len(pairs) < 3:
        return None
    tot = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / tot
    q05 = weighted_quantile(pairs, 0.05)
    q95 = weighted_quantile(pairs, 0.95)
    return {"mean": mean, "q05": q05, "q95": q95, "n": len(pairs)}


def onset_reg_predict(oni, soi, seq, y_t):
    """爆发候线性回归：特征 [DJF ONI, 4 月 SOI, 前一年爆发候]，训练集严格早于 y_t。
    返回 {"mean":..., "n":..., "coef":[...]} 或 None（特征缺失/样本不足）。"""
    X, Y = [], []
    for y in sorted(seq):
        if y >= y_t or (y - 1) not in seq:
            continue
        o = oni.get((y, 2))
        s = soi.get((y, 4))
        if o is None or s is None:
            continue
        X.append([o, s, seq[y - 1]])
        Y.append(seq[y])
    if len(Y) < 5:
        return None
    coef = _ols(X, Y)
    if coef is None:
        return None
    o = oni.get((y_t, 2))
    s = soi.get((y_t, 4))
    lag = seq.get(y_t - 1)
    if o is None or s is None or lag is None:
        return None
    pred = coef[0] + coef[1] * o + coef[2] * s + coef[3] * lag
    return {"mean": pred, "n": len(Y), "coef": coef}


def _onset_metrics(triples):
    """[(预测, 观测, 当年气候基准) ...] -> 验证指标 dict；样本不足返回 None。"""
    good = [(p, o, c) for p, o, c in triples if p is not None and c is not None]
    if len(good) < 5:
        return None
    n = len(good)
    r = _pearson([t[0] for t in good], [t[1] for t in good])
    mae = sum(abs(p - o) for p, o, _ in good) / n
    mae_clim = sum(abs(o - c) for _, o, c in good) / n
    skill = 1.0 - mae / mae_clim if mae_clim > 0 else None
    hits = sum(1 for p, o, c in good if (p > c) == (o > c))
    return {"n": n, "r": r, "mae": mae, "mae_clim": mae_clim,
            "skill": skill, "hit_rate": hits / n}


def onset_cross_validate(oni, soi, scsmi, seq, verify_start=ONSET_VERIFY_START,
                         dmi=None):
    """爆发候样本外（时序）验证：逐年以当年之前的数据分别做
    气候态 / 相似型（起报 4 月）/ 线性回归预测，统计 r、MAE、相对气候态技能、
    距平符号命中率。相似型复用主预测的 4 月起报口径，候选年严格早于目标年。"""
    years = [y for y in sorted(seq) if y >= verify_start]
    tri = {"analog": [], "reg": []}
    for y_t in years:
        obs = seq[y_t]
        clim = _clim_before(seq, y_t)
        # 相似型：起报 4 月（业务口径），目标同年
        ana = None
        if len(seq) >= 3:
            a_pool = analog_distribution(oni, soi, scsmi, y_t, 4, dmi=dmi)
            if a_pool is not None:
                a_m_res, _, _ = monthly_analog(oni, soi, scsmi, y_t, 4, dmi=dmi)
                am = a_m_res or a_pool
                ana = onset_analog_forecast(seq, am["years"], am["weights"])
        tri["analog"].append((ana["mean"] if ana else None, obs, clim))
        reg = onset_reg_predict(oni, soi, seq, y_t)
        tri["reg"].append((reg["mean"] if reg else None, obs, clim))
    out = {"verify_start": verify_start, "n_years": len(years)}
    for key, t in tri.items():
        out[key] = _onset_metrics(t)
    return out


def build_onset_result(oni, soi, scsmi, u850, y_t, comp, dmi=None):
    """爆发日期全链结果（供 forecast/report 使用）：序列、统计、本轮两类预测、
    样本外验证。u850 缺失或无爆发年份时返回 None。"""
    seq = onset_series(u850)
    if len(seq) < 25:
        return None
    stats = onset_stats(seq, oni, soi)
    if stats is None:
        return None
    am = comp.get("analog_monthly") or comp.get("analog_pooled")
    ana = None
    near = None
    if am:
        ok = [(y, w) for y, w in zip(am["years"], am["weights"]) if y < y_t]  # 无泄漏防线
        if len(ok) >= 3:
            ana = onset_analog_forecast(seq, [y for y, _ in ok], [w for _, w in ok])
        near = sorted((seq[y], y) for y, _ in ok if y in seq)[:5]
    reg = onset_reg_predict(oni, soi, seq, y_t)
    clim = _clim_before(seq, y_t)
    cv = onset_cross_validate(oni, soi, scsmi, seq, dmi=dmi)
    return {
        "target_year": y_t, "series": seq, "stats": stats,
        "analog": ana, "reg": reg, "clim": clim, "cv": cv,
        "near": near,
        "_terms": ("南海季风爆发候=区域%s850hPa纬向风转正并维持3候（王斌等2004）"
                   % ONSET_REGION_LABEL),
    }


def update_data():
    """update 子命令：刷新五个数据源并写缓存（U850 只补齐缺失年份）。"""
    print("刷新数据源：")
    for name, url, cache in [("ONI", ONI_URL, ONI_CACHE),
                             ("SOI", SOI_URL, SOI_CACHE),
                             ("SCSMI", SCSMI_URL, SCSMI_CACHE),
                             ("DMI", DMI_URL, DMI_CACHE)]:
        text, from_cache = fetch_text(url, cache, refresh=True, allow_network=True)
        print("  %-6s %s %s" % (name, "缓存回退" if from_cache else "OK", os.path.basename(cache)))
    data = _load_u850_cache()
    missing = [y for y in range(1948, U850_MAX_YEAR + 1) if y not in data]
    if missing:
        print("  U850   补齐 %d 个年份（第一次运行需抓取 %d-%d 全部爆发窗口，几分钟）..."
              % (len(missing), min(missing), max(missing)))
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(fetch_u850_year, missing))
        ok = 0
        for y, vals in zip(missing, results):
            if vals is not None:
                data[y] = vals
                ok += 1
        if data:
            _save_u850_cache(data)
        print("  U850   %s 成功 %d/%d" % ("缓存回退" if ok == 0 else "OK", ok, len(missing)))
    else:
        print("  U850   OK（缓存完整）%s" % os.path.basename(U850_CACHE))


def target_years(scsmi):
    """JJAS 标准化值完整的年份升序列表。"""
    return sorted(y for y, d in scsmi.items() if d["norm"][4] is not None)


def jjas_norm(scsmi, y):
    d = scsmi.get(y)
    if d is None:
        return None
    return d["norm"][4]


# ------------------------------------------------------- ENSO 状态判定 ----

def oni_state_at(oni, y, m):
    """起报月 m（数据截至 m-1）时的 ENSO 状态：连续 5 个重叠季均超阈值。
    返回 "E" / "N" / "L"，数据不足返回 None。"""
    months = prev_months(y, m - 1, 5)
    vals = [oni.get((yy, mm)) for yy, mm in months]
    if any(v is None for v in vals):
        return None
    if all(v >= 0.5 for v in vals):
        return "E"
    if all(v <= -0.5 for v in vals):
        return "L"
    return "N"


def oni_streak(oni, y, m, sign):
    """从 (y, m-1) 往回数 ONI 连续 >= +0.5（sign>0）或 <= -0.5（sign<0）的月数。"""
    yy, mm = mon_norm(y, m - 1)
    k = 0
    while True:
        v = oni.get((yy, mm))
        if v is None or (sign > 0 and v < 0.5) or (sign < 0 and v > -0.5):
            break
        k += 1
        mm -= 1
        if mm == 0:
            mm = 12
            yy -= 1
    return k


def describe_oni_state(oni, y, m):
    """终端展示用的 ENSO 状态描述。"""
    st = oni_state_at(oni, y, m)
    if st == "E":
        return "厄尔尼诺（近 5 个重叠季 ONI ≥ +0.5）"
    if st == "L":
        return "拉尼娜（近 5 个重叠季 ONI ≤ -0.5）"
    if st == "N":
        # 临近判定
        last = None
        yy, mm = mon_norm(y, m - 1)
        while last is None and yy >= 1900:
            last = oni.get((yy, mm))
            mm -= 1
            if mm == 0:
                mm = 12
                yy -= 1
        if last is not None and last >= 0.5:
            k = oni_streak(oni, y, m, +1)
            return "中性（临近判定：近 %d 个月 ONI ≥ +0.5，若持续将转入厄尔尼诺）" % k
        if last is not None and last <= -0.5:
            k = oni_streak(oni, y, m, -1)
            return "中性（临近判定：近 %d 个月 ONI ≤ -0.5，若持续将转入拉尼娜）" % k
        return "中性"
    return "未知（数据不足）"


def latest_oni(oni):
    best = None
    for (y, m), v in oni.items():
        if v is None:
            continue
        if best is None or (y, m) > best[0]:
            best = ((y, m), v)
    return best


def latest_soi(soi):
    best = None
    for (y, m), v in soi.items():
        if v is None:
            continue
        if best is None or (y, m) > best[0]:
            best = ((y, m), v)
    return best


def latest_dmi(dmi):
    """最新一个非缺失 DMI 值 -> ((年, 月), 值)；无数据返回 None。"""
    best = None
    for (y, m), v in dmi.items():
        if v is None:
            continue
        if best is None or (y, m) > best[0]:
            best = ((y, m), v)
    return best


# ------------------------------------------------------------ 特征构造 ----

ENSO_PHASES = [("D", "发展(增强)"), ("P", "峰值(平台)"), ("C", "衰减(减弱)"), ("N", "中性")]
ENSO_PHASE_TREND = 0.2   # 3 个月 ONI 变化阈值（发展/衰减判定）


def enso_phase_at(oni, y, m, win=6):
    """起报 (y, m) 时点的 ENSO 发展阶段，只用 m 及之前的数据（无泄漏）。
    以近 3 个月 ONI 趋势在 ±0.2 内判峰值平台：
      'D' 发展（|ONI|≥0.5 且 3 个月变化 ≥ +0.2）
      'C' 衰减（|ONI|≥0.5 且 3 个月变化 ≤ -0.2）
      'P' 峰值（|ONI|≥0.5 且变化平缓）
      'N' 中性（近 win 个月 |ONI| 全 < 0.5）
    数据不足返回 None。"""
    vals = [oni.get((yy, mm)) for yy, mm in prev_months(y, m, win)]  # 时间正序，末位=当月
    vals = [v for v in vals if v is not None]
    if len(vals) < 4:
        return None
    if all(abs(v) < 0.5 for v in vals):
        return "N"
    cur = vals[-1]
    if abs(cur) < 0.5:
        return "N"
    delta = cur - vals[-4]
    if delta >= ENSO_PHASE_TREND:
        return "D"
    if delta <= -ENSO_PHASE_TREND:
        return "C"
    return "P"


def build_feature_axes(y, m):
    """起报 (年 y, 月 m) 的特征轴 id 列表：
    6 个 ONI 月 + 6 个 SOI 月 + 6 个 DMI 月（截至 m-1）+ 前一年 SCSMI。"""
    wins = prev_months(y, m - 1, WINDOW)
    axes = [("oni", yy, mm) for yy, mm in wins]
    axes += [("soi", yy, mm) for yy, mm in wins]
    axes += [("dmi", yy, mm) for yy, mm in wins]
    axes += [("prev", y - 1)]
    return axes


def axis_value(ax, oni, soi, scsmi, for_year, dmi=None):
    kind = ax[0]
    if kind == "oni":
        return oni.get((ax[1], ax[2]))
    if kind == "soi":
        return soi.get((ax[1], ax[2]))
    if kind == "dmi":
        return (dmi or {}).get((ax[1], ax[2]))
    return jjas_norm(scsmi, for_year - 1)


def feature_vector(axes, oni, soi, scsmi, y, dmi=None):
    return {ax: axis_value(ax, oni, soi, scsmi, y, dmi) for ax in axes}


# ------------------------------------------------------------ 模型样本池 ----

POS_AXES = ([("oni", k) for k in range(WINDOW)] +
            [("soi", k) for k in range(WINDOW)] +
            [("dmi", k) for k in range(WINDOW)] + [("prev",)])

_POOL_CACHE = {}


def _build_pool(oni, soi, scsmi, dmi=None):
    """按数据身份构建一次性的模型样本池：
    t_rows: (目标年 T, 前一年状态, 当年状态)，以及各起报月 m2 的 ENSO 态
    a_rows: (目标年 T, 窗口结束月 w, 窗口 6 个月列表) —— w=1..5 同年、w=6..12 前一年"""
    t_rows, a_rows = [], []
    st_map = {}
    for T in target_years(scsmi):
        s_cur = state_of(jjas_norm(scsmi, T))
        if s_cur is None:
            continue
        s_prev = state_of(jjas_norm(scsmi, T - 1))
        t_rows.append((T, s_prev, s_cur))
        st_map[T] = [None] + [oni_state_at(oni, T - (1 if m2 >= 6 else 0), m2)
                              for m2 in range(1, 13)]
        for w in range(1, 13):
            Yw = T - 1 if w >= 6 else T
            a_rows.append((T, w, prev_months(Yw, w, WINDOW)))
    return {"t_rows": t_rows, "a_rows": a_rows, "st_map": st_map,
            "oni": oni, "soi": soi, "scsmi": scsmi, "dmi": dmi or {}}


def get_pool(oni, soi, scsmi, dmi=None):
    key = (id(oni), id(soi), id(scsmi), id(dmi))
    p = _POOL_CACHE.get(key)
    if p is None:
        p = _build_pool(oni, soi, scsmi, dmi)
        if len(_POOL_CACHE) > 12:
            _POOL_CACHE.pop(next(iter(_POOL_CACHE)))
        _POOL_CACHE[key] = p
    return p


def _candidate_years(scsmi, y_t):
    return [y2 for y2 in target_years(scsmi) if y2 < y_t]


# ------------------------------------------------------------ 转移模型 ----

def transition_distribution(oni, soi, scsmi, y_t, m, prior=PRIOR_STRENGTH, phase=None):
    """状态转移：P(今年季风态 | 去年季风态, 起报时 ENSO 态)。
    phase=None 为全历史（全部 12 个起报月混合计数），phase=m 为同月起报分层。"""
    pool = get_pool(oni, soi, scsmi)
    scsmi, oni = pool["scsmi"], pool["oni"]
    counts = defaultdict(int)  # (prev_s, oni_s, s)
    marg = defaultdict(int)
    n_used = 0
    for T, s_prev, s_cur in pool["t_rows"]:
        if T >= y_t:
            continue  # 只允许评估时点之前完整闭合的目标年，杜绝未来泄漏
        if s_prev is None:
            continue
        ms = [phase] if phase is not None else list(range(1, 13))
        for m2 in ms:
            o = pool["st_map"].get(T, [None] * 13)[m2]
            if o is None:
                continue
            counts[(s_prev, o, s_cur)] += 1
            marg[s_cur] += 1
            n_used += 1
    total = sum(marg.values())
    if total == 0:
        return {"p": None, "n": 0, "marginal": {s: 1.0 / 3.0 for s in STATES}}
    marginal = {s: marg[s] / total for s in STATES}
    prev_now = state_of(jjas_norm(scsmi, y_t - 1))
    oni_now = oni_state_at(oni, y_t - (1 if m >= 6 else 0), m)
    if oni_now is None:
        conds = [(prev_now, o) for o in ("E", "N", "L")]
    else:
        conds = [(prev_now, oni_now)]
    if prev_now is None:
        conds = [(p, o) for (p, o) in conds for p in STATES]
    agg = defaultdict(int)
    n_cond = 0
    for (p, o) in conds:
        for s in STATES:
            agg[s] += counts.get((p, o, s), 0)
            n_cond += counts.get((p, o, s), 0)
    if n_cond == 0:
        return {"p": dict(marginal), "n": n_used, "fallback": True,
                "marginal": dict(marginal)}
    p = {s: (agg[s] + prior * marginal[s]) / (n_cond + prior) for s in STATES}
    return {"p": p, "n": n_used, "fallback": False, "marginal": dict(marginal)}


def monthly_transition(oni, soi, scsmi, y_t, m, prior=PRIOR_STRENGTH):
    """转移模型的同月起报版本。"""
    r = transition_distribution(oni, soi, scsmi, y_t, m, prior, phase=m)
    if r is None or r.get("p") is None:
        return None, 0
    return r["p"], r["n"]


# ------------------------------------------------------------ 相似型模型 ----

def _vec_from_months(oni, soi, scsmi, T, months, dmi=None):
    v = {}
    dd = dmi or {}
    for k, (yy, mm) in enumerate(months):
        v[("oni", k)] = oni.get((yy, mm))
        v[("soi", k)] = soi.get((yy, mm))
        v[("dmi", k)] = dd.get((yy, mm))
    v[("prev",)] = jjas_norm(scsmi, T - 1)
    return v


def _analog_core_pool(pool, y_t, m, phase, axes, min_candidates, bw_floor):
    """给定特征轴集合，从样本池计算高斯核加权状态概率与近邻明细。"""
    oni, soi, scsmi = pool["oni"], pool["soi"], pool["scsmi"]
    dmi = pool.get("dmi") or {}
    Yw0, w0 = mon_norm(y_t - (1 if m >= 6 else 0), m - 1)
    q_months = prev_months(Yw0, w0, WINDOW)
    qv = _vec_from_months(oni, soi, scsmi, y_t, q_months, dmi)
    rows = []
    for T, w, months in pool["a_rows"]:
        if T >= y_t:
            continue
        if phase is not None and w != w0:
            continue
        rows.append((T, w, months))
    if len(rows) < min_candidates:
        return None
    vecs = {i: _vec_from_months(oni, soi, scsmi, T, months, dmi)
            for i, (T, w, months) in enumerate(rows)}
    mu, sd = {}, {}
    for ax in axes:
        vals = [vecs[i].get(ax) for i in range(len(rows))
                if vecs[i].get(ax) is not None]
        if len(vals) < 2:
            mu[ax], sd[ax] = None, None
            continue
        m_ = sum(vals) / len(vals)
        mu[ax], sd[ax] = m_, max(statistics_stdev(vals), 1e-6)
    out = []
    for i, (T, w, months) in enumerate(rows):
        v2 = vecs[i]
        common = [ax for ax in axes
                  if qv.get(ax) is not None and v2.get(ax) is not None
                  and mu.get(ax) is not None]
        if len(common) < MIN_AXES:
            continue
        d2 = sum(((qv[ax] - v2[ax]) / sd[ax]) ** 2 for ax in common)
        out.append((T, w, math.sqrt(d2 / len(common))))
    if not out:
        return None
    h = silverman_bw([d for _, _, d in out], floor_bw=bw_floor)
    wmap = {i: math.exp(-d * d / (2.0 * h * h)) for i, (_, _, d) in enumerate(out)}
    total_w = sum(wmap.values())
    if total_w <= 1e-12:
        return None
    probs = {s: 0.0 for s in STATES}
    for i, (T, w, d) in enumerate(out):
        s = state_of(jjas_norm(scsmi, T))
        probs[s] += wmap[i] / total_w
    n_eff = total_w * total_w / sum(w * w for w in wmap.values())
    order = sorted(range(len(out)), key=lambda i: out[i][2])
    years_sorted = [out[i][0] for i in order]
    dists_sorted = [out[i][2] for i in order]
    w_sorted = [wmap[i] for i in order]
    return {
        "p": probs,
        "years": years_sorted,
        "dists": dists_sorted,
        "weights": w_sorted,
        "n_eff": n_eff,
        "h": h,
        "n_axes": len(axes),
        "n_cand": len(out),
        "degraded": False,
        "phase_w": w0,
    }


def analog_distribution(oni, soi, scsmi, y_t, m,
                        min_candidates=MIN_CANDIDATES, bw_floor=BW_FLOOR, phase=None,
                        dmi=None):
    """历史相似型（phase=None 为全窗口混合，phase=w0 为同相）。
    候选样本不足时逐级降维（去 DMI -> 去 SOI -> 去前一年 -> 仅 ONI）。"""
    pool = get_pool(oni, soi, scsmi, dmi)
    deg_chain = [
        ("全特征", POS_AXES),
        ("无DMI", [ax for ax in POS_AXES if ax[0] != "dmi"]),
        ("无SOI", [ax for ax in POS_AXES if ax[0] not in ("soi", "dmi")]),
        ("无前一年", [ax for ax in POS_AXES if ax[0] not in ("prev", "dmi")]),
        ("仅ONI", [ax for ax in POS_AXES if ax[0] == "oni"]),
    ]
    last_res = None
    for name, axes in deg_chain:
        res = _analog_core_pool(pool, y_t, m, phase, axes, min_candidates, bw_floor)
        if res is not None:
            if res["n_cand"] >= max(min_candidates, 4):
                res["deg_label"] = name
                res["degraded"] = (name != "全特征")
                return res
            last_res = res
    if last_res is not None and last_res["n_cand"] >= 3:
        last_res["deg_label"] = "降级"
        last_res["degraded"] = True
        return last_res
    return None


# ------------------------------------------------------ 分层收缩与组合 ----

def monthly_analog(oni, soi, scsmi, y_t, m, min_candidates=MIN_CANDIDATES, dmi=None):
    """相似型的同相（同窗口结束月）版本。"""
    pool = get_pool(oni, soi, scsmi, dmi)
    _, w0 = mon_norm(y_t - (1 if m >= 6 else 0), m - 1)
    res = _analog_core_pool(pool, y_t, m, phase=w0, axes=POS_AXES,
                            min_candidates=min_candidates, bw_floor=BW_FLOOR)
    n_all = sum(1 for T, w, _m in pool["a_rows"] if T < y_t and w == w0)
    if res is None:
        return None, 0, n_all
    return res, res["n_cand"], n_all


def shrunk_components(oni, soi, scsmi, y_t, m, prior=PRIOR_STRENGTH, k_shrink=SHRINK_K,
                      dmi=None):
    """经验贝叶斯收缩后的两个模型组件与混合参数。"""
    t_pool = transition_distribution(oni, soi, scsmi, y_t, m, prior, phase=None)
    t_m, n_tm = monthly_transition(oni, soi, scsmi, y_t, m, prior)
    a_pool = analog_distribution(oni, soi, scsmi, y_t, m, phase=None, dmi=dmi)
    a_m_res, n_am, _ = monthly_analog(oni, soi, scsmi, y_t, m, dmi=dmi)

    n_tpool = t_pool.get("n", 0)
    lam_t = n_tm / (n_tm + k_shrink) if n_tm > 0 else 0.0
    p_t = {}
    if t_m is not None and lam_t > 0:
        for s in STATES:
            p_t[s] = lam_t * t_m[s] + (1 - lam_t) * t_pool["p"][s]
    else:
        p_t = dict(t_pool["p"])

    p_a = None
    n_eff_a = 0.0
    lam_a = 0.0
    a_degraded = None
    if a_pool is not None:
        if a_m_res is not None:
            lam_a = n_am / (n_am + k_shrink)
            p_a = {s: lam_a * a_m_res["p"][s] + (1 - lam_a) * a_pool["p"][s]
                   for s in STATES}
            n_eff_a = lam_a * a_m_res["n_eff"] + (1 - lam_a) * a_pool["n_eff"]
        else:
            p_a = dict(a_pool["p"])
            n_eff_a = a_pool["n_eff"]
            lam_a = 0.0
        a_degraded = a_pool.get("degraded", False) or (a_m_res is None)

    # 蒙特卡洛 Dirichlet 后验参数
    marginal = t_pool.get("marginal", {s: 1.0 / 3.0 for s in STATES})
    alpha_t = {s: prior * marginal.get(s, 0.0) + max(n_tpool, 0.5) * p_t[s] for s in STATES}
    alpha_a = None
    if p_a is not None:
        alpha_a = {s: prior * marginal.get(s, 0.0) + max(n_eff_a, 0.5) * p_a[s] for s in STATES}
    return {
        "p_trans": p_t, "alpha_trans": alpha_t, "lam_trans": lam_t,
        "n_trans_pooled": n_tpool, "n_trans_monthly": n_tm,
        "p_analog": p_a, "alpha_analog": alpha_a, "lam_analog": lam_a,
        "n_eff_analog": n_eff_a, "analog_degraded": a_degraded,
        "analog_pooled": a_pool, "analog_monthly": a_m_res,
    }


def skill_weights(skill_t, skill_a):
    """按正技能分比例分配权重；负技能剔除；双负退化等权。"""
    flag = None
    t = max(0.0, skill_t)
    a = max(0.0, skill_a)
    if skill_t <= 0.0 and skill_a <= 0.0:
        return 0.5, "两模型技能均不优于气候态，组合已退化为等权"
    denom = t + a
    if denom <= 0:
        return 0.5, "组合权重退化为等权"
    return t / denom, flag


def blend_forecast(p_t, p_a, w_t, alpha_t, alpha_a, seed=MC_SEED, draws=MC_DRAWS):
    """Dirichlet MC 混合 -> 各状态均值与 90% 区间。"""
    rng = random.Random(seed)

    def dirichlet_draw(alpha, rng):
        xs = [rng.gammavariate(max(alpha[s], 1e-9), 1.0) for s in STATES]
        tot = sum(xs)
        return [x / tot for x in xs]

    p_a_ = p_a if p_a is not None else dict(p_t)
    alpha_a_ = alpha_a if alpha_a is not None else dict(alpha_t)
    w = w_t
    acc = {s: [] for s in STATES}
    for _ in range(draws):
        x1 = dirichlet_draw(alpha_t, rng)
        x2 = dirichlet_draw(alpha_a_, rng)
        for i, s in enumerate(STATES):
            acc[s].append(w * x1[i] + (1 - w) * x2[i])
    mean = {s: sum(acc[s]) / draws for s in STATES}
    lo, hi = {}, {}
    for s in STATES:
        v = sorted(acc[s])
        lo[s] = v[int(0.05 * (draws - 1))]
        hi[s] = v[int(0.95 * (draws - 1))]
    return mean, lo, hi


# ------------------------------------------------------------ 回测 ----

def compute_skills(oni, soi, scsmi, verify_years=None, verbose=False, dmi=None):
    """对每个起报月 m 做样本外回测：
    返回 dict: {m: {n, bs_model_t, bs_model_a, bs_blend, bs_clim,
                    skill_t, skill_a, skill_blend, w_t, flag}}"""
    ys = verify_years if verify_years is not None else target_years(scsmi)
    ys = [y for y in ys if y >= VERIFY_START and jjas_norm(scsmi, y - 1) is not None]
    skills = {}
    for m in range(1, 13):
        entries = []
        for y in ys:
            t_pool = transition_distribution(oni, soi, scsmi, y, m)
            if t_pool is None or t_pool.get("p") is None:
                continue
            a_pool = analog_distribution(oni, soi, scsmi, y, m, dmi=dmi)
            if a_pool is None or a_pool.get("degraded"):
                continue  # 两模型都必须给出非降级估计才计入技能比较
            comp = shrunk_components(oni, soi, scsmi, y, m, dmi=dmi)
            s_obs = state_of(jjas_norm(scsmi, y))
            entries.append((y, comp["p_trans"], comp["p_analog"], s_obs))
        if not entries:
            skills[m] = {"n": 0}
            continue
        bs_t = bs_a = bs_blend = 0.0
        freqs = defaultdict(int)
        for y, p_t, p_a, s_obs in entries:
            freqs[s_obs] += 1
        n = len(entries)
        clim = {s: freqs[s] / n for s in STATES}

        def bs_entry(p, obs):
            return sum((p.get(s, 0.0) - (1.0 if s == obs else 0.0)) ** 2
                       for s in STATES) / 3.0

        for y, p_t, p_a, s_obs in entries:
            bs_t += bs_entry(p_t, s_obs)
            bs_a += bs_entry(p_a, s_obs)
        bs_t /= n
        bs_a /= n
        bs_clim = sum(bs_entry(clim, s_obs) for y, _, _, s_obs in entries) / n
        skill_t = 1.0 - bs_t / bs_clim if bs_clim > 0 else 0.0
        skill_a = 1.0 - bs_a / bs_clim if bs_clim > 0 else 0.0
        w_t, flag = skill_weights(skill_t, skill_a)
        for y, p_t, p_a, s_obs in entries:
            p_blend = {s: w_t * p_t[s] + (1 - w_t) * p_a[s] for s in STATES}
            bs_blend += bs_entry(p_blend, s_obs)
        bs_blend /= n
        skill_blend = 1.0 - bs_blend / bs_clim if bs_clim > 0 else 0.0
        skills[m] = {
            "n": n, "bs_t": bs_t, "bs_a": bs_a, "bs_blend": bs_blend,
            "bs_clim": bs_clim, "skill_t": skill_t, "skill_a": skill_a,
            "skill_blend": skill_blend, "w_t": w_t, "flag": flag,
        }
    return skills


def compute_calibration(oni, soi, scsmi, verify_years=None, dmi=None):
    """可靠性校准：把历史（m × y）组合概率分 10 桶，每状态一条曲线。"""
    ys = verify_years if verify_years is not None else target_years(scsmi)
    ys = [y for y in ys if y >= VERIFY_START and jjas_norm(scsmi, y - 1) is not None]
    buckets = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))  # s -> bin -> [Σp, count, hits]
    # 需要组合权重：用全部验证年预先算一遍全局技能 -> 权重
    skills = compute_skills(oni, soi, scsmi, ys, dmi=dmi)
    for m in range(1, 13):
        w_t = skills[m].get("w_t")
        if w_t is None:
            continue
        for y in ys:
            comp = shrunk_components(oni, soi, scsmi, y, m, dmi=dmi)
            if comp["p_analog"] is None:
                continue
            p_blend = {s: w_t * comp["p_trans"][s] + (1 - w_t) * comp["p_analog"][s]
                       for s in STATES}
            s_obs = state_of(jjas_norm(scsmi, y))
            for s in STATES:
                b = int(min(9, max(0, math.floor(p_blend[s] * 10.0))))
                if p_blend[s] >= 0.999999:
                    b = 9
                buckets[s][b][0] += p_blend[s]
                buckets[s][b][1] += 1
                if s == s_obs:
                    buckets[s][b][2] += 1.0
    series = {}
    for s in STATES:
        pts = []
        for b in range(10):
            cnt = buckets[s][b][1]
            if cnt == 0:
                pts.append(None)
                continue
            avg_p = buckets[s][b][0] / cnt
            hit = buckets[s][b][2]
            pts.append((b, hit / cnt, avg_p, cnt))
        series[s] = pts
    return series


# ------------------------------------------------------------ 相似年与点预测 ----

def top_analogs(comp, scsmi, top_n=TOP_N_ANALOGS):
    """同月起报相似型的最近邻明细。"""
    am = comp.get("analog_monthly") or comp.get("analog_pooled")
    if am is None:
        return []
    out = []
    for i in range(min(top_n, len(am["years"]))):
        y2 = am["years"][i]
        d = am["dists"][i]
        w = am["weights"][i]
        jj = jjas_norm(scsmi, y2)
        row = scsmi.get(y2, {}).get("norm", None)
        out.append({"year": y2, "dist": d, "weight": w,
                    "state": state_of(jj), "jjas": jj,
                    "months": row[:4] if row else None})
    return out


def point_forecast(comp, scsmi):
    """夏季逐月（6/7/8/9 月）与 JJAS 的相似型加权点预测 + 加权 q5/q95。"""
    am = comp.get("analog_monthly") or comp.get("analog_pooled")
    if am is None:
        return None
    wmap = dict(zip(am["years"], am["weights"]))
    series = []
    for idx, mname in enumerate(["6月", "7月", "8月", "9月", "JJAS"]):
        pairs = []
        for y2 in am["years"]:
            row = scsmi.get(y2, {}).get("norm")
            if row is None or row[idx] is None:
                continue
            pairs.append((row[idx], wmap[y2]))
        if not pairs:
            series.append({"name": mname, "mean": None})
            continue
        mean = sum(v * w for v, w in pairs) / sum(w for _, w in pairs)
        q05 = weighted_quantile(pairs, 0.05)
        q95 = weighted_quantile(pairs, 0.95)
        entry = {"name": mname, "mean": mean, "q05": q05, "q95": q95}
        if idx < 4:
            mp = monthly_state_probs(am, scsmi, idx)
            if mp is not None:
                entry.update(mp)
        series.append(entry)
    return series


# ------------------------------------------------------------ 逐月细化 ----

def monthly_climatology(scsmi, idx):
    """第 idx 月（0=6月..3=9月）全样本三档气候态频率（Dirichlet 先验用）。"""
    cnt = {s: 0.0 for s in STATES}
    n = 0.0
    for y in scsmi:
        row = scsmi.get(y, {}).get("norm")
        if row is None or row[idx] is None:
            continue
        cnt[state_of(row[idx])] += 1.0
        n += 1.0
    if n <= 0:
        return {s: 1.0 / 3.0 for s in STATES}
    return {s: cnt[s] / n for s in STATES}


def monthly_state_probs(am, scsmi, idx, marginal=None, prior=PRIOR_STRENGTH,
                        seed=MC_SEED, draws=MC_DRAWS, mc=True):
    """相似型近邻对第 idx 月标准化值的三档加权计票 + Dirichlet 后验 90% 区间。
    与 JJAS 组合区间同一 MC 约定（固定种子，可复现）；mc=False 时返回解析后验均值。"""
    if am is None:
        return None
    wmap = dict(zip(am["years"], am["weights"]))
    counts = {s: 0.0 for s in STATES}
    tw = 0.0
    sqw = 0.0
    for y2 in am["years"]:
        row = scsmi.get(y2, {}).get("norm")
        if row is None or row[idx] is None:
            continue
        w = wmap.get(y2, 0.0)
        counts[state_of(row[idx])] += w
        tw += w
        sqw += w * w
    if tw <= 1e-12:
        return None
    p = {s: counts[s] / tw for s in STATES}
    n_eff = tw * tw / sqw if sqw > 0 else 0.0
    if marginal is None:
        marginal = monthly_climatology(scsmi, idx)
    alpha = {s: prior * marginal.get(s, 0.0) + max(n_eff, 0.5) * p[s] for s in STATES}
    if not mc:
        tot_a = sum(alpha.values())
        return {"probs": {s: alpha[s] / tot_a for s in STATES},
                "ci_lo": {}, "ci_hi": {}, "n_eff": n_eff}
    rng = random.Random(seed + idx)
    acc = {s: [] for s in STATES}
    for _ in range(draws):
        xs = [rng.gammavariate(max(alpha[s], 1e-9), 1.0) for s in STATES]
        tot = sum(xs)
        for i, s in enumerate(STATES):
            acc[s].append(xs[i] / tot)
    mean = {s: sum(acc[s]) / draws for s in STATES}
    lo, hi = {}, {}
    for s in STATES:
        v = sorted(acc[s])
        lo[s] = v[int(0.05 * (draws - 1))]
        hi[s] = v[int(0.95 * (draws - 1))]
    return {"probs": mean, "ci_lo": lo, "ci_hi": hi, "n_eff": n_eff}


def strong_month_dist(am, scsmi):
    """偏强月数（6-9 月中标准化值 >= +0.5 的个数）的相似型加权分布。
    返回长度 5 的列表（k=0..4）或 None。"""
    if am is None:
        return None
    wmap = dict(zip(am["years"], am["weights"]))
    dist = [0.0] * 5
    tw = 0.0
    for y2 in am["years"]:
        row = scsmi.get(y2, {}).get("norm")
        if row is None or any(row[i] is None for i in range(4)):
            continue
        k = sum(1 for i in range(4) if row[i] >= THRESHOLD)
        dist[k] += wmap.get(y2, 0.0)
        tw += wmap.get(y2, 0.0)
    if tw <= 1e-12:
        return None
    return [d / tw for d in dist]


def compute_monthly_calibration(oni, soi, scsmi, verify_years=None, dmi=None):
    """逐月（6/7/8/9）校准：全部起报月合并的三分类逐月 Brier 技能 + 10 桶命中率。
    与 JJAS 校准同一无泄漏口径（历史预测只用起报时点之前的数据）。"""
    ys = verify_years if verify_years is not None else target_years(scsmi)
    ys = [y for y in ys if y >= VERIFY_START]
    marginals = [monthly_climatology(scsmi, i) for i in range(4)]
    acc = [{
        "buckets": {s: [[0.0, 0.0, 0.0] for _ in range(10)] for s in STATES},
        "counts": {s: 0 for s in STATES},
        "bs": 0.0, "obs": [], "n": 0.0,
    } for _ in range(4)]
    for m in range(1, 13):
        for y in ys:
            comp = shrunk_components(oni, soi, scsmi, y, m, dmi=dmi)
            am = comp.get("analog_monthly") or comp.get("analog_pooled")
            if am is None:
                continue
            row = scsmi.get(y, {}).get("norm")
            for idx in range(4):
                if row is None or row[idx] is None:
                    continue
                mp = monthly_state_probs(am, scsmi, idx, marginal=marginals[idx])
                if mp is None:
                    continue
                obs_s = state_of(row[idx])
                a = acc[idx]
                a["n"] += 1.0
                a["counts"][obs_s] += 1
                a["obs"].append(obs_s)
                for s in STATES:
                    p = mp["probs"][s]
                    a["bs"] += (p - (1.0 if s == obs_s else 0.0)) ** 2 / 3.0
                    b = int(min(9, max(0, math.floor(p * 10.0))))
                    if p >= 0.999999:
                        b = 9
                    a["buckets"][s][b][0] += p
                    a["buckets"][s][b][1] += 1
                    if s == obs_s:
                        a["buckets"][s][b][2] += 1.0
    out = {}
    for idx, a in enumerate(acc):
        n = a["n"]
        if n <= 0:
            out[idx] = None
            continue
        clim = {s: a["counts"][s] / n for s in STATES}
        bs_clim = sum(sum((clim[s] - (1.0 if s == o else 0.0)) ** 2 for s in STATES) / 3.0
                       for o in a["obs"]) / n
        skill = 1.0 - (a["bs"] / n) / bs_clim if bs_clim > 0 else 0.0
        series = {}
        for s in STATES:
            pts = []
            for b in range(10):
                cnt = a["buckets"][s][b][1]
                if cnt == 0:
                    pts.append(None)
                    continue
                pts.append((b, a["buckets"][s][b][2] / cnt,
                            a["buckets"][s][b][0] / cnt, cnt))
            series[s] = pts
        out[idx] = {"buckets": series, "skill": skill,
                    "bs": a["bs"] / n, "bs_clim": bs_clim,
                    "n": int(n), "clim": clim}
    return out


# ------------------------------------------------------------ 转折诊断 ----

def _layer_stats(entries):
    """转折诊断的纯统计层：entries=[(p_blend, s_prev, s_obs)]。
    按 前一年状态==当年状态（持续）/ !=（转折）分层，各层统计组合 Brier 分与技能；
    层样本 <5 或气候 Brier 为 0 时对应层返回 None。"""
    def stat(sub):
        n = len(sub)
        if n < 5:
            return None
        freqs = defaultdict(int)
        for _, _, so in sub:
            freqs[so] += 1
        clim = {s: freqs.get(s, 0) / n for s in STATES}
        sse_m = sum(sum((p.get(s, 0.0) - (1.0 if s == so else 0.0)) ** 2
                        for s in STATES) / 3.0 for p, _, so in sub)
        sse_c = sum(sum((clim[s] - (1.0 if s == so else 0.0)) ** 2
                        for s in STATES) / 3.0 for _, _, so in sub)
        bs_m, bs_c = sse_m / n, sse_c / n
        skill = 1.0 - bs_m / bs_c if bs_c > 0 else None
        return {"n": n, "bs": bs_m, "bs_clim": bs_c, "skill": skill,
                "freq": {s: clim[s] for s in STATES}}
    with_prev = [e for e in entries if e[1] is not None]
    persist = [e for e in with_prev if e[1] == e[2]]
    transit = [e for e in with_prev if e[1] != e[2]]
    return {"persist": stat(persist), "transition": stat(transit),
            "n_total": len(with_prev), "n_no_prev": len(entries) - len(with_prev)}


def transition_diagnostics(oni, soi, scsmi, verify_years=None, dmi=None):
    """转折诊断：把历史回测样本按"前一年状态==当年状态（持续）/!=(转折)"分层，
    统计各层组合模型的 Brier 分与技能。分层只用前一年状态——起报时已知，不涉未来。"""
    ys = verify_years if verify_years is not None else target_years(scsmi)
    ys = [y for y in ys if y >= VERIFY_START and jjas_norm(scsmi, y - 1) is not None]
    skills = compute_skills(oni, soi, scsmi, ys, dmi=dmi)
    entries = []
    for m in range(1, 13):
        w_t = skills[m].get("w_t")
        if w_t is None:
            continue
        for y in ys:
            comp = shrunk_components(oni, soi, scsmi, y, m, dmi=dmi)
            if comp["p_analog"] is None:
                continue
            s_prev = state_of(jjas_norm(scsmi, y - 1))
            s_obs = state_of(jjas_norm(scsmi, y))
            if s_prev is None or s_obs is None:
                continue
            p_blend = {s: w_t * comp["p_trans"][s] + (1 - w_t) * comp["p_analog"][s]
                       for s in STATES}
            entries.append((p_blend, s_prev, s_obs))
    return _layer_stats(entries)


def print_transition_diagnostics(diag):
    """终端输出：转折诊断面板。"""
    print("=" * 78)
    print("转折诊断：持续性年份 vs 转折年份 的组合模型 Brier 分与技能")
    print(f"  有效分层样本  n={diag['n_total']}"
          f"（缺少前一年状态被跳过 {diag['n_no_prev']} 条）")
    for key, label in (("persist", "持续年（今年状态=去年状态）"),
                       ("transition", "转折年（今年状态≠去年状态）")):
        st = diag[key]
        if st is None:
            print(f"  {label:<18} 样本不足（<5），未统计")
            continue
        sk = st["skill"]
        sk_s = f"{sk:+.3f}" if sk is not None else "  n/a "
        freq = " / ".join(f"{s}={st['freq'][s]:.0%}" for s in STATES)
        print(f"  {label:<18} n={st['n']:>3d}  Brier={st['bs']:.4f}"
              f"（气候态 {st['bs_clim']:.4f}）  技能分 {sk_s}   频率 {freq}")
    print("  解读：若转折年技能分显著低于持续年，说明模型在状态转折时滞后失效。")
    print("=" * 78)


# -------------------------------------------------------- 因子分层与年代际 ----

def enso_phase_diagnostics(oni, soi, scsmi, dmi=None, verify_years=None):
    """因子分层：把回测样本按起报时点 ENSO 发展阶段（发展/峰值/衰减/中性）分层，
    统计各层组合 Brier 技能。阶段用起报月（含）之前的数据判定，无未来泄漏。"""
    ys = verify_years if verify_years is not None else target_years(scsmi)
    ys = [y for y in ys if y >= VERIFY_START and jjas_norm(scsmi, y - 1) is not None]
    skills = compute_skills(oni, soi, scsmi, ys, dmi=dmi)
    entries = []
    for m in range(1, 13):
        w_t = skills[m].get("w_t")
        if w_t is None:
            continue
        for y in ys:
            ph = enso_phase_at(oni, y if m <= 5 else y - 1, m)
            if ph is None:
                continue
            comp = shrunk_components(oni, soi, scsmi, y, m, dmi=dmi)
            if comp["p_analog"] is None:
                continue
            s_obs = state_of(jjas_norm(scsmi, y))
            if s_obs is None:
                continue
            p_blend = {s: w_t * comp["p_trans"][s] + (1 - w_t) * comp["p_analog"][s]
                       for s in STATES}
            entries.append((p_blend, ph, s_obs))
    return _phase_layer_stats(entries)


def _phase_layer_stats(entries):
    """entries=[(p_blend, phase, s_obs)]，按 ENSO 发展阶段分层统计 Brier 分与技能。"""
    def stat(sub):
        n = len(sub)
        if n < 5:
            return None
        freqs = defaultdict(int)
        for _, _, so in sub:
            freqs[so] += 1
        clim = {s: freqs.get(s, 0) / n for s in STATES}
        sse_m = sum(sum((p.get(s, 0.0) - (1.0 if s == so else 0.0)) ** 2
                        for s in STATES) / 3.0 for p, _, so in sub)
        sse_c = sum(sum((clim[s] - (1.0 if s == so else 0.0)) ** 2
                        for s in STATES) / 3.0 for _, _, so in sub)
        bs_m, bs_c = sse_m / n, sse_c / n
        skill = 1.0 - bs_m / bs_c if bs_c > 0 else None
        return {"n": n, "bs": bs_m, "bs_clim": bs_c, "skill": skill,
                "freq": {s: clim[s] for s in STATES}}
    layers = {}
    for key, _label in ENSO_PHASES:
        layers[key] = stat([(p, ph, so) for p, ph, so in entries if ph == key])
    return {"layers": layers, "n_total": len(entries)}


def print_enso_phase_diagnostics(diag):
    """终端输出：ENSO 发展阶段分层技能面板。"""
    print("=" * 78)
    print("因子分层诊断：起报时 ENSO 发展阶段（发展/峰值/衰减/中性）的组合 Brier 技能")
    print("  有效分层样本  n=%d（阶段判定只用起报月及之前 ONI，无未来泄漏）" % diag["n_total"])
    for key, label in ENSO_PHASES:
        st = diag["layers"].get(key)
        if st is None:
            print("  %-8s 样本不足（<5），未统计" % label)
            continue
        sk = st["skill"]
        sk_s = f"{sk:+.3f}" if sk is not None else "  n/a "
        freq = " / ".join(f"{s}={st['freq'][s]:.0%}" for s in STATES)
        print("  %-8s n=%3d  Brier=%.4f（气候态 %.4f）  技能分 %s   频率 %s"
              % (label, st["n"], st["bs"], st["bs_clim"], sk_s, freq))
    print("  解读：不同 ENSO 阶段可预报性差异大（如衰减年可预报性高、发展年低），"
          "跨层合并会掩盖信号。")
    print("=" * 78)


def decadal_skill_windows(oni, soi, scsmi, dmi=None, length=30, step=5):
    """年代际非平稳检验：30 年滚动窗（5 年步长）的组合 Brier 技能。
    返回 [{start, end, n, skill}]，skill 为该窗各起报月组合技能均值。"""
    ys = [y for y in target_years(scsmi)
          if y >= VERIFY_START and jjas_norm(scsmi, y - 1) is not None]
    if len(ys) < length:
        return []
    first, last = ys[0], ys[-1]
    starts = list(range(first, last - length + 2, step))
    if not starts or starts[-1] < last - length + 1:
        starts.append(last - length + 1)
    out = []
    for s in starts:
        win = [y for y in ys if s <= y <= s + length - 1]
        if len(win) < max(10, length // 2):
            continue
        skills = compute_skills(oni, soi, scsmi, win, dmi=dmi)
        vals = [skills[m]["skill_blend"] for m in range(1, 13)
                if skills[m].get("skill_blend") is not None]
        out.append({"start": s, "end": s + length - 1, "n": len(win),
                    "skill": (sum(vals) / len(vals)) if vals else None})
    return out


def print_decadal_skill_windows(wins):
    """终端输出：年代际滚动窗技能对照。"""
    print("=" * 78)
    print("年代际非平稳检验：30 年滚动窗（5 年步长）组合 Brier 技能")
    print("  窗宽 30 年，每窗取 12 个起报月组合技能均值；绿=有技能，红=负技能。")
    for w in wins:
        sk = w["skill"]
        sks = "%+.3f" % sk if sk is not None else "  n/a"
        marker = "+" if (sk or 0) > 0 else "-"
        print("  %d-%d  n=%2d  组合技能 %s %s" % (w["start"], w["end"], w["n"], sks, marker))
    has = [w for w in wins if w["skill"] is not None]
    if has:
        print("  解读：窗间技能若正负摆动明显，说明可预报性存在年代际非平稳，"
              "整体平均技能会掩盖窗口差异。")
    print("=" * 78)


# ------------------------------------------------------------ 在线预测 ----

def live_context(today=None):
    """由当前日期确定起报（目标年, 起报月）。"""
    d = today or datetime.date.today()
    if d.month <= 5:
        return d.year, d.month
    return d.year + 1, d.month


def run_forecast(allow_network=True, today=None, json_out=None):
    """在线预测主流程 -> dict 结果。"""
    oni = load_oni(allow_network)
    soi = load_soi(allow_network)
    scsmi = load_scsmi(allow_network)
    dmi = load_dmi(allow_network)
    y_t, m = live_context(today)
    comp = shrunk_components(oni, soi, scsmi, y_t, m, dmi=dmi)
    u850 = load_u850(allow_network)
    onset = build_onset_result(oni, soi, scsmi, u850, y_t, comp, dmi=dmi)

    ys = target_years(scsmi)
    last_target = ys[-1] if ys else None
    skills = compute_skills(oni, soi, scsmi, ys, dmi=dmi)
    sk = skills.get(m, {})
    w_t = sk.get("w_t")
    if w_t is None:
        w_t, wflag = skill_weights(sk.get("skill_t", 0.0), sk.get("skill_a", 0.0))
    else:
        wflag = sk.get("flag")
    p_analog = comp["p_analog"]
    if p_analog is None:
        p_a_use = None
    else:
        p_a_use = p_analog
    mean, lo, hi = blend_forecast(comp["p_trans"], p_a_use, w_t,
                                  comp["alpha_trans"], comp["alpha_analog"])
    analogs = top_analogs(comp, scsmi)
    pf = point_forecast(comp, scsmi)
    smd = strong_month_dist(comp.get("analog_monthly") or comp.get("analog_pooled"), scsmi)

    result = {
        "target_year": y_t, "issue_month": m,
        "lead": lead_months(m),
        "probs": mean, "ci_lo": lo, "ci_hi": hi,
        "w_trans": w_t, "w_analog": 1 - w_t, "weight_flag": wflag,
        "skill_trans": sk.get("skill_t"), "skill_analog": sk.get("skill_a"),
        "skill_blend": sk.get("skill_blend"),
        "n_verify": sk.get("n"),
        "lam_trans": comp["lam_trans"], "lam_analog": comp["lam_analog"],
        "analog_degraded": comp["analog_degraded"],
        "n_eff_analog": comp["n_eff_analog"],
        "analogs": analogs, "point_forecast": pf, "strong_month_dist": smd,
        "onset": onset,
        "oni_latest": latest_oni(oni), "soi_latest": latest_soi(soi),
        "dmi_latest": latest_dmi(dmi),
        "enso_phase": enso_phase_at(oni, y_t if m <= 5 else y_t - 1, m),
        "oni_state_desc": describe_oni_state(oni, y_t if m <= 5 else y_t - 1, m),
        "last_scsmi_year": last_target,
        "prev_scsmi": jjas_norm(scsmi, y_t - 1),
        "prev_state": state_of(jjas_norm(scsmi, y_t - 1)),
        "target_years_full": ys,
        "data_end_year": last_target,
    }
    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    return result, (oni, soi, scsmi)


def print_onset(on):
    """终端输出：南海季风爆发日期历史统计 + 预测 + 验证。"""
    st = on["stats"]
    print("=" * 72)
    print("南海季风爆发日期（东亚夏季风第一次北跳）")
    print("=" * 72)
    print("判定标准   区域%s850hPa纬向风转正并维持3候（王斌等2004）" % ONSET_REGION_LABEL)
    print("历史序列   %d-%d 共 %d 年，均值 %.1f 候，标准差 %.1f 候" % (
        st["first_year"], st["last_year"], st["n"], st["mean"], st["sd"]))
    print("常态       %s 前后；最早 %d 年%s；最晚 %d 年%s" % (
        pentad_label(st["mean"]),
        st["earliest"][0], pentad_label(st["earliest"][1]),
        st["latest"][0], pentad_label(st["latest"][1])))
    print("趋势       %+.2f 候/10年；滞后1年自相关 %s%s" % (
        st["trend_per_decade"] or 0.0,
        ("%+.2f" % st["lag1_r"]) if st["lag1_r"] is not None else "NA",
        "" if st["lag1_r"] is None else "（上一年爆发候的持续性）"))
    if st.get("corr_oni_djf") is not None or st.get("corr_soi_apr") is not None:
        print("ENSO 联系  爆发候 vs 冬 DJF ONI: %s；vs 4月 SOI: %s" % (
            ("%+.2f" % st["corr_oni_djf"]) if st["corr_oni_djf"] is not None else "NA",
            ("%+.2f" % st["corr_soi_apr"]) if st["corr_soi_apr"] is not None else "NA"))
    print("-" * 72)
    print("%d 年南海季风爆发日期预测（气候态 %s）：" % (
        on["target_year"], pentad_label(on["clim"])))
    if on["analog"]:
        a = on["analog"]
        print("  相似型     %s [%s - %s]，近邻 %d 个" % (
            pentad_label(a["mean"]), pentad_label(a["q05"]), pentad_label(a["q95"]), a["n"]))
    else:
        print("  相似型     近邻样本不足")
    if on["reg"]:
        r = on["reg"]
        print("  线性回归   %s（DJF ONI + 4月SOI + 前一年爆发候，训练 %d 年）" % (
            pentad_label(r["mean"]), r["n"]))
    else:
        print("  线性回归   不可用（需前一年爆发候，%d 年暂缺或特征不足）" % (on["target_year"] - 1))
    if on["near"]:
        print("  参考近邻   %s" % "  ".join(
            "%d年%s" % (yy, pentad_label(v)) for v, yy in on["near"]))
    cv = on["cv"]
    print("样本外验证（%d-%d，逐年只用当年之前的数据）：" % (
        cv["verify_start"], st["last_year"]))
    for key, name in (("analog", "相似型"), ("reg", "回归")):
        d = cv.get(key)
        if d is None:
            print("  %-6s 样本不足" % name)
            continue
        print("  %-6s n=%d  r=%s  MAE=%.1f候 (气候态%.1f候, 技能%+.0f%%)  距平符号命中率 %.0f%%" % (
            name, d["n"], ("%+.2f" % d["r"]) if d["r"] is not None else "NA",
            d["mae"], d["mae_clim"], 100 * d["skill"],
            100 * d["hit_rate"]))
    print("注：爆发日期由 NCEP/NCAR R1 推算（该数据 2026-03 停产），2026 年起暂无法更新。")


def print_forecast(res):
    y_t = res["target_year"]
    m = res["issue_month"]
    bar = lambda name, key: "%-6s %5.1f%%  [%5.1f%% - %5.1f%%]" % (
        name, 100 * res["probs"][key], 100 * res["ci_lo"][key], 100 * res["ci_hi"][key])
    print("=" * 72)
    print("中国夏季东南季风（南海夏季风）概率预测")
    print("=" * 72)
    print("目标季    %d 年 6-9 月（JJAS）南海夏季风指数 SCSMI" % y_t)
    print("起报      %d 年 %s（提前约 %d 个月）" % (y_t if m <= 5 else y_t - 1,
              MONTH_CN[m], res["lead"]))
    print("最新 ONI  %s" % ("%+.2f（%04d %s）" % (om[1], om[0][0], SEASON_LABELS[om[0][1]])
                            if (om := res["oni_latest"]) else "未知"))
    if res["soi_latest"]:
        sm = res["soi_latest"]
        print("最新 SOI  %+.1f（%04d-%02d）" % (sm[1], sm[0][0], sm[0][1]))
    if res.get("dmi_latest"):
        dm = res["dmi_latest"]
        print("最新 DMI  %+.2f（%04d-%02d）" % (dm[1], dm[0][0], dm[0][1]))
    print("ENSO 态   %s" % res["oni_state_desc"])
    if res.get("enso_phase"):
        print("ENSO 阶段  %s" % dict(ENSO_PHASES).get(res["enso_phase"], res["enso_phase"]))
    pv = res["prev_scsmi"]
    print("前一年    若可得: SCSMI %s -> %s" % (
        ("%+.2f" % pv) if pv is not None else "未更新",
        res["prev_state"] or "未知"))
    print("-" * 72)
    print("%d 年夏季季风状态概率（%s / %s / %s）" % (y_t, STATE_STRONG, STATE_NORMAL, STATE_WEAK))
    for s in STATES:
        print("  " + bar(s, s))
    print("  模型权重   转移模型 %.2f / 历史相似型 %.2f" % (res["w_trans"], res["w_analog"]))
    if res["weight_flag"]:
        print("  提示       %s" % res["weight_flag"])
    skt = res["skill_trans"]
    ska = res["skill_analog"]
    skb = res["skill_blend"]
    print("  回测技能   " + "转移 %+.3f / 相似 %+.3f / 组合 %+.3f （样本 %s，起报月 %d）" % (
        skt if skt is not None else float("nan"),
        ska if ska is not None else float("nan"),
        skb if skb is not None else float("nan"),
        res["n_verify"] if res["n_verify"] is not None else "0", m))
    if res["analog_degraded"]:
        print("  注意       相似型因特征缺失发生降维")
    print("-" * 72)
    if res["analogs"]:
        print("最相似的历史年份（其后夏季的实际 SCSMI 状态）：")
        for a in res["analogs"]:
            print("  %d 年  距离 %.2f  权重 %.3f  其后: %s (%s)" % (
                a["year"], a["dist"], a["weight"], a["state"],
                ("%+.2f" % a["jjas"]) if a["jjas"] is not None else "NA"))
    if res["point_forecast"]:
        print("夏季逐月强度点预测（相似型加权，标准化值）：")
        parts = []
        for p in res["point_forecast"]:
            if p["mean"] is None:
                parts.append("%s NA" % p["name"])
            else:
                parts.append("%s %+.2f [%+.2f, %+.2f]" % (p["name"], p["mean"], p["q05"], p["q95"]))
        print("  " + "  ".join(parts))
    mp_rows = [p for p in (res.get("point_forecast") or []) if p.get("probs")]
    if mp_rows:
        print("夏季逐月状态概率（相似型加权 ±0.5σ 三档，90% 区间）：")
        for p in mp_rows:
            pr, lo_, hi_ = p["probs"], p["ci_lo"], p["ci_hi"]
            print("  %s   偏强 %5.1f%% [%4.1f - %4.1f]   正常 %5.1f%% [%4.1f - %4.1f]"
                  "   偏弱 %5.1f%% [%4.1f - %4.1f]" % (
                      p["name"],
                      100 * pr[STATE_STRONG], 100 * lo_[STATE_STRONG], 100 * hi_[STATE_STRONG],
                      100 * pr[STATE_NORMAL], 100 * lo_[STATE_NORMAL], 100 * hi_[STATE_NORMAL],
                      100 * pr[STATE_WEAK], 100 * lo_[STATE_WEAK], 100 * hi_[STATE_WEAK]))
        if res.get("strong_month_dist"):
            print("  偏强月数(6-9月)分布  %s" % "  ".join(
                "%d个月 %5.1f%%" % (i, 100 * q)
                for i, q in enumerate(res["strong_month_dist"])))
    if res.get("onset"):
        print_onset(res["onset"])
    else:
        print("南海季风爆发日期模块不可用（缺少 U850 缓存，请先运行 update 子命令）。")
    print("-" * 72)
    print("常规起报窗口：每年 1-5 月（提前 5-1 个月）。本输出为 %d 月提前展望，" % m)
    print("回测表明提前期越长技能越低，请结合技能面板解读。本工具为统计演示，不构成任何决策依据。")


# ------------------------------------------------------------ 终端/HTML 输出 ----

def print_backtest(skills):
    print("=" * 92)
    print("样本外回测：起报月 × 提前期 的三分类 Brier 技能分（vs 该窗口气候态）")
    print("=" * 92)
    hdr = ("起报月", "提前", "样本", "BS气候", "转移技能", "相似技能", "组合技能", "权重转")
    print("%-6s %-4s %-5s %-8s %-9s %-9s %-9s %-6s" % hdr)
    print("-" * 92)
    for m in range(1, 13):
        d = skills.get(m, {})
        if not d.get("n"):
            print("%-6s %-4d %-5s  --" % (MONTH_CN[m], lead_months(m), "0"))
            continue
        print("%-6s %-4d %-5d %-8.4f %-9.3f %-9.3f %-9.3f %-6.2f" % (
            MONTH_CN[m], lead_months(m), d["n"], d["bs_clim"],
            d["skill_t"], d["skill_a"], d["skill_blend"], d["w_t"]))
    print("-" * 92)
    valid = [d for d in skills.values() if d.get("n")]
    if valid:
        avg = lambda k: sum(d[k] for d in valid) / len(valid)
        print("均值      组合技能 %+.3f | 转移 %+.3f | 相似 %+.3f" % (
            avg("skill_blend"), avg("skill_t"), avg("skill_a")))
    print("注：负技能 = 不优于该窗口气候态；10 个月以上提前期通常无技能。")

# ------------------------------------------------------------ SVG ----

def _svg_fc_course(pf):
    """点预测曲线：x = 6/7/8/9/JJAS。"""
    W, H = 640, 300
    pad_l, pad_r, pad_t, pad_b = 46, 30, 30, 44
    idxs = [i for i, p in enumerate(pf) if p["mean"] is not None]
    if not idxs:
        return ""
    vals = [p["mean"] for p in pf if p["mean"] is not None]
    vals += [p["q05"] for p in pf if p["mean"] is not None]
    vals += [p["q95"] for p in pf if p["mean"] is not None]
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, 1.0)
    vmin -= 0.15 * span
    vmax += 0.15 * span
    span = vmax - vmin
    xs = [pad_l + (W - pad_l - pad_r) * i / (len(pf) - 1) for i in range(len(pf))]

    def sy(v):
        return pad_t + (H - pad_t - pad_b) * (1 - (v - vmin) / span)

    parts = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img">' % (W, H)]
    parts.append('<rect x="0" y="0" width="%d" height="%d" fill="#fbfcfe"/>' % (W, H))
    y0 = sy(0)
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#3b6ea5" stroke-width="1" stroke-dasharray="4,3" opacity="0.35"/>' % (pad_l, y0, W - pad_r, y0))
    # 网格
    for frac in (0.25, 0.5, 0.75):
        yy = pad_t + (H - pad_t - pad_b) * (1 - frac)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#dbe3ec" stroke-width="1"/>' % (pad_l, yy, W - pad_r, yy))
    # 阈值虚线
    for tv in (THRESHOLD, -THRESHOLD):
        if vmin <= tv <= vmax:
            parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#c78a3c" stroke-width="1" stroke-dasharray="5,4" opacity="0.6"/>' % (pad_l, sy(tv), W - pad_r, sy(tv)))
            parts.append('<text x="%.1f" y="%.1f" font-size="9" fill="#c78a3c">%+.1fσ</text>' % (pad_l + 3, sy(tv) - 3, tv))
    # 90% 区间带
    lo_pts = [(xs[i], sy(pf[i]["q05"])) for i in idxs]
    hi_pts = [(xs[i], sy(pf[i]["q95"])) for i in idxs]
    band = "M" + " L".join("%.1f,%.1f" % (x, y) for x, y in lo_pts)
    band += " L" + " L".join("%.1f,%.1f" % (x, y) for x, y in reversed(hi_pts)) + " Z"
    parts.append('<path d="%s" fill="#4f81bd" opacity="0.15"/>' % band)
    mean_pts = " ".join("%.1f,%.1f" % (xs[i], sy(pf[i]["mean"])) for i in idxs)
    parts.append('<polyline points="%s" fill="none" stroke="#274b78" stroke-width="2.2"/>' % mean_pts)
    for i in idxs:
        p = pf[i]
        parts.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="#274b78"/>' % (xs[i], sy(p["mean"])))
        parts.append('<text x="%.1f" y="%.1f" font-size="10.5" text-anchor="middle" fill="#274b78" font-weight="600">%+.2f</text>' % (xs[i], sy(p["mean"]) - 12, p["mean"]))
        parts.append('<text x="%.1f" y="%.1f" font-size="10.5" text-anchor="middle" fill="#7a7f88">[%+.2f, %+.2f]</text>' % (xs[i], sy(p["mean"]) + 16, p["q05"], p["q95"]))
        parts.append('<text x="%.1f" y="%.1f" font-size="11" text-anchor="middle" fill="#444">%s</text>' % (xs[i], H - pad_b + 16, p["name"]))
    # 轴
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#8894a3" stroke-width="1"/>' % (pad_l, pad_t + H - pad_t - pad_b, W - pad_r, pad_t + H - pad_t - pad_b))
    parts.append('</svg>')
    return "".join(parts)


def _svg_history(scsmi):
    ys = target_years(scsmi)
    if not ys:
        return ""
    W, H = 640, 300
    pad_l, pad_r, pad_t, pad_b = 44, 24, 26, 40
    vals = [jjas_norm(scsmi, y) for y in ys]
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, 1.0)
    vmin -= 0.12 * span
    vmax += 0.12 * span
    span = vmax - vmin
    x0, x1 = ys[0], ys[-1]
    n = len(ys)

    def sx(y):
        return pad_l + (W - pad_l - pad_r) * (y - x0) / max(x1 - x0, 1)

    def sy(v):
        return pad_t + (H - pad_t - pad_b) * (1 - (v - vmin) / span)

    parts = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img">' % (W, H)]
    parts.append('<rect x="0" y="0" width="%d" height="%d" fill="#fbfcfe"/>' % (W, H))
    # 事件色带
    for y, v in zip(ys, vals):
        s = state_of(v)
        if s == STATE_STRONG:
            color, alpha = "#c0392b", 0.16
        elif s == STATE_WEAK:
            color, alpha = "#2471a3", 0.16
        else:
            continue
        w = max((W - pad_l - pad_r) / max(n, 1) - 1.0, 1.0)
        parts.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="%s" opacity="%.2f"/>' % (
            sx(y) - w / 2, pad_t, w, H - pad_t - pad_b, color, alpha))
    # 阈值线
    for tv in (THRESHOLD, -THRESHOLD):
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#c78a3c" stroke-width="1" stroke-dasharray="5,4" opacity="0.55"/>' % (pad_l, sy(tv), W - pad_r, sy(tv)))
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#8894a3" stroke-width="1"/>' % (pad_l, sy(0), W - pad_r, sy(0)))
    line = " ".join("%.1f,%.1f" % (sx(y), sy(v)) for y, v in zip(ys, vals))
    parts.append('<polyline points="%s" fill="none" stroke="#222" stroke-width="1.6"/>' % line)
    for tk in range(x0, x1 + 1, 10):
        parts.append('<text x="%.1f" y="%d" font-size="10" text-anchor="middle" fill="#555">%d</text>' % (sx(tk), H - pad_b + 15, tk))
    # 图例
    parts.append('<rect x="%.1f" y="%d" width="10" height="10" fill="#c0392b" opacity="0.55"/><text x="%.1f" y="%d" font-size="10.5" fill="#555">偏强年</text>' % (W - 130, pad_t + 10, W - 116, pad_t + 19))
    parts.append('<rect x="%.1f" y="%d" width="10" height="10" fill="#2471a3" opacity="0.55"/><text x="%.1f" y="%d" font-size="10.5" fill="#555">偏弱年</text>' % (W - 130, pad_t + 26, W - 116, pad_t + 35))
    parts.append('<text x="%d" y="%d" font-size="10" fill="#7a7f88">SCSMI JJAS 标准化值</text>' % (pad_l, pad_t - 6))
    parts.append('</svg>')
    return "".join(parts)


def _svg_onset_history(on):
    """历年南海季风爆发候折线 + 气候均值 + 趋势线 + 本轮预测区间。"""
    seq = on.get("series") or {}
    st = on.get("stats") or {}
    if not seq:
        return ""
    ys = sorted(seq)
    W, H = 640, 300
    pad_l, pad_r, pad_t, pad_b = 44, 24, 26, 40
    vmin, vmax = 20.0, 39.0
    x0, x1 = ys[0], ys[-1]

    def sx(y):
        return pad_l + (W - pad_l - pad_r) * (y - x0) / max(x1 - x0, 1)

    def sy(v):
        return pad_t + (H - pad_t - pad_b) * (1 - (v - vmin) / (vmax - vmin))
    parts = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img">' % (W, H)]
    parts.append('<rect x="0" y="0" width="%d" height="%d" fill="#fbfcfe"/>' % (W, H))
    # 气候均值与 ±1σ 带
    if st:
        for k, dash in ((st["mean"] - 1.0 * st["sd"], "3,5"), (st["mean"] + 1.0 * st["sd"], "3,5")):
            parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#b8c4d4" stroke-width="1" stroke-dasharray="%s" opacity="0.6"/>' % (pad_l, sy(k), W - pad_r, sy(k), dash))
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#c78a3c" stroke-width="1.2" stroke-dasharray="6,4"/>' % (pad_l, sy(st["mean"]), W - pad_r, sy(st["mean"])))
    line = " ".join("%.1f,%.1f" % (sx(y), sy(seq[y])) for y in ys)
    parts.append('<polyline points="%s" fill="none" stroke="#2e5d87" stroke-width="1.6"/>' % line)
    # 本轮预测：点 + 90% 区间竖条
    yt = on.get("target_year")
    if yt is not None and on.get("analog"):
        a = on["analog"]
        xp = min(max(sx(max(yt, x0 + 1)), pad_l + 1), W - pad_r - 52)
        parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0392b" stroke-width="5" opacity="0.5"/>' % (xp, sy(a["q05"]), xp, sy(a["q95"])))
        parts.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="#c0392b"/>' % (xp, sy(a["mean"])))
        parts.append('<text x="%.1f" y="%d" font-size="10.5" fill="#c0392b" text-anchor="middle">%d年预测</text>' % (xp, pad_t + 12, yt))
    for tk in range(x0, x1 + 1, 10):
        parts.append('<text x="%.1f" y="%d" font-size="10" text-anchor="middle" fill="#555">%d</text>' % (sx(tk), H - pad_b + 15, tk))
    for v in range(20, 40, 4):
        parts.append('<text x="%d" y="%.1f" font-size="10" text-anchor="middle" fill="#555">第%d候</text>' % (pad_l - 18, sy(v) + 3, v))
    parts.append('<text x="%d" y="%d" font-size="10.5" fill="#7a7f88">爆发候（上=偏早）·  橙虚线=气候均值' % (pad_l, pad_t - 6))
    parts.append('<text x="%d" y="%d" font-size="10.5" fill="#c78a3c">' % (pad_l + 205, pad_t - 6))
    parts.append('</svg>')
    return "".join(parts)


def _svg_reliability(series):
    W, H = 430, 330
    pad = 44
    parts = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" role="img">' % (W, H)]
    parts.append('<rect x="0" y="0" width="%d" height="%d" fill="#fbfcfe"/>' % (W, H))
    colors = {STATE_STRONG: "#c0392b", STATE_NORMAL: "#5b5f66", STATE_WEAK: "#2471a3"}

    def sx(q):
        return pad + (W - pad - 26) * q

    def sy(q):
        return pad + (H - pad - 46) * (1 - q)

    # 对角线
    parts.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%d" stroke="#333" stroke-width="1.2"/>' % (pad, sy(0), sx(1), sy(1)))
    for q in (0.25, 0.5, 0.75):
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%.1f" stroke="#e2e7ee" stroke-width="1"/>' % (sx(q), pad, sx(q), sy(1)))
        parts.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e2e7ee" stroke-width="1"/>' % (pad, sy(q), sx(1), sy(q)))
    for s in STATES:
        pts = []
        for b, row in enumerate(series.get(s, [])):
            if row is None:
                continue
            freq = row[1] if isinstance(row, tuple) else row
            item = series[s][b]
            avg_p = item[2] if item is not None else (b + 0.5) / 10
            frac = (b + 0.5) / 10
            pts.append((sx(avg_p if avg_p is not None else frac), sy(freq if freq is not None else frac)))
        if pts:
            d = " ".join("%.1f,%.1f" % (x, y) for x, y in pts)
            parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (d, colors[s]))
            parts.append('<text x="%.1f" y="%d" font-size="10.5" fill="%s">%s</text>' % (pad + 8, pad + 14 + 12 * STATES.index(s), colors[s], s))
    parts.append('<text x="%d" y="%d" font-size="10" fill="#7a7f88">预测概率</text>' % (pad, H - pad + 40))
    parts.append('<text x="%d" y="%d" font-size="10" fill="#7a7f88" transform="rotate(-90 %d %d)">实际频率</text>' % (14, H / 2 + 40, 14, H / 2 + 40))
    parts.append('</svg>')
    return "".join(parts)


def _skill_table(skills):
    rows = []
    for m in range(1, 13):
        d = skills.get(m, {})
        if not d.get("n"):
            cell = '<td class="cell" style="background:#f0f0f0;color:#999">—</td>'
        else:
            sk = d["skill_blend"]
            # 绿 = 有技能，红 = 负技能
            if sk > 0:
                t = min(sk / 0.4, 1.0)
                bg = "rgba(46,125,50,%.2f)" % (0.12 + 0.5 * t)
                fg = "#1c5c22"
            else:
                t = min(-sk / 0.4, 1.0)
                bg = "rgba(198,40,40,%.2f)" % (0.10 + 0.42 * t)
                fg = "#8e1d1d"
            cell = '<td class="cell" style="background:%s;color:%s">%+.2f</td>' % (bg, fg, sk)
        rows.append('<tr><td class="cell lab">%s</td><td class="cell lab">%d 个月</td>%s</tr>'
                    % (MONTH_CN[m], lead_months(m), cell))
    return '<table class="heat"><thead><tr><th>起报月</th><th>提前期</th><th>组合技能分</th></tr></thead><tbody>%s</tbody></table>' % "".join(rows)


_REPORT_CSS = """
body{font-family:'Segoe UI','Microsoft YaHei',system-ui,sans-serif;background:#f2f4f8;color:#222;margin:0;padding:24px 8px}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}
.sub{color:#667;margin:0 0 6px;font-size:13.5px}
.meta{color:#889;font-size:12px;margin-bottom:22px}
.card{background:#fff;border:1px solid #e3e7ee;border-radius:10px;padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(20,30,60,.05)}
h2{font-size:17px;margin:0 0 12px;border-left:4px solid #3b6ea5;padding-left:10px}
.grid{display:flex;gap:14px;flex-wrap:wrap}
.prob{flex:1 1 220px;border:1px solid #dde3ec;border-radius:10px;padding:14px;text-align:center;background:#fbfcfe}
.prob .st{font-size:16px;font-weight:700}
.prob .val{font-size:30px;font-weight:700;margin:6px 0}
.prob .ci{color:#778;font-size:12px}
.strong{color:#c0392b}.normal{color:#5b5f66}.weak{color:#2471a3}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e2e7ee;padding:7px 9px;text-align:center}
th{background:#f0f3f8}
table.heat .cell.lab{background:#f7f9fc}
.note{font-size:12.5px;color:#667;line-height:1.65}
.warn{background:#fff7e8;border:1px solid #f0d9a8;border-radius:8px;padding:10px 14px;font-size:13px;color:#7a5b1e;margin-bottom:14px}
.footer{color:#99a;font-size:12px;text-align:center;margin:26px 0 8px;line-height:1.7}
.tag{display:inline-block;background:#eef2f8;border-radius:4px;padding:1px 7px;font-size:11.5px;color:#556;margin-right:5px}
a{color:#2d5f9e}
.dim{color:#889;font-size:11px}
.sm{display:flex;align-items:center;gap:8px;margin:4px 0}
.smk{width:52px;font-size:12px;color:#556;flex:none}
.smbar{flex:1;background:#e8edf5;height:10px;border-radius:5px;overflow:hidden}
.smbar div{height:10px;border-radius:5px}
.smp{width:52px;font-size:12px;color:#334;text-align:right;flex:none}
h3{font-size:13.5px;color:#445;margin:14px 0 6px}
svg{max-width:100%;display:block;margin:0 auto}
"""


def build_report_html(res, skills, calib, scsmi, allow_network, mcalib=None, tdiag=None,
                      pdiag=None, dwin=None):
    y_t = res["target_year"]
    m = res["issue_month"]
    oni_l, soi_l = res["oni_latest"], res["soi_latest"]
    dmi_l = res.get("dmi_latest")
    now_s = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def esc(v):
        return html.escape(str(v), quote=True)

    cards = []
    colors = {STATE_STRONG: "strong", STATE_NORMAL: "normal", STATE_WEAK: "weak"}
    for s in STATES:
        bar_w = int(round(100 * res["probs"][s]))
        cards.append(
            '<div class="prob"><div class="st %s">%s</div>'
            '<div class="val %s">%.1f%%</div>'
            '<div class="ci">90%% 区间 %.1f%% - %.1f%%</div>'
            '<div style="background:#e8edf5;height:8px;border-radius:4px;margin-top:8px">'
            '<div style="width:%d%%;height:8px;border-radius:4px;background:#3b6ea5"></div></div></div>'
            % (colors[s], s, colors[s], 100 * res["probs"][s],
               100 * res["ci_lo"][s], 100 * res["ci_hi"][s], bar_w))
    cards_html = '<div class="grid">%s</div>' % "".join(cards)

    warn = ""
    if res["analog_degraded"]:
        warn += ('<div class="warn">相似型模型发生特征降维（部分起报前因子缺失），其估计的不确定性增大，'
                 '解读时请谨慎。</div>')
    if res["weight_flag"]:
        warn += '<div class="warn">%s</div>' % esc(res["weight_flag"])
    if (res["prev_scsmi"] is None):
        warn += ('<div class="warn">前一年（%d 年）SCSMI 尚未在源站点更新，相似型暂缺"前一年季风状态"特征，'
                 '待官方更新后本窗口预测将自动纳入该特征。</div>' % (y_t - 1))
    if res["lead"] >= 8:
        warn += ('<div class="warn">当前属提前 %d 个月的超长提前期展望，回测显示该时段技能接近气候态，'
                 '请仅作背景参考。</div>' % res["lead"])

    analog_rows = ""
    if res["analogs"]:
        for a in res["analogs"]:
            mo = " / ".join("%+.2f" % v for v in a["months"]) if a["months"] else "NA"
            analog_rows += ('<tr><td>%d</td><td>%.2f</td><td>%.3f</td><td>%s</td>'
                            '<td>%s</td><td>%s</td></tr>' % (
                                a["year"], a["dist"], a["weight"], a["state"],
                                ("%+.2f" % a["jjas"]) if a["jjas"] is not None else "NA", mo))
        analog_table = ('<table><thead><tr><th>年份</th><th>距离</th><th>权重</th>'
                        '<th>其后 JJAS 状态</th><th>JJAS 值</th><th>6/7/8/9 月值</th></tr></thead>'
                        '<tbody>%s</tbody></table>' % analog_rows)
    else:
        analog_table = '<p class="note">相似型样本不足，无法列出近邻。</p>'

    pf_html = _svg_fc_course(res["point_forecast"] or [])
    hist_html = _svg_history(scsmi)
    rel_html = _svg_reliability(calib)
    heat_html = _skill_table(skills)

    # 逐月状态概率 / 偏强月数分布 / 逐月校准
    mp_rows = ""
    if res.get("point_forecast"):
        for p in res["point_forecast"]:
            if not p.get("probs"):
                continue
            cells = "".join(
                '<td>%4.1f%% <span class="dim">(%4.1f - %4.1f)</span></td>' % (
                    100 * p["probs"][s], 100 * p["ci_lo"][s], 100 * p["ci_hi"][s])
                for s in STATES)
            mp_rows += "<tr><td>%s</td>%s</tr>" % (esc(p["name"]), cells)
    smd_html = ""
    if res.get("strong_month_dist"):
        bars = "".join(
            '<div class="sm"><span class="smk">%d个月</span>'
            '<div class="smbar"><div style="width:%d%%"></div></div>'
            '<span class="smp">%4.1f%%</span></div>'
            % (i, max(0, int(round(100 * q))), 100 * q)
            for i, q in enumerate(res["strong_month_dist"]))
        smd_html = '<h3>偏强月数（6-9 月中 ≥ +0.5σ 的月数）分布</h3>%s' % bars
    mcl_html = ""
    if mcalib:
        rows_mc = ""
        names = ["6月", "7月", "8月", "9月"]
        for idx in range(4):
            d = mcalib.get(idx)
            if d is None:
                rows_mc += '<tr><td>%s</td><td colspan="3">样本不足</td></tr>' % names[idx]
                continue
            rows_mc += ('<tr><td>%s</td><td>%d</td><td>%+.3f</td>'
                        '<td>%4.1f%% / %4.1f%% / %4.1f%%</td></tr>' % (
                            names[idx], d["n"], d["skill"],
                            100 * d["clim"][STATE_STRONG], 100 * d["clim"][STATE_NORMAL],
                            100 * d["clim"][STATE_WEAK]))
        mcl_html = ('<h3>逐月三档概率的历史校准（全部起报月合并）</h3>'
                    '<table><thead><tr><th>月份</th><th>样本</th>'
                    '<th>Brier 技能（vs 逐月气候态）</th>'
                    '<th>气候态频率 强/正常/弱</th></tr></thead><tbody>%s</tbody></table>'
                    '<p class="note">技能为负表示逐月三档概率同样不优于气候态，'
                    '表中概率应谨慎解读，仅作结构展示。</p>' % rows_mc)
    mon_card = ""
    if mp_rows or smd_html or mcl_html:
        head = ""
        if mp_rows:
            head = ('<table><thead><tr><th>月份</th><th>偏强</th><th>正常</th><th>偏弱</th>'
                    '</tr></thead><tbody>%s</tbody></table>' % mp_rows)
        mon_card = ('<div class="card"><h2>夏季逐月状态概率（相似型加权计票）</h2>'
                    + head + smd_html + mcl_html
                    + '<p class="note">逐月三档概率由相似型近邻对当月标准化值加权计票 + '
                      'Dirichlet 后验蒙特卡洛 90% 区间。季节平均态会掩盖月内反向信号，'
                      '请与上方逐月强度曲线交叉解读。</p>'
                    + '</div>')

    tdiag_card = ""
    if tdiag:
        rows_td = ""
        for key, label in (("persist", "持续年（今年状态=去年状态）"),
                           ("transition", "转折年（今年状态≠去年状态）")):
            st = tdiag[key]
            if st is None:
                rows_td += '<tr><td>%s</td><td colspan="5">样本不足（&lt;5）</td></tr>' % label
                continue
            sk = st["skill"]
            sk_s = ("%+.3f" % sk) if sk is not None else "n/a"
            rows_td += ('<tr><td>%s</td><td>%d</td><td>%.4f</td><td>%.4f</td><td>%s</td>'
                        '<td>%d%% / %d%% / %d%%</td></tr>' % (
                            label, st["n"], st["bs"], st["bs_clim"], sk_s,
                            round(100 * st["freq"][STATE_STRONG]),
                            round(100 * st["freq"][STATE_NORMAL]),
                            round(100 * st["freq"][STATE_WEAK])))
        tdiag_card = (
            '<div class="card"><h2>转折诊断：持续年 vs 转折年</h2>'
            '<table><thead><tr><th>层</th><th>样本</th><th>组合 Brier</th>'
            '<th>气候态 Brier</th><th>Brier 技能</th><th>实际频率 强/正常/弱</th>'
            '</tr></thead><tbody>%s</tbody></table>'
            '<p class="note">夏季风并非渐变：南海季风爆发本身就是一次状态"跳跃"，'
            '季节平均态同样存在持续年与转折年之别。本表把回测样本按'
            '"去年状态 == / != 当年实际状态"分层（分层只用前一年状态，起报时已知，无未来信息）。'
            '若转折年技能显著低于持续年，说明模型在状态跳跃时存在滞后失效，'
            '转折年的概率应比持续年更谨慎解读。</p></div>' % rows_td)

    pdiag_card = ""
    if pdiag:
        rows_pd = ""
        for key, label in ENSO_PHASES:
            st = pdiag["layers"].get(key)
            if st is None:
                rows_pd += '<tr><td>%s</td><td colspan="5">样本不足（&lt;5）</td></tr>' % label
                continue
            sk = st["skill"]
            sk_s = ("%+.3f" % sk) if sk is not None else "n/a"
            rows_pd += ('<tr><td>%s</td><td>%d</td><td>%.4f</td><td>%.4f</td><td>%s</td>'
                        '<td>%d%% / %d%% / %d%%</td></tr>' % (
                            label, st["n"], st["bs"], st["bs_clim"], sk_s,
                            round(100 * st["freq"][STATE_STRONG]),
                            round(100 * st["freq"][STATE_NORMAL]),
                            round(100 * st["freq"][STATE_WEAK])))
        pdiag_card = (
            '<div class="card"><h2>因子分层：起报时 ENSO 发展阶段</h2>'
            '<table><thead><tr><th>起报时阶段</th><th>样本</th><th>组合 Brier</th>'
            '<th>气候态 Brier</th><th>Brier 技能</th><th>实际频率 强/正常/弱</th>'
            '</tr></thead><tbody>%s</tbody></table>'
            '<p class="note">发展阶段由起报月（含）之前的 3 个月 ONI 趋势判定'
            '（变化 ≥ +0.2 为发展、≤ -0.2 为衰减、平缓为峰值，近 6 个月 |ONI| 全 &lt;0.5 为中性），'
            '只用起报时已知数据，无未来泄漏。跨阶段可预报性差异越大，'
            '说明按阶段分层解读越有必要。</p></div>' % rows_pd)

    dwin_card = ""
    if dwin:
        bars = ""
        mx = max((abs(w["skill"]) for w in dwin if w["skill"] is not None),
                 default=0.01) or 0.01
        for w in dwin:
            sk = w["skill"]
            pct = 0 if sk is None else abs(sk) / mx * 100
            cls = "bar-pos" if (sk or 0) >= 0 else "bar-neg"
            bars += ('<div class="drow"><span class="dlab">%d-%d</span>'
                     '<div class="dbar"><div class="dfill %s" style="width:%.1f%%"></div></div>'
                     '<span class="dval">%s</span></div>' % (
                         w["start"], w["end"], cls, pct,
                         ("%+.3f" % sk) if sk is not None else "n/a"))
        dwin_card = (
            '<div class="card"><h2>年代际非平稳检验：30 年滚动窗组合技能</h2>'
            '<p class="note">窗宽 30 年、5 年步长，每窗取 12 个起报月组合 Brier 技能均值。'
            '绿=正技能，红=负技能。窗间摆动越大，说明季风可预报性越存在年代际非平稳，'
            '整体平均技能会掩盖窗口差异。</p>%s'
            '<style>.drow{display:flex;align-items:center;margin:2px 0}'
            '.dlab{width:76px;font-size:11px;color:#888}.dbar{flex:1;background:#222;height:10px;margin:0 8px}'
            '.dfill{height:10px}.bar-pos{background:#2e9e5b}.bar-neg{background:#c0392b}'
            '.dval{width:52px;text-align:right;font-size:11px}</style></div>' % bars)

    skt = res.get("skill_trans")
    ska = res.get("skill_analog")
    skb = res.get("skill_blend")
    def fmt3(x):
        return ("%+.3f" % x) if x is not None else "NA"

    oni_s = "NA"
    if oni_l:
        oni_s = "%+.2f（%04d %s）" % (oni_l[1], oni_l[0][0], SEASON_LABELS[oni_l[0][1]])
    soi_s = "NA"
    if soi_l:
        soi_s = "%+.1f（%04d-%02d）" % (soi_l[1], soi_l[0][0], soi_l[0][1])
    dmi_s = "NA"
    if dmi_l:
        dmi_s = "%+.2f（%04d-%02d）" % (dmi_l[1], dmi_l[0][0], dmi_l[0][1])
    ph_s = "未知"
    if res.get("enso_phase"):
        ph_s = dict(ENSO_PHASES).get(res["enso_phase"], res["enso_phase"])
    py = res["target_years_full"]
    first_yr, last_yr = (py[0], py[-1]) if py else (1948, 1948)

    onset_card = ""
    if res.get("onset"):
        on = res["onset"]
        st = on["stats"]
        preds = []
        if on["analog"]:
            a = on["analog"]
            preds.append('相似型：<b>%s</b>（90%% 区间 %s - %s，近邻 %d 个）' % (
                pentad_label(a["mean"]), pentad_label(a["q05"]),
                pentad_label(a["q95"]), a["n"]))
        else:
            preds.append("相似型：近邻样本不足")
        if on["reg"]:
            preds.append("线性回归：<b>%s</b>（DJF ONI + 4 月 SOI + 前一年爆发候，训练 %d 年）" % (
                pentad_label(on["reg"]["mean"]), on["reg"]["n"]))
        else:
            preds.append("线性回归：不可用（需前一年爆发候，%d 年暂缺或特征不足）" % (on["target_year"] - 1))
        corr_oni = ("%+.2f" % st["corr_oni_djf"]) if st.get("corr_oni_djf") is not None else "不足"
        corr_soi = ("%+.2f" % st["corr_soi_apr"]) if st.get("corr_soi_apr") is not None else "不足"
        lag1 = ("%+.2f" % st["lag1_r"]) if st.get("lag1_r") is not None else "NA"
        ov_rows = ""
        for key, name in (("analog", "相似型（4 月起报口径）"), ("reg", "线性回归")):
            d = on["cv"].get(key)
            if d is None:
                ov_rows += '<tr><td>%s</td><td colspan="5">样本不足</td></tr>' % name
                continue
            ov_rows += ('<tr><td>%s</td><td>%d</td><td>%s</td><td>%.1f / %.1f</td>'
                        '<td>%+.0f%%</td><td>%.0f%%</td></tr>' % (
                            name, d["n"], ("%+.2f" % d["r"]) if d["r"] is not None else "NA",
                            d["mae"], d["mae_clim"], 100 * d["skill"], 100 * d["hit_rate"]))
        onset_card = (
            '<div class="card"><h2>南海季风爆发日期（东亚夏季风第一次北跳）</h2>'
            '<p class="note">判定标准：区域(10°-20°N, 110°-120°E)850 hPa 纬向风由负转正'
            '且其后连续 2 候维持正（王斌等 2004，候平均）。历史序列 %d-%d 共 %d 年：'
            '均值 %s，最早 %d 年%s，最晚 %d 年%s；趋势 %+.2f 候/10 年，滞后 1 年自相关 %s。'
            '与冬季 DJF ONI 相关 %s，与 4 月 SOI 相关 %s。</p>'
            '<p>%d 年 <b>气候态</b> %s；%s</p>%s'
            '<h3>样本外验证（%d-%d，逐年只用当年之前的数据）</h3>'
            '<table><thead><tr><th>方法</th><th>样本</th><th>相关系数 r</th>'
            '<th>MAE 模型/气候态（候）</th><th>MAE 技能</th><th>距平符号命中率</th>'
            '</tr></thead><tbody>%s</tbody></table>'
            '<p class="note">判定所用的 NCEP/NCAR R1 再分析已于 2026 年 3 月停产，'
            '爆发日期序列暂止于 %d 年。统计学上，爆发候的可预报性弱于季节平均强度：'
            '若 MAE 技能为负，说明其结果不优于直接用气候态日期，仅作结构展示。</p></div>'
            % (st["first_year"], st["last_year"], st["n"],
               pentad_label(st["mean"]),
               st["earliest"][0], pentad_label(st["earliest"][1]),
               st["latest"][0], pentad_label(st["latest"][1]),
               st["trend_per_decade"], lag1, corr_oni, corr_soi,
               on["target_year"], pentad_label(on["clim"]),
               "；".join(preds), _svg_onset_history(on),
               on["cv"]["verify_start"], st["last_year"], ov_rows,
               st["last_year"]))

    html_out = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>中国夏季东南季风概率预测（%d 年 6-9 月）</title>
<style>%s</style>
</head>
<body>
<div class="wrap">
<h1>中国夏季东南季风（南海夏季风）概率预测</h1>
<p class="sub">统计相似性外推 · 起报前信息无泄漏 · 经验贝叶斯收缩 · 非官方季节预报</p>
<p class="meta">生成于 %s · 起报 %d 年 %s · 目标 %d 年 6-9 月南海夏季风指数 SCSMI（提前约 %d 个月）</p>
%s
<div class="card">
<h2>当前气候背景</h2>
<table>
<thead><tr><th>最新 ONI</th><th>最新 SOI</th><th>最新 DMI</th><th>ENSO 状态</th>
<th>起报时 ENSO 阶段</th><th>前一年 SCSMI</th><th>前一年状态</th></tr></thead>
<tbody><tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr></tbody>
</table>
<p class="note">ENSO 发展阶段（发展/峰值/衰减/中性）由起报月及之前 3 个月 ONI 趋势判定，只反映起报时已知信息。</p>
</div>
<div class="card">
<h2>%d 年夏季（6-9 月）季风状态概率</h2>
%s
<p class="note" style="margin-top:12px">模型权重：状态转移 %.2f / 历史相似型 %.2f；
起报月 %s 的回测技能：转移 %s / 相似 %s / 组合 %s（样本 %s）。</p>
</div>
<div class="card">
<h2>最相似的历史年份（其后夏季实际走势）</h2>
%s
<p class="note">距离为标准化后的多维特征（ONI/SOI/DMI 轨迹 + 前一年 SCSMI）欧氏距离；权重为高斯核权重。相似年只取自目标年之前的年份，无未来信息。</p>
</div>
<div class="card">
<h2>夏季逐月强度点预测（相似型加权 + 90%% 区间）</h2>
%s
<p class="note">纵轴为 SCSMI 标准化值；±0.5σ 线即三档判定的阈值。该曲线由相似型近邻加权得到，转移模型不参与。</p>
</div>
%s
<div class="card">
<h2>历史南海夏季风指数（%d-%d 年，JJAS 标准化）</h2>
%s
<p class="note">红带=偏强年，蓝带=偏弱年。指数来源：李建平团队（Li &amp; Zeng 2002/2005 统一季风指数框架，DNS 动力学标准化季节变率）。</p>
</div>
%s
<div class="card">
<h2>样本外回测技能面板（起报月 × 提前期）</h2>
%s
<p class="note">绿=优于气候态，红=劣于气候态。1-5 月起报逐月递进（提前 5→1 个月），技能一般随起报月临近夏季而提升，
与春季预报障碍一致；10 个月以上提前期通常无技能。</p>
</div>
<div class="card">
<h2>概率可靠性校准</h2>
%s
<p class="note">横轴=模型给出的预测概率（分桶平均），纵轴=该桶内实际发生频率；曲线越贴近对角线越可靠。
校准样本为全部起报月×验证年的组合预测（同一年会被多次计样本）。</p>
</div>
%s
%s
%s
<div class="card">
<h2>方法与数据</h2>
<p class="note">
<b>目标量：</b>南海夏季风指数 SCSMI（6-9 月，标准化序列）。该指数基于动力学标准化季节变率（DNS），
度量南海季风区低层环流季节性反转的强度，是中国东南部夏季风（东南季风）的直接强度指标；
偏强年对应华南-东南沿海夏季水汽输送总体偏强、降水易偏多（后汛期尤其明显），但指数与区域降水的相关并非 1:1，<b>不能把它当作降水预报直接使用</b>。<br>
<b>阈值：</b>JJAS 标准化值 ≥ +0.5σ 记为偏强，≤ -0.5σ 记为偏弱，其余为正常。<br>
<b>预报因子：</b>起报前 6 个月 ONI 轨迹、SOI 轨迹与印度洋偶极子 DMI 轨迹（NOAA CPC/PSL 逐月更新）+ 前一年 SCSMI（季风自身持续性）。<br>
<b>状态转移模型：</b>P(今夏状态 | 去夏状态, 起报时 ENSO 态)，条件计数 + Dirichlet 先验（强度 4×边际频率）。<br>
<b>历史相似型：</b>多维特征（6 月 ONI + 6 月 SOI + 6 月 DMI 轨迹 + 前一年 SCSMI）滚动标准化欧氏距离 + 高斯核加权（Silverman 带宽，下限 0.35）；
候选年全部早于目标年，滚动标准化只使用候选年统计量，无未来泄漏。<br>
<b>收缩与组合：</b>起报月分层（同月样本）与全历史估计按 λ = n/(n+24) 经验贝叶斯收缩；
组合权重按样本外三分类 Brier 技能分分配，负技能剔除，双负等权。<br>
<b>区间：</b>90%% 区间来自 Dirichlet 后验蒙特卡洛（固定种子，1500 次）。<br>
<b>逐月细化：</b>6/7/8/9 月各自的三档概率由相似型近邻对当月标准化值加权计票 + Dirichlet 后验
90%% 区间；偏强月数（≥+0.5σ 的月数）分布同源统计。<br>
<b>提前期约定：</b>起报月 m≤5 时目标为同年 JJAS（6-9 月），m≥6 时目标为次年 JJAS；提前期 = 距 6 月的月数。<br>
<b>数据源：</b>
<span class="tag">SCSMI <a href="http://lijianping.cn/dct/page/65578">lijianping.cn</a></span>
<span class="tag">ONI <a href="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt">NOAA CPC</a></span>
<span class="tag">SOI <a href="https://www.cpc.ncep.noaa.gov/data/indices/soi">NOAA CPC</a></span>
<span class="tag">DMI <a href="https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/">NOAA PSL</a></span>
</p>
</div>
<div class="card">
<h2>局限性声明</h2>
<p class="note">
1. 本工具为<b>科研与教学演示</b>：概率完全来自历史统计外推，不代表官方季节预报，不构成任何决策依据。<br>
2. SCSMI 由源站点在每年夏季结束后更新，存在约一年滞后（当前数据至 %d 年）；预测目标年的真值尚不可得。<br>
3. 样本仅约 %d 年（%d-%d），条件计数单元稀疏，先验影响较大。<br>
4. 提前 8 个月以上的预测技能接近气候态；转折年（如 ENSO 快速转型年）的滞后失效同样存在。<br>
5. 模型未纳入海温次表层、高原积雪、印度洋-太平洋耦合、台风活动等信息；季风指数与区域旱涝的关系存在年代际变化。<br>
</p>
</div>
<div class="footer">中国夏季东南季风概率预测工具 · 纯 Python 标准库 · 姊妹项目 enso-probability 同一架构<br>
方法参考：Li J. &amp; Zeng Q. (2002, 2003, 2005) 统一季风指数；NOAA CPC ONI/SOI 数据。</div>
</div>
</body>
</html>
""" % (
        y_t, _REPORT_CSS, now_s,
        y_t if m <= 5 else y_t - 1, MONTH_CN[m], y_t, res["lead"],
        warn,
        esc(oni_s), esc(soi_s), esc(dmi_s), esc(res["oni_state_desc"]), esc(ph_s),
        esc(("%+.2f" % res["prev_scsmi"]) if res["prev_scsmi"] is not None else "未更新"),
        esc(res["prev_state"] or "未知"),
        y_t, cards_html,
        res["w_trans"], res["w_analog"], MONTH_CN[m], fmt3(skt), fmt3(ska), fmt3(skb),
        res["n_verify"] if res["n_verify"] is not None else 0,
        analog_table, pf_html, mon_card,
        first_yr, last_yr, hist_html, onset_card, heat_html, rel_html,
        tdiag_card, pdiag_card, dwin_card,
        res["data_end_year"],
        len(res["target_years_full"]), first_yr, last_yr,
    )
    return html_out


# ------------------------------------------------------------ 自测 ----

_SELFTEST_PASS = 0
_SELFTEST_FAIL = 0
_SELFTEST_MSGS = []


def _check(name, cond, detail=""):
    global _SELFTEST_PASS, _SELFTEST_FAIL
    if cond:
        _SELFTEST_PASS += 1
    else:
        _SELFTEST_FAIL += 1
        _SELFTEST_MSGS.append("  FAIL %s %s" % (name, detail))


def _close(a, b, tol):
    return abs(a - b) <= tol


def _synthetic_monsoon(period=True):
    """合成多年季风序列（有 ENSO 依赖的马尔可夫链，主观构造用于自测）。"""
    rng = random.Random(777)
    n = 200
    yrs = list(range(1960, 1960 + n))
    oni_phase = 0  # 0 N, 1 E, 2 L
    rows = {}
    prev = STATE_NORMAL
    for i, y in enumerate(yrs):
        # ENSO 相缓慢随机游走
        if rng.random() < 0.18:
            oni_phase = rng.choice([0, 1, 2])
        pmat = {
            STATE_STRONG: {STATE_STRONG: 0.25, STATE_NORMAL: 0.45, STATE_WEAK: 0.30},
            STATE_NORMAL: {STATE_STRONG: 0.28, STATE_NORMAL: 0.44, STATE_WEAK: 0.28},
            STATE_WEAK: {STATE_STRONG: 0.30, STATE_NORMAL: 0.45, STATE_WEAK: 0.25},
        }
        # ENSO 相调制：E 年偏弱概率上浮，L 年偏强概率上浮
        shift = {0: 0.0, 1: -0.12, 2: 0.12}[oni_phase]
        q = rng.random() + shift
        if q < pmat[prev][STATE_STRONG]:
            s = STATE_STRONG
        elif q < pmat[prev][STATE_STRONG] + pmat[prev][STATE_NORMAL]:
            s = STATE_NORMAL
        else:
            s = STATE_WEAK
        v = {STATE_STRONG: 0.8, STATE_NORMAL: 0.0, STATE_WEAK: -0.8}[s] + rng.gauss(0, 0.15)
        rows[y] = {"norm": [v, v, v, v, v]}
        prev = s
    return rows


def run_selftest(verbose=True):
    global _SELFTEST_PASS, _SELFTEST_FAIL, _SELFTEST_MSGS
    _SELFTEST_PASS = _SELFTEST_FAIL = 0
    _SELFTEST_MSGS = []
    here = os.path.dirname(os.path.abspath(__file__))
    # ---- 用真实缓存文件测解析 ----
    oni = load_oni(allow_network=False)
    soi = load_soi(allow_network=False)
    scsmi = load_scsmi(allow_network=False)

    _check("ONI 解析-已知值", oni.get((1950, 2)) == -1.32, "1950 DJF(结束2月) 应为 -1.32")
    _check("ONI 解析-JAS2026", oni.get((2026, 9)) == 2.16, "2026 JAS(结束9月) 应为 +2.16")
    _check("ONI 解析-月轴映射", oni.get((2026, 8)) == 1.80, "2026 JJA(结束8月) 应为 +1.80")
    _check("ONI 解析-规模", len(oni) > 800, "样本应超过800")

    _check("SOI 解析-1970 年 1 月", soi.get((1970, 1)) == -1.1, "标准化段 1970-01 应为 -1.1")
    _check("SOI 解析-2026 缺测", soi.get((2026, 10)) is None and soi.get((2026, 9)) == -2.0,
           "2026-09=-2.0 且 2026-10 为 None")
    _check("SOI 解析-未来年", (2027, 1) in soi and soi.get((2027, 1)) is None, "2027 年为缺测")
    _check("SOI 解析-规模", len(soi) > 700, "样本应超过700")

    _check("SCSMI 解析-1948", scsmi.get(1948, {}).get("norm", [None] * 5)[4] == 2.656,
           "1948 JJAS norm=2.656")
    _check("SCSMI 解析-2022", scsmi.get(2022, {}).get("norm", [None] * 5)[4] == -1.862,
           "2022 JJAS norm=-1.862")
    _check("SCSMI 解析-年份范围", 1948 in scsmi and 2025 in scsmi, "包含 1948 与 2025")
    _check("SCSMI 解析-规模", len(scsmi) >= 78, "78 年左右")

    # ---- 状态阈值 ----
    _check("状态阈值-上边界", state_of(0.5) == STATE_STRONG and state_of(0.49) == STATE_NORMAL)
    _check("状态阈值-下边界", state_of(-0.5) == STATE_WEAK and state_of(-0.51) == STATE_WEAK)
    _check("状态阈值-中间", state_of(0.0) == STATE_NORMAL)

    # ---- 月运算与提前期 ----
    _check("月份归一", mon_norm(2027, 0) == (2026, 12) and mon_norm(2026, 13) == (2027, 1))
    _check("窗口-跨年", prev_months(2027, 3, 6) == [(2026, 10), (2026, 11), (2026, 12),
                                              (2027, 1), (2027, 2), (2027, 3)])
    _check("窗口-同年", prev_months(2026, 10, 6) == [(2026, 5), (2026, 6), (2026, 7),
                                                (2026, 8), (2026, 9), (2026, 10)])
    _check("提前期公式", [lead_months(m) for m in (1, 5, 6, 12)] == [5, 1, 12, 6])

    # ---- ENSO 状态判定 ----
    # 用真实 ONI 找一个强厄尔尼诺窗口：2015 年 12 月（截至 11 月数据 AMJ..SON 全部高温? 用 2016-01 起报）
    # 稳妥：直接构造合成 ONI
    syn_oni = {}
    for i in range(60):
        y = 2000 + i // 12
        m = i % 12 + 1
        v = 1.0 if (i % 24) < 12 else (0.0 if (i % 24) < 18 else -1.0)
        syn_oni[(y, m)] = v  # 交替 12 个月 +1.0 / 6 个月 0 / 6 个月 -1.0
    # 在 +1.0 段中部（第 6 个月）起报，5 连季应判 E
    start = 2000 + (6) // 12
    _check("ENSO态-E", oni_state_at(syn_oni, 2000, 6) == "E", "")
    _check("ENSO态-转换带", oni_state_at(syn_oni, 2001, 2) in ("E", "N"), "")
    _check("ENSO态-不足", oni_state_at({(2000, 1): 1.0, (2000, 2): 1.0}, 2000, 5) is None, "")

    # ---- 特征轴 ----
    axes = build_feature_axes(2026, 5)
    _check("特征轴-来自窗口", len([a for a in axes if a[0] == "oni"]) == 6 and
           len([a for a in axes if a[0] == "soi"]) == 6 and
           len([a for a in axes if a[0] == "prev"]) == 1, "6+6+1")

    # ---- 转移模型参数回收（合成序列）----
    syn = _synthetic_monsoon()
    syn_oni2 = {}
    for i in range(200 * 12):
        y = 1960 + i // 12
        m = i % 12 + 1
        syn_oni2[(y, m)] = random.Random(123 + i).gauss(0.0, 0.4)
    syn_soi = {(y, m): 0.0 for y in range(1950, 2160) for m in range(1, 13)}
    t_rec = transition_distribution(syn_oni2, syn_soi, syn, 2005, 3)
    _check("转移-概率归一", abs(sum(t_rec["p"].values()) - 1.0) < 1e-9 and
           all(0 <= v <= 1 for v in t_rec["p"].values()), "")
    # 独立计数对照
    n_manual = 0
    for y2 in _candidate_years(syn, 2005):
        if state_of(jjas_norm(syn, y2 - 1)) is not None and state_of(jjas_norm(syn, y2)) is not None:
            n_manual += 1
    _check("转移-样本量合理", t_rec["n"] >= n_manual and n_manual >= 20,
           "n=%d manual=%d" % (t_rec["n"], n_manual))
    t_m2, n_m2 = monthly_transition(syn_oni2, syn_soi, syn, 2005, 3)
    _check("分层-同相计数≤全历史", 0 < n_m2 <= t_rec["n"], "n_m=%d n_all=%d" % (n_m2, t_rec["n"]))

    # ---- 相似型-无泄漏 & 核性质 ----
    a1 = analog_distribution(syn_oni2, syn_soi, syn, 1990, 3)
    _check("相似型-输出结构", a1 is not None and set(a1["p"]) == set(STATES) and
           abs(sum(a1["p"].values()) - 1.0) < 1e-9, "")
    # 改变"未来"年份（1990 之后）目标值，不应影响 1990 年预测
    syn2 = {y: dict(d) for y, d in syn.items()}
    for y in list(syn2):
        if y > 1990:
            syn2[y] = {"norm": [5.0, 5.0, 5.0, 5.0, 5.0]}
    a2 = analog_distribution(syn_oni2, syn_soi, syn2, 1990, 3)
    _check("相似型-无未来泄漏", all(_close(a1["p"][s], a2["p"][s], 1e-12) for s in STATES), "")
    _check("相似型-权重递减", all(w1 >= w2 - 1e-12 for w1, w2 in zip(a1["weights"], a1["weights"][1:])),
           "距离升序 => 权重应单调不减")
    _check("相似型-有效样本", 0 < a1["n_eff"] <= a1["n_cand"], "")
    # 距离尺度不变性：某轴线性放大（正斜率）应不变
    syn_oni3 = {(y, m): v * 2.0 for (y, m), v in syn_oni2.items()}
    a3 = analog_distribution(syn_oni3, syn_soi, syn, 1990, 3)
    _check("相似型-尺度不变", all(_close(a1["p"][s], a3["p"][s], 1e-6) for s in STATES), "")

    # ---- 相似型-蒙特卡洛对照（合成持久序列上应有技能）----
    pers = {}
    p_oni = {}
    pers_soi = {}
    rng = random.Random(4242)
    v = {}
    for y in range(1948, 2070):
        v[y] = 0.75 * v.get(y - 1, 0.0) + rng.gauss(0, 0.5)
    for y, val in v.items():
        pers[y] = {"raw": [val] * 5, "norm": [0.0, 0.0, 0.0, 0.0, 1.9 * val]}
        for m in range(1, 13):
            p_oni[(y, m)] = 0.6 * val + rng.gauss(0, 0.12)
            pers_soi[(y, m)] = -0.6 * val + rng.gauss(0, 0.12)
    pers_valid = list(range(1970, 2020))
    sk_pers = compute_skills(p_oni, pers_soi, pers, verify_years=pers_valid)
    sk_avg = sum(d["skill_blend"] for d in sk_pers.values() if d.get("n")) / 12.0
    _check("相似型-MC对照(持久性)", sk_avg > 0.02, "持久序列上组合技能应 > 0.02，实际 %.3f" % sk_avg)

    # ---- 收缩 ----
    comp = shrunk_components(oni, soi, scsmi, 2000, 3)
    _check("收缩-λ在[0,1]", 0 <= comp["lam_trans"] <= 1 and 0 <= comp["lam_analog"] <= 1, "")
    _check("收缩-输出归一", abs(sum(comp["p_trans"].values()) - 1.0) < 1e-9 and
           abs(sum(comp["p_analog"].values()) - 1.0) < 1e-9, "")
    t_m, n_tm = monthly_transition(oni, soi, scsmi, 2000, 3)
    _check("收缩-月度计数一致性", 0 < n_tm <= sum(
        1 for y2 in _candidate_years(scsmi, 2000)
        if state_of(jjas_norm(scsmi, y2 - 1)) is not None), "n_tm=%d" % n_tm)

    # ---- 技能自洽 & 权重 ----
    skills = compute_skills(oni, soi, scsmi)
    for m in range(1, 13):
        d = skills[m]
        if not d.get("n"):
            continue
        s1 = 1.0 - d["bs_t"] / d["bs_clim"] if d["bs_clim"] > 0 else 0.0
        _check("技能自洽-转[%d]" % m, _close(s1, d["skill_t"], 1e-9), "")
        break  # 抽查一个起报月即可
    _check("技能面板结构", all("skill_blend" in skills[m] for m in range(1, 13)), "")

    # 权重规则
    _check("权重-正技能按比例", _close(skill_weights(0.4, 0.2)[0], 2.0 / 3.0, 1e-9), "")
    _check("权重-负技能剔除", _close(skill_weights(-0.1, 0.2)[0], 0.0, 1e-9), "")
    _check("权重-双负等权", skill_weights(-0.2, -0.1)[0] == 0.5 and
           skill_weights(-0.2, -0.1)[1] is not None, "")

    # ---- 组合与 MC 区间 ----
    comp_live = shrunk_components(oni, soi, scsmi, 1995, 5)
    mean, lo, hi = blend_forecast(comp_live["p_trans"], comp_live["p_analog"], 0.6,
                                  comp_live["alpha_trans"], comp_live["alpha_analog"])
    _check("MC-均值归一", abs(sum(mean.values()) - 1.0) < 1e-6, "")
    _check("MC-区间覆盖", all(lo[s] <= mean[s] <= hi[s] for s in STATES) and
           all(0 <= lo[s] and hi[s] <= 1 for s in STATES), "")
    _check("MC-可复现", blend_forecast(comp_live["p_trans"], comp_live["p_analog"], 0.6,
                                       comp_live["alpha_trans"], comp_live["alpha_analog"])[0] == mean, "")

    # ---- 加权分位 ----
    _check("加权分位-点质量", weighted_quantile([(0.0, 1), (1.0, 0), (2.0, 0)], 0.95) == 0.0, "")
    _check("加权分位-中位", _close(weighted_quantile([(0, 1.0), (10, 1.0)], 0.5), 5.0, 1e-9), "")

    # ---- 校准结构 ----
    calib = compute_calibration(oni, soi, scsmi)
    _check("校准-结构", set(calib) == set(STATES) and
           all(len(calib[s]) == 10 for s in STATES), "")
    for s in STATES:
        for row in calib[s]:
            if row is None or row[1] is None:
                continue
            freq, avgp, cnt = row[1], row[2], row[3]
            _check("校准-频率合法", 0 <= freq <= 1 and 0 <= avgp <= 1 and cnt >= 1, "")

    # ---- 相似年与点预测 ----
    ana = top_analogs(comp_live, scsmi)
    _check("相似年-数量与顺序", len(ana) == TOP_N_ANALOGS and
           all(ana[i]["dist"] <= ana[i + 1]["dist"] for i in range(len(ana) - 1)), "")
    pf = point_forecast(comp_live, scsmi)
    _check("点预测-结构", pf is not None and len(pf) == 5 and all(p["name"] for p in pf), "")
    _check("点预测-分位有序", all(p["mean"] is None or (p["q05"] <= p["mean"] <= p["q95"])
                                   for p in pf), "")

    # ---- 逐月细化 ----
    am_live = comp_live.get("analog_monthly") or comp_live.get("analog_pooled")
    mp0 = monthly_state_probs(am_live, scsmi, 0)
    _check("逐月-结构", mp0 is not None and abs(sum(mp0["probs"].values()) - 1.0) < 1e-6 and
           mp0["n_eff"] > 0, "")
    _check("逐月-区间覆盖并含均值", all(mp0["ci_lo"][s] <= mp0["probs"][s] <= mp0["ci_hi"][s]
                                      for s in STATES), "")
    _check("逐月-并入点预测", all((pf[i].get("probs") is not None) == (pf[i]["mean"] is not None)
                                for i in range(4)) and not pf[4].get("probs"), "")
    # 符号翻转对称性：翻转全部 SCSMI 值后偏强/偏弱逐月后验概率应精确互换（解析模式）
    flip = {y: {"norm": [(-v if v is not None else None) for v in d["norm"]]}
            for y, d in scsmi.items()}
    comp_flip = shrunk_components(oni, soi, flip, 1995, 5)
    am_flip = comp_flip.get("analog_monthly") or comp_flip.get("analog_pooled")
    ok_flip = am_flip is not None and am_flip["years"] == am_live["years"] and \
        all(abs(a - b) < 1e-12 for a, b in zip(am_live["weights"], am_flip["weights"]))
    for idx in range(4):
        m1 = monthly_state_probs(am_live, scsmi, idx, mc=False)
        m2 = monthly_state_probs(am_flip, flip, idx, mc=False)
        ok_flip = ok_flip and m1 is not None and m2 is not None and \
            _close(m1["probs"][STATE_STRONG], m2["probs"][STATE_WEAK], 1e-9) and \
            _close(m1["probs"][STATE_WEAK], m2["probs"][STATE_STRONG], 1e-9) and \
            _close(m1["probs"][STATE_NORMAL], m2["probs"][STATE_NORMAL], 1e-9)
    _check("逐月-符号翻转对称", ok_flip, "")
    smd = strong_month_dist(am_live, scsmi)
    _check("逐月-偏强月数分布", smd is not None and _close(sum(smd), 1.0, 1e-9) and
           all(0 <= q <= 1 for q in smd), "")
    # 构造极小合成样例验证计票与守卫
    mini = {1990: {"norm": [0.9, 0.9, 0.9, 0.9, 1.0]},
            1991: {"norm": [-0.9, -0.9, -0.9, -0.9, -1.0]}}
    mini_am = {"years": [1990, 1991], "weights": [0.7, 0.3]}
    smd2 = strong_month_dist(mini_am, mini)
    _check("逐月-偏强月数合成对照", smd2 is not None and _close(smd2[4], 0.7, 1e-9) and
           _close(smd2[0], 0.3, 1e-9), "")
    mp_mini = monthly_state_probs(mini_am, mini, 0)
    _check("逐月-计票方向性", mp_mini is not None and
           mp_mini["probs"][STATE_STRONG] > mp_mini["probs"][STATE_WEAK], "")
    mini_nom = {y: {"norm": [None, None, None, None, None]} for y in (1990, 1991)}
    _check("逐月-缺测守卫", monthly_state_probs(mini_am, mini_nom, 0) is None and
           strong_month_dist(mini_am, mini_nom) is None, "")
    # 逐月校准结构
    mcal = compute_monthly_calibration(oni, soi, scsmi)
    ok_mcal = all(k in mcal for k in range(4))
    for idx in range(4):
        d = mcal.get(idx)
        if d is None:
            continue
        ok_mcal = ok_mcal and d["n"] >= 1 and -2.0 <= d["skill"] <= 1.0 and \
            all(len(d["buckets"][s]) == 10 for s in STATES) and \
            abs(sum(d["clim"].values()) - 1.0) < 1e-9
        for s in STATES:
            for row in d["buckets"][s]:
                if row is not None:
                    ok_mcal = ok_mcal and 0 <= row[1] <= 1 and 0 <= row[2] <= 1 and row[3] >= 1
    _check("逐月校准-结构", ok_mcal, "")

    # ---- 在线上下文 ----
    _check("在线上下文-10月", live_context(datetime.date(2026, 10, 23)) == (2027, 10), "")
    _check("在线上下文-3月", live_context(datetime.date(2027, 3, 15)) == (2027, 3), "")

    # ---- 爆发日期：候历与窗口 ----
    _check("候历-4/30=第24候", pentad_of_md(4, 30) == 24 and pentad_of_md(5, 1) == 25, "")
    _check("候历-端点", pentad_of_md(1, 1) == 1 and pentad_of_md(12, 31) == 73, "")
    _check("候历-中心日", pentad_center_md(2001, 24) == (4, 28) and
           pentad_center_md(2001, 26) == (5, 8), "")
    _check("候历-标签", pentad_label(24) == "第24候(04月28日)", "")
    _check("窗口-闰年4/1索引", _apr1_index(1948) == 91 and _apr1_index(1949) == 90, "")

    # ---- 爆发日期：OPeNDAP ASCII 解析 ----
    _s_lines = []
    for _t in range(3):
        for _lat in range(5):
            _s_lines.append("[%d][0][%d], %s" % (
                _t, _lat, ", ".join(["%.1f" % (_t + 1)] * 5)))
    _s_text = ("Dataset { ... }\n...\nuwnd.uwnd[3][1][5][5]\n" +
               "\n".join(_s_lines) + "\nuwnd.time[3]\n1.0\n")
    _parsed = parse_u850_ascii(_s_text, n_days=3)
    _check("U850解析-合成样例", _parsed is not None and len(_parsed) == 3 and
           _close(_parsed[0], 1.0, 1e-9) and _close(_parsed[2], 3.0, 1e-9), "")
    _check("U850解析-空文本守卫", parse_u850_ascii("", 3) is None, "")
    _check("U850解析-截断守卫", parse_u850_ascii(_s_text, n_days=4) is None, "")

    # ---- 爆发日期：候聚合与检测 ----
    _d_syn = [-1.0] * 30 + [1.0] * 93  # 4/1-4/30 为负，5/1 起为正
    _pents_daily = daily_to_pentads(_d_syn)
    _check("候聚合-正负与末候", _pents_daily[24] < 0 and _pents_daily[25] > 0 and
           _close(_pents_daily[43], 1.0, 1e-9), "")
    _p_all_neg = {p: -1.0 for p in range(19, 44)}
    _check("爆发检测-全负无爆发", detect_onset(_p_all_neg, 2001) == (None, None), "")
    _p_brk = {p: -1.0 for p in range(19, 44)}
    _p_brk[26], _p_brk[28] = 1.0, 1.0  # 26 正但 27 负：连续 3 候不满足
    _check("爆发检测-转正不足3候", detect_onset(_p_brk, 2001) == (None, None), "")
    _p_ok = {p: -1.0 for p in range(19, 44)}
    for _p in (26, 27, 28):
        _p_ok[_p] = 1.0
    _check("爆发检测-第26候爆发", detect_onset(_p_ok, 2001) == (26, "5月8日"), "")
    _p_ear = {p: -1.0 for p in range(19, 44)}
    for _p in (22, 23, 24):
        _p_ear[_p] = 1.0  # 第 22 候转正但早于判定起点 24，且 25/26 候回负
    _check("爆发检测-起点前转正忽略", detect_onset(_p_ear, 2001) == (None, None), "")
    _p_first = {p: -1.0 for p in range(19, 44)}
    for _p in (24, 25, 26):
        _p_first[_p] = 1.0
    _check("爆发检测-第24候即爆发", detect_onset(_p_first, 2001) == (24, "4月28日"), "")
    _p_gap = {p: -1.0 for p in range(19, 44)}
    for _p in (24, 25, 27, 28, 29):
        _p_gap[_p] = 1.0
    _p_gap[26] = None  # 缺测打断 24/25 起算，但 27/28/29 连续满足
    _check("爆发检测-缺测不妨碍后续", detect_onset(_p_gap, 2001) == (27, "5月13日"), "")

    # ---- 爆发日期：序列与统计 ----
    _u_syn = {1980: [-1.0] * 123, 1981: _d_syn, 1982: _d_syn}
    _seq_syn = onset_series(_u_syn)
    _check("爆发序列-无爆发年剔除", 1980 not in _seq_syn and _seq_syn.get(1981) == 25, "")
    _st_syn = onset_stats({1980: 25, 1981: 25, 1982: 31})
    _check("爆发统计-基本量", _st_syn is not None and _close(_st_syn["mean"], 27.0, 1e-9) and
           _st_syn["earliest"] == (1980, 25) and _st_syn["latest"] == (1982, 31) and
           _close(_st_syn["sd"], math.sqrt(12.0), 1e-6), "")

    # ---- 爆发日期：相关与 OLS 工具 ----
    _check("相关-完全正/负", _close(_pearson([1, 2, 3], [1, 2, 3]), 1.0, 1e-9) and
           _close(_pearson([1, 2, 3], [3, 2, 1]), -1.0, 1e-9), "")
    _check("相关-常量守卫", _pearson([1, 1, 1], [1, 2, 3]) is None, "")
    _coef2 = _ols([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    _check("OLS-两点直线", _coef2 is not None and _close(_coef2[0], 0.0, 1e-9) and
           _close(_coef2[1], 1.0, 1e-9), "")
    _check("OLS-样本不足守卫", _ols([[1.0]], [1.0]) is None, "")

    # ---- 爆发日期：预测与守卫 ----
    _ofa = onset_analog_forecast({1990: 24, 1991: 28, 1992: 29}, [1990, 1991, 1992],
                                 [0.5, 0.3, 0.2])
    _check("爆发预测-相似型加权", _ofa is not None and _close(_ofa["mean"], 26.2, 1e-9) and
           _ofa["q05"] <= _ofa["mean"] <= _ofa["q95"], "")
    _check("爆发预测-相似型样本不足", onset_analog_forecast(
        {1990: 24}, [1990, 1991], [0.5, 0.5]) is None, "")
    _check("爆发预测-近邻缺值剔除", onset_analog_forecast(
        {1990: 24}, [1990, 1992], [0.5, 0.5]) is None, "")
    _oni_syn = {(y, 2): 0.05 * (y - 1975) for y in range(1960, 2010)}
    _soi_syn = {(y, 4): 0.2 * (y % 5 - 2) for y in range(1960, 2010)}
    _rseq = {y: 25 + (y % 3) for y in range(1960, 2006)}
    _rr = onset_reg_predict(_oni_syn, _soi_syn, _rseq, 1995)
    _check("爆发回归-合成输出", _rr is not None and 19 <= _rr["mean"] <= 44 and _rr["n"] >= 5, "")
    _oni_miss = {k: v for k, v in _oni_syn.items() if k != (1995, 2)}
    _check("爆发回归-特征缺失守卫", onset_reg_predict(_oni_miss, _soi_syn, _rseq, 1995) is None, "")
    _check("爆发气候态基准", _close(_clim_before({1988: 22, 1989: 26, 1990: 30}, 1991), 26.0, 1e-9)
           and _clim_before({1990: 24}, 1991) is None, "")
    if am_live is not None:
        _check("爆发预测-近邻无泄漏", all(y < 1995 for y in am_live["years"]), "")

    # ---- 爆发日期：真实缓存集成（缓存缺失时降级为提示通过） ----
    _u850_real = _load_u850_cache()
    if _u850_real:
        _seq_real = onset_series(_u850_real)
        _check("爆发序列-真实缓存覆盖", len(_seq_real) >= 70 and 1948 in _seq_real and
               2025 in _seq_real and 2026 not in _seq_real, "R1 完整窗口应到 2025")
        _st_real = onset_stats(_seq_real)
        _check("爆发序列-气候均值范围", 24 <= _st_real["mean"] <= 31, "气候态约 5 月中下旬")
        _check("爆发序列-年份有序", list(_seq_real) == sorted(_seq_real), "")
    else:
        _check("爆发序列-真实缓存覆盖(跳过)", True, "U850 缓存缺失：运行 update 后强校验")
        _check("爆发序列-气候均值范围(跳过)", True, "")
        _check("爆发序列-年份有序(跳过)", True, "")

    # ---- SVG / HTML 组件 ----
    pf_html = _svg_fc_course(pf)
    _check("SVG-点预测曲线", "<svg" in pf_html and "polyline" in pf_html and "</svg>" in pf_html, "")
    hist_html = _svg_history(scsmi)
    _check("SVG-历史曲线", "<svg" in hist_html and "1948" in hist_html and "</svg>" in hist_html, "")
    rel_html = _svg_reliability(calib)
    _check("SVG-可靠性", "<svg" in rel_html and "rotate" in rel_html, "")
    _check("SVG-配对标签", sum(t.count("<svg") - t.count("</svg>") for t in
                                (pf_html, hist_html, rel_html)) == 0, "")
    heat_html = _skill_table(skills)
    _check("热力表格-结构", heat_html.count("<tr>") >= 13 and "技能分" in heat_html, "")

    # ---- 转折诊断：纯统计层 ----
    _p_unif = {s: 1.0 / 3.0 for s in STATES}
    _td_rnd = random.Random(42)
    _td_entries = []
    for _ in range(60):
        sp = _td_rnd.choice(STATES)
        so = sp if _td_rnd.random() < 0.7 else _td_rnd.choice([s for s in STATES if s != sp])
        _td_entries.append((dict(_p_unif), sp, so))
    _ls = _layer_stats(_td_entries)
    _check("转折层统计-结构", _ls["n_total"] == 60 and _ls["n_no_prev"] == 0, "")
    _check("转折层统计-层样本和", _ls["persist"]["n"] + _ls["transition"]["n"] == 60, "")
    _check("转折层统计-两层均有样本", None not in (_ls["persist"], _ls["transition"]), "")
    for _k in ("persist", "transition"):
        _st = _ls[_k]
        _check("转折层统计-Brier技能上限[%s]" % _k,
               _st["skill"] is None or _st["skill"] <= 1.0 + 1e-9, "")
        _check("转折层统计-频率归一[%s]" % _k,
               _close(sum(_st["freq"].values()), 1.0, 1e-9), "")
    _td_perf = [({s: (1.0 if s == si else 0.0) for s in STATES}, si, si)
                for si in [STATE_STRONG] * 4 + [STATE_WEAK] * 4]
    _ls_perf = _layer_stats(_td_perf)
    _check("转折层统计-确定性手算", _ls_perf["persist"]["n"] == 8 and
           _close(_ls_perf["persist"]["bs"], 0.0, 1e-9) and
           (_ls_perf["persist"]["skill"] is None or
            _close(_ls_perf["persist"]["skill"], 1.0, 1e-9)) and
           _ls_perf["transition"] is None, "")
    _check("转折层统计-空层安全", _layer_stats([])["persist"] is None and
           _layer_stats([])["transition"] is None and
           _layer_stats([])["n_total"] == 0, "")
    _check("转折层统计-缺失前态剔除",
           _layer_stats([(_p_unif, None, STATE_NORMAL)] * 8)["n_no_prev"] == 8 and
           _layer_stats([(_p_unif, None, STATE_NORMAL)] * 8)["n_total"] == 0, "")

    # ---- 转折诊断：真实数据小子集集成（控时，无未来泄漏） ----
    _td = transition_diagnostics(oni, soi, scsmi, verify_years=list(range(1990, 1996)))
    _t_pd, _t_tr = _td["persist"], _td["transition"]
    _check("转折诊断-真实子集结构", isinstance(_td, dict) and _td["n_total"] >= 0, "")
    _check("转折诊断-真实子集分层守恒",
           (_t_pd is None or _t_pd["n"] >= 5) and (_t_tr is None or _t_tr["n"] >= 5) and
           (_t_pd is None or _t_tr is None or _t_pd["n"] + _t_tr["n"] == _td["n_total"]), "")

    # ---- DMI 解析与特征集成 ----
    _dmi_txt = (" 1870 2026\n"
                "1870  -0.10  -0.20  -0.30  -0.40  -0.50  -0.60  -0.70  -0.80  -0.90"
                "  -1.00  -1.10  -1.20\n"
                "1997   0.10 -9999.000   0.30   0.40   0.50   0.60   0.70   0.80"
                "   0.90   1.00   1.10   1.20\n"
                "Timeseries output created at NOAA PSL\nhttps://psl.noaa.gov/gcos_wgsp/timeseries/DMI\n")
    _pd = parse_dmi(_dmi_txt)
    _check("DMI解析-年行与值", _close(_pd.get((1870, 1)), -0.10, 1e-9) and
           _close(_pd.get((1870, 12)), -1.20, 1e-9) and _close(_pd.get((1997, 1)), 0.10, 1e-9), "")
    _check("DMI解析-缺失", _pd.get((1997, 2)) is None and _close(_pd.get((1997, 3)), 0.30, 1e-9), "")
    _check("DMI解析-页脚表头忽略", (2026, 1) not in _pd and (1870, 13) not in _pd, "")
    _dl = load_dmi(allow_network=False)
    if _dl:
        _check("DMI真实缓存-覆盖", len(_dl) > 1500 and
               any(y >= 2020 and v is not None for (y, _m), v in _dl.items()), "")
    else:
        print("  提示：暂无 DMI 本地缓存（运行 update 后自动补齐），DMI 真实缓存测试跳过")
    _dmi_zero = {(y, m): 0.0 for y in range(1950, 2027) for m in range(1, 13)}
    _ax19 = build_feature_axes(2020, 4)
    _check("DMI因子-轴数", len(POS_AXES) == 19 and
           len([a for a in _ax19 if a[0] == "dmi"]) == 6 and len(_ax19) == 19, "")
    _vq = feature_vector(_ax19, oni, soi, scsmi, 2020, dmi=_dmi_zero)
    _check("DMI因子-向量取值", all(v == 0.0 for ax, v in _vq.items() if ax[0] == "dmi") and
           _vq[("prev", 2019)] == jjas_norm(scsmi, 2019), "")
    _check("DMI因子-缺失降级", axis_value(("dmi", 2000, 1), oni, soi, scsmi, 2020) is None, "")
    _p_no_d = analog_distribution(oni, soi, scsmi, 2020, 4)
    _p_no_d2 = analog_distribution(oni, soi, scsmi, 2020, 4, dmi=None)
    _p_z_d = analog_distribution(oni, soi, scsmi, 2020, 4, dmi=_dmi_zero)
    _check("DMI因子-无DMI等价", bool(_p_no_d) and bool(_p_no_d2) and
           _p_no_d["p"] == _p_no_d2["p"] and _p_no_d["n_cand"] == _p_no_d2["n_cand"], "")
    _check("DMI因子-零常数候选守恒与分布合法", bool(_p_z_d) and
           abs(_p_z_d["n_cand"] - _p_no_d["n_cand"]) <= 5 and
           _close(sum(_p_z_d["p"].values()), 1.0, 1e-12) and
           all(0.0 <= _p_z_d["p"][s] <= 1.0 for s in STATES), "")

    # ---- ENSO 发展阶段判定（纯函数） ----
    def _mk_oni(seq):
        d = {}
        yr, mo = 1990, 1
        for v in seq:
            d[(yr, mo)] = v
            mo += 1
            if mo > 12:
                mo = 1
                yr += 1
        return d
    _check("ENSO阶段-发展", enso_phase_at(
        _mk_oni([0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]), 1990, 12) == "D", "")
    _check("ENSO阶段-衰减", enso_phase_at(
        _mk_oni([1.3, 1.1, 0.9, 0.7, 0.5, 0.3, -0.1, -0.3, -0.5, -0.7, -0.9, -1.1]), 1990, 12) == "C", "")
    _check("ENSO阶段-峰值", enso_phase_at(
        _mk_oni([0.8, 0.9, 1.0, 1.0, 0.95, 1.0, 1.0, 1.05, 1.0, 1.05, 1.0, 1.0]), 1990, 12) == "P", "")
    _check("ENSO阶段-中性", enso_phase_at(
        _mk_oni([0.2, 0.1, -0.2, 0.1, 0.0, 0.1, -0.1, -0.2, 0.0, 0.1, -0.1, 0.0]), 1990, 12) == "N", "")
    _check("ENSO阶段-数据不足", enso_phase_at({(1990, 1): 1.0}, 1990, 3) is None, "")

    # ---- ENSO 阶段分层层统计 ----
    _perf2 = [({s: (1.0 if s == si else 0.0) for s in STATES}, "D", si)
              for si in [STATE_STRONG] * 3 + [STATE_NORMAL] * 3 + [STATE_WEAK] * 3]
    _ls2 = _phase_layer_stats(_perf2)
    _check("阶段层统计-确定性手算", _ls2["n_total"] == 9 and _ls2["layers"]["D"]["n"] == 9 and
           _close(_ls2["layers"]["D"]["bs"], 0.0, 1e-9) and
           _ls2["layers"]["D"]["skill"] is not None, "")
    _check("阶段层统计-其余层为空", all(_ls2["layers"][k] is None for k in ("P", "C", "N")), "")
    _check("阶段层统计-空输入安全", _phase_layer_stats([])["n_total"] == 0 and
           all(v is None for v in _phase_layer_stats([])["layers"].values()), "")
    _ph = enso_phase_diagnostics(oni, soi, scsmi, dmi=_dl or _dmi_zero,
                                 verify_years=list(range(1990, 1996)))
    _check("阶段分层诊断-结构", isinstance(_ph, dict) and "layers" in _ph and
           _ph["n_total"] >= 0, "")
    _check("阶段分层诊断-分层守恒", sum(st["n"] for st in _ph["layers"].values() if st) <=
           _ph["n_total"], "")
    _check("阶段分层诊断-小样本层为None", all(st is None or st["n"] >= 5
                                               for st in _ph["layers"].values()), "")
    _check("阶段分层诊断-技能上限", all(st["skill"] is None or st["skill"] <= 1.0 + 1e-9
                                         for st in _ph["layers"].values() if st), "")

    # ---- 年代际滚动窗（合成序列 + 真实因子，控时长） ----
    _rows = _synthetic_monsoon()
    _scsmi_sub = {y: v for y, v in _rows.items() if 1961 <= y <= 2020}
    _wins = decadal_skill_windows(oni, soi, _scsmi_sub, dmi=_dmi_zero, length=20, step=10)
    _check("年代际窗-数量", 3 <= len(_wins) <= 6, "应生成若干滚动窗")
    _check("年代际窗-窗长与样本", all(w["end"] - w["start"] + 1 == 20 and w["n"] == 20
                                       for w in _wins), "")
    _check("年代际窗-排序递增", all(_wins[i]["start"] < _wins[i + 1]["start"]
                                     for i in range(len(_wins) - 1)), "")
    _check("年代际窗-技能上限", all(w["skill"] is None or w["skill"] <= 1.0 + 1e-9
                                     for w in _wins), "")

    # ---- 完整报告 ----
    res, _ = run_forecast(allow_network=False, today=datetime.date(2026, 3, 15))
    report = build_report_html(res, compute_skills(oni, soi, scsmi), calib, scsmi, False)
    for kw in ("概率", "相似", "技能", "可靠性", "局限性", "SCSMI", "SVG".lower()):
        _check("报告-关键词[%s]" % kw, kw in report or kw in report.lower(), "")
    report_td = build_report_html(res, compute_skills(oni, soi, scsmi), calib, scsmi, False,
                                  tdiag=_td)
    _check("报告-转折诊断卡", "转折诊断" in report_td and "持续年" in report_td and
           "转折年" in report_td, "")
    _check("报告-转折标签闭合", report_td.count("<html") == report_td.count("</html>") == 1 and
           report_td.count("<body>") == report_td.count("</body>") == 1, "")
    _check("报告-标签闭合", report.count("<html") == report.count("</html>") == 1 and
           report.count("<body>") == report.count("</body>") == 1, "")

    # ---- 爆发日期：完整链路集成（依赖 U850 缓存） ----
    if res.get("onset"):
        on = res["onset"]
        _check("爆发集成-结构", on["stats"] is not None and
               on["cv"]["verify_start"] == ONSET_VERIFY_START, "")
        _check("爆发集成-预测区间", on["analog"] is None or
               on["analog"]["q05"] <= on["analog"]["mean"] <= on["analog"]["q95"], "")
        _check("爆发集成-近邻早于目标年", all(y < res["target_year"]
                                              for _, y in (on["near"] or [])), "")
        svg_on = _svg_onset_history(on)
        _check("爆发集成-SVG", "<svg" in svg_on and "polyline" in svg_on, "")
        _check("爆发集成-报告卡", "南海季风爆发日期" in report, "")
        for _k in ("analog", "reg"):
            _d = on["cv"].get(_k)
            if _d is not None:
                _check("爆发集成-CV结构[%s]" % _k, _d["n"] >= 5 and
                       (_d["r"] is None or -1.001 <= _d["r"] <= 1.001) and
                       _d["mae"] >= 0 and -5 <= _d["skill"] <= 1.5 and
                       0 <= _d["hit_rate"] <= 1, "")

    print("自测完成：%d 项通过，%d 项失败" % (_SELFTEST_PASS, _SELFTEST_FAIL))
    for msg in _SELFTEST_MSGS:
        print(msg)
    return _SELFTEST_FAIL == 0


# ------------------------------------------------------------ CLI ----

def main():
    ap = argparse.ArgumentParser(
        description="中国夏季东南季风（南海夏季风）概率预测工具（纯 Python 标准库）")
    ap.add_argument("command", nargs="?", default="forecast",
                    choices=["update", "forecast", "backtest", "report", "selftest"],
                    help="子命令（默认 forecast）")
    ap.add_argument("--offline", action="store_true", help="禁止联网，只用本地缓存")
    ap.add_argument("--json", nargs="?", const=os.path.join(OUTPUT_DIR, "forecast.json"),
                    help="把预测结果写入 JSON 文件（可带路径）")
    ap.add_argument("--out", default=None, help="report 子命令的输出文件路径（默认 output/monsoon_report.html）")
    args = ap.parse_args()

    allow_network = not args.offline

    if args.command == "update":
        update_data()
        return 0
    if args.command == "selftest":
        ok = run_selftest()
        return 0 if ok else 1
    if args.command == "backtest":
        oni = load_oni(allow_network)
        soi = load_soi(allow_network)
        scsmi = load_scsmi(allow_network)
        dmi = load_dmi(allow_network)
        skills = compute_skills(oni, soi, scsmi, dmi=dmi)
        print_backtest(skills)
        u850 = load_u850(allow_network)
        seq = onset_series(u850)
        if len(seq) >= 25:
            st = onset_stats(seq)
            cv = onset_cross_validate(oni, soi, scsmi, seq, dmi=dmi)
            print("=" * 92)
            print("南海季风爆发日期：样本外验证（%d-%d，共 %d 年序列，均值 %s）" % (
                cv["verify_start"], st["last_year"], st["n"], pentad_label(st["mean"])))
            print("=" * 92)
            for key, name in (("analog", "相似型(4月起报)"), ("reg", "线性回归")):
                d = cv.get(key)
                if d is None:
                    print("%-12s 样本不足" % name)
                    continue
                print("%-12s n=%d  r=%s  MAE=%.1f候 (气候态%.1f候,技能%+.0f%%)  距平符号命中率 %.0f%%" % (
                    name, d["n"], ("%+.2f" % d["r"]) if d["r"] is not None else "NA",
                    d["mae"], d["mae_clim"], 100 * d["skill"], 100 * d["hit_rate"]))
        else:
            print("南海季风爆发日期序列不足（缺 U850 缓存），跳过。请先运行 update。")
        print_transition_diagnostics(transition_diagnostics(oni, soi, scsmi, dmi=dmi))
        print_enso_phase_diagnostics(enso_phase_diagnostics(oni, soi, scsmi, dmi=dmi))
        print_decadal_skill_windows(decadal_skill_windows(oni, soi, scsmi, dmi=dmi))
        return 0
    if args.command == "report":
        oni = load_oni(allow_network)
        soi = load_soi(allow_network)
        scsmi = load_scsmi(allow_network)
        dmi = load_dmi(allow_network)
        res, _ = run_forecast(allow_network=allow_network)
        ys = target_years(scsmi)
        skills = compute_skills(oni, soi, scsmi, ys, dmi=dmi)
        calib = compute_calibration(oni, soi, scsmi, ys, dmi=dmi)
        mcalib = compute_monthly_calibration(oni, soi, scsmi, ys, dmi=dmi)
        tdiag = transition_diagnostics(oni, soi, scsmi, ys, dmi=dmi)
        pdiag = enso_phase_diagnostics(oni, soi, scsmi, dmi=dmi, verify_years=ys)
        dwin = decadal_skill_windows(oni, soi, scsmi, dmi=dmi)
        report = build_report_html(res, skills, calib, scsmi, allow_network, mcalib,
                                   tdiag, pdiag, dwin)
        out_path = args.out or os.path.join(OUTPUT_DIR, "monsoon_report.html")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print("报告已写入 %s" % out_path)
        return 0
    # forecast（默认）
    res, _ = run_forecast(allow_network=allow_network, json_out=None)
    print_forecast(res)
    if args.json:
        out_path = args.json
        res2, _ = run_forecast(allow_network=allow_network, json_out=out_path)
        print("JSON 已写入 %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())