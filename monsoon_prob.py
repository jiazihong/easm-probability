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
  2. 历史相似型模型：13 维特征（经滚动标准化）欧氏距离 + 高斯核加权
  3. 起报月分层 + 经验贝叶斯收缩：p = λ·p_同月 + (1-λ)·p_全历史，λ = n/(n+24)
  4. 组合权重按样本外三分类 Brier 技能分分配，负技能剔除，双负退化等权
  5. 90% 区间来自 Dirichlet 蒙特卡洛抽样（固定种子）
  6. 相似年份透明化 + 夏季逐月强度（6/7/8/9 月与 JJAS）加权点预测
  7. 技能面板：起报月（1-12 月）× 其唯一提前期（1-12 个月）的 Brier 技能
  8. 概率可靠性校准曲线

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
            header = "# fetched=%sZ source=%s\n" % (datetime.datetime.utcnow().isoformat(timespec="seconds"), url)
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
            header = "# fetched=%sZ source=%s\n" % (datetime.datetime.utcnow().isoformat(timespec="seconds"), url)
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


def update_data():
    """update 子命令：刷新三个数据源并写缓存。"""
    print("刷新数据源：")
    for name, url, cache in [("ONI", ONI_URL, ONI_CACHE),
                             ("SOI", SOI_URL, SOI_CACHE),
                             ("SCSMI", SCSMI_URL, SCSMI_CACHE)]:
        text, from_cache = fetch_text(url, cache, refresh=True, allow_network=True)
        print("  %-6s %s %s" % (name, "缓存回退" if from_cache else "OK", os.path.basename(cache)))


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


# ------------------------------------------------------------ 特征构造 ----

def build_feature_axes(y, m):
    """起报 (年 y, 月 m) 的特征轴 id 列表：
    6 个 ONI 月 + 6 个 SOI 月（截至 m-1）+ 前一年 SCSMI。"""
    wins = prev_months(y, m - 1, WINDOW)
    axes = [("oni", yy, mm) for yy, mm in wins]
    axes += [("soi", yy, mm) for yy, mm in wins]
    axes += [("prev", y - 1)]
    return axes


def axis_value(ax, oni, soi, scsmi, for_year):
    kind = ax[0]
    if kind == "oni":
        return oni.get((ax[1], ax[2]))
    if kind == "soi":
        return soi.get((ax[1], ax[2]))
    return jjas_norm(scsmi, for_year - 1)


def feature_vector(axes, oni, soi, scsmi, y):
    return {ax: axis_value(ax, oni, soi, scsmi, y) for ax in axes}


# ------------------------------------------------------------ 模型样本池 ----

POS_AXES = [("oni", k) for k in range(WINDOW)] + \
           [("soi", k) for k in range(WINDOW)] + [("prev",)]

_POOL_CACHE = {}


def _build_pool(oni, soi, scsmi):
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
            "oni": oni, "soi": soi, "scsmi": scsmi}


def get_pool(oni, soi, scsmi):
    key = (id(oni), id(soi), id(scsmi))
    p = _POOL_CACHE.get(key)
    if p is None:
        p = _build_pool(oni, soi, scsmi)
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

def _vec_from_months(oni, soi, scsmi, T, months):
    v = {}
    for k, (yy, mm) in enumerate(months):
        v[("oni", k)] = oni.get((yy, mm))
        v[("soi", k)] = soi.get((yy, mm))
    v[("prev",)] = jjas_norm(scsmi, T - 1)
    return v


def _analog_core_pool(pool, y_t, m, phase, axes, min_candidates, bw_floor):
    """给定特征轴集合，从样本池计算高斯核加权状态概率与近邻明细。"""
    oni, soi, scsmi = pool["oni"], pool["soi"], pool["scsmi"]
    Yw0, w0 = mon_norm(y_t - (1 if m >= 6 else 0), m - 1)
    q_months = prev_months(Yw0, w0, WINDOW)
    qv = _vec_from_months(oni, soi, scsmi, y_t, q_months)
    rows = []
    for T, w, months in pool["a_rows"]:
        if T >= y_t:
            continue
        if phase is not None and w != w0:
            continue
        rows.append((T, w, months))
    if len(rows) < min_candidates:
        return None
    vecs = {i: _vec_from_months(oni, soi, scsmi, T, months)
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
                        min_candidates=MIN_CANDIDATES, bw_floor=BW_FLOOR, phase=None):
    """历史相似型（phase=None 为全窗口混合，phase=w0 为同相）。
    候选样本不足时逐级降维（去 SOI -> 去前一年 -> 仅 ONI）。"""
    pool = get_pool(oni, soi, scsmi)
    deg_chain = [
        ("全特征", POS_AXES),
        ("无SOI", [ax for ax in POS_AXES if ax[0] != "soi"]),
        ("无前一年", [ax for ax in POS_AXES if ax[0] != "prev"]),
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

def monthly_analog(oni, soi, scsmi, y_t, m, min_candidates=MIN_CANDIDATES):
    """相似型的同相（同窗口结束月）版本。"""
    pool = get_pool(oni, soi, scsmi)
    _, w0 = mon_norm(y_t - (1 if m >= 6 else 0), m - 1)
    res = _analog_core_pool(pool, y_t, m, phase=w0, axes=POS_AXES,
                            min_candidates=min_candidates, bw_floor=BW_FLOOR)
    n_all = sum(1 for T, w, _m in pool["a_rows"] if T < y_t and w == w0)
    if res is None:
        return None, 0, n_all
    return res, res["n_cand"], n_all


def shrunk_components(oni, soi, scsmi, y_t, m, prior=PRIOR_STRENGTH, k_shrink=SHRINK_K):
    """经验贝叶斯收缩后的两个模型组件与混合参数。"""
    t_pool = transition_distribution(oni, soi, scsmi, y_t, m, prior, phase=None)
    t_m, n_tm = monthly_transition(oni, soi, scsmi, y_t, m, prior)
    a_pool = analog_distribution(oni, soi, scsmi, y_t, m, phase=None)
    a_m_res, n_am, _ = monthly_analog(oni, soi, scsmi, y_t, m)

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

def compute_skills(oni, soi, scsmi, verify_years=None, verbose=False):
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
            a_pool = analog_distribution(oni, soi, scsmi, y, m)
            if a_pool is None or a_pool.get("degraded"):
                continue  # 两模型都必须给出非降级估计才计入技能比较
            comp = shrunk_components(oni, soi, scsmi, y, m)
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


def compute_calibration(oni, soi, scsmi, verify_years=None):
    """可靠性校准：把历史（m × y）组合概率分 10 桶，每状态一条曲线。"""
    ys = verify_years if verify_years is not None else target_years(scsmi)
    ys = [y for y in ys if y >= VERIFY_START and jjas_norm(scsmi, y - 1) is not None]
    buckets = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))  # s -> bin -> [Σp, count, hits]
    # 需要组合权重：用全部验证年预先算一遍全局技能 -> 权重
    skills = compute_skills(oni, soi, scsmi, ys)
    for m in range(1, 13):
        w_t = skills[m].get("w_t")
        if w_t is None:
            continue
        for y in ys:
            comp = shrunk_components(oni, soi, scsmi, y, m)
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
        series.append({"name": mname, "mean": mean, "q05": q05, "q95": q95})
    return series


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
    y_t, m = live_context(today)
    comp = shrunk_components(oni, soi, scsmi, y_t, m)

    ys = target_years(scsmi)
    last_target = ys[-1] if ys else None
    skills = compute_skills(oni, soi, scsmi, ys)
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
        "analogs": analogs, "point_forecast": pf,
        "oni_latest": latest_oni(oni), "soi_latest": latest_soi(soi),
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
    print("ENSO 态   %s" % res["oni_state_desc"])
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
svg{max-width:100%;display:block;margin:0 auto}
"""


def build_report_html(res, skills, calib, scsmi, allow_network):
    y_t = res["target_year"]
    m = res["issue_month"]
    oni_l, soi_l = res["oni_latest"], res["soi_latest"]
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
    py = res["target_years_full"]
    first_yr, last_yr = (py[0], py[-1]) if py else (1948, 1948)

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
<thead><tr><th>最新 ONI</th><th>最新 SOI</th><th>ENSO 状态</th><th>前一年 SCSMI</th><th>前一年状态</th></tr></thead>
<tbody><tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr></tbody>
</table>
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
<p class="note">距离为标准化后的 13 维特征欧氏距离；权重为高斯核权重。相似年只取自目标年之前的年份，无未来信息。</p>
</div>
<div class="card">
<h2>夏季逐月强度点预测（相似型加权 + 90%% 区间）</h2>
%s
<p class="note">纵轴为 SCSMI 标准化值；±0.5σ 线即三档判定的阈值。该曲线由相似型近邻加权得到，转移模型不参与。</p>
</div>
<div class="card">
<h2>历史南海夏季风指数（%d-%d 年，JJAS 标准化）</h2>
%s
<p class="note">红带=偏强年，蓝带=偏弱年。指数来源：李建平团队（Li &amp; Zeng 2002/2005 统一季风指数框架，DNS 动力学标准化季节变率）。</p>
</div>
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
<div class="card">
<h2>方法与数据</h2>
<p class="note">
<b>目标量：</b>南海夏季风指数 SCSMI（6-9 月，标准化序列）。该指数基于动力学标准化季节变率（DNS），
度量南海季风区低层环流季节性反转的强度，是中国东南部夏季风（东南季风）的直接强度指标；
偏强年对应华南-东南沿海夏季水汽输送总体偏强、降水易偏多（后汛期尤其明显），但指数与区域降水的相关并非 1:1，<b>不能把它当作降水预报直接使用</b>。<br>
<b>阈值：</b>JJAS 标准化值 ≥ +0.5σ 记为偏强，≤ -0.5σ 记为偏弱，其余为正常。<br>
<b>预报因子：</b>起报前 6 个月 ONI 轨迹与 SOI 轨迹（NOAA CPC 逐月更新）+ 前一年 SCSMI（季风自身持续性）。<br>
<b>状态转移模型：</b>P(今夏状态 | 去夏状态, 起报时 ENSO 态)，条件计数 + Dirichlet 先验（强度 4×边际频率）。<br>
<b>历史相似型：</b>13 维特征滚动标准化欧氏距离 + 高斯核加权（Silverman 带宽，下限 0.35）；
候选年全部早于目标年，滚动标准化只使用候选年统计量，无未来泄漏。<br>
<b>收缩与组合：</b>起报月分层（同月样本）与全历史估计按 λ = n/(n+24) 经验贝叶斯收缩；
组合权重按样本外三分类 Brier 技能分分配，负技能剔除，双负等权。<br>
<b>区间：</b>90%% 区间来自 Dirichlet 后验蒙特卡洛（固定种子，1500 次）。<br>
<b>提前期约定：</b>起报月 m≤5 时目标为同年 JJAS（6-9 月），m≥6 时目标为次年 JJAS；提前期 = 距 6 月的月数。<br>
<b>数据源：</b>
<span class="tag">SCSMI <a href="http://lijianping.cn/dct/page/65578">lijianping.cn</a></span>
<span class="tag">ONI <a href="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt">NOAA CPC</a></span>
<span class="tag">SOI <a href="https://www.cpc.ncep.noaa.gov/data/indices/soi">NOAA CPC</a></span>
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
        esc(oni_s), esc(soi_s), esc(res["oni_state_desc"]),
        esc(("%+.2f" % res["prev_scsmi"]) if res["prev_scsmi"] is not None else "未更新"),
        esc(res["prev_state"] or "未知"),
        y_t, cards_html,
        res["w_trans"], res["w_analog"], MONTH_CN[m], fmt3(skt), fmt3(ska), fmt3(skb),
        res["n_verify"] if res["n_verify"] is not None else 0,
        analog_table, pf_html,
        first_yr, last_yr, hist_html, heat_html, rel_html,
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

    # ---- 在线上下文 ----
    _check("在线上下文-10月", live_context(datetime.date(2026, 10, 23)) == (2027, 10), "")
    _check("在线上下文-3月", live_context(datetime.date(2027, 3, 15)) == (2027, 3), "")

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

    # ---- 完整报告 ----
    res, _ = run_forecast(allow_network=False, today=datetime.date(2026, 3, 15))
    report = build_report_html(res, compute_skills(oni, soi, scsmi), calib, scsmi, False)
    for kw in ("概率", "相似", "技能", "可靠性", "局限性", "SCSMI", "SVG".lower()):
        _check("报告-关键词[%s]" % kw, kw in report or kw in report.lower(), "")
    _check("报告-标签闭合", report.count("<html") == report.count("</html>") == 1 and
           report.count("<body>") == report.count("</body>") == 1, "")

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
        skills = compute_skills(oni, soi, scsmi)
        print_backtest(skills)
        return 0
    if args.command == "report":
        oni = load_oni(allow_network)
        soi = load_soi(allow_network)
        scsmi = load_scsmi(allow_network)
        res, _ = run_forecast(allow_network=allow_network)
        ys = target_years(scsmi)
        skills = compute_skills(oni, soi, scsmi, ys)
        calib = compute_calibration(oni, soi, scsmi, ys)
        report = build_report_html(res, skills, calib, scsmi, allow_network)
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