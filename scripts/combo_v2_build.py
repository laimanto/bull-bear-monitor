"""
Combo Strategy v2 — all markets. Extends qqq_combo2_build.py's single-
market build to all 6 monitor markets (QQQ, SPY, HSI, HSCEI, Nikkei 225,
FTSE 100), applying rounds 62-66's conclusions:

  - Same 3-signal structure everywhere: VB (Volatility Breaker) + MA
    (50/200 golden/death cross) + VM (Volume Momentum recovery override,
    = RC2 in the research scripts). No separate VB-alone "Maximizer" --
    round 61 showed the full combo already beats VB-alone at every
    capital-split weight, so there is nothing left for a Maximizer sleeve
    to add.
  - VB's absolute level is calibrated per VOLATILITY CLUSTER, not one
    global QQQ number (round 66): markets split cleanly by median vol10
    into a volatile cluster (QQQ, HSI, N225, HSCEI, median 18-23%) and a
    stable cluster (SPY, FTSE, median 12.7-13.8%). The volatile cluster
    uses QQQ's own percentile position (45%=own p92, 18%=own p50)
    translated into each market's own vol10 distribution; the stable
    cluster uses the raw QQQ 45/18 unchanged (round 66: no percentile
    recipe beat that for SPY/FTSE).

Reads data/<ticker>.csv (already cached, no network calls). Writes
results/combo_v2_results.json and dashboard/Bull_Bear_Monitor_v2.html --
the latter is the live GitHub Pages dashboard (wired into the daily
workflow), superseding the older dashboard/Bull_Bear_Monitor.html.
Run from scripts/.
"""

import json
import math
import os

import numpy as np
import pandas as pd

import examine_cutoffs as ec
import leading_pairs as lp
import market_regime as mr
import regime_dashboard_data as rd
import tune_regime as tr
from monitor_build import detect_crises, pfmt, grace

STRATEGY_NAME = "Combo Strategy"
MA = (50, 200)
QQQ_TRIG, QQQ_CALM = 0.45, 0.18
VM_SHARE, VM_WIN, VM_CALM, VM_GUARD = 0.60, 12, 0.20, -0.15
REGIME_SPLIT = "2009-01-01"

# (key, yahoo symbol, tab label, full name, currency, volatile-cluster?)
MARKETS = [
    ("QQQ",   "QQQ",   "QQQ",         "Invesco QQQ (Nasdaq-100)",    "$", True),
    ("SPY",   "SPY",   "SPY",         "SPDR S&P 500",                 "$", False),
    ("HSI",   "^HSI",  "HSI",         "Hang Seng Index",              "",  True),
    ("HSCEI", "^HSCE", "HSCEI",       "Hang Seng China Enterprises",  "",  True),
    ("N225",  "^N225", "Nikkei 225",  "Nikkei 225",                   "",  True),
    ("FTSE",  "^FTSE", "FTSE 100",    "FTSE 100",                     "",  False),
]

RULES = [
    {"name": "VB — Volatility Breaker", "detects": "Fast, violent crashes",
     "desc": ("10-day realized volatility, annualized — the standard deviation of the last 10 "
              "daily price changes, scaled to a yearly figure. A direct, standard statistical "
              "measure of how turbulently the market has been trading: it spikes sharply in a "
              "real crash because large daily swings in either direction are the signature of "
              "panic-driven markets. LEVEL IS MARKET-SPECIFIC (docs/EXPERIMENT_LOG.md rounds "
              "62-66): QQQ's own trigger/calm (45%/18%) sit at QQQ's own 92nd/50th percentile "
              "of its historical volatility; markets whose typical volatility runs at or above "
              "QQQ's own (HSI, Nikkei 225, HSCEI) use that SAME percentile position translated "
              "into their own volatility history; markets whose typical volatility runs well "
              "below QQQ's (SPY, FTSE) keep the raw QQQ number unchanged — testing showed no "
              "percentile-based recalibration improves on it for those two."),
     "buy": "10-day volatility settles back below the market's calm level — the market has genuinely calmed",
     "sell": "10-day volatility rises to the market's trigger level"},
    {"name": "MA — Moving Average cross (50/200)", "detects": "Slow, grinding bear markets",
     "desc": ("Compares the 50-day and 200-day simple moving averages of price — the classic "
              "“golden cross / death cross” trend signal used across technical "
              "analysis. When the shorter average falls below the longer one, the market's "
              "medium-term trend has turned down; when it climbs back above, the trend has "
              "turned back up. Catches slow, grinding bear markets that don't necessarily show "
              "up as a volatility spike."),
     "buy": "The 50-day average closes back above the 200-day average (golden cross)",
     "sell": "The 50-day average closes below the 200-day average (death cross)"},
    {"name": "VM — Volume Momentum", "detects": "A real, volume-confirmed rebound taking hold",
     "desc": ("NOT a standard textbook indicator — developed and validated specifically for "
              "this project (docs/EXPERIMENT_LOG.md rounds 32-33, 48, 53), on QQQ, then "
              "transferred unmodified to every other market (round 62). Formula: (trading "
              "volume on up days) ÷ (total trading volume), summed over the last 12 sessions. "
              "A high value means recent buying activity is concentrated on up days — broad, "
              "volume-backed participation in a rally, the signature of a genuine recovery "
              "rather than a thin, low-conviction bounce."),
     "buy": ("At least 60% of the last 12 sessions' trading volume fell on UP days, AND "
             "10-day volatility is under 20%, AND the close is within 15% of the 200-day "
             "average — buys back even while VB/MA are still on"),
     "sell": "Never sells — a buy-side override only"},
]


def vb_flags(vol10, trig, calm):
    n = len(vol10)
    out = [False] * n
    active = False
    for i in range(n):
        v = vol10[i]
        if not active and not math.isnan(v) and v >= trig:
            active = True
        elif active and not math.isnan(v) and v <= calm:
            active = False
        out[i] = active
    return out


def vb_levels_for(key, vol10_this, vol10_qqq, volatile):
    """Round 66: volatile-cluster markets get QQQ's own percentile
    position (45%/18% = QQQ's own p92/p50) translated into their own
    vol10 distribution; stable-cluster markets (and QQQ itself) keep the
    raw QQQ number unchanged."""
    if key == "QQQ" or not volatile:
        return QQQ_TRIG, QQQ_CALM
    vq = vol10_qqq[~np.isnan(vol10_qqq)]
    trig_pctile = (vq < QQQ_TRIG).mean() * 100
    calm_pctile = (vq < QQQ_CALM).mean() * 100
    v = vol10_this[~np.isnan(vol10_this)]
    return float(np.percentile(v, trig_pctile)), float(np.percentile(v, calm_pctile))


def vm_flags(d, p):
    c = pd.Series(d["close"])
    v = pd.Series(d["volume"])
    up = c.pct_change() > 0
    ushare = ((v.where(up, 0.0)).rolling(VM_WIN).sum()
              / v.rolling(VM_WIN).sum()).to_numpy()
    vol10 = p["vol10"]
    s200 = p["s200"]
    p200 = c.to_numpy() / s200 - 1
    n = len(c)
    flags = [False] * n
    for i in range(n):
        if math.isnan(ushare[i]) or math.isnan(vol10[i]) or math.isnan(p200[i]):
            continue
        flags[i] = (ushare[i] >= VM_SHARE and vol10[i] <= VM_CALM
                    and p200[i] >= VM_GUARD)
    return flags, ushare, p200


def era_stats(cl, dates, flags, lo, hi, force_in):
    i0 = next(i for i, dt in enumerate(dates) if dt >= lo)
    i1 = next((i for i, dt in enumerate(dates) if dt >= hi), len(dates) - 1)
    wf = list(flags[i0:i1 + 1])
    if force_in:
        wf[0] = False
    eq, _, _ = mr.equity(cl[i0:i1 + 1], wf)
    return {"roi": round((eq[-1] - 1) * 100, 1), "dd": round(rd.max_dd(eq), 1),
            "cap": round(eq[-1] / (cl[i1] / cl[i0]), 2)}


def run_one(key, symbol, tab, full_name, cur, volatile, vol10_qqq):
    d = ec.fetch(symbol)
    p = tr.prep(d)
    cl, dates = p["cl"], p["dates"]
    n = len(cl)
    vol10 = p["vol10"]

    vb_trig, vb_calm = vb_levels_for(key, vol10, vol10_qqq, volatile)

    ma_out = lp.flags_for(cl, *MA)
    vb_out = vb_flags(vol10, vb_trig, vb_calm)
    off = [False] * n

    momo, ushare, p200 = vm_flags(d, p)
    comb = rd.combine_all(ma_out, vb_out, off, momo)
    eq_strat, _, _ = mr.equity(cl, comb)
    eq_ma, _, _ = mr.equity(cl, ma_out)
    eq_bh = [c / cl[0] for c in cl]

    c_ser = pd.Series(d["close"])
    smaF = c_ser.rolling(MA[0]).mean().to_numpy()
    smaS = c_ser.rolling(MA[1]).mean().to_numpy()

    def vpct(v):
        return f"{v * 100:.0f}%"

    def ma_reading(i, below):
        rel = "below" if below else "above"
        return f"50-day avg {pfmt(cur, smaF[i])} {rel} 200-day avg {pfmt(cur, smaS[i])}"

    def out_reading(i, cause):
        parts = []
        if cause in ("gc", "both"):
            parts.append(ma_reading(i, True))
        if cause in ("cb", "both"):
            parts.append(f"10-day vol {vpct(vol10[i])} ≥ trigger {vpct(vb_trig)}")
        return " · ".join(parts)

    def in_fields(e):
        if momo[e] and (ma_out[e] or vb_out[e]):
            return "momo", (f"{ushare[e] * 100:.0f}% of last 12d volume on up days ≥ 60% · "
                            f"10-day vol {vpct(vol10[e])} ≤ 20% · "
                            f"price {p200[e] * 100:+.0f}% vs 200-day avg (≥ −15% required)")
        v = vol10[e]
        parts = [f"10-day vol {vpct(v)} ≤ calm {vpct(vb_calm)}" if v <= vb_calm
                 else f"10-day vol {vpct(v)} (breaker not tripped)",
                 ma_reading(e, False)]
        return "clear", " · ".join(parts)

    def cause_at(i):
        new_ma = ma_out[i] and (i == 0 or not ma_out[i - 1])
        new_vb = vb_out[i] and (i == 0 or not vb_out[i - 1])
        return "both" if (new_ma and new_vb) else "gc" if new_ma else "cb"

    crises = detect_crises(key, cl, dates)
    episodes = rd.extract_episodes(comb, cl, dates, cause_at)
    cum = 1.0
    for e in episodes:
        e["out_reading"] = out_reading(e["exit_i"], e["cause"])
        e["in_cause"], e["in_reading"] = (None, None) if e["open"] else in_fields(e["end_i"])
        cum *= e["wealth_effect"]
        e["cum"] = round(cum, 4)
        to = e["reenter_date"] or dates[-1]
        e["during"] = ", ".join(
            c["name"] for c in crises
            if e["exit_date"] <= grace(c["end"]) and to >= c["start"])
        e["reenter_i"] = None if e["open"] else e["end_i"]

    edge = eq_strat[-1] / eq_bh[-1]
    assert abs(cum - edge) / edge < 0.01, f"{key}: cum {cum} vs edge {edge}"

    crosses = []
    for i in range(1, n):
        if math.isnan(smaS[i]) or math.isnan(smaS[i - 1]):
            continue
        if ma_out[i] and not ma_out[i - 1]:
            crosses.append({"i": i, "dir": "death"})
        elif ma_out[i - 1] and not ma_out[i]:
            crosses.append({"i": i, "dir": "golden"})

    for c in crises:
        i0, i1 = c["peak_i"], c["trough_i"]
        c["bh"] = c["depth"]
        c["gc"] = round((eq_ma[i1] / eq_ma[i0] - 1) * 100, 1)
        c["combo"] = round((eq_strat[i1] / eq_strat[i0] - 1) * 100, 1)

    def series(arr, nd=2):
        return [None if math.isnan(v) else round(float(v), nd) for v in arr]

    ma_exits = sum(1 for i in range(n) if ma_out[i] and (i == 0 or not ma_out[i - 1]))

    eras = {}
    for label, lo, hi in [("pre2009", dates[0], "2008-12-31"),
                          ("post2009", REGIME_SPLIT, dates[-1])]:
        eras[label] = {
            "bh": era_stats(cl, dates, off, lo, hi, force_in=False),
            "ma": era_stats(cl, dates, ma_out, lo, hi, force_in=True),
            "combo2": era_stats(cl, dates, comb, lo, hi, force_in=True),
        }

    return {
        "key": key, "symbol": symbol, "tab": tab,
        "full_name": full_name, "cur": cur, "is_max": False,
        "strategy_name": STRATEGY_NAME,
        "strategy_label": f"{STRATEGY_NAME}: VB (Volatility Breaker) + MA (50/200 cross) + VM (Volume Momentum)",
        "rules": RULES,
        "first_date": dates[0], "last_date": dates[-1],
        "dates": dates,
        "close": [round(float(v), 2) for v in cl],
        "open": series(d["open"]), "high": series(d["high"]), "low": series(d["low"]),
        "volume": [None if math.isnan(v) else round(float(v)) for v in d["volume"]],
        "smaF": series(smaF), "smaS": series(smaS),
        "vol": series([v * 100 for v in vol10], 1),
        "vm": series([v * 100 for v in ushare], 1),
        "out": [1 if o else 0 for o in comb],
        "crosses": crosses,
        "thresholds": {"ma_fast": MA[0], "ma_slow": MA[1],
                       "vol_trig": round(vb_trig * 100, 1), "vol_calm": round(vb_calm * 100, 1),
                       "vm_share": VM_SHARE * 100, "vm_win": VM_WIN,
                       "vm_vol_gate": VM_CALM * 100, "vm_guard": VM_GUARD * 100},
        "episodes": episodes,
        "crises": crises,
        "now": {"vb": bool(vb_out[-1]), "ma_below": bool(ma_out[-1]),
                "vm_value": round(float(ushare[-1]) * 100, 1) if not math.isnan(ushare[-1]) else None,
                "p200_now": round(float(p200[-1]) * 100, 1) if not math.isnan(p200[-1]) else None},
        "summary": {
            "roi_bh": round((eq_bh[-1] - 1) * 100, 1),
            "roi_ma": round((eq_ma[-1] - 1) * 100, 1),
            "ma_exits": ma_exits,
            "ma_exposure_pct": round(100 * (1 - sum(ma_out) / n), 1),
            "roi_strat": round((eq_strat[-1] - 1) * 100, 1),
            "mdd_bh": round(rd.max_dd(eq_bh), 1),
            "mdd_ma": round(rd.max_dd(eq_ma), 1),
            "mdd_strat": round(rd.max_dd(eq_strat), 1),
            "edge_vs_bh": round(edge, 3),
            "exits": len(episodes),
            "exposure_pct": round(100 * (1 - sum(comb) / n), 1),
            "avoided": sum(1 for e in episodes if not e["open"] and e["market_chg_pct"] < 0),
            "eras": eras,
        },
    }


def main():
    dq = ec.fetch("QQQ")
    pq = tr.prep(dq)
    vol10_qqq = pq["vol10"]

    out = {}
    for key, symbol, tab, full_name, cur, volatile in MARKETS:
        print(f"processing {key} ({symbol}) ...", flush=True)
        res = run_one(key, symbol, tab, full_name, cur, volatile, vol10_qqq)
        out[key] = res
        s = res["summary"]
        pos = "OUT" if res["out"][-1] else "IN"
        th = res["thresholds"]
        print(f"  VB {th['vol_trig']}%/{th['vol_calm']}% | {pos} @ {res['last_date']} | "
              f"{STRATEGY_NAME} {s['roi_strat']}% (bh {s['roi_bh']}%) | maxDD {s['mdd_strat']}% | "
              f"x{s['edge_vs_bh']} | exits {s['exits']}")
        e = s["eras"]
        print(f"    post-2009: bh {e['post2009']['bh']['roi']}% | "
              f"{STRATEGY_NAME} {e['post2009']['combo2']['roi']}% (cap {e['post2009']['combo2']['cap']})")

    base = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base, "..", "results", "combo_v2_results.json")
    dash_dir = os.path.join(base, "..", "dashboard")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    data = open(results_path, encoding="utf-8").read()
    tpl = open(os.path.join(dash_dir, "combo_v2_template.html"), encoding="utf-8").read()
    body = tpl.replace("__DATA_JSON__", data)
    tor = pd.Timestamp.now(tz="America/Toronto").strftime("%b %d, %Y %H:%M")
    hk = pd.Timestamp.now(tz="Asia/Hong_Kong").strftime("%b %d, %Y %H:%M")
    body = body.replace("__BUILD_TS__", f"Built {tor} (Toronto)<br>{hk} (Hong Kong)")
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                  '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    dash_path = os.path.join(dash_dir, "Bull_Bear_Monitor_v2.html")
    open(dash_path, "w", encoding="utf-8").write(standalone)
    print(f"wrote {dash_path} ({len(standalone):,} bytes)")


if __name__ == "__main__":
    main()
