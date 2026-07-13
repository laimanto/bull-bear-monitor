"""
Bull-Bear Monitor — v1: compute + build.

Per-market rule set, validated in docs/EXPERIMENT_LOG.md (rounds 26-28
for the base combo shape; rounds 32/33/53 for RC2; round 66 for the
per-market VB calibration):

  Golden cross 50/200   sell when the 50-day average closes below the
                        200-day average; buy back when it closes above
  Volatility breaker    sell when 10-day realized vol reaches the
                        market's own trigger (see VB_LEVELS); buy back
                        once it settles back to that market's own calm level
  Recovery override     up-day volume share of the trailing 12 sessions
  (RC2)                 >= 60% AND vol <= 20% AND price within 15% of
                        its 200-day average: buy back even while alarms
                        are on

VB is calibrated per market (round 66): QQQ/SPY/FTSE keep the original
45%/18% (a percentile-pair search found raw levels beat any percentile-
matched pair for this low-volatility cluster); HSI/Nikkei 225/HSCEI —
the higher-volatility cluster — are percentile-matched to QQQ's own
45%/18% position (its p92/p50) on each market's own vol10 history.

Plus the QQQ MAXIMIZER: the volatility breaker 45/18 alone (no trend
filter) — QQQ-specific aggressive variant (rounds 9/21/24).

Markets: QQQ, SPY, ^HSI, ^HSCE (HSCEI), ^N225 (Nikkei 225), ^FTSE.
Financial-crisis windows are auto-detected per market: every closing
peak -> trough decline of 20%+ (episode ends when the market rallies
25% off the trough or regains the peak), labeled from a date map.

Reads data/<symbol>.csv (run update_data.py first for fresh closes).
Writes results/monitor_results.json and dashboard/Bull_Bear_Monitor.html.
Run from scripts/.
"""

import json
import math
import os

import pandas as pd

import examine_cutoffs as ec
import leading_pairs as lp
import market_regime as mr
import regime_dashboard_data as rd
import tune_regime as tr

GC = (50, 200)
# per-market VB calibration (round 66): QQQ/SPY/FTSE raw 45/18 (own
# percentile pair already, and beats any re-matched pair for this
# cluster); HSI/N225/HSCEI percentile-matched to QQQ's own p92/p50.
VB_LEVELS = {
    "QQQ": (0.45, 0.18), "QQQMAX": (0.45, 0.18),
    "SPY": (0.45, 0.18), "FTSE": (0.45, 0.18),
    "HSI": (0.375, 0.184), "N225": (0.346, 0.186), "HSCEI": (0.505, 0.230),
}
# RC2 (rounds 32/33/53): replaces the older 10/12-up-days Recovery rule
RC2_SHARE, RC2_WIN, RC2_CALM, RC2_P200_FLOOR = 0.60, 12, 0.20, -0.15


def pct1(x):
    s = f"{x * 100:.1f}"
    return s[:-2] if s.endswith(".0") else s


# VB first — it is the dominant alarm; GC and RC complete the combo
def rules_for(vb_trig, vb_calm):
    return [
        {"alarm": f"VB — Volatility breaker {pct1(vb_trig)}/{pct1(vb_calm)}",
         "detects": "Fast, violent crashes",
         "sell": f"10-day volatility rises to {pct1(vb_trig)}% annualized",
         "buy": f"10-day volatility settles back below {pct1(vb_calm)}% annualized — the market has genuinely calmed"},
        {"alarm": "GC — Golden cross 50/200", "detects": "Slow, grinding bear markets",
         "sell": "The 50-day average closes below the 200-day average (death cross)",
         "buy": "The 50-day average closes back above the 200-day average (golden cross)"},
        {"alarm": "RC — Recovery override (RC2)", "detects": "A real rebound taking hold",
         "sell": "Never sells — a buy-side override only",
         "buy": (f"Up-day volume share of the last {RC2_WIN} sessions' total volume is >= "
                 f"{pct1(RC2_SHARE)}% AND 10-day volatility is <= {pct1(RC2_CALM)}% AND price is within "
                 f"{pct1(-RC2_P200_FLOOR)}% of its 200-day average — buys back even while the alarms above are still on")},
    ]


def rc2_flags(d, p):
    """RC2 (rounds 32/33/53): up-day volume share of the trailing RC2_WIN
    sessions >= RC2_SHARE, AND 10-day vol <= RC2_CALM, AND price within
    -RC2_P200_FLOOR of its 200-day average. Returns (flags, up_share_series)."""
    c = pd.Series(d["close"])
    v = pd.Series(d["volume"])
    up = c.pct_change() > 0
    us = (v.where(up, 0.0)).rolling(RC2_WIN).sum() / v.rolling(RC2_WIN).sum()
    vol10, s200, cl = p["vol10"], p["s200"], p["cl"]
    out = []
    for i in range(len(cl)):
        u, vv, s = us.iloc[i], vol10[i], s200[i]
        ok = (not math.isnan(u) and u >= RC2_SHARE
              and not math.isnan(vv) and vv <= RC2_CALM
              and not math.isnan(s) and (cl[i] / s - 1) >= RC2_P200_FLOOR)
        out.append(ok)
    return out, us


#           key       symbol    tab label      full name                        currency
MARKETS = [("QQQ",    "QQQ",    "QQQ",         "Invesco QQQ (Nasdaq-100)",      "$"),
           ("QQQMAX", "QQQ",    "QQQ Maximizer", "Invesco QQQ (Nasdaq-100)",    "$"),
           ("SPY",    "SPY",    "SPY",         "SPDR S&P 500",                  "$"),
           ("HSI",    "^HSI",   "HSI",         "Hang Seng Index",               ""),
           ("HSCEI",  "^HSCE",  "HSCEI",       "Hang Seng China Enterprises",   ""),
           ("N225",   "^N225",  "Nikkei 225",  "Nikkei 225",                    ""),
           ("FTSE",   "^FTSE",  "FTSE 100",    "FTSE 100",                      "")]

# (name, first trough date, last trough date, restricted-to-keys or None)
CRISIS_LABELS = [
    ("Asian Financial Crisis", "1997-01-01", "1999-03-31", None),
    ("Dot-com bust", "2000-01-01", "2003-06-30", None),
    ("Global Financial Crisis", "2007-06-01", "2009-06-30", None),
    ("2011 Tōhoku earthquake / euro crisis", "2010-06-01", "2012-06-30", {"N225"}),
    ("Euro debt crisis", "2010-01-01", "2012-12-31", None),
    ("2015-16 global selloff", "2015-01-01", "2016-12-31", None),
    ("Q4 2018 selloff", "2018-09-01", "2019-01-31", None),
    ("COVID crash", "2020-01-01", "2020-06-30", None),
    ("China property / regulation bear", "2021-06-01", "2024-12-31", {"HSI", "HSCEI"}),
    ("2022 inflation bear", "2021-10-01", "2023-06-30", None),
    ("2025 tariff crash", "2025-01-01", "2025-06-30", None),
]


def detect_crises(key, cl, dates):
    """Per-market bear windows: closing peak -> trough with a drop >= 20%;
    a raw episode ends when the close rallies 25% off the trough or regains
    the peak. Raw legs separated by only a few months of rebound (and no
    full recovery of the earlier peak) are MERGED into one window — the
    overall peak and trough of the whole crisis."""
    n = len(cl)
    eps = []
    peak, peak_i, in_bear, tro, tro_i = cl[0], 0, False, 0.0, 0
    for i in range(1, n):
        c = cl[i]
        if not in_bear:
            if c >= peak:
                peak, peak_i = c, i
            elif c <= peak * 0.80:
                in_bear, tro, tro_i = True, c, i
        else:
            if c < tro:
                tro, tro_i = c, i
            if c >= tro * 1.25 or c >= peak:
                eps.append((peak_i, tro_i))
                in_bear, peak, peak_i = False, c, i
    if in_bear:
        eps.append((peak_i, tro_i))

    # merge legs: next leg's peak within ~8 months of the previous trough
    # and below the previous peak = a rebound inside the same crisis
    merged = []
    for pi, ti in eps:
        if merged:
            mpi, mti = merged[-1]
            gap = days_gap(dates[mti], dates[pi])
            if gap <= 240 and cl[pi] < cl[mpi]:
                if cl[ti] < cl[mti]:
                    merged[-1] = (mpi, ti)
                continue
        merged.append((pi, ti))
    eps = merged

    out = []
    for pi, ti in eps:
        t_date = dates[ti]
        name = next((nm for nm, lo, hi, keys in CRISIS_LABELS
                     if lo <= t_date <= hi and (keys is None or key in keys)),
                    f"Bear market of {t_date[:4]}")
        out.append({"name": name, "peak_i": pi, "trough_i": ti,
                    "start": dates[pi], "end": t_date,
                    "depth": round((cl[ti] / cl[pi] - 1) * 100, 1)})

    # a date-range label may match several separate declines on one market
    # (e.g. HSCEI fell three distinct times during the dot-com years):
    # the DEEPEST keeps the crisis name, the others become plain bears
    for name in {c["name"] for c in out}:
        same = [c for c in out if c["name"] == name]
        if len(same) > 1 and not name.startswith("Bear market of"):
            for c in sorted(same, key=lambda c: c["depth"])[1:]:
                c["name"] = f"Bear market of {c['end'][:4]}"
    return out


def pfmt(cur, v):
    """Price formatting: cents for ETFs, whole points for big indexes."""
    return f"{cur}{v:,.0f}" if v >= 1000 else f"{cur}{v:,.2f}"


def run_market(key, symbol, tab, full_name, cur):
    d = ec.fetch(symbol)
    p = tr.prep(d)
    cl, dates = p["cl"], p["dates"]
    n = len(cl)
    is_max = key == "QQQMAX"

    vb_trig, vb_calm = VB_LEVELS[key]
    gc_out = lp.flags_for(cl, *GC)
    vb_out = rd.vol_breaker(p, vb_trig, vb_calm)
    rc2, up_share = rc2_flags(d, p)
    off = [False] * n
    active_gc = off if is_max else gc_out
    active_rc2 = off if is_max else rc2
    comb_out = rd.combine_all(active_gc, vb_out, off, active_rc2)

    eq_bh = [c / cl[0] for c in cl]
    eq_gc, _, _ = mr.equity(cl, gc_out)
    eq_cc, _, _ = mr.equity(cl, comb_out)

    c_ser = pd.Series(d["close"])
    smaF = c_ser.rolling(GC[0]).mean().to_numpy()
    smaS = c_ser.rolling(GC[1]).mean().to_numpy()
    vol10 = p["vol10"]

    def vpct(v):
        return f"{v * 100:.0f}%"

    def gc_reading(i, below):
        rel = "below" if below else "above"
        return f"50-day avg {pfmt(cur, smaF[i])} {rel} 200-day avg {pfmt(cur, smaS[i])}"

    def out_reading(i, cause):
        parts = []
        if cause in ("gc", "both"):
            parts.append(gc_reading(i, True))
        if cause in ("cb", "both"):
            parts.append(f"10-day vol {vpct(vol10[i])} ≥ trigger {vpct(vb_trig)}")
        return " · ".join(parts)

    def in_fields(e):
        if active_rc2[e] and (active_gc[e] or vb_out[e]):
            p200 = p["cl"][e] / p["s200"][e] - 1
            return "momo", (f"up-vol share {vpct(up_share.iloc[e])} ≥ {vpct(RC2_SHARE)} · 10-day vol "
                            f"{vpct(vol10[e])} ≤ {vpct(RC2_CALM)} · price {p200*100:+.0f}% vs 200-day avg")
        v = vol10[e]
        parts = [f"10-day vol {vpct(v)} ≤ calm {vpct(vb_calm)}" if v <= vb_calm
                 else f"10-day vol {vpct(v)} (breaker not tripped)"]
        if not is_max:
            parts.append(gc_reading(e, False))
        return "clear", " · ".join(parts)

    def cause_at(i):
        new_gc = active_gc[i] and (i == 0 or not active_gc[i - 1])
        new_vb = vb_out[i] and (i == 0 or not vb_out[i - 1])
        return "both" if (new_gc and new_vb) else "gc" if new_gc else "cb"

    crises = detect_crises(key, cl, dates)

    episodes = rd.extract_episodes(comb_out, cl, dates, cause_at)
    cum = 1.0
    for e in episodes:
        e["out_reading"] = out_reading(e["exit_i"], e["cause"])
        e["in_cause"], e["in_reading"] = (None, None) if e["open"] else in_fields(e["end_i"])
        cum *= e["wealth_effect"]
        e["cum"] = round(cum, 4)
        to = e["reenter_date"] or dates[-1]
        # an exit up to ~1 month after the trough still belongs to that
        # crisis (detector lag), hence the 21-day grace on the window end
        e["during"] = ", ".join(
            c["name"] for c in crises
            if e["exit_date"] <= grace(c["end"]) and to >= c["start"])

    # tie-out: product of episode wealth effects == strategy wealth / B&H
    # wealth (effects are rounded to 4dp, hence the loose tolerance)
    edge = eq_cc[-1] / eq_bh[-1]
    assert abs(cum - edge) / edge < 0.01, f"{key}: cum {cum} vs edge {edge}"

    # golden/death cross events (chart markers; reference-only on QQQMAX)
    crosses = []
    for i in range(1, n):
        if math.isnan(smaS[i]) or math.isnan(smaS[i - 1]):
            continue
        if gc_out[i] and not gc_out[i - 1]:
            crosses.append({"i": i, "dir": "death"})
        elif gc_out[i - 1] and not gc_out[i]:
            crosses.append({"i": i, "dir": "golden"})

    for c in crises:
        i0, i1 = c["peak_i"], c["trough_i"]
        c["bh"] = c["depth"]
        c["gc"] = round((eq_gc[i1] / eq_gc[i0] - 1) * 100, 1)
        c["combo"] = round((eq_cc[i1] / eq_cc[i0] - 1) * 100, 1)

    def series(arr, nd=2):
        return [None if math.isnan(v) else round(float(v), nd) for v in arr]

    res = {
        "key": key, "symbol": symbol, "tab": tab, "full_name": full_name,
        "cur": cur, "is_max": is_max,
        "strategy_label": (f"Maximizer: VB (volatility breaker {pct1(vb_trig)}/{pct1(vb_calm)}) alone — no trend filter "
                           "(QQQ-specific aggressive variant)") if is_max else
                          f"Combo: VB (volatility breaker {pct1(vb_trig)}/{pct1(vb_calm)}) + GC (golden cross 50/200) + RC (Recovery override, RC2)",
        "rules": [rules_for(vb_trig, vb_calm)[0]] if is_max else rules_for(vb_trig, vb_calm),
        "first_date": dates[0], "last_date": dates[-1],
        "dates": dates,
        "close": [round(float(v), 2) for v in cl],
        "smaF": series(smaF), "smaS": series(smaS),
        "vol": series([v * 100 for v in vol10], 1),
        "out": [1 if o else 0 for o in comb_out],
        "crosses": crosses,
        "thresholds": {"gc_fast": GC[0], "gc_slow": GC[1],
                       "vol_trig": round(vb_trig * 100, 1), "vol_calm": round(vb_calm * 100, 1),
                       "rc2_share": RC2_SHARE * 100, "rc2_win": RC2_WIN, "rc2_calm": RC2_CALM * 100,
                       "rc2_p200_floor": -RC2_P200_FLOOR * 100},
        "episodes": episodes,
        "crises": crises,
        "now": {"vb": bool(vb_out[-1]), "gc_below": bool(gc_out[-1])},
        "summary": {
            "roi_bh": round((eq_bh[-1] - 1) * 100, 1),
            "roi_gc": round((eq_gc[-1] - 1) * 100, 1),
            "gc_exits": sum(1 for i in range(n)
                            if gc_out[i] and (i == 0 or not gc_out[i - 1])),
            "gc_exposure_pct": round(100 * (1 - sum(gc_out) / n), 1),
            "roi_strat": round((eq_cc[-1] - 1) * 100, 1),
            "mdd_bh": round(rd.max_dd(eq_bh), 1),
            "mdd_gc": round(rd.max_dd(eq_gc), 1),
            "mdd_strat": round(rd.max_dd(eq_cc), 1),
            "edge_vs_bh": round(edge, 3),
            "exits": len(episodes),
            "exposure_pct": round(100 * (1 - sum(comb_out) / n), 1),
            "avoided": sum(1 for e in episodes if not e["open"] and e["market_chg_pct"] < 0),
        },
    }
    for e in episodes:  # keep exit_i/end_i for chart-marker <-> log linking
        e["reenter_i"] = None if e["open"] else e["end_i"]
    return res


def grace(iso, n=21):
    return (pd.Timestamp(iso) + pd.Timedelta(days=n)).strftime("%Y-%m-%d")


def days_gap(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days


def main():
    out = {}
    for key, symbol, tab, full_name, cur in MARKETS:
        print(f"processing {key} ({symbol}) ...", flush=True)
        out[key] = run_market(key, symbol, tab, full_name, cur)
        s = out[key]["summary"]
        pos = "OUT" if out[key]["out"][-1] else "IN"
        print(f"  {pos} @ {out[key]['last_date']} | strat {s['roi_strat']}% "
              f"(bh {s['roi_bh']}%, gc {s['roi_gc']}%) | maxDD {s['mdd_strat']}% "
              f"(bh {s['mdd_bh']}%) | ×{s['edge_vs_bh']} | exits {s['exits']}")

    # the Maximizer tab also displays the QQQ combo row for comparison
    out["QQQMAX"]["combo_ref"] = {k: out["QQQ"]["summary"][k]
                                  for k in ("roi_strat", "mdd_strat", "edge_vs_bh",
                                            "exits", "exposure_pct")}
    # share the heavy per-day arrays with the QQQ tab (template re-links)
    for k in ("dates", "close", "smaF", "smaS", "vol"):
        out["QQQMAX"][k] = None
    out["QQQMAX"]["base"] = "QQQ"

    base = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base, "..", "results", "monitor_results.json")
    dash_dir = os.path.join(base, "..", "dashboard")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    data = open(results_path, encoding="utf-8").read()
    tpl = open(os.path.join(dash_dir, "monitor_template.html"), encoding="utf-8").read()
    body = tpl.replace("__DATA_JSON__", data)
    tor = pd.Timestamp.now(tz="America/Toronto").strftime("%b %d, %Y %H:%M")
    hk = pd.Timestamp.now(tz="Asia/Hong_Kong").strftime("%b %d, %Y %H:%M")
    body = body.replace("__BUILD_TS__", f"Built {tor} (Toronto)<br>{hk} (Hong Kong)")
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                  '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    dash_path = os.path.join(dash_dir, "Bull_Bear_Monitor.html")
    open(dash_path, "w", encoding="utf-8").write(standalone)
    print(f"wrote {dash_path} ({len(standalone):,} bytes)")


if __name__ == "__main__":
    main()
