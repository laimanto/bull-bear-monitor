"""
Data + build for the market-regime dashboard (SPY, QQQ) — tuned version.

  SPY: leading golden cross 45/195 + volatility breaker: exit when 10d
       realized vol >= 40% annualized, re-enter when it settles <= 20%
  QQQ: leading golden cross 45/200 + volatility breaker: exit >= 50%
       (QQQ's higher baseline volatility needs a higher panic threshold),
       re-enter <= 18%

  Both sides of the breaker are volatility levels (hysteresis) — no
  hardcoded rebound %. See spy_pairs_reentry.py for the sweep; the calm
  re-entry beat the old "+10% off trough" rule at every trigger level on
  SPY and roughly tied on QQQ with half the trades.

  The 45-day fast SMA makes the cross fire a few days before the classic
  50/200 watched by algo traders.

Cash earns nothing while out of the market.
Outputs regime_results.json and builds Regime_Dashboard_SPY_QQQ.html.
"""

import json
import math
import os

import examine_cutoffs as ec
import leading_pairs as lp
import market_regime as mr
import tune_regime as tr
from crash_windows import PER_TICKER

# panic-day sell (same for both tickers): the day drops >= PANIC_MOVE x the
# average absolute daily move of the prior PANIC_N days, on >= PANIC_VOL x
# the average volume of those days. Sell same close; back in when the other
# alarms are clear. (The mirror-image euphoria BUY day was tested in
# round6.py and added nothing, so only the sell side is used.)
PANIC_N, PANIC_MOVE, PANIC_VOL = 10, 2.0, 2.5
PANIC_RULE = {
    "alarm": "Crash", "detects": "Panic selling",
    "sell": ("The close drops at least 2× the average daily move of the last 10 days "
             "on at least 2.5× the average volume"),
    "buy": "No re-entry of its own — back in as soon as the other alarms are off",
}

# momentum build-up buy override (same for both tickers): recovery confidence
# builds over ~2 weeks, so buy back in — even while the alarms are still on —
# once 10 of the last 12 sessions closed up AND 10-day vol is back under 20%.
# Robust plateau: 10/11, 10/12, 11/13, 12/14 all work on both tickers; looser
# ratios (<=10/13) buy 2000-02 bear rallies and collapse (round7.py).
MOMO_K, MOMO_N, MOMO_CALM = 10, 12, 0.20
MOMO_RULE = {
    "alarm": "Recovery", "detects": "Momentum build-up (buy override)",
    "sell": "Never sells — a buy-side override only",
    "buy": ("At least 10 of the last 12 sessions closed up and 10-day volatility is "
            "back under 20% — buys back in even while the alarms above are still on"),
}

CONFIG = {
    "SPY": {
        # SPY keeps the trend filter: its 2000-02 bear was a LOW-volatility
        # grind that only the golden cross catches (crisis-only SPY: DD -55%).
        "gc": (45, 195), "use_gc": True, "vol": 0.40, "calm": 0.20,
        "label": "Golden cross 45/195 + volatility breaker (40% / 20%) + crash exit + recovery re-entry",
        "rules": [
            {"alarm": "Golden cross", "detects": "Slow bear markets",
             "sell": "45-day average closes below the 195-day average",
             "buy": "45-day average closes back above the 195-day average"},
            {"alarm": "Volatility breaker", "detects": "Fast crashes",
             "sell": "10-day volatility rises to 40% annualized",
             "buy": "10-day volatility settles back below 20% annualized"},
            PANIC_RULE,
            MOMO_RULE,
        ],
    },
    "QQQ": {
        # QQQ is CRISIS-ONLY: every QQQ disaster (incl. dot-com) is violent
        # enough for the volatility machinery; the golden cross only added
        # whipsaw and late re-entries (rounds 9-11). The 50/200 cross is
        # still computed and shown as the "trend only" reference line.
        "gc": (50, 200), "use_gc": False, "vol": 0.40, "calm": 0.18,
        "label": "Crisis-only: volatility breaker (40% / 18%) + crash exit + recovery re-entry — no trend filter",
        "rules": [
            {"alarm": "Volatility breaker", "detects": "Fast crashes",
             "sell": "10-day volatility rises to 40% annualized",
             "buy": "10-day volatility settles back below 18% annualized"},
            PANIC_RULE,
            MOMO_RULE,
        ],
    },
}

# out-of-sample validation (walk_forward.py / round11.py): at each split the
# volatility-breaker grid was re-tuned on pre-split data ONLY, then frozen
# and measured on the years after. Buy & hold shown for the same window.
WALK_FORWARD = {
    "SPY": [
        {"window": "2013 – 2026", "pick": "vol 45% / calm 22%",
         "roi": 546, "dd": -19, "bh_roi": 557, "bh_dd": -34},
        {"window": "2018 – 2026", "pick": "vol 45% / calm 22%",
         "roi": 244, "dd": -19, "bh_roi": 217, "bh_dd": -34},
    ],
    "QQQ": [
        {"window": "2013 – 2026", "pick": "vol 40% / calm 18%",
         "roi": 972, "dd": -22, "bh_roi": 1113, "bh_dd": -35},
        {"window": "2018 – 2026", "pick": "vol 40% / calm 18%",
         "roi": 316, "dd": -22, "bh_roi": 379, "bh_dd": -35},
    ],
}


def max_dd(eq):
    pk, mdd = eq[0], 0.0
    for v in eq:
        pk = max(pk, v)
        mdd = min(mdd, v / pk - 1)
    return mdd * 100


def vol_breaker(p, trig, calm):
    """Volatility hysteresis: OUT once 10d vol reaches `trig`, back IN only
    when it has settled below `calm` (both annualized)."""
    vol10 = p["vol10"]
    n = len(vol10)
    out = [False] * n
    active = False
    for i in range(n):
        if math.isnan(vol10[i]):
            out[i] = False
            continue
        if not active and vol10[i] >= trig:
            active = True
        elif active and vol10[i] <= calm:
            active = False
        out[i] = active
    return out


def extract_episodes(flags, cl, dates, cause_at):
    episodes = []
    n = len(cl)
    i = 0
    while i < n:
        if flags[i] and (i == 0 or not flags[i - 1]):
            j = i
            while j < n and flags[j]:
                j += 1
            reentered = j < n
            end = j if reentered else n - 1
            chg = (cl[end] / cl[i] - 1) * 100
            episodes.append({
                "exit_i": i,
                "end_i": end,
                "exit_date": dates[i],
                "reenter_date": dates[end] if reentered else None,
                "cause": cause_at(i),
                "market_chg_pct": round(chg, 2),
                # sidestepping a move m multiplies wealth (vs B&H) by 1/(1+m)
                "wealth_effect": round(1 / (1 + chg / 100), 4),
                "days_out": end - i,
                "open": not reentered,
            })
            i = j
        else:
            i += 1
    return episodes


def lead_vs_classic(cl, pair):
    """How many trading days before the classic 50/200 cross the leading
    pair's same-direction cross fires (matched within +/-45 trading days)."""
    base = lp.flags_for(cl, 50, 200)
    ours = lp.flags_for(cl, *pair)
    bex, bre = lp.edges(base)
    pex, pre = lp.edges(ours)

    def leads(b_edges, p_edges):
        out = []
        for i0 in b_edges:
            cands = [i for i in p_edges if abs(i - i0) <= 45]
            if cands:
                out.append(i0 - min(cands, key=lambda i: abs(i - i0)))
        return out

    sell, buy = leads(bex, pex), leads(bre, pre)
    eq, _, _ = mr.equity(cl, base)
    return {
        "classic_roi": round((eq[-1] - 1) * 100, 1),
        "classic_mdd": round(max_dd(eq), 1),
        "sell_lead_mean": round(sum(sell) / len(sell), 1) if sell else None,
        "sell_lead_n": len(sell),
        "buy_lead_mean": round(sum(buy) / len(buy), 1) if buy else None,
        "buy_lead_n": len(buy),
    }


def panic_days(d):
    import pandas as pd
    cl = pd.Series(d["close"])
    vol = pd.Series(d["volume"])
    ret = cl.pct_change()
    avgmove = ret.abs().rolling(PANIC_N).mean().shift(1)  # prior days only
    avgvol = vol.rolling(PANIC_N).mean().shift(1)
    return ((ret <= -PANIC_MOVE * avgmove) & (vol >= PANIC_VOL * avgvol)).fillna(False).tolist()


def momo_days(d, p):
    """Momentum build-up: >= MOMO_K of the last MOMO_N sessions closed up
    and 10-day vol has settled below MOMO_CALM."""
    import pandas as pd
    up = (pd.Series(d["close"]).pct_change() > 0)
    pattern = (up.rolling(MOMO_N).sum() >= MOMO_K).fillna(False).tolist()
    return [pt and not math.isnan(v) and v <= MOMO_CALM
            for pt, v in zip(pattern, p["vol10"])]


def combine_all(gc_out, vb_out, panic, momo):
    """Alarms give the baseline posture. A panic day forces a same-close
    sell even when the alarms are quiet; the momentum build-up buys back in
    even while the alarms are still on. After an override buy, only a NEW
    alarm edge or a panic day can force the next exit."""
    n = len(gc_out)
    out = [False] * n
    in_mkt = True
    for i in range(n):
        new_alarm = (gc_out[i] and (i == 0 or not gc_out[i - 1])) or \
                    (vb_out[i] and (i == 0 or not vb_out[i - 1]))
        if in_mkt:
            if new_alarm or panic[i]:
                in_mkt = False
        elif not (gc_out[i] or vb_out[i]) or momo[i]:
            in_mkt = True
        out[i] = not in_mkt
    return out


def run(ticker, cfg):
    import pandas as pd
    d = ec.fetch(ticker)
    p = tr.prep(d)
    cl, dates = p["cl"], p["dates"]
    n = len(cl)
    gc_out = lp.flags_for(cl, *cfg["gc"])
    # for QQQ (use_gc False) the cross is reference-only: shown as the
    # "trend only" line but not part of the traded strategy
    active_gc = gc_out if cfg["use_gc"] else [False] * n
    vol_out = vol_breaker(p, cfg["vol"], cfg["calm"])
    panic = panic_days(d)
    momo = momo_days(d, p)
    comb_out = combine_all(active_gc, vol_out, panic, momo)

    eq_bh = [c / cl[0] for c in cl]
    eq_gc, _, _ = mr.equity(cl, gc_out)
    eq_cc, _, _ = mr.equity(cl, comb_out)

    # daily indicator readings (for episode tables + the dashboard chart)
    fN, sN = cfg["gc"]
    c_ser = pd.Series(d["close"])
    smaF = c_ser.rolling(fN).mean().to_numpy()
    smaS = c_ser.rolling(sN).mean().to_numpy()
    vol10 = p["vol10"]
    ret = c_ser.pct_change()
    avgmove = ret.abs().rolling(PANIC_N).mean().shift(1).to_numpy()
    avgvol = pd.Series(d["volume"]).rolling(PANIC_N).mean().shift(1).to_numpy()
    ret = ret.to_numpy()
    up12 = (c_ser.pct_change() > 0).rolling(MOMO_N).sum().to_numpy()

    def vpct(v):
        return f"{v * 100:.0f}%"

    def gc_reading(i, below):
        rel = "below" if below else "above"
        return (f"{fN}-day avg {smaF[i]:,.2f} {rel} {sN}-day avg {smaS[i]:,.2f}")

    def out_reading(i, cause):
        parts = []
        if cause in ("gc", "both"):
            parts.append(gc_reading(i, True))
        if cause in ("cb", "both"):
            parts.append(f"10-day vol {vpct(vol10[i])} ≥ trigger {vpct(cfg['vol'])}")
        if cause == "pd":
            parts.append(f"{ret[i] * 100:+.1f}% day ({-ret[i] / avgmove[i]:.1f}× avg move) "
                         f"on {d['volume'][i] / avgvol[i]:.1f}× avg volume")
        return " · ".join(parts)

    def in_fields(e):
        """Which condition let the strategy back in, and its reading that day."""
        if momo[e] and (active_gc[e] or vol_out[e]):
            return "momo", (f"{int(up12[e])}/{MOMO_N} up days · 10-day vol "
                            f"{vpct(vol10[e])} ≤ {vpct(MOMO_CALM)}")
        v = vol10[e]
        parts = [f"10-day vol {vpct(v)} ≤ calm {vpct(cfg['calm'])}" if v <= cfg["calm"]
                 else f"10-day vol {vpct(v)} (breaker not tripped)"]
        if cfg["use_gc"]:
            parts.append(gc_reading(e, False))
        return "clear", " · ".join(parts)

    def cause_at(i):
        new_gc = active_gc[i] and (i == 0 or not active_gc[i - 1])
        new_vb = vol_out[i] and (i == 0 or not vol_out[i - 1])
        if new_gc and new_vb:
            return "both"
        if new_gc:
            return "gc"
        if new_vb:
            return "cb"  # cb = volatility breaker
        return "pd"      # panic day
    episodes = extract_episodes(comb_out, cl, dates, cause_at)
    for e in episodes:
        e["out_reading"] = out_reading(e["exit_i"], e["cause"])
        if e["open"]:
            e["in_cause"], e["in_reading"] = None, None
        else:
            e["in_cause"], e["in_reading"] = in_fields(e["end_i"])
        del e["exit_i"], e["end_i"]

    crash = []
    for name, sev, per in PER_TICKER:
        if ticker not in per:
            continue
        s, e = per[ticker]  # this ticker's own closing peak -> closing trough
        i0, i1 = ec.idx_at(dates, s), ec.idx_at(dates, e)
        if i0 < 0 or i1 <= i0:
            continue
        crash.append({
            "name": name, "sev": sev, "start": s, "end": e,
            "bh": round((eq_bh[i1] / eq_bh[i0] - 1) * 100, 1),
            "gc": round((eq_gc[i1] / eq_gc[i0] - 1) * 100, 1),
            "gccb": round((eq_cc[i1] / eq_cc[i0] - 1) * 100, 1),
        })

    def series(arr, nd=2):
        return [None if math.isnan(v) else round(float(v), nd) for v in arr]

    return {
        "ticker": ticker,
        "strategy_label": cfg["label"],
        "rules": cfg["rules"],
        "first_date": dates[0],
        "last_date": dates[-1],
        "dates": dates,
        "close": [round(float(v), 2) for v in cl],
        "smaF": series(smaF),
        "smaS": series(smaS),
        "vol": series([v * 100 for v in vol10], 1),
        "thresholds": {
            "gc_fast": fN, "gc_slow": sN,
            "vol_trig": cfg["vol"] * 100, "vol_calm": cfg["calm"] * 100,
            "momo_k": MOMO_K, "momo_n": MOMO_N, "momo_calm": MOMO_CALM * 100,
        },
        "out": [1 if o else 0 for o in comb_out],
        "episodes": episodes,
        "use_gc": cfg["use_gc"],
        "gc_pair": f"{cfg['gc'][0]}/{cfg['gc'][1]}" + ("" if cfg["use_gc"] else " (reference only)"),
        "lead": lead_vs_classic(cl, cfg["gc"]),
        "walkforward": WALK_FORWARD[ticker],
        "summary": {
            "roi_bh": round((eq_bh[-1] - 1) * 100, 1),
            "roi_gc": round((eq_gc[-1] - 1) * 100, 1),
            "roi_gccb": round((eq_cc[-1] - 1) * 100, 1),
            "mdd_bh": round(max_dd(eq_bh), 1),
            "mdd_gc": round(max_dd(eq_gc), 1),
            "mdd_gccb": round(max_dd(eq_cc), 1),
            # strategy wealth ÷ buy-and-hold wealth == product of episode effects
            "edge_vs_bh": round(eq_cc[-1] / eq_bh[-1], 3),
            "edge_gc": round(eq_gc[-1] / eq_bh[-1], 3),
            "exits": len(episodes),
            "exposure_pct": round(100 * (1 - sum(comb_out) / n), 1),
            "avoided": sum(1 for e in episodes if not e["open"] and e["market_chg_pct"] < 0),
            "crash": crash,
        },
    }


def main():
    out = {}
    for t, cfg in CONFIG.items():
        print(f"processing {t} ...", flush=True)
        out[t] = run(t, cfg)
        s = out[t]["summary"]
        print(f"  ROI gc+vol {s['roi_gccb']}% | gc {s['roi_gc']}% | bh {s['roi_bh']}% | "
              f"maxDD {s['mdd_gccb']}% (bh {s['mdd_bh']}%) | exits {s['exits']} | "
              f"avoided {s['avoided']}/{len(out[t]['episodes'])} | edge x{s['edge_vs_bh']}")
        for c in s["crash"]:
            print(f"    {c['name']}: bh {c['bh']}% | gc {c['gc']}% | gc+vol {c['gccb']}%")
    base = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base, "..", "results", "regime_results.json")
    dash_dir = os.path.join(base, "..", "dashboard")
    with open(results_path, "w") as f:
        json.dump(out, f)

    data = open(results_path, encoding="utf-8").read()
    tpl = open(os.path.join(dash_dir, "dashboard_regime_template.html"), encoding="utf-8").read()
    body = tpl.replace("__DATA_JSON__", data)
    import pandas as pd
    tor = pd.Timestamp.now(tz="America/Toronto").strftime("%b %d, %Y %H:%M")
    body = body.replace("__BUILD_TS__",
                        f"Data frozen at the {ec.FREEZE} close<br>Built {tor} (Toronto)")
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                  '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    dash_path = os.path.join(dash_dir, "Regime_Dashboard_SPY_QQQ.html")
    open(dash_path, "w", encoding="utf-8").write(standalone)
    print(f"wrote {dash_path} ({len(standalone):,} bytes)")


if __name__ == "__main__":
    main()
