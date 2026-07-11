"""
Threshold sweep for the regime strategy on SPY and QQQ (cash earns 0% here;
T-bill yield adds roughly the same bonus to every variant).

Detectors combined with the golden cross (out if ANY alarm active):
  cbD   : drawdown breaker -- close <= (1-D) * 20d high; re-enter +D off trough
  volV  : 10d realized vol >= V% annualized; re-enter +10% off trough
  velN/D: velocity -- close fell D% within N days; re-enter +10% off trough

Reports total ROI, max DD, exits, and P/L inside each crash window.
Also prints the exit dates around COVID for QQQ configs (speed check).
"""

import math

import pandas as pd

import examine_cutoffs as ec
import market_regime as mr
from crash_windows import WINDOWS

TICKERS = ["SPY", "QQQ"]


def prep(d):
    c = pd.Series(d["close"])
    return {
        "cl": c.to_numpy(),
        "dates": d["dates"],
        "hi20": c.rolling(20).max().to_numpy(),
        "s50": c.rolling(50).mean().to_numpy(),
        "s200": c.rolling(200).mean().to_numpy(),
        "vol10": (c.pct_change().rolling(10).std() * math.sqrt(252)).to_numpy(),
    }


def breaker(p, trigger, re_frac):
    cl = p["cl"]
    n = len(cl)
    out = [False] * n
    active = False
    trough = None
    for i in range(n):
        if not active:
            if trigger(i):
                active = True
                trough = cl[i]
        else:
            trough = min(trough, cl[i])
            if cl[i] >= trough * (1 + re_frac):
                active = False
        out[i] = active
    return out


def gc_flags(p):
    n = len(p["cl"])
    return [(not math.isnan(p["s200"][i])) and p["s50"][i] < p["s200"][i] for i in range(n)]


def cb_flags(p, drop):
    return breaker(p, lambda i: not math.isnan(p["hi20"][i]) and
                   p["cl"][i] <= p["hi20"][i] * (1 - drop), drop)


def vol_flags(p, trig):
    return breaker(p, lambda i: not math.isnan(p["vol10"][i]) and
                   p["vol10"][i] >= trig, 0.10)


def vel_flags(p, ndays, drop):
    cl = p["cl"]
    return breaker(p, lambda i: i >= ndays and cl[i] / cl[i - ndays] - 1 <= -drop, 0.10)


def combine(*flag_lists):
    return [any(f) for f in zip(*flag_lists)]


def evaluate(p, flags):
    eq, exits, _ = mr.equity(p["cl"], flags)
    pk, mdd = eq[0], 0.0
    for v in eq:
        pk = max(pk, v)
        mdd = min(mdd, v / pk - 1)
    crash = []
    for _, s, e in WINDOWS:
        i0, i1 = ec.idx_at(p["dates"], s), ec.idx_at(p["dates"], e)
        crash.append((eq[i1] / eq[i0] - 1) * 100 if 0 <= i0 < i1 else None)
    return eq, exits, mdd * 100, crash


def main():
    for t in TICKERS:
        print(f"\n########## {t} ##########")
        p = prep(ec.fetch(t))
        gc = gc_flags(p)

        configs = [("gc only", gc)]
        for dr in [0.08, 0.10, 0.12, 0.15, 0.20]:
            configs.append((f"gc+cb{int(dr * 100)}", combine(gc, cb_flags(p, dr))))
        for v in [0.30, 0.35, 0.40, 0.50]:
            configs.append((f"gc+vol{int(v * 100)}", combine(gc, vol_flags(p, v))))
        for dr in [0.10, 0.12, 0.15]:
            for v in [0.35, 0.40]:
                configs.append((f"gc+cb{int(dr * 100)}+vol{int(v * 100)}",
                                combine(gc, cb_flags(p, dr), vol_flags(p, v))))
        configs.append(("gc+vel5d10", combine(gc, vel_flags(p, 5, 0.10))))
        configs.append(("gc+cb15+vel5d10", combine(gc, cb_flags(p, 0.15), vel_flags(p, 5, 0.10))))

        print(f"{'config':20}{'ROI %':>10}{'maxDD':>7}{'exits':>6}" +
              "".join(f"{lbl[:9]:>11}" for lbl, _, _ in WINDOWS))
        for name, flags in configs:
            eq, exits, mdd, crash = evaluate(p, flags)
            row = f"{name:20}{(eq[-1] - 1) * 100:>10,.0f}{mdd:>7.0f}{exits:>6}"
            for c in crash:
                row += f"{c:>11.1f}" if c is not None else f"{'n/a':>11}"
            print(row)

        if t == "QQQ":
            print("\nCOVID speed check (QQQ, 2020): first out-day of each detector")
            dets = {"cb15": cb_flags(p, 0.15), "cb12": cb_flags(p, 0.12),
                    "cb10": cb_flags(p, 0.10), "vol35": vol_flags(p, 0.35),
                    "vol40": vol_flags(p, 0.40), "vel5d10": vel_flags(p, 5, 0.10)}
            for nm, fl in dets.items():
                for i, dt in enumerate(p["dates"]):
                    if "2020-02-19" <= dt <= "2020-04-30" and fl[i]:
                        drop = (p["cl"][i] / max(p["cl"][j] for j, dd in enumerate(p["dates"])
                                                 if "2020-02-01" <= dd <= "2020-02-19") - 1) * 100
                        print(f"  {nm:8} exits {dt}  ({drop:+.1f}% from Feb 19 peak)")
                        break


if __name__ == "__main__":
    main()
