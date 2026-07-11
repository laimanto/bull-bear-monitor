"""
Market-level regime strategies on SPY and QQQ, 1996/1999 -> now.

  bh    : buy & hold
  gc    : golden cross -- in while 50d SMA > 200d SMA (slow-bear detector)
  cb    : circuit breaker -- exit when close <= 0.90 * 20d rolling max;
          re-enter when close >= 1.10 * lowest close since exit (fast-crash)
  vol   : volatility trigger -- exit when 10d realized vol >= 40% annualized;
          re-enter when close >= 1.10 * lowest close since exit
  gc+cb : out of market whenever gc is bearish OR the circuit breaker is
          active; in only when both are calm

Signals on daily closes, executed at the same close. No costs/taxes.
"""

import math

import numpy as np
import pandas as pd

import examine_cutoffs as ec
from crash_windows import WINDOWS

TICKERS = ["SPY", "QQQ"]
STRATS = ["bh", "gc", "cb", "vol", "gc+cb"]

CB_DROP = 0.10     # exit: 10% below 20d high
CB_REENTER = 0.10  # re-enter: 10% above trough since exit
VOL_TRIG = 0.40    # exit: 10d realized vol >= 40% annualized


def signals(d):
    """Per-day alarm states; True = that detector says OUT."""
    c = pd.Series(d["close"])
    n = len(c)
    s50 = c.rolling(50).mean().to_numpy()
    s200 = c.rolling(200).mean().to_numpy()
    hi20 = c.rolling(20).max().to_numpy()
    vol10 = (c.pct_change().rolling(10).std() * math.sqrt(252)).to_numpy()
    cl = c.to_numpy()

    gc_out = [False] * n
    for i in range(n):
        gc_out[i] = (not math.isnan(s200[i])) and s50[i] < s200[i]

    def breaker(trigger):
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
                if cl[i] >= trough * (1 + CB_REENTER):
                    active = False
            out[i] = active
        return out

    cb_out = breaker(lambda i: not math.isnan(hi20[i]) and cl[i] <= hi20[i] * (1 - CB_DROP))
    vol_out = breaker(lambda i: not math.isnan(vol10[i]) and vol10[i] >= VOL_TRIG)
    return {"gc": gc_out, "cb": cb_out, "vol": vol_out,
            "gc+cb": [g or b for g, b in zip(gc_out, cb_out)]}


def equity(cl, out_flags):
    n = len(cl)
    eq = [1.0] * n
    mult = 1.0
    in_mkt = True
    entry = cl[0]
    switches = 0
    days_in = 0
    for i in range(n):
        if in_mkt and out_flags[i]:
            mult *= cl[i] / entry
            in_mkt = False
            switches += 1
        elif not in_mkt and not out_flags[i]:
            in_mkt = True
            entry = cl[i]
        if in_mkt:
            days_in += 1
        eq[i] = mult * (cl[i] / entry) if in_mkt else mult
    return eq, switches, days_in / n


def main():
    data, curves, stats = {}, {}, {}
    for t in TICKERS:
        print(f"fetching {t} ...", flush=True)
        data[t] = ec.fetch(t)
        sig = signals(data[t])
        cl = data[t]["close"]
        curves[t] = {"bh": (cl / cl[0]).tolist()}
        stats[t] = {"bh": (0, 1.0)}
        for s in STRATS[1:]:
            eq, sw, exp = equity(cl, sig[s])
            curves[t][s] = eq
            stats[t][s] = (sw, exp)

    for t in TICKERS:
        first = data[t]["dates"][0]
        print(f"\n########## {t} (data from {first}) ##########")

        print(f"{'':14}" + "".join(f"{s:>10}" for s in STRATS))
        row_roi = f"{'total ROI %':14}"
        row_dd = f"{'max DD %':14}"
        row_sw = f"{'exits':14}"
        row_ex = f"{'in-mkt %':14}"
        for s in STRATS:
            eq = curves[t][s]
            pk, mdd = eq[0], 0.0
            for v in eq:
                pk = max(pk, v)
                mdd = min(mdd, v / pk - 1)
            row_roi += f"{(eq[-1] - 1) * 100:>10,.0f}"
            row_dd += f"{mdd * 100:>10.0f}"
            row_sw += f"{stats[t][s][0]:>10}"
            row_ex += f"{stats[t][s][1] * 100:>10.0f}"
        print(row_roi); print(row_dd); print(row_sw); print(row_ex)

        print(f"\n{'crash P/L %':14}" + "".join(f"{s:>10}" for s in STRATS))
        for label, start, end in WINDOWS:
            i0 = ec.idx_at(data[t]["dates"], start)
            i1 = ec.idx_at(data[t]["dates"], end)
            if i0 < 0 or i1 <= i0:
                print(f"{label:14}  (no data)")
                continue
            row = f"{label[:13]:14}"
            for s in STRATS:
                eq = curves[t][s]
                row += f"{(eq[i1] / eq[i0] - 1) * 100:>10.1f}"
            print(row)

        print(f"\n{'ROI at cutoff':14}" + "".join(f"{s:>10}" for s in STRATS))
        for label, cutoff in ec.CUTOFFS:
            k = ec.idx_at(data[t]["dates"], cutoff)
            if k < 0:
                continue
            row = f"{label[:13]:14}"
            for s in STRATS:
                row += f"{(curves[t][s][k] - 1) * 100:>10,.0f}"
            print(row)


if __name__ == "__main__":
    main()
