"""
Leading MA pairs: find fast/slow SMA lengths near 50/200 whose crosses
systematically occur BEFORE the classic 50/200 cross (which algo traders
watch), and measure the lead time and the performance cost.

For each pair: match every classic 50/200 death/golden cross to the
nearest same-direction cross of the pair (within +/-45 trading days) and
record the lead (positive = pair crossed earlier) and the price at both
cross dates (sell advantage = pair-cross price vs classic-cross price).
"""

import math
import statistics

import pandas as pd

import examine_cutoffs as ec
import market_regime as mr
from crash_windows import WINDOWS, MODERATE

TICKERS = ["SPY", "QQQ"]
PAIRS = [
    (50, 200),  # classic baseline
    (45, 200), (40, 200), (35, 200), (30, 200),
    (50, 205), (50, 210), (50, 220),
    (45, 205), (45, 210), (40, 210), (35, 220),
]
EVAL_WINDOWS = WINDOWS + [MODERATE[-1]]


def flags_for(cl, f, s):
    c = pd.Series(cl)
    fa = c.rolling(f).mean().to_numpy()
    sl = c.rolling(s).mean().to_numpy()
    return [(not math.isnan(sl[i])) and fa[i] < sl[i] for i in range(len(cl))]


def edges(flags):
    ex, re = [], []
    for i in range(1, len(flags)):
        if flags[i] and not flags[i - 1]:
            ex.append(i)
        elif not flags[i] and flags[i - 1]:
            re.append(i)
    return ex, re


def lead_stats(base_edges, pair_edges, cl):
    """median lead in trading days (positive = pair earlier) and avg price adv %."""
    leads, advs = [], []
    for i0 in base_edges:
        cands = [i for i in pair_edges if abs(i - i0) <= 45]
        if not cands:
            continue
        ia = min(cands, key=lambda i: abs(i - i0))
        leads.append(i0 - ia)
        advs.append((cl[ia] / cl[i0] - 1) * 100)
    if not leads:
        return None, None, None
    return statistics.median(leads), min(leads), sum(advs) / len(advs)


def main():
    for t in TICKERS:
        print(f"\n########## {t} ##########")
        d = ec.fetch(t)
        cl, dates = d["close"], d["dates"]
        base = flags_for(cl, 50, 200)
        bex, bre = edges(base)

        print(f"{'pair':10}{'ROI %':>9}{'maxDD':>7}{'exits':>6}"
              f"{'dLead':>7}{'dMin':>6}{'sellAdv':>9}{'gLead':>7}{'buyAdv':>9}" +
              "".join(f"{lbl[:8]:>10}" for lbl, _, _ in EVAL_WINDOWS))
        for f, s in PAIRS:
            fl = flags_for(cl, f, s)
            eq, exits, _ = mr.equity(cl, fl)
            pk, mdd = eq[0], 0.0
            for v in eq:
                pk = max(pk, v)
                mdd = min(mdd, v / pk - 1)
            pex, pre = edges(fl)
            dl, dmin, dadv = lead_stats(bex, pex, cl)
            gl, _, gadv = lead_stats(bre, pre, cl)
            # buy advantage: positive = bought cheaper than classic cross day
            gadv = -gadv if gadv is not None else None
            row = (f"{f:>3}/{s:<6}{(eq[-1] - 1) * 100:>9,.0f}{mdd * 100:>7.0f}{exits:>6}"
                   f"{dl if dl is not None else '–':>7}{dmin if dmin is not None else '–':>6}"
                   f"{dadv:>+8.1f}%{gl if gl is not None else '–':>7}{gadv:>+8.1f}%")
            for _, ws, we in EVAL_WINDOWS:
                i0, i1 = ec.idx_at(dates, ws), ec.idx_at(dates, we)
                row += f"{(eq[i1] / eq[i0] - 1) * 100:>10.1f}" if 0 <= i0 < i1 else f"{'n/a':>10}"
            print(row)
        print("dLead/gLead = median lead in trading days before the classic 50/200 death/golden cross"
              "\ndMin = worst (smallest) death-cross lead · sellAdv/buyAdv = avg price vs classic cross day")


if __name__ == "__main__":
    main()
