"""
How much did each strategy LOSE during each market disaster?

For each crash window (market peak -> trough), report the change in each
strategy's equity across the window: eq[trough] / eq[peak] - 1.
Equity curves are simulated from 1996 so the position state at the start of
each window is realistic. Strategies as in examine_cutoffs.py.
"""

import examine_cutoffs as ec

WINDOWS = [
    ("2000 dot-com crash", "2000-03-24", "2002-10-09"),
    ("2008 global financial crisis", "2007-10-09", "2009-03-09"),
    ("2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 rate-hike bear market", "2022-01-03", "2022-10-12"),
]

# moderate corrections (peak -> trough), used by the regime dashboard
MODERATE = [
    ("1998 Russia default crisis", "1998-07-17", "1998-08-31"),
    ("2011 debt-ceiling crisis", "2011-04-29", "2011-10-03"),
    ("2015 China slowdown scare", "2015-05-21", "2016-02-11"),
    ("2018 trade-war selloff", "2018-09-20", "2018-12-24"),
    ("2025 tariff crash", "2025-02-19", "2025-04-08"),
]

# per-ticker windows: each ticker's own closing peak -> its own closing
# trough (audited against the data by verify_crashes.py; on closes the
# 1998 SPY bottom is Aug 31 -- the Oct 8 low was intraday only).
# WINDOWS/MODERATE above stay SPY-referenced for the older sweep scripts.
PER_TICKER = [
    # name, severity, {ticker: (peak, trough)}
    ("2000 dot-com crash", "major",
     {"SPY": ("2000-03-24", "2002-10-09"), "QQQ": ("2000-03-27", "2002-10-09")}),
    ("2008 global financial crisis", "major",
     {"SPY": ("2007-10-09", "2009-03-09"), "QQQ": ("2007-10-23", "2008-11-20")}),
    ("2020 COVID crash", "major",
     {"SPY": ("2020-02-19", "2020-03-23"), "QQQ": ("2020-02-19", "2020-03-16")}),
    ("2022 rate-hike bear market", "major",
     {"SPY": ("2022-01-03", "2022-10-12"), "QQQ": ("2021-12-27", "2022-11-03")}),
    ("1998 Russia default crisis", "correction",
     {"SPY": ("1998-07-17", "1998-08-31")}),
    ("2011 debt-ceiling crisis", "correction",
     {"SPY": ("2011-04-29", "2011-10-03"), "QQQ": ("2011-04-27", "2011-08-19")}),
    ("2015 China slowdown scare", "correction",
     {"SPY": ("2015-05-21", "2016-02-11"), "QQQ": ("2015-05-27", "2016-02-09")}),
    ("2018 trade-war selloff", "correction",
     {"SPY": ("2018-09-20", "2018-12-24"), "QQQ": ("2018-08-29", "2018-12-24")}),
    ("2025 tariff crash", "correction",
     {"SPY": ("2025-02-19", "2025-04-08"), "QQQ": ("2025-02-19", "2025-04-08")}),
]


def main():
    data, curves = {}, {}
    for t in ec.TICKERS:
        print(f"fetching {t} ...", flush=True)
        data[t] = ec.fetch(t)
        curves[t] = {s: ec.equity_curve(data[t], s) for s in ec.STRATS}

    for label, start, end in WINDOWS:
        print(f"\n=== {label}: strategy P/L from {start} to {end} ===")
        print(f"{'ticker':8}" + "".join(f"{s:>12}" for s in ec.STRATS))
        geo = {s: 1.0 for s in ec.STRATS}
        cnt = 0
        for t in ec.TICKERS:
            i0 = ec.idx_at(data[t]["dates"], start)
            i1 = ec.idx_at(data[t]["dates"], end)
            if i0 < 0 or i1 <= i0:
                continue
            cnt += 1
            row = f"{t:8}"
            for s in ec.STRATS:
                ratio = curves[t][s][i1] / curves[t][s][i0]
                geo[s] *= ratio
                row += f"{(ratio - 1) * 100:>12,.1f}"
            print(row)
        print(f"{'geo-avg':8}" + "".join(
            f"{(geo[s] ** (1 / cnt) - 1) * 100:>12,.1f}" for s in ec.STRATS))


if __name__ == "__main__":
    main()
