"""Rebuild data/hk9988_long.csv = BABA (NYSE) spliced to 9988.HK.

Alibaba's HK line (9988.HK) only lists on 2019-11-26, which leaves 6.7 years of
history and about 2 testable years. BABA has traded on the NYSE since 2014-09-19 and
is the SAME security - verified over the overlap by a weekly return correlation of
0.884 and a stable 0.97-1.02 price ratio. (The DAILY correlation is only 0.53 purely
because Hong Kong closes at 04:00 ET, before the US session opens; that is a session
artifact, not divergence.) Splicing gives 11.9 years and 8 testable ones.

Two corrections to the common story, both checked: BABA never delisted, it is still
trading; and the 2007 Hong Kong listing was *Alibaba.com Ltd*, the B2B subsidiary
taken private in 2012 - a different security whose old ticker 1688.HK has since been
reused by an unrelated company.

THE JOIN. BABA is rescaled onto the HK line, not the other way round, so the live
end of the series is untouched real 9988.HK data.

  price  - scaled so the return ACROSS the join is exactly zero. The raw gap is an
           IPO-pricing artifact (the HK line was priced at a discount), not a market
           move, and leaving it in would inject a fake overnight jump into every
           momentum feature that spans 2019-11-26.
  volume - scaled to match 60-day median volume on each side of the join. Volume
           enters the model only through its own z-scores, so the level is arbitrary,
           but a step change in it would read as a volume shock.

This existed only as ad-hoc work until 2026-08-02, which meant a data refresh updated
baba.csv and hk9988.csv while hk9988_long.csv - the file actually scored - stayed
frozen. Run this after update_data.py, always.
"""
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
JOIN = pd.Timestamp("2019-11-26")      # first 9988.HK session
MEDIAN_WIN = 60                        # trading days each side, for the volume match
COLS = ["Open", "High", "Low", "Close", "Volume"]


def build():
    baba = pd.read_csv(os.path.join(DATA_DIR, "baba.csv"), index_col=0, parse_dates=True)
    hk = pd.read_csv(os.path.join(DATA_DIR, "hk9988.csv"), index_col=0, parse_dates=True)
    assert hk.index[0] == JOIN, f"9988.HK now starts {hk.index[0].date()}, expected {JOIN.date()}"

    pre = baba.loc[baba.index < JOIN]
    assert len(pre) > 1000, f"only {len(pre)} BABA rows before the join"

    # price: anchor so the join return is exactly 0.000%
    pf = hk["Close"].iloc[0] / pre["Close"].iloc[-1]
    # volume: match the 60-day medians on each side
    vf = hk["Volume"].iloc[:MEDIAN_WIN].median() / pre["Volume"].iloc[-MEDIAN_WIN:].median()

    scaled = pre.copy()
    for c in ("Open", "High", "Low", "Close"):
        scaled[c] = scaled[c] * pf
    scaled["Volume"] = scaled["Volume"] * vf

    out = pd.concat([scaled[COLS], hk[COLS]])
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates
    join_ret = out["Close"].loc[JOIN] / out["Close"].shift(1).loc[JOIN] - 1
    assert abs(join_ret) < 1e-6, f"join return {join_ret:.6%} is not flat"
    return out, pf, vf


if __name__ == "__main__":
    out, pf, vf = build()
    path = os.path.join(DATA_DIR, "hk9988_long.csv")
    out.to_csv(path)
    print(f"hk9988_long.csv: {len(out):,} rows  {out.index[0].date()} -> {out.index[-1].date()}")
    print(f"  price factor {pf:.6f}, volume factor {vf:.6f}, join {JOIN.date()} flat")
