"""Re-score every CACHED raw state, gated and ungated, with churn on BOTH.

No fitting happens here. Every arm's raw state is already on disk from
ma_family.run_raw()'s cache, so this answers a new question - "what is the flip
behaviour of the GATED state, not just the raw one?" - in seconds rather than the
~10 minutes per arm a refit would cost. This is exactly what the cache was for.

Columns, reported identically for raw and gated:
    cap     capture vs buy-and-hold
    prot    peak-to-trough protection in the worst drawdown
    auc     state AUC against actual crash windows
    fl/yr   flips per year
    avg d   average run length in trading days (the mean gap between flips)
    <8d     runs shorter than the 8-day dwell

avg-d is the natural companion to fl/yr: fl/yr says how OFTEN it changes its mind,
avg-d says how long a typical stance survives. The gate can only raise avg-d - it
never shortens a run - so the pair shows what the gate actually buys.
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# pipeline scripts all live in this directory - no path insert needed
warnings.filterwarnings("ignore")

import decode as dc                          # noqa: E402
import final_config as fc                    # noqa: E402
import configs as cf                         # noqa: E402

ORDER = ["ret70flat", "ma3_70", "ma200_70", "ret50flat", "ret60flat",
         "ret80flat", "ret70fast", "ret70slow"]
LABEL = {"ret70flat": "ret 5/21/63 @70", "ma3_70": "MA 21/50/200 @70",
         "ma200_70": "ma200 alone @70", "ret50flat": "ret @50 flat",
         "ret60flat": "ret @60 flat", "ret80flat": "ret @80 flat",
         "ret70fast": "ret @70 fast 3:2:1", "ret70slow": "ret @70 slow 1:2:3"}
TEST = {"ret70flat": "anchor", "ma3_70": "anchor", "ma200_70": "anchor",
        "ret50flat": "weight", "ret60flat": "weight", "ret80flat": "weight",
        "ret70fast": "split", "ret70slow": "split"}


def churn(sig):
    """flips/yr, average run length in trading days, count of runs under 8 days."""
    runs = [len(g) for _, g in sig.groupby((sig != sig.shift()).cumsum())]
    yrs = (sig.index[-1] - sig.index[0]).days / 365.25
    flips = int((sig.diff().abs() == 1).sum())
    return flips / yrs, float(np.mean(runs)), sum(1 for r in runs if r < 8)


def load_cache():
    """market -> {config tag: (raw, close, ret, rf)} from the on-disk cache."""
    out = {}
    for j in glob.glob(os.path.join(HERE, "featcache", "raw_*.json")):
        meta = json.load(open(j))
        c = pd.read_csv(j.replace(".json", ".csv"), index_col=0, parse_dates=True)
        out.setdefault(meta["market"], {})[meta.get("tag") or "?"] = (
            c["state"].astype(int), c["close"], c["ret"], c["rf"])
    return out


def row(state, close, ret, rf):
    s = fc.score(close, ret, rf, state)
    f, avg, short = churn(state)
    return s["cap"], s["prot"], s["auc"], f, avg, short


if __name__ == "__main__":
    MK = sys.argv[1:] or ["FTSE", "NDX"]
    cache = load_cache()
    print("RE-SCORED FROM CACHE - raw vs gated, with flip behaviour on both")
    print(f"{'market':<6}{'arm':<21}{'test':<7}"
          f"{'cap':>7}{'prot':>8}{'auc':>7}{'fl/yr':>7}{'avg d':>7}{'<8d':>5}"
          f"{'  ||':>4}{'cap':>7}{'prot':>8}{'auc':>7}{'fl/yr':>7}{'avg d':>7}{'<8d':>5}")
    print(f"{'':<34}{'--------- UNGATED (raw) ----------':^38}"
          f"{'---------- GATED (2/3 + 8d) ----------':^42}")
    for m in MK:
        if m not in cache:
            continue
        for name in ORDER:
            if name not in cache[m]:
                continue
            raw, close, ret, rf = cache[m][name]
            pub = dc.confirm(raw, np.ones(len(raw), dtype=bool))
            a = row(raw, close, ret, rf)
            b = row(pub, close, ret, rf)
            print(f"{m:<6}{LABEL[name]:<21}{TEST[name]:<7}"
                  f"{a[0]:>7.3f}{a[1]:>+8.1%}{a[2]:>7.3f}{a[3]:>7.1f}{a[4]:>7.1f}{a[5]:>5.0f}"
                  f"{'  ||':>4}"
                  f"{b[0]:>7.3f}{b[1]:>+8.1%}{b[2]:>7.3f}{b[3]:>7.1f}{b[4]:>7.1f}{b[5]:>5.0f}")
        print()
