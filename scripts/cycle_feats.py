"""dd is memoryless about the trough - add mdd and rc so the cycle position is complete.

User's argument (2026-07-30), and it is correct: dd = close/running_max - 1 reports the
CURRENT distance below the peak and nothing else. It cannot tell
    (a) fallen 20% and still falling      from
    (b) fell 50%, bottomed, recovered to -20%
Both read dd = -20%, yet (b) is exactly the recovery regime the model has been blind to.

My earlier rejection ("dd already reflects it") was wrong, and the evidence behind it
was weak anyway: a rolling-window run-up, scored by logistic regression - the criterion
that also wrongly dismissed sortino - at 1.9% weight among 24 features.

ENCODING. Three features that jointly pin the cycle position:
    dd  = close / running peak - 1          (<=0)  where we are
    mdd = running min of dd this episode    (<=0)  how bad it got   <- the memory
    rc  = dd - mdd                          (>=0)  recovered off the trough
An "episode" resets when dd returns to 0, i.e. a new all-time high.
"""
import numpy as np
import pandas as pd


def cycle_features(close, win=252):
    """Windowed, so the episode cannot stay open for decades.

    Resetting only at a new ALL-TIME high makes mdd sticky across regimes - NDX made
    no new high from 2000 to 2015, so the whole period carried mdd from the 2002
    trough and the feature stopped describing the current cycle. A rolling window
    bounds the memory to something a regime model can use.
    """
    dd = close / close.rolling(win, min_periods=20).max() - 1.0
    mdd = dd.rolling(win, min_periods=20).min()
    rc = dd - mdd
    return dd, mdd, rc


if __name__ == "__main__":
    import sys, os, warnings
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "..", "work", "scripts"))
    warnings.filterwarnings("ignore")
    sys.argv = ["x"]
    import final_config as fc
    CR = fc.CR
    print("DOES dd ALONE CONFUSE 'still falling' WITH 'recovered'?")
    print("mean feature values at two points that share a similar dd\n")
    for m in ["NDX", "FTSE"]:
        X, ret, rf, close = fc.feats(m)
        dd, mdd, rc = cycle_features(close)
        rows = []
        for a, b in CR:
            sl = close.loc[a:b]
            if len(sl) < 20:
                continue
            tr = (sl / sl.cummax() - 1).idxmin()
            pk = sl.loc[:tr].idxmax()
            if pk >= tr:
                continue
            idx = close.index
            half = dd.loc[pk:tr]
            target = half.min() / 2.0                      # halfway down
            falling = half.sub(target).abs().idxmin()      # on the way down
            after = dd.loc[tr:]
            recov = after.sub(target).abs().idxmin()       # same dd, on the way up
            rows.append((float(dd[falling]), float(mdd[falling]), float(rc[falling]),
                         float(dd[recov]), float(mdd[recov]), float(rc[recov])))
        if rows:
            a = np.mean(rows, axis=0)
            print(f"  {m}")
            print(f"    FALLING  : dd {a[0]:+.1%}   mdd {a[1]:+.1%}   rc {a[2]:+.1%}")
            print(f"    RECOVERED: dd {a[3]:+.1%}   mdd {a[4]:+.1%}   rc {a[5]:+.1%}")
            print(f"    -> dd differs by {abs(a[0]-a[3]):.1%}, but rc differs by {abs(a[2]-a[5]):.1%}")
