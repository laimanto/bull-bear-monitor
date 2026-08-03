"""Turn VM's alarm level into an honest flip probability, walk-forward.

THE PROBLEM (measured 2026-08-01). VM's score ranks well - its correlation with
actually flipping within 21 days is +0.383 on KOSPI, as good as the jump model's
+0.394 on NDX - but it is not CALIBRATED. Because both of its alarms are clipped to
[0,1], the score pins at exactly 1.00 on 46% of KOSPI's bear days, and the dashboard
rendered "flip to bull = 1 - 1.00 = 0%". The measured rate on those very days was
30%. The information is real; the number was wrong.

THE FIX. Learn the mapping alarm -> P(flip within 21 trading days) from history, and
publish that instead. Isotonic regression, so the mapping is monotone by
construction (a louder alarm can never imply a higher chance of recovery).

WALK-FORWARD, with the label's own lookahead handled:
  - refit each January on data through 31 Dec of the prior year, like everything else
  - the label "flipped within 21 days" needs 21 days of future data to observe, so the
    fit uses only training days at least 21 trading days before the cut. Without this
    the last month of each training window would carry a label peeked from the test
    period.
  - separate mappings per state: from BULL, alarm -> P(flip to bear), increasing;
    from BEAR, alarm -> P(flip to bull), decreasing.

OUTPUT CONVENTION. The dashboard computes the displayed number as
`stateIsBull ? p_bear : 1 - p_bear`, so to make the display equal the calibrated
probability we store:
    bull day  ->  p_bear = P(flip to bear)
    bear day  ->  p_bear = 1 - P(flip to bull)
The template needs no change, and p_bear keeps its "high = bearish" direction.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# pipeline scripts all live in this directory - no path insert needed
warnings.filterwarnings("ignore")

from sklearn.isotonic import IsotonicRegression      # noqa: E402

HORIZON = 21
MIN_FIT = 250          # training days needed before a mapping is trusted
MIN_POS = 10           # need some flips of each kind to fit anything


def flip_labels(state, horizon=HORIZON):
    """1 if the state changes within the next `horizon` rows, else 0. NaN where the
    window runs past the end of the data (label not observable)."""
    s = state.values
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(n):
        j = min(i + horizon + 1, n)
        if j - i <= 1 and i + horizon >= n:
            continue
        w = s[i + 1:j]
        if len(w) == 0:
            continue
        if i + horizon >= n:
            # partial window: only a positive is conclusive
            out[i] = 1.0 if (w != s[i]).any() else np.nan
        else:
            out[i] = 1.0 if (w != s[i]).any() else 0.0
    return pd.Series(out, index=state.index)


def _fit(alarm, lab, increasing):
    m = alarm.notna() & lab.notna()
    a, y = alarm[m], lab[m]
    if len(a) < MIN_FIT or y.sum() < MIN_POS or (1 - y).sum() < MIN_POS:
        return None
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=increasing,
                            out_of_bounds="clip")
    ir.fit(a.values, y.values)
    return ir


def calibrate(state, alarm, first_year=None):
    """Walk-forward calibrated p_bear (dashboard convention). NaN before the first
    year that can be fitted."""
    idx = state.index
    lab = flip_labels(state)
    out = pd.Series(np.nan, index=idx)
    years = sorted({t.year for t in idx})
    for y in years:
        cut = pd.Timestamp(f"{y-1}-12-31")
        tr = idx[idx <= cut]
        if len(tr) < MIN_FIT:
            continue
        # drop the tail whose label needed data from the test period
        usable = tr[:-HORIZON] if len(tr) > HORIZON else tr[:0]
        if len(usable) < MIN_FIT:
            continue
        st_tr, al_tr, lb_tr = state.loc[usable], alarm.loc[usable], lab.loc[usable]
        bull = st_tr == 0
        f_bull = _fit(al_tr[bull], lb_tr[bull], increasing=True)     # P(flip to bear)
        f_bear = _fit(al_tr[~bull], lb_tr[~bull], increasing=False)  # P(flip to bull)
        te = idx[(idx > cut) & (idx <= pd.Timestamp(f"{y}-12-31"))]
        if not len(te):
            continue
        for t in te:
            a = alarm.loc[t]
            if np.isnan(a):
                continue
            if state.loc[t] == 0:
                if f_bull is not None:
                    out.loc[t] = float(f_bull.predict([a])[0])
            else:
                if f_bear is not None:
                    out.loc[t] = 1.0 - float(f_bear.predict([a])[0])
    return out


if __name__ == "__main__":
    import build_regimes as jm
    import combo_v2 as cv
    R = os.path.join(HERE, "..", "work", "results")
    print("CALIBRATION CHECK - displayed % vs what actually happened, out-of-sample\n")
    for m in ["KOSPI", "HSI"]:
        cfg = jm.SYMBOLS[m]
        close, ret, rf = jm.load_data(cfg)
        raw = pd.read_csv(os.path.join(jm.DATA_DIR, cfg["index"]), index_col=0, parse_dates=True)
        vol = raw["Volume"].astype(float).reindex(close.index)
        v0 = jm.VOL_START.get(m)
        if v0:
            close, vol = close.loc[v0:], vol.loc[v0:]
        cb = cv.build_walkforward(m, close, vol, cfg["first_test"])
        alarm = cb["combo_p_bear"]
        state = (alarm >= 0.60).astype(int)
        cal = calibrate(state, alarm)
        lab = flip_labels(state)
        disp_raw = np.where(state == 0, alarm, 1 - alarm)
        disp_cal = np.where(state == 0, cal, 1 - cal)
        df = pd.DataFrame({"raw": disp_raw, "cal": disp_cal, "act": lab,
                           "state": state}, index=state.index).dropna()
        print(f"  {m}  ({len(df)} days with a known outcome)")
        print(f"    {'displayed band':<16}{'n':>6}{'RAW says':>10}{'CAL says':>10}{'ACTUAL':>9}")
        for lo, hi in [(0, .001), (.001, .2), (.2, .4), (.4, .6), (.6, 1.01)]:
            g = df[(df["cal"] >= lo) & (df["cal"] < hi)]
            if len(g) < 20:
                continue
            print(f"    {f'{lo:.0%}-{hi:.0%}':<16}{len(g):>6}{g['raw'].mean():>9.0%}"
                  f"{g['cal'].mean():>10.0%}{g['act'].mean():>9.0%}")
        z = df[df["raw"] <= 1e-9]
        if len(z):
            print(f"    days the OLD tile showed 0%: n={len(z)}, "
                  f"actual flip rate {z['act'].mean():.0%}, calibrated now says "
                  f"{z['cal'].mean():.0%}")
        err_raw = float((df["raw"] - df["act"]).abs().mean())
        err_cal = float((df["cal"] - df["act"]).abs().mean())
        print(f"    mean absolute calibration error:  raw {err_raw:.3f}  ->  "
              f"calibrated {err_cal:.3f}\n")
