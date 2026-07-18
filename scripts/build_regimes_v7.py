"""v7: confidence-averaging ensemble of JM (v6's model) and Combo v2, decided
at decision-level, not model-internals-level - each model keeps its own
complete, unmodified signal; neither gates or vetoes the other.

  combo_p_bear   = combo_v2.build_walkforward(...)'s continuous score
  combined_p_bear = (JM p_bear + combo_p_bear) / 2
  published state = v6's EXACT 2/3-gate + 8-day-hold + bear-entry-conviction
                     decode (confirm_v6 in build_regimes.py), just fed
                     combined_p_bear instead of JM's own p_bear alone - not
                     a naive continuous blend + re-threshold, which an
                     earlier lambda-ensemble study showed destroys
                     persistence (the blend has no jump-penalty of its own).

Validated in the research repo (exp_combov2_confidence.py, 2026-07-18):
mean relative capture +5.9% vs the previously-adopted bear-only-veto design,
+16.8% vs plain JM v6, across the 11 production markets; fixes both of that
design's known bear-side losses (HSCEI, KOSPI). One real, accepted trade-off:
NVDA regresses (0.479 vs v6-alone's 0.535) - diagnosed directly (see the
research repo's nvda_diagnose.py): NVDA's own walk-forward VB trigger sits
at ~130% annualized vol (its 92nd percentile), so combo_p_bear stays pinned
near 0 for weeks during real-but-moderate pullbacks even while JM is
confidently elevated: blind averaging waters down JM's correct, timely call.
Tried and rejected as fixes: walk-forward-percentile MA-gap calibration
(worse in aggregate AND for NVDA), confidence-magnitude weighting and
online track-record adaptive weighting (both worse for NVDA specifically,
because combo_p_bear=~0 is itself a confident "no bear evidence" signal, not
an uncertain one - discounting it by "distance from 0.5" discounts the
wrong thing). The plain fixed 50/50 average remains the best validated
design; accepted with NVDA's regression as a known, deliberate trade-off.

Both models' walk-forward parameters are cached (JM's own models/{market}_
stage*_{year}.joblib via build_regimes.compute_raw - a cache hit if
build_regimes.py already ran for this market; Combo v2's models/{market}_
combo_{year}.joblib via combo_v2.build_walkforward) - v7 costs no extra
retraining beyond running build_regimes.py first.

Usage: python build_regimes_v7.py [MARKET ...]   (default: all 11; requires
build_regimes.py to have already populated the JM model cache for each
market - run that first if starting cold.)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

import build_regimes as jm
import combo_v2

warnings.filterwarnings("ignore")

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "..", "data")
RESULTS_DIR = os.path.join(DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BEAR, N_BULL = 2, 3
TB = 0.60
DWELL = 8


def confirm_v7(raw_state, combined_p_bear):
    """v6's exact decode structure (2/3 gate, 8-day hold, bear-entry
    conviction veto), fed the combined confidence instead of JM's own
    p_bear alone - see build_regimes.confirm_v6, kept in lockstep."""
    s, p = raw_state.values, combined_p_bear.values
    n = len(s)
    out = s.copy()
    last_flip = -10**9
    for i in range(1, n):
        cnt = N_BEAR if s[i] == 1 else N_BULL
        persist = all(s[i - j] == s[i] for j in range(min(cnt, i + 1)))
        if persist and s[i] != out[i - 1]:
            if i - last_flip < DWELL:
                persist = False
            elif s[i] == 1:
                persist = not (np.isnan(p[i]) or p[i] < TB)
        out[i] = s[i] if persist else out[i - 1]
        if out[i] != out[i - 1]:
            last_flip = i
    return pd.Series(out, index=raw_state.index)


def build_market(market):
    close, ret, rf, raw_states, p_bear_full, lam_full = jm.compute_raw(market)

    cfg = jm.SYMBOLS[market]
    raw = pd.read_csv(os.path.join(DATA_DIR, cfg["index"]), index_col=0, parse_dates=True)
    close_full = raw["Close"].astype(float)
    volume_full = raw["Volume"].astype(float)
    v0 = jm.VOL_START.get(market)
    if v0:
        close_full, volume_full = close_full.loc[v0:], volume_full.loc[v0:]

    combo = combo_v2.build_walkforward(market, close_full, volume_full, cfg["first_test"])

    common = raw_states.index.intersection(combo.index)
    raw_c = raw_states.loc[common]
    p_jm = p_bear_full.loc[common]
    p_combo = combo.loc[common, "combo_p_bear"]
    p_combined = (p_jm + p_combo) / 2

    published = confirm_v7(raw_c, p_combined)

    pd.DataFrame({"close": close.reindex(common),
                  "ret": ret.reindex(common),
                  "rf": rf.reindex(common),
                  "state": published,
                  "lam": lam_full.reindex(common),
                  "p_bear": p_combined,
                  "p_bear_jm": p_jm,
                  "p_bear_combo": p_combo}
                 ).to_csv(os.path.join(RESULTS_DIR, f"regimes_{market}_V7.csv"))
    print(f"{market}: regimes_{market}_V7.csv written, {len(common)} rows, "
          f"published flips {int(published.diff().abs().sum())}")


def main():
    markets = [m.upper() for m in sys.argv[1:]] or list(jm.SYMBOLS)
    for m in markets:
        build_market(m)


if __name__ == "__main__":
    main()
