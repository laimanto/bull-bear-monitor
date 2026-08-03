"""Build regimes_{MARKET}_V11.csv for the 7 INDEX markets.

V11 = ret70flat + confirmation-only decode. The settled configuration:

  FEATURES  21 raw: ret/clv x 5,21,63 | vltup/vltdn/volup/voldn x 5,21,63 | dd, dd63, mdd
  WEIGHTS   ret_5/ret_21/ret_63 hold 70% of the distance metric, 23.3% each
            (flat); the other 18 features share the remaining 30%. n_eff 5.9.
  DECODE    2/3 confirmation ONLY - bear needs 2 consecutive raw days, bull 3.
            NO 8-day dwell, NO conviction veto, NO stage-2, NO masking.
  FIT       plain JumpModel, annual refit, lambda by validation Sharpe, 3y tail.
  EXECUTION 2-day lag, 10bp per switch.

WHAT CHANGED FROM v8, and why (all measured 2026-07-31 across 11 markets):
  - conviction veto (p_bear>=0.60) REMOVED - a hardcoded bar that mistimed Gold
  - 8-day dwell REMOVED - it was the harmful half of the decode. Isolated:
    sum capture raw 9.841 / confirm-only 11.835 / dwell-only 10.527 / both 11.321.
    Confirm-only beats raw on 9/11 markets and beats the both-gate in aggregate.
    NVDA is the clearest case: confirm 1.192, dwell 1.308, BOTH 0.756 - each
    filter helps alone, together they destroy a third of the capture.
  - feature set rebuilt from raw indicators, no stage-1/stage-2, no overrides.

p_bear is still recorded (distance to the bull centroid over total distance, in
the WEIGHTED space, since feat_weights scales columns before clustering) but it
no longer gates anything - it is diagnostic only.
"""
import hashlib
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# pipeline scripts all live in this directory - no path insert needed
warnings.filterwarnings("ignore")

import build_regimes as jm                              # noqa: E402
from jumpmodels.jump import JumpModel                   # noqa: E402
from jumpmodels.preprocess import StandardScalerPD      # noqa: E402

import configs as cf                                    # noqa: E402
import decode as dc                                     # noqa: E402
import ma_family as mf                                  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
FITCACHE = os.path.join(HERE, "..", "fitcache")
INDEX_MARKETS = ["NDX", "SPX", "HSI", "HSCEI", "KOSPI", "NIKKEI", "FTSE"]
CONFIG = "ret70flat"


def _fit_key(m, year, cols, w, close_tr, ret_tr, rf_tr, n_rows):
    """Everything a fit depends on, hashed - on PLATFORM-STABLE inputs only.

    Keyed on the TRAINING DATA ITSELF, not just the config. `ma_family._cache_key`
    hashes only (market, columns, weights), which silently survives a data refresh -
    it would hand back a pre-refresh fit for a post-refresh series. Hashing the
    values also catches a yfinance re-adjustment (a split rescales history without
    changing any date or row count).

    WHY close/ret/rf AND NOT THE FEATURE MATRIX: the first CI run (2026-08-03)
    missed all 491 cached fits, because the features go through ewm/log and libm's
    transcendentals differ in the last bit between platforms (Windows/py3.14 wrote
    the cache, Linux/py3.12 read it). close, ret and rf are produced from the
    committed CSV bytes by pure IEEE arithmetic (correctly-rounded, bit-identical
    everywhere), so this key is portable while still rotating on any real data
    change - a split or dividend re-adjustment rescales every close. Known gap:
    a volume-ONLY revision does not rotate the key; in practice volume revisions
    accompany price re-adjustments.
    """
    h = hashlib.md5()
    h.update(f"{m}|{CONFIG}|{year}|{','.join(cols)}|{n_rows}|".encode())
    h.update(np.ascontiguousarray(w, dtype=np.float64).tobytes())
    for s in (close_tr, ret_tr, rf_tr):
        h.update(np.ascontiguousarray(s.to_numpy(), dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def load_fit(key):
    path = os.path.join(FITCACHE, f"fit_{key}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None      # a corrupt cache entry must never fail a build


def save_fit(key, model, scaler, lam, meta):
    """Persist the trained model. Standing rule (user, 2026-08-02): never throw a fit
    away - refits are the entire cost of this pipeline, and a discarded fit also
    destroys any later ability to ask why a given year called what it called."""
    os.makedirs(FITCACHE, exist_ok=True)
    with open(os.path.join(FITCACHE, f"fit_{key}.pkl"), "wb") as fh:
        pickle.dump({"model": model, "scaler": scaler, "lam": lam}, fh)
    with open(os.path.join(FITCACHE, f"fit_{key}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


CACHE = {"hit": 0, "miss": 0}


def build(m):
    cols, share, spl = cf.CONFIGS[CONFIG]
    X, ret, rf, close = mf.feats(m)
    use = cf.columns(CONFIG)
    X = X[use]
    w, _ = cf.weights(CONFIG)
    first = jm.SYMBOLS[m]["first_test"]
    st = pd.Series(index=X.index, dtype=float)
    lam_s = pd.Series(index=X.index, dtype=float)
    pb_s = pd.Series(index=X.index, dtype=float)
    for year in range(first, X.index[-1].year + 1):
        cut = pd.Timestamp(f"{year - 1}-12-31")
        Xtr = X.loc[:cut]
        if len(Xtr) < 500:
            continue
        vs = pd.Timestamp(f"{year - 1 - jm.VAL_YEARS}-12-31")
        Xfit, vm = Xtr.loc[:vs], Xtr.index > vs
        if len(Xfit) < 250 or vm.sum() < 100:
            continue
        # An annual walk-forward only ever GAINS a year: appending recent bars leaves
        # every prior year's training slice untouched, so a data refresh should fit
        # one model, not all of them.
        key = _fit_key(m, year, list(X.columns), w, close.reindex(Xtr.index),
                       ret.reindex(Xtr.index), rf.reindex(Xtr.index), len(Xtr))
        hit = load_fit(key)
        if hit is not None:
            b, sf, blam = hit["model"], hit["scaler"], hit["lam"]
            CACHE["hit"] += 1
        else:
            CACHE["miss"] += 1
            _cap = int(os.environ.get("MAX_NEW_FITS", 10**9))
            if CACHE["miss"] > _cap:
                raise SystemExit(
                    f"ABORT: {CACHE['miss']} cache misses exceeds MAX_NEW_FITS={_cap}. "
                    "The fitcache no longer matches the committed data - rebuild "
                    "locally and commit the refreshed fitcache/ instead of training in CI.")
            sv = StandardScalerPD()
            Xf, Xv = sv.fit_transform(Xfit), sv.transform(Xtr)
            best, blam = -np.inf, jm.LAMBDA_GRID[0]
            for lam in jm.LAMBDA_GRID:
                a = JumpModel(n_components=2, jump_penalty=lam, cont=False)
                a.fit(Xf, ret_ser=ret.reindex(Xfit.index), feat_weights=w, sort_by="cumret")
                s = pd.Series(a.predict_online(Xv), index=Xv.index)[vm]
                sh = jm.strategy_sharpe(ret.reindex(s.index), rf.reindex(s.index), s)
                if sh > best:
                    best, blam = sh, lam
            sf = StandardScalerPD()
            Xt = sf.fit_transform(Xtr)
            b = JumpModel(n_components=2, jump_penalty=blam, cont=False)
            b.fit(Xt, ret_ser=ret.reindex(Xtr.index), feat_weights=w, sort_by="cumret")
            save_fit(key, b, sf, blam, dict(
                market=m, config=CONFIG, year=year, lam=float(blam),
                train_start=str(Xtr.index[0].date()), train_end=str(Xtr.index[-1].date()),
                n_train=int(len(Xtr)), columns=list(X.columns),
                weights=[float(x) for x in w],
                centers=[[float(x) for x in row] for row in np.asarray(b.centers_)]))
        Xs = sf.transform(X.loc[:pd.Timestamp(f"{year}-12-31")])
        s = pd.Series(b.predict_online(Xs), index=Xs.index)
        # p_bear in the WEIGHTED space - centers_ were fitted on X*w, so the
        # distance must be measured there too or the ratio is meaningless.
        c = np.asarray(b.centers_)
        Xw = np.asarray(Xs) * w
        d = ((Xw[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        pb = pd.Series(d[:, 0] / (d[:, 0] + d[:, 1] + 1e-12), index=Xs.index)
        tm = (Xs.index > cut) & (Xs.index <= pd.Timestamp(f"{year}-12-31"))
        st.loc[Xs.index[tm]] = s[tm]
        lam_s.loc[Xs.index[tm]] = blam
        pb_s.loc[Xs.index[tm]] = pb[tm]
    raw = st.dropna().astype(int)
    pub = dc.confirm(raw, np.ones(len(raw), dtype=bool))   # DWELL=1: confirm only
    out = pd.DataFrame({
        "close": close.reindex(pub.index), "ret": ret.reindex(pub.index),
        "rf": rf.reindex(pub.index), "state": pub.astype(float),
        "lam": lam_s.reindex(pub.index), "p_bear": pb_s.reindex(pub.index),
        "raw_state": raw.astype(float),
    })
    out.index.name = "Date"
    path = os.path.join(RESULTS, f"regimes_{m}_V11.csv")
    out.to_csv(path)
    return out, path


if __name__ == "__main__":
    MK = sys.argv[1:] or INDEX_MARKETS
    print(f"BUILD V11 REGIMES - {CONFIG}, decode 2/3 no dwell "
          f"(N_BEAR={dc.N_BEAR}, N_BULL={dc.N_BULL}, DWELL={dc.DWELL})")
    assert dc.DWELL == 1, "decode.DWELL must be 1 for V11"
    for m in MK:
        t0 = time.time()
        out, path = build(m)
        flips = int(out["state"].diff().abs().sum())
        print(f"  {m:<8}{len(out):>6} rows  {out.index[0].date()} -> {out.index[-1].date()}"
              f"  flips={flips:<4} cur={'BEAR' if out['state'].iloc[-1] else 'BULL'}"
              f"  MARKETDONE [{time.time()-t0:.0f}s]", flush=True)
    print(f"fits: {CACHE['miss']} trained, {CACHE['hit']} reused from {FITCACHE}")
    print("V11REGIMES COMPLETE")
