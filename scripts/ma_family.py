"""Tests 3 & 4: is the trend anchor better as a FAMILY than as ma200 alone?

ma200 as a single feature carrying 60% of the metric beat ret x3 on NDX (2.188 vs
1.648), but the comparison was not like-for-like: ret spreads 60% over three
dimensions, ma200 concentrates it in one. These arms give the trend anchor three
features so both are equally concentrated (20% each).

  Test 3  MA long  : ma21, ma50, ma200      spans a month to a year
  Test 4  MA short : ma5, ma21, ma63        matches ret's 5/21/63 horizons

ma_n = log(close / n-day simple mean), the same encoding as the existing ma200.

Also runs share = 50/60/70 (test 2) on each anchor, so the core-weight question is
answered for the new families rather than assumed from ret's curve.
"""
import sys, time, warnings, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
# pipeline scripts all live in this directory - no path insert needed
warnings.filterwarnings("ignore")
import build_regimes as jm                  # noqa: E402
from jumpmodels.jump import JumpModel       # noqa: E402
from jumpmodels.preprocess import StandardScalerPD   # noqa: E402
import decode as dc                         # noqa: E402
import final_config as fc                   # noqa: E402
import raw_signed as rs                     # noqa: E402
from cycle_feats import cycle_features      # noqa: E402

H = (5, 21, 63)
MA_N = (5, 21, 50, 63, 200)


def feats(m):
    X, ret, rf, close = fc.feats(m)
    # DOUBLE WARM-UP FIX (2026-08-02, found while adding short-history markets).
    # `close` as returned by fc.feats is ALREADY truncated to X.index, i.e. past that
    # frame's own ~200-day ma200 warm-up. Computing another rolling(200) on it burned a
    # SECOND 200 days, so every market silently lost ~10 months of usable history:
    # HK1810's features began 2020-02-18 when 2019-04-30 was available. Rebuild these
    # columns from the FULL price series, then reindex - identical values, no lost rows.
    full = _full_close(m)
    dd, mdd, rc = cycle_features(full, 252)
    add = {"mdd": mdd.reindex(X.index)}
    # ret_126 (user, 2026-07-31): the anchor currently caps at 63d, but HSI's bears
    # run longer - ret126 separates its own bears at d=-0.76 vs the existing best
    # -0.70. Only enters a model if an anchor in configs.py names it.
    add["ret_126"] = full.pct_change(126).reindex(X.index)
    for n in MA_N:
        add[f"ma{n}"] = np.log(full / full.rolling(n).mean()).reindex(X.index)
    # Same drop-before-dropna rule as fc.feats: the ma* and ret_126 columns are built so
    # an anchor CAN name them, but no current config does, and letting them into the
    # completeness test cost ~200 rows per market. Only mdd is required here.
    X = X.assign(**add).replace([np.inf, -np.inf], np.nan)
    required = [c for c in X.columns
                if not (c.startswith("ma") or c == "ret_126" or c.startswith("sortino_"))]
    X = X.loc[X[required].notna().all(axis=1)]
    return X, ret.reindex(X.index), rf.reindex(X.index), close.reindex(X.index)


def _full_close(m):
    """The raw, fully untruncated close.

    Deliberately does NOT apply VOL_START. fc.feats already builds its own ma200 from
    the untruncated series, so truncating here would (a) disagree with it and (b) push
    the warm-up LATER on any market that has a VOL_START - HSI lost 120 rows when this
    honoured it. VOL_START exists to protect the VOLUME features from bad volume data;
    price-only features have no reason to respect it.
    """
    cfg = jm.SYMBOLS[m]
    raw = pd.read_csv(os.path.join(jm.DATA_DIR, cfg["index"]), index_col=0, parse_dates=True)
    return raw["Close"].astype(float).sort_index()


BASE = rs.RAW18 + ["dd", "dd63", "mdd"]        # ma200 lives in the anchor sets below
ANCHORS = {
    "ret 5/21/63":      [f"ret_{h}" for h in H],
    "ma200 alone":      ["ma200"],
    "MA 21/50/200":     ["ma21", "ma50", "ma200"],
    "MA 5/21/63":       ["ma5", "ma21", "ma63"],
}
# Reduced grid (user, 2026-07-31): sweep the core weight only on the two anchors
# being compared for weighting (ret and ma200). The MA FAMILIES are a separate
# question - each is tested against ma200 at a single share, not swept.
SHARES_BY_ANCHOR = {"ret 5/21/63": (0.50, 0.60, 0.70),
                    "ma200 alone": (0.50, 0.60, 0.70),
                    "MA 21/50/200": (0.60,),
                    "MA 5/21/63": (0.60,)}


def run(m, anchor, share):
    """Back-compat wrapper: returns the GATED (published) state, as before."""
    raw, close, ret, rf = run_raw(m, anchor, share)
    pub = dc.confirm(raw, np.ones(len(raw), dtype=bool))
    return pub, close.reindex(pub.index), ret.reindex(pub.index), rf.reindex(pub.index)


def _cache_key(m, cols, w):
    """Stable id for (market, feature set, weight vector) — the full determinant
    of a fit, given the protocol constants are fixed."""
    import hashlib
    sig = m + "|" + ",".join(cols) + "|" + ",".join(f"{x:.6f}" for x in w)
    return hashlib.md5(sig.encode()).hexdigest()[:10]


def run_raw(m, anchor, share, shares=None, cache=True, tag=""):
    """The walk-forward fit, returning the RAW state with no decode applied.

    Split out 2026-07-31: every conclusion so far (anchor, core weight, internal
    split) was measured on dc.confirm() output, so the gate was a confound in all
    of them. The fit is the expensive part - scoring the same raw state both gated
    and gate-free costs nothing extra, so both rankings come from one run.
    """
    X, ret, rf, close = feats(m)
    cols = BASE + [c for c in anchor if c not in BASE]
    X = X[cols]
    tgt = np.array([c in anchor for c in cols])
    w = np.zeros(len(cols))
    if shares is None:
        w[tgt] = np.sqrt(share / tgt.sum())
    else:
        # explicit per-anchor-column shares (for the internal-split arms), given
        # in the order of `anchor`; must sum to `share`
        assert abs(sum(shares) - share) < 1e-9, (shares, share)
        pos = {c: i for i, c in enumerate(cols)}
        for c, sc in zip(anchor, shares):
            w[pos[c]] = np.sqrt(sc)
    w[~tgt] = np.sqrt((1 - share) / (~tgt).sum())

    # ---- cache: the walk-forward is the whole cost; scoring it is free -------
    # Persisting the RAW state means any later question (different gate, different
    # metric, different crisis windows) is answered by re-scoring, not refitting.
    key = _cache_key(m, cols, w)
    cdir = os.path.join(HERE, "featcache")
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, f"raw_{m}_{key}.csv")
    if cache and os.path.exists(cpath):
        c = pd.read_csv(cpath, index_col=0, parse_dates=True)
        return (c["state"].astype(int), c["close"], c["ret"], c["rf"])

    first = jm.SYMBOLS[m]["first_test"]
    st = pd.Series(index=X.index, dtype=float)
    for year in range(first, X.index[-1].year + 1):
        cut = pd.Timestamp(f"{year-1}-12-31"); Xtr = X.loc[:cut]
        if len(Xtr) < 500: continue
        vs = pd.Timestamp(f"{year-1-jm.VAL_YEARS}-12-31")
        Xfit, vm = Xtr.loc[:vs], Xtr.index > vs
        if len(Xfit) < 250 or vm.sum() < 100: continue
        sv = StandardScalerPD(); Xf, Xv = sv.fit_transform(Xfit), sv.transform(Xtr)
        best, blam = -np.inf, jm.LAMBDA_GRID[0]
        for lam in jm.LAMBDA_GRID:
            a = JumpModel(n_components=2, jump_penalty=lam, cont=False)
            a.fit(Xf, ret_ser=ret.reindex(Xfit.index), feat_weights=w, sort_by="cumret")
            s = pd.Series(a.predict_online(Xv), index=Xv.index)[vm]
            sh = jm.strategy_sharpe(ret.reindex(s.index), rf.reindex(s.index), s)
            if sh > best: best, blam = sh, lam
        sf = StandardScalerPD(); Xt = sf.fit_transform(Xtr)
        b = JumpModel(n_components=2, jump_penalty=blam, cont=False)
        b.fit(Xt, ret_ser=ret.reindex(Xtr.index), feat_weights=w, sort_by="cumret")
        Xs = sf.transform(X.loc[:pd.Timestamp(f"{year}-12-31")])
        s = pd.Series(b.predict_online(Xs), index=Xs.index)
        tm = (Xs.index > cut) & (Xs.index <= pd.Timestamp(f"{year}-12-31"))
        st.loc[Xs.index[tm]] = s[tm]
    raw = st.dropna().astype(int)
    close, ret, rf = (close.reindex(raw.index), ret.reindex(raw.index),
                      rf.reindex(raw.index))
    if cache:
        pd.DataFrame({"state": raw, "close": close, "ret": ret, "rf": rf}).to_csv(cpath)
        # self-describing sidecar, so a cache file is never an anonymous hash
        import json
        with open(cpath.replace(".csv", ".json"), "w") as fh:
            json.dump({"market": m, "tag": tag, "anchor": list(anchor),
                       "share": share, "shares": (list(shares) if shares else None),
                       "columns": cols, "weights": [float(x) for x in w],
                       "n_effective": float(1.0 / np.sum((w**2 / (w**2).sum())**2))},
                      fh, indent=2)
    return raw, close, ret, rf


if __name__ == "__main__":
    # Imported HERE, not at module level: run_trend.py is an experiment script that
    # loads 11 pickles and runs a full evaluation grid as a SIDE EFFECT OF IMPORT.
    # Every build that touched ma_family was silently re-running that experiment and
    # printing its two tables into the build log. Only this __main__ block needs it.
    import run_trend as t                       # noqa: E402
    MK = sys.argv[1:] or ["FTSE", "NDX"]
    print("TESTS 2/3/4 — core weight on ret & ma200; MA families vs ma200 at 60%")
    print(f"{'market':<7}{'anchor':<16}{'share':<7}{'out':>7}{'capture':>9}"
          f"{'protection':>12}{'exit':>7}{'re-entry':>10}{'park':>7}")
    for m in MK:
        r0 = t.evaluate(m, t.JM[m]); d = t.D[m]
        e0 = fc.score(d["close"], d["ret"], d["rf"], t.JM[m])
        print(f"{m:<7}{'v8 live':<16}{'':<7}{float((t.JM[m]==1).mean()):>7.0%}"
              f"{r0['cap']:>9.3f}{r0['prot']:>+12.1%}{e0['auc']:>7.3f}{e0['exit']:>7.0f}"
              f"{e0['reentry']:>10.0f}{e0['longest']:>7.0f}")
        for aname, acols in ANCHORS.items():
            for sh in SHARES_BY_ANCHOR[aname]:
                t0 = time.time()
                pub, close, ret, rf = run(m, acols, sh)
                s = fc.score(close, ret, rf, pub)
                print(f"{'':<7}{aname:<16}{sh:<7.0%}{s['out']:>7.0%}{s['cap']:>9.3f}"
                      f"{s['prot']:>+12.1%}{s['auc']:>7.3f}{s['exit']:>7.0f}{s['reentry']:>10.0f}"
                      f"{s['longest']:>7.0f}  ARM  [{time.time()-t0:.0f}s]", flush=True)
            print()
