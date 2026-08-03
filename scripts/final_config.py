"""The redesigned configuration, tested across every asset type available.

CONFIG (what the session converged on)
  - 24 raw features, no masking, no stage-1, no overrides:
        vltup/vltdn x 5,21,63     direction-split volatility
        volup/voldn x 5,21,63     up-day / down-day volume vs long-run average
        ret         x 5,21,63     drift
        clv         x 5,21,63     intraday close position
        sortino     x 5,21,63     risk-adjusted return  <- the one that mattered
        dd, dd63, ma200           position in cycle, slow trend
  - PLAIN JumpModel (uniform weights). Forced selection lost at every budget on
    every market tested, and WJM's ceiling was to imitate this.
  - lambda selected per year on validation strategy Sharpe (production protocol).
  - same decode as v8 (2/3 confirmation + 8-day hold), same 2-day lag, 10bp cost.

sortino is included because it is a RATIO - a Euclidean-distance model cannot build
it from ret and downside-deviation however correlated they are. The linear screens
that pruned it were the wrong test, and restoring it moved NDX capture 1.335 -> 2.099
and exit lag 102d -> 30d.

ASSET TYPES: developed indices (NDX SPX FTSE NIKKEI), emerging Asia (HSI HSCEI
KOSPI), thematic ETF (ARKQ), single stocks (MSFT NVDA), commodity (GOLD).

Scored on breadth per the user's criterion - a market counts BETTER only if capture
AND protection both improve - plus the circular-shift placebo, since every arm so far
has run higher exposure than v8 and protection rises mechanically with time out.
"""
import os
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

import decode as dc                                     # noqa: E402

CR = [("2000-03-10", "2002-10-09"), ("2007-10-31", "2009-03-09"),
      ("2011-07-22", "2011-10-03"), ("2015-07-20", "2016-02-11"),
      ("2018-08-29", "2018-12-24"), ("2020-02-19", "2020-03-23"),
      ("2022-01-03", "2022-12-28"), ("2025-02-19", "2025-04-08")]
COST = 10 / 1e4
TYPE = {"NDX": "dev index", "SPX": "dev index", "FTSE": "dev index",
        "NIKKEI": "dev index", "HSI": "EM Asia", "HSCEI": "EM Asia",
        "KOSPI": "EM Asia", "ARKQ": "thematic ETF", "MSFT": "single stock",
        "NVDA": "single stock", "GOLD": "commodity"}


def feats(m):
    cfg = jm.SYMBOLS[m]
    close, ret, rf = jm.load_data(cfg)
    raw = pd.read_csv(os.path.join(jm.DATA_DIR, cfg["index"]),
                      index_col=0, parse_dates=True)
    cf = raw["Close"].astype(float)
    v = raw["Volume"].astype(float).reindex(ret.index)
    hi = raw["High"].astype(float).reindex(ret.index)
    lo = raw["Low"].astype(float).reindex(ret.index)
    c = cf.reindex(ret.index)
    vb = v.ewm(halflife=252).mean()
    pos, neg = np.maximum(ret, 0.0), np.minimum(ret, 0.0)
    vup, vdn = v.where(ret > 0, 0.0), v.where(ret < 0, 0.0)
    clv = ((c - lo) / (hi - lo) - 0.5).fillna(0.0)
    f = {}
    for h in (5, 21, 63):
        dd_ = np.sqrt(pd.Series(neg, index=ret.index).pow(2).ewm(halflife=h).mean())
        f[f"vltup_{h}"] = np.log(np.sqrt(pd.Series(pos, index=ret.index)
                                         .pow(2).ewm(halflife=h).mean()))
        f[f"vltdn_{h}"] = np.log(dd_)
        f[f"volup_{h}"] = vup.ewm(halflife=h).mean() / vb
        f[f"voldn_{h}"] = vdn.ewm(halflife=h).mean() / vb
        f[f"ret_{h}"] = ret.ewm(halflife=h).mean()
        f[f"clv_{h}"] = clv.ewm(halflife=h).mean()
        f[f"sortino_{h}"] = ret.ewm(halflife=h).mean().div(dd_)
    f["dd"] = c / c.cummax() - 1
    f["dd63"] = c / c.rolling(63).max() - 1
    f["ma200"] = np.log(cf / cf.rolling(200).mean()).reindex(ret.index)
    # DROP-BEFORE-DROPNA (2026-08-02, user). ma200 and sortino_* are built here but are
    # in NO current config - v11 rebuilt the feature set from raw indicators and uses
    # neither. They were still inside the dropna(), so an unused column with a 200-day
    # warm-up decided where every market's history could begin, costing ~200 rows each.
    # That matters more than it looks: the JM-vs-VM assignment compares the two models on
    # their SHARED rows, and JM starts later than VM on every market tested, so JM's
    # warm-up bound the whole comparison. Dropping them moves the earliest usable test
    # year a full year earlier on HK1810, BTC, ARKQ and META.
    # Kept as columns (callers may ask for them; ma_family's anchors can name ma200) but
    # excluded from the completeness test that trims the frame.
    OPTIONAL = [k for k in f if k == "ma200" or k.startswith("sortino_")]
    X = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)
    X = X.loc[X.drop(columns=OPTIONAL).notna().all(axis=1)]
    return X, ret.reindex(X.index), rf.reindex(X.index), close.reindex(X.index)


def run(m):
    X, ret, rf, close = feats(m)
    first = jm.SYMBOLS[m]["first_test"]
    st = pd.Series(index=X.index, dtype=float)
    for year in range(first, X.index[-1].year + 1):
        cut = pd.Timestamp(f"{year - 1}-12-31")
        Xtr = X.loc[:cut]
        if len(Xtr) < 500:
            continue
        vs = pd.Timestamp(f"{year - 1 - jm.VAL_YEARS}-12-31")
        Xfit, vm = Xtr.loc[:vs], Xtr.index > vs
        if len(Xfit) < 250 or vm.sum() < 100:
            continue
        sv = StandardScalerPD()
        Xf, Xv = sv.fit_transform(Xfit), sv.transform(Xtr)
        best, blam = -np.inf, jm.LAMBDA_GRID[0]
        for lam in jm.LAMBDA_GRID:
            a = JumpModel(n_components=2, jump_penalty=lam, cont=False)
            a.fit(Xf, ret_ser=ret.reindex(Xfit.index), sort_by="cumret")
            s = pd.Series(a.predict_online(Xv), index=Xv.index)[vm]
            sh = jm.strategy_sharpe(ret.reindex(s.index), rf.reindex(s.index), s)
            if sh > best:
                best, blam = sh, lam
        sf = StandardScalerPD()
        Xt = sf.fit_transform(Xtr)
        b = JumpModel(n_components=2, jump_penalty=blam, cont=False)
        b.fit(Xt, ret_ser=ret.reindex(Xtr.index), sort_by="cumret")
        Xs = sf.transform(X.loc[:pd.Timestamp(f"{year}-12-31")])
        s = pd.Series(b.predict_online(Xs), index=Xs.index)
        tm = (Xs.index > cut) & (Xs.index <= pd.Timestamp(f"{year}-12-31"))
        st.loc[Xs.index[tm]] = s[tm]
    raw = st.dropna().astype(int)
    pub = dc.confirm(raw, np.ones(len(raw), dtype=bool))
    return pub, close.reindex(pub.index), ret.reindex(pub.index), rf.reindex(pub.index)


def own_bear_labels(close, thresh=0.20):
    """Label peak->trough episodes of >= `thresh` decline on the market's OWN price.

    REPLACES the fixed CR crisis calendar for AUC (user, 2026-07-31). Those windows
    are NDX-derived - 2000-03-10 to 2002-10-09 is the Nasdaq top and bottom - and
    applying them to other assets mislabels in BOTH directions:

      - non-crashes inside a window get labelled bear. The old code took the worst
        peak-to-trough in each window with no minimum depth, so GOLD had a -3.0%
        dip over 9 days and a -6.0% dip over 4 days scored as bear markets.
      - real crashes outside every window are labelled calm, so a model that calls
        them correctly is marked WRONG. Gold's April 2013 collapse (~-25%) falls in
        no CR window at all.

    Scoring each market against its own >=20% bears changed real conclusions:
    KOSPI 0.624 -> 0.427 (BELOW chance - the signal is anti-correlated with its own
    bears), ARKQ 0.742 -> 0.597 (it was never the second-best detector), GOLD
    0.595 -> 0.641 (it does have some skill, the calendar was hiding it).

    Only AUC is affected - capture and protection come from returns and are
    unchanged, so conclusions decided on capture do not move.
    """
    lab = pd.Series(0, index=close.index)
    peak, peak_i = close.iloc[0], close.index[0]
    trough, trough_i = np.inf, None
    for d, p in close.items():
        if p > peak:
            if trough_i is not None and (peak - trough) / peak >= thresh:
                lab.loc[peak_i:trough_i] = 1
            peak, peak_i, trough, trough_i = p, d, np.inf, None
        elif p < trough:
            trough, trough_i = p, d
    if trough_i is not None and (peak - trough) / peak >= thresh:
        lab.loc[peak_i:trough_i] = 1
    return lab


def crisis_calendar_labels(close, pub):
    """The OLD fixed-window labels. Kept only to reproduce pre-2026-07-31 numbers."""
    lab = pd.Series(0, index=pub.index)
    for a2, b2 in CR:
        sl2 = close.loc[a2:b2]
        if len(sl2) < 20:
            continue
        tr2 = (sl2 / sl2.cummax() - 1).idxmin()
        pk2 = sl2.loc[:tr2].idxmax()
        if pk2 < tr2:
            lab.loc[pk2:tr2] = 1
    return lab


def score(close, ret, rf, pub, auc_labels="own"):
    # The 2-day execution lag leaves the first two rows undefined. Seed them from the
    # FIRST STATE, not with a blanket 1.0 (bug found 2026-08-01, user sanity check).
    # The old fillna(1.0) counted a market as invested for two days even when the
    # signal opened in bear - on KOSPI, whose first state is bear and whose first two
    # days gained +5.35%, that inflated capture from 0.504 to 0.530. This now matches
    # build_payloads.py, which always seeded from the first state, so the summary table
    # and the market tabs can no longer disagree.
    pos = (pub == 0).astype(float).shift(2).fillna(1.0 if pub.iloc[0] == 0 else 0.0)
    st = pos * ret + (1 - pos) * rf - pos.diff().abs().fillna(0.0) * COST
    eqs, eqb = (1 + st).cumprod(), (1 + ret).cumprod()
    tr = (close / close.cummax() - 1).idxmin()
    pk = close.loc[:tr].idxmax()
    prot = float((1 + st.loc[pk:tr]).prod() - (1 + ret.loc[pk:tr]).prod()) if pk < tr else 0.0
    E = []
    for a, b in CR:
        sl = close.loc[a:b]
        if len(sl) < 20:
            continue
        t2 = (sl / sl.cummax() - 1).idxmin()
        p2 = sl.loc[:t2].idxmax()
        if p2 >= t2:
            continue
        w = pos.loc[p2:b]
        o = w[w == 0]
        if len(o):
            E.append(close.index.get_loc(o.index[0]) - close.index.get_loc(p2))
    # RE-ENTRY: days from the crisis TROUGH until the signal returns to long. A
    # signal that exits fast and never comes back scores a great exit lag while
    # simply sitting in cash - exit lag alone cannot distinguish skill from inertia.
    R = []
    for a, b in CR:
        sl = close.loc[a:b]
        if len(sl) < 20:
            continue
        t2 = (sl / sl.cummax() - 1).idxmin()
        w2 = pos.loc[t2:]
        bk = w2[w2 == 1]
        if len(bk) and (bk.index[0] - t2).days < 800:
            R.append(close.index.get_loc(bk.index[0]) - close.index.get_loc(t2))
    # is it actually trading, or parked? count completed round trips and the
    # longest single stretch out of the market.
    runs = pub.groupby((pub != pub.shift()).cumsum())
    bear_runs = [len(g) for _, g in runs if g.iloc[0] == 1]
    # STATE AUC: does the published signal line up with this market's OWN bears?
    # Binary predictor, so this equals balanced accuracy - 0.50 is chance, and it is
    # scale-free, unlike capture/protection which depend on the market's own returns.
    lab = (own_bear_labels(close).reindex(pub.index).fillna(0)
           if auc_labels == "own" else crisis_calendar_labels(close, pub))
    auc = np.nan
    if lab.nunique() > 1:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(lab.values, pub.values))
    return dict(cap=float(eqs.iloc[-1] / eqb.iloc[-1]), prot=prot,
                out=float((pub == 1).mean()), exit=(np.mean(E) if E else np.nan),
                reentry=(np.mean(R) if R else np.nan),
                auc=auc,
                trips=len(bear_runs),
                longest=(max(bear_runs) if bear_runs else 0))


if __name__ == "__main__":
    # Imported HERE, not at module level. run_trend.py is an experiment script that
    # loads 11 pickles and runs a whole evaluation grid as a SIDE EFFECT OF IMPORT,
    # printing two tables as it goes. final_config is imported by nearly every script
    # in this directory, so every one of them was paying that cost and polluting its
    # own output with it. Only this __main__ block uses `t`.
    import run_trend as t                               # noqa: E402
    MK = sys.argv[1:] or list(TYPE)
    rng = np.random.default_rng(20260729)
    print("REDESIGNED CONFIG (24 raw features + sortino, uniform weights, no overrides)")
    print(f"{'market':<8}{'type':<13}{'out':>12}{'capture':>16}{'protection':>19}"
          f"{'exit':>10}{'re-entry':>12}{'trips':>12}{'longest':>9}{'plc':>6}  verdict")
    tally = [0, 0, 0]
    for m in MK:
        d = t.D[m]
        b0 = t.JM[m]
        r0 = t.evaluate(m, b0)
        e0 = score(d["close"], d["ret"], d["rf"], b0)
        t0 = time.time()
        pub, close, ret, rf = run(m)
        pd.DataFrame({"state": pub, "close": close, "ret": ret, "rf": rf}).to_csv(
            os.path.join(HERE, "featcache", f"redesign_{m}.csv"))
        s = score(close, ret, rf, pub)
        up = s["cap"] > r0["cap"] and s["prot"] > r0["prot"]
        dn = s["cap"] < r0["cap"] and s["prot"] < r0["prot"]
        tally[0 if up else (2 if dn else 1)] += 1
        sh = rng.integers(252, len(pub) - 252, size=200)
        nl = [score(close, ret, rf, pd.Series(np.roll(pub.values, int(k)), index=pub.index))
              for k in sh]
        pp = 100 * np.mean([x["prot"] < s["prot"] for x in nl])
        v = "BETTER" if up else ("worse" if dn else "mixed")
        print(f"{m:<8}{TYPE[m]:<13}{float((b0==1).mean()):>5.0%}->{s['out']:<6.0%}"
              f"{r0['cap']:>7.3f}->{s['cap']:<8.3f}"
              f"{r0['prot']:>+9.1%}->{s['prot']:<+9.1%}"
              f"{e0['exit']:>4.0f}->{s['exit']:<5.0f}"
              f"{e0['reentry']:>5.0f}->{s['reentry']:<6.0f}"
              f"{e0['trips']:>5.0f}->{s['trips']:<6.0f}"
              f"{s['longest']:>9.0f}{pp:>5.0f}%  {v}", flush=True)
    print()
    print(f"  BREADTH: better on both {tally[0]}/{len(MK)}   mixed {tally[1]}   worse on both {tally[2]}")
