"""User's proposed feature design: raw indicators only, signed vs unsigned.

PROPOSAL (2026-07-30). Group features by whether they carry a SIGN:
  signed   ret x3, clv x3     - already directional, no split needed
                               (clv = (close-low)/(high-low) - 0.5, spans [-0.5,+0.5])
  unsigned vltup/vltdn x3, volup/voldn x3 - magnitudes, so split by direction to
                               give them sign
and give the signed group 60% of the metric (10% each). Test these RAW indicators
before adding the derived/slow features dd, dd63, ma200.

18 features total. This is the first grouping in the session with a principled basis
(does the feature carry direction?) rather than an empirical one.

CAVEAT worth measuring: at 60% over six features, the twelve vlt/vol features fall to
3.33% each. vltdn has been the strongest single discriminator all session (effect size
1.21-1.78 vs clv's 0.50-1.35), so this allocation demotes it well below clv.

ARMS (each isolates one thing):
  A  ret+clv 60%, 18 raw features        <- the proposal
  B  ret only 60%, 18 raw features       <- does clv earn its half?
  C  ret+clv 60%, 24 features (with dd/ma200)  <- does dropping the derived ones help?
  D  uniform, 18 raw features            <- control: is the 60% doing anything?
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
# import run_trend moved into __main__ - it runs an experiment grid on import
import final_config as fc                               # noqa: E402

H = (5, 21, 63)
SIGNED = [f"{a}_{h}" for a in ("ret", "clv") for h in H]
UNSIGNED = [f"{a}_{h}" for a in ("vltup", "vltdn", "volup", "voldn") for h in H]
RAW18 = SIGNED + UNSIGNED
DERIVED = ["dd", "dd63", "ma200"]

ARMS = [
    ("A proposal  ret+clv 60%", RAW18, SIGNED, 0.60),
    ("B ret only 60%", RAW18, [f"ret_{h}" for h in H], 0.60),
    ("C +dd/ma200 60%", RAW18 + DERIVED, SIGNED, 0.60),
    ("D uniform 18 raw", RAW18, SIGNED, None),
]


def weights_for(cols, target, share):
    if share is None:
        return np.ones(len(cols)) / np.sqrt(len(cols))
    tgt = np.array([c in target for c in cols])
    nt, no = int(tgt.sum()), int((~tgt).sum())
    w = np.zeros(len(cols))
    w[tgt] = np.sqrt(share / nt)
    if no:
        w[~tgt] = np.sqrt((1.0 - share) / no)
    return w


def run(m, cols, target, share):
    X, ret, rf, close = fc.feats(m)
    X = X[cols]
    fw = weights_for(cols, target, share)
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
            a.fit(Xf, ret_ser=ret.reindex(Xfit.index), feat_weights=fw, sort_by="cumret")
            s = pd.Series(a.predict_online(Xv), index=Xv.index)[vm]
            sh = jm.strategy_sharpe(ret.reindex(s.index), rf.reindex(s.index), s)
            if sh > best:
                best, blam = sh, lam
        sf = StandardScalerPD()
        Xt = sf.fit_transform(Xtr)
        b = JumpModel(n_components=2, jump_penalty=blam, cont=False)
        b.fit(Xt, ret_ser=ret.reindex(Xtr.index), feat_weights=fw, sort_by="cumret")
        Xs = sf.transform(X.loc[:pd.Timestamp(f"{year}-12-31")])
        s = pd.Series(b.predict_online(Xs), index=Xs.index)
        tm = (Xs.index > cut) & (Xs.index <= pd.Timestamp(f"{year}-12-31"))
        st.loc[Xs.index[tm]] = s[tm]
    raw = st.dropna().astype(int)
    pub = dc.confirm(raw, np.ones(len(raw), dtype=bool))
    return (pub, close.reindex(pub.index), ret.reindex(pub.index),
            rf.reindex(pub.index))


if __name__ == "__main__":
    import run_trend as t                # noqa: E402
    MK = sys.argv[1:] or ["FTSE", "NDX"]
    print("RAW SIGNED/UNSIGNED DESIGN — signed group given 60% of the metric")
    print(f"{'market':<8}{'arm':<24}{'n':>4}{'out':>7}{'capture':>9}{'protection':>12}"
          f"{'exit':>7}{'re-entry':>10}{'longest':>9}")
    for m in MK:
        r0 = t.evaluate(m, t.JM[m])
        d = t.D[m]
        e0 = fc.score(d["close"], d["ret"], d["rf"], t.JM[m])
        print(f"{m:<8}{'v8 live':<24}{14:>4}{float((t.JM[m]==1).mean()):>7.0%}"
              f"{r0['cap']:>9.3f}{r0['prot']:>+12.1%}{e0['exit']:>7.0f}"
              f"{e0['reentry']:>10.0f}{e0['longest']:>9.0f}")
        for lab, cols, tgt, sh in ARMS:
            t0 = time.time()
            pub, close, ret, rf = run(m, cols, tgt, sh)
            s = fc.score(close, ret, rf, pub)
            print(f"{'':<8}{lab:<24}{len(cols):>4}{s['out']:>7.0%}{s['cap']:>9.3f}"
                  f"{s['prot']:>+12.1%}{s['exit']:>7.0f}{s['reentry']:>10.0f}"
                  f"{s['longest']:>9.0f}   [{time.time()-t0:.0f}s]", flush=True)
        print()
