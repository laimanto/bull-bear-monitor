"""The V11 diagnostics table: detection, economics, and reaction speed.

Exit / re-entry delays are measured against each market's OWN >=20% bear episodes,
not the fixed NDX-derived crisis calendar. That calendar mislabels non-US assets in
both directions (it scored a -3.0% Gold dip as a bear market while ignoring Gold's
April 2013 collapse), so timing measured against it is not comparable across markets.

  exit delay     trading days from the episode PEAK until the signal leaves the market
  re-entry delay trading days from the episode TROUGH until the signal returns
  missed         episodes where the signal never left at all
"""
import os, sys, warnings
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
# pipeline scripts all live in this directory - no path insert needed
warnings.filterwarnings("ignore")
import decode as dc, final_config as fc, rescore as rs

# All 11 production markets (2026-08-01). The table is rendered once per variant
# in the Model tab, so it appears on BOTH dashboard pages - index_v11.html and
# monitor2_v11.html. Restricting it to the 7 indexes left the stock page showing
# diagnostics for markets it does not display.
INDEX = ["NDX", "SPX", "HSI", "HSCEI", "KOSPI", "NIKKEI", "FTSE",
         "GOLD", "ARKQ", "MSFT", "NVDA"]


EPISODE_WIN = 504          # 2 trading years


def episodes(close, thresh=0.20, win=EPISODE_WIN):
    """(peak_date, trough_date, depth) for every >=thresh peak-to-trough decline.

    The reference peak resets at a `win`-day rolling high, NOT an all-time high
    (fixed 2026-08-01). Resetting only at all-time highs made episodes swallow each
    other whenever a market took years to recover: NDX peaked in 2000, bottomed in
    2002 and did not regain that level until Nov 2015, so the 2007-09 GFC - a -54%
    decline - was absorbed into the dot-com episode and vanished from every table.
    A 2-year window bounds that memory, and NDX now shows both. Same reasoning as
    cycle_feats.cycle_features, which windows mdd for the same reason.
    """
    out = []
    ref = close.rolling(win, min_periods=1).max()
    peak, peak_i = close.iloc[0], close.index[0]
    trough, trough_i = np.inf, None
    for i, (d, p) in enumerate(close.items()):
        if p >= ref.iloc[i] - 1e-12:
            if trough_i is not None and (peak - trough) / peak >= thresh:
                out.append((peak_i, trough_i, trough / peak - 1))
            peak, peak_i, trough, trough_i = p, d, np.inf, None
        elif p < trough:
            trough, trough_i = p, d
    if trough_i is not None and (peak - trough) / peak >= thresh:
        out.append((peak_i, trough_i, trough / peak - 1))
    return out


def timing(close, pos, eps):
    ex, re_, missed = [], [], 0
    idx = close.index
    for pk, tr, _ in eps:
        w = pos.loc[pk:tr]
        o = w[w == 0]
        if len(o):
            ex.append(idx.get_loc(o.index[0]) - idx.get_loc(pk))
        else:
            missed += 1
        w2 = pos.loc[tr:]
        bk = w2[w2 == 1]
        if len(bk):
            re_.append(idx.get_loc(bk.index[0]) - idx.get_loc(tr))
    return (np.mean(ex) if ex else np.nan, np.mean(re_) if re_ else np.nan,
            missed, len(eps))


def row(m, cache):
    raw, close, ret, rf = cache[m]["ret70flat"]
    pub = dc.confirm(raw, np.ones(len(raw), dtype=bool))
    s = fc.score(close, ret, rf, pub)
    f, avgd, short = rs.churn(pub)
    pos = (pub == 0).astype(float).shift(2).fillna(1.0)
    eps = episodes(close)
    ex, re_, missed, n = timing(close, pos, eps)
    return dict(market=m, auc=s["auc"], cap=s["cap"], prot=s["prot"],
                out=float((pub == 1).mean()), fl=f, avgd=avgd,
                exitd=ex, reentry=re_, missed=missed, n_ep=n)


if __name__ == "__main__":
    cache = rs.load_cache()
    rows = [row(m, cache) for m in INDEX if m in cache]
    rows.sort(key=lambda r: -r["auc"])
    print("V11 DIAGNOSTICS - ret70flat, decode 2/3 no dwell, all 11 markets")
    print(f"{'market':<8}{'AUC':>7}{'cap':>7}{'prot':>8}{'out%':>6}"
          f"{'fl/yr':>7}{'avg hold':>9}{'exit lag':>9}{'re-entry':>9}"
          f"{'bears':>7}{'missed':>7}")
    for r in rows:
        print(f"{r['market']:<8}{r['auc']:>7.3f}{r['cap']:>7.3f}{r['prot']:>+8.1%}"
              f"{r['out']:>6.0%}{r['fl']:>7.1f}{r['avgd']:>9.0f}"
              f"{r['exitd']:>9.0f}{r['reentry']:>9.0f}{r['n_ep']:>7}{r['missed']:>7}")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "v11_metrics.csv"), index=False)
    print("\nwrote v11_metrics.csv")
