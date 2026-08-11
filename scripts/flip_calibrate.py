"""Honest flip probability for BOTH models: P(the published signal flips soon).

WHY THIS REPLACED THE OLD NUMBER (user, 2026-08-10/11)
------------------------------------------------------
The tile used to show a 21-day probability. It saturated: GOOGL sat at "Flip to Bull 100%"
for days while still published BEAR, and capping the display at 95% only hid it. Measured,
the old number was frozen across days on which the actual proximity was changing - it read
100% both two days out and one day out, and 96% on a day when three confirmation days were
still outstanding.

The cause is that a single level cannot express this. The published signal only moves when
the RAW state persists long enough to clear the confirmation gate (bear needs 2 consecutive
raw-bear days, bull needs 3 - `decode.N_BEAR/N_BULL`). So:

    P(flip next session) is EXACTLY 0 with 2+ confirmation days outstanding
    - 0.000% over 123,400 market-days, no exceptions - and 88.0% with 1 outstanding.

No rescaling of a level can repair that, because the fact is mechanical.

THE FIX: two inputs, not one
----------------------------
  need  how many consecutive raw days are still outstanding before a flip may publish
  gap   how far the model's own driver is past the threshold that turns the raw state

Neither works alone: `need` alone is near-binary (~3% vs 88%), and `gap` alone cannot see
the gate. Together they give a smooth, fully-populated ladder from ~2% to ~93%:

    P(flip within 5 sessions)      need 3    need 2    need 1
      driver > 0.35 from threshold     3%        2%         -
      0.35 - 0.22                      9%        7%         -
      0.22 - 0.12                     15%       11%         -
      0.12 - 0.05                     21%       20%       80%
      within 0.05                     37%       38%       80%
      already past it                 14%       54%       93%

HORIZON = 5 sessions, and it is deliberately NOT shown on the tile (user, 2026-08-11). The
number already encodes urgency - median sessions-to-flip runs 132 / 55 / 34 / 19 / 14 / 8 /
2 / 1 down that ladder - so a reader takes 93% as "tomorrow" and 20% as "not soon", both
correctly. At the top of the scale the 5-session number IS a next-day number: 89.6% next
session vs 93.8% within five. The horizon label was only hedging the bottom end, and the
countdown line on the tile states the mechanical floor far more precisely than a label could.

FITTED PER MODEL FAMILY, POOLED ACROSS MARKETS
---------------------------------------------
VM is the stronger case and the gap is the MECHANISM there, not a proxy: its raw state is
literally `alarm >= 0.60`, so distance-to-0.60 determines it. VM's ladder is monotone and
reaches 81% in the middle band.

JM is weaker and this is a real limitation, not a bug. JM's raw decode is NOT a threshold on
p_bear - the jump penalty makes it path-dependent, so the two disagree ~11% of days - which
is why JM's middle band tops out near 27% and its `need 3` column is non-monotone in the raw
counts. Isotonic enforces monotonicity; JM's gauge is still coarser between the extremes and
leans on the countdown.

Pooled across markets WITHIN a family because armed days are only ~1.7% of the record, i.e.
~30-50 per market - far too thin for a per-market fit. Refit every January on prior data
only, with the label's own 5-session lookahead trimmed off the training tail, exactly as
vm_calibrate does.

OUTPUT CONVENTION
-----------------
The dashboard computes the displayed number as `stateIsBull ? p_bear : 1 - p_bear`, so to
make the display equal P(flip away from today's published signal) this writes:
    bull day  ->  p_bear = P(flip to bear)
    bear day  ->  p_bear = 1 - P(flip to bull)
which keeps p_bear's "high = bearish" direction and needs no change to flipPct/flipZone.

The RAW DRIVER is preserved in `p_bear_raw` before p_bear is overwritten, so re-running is
idempotent and JM's bear-proximity score is not destroyed - the same mistake that was found
in build_boards' handling of `raw_state`.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

from sklearn.isotonic import IsotonicRegression      # noqa: E402

import build_boards as bb                            # noqa: E402
import decode as dc                                  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
HORIZON = 5            # sessions; see the module docstring for why 5 and why it is hidden
MIN_FIT = 400          # pooled rows needed before an isotonic fit is trusted
MIN_POS = 20           # need some flips AND some non-flips
MIN_TR = 5             # raw transitions of each kind needed to place a JM threshold
VM_THRESH = bb.VM_THRESH


def gate_for(family, variant):
    """(days needed to publish BEAR, days needed to publish BULL)."""
    if family == "VM":
        g = bb.VM_CFG[variant]["gate"]
        return int(g[0]), int(g[1])
    return int(dc.N_BEAR), int(dc.N_BULL)


def countdown(pub, raw, n_bear, n_bull):
    """Consecutive raw days still outstanding before a flip may publish, per day."""
    r, p = raw.values, pub.values
    n = len(r)
    run = np.ones(n, dtype=int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if r[i] == r[i - 1] else 1
    need = np.empty(n, dtype=int)
    req = np.empty(n, dtype=int)
    for i in range(n):
        tgt = 1 - p[i]                        # the raw value a flip would need
        req[i] = n_bear if tgt == 1 else n_bull
        need[i] = max(0, req[i] - (run[i] if r[i] == tgt else 0))
    return pd.Series(need, index=pub.index), pd.Series(req, index=pub.index)


def jm_levels(drv, raw, upto):
    """(U, L) = median driver value at this market's own prior raw transitions."""
    s, d = raw.loc[:upto], drv.loc[:upto]
    ch = s.diff().fillna(0)
    tb, tu = d[ch == 1].dropna(), d[ch == -1].dropna()
    if len(tb) < MIN_TR or len(tu) < MIN_TR:
        return None
    U, L = float(tb.median()), float(tu.median())
    return (U, L) if U > L else None


def flip_labels(pub, horizon=HORIZON):
    """1 if the PUBLISHED state changes within the next `horizon` rows, else 0. NaN where
    the window runs past the end of the data, unless a flip already settles it."""
    s = pub.values
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(n):
        w = s[i + 1:min(i + horizon + 1, n)]
        if len(w) == 0:
            continue
        if (w != s[i]).any():
            out[i] = 1.0
        elif i + horizon < n:
            out[i] = 0.0
        # else: inconclusive tail - leave NaN
    return pd.Series(out, index=pub.index)


def _load(market, variant):
    p = os.path.join(RESULTS, f"regimes_{market}_{variant}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    if "raw_state" not in d or "state" not in d:
        return None
    # idempotence: once p_bear has been overwritten, the driver lives in p_bear_raw
    if "p_bear_raw" not in d and "p_bear" in d:
        d["p_bear_raw"] = d["p_bear"]
    family = "VM" if "vm_alarm" in d else "JM"
    return d, family, p


def prepare(market, variant):
    """need / gap / label for one market, plus what is needed to write it back."""
    got = _load(market, variant)
    if got is None:
        return None
    d, family, path = got
    pub = d["state"].astype(int)
    raw = d["raw_state"].astype(int)
    n_bear, n_bull = gate_for(family, variant)
    need, req = countdown(pub, raw, n_bear, n_bull)
    drv = (d["vm_alarm"] if family == "VM" else d["p_bear_raw"]).astype(float)

    gap = pd.Series(np.nan, index=d.index)
    if family == "VM":
        # the raw state IS drv >= VM_THRESH, so the distance to it is the mechanism
        gap = np.where(pub == 0, drv - VM_THRESH, VM_THRESH - drv)
        gap = pd.Series(gap, index=d.index)
    else:
        # JM: per-market thresholds, refit each January on prior transitions only
        for y in sorted({t.year for t in d.index}):
            lv = jm_levels(drv, raw, pd.Timestamp(f"{y - 1}-12-31"))
            if lv is None:
                continue
            U, L = lv
            sel = (d.index >= pd.Timestamp(f"{y}-01-01")) & (d.index <= pd.Timestamp(f"{y}-12-31"))
            gap[sel] = np.where(pub[sel] == 0, drv[sel] - U, L - drv[sel])

    return dict(market=market, family=family, path=path, d=d, pub=pub,
                need=need, req=req, gap=gap, lab=flip_labels(pub))


def fit_predict(parts):
    """Pooled per (family, need) walk-forward isotonic. Returns {market: Series P(flip)}."""
    years = sorted({y for p in parts for y in {t.year for t in p["d"].index}})
    out = {p["market"]: pd.Series(np.nan, index=p["d"].index) for p in parts}
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        # ---- assemble the pooled training set, trimming the label's lookahead
        tr = {}
        for p in parts:
            idx = p["d"].index
            usable = idx[idx <= cut]
            if len(usable) > HORIZON:
                usable = usable[:-HORIZON]
            else:
                continue
            for k in (1, 2, 3):
                m = (p["need"].loc[usable] == k) & p["gap"].loc[usable].notna() \
                    & p["lab"].loc[usable].notna()
                if not m.any():
                    continue
                key = (p["family"], k)
                g, l = p["gap"].loc[usable][m], p["lab"].loc[usable][m]
                tr.setdefault(key, [[], []])
                tr[key][0] += list(g.values)
                tr[key][1] += list(l.values)
        # ---- fit one isotonic per (family, need); fall back to the cell mean
        model, base = {}, {}
        for key, (g, l) in tr.items():
            g, l = np.asarray(g, float), np.asarray(l, float)
            base[key] = float(l.mean())
            if len(g) < MIN_FIT or l.sum() < MIN_POS or (1 - l).sum() < MIN_POS:
                continue
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                    out_of_bounds="clip")
            ir.fit(g, l)
            model[key] = ir
        # ---- predict this year
        for p in parts:
            idx = p["d"].index
            te = idx[(idx > cut) & (idx <= pd.Timestamp(f"{y}-12-31"))]
            for t in te:
                k = int(p["need"].loc[t])
                key = (p["family"], k)
                if k == 0:                       # gate already satisfied
                    out[p["market"]].loc[t] = 1.0
                    continue
                gv = p["gap"].loc[t]
                if key in model and not np.isnan(gv):
                    out[p["market"]].loc[t] = float(model[key].predict([gv])[0])
                elif key in base:
                    out[p["market"]].loc[t] = base[key]
    return out


def run(markets, variant):
    parts = [q for q in (prepare(m, variant) for m in markets) if q]
    if not parts:
        raise SystemExit("flip_calibrate: nothing to fit")
    pf = fit_predict(parts)
    for p in parts:
        d, pub = p["d"], p["pub"]
        P = pf[p["market"]]                     # P(flip AWAY from today's published state)
        # dashboard convention: bull -> p_bear = P(to bear); bear -> p_bear = 1 - P(to bull)
        d["p_bear"] = np.where(pub == 0, P, 1.0 - P)
        d["flip_p"] = P
        d["flip_need"] = p["need"]
        d["flip_req"] = p["req"]
        d["flip_gap"] = p["gap"]
        d.to_csv(p["path"])
        last = d.index[-1]
        print(f"  {p['market']:8} {p['family']}  need {int(p['need'].loc[last])}"
              f"/{int(p['req'].loc[last])}  P(flip)="
              f"{'--' if pd.isna(P.loc[last]) else format(P.loc[last], '.0%')}"
              f"  covered {P.notna().mean():.0%} of rows", flush=True)


if __name__ == "__main__":
    variant = "V17"
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--variant="):
            variant = a.split("=", 1)[1]
        else:
            args.append(a.upper())
    markets = args or [m for b in bb.BOARDS.values() for m in b]
    print(f"FLIP CALIBRATE {variant} - horizon {HORIZON} sessions, "
          f"pooled per model family ({len(markets)} markets)")
    run(markets, variant)
    print("FLIP CALIBRATE COMPLETE")
