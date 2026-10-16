"""CLOSE TO FLIP: how far along the road to a flip each market is, 0-100%.

WHY THIS REPLACED THE 5-SESSION PROBABILITY (user, 2026-10-16)
-------------------------------------------------------------
The tile showed P(flip within 5 sessions), fitted isotonic on (countdown, distance). It read
1% or 4% so often it looked hard-coded - 14 of the 20 VM markets showed one of those two
numbers on 2026-08-14. Three causes, all measured:
  1. VM's alarm is CLIPPED: exactly 0 on 43% of days, exactly 1 on 13%. A pinned alarm gives
     every market the same distance to the 0.60 line, so the same output (1.5% / 3.8%).
  2. Isotonic regression is a staircase, so nearby inputs share a step.
  3. Over 5 sessions the honest answer in a quiet market really is ~2% (1.6% of pinned-bull
     days flip). True, but it cannot tell MU from ARKQ.

And the reading the user actually wanted - "as it gets closer to the threshold, the number
climbs toward 100%" - cannot be a next-day probability at all. Measured on VM:
    signal right at the line (within 0.02)   crosses next day only 36% - it wobbles
    crossed, 2+ confirming days outstanding  flips next day 0.000% - the gate forbids it
    crossed, 1 confirming day outstanding    flips next day 90.5%
A next-day probability goes 0 -> 0 -> 0 -> 90%. It does not climb; it jumps on the last day.

THE READING: a progress measure, not a forecast
-----------------------------------------------
    0-70%   APPROACH  the driver's distance to the line that turns the raw state.
                      70 * (1 - distance / D0), capped at 69 so it never reads "crossed";
                      past 0.9*D0 (7%) it decays smoothly instead of hitting 0, because
                      0% reads as "it only just flipped" (user, 2026-10-17).
   70-100%  CONFIRM   once crossed, each confirming day fills an equal share:
                      to bear (2 days): 85 -> 100;  to bull (3 days): 80 -> 90 -> 100.

WHY THE LINE SITS AT 70 (user, 2026-10-16): once the signal crosses, the flip usually goes
through - 90% of 797 crossings toward bear published, 76% of 932 toward bull. Reaching the
line is most of the journey, and the confirm steps then land near the real odds (first bull
crossing reads 80% vs 76% that complete; first bear crossing 85% vs 90%).

The approach half is SMOOTH because the distance uses the UNCLIPPED driver:
  VM  `vm_alarm_u` (combo_v2's raw legs carried past the 0/1 clip; equal to the alarm inside
      (0,1), and on the same side of 0.60 everywhere, so it never contradicts raw_state)
  JM  p_bear against per-market fitted crossing levels (U to bear, L to bull), refit each
      January on prior transitions; markets without enough of their own use the pooled JM
      levels.
D0 = the 95th percentile of approach distances, per model family, pooled across markets and
refit each January on prior years only - so a year's scale is fixed once set and the "1 wk
ago" numbers do not drift. Years before enough history exists use the first fitted scale.

IT IS NOT A PROBABILITY and the tile must not call it a chance. At 65% the signal is near
the line, yet it crosses next day only about a third of the time. The countdown line beside
it ("1 of 3 bull days confirmed - one more does it") is what states tomorrow exactly.

OUTPUT CONVENTION (unchanged, so flipPct / flipZone / p_bear_hist / the mail need no
re-plumbing): bull day -> p_bear = closeness to bear; bear day -> p_bear = 1 - closeness to
bull. The raw driver is preserved in `p_bear_raw` before p_bear is overwritten, so re-running
is idempotent.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

import build_boards as bb                            # noqa: E402
import decode as dc                                  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
LINE = 0.70            # where "crossed the trigger" sits on the 0-100 scale
APPROACH_CAP = 0.69    # the approach half never reads as crossed
SCALE_Q = 95           # D0 = this percentile of approach distances
TAIL_X, TAIL_K = 0.9, 0.5   # beyond 0.9*D0 (7%), decay instead of reaching 0
MIN_SCALE_ROWS = 2000  # pooled approach rows needed before a year's D0 is fitted
MIN_TR = 5             # raw transitions of each kind needed to place a JM level
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


def transitions(drv, raw, upto):
    """Driver values at this market's raw transitions up to `upto`: (to bear, to bull)."""
    s, d = raw.loc[:upto], drv.loc[:upto]
    ch = s.diff().fillna(0)
    return d[ch == 1].dropna().values, d[ch == -1].dropna().values


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
    got = _load(market, variant)
    if got is None:
        return None
    d, family, path = got
    pub = d["state"].astype(int)
    raw = d["raw_state"].astype(int)
    n_bear, n_bull = gate_for(family, variant)
    need, req = countdown(pub, raw, n_bear, n_bull)
    if family == "VM":
        col = "vm_alarm_u" if "vm_alarm_u" in d and d["vm_alarm_u"].notna().any() else "vm_alarm"
        drv = d[col].astype(float)
    else:
        drv = d["p_bear_raw"].astype(float)
    return dict(market=market, family=family, path=path, d=d, pub=pub, raw=raw,
                need=need, req=req, drv=drv)


def distances(parts):
    """Distance from the driver to the line that would turn the raw state, toward a flip.
    VM: the line is 0.60 itself. JM: per-market levels refit each January (pooled JM
    levels where a market has too few transitions of its own)."""
    for p in parts:
        p["dist"] = pd.Series(np.nan, index=p["d"].index)
        if p["family"] == "VM":
            p["dist"][:] = np.where(p["pub"] == 0, VM_THRESH - p["drv"], p["drv"] - VM_THRESH)
    jm = [p for p in parts if p["family"] == "JM"]
    years = sorted({y for p in jm for y in p["d"].index.year})
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        own, pool_b, pool_u = {}, [], []
        for p in jm:
            tb, tu = transitions(p["drv"], p["raw"], cut)
            pool_b += list(tb)
            pool_u += list(tu)
            if len(tb) >= MIN_TR and len(tu) >= MIN_TR and np.median(tb) > np.median(tu):
                own[p["market"]] = (float(np.median(tb)), float(np.median(tu)))
        pooled = ((float(np.median(pool_b)), float(np.median(pool_u)))
                  if len(pool_b) >= MIN_TR and len(pool_u) >= MIN_TR else None)
        for p in jm:
            lv = own.get(p["market"], pooled)
            if lv is None:
                continue
            U, L = lv
            idx = p["d"].index
            sel = idx.year == y
            drv, pub = p["drv"][sel], p["pub"][sel]
            p["dist"][sel] = np.where(pub == 0, U - drv, drv - L)


def closeness(parts):
    """The 0-100% reading per market (as 0-1). D0 per family, walk-forward by year."""
    out = {}
    for fam in {p["family"] for p in parts}:
        fp = [p for p in parts if p["family"] == fam]
        years = sorted({y for p in fp for y in p["d"].index.year})
        scale = {}
        for y in years:
            cut = pd.Timestamp(f"{y - 1}-12-31")
            pool = np.concatenate([
                p["dist"][(p["d"].index <= cut) & (p["need"] == p["req"]).values].dropna()
                .clip(lower=0).values for p in fp])
            if len(pool) >= MIN_SCALE_ROWS:
                scale[y] = float(np.percentile(pool, SCALE_Q))
        if not scale:
            raise SystemExit(f"flip_calibrate: no {fam} scale could be fitted")
        first = scale[min(scale)]
        for y in years:                      # early years borrow the first fitted scale
            scale.setdefault(y, first)
        print(f"  {fam} scale D0 (distance that reads 0%): " + ", ".join(
            f"{y}:{scale[y]:.2f}" for y in sorted(scale) if y >= max(scale) - 2), flush=True)
        for p in fp:
            idx = p["d"].index
            D0 = pd.Series([scale[y] for y in idx.year], index=idx)
            need, req = p["need"], p["req"]
            x = p["dist"].clip(lower=0) / D0
            appr = LINE * (1 - x)
            # NO HARD ZERO (user, 2026-10-17): 0% reads as "it only just flipped". Past
            # TAIL_X the straight line bends into a decaying tail that meets it at the same
            # value, so the far end still moves with the market and only approaches 0.
            tail = LINE * (1 - TAIL_X) * np.exp(-(x - TAIL_X) / TAIL_K)
            appr = appr.where(x <= TAIL_X, tail).clip(upper=APPROACH_CAP)
            conf = LINE + (1 - LINE) * (req - need) / req
            c = np.where(need == req, appr, np.where(need <= 0, 1.0, conf))
            out[p["market"]] = pd.Series(c, index=idx)
    return out


def run(markets, variant):
    parts = [q for q in (prepare(m, variant) for m in markets) if q]
    if not parts:
        raise SystemExit("flip_calibrate: nothing to compute")
    distances(parts)
    cl = closeness(parts)
    for p in parts:
        d, pub = p["d"], p["pub"]
        C = cl[p["market"]]                    # closeness to flipping AWAY from today's state
        d["p_bear"] = np.where(pub == 0, C, 1.0 - C)
        d["flip_p"] = C                        # kept name: now closeness, not a probability
        d["flip_need"] = p["need"]
        d["flip_req"] = p["req"]
        d["flip_gap"] = -p["dist"]             # old sign convention: >0 = past the line
        d.to_csv(p["path"])
        last = d.index[-1]
        print(f"  {p['market']:8} {p['family']}  need {int(p['need'].loc[last])}"
              f"/{int(p['req'].loc[last])}  close to flip="
              f"{'--' if pd.isna(C.loc[last]) else format(C.loc[last], '.0%')}"
              f"  covered {C.notna().mean():.0%} of rows", flush=True)


if __name__ == "__main__":
    variant = "V17"
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--variant="):
            variant = a.split("=", 1)[1]
        else:
            args.append(a.upper())
    markets = args or [m for b in bb.BOARDS.values() for m in b]
    print(f"FLIP CLOSENESS {variant} - line at {LINE:.0%}, scale = p{SCALE_Q} of approach "
          f"distance per model family ({len(markets)} markets)")
    run(markets, variant)
    print("FLIP CLOSENESS COMPLETE")
