"""Combo v2 (VB/MA/VM) walk-forward rebuild, ported from the research repo's
combo_v2_walkforward.py (2026-07-18), for the v7 ensemble (JM p_bear averaged
50/50 with this module's continuous bear-confidence, both fed through v6's
own 2/3-gate + 8-day-hold decode structure - see build_regimes_v7.py).

Why walk-forward, not the original fixed VB45/18 and VM60%: those were
selected by looking at the FULL backtest history - using them as-is would
let Combo v2 "see the answer" JM never gets. This rebuild re-derives both
walk-forward, refit annually (matching JM's own cadence):
  VB: trig/calm = 92nd/50th percentile of vol10, expanding window through
      last year-end only.
  VM_SHARE: grid-searched each year over {0.50..0.75} maximizing strategy
      Sharpe on the training window's trailing 3 years (mirrors JM's own
      VAL_YEARS=3 lambda selection).
  MA(50,200): fixed - canonical convention, not a tuned free parameter.

Continuous confidence score (mirrors Combo v2's own OR/override logic):
  vb_conf = clip((vol10-vb_calm)/(vb_trig-vb_calm), 0, 1)
  ma_conf = clip(ma_gap * 10, 0, 1)              [fixed x10 scale - TESTED
            walk-forward-percentile calibration instead and it made things
            WORSE in aggregate and much worse for NVDA specifically, so the
            fixed scale is kept deliberately, not out of neglect]
  vm_conf = clip((ushare-0.60)/(1-0.60), 0, 1) if vol10<=0.20 else 0
            [ALSO tested the real per-year grid-selected vm_share here
            instead of the 0.60 proxy - produced IDENTICAL results in every
            market, since this gate rarely binds during the ma/vb-driven
            windows that matter, so the fixed proxy is kept]
  combo_p_bear = max(vb_conf, ma_conf) * (1 - vm_conf)

Caching (mirrors build_regimes.py's model cache exactly, so pushing this to
git means no retraining is needed for Combo v2 either): each year's walk-
forward parameters (vb_trig, vb_calm, vm_share) only ever need to be derived
ONCE per year - models/{market}_combo_{year}.joblib caches them; a COMPLETE
year's params never change once cached, only the current (still-
accumulating) year is ever recomputed.
"""
import os
import numpy as np
import pandas as pd
import joblib

DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

VAL_YEARS = 3
MA_FAST, MA_SLOW = 50, 200
VM_WIN, VM_CALM, VM_GUARD = 12, 0.20, -0.15
VM_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
VM_SHARE_FIXED = 0.60
COST = 10 / 1e4


def signed_vol(ret, win=10):
    """vltup / vltdn - JM's split, brought to VM (user, 2026-08-02).

    JM has carried separate upside and downside volatility since v11; VM never did,
    and `vol10` - a plain standard deviation - is the one unsigned input it has. It
    reads MAGNITUDE only, so a +15.5% day counts exactly like -15.5%. Measured on
    MSFT 2026-07-30: total vol10 leapt 26.4% -> 83.1% on a RALLY, pinning the bear
    alarm at 1.000, while downside volatility never moved (14.5% -> 14.5%).

    Semi-deviation about zero: the downside leg uses only negative days, the upside
    leg only positive ones, so neither is contaminated by the other's moves.
    """
    r = pd.Series(ret)
    dn = r.where(r < 0, 0.0)
    up = r.where(r > 0, 0.0)
    f = np.sqrt(252)
    return (dn.rolling(win).std().to_numpy() * f, up.rolling(win).std().to_numpy() * f)


def signed_volume(close, volume, win=VM_WIN):
    """volup / voldn - the volume analogue, also mirroring JM.

    `ushare` already encoded the up-share, but only ever as a BUYER veto. Returning
    both legs lets the down leg add bear conviction instead of only the up leg
    removing it, which is what JM's volup/voldn pair does.
    """
    r = pd.Series(close).pct_change()
    v = pd.Series(volume)
    tot = v.rolling(win).sum()
    up = v.where(r > 0, 0.0).rolling(win).sum() / tot
    dn = v.where(r < 0, 0.0).rolling(win).sum() / tot
    return up.to_numpy(), dn.to_numpy()


def vb_flags(vol10, trig_arr, calm_arr):
    n = len(vol10)
    out = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        v = vol10[i]
        if not active and not np.isnan(v) and v >= trig_arr[i]:
            active = True
        elif active and not np.isnan(v) and v <= calm_arr[i]:
            active = False
        out[i] = active
    return out


def vm_flags(close, volume, vol10, p200, share_arr):
    n = len(close)
    up = pd.Series(close).pct_change() > 0
    ushare = ((pd.Series(volume).where(up, 0.0)).rolling(VM_WIN).sum()
              / pd.Series(volume).rolling(VM_WIN).sum()).to_numpy()
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(ushare[i]) or np.isnan(vol10[i]) or np.isnan(p200[i]):
            continue
        out[i] = (ushare[i] >= share_arr[i] and vol10[i] <= VM_CALM
                   and p200[i] >= VM_GUARD)
    return out


def strategy_sharpe(ret, pos):
    strat = pos * ret
    sw = pd.Series(pos).diff().abs().fillna(0.0).to_numpy()
    strat = strat - sw * COST
    sd = np.nanstd(strat)
    return np.nanmean(strat) / sd * np.sqrt(252) if sd > 0 else -np.inf


def _fit_year(vol10, ma_out, close, volume, p200, ret, idx, train_mask, val_mask):
    """The expensive path: VB percentiles (cheap) + VM_SHARE grid search
    (the part worth caching - re-evaluates the full VM/VB/MA combination
    across the grid on every candidate share value)."""
    past_vol = vol10[train_mask]
    past_vol = past_vol[~np.isnan(past_vol)]
    vb_trig = float(np.percentile(past_vol, 92))
    vb_calm = float(np.percentile(past_vol, 50))

    n = len(idx)
    best_share, best_sharpe = VM_GRID[0], -np.inf
    for share in VM_GRID:
        share_arr_val = np.full(n, share)
        vb_val = vb_flags(vol10, np.where(train_mask, vb_trig, np.nan),
                          np.where(train_mask, vb_calm, np.nan))
        vm_val = vm_flags(close, volume, vol10, p200, share_arr_val)
        out_val = (ma_out | vb_val) & ~vm_val
        pos_val = (~out_val).astype(float)
        s = strategy_sharpe(ret[val_mask], pos_val[val_mask])
        if s > best_sharpe:
            best_sharpe, best_share = s, share
    return vb_trig, vb_calm, best_share


def yearly_params(vol10, ma_out, close, volume, p200, ret, idx, year, is_complete, cache_path=None):
    """Returns (vb_trig, vb_calm, vm_share) for this year - the walk-forward-
    honest analogue of build_regimes.py's yearly(): a COMPLETE year's params
    are cached permanently (a finished year's training data never changes);
    the current year is always recomputed since its expanding training
    window grows by one day on every run."""
    if is_complete and cache_path and os.path.exists(cache_path):
        return joblib.load(cache_path)

    train_end = pd.Timestamp(f"{year - 1}-12-31")
    val_start = pd.Timestamp(f"{year - 1 - VAL_YEARS}-12-31")
    train_mask = idx <= train_end
    val_mask = train_mask & (idx > val_start)
    params = _fit_year(vol10, ma_out, close, volume, p200, ret, idx, train_mask, val_mask)

    if is_complete and cache_path:
        joblib.dump(params, cache_path)
    return params


def build_walkforward(market, close_full, volume_full, first_test, signed=False,
                      sv=None, svol=None):
    """sv   - signed VOLATILITY: downside leg drives the burst and gates the buyer veto,
              upside leg damps the burst (never the trend).
       svol - signed VOLUME: the seller share adds bear conviction, mirroring the buyer
              veto that already existed.
       `signed=True` sets both, kept so earlier callers behave unchanged."""
    """Walk-forward Combo v2 output for one market, using its OWN per-year
    cache. `close_full`/`volume_full` must already be truncated to the
    market's volume-clean start (build_regimes_v7.py handles this, matching
    JM's own VOL_START truncation)."""
    close, volume = close_full, volume_full
    ret = close.pct_change()
    vol10 = (ret.rolling(10).std() * np.sqrt(252)).to_numpy()
    # SIGNED VARIANT (v16 candidate). vol10 drives BOTH the burst confidence and the
    # gate on the buyer veto, so leaving it unsigned damages the model twice: an upside
    # spike raises bear conviction AND switches off the one input that knows direction.
    # Both uses move to the downside leg; the upside leg becomes an explicit damper.
    sv = signed if sv is None else sv
    svol = signed if svol is None else svol
    vol_dn, vol_up = signed_vol(ret.to_numpy(), 10)
    if sv:
        vol10 = vol_dn                       # burst + veto gate now read downside only
    sma_fast = close.rolling(MA_FAST).mean()
    sma_slow = close.rolling(MA_SLOW).mean()
    ma_out = (sma_fast < sma_slow).to_numpy()
    ma_gap = (sma_slow / sma_fast - 1).to_numpy()
    sma200 = close.rolling(200).mean()
    p200 = (close / sma200 - 1).to_numpy()
    up = ret > 0
    ushare = ((volume.where(up, 0.0)).rolling(VM_WIN).sum()
              / volume.rolling(VM_WIN).sum())
    volup, voldn = signed_volume(close.to_numpy(), volume.to_numpy())

    idx = close.index
    n = len(idx)
    vb_trig_arr = np.full(n, np.nan)
    vb_calm_arr = np.full(n, np.nan)
    vm_share_arr = np.full(n, np.nan)
    ret_np = ret.to_numpy()

    for year in range(first_test, idx[-1].year + 1):
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        test_end = pd.Timestamp(f"{year}-12-31")
        test_mask = (idx > train_end) & (idx <= test_end)
        train_mask = idx <= train_end
        if not test_mask.any() or train_mask.sum() < 260:
            continue
        is_complete = year < idx[-1].year
        # separate cache namespace, or the signed run would silently reuse thresholds
        # fitted on TOTAL volatility - a different quantity with a different scale
        # keyed on `sv` ONLY: svol changes the final confidence blend, not the inputs
        # _fit_year sees, so the yearly percentiles/grid are identical with or without it
        tag = "combo_sv" if sv else "combo"
        cache_path = os.path.join(MODELS_DIR, f"{market}_{tag}_{year}.joblib")
        vb_trig, vb_calm, vm_share = yearly_params(
            vol10, ma_out, close.to_numpy(), volume.to_numpy(), p200, ret_np,
            idx, year, is_complete, cache_path=cache_path)
        vb_trig_arr[test_mask] = vb_trig
        vb_calm_arr[test_mask] = vb_calm
        vm_share_arr[test_mask] = vm_share

    valid = ~np.isnan(vb_trig_arr)
    vb_out = vb_flags(vol10, vb_trig_arr, vb_calm_arr)
    vm_out = vm_flags(close.to_numpy(), volume.to_numpy(), vol10, p200,
                       np.nan_to_num(vm_share_arr, nan=VM_SHARE_FIXED))
    out = (ma_out | vb_out) & ~vm_out
    out = np.where(valid, out, True)  # untested warm-up years: conservatively "out"

    vb_conf = np.clip((vol10 - vb_calm_arr) / (vb_trig_arr - vb_calm_arr + 1e-9), 0, 1)
    ma_conf = np.clip(ma_gap * 10, 0, 1)
    vm_gap = (ushare.to_numpy() - VM_SHARE_FIXED) / (1 - VM_SHARE_FIXED)
    vm_conf = np.where(vol10 <= VM_CALM, np.clip(vm_gap, 0, 1), 0.0)
    combo_p_bear = np.maximum(np.nan_to_num(vb_conf), np.nan_to_num(ma_conf)) * (1 - np.nan_to_num(vm_conf))
    if sv or svol:
        tot = np.nan_to_num(vol_up) + np.nan_to_num(vol_dn) + 1e-9
        up_dom = np.clip((np.nan_to_num(vol_up) - np.nan_to_num(vol_dn)) / tot, 0, 1)
        # The damper applies to the BURST ONLY, never to the trend. ma_conf reads the
        # 50/200 cross, which is already signed and is a long-horizon statement; a
        # two-day rally must not cancel it. An earlier version damped the max() of both
        # and drove MSFT's alarm to 0.497 - flipping a market to bull off two days,
        # exactly the over-reaction we are trying to avoid.
        vb_signed = np.nan_to_num(vb_conf) * (1 - up_dom) if sv else np.nan_to_num(vb_conf)
        # Sellers ADD conviction: the voldn leg, mirroring the buyer veto that already
        # existed. JM has carried both legs since v11; VM only ever had the up one.
        sell = np.clip((np.nan_to_num(voldn) - VM_SHARE_FIXED) / (1 - VM_SHARE_FIXED), 0, 1)
        core = np.maximum(vb_signed, np.nan_to_num(ma_conf))
        add = 0.5 * sell * (1 - up_dom) if svol else 0.0
        combo_p_bear = np.clip(core * (1 - np.nan_to_num(vm_conf)) + add, 0, 1)
    combo_p_bear = np.where(valid, combo_p_bear, np.nan)

    result = pd.DataFrame({"close": close, "ret": ret, "out": out,
                           "combo_p_bear": combo_p_bear}, index=idx)
    return result[valid]
