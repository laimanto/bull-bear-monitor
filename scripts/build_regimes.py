"""Statistical Jump Model v6 — full walk-forward + decode, all 11 markets.

Ported from the research repo's run_jm.py / exp_v2_masked.py / exp_v3_transfer.py
/ make_v6.py (2026-07-17), consolidated into one production script. No logic
changed from the tested/adopted versions - only paths and market-list
trimming (research variants BL/V1/V2/QQQ/SPY/FUTU dropped; only the 11
production markets kept).

Pipeline per market, per test year (expanding window, refit every January 1):
  1. Stage 1: fit a 2-state jump model on the 9-feature v1 architecture
     (EWM return / log EWM downside deviation / Sortino ratio at 5/21/63-day
     halflives, no clipping). Jump penalty re-selected each year by maximizing
     the walk-forward strategy Sharpe over the training window's last 3 years.
     Its ONLINE state path supplies a lagged bear-mask for stage 2.
  2. Stage 2: same 9 features + 3 bear-masked recovery dims (up-volume-share
     deviation at 5/21-day halflives, log distance to the 200d average - live
     only when stage 1 said bear YESTERDAY) + 2 unconditional close-location-
     in-range dims (5/21-day halflives). Same yearly lambda re-selection.
     This stage's state/bear-probability is the raw model output.
  3. Decode (v6): a flip to BEAR publishes after 2 consecutive raw bear days
     AND the model's own bear-probability >= 0.60 that day (low-conviction
     bear alarms are historically false and held back); a flip to BULL
     publishes after 3 consecutive raw bull days (exits confirm faster than
     re-entries; re-entries are deliberately not conviction-gated - every
     such filter tested costs real money at V-shaped bottoms). On top of
     both: once published, a flip holds for >= 8 trading days before another
     flip may publish (a blocked flip retries daily; it publishes the moment
     the hold expires if its raw run still stands, or never publishes if the
     raw run dies first) - this guarantees the published signal never
     reverses within 8 trading days, which is the whole point of a bull/bear
     MONITOR: a signal that flip-flops in days destroys user confidence
     regardless of backtest economics. Full reasoning and the decode studies
     that ruled out every alternative live in the research repo's memory
     notes; this file only carries the adopted result forward.

Some markets' volume data has a real Yahoo data-quality gap (all-zero) before
a cutover date, not real illiquidity - VOL_START truncates history to each
market's own volume-clean start before any feature is built.

Model caching (2026-07-17, so the DAILY run doesn't re-walk 20-40 years of
history just to add one new day): each year's fit only ever needs to happen
ONCE - the walk-forward already refits on every January 1st and holds that
one model+scaler fixed for the rest of the year, so a persisted model is
valid for every subsequent day of the same calendar year. models/{market}_
{stage}_{year}.joblib caches (scaler, lambda, model) per market/stage/year;
if a cache file exists it is loaded and only predict_online (cheap - a
forward pass through an already-fit clustering rule, not a refit) is re-run
on the now-slightly-longer feature history; if it doesn't exist (a new
year, or the one-time historical backfill), the expensive fit + 3-year
validation lambda search runs once and the result is cached before moving
on. Every subsequent daily run for that year is then just a cache hit.

Usage: python build_regimes.py [MARKET ...]   (default: all 11 production
markets). Writes results/regimes_{MARKET}_V6.csv and populates models/.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import StandardScalerPD

warnings.filterwarnings("ignore")

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "..", "data")
MODELS_DIR = os.path.join(DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HALFLIVES = (5, 21, 63)
LAMBDA_GRID = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0]
VAL_YEARS = 3
COST_BPS = 10
TRADING_DAYS = 252

# etf == index (no splice) for every production market; etf_start is each
# series' own true earliest date; first_test is chosen so the training
# window contains >=1 real crisis (checked against peak-to-trough returns).
SYMBOLS = {
    # --- 10-year decliners, added 2026-08-05 for the defensive board. NOT on any live
    # board and NOT in BOARDS: these exist so the JM/VM protocol can be scored on
    # assets that FELL, where "beat buy-and-hold" is a meaningless bar and the real
    # question is whether the timing beats holding less. first_test = listing + 6y,
    # the protocol's documented warm-up (10mo features + 2y train + 3y validate).
    "LUMN": {"etf": "lumn.csv", "index": "lumn.csv", "etf_start": "1985-01-02", "first_test": 1991},
    "CCL": {"etf": "ccl.csv", "index": "ccl.csv", "etf_start": "1987-07-24", "first_test": 1994},
    "SIRI": {"etf": "siri.csv", "index": "siri.csv", "etf_start": "1994-09-13", "first_test": 2001},
    "BIDU": {"etf": "bidu.csv", "index": "bidu.csv", "etf_start": "2005-08-05", "first_test": 2012},
    "NCLH": {"etf": "nclh.csv", "index": "nclh.csv", "etf_start": "2013-01-18", "first_test": 2019},

    "NDX":    dict(etf="ndx.csv",    index="ndx.csv",    etf_start="1985-10-02", first_test=1999),
    "SPX":    dict(etf="gspc.csv",   index="gspc.csv",   etf_start="1985-01-02", first_test=1999),
    "HSI":    dict(etf="hsi.csv",    index="hsi.csv",    etf_start="1986-12-31", first_test=2010),
    "HSCEI":  dict(etf="hscei.csv",  index="hscei.csv",  etf_start="1993-07-15", first_test=2010),
    "KOSPI":  dict(etf="ks11.csv",   index="ks11.csv",   etf_start="1996-12-11", first_test=2002),
    "NIKKEI": dict(etf="nikkei.csv", index="nikkei.csv", etf_start="1985-01-02", first_test=2010),
    "FTSE":   dict(etf="ftse.csv",   index="ftse.csv",   etf_start="1985-01-02", first_test=2004),
    # Singapore (2026-08-09). The series is the SPDR Straits Times Index ETF (ES3.SI),
    # NOT ^STI: Yahoo gives ^STI zero volume on every session of 2008-2011, 93% of 2012
    # and 99.6% of 2016, and that gap sits in the MIDDLE of the history, where VOL_START
    # (which only truncates from the front) cannot reach it - trimming to the clean 2018+
    # stretch would leave 8.6 years, less than the protocol's own warm-up. Same defect,
    # same remedy as SMH standing in for ^SOX. ES3.SI tracks the index directly: daily
    # return correlation 0.937 against ^STI, weekly 0.980, and it is quoted in SGD on
    # SGX, so unlike a US-listed proxy it carries no FX or time-zone offset.
    # Cost of the short series, stated plainly: 12.6 forecast years containing only TWO
    # >=20% episodes (2015-16 China, and the 2018->COVID slide). Read it as two
    # observations, not a track record. first_test = VOL_START + 5y, as for the HK names.
    "STI":    dict(etf="sti.csv",    index="sti.csv",    etf_start="2008-01-02", first_test=2014),
    # The SAME market through a US-listed proxy, carried as its own row next to STI so
    # the two can be compared on measured results rather than on priors (user,
    # 2026-08-09). EWS tracks MSCI Singapore, not the STI's 30 names, and is quoted in
    # USD on NYSE Arca (weekly correlation to ^STI 0.868, against ES3.SI's 0.980; the
    # daily figure of 0.569 is almost entirely the time-zone offset - SGX closes 05:00
    # ET, before the US open).
    #
    # The CURRENCY is not the problem, which is worth recording because it is the
    # obvious thing to assume: SGD/USD is 6.1% of EWS's weekly variance, and stripping
    # FX out does not improve the match to ES3.SI at all (0.857 de-FX'd vs 0.861 raw).
    # The residual gap is index composition plus the session offset.
    #
    # What EWS buys is record length: clean volume from 1997 gives 24.6 forecast years
    # spanning SIX >=20% episodes (2002, GFC, 2011, 2015-16, 2018->COVID, 2022) where
    # ES3.SI has two - and it avoided loss in ALL six (+20% to +78%).
    #
    # MEASURED 2026-08-09, and it decides which row to believe: run EWS's published
    # signal against ES3.SI's OWN returns over the common 2014+ window and it avoids
    # 14.4% of the loss at the 88th placebo percentile while sitting out only 15% of the
    # time, where ES3.SI's own signal avoids 4.6% at the 58th percentile and sits out
    # 42%. The longer record TRANSFERS. ES3.SI's own AUC of 0.815 is the AUC-vs-dollars
    # trap: it is high because the signal is absent through a long decline, and a
    # randomly re-timed copy of it scores about as well.
    "EWS":    dict(etf="ews.csv",    index="ews.csv",    etf_start="1996-03-18", first_test=2002),
    "GOLD":   dict(etf="gold.csv",   index="gold.csv",   etf_start="2000-08-30", first_test=2014),
    "ARKQ":   dict(etf="arkq.csv",   index="arkq.csv",   etf_start="2014-09-30", first_test=2019),
    "MSFT":   dict(etf="msft.csv",   index="msft.csv",   etf_start="1986-03-13", first_test=1999),
    "NVDA":   dict(etf="nvda.csv",   index="nvda.csv",   etf_start="1999-01-22", first_test=2003),
    # --- US tech dashboard (added 2026-08-01): the Magnificent 7 as INDIVIDUAL names
    # (user's choice, not a basket), plus Micron and SMH.
    # first_test is set ~5-6 years after each listing: the walk-forward needs roughly
    # 2 years to fit plus a 3-year validation window before its first honest call.
    # SMH stands in for .SOX - Yahoo reports ZERO volume on ^SOX for 7,956 of 8,115
    # days, and both models use volume features, so the index itself cannot be fitted.
    "AAPL":   dict(etf="aapl.csv",   index="aapl.csv",   etf_start="1985-01-02", first_test=1999),
    "GOOGL":  dict(etf="googl.csv",  index="googl.csv",  etf_start="2004-08-19", first_test=2010),
    "AMZN":   dict(etf="amzn.csv",   index="amzn.csv",   etf_start="1997-05-15", first_test=2003),
    "META":   dict(etf="meta.csv",   index="meta.csv",   etf_start="2012-05-18", first_test=2017),
    "TSLA":   dict(etf="tsla.csv",   index="tsla.csv",   etf_start="2010-06-29", first_test=2015),
    "MU":     dict(etf="mu.csv",     index="mu.csv",     etf_start="1985-01-02", first_test=1999),
    "SMH":    dict(etf="smh.csv",    index="smh.csv",    etf_start="2000-06-05", first_test=2006),
    # --- HK dashboard (2026-08-02). first_test is 5y past each name's VOL_START below.
    "HK0005": dict(etf="hk0005.csv", index="hk0005.csv", etf_start="2000-01-03", first_test=2010),
    "HK0388": dict(etf="hk0388.csv", index="hk0388.csv", etf_start="2000-06-27", first_test=2014),
    "HK0700": dict(etf="hk0700.csv", index="hk0700.csv", etf_start="2004-06-16", first_test=2010),
    "HK0941": dict(etf="hk0941.csv", index="hk0941.csv", etf_start="2000-01-04", first_test=2010),
    "HK0939": dict(etf="hk0939.csv", index="hk0939.csv", etf_start="2005-10-27", first_test=2016),
    "HK1800": dict(etf="hk1800.csv", index="hk1800.csv", etf_start="2006-12-15", first_test=2014),
    "HK1810": dict(etf="hk1810.csv", index="hk1810.csv", etf_start="2018-07-09", first_test=2023),
    # Alibaba: BABA (NYSE, from 2014-09) spliced to 9988.HK at its 2019-11-26 HK listing.
    # Same company, same shares - weekly return correlation 0.884 and a stable
    # 0.97-1.02 price ratio across the overlap. The daily correlation looks low (0.53)
    # only because HK closes at 04:00 ET, before the US session opens.
    # 11.9 years instead of 6.7. See hk9988_long.csv for how the join is anchored.
    "HK9988": dict(etf="hk9988_long.csv", index="hk9988_long.csv",
                   etf_start="2014-09-19", first_test=2020),
    # BABA as its own market - the US listing, one continuous series, no splice.
    # Kept alongside HK9988 to test whether the splice costs anything.
    "BABA":   dict(etf="baba.csv",   index="baba.csv",   etf_start="2014-09-19", first_test=2020),
    # --- Commodity & crypto (2026-08-02). GOLD already exists above.
    # WTI uses the USO ETF, not CL=F: the CL=F front-month settled at -$37.63 on
    # 2020-04-20, which makes pct_change (-306%) and any compounded equity meaningless
    # for a long-only backtest - capture came out at -2.503 before this swap.
    # SILVER uses the SLV ETF, not SI=F: silver futures carry 9-13% zero-volume days
    # even in recent years, which the volume features cannot survive. WTI keeps the
    # CL=F future (only 0.2% zero-volume).
    # BTC/ETH CSVs are truncated to WEEKDAYS - the pipeline annualises on 252 bars and
    # charges the cash leg irx/252 per bar, so feeding 365 bars/yr would overstate the
    # risk-free leg by ~1.45x. Weekend moves still land in the Fri->Mon return.
    "SILVER": dict(etf="silver.csv", index="silver.csv", etf_start="2006-04-28", first_test=2012),
    "WTI":    dict(etf="wti.csv",    index="wti.csv",    etf_start="2006-04-10", first_test=2012),
    "BTC":    dict(etf="btc.csv",    index="btc.csv",    etf_start="2014-09-17", first_test=2019),
    "ETH":    dict(etf="eth.csv",    index="eth.csv",    etf_start="2017-11-09", first_test=2023),
}

# Volume-clean truncation - Yahoo's volume data is a hard zero-gap (data
# quality, not real illiquidity) before these dates for these markets.
VOL_START = {
    "HSI": "2002-01-01", "HSCEI": "2001-10-17", "NIKKEI": "2002-06-10",
    "FTSE": "1999-01-04", "GOLD": "2008-01-01",
    # ES3.SI listed 2008-01-02 but Yahoo carries zero volume for all of 2008; every
    # year from 2009 on is <=1.3% zero-volume.
    "STI": "2009-01-01",
    # EWS listed 1996-03-18; 1996 carries 1.0% zero-volume days, every year after 0.0%.
    "EWS": "1997-01-01",
    # HK single names (2026-08-02): each set to the first year after which every later
    # year carries <3% zero-volume days. Yahoo's early HK volume is patchy - HKEX
    # (0388) alone had 52 zero-volume days in 2008.
    "HK0005": "2005-01-01", "HK0388": "2009-01-01", "HK0700": "2005-01-01",
    "HK0941": "2005-01-01", "HK0939": "2011-01-01", "HK1800": "2009-01-01",
    "HK1810": "2018-07-09", "HK9988": "2014-09-19",
    "BABA": "2014-09-19", "SILVER": "2006-04-28", "WTI": "2006-04-10", "BTC": "2014-09-17", "ETH": "2017-11-09",
}

# v6 decode parameters
N_BEAR, N_BULL = 2, 3
TB = 0.60
DWELL = 8


def load_data(cfg):
    etf = pd.read_csv(os.path.join(DATA_DIR, cfg["etf"]), index_col=0, parse_dates=True)
    idx = pd.read_csv(os.path.join(DATA_DIR, cfg["index"]), index_col=0, parse_dates=True)
    irx = pd.read_csv(os.path.join(DATA_DIR, "irx.csv"), index_col=0, parse_dates=True)
    etf_close = etf["Close"].astype(float)
    etf_ret = etf_close.pct_change()
    idx_ret = idx["Close"].astype(float).pct_change()
    start = pd.Timestamp(cfg["etf_start"])
    ret = pd.concat([idx_ret[idx_ret.index < start], etf_ret[etf_ret.index >= start]])
    rf = (irx["Close"].astype(float) / 100 / TRADING_DAYS).reindex(ret.index).ffill().fillna(0.0)
    return etf_close, ret, rf


def make_features(ret: pd.Series) -> pd.DataFrame:
    feats = {}
    ret_neg = np.minimum(ret, 0.0)
    for hl in HALFLIVES:
        mean = ret.ewm(halflife=hl).mean()
        dd = np.sqrt(pd.Series(ret_neg, index=ret.index).pow(2).ewm(halflife=hl).mean())
        feats[f"ret_{hl}"] = mean
        feats[f"logdd_{hl}"] = np.log(dd)
        feats[f"sortino_{hl}"] = mean.div(dd)
    X = pd.DataFrame(feats)
    return X.replace([np.inf, -np.inf], np.nan).dropna()


def strategy_sharpe(ret, rf, states):
    pos = (states == 0).astype(float).shift(2)
    pos = pos.reindex(ret.index).ffill().fillna(0.0)
    strat = pos * ret + (1 - pos) * rf - pos.diff().abs().fillna(0.0) * COST_BPS / 1e4
    excess = strat - rf
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return -np.inf
    return excess.mean() / sd * np.sqrt(TRADING_DAYS)


def eval_lambda(lam, Xf, Xv, val_mask, ret_fit, ret_val, rf_val):
    jm = JumpModel(n_components=2, jump_penalty=lam, cont=False)
    jm.fit(Xf, ret_ser=ret_fit, sort_by="cumret")
    states_val = pd.Series(jm.predict_online(Xv), index=Xv.index)[val_mask]
    return lam, strategy_sharpe(ret_val, rf_val, states_val)


def _fit_year(X, ret, rf, year):
    """The expensive path: 3-year-validation lambda search + final fit on
    the full expanding training window. Only ever called once per
    market/stage/year - yearly() below skips straight to inference if a
    cached fit already exists."""
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    X_train = X.loc[:train_end]
    val_start = pd.Timestamp(f"{year - 1 - VAL_YEARS}-12-31")
    X_fit = X_train.loc[:val_start]
    val_mask = (X_train.index > val_start)
    scale = StandardScalerPD()
    Xf = scale.fit_transform(X_fit)
    Xv = scale.transform(X_train)
    results = Parallel(n_jobs=-1)(
        delayed(eval_lambda)(lam, Xf, Xv, val_mask, ret.reindex(X_fit.index),
                             ret.reindex(X_train.index[val_mask]), rf.reindex(X_train.index[val_mask]))
        for lam in LAMBDA_GRID)
    lam, _ = max(results, key=lambda r: r[1])
    scaler = StandardScalerPD()
    Xt = scaler.fit_transform(X_train)
    jm = JumpModel(n_components=2, jump_penalty=lam, cont=False)
    jm.fit(Xt, ret_ser=ret.reindex(X_train.index), sort_by="cumret")
    return scaler, jm, lam


def yearly(X, ret, rf, year, is_complete, model_cache=None, output_cache=None):
    """Online state + bear-probability, covering all of X's index through
    this year's Dec 31 (matching the original API - callers use the shift(1)
    of the WHOLE returned series to build the next stage's mask).

    Two independent caches:
      model_cache  - the fitted (scaler, model, lambda) for this year. Once
                     a year is done, its model never needs refitting again;
                     while a year is still in progress, the SAME model
                     (fit once at the year's first run) is reused all year.
      output_cache - the (states, lam, p_bear) RESULT for a COMPLETE year.
                     A finished year's result can never change (no future
                     data alters an already-settled past year), so once
                     cached it is loaded directly and neither transform()
                     nor predict_online() run again for it at all - this is
                     what keeps a daily run from re-walking the entire
                     history just to add one new day. Never used for the
                     current (still-accumulating) year, whose result
                     legitimately changes every day.

    Caveat: caches are only valid for the data alignment they were built on.
    If update_data.py --full ever re-aligns a market's whole history (the
    dividend-adjustment drift fix noted in the README - worth doing
    quarterly), delete that market's models/{market}_*.joblib files first
    so every year rebuilds against the realigned series; otherwise the
    cached years silently keep using the pre-realignment fit."""
    if is_complete and output_cache and os.path.exists(output_cache):
        return joblib.load(output_cache)

    test_end = pd.Timestamp(f"{year}-12-31")
    if model_cache and os.path.exists(model_cache):
        scaler, jm, lam = joblib.load(model_cache)
    else:
        scaler, jm, lam = _fit_year(X, ret, rf, year)
        if model_cache:
            joblib.dump((scaler, jm, lam), model_cache)
    Xs = scaler.transform(X.loc[:test_end])
    states = pd.Series(jm.predict_online(Xs), index=Xs.index)
    c = np.asarray(jm.centers_)
    d = ((Xs.values[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    p_bear = pd.Series(d[:, 0] / (d[:, 0] + d[:, 1] + 1e-12), index=Xs.index)

    if is_complete and output_cache:
        joblib.dump((states, lam, p_bear), output_cache)
    return states, lam, p_bear


def confirm_v6(states, p_bear):
    s, p = states.values, p_bear.values
    out = s.copy()
    last_flip = -10**9
    for i in range(1, len(s)):
        n = N_BEAR if s[i] == 1 else N_BULL
        persist = all(s[i - j] == s[i] for j in range(min(n, i + 1)))
        if persist and s[i] != out[i - 1]:
            if i - last_flip < DWELL:
                persist = False
            elif s[i] == 1:
                persist = p[i] >= TB
        out[i] = s[i] if persist else out[i - 1]
        if out[i] != out[i - 1]:
            last_flip = i
    return pd.Series(out, index=states.index)


def compute_raw(market):
    """JM's raw (pre-decode) walk-forward output: close/ret/rf plus the raw
    stage-2 state path and continuous p_bear, before any decode (v6/v7/...)
    is applied. Fully cache-driven (models/{market}_stage*_{year}.joblib) -
    calling this after build_regimes.py has already run for `market` is a
    cache hit, not a recompute; used by build_regimes_v7.py so the ensemble
    build never re-walks JM's own history."""
    cfg = SYMBOLS[market]
    close, ret, rf = load_data(cfg)
    v0 = VOL_START.get(market)
    if v0:
        close, ret, rf = close.loc[v0:], ret.loc[v0:], rf.loc[v0:]

    raw = pd.read_csv(os.path.join(DATA_DIR, cfg["index"]), index_col=0, parse_dates=True)
    volume = raw["Volume"].astype(float).reindex(ret.index)
    close_full = raw["Close"].astype(float)
    h, l = (raw["High"].astype(float).reindex(ret.index),
            raw["Low"].astype(float).reindex(ret.index))

    X1 = make_features(ret)
    up_vol = volume.where(ret > 0, 0.0)
    recov = {}
    for hl in (5, 21):
        recov[f"vm{hl}"] = (up_vol.ewm(halflife=hl).mean() / volume.ewm(halflife=hl).mean() - 0.5)
    recov["ma200"] = np.log(close_full / close_full.rolling(200).mean())
    recov = pd.DataFrame(recov).reindex(X1.index)
    c = close_full.reindex(ret.index)
    clv_daily = ((c - l) / (h - l) - 0.5).fillna(0.0)
    clv_feats = pd.DataFrame({f"clv_{hl}": clv_daily.ewm(halflife=hl).mean() for hl in (5, 21)})
    base = X1.join(recov.add_prefix("raw_")).join(clv_feats).dropna()
    X1 = X1.loc[base.index]
    idx = X1.index

    first_test = cfg["first_test"]
    state_out = pd.Series(index=idx, dtype=float)
    lam_out = pd.Series(index=idx, dtype=float)
    pbear_out = pd.Series(index=idx, dtype=float)

    # A year is "complete" once a LATER year has data - i.e. every iteration
    # except the very last (idx[-1].year is always the current, still-
    # accumulating year). Complete years' output caches are permanent;
    # only the current year is ever recomputed on a later run.
    for year in range(first_test, idx[-1].year + 1):
        test_end = pd.Timestamp(f"{year}-12-31")
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        if X1.loc[str(year)].empty:
            continue
        is_complete = year < idx[-1].year
        model1 = os.path.join(MODELS_DIR, f"{market}_stage1_{year}.joblib")
        model2 = os.path.join(MODELS_DIR, f"{market}_stage2_{year}.joblib")
        out1 = os.path.join(MODELS_DIR, f"{market}_stage1_output_{year}.joblib")
        out2 = os.path.join(MODELS_DIR, f"{market}_stage2_output_{year}.joblib")
        s1, lam1, _ = yearly(X1, ret, rf, year, is_complete, model_cache=model1, output_cache=out1)
        mask = (s1.shift(1) == 1).reindex(idx).fillna(False)
        X2 = X1.copy()
        for col in ("vm5", "vm21", "ma200"):
            X2[col] = base[f"raw_{col}"].where(mask, 0.0)
        for col in ("clv_5", "clv_21"):
            X2[col] = base[col]
        s2, lam2, pb2 = yearly(X2, ret, rf, year, is_complete, model_cache=model2, output_cache=out2)
        tm = (s2.index > train_end) & (s2.index <= test_end)
        state_out.loc[s2.index[tm]] = s2[tm]
        lam_out.loc[s2.index[tm]] = lam2
        pbear_out.loc[s2.index[tm]] = pb2[tm]
        print(f"{market} {year}: lam1={lam1:>4.0f} lam2={lam2:>4.0f}  "
              f"bear_days={int((s2[tm] == 1).sum())}/{int(tm.sum())}", flush=True)

    raw_states = state_out.dropna()
    p_bear_full = pbear_out.reindex(raw_states.index)
    lam_full = lam_out.reindex(raw_states.index)
    return close, ret, rf, raw_states, p_bear_full, lam_full


def build_market(market):
    close, ret, rf, raw_states, p_bear_full, lam_full = compute_raw(market)
    published = confirm_v6(raw_states, p_bear_full)

    pd.DataFrame({"close": close.reindex(raw_states.index),
                  "ret": ret.reindex(raw_states.index),
                  "rf": rf.reindex(raw_states.index),
                  "state": published,
                  "lam": lam_full,
                  "p_bear": p_bear_full}
                 ).to_csv(os.path.join(RESULTS_DIR, f"regimes_{market}_V6.csv"))
    print(f"{market}: regimes_{market}_V6.csv written, {len(raw_states)} rows, "
          f"published flips {int(published.diff().abs().sum())}")


def main():
    markets = [m.upper() for m in sys.argv[1:]] or list(SYMBOLS)
    for m in markets:
        build_market(m)


if __name__ == "__main__":
    main()
