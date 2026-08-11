"""Backtest results/regimes_{MARKET}_V6.csv and build the dashboard payload
results/payload_{MARKET}_V6.json - ported verbatim (logic unchanged) from the
research repo's backtest.py, trimmed to the 11 production markets.

Accounting: pos_t applies to the return of day t (close_{t-1} -> close_t).
pos = (state==0).shift(2): signal at close of day t, executed at close of
day t+1, first return earned day t+2. 10bps cost per switch.

Usage: python build_payloads.py [MARKET ...]   (default: all 11)
"""
import json
import os
import sys
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(DIR, "..", "results")
COST = 10 / 1e4
TD = 252
# "Changed recently" window for the board's ticker highlight (user, 2026-08-04).
# Trading days, not calendar days - over a weekend a calendar window would silently
# shrink to one session. Keep in step with RECENT_N in dashboard_template_prod.html.
RECENT_N = 3

MARKETS = ["NDX", "SPX", "HSI", "HSCEI", "KOSPI", "NIKKEI", "FTSE",
           "GOLD", "ARKQ", "MSFT", "NVDA"]

NAMES = {"NDX": "NASDAQ-100 Index (^NDX)", "HSI": "Hang Seng Index (^HSI)",
         "KOSPI": "KOSPI Composite Index (^KS11)",
         "SPX": "S&P 500 Index (^GSPC)", "HSCEI": "Hang Seng China Enterprises (^HSCEI)",
         "NIKKEI": "Nikkei 225 (^N225)", "FTSE": "FTSE 100 (^FTSE)",
         "STI": "Straits Times Index (SPDR STI ETF, ES3.SI) - proxy for ^STI",
         "EWS": "Singapore (iShares MSCI Singapore ETF, EWS) - stands in for ^STI",
         "GOLD": "Gold (COMEX futures, GC=F)", "ARKQ": "ARK Autonomous Tech & Robotics ETF (ARKQ)",
         "MSFT": "Microsoft Corp (MSFT)", "NVDA": "NVIDIA Corp (NVDA)",
         "AAPL": "Apple Inc (AAPL)", "GOOGL": "Alphabet Inc (GOOGL)",
         "AMZN": "Amazon.com Inc (AMZN)", "META": "Meta Platforms (META)",
         "TSLA": "Tesla Inc (TSLA)", "MU": "Micron Technology (MU)",
         "SMH": "VanEck Semiconductor ETF (SMH) - proxy for .SOX",
         "HK0005": "HSBC Holdings (0005.HK)", "HK0388": "HK Exchanges & Clearing (0388.HK)",
         "HK0700": "Tencent Holdings (0700.HK)", "HK0941": "China Mobile (0941.HK)",
         "HK0939": "China Construction Bank (0939.HK)",
         "HK1800": "China Communications Construction (1800.HK)",
         "HK1810": "Xiaomi Corp (1810.HK)", "HK9988": "Alibaba Group (9988.HK)",
         "SILVER": "Silver (iShares Silver Trust, SLV)",
         "WTI": "WTI Crude Oil (United States Oil Fund, USO)",
         "BTC": "Bitcoin (BTC-USD, weekdays only)",
         "ETH": "Ethereum (ETH-USD, weekdays only)",
         }
INDEX_SYMS = {"NDX", "HSI", "KOSPI", "SPX", "HSCEI", "NIKKEI", "FTSE"}
# everything else is price-quoted in its own currency; HK names in HKD, crypto in USD
# STI sits on the index board but the series IS an ETF share price, ~S$3-4, not an
# index level - so it takes a currency prefix rather than the bare index formatting,
# and the prefix says SGD outright because the number is small enough to be mistaken
# for USD.
UNITS = {"STI": "S$"}

TRAINING_START = {
    "NDX": "1985-10-02", "SPX": "1985-01-02",
    "HSI": "2002-01-01", "HSCEI": "2001-10-17", "KOSPI": "1996-12-11",
    "NIKKEI": "2002-06-10", "FTSE": "1999-01-04", "GOLD": "2008-01-01",
    "STI": "2009-01-01", "EWS": "1997-01-01",
    "ARKQ": "2014-09-30", "MSFT": "1986-03-13", "NVDA": "1999-01-22",
}

def periods(first_year):
    return [
        ("full", "Full period", None, None),
        ("split_pre", "Pre-2009 (dot-com + GFC)", None, "2008-12-31"),
        ("split_post", "Post-2009 (Fed-put + inflation eras)", "2009-01-01", None),
        ("era1", f"{first_year}–2009 (dot-com + GFC)", None, "2009-12-31"),
        ("era2", "2010–2021 (Fed-put era)", "2010-01-01", "2021-12-31"),
        ("era3", "2022–now (inflation era)", "2022-01-01", None),
    ]

CRISES = [
    ("1990 Gulf War bear", "1990-07-16", "1990-10-11"),
    ("1998 LTCM / Asia crisis", "1998-07-20", "1998-10-08"),
    ("Dot-com bust", "2000-03-10", "2002-10-09"),
    ("Global Financial Crisis", "2007-10-31", "2009-03-09"),
    ("2011 debt-ceiling / euro", "2011-07-22", "2011-10-03"),
    ("2015–16 China / oil", "2015-07-20", "2016-02-11"),
    ("Q4 2018 rate scare", "2018-08-29", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 inflation bear", "2022-01-03", "2022-12-28"),
    ("2025 tariff shock", "2025-02-19", "2025-04-08"),
]

CRISES_GOLD = [
    ("2008 GFC gold sell-off", "2008-03-18", "2008-11-13"),
    ("2010 correction", "2009-12-03", "2010-02-05"),
    ("2011–15 gold bear market", "2011-08-22", "2015-12-17"),
    ("2020–22 gold correction", "2020-08-06", "2022-09-26"),
    ("2026 pullback", "2026-01-29", "2026-06-24"),
]
CRISES_BY_SYMBOL = {"GOLD": CRISES_GOLD}


def _num(v):
    """float, or None for NaN/missing - so json.dump writes null, never NaN."""
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) else v


def metrics(strat, bh, rf):
    def side(r):
        eq = (1 + r).cumprod()
        yrs = len(r) / TD
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        vol = r.std() * np.sqrt(TD)
        ex = r - rf.reindex(r.index)
        sharpe = ex.mean() / ex.std() * np.sqrt(TD) if ex.std() > 0 else 0.0
        downside = ex.where(ex < 0, 0.0)
        dstd = np.sqrt((downside ** 2).mean()) * np.sqrt(TD)
        sortino = ex.mean() * TD / dstd if dstd > 0 else 0.0
        mdd = (eq / eq.cummax() - 1).min()
        return dict(cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
                    mdd=mdd, total=eq.iloc[-1] - 1, end10k=10000 * eq.iloc[-1])
    return {"strat": side(strat), "bh": side(bh)}


def _protection(crises):
    """Share of buy-and-hold's bear loss that the strategy avoided. HIGHER IS BETTER.

    = 1 - (avg strategy loss / avg B&H loss) over this market's own >=20% declines.
    0.47 means it gave up 47% less than holding did; 0 means it lost exactly as much;
    negative means it lost MORE. Stated this way (user, 2026-08-02) so it runs the
    same direction as profit vs B&H - both larger-is-better - rather than one column
    being good at 0.7 and the other bad at 0.7.

    Reported as 1 - ratio rather than the reciprocal: 1/ratio renders HK0939 as 33x,
    which is unreadable, while this is bounded and says plainly how much was avoided.

    Ratio of averages, not average of ratios: a single shallow episode where B&H fell
    2% would otherwise dominate the mean through its tiny denominator.
    """
    if not crises:
        return None
    bh = sum(c["bh"] for c in crises) / len(crises)
    st = sum(c["strat"] for c in crises) / len(crises)
    return None if bh == 0 else 1.0 - st / bh


def build_trades(df):
    pos = df["pos"].values
    close = df["close"].values
    dates = df.index
    trades, i, n = [], 0, len(df)
    while i < n:
        if pos[i] == 1:
            j = i
            while j < n and pos[j] == 1:
                j += 1
            buy_i, sell_i = max(i - 1, 0), j - 1
            open_trade = j >= n
            buy_p, sell_p = close[buy_i], close[sell_i]
            roi = sell_p / buy_p * (1 - COST) ** 2 - 1
            out_move = None
            if not open_trade:
                k = j
                while k < n and pos[k] == 0:
                    k += 1
                reentry_i = max(k - 1, sell_i) if k < n else n - 1
                out_move = close[reentry_i] / close[sell_i] - 1
            trades.append(dict(
                buy_date=str(dates[buy_i].date()), buy_price=round(float(buy_p), 2),
                sell_date=None if open_trade else str(dates[sell_i].date()),
                sell_price=None if open_trade else round(float(sell_p), 2),
                days=int(sell_i - buy_i), roi=float(roi),
                out_move=None if out_move is None else float(out_move),
                open=open_trade,
            ))
            if open_trade:
                trades[-1]["roi"] = float(close[n - 1] / buy_p * (1 - COST) - 1)
                trades[-1]["sell_price"] = round(float(close[n - 1]), 2)
                trades[-1]["days"] = int(n - 1 - buy_i)
            i = j
        else:
            i += 1
    return trades


def main(sym, variant="V6"):
    base = sym
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"regimes_{sym}_{variant}.csv"), index_col=0, parse_dates=True)
    df["pos"] = (df["state"] == 0).astype(float).shift(2)
    df["pos"] = df["pos"].fillna(df["state"].iloc[0] == 0 and 1.0 or 0.0)
    switch = df["pos"].diff().abs().fillna(0.0)
    df["strat_ret"] = df["pos"] * df["ret"] + (1 - df["pos"]) * df["rf"] - switch * COST
    df["bh_ret"] = df["ret"]
    df = df.dropna(subset=["strat_ret", "bh_ret"])

    PERIODS = periods(df.index[0].year)
    panels = {}
    for key, label, a, b in PERIODS:
        sl = df.loc[a:b] if (a or b) else df
        if sl.empty:
            continue
        m = metrics(sl["strat_ret"], sl["bh_ret"], sl["rf"])
        m["label"] = label
        m["start"], m["end"] = str(sl.index[0].date()), str(sl.index[-1].date())
        m["n_switches"] = int(sl["pos"].diff().abs().sum())
        m["time_in_mkt"] = float(sl["pos"].mean())
        m["n_trades"] = int((sl["pos"].diff() == 1).sum()) + int(sl["pos"].iloc[0] == 1)
        panels[key] = m

    trades = build_trades(df)

    pos_arr, dates_arr = df["pos"].values, df.index
    out_runs, i, n = [], 0, len(df)
    while i < n:
        if pos_arr[i] == 0:
            j = i
            while j < n and pos_arr[j] == 0:
                j += 1
            out_runs.append((dates_arr[i], dates_arr[j - 1]))
            i = j
        else:
            i += 1

    # ---- Crisis table, rebuilt on EACH MARKET'S OWN episodes (2026-08-01, user) ----
    # It used to walk a FIXED, NDX-derived calendar of crisis windows and find the worst
    # decline inside each. Two things were wrong with that for non-US markets:
    #   1. the window BOUNDARIES are US dates, so a market that topped earlier had its
    #      decline truncated - Hang Seng peaked Jan 2018 and bottomed Oct 2022, but the
    #      calendar chopped that single -56% slide into three separate rows, each
    #      starting long after the real peak, making the signal look far quicker than it
    #      was (2 days instead of 82);
    #   2. a crisis specific to one market appears in no window at all, so it was
    #      silently dropped.
    # Now each market's own >=20% peak-to-trough episodes ARE the rows, and a known
    # crisis name is attached when one overlaps. This is the same episode definition
    # the Model tab's summary table uses, so the two pages finally agree.
    def _own_episodes(close, thresh=0.20, win=504):
        # The reference peak resets at a 2-YEAR rolling high, not an all-time high.
        # With all-time highs, a market that takes years to recover swallows the next
        # crash whole: NDX peaked in 2000, bottomed 2002 and only regained that level in
        # Nov 2015, so the 2007-09 GFC (-54%) disappeared from its table entirely.
        # Must match v11_metrics.episodes(), which feeds the Model tab's summary.
        out, peak, pi, trough, ti = [], close.iloc[0], close.index[0], np.inf, None
        ref = close.rolling(win, min_periods=1).max()
        for i_, (d_, p_) in enumerate(close.items()):
            if p_ >= ref.iloc[i_] - 1e-12:
                if ti is not None and (peak - trough) / peak >= thresh:
                    out.append((pi, ti, trough / peak - 1))
                peak, pi, trough, ti = p_, d_, np.inf, None
            elif p_ < trough:
                trough, ti = p_, d_
        if ti is not None and (peak - trough) / peak >= thresh:
            out.append((pi, ti, trough / peak - 1))
        return out

    def _label(peak_date, trough_date):
        """Name the episode after whichever known crises it overlaps."""
        hits = [nm for nm, a_, b_ in CRISES_BY_SYMBOL.get(base, CRISES)
                if pd.Timestamp(a_) <= trough_date and pd.Timestamp(b_) >= peak_date]
        if not hits:
            return f"{peak_date.year} decline"          # market-specific, no US analogue
        return hits[0] if len(hits) == 1 else f"{hits[0]} → {hits[-1]}"

    crises = []
    for peak_date, trough_date, depth in _own_episodes(df["close"]):
        pk = df.loc[peak_date:trough_date]
        if len(pk) < 5:
            continue
        bh_dd = float((1 + pk["bh_ret"]).cumprod().iloc[-1] - 1)
        st_dd = float((1 + pk["strat_ret"]).cumprod().iloc[-1] - 1)
        pct_out = float((pk["pos"] == 0).mean())
        peak_idx = df.index.get_loc(peak_date)
        trough_idx = df.index.get_loc(trough_date)
        # exit lag: trading days from THIS MARKET'S peak to the first day out of the
        # market; re-entry lag: from its trough to the first day back in. Identical
        # definition to v11_metrics.timing(), which feeds the summary table.
        w = pk[pk["pos"] == 0]
        exit_lag = int(df.index.get_loc(w.index[0]) - peak_idx) if len(w) else None
        after = df.loc[trough_date:]
        back = after[after["pos"] == 1]
        reentry_lag = int(df.index.get_loc(back.index[0]) - trough_idx) if len(back) else None
        crises.append(dict(name=_label(peak_date, trough_date),
                           start=str(peak_date.date()), end=str(trough_date.date()),
                           peak=str(peak_date.date()), trough=str(trough_date.date()),
                           bh=bh_dd, strat=st_dd,
                           exit_lag_days=exit_lag, reentry_lag_days=reentry_lag,
                           pct_out=pct_out))

    s = df["state"].values.astype(int)
    run_starts = [0] + [i for i in range(1, len(s)) if s[i] != s[i - 1]]
    runs = [(s[a], b - a) for a, b in zip(run_starts, run_starts[1:])]
    cur_state, age = int(s[-1]), len(s) - run_starts[-1]
    durs = np.array([l for st, l in runs if st == cur_state])
    surv = durs[durs >= age]
    hazard = {}
    for horizon in (21, 63):
        hazard[f"h{horizon}"] = (float((surv < age + horizon).mean())
                                 if len(surv) >= 3 else None)
    tr_last = trades[-1] if trades else None
    if tr_last and tr_last["open"]:
        sl = df.loc[tr_last["buy_date"]:]
        peak = sl["close"].cummax()
        trade = dict(status="in", entry=tr_last["buy_date"], days=tr_last["days"],
                     entry_price=tr_last["buy_price"], roi=tr_last["roi"],
                     max_dd=float((sl["close"] / peak - 1).min()),
                     off_peak=float(sl["close"].iloc[-1] / peak.iloc[-1] - 1))
    elif tr_last:
        sl = df.loc[tr_last["sell_date"]:]
        held = df.loc[tr_last["buy_date"]:tr_last["sell_date"]]
        peak = held["close"].cummax()
        trade = dict(status="out", exit=tr_last["sell_date"], days=len(sl) - 1,
                     exit_price=tr_last["sell_price"], roi=tr_last["roi"],
                     max_dd=float((held["close"] / peak - 1).min()),
                     off_peak=float(held["close"].iloc[-1] / peak.iloc[-1] - 1),
                     move_since_exit=float(sl["close"].iloc[-1] / sl["close"].iloc[0] - 1))
    else:
        trade = None
    signal = dict(
        regime_age=age,
        n_hist_runs=int(len(durs)), n_survivors=int(len(surv)),
        med_dur=(float(np.median(durs)) if len(durs) else None),
        max_dur=(int(durs.max()) if len(durs) else None),
        **hazard, trade=trade,
        # NaN must become JSON null, not NaN: the template's flip-risk light tests
        # `pBear == null` to blank itself, and NaN passes that test in JS, yielding a
        # displayed "NaN%". Variants that legitimately have no p_bear (V13's VM
        # markets, whose alarm level is not a probability) rely on this.
        p_bear=_num(df["p_bear"].iloc[-1] if "p_bear" in df else None),
        p_bear_1w=_num(df["p_bear"].iloc[-6] if "p_bear" in df and len(df) > 6 else None),
        p_bear_2w=_num(df["p_bear"].iloc[-11] if "p_bear" in df and len(df) > 11 else None),
        p_bear_1m=_num(df["p_bear"].iloc[-22] if "p_bear" in df and len(df) > 22 else None),
        # The last RECENT_N+1 sessions of p_bear, oldest first, so the board can tell
        # whether the flip-risk light CHANGED colour recently and mark the ticker.
        # The zone thresholds live in the template's flipZone() and depend on the day's
        # own bull/bear state, so the history is shipped raw and classified there -
        # duplicating the thresholds here is how the two would silently drift apart.
        p_bear_hist=([_num(v) for v in df["p_bear"].iloc[-(RECENT_N + 1):]]
                     if "p_bear" in df and len(df) > RECENT_N else None),
        # The confirmation COUNTDOWN behind the flip number (flip_calibrate.py). `flip_req`
        # consecutive raw days are needed to publish a flip - 2 for bear, 3 for bull - and
        # `flip_need` are still outstanding. The tile states this beside the probability
        # because it is exact where the probability is estimated, and because it is what
        # visibly reverts when a setup breaks: "1 of 2 confirmed" dropping back to "0 of 2"
        # is the reader's signal that the move fell apart. It also carries the one fact a
        # bare percentage cannot - with 2+ outstanding, a flip tomorrow is impossible.
        flip_need=_num(df["flip_need"].iloc[-1] if "flip_need" in df else None),
        flip_req=_num(df["flip_req"].iloc[-1] if "flip_req" in df else None),
    )

    eq_s = (1 + df["strat_ret"]).cumprod() * 10000
    eq_b = (1 + df["bh_ret"]).cumprod() * 10000
    lam_by_year = df["lam"].groupby(df.index.year).first()

    train_start = pd.Timestamp(TRAINING_START.get(base, df.index[0]))
    test_start = df.index[0]
    training_crises = [name for name, a, b in CRISES_BY_SYMBOL.get(base, CRISES)
                       if pd.Timestamp(a) < test_start and pd.Timestamp(b) >= train_start]

    payload = dict(
        symbol=base, variant=variant, full_name=NAMES.get(base, base),
        unit=UNITS.get(base, "" if base in INDEX_SYMS else "$"),
        fixed_lam=(float(df["lam"].iloc[-1]) if df["lam"].nunique() == 1 else None),
        generated=str(pd.Timestamp.now().date()),
        start=str(df.index[0].date()), end=str(df.index[-1].date()),
        train_start=str(train_start.date()), training_crises=training_crises,
        current_state="bull" if df["state"].iloc[-1] == 0 else "bear",
        # WHAT YOU HOLD NOW, i.e. after the last bar's close - not the position DURING
        # that bar. df["pos"] is bar-level: pos[t] covers close[t-1] -> close[t], so
        # pos[-1] says whether you were invested across the final day, which is a
        # different question. The trade at close[T] is decided by state[T-1], so the
        # holding after that close is state[-2]. (Found 2026-08-02: ARKQ turned bear on
        # 07-30 and therefore sold at 07-31's close, yet the tab still read "in".)
        current_pos=("in" if (df["state"].iloc[-2] == 0) else "out")
                    if len(df) > 1 else ("in" if df["state"].iloc[-1] == 0 else "out"),
        last_close=float(df["close"].iloc[-1]),
        dates=[str(d.date()) for d in df.index],
        close=[round(float(v), 2) for v in df["close"]],
        state=[int(v) for v in df["state"]],
        pos=[int(v) for v in df["pos"]],
        eq_strat=[round(float(v), 2) for v in eq_s],
        eq_bh=[round(float(v), 2) for v in eq_b],
        panels=panels, trades=trades, crises=crises, signal=signal,
        # LOSS vs B&H: average BBM loss / average B&H loss across THIS market's own
        # >=20% declines. Computed here, from the very list the crisis table on the
        # tab is rendered from, so the summary and the tab can never disagree.
        #
        # This REPLACES "protection", which was the single worst drawdown only and
        # systematically flattered: NVDA read +44.5% protection off the 2008 crash
        # while its median episode was +0.0% and it never exited in 6 of 14 bears
        # (user, 2026-08-02). Averaging over every episode removes that.
        protection_vs_bh=_protection(crises),
        lam_by_year={str(k): _num(v) for k, v in lam_by_year.items()},
    )
    with open(os.path.join(RESULTS_DIR, f"payload_{sym}_{variant}.json"), "w") as f:
        json.dump(payload, f)
    print(f"{len(trades)} round trips; payload_{sym}_{variant}.json written.")


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = "V6"
    if args and args[0].startswith("--variant="):
        variant = args[0].split("=", 1)[1]
        args = args[1:]
    markets = [m.upper() for m in args] or MARKETS
    for m in markets:
        main(m, variant)
