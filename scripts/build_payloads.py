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

MARKETS = ["NDX", "SPX", "HSI", "HSCEI", "KOSPI", "NIKKEI", "FTSE",
           "GOLD", "ARKQ", "MSFT", "NVDA"]

NAMES = {"NDX": "NASDAQ-100 Index (^NDX)", "HSI": "Hang Seng Index (^HSI)",
         "KOSPI": "KOSPI Composite Index (^KS11)",
         "SPX": "S&P 500 Index (^GSPC)", "HSCEI": "Hang Seng China Enterprises (^HSCEI)",
         "NIKKEI": "Nikkei 225 (^N225)", "FTSE": "FTSE 100 (^FTSE)",
         "GOLD": "Gold (COMEX futures, GC=F)", "ARKQ": "ARK Autonomous Tech & Robotics ETF (ARKQ)",
         "MSFT": "Microsoft Corp (MSFT)", "NVDA": "NVIDIA Corp (NVDA)"}
INDEX_SYMS = {"NDX", "HSI", "KOSPI", "SPX", "HSCEI", "NIKKEI", "FTSE"}

TRAINING_START = {
    "NDX": "1985-10-02", "SPX": "1985-01-02",
    "HSI": "2002-01-01", "HSCEI": "2001-10-17", "KOSPI": "1996-12-11",
    "NIKKEI": "2002-06-10", "FTSE": "1999-01-04", "GOLD": "2008-01-01",
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


def main(sym):
    base, variant = sym, "V6"
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"regimes_{sym}_V6.csv"), index_col=0, parse_dates=True)
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

    crises = []
    for name, a, b in CRISES_BY_SYMBOL.get(base, CRISES):
        sl = df.loc[a:b]
        if sl.empty:
            continue
        roll_max = sl["close"].cummax()
        dd = sl["close"] / roll_max - 1
        trough_date = dd.idxmin()
        peak_date = sl["close"].loc[:trough_date].idxmax()
        pk = df.loc[peak_date:trough_date]
        bh_dd = float((1 + pk["bh_ret"]).cumprod().iloc[-1] - 1)
        st_dd = float((1 + pk["strat_ret"]).cumprod().iloc[-1] - 1)
        pct_out = float((pk["pos"] == 0).mean())
        peak_idx, trough_idx = df.index.get_loc(peak_date), df.index.get_loc(trough_date)
        b_ts = df.index[df.index <= pd.Timestamp(b)].max() if len(df.index[df.index <= pd.Timestamp(b)]) else None
        search = df.loc[peak_date:b]
        out_days = search.index[search["pos"] == 0]
        exit_run = None
        if len(out_days):
            target = peak_date if out_days[0] == peak_date else out_days[0]
            for r_start, r_end in out_runs:
                if r_start <= target <= r_end:
                    exit_run = (r_start, r_end)
                    break
        if exit_run is not None:
            exit_lag = df.index.get_loc(exit_run[0]) - peak_idx
            candidates = [df.index.get_loc(r_end) + 1 for r_start, r_end in out_runs
                          if exit_run[0] <= r_start <= b_ts and df.index.get_loc(r_end) + 1 < n]
            if candidates:
                reentry_idx = min(candidates, key=lambda x: abs(x - trough_idx))
                reentry_lag = reentry_idx - trough_idx
            else:
                reentry_lag = None
        else:
            exit_lag = reentry_lag = None
        crises.append(dict(name=name, start=a, end=b,
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
        p_bear=(float(df["p_bear"].iloc[-1]) if "p_bear" in df else None),
        p_bear_1w=(float(df["p_bear"].iloc[-6]) if "p_bear" in df and len(df) > 6 else None),
        p_bear_2w=(float(df["p_bear"].iloc[-11]) if "p_bear" in df and len(df) > 11 else None),
        p_bear_1m=(float(df["p_bear"].iloc[-22]) if "p_bear" in df and len(df) > 22 else None),
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
        unit="" if base in INDEX_SYMS else "$",
        fixed_lam=(float(df["lam"].iloc[-1]) if df["lam"].nunique() == 1 else None),
        generated=str(pd.Timestamp.now().date()),
        start=str(df.index[0].date()), end=str(df.index[-1].date()),
        train_start=str(train_start.date()), training_crises=training_crises,
        current_state="bull" if df["state"].iloc[-1] == 0 else "bear",
        current_pos="in" if df["pos"].iloc[-1] == 1 else "out",
        last_close=float(df["close"].iloc[-1]),
        dates=[str(d.date()) for d in df.index],
        close=[round(float(v), 2) for v in df["close"]],
        state=[int(v) for v in df["state"]],
        pos=[int(v) for v in df["pos"]],
        eq_strat=[round(float(v), 2) for v in eq_s],
        eq_bh=[round(float(v), 2) for v in eq_b],
        panels=panels, trades=trades, crises=crises, signal=signal,
        lam_by_year={str(k): float(v) for k, v in lam_by_year.items()},
    )
    with open(os.path.join(RESULTS_DIR, f"payload_{sym}_V6.json"), "w") as f:
        json.dump(payload, f)
    print(f"{len(trades)} round trips; payload_{sym}_V6.json written.")


if __name__ == "__main__":
    markets = [m.upper() for m in sys.argv[1:]] or MARKETS
    for m in markets:
        main(m)
