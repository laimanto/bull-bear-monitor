"""
Strategy ROI measured at bear-market trough cutoffs, 1996 -> cutoff.

Strategies (all long-only, one position):
  bh        : buy & hold from first data point
  baseline  : Bollinger band strategy, no stop
  intra25   : + 25% intraday trailing stop (peak = intraday high; trigger on
              low; fill at stop or open if gapped below)
  eod15     : + 15% trail on closes; trigger at close, sell at next open

Each strategy produces a daily equity curve (mark-to-market at close, open
positions valued at the day's close), so ROI can be read at any cutoff date.
"""

import math
import os

import pandas as pd

TICKERS = ["SPY", "AAPL", "MSFT", "AMZN", "INTC", "CSCO", "GE", "C", "KO"]
START = "1996-01-01"
WINDOW = 20
NUM_STD = 2.0

# The project is frozen at this date: full OHLCV history lives in
# ../data/<ticker>.csv and is loaded from there. yfinance is only touched
# when a ticker's CSV does not exist yet (one-time bootstrap, clipped to
# FREEZE so a later bootstrap cannot silently extend the period).
FREEZE = "2026-07-08"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

CUTOFFS = [
    ("dot-com trough", "2002-10-09"),
    ("GFC trough", "2009-03-09"),
    ("COVID trough", "2020-03-23"),
    ("2022 bear trough", "2022-10-14"),
    ("today", "2026-12-31"),
]

STRATS = ["bh", "baseline", "intra25", "eod15"]


def fetch(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:FREEZE, ["Open", "High", "Low", "Close", "Volume"]]
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path)
    df = df.dropna(subset=["Close"])
    close = df["Close"]
    mid = close.rolling(WINDOW).mean()
    std = close.rolling(WINDOW).std(ddof=0)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "open": df["Open"].to_numpy(dtype=float),
        "high": df["High"].to_numpy(dtype=float),
        "low": df["Low"].to_numpy(dtype=float),
        "close": close.to_numpy(dtype=float),
        "volume": df["Volume"].to_numpy(dtype=float),
        "mid": mid.to_numpy(dtype=float),
        "upper": (mid + NUM_STD * std).to_numpy(dtype=float),
        "lower": (mid - NUM_STD * std).to_numpy(dtype=float),
    }


def equity_curve(d, strat):
    n = len(d["close"])
    if strat == "bh":
        return [c / d["close"][0] for c in d["close"]]

    eq = [1.0] * n
    mult = 1.0
    in_pos = armed_buy = armed_sell = pending_exit = False
    entry = peak = None

    for i in range(n):
        c, m, u, l = d["close"][i], d["mid"][i], d["upper"][i], d["lower"][i]
        if math.isnan(m):
            eq[i] = mult * (c / entry if in_pos else 1.0)
            continue

        if in_pos and pending_exit:  # eod15: sell at today's open
            mult *= d["open"][i] / entry
            in_pos = armed_sell = pending_exit = False

        if not in_pos:
            if c < l:
                armed_buy = True
            elif armed_buy and c > m:
                in_pos, armed_buy, armed_sell = True, False, False
                entry = peak = c
        else:
            exited = False
            if strat == "intra25":
                stop = peak * (1 - 0.25)  # peak as of prior day
                if d["low"][i] <= stop:
                    mult *= min(d["open"][i], stop) / entry
                    in_pos = armed_sell = False
                    exited = True
                else:
                    peak = max(peak, d["high"][i])
            elif strat == "eod15":
                peak = max(peak, c)
                if c <= peak * (1 - 0.15):
                    pending_exit = True  # band exit suppressed today

            if in_pos and not (strat == "eod15" and pending_exit):
                if c > u:
                    armed_sell = True
                elif armed_sell and c < m:
                    mult *= c / entry
                    in_pos = armed_sell = False

        eq[i] = mult * (c / entry if in_pos else 1.0)
    return eq


def idx_at(dates, cutoff):
    k = -1
    for i, dt in enumerate(dates):
        if dt <= cutoff:
            k = i
        else:
            break
    return k


def main():
    data, curves = {}, {}
    for t in TICKERS:
        print(f"fetching {t} ...", flush=True)
        data[t] = fetch(t)
        curves[t] = {s: equity_curve(data[t], s) for s in STRATS}

    for label, cutoff in CUTOFFS:
        print(f"\n=== ROI % at {label} ({cutoff}), from 1996 ===")
        print(f"{'ticker':8}" + "".join(f"{s:>12}" for s in STRATS))
        geo = {s: 1.0 for s in STRATS}
        cnt = 0
        for t in TICKERS:
            k = idx_at(data[t]["dates"], cutoff)
            if k < 0:
                continue
            cnt += 1
            row = f"{t:8}"
            for s in STRATS:
                v = curves[t][s][k]
                geo[s] *= v
                row += f"{(v - 1) * 100:>12,.0f}"
            print(row)
        print(f"{'geo-avg':8}" + "".join(
            f"{(geo[s] ** (1 / cnt) - 1) * 100:>12,.0f}" for s in STRATS))


if __name__ == "__main__":
    main()
