"""
Incremental daily data updater for the Bull-Bear Monitor.

For each monitored ticker, appends only the rows missing since the CSV's
last date (one small yfinance call per ticker). This intentionally ends
the 2026-07-08 research freeze: from now on data/<ticker>.csv grows with
live closes and ec.FREEZE is only the historical bootstrap origin.

    python update_data.py            # incremental (normal daily run)
    python update_data.py --full     # re-download full history

Why --full exists: yfinance's auto_adjust rescales the WHOLE history
each time a dividend is paid (QQQ/SPY), so an incrementally-stitched
series slowly drifts from a true total-return series (~0.3%/quarter on
the ETFs; zero for the price indexes). Signals are insensitive to this,
but run --full once in a while (e.g. quarterly) to re-align history.

Run from scripts/.
"""

import os
import sys

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
START = "1996-01-01"
TICKERS = ["QQQ", "SPY", "^HSI", "^HSCE", "^N225", "^FTSE"]
COLS = ["Open", "High", "Low", "Close", "Volume"]


def download(ticker, start):
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[COLS].dropna(subset=["Close"])


def update(ticker, full=False):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if full or not os.path.exists(path):
        df = download(ticker, START)
        if df.empty:
            return f"{ticker}: DOWNLOAD FAILED (kept existing file)"
        df.to_csv(path)
        return f"{ticker}: full history rewritten — {len(df):,} rows through {df.index[-1].date()}"

    old = pd.read_csv(path, index_col=0, parse_dates=True)
    last = old.index.max()
    # re-fetch from the last stored date INCLUSIVE and overwrite that row:
    # if a previous run grabbed a provisional (intraday) close, the next
    # run self-heals it with the final value
    new = download(ticker, last.strftime("%Y-%m-%d"))
    new = new[new.index >= last]
    if new.empty:
        return f"{ticker}: up to date (last close {last.date()})"
    added = new[new.index > last]
    pd.concat([old[old.index < last], new]).to_csv(path)
    if added.empty:
        return f"{ticker}: refreshed last close ({last.date()})"
    return (f"{ticker}: +{len(added)} row(s) — "
            f"{', '.join(str(d.date()) for d in added.index)}")


def main():
    full = "--full" in sys.argv
    failures = 0
    for t in TICKERS:
        try:
            msg = update(t, full)
        except Exception as e:
            msg = f"{t}: ERROR {e}"
        if "FAILED" in msg or "ERROR" in msg:
            failures += 1
        print(msg, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
