"""Incremental daily data updater for the Jump Model v6 pipeline.

For each ticker, appends only the rows missing since the CSV's last date
(one small yfinance call per ticker), then re-fetches the last stored date
INCLUSIVE so a provisional (intraday) close self-heals on the next run.
Adapted from this repo's original update_data.py (same self-healing pattern),
retargeted at the 11 markets + risk-free ticker the v6 pipeline needs.

    python update_data.py            # incremental (normal daily run)
    python update_data.py --full     # re-download full history

--full exists because yfinance's auto_adjust rescales the WHOLE history each
time a dividend/split occurs - an incrementally-stitched series slowly
drifts from a true adjusted series. Run --full occasionally (e.g. quarterly)
to re-align history; daily incremental runs are otherwise correct.

Run from scripts/.
"""
import os
import sys

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
START = "1985-01-01"

# ticker -> filename (matches build_regimes.py's SYMBOLS/VOL_START expectations)
TICKERS = {
    "^NDX": "ndx.csv", "^GSPC": "gspc.csv", "^HSI": "hsi.csv", "^HSCE": "hscei.csv",
    "^KS11": "ks11.csv", "^N225": "nikkei.csv", "^FTSE": "ftse.csv", "GC=F": "gold.csv",
    "ARKQ": "arkq.csv", "MSFT": "msft.csv", "NVDA": "nvda.csv",
    "^IRX": "irx.csv",  # 13-week T-bill discount rate, shared risk-free leg
}
COLS = ["Open", "High", "Low", "Close", "Volume"]


def download(ticker, start):
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[COLS].dropna(subset=["Close"])


def update(ticker, fname, full=False):
    path = os.path.join(DATA_DIR, fname)
    if full or not os.path.exists(path):
        df = download(ticker, START)
        if df.empty:
            return f"{ticker}: DOWNLOAD FAILED (kept existing file)"
        df.to_csv(path)
        return f"{ticker}: full history rewritten — {len(df):,} rows through {df.index[-1].date()}"

    old = pd.read_csv(path, index_col=0, parse_dates=True)
    last = old.index.max()
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
    for ticker, fname in TICKERS.items():
        try:
            msg = update(ticker, fname, full)
        except Exception as e:
            msg = f"{ticker}: ERROR {e}"
        if "FAILED" in msg or "ERROR" in msg:
            failures += 1
        print(msg, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
