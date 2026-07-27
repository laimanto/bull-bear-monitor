"""Incremental daily data updater for the Jump Model v6 pipeline.

For each ticker, appends only the rows missing since the CSV's last date
(one small yfinance call per ticker), then re-fetches the last stored date
INCLUSIVE so a provisional (intraday) close self-heals on the next run.
Adapted from this repo's original update_data.py (same self-healing pattern),
retargeted at the 11 markets + risk-free ticker the v6 pipeline needs.

v8 (2026-07-27): a bar is now only stored once its session has provably
closed - see SETTLE / drop_unsettled below. This makes the pipeline correct
at ANY run time rather than relying on the cron landing in a safe window,
and the self-heal above becomes a second line of defence instead of the
first.

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


# --- Bar-settlement guard (v8, 2026-07-27) --------------------------------
# yfinance happily returns the CURRENT, still-trading session as a completed
# daily bar. Before this guard the daily job ran while Tokyo/Seoul/Hong Kong
# were mid-session and GC=F was mid-Globex, so five of the eleven markets were
# scored on an intraday snapshot and then silently restated the next day
# (measured: KOSPI -3.58% on 2026-07-22, NIKKEI -1.25%, GOLD -1.61%, and a
# published p_bear off by up to 0.086). With the 8-day hold, a flip triggered
# by such a phantom price would be locked in for 8 trading days.
#
# So: never store a bar until its session is provably over. `SETTLE` gives,
# per ticker, when the bar DATED D becomes final, as (calendar days after D,
# UTC hour, UTC minute). Values sit comfortably past each venue's close in
# BOTH DST regimes - the cost of being late is one stale day (self-corrects
# on the next run), the cost of being early is a wrong published signal.
#
# GC=F is the awkward one: verified empirically against Yahoo (2026-07-27)
# that a GC=F bar dated D is STILL being revised at D+1 04:00 UTC - the
# 2026-07-24 bar read 4055.70 when fetched at 2026-07-25 03:49 UTC but
# settled at 4067.60. Its Globex session only finishes at 17:00 ET on D+1,
# so gold is structurally one day behind the equity markets and no schedule
# can change that. (Switching the gold tab to GLD would remove the lag and
# add ~4 years of training history - tracked separately as a model change,
# not a bug fix.)
SETTLE = {
    "^NDX": (0, 21, 30), "^GSPC": (0, 21, 30),   # 16:00 ET
    "MSFT": (0, 21, 30), "NVDA": (0, 21, 30), "ARKQ": (0, 21, 30),
    "^IRX": (0, 21, 30),
    "^FTSE": (0, 17, 30),                        # 16:30 London
    "^N225": (0, 7, 0),                          # 15:00 Tokyo
    "^KS11": (0, 7, 30),                         # 15:30 Seoul
    "^HSI": (0, 9, 0), "^HSCE": (0, 9, 0),       # 16:00 Hong Kong
    "GC=F": (1, 22, 30),                         # 17:00 ET the NEXT day
}
DEFAULT_SETTLE = (1, 22, 30)   # unknown ticker: assume the slowest case


def drop_unsettled(df, ticker, now=None):
    """Drop trailing bars whose trading session has not provably closed."""
    if df.empty:
        return df
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    days, hh, mm = SETTLE.get(ticker, DEFAULT_SETTLE)
    final_at = (df.index.normalize() + pd.Timedelta(days=days)
                + pd.Timedelta(hours=hh, minutes=mm)).tz_localize("UTC")
    return df[final_at <= now]


def download(ticker, start):
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return drop_unsettled(df[COLS].dropna(subset=["Close"]), ticker)


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
        # Either nothing new, or everything on offer is still unsettled.
        # Never write in this branch - `old[old.index < last]` would drop the
        # last stored row rather than refresh it.
        return f"{ticker}: up to date (last settled close {last.date()})"
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
