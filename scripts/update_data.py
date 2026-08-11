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
    # --- the three boards added for v13 (2026-08-02) -------------------------
    # These were downloaded once when their boards were built and then never
    # refreshed, because this dict was the ONLY place a ticker gets registered
    # and nobody extended it. Result: the original 12 tracked the daily job
    # while the other 18 froze on their download date. Register them here or
    # they silently go stale again.
    "AAPL": "aapl.csv", "GOOGL": "googl.csv", "AMZN": "amzn.csv", "META": "meta.csv",
    "TSLA": "tsla.csv", "MU": "mu.csv", "SMH": "smh.csv",
    "0005.HK": "hk0005.csv", "0388.HK": "hk0388.csv", "0700.HK": "hk0700.csv",
    "0941.HK": "hk0941.csv", "0939.HK": "hk0939.csv", "1800.HK": "hk1800.csv",
    "1810.HK": "hk1810.csv", "9988.HK": "hk9988.csv", "BABA": "baba.csv",
    "SLV": "silver.csv",    # not SI=F: futures carry 9-13% zero-volume days
    "USO": "wti.csv",       # not CL=F: front-month settled at -$37.63 in 2020
    "BTC-USD": "btc.csv", "ETH-USD": "eth.csv",   # truncated to weekdays below
    # Singapore (2026-08-09): the SPDR Straits Times Index ETF, NOT ^STI. Yahoo
    # reports ZERO volume on ^STI for every session of 2008-2012 and most of
    # 2015-17 (100% of 2008-2011, 93% of 2012, 99.6% of 2016), and both models
    # are volume-driven, so the index itself cannot be fitted - the same defect
    # that put SMH on the board in place of ^SOX. ES3.SI tracks STI directly
    # (daily return correlation 0.937, weekly 0.980) and its volume is clean
    # from 2009 on (<=1.3% zero-volume in every year since).
    "ES3.SI": "sti.csv",
    # ...and the same market seen through a US-listed proxy, carried ALONGSIDE STI for
    # comparison (user, 2026-08-09). EWS holds MSCI Singapore rather than the STI's 30
    # names and is quoted in USD on NYSE Arca - weekly return correlation against ^STI
    # is 0.868, against ES3.SI's 0.980. The currency is NOT what costs it that: SGD/USD
    # is 6.1% of its weekly variance and removing FX does not improve the match. What
    # it buys is history: clean volume from 1997 and 6 testable bear episodes against
    # ES3.SI's 2. See build_regimes.SYMBOLS for the test that decides between them.
    "EWS": "ews.csv",
}
# HK9988 is scored off hk9988_long.csv (BABA spliced to 9988.HK), which is REBUILT
# from baba.csv + hk9988.csv by splice_hk9988.py - run that after this script.
SPLICED = {"hk9988_long.csv"}
# Crypto trades 7 days a week, but the pipeline annualises on 252 bars and charges
# cash at irx/252, so weekend bars would distort both. Drop them at the source.
WEEKDAY_ONLY = {"BTC-USD", "ETH-USD"}
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
    # US equities and US-listed ETFs/ADRs all close 16:00 ET
    "AAPL": (0, 21, 30), "GOOGL": (0, 21, 30), "AMZN": (0, 21, 30), "META": (0, 21, 30),
    "TSLA": (0, 21, 30), "MU": (0, 21, 30), "SMH": (0, 21, 30), "BABA": (0, 21, 30),
    "SLV": (0, 21, 30), "USO": (0, 21, 30),
    # Hong Kong closes 16:00 HKT = 08:00 UTC
    "0005.HK": (0, 9, 0), "0388.HK": (0, 9, 0), "0700.HK": (0, 9, 0), "0941.HK": (0, 9, 0),
    "0939.HK": (0, 9, 0), "1800.HK": (0, 9, 0), "1810.HK": (0, 9, 0), "9988.HK": (0, 9, 0),
    # Crypto never closes; a bar dated D is final at 00:00 UTC on D+1
    "BTC-USD": (1, 0, 30), "ETH-USD": (1, 0, 30),
    "ES3.SI": (0, 10, 0),                        # SGX closes 17:00 SGT = 09:00 UTC
    "EWS": (0, 21, 30),                          # NYSE Arca, 16:00 ET
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
    df = df[COLS].dropna(subset=["Close"])
    if ticker in WEEKDAY_ONLY:
        df = df[df.index.dayofweek < 5]
    return drop_unsettled(df, ticker)


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
