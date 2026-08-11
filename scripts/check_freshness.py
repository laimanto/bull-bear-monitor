"""Refuse to publish a board built on stale data.

WHY THIS EXISTS (2026-08-04)
----------------------------
The daily run of 2026-08-04 06:12 UTC committed and published boards whose
newest price was Friday 2026-07-31 - it silently missed Monday 2026-08-03 for
every US ticker, every Asian INDEX (^HSI/^HSCE/^KS11/^N225), crypto and gold.
Only the eight HK single stocks and ^FTSE picked the day up.

The cause was NOT the settlement guard in update_data.py and NOT a code bug:
^HSI and 0005.HK share the same SETTLE entry (09:00 UTC same day) and were
fetched one second apart, yet only the stock had an 08-03 bar. Yahoo simply
had not published the consolidated daily bar yet. The same lag hit the 08-03
18:24 UTC run (HK stocks still absent 10h after their close). Yahoo had
everything by 17:00 UTC the next day.

So a run can fetch nothing and still exit 0. That mattered because the
fallback cron slot is gated on "has any run SUCCEEDED today?" - a stale run
counted as success, so the retry was skipped and the miss stuck for 24h.

This script closes that hole: it compares each venue's newest stored bar
against the newest bar that venue should by now have PUBLISHED, and exits
non-zero when they disagree. A non-zero exit leaves the day's gate open, so
the later cron slots retry instead of skipping.

SETTLED vs PUBLISHED - the distinction that shapes the table below.
update_data.py's SETTLE says when a bar becomes FINAL at the venue (Tokyo
07:00 UTC, Hong Kong 09:00 UTC, ...). This file needs something different:
when the bar shows up in Yahoo. For the Asian indices that is roughly 20h
after the close, already documented in daily.yml. Keying the expectation to
settlement instead would demand Tokyo's same-day bar at any slot after 07:00
UTC - a bar Yahoo will not serve for another 20 hours - so every late slot
would fail forever. Hence one uniform rule: a bar dated D is expected to be
present once D+1 03:00 UTC has passed, which is the same reasoning that put
the primary cron at 03:30 UTC. Gold is the one exception: GC=F's Globex
session for bar D only finishes 17:00 ET on D+1, so it is structurally a day
behind and gets D+2.

Holidays deliberately trip this too - on a venue holiday there IS no new bar
and the check cannot tell "holiday" from "Yahoo is late". Both want the same
response (retry later), so the LAST slot of the day runs with ALLOW_STALE=1
and publishes whatever exists. Cost of a false trip: the board rebuilds at the
late slot instead of the early one. Cost of not having the check: a day-stale
board published as if it were current.

    python check_freshness.py                 # strict: exit 1 if anything is behind
    ALLOW_STALE=1 python check_freshness.py   # report only, always exit 0

Run from scripts/.
"""
import os
import sys

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# One representative ticker per trading venue, with the point at which a bar
# dated D is expected to be AVAILABLE from Yahoo, as (calendar days after D,
# UTC hour, UTC minute). Checking one name per venue rather than all 31 keeps
# a venue holiday to one trip instead of eight.
PUBLISHED = (1, 3, 0)        # every equity/crypto venue: D+1 03:00 UTC
GOLD_PUBLISHED = (2, 3, 0)   # GC=F settles 17:00 ET on D+1, so D+2
VENUES = [
    ("US equities", "gspc.csv",   PUBLISHED),
    ("London",      "ftse.csv",   PUBLISHED),
    ("Tokyo",       "nikkei.csv", PUBLISHED),
    ("Seoul",       "ks11.csv",   PUBLISHED),
    ("Hong Kong",   "hsi.csv",    PUBLISHED),
    # NO Singapore venue. The Singapore row is EWS, which trades on NYSE Arca and is
    # already covered by "US equities" - sti.csv (ES3.SI, on SGX) is downloaded but no
    # longer feeds a board. Listing SGX here would fail the daily run on Singapore
    # public holidays, which fall on different days from every other venue, for a file
    # nothing is built from.
    ("Crypto",      "btc.csv",    PUBLISHED),
    ("Gold",        "gold.csv",   GOLD_PUBLISHED),
]


def expected_bar(published, now):
    """Newest weekday bar that should be available from Yahoo by `now`."""
    days, hh, mm = published
    d = now.normalize().tz_localize(None)
    for _ in range(14):
        if d.dayofweek < 5:
            available_at = (d + pd.Timedelta(days=days)
                            + pd.Timedelta(hours=hh, minutes=mm)).tz_localize("UTC")
            if available_at <= now:
                return d
        d -= pd.Timedelta(days=1)
    raise RuntimeError("no published weekday found in the last 14 days")


def main():
    allow_stale = os.environ.get("ALLOW_STALE", "0") == "1"
    now = pd.Timestamp.now(tz="UTC")
    print(f"Freshness check at {now:%Y-%m-%d %H:%M} UTC "
          f"({'report-only' if allow_stale else 'strict'})")

    stale = []
    for venue, fname, published in VENUES:
        path = os.path.join(DATA_DIR, fname)
        try:
            stored = pd.read_csv(path, index_col=0, parse_dates=True).index.max()
        except Exception as e:
            print(f"  {venue:12s} {fname:12s} UNREADABLE: {e}")
            stale.append(venue)
            continue
        want = expected_bar(published, now)
        ok = stored.normalize() >= want
        print(f"  {venue:12s} {fname:12s} stored={stored.date()} "
              f"expected={want.date()}  {'ok' if ok else 'STALE'}")
        if not ok:
            stale.append(venue)

    if not stale:
        print("All venues current.")
        return 0
    msg = f"{len(stale)} venue(s) behind: {', '.join(stale)}"
    if allow_stale:
        print(f"WARNING: {msg} - publishing anyway (ALLOW_STALE=1).")
        return 0
    print(f"ERROR: {msg}. Not building - a later slot will retry.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
