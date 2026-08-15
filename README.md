# Bull-Bear Monitor — BBM v17

An IN/OUT market-regime signal for 30 markets, published as four linked
self-contained dashboards (BBM v17, live 2026-08-11) served at this repo's
GitHub Pages URL:

- [`dashboard/index.html`](dashboard/index.html) — Global indexes (NDX, SPX, HSI, KOSPI, Nikkei, Singapore/EWS, FTSE)
- [`dashboard/us.html`](dashboard/us.html) — US tech (AAPL, AMZN, ARKQ, GOOGL, META, MSFT, MU, NVDA, SMH, TSLA)
- [`dashboard/hk.html`](dashboard/hk.html) — Hong Kong (0005, 0388, 0700, 0939, 0941, 1800, 1810, 9988)
- [`dashboard/commodity.html`](dashboard/commodity.html) — Commodity & crypto (Gold, Silver, WTI, BTC, ETH)

BBM is a dual-model system: per market it uses either the Jump Model (JM) or
the Volatility Model (VM), assigned by a rule evaluated on the jump model
alone. The current model write-up, per-market ratings and full trade history
are on each dashboard's **Model** tab. The daily workflow refreshes the price
data and rebuilds all four boards end-to-end (`scripts/rebuild_daily.py`);
JumpModel fits are cached in `fitcache/` keyed on the training data, so daily
runs do warm inference only and a new fit happens once per market at the
January rollover. The previous JM v6/v7 pipeline was retired 2026-08-03.

## The model (HISTORICAL — describes the earlier JM v6/v7 system)

A Statistical Jump Model (Shu, Yu & Mulvey 2024,
[arxiv.org/abs/2402.05272](https://arxiv.org/abs/2402.05272)) walk-forward:
refit every January on an expanding window, online (day-ahead) inference for
the coming year — day t's call uses only data through day t. Two features
stages: stage 1 clusters 9 trend/downside-risk/Sortino features (no clip)
into a rough bull/bear call that supplies a lagged bear-mask; stage 2 adds
3 bear-masked recovery features (up-volume-share deviation, distance to the
200-day average) and 2 unconditional close-location-in-range features. The
jump penalty (switching cost) is re-selected every January on the training
window's last 3 years.

**v6 decode**, on top of the raw model call:
- A flip to **BEAR** publishes after 2 consecutive raw bear days *and* the
  model's own bear-probability ≥ 60% that day — low-conviction bear alarms
  are historically false and held back (they still publish the moment
  conviction arrives; genuine bear flips pass with zero added delay 91% of
  the time).
- A flip to **BULL** publishes after 3 consecutive raw bull days. Re-entries
  are deliberately *not* conviction-gated — every such filter tested costs
  real money at V-shaped bottoms, where the first rebound days are the most
  valuable of the whole cycle.
- Once published, a flip **holds for at least 8 trading days** before
  another flip may publish. This is the whole point of a bull/bear
  *monitor*: a signal that reverses within days destroys user confidence
  regardless of backtest economics. A blocked flip is retried daily and
  publishes the moment the hold expires if it still stands — genuine moves
  are delayed, never lost.

Net of 10bps switch costs and a 1-day execution delay (signal at close t →
trade at close t+1), cash earns the 13-week T-bill while out.

## How it runs

- `scripts/update_data.py` — appends missing daily rows to `data/*.csv` (one
  small yfinance call per ticker, 34 of them; re-fetches the last stored row so
  a provisional close self-heals). A **settlement guard** refuses to store any
  bar whose session has not provably closed, per venue — that is what keeps
  partial intraday bars out of the history, and why gold (`GC=F`, Globex
  settling 17:00 ET on D+1) is structurally a day behind the equity markets.
  `--full` re-downloads a series from scratch.
- `scripts/splice_hk9988.py` — rebuilds the derived `data/hk9988_long.csv`
  (BABA spliced to 9988.HK). Must run after every data refresh or the scored
  file goes stale.
- `scripts/check_freshness.py` — fails before the build if any venue is behind,
  so a green run cannot publish a stale board. `ALLOW_STALE=1` downgrades it to
  a warning; the last slot of the day sets it, because a venue holiday is
  indistinguishable from a Yahoo delay and the day still has to publish.
- `scripts/rebuild_daily.py` — **the whole build, one command**, and the order
  is load-bearing (its docstring says why each step cannot move):
  `build_v11` (JM walk-forward) → `build_boards` (JM vs VM per market) →
  `flip_calibrate` (P(flip within 5 sessions)) → `build_payloads` →
  `refresh_v13_all` → `build_dashboard` ×4 → **verify**, which renders every
  page in headless Chrome and asserts on the real DOM. Every past breakage of
  this template still produced a "successful" build, hence the last step.
- JumpModel fits live in `fitcache/`, keyed on the training data itself, so
  daily runs do inference only — expect a new fit once per market at the
  January rollover. `MAX_NEW_FITS` aborts the run rather than training for
  hours in CI if the cache stops matching the data; rebuild it locally and
  commit the refreshed cache instead. VM's combo fits cache in `models/`.
- `scripts/notify_changes.py` — after the build, lists the markets whose signal
  (bull ↔ bear) or flip-risk light (green/amber/red) moved since the last board
  went out, and the workflow mails that list by opening an issue. The comparison
  is against `results/signal_state.json`, the readings last reported, so a change
  is mailed once on the day it appears no matter which cron slots ran. The
  traffic-light thresholds are parsed out of the template's `flipZone()` rather
  than copied, and the script fails the run if it cannot find them. To test the
  alert on a quiet day, run the workflow manually with **sample_notice** ticked:
  it mails a marked SAMPLE and leaves the baseline untouched.
- `.github/workflows/daily.yml` — GitHub Actions, **03:30 UTC Tue-Sat**, the
  first moment all four venues' day-D bars are both settled and published, with
  retry slots at 05:30, 07:30 and 13:00 UTC. Sequence: update → freshness gate →
  rebuild → notify → commit → deploy to GitHub Pages. Manual run: Actions tab →
  "Daily update & publish" → Run workflow.
  **"Data not published yet" is a green skip, not a failure** — Yahoo does not
  publish the BTC-USD bar for day D until ~D+1 07:00-10:00 UTC, so the first two
  slots structurally cannot pass the gate and the board normally goes out at
  07:30. A run that finds the data unready therefore ends green having done
  nothing, which means **a successful run no longer implies a board was
  published**: the retry gate asks whether `main` carries today's
  `Daily update <UTC date>` commit. That message is the receipt — do not change
  its prefix.

Run locally — one command rebuilds all four boards:

```
pip install -r requirements.txt
cd scripts
python update_data.py && python splice_hk9988.py
python rebuild_daily.py
```

## One-time repo setup

1. Settings → Pages → **Source: GitHub Actions**.
2. Actions tab → enable workflows if prompted.

## Disclaimer

Research/educational project. Nothing here is investment advice; signals are
generated from historical prices and can be wrong. Do your own diligence.
