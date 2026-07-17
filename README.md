# Bull-Bear Monitor — v6

A daily IN/OUT market-regime signal for 11 markets — **NASDAQ-100, S&P 500,
Hang Seng, HSCEI, KOSPI, Nikkei 225, FTSE 100, Gold (COMEX futures), ARKQ,
Microsoft, NVIDIA** — published as two self-contained dashboards:
[`dashboard/index.html`](dashboard/index.html) (the 7 indexes) and
[`dashboard/monitor2.html`](dashboard/monitor2.html) (Gold/ARKQ/MSFT/NVDA),
both served at this repo's GitHub Pages URL.

## The model

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
  small yfinance call per ticker; re-fetches the last stored row so a
  provisional close self-heals). `--full` re-downloads a series from scratch.
- `scripts/build_regimes.py` — the walk-forward + v6 decode, all 11 markets.
- `scripts/build_payloads.py` — backtests each market's regime file into the
  dashboard's JSON payload (era performance, named crises, trade log).
- `scripts/build_dashboard.py` — merges payloads + template into the final
  standalone HTML.
- `.github/workflows/daily.yml` — GitHub Actions, **00:30 UTC Tue-Sat**
  (~8pm US Eastern Mon-Fri evenings — after the US close AND gold futures'
  later settlement, hours after Asia/Europe closed): update → rebuild →
  commit → deploy to GitHub Pages. Manual run: Actions tab → "Daily update &
  publish" → Run workflow.

Run locally:

```
pip install -r requirements.txt
cd scripts
python update_data.py
python build_regimes.py
python build_payloads.py
python build_dashboard.py ../dashboard/index.html NDX SPX HSI HSCEI KOSPI NIKKEI FTSE
python build_dashboard.py ../dashboard/monitor2.html GOLD ARKQ MSFT NVDA NDX:ref
```

## One-time repo setup

1. Settings → Pages → **Source: GitHub Actions**.
2. Actions tab → enable workflows if prompted.

## Disclaimer

Research/educational project. Nothing here is investment advice; signals are
generated from historical prices and can be wrong. Do your own diligence.
