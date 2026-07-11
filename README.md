# Bull-Bear Monitor — v1

A daily IN/OUT market-regime signal for six markets — **QQQ, SPY, Hang Seng,
HSCEI, Nikkei 225, FTSE 100** — published as a single self-contained dashboard:
[`dashboard/Bull_Bear_Monitor.html`](dashboard/Bull_Bear_Monitor.html)
(served at this repo's GitHub Pages URL).

## The rules (identical on every market)

| Alarm | Detects | Sell when… | Buy back when… |
|---|---|---|---|
| **VB** — volatility breaker 45/18 | Fast, violent crashes | 10-day realized volatility ≥ 45% annualized | it settles back ≤ 18% |
| **GC** — golden cross 50/200 | Slow, grinding bears | 50-day average closes below the 200-day (death cross) | it closes back above (golden cross) |
| **RC** — Recovery override | A real rebound taking hold | never sells | ≥ 10 of the last 12 sessions up AND vol ≤ 20% — buys back even while alarms are on |

All-in or all-out at daily closes: **sell** when ANY alarm fires, **buy back**
when every alarm is off — or earlier on RC. The **QQQ Maximizer** tab runs VB
alone (a QQQ-specific aggressive variant). Cash earns nothing while out; no
costs/taxes modeled. Out-of-sample honesty: expect the strategy to trail buy &
hold in long bull decades and earn its keep in extended bears — the constant
is the drawdown cut (validated across 10 indexes).

## How it runs

- `scripts/update_data.py` — appends the missing daily rows to `data/*.csv`
  (one small yfinance call per market; re-fetches the last stored row so a
  provisional close self-heals). `--full` re-downloads a series from scratch —
  worth a quarterly run for the dividend-adjusted ETFs (QQQ/SPY).
- `scripts/monitor_build.py` — recomputes all signals and rebuilds the HTML.
- `.github/workflows/daily.yml` — GitHub Actions, **Mon-Fri 22:30 UTC**
  (after the US close; Asia and Europe closed hours earlier): update → build →
  commit → deploy to GitHub Pages. Manual run: Actions tab → "Daily update &
  publish" → Run workflow.

Run locally:

```
pip install -r requirements.txt
cd scripts
python update_data.py
python monitor_build.py
# open dashboard/Bull_Bear_Monitor.html
```

## One-time repo setup

1. Settings → Pages → **Source: GitHub Actions**.
2. Actions tab → enable workflows if prompted.

## Disclaimer

Research/educational project. Nothing here is investment advice; signals are
generated from historical prices and can be wrong. Do your own diligence.
