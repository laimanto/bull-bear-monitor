"""Rewrite the template's `const V13_ALL = [...]` from v13_metrics_all.csv.

Run after build_boards.py and before build_dashboard.py. Previously this was done with
ad-hoc inline python on each rebuild, which is how the table twice ended up describing
markets whose payloads said something else.

WHAT IS AND IS NOT PUBLISHED. The metrics file still carries `prot` (single worst
drawdown) and the placebo percentiles `capP`/`protP`. Those stay in the CSV because
the JM-vs-VM assignment rule in build_boards.py needs the placebo, and because it is
the right test when judging whether a proposed model change is real. They are NOT
emitted here (user, 2026-08-02): a percentile computed on a one-episode statistic read
as a grade, and produced the NVDA contradiction - 98th percentile protection sitting
beside 6 of 14 bears missed and 0.21x profit.

What the table shows instead:
  profit vs B&H  - `cap`, renamed. Whole-period final value against holding.
  loss vs B&H    - NOT here. It comes from each market's payload, so that it is
                   computed from the very episode list the tab's crisis table renders.
"""
import json
import os
import pathlib
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = pathlib.Path(HERE, "..", "dashboard", "dashboard_template_prod.html").resolve()
VARIANT = "V13"
CSV = pathlib.Path(HERE, "..", "results", "v13_metrics_all.csv")


def rows():
    df = pd.read_csv(CSV)
    out = []
    for r in df.itertuples():
        out.append(dict(
            m=r.market, board=r.board, mdl=r.model, thin=bool(r.thin),
            yrs=round(float(r.years), 1), hist=round(float(r.hist), 1),
            need=round(float(r.need), 1),
            auc=round(r.auc, 3),
            profit=round(r.cap, 3),          # renamed: "capture" -> "profit vs B&H"
            out=round(r.out, 4), fl=round(r.fl, 1), hold=int(round(r.avgd)),
            n=int(r.n_ep), miss=int(r.missed),
            ex=(None if pd.isna(r.exitd) else int(round(r.exitd))),
            re=(None if pd.isna(r.reentry) else int(round(r.reentry)))))
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("--variant="):
            VARIANT = a.split("=", 1)[1]
            CSV = pathlib.Path(HERE, "..", "results", f"{VARIANT.lower()}_metrics_all.csv")
    data = rows()
    s = TPL.read_text(encoding="utf-8")
    new = "const V13_ALL = " + json.dumps(data) + ";"
    s, n = re.subn(r"const V13_ALL = \[.*?\];", lambda _: new, s, count=1, flags=re.S)
    assert n == 1, "V13_ALL anchor not found"
    assert s.startswith("<title>"), "template corrupted"
    TPL.write_text(s, encoding="utf-8")
    print(f"V13_ALL refreshed from {CSV.name}: {len(data)} markets, "
          f"{sum(d['thin'] for d in data)} flagged short-history")
    print("  " + ", ".join(f"{d['m']} {d['profit']:.2f}x" for d in data[:6]) + " ...")
