"""Merge per-market v6 payloads into the dashboard template - ported from the
research repo's make_dashboard_prod.py (unchanged logic; only paths and the
single-variant "_V6" convention are baked in since this repo only ever runs
the current production model).

Usage: python build_dashboard.py OUT_PATH MARKET [MARKET:ref ...]
Example:
  python build_dashboard.py ../dashboard/index.html NDX SPX HSI HSCEI KOSPI NIKKEI FTSE
  python build_dashboard.py ../dashboard/monitor2.html GOLD ARKQ MSFT NVDA NDX:ref
"""
import json
import os
import sys
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(DIR, "..", "results")
DASHBOARD_DIR = os.path.join(DIR, "..", "dashboard")
DATA_DIR = os.path.join(DIR, "..", "data")

out_path = sys.argv[1]
markets = sys.argv[2:]

data = {}
for m in markets:
    m, _, flag = m.partition(":")
    key = f"{m}_V6"
    with open(os.path.join(RESULTS_DIR, f"payload_{key}.json")) as f:
        data[key] = json.load(f)
    if flag == "ref":
        data[key]["ref_only"] = True

with open(os.path.join(DASHBOARD_DIR, "dashboard_template_prod.html"), encoding="utf-8") as f:
    tpl = f.read()

history = []
hist_path = os.path.join(DATA_DIR, "JM_history.xlsx")
if os.path.exists(hist_path):
    import pandas as pd
    hdf = pd.read_excel(hist_path).dropna(how="all")
    cols = list(hdf.columns)
    history = [dict(stage=str(row[cols[0]]), org=str(row[cols[1]]),
                    who=str(row[cols[2]]), what=str(row[cols[3]]))
               for _, row in hdf.iterrows()]

html = tpl.replace("__PAYLOAD_JSON__", json.dumps(data))
html = html.replace("__HISTORY_JSON__", json.dumps(history))
html = html.replace("__BUILT_AT_ISO__", datetime.now(timezone.utc).isoformat())
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {out_path} ({len(html):,} bytes)")
