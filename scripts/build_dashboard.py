"""Merge per-market payloads into the dashboard template - ported from the
research repo's make_dashboard_prod.py (unchanged logic; only paths are
baked in). Variant defaults to V6; pass --variant=V7 to build from the
ensemble's payloads instead (this is what index.html/monitor2.html use as
of the v7 switch - see daily.yml).

Usage: python build_dashboard.py [--variant=V7] OUT_PATH MARKET [MARKET:ref ...]
Example:
  python build_dashboard.py --variant=V7 ../dashboard/index.html NDX SPX HSI HSCEI KOSPI NIKKEI FTSE
  python build_dashboard.py --variant=V7 ../dashboard/monitor2.html GOLD ARKQ MSFT NVDA NDX:ref
"""
import json
import os
import sys
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(DIR, "..", "results")
DASHBOARD_DIR = os.path.join(DIR, "..", "dashboard")
DATA_DIR = os.path.join(DIR, "..", "data")

args = sys.argv[1:]
variant = "V6"
if args and args[0].startswith("--variant="):
    variant = args[0].split("=", 1)[1]
    args = args[1:]
# Which board this page IS - drives the cross-board nav's current-item highlight.
board = ""
if args and args[0].startswith("--board="):
    board = args[0].split("=", 1)[1]
    args = args[1:]
out_path = args[0]
markets = args[1:]

data = {}
for m in markets:
    m, _, flag = m.partition(":")
    key = f"{m}_{variant}"
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
html = html.replace("__BOARD_NAME__", board)
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {out_path} ({len(html):,} bytes)")
