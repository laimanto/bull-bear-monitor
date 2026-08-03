"""The daily BBM v16 build - one command, run by .github/workflows/daily.yml.

Adapted from the research repo's rebuild_all.py. ORDER MATTERS, and getting it
wrong has produced silent, plausible-looking wrong pages more than once:

  1. build_v11.py         JM walk-forward, all markets -> regimes_{M}_V11.csv.
                          Warm from fitcache/ (inference only, seconds/market);
                          a NEW fit happens only at the January rollover. The
                          MAX_NEW_FITS env guard aborts if the cache stops
                          matching the data rather than training for hours in CI.
  2. build_boards.py      JM vs VM per market -> regimes_{M}_V16.csv + metrics
  3. build_payloads.py    per-market tab data, INCLUDING loss_vs_bh
  4. refresh_v13_all.py   inject the metrics into the template's V13_ALL
  5. build_dashboard.py   x4 -> dashboard/{index,us,hk,commodity}.html
  6. verify               headless render of each page, assert on the real DOM

Skipping step 4 leaves the summary table describing the previous run while the
tabs show the new one. Running step 5 before step 3 builds pages from stale
payloads. Both have happened; neither announced itself.

VERIFICATION is not optional. Every past breakage of this template still
produced a "successful" build - the failures were runtime failures or plain
wrong numbers, so each page is rendered in headless Chrome and the rendered DOM
is checked: row counts, row order, every market rated, no NaN/undefined, no
stray text before the body.
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.join(HERE, "..", "dashboard")
sys.path.insert(0, HERE)
from build_boards import BOARDS, BOARD_TITLE          # noqa: E402

# board key -> (page title for the nav highlight, LIVE output filename)
PAGES = {"index": ("Global indexes", "index.html"),
         "us": ("US tech", "us.html"),
         "hk": ("Hong Kong", "hk.html"),
         "cc": ("Commodity & crypto", "commodity.html")}
VARIANT = "V16"


def chrome():
    for c in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
        p = shutil.which(c)
        if p:
            return p
    win = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if os.path.exists(win):
        return win
    raise SystemExit("no Chrome/Chromium found for page verification")


def run(cmd, label):
    print(f"\n>>> {label}", flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    for line in (r.stdout or "").strip().splitlines()[-4:]:
        print("    " + line, flush=True)
    if r.returncode:
        print((r.stderr or "")[-1500:])
        raise SystemExit(f"FAILED: {label}")


def verify(board, fname, browser):
    """Render the real page and assert on the real DOM. Nothing else is proof."""
    src = os.path.abspath(os.path.join(DASH, fname))
    out = subprocess.run([browser, "--headless", "--disable-gpu", "--no-sandbox",
                          "--virtual-time-budget=9000", "--dump-dom",
                          "file:///" + src.replace("\\", "/")],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    s = out.stdout or ""
    want = BOARDS[board]
    problems = []
    if not s.lstrip().lower().startswith("<!doctype"):
        problems.append("stray content before <!doctype>")
    body = re.search(r"<body[^>]*>(.*)", s, re.S)
    if body and body.group(1)[:body.group(1).index("<")].strip():
        problems.append("stray text at top of <body>")
    tbl = re.findall(r'<table class="perftbl">(.*?)</table>', s, re.S)
    if not tbl:
        problems.append("performance table did not render")
    else:
        rows = [r for r in re.findall(r"<tr.*?</tr>", tbl[0], re.S) if "<td" in r]
        if len(rows) != len(want):
            problems.append(f"table has {len(rows)} rows, expected {len(want)}")
        order = [re.sub("<.*?>", "", re.findall(r"<td.*?</td>", r, re.S)[0])
                 .replace("*", "").strip() for r in rows]
        if order != want:
            problems.append(f"row order {order} != tab order {want}")
        rated = [re.sub("<.*?>", "", re.findall(r"<td.*?</td>", r, re.S)[2]).strip()
                 for r in rows]
        bad = [m for m, v in zip(order, rated)
               if v not in ("Strong", "Good", "Fair", "Weak", "Poor")]
        if bad:
            problems.append(f"no signal-strength rating for {bad}")
    if not re.search(r'<table class="fml">', s):
        problems.append("scoring formula box missing")
    for tok in ("NaN", "undefined", "[object Object]"):
        if tok in s:
            problems.append(f"{s.count(tok)}x '{tok}' in rendered page")
    # the cross-board nav must be populated with the LIVE filenames
    others = [f for _, (t, f) in PAGES.items() if f != fname]
    missing = [f for f in others if f'href="{f}"' not in s]
    if missing:
        problems.append(f"board nav missing links to {missing}")
    return problems


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("--variant="):
            VARIANT = a.split("=", 1)[1]
    markets = [m for b in BOARDS.values() for m in b]
    print(f"=== daily rebuild {VARIANT} ({len(markets)} markets) ===", flush=True)

    run([sys.executable, "build_v11.py"] + markets,
        f"1/6 build_v11 (JM walk-forward, warm from fitcache)")
    run([sys.executable, "build_boards.py", f"--variant={VARIANT}"],
        f"2/6 build_boards ({VARIANT}, JM vs VM)")
    run([sys.executable, "build_payloads.py", f"--variant={VARIANT}"] + markets,
        f"3/6 build_payloads ({len(markets)} markets)")
    run([sys.executable, "refresh_v13_all.py", f"--variant={VARIANT}"],
        "4/6 refresh V13_ALL in template")
    for b, (title, fname) in PAGES.items():
        run([sys.executable, "build_dashboard.py", f"--variant={VARIANT}",
             f"--board={title}", os.path.join("..", "dashboard", fname)] + BOARDS[b],
            f"5/6 build {BOARD_TITLE[b]} -> {fname}")

    print("\n>>> 6/6 verify (headless render, assertions on the real DOM)", flush=True)
    browser = chrome()
    failed = False
    for b, (_, fname) in PAGES.items():
        probs = verify(b, fname, browser)
        print(f"    {fname:<16} " + ("OK" if not probs else "FAILED"), flush=True)
        for p in probs:
            print(f"           - {p}", flush=True)
        failed |= bool(probs)
    if failed:
        raise SystemExit("\nVERIFICATION FAILED")
    print("\nDAILY REBUILD COMPLETE")
