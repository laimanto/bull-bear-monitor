"""The daily BBM build - one command, run by .github/workflows/daily.yml.

Adapted from the research repo's rebuild_all.py. ORDER MATTERS, and getting it
wrong has produced silent, plausible-looking wrong pages more than once:

  1. build_v11.py         JM walk-forward, all markets -> regimes_{M}_V11.csv.
                          Warm from fitcache/ (inference only, seconds/market);
                          a NEW fit happens only at the January rollover. The
                          MAX_NEW_FITS env guard aborts if the cache stops
                          matching the data rather than training for hours in CI.
  2. build_boards.py      JM vs VM per market -> regimes_{M}_{VARIANT}.csv + metrics
  3. flip_calibrate.py    P(flip within 5 sessions) from (countdown, distance) ->
                          overwrites p_bear in every regimes CSV and adds the
                          countdown columns. Pooled ACROSS markets per model
                          family, so it must see every market in one pass and
                          cannot be folded into the per-market loops either side.
  4. build_payloads.py    per-market tab data, INCLUDING loss_vs_bh
  5. refresh_v13_all.py   inject the metrics into the template's V13_ALL
  6. build_dashboard.py   x4 -> dashboard/{index,us,hk,commodity}.html
  7. verify               headless render of each page, assert on the real DOM

Skipping step 5 leaves the summary table describing the previous run while the
tabs show the new one. Running step 6 before step 4 builds pages from stale
payloads. Both have happened; neither announced itself. Step 3 must come after 2
(it reads the regimes CSVs build_boards writes, including VM's raw_state) and
before 4 (the payload ships the p_bear it produces).

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
# V17 went live 2026-08-11 (user-authorised). v17 = v16's VM byte-for-byte, with the
# all-episode placebo (ALL_EP_PLACEBO in build_boards), money-based signal strength,
# Singapore as EWS, and the rebuilt flip tile. 13 of 30 markets change vs V16; realised
# protection is identical in 28 of them and improves in the other 2 (FTSE, SMH, the only
# two where the JM/VM choice flips) - what moved is the placebo yardstick, not the
# strategy. To roll back, set this to "V16": every V16 artefact is still reproducible
# because the old single-episode placebo was kept for v13-v16.
VARIANT = "V17"
# Where the four pages are written, relative to dashboard/. Empty = the live pages.
# `--outdir=v17` builds a full four-page board into dashboard/v17/ WITHOUT touching the
# published ones - the cross-board nav uses bare filenames, so a self-contained subfolder
# links to its own siblings and the verification's nav check still passes.
OUTSUB = ""


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


def outdir():
    d = os.path.join(DASH, OUTSUB) if OUTSUB else DASH
    os.makedirs(d, exist_ok=True)
    return d


def verify(board, fname, browser):
    """Render the real page and assert on the real DOM. Nothing else is proof."""
    src = os.path.abspath(os.path.join(outdir(), fname))
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
        elif a.startswith("--outdir="):
            OUTSUB = a.split("=", 1)[1].strip("/\\")
    markets = [m for b in BOARDS.values() for m in b]
    dest = os.path.join("dashboard", OUTSUB) if OUTSUB else "dashboard"
    print(f"=== daily rebuild {VARIANT} ({len(markets)} markets) -> {dest}/ ===", flush=True)

    run([sys.executable, "build_v11.py"] + markets,
        f"1/7 build_v11 (JM walk-forward, warm from fitcache)")
    run([sys.executable, "build_boards.py", f"--variant={VARIANT}"],
        f"2/7 build_boards ({VARIANT}, JM vs VM)")
    run([sys.executable, "flip_calibrate.py", f"--variant={VARIANT}"] + markets,
        f"3/7 flip_calibrate (P(flip<=5 sessions), pooled per model family)")
    run([sys.executable, "build_payloads.py", f"--variant={VARIANT}"] + markets,
        f"4/7 build_payloads ({len(markets)} markets)")
    run([sys.executable, "refresh_v13_all.py", f"--variant={VARIANT}"],
        "5/7 refresh V13_ALL in template")
    for b, (title, fname) in PAGES.items():
        out = os.path.join("..", "dashboard", OUTSUB, fname) if OUTSUB \
              else os.path.join("..", "dashboard", fname)
        os.makedirs(os.path.dirname(os.path.join(HERE, out)), exist_ok=True)
        run([sys.executable, "build_dashboard.py", f"--variant={VARIANT}",
             f"--board={title}", out] + BOARDS[b],
            f"6/7 build {BOARD_TITLE[b]} -> {dest}/{fname}")

    print("\n>>> 7/7 verify (headless render, assertions on the real DOM)", flush=True)
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
