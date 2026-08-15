"""The daily change notice: which markets' published readings moved, as a mailed list.

Run by .github/workflows/daily.yml AFTER the build and BEFORE the commit. It compares
today's published reading for every market on the four boards against the reading
recorded the last time a board went out, and writes a markdown list of what moved:

    HSI    BULL -> BEAR    AMBER -> RED
    NDX    BULL            GREEN -> AMBER

Two readings are watched - the same two the board already highlights a ticker for:
  * the signal itself   - bull <-> bear
  * the flip-risk light - GREEN / AMBER / RED

WHY A COMMITTED BASELINE, and not the payload's own regime_age / p_bear_hist. Those
carry a 3-session window (RECENT_N), so one flip would mail on three consecutive days,
and any day the job skips - stale data, a venue holiday, a cron slot GitHub silently
drops - shifts that window without saying so. results/signal_state.json instead holds
the exact readings that were last REPORTED, so every change is mailed once, on the day
it appears, however irregular the publishing schedule turns out to be.

ORDERING IS LOAD-BEARING. This step writes the baseline, but the workflow commits it
only after the notification has been delivered. If delivery fails, the baseline stays
uncommitted and the next run re-detects the same change rather than losing it. Moving
this step after the commit would make a failed send a permanently missed signal.

THE ZONE THRESHOLDS ARE NOT COPIED HERE. flipZone() in
dashboard/dashboard_template_prod.html is the one definition of the traffic light, and
build_payloads.py ships p_bear raw precisely so nothing downstream re-implements it.
This script PARSES the thresholds and the display clamp out of that function and
refuses to run if it cannot find them, so a change to the template either reaches the
mail too or fails the build loudly - it cannot drift quietly.

Usage:  python notify_changes.py [--variant=V17] [--dry-run] [--sample]
        --dry-run prints the notice and leaves results/signal_state.json alone.
        --sample  builds a clearly-marked demonstration notice out of today's real
                  readings against INVENTED previous ones, so the delivery path can be
                  exercised on a day when nothing has actually moved. It never touches
                  the baseline. The workflow exposes it as the `sample_notice` input on
                  a manual run - that is the way to test this end to end without
                  committing a false baseline to make a change appear.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_boards import BOARDS, BOARD_TITLE          # noqa: E402
from rebuild_daily import PAGES, VARIANT              # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
TEMPLATE = os.path.join(HERE, "..", "dashboard", "dashboard_template_prod.html")
STATE_FILE = os.path.join(RESULTS, "signal_state.json")
# The notice itself is NOT committed: the repo root is gitignored apart from an explicit
# whitelist, so a file written here is picked up by the workflow and by nothing else.
NOTICE = os.path.join(HERE, "..", "change_notice.md")
SITE = "https://laimanto.github.io/bull-bear-monitor/"


def zone_rules(path=TEMPLATE):
    """Read the traffic-light thresholds out of the template's flipZone().

    Returns (clamp_lo, clamp_hi, bull_ladder, bear_ladder), each ladder being
    ([(threshold, name), ...], name_above_the_last_threshold).

    Every failure here is fatal on purpose. The alternative - falling back to a copy of
    last known thresholds - is exactly the silent divergence between the tile and this
    mail that shipping p_bear raw was meant to prevent.
    """
    src = open(path, encoding="utf-8").read()
    clamp = re.search(r"const FLIP_MIN = ([0-9.]+), FLIP_MAX = ([0-9.]+);", src)
    body = re.search(r"function flipZone\(.*?\n\}", src, re.S)
    if not clamp or not body:
        raise SystemExit("notify_changes: cannot find FLIP_MIN/FLIP_MAX or flipZone() in "
                         f"{os.path.relpath(path, HERE)} - the template changed shape. "
                         "Update the parser here so the mail keeps matching the tile.")
    steps = re.findall(r'pct <= ([0-9.]+) \? \{ name: "([A-Z]+)"', body.group(0))
    tails = re.findall(r':\s*\{ name: "([A-Z]+)"', body.group(0))
    if len(steps) != 4 or len(tails) != 2:
        raise SystemExit("notify_changes: flipZone() no longer reads as two 2-step "
                         f"ladders (found {len(steps)} steps, {len(tails)} tails). "
                         "Update the parser here so the mail keeps matching the tile.")
    return (float(clamp.group(1)), float(clamp.group(2)),
            ([(float(t), n) for t, n in steps[:2]], tails[0]),
            ([(float(t), n) for t, n in steps[2:]], tails[1]))


def light(state, p_bear, rules):
    """The tile's flip percentage and traffic-light colour for one market.

    The percentage is inverted for a bear market - it is always "chance of flipping AWAY
    from where we are" - and that inversion also flips which end of the scale is green.
    Both facts come from flipZone(); this only applies them.
    """
    lo, hi, bull, bear = rules
    if p_bear is None:
        return None, None
    is_bull = state == "bull"
    pct = min(hi, max(lo, p_bear if is_bull else 1 - p_bear))
    ladder, tail = bull if is_bull else bear
    for threshold, name in ladder:
        if pct <= threshold:
            return name, pct
    return tail, pct


def read_today(variant, rules):
    """Today's published reading per market, straight off the payloads just built."""
    out, missing = {}, []
    for board, markets in BOARDS.items():
        for m in markets:
            path = os.path.join(RESULTS, f"payload_{m}_{variant}.json")
            if not os.path.exists(path):
                missing.append(m)
                continue
            with open(path) as f:
                d = json.load(f)
            name, pct = light(d["current_state"], d.get("signal", {}).get("p_bear"), rules)
            out[m] = dict(board=board, full_name=d.get("full_name", m),
                          state=str(d["current_state"]).upper(), zone=name, pct=pct,
                          end=d["end"])
    if missing:
        # build_payloads has just run over every market on every board, so a gap here is
        # a broken build, not an empty day. Reporting "nothing moved" would be a lie.
        raise SystemExit(f"notify_changes: no {variant} payload for {', '.join(missing)}")
    return out


def diff(today, before):
    """Markets whose signal or flip light differs from the last reported reading.

    A market with no baseline is seeded silently: on the first run that is all 30, and
    afterwards it is a market newly added to a board - neither is a change in a reading.
    """
    moved = []
    for m, now in today.items():
        was = before.get(m)
        if not was:
            continue
        if was.get("state") != now["state"] or was.get("zone") != now["zone"]:
            moved.append((m, was, now))
    order = [m for b in BOARDS.values() for m in b]
    moved.sort(key=lambda t: order.index(t[0]))
    return moved


def arrow(was, now):
    return f"{was} → **{now}**" if was != now else f"{was}"


def fake(today):
    """Invented "previous" readings for --sample, one market off each board.

    Both columns are made to move so the layout is exercised in full. The banner
    notice() puts on top of a sample is not optional: the body is otherwise
    indistinguishable from a real alert, and a test mail that reads as a genuine
    bull-to-bear call on NDX is worse than no test at all.
    """
    rotate = {"GREEN": "AMBER", "AMBER": "RED", "RED": "AMBER"}
    picks = [markets[0] for markets in BOARDS.values() if markets]
    return [(m, {"state": "BEAR" if today[m]["state"] == "BULL" else "BULL",
                 "zone": rotate.get(today[m]["zone"], "AMBER")}, today[m])
            for m in picks if m in today]


def notice(moved, data_through, sample=False):
    """The mail body, grouped by board, one row per market that moved."""
    n = len(moved)
    head = (f"**{n} market{'' if n == 1 else 's'} moved** on the session ending "
            f"**{data_through}**.\n")
    if sample:
        head = ("> ⚠️ **SAMPLE — not a real signal change.** Sent by hand to test the "
                "daily notifier. The **before** values in the left of each pair are "
                "invented; the current readings are real. Nothing has moved.\n")
    parts = [head]
    for board, markets in BOARDS.items():
        rows = [(m, was, now) for m, was, now in moved if now["board"] == board]
        if not rows:
            continue
        _, page = PAGES[board]
        parts.append(f"\n### [{BOARD_TITLE[board]}]({SITE}{page})\n")
        parts.append("| Market | Signal | Flip light | Chance of flipping |")
        parts.append("| --- | --- | --- | --- |")
        for m, was, now in rows:
            pct = "—" if now["pct"] is None else f"{round(now['pct'] * 100)}%"
            parts.append(f"| **{m}** — {now['full_name']} "
                         f"| {arrow(was.get('state'), now['state'])} "
                         f"| {arrow(was.get('zone'), now['zone'])} | {pct} |")
    parts.append(
        "\n---\n"
        "Boards: " + " · ".join(f"[{BOARD_TITLE[b]}]({SITE}{f})"
                                for b, (_, f) in PAGES.items()) + "\n\n"
        "*Chance of flipping* is the board's own tile: for a **BULL** market the measured "
        "chance of turning bear within 5 sessions, for a **BEAR** market the chance of "
        "turning back to bull. The light is green when that reading is comfortable, red "
        "when the signal is close to going the wrong way. Sent by the daily job "
        "(`.github/workflows/daily.yml`); each change is reported once, on the day it "
        "first appears.\n")
    return "\n".join(parts)


def subject(moved, data_through):
    """Short enough to survive an inbox list, specific enough to read without opening."""
    bits = []
    for m, was, now in moved[:3]:
        if was.get("state") != now["state"]:
            bits.append(f"{m} {str(was.get('state')).lower()}->{now['state'].lower()}")
        else:
            bits.append(f"{m} {str(was.get('zone')).lower()}->{str(now['zone']).lower()}")
    if len(moved) > 3:
        bits.append(f"+{len(moved) - 3} more")
    return f"Signal change {data_through}: " + ", ".join(bits)


def emit(**outputs):
    """Hand the workflow its gate and its subject line."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        for k, v in outputs.items():
            f.write(f"{k}={v}\n")


if __name__ == "__main__":
    variant, dry, sample = VARIANT, False, False
    for a in sys.argv[1:]:
        if a.startswith("--variant="):
            variant = a.split("=", 1)[1]
        elif a == "--dry-run":
            dry = True
        elif a == "--sample":
            sample = True

    rules = zone_rules()
    today = read_today(variant, rules)
    before = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            before = json.load(f).get("markets", {})
    data_through = max(r["end"] for r in today.values())
    moved = fake(today) if sample else diff(today, before)

    if moved:
        with open(NOTICE, "w", encoding="utf-8") as f:
            f.write(notice(moved, data_through, sample=sample))
        title = subject(moved, data_through)
        if sample:
            title = "[SAMPLE, ignore] " + title
        print(f"{len(moved)} of {len(today)} markets moved (data through {data_through}):")
        for m, was, now in moved:
            bits = [f"{was.get(k)} -> {now[k]}" for k in ("state", "zone")
                    if was.get(k) != now[k]]
            print(f"    {m:<8} " + ", ".join(bits))
        print(f"  subject: {title}")
    else:
        title = ""
        seeded = len(today) - len(before)
        print(f"nothing moved (data through {data_through}"
              + (f"; {seeded} market(s) seeded" if seeded > 0 else "") + ")")

    # A sample must never move the baseline: its "before" values are invented, so
    # recording its aftermath would hide the next real change on those markets.
    if dry or sample:
        print(("--sample" if sample else "--dry-run") + ": baseline not written")
        emit(changed=str(bool(moved)).lower(), count=len(moved), title=title)
        sys.exit(0)

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "data_through": data_through,
                   "markets": {m: {k: r[k] for k in ("state", "zone", "pct", "end")}
                               for m, r in sorted(today.items())}}, f, indent=1)
        f.write("\n")
    emit(changed=str(bool(moved)).lower(), count=len(moved), title=title)
