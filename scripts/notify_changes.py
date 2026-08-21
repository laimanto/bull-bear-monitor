"""The daily change notice: which markets' published readings moved, as a mailed list.

Run by .github/workflows/daily.yml AFTER the build and BEFORE the commit. It compares
today's published reading for every market on the four boards against the reading
recorded the last time a board went out, and writes a markdown list of what moved:

    HSI    BULL -> BEAR    AMBER -> RED    Fair
    NDX    BULL            GREEN -> AMBER  Strong

Two readings are watched - the same two the board already highlights a ticker for:
  * the signal itself   - bull <-> bear
  * the flip-risk light - GREEN / AMBER / RED

A third, SIGNAL STRENGTH, is REPORTED BUT NEVER TRIGGERS (user, 2026-08-21). It is the
board's own Strong/Good/Fair/Weak/Poor rating of the market's whole timing record, so it
answers "how much should I trust this call?" beside the call itself. It is a property of
the market rather than of today's session and moves only when a metric crosses a cut, so
mailing on it would be noise; it is carried in the baseline purely so the notice can show
an arrow on the rare day the rating itself shifted.

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

NO THRESHOLD IS COPIED HERE - not the light's, not the rating's. flipZone() and
strengthOf() in dashboard/dashboard_template_prod.html are the one definition of both,
and build_payloads.py ships p_bear raw precisely so nothing downstream re-implements it.
This script PARSES the light's thresholds and clamp, and the rating's cuts, ceilings and
tier names, out of those two functions, and refuses to run if it cannot find them - so a
change to the template either reaches the mail too or fails the build loudly, and cannot
drift quietly. Strength is then scored on the two inputs the tile itself uses: the
market's row in the template's own V13_ALL, and protection_vs_bh off its payload.

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


def strength_rules(path=TEMPLATE):
    """Read signal strength's scoring constants out of the template's strengthOf().

    Same contract as zone_rules(), and fatal for the same reason: the rating is the
    board's, and a mail that graded a market differently from the page it links to would
    be worse than a mail with no rating in it.

    Returns the four things that decide a rating - the two money ladders, the tier
    labels, and the two evidence ceilings - in the form the JS uses them.
    """
    src = open(path, encoding="utf-8").read()
    body = re.search(r"function strengthOf\(.*?\n\}", src, re.S)
    if not body:
        raise SystemExit("notify_changes: cannot find strengthOf() in "
                         f"{os.path.relpath(path, HERE)} - the template changed shape. "
                         "Update the parser here so the mail keeps matching the board.")
    b = body.group(0)
    profit = re.search(r"band\(d\.profit,\s*\[([0-9., ]+)\]\)", b)
    prot = re.search(r"band\(prot,\s*\[([0-9., ]+)\]\)", b)
    step = re.search(r"return ([0-9.]+) - k \* ([0-9.]+);", b)
    tiers = re.findall(r'\["([A-Za-z]+)", "var\(--[a-z]+\)", \d+\]', b)
    money = re.search(r"const byMoney\s*=\s*\[([0-9., ]+)\]\.findIndex", b)
    plcb = re.findall(r"d\.plcb >= ([0-9.]+) \? (\d)", b)
    plcb_tail = re.search(r"d\.plcb >= [0-9.]+ \? \d : (\d);", b)
    eps = re.findall(r"d\.n >= (\d+) \? (\d)", b)
    eps_tail = re.search(r"d\.n >= \d+ \? \d : (\d);", b)
    if not (profit and prot and step and money and plcb_tail and eps_tail) or not tiers:
        raise SystemExit("notify_changes: strengthOf() no longer reads as two money "
                         "ladders, a tier list and two ceilings. Update the parser here "
                         "so the mail keeps matching the board.")
    nums = lambda g: [float(x) for x in g.split(",")]

    def ladder(pairs, tail, what):
        # The tier INDEXES are checked, not just the count. Half-editing a ceiling -
        # deleting only its top rung, say - leaves a ladder that still parses and still
        # scores every market, just never applying the cap that was removed. That is the
        # one failure mode this whole parse-don't-copy design exists to make impossible,
        # and it is invisible unless the rungs are required to run 0, 1, 2, ... as the JS
        # writes them.
        rungs = [(float(t), int(i)) for t, i in pairs] + [(None, int(tail))]
        if [i for _, i in rungs] != list(range(len(rungs))) or len(rungs) > len(tiers):
            raise SystemExit(f"notify_changes: strengthOf()'s {what} ceiling reads as "
                             f"tiers {[i for _, i in rungs]} against {len(tiers)} tiers "
                             "- it has changed shape. Update the parser here so the mail "
                             "keeps matching the board.")
        return rungs

    if len(nums(money.group(1))) != len(tiers):
        raise SystemExit(f"notify_changes: strengthOf() has {len(tiers)} tiers but "
                         f"{len(nums(money.group(1)))} money cuts. Update the parser "
                         "here so the mail keeps matching the board.")
    return dict(profit=nums(profit.group(1)), prot=nums(prot.group(1)),
                base=float(step.group(1)), step=float(step.group(2)),
                tiers=tiers, money=nums(money.group(1)),
                plcb=ladder(plcb, plcb_tail.group(1), "placebo"),
                eps=ladder(eps, eps_tail.group(1), "episode"))


def board_metrics(path=TEMPLATE):
    """Each market's row of the template's own V13_ALL, keyed by symbol.

    Read from the TEMPLATE, not from results/*_metrics_all.csv, so the mail scores the
    rating off the same rounded numbers the rendered page does - refresh_v13_all.py has
    already run by the time this does, and a market sitting exactly on a cut must not be
    graded one way here and another way on the board.
    """
    src = open(path, encoding="utf-8").read()
    m = re.search(r"const V13_ALL = (\[.*?\]);", src, re.S)
    if not m:
        raise SystemExit("notify_changes: cannot find V13_ALL in "
                         f"{os.path.relpath(path, HERE)} - has refresh_v13_all.py run?")
    return {r["m"]: r for r in json.loads(m.group(1))}


def strength(d, prot, rules):
    """The board's Signal-strength word for one market, or None if it cannot be rated.

    Mirrors strengthOf(): money sets the tier, and the protection placebo and the bear
    episode count can only pull it DOWN - evidence never adds points. Worth mailing
    because BULL -> BEAR on a Strong market and the same move on a Poor one are not the
    same message, and nothing else in the notice says which one this is.
    """
    if not d or prot is None or d.get("plcb") is None or d.get("n") is None:
        return None

    def band(x, cuts):
        for k, c in enumerate(cuts):
            if x >= c:
                return rules["base"] - k * rules["step"]
        return 0.0

    def rung(v, ladder):
        return next(i for t, i in ladder if t is None or v >= t)

    sc = band(d["profit"], rules["profit"]) + band(prot, rules["prot"])
    by_money = next((i for i, c in enumerate(rules["money"]) if sc >= c),
                    len(rules["tiers"]) - 1)
    return rules["tiers"][max(by_money, rung(d["plcb"], rules["plcb"]),
                              rung(d["n"], rules["eps"]))]


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


def read_today(variant, rules, srules, mets):
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
            met = mets.get(m)
            out[m] = dict(board=board, full_name=d.get("full_name", m),
                          state=str(d["current_state"]).upper(), zone=name, pct=pct,
                          end=d["end"],
                          strength=strength(met, d.get("protection_vs_bh"), srules),
                          thin=bool((met or {}).get("thin")))
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
        # Deliberately NOT `or was.get("strength") != now["strength"]`: the rating
        # describes the market's whole record, so a cut crossed by a rounding-sized
        # move in one of its inputs is not news about today. It rides along in the
        # notice; it never causes one.
        if was.get("state") != now["state"] or was.get("zone") != now["zone"]:
            moved.append((m, was, now))
    order = [m for b in BOARDS.values() for m in b]
    moved.sort(key=lambda t: order.index(t[0]))
    return moved


def arrow(was, now):
    return f"{was} → **{now}**" if was != now else f"{was}"


def rating(was, now):
    """The strength cell: the current rating, arrowed only if it really moved.

    A baseline written before the rating was reported has no "strength" key at all,
    and every market would then show `None → Fair` on the first notice after this
    shipped - an invented change. A missing key therefore prints the rating plain.
    """
    if now["strength"] is None:
        return "&mdash;"
    mark = " \\*" if now.get("thin") else ""
    prev = was.get("strength")
    if prev and prev != now["strength"]:
        return f"{prev} → **{now['strength']}**{mark}"
    return f"{now['strength']}{mark}"


def fake(today):
    """Invented "previous" readings for --sample, one market off each board.

    Both columns are made to move so the layout is exercised in full. The banner
    notice() puts on top of a sample is not optional: the body is otherwise
    indistinguishable from a real alert, and a test mail that reads as a genuine
    bull-to-bear call on NDX is worse than no test at all.
    """
    rotate = {"GREEN": "AMBER", "AMBER": "RED", "RED": "AMBER"}
    regrade = {"Strong": "Good", "Good": "Fair", "Fair": "Good", "Weak": "Fair",
               "Poor": "Weak"}
    picks = [markets[0] for markets in BOARDS.values() if markets]
    return [(m, {"state": "BEAR" if today[m]["state"] == "BULL" else "BULL",
                 "zone": rotate.get(today[m]["zone"], "AMBER"),
                 "strength": regrade.get(today[m]["strength"])}, today[m])
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
        parts.append("| Market | Signal | Flip light | Chance of flipping "
                     "| Signal strength |")
        parts.append("| --- | --- | --- | --- | --- |")
        for m, was, now in rows:
            pct = "—" if now["pct"] is None else f"{round(now['pct'] * 100)}%"
            parts.append(f"| **{m}** — {now['full_name']} "
                         f"| {arrow(was.get('state'), now['state'])} "
                         f"| {arrow(was.get('zone'), now['zone'])} | {pct} "
                         f"| {rating(was, now)} |")
    parts.append(
        "\n---\n"
        "Boards: " + " · ".join(f"[{BOARD_TITLE[b]}]({SITE}{f})"
                                for b, (_, f) in PAGES.items()) + "\n\n"
        "*Chance of flipping* is the board's own tile: for a **BULL** market the measured "
        "chance of turning bear within 5 sessions, for a **BEAR** market the chance of "
        "turning back to bull. The light is green when that reading is comfortable, red "
        "when the signal is close to going the wrong way.\n\n"
        "*Signal strength* is the board's own **Strong / Good / Fair / Weak / Poor** "
        "rating of that market's whole timing record — what it earned against buy-and-hold "
        "and how much of the bear-market loss it avoided, capped by how much evidence "
        "stands behind it. It rates the market, not today's move, so it rarely changes "
        "and never sets off a notice on its own; an arrow there means the rating itself "
        "shifted since the last one." + (
            " A \\* marks a short forecast history — same rating, fewer years behind it."
            if any(now.get("thin") and now["strength"] for _, _, now in moved) else "")
        + "\n\nSent by the daily job (`.github/workflows/daily.yml`); each change is "
        "reported once, on the day it first appears.\n")
    return "\n".join(parts)


def subject(moved, data_through):
    """Short enough to survive an inbox list, specific enough to read without opening."""
    bits = []
    for m, was, now in moved[:3]:
        # The rating rides in the subject too: an inbox line saying a Poor-rated market
        # turned bear is a different call to action from the same move on a Strong one,
        # and that is exactly the judgement made without opening the mail.
        grade = f" ({now['strength']})" if now["strength"] else ""
        if was.get("state") != now["state"]:
            bits.append(f"{m} {str(was.get('state')).lower()}->{now['state'].lower()}{grade}")
        else:
            bits.append(f"{m} {str(was.get('zone')).lower()}->"
                        f"{str(now['zone']).lower()}{grade}")
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
    today = read_today(variant, rules, strength_rules(), board_metrics())
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
                   "markets": {m: {k: r[k]
                                   for k in ("state", "zone", "pct", "end",
                                             "strength")}
                               for m, r in sorted(today.items())}}, f, indent=1)
        f.write("\n")
    emit(changed=str(bool(moved)).lower(), count=len(moved), title=title)
