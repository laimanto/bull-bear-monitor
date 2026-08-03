"""Build all four v13 board datasets: global indexes, US tech, Hong Kong, commodity & crypto.

Named without a version on purpose. This file was called v14_build.py, which twice led
to it being read as "v14" - there is no v14. It builds V13 for every board.

Same dual-model design as V13. A market uses the Volatility Model instead of the jump
model only if BOTH hold, and both are decided from the JUMP model alone:
    (a) JM's own protection placebo percentile < 90  - it cannot defend that market, and
    (b) VM's AUC > JM's AUC                          - VM actually detects its bears better
Condition (b) is the one GOLD failed in the V12 draft: its VM protection percentile was
94 but its VM AUC was 0.505 (chance), so the percentile was a single-episode fluke.

THIN MARKETS. Some names simply have not existed long enough for this protocol, which
consumes ~6 years before its first call (~10 months feature warm-up + 2 years training
+ a 3-year validation window). Anything under MIN_TEST_YEARS is still shown - the user
asked to see them - but flagged, and its placebo percentiles are suppressed rather than
published, because a percentile computed on 2 years of one market is not evidence.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# pipeline scripts all live in this directory - no path insert needed
warnings.filterwarnings("ignore")

import build_regimes as jm                     # noqa: E402
import decode as dc                            # noqa: E402
import combo_v2 as cv                          # noqa: E402
import final_config as fc                      # noqa: E402
import rescore as rs                           # noqa: E402
import vm_calibrate as vcal                    # noqa: E402
from v11_metrics import episodes, timing       # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
VM_THRESH = 0.60
# VM MINIMUM HOLD (user, 2026-08-02). VM was an ungated threshold and 27% of its round
# trips lasted <= 3 days, which reads badly on the dashboard. A 5-day minimum hold with
# NO confirmation removes them entirely (27% -> 0%) and was chosen over the confirmation
# gates after sweeping 1/2, 2/2, 2/3, 3/3, 3/4, 2/3+5d and 1/1+5d across all 16 VM
# markets - see exp/vm_gate_sweep.csv.
#
# Two honest caveats, both measured:
#  - NO arm improved the placebo percentiles (profit 52-58, protection 69-72, versus
#    56/71 ungated). This adds no timing skill. It is a turnover and presentation
#    change, and the money differences between arms sit inside the noise.
#  - EVERY arm degrades the worst episode. 1/1+5d degrades it least (-2.3% vs -2.1%
#    ungated, and not worse than ungated on 14 of 16 markets), which is why it was
#    preferred over 2/3+5d (-2.9%) even though 2/3+5d scores better on mean profit.
#
# NOTE this contradicts the JM finding, where the dwell was the HARMFUL half of the
# decode and was removed. It does not transfer: on VM the dwell arms are among the
# better performers. Different model, different answer.
# v13 = (1, 1, 5): 5-day minimum hold, no confirmation.
# v14 = (2, 3, 1): JM's 2/3 confirmation, no hold.  Chosen after v13 shipped and the
#   holding distribution showed the dwell had RELABELLED the churn rather than removed
#   it - 101 of 578 VM trades landed on exactly 5 days, a 17.5% spike sitting on the
#   floor. A hard minimum clamps short trades to its boundary; a confirmation gate has
#   no floor, so the distribution stays smooth.
VM_CFG = {
    # gate = (n_bear, n_bull, dwell);  sv = signed volatility features
    "V13": dict(gate=(1, 1, 5), sv=False),   # 5-day minimum hold, no confirmation
    "V14": dict(gate=(2, 3, 1), sv=False),   # JM's 2/3 confirmation
    "V15": dict(gate=(2, 3, 5), sv=False),   # both
    # V16 (user, 2026-08-03). SIGNED VOLATILITY + 2/3 confirmation.
    #
    # Adopted for CORRECTNESS, not performance, and that distinction should survive:
    # the placebo is FLAT against the shipped model (profit 57th percentile vs 56th,
    # protection 66th vs 71st), so this does not make money. What it fixes is that VM's
    # only unsigned input, vol10, made a RALLY raise bear conviction - measured across
    # all 16 VM markets, the bear alarm rose on 43% of days a market gained >5%, versus
    # 49% of days it fell >5%. On MSFT 2026-07-30 a +15.5% day pinned the alarm at 1.000
    # and drove flip-to-bull to 13%.
    #
    # What it costs: the worst episode goes -2.1% -> -3.3%. That is consistent across
    # every sv arm tested, so unlike the protection difference it is not noise.
    #
    # Only the VOLATILITY half is taken. Signed VOLUME (the seller-conviction term) was
    # tested and rejected: its protection gain sat at the 70th percentile against the
    # baseline's 71st, i.e. noise, and it does not address the defect.
    #
    # The gate is plain 2/3. A conviction-weighted gate (strong alarms skip the wait)
    # was built and REJECTED - it worsened the tail in 3 of 5 variants and never
    # improved it, because a maximal alarm is a LAGGING signal: at alarm >= 0.90 the
    # prior 10 days average -2.47% and the NEXT 10 average +1.04%. Acting faster on
    # high conviction means exiting into the bounce.
    "V16": dict(gate=(2, 3, 1), sv=True),
}

# Under this many FORECAST years a market is shown but marked as unproven. Set to 7.0
# (user, 2026-08-02): at 8.0 it separated ARKQ/BTC at 7.6y from META at 9.6y, and those
# are not meaningfully different in evidence.
MIN_TEST_YEARS = 7.0
JM_PROT_BAR = 90            # JM protection placebo percentile below which VM is considered
N_SHIFT = 300

# Tab order is the order below. US and HK are alphabetical / numeric-ascending; the
# commodity board follows Gold > Silver > Oil > Bitcoin > Ethereum.
BOARDS = {
    "index": ["NDX", "SPX", "HSI", "KOSPI", "NIKKEI", "FTSE"],
    "us": ["AAPL", "AMZN", "ARKQ", "GOOGL", "META", "MSFT", "MU", "NVDA", "SMH", "TSLA"],
    "hk": ["HK0005", "HK0388", "HK0700", "HK0939", "HK0941", "HK1800", "HK1810", "HK9988"],
    "cc": ["GOLD", "SILVER", "WTI", "BTC", "ETH"],
}
VARIANT = "V13"      # overridden by --variant=
BOARD_TITLE = {"index": "Global indexes", "us": "US tech", "hk": "Hong Kong",
               "cc": "Commodity & crypto"}


def vm_alarm(m):
    cfg = jm.SYMBOLS[m]
    close, ret, rf = jm.load_data(cfg)
    raw = pd.read_csv(os.path.join(jm.DATA_DIR, cfg["index"]), index_col=0, parse_dates=True)
    vol = raw["Volume"].astype(float).reindex(close.index)
    v0 = jm.VOL_START.get(m)
    if v0:
        close, vol = close.loc[v0:], vol.loc[v0:]
    return cv.build_walkforward(m, close, vol, cfg["first_test"],
                                sv=VM_CFG[VARIANT]["sv"])["combo_p_bear"]


def vm_state(alarm):
    """VM's published state: threshold, then a VM_DWELL-day minimum hold.

    N_BEAR=N_BULL=1 makes the confirmation a no-op, so the dwell is the only filter -
    a flip may publish only if at least VM_DWELL trading days have passed since the
    last one. That enforces the minimum holding period directly rather than achieving
    it as a side effect of confirmation.
    """
    raw = (alarm >= VM_THRESH).astype(int)
    old = (dc.N_BEAR, dc.N_BULL, dc.DWELL)
    dc.N_BEAR, dc.N_BULL, dc.DWELL = VM_CFG[VARIANT]["gate"]
    try:
        return dc.confirm(raw, np.ones(len(raw), dtype=bool)).astype(int)
    finally:
        dc.N_BEAR, dc.N_BULL, dc.DWELL = old


def placebo(c, r, f, sig, n=N_SHIFT):
    s = fc.score(c, r, f, sig)
    rng = np.random.default_rng(0)
    sh = rng.integers(252, max(253, len(sig) - 252), size=n)
    nl = [fc.score(c, r, f, pd.Series(np.roll(sig.values, int(k)), index=sig.index))
          for k in sh]
    return (s, 100 * float(np.mean([x["cap"] < s["cap"] for x in nl])),
            100 * float(np.mean([x["prot"] < s["prot"] for x in nl])))


def assess(m):
    """Score JM and VM on the same rows, then apply the rule."""
    d = pd.read_csv(os.path.join(RESULTS, f"regimes_{m}_V11.csv"), index_col=0, parse_dates=True)
    alarm = vm_alarm(m).reindex(d.index)
    idx = d.index[alarm.notna()]
    d = d.loc[idx]
    alarm = alarm.loc[idx]
    c, r, f = d["close"], d["ret"], d["rf"]
    J = d["state"].astype(int)
    V = vm_state(alarm)
    sj, jcp, jpp = placebo(c, r, f, J)
    sv, vcp, vpp = placebo(c, r, f, V)
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    use_vm = (jpp < JM_PROT_BAR) and (sv["auc"] > sj["auc"])
    return dict(market=m, years=yrs, thin=yrs < MIN_TEST_YEARS, use_vm=use_vm,
                d=d, alarm=alarm, J=J, V=V,
                jm=dict(**sj, capP=jcp, protP=jpp), vm=dict(**sv, capP=vcp, protP=vpp))


def build(a, variant):
    m = a["market"]
    d = a["d"].copy()
    if a["use_vm"]:
        d["state"] = a["V"].astype(float)
        d["raw_state"] = d["state"]
        d["vm_alarm"] = a["alarm"]
        d["p_bear"] = vcal.calibrate(a["V"], a["alarm"])   # honest flip probability
        d["lam"] = np.nan
    d.to_csv(os.path.join(RESULTS, f"regimes_{m}_{variant}.csv"))
    pub = d["state"].astype(int)
    s = fc.score(d["close"], d["ret"], d["rf"], pub)
    fl, avgd, _ = rs.churn(pub)
    pos = (pub == 0).astype(float).shift(2).fillna(1.0 if pub.iloc[0] == 0 else 0.0)
    ex, re_, missed, n = timing(d["close"], pos, episodes(d["close"]))
    src = a["vm"] if a["use_vm"] else a["jm"]
    # Data budget, for the "not enough history" note: total history the market has,
    # how much of it the model consumes before its first call, and what is left to
    # forecast on. Reported per market because calendar quantisation moves it around.
    cfg = jm.SYMBOLS[m]
    raw = pd.read_csv(os.path.join(jm.DATA_DIR, cfg["index"]), index_col=0, parse_dates=True)
    v0 = jm.VOL_START.get(m)
    px = raw.index if not v0 else raw.loc[v0:].index
    hist_y = (d.index[-1] - px[0]).days / 365.25
    need_y = (d.index[0] - px[0]).days / 365.25
    return dict(market=m, model=("VM" if a["use_vm"] else "JM"), thin=a["thin"],
                hist=round(hist_y, 1), need=round(need_y, 1),
                years=round(a["years"], 1), auc=s["auc"], cap=s["cap"], prot=s["prot"],
                out=float((pub == 1).mean()), fl=fl, avgd=avgd, exitd=ex, reentry=re_,
                missed=missed, n_ep=n,
                capP=src["capP"], protP=src["protP"],
                jm_auc=a["jm"]["auc"], vm_auc=a["vm"]["auc"], jm_protP=a["jm"]["protP"])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--variant="):
            VARIANT = a.split("=", 1)[1]
    assert VARIANT in VM_CFG, f"no VM config defined for {VARIANT}"
    print(f"variant {VARIANT}; VM {VM_CFG[VARIANT]}")
    want = args or list(BOARDS)
    allrows = []
    for board in want:
        rows = []
        print(f"\n=== {board}  ({BOARD_TITLE[board]}) ===", flush=True)
        for m in BOARDS[board]:
            a = assess(m)
            rows.append(build(a, VARIANT))          # always regimes_{m}_V13.csv
            r = rows[-1]
            r["board"] = BOARD_TITLE[board]
            print(f"  {m:<7} {r['model']}  {r['years']:>5.1f}y  AUC {r['auc']:.3f}  "
                  f"cap {r['cap']:.3f}  prot {r['prot']:+.1%}  "
                  f"(JM auc {r['jm_auc']:.3f}/protP {r['jm_protP']:.0f}% | VM auc {r['vm_auc']:.3f})"
                  f"{'  THIN' if r['thin'] else ''}", flush=True)
        df = pd.DataFrame(rows).sort_values("auc", ascending=False)
        nvm = int((df["model"] == "VM").sum())
        print(f"  -> {nvm} VM / {len(df)-nvm} JM;  thin: "
              f"{', '.join(df[df.thin]['market']) or 'none'}")
        allrows.append(df)
    # ONE merged metrics file, written only when every board was rebuilt, so a partial
    # run can never leave the table describing markets whose payloads say otherwise.
    if set(want) == set(BOARDS):
        out = pd.concat(allrows, ignore_index=True)
        out.to_csv(os.path.join(RESULTS, f"{VARIANT.lower()}_metrics_all.csv"), index=False)
        print(f"\nwrote v13_metrics_all.csv ({len(out)} markets, all 4 boards)")
    else:
        print(f"\nPARTIAL RUN ({', '.join(want)}) - {VARIANT.lower()}_metrics_all.csv NOT rewritten; "
              f"re-run with no arguments before rebuilding dashboards.")
