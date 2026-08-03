"""v6/v7 decode, generalised so the conviction rule can be swapped out."""
import numpy as np
import pandas as pd

# DWELL dropped from 8 to 1 (user, 2026-07-31): the 8-day minimum hold was the
# harmful half of the decode. Isolating the two overrides across 11 markets:
#     sum capture   raw 9.841   confirm-only 11.835   dwell-only 10.527   both 11.321
# Confirmation alone beats raw on 9/11 markets AND beats the both-gate in aggregate,
# with one fewer override. The dwell is negligible on slow markets and catastrophic
# on fast ones - NVDA: confirm 1.192, dwell 1.308, BOTH 0.756, i.e. each filter helps
# alone but together they destroy a third of the capture. Even NDX, the market the
# decode was originally tuned on, prefers confirm-only (4.110 vs 4.032).
# Set DWELL = 8 to reproduce pre-2026-07-31 (v4-v8) numbers.
N_BEAR, N_BULL, DWELL = 2, 3, 1


def confirm(raw_state, allow_bear):
    """v6/v7's exact structure: 2/3 gate + 8-day hold + bear-entry veto.

    `allow_bear` is a boolean array - may a bear flip publish on day i?
    That is the ONLY thing the candidates below vary."""
    s = raw_state.values
    ok = np.asarray(allow_bear)
    out = s.copy()
    last_flip = -10 ** 9
    for i in range(1, len(s)):
        n = N_BEAR if s[i] == 1 else N_BULL
        persist = all(s[i - j] == s[i] for j in range(min(n, i + 1)))
        if persist and s[i] != out[i - 1]:
            if i - last_flip < DWELL:
                persist = False
            elif s[i] == 1:
                persist = bool(ok[i])
        out[i] = s[i] if persist else out[i - 1]
        if out[i] != out[i - 1]:
            last_flip = i
    return pd.Series(out, index=raw_state.index)


# ---- conviction rules -------------------------------------------------------

def v7_blend(d, tb=0.60):
    """LIVE: average JM's centroid score with VM, one threshold."""
    return ((d["p_jm"] + d["p_combo"]) / 2 >= tb).values


def blend_with(d, col, tb):
    """Same blend, different JM-side score (E1)."""
    return ((d[col] + d["p_combo"]) / 2 >= tb).values


def two_threshold(d, a, b, col="p_jm"):
    """E2: explicit OR rule instead of a blend + single threshold."""
    return ((d[col] >= a) | (d["p_combo"] >= b)).values


def jm_only(d, tb=0.60, col="p_jm"):
    """v6-style: JM's conviction alone."""
    return (d[col] >= tb).values


def calibrate(d, col, target_rate, blend=True, lo=0.0, hi=1.0):
    """Pick the threshold that reproduces a target bear-gate pass RATE, so a
    new score is compared on its ranking quality, not on an arbitrary cut."""
    x = (d[col] + d["p_combo"]) / 2 if blend else d[col]
    q = float(np.nanquantile(x.values, 1 - target_rate))
    return min(max(q, lo), hi)
