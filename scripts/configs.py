"""REGISTRY of model weightings — the single source of truth for every arm tested.

WHY THIS EXISTS (user, 2026-07-31). The weighting configurations were previously
scattered across ma_family.ANCHORS / SHARES_BY_ANCHOR, ret_split.SPLITS,
ret_limit, raw_signed.ARMS and gatefree.ARMS, each re-deriving its own weight
vector. That made arms hard to compare and easy to get wrong - the split arms were
written as literal 30/20/10 (sum 0.60) and then paired with a 0.70 core weight,
which silently changed two variables at once.

Here every configuration is named, and the weight vector is MATERIALISED by one
function, so:
  - any test can be re-run by name, without re-deriving weights
  - a config's exact weights can be inspected without running anything
  - two arms differ only in what the name says they differ in

WEIGHT MECHANICS. feat_weights scales columns before clustering. Euclidean
distance is quadratic in that scaling, so a column's SHARE of squared distance is
w**2 (given sum(w**2) == 1). Hence w = sqrt(share). A family of n features sharing
`s` in total gets w = sqrt(s/n) each. `realised_shares()` verifies this rather
than trusting it.

USAGE
    import configs as cf
    w, cols = cf.weights("ret70flat")            # the vector and its column order
    cf.describe("ret70flat")                      # human-readable share table
    cf.dump_all("weights_registry.json")          # write every config to disk
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = (5, 21, 63)
RET = [f"ret_{h}" for h in H]


def _scale(ratio, share):
    """Ratio like 3:2:1 -> per-column shares summing exactly to `share`."""
    tot = float(sum(ratio))
    return tuple(share * r / tot for r in ratio)


# ---------------------------------------------------------------------------
# The feature base. ma* columns are added by the anchor, not carried in BASE,
# so an ma-anchored config does not also leak ma200 into the background block.
# ---------------------------------------------------------------------------
SIGNED = [f"{a}_{h}" for a in ("ret", "clv") for h in H]
UNSIGNED = [f"{a}_{h}" for a in ("vltup", "vltdn", "volup", "voldn") for h in H]
RAW18 = SIGNED + UNSIGNED
BASE = RAW18 + ["dd", "dd63", "mdd"]                       # 21 features

# name -> (anchor columns, total share, per-column shares or None for flat)
CONFIGS = {
    # --- test 1: which anchor, at a fixed core weight -----------------------
    "ret70flat":   (RET, 0.70, None),
    "ma3_70":      (["ma21", "ma50", "ma200"], 0.70, None),
    "ma200_70":    (["ma200"], 0.70, None),
    "maShort_70":  (["ma5", "ma21", "ma63"], 0.70, None),

    # --- test 2: core weight sweep on the ret anchor ------------------------
    "ret50flat":   (RET, 0.50, None),
    "ret60flat":   (RET, 0.60, None),
    "ret80flat":   (RET, 0.80, None),
    "ret90flat":   (RET, 0.90, None),
    "ret100flat":  (RET, 1.00, None),

    # --- test 3: internal split of the ret anchor, held at 70% --------------
    # expressed as RATIOS so the split question stays independent of the
    # weight question; the 60%-era literals 30/20/10 are reproduced by
    # _scale((3,2,1), 0.60) if that comparison is ever needed again.
    "ret70fast":   (RET, 0.70, _scale((3, 2, 1), 0.70)),
    "ret70slow":   (RET, 0.70, _scale((1, 2, 3), 0.70)),
    # 1:3:2 - mid-horizon heaviest (user, 2026-07-31). Untested: fast/slow both
    # tilt monotonically, so neither can tell whether ret_21 carries more signal
    # than its neighbours. This is the only shape that isolates the middle.
    "ret70mid":    (RET, 0.70, _scale((1, 3, 2), 0.70)),
    "ret70only5":  (RET, 0.70, (0.70, 0.0, 0.0)),
    "ret70only21": (RET, 0.70, (0.0, 0.70, 0.0)),
    "ret70only63": (RET, 0.70, (0.0, 0.0, 0.70)),

    # --- controls -----------------------------------------------------------
    # --- test 4: does the 63d ceiling bind? (user, 2026-07-31) -------------
    # ret126 out-separates every current feature on HSI. Tested as an ADD (4 dims)
    # and as two SWAPS, per the standing preference for swapping over adding.
    "ret4_70":     (["ret_5","ret_21","ret_63","ret_126"], 0.70, None),
    "retNo5_70":   (["ret_21","ret_63","ret_126"], 0.70, None),
    "retNo21_70":  (["ret_5","ret_63","ret_126"], 0.70, None),

    "uniform":     ([], 0.0, None),          # no anchor: every feature equal
}

MA_N = (5, 21, 50, 63, 200)                  # ma columns the feature builder makes


def columns(name):
    """Full column list for a config: BASE plus any anchor columns not in BASE."""
    anchor, _, _ = CONFIGS[name]
    return BASE + [c for c in anchor if c not in BASE]


def weights(name):
    """Materialise (weight_vector, columns) for a named config."""
    anchor, share, shares = CONFIGS[name]
    cols = columns(name)
    w = np.zeros(len(cols))
    if not anchor:                                    # uniform control
        w[:] = np.sqrt(1.0 / len(cols))
        return w, cols
    tgt = np.array([c in anchor for c in cols])
    if shares is None:
        w[tgt] = np.sqrt(share / tgt.sum())
    else:
        assert abs(sum(shares) - share) < 1e-9, (name, shares, share)
        pos = {c: i for i, c in enumerate(cols)}
        for c, sc in zip(anchor, shares):
            w[pos[c]] = np.sqrt(sc)
    if share < 1.0:
        w[~tgt] = np.sqrt((1.0 - share) / (~tgt).sum())
    return w, cols


def realised_shares(name):
    """What share of squared distance each column ACTUALLY gets. Verifies intent."""
    w, cols = weights(name)
    s = w ** 2
    return dict(zip(cols, s / s.sum()))


def n_effective(name):
    """Participation ratio 1/sum(share**2) - how many features the metric really uses.

    Reported per the standing request to quote effective feature count rather than
    just weight-by-family: a config can name 21 features and behave like 4.
    """
    s = np.array(list(realised_shares(name).values()))
    return float(1.0 / np.sum(s ** 2))


def describe(name):
    rs = realised_shares(name)
    anchor, share, _ = CONFIGS[name]
    lines = [f"{name}: {len(rs)} features, anchor={anchor or 'none'} "
             f"target share={share:.0%}, n_eff={n_effective(name):.1f}"]
    if anchor:
        lines.append(f"  anchor realised: {sum(rs[c] for c in anchor):.4f}")
    for c in (anchor or []):
        lines.append(f"    {c:<10} {rs[c]:.4f}")
    bg = [c for c in rs if c not in (anchor or [])]
    if bg:
        lines.append(f"  background: {len(bg)} features x {rs[bg[0]]:.4f} "
                     f"= {sum(rs[c] for c in bg):.4f}")
    return "\n".join(lines)


def dump_all(path=None):
    """Write every config's full weight vector to JSON for the record."""
    path = path or os.path.join(HERE, "weights_registry.json")
    out = {}
    for name in CONFIGS:
        w, cols = weights(name)
        out[name] = {
            "columns": cols,
            "weights": [round(float(x), 8) for x in w],
            "realised_shares": {k: round(float(v), 8)
                                for k, v in realised_shares(name).items()},
            "n_features": len(cols),
            "n_effective": round(n_effective(name), 3),
            "anchor": CONFIGS[name][0],
            "target_share": CONFIGS[name][1],
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(CONFIGS)
    for n in names:
        print(describe(n))
        print()
    p = dump_all()
    print(f"registry written -> {p}")
