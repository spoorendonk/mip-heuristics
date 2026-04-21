#!/usr/bin/env python3
"""Aggregate `[Sequential]` per-heuristic rates across a results tree.

Feeds the calibration loop for `kWeight*` in `src/mode_dispatch.cpp`
(issue #71).  Each run in `bench/results/<config>/seed<N>/<inst>.log`
emits one `[Sequential] heur=<name> effort=<N> wall_ms=<X>` line per
heuristic.  We:

  * geomean `effort_per_ms` per heuristic across all instances/seeds
    (geometric mean — `effort_per_ms` spans multiple orders of magnitude
    between heuristics and instances, so arithmetic mean is dominated
    by a handful of outlier instances);
  * compute the max/min ratio ("drift") between heuristics;
  * suggest weight multipliers proportional to `effort_per_ms` so that
    the weighted allocation in `run_sequential` yields equal wall-clock
    spend per heuristic.

Why weights are proportional to `effort_per_ms`.  Given a single
`budget` in effort units split by weights, each heuristic's share
`share_i = budget * w_i / sum(w)` is consumed at its rate `r_i`
(`effort_per_ms_i`), so wall-ms is `share_i / r_i`.  Equal wall-ms
across heuristics requires `w_i / r_i = const`, i.e. `w_i ∝ r_i`.
Fast-per-effort heuristics (high `effort_per_ms`) get a larger share;
slow-per-effort heuristics get less effort because they'd otherwise
run longer for the same effort allocation.

Fails (exit 2) if drift between heuristics exceeds `--max-drift`
(default 3× per the acceptance criterion in issue #71).  That makes
this script usable as a regression gate in CI, and as a one-shot
calibration helper when editing the constants.

Exit codes:
  0  drift within threshold (or --max-drift unreachable with one heuristic)
  1  no `[Sequential]` samples, or a bad `--reference` argument
  2  drift exceeds `--max-drift`
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

from parse_highs_log import parse_log_file


def walk_logs(root: str):
    """Yield `.log` paths under `root`."""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".log"):
                yield os.path.join(dirpath, name)


def geomean(values: list[float]) -> float:
    """Geometric mean, ignoring non-positive entries."""
    clean = [v for v in values if v > 0]
    if not clean:
        return 0.0
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def aggregate(root: str) -> dict[str, list[float]]:
    """Return {heuristic: [effort_per_ms samples]} across all logs."""
    by_heur: dict[str, list[float]] = defaultdict(list)
    for path in walk_logs(root):
        try:
            result = parse_log_file(path)
        except OSError:
            continue
        for s in result.sequential_samples:
            if s.effort_per_ms > 0:
                by_heur[s.heuristic].append(s.effort_per_ms)
    return by_heur


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Directory tree containing HiGHS .log files")
    parser.add_argument(
        "--max-drift",
        type=float,
        default=3.0,
        help="Fail if max/min geomean(effort_per_ms) across heuristics exceeds this ratio (default 3.0).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Normalise suggested weights so this heuristic's weight is 1.0 (default: slowest).",
    )
    args = parser.parse_args()

    by_heur = aggregate(args.root)
    if not by_heur:
        print(f"No [Sequential] samples found under {args.root}", file=sys.stderr)
        return 1

    # Geomean effort_per_ms per heuristic; suggested weight is proportional
    # to this rate so that each heuristic's share of a common effort budget
    # translates to equal wall-clock spend (see module docstring).
    summary = {}
    for heur, samples in by_heur.items():
        gm = geomean(samples)
        summary[heur] = (gm, len(samples))

    expected = {"fj", "fpr", "local_mip", "scylla"}
    missing = expected - summary.keys()
    if missing:
        # local_mip is the usual offender: it early-returns in
        # `src/mode_dispatch.cpp`'s sequential chain when no incumbent
        # exists yet, so its [Sequential] line is absent on cold solves.
        # Flag so kWeightLocalMip doesn't get silently skipped in a
        # calibration pass.
        print(
            f"WARNING: no [Sequential] samples for: {', '.join(sorted(missing))} "
            f"— their kWeight* cannot be recalibrated from this run.",
            file=sys.stderr,
        )

    if args.reference is not None and args.reference not in summary:
        print(
            f"ERROR: --reference '{args.reference}' not found in samples (have: "
            f"{', '.join(sorted(summary))}).",
            file=sys.stderr,
        )
        return 1

    heurs = sorted(summary.keys())
    print(f"{'heuristic':<12} {'n':>5} {'gm(eff/ms)':>14} {'ms/eff':>12} {'suggested_w':>14}")
    effs = [summary[h][0] for h in heurs]
    max_eff = max(effs)
    min_eff = min(effs)
    drift = (max_eff / min_eff) if min_eff > 0 else float("inf")

    # Normalise weights against the heuristic with the smallest effort_per_ms
    # (i.e. the slowest-per-effort heuristic anchors w=1.0; faster ones scale
    # up proportionally).  --reference overrides which heuristic anchors.
    ref = args.reference if args.reference is not None else min(summary, key=lambda h: summary[h][0])
    scale = summary[ref][0]
    for h in heurs:
        gm, n = summary[h]
        ms_per_effort = (1.0 / gm) if gm > 0 else float("inf")
        w = (gm / scale) if scale > 0 else float("inf")
        print(f"{h:<12} {n:>5d} {gm:>14.3f} {ms_per_effort:>12.6f} {w:>14.3f}")

    print(f"\nDrift (max/min effort_per_ms) = {drift:.2f}× (limit {args.max_drift:.2f}×)")
    if drift > args.max_drift:
        print("FAIL: drift exceeds threshold — recalibrate kWeight* in src/mode_dispatch.cpp.",
              file=sys.stderr)
        return 2
    print("OK: drift within threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
