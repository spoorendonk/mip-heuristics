#!/usr/bin/env python3
"""Read an Ablation C tree (issue #165): does `fpr_lp` fire, and does it yield?

Two counts per instance, and they answer different questions:

* **dispatches** — `[Heur] name=fpr_lp phase=dive` lines, one per dive-time
  dispatch.  Emitted at `log_dev_level=3` only.  Whether a run was traced is
  read from its sibling `.opts`, **not** from whether it produced any such
  line: this stage runs `suite=fpr_lp`, so the only heuristic that can emit a
  `[Heur]` line at all is the one being measured, and zero lines is therefore
  *data* — it says the dive never dispatched.  Inferring "untraced" from it
  would silently discard the instances where the answer is no.  (The first
  version of this file did exactly that and reported 11 of 49 runs as
  unreadable when every one of them carried `log_dev_level = 3`.)
  `abandoned_setup=1` marks a dispatch that gave up in setup against the wall
  clock without searching, which is a third thing distinct from both "ran and
  found nothing" and "never ran" (#119).
* **yields** — incumbent lines carrying the `D` solution-source character,
  which is `kSolutionSourceFprLp`.  These are in the *ordinary* log at any
  level, because HiGHS prints the source of every incumbent it accepts.  A
  yield is therefore an accepted improvement, not merely an offer.

The gate this exists for: **plentiful dispatches with zero yields is a
decisive null** — `fpr_lp` cannot produce an accepted solution even given
every advantage — and it needs no power calculation, because a count of zero
is a count of zero at any n.  **Few dispatches is not a null at all**: it says
the run was too short for B&B to reach dive nodes, and the answer is to raise
the limit rather than to conclude.  The report states which of the two it is
instead of leaving the reader to infer it.

A **killed** run is excluded from the fire rate rather than counted as a zero:
the harness SIGKILLs a solve that overran its limit, so the log stops
mid-solve and its dispatch count is a lower bound, not an observation.

Usage:
    bench/analyze_ablation_c.py bench/results/ablation_c/capability/fpr_lp/seed0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_highs_log import parse_log

# `kSolutionSourceFprLp` prints as `D` (third_party/highs_patch/apply_patch.cmake
# writes the source-to-string entry).  Named here rather than inlined so the
# one place it is spelled is greppable from the patch.
FPR_LP_SOURCE = "D"

# Below this many dispatches over the whole tree, a zero yield count says
# nothing about the heuristic -- it says the runs were too short.  Deliberately
# a *reported threshold* rather than a silent one: the verdict line names it.
MIN_DISPATCHES_FOR_A_NULL = 20


@dataclass
class InstanceCounts:
    instance: str
    dispatches: int
    abandoned: int
    yields: int
    traced: bool
    killed: bool


def was_traced(log: Path, result) -> bool:
    """Whether this run requested `log_dev_level >= 3`.

    From the `.opts` the run was given, because that is the *request* rather
    than a consequence of it -- `make_archive.py` records instrumentation both
    ways for the same reason.  The observed fallback is only for a log with no
    `.opts` beside it, and it is one-directional: `[Heur]` lines prove
    tracing, their absence proves nothing.
    """
    opts = log.with_suffix(".opts")
    if opts.exists():
        for line in opts.read_text(errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "log_dev_level":
                try:
                    return int(value.strip()) >= 3
                except ValueError:
                    return False
        return False
    return bool(result.heuristic_samples)


def count_one(path: Path) -> InstanceCounts:
    result = parse_log(path.read_text(errors="replace"))
    dive = [
        s for s in result.heuristic_samples if s.name == "fpr_lp" and s.phase == "dive"
    ]
    return InstanceCounts(
        instance=path.stem,
        dispatches=len(dive),
        # `abandoned_setup` is None on a log written before #119, which is not
        # the same as False; count only what is known to be a bail.
        abandoned=sum(1 for s in dive if s.abandoned_setup),
        yields=sum(1 for inc in result.incumbents if inc.source == FPR_LP_SOURCE),
        traced=was_traced(path, result),
        # The harness kills a solve that overran its limit, so the log stops
        # mid-solve: this run's counts are lower bounds, not observations.
        killed=result.killed,
    )


def report(rows: list[InstanceCounts], out=sys.stdout) -> None:
    traced = [r for r in rows if r.traced]
    killed = [r for r in rows if r.killed]
    # The fire rate's denominator: traced, and not cut short mid-solve.
    observable = [r for r in traced if not r.killed]
    total_d = sum(r.dispatches for r in rows)
    total_a = sum(r.abandoned for r in rows)
    total_y = sum(r.yields for r in rows)
    fired = [r for r in observable if r.dispatches > 0]
    yielded = [r for r in rows if r.yields > 0]

    print(
        f"{len(rows)} run(s), {len(traced)} traced, {len(killed)} killed\n",
        file=out,
    )
    print(f"{'instance':34s} {'dispatch':>8s} {'bailed':>7s} {'yield':>6s}", file=out)
    print("-" * 59, file=out)
    # Most dispatches first: the instances that exercise the heuristic hardest
    # are the ones a zero yield has to be read against.
    for r in sorted(rows, key=lambda r: (-r.dispatches, r.instance)):
        note = ""
        if not r.traced:
            note = "  (untraced)"
        elif r.killed:
            note = "  (killed -- lower bound)"
        print(
            f"{r.instance:34s} {r.dispatches:8d} {r.abandoned:7d} {r.yields:6d}{note}",
            file=out,
        )
    print("-" * 59, file=out)
    print(f"{'total':34s} {total_d:8d} {total_a:7d} {total_y:6d}\n", file=out)

    if observable:
        print(
            f"fired on {len(fired)}/{len(observable)} observable instances "
            f"(traced, not killed); yielded on {len(yielded)}/{len(rows)}",
            file=out,
        )
    if len(traced) < len(rows):
        print(
            f"WARNING: {len(rows) - len(traced)} run(s) carry no [Heur] lines. "
            "Dispatch counts are unreadable there -- rerun with PLATO_DEV_LOG=1. "
            "Yield counts are still valid: the source character is in every log.",
            file=out,
        )

    print(file=out)
    if total_y > 0:
        print(
            f"VERDICT: fpr_lp yielded {total_y} accepted incumbent(s) on "
            f"{len(yielded)} instance(s). Not a null -- proceed to the paired "
            "contribution stage.",
            file=out,
        )
    elif total_d >= MIN_DISPATCHES_FOR_A_NULL:
        print(
            f"VERDICT: {total_d} dispatches, zero accepted incumbents. "
            "Decisive null: fpr_lp fires and produces nothing.",
            file=out,
        )
    elif traced:
        print(
            f"VERDICT: inconclusive -- only {total_d} dispatch(es), below the "
            f"{MIN_DISPATCHES_FOR_A_NULL} this reads as adequate exercise. "
            "Raise the time limit; do not read this as a null.",
            file=out,
        )
    else:
        print("VERDICT: unreadable -- no traced runs.", file=out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("results_dir", type=Path, help="a seed directory of .log files")
    args = p.parse_args(argv)

    logs = sorted(args.results_dir.glob("*.log"))
    if not logs:
        print(f"no .log files under {args.results_dir}", file=sys.stderr)
        return 2
    report([count_one(f) for f in logs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
