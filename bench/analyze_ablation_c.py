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

The second half of the ablation is the paired contribution: the same
configuration with and without `fpr_lp`, on identical instances, scored on the
campaign metric.  It lives here rather than in `analyze_results.py` because it
needs C0's labels -- and because the labels are what make it readable.  The
fire rate is bimodal, so a mean over all instances mixes a real effect with the
instances where the heuristic never ran; **the instances where C0 saw no
dispatch are the null control**, and their ratio is the check that the pairing
is tight enough for anything else here to mean something.

Usage:
    bench/analyze_ablation_c.py capability <seed-dir>
    bench/analyze_ablation_c.py contribution <control-dir> <arm-dir> [--labels <c0-dir>]
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_results import build_best_known, load_results, parse_solu_file
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


# The campaign's own limit; the primal integral is defined against it.
TIME_LIMIT = 600.0
# Below this many instances a subgroup's interval is too wide to print as
# anything but a count.
MIN_SUBGROUP = 3
# Matches `analyze_results.py`'s SGM shift, so an instance solved before the
# first display row keeps a finite ratio.
SHIFT = 1e-3


@dataclass
class Paired:
    """Two arms compared on the log of the primal-integral ratio.

    Log-ratio rather than difference because the integral spans orders of
    magnitude across instances.  Paired because instance difficulty is the
    dominant variance component in cross-instance MIP benchmarking, and it
    cancels exactly when both arms run the same instance.
    """

    n: int
    ratio: float
    lo: float
    hi: float
    t: float
    sd: float

    @property
    def separated(self) -> bool:
        """Whether the 95% interval excludes no-effect."""
        return not (self.lo <= 1.0 <= self.hi)

    def detectable(self) -> float:
        """Smallest difference this n resolves at 80% power."""
        return math.exp(math.sqrt(8 * self.sd**2 / self.n)) - 1

    def n_for(self, ratio: float) -> int:
        """The n that would resolve `ratio` at 80% power."""
        return math.ceil(8 * self.sd**2 / math.log(ratio) ** 2)


def paired(values: list[float]) -> Paired | None:
    if len(values) < 2:
        return None
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(n)
    # A zero-variance sample is a legitimate input -- every instance moved by
    # the same factor, or none moved at all -- and it has no finite t.  The
    # sign carries the direction so `separated` still reads correctly: a
    # degenerate sample away from 1.0 has an interval of zero width that
    # excludes it.
    if sd == 0.0:
        t = math.copysign(math.inf, mean) if mean else 0.0
    else:
        t = mean / (sd / math.sqrt(n))
    return Paired(
        n=n,
        ratio=math.exp(mean),
        lo=math.exp(mean - half),
        hi=math.exp(mean + half),
        t=t,
        sd=sd,
    )


def log_ratios(control, arm, refs, instances: list[str]) -> list[float]:
    return [
        math.log(
            (arm[i].primal_integral(TIME_LIMIT, refs[i]) + SHIFT)
            / (control[i].primal_integral(TIME_LIMIT, refs[i]) + SHIFT)
        )
        for i in instances
    ]


def contribution(
    control_dir: Path, arm_dir: Path, labels_dir: Path | None, out=sys.stdout
) -> int:
    dirs = {"control": str(control_dir), "arm": str(arm_dir)}
    results = load_results(str(control_dir.parent), list(dirs), config_dirs=dirs)
    if "control" not in results or "arm" not in results:
        print("both directories must hold a seed0/ of logs", file=sys.stderr)
        return 2
    shared = sorted(set(results["control"][0]) & set(results["arm"][0]))
    if not shared:
        print("no instances in common", file=sys.stderr)
        return 2
    refs = build_best_known(
        results, list(dirs), shared, parse_solu_file("bench/miplib2017-v36.solu")
    )
    control, arm = results["control"][0], results["arm"][0]

    def row(label: str, instances: list[str]) -> Paired | None:
        p = paired(log_ratios(control, arm, refs, instances))
        if p is None or p.n < MIN_SUBGROUP:
            print(f"{label:32s} n={len(instances):3d}  (too few)", file=out)
            return None
        mark = "  *" if p.separated else ""
        print(
            f"{label:32s} n={p.n:3d}  ratio={p.ratio:.3f}  "
            f"CI=[{p.lo:.2f}, {p.hi:.2f}]  t={p.t:+.2f}{mark}",
            file=out,
        )
        return p

    print(
        f"{len(shared)} paired instance(s), arm over control (lower is better)\n",
        file=out,
    )
    overall = row("overall", shared)

    if labels_dir is not None:
        counts = {
            r.instance: r for r in (count_one(f) for f in labels_dir.glob("*.log"))
        }
        # Partitioned by an *independent* run rather than by the outcome being
        # measured, so this is not outcome selection -- but the labels come
        # from C0's configuration and not from these runs, so they are a proxy
        # for "an instance where fpr_lp engages", and the split is post-hoc.
        # Read it as a mechanism, not as a second headline.
        known = [
            i
            for i in shared
            if i in counts and counts[i].traced and not counts[i].killed
        ]
        print(
            "\nby Stage C0 label (proxy: C0's configuration, not these runs)", file=out
        )
        row("  C0 yielded", [i for i in known if counts[i].yields > 0])
        row(
            "  C0 fired, never yielded",
            [i for i in known if counts[i].dispatches > 0 and counts[i].yields == 0],
        )
        null = row(
            "  C0 never fired (null control)",
            [i for i in known if counts[i].dispatches == 0],
        )
        if null is not None and null.separated:
            print(
                "\nWARNING: the null control is separated. The arms differ where "
                "the heuristic never ran, so something other than fpr_lp moved -- "
                "do not read the other subgroups until that is explained.",
                file=out,
            )

    if overall is not None:
        print(
            f"\npower: n={overall.n} resolves {overall.detectable():.1%}; "
            f"the observed {abs(overall.ratio - 1):.1%} would need "
            f"n={overall.n_for(overall.ratio)}",
            file=out,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="stage", required=True)

    cap = sub.add_parser("capability", help="stage C0: does fpr_lp fire, and yield?")
    cap.add_argument("results_dir", type=Path, help="a seed directory of .log files")

    con = sub.add_parser("contribution", help="stage C1: the paired campaign metric")
    con.add_argument(
        "control_dir", type=Path, help="the arm without fpr_lp, up to seed0/"
    )
    con.add_argument("arm_dir", type=Path, help="the arm with it, up to seed0/")
    con.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="a stage C0 seed directory, to partition by fire/yield",
    )

    args = p.parse_args(argv)

    if args.stage == "capability":
        logs = sorted(args.results_dir.glob("*.log"))
        if not logs:
            print(f"no .log files under {args.results_dir}", file=sys.stderr)
            return 2
        report([count_one(f) for f in logs])
        return 0
    return contribution(args.control_dir, args.arm_dir, args.labels)


if __name__ == "__main__":
    raise SystemExit(main())
