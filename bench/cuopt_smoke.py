#!/usr/bin/env python3
"""Smoke A/B of the cuOpt ideas against main, one script for every host (#238).

    bench/cuopt_smoke.py size              what `run` has left, in core-hours
    bench/cuopt_smoke.py run               every pending solve, resumable
    bench/cuopt_smoke.py report            the ratio table, in Markdown

Every arm is `all` at the binary's defaults with one option moved, so each idea
is measured against main with the others off.  Main against itself at another
seed is the noise floor an arm's ratio has to clear, which matters at few seeds
and on a machine in use: at one seed per arm `main` runs one more for it, at
several its own seeds serve.

Results land under `bench/results/cuopt-smoke/<host>`, one tree per host, since
times are never compared across hosts.  The first `run` copies the binary into
the tree and every solve uses that copy, so a rebuild cannot put two binaries in
one comparison; a later `run` with a different binary is refused.

`SMOKE_WRAP` runs each solve under a wrapper, so the host's lock or budget is
held per solve and a build can run between solves:

    SMOKE_WRAP="xk lock --"                   Linux: the machine lock, exclusive
    SMOKE_WRAP="wcp build --here --jobs 8 --" macOS: no lock there; a solve charged
                                              its cores takes all the build budget,
                                              which reduces overlap with a build
                                              rather than excluding it

Smoke only at one seed: a verdict from those runs is a choice of what to measure
properly, never a default change.  `run_unsolved15.sh` is the measured run (#244).
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from analyze_results import (
    SeedPairing,
    apply_reference_formula,
    build_best_known,
    integral_shift,
    load_results,
    paired,
    parse_mipfeas_z,
    parse_solu_file,
    read_instance_list,
    score_integral,
    seed_paired_values,
)
from parse_highs_log import SolveResult
from run_benchmark import existing_log

BENCH = Path(__file__).resolve().parent
TIME_LIMIT = 60.0
BASELINE = "main"
CONFIG = "all"

# Where #206's scan found a structural persona.  On the other instances of the
# list every slot stays seeded, so that arm is main plus one O(nnz) detection
# pass and a solve there would measure seed noise, which `main` already does.
PERSONA_INSTANCES = frozenset(
    {
        "csched008",
        "dano3_3",
        "fiball",
        "neos5",
        "nw04",
        "qap10",
        "roll3000",
        "satellites2-60-fs",
        "seymour1",
        "triptim1",
        "unitcal_7",
    }
)

# The shift of the time-to-first-incumbent ratio, in seconds: a difference
# under about a second is not one a 60 s smoke run can tell from noise.
T_FIRST_SHIFT = 1.0

Results = dict[str, dict[int, dict[str, SolveResult]]]


@dataclass(frozen=True)
class Arm:
    name: str
    options: tuple[str, ...] = ()
    instances: frozenset[str] | None = None  # None: the whole list


ARMS: tuple[Arm, ...] = (
    Arm(BASELINE),
    Arm("early_fj", ("mip_heuristic_early_fj_effort=1.0",)),
    Arm("personas", ("mip_heuristic_fj_personas=true",), PERSONA_INSTANCES),
    Arm("ruin", ("mip_heuristic_ruin_effort=1.0",)),
    Arm("cut", ("mip_heuristic_objective_cut=true",)),
    Arm("recombine", ("mip_heuristic_recombine_effort=1.0",)),
)


@dataclass(frozen=True)
class Solve:
    arm: Arm
    instance: str
    seed: int


def noise_seed(seeds: list[int]) -> int:
    """The extra seed `main` runs when the arms have only one."""
    return max(seeds) + 1


def main_seeds(seeds: list[int]) -> list[int]:
    """`main`'s seeds: two or more already are its noise floor, so none is added."""
    return seeds if len(seeds) > 1 else [*seeds, noise_seed(seeds)]


def plan(instances: list[str], seeds: list[int], arms: tuple[Arm, ...]) -> list[Solve]:
    """Every solve, instance by instance, so a run cut short is still paired."""
    solves: list[Solve] = []
    for instance in instances:
        for arm in arms:
            if arm.instances is not None and instance not in arm.instances:
                continue
            arm_seeds = main_seeds(seeds) if arm.name == BASELINE else seeds
            solves.extend(Solve(arm, instance, s) for s in arm_seeds)
    return solves


def pending(solves: list[Solve], out: Path) -> list[Solve]:
    """What is left, by the test `run_benchmark --skip-existing` applies."""
    return [
        s
        for s in solves
        if not existing_log(str(out / s.arm.name), CONFIG, s.instance, s.seed)
    ]


def pin_binary(binary: str, out: Path) -> str:
    """The tree's own copy of `binary`; raises if the tree holds another build."""
    pinned = out / "bin" / "highs"
    if not pinned.exists():
        pinned.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, pinned)
    elif not filecmp.cmp(binary, pinned, shallow=False):
        raise ValueError(
            f"{out} was run with another binary than {binary}: its arms would pair "
            "across builds.  Use a fresh --output, or --binary "
            f"{pinned} to resume."
        )
    return str(pinned)


def core_hours(runs: int, cores: int) -> float:
    """Rule 17's size: an upper bound, since a solve may end before the limit."""
    return runs * TIME_LIMIT / 3600.0 * cores


def solve_command(solve: Solve, args: argparse.Namespace) -> list[str]:
    """One solve, from a one-line list: run_benchmark's --start counts only the
    instances it found on disk, so an index into the full list could name another.
    """
    one_instance = args.output / "one-instance" / f"{solve.instance}.txt"
    one_instance.parent.mkdir(parents=True, exist_ok=True)
    one_instance.write_text(f"{solve.instance}\n")
    command = [
        *shlex.split(os.environ.get("SMOKE_WRAP", "")),
        sys.executable,
        str(BENCH / "run_benchmark.py"),
        "--instances",
        str(one_instance),
        "--binary",
        args.binary,
        "--time-limit",
        str(TIME_LIMIT),
        "--seeds",
        str(solve.seed),
        "--configs",
        CONFIG,
        "--skip-existing",
        "--output",
        str(args.output / solve.arm.name),
    ]
    if args.data_dir:
        command += ["--data-dir", args.data_dir]
    if solve.arm.options:
        command += ["--extra-options", *solve.arm.options]
    return command


def run(todo: list[Solve], args: argparse.Namespace) -> int:
    """Run every pending solve; rerun to resume.

    3 when the wall cap stopped it.  4 when a solve left no log: run_benchmark
    exits 0 after a failed solve (`.log.err`) or a missing instance file, and
    that solve would otherwise only be a smaller n in the report.
    """
    cap = 1.5 * len(todo) * TIME_LIMIT if args.wall_cap is None else args.wall_cap
    args.binary = pin_binary(args.binary, args.output)
    start = time.monotonic()
    for i, solve in enumerate(todo, 1):
        if time.monotonic() - start > cap:
            print(f"wall cap of {cap:.0f} s hit, {len(todo) - i + 1} solves left")
            return 3
        print(
            f"[{i}/{len(todo)}] {solve.arm.name} {solve.instance} seed {solve.seed}",
            flush=True,
        )
        code = subprocess.call(solve_command(solve, args))
        if code != 0:
            return code
    left = pending(todo, args.output)
    for solve in left:
        print(f"no log: {solve.arm.name} {solve.instance} seed {solve.seed}")
    return 4 if left else 0


def t_first(run_result: SolveResult, _instance: str) -> float:
    """Time to the first incumbent, capped at the limit, which is also what a run
    without one scores: a late incumbent must not rank below none, and the primal
    integral ignores one past the limit too.
    """
    found = run_result.time_to_first_feasible
    return TIME_LIMIT if found is None else min(found, TIME_LIMIT)


def table_row(
    label: str,
    pi: SeedPairing,
    t1: SeedPairing,
    feasible: tuple[int, int],
    pi_shift: float,
) -> str:
    """One row: an arm's ratios to its baseline, lower is better."""

    def cells(pairing: SeedPairing, shift: float) -> str:
        if len(pairing.a) < 2:
            return "n/a | n/a | n/a | n/a"
        st = paired(pairing.a, pairing.b, shift)
        ratios = [(x + shift) / (y + shift) for x, y in zip(pairing.a, pairing.b)]
        return (
            f"{st.ratio:.3f} | {st.median_ratio:.3f} | "
            f"[{st.ci_lo:.3f}, {st.ci_hi:.3f}] | {st.better}/{st.tied}/{st.worse} "
            f"(per instance {min(ratios):.3g} to {max(ratios):.3g})"
        )

    return (
        f"| {label} | {len(pi.a)} | {cells(pi, pi_shift)} | "
        f"{feasible[0]} vs {feasible[1]} | {cells(t1, T_FIRST_SHIFT)} |"
    )


def feasible_counts(results: Results, arm: str, base: str) -> tuple[int, int]:
    """Runs with an incumbent inside the limit, over the runs the two share."""

    def feasible(r: SolveResult) -> bool:
        return any(inc.time < TIME_LIMIT for inc in r.incumbents)

    in_arm = in_base = 0
    for seed, runs in results[arm].items():
        for instance, r in runs.items():
            other = results[base].get(seed, {}).get(instance)
            if other is not None:
                in_arm += feasible(r)
                in_base += feasible(other)
    return in_arm, in_base


def report_rows(
    results: Results,
    seeds: list[int],
    instances: list[str],
    integral: Callable[[SolveResult, str], float],
) -> tuple[list[str], list[str]]:
    """The table's rows, the noise floor first, and what each row left out."""
    shift = integral_shift()

    scope = {arm.name: arm.instances for arm in ARMS}
    notes: list[str] = []

    def row(label: str, tree: Results, arm: str, base: str) -> str:
        pi = seed_paired_values(tree, arm, base, instances, integral, shift)
        t1 = seed_paired_values(tree, arm, base, instances, t_first, T_FIRST_SHIFT)
        planned = scope.get(arm)
        missing = [i for i in pi.unpaired if planned is None or i in planned]
        if missing:
            notes.append(f"{label}: no paired run on {', '.join(missing)}")
        # Ignoring a noise seed only `main` runs, which is no failed run.
        short = [
            i
            for i, (ran, base_ran) in pi.differing.items()
            if (set(ran) ^ set(base_ran)) & set(seeds)
        ]
        if short:
            notes.append(f"{label}: fewer paired seeds on {', '.join(short)}")
        return table_row(label, pi, t1, feasible_counts(tree, arm, base), shift)

    # Main on one seed as "main", and on a later seed under the same seed number
    # of its own name, so the seed pairing compares the two.
    main_runs = results.get(BASELINE, {})
    rows: list[str] = []
    for first, other in combinations(main_seeds(seeds), 2):
        if first in main_runs and other in main_runs:
            noise_tree: Results = {
                "noise": {first: main_runs[other]},
                BASELINE: {first: main_runs[first]},
            }
            label = f"noise floor (main seed {other} vs {first})"
            rows.append(row(label, noise_tree, "noise", BASELINE))
    # Arms pair on the seeds they ran, so a noise seed of `main`'s takes no part.
    rows.extend(
        row(arm, results, arm, BASELINE)
        for arm in results
        if arm != BASELINE and results[arm]
    )
    return rows, notes


def seed_spread(
    results: Results,
    seeds: list[int],
    instances: list[str],
    integral: Callable[[SolveResult, str], float],
) -> list[str]:
    """Each arm's PI ratio at each seed: one seed against one, the same footing as
    the noise-floor rows, where an arm's table row averages over its seeds.
    """
    shift = integral_shift()
    lines: list[str] = []
    for arm in results:
        if arm == BASELINE:
            continue
        cells: list[str] = []
        ratios: list[float] = []
        for seed in seeds:
            tree: Results = {
                arm: {seed: results[arm].get(seed, {})},
                BASELINE: {seed: results[BASELINE].get(seed, {})},
            }
            pi = seed_paired_values(tree, arm, BASELINE, instances, integral, shift)
            if len(pi.a) < 2:
                cells.append(f"seed {seed} n/a")
                continue
            ratios.append(paired(pi.a, pi.b, shift).ratio)
            short = f" (n={len(pi.a)})" if len(pi.a) < len(instances) else ""
            cells.append(f"seed {seed} {ratios[-1]:.3f}{short}")
        spread = f" (range {max(ratios) - min(ratios):.3f})" if len(ratios) > 1 else ""
        lines.append(f"{arm}: {', '.join(cells)}{spread}")
    return lines


TABLE_HEADER = (
    "| arm | n | PI ratio | PI median | PI 95% CI | better/tied/worse | "
    "feasible runs (arm vs main) | T1st ratio | T1st median | T1st 95% CI | "
    "better/tied/worse |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def report(args: argparse.Namespace) -> int:
    names = [arm.name for arm in ARMS]
    config_dirs = {name: str(args.output / name / CONFIG) for name in names}
    results = {
        name: runs
        for name, runs in load_results(str(args.output), names, config_dirs).items()
        if any(runs.values())
    }
    if BASELINE not in results:
        print(f"no {BASELINE} runs under {args.output}", file=sys.stderr)
        return 1
    instances = read_instance_list(args.instances)
    solu = parse_solu_file(str(BENCH / "miplib2017-v36.solu"))
    refs = apply_reference_formula(
        build_best_known(results, list(results), instances, solu, TIME_LIMIT),
        parse_mipfeas_z(str(BENCH / "mipfeas_optimal_objective.csv")),
    )

    def integral(r: SolveResult, instance: str) -> float:
        return score_integral(r, TIME_LIMIT, refs.get(instance))

    print(f"{args.output}: {TIME_LIMIT:.0f} s, ratios to main, lower is better\n")
    print(TABLE_HEADER)
    rows, notes = report_rows(results, args.seeds, instances, integral)
    print("\n".join(rows))
    if notes:
        print("\n" + "\n".join(notes))
    if len(args.seeds) > 1:
        print("\nPI ratio to main per seed, against the noise-floor rows:\n")
        print("\n".join(seed_spread(results, args.seeds, instances, integral)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["size", "run", "report"])
    parser.add_argument("--instances", default=str(BENCH / "instances_small.txt"))
    parser.add_argument("--binary", default=str(BENCH.parent / "build/bin/highs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCH / "results" / "cuopt-smoke" / socket.gethostname().split(".")[0],
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0], help="Seeds of every arm"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[arm.name for arm in ARMS],
        choices=[arm.name for arm in ARMS],
        help="Arms to run; main is what the others pair against",
    )
    parser.add_argument(
        "--wall-cap",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Stop launching solves past this (default 1.5 x the pending size)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "report":
        return report(args)
    arms = tuple(arm for arm in ARMS if arm.name in args.arms)
    instances = read_instance_list(args.instances)
    todo = pending(plan(instances, args.seeds, arms), args.output)
    if args.command == "run":
        try:
            return run(todo, args)
        except (ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    cores = os.cpu_count() or 1
    print(
        f"{len(todo)} solves x {TIME_LIMIT:.0f} s x {cores} cores: at most "
        f"{core_hours(len(todo), cores):.1f} core-h, "
        f"{core_hours(len(todo), 1):.2f} h wall"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
