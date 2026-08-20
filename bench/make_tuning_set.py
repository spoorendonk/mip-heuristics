#!/usr/bin/env python3
"""Derive a stratified tuning subset from a vanilla PLATO results tree.

Issue #103.  The campaign needs a tuning/dev subset that is representative of
what the benchmark *measures*.  PLATO `mipfeas` scores the primal integral,
which responds to how hard it is to **find a feasible solution** — so the
stratification variable here is vanilla's time to first feasible solution, not
the time to prove optimality that `bench/instances_small.txt` was selected on.
(That list stays exactly as it is: it is the recorded input of the `kWeight*`
calibration and a re-measurement on another set is not comparable to it.)

Sampling proportionally across the whole spectrum — including the instances
that find a solution immediately — is deliberate.  The budget question is
two-sided: presolve effort buys feasibility where feasibility is hard and is
pure overhead where branch-and-bound has an incumbent in the first second.  A
subset drawn only from the hard end sees the benefit without the cost.

Strata are half-open intervals on the aggregated time, split at `--boundaries`,
plus a bucket for runs that never became feasible.  The default split
`1,10,100,600` gives the five suggested strata (immediate / fast / moderate /
hard / never) and one overflow bucket, `>=600s`, for a solution found at or
past the time limit — such a run is *not* never-feasible and must not be
filed as one.

Output is an instance list in the same commented format the other lists in
`bench/` use, with a header recording the source tree, the seed, the
boundaries and the per-stratum counts of both the full set and the sample.  It
carries no timestamp on purpose: the same tree and the same `--seed` must
produce a byte-identical file, which is what makes the subset a recorded,
reproducible input rather than a hand-picked one.

The list goes to `--output` (stdout by default); the distribution report and
every diagnostic go to stderr, so redirecting stdout to a file yields the list
alone and still shows the report.

Exit codes:
  0  a subset was produced
  1  bad arguments or an unreadable tree
  2  the tree does not cover the reference instance list (see --help)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# "Which directory holds the vanilla arm?" is the same question the
# cannibalization tables answer, so the preference order is shared rather than
# copied — a second list here would drift the first time a config is renamed.
from analyze_results import CANNIBALIZATION_BASELINE_NAMES as VANILLA_CONFIG_NAMES
from analyze_results import load_results
from parse_highs_log import SolveResult
from run_benchmark import load_instances

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

# Split points, in seconds, of the time-to-first-feasible axis.  The last one
# is the benchmark time limit; see the module docstring on the overflow
# bucket it creates.
DEFAULT_BOUNDARIES: tuple[float, ...] = (1.0, 10.0, 100.0, 600.0)

# The bucket for runs that never produced a feasible solution.  Not a time,
# so it is not derived from the boundaries.
NEVER_LABEL = "never"

DEFAULT_SIZE = 40
DEFAULT_MIN_PER_STRATUM = 1
DEFAULT_INSTANCES = os.path.join(BENCH_DIR, "instances_plato.txt")

# How many names a refusal message spells out before summarising the rest.
MAX_LISTED = 20


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    """Render a boundary the way it was typed: `1`, not `1.0`; `0.5` stays."""
    return str(int(value)) if value == int(value) else f"{value:g}"


def stratum_labels(boundaries: tuple[float, ...]) -> list[str]:
    """Labels for `len(boundaries) + 1` time strata plus the never bucket.

    Derived from the boundaries themselves rather than a fixed name table, so
    a custom `--boundaries` can never end up with a label that describes a
    different interval than the one it bins.
    """
    labels = [f"<{_num(boundaries[0])}s"]
    labels += [f"{_num(lo)}-{_num(hi)}s" for lo, hi in pairwise(boundaries)]
    labels.append(f">={_num(boundaries[-1])}s")
    labels.append(NEVER_LABEL)
    return labels


def assign_stratum(seconds: float | None, boundaries: tuple[float, ...]) -> str:
    """Bin one aggregated time-to-first-feasible.

    `None` and any non-finite value mean the run never became feasible.
    Intervals are half-open on the right, so a boundary value lands in the
    stratum above it: at the default split a solution at exactly 1.000s is
    `1-10s`, not `<1s`.
    """
    labels = stratum_labels(boundaries)
    if seconds is None or not math.isfinite(seconds):
        return NEVER_LABEL
    for i, bound in enumerate(boundaries):
        if seconds < bound:
            return labels[i]
    return labels[len(boundaries)]


def parse_boundaries(text: str) -> tuple[float, ...]:
    """Parse and validate a `--boundaries` argument."""
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"not a number: {part!r}") from exc
    if not values:
        raise ValueError("at least one boundary is required")
    # Before the comparisons below: `inf` passes every one of them and then
    # blows up in `_num` with an unhandled OverflowError, and `nan` fails
    # each comparison silently.
    if any(not math.isfinite(v) for v in values):
        raise ValueError("boundaries must be finite")
    if any(v <= 0 for v in values):
        raise ValueError("boundaries must be positive")
    if any(hi <= lo for lo, hi in pairwise(values)):
        raise ValueError("boundaries must be strictly increasing")
    return tuple(values)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def allocate(
    counts: list[int], size: int, min_per_stratum: int = DEFAULT_MIN_PER_STRATUM
) -> list[int]:
    """Split `size` draws across strata of the given populations.

    Largest remainder: every stratum takes the floor of its proportional
    share and the leftover seats go to the largest fractional remainders,
    ties broken by stratum order.  The result is a function of the inputs
    alone — no rounding rule that depends on iteration order, and the total
    is exactly `size` rather than `size` give or take the rounding.

    One deviation from strict proportionality, recorded in the emitted
    header: `min_per_stratum` is reserved off the top for each non-empty
    stratum.  The point of stratifying is to span the spectrum, and a stratum
    holding 3 of 233 instances rounds to zero at every subset size below 39 —
    which drops exactly the stratum the stratification existed to keep.

    No stratum can come out with more draws than it holds.  That bound is a
    property of the arithmetic rather than a clamp applied afterwards: the
    proof sits beside the code, and the sweep in `test_make_tuning_set.py`
    checks it exhaustively over small inputs instead of the implementation
    carrying a cap that can never bind.

    Raises ValueError when `size` cannot be met at all: more draws than
    instances, or too few draws to give every non-empty stratum its minimum.
    """
    if size < 0:
        raise ValueError("size must be non-negative")
    if min_per_stratum < 0:
        raise ValueError("min_per_stratum must be non-negative")
    total = sum(counts)
    if size > total:
        raise ValueError(f"requested {size} instances but the tree holds {total}")

    base = [min(min_per_stratum, n) for n in counts]
    reserved = sum(base)
    if reserved > size:
        nonempty = sum(1 for n in counts if n > 0)
        raise ValueError(
            f"size {size} is below the {reserved} draws needed to give each of "
            f"the {nonempty} non-empty strata a minimum of {min_per_stratum}"
        )

    rest = size - reserved
    room = [n - b for n, b in zip(counts, base)]
    room_total = sum(room)
    if rest == 0 or room_total == 0:
        return base

    # Integer arithmetic, not floats.  The shares of `room=[1, 1, 10]` with
    # `rest=4` are 0.3333333333333333, 0.3333333333333333, 3.3333333333333335,
    # whose fractional parts are a mathematical three-way tie (4/12 each) but
    # sort as 2, 0, 1 in binary floating point — so the seat goes to the third
    # stratum where the stated rule ("ties by stratum order") gives it to the
    # first.  That is a real regression this caught, not a hypothetical.
    # `divmod` keeps the quotient and the remainder exact.
    shares = [divmod(rest * r, room_total) for r in room]
    extra = [q for q, _ in shares]
    leftover = rest - sum(extra)
    order = sorted(range(len(counts)), key=lambda i: (-shares[i][1], i))
    # One pass over the head of `order` places every leftover seat, and none
    # of them can overrun a stratum.  `sum(rem_i) == leftover * room_total`
    # with every `rem_i <= room_total - 1`, so strictly more than `leftover`
    # strata have `rem_i > 0`; the first `leftover` entries of `order` are
    # therefore all positive-remainder strata, and `rem_i > 0` means
    # `q_i < rest * r_i / room_total <= r_i`, i.e. room for one more.
    for i in order[:leftover]:
        extra[i] += 1
    return [b + e for b, e in zip(base, extra)]


# ---------------------------------------------------------------------------
# Reading a results tree
# ---------------------------------------------------------------------------


def unusable_reason(result: SolveResult) -> str | None:
    """Why a parsed log cannot be stratified, or None when it can be.

    Both cases are logs that *look* never-feasible and are not, which is the
    one misclassification this script cannot afford: `never` is normally the
    smallest stratum, so `--min-per-stratum` reserves it a seat and whatever
    was misfiled into it is then over-weighted rather than diluted.

    * A truncated or zero-byte `.log` parses into a default `SolveResult`
      with no incumbents.  HiGHS prints its `Solving report` on every exit
      path it survives, so a log with neither a status line nor an incumbent
      never recorded a solve.
    * A log reporting a finite primal bound with no incumbent line *did*
      find a solution and cannot say when.  That is what a source code
      missing from `parse_highs_log._INCUMBENT_SOURCES` looks like — the
      case a HiGHS bump introduces, i.e. exactly when the tree is least
      trustworthy.  The parser already detects it and warns; mirroring its
      condition here turns that warning into a refusal instead of letting
      the instance be binned as never-feasible.
    """
    if not result.status and not result.incumbents:
        return "truncated log (no solving report)"
    if math.isfinite(result.primal_bound) and not result.incumbents:
        return (
            "primal bound but no incumbent line (source code missing from "
            "parse_highs_log._INCUMBENT_SOURCES?)"
        )
    return None


def observed_time(result: SolveResult) -> float:
    """Time to first feasible, with `inf` standing for never feasible."""
    t = result.time_to_first_feasible
    return float("inf") if t is None else t


def aggregate_time(values: list[float]) -> float:
    """Aggregate one instance's per-seed times into the value it is binned on.

    Median, taking the lower of the two middles on an even seed count.  The
    lower middle is what keeps a single unlucky seed from moving an instance
    that is usually feasible into the never bucket, while a majority of
    never-feasible seeds still lands there — and unlike the mean it survives
    the `inf` that a never-feasible seed contributes.
    """
    return statistics.median_low(sorted(values))


@dataclass
class TreeScan:
    """What a results tree says about the reference instance list."""

    config: str
    config_dir: str
    seeds: list[int]
    # instance -> aggregated time to first feasible; inf = never feasible.
    observations: dict[str, float]
    # Reference instances the tree cannot stratify at all, and ones it covers
    # for only some of the config's seeds.  Both map to a printable reason.
    missing: dict[str, str]
    incomplete: dict[str, str]
    # Instances in the tree that the reference list does not name.  Not an
    # error — but a large count usually means the wrong `--instances`.
    extra: list[str]


def discover_configs(results_dir: str) -> list[str]:
    """Config subdirectories of a results tree, in sorted order."""
    root = Path(results_dir)
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and looks_like_config_dir(str(child)):
            found.append(child.name)
    return found


def looks_like_config_dir(path: str) -> bool:
    """Whether `path` holds run logs, directly or under `seed<N>/`."""
    directory = Path(path)
    if not directory.is_dir():
        return False
    if any(directory.glob("*.log")):
        return True
    return any(sd.is_dir() and any(sd.glob("*.log")) for sd in directory.glob("seed*"))


def resolve_config(results_dir: str, explicit: str | None) -> tuple[str, str]:
    """Pick the config arm to stratify on, returning `(name, directory)`.

    An explicit `--config` wins.  Otherwise: a tree that *is* a config
    directory is used as itself, a tree with one config uses it, and a tree
    with several prefers the first vanilla-ish name — anything else is a
    refusal rather than a guess, since stratifying on a patched arm would
    measure the thing under test.
    """
    if explicit:
        config_dir = os.path.join(results_dir, explicit)
        if os.path.isdir(config_dir):
            return explicit, config_dir
        # `--config vanilla` against a tree that already *is* `.../vanilla`
        # names the same arm; resolving it to `.../vanilla/vanilla` would
        # reject a command that is merely redundant.
        if os.path.basename(os.path.normpath(results_dir)) == explicit:
            return explicit, results_dir
        raise ValueError(f"no such config directory: {config_dir}")

    # A directory that holds logs or `seed<N>/` directories *is* the config
    # arm.  Tested before the config scan, not after: the scan would
    # otherwise see this directory's own `seed0/` as a config named `seed0`.
    if looks_like_config_dir(results_dir):
        return os.path.basename(os.path.normpath(results_dir)), results_dir

    configs = discover_configs(results_dir)
    if not configs:
        raise ValueError(f"no run logs found under {results_dir}")
    if len(configs) == 1:
        return configs[0], os.path.join(results_dir, configs[0])
    for name in VANILLA_CONFIG_NAMES:
        if name in configs:
            return name, os.path.join(results_dir, name)
    raise ValueError(
        f"{results_dir} holds several configs ({', '.join(configs)}) and none is "
        "a recognised vanilla arm; name one with --config"
    )


def err_files_by_seed(config_dir: str) -> dict[int, set[str]]:
    """Instances parked as `<instance>.log.err`, per seed.

    `run_benchmark.py` writes a failed or misconfigured run beside the log
    under that name precisely so it is not mistaken for a result.  Reading it
    here turns "instance absent" into "instance failed", which is the
    difference between a mystery and a rerun.
    """
    found: dict[int, set[str]] = {}
    suffix = ".log.err"
    for dirpath, _dirs, files in os.walk(config_dir):
        base = os.path.basename(dirpath)
        seed = int(base.removeprefix("seed")) if base.startswith("seed") else 0
        for name in files:
            if name.endswith(suffix):
                found.setdefault(seed, set()).add(name[: -len(suffix)])
    return found


def _seed_list(seeds: list[int]) -> str:
    return ",".join(str(s) for s in seeds)


def _coverage_reason(
    absent: list[int], unusable: dict[int, str], errored: list[int]
) -> str:
    """One printable sentence per way the tree failed to cover an instance."""
    parts = []
    if absent:
        parts.append("no log for seed(s) " + _seed_list(absent))
    # Grouped by reason rather than one clause per seed: a campaign that lost
    # a machine loses the same way on every seed of the instance.
    for reason in sorted(set(unusable.values())):
        seeds = [s for s in sorted(unusable) if unusable[s] == reason]
        parts.append(f"{reason} for seed(s) " + _seed_list(seeds))
    if errored:
        parts.append("failed run parked as .log.err for seed(s) " + _seed_list(errored))
    return "; ".join(parts)


def scan_tree(
    results_dir: str,
    config: str,
    instances: list[str],
    config_dir: str | None = None,
) -> TreeScan:
    """Read every log of one config arm and stratify-ready the reference set."""
    if config_dir is None:
        _, config_dir = resolve_config(results_dir, config)
    loaded = load_results(results_dir, [config], config_dirs={config: config_dir})
    by_seed = loaded.get(config, {})
    seeds = sorted(by_seed)
    errored = err_files_by_seed(config_dir)

    observations: dict[str, float] = {}
    missing: dict[str, str] = {}
    incomplete: dict[str, str] = {}
    for instance in instances:
        times: dict[int, float] = {}
        absent: list[int] = []
        unusable: dict[int, str] = {}
        for seed in seeds:
            result = by_seed[seed].get(instance)
            if result is None:
                absent.append(seed)
                continue
            reason = unusable_reason(result)
            if reason is not None:
                unusable[seed] = reason
            else:
                times[seed] = observed_time(result)
        errs = sorted(s for s in seeds if instance in errored.get(s, set()))
        if not times:
            missing[instance] = _coverage_reason(absent, unusable, errs)
            continue
        if len(times) < len(seeds):
            incomplete[instance] = _coverage_reason(absent, unusable, errs)
        observations[instance] = aggregate_time(list(times.values()))

    reference = set(instances)
    seen: set[str] = set()
    for seed_results in by_seed.values():
        seen.update(seed_results)
    return TreeScan(
        config=config,
        config_dir=config_dir,
        seeds=seeds,
        observations=observations,
        missing=missing,
        incomplete=incomplete,
        extra=sorted(seen - reference),
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """A drawn subset plus everything needed to justify it in the header."""

    scan: TreeScan
    boundaries: tuple[float, ...]
    labels: list[str]
    members: dict[str, list[str]]
    allocation: dict[str, int]
    sample: dict[str, list[str]]
    size: int
    seed: int
    min_per_stratum: int
    reference_path: str
    reference_count: int
    allowed_incomplete: bool

    @property
    def instances(self) -> list[str]:
        """Every drawn instance, in emitted order (by stratum, then name)."""
        return [name for label in self.labels for name in self.sample[label]]


def sample_stratum(names: list[str], count: int, seed: int, label: str) -> list[str]:
    """Draw `count` names deterministically from one stratum.

    The RNG is seeded per stratum (`<seed>:<label>`) rather than once for the
    whole run, so one stratum's draw does not depend on what its neighbours
    drew.  It is *not* nested in `count`: `random.sample` draws a fresh subset
    for each k, so a different `--size` redraws every stratum whose allocation
    moved — the per-stratum seeding buys independence between strata, not
    stability across sizes.

    Both the candidate order and the emitted order are sorted.  That is what
    makes the draw a function of the tree's contents rather than of the
    reference list's order, which is how the instances reach `members` in the
    first place.  This sort and the one in `build_selection` are mutually
    redundant — removing either alone is unobservable because the other
    re-establishes the order, and removing both makes the draw follow the
    reference list.  `test_draw_ignores_reference_list_order` kills that pair;
    keep both rather than reasoning about which one is load-bearing today.
    """
    rng = random.Random(f"{seed}:{label}")
    return sorted(rng.sample(sorted(names), count))


def build_selection(
    scan: TreeScan,
    boundaries: tuple[float, ...],
    size: int,
    seed: int,
    min_per_stratum: int,
    reference_path: str,
    reference_count: int,
    allowed_incomplete: bool = False,
) -> Selection:
    """Bin the scanned instances, allocate the draws, and sample each stratum."""
    labels = stratum_labels(boundaries)
    members: dict[str, list[str]] = {label: [] for label in labels}
    for name in sorted(scan.observations):
        members[assign_stratum(scan.observations[name], boundaries)].append(name)

    counts = [len(members[label]) for label in labels]
    drawn = allocate(counts, size, min_per_stratum)
    allocation = dict(zip(labels, drawn))
    sample = {
        label: sample_stratum(members[label], allocation[label], seed, label)
        for label in labels
    }
    return Selection(
        scan=scan,
        boundaries=boundaries,
        labels=labels,
        members=members,
        allocation=allocation,
        sample=sample,
        size=size,
        seed=seed,
        min_per_stratum=min_per_stratum,
        reference_path=reference_path,
        reference_count=reference_count,
        allowed_incomplete=allowed_incomplete,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def distribution_rows(sel: Selection) -> list[str]:
    """The stratum table: full set beside sample, with the ideal share.

    `ideal` is the unrounded proportional draw.  Printing it next to the
    actual allocation is what makes over- and under-representation visible
    instead of implied — the minimum-per-stratum floor and the largest
    remainder both show up as a gap between the two columns.
    """
    total_full = sum(len(sel.members[label]) for label in sel.labels)
    total_sample = sum(len(sel.sample[label]) for label in sel.labels)
    rows = [f"{'stratum':<12}{'full':>6}{'full%':>8}{'sample':>8}{'sample%':>9}{'ideal':>8}"]
    for label in sel.labels:
        full = len(sel.members[label])
        drawn = len(sel.sample[label])
        full_pct = 100.0 * full / total_full if total_full else 0.0
        sample_pct = 100.0 * drawn / total_sample if total_sample else 0.0
        ideal = sel.size * full / total_full if total_full else 0.0
        rows.append(
            f"{label:<12}{full:>6}{full_pct:>7.1f}%{drawn:>8}{sample_pct:>8.1f}%"
            f"{ideal:>8.1f}"
        )
    rows.append(
        f"{'TOTAL':<12}{total_full:>6}{100.0:>7.1f}%{total_sample:>8}"
        f"{100.0:>8.1f}%{float(sel.size):>8.1f}"
    )
    return rows


def regeneration_command(sel: Selection, results_dir: str) -> str:
    """The exact command that reproduces this file, for the header."""
    parts = [
        "bench/make_tuning_set.py",
        results_dir,
        f"--config {sel.scan.config}",
        f"--instances {sel.reference_path}",
        f"--size {sel.size}",
        f"--seed {sel.seed}",
    ]
    if tuple(sel.boundaries) != DEFAULT_BOUNDARIES:
        parts.append("--boundaries " + ",".join(_num(b) for b in sel.boundaries))
    if sel.min_per_stratum != DEFAULT_MIN_PER_STRATUM:
        parts.append(f"--min-per-stratum {sel.min_per_stratum}")
    if sel.allowed_incomplete:
        parts.append("--allow-incomplete-seeds")
    return " ".join(parts)


def _fmt_time(seconds: float) -> str:
    return NEVER_LABEL if not math.isfinite(seconds) else f"{seconds:.2f}s"


def render_list(sel: Selection, results_dir: str) -> str:
    """Render the instance list, header and all.

    Deliberately carries no timestamp and no hostname: the acceptance
    criterion is that the same tree and seed produce a byte-identical file,
    and a generation date would break it on every rerun.
    """
    scan = sel.scan
    seeds = ", ".join(str(s) for s in scan.seeds) or "(none)"
    lines = [
        "# Tuning subset of the PLATO mipfeas set, stratified on vanilla",
        "# time-to-first-feasible (issue #103) — the axis the primal integral",
        "# responds to, unlike the optimality solve time that",
        "# bench/instances_small.txt was selected on.",
        "#",
        "# Generated by bench/make_tuning_set.py; do not hand-edit.  It is a",
        "# derived, reproducible input: the same tree and --seed regenerate it",
        "# byte for byte, which is why it carries no generation date.",
        "#",
        f"#   source_tree      {results_dir}",
        f"#   config           {scan.config}",
        f"#   config_seeds     {seeds}",
        f"#   reference_list   {sel.reference_path} ({sel.reference_count} instances)",
        f"#   sample_seed      {sel.seed}",
        f"#   sample_size      {sel.size}",
        "#   boundaries_s     " + ", ".join(_num(b) for b in sel.boundaries),
        f"#   min_per_stratum  {sel.min_per_stratum}",
        (
            "#   aggregation      median-low over the config's seeds;"
            " never-feasible = +inf"
        ),
        (
            "#   allocation       proportional, largest remainder (integer"
            " shares, ties by stratum order)"
        ),
    ]
    if sel.allowed_incomplete:
        lines.append(
            f"#   partial_seeds    allowed for {len(scan.incomplete)} instance(s)"
            " (--allow-incomplete-seeds)"
        )
    lines.append("#")
    lines += [f"#   {row}" for row in distribution_rows(sel)]
    lines += [
        "#",
        "# Regenerate with:",
        f"#   {regeneration_command(sel, results_dir)}",
        "",
    ]

    chosen = sel.instances
    width = max((len(name) for name in chosen), default=0) + 2
    for label in sel.labels:
        names = sel.sample[label]
        if not names:
            continue
        lines.append(f"# {label} ({len(names)} of {len(sel.members[label])})")
        for name in names:
            seconds = scan.observations[name]
            lines.append(f"{name:<{width}}# {_fmt_time(seconds)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Coverage refusal
# ---------------------------------------------------------------------------


def _listing(names: list[str], reasons: dict[str, str]) -> list[str]:
    shown = names[:MAX_LISTED]
    lines = [f"    {name}: {reasons[name]}" for name in shown]
    if len(names) > len(shown):
        lines.append(f"    ... and {len(names) - len(shown)} more")
    return lines


def coverage_errors(scan: TreeScan, allow_incomplete: bool) -> list[str]:
    """Refusal messages for a tree that does not cover the reference list.

    Sampling a partial tree is worse than not sampling: the instances a
    campaign failed to run are not a random subset of it — they are the ones
    that crashed, timed out at the harness level, or were never launched —
    so a subset drawn around them is biased in exactly the direction the
    stratification is trying to measure.
    """
    errors: list[str] = []
    if not scan.seeds:
        errors.append(f"ERROR: no seed directories or logs under {scan.config_dir}")
        return errors
    if scan.missing:
        names = sorted(scan.missing)
        errors.append(
            f"ERROR: {len(names)} reference instance(s) have no usable log in "
            f"config '{scan.config}':"
        )
        errors += _listing(names, scan.missing)
    if scan.incomplete and not allow_incomplete:
        names = sorted(scan.incomplete)
        errors.append(
            f"ERROR: {len(names)} reference instance(s) are covered for only some "
            f"of seeds {', '.join(str(s) for s in scan.seeds)}:"
        )
        errors += _listing(names, scan.incomplete)
        errors.append(
            "    (rerun those instances, or pass --allow-incomplete-seeds to "
            "aggregate over the seeds that are present)"
        )
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a stratified tuning subset from a vanilla PLATO results "
            "tree, stratifying on time to first feasible solution."
        ),
        epilog=(
            "The tree must cover every instance of the reference list for "
            "every seed of the chosen config; anything else is a refusal "
            "(exit 2), because the instances a campaign failed to run are not "
            "a random subset of it."
        ),
    )
    parser.add_argument(
        "results_dir",
        help="results tree written by run_benchmark.py, or one config directory",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "config arm to stratify on; default is the tree's only config, or "
            f"the first of {', '.join(VANILLA_CONFIG_NAMES)} that it holds"
        ),
    )
    parser.add_argument(
        "--instances",
        default=DEFAULT_INSTANCES,
        help="reference instance list the tree must cover (default: %(default)s)",
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE, help="subset size (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="sampling seed (default: %(default)s)"
    )
    parser.add_argument(
        "--boundaries",
        default=",".join(_num(b) for b in DEFAULT_BOUNDARIES),
        help=(
            "comma-separated split points in seconds; N boundaries give N+1 "
            "time strata plus the never-feasible bucket (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--min-per-stratum",
        type=int,
        default=DEFAULT_MIN_PER_STRATUM,
        help=(
            "draws reserved for every non-empty stratum before proportional "
            "allocation; 0 for strict proportionality (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output",
        default="-",
        help="write the list here; '-' is stdout (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-incomplete-seeds",
        action="store_true",
        help=(
            "aggregate instances covered for only some seeds instead of "
            "refusing; recorded in the emitted header"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: no such results tree: {args.results_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.instances):
        print(f"ERROR: no such instance list: {args.instances}", file=sys.stderr)
        return 1
    try:
        boundaries = parse_boundaries(args.boundaries)
        config, config_dir = resolve_config(args.results_dir, args.config)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    reference = load_instances(args.instances)
    if not reference:
        print(f"ERROR: {args.instances} names no instances", file=sys.stderr)
        return 1

    scan = scan_tree(args.results_dir, config, reference, config_dir=config_dir)
    errors = coverage_errors(scan, args.allow_incomplete_seeds)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2
    if scan.extra:
        print(
            f"Note: {len(scan.extra)} instance(s) in the tree are not in "
            f"{args.instances} and were ignored.",
            file=sys.stderr,
        )

    try:
        selection = build_selection(
            scan=scan,
            boundaries=boundaries,
            size=args.size,
            seed=args.seed,
            min_per_stratum=args.min_per_stratum,
            reference_path=args.instances,
            reference_count=len(reference),
            allowed_incomplete=bool(scan.incomplete) and args.allow_incomplete_seeds,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    text = render_list(selection, args.results_dir)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        try:
            with open(args.output, "w") as f:
                f.write(text)
        except OSError as exc:
            print(f"ERROR: cannot write {args.output}: {exc}", file=sys.stderr)
            return 1

    seed_list = ", ".join(str(s) for s in scan.seeds)
    report = [
        (
            f"Stratified {len(scan.observations)} instances from "
            f"{args.results_dir} (config {scan.config}, seeds {seed_list})."
        ),
        *distribution_rows(selection),
    ]
    if args.output != "-":
        report.append(f"Wrote {len(selection.instances)} instances to {args.output}.")
    print("\n".join(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
