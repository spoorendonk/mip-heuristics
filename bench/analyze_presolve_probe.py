#!/usr/bin/env python3
"""Derive the informative instance set and the effort trajectories from a
presolve-only probe results tree.

Issue #113.  The probe runs the full PLATO list once at a deliberately
generous configuration — all four presolve heuristics, effort 1.0, stall
gates disabled, `mip_heuristic_presolve_only` — and this script turns that
tree into the three things the tuning stage cannot proceed without:

1. **The informative set.**  Instances that produced at least one accepted
   solution during presolve.  An instance that produces nothing even at the
   generous configuration is a constant in every comparison the search makes,
   so excluding it removes no signal — but the *excluded* list is a result in
   its own right: it says how much of the PLATO set a presolve-only screen
   can reason about at all.

2. **The retained hard tier.**  Those excluded instances, kept and listed
   rather than discarded, to be scored on a different question — *did any
   configuration crack it* — reported separately so a breakthrough shows up
   without diluting the quality ranking.

3. **The effort trajectories.**  Per heuristic: productive effort (charged
   effort at the last accepted solution), stale effort (the rest), and the
   inter-acceptance effort-gap distribution normalised by `nnz`, with its
   quantiles.  The stall thresholds are per-nonzero integers, so a high
   quantile of that distribution is directly the value to set — see the
   `stall_p95` column and the proposal block under the table.

**The filter is a union over configurations, never one config's outcome.**
Which instances yield a solution depends on which configuration ran (at
effort 0 none do), so selecting by what the current defaults are good at
biases the set toward them — and for a feasibility campaign, cracking a
previously-unsolved instance is the headline capability.  `--configs`
therefore defaults to *every* config in the tree and the predicate is a union
over `config x seed`; naming one config is the narrowing special case, and
the report says so.

Trajectories are read from the `[HeurSol]` trace, one line per
`IncumbentSink::offer`, emitted at `log_dev_level=3`.  Gaps are taken **within
one `(name, dispatch, worker)` triple** and only then pooled: the probe runs
at the machine's normal worker count, so a dispatch's accepted offers
interleave every worker and differencing them across workers would measure
the interleaving rather than the heuristic.  `effort_at` is contractually
monotone within a triple; a violation is surfaced and the whole dispatch
dropped, never repaired by discarding the negative difference — clipping
biases the p90-p95 downward, which is the direction that sets too tight a
stall threshold and costs solutions.

Both emitted lists carry the same reproducibility contract as
`bench/make_tuning_set.py`, whose `--informative-instances` consumes them:
sorted, header recording the derivation and the regeneration command, and no
timestamp, so the same tree and the same arguments produce a byte-identical
file.

Exit codes:
  0  the tree was analysed
  1  bad arguments or an unreadable tree
  2  the tree does not cover the reference instance list
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import load_results
from make_tuning_set import (
    MAX_LISTED,
    discover_configs,
    err_files_by_seed,
    looks_like_config_dir,
    sample_stratum,
)
from parse_highs_log import SolveResult
from run_benchmark import load_instances

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INSTANCES = os.path.join(BENCH_DIR, "instances_plato.txt")

# The presolve chain, in `kChain` order.  `fpr_lp` is deliberately absent: it
# is the dive-time heuristic, it runs on the far side of the root LP that a
# presolve-only run never reaches, and an offer it makes is not evidence that
# a presolve screen can see the instance.
PRESOLVE_HEURISTICS: tuple[str, ...] = ("fj", "fpr", "local_mip", "scylla")

# Whether a heuristic's stall option is divided by the worker count on its way
# to the per-worker gate that the measured gaps are a distribution of.
#
#   fj        the option's scope is per worker; the runner multiplies it by N
#             and `make_budget` divides it back, so the worker sees `per_nnz`.
#   fpr       whole-dispatch scope; `make_budget` divides by N, so the worker
#   local_mip sees `per_nnz / N` and the option must be N times a worker gap.
#   scylla    whole-dispatch scope, but it takes the dispatch-level threshold
#             as its *worker* threshold too, because its per-worker counter is
#             already charged the PDLP cost divided by N.
#
# This is the mapping from a measured per-worker gap to an option value; it
# mirrors `run_sequential` / `make_budget`, so it has to move if they do.
STALL_SCALES_WITH_WORKERS: dict[str, bool] = {
    "fj": False,
    "fpr": True,
    "local_mip": True,
    "scylla": False,
}

# Quantiles of the inter-acceptance gap distribution.  p90-p95 is the natural
# stall-threshold setting and the tail beyond it is the sharpness, so both are
# in the default set.
DEFAULT_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)

# The quantile the suggested stall threshold is read off, independently of
# `--quantiles`, so the suggestion never disappears because someone asked for
# a different table.
STALL_QUANTILE = 0.95


def stall_option(name: str) -> str:
    """The HiGHS option whose value one heuristic's gap quantile sets."""
    return f"mip_heuristic_{name}_stall"


# ---------------------------------------------------------------------------
# The `[HeurSol]` adapter
#
# This is the *only* place that knows how a per-solution trace sample reaches
# us.  `bench/parse_highs_log.py` is another track's file and its sample type
# is not nameable from here, so the adapter prefers whatever attribute that
# parser exposes and falls back to the contract's own line format when the
# parser predates the trace.  Everything below consumes `HeurSolSample`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeurSolSample:
    """One `IncumbentSink::offer`, as the frozen #106 contract spells it.

        [HeurSol] name=<n> dispatch=<i> worker=<w> effort_at=<E> wall_ms=<X> \
    obj=<O> accepted=<0|1>

    `effort_at` is the charged effort of the *offering worker* at the moment
    of the offer, monotone non-decreasing within a `(name, dispatch, worker)`
    triple; `(name, dispatch)` identifies one dispatch within a solve; and
    `wall_ms` may be negative because the solver clock is not monotonic.

    `worker` is `None` only when the producing build predates the field.  It
    is not defaulted to 0: merging every worker into one series is exactly the
    interleaving artefact the field was added to remove, so a sample without
    it disables the trajectory pass for that run rather than quietly
    corrupting the gap distribution.
    """

    name: str
    dispatch: int
    worker: int | None
    effort_at: int
    wall_ms: float
    obj: float
    accepted: bool


# Attribute names `SolveResult` might expose the samples under.  First one
# present wins; an empty list counts as present, since a log below
# `log_dev_level=3` legitimately has no samples.
_HEURSOL_ATTRS: tuple[str, ...] = (
    "heursol_samples",
    "heur_sol_samples",
    "heuristic_solution_samples",
    "solution_samples",
    "incumbent_samples",
)

# Field aliases, so a sample type that spells `effort_at` as `effort` or `obj`
# as `objective` still adapts.  The keys are this module's field names.
_HEURSOL_FIELDS: dict[str, tuple[str, ...]] = {
    "name": ("name", "heuristic"),
    "dispatch": ("dispatch", "dispatch_index"),
    "worker": ("worker", "worker_index"),
    "effort_at": ("effort_at", "effort"),
    "wall_ms": ("wall_ms",),
    "obj": ("obj", "objective"),
    "accepted": ("accepted",),
}

_HEURSOL_PREFIX = "[HeurSol]"

# Keys a line must carry.  `worker` is deliberately not here: a build that
# predates the amended contract still yields a usable informative set, and
# the trajectory pass reports its absence rather than the loader refusing the
# whole tree.
_HEURSOL_REQUIRED = ("name", "dispatch", "effort_at", "wall_ms", "obj", "accepted")


class AdapterError(RuntimeError):
    """A `[HeurSol]` sample this module cannot read."""


def parser_supports_heursol() -> bool:
    """Whether `parse_highs_log.SolveResult` carries the trace itself."""
    probe = SolveResult()
    return any(getattr(probe, attr, None) is not None for attr in _HEURSOL_ATTRS)


def _to_float(text: str) -> float:
    """Parse an objective field, tolerating HiGHS's infinities."""
    text = text.strip()
    if text in ("inf", "+inf", "1e+999"):
        return float("inf")
    if text in ("-inf", "-1e+999"):
        return float("-inf")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _to_bool(value: object) -> bool:
    """Read an `accepted` field written either as a flag or as `0`/`1`."""
    if isinstance(value, str):
        return value.strip() not in ("0", "false", "False", "")
    return bool(value)


def _field_of(item: object, key: str, required: bool = True) -> object:
    for alias in _HEURSOL_FIELDS[key]:
        if hasattr(item, alias):
            return getattr(item, alias)
    if not required:
        return None
    raise AdapterError(
        f"a [HeurSol] sample of type {type(item).__name__} exposes none of "
        f"{', '.join(_HEURSOL_FIELDS[key])} for field {key!r}"
    )


def _coerce(item: object) -> HeurSolSample:
    worker = _field_of(item, "worker", required=False)
    return HeurSolSample(
        name=str(_field_of(item, "name")),
        dispatch=int(_field_of(item, "dispatch")),  # type: ignore[arg-type]
        worker=None if worker is None else int(worker),  # type: ignore[arg-type]
        effort_at=int(_field_of(item, "effort_at")),  # type: ignore[arg-type]
        wall_ms=float(_field_of(item, "wall_ms")),  # type: ignore[arg-type]
        obj=float(_field_of(item, "obj")),  # type: ignore[arg-type]
        accepted=_to_bool(_field_of(item, "accepted")),
    )


def parse_heursol_line(line: str) -> HeurSolSample | None:
    """Read one `[HeurSol]` line, or None when the line is not one.

    Parsed as a `key=value` dict rather than a positional regex, which the
    shared contract requires: a field added to the trace later must not break
    this adapter or the release archive's verifier.  Unknown keys are
    ignored, and field order does not matter.
    """
    stripped = line.strip()
    if not stripped.startswith(_HEURSOL_PREFIX):
        return None
    fields: dict[str, str] = {}
    for token in stripped[len(_HEURSOL_PREFIX) :].split():
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    absent = [key for key in _HEURSOL_REQUIRED if key not in fields]
    if absent:
        raise AdapterError(
            f"[HeurSol] line is missing {', '.join(absent)}: {stripped[:120]}"
        )
    try:
        return HeurSolSample(
            name=fields["name"],
            dispatch=int(fields["dispatch"]),
            worker=int(fields["worker"]) if "worker" in fields else None,
            effort_at=int(fields["effort_at"]),
            wall_ms=float(fields["wall_ms"]),
            obj=_to_float(fields["obj"]),
            accepted=_to_bool(fields["accepted"]),
        )
    except ValueError as exc:
        raise AdapterError(f"unreadable [HeurSol] line: {stripped[:120]}") from exc


def heursol_from_text(log_text: str) -> list[HeurSolSample]:
    """Scan raw log text for `[HeurSol]` lines.

    The fallback path, used only while `parse_highs_log.py` does not carry
    the sample type.  It is not a second log parser: it reads one line shape
    that the shared contract froze, and every other number in this module
    still comes out of `parse_highs_log`.
    """
    samples = []
    for line in log_text.splitlines():
        sample = parse_heursol_line(line)
        if sample is not None:
            samples.append(sample)
    return samples


def heursol_samples(
    result: SolveResult, log_text: str | None = None
) -> list[HeurSolSample]:
    """Every `[HeurSol]` sample of one run, whatever the parser exposes."""
    for attr in _HEURSOL_ATTRS:
        raw = getattr(result, attr, None)
        if raw is not None:
            return [_coerce(item) for item in raw]
    if log_text is None:
        return []
    return heursol_from_text(log_text)


# ---------------------------------------------------------------------------
# Loading the tree
# ---------------------------------------------------------------------------


@dataclass
class ProbeRun:
    """One `(config, seed, instance)` run of the probe."""

    config: str
    seed: int
    instance: str
    result: SolveResult
    heursols: list[HeurSolSample]


@dataclass
class ProbeTree:
    """Every run of the probe, plus how the tree failed to be one."""

    root: str
    configs: list[str]
    config_dirs: dict[str, str]
    seeds: dict[str, list[int]]
    # instance -> runs, in (config, seed) order.
    runs: dict[str, list[ProbeRun]]
    # Reference instances the tree cannot analyse, mapped to a printable
    # reason.  Any entry here is a refusal unless --allow-missing.
    missing: dict[str, str]
    heursol_source: str
    parse_warnings: int


def resolve_configs(results_dir: str, explicit: list[str] | None) -> dict[str, str]:
    """Map every participating config name to its directory.

    A tree that *is* a config directory counts as one config named after
    itself, which is what a probe launched into `results/probe/` looks like.
    Tested before the config scan for the same reason `make_tuning_set` does
    it: the scan would otherwise see this directory's own `seed0/` as a
    config.
    """
    if explicit:
        dirs = {}
        for name in explicit:
            path = os.path.join(results_dir, name)
            if os.path.isdir(path):
                dirs[name] = path
            elif os.path.basename(os.path.normpath(results_dir)) == name:
                dirs[name] = results_dir
            else:
                raise ValueError(f"no such config directory: {path}")
        return dirs
    if looks_like_config_dir(results_dir):
        name = os.path.basename(os.path.normpath(results_dir))
        return {name: results_dir}
    found = discover_configs(results_dir)
    if not found:
        raise ValueError(f"no run logs found under {results_dir}")
    return {name: os.path.join(results_dir, name) for name in found}


def _log_path(config_dir: str, seed: int, instance: str) -> str:
    return os.path.join(config_dir, f"seed{seed}", f"{instance}.log")


def unusable_reason(result: SolveResult) -> str | None:
    """Why a probe log cannot be classified at all, or None when it can.

    Only one shape qualifies: a log with neither a solving report nor an
    incumbent nor the runner's `TIMEOUT:` marker, i.e. a run that left no
    evidence it ever started.

    This is deliberately *narrower* than `make_tuning_set.unusable_reason`,
    and the two must not be merged.

    * A **killed** run is a legitimate probe result, not missing data.  The
      probe's per-run cap has to be a wall-clock kill because HiGHS checks
      its clock between work units and an instance that never returns from
      its own presolve never looks at it (`ns1760995` spends the entire
      600 s limit there).  `run_benchmark.py` keeps the partial log with a
      `TIMEOUT:` marker, `parse_highs_log` surfaces it as `killed`, and this
      script routes it to the hard tier as *unreached* — never as
      never-feasible and never as a refusal.
    * A **finite primal bound with no incumbent line** is the expected shape
      of a *successful* presolve-only run: the solve exits before the root LP
      and may never print a display-table row, so the Solving report is the
      only place the presolve-found solution appears.  In the vanilla
      full-solve tree that `make_tuning_set` reads, the same shape means a
      source code missing from `parse_highs_log._INCUMBENT_SOURCES` and is
      rightly a refusal there.
    """
    if not result.status and not result.incumbents and not result.killed:
        return "no solving report, no incumbent and no TIMEOUT marker"
    return None


def load_probe_tree(
    results_dir: str,
    config_dirs: dict[str, str],
    instances: list[str],
) -> ProbeTree:
    """Read every log of every participating config arm."""
    configs = list(config_dirs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_results(results_dir, configs, config_dirs=config_dirs)
    # `parse_highs_log` warns once per log that reports a primal bound with
    # no incumbent line — the normal shape of a successful presolve-only run,
    # so it would otherwise fire on most of the tree.  Tallied, not echoed.
    parse_warnings = len(caught)

    from_parser = parser_supports_heursol()
    seeds = {c: sorted(loaded.get(c, {})) for c in configs}
    errored = {c: err_files_by_seed(config_dirs[c]) for c in configs}

    runs: dict[str, list[ProbeRun]] = {}
    missing: dict[str, str] = {}
    for instance in instances:
        found: list[ProbeRun] = []
        problems: list[str] = []
        for config in configs:
            for seed in seeds[config]:
                where = f"{config}/seed{seed}"
                result = loaded[config][seed].get(instance)
                if result is None:
                    if instance in errored[config].get(seed, set()):
                        problems.append(f"failed run parked as .log.err in {where}")
                    else:
                        problems.append(f"no log in {where}")
                    continue
                reason = unusable_reason(result)
                if reason is not None:
                    problems.append(f"{reason} in {where}")
                    continue
                text = None
                if not from_parser:
                    try:
                        with open(_log_path(config_dirs[config], seed, instance)) as f:
                            text = f.read()
                    except OSError:
                        text = None
                found.append(
                    ProbeRun(
                        config=config,
                        seed=seed,
                        instance=instance,
                        result=result,
                        heursols=heursol_samples(result, text),
                    )
                )
        if problems:
            missing[instance] = "; ".join(problems[:MAX_LISTED])
        if found:
            runs[instance] = found

    return ProbeTree(
        root=results_dir,
        configs=configs,
        config_dirs=config_dirs,
        seeds=seeds,
        runs=runs,
        missing=missing,
        heursol_source="parse_highs_log" if from_parser else "log scan (fallback)",
        parse_warnings=parse_warnings,
    )


# ---------------------------------------------------------------------------
# The informative set
# ---------------------------------------------------------------------------

# How a run's verdict was reached, strongest first.
EVIDENCE_HEURSOL = "heursol"
EVIDENCE_HEUR = "heur"
EVIDENCE_BOUND = "bound"

REASON_NO_ACCEPTANCE = "no-acceptance"
REASON_UNREACHED = "unreached"


@dataclass(frozen=True)
class RunVerdict:
    """What one run says about whether a presolve screen can see the model."""

    config: str
    seed: int
    evidence: str
    informative: bool
    killed: bool
    # Accepted presolve offers, or None when the evidence tier cannot count
    # them (no `[HeurSol]` trace in the log).
    accepted: int | None


def classify_run(run: ProbeRun) -> RunVerdict:
    """Did this run produce an accepted solution during presolve?

    Three tiers of evidence, strongest first, because the filtering pass is
    prescribed to run without `--dev-log` (level 3 costs 1.1-4.4x the wall
    time, concentrated in exactly the window being measured):

    1. `[HeurSol]`: count the accepted offers of the presolve chain directly.
    2. `[Heur]`: its `found` flag is the same `IncumbentSink` accept counter,
       aggregated per dispatch.
    3. Neither: the Solving report.  A presolve-only run exits before the
       root LP, so *any* solution it reports came from the chain — an
       incumbent line if one was printed, and otherwise the finite primal
       bound, which is the only place a presolve-found solution appears when
       the exit path prints no display-table row.
    """
    result = run.result
    chain = [s for s in run.heursols if s.name in PRESOLVE_HEURISTICS]
    if chain:
        accepted = sum(1 for s in chain if s.accepted)
        return RunVerdict(
            config=run.config,
            seed=run.seed,
            evidence=EVIDENCE_HEURSOL,
            informative=accepted > 0,
            killed=result.killed,
            accepted=accepted,
        )
    presolve = [
        s
        for s in result.heuristic_samples
        if s.phase == "presolve" and s.name in PRESOLVE_HEURISTICS
    ]
    if presolve:
        return RunVerdict(
            config=run.config,
            seed=run.seed,
            evidence=EVIDENCE_HEUR,
            informative=any(s.found for s in presolve),
            killed=result.killed,
            accepted=None,
        )
    return RunVerdict(
        config=run.config,
        seed=run.seed,
        evidence=EVIDENCE_BOUND,
        informative=bool(result.incumbents) or math.isfinite(result.primal_bound),
        killed=result.killed,
        accepted=None,
    )


@dataclass
class InformativeScan:
    """The union-over-configurations verdict for every reference instance."""

    informative: list[str]
    excluded: list[str]
    verdicts: dict[str, list[RunVerdict]]
    reasons: dict[str, str]
    details: dict[str, str]
    evidence_counts: dict[str, int]

    @property
    def covered(self) -> int:
        return len(self.informative) + len(self.excluded)


def informative_set(runs: dict[str, list[ProbeRun]]) -> InformativeScan:
    """Split the analysed instances into informative and hard-tier.

    The predicate is a **union over every run in `runs`** — every config and
    every seed.  A single-config mapping is the special case, not the shape
    the function is built around: conditioning the instance set on one
    configuration's outcome biases it toward what that configuration is
    already good at, and a candidate that would crack a different instance
    then gets no credit for it.

    An excluded instance is *unreached* when no run showed an acceptance and
    every run was killed — the screen never got to look at the model, which
    is a different fact from "the heuristics ran and found nothing".  Any
    clean run that found nothing makes it *no-acceptance*, with the killed
    count annotated when the runs are mixed.
    """
    informative: list[str] = []
    excluded: list[str] = []
    verdicts: dict[str, list[RunVerdict]] = {}
    reasons: dict[str, str] = {}
    details: dict[str, str] = {}
    evidence_counts: dict[str, int] = defaultdict(int)

    for instance in sorted(runs):
        vs = [classify_run(run) for run in runs[instance]]
        verdicts[instance] = vs
        for v in vs:
            evidence_counts[v.evidence] += 1
        if any(v.informative for v in vs):
            informative.append(instance)
            continue
        excluded.append(instance)
        killed = sum(1 for v in vs if v.killed)
        if killed == len(vs):
            reasons[instance] = REASON_UNREACHED
            details[instance] = f"all {len(vs)} run(s) killed before the chain reported"
        else:
            reasons[instance] = REASON_NO_ACCEPTANCE
            details[instance] = (
                f"{killed} of {len(vs)} run(s) killed"
                if killed
                else f"{len(vs)} run(s), none produced a solution"
            )
    return InformativeScan(
        informative=informative,
        excluded=excluded,
        verdicts=verdicts,
        reasons=reasons,
        details=details,
        evidence_counts=dict(evidence_counts),
    )


# ---------------------------------------------------------------------------
# Effort trajectories
# ---------------------------------------------------------------------------


def quantile(values: list[float], p: float) -> float:
    """Linear-interpolated quantile of an already-sorted list.

    Type 7 (the `numpy.percentile` default), written out because nothing else
    in `bench/` pulls numpy in and a quantile that moves with a dependency
    version is not a recorded input.
    """
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    h = (len(values) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return values[lo]
    return values[lo] + (h - lo) * (values[hi] - values[lo])


@dataclass(frozen=True)
class WorkerSeries:
    """One worker's accepted offers within one dispatch, in emission order."""

    worker: int
    accepted_efforts: tuple[int, ...]

    @property
    def gaps(self) -> list[int]:
        """Completed inter-acceptance gaps of this worker.

        The first gap runs from the start of the dispatch, because that is
        what the staleness gate measures: a worker's effort-since-improvement
        starts at zero when the dispatch does.  The interval after the last
        acceptance is *not* here — nothing ended it, so pooling it with the
        completed gaps would report a censored observation as a finished one.
        It is part of the dispatch's `stale`, reported in its own column.
        """
        out: list[int] = []
        previous = 0
        for effort in self.accepted_efforts:
            out.append(effort - previous)
            previous = effort
        return out


@dataclass(frozen=True)
class DispatchTrace:
    """One heuristic's one dispatch: where its accepted solutions landed."""

    instance: str
    config: str
    seed: int
    name: str
    dispatch: int
    nnz: int
    total_effort: int
    workers: int | None
    series: tuple[WorkerSeries, ...]

    @property
    def productive(self) -> int:
        """Charged effort at each worker's last accepted solution, summed.

        The dispatch-level definition the contract states: `effort_at` is a
        per-worker counter, so the dispatch's productive spend is the sum
        over workers, not the maximum.
        """
        return sum(s.accepted_efforts[-1] for s in self.series if s.accepted_efforts)

    @property
    def stale(self) -> int:
        """Dispatch effort spent after the last acceptance — censored."""
        return max(self.total_effort - self.productive, 0)

    @property
    def accepts(self) -> int:
        return sum(len(s.accepted_efforts) for s in self.series)

    @property
    def gaps(self) -> list[int]:
        return [gap for s in self.series for gap in s.gaps]


def _monotone(values: list[int]) -> bool:
    return all(b >= a for a, b in pairwise(values))


def dispatch_traces(
    run: ProbeRun, single_worker_only: bool = False
) -> tuple[list[DispatchTrace], list[str]]:
    """Every usable dispatch of one run, plus why any were dropped.

    `[Heur]` is the authority on how many dispatches a heuristic had and what
    each one spent, since a dispatch that made no offer at all leaves no
    `[HeurSol]` line.  **`dispatch` is assumed to be a 0-based per-name
    counter**, so it indexes that heuristic's `[Heur]` samples in emission
    order; an id outside that range drops the heuristic from this run with a
    diagnostic rather than being silently re-interpreted.

    Three things drop a dispatch outright, all reported rather than repaired:
    a `(dispatch, worker)` series whose `effort_at` is not monotone (the
    contract guarantees it is, so a violation is a data error and clipping it
    would bias the quantiles downward); a `[HeurSol]` sample with no `worker`
    field (a build predating the amended contract, where every worker would
    otherwise merge into one bogus series); and a `[Heur]` sample reporting
    `found=1` whose dispatch shows no accepted offer, which means the trace
    and the ledger disagree.
    """
    diagnostics: list[str] = []
    where = f"{run.instance} [{run.config}/seed{run.seed}]"
    result = run.result

    if result.num_nonzeros is None:
        return [], [f"{where}: no model header, so nnz is unknown"]
    if single_worker_only and result.thread_count != 1:
        observed = (
            "unknown" if result.thread_count is None else str(result.thread_count)
        )
        return [], [f"{where}: thread count {observed}, not 1"]

    chain_offers = [s for s in run.heursols if s.name in PRESOLVE_HEURISTICS]
    if any(s.worker is None for s in chain_offers):
        return [], [f"{where}: [HeurSol] lines carry no worker= field"]

    by_name: dict[str, list] = defaultdict(list)
    for sample in result.heuristic_samples:
        if sample.phase == "presolve" and sample.name in PRESOLVE_HEURISTICS:
            by_name[sample.name].append(sample)

    offers: dict[str, list[HeurSolSample]] = defaultdict(list)
    for sample in chain_offers:
        offers[sample.name].append(sample)

    traces: list[DispatchTrace] = []
    for name in PRESOLVE_HEURISTICS:
        totals = by_name.get(name, [])
        mine = offers.get(name, [])
        if not totals:
            if mine:
                diagnostics.append(
                    f"{where}: {len(mine)} [HeurSol] line(s) for {name} with no "
                    "[Heur] dispatch to charge them against"
                )
            continue
        if any(s.dispatch < 0 or s.dispatch >= len(totals) for s in mine):
            diagnostics.append(
                f"{where}: {name} dispatch id outside 0..{len(totals) - 1}; the "
                "0-based per-name counter assumption does not hold"
            )
            continue
        # (dispatch, worker) -> every offer, and the accepted subset, both in
        # emission order.  Monotonicity is checked over *every* offer, which
        # is what the contract guarantees and the wider net.
        seen: dict[tuple[int, int], list[int]] = defaultdict(list)
        taken: dict[tuple[int, int], list[int]] = defaultdict(list)
        for sample in mine:
            key = (sample.dispatch, sample.worker)  # type: ignore[arg-type]
            seen[key].append(sample.effort_at)
            if sample.accepted:
                taken[key].append(sample.effort_at)
        for index, heur in enumerate(totals):
            keys = sorted(k for k in seen if k[0] == index)
            broken = [k for k in keys if not _monotone(seen[k])]
            if broken:
                diagnostics.append(
                    f"{where}: {name} dispatch {index} has non-monotone effort_at "
                    f"for worker(s) {', '.join(str(k[1]) for k in broken)}; the "
                    "dispatch is dropped rather than clipped"
                )
                continue
            series = tuple(
                WorkerSeries(worker=k[1], accepted_efforts=tuple(taken[k]))
                for k in keys
                if taken[k]
            )
            if heur.found and not series:
                diagnostics.append(
                    f"{where}: {name} dispatch {index} reports found=1 with no "
                    "accepted [HeurSol] offer"
                )
                continue
            traces.append(
                DispatchTrace(
                    instance=run.instance,
                    config=run.config,
                    seed=run.seed,
                    name=name,
                    dispatch=index,
                    nnz=result.num_nonzeros,
                    total_effort=heur.effort,
                    workers=result.thread_count,
                    series=series,
                )
            )
    return traces, diagnostics


@dataclass
class HeuristicTrajectory:
    """One heuristic's pooled trajectory over the whole probe."""

    name: str
    dispatches: int = 0
    accepts: int = 0
    productive_effort: int = 0
    total_effort: int = 0
    # Sorted, in effort units per matrix nonzero — the units the stall
    # options take.  Gaps are per `(dispatch, worker)` series; tails are
    # per dispatch, hence a sum over that dispatch's workers.
    gaps_per_nnz: list[float] = field(default_factory=list)
    tails_per_nnz: list[float] = field(default_factory=list)

    @property
    def stale_effort(self) -> int:
        return self.total_effort - self.productive_effort

    @property
    def stale_fraction(self) -> float:
        if self.total_effort <= 0:
            return float("nan")
        return self.stale_effort / self.total_effort

    def suggested_stall(self, workers: int | None) -> int | None:
        """`mip_heuristic_<name>_stall` implied by the gap distribution.

        The p95 of the completed per-worker inter-acceptance gaps, rounded
        up: a threshold there truncates a worker before its next acceptance
        in one series in twenty, below it the gate cuts productive work, and
        above it the tail is what is being paid for.

        Scaled by the worker count for the two heuristics whose option is
        divided by N on its way to the per-worker gate — see
        `STALL_SCALES_WITH_WORKERS`.  A suggestion is therefore only valid at
        the worker count it was measured at, which the report prints.
        """
        if not self.gaps_per_nnz:
            return None
        value = quantile(self.gaps_per_nnz, STALL_QUANTILE)
        if STALL_SCALES_WITH_WORKERS[self.name] and workers:
            value *= workers
        return math.ceil(value)


def summarise_traces(traces: list[DispatchTrace]) -> dict[str, HeuristicTrajectory]:
    """Pool per-dispatch traces into one trajectory per heuristic."""
    out = {name: HeuristicTrajectory(name=name) for name in PRESOLVE_HEURISTICS}
    for trace in traces:
        t = out[trace.name]
        t.dispatches += 1
        t.accepts += trace.accepts
        t.productive_effort += trace.productive
        t.total_effort += trace.total_effort
        t.gaps_per_nnz.extend(gap / trace.nnz for gap in trace.gaps)
        t.tails_per_nnz.append(trace.stale / trace.nnz)
    for t in out.values():
        t.gaps_per_nnz.sort()
        t.tails_per_nnz.sort()
    return out


def worker_counts(traces: list[DispatchTrace]) -> list[int]:
    """The distinct observed worker counts of the traced dispatches, sorted.

    More than one means the tree mixes machines, which the report says
    outright: a stall suggestion is only valid at the worker count it was
    measured at, so a mixed tree's single summary number is a fiction.
    """
    return sorted({t.workers for t in traces if t.workers is not None})


def observed_workers(traces: list[DispatchTrace]) -> int | None:
    """The worker count the trajectories were measured at, median-low."""
    counts = [t.workers for t in traces if t.workers is not None]
    if not counts:
        return None
    return statistics.median_low(sorted(counts))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def parse_quantiles(text: str) -> tuple[float, ...]:
    """Parse and validate a `--quantiles` argument."""
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"not a number: {part!r}") from exc
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"quantiles must lie in (0, 1]: {part}")
        values.append(value)
    if not values:
        raise ValueError("at least one quantile is required")
    if any(b <= a for a, b in pairwise(values)):
        raise ValueError("quantiles must be strictly increasing")
    return tuple(values)


def _q_label(p: float) -> str:
    return f"p{100 * p:g}"


def _effort(value: float) -> str:
    return f"{value:.3g}"


def _cell(value: float, width: int) -> str:
    return f"{'-' if math.isnan(value) else f'{value:.1f}':>{width}}"


def trajectory_rows(
    trajectories: dict[str, HeuristicTrajectory],
    quantiles: tuple[float, ...],
    workers: int | None,
) -> list[str]:
    """The per-heuristic trajectory table.

    Effort columns are raw charged-effort units and are **not comparable
    across heuristics** — each counter counts what its own heuristic knows
    how to count.  Gap columns are per matrix nonzero per worker series; the
    tail column is per matrix nonzero per dispatch, i.e. summed over that
    dispatch's workers.
    """
    head = (
        f"{'heur':<10}{'disp':>6}{'accept':>7}{'productive':>12}{'stale':>12}"
        f"{'stale%':>8}{'gaps':>6}"
    )
    head += "".join(f"{_q_label(p):>10}" for p in quantiles)
    head += f"{'tail_p90':>10}{'stall_p95':>11}"
    rows = [head]
    for name in PRESOLVE_HEURISTICS:
        t = trajectories[name]
        fraction = t.stale_fraction
        stale_pct = "-" if math.isnan(fraction) else f"{100 * fraction:.1f}"
        row = (
            f"{name:<10}{t.dispatches:>6}{t.accepts:>7}"
            f"{_effort(t.productive_effort):>12}{_effort(t.stale_effort):>12}"
            f"{stale_pct:>8}{len(t.gaps_per_nnz):>6}"
        )
        for p in quantiles:
            row += _cell(quantile(t.gaps_per_nnz, p), 10)
        row += _cell(quantile(t.tails_per_nnz, 0.9), 10)
        suggested = t.suggested_stall(workers)
        row += f"{'-' if suggested is None else suggested:>11}"
        rows.append(row)
    return rows


def stall_suggestions(
    trajectories: dict[str, HeuristicTrajectory], workers: int | None
) -> list[str]:
    """The proposed stall option values, spelled as options."""
    lines = []
    for name in PRESOLVE_HEURISTICS:
        t = trajectories[name]
        value = t.suggested_stall(workers)
        if value is None:
            lines.append(f"  {stall_option(name):<32} (no accepted solutions traced)")
            continue
        scaled = STALL_SCALES_WITH_WORKERS[name] and workers
        note = f"p95 x {workers} workers" if scaled else "p95, per-worker scope"
        lines.append(f"  {stall_option(name):<32} {value:<10} ({note})")
    return lines


def probe_command(args: argparse.Namespace) -> str:
    """The exact command that reproduces the emitted lists, for the header."""
    parts = ["bench/analyze_presolve_probe.py", args.results_dir]
    if args.configs:
        parts.append("--configs " + " ".join(args.configs))
    parts.append(f"--instances {args.instances}")
    if args.informative_output:
        parts.append(f"--informative-output {args.informative_output}")
    if args.hard_tier_output:
        parts.append(f"--hard-tier-output {args.hard_tier_output}")
    if args.hard_tier_size is not None:
        parts.append(f"--hard-tier-size {args.hard_tier_size}")
        parts.append(f"--hard-tier-seed {args.hard_tier_seed}")
    if args.single_worker_trajectories:
        parts.append("--single-worker-trajectories")
    if args.allow_missing:
        parts.append("--allow-missing")
    return " ".join(parts)


def _provenance(
    tree: ProbeTree, reference_path: str, reference_count: int
) -> list[str]:
    pairs = sum(len(tree.seeds[c]) for c in tree.configs)
    return [
        f"#   probe_tree       {tree.root}",
        f"#   configs          {', '.join(tree.configs)}",
        (
            f"#   union_over       {len(tree.configs)} config(s), "
            f"{pairs} config-seed pair(s)"
        ),
        f"#   reference_list   {reference_path} ({reference_count} instances)",
        f"#   trace_source     {tree.heursol_source}",
    ]


def render_informative_list(
    tree: ProbeTree,
    scan: InformativeScan,
    args: argparse.Namespace,
    reference_count: int,
) -> str:
    """The informative instance list, header and all.

    Carries no timestamp, for the same reason `bench/make_tuning_set.py`
    does not: it is a derived input whose bytes are pinned into that script's
    header, so the same tree and the same arguments must regenerate it byte
    for byte.
    """
    lines = [
        "# Informative subset of the PLATO mipfeas set: instances a",
        "# presolve-only screen can see at all (issue #113).",
        "#",
        "# Generated by bench/analyze_presolve_probe.py; do not hand-edit.  It",
        "# is a derived, reproducible input — the same tree and arguments",
        "# regenerate it byte for byte, which is why it carries no date.",
        "#",
        *_provenance(tree, args.instances, reference_count),
        (
            "#   rule             at least one accepted presolve solution in ANY"
            " run (union over configs)"
        ),
        f"#   informative      {len(scan.informative)} of {scan.covered} analysed",
        f"#   hard_tier        {len(scan.excluded)} excluded, kept and scored apart",
        "#",
        "# Regenerate with:",
        f"#   {probe_command(args)}",
        "",
    ]
    lines += list(scan.informative)
    return "\n".join(lines).rstrip("\n") + "\n"


def render_hard_tier_list(
    tree: ProbeTree,
    scan: InformativeScan,
    chosen: list[str],
    args: argparse.Namespace,
    reference_count: int,
) -> str:
    """The retained hard tier, with the rule it is scored under."""
    width = max((len(name) for name in chosen), default=0) + 2
    lines = [
        "# Retained hard tier of the PLATO mipfeas set (issue #113): instances",
        "# a presolve-only screen produced nothing on at the generous probe",
        "# configuration.  Kept, not discarded.",
        "#",
        "# Generated by bench/analyze_presolve_probe.py; do not hand-edit.",
        "#",
        *_provenance(tree, args.instances, reference_count),
        (
            "#   rule             no accepted presolve solution in ANY run"
            " (union over configs)"
        ),
        f"#   excluded         {len(scan.excluded)} of {scan.covered} analysed",
        f"#   retained         {len(chosen)}",
    ]
    if len(chosen) < len(scan.excluded):
        lines.append(f"#   sample_seed      {args.hard_tier_seed}")
    lines += [
        "#",
        "#   scoring          scored on a different question from the tuning",
        "#                    set: *did any configuration crack it*, i.e. the",
        "#                    count of instances a candidate found any solution",
        "#                    for.  Reported separately and never pooled into",
        "#                    the quality ranking, so it cannot dilute the",
        "#                    comparison but a breakthrough still shows up.",
        "#   unreached        the run was killed before the chain reported, so",
        "#                    the screen never looked at the model; distinct",
        "#                    from no-acceptance, where it looked and found",
        "#                    nothing.",
        "#",
        "# Regenerate with:",
        f"#   {probe_command(args)}",
        "",
    ]
    for name in chosen:
        lines.append(f"{name:<{width}}# {scan.reasons[name]}: {scan.details[name]}")
    return "\n".join(lines).rstrip("\n") + "\n"


def _header_notes(tree: ProbeTree, scan: InformativeScan) -> list[str]:
    notes = []
    if len(tree.configs) < 2:
        notes.append(
            "  NOTE: one config only — the informative filter is meant to be a "
            "union over configurations; a single-config filter conditions the "
            "instance set on that configuration's outcome."
        )
    if scan.evidence_counts.get(EVIDENCE_HEURSOL, 0) == 0:
        notes.append(
            "  NOTE: no [HeurSol] trace in this tree; informativeness was "
            "inferred without per-solution attribution.  Rerun with --dev-log "
            "for the trajectories."
        )
    if tree.parse_warnings:
        notes.append(
            f"  NOTE: {tree.parse_warnings} log(s) report a primal bound with no "
            "incumbent line — the normal shape of a presolve-only exit."
        )
    if tree.missing:
        notes.append(f"  NOTE: {len(tree.missing)} instance(s) not fully covered.")
    return notes


def render_report(
    tree: ProbeTree,
    scan: InformativeScan,
    hard_tier: list[str],
    traces: list[DispatchTrace],
    trajectories: dict[str, HeuristicTrajectory],
    diagnostics: list[str],
    traced_runs: int,
    workers: int | None,
    quantiles: tuple[float, ...],
    reference_count: int,
) -> str:
    """The human-facing report: counts, both listings, and the trajectories."""
    seeds = ", ".join(f"{c}:{len(tree.seeds[c])}" for c in tree.configs)
    lines = [
        f"Presolve probe: {tree.root}",
        f"  configs        {', '.join(tree.configs)} (seeds per config: {seeds})",
        f"  reference      {reference_count} instances",
        f"  analysed       {scan.covered} instances",
        f"  trace source   {tree.heursol_source}",
        "  evidence       "
        + ", ".join(
            f"{tier}={scan.evidence_counts.get(tier, 0)}"
            for tier in (EVIDENCE_HEURSOL, EVIDENCE_HEUR, EVIDENCE_BOUND)
        ),
    ]
    lines += _header_notes(tree, scan)

    pct = 100.0 * len(scan.informative) / scan.covered if scan.covered else 0.0
    lines += [
        "",
        f"Informative set: {len(scan.informative)} of {scan.covered} ({pct:.1f}%)",
        "",
        f"Excluded (hard tier): {len(scan.excluded)}",
    ]
    by_reason: dict[str, int] = defaultdict(int)
    for name in scan.excluded:
        by_reason[scan.reasons[name]] += 1
    for reason in (REASON_NO_ACCEPTANCE, REASON_UNREACHED):
        lines.append(f"  {reason:<16}{by_reason.get(reason, 0)}")
    width = max((len(n) for n in scan.excluded), default=0) + 2
    for name in scan.excluded:
        lines.append(f"  {name:<{width}}{scan.reasons[name]}: {scan.details[name]}")
    if len(hard_tier) < len(scan.excluded):
        lines.append(f"  retained tier: {len(hard_tier)} sampled from the above")

    worker_note = "unknown" if workers is None else str(workers)
    lines += [
        "",
        (
            f"Effort trajectories ({traced_runs} run(s) traced, "
            f"{len(diagnostics)} skipped, {worker_note} worker(s))"
        ),
        "  effort columns are raw charged units and are NOT comparable across",
        "  heuristics; gap columns are per matrix nonzero per (dispatch, worker)",
        "  series, and the tail column is per nonzero per dispatch, i.e. summed",
        "  over that dispatch's workers.",
    ]
    counts = worker_counts(traces)
    if len(counts) > 1:
        lines.append(
            "  NOTE: the traced dispatches ran at "
            + ", ".join(str(c) for c in counts)
            + " workers, so this tree mixes machines and the single summary "
            "worker count above is a fiction; the stall suggestions below are "
            "only valid at one worker count."
        )
    lines.append("")
    lines += ["  " + row for row in trajectory_rows(trajectories, quantiles, workers)]
    lines += [
        "",
        (
            "Proposed stall thresholds (p95 of the inter-acceptance gap per nnz, "
            f"at {worker_note} worker(s)):"
        ),
        *stall_suggestions(trajectories, workers),
    ]
    if diagnostics:
        lines += ["", "Trajectory diagnostics:"]
        lines += [f"  {d}" for d in diagnostics[:MAX_LISTED]]
        if len(diagnostics) > MAX_LISTED:
            lines.append(f"  ... and {len(diagnostics) - MAX_LISTED} more")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def coverage_errors(tree: ProbeTree) -> list[str]:
    """Refusal messages for a tree that does not cover the reference list."""
    if not any(tree.seeds.values()):
        return [f"ERROR: no seed directories or logs under {tree.root}"]
    if not tree.missing:
        return []
    names = sorted(tree.missing)
    errors = [
        (
            f"ERROR: {len(names)} reference instance(s) are not fully covered by "
            f"config(s) {', '.join(tree.configs)}:"
        )
    ]
    errors += [f"    {name}: {tree.missing[name]}" for name in names[:MAX_LISTED]]
    if len(names) > MAX_LISTED:
        errors.append(f"    ... and {len(names) - MAX_LISTED} more")
    errors.append(
        "    (rerun those instances, or pass --allow-missing to analyse the "
        "runs that are present)"
    )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the informative instance set, the retained hard tier and "
            "the per-heuristic effort trajectories from a presolve-only probe."
        ),
        epilog=(
            "The informative filter is a union over every config in the tree: "
            "selecting instances by what one configuration is good at biases "
            "the set toward it.  Inter-acceptance gaps are taken within one "
            "(name, dispatch, worker) triple and only then pooled."
        ),
    )
    parser.add_argument(
        "results_dir", help="probe results tree written by run_benchmark.py"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="configs to union over (default: every config in the tree)",
    )
    parser.add_argument(
        "--instances",
        default=DEFAULT_INSTANCES,
        help="reference instance list the probe covered (default: %(default)s)",
    )
    parser.add_argument(
        "--informative-output", default=None, help="write the informative list here"
    )
    parser.add_argument(
        "--hard-tier-output", default=None, help="write the retained hard tier here"
    )
    parser.add_argument(
        "--hard-tier-size",
        type=int,
        default=None,
        help="retain this many excluded instances (default: all of them)",
    )
    parser.add_argument(
        "--hard-tier-seed",
        type=int,
        default=0,
        help="sampling seed for --hard-tier-size (default: %(default)s)",
    )
    parser.add_argument(
        "--quantiles",
        default=",".join(f"{p:g}" for p in DEFAULT_QUANTILES),
        help="gap-distribution quantiles to report (default: %(default)s)",
    )
    parser.add_argument(
        "--report-output", default="-", help="write the report here; '-' is stdout"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "analyse a tree that does not cover the reference list instead of refusing"
        ),
    )
    parser.add_argument(
        "--single-worker-trajectories",
        action="store_true",
        help=(
            "trace only logs whose observed thread count is 1, the project's "
            "reproducible configuration; the default traces every log, since "
            "[HeurSol] carries a worker id"
        ),
    )
    return parser


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def collect_traces(
    tree: ProbeTree, single_worker_only: bool
) -> tuple[list[DispatchTrace], list[str], int]:
    """Trace every run of the tree, in a deterministic order."""
    traces: list[DispatchTrace] = []
    diagnostics: list[str] = []
    traced_runs = 0
    for instance in sorted(tree.runs):
        for run in tree.runs[instance]:
            got, notes = dispatch_traces(run, single_worker_only)
            if got:
                traced_runs += 1
            traces.extend(got)
            diagnostics.extend(notes)
    return traces, diagnostics, traced_runs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: no such results tree: {args.results_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.instances):
        print(f"ERROR: no such instance list: {args.instances}", file=sys.stderr)
        return 1
    try:
        quantiles = parse_quantiles(args.quantiles)
        config_dirs = resolve_configs(args.results_dir, args.configs)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    reference = load_instances(args.instances)
    if not reference:
        print(f"ERROR: {args.instances} names no instances", file=sys.stderr)
        return 1

    try:
        tree = load_probe_tree(args.results_dir, config_dirs, reference)
    except AdapterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    errors = coverage_errors(tree)
    if errors and not (args.allow_missing and any(tree.seeds.values())):
        print("\n".join(errors), file=sys.stderr)
        return 2
    if errors:
        print(errors[0], file=sys.stderr)

    scan = informative_set(tree.runs)
    traces, diagnostics, traced_runs = collect_traces(
        tree, args.single_worker_trajectories
    )
    trajectories = summarise_traces(traces)
    workers = observed_workers(traces)

    hard_tier = list(scan.excluded)
    if args.hard_tier_size is not None:
        if not 0 <= args.hard_tier_size <= len(scan.excluded):
            print(
                f"ERROR: --hard-tier-size {args.hard_tier_size} outside "
                f"0..{len(scan.excluded)} excluded instance(s)",
                file=sys.stderr,
            )
            return 1
        hard_tier = sample_stratum(
            scan.excluded, args.hard_tier_size, args.hard_tier_seed, "hard-tier"
        )

    try:
        if args.informative_output:
            _write(
                args.informative_output,
                render_informative_list(tree, scan, args, len(reference)),
            )
        if args.hard_tier_output:
            _write(
                args.hard_tier_output,
                render_hard_tier_list(tree, scan, hard_tier, args, len(reference)),
            )
        report = render_report(
            tree,
            scan,
            hard_tier,
            traces,
            trajectories,
            diagnostics,
            traced_runs,
            workers,
            quantiles,
            len(reference),
        )
        if args.report_output == "-":
            sys.stdout.write(report)
        else:
            _write(args.report_output, report)
    except OSError as exc:
        print(f"ERROR: cannot write output: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
