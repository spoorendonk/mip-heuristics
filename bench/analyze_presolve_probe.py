#!/usr/bin/env python3
"""Derive the informative instance set and the effort trajectories from a
presolve-only probe results tree.

Issue #113.  The probe runs the full PLATO list once at a deliberately
generous configuration — all four presolve heuristics, effort 1.0, stall
gates disabled, `mip_heuristic_presolve_only` — and this script turns that
tree into the three things the tuning stage cannot proceed without:

1. **The informative set.**  Instances the presolve *chain* produced at least
   one accepted solution on.  An instance that produces nothing even at the
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
   quantile of that distribution is directly the value to set.

**The filter is a union over configurations, never one config's outcome.**
Which instances yield a solution depends on which configuration ran (at
effort 0 none do), so selecting by what the current defaults are good at
biases the set toward them — and for a feasibility campaign, cracking a
previously-unsolved instance is the headline capability.  `--configs`
therefore defaults to *every* config in the tree and the predicate is a union
over `config x seed`; naming one config is the narrowing special case, and
the report says so.

**A solution is only evidence if the chain produced it.**  HiGHS runs its own
trivial heuristics inside `runSetup()`, before the chain, and a
presolve-only run reports what they found like any other solution — the
display sources `l`, `p`, `u`, `z`, `X`, `Y`.  An instance solved only by
Trivial-upper carries no signal about any candidate configuration, so it
belongs in the hard tier; `CHAIN_SOURCES` is the set of codes the patch
assigns to our four heuristics and is what the evidence tests against.

Trajectories come from the `[HeurSol]` trace, one line per
`IncumbentSink::offer`, emitted at `log_dev_level=3`.  Everything about how
those lines group into dispatches — the process-global `dispatch` id, the
`heur_index` that binds a dispatch to the `[Heur]` line closing it, and the
model's nonzero count — is `bench/parse_highs_log.py`'s to decide, and this
module consumes `SolveResult.dispatch_traces()` rather than re-deriving any
of it.

Exit codes:
  0  the tree was analysed
  1  bad arguments or an unreadable tree
  2  the tree does not cover the reference list, or is not a probe tree
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import load_results
from make_archive import PATCH_MARKER, read_options_file
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

# The display source codes `apply_patch.cmake` assigns to the chain: A=FPR,
# M=LocalMIP, G=Scylla, J=FJ.  `D` (fpr_lp) is excluded for the same reason
# `fpr_lp` is absent above.  Everything else on a display row — `l`/`p`/`u`/
# `z` (HiGHS's trivial heuristics), `X`/`Y`, `T`, `B` — is not ours.
CHAIN_SOURCES = frozenset("AMGJ")

# `IncumbentSink::offer` tags an offer made off any worker slot with this,
# for LocalMIP's cold-start publish on the dispatching thread.  Such an offer
# is a real solution but not a worker's improvement-free interval, so it
# counts for the informative set and never enters the gap distribution.
OFF_SLOT_WORKER = -1

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

# How many lines of a log are scanned for the patch marker.  It is line 3 of
# every patched run; a dev-log run is gigabytes, so this is a bounded prefix
# read and never a slurp.
MARKER_PREFIX_LINES = 200

# Option values HiGHS's own options-file loader accepts as true.
_TRUE_VALUES = frozenset({"true", "on", "1", "yes"})


def stall_option(name: str) -> str:
    """The HiGHS option whose value one heuristic's gap quantile sets."""
    return f"mip_heuristic_{name}_stall"


# ---------------------------------------------------------------------------
# The `[HeurSol]` adapter
#
# The only place that knows how a per-solution trace sample reaches us.
# `bench/parse_highs_log.py` is another track's file, so the adapter prefers
# what that parser exposes — `SolveResult.dispatch_traces()` first, then a
# raw sample list — and falls back to the contract's own line format when the
# parser predates the trace.  Everything below consumes `DispatchView`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeurSolSample:
    """One `IncumbentSink::offer`, as the #106 contract spells it.

        [HeurSol] name=<n> dispatch=<i> worker=<w> effort_at=<E> wall_ms=<X> \
    obj=<O> accepted=<0|1>

    `effort_at` is the charged effort of the *offering worker* at the moment
    of the offer, monotone non-decreasing within a `(name, dispatch, worker)`
    triple; `dispatch` is **process-global** and neither zero-based nor dense
    within a solve; `worker` is `-1` for an offer made off any slot; and
    `wall_ms` may be negative because the solver clock is not monotonic.

    `heur_index` is not a field of the line.  It is the position of the
    `[Heur]` sample that closed this dispatch, derived by the parser and
    mirrored by this module's text fallback, and is `None` when the log ended
    before that line (a killed run).
    """

    name: str
    dispatch: int
    worker: int
    effort_at: int
    wall_ms: float
    obj: float
    accepted: bool
    heur_index: int | None = None


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
    "heur_index": ("heur_index",),
}

_HEURSOL_PREFIX = "[HeurSol]"
_HEUR_PREFIX = "[Heur]"

_HEURSOL_REQUIRED = (
    "name",
    "dispatch",
    "worker",
    "effort_at",
    "wall_ms",
    "obj",
    "accepted",
)


class AdapterError(RuntimeError):
    """A `[HeurSol]` sample this module cannot read."""


def parser_supports_heursol() -> bool:
    """Whether `parse_highs_log.SolveResult` carries the trace itself."""
    probe = SolveResult()
    return any(getattr(probe, attr, None) is not None for attr in _HEURSOL_ATTRS)


def parser_groups_dispatches() -> bool:
    """Whether the parser groups the trace into dispatches for us.

    When it does, this module never derives a dispatch binding of its own:
    the `dispatch` id is process-global and the `[Heur]` line closing a
    dispatch is bound by position, both of which are the parser's to know.
    """
    return hasattr(SolveResult(), "dispatch_traces")


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
    index = _field_of(item, "heur_index", required=False)
    return HeurSolSample(
        name=str(_field_of(item, "name")),
        dispatch=int(_field_of(item, "dispatch")),  # type: ignore[arg-type]
        worker=int(_field_of(item, "worker")),  # type: ignore[arg-type]
        effort_at=int(_field_of(item, "effort_at")),  # type: ignore[arg-type]
        wall_ms=float(_field_of(item, "wall_ms")),  # type: ignore[arg-type]
        obj=float(_field_of(item, "obj")),  # type: ignore[arg-type]
        accepted=_to_bool(_field_of(item, "accepted")),
        heur_index=None if index is None else int(index),  # type: ignore[arg-type]
    )


def parse_heursol_line(line: str) -> HeurSolSample | None:
    """Read one `[HeurSol]` line, or None when the line is not one.

    Parsed as a `key=value` dict rather than a positional regex, which the
    shared contract requires: the line already gained a field once, and a
    positional pattern turns the next such addition into a silent parse
    failure in every archived log.  Unknown keys are ignored and field order
    does not matter.
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
            worker=int(fields["worker"]),
            effort_at=int(fields["effort_at"]),
            wall_ms=float(fields["wall_ms"]),
            obj=_to_float(fields["obj"]),
            accepted=_to_bool(fields["accepted"]),
        )
    except ValueError as exc:
        raise AdapterError(f"unreadable [HeurSol] line: {stripped[:120]}") from exc


def heursol_from_lines(lines) -> list[HeurSolSample]:
    """Scan log lines for `[HeurSol]`, binding each to its `[Heur]` line.

    The fallback path, used only while `parse_highs_log.py` does not carry
    the sample type.  It streams: a `--dev-log` probe log runs to gigabytes,
    so nothing here holds the file.

    `heur_index` is derived the way the parser derives it — a dispatch's
    `[HeurSol]` lines all precede the `[Heur]` line for the same name — by
    holding each name's unbound samples until that line arrives.  Samples
    still unbound at end of file keep `None`, which is what a killed run
    leaves behind.
    """
    samples: list[HeurSolSample] = []
    pending: dict[str, list[int]] = defaultdict(list)
    heur_seen = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(_HEURSOL_PREFIX):
            sample = parse_heursol_line(line)
            if sample is not None:
                pending[sample.name].append(len(samples))
                samples.append(sample)
            continue
        if stripped.startswith(_HEUR_PREFIX):
            name = ""
            for token in stripped[len(_HEUR_PREFIX) :].split():
                key, sep, value = token.partition("=")
                if sep and key == "name":
                    name = value
                    break
            for position in pending.pop(name, []):
                held = samples[position]
                samples[position] = HeurSolSample(
                    name=held.name,
                    dispatch=held.dispatch,
                    worker=held.worker,
                    effort_at=held.effort_at,
                    wall_ms=held.wall_ms,
                    obj=held.obj,
                    accepted=held.accepted,
                    heur_index=heur_seen,
                )
            heur_seen += 1
    return samples


def heursol_samples(result: SolveResult, log_path: str | None = None):
    """Every `[HeurSol]` sample of one run, whatever the parser exposes."""
    for attr in _HEURSOL_ATTRS:
        raw = getattr(result, attr, None)
        if raw is not None:
            return [_coerce(item) for item in raw]
    if log_path is None:
        return []
    with open(log_path, errors="replace") as handle:
        return heursol_from_lines(handle)


# ---------------------------------------------------------------------------
# Dispatches
# ---------------------------------------------------------------------------


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
        acceptance is *not* here — nothing ended it.  It is right-censored,
        and `HeuristicTrajectory` carries it as such.
        """
        out: list[int] = []
        previous = 0
        for effort in self.accepted_efforts:
            out.append(effort - previous)
            previous = effort
        return out


@dataclass(frozen=True)
class DispatchView:
    """One heuristic's one dispatch, reduced to what the calibration needs.

    Built from the parser's own grouping.  The one thing it does *not* take
    from the parser is the gap arithmetic: `DispatchTrace.acceptance_gaps()`
    pools every worker id including `-1`, and an off-slot publish is not a
    worker's improvement-free interval.
    """

    instance: str
    config: str
    seed: int
    name: str
    dispatch: int
    nnz: int | None
    total_effort: int | None
    workers: int | None
    series: tuple[WorkerSeries, ...]
    off_slot_accepts: int = 0

    @property
    def productive(self) -> int:
        """Charged effort at each worker's last accepted solution, summed.

        `effort_at` is a per-worker counter, so the dispatch's productive
        spend is the sum over workers, not the maximum.
        """
        return sum(s.accepted_efforts[-1] for s in self.series if s.accepted_efforts)

    @property
    def stale(self) -> int | None:
        """Dispatch effort spent after the last acceptance — right-censored."""
        if self.total_effort is None:
            return None
        return max(self.total_effort - self.productive, 0)

    @property
    def accepts(self) -> int:
        return sum(len(s.accepted_efforts) for s in self.series)

    @property
    def gaps(self) -> list[int]:
        return [gap for s in self.series for gap in s.gaps]


def _monotone(values: list[int]) -> bool:
    return all(b >= a for a, b in pairwise(values))


def _heur_nnz(sample: object, result: SolveResult) -> int | None:
    """The nonzero count for one `[Heur]` dispatch.

    Prefers the field on the line itself, which is the only source that is
    both present at `log_dev_level=3` and the *post-presolve* matrix the
    stall options are expressed against; the one-line model header is absent
    at that level and the block form reports the original matrix.
    """
    own = getattr(sample, "nnz", None)
    return own or result.num_nonzeros


def _barren_dispatches(
    result: SolveResult, claimed: set[int], has_trace: bool
) -> tuple[list[tuple[str, int, int | None, int | None]], list[str]]:
    """Presolve dispatches that made no offer at all, from `[Heur]` alone.

    A dispatch that never offered a solution emits no `[HeurSol]` line, so it
    appears in no grouping built from the trace — and it is precisely the
    dispatch a staleness gate exists to cut.  Leaving it out is what makes an
    events-only quantile too tight, so its whole spend is recovered here as
    one right-censored interval.  `found=0` is enough to know the whole
    dispatch was improvement-free, so this works even on a log with no trace
    at all.

    `found=1` on an unclaimed dispatch is only a disagreement when the log
    *has* a trace; when it has none — a run without `--dev-log`, or a binary
    predating the trace — the dispatch is simply unobservable, and saying so
    is the useful diagnostic.

    The returned dispatch id is `-1 - index`: negative, so it cannot collide
    with the process-global counter, and distinct per `[Heur]` line.  It is
    an identity, never an index into anything.
    """
    out = []
    diagnostics = []
    for index, sample in enumerate(getattr(result, "heuristic_samples", [])):
        if index in claimed:
            continue
        if sample.phase != "presolve" or sample.name not in PRESOLVE_HEURISTICS:
            continue
        if sample.found:
            diagnostics.append(
                f"{sample.name} produced a solution the trace does not show"
                + (
                    " ([Heur] found=1 with no accepted [HeurSol] offer)"
                    if has_trace
                    else " (this log carries no [HeurSol] lines at all)"
                )
            )
            continue
        out.append((sample.name, -1 - index, sample.effort, _heur_nnz(sample, result)))
    return out, diagnostics


def _grouped_dispatches(result: SolveResult, samples: list[HeurSolSample]):
    """`(name, dispatch, total_effort, nnz, samples)` for every dispatch.

    Delegates to `SolveResult.dispatch_traces()` when the parser has it: the
    `dispatch` id is process-global, the `[Heur]` line closing a dispatch is
    bound by position, and the nonzero count a `--dev-log` run reports is not
    the one-line model header, so all three are the parser's to know.

    The fallback exists only while that method does not, and mirrors it.
    """
    traces = getattr(result, "dispatch_traces", None)
    if callable(traces):
        return [
            (t.name, t.dispatch, t.total_effort, t.nnz, list(t.samples))
            for t in traces()
        ]
    heur = getattr(result, "heuristic_samples", [])
    grouped: dict[tuple[str, int], list[HeurSolSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.name, sample.dispatch), []).append(sample)
    out = []
    for (name, dispatch), members in grouped.items():
        index = members[0].heur_index
        closing = heur[index] if index is not None and index < len(heur) else None
        total = None if closing is None else closing.effort
        nnz = result.num_nonzeros if closing is None else _heur_nnz(closing, result)
        out.append((name, dispatch, total, nnz, members))
    return out


def dispatch_views(
    instance: str,
    config: str,
    seed: int,
    result: SolveResult,
    samples: list[HeurSolSample],
    single_worker_only: bool = False,
) -> tuple[list[DispatchView], list[str]]:
    """Every usable dispatch of one run, plus why any were dropped.

    Two things drop a dispatch, both reported rather than repaired: a
    `(dispatch, worker)` series whose `effort_at` is not monotone — the
    parser guarantees it is, so a violation is a data error and clipping it
    would bias the quantiles downward — and an unknown nonzero count, which
    is what a `--dev-log` log looks like until the `[Heur]` line carries
    `nnz` (the one-line model header is absent at that level, and the block
    form reports the *original* matrix rather than the post-presolve one the
    stall options are expressed against).
    """
    diagnostics: list[str] = []
    where = f"{instance} [{config}/seed{seed}]"
    workers = result.thread_count

    if single_worker_only and workers != 1:
        observed = "unknown" if workers is None else str(workers)
        return [], [f"{where}: thread count {observed}, not 1"]

    views: list[DispatchView] = []
    grouped = _grouped_dispatches(result, samples)
    claimed = {
        s.heur_index
        for _, _, _, _, members in grouped
        for s in members
        if getattr(s, "heur_index", None) is not None
    }
    barren, notes = _barren_dispatches(result, claimed, bool(samples))
    diagnostics += [f"{where}: {note}" for note in notes]
    for name, dispatch, total, nnz in barren:
        if nnz is None:
            diagnostics.append(
                f"{where}: {name} barren dispatch has no nonzero count "
                "(the [Heur] line carries none and no model header was printed)"
            )
            continue
        views.append(
            DispatchView(
                instance=instance,
                config=config,
                seed=seed,
                name=name,
                dispatch=dispatch,
                nnz=nnz,
                total_effort=total,
                workers=workers,
                series=(),
            )
        )
    for name, dispatch, total, nnz, members in grouped:
        if name not in PRESOLVE_HEURISTICS:
            continue
        if nnz is None:
            diagnostics.append(
                f"{where}: {name} dispatch {dispatch} has no nonzero count "
                "(the [Heur] line carries none and no model header was printed)"
            )
            continue
        seen: dict[int, list[int]] = defaultdict(list)
        taken: dict[int, list[int]] = defaultdict(list)
        off_slot = 0
        for sample in members:
            if sample.worker == OFF_SLOT_WORKER:
                off_slot += int(bool(sample.accepted))
                continue
            seen[sample.worker].append(sample.effort_at)
            if sample.accepted:
                taken[sample.worker].append(sample.effort_at)
        broken = sorted(w for w, values in seen.items() if not _monotone(values))
        if broken:
            diagnostics.append(
                f"{where}: {name} dispatch {dispatch} has non-monotone effort_at "
                f"for worker(s) {', '.join(str(w) for w in broken)}; the dispatch "
                "is dropped rather than clipped"
            )
            continue
        views.append(
            DispatchView(
                instance=instance,
                config=config,
                seed=seed,
                name=name,
                dispatch=dispatch,
                nnz=nnz,
                total_effort=total,
                workers=workers,
                series=tuple(
                    WorkerSeries(worker=w, accepted_efforts=tuple(taken[w]))
                    for w in sorted(taken)
                ),
                off_slot_accepts=off_slot,
            )
        )
    return views, diagnostics


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
    presolve_only: bool | None
    patched: bool


@dataclass
class ProbeTree:
    """Every run of the probe, plus how the tree failed to be one."""

    root: str
    configs: list[str]
    config_dirs: dict[str, str]
    seeds: dict[str, list[int]]
    runs: dict[str, list[ProbeRun]]
    missing: dict[str, str]
    heursol_source: str
    parse_warnings: int

    @property
    def all_runs(self) -> list[ProbeRun]:
        return [run for runs in self.runs.values() for run in runs]


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


def log_is_patched(path: str) -> bool:
    """Whether a log carries the `mip-heuristics patch active` marker.

    A bounded prefix read: the marker is the third line of every patched run
    and a `--dev-log` probe log runs to gigabytes.
    """
    try:
        with open(path, errors="replace") as handle:
            for number, line in enumerate(handle):
                if PATCH_MARKER in line:
                    return True
                if number >= MARKER_PREFIX_LINES:
                    break
    except OSError:
        return False
    return False


def run_is_presolve_only(opts_path: str) -> bool | None:
    """Whether a run's `.opts` set `mip_heuristic_presolve_only`.

    None when there is no `.opts` beside the log, which is a run whose
    configuration was not recorded rather than one known to be wrong.
    """
    if not os.path.isfile(opts_path):
        return None
    value = read_options_file(Path(opts_path)).get("mip_heuristic_presolve_only")
    return value is not None and value.strip().lower() in _TRUE_VALUES


def unusable_reason(result: SolveResult) -> str | None:
    """Why a probe log cannot be classified at all, or None when it can.

    Only one shape qualifies: a log with neither a solving report nor an
    incumbent nor the runner's `TIMEOUT:` marker, i.e. a run that left no
    evidence it ever started.

    This is deliberately *narrower* than `make_tuning_set.unusable_reason`,
    and the two must not be merged.  A **killed** run is a legitimate probe
    result, not missing data: the probe's per-run cap has to be a wall-clock
    kill because HiGHS checks its clock between work units and an instance
    that never returns from its own presolve never looks at it.
    `run_benchmark.py` keeps the partial log with a `TIMEOUT:` marker,
    `parse_highs_log` surfaces it as `killed`, and this script routes it to
    the hard tier as *unreached* — never as a refusal.
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
                path = _log_path(config_dirs[config], seed, instance)
                found.append(
                    ProbeRun(
                        config=config,
                        seed=seed,
                        instance=instance,
                        result=result,
                        heursols=heursol_samples(result, None if from_parser else path),
                        presolve_only=run_is_presolve_only(
                            path[: -len(".log")] + ".opts"
                        ),
                        patched=log_is_patched(path),
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
        heursol_source=("parse_highs_log" if from_parser else "log scan (fallback)"),
        parse_warnings=parse_warnings,
    )


# ---------------------------------------------------------------------------
# Is this a probe tree at all?
# ---------------------------------------------------------------------------


@dataclass
class ProbeCheck:
    """Whether the tree is what this script is written to read."""

    runs: int
    presolve_only: int
    unrecorded: int
    patched: int

    @property
    def problems(self) -> list[str]:
        out = []
        if self.runs and self.patched < self.runs:
            out.append(
                f"{self.runs - self.patched} of {self.runs} run(s) carry no "
                f"'{PATCH_MARKER}' marker, so they came from an unpatched binary "
                "and ran none of the presolve chain"
            )
        wrong = self.runs - self.presolve_only - self.unrecorded
        if wrong:
            out.append(
                f"{wrong} of {self.runs} run(s) did not set "
                "mip_heuristic_presolve_only, so they measure a full solve "
                "rather than what a presolve-only screen can see"
            )
        if self.unrecorded:
            out.append(
                f"{self.unrecorded} of {self.runs} run(s) have no .opts beside "
                "the log, so their configuration is unrecorded"
            )
        return out


def check_is_probe(tree: ProbeTree) -> ProbeCheck:
    """Audit the tree against the probe the analysis assumes it is reading.

    Without this the script happily reports on a **full-solve vanilla** tree:
    every run falls to the weakest evidence tier, the "informative" count
    becomes "vanilla found something inside its time limit", and the emitted
    list is then pinned by digest into a tuning-set header as though it meant
    something.  That is not a hypothetical — it is how this script was first
    validated.
    """
    runs = tree.all_runs
    return ProbeCheck(
        runs=len(runs),
        presolve_only=sum(1 for r in runs if r.presolve_only is True),
        unrecorded=sum(1 for r in runs if r.presolve_only is None),
        patched=sum(1 for r in runs if r.patched),
    )


# ---------------------------------------------------------------------------
# The informative set
# ---------------------------------------------------------------------------

EVIDENCE_HEURSOL = "heursol"
EVIDENCE_HEUR = "heur"
EVIDENCE_SOURCE = "source"

REASON_NO_ACCEPTANCE = "no-acceptance"
REASON_UNREACHED = "unreached"
REASON_TRIVIAL_ONLY = "trivial-only"


@dataclass(frozen=True)
class RunVerdict:
    """What one run says about whether a presolve screen can see the model."""

    config: str
    seed: int
    evidence: str
    informative: bool
    killed: bool
    # A solution HiGHS's own trivial heuristics found before the chain ran.
    trivial_only: bool
    accepted: int | None


def classify_run(run: ProbeRun) -> RunVerdict:
    """Did the presolve *chain* produce an accepted solution in this run?

    Three tiers of evidence, strongest first, because the filtering pass is
    prescribed to run without `--dev-log` (level 3 costs 1.1-4.4x the wall
    time, concentrated in exactly the window being measured):

    1. `[HeurSol]`: count the accepted offers of the presolve chain directly.
    2. `[Heur]`: its `found` flag is the same `IncumbentSink` accept counter,
       aggregated per dispatch.
    3. Neither: the display rows, filtered to `CHAIN_SOURCES`.  This is the
       *primary* path, and it is a source test rather than a "did the run
       report any solution" test — HiGHS's own trivial heuristics run inside
       `runSetup()`, before the chain, and a run whose only solution came
       from Trivial-upper (`u`) carries no signal about any candidate
       configuration.  `trivial_only` records that case so the hard tier can
       say which kind of nothing it found.
    """
    result = run.result
    chain = [s for s in run.heursols if s.name in PRESOLVE_HEURISTICS]
    trivial_only = bool(result.incumbents) and not any(
        inc.source in CHAIN_SOURCES for inc in result.incumbents
    )
    if chain:
        accepted = sum(1 for s in chain if s.accepted)
        return RunVerdict(
            config=run.config,
            seed=run.seed,
            evidence=EVIDENCE_HEURSOL,
            informative=accepted > 0,
            killed=result.killed,
            trivial_only=trivial_only and accepted == 0,
            accepted=accepted,
        )
    presolve = [
        s
        for s in result.heuristic_samples
        if s.phase == "presolve" and s.name in PRESOLVE_HEURISTICS
    ]
    if presolve:
        found = any(s.found for s in presolve)
        return RunVerdict(
            config=run.config,
            seed=run.seed,
            evidence=EVIDENCE_HEUR,
            informative=found,
            killed=result.killed,
            trivial_only=trivial_only and not found,
            accepted=None,
        )
    ours = any(inc.source in CHAIN_SOURCES for inc in result.incumbents)
    return RunVerdict(
        config=run.config,
        seed=run.seed,
        evidence=EVIDENCE_SOURCE,
        informative=ours,
        killed=result.killed,
        trivial_only=trivial_only,
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

    An excluded instance is *unreached* when every run was killed — the
    screen never got to look at the model; *trivial-only* when the only
    solutions anywhere came from HiGHS's own pre-chain heuristics; and
    *no-acceptance* otherwise.  All three are hard tier, and which one it is
    is the difference between "our heuristics failed" and "we never asked".
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
        trivial = sum(1 for v in vs if v.trivial_only)
        if killed == len(vs):
            reasons[instance] = REASON_UNREACHED
            details[instance] = f"all {len(vs)} run(s) killed before the chain reported"
        elif trivial:
            reasons[instance] = REASON_TRIVIAL_ONLY
            details[instance] = (
                f"{trivial} of {len(vs)} run(s) found a solution, none from the "
                "chain (HiGHS's own trivial heuristics)"
            )
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
class Observation:
    """One improvement-free interval, in effort units per nonzero.

    `event` means an acceptance ended it; otherwise the dispatch ended first
    and the interval is right-censored — a lower bound on the gap that would
    have been observed.  `weight` is the observation's share, and `instance`
    is what the weights are equalised over.
    """

    value: float
    instance: str
    event: bool
    weight: float = 1.0


def km_quantile(observations: list[Observation], p: float) -> float | None:
    """Kaplan-Meier quantile over right-censored, weighted observations.

    Why not the plain quantile of the completed gaps: a dispatch that stops
    producing contributes no completed gap at all, so an events-only p95 is
    computed *conditional on the heuristic eventually producing again* and
    systematically understates the threshold — precisely the direction that
    sets too tight a gate and costs solutions.  Treating each dispatch's
    post-last-acceptance spend as a right-censored observation puts those
    intervals back in, without pretending to know how long they would have
    run.

    Returns None when the estimate never reaches `1 - p`, i.e. when the
    censoring is heavy enough that the quantile is **not identifiable**.
    That is a real answer — "no constant bounds this" — and reporting it
    beats extrapolating one.
    """
    if not observations:
        return None
    ordered = sorted(observations, key=lambda o: o.value)
    total = sum(o.weight for o in ordered)
    survival = 1.0
    target = 1.0 - p
    index = 0
    while index < len(ordered):
        value = ordered[index].value
        events = 0.0
        leaving = 0.0
        while index < len(ordered) and ordered[index].value == value:
            leaving += ordered[index].weight
            if ordered[index].event:
                events += ordered[index].weight
            index += 1
        if events and total > 0:
            survival *= 1.0 - events / total
            if survival <= target:
                return value
        total -= leaving
    return None


@dataclass
class HeuristicTrajectory:
    """One heuristic's pooled trajectory over the whole probe."""

    name: str
    dispatches: int = 0
    accepts: int = 0
    off_slot_accepts: int = 0
    productive_effort: int = 0
    total_effort: int = 0
    unknown_total: int = 0
    # Every improvement-free interval, per nonzero: completed gaps as events
    # and post-last-acceptance spend as right-censored.
    observations: list[Observation] = field(default_factory=list)

    @property
    def stale_effort(self) -> int:
        return self.total_effort - self.productive_effort

    @property
    def stale_fraction(self) -> float:
        if self.total_effort <= 0:
            return float("nan")
        return self.stale_effort / self.total_effort

    @property
    def gaps_per_nnz(self) -> list[float]:
        """Completed gaps only, sorted — the events-only view."""
        return sorted(o.value for o in self.observations if o.event)

    @property
    def censored_per_nnz(self) -> list[float]:
        return sorted(o.value for o in self.observations if not o.event)

    def weighted(self) -> list[Observation]:
        """The observations with each instance carrying equal total weight.

        Flat pooling lets one easy instance with many short, high-acceptance
        dispatches dominate the quantile that sets a shipped default.  The
        gate fires per instance in the end, so each instance gets one vote.
        """
        by_instance: dict[str, int] = defaultdict(int)
        for o in self.observations:
            by_instance[o.instance] += 1
        return [
            Observation(
                value=o.value,
                instance=o.instance,
                event=o.event,
                weight=1.0 / by_instance[o.instance],
            )
            for o in self.observations
        ]

    def events_p95(self) -> float | None:
        """The events-only p95 — a lower bound on the threshold."""
        gaps = self.gaps_per_nnz
        return quantile(gaps, STALL_QUANTILE) if gaps else None

    def censored_p95(self) -> float | None:
        """The censoring-aware p95, or None when it is not identifiable."""
        return km_quantile(self.weighted(), STALL_QUANTILE)

    def stall_range(self, workers: int | None) -> tuple[int | None, int | None]:
        """The `mip_heuristic_<name>_stall` range #107 should search.

        Lower bound from the events-only p95, upper from the censoring-aware
        one; `None` upper means the censoring never reaches 5 %, i.e. no
        constant bounds this heuristic on this data.  Both are scaled by the
        worker count for the two options divided by N on the way to the
        per-worker gate, so both are valid only at the measured worker count.
        """
        scale = workers if (STALL_SCALES_WITH_WORKERS[self.name] and workers) else 1
        low = self.events_p95()
        high = self.censored_p95()
        return (
            None if low is None else math.ceil(low * scale),
            None if high is None else math.ceil(high * scale),
        )


def summarise_traces(views: list[DispatchView]) -> dict[str, HeuristicTrajectory]:
    """Pool per-dispatch views into one trajectory per heuristic.

    A dispatch's post-last-acceptance spend is split evenly across its
    workers before entering the distribution, because the measured gaps are
    per worker and `make_budget` makes the same evenness assumption when it
    divides the dispatch threshold by N.  A wholly barren dispatch therefore
    contributes N censored intervals of `total / N` each, which is what the
    gate would have been asked to cut.
    """
    out = {name: HeuristicTrajectory(name=name) for name in PRESOLVE_HEURISTICS}
    for view in views:
        t = out[view.name]
        t.dispatches += 1
        t.accepts += view.accepts
        t.off_slot_accepts += view.off_slot_accepts
        nnz = view.nnz or 1
        for gap in view.gaps:
            t.observations.append(Observation(gap / nnz, view.instance, True))
        stale = view.stale
        if stale is None:
            t.unknown_total += 1
            continue
        t.productive_effort += view.productive
        t.total_effort += view.total_effort or 0
        workers = view.workers if view.workers and view.workers > 0 else 1
        share = stale / (nnz * workers)
        for _ in range(workers):
            t.observations.append(Observation(share, view.instance, False))
    return out


def worker_counts(views: list[DispatchView]) -> list[int]:
    """The distinct observed worker counts of the traced dispatches, sorted.

    More than one means the tree mixes machines, which the report says
    outright: a stall suggestion is only valid at the worker count it was
    measured at, so a mixed tree's single summary number is a fiction.
    """
    return sorted({v.workers for v in views if v.workers is not None})


def observed_workers(views: list[DispatchView]) -> int | None:
    """The worker count the trajectories were measured at, median-low."""
    counts = [v.workers for v in views if v.workers is not None]
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
    how to count.  Gap columns are per matrix nonzero per worker series, and
    are the events-only view; `cens` counts the right-censored intervals that
    the `stall` range's upper end accounts for and these columns do not.
    """
    head = (
        f"{'heur':<10}{'disp':>6}{'accept':>7}{'productive':>12}{'stale':>12}"
        f"{'stale%':>8}{'gaps':>6}"
    )
    head += "".join(f"{_q_label(p):>10}" for p in quantiles)
    head += f"{'cens':>6}{'stall_lo':>10}{'stall_hi':>10}"
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
        gaps = t.gaps_per_nnz
        for p in quantiles:
            row += _cell(quantile(gaps, p) if gaps else float("nan"), 10)
        low, high = t.stall_range(workers)
        row += f"{len(t.censored_per_nnz):>6}"
        row += f"{'-' if low is None else low:>10}"
        # `unbnd` is an answer — "the censoring never reaches 5 %" — and must
        # not be shown for a heuristic nothing was traced for, where the
        # honest cell is empty.
        if not t.observations:
            row += f"{'-':>10}"
        else:
            row += f"{'unbnd' if high is None else high:>10}"
        rows.append(row)
    return rows


def stall_suggestions(
    trajectories: dict[str, HeuristicTrajectory], workers: int | None
) -> list[str]:
    """The proposed stall option ranges, spelled as options."""
    lines = []
    for name in PRESOLVE_HEURISTICS:
        t = trajectories[name]
        low, high = t.stall_range(workers)
        if not t.observations:
            lines.append(f"  {stall_option(name):<32} (nothing traced)")
            continue
        scaled = STALL_SCALES_WITH_WORKERS[name] and workers
        scope = f"x {workers} workers" if scaled else "per-worker scope"
        top = "unbounded (censoring never reaches 5%)" if high is None else str(high)
        lines.append(
            f"  {stall_option(name):<32} {'-' if low is None else low} .. {top}"
            f"  ({scope})"
        )
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
    if args.allow_non_probe:
        parts.append("--allow-non-probe")
    return " ".join(parts)


def _provenance(
    tree: ProbeTree, check: ProbeCheck, reference_path: str, reference_count: int
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
        (
            f"#   probe_runs       {check.runs} run(s), "
            f"{check.presolve_only} presolve-only, {check.patched} patched"
        ),
        f"#   trace_source     {tree.heursol_source}",
    ]


def render_informative_list(
    tree: ProbeTree,
    check: ProbeCheck,
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
        *_provenance(tree, check, args.instances, reference_count),
        (
            "#   rule             at least one accepted solution from the presolve"
            " chain in ANY"
        ),
        (
            "#                    run (union over configs).  HiGHS's own trivial"
            " heuristics run"
        ),
        ("#                    before the chain and their solutions do not count."),
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
    check: ProbeCheck,
    scan: InformativeScan,
    chosen: list[str],
    args: argparse.Namespace,
    reference_count: int,
) -> str:
    """The retained hard tier, with the rule it is scored under."""
    width = max((len(name) for name in chosen), default=0) + 2
    lines = [
        "# Retained hard tier of the PLATO mipfeas set (issue #113): instances",
        "# the presolve chain produced nothing on at the generous probe",
        "# configuration.  Kept, not discarded.",
        "#",
        "# Generated by bench/analyze_presolve_probe.py; do not hand-edit.",
        "#",
        *_provenance(tree, check, args.instances, reference_count),
        (
            "#   rule             no accepted chain solution in ANY run"
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
        "#   unreached        killed before the chain reported: the screen never",
        "#                    looked at the model.",
        "#   trivial-only     a solution was found, but by HiGHS's own pre-chain",
        "#                    heuristics (sources l/p/u/z/X/Y), so it says",
        "#                    nothing about any candidate configuration.",
        "#   no-acceptance    the chain ran and produced nothing.",
        "#",
        "# Regenerate with:",
        f"#   {probe_command(args)}",
        "",
    ]
    for name in chosen:
        lines.append(f"{name:<{width}}# {scan.reasons[name]}: {scan.details[name]}")
    return "\n".join(lines).rstrip("\n") + "\n"


def _header_notes(
    tree: ProbeTree, check: ProbeCheck, scan: InformativeScan
) -> list[str]:
    notes = []
    for problem in check.problems:
        notes.append(f"  WARNING: {problem}")
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
            "incumbent line, which parse_highs_log flags as a possibly missing "
            "source code; such a run cannot be attributed and is not counted "
            "informative."
        )
    if tree.missing:
        notes.append(f"  NOTE: {len(tree.missing)} instance(s) not fully covered.")
    return notes


def render_report(
    tree: ProbeTree,
    check: ProbeCheck,
    scan: InformativeScan,
    hard_tier: list[str],
    views: list[DispatchView],
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
        (
            f"  probe check    {check.presolve_only}/{check.runs} presolve-only, "
            f"{check.patched}/{check.runs} patched"
        ),
        f"  trace source   {tree.heursol_source}",
        "  evidence       "
        + ", ".join(
            f"{tier}={scan.evidence_counts.get(tier, 0)}"
            for tier in (EVIDENCE_HEURSOL, EVIDENCE_HEUR, EVIDENCE_SOURCE)
        ),
    ]
    lines += _header_notes(tree, check, scan)

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
    for reason in (REASON_NO_ACCEPTANCE, REASON_TRIVIAL_ONLY, REASON_UNREACHED):
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
            f"{len(diagnostics)} dispatch(es) skipped, {worker_note} worker(s))"
        ),
        "  effort columns are raw charged units and are NOT comparable across",
        "  heuristics; gap columns are per matrix nonzero per worker series.",
        "  stall_lo is the events-only p95 -- a lower bound, since a dispatch",
        "  that stopped producing contributes no completed gap.  stall_hi adds",
        "  those censored intervals back with Kaplan-Meier, per-instance",
        "  equally weighted; 'unbnd' means no constant bounds this heuristic.",
    ]
    counts = worker_counts(views)
    if len(counts) > 1:
        lines.append(
            "  NOTE: the traced dispatches ran at "
            + ", ".join(str(c) for c in counts)
            + " workers, so this tree mixes machines and the single summary "
            "worker count above is a fiction; the stall ranges below are only "
            "valid at one worker count."
        )
    off_slot = sum(t.off_slot_accepts for t in trajectories.values())
    if off_slot:
        lines.append(
            f"  NOTE: {off_slot} accepted offer(s) came from worker -1 (off any "
            "slot, e.g. LocalMIP's cold-start publish); they count for the "
            "informative set and are excluded from the gap distribution."
        )
    unknown = sum(t.unknown_total for t in trajectories.values())
    if unknown:
        lines.append(
            f"  NOTE: {unknown} dispatch(es) have no [Heur] line (a killed run), "
            "so they contribute gaps but no productive/stale split."
        )
    lines.append("")
    lines += ["  " + row for row in trajectory_rows(trajectories, quantiles, workers)]
    lines += [
        "",
        (f"Proposed stall ranges for #107 (per nnz, at {worker_note} worker(s)):"),
        *stall_suggestions(trajectories, workers),
    ]
    if diagnostics:
        lines += ["", "Trajectory diagnostics:"]
        counted: dict[str, int] = defaultdict(int)
        for note in diagnostics:
            counted[note.split(": ", 1)[-1].split(";")[0]] += 1
        for note, count in sorted(counted.items(), key=lambda kv: -kv[1])[:MAX_LISTED]:
            lines.append(f"  {count:>6}x  {note}")
        if len(counted) > MAX_LISTED:
            lines.append(f"  ... and {len(counted) - MAX_LISTED} more kinds")
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
            "The informative filter is a union over every config in the tree, "
            "and counts only solutions the presolve chain itself produced. "
            "The tree must be a presolve-only probe run by a patched binary; "
            "anything else is a refusal."
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
        "--allow-non-probe",
        action="store_true",
        help=(
            "analyse a tree whose runs are not presolve-only, or not patched, "
            "instead of refusing; the warning is recorded in both emitted lists"
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


def collect_views(
    tree: ProbeTree, single_worker_only: bool
) -> tuple[list[DispatchView], list[str], int]:
    """Trace every run of the tree, in a deterministic order."""
    views: list[DispatchView] = []
    diagnostics: list[str] = []
    traced_runs = 0
    for instance in sorted(tree.runs):
        for run in tree.runs[instance]:
            got, notes = dispatch_views(
                run.instance,
                run.config,
                run.seed,
                run.result,
                run.heursols,
                single_worker_only,
            )
            if got:
                traced_runs += 1
            views.extend(got)
            diagnostics.extend(notes)
    return views, diagnostics, traced_runs


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

    check = check_is_probe(tree)
    if check.problems and not args.allow_non_probe:
        print(
            "ERROR: this is not a presolve-only probe tree:\n"
            + "\n".join(f"    {p}" for p in check.problems)
            + "\n    (analysing a full-solve tree measures what the solver found "
            "in its whole time limit, not what a presolve screen can see; pass "
            "--allow-non-probe to do it anyway)",
            file=sys.stderr,
        )
        return 2

    scan = informative_set(tree.runs)
    views, diagnostics, traced_runs = collect_views(
        tree, args.single_worker_trajectories
    )
    trajectories = summarise_traces(views)
    workers = observed_workers(views)

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
                render_informative_list(tree, check, scan, args, len(reference)),
            )
        if args.hard_tier_output:
            _write(
                args.hard_tier_output,
                render_hard_tier_list(
                    tree, check, scan, hard_tier, args, len(reference)
                ),
            )
        report = render_report(
            tree,
            check,
            scan,
            hard_tier,
            views,
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
