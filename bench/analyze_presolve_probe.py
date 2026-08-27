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

**Informative means the chain produced the reported incumbent.**  Two
signals are available and they do not agree, so the choice is explicit.
`[Heur] found=1` says the solution pool accepted a chain offer — CLAUDE.md's
"single definition of production".  A chain-sourced display row says that
offer actually became the incumbent.  They diverge whenever the chain
produced something that never beat what was already there: on
`supportcase10` FJ reports `found=1` while the only display row is `u`,
HiGHS's Trivial-upper at 70.

This module decides on the **display row**, at every log level, for two
reasons.  It is the predicate the tuning objective scores — #107 ranks
candidates on the presolve-exit primal bound, so an instance where no
candidate's solution can become the incumbent is a constant in every
comparison, which is exactly what the hard tier is for.  And it is the only
predicate both passes can use: `[Heur]` needs `log_dev_level=3`, the
filtering pass is prescribed to run without it and the trajectory pass with
it, so deciding on acceptance would let the same solve classify differently
in the two passes — a reproducibility hazard in the one artifact that gets
pinned by digest into a tuning-set header.

Acceptance is still read.  It splits the hard tier's *reason*
(`produced-not-improved` — the chain works here and is never good enough,
which is a datum about the heuristic) and it raises a diagnostic wherever
the two signals disagree.  That refinement needs the trace; membership does
not.  `CHAIN_SOURCES` is the set of display codes the patch assigns to our
four heuristics.

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
import itertools
import json
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
from analyze_results import load_results, parse_solu_file, resolve_reference
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
from run_target import presolve_objective, primal_gap

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

# Whether a heuristic's effort option sizes one worker's allowance rather
# than a whole dispatch.  FJ is the only one, flagged `budget_is_per_worker`
# in `kChain`; the other three are divided by N in `make_budget`.  It is
# here for the same reason `STALL_SCALES_WITH_WORKERS` is: a measured
# dispatch total is only comparable to an option value through this.
BUDGET_IS_PER_WORKER: dict[str, bool] = {
    "fj": True,
    "fpr": False,
    "local_mip": False,
    "scylla": False,
}

# `heuristic_effort_budget`'s anchor and base shift, from
# `src/heuristic_common.h`: the budget is `nnz << 12` effort units at effort
# 0.05, scaling linearly.
EFFORT_BASE_SHIFT = 12
EFFORT_ANCHOR = 0.05

# What counts as having reached the budget.  Not equality: concurrent
# workers overshoot `budget.total` by up to `n * attempt_cap`, and a
# dispatch that stopped a hair short still stopped *because of* the budget.
BUDGET_BOUND_FRACTION = 0.95

# The shipped effort defaults, for the report's comparison column only.  They
# are inherited rather than measured — fj is pinned to vanilla HiGHS's
# hardcoded `nnz << 10` per worker, and the other three are `0.30 x w/Sw` for
# weights proportional to a geomean `effort_per_ms` measured on a different
# instance set — which is the whole reason #113 derives a vector instead.
SHIPPED_EFFORT: dict[str, float] = {
    "fj": 0.0125,
    "fpr": 0.0884,
    "local_mip": 0.1821,
    "scylla": 0.0296,
}

# The quantile the proposed effort is read off: a budget that suffices to
# reach the last acceptance on 90 % of the dispatches that produced anything.
KNEE_QUANTILE = 0.9

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
    obj=<O> accepted=<0|1>   (only accepted offers are emitted since #113)

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


def objective_sense(result: SolveResult) -> str:
    """ "min" or "max", inferred from the run's own incumbent trajectory.

    HiGHS does not print the sense, but incumbents only ever move one way, so
    the direction of the display rows is the sense.  A run with fewer than two
    distinct incumbent objectives falls back to "min", the MIPLIB benchmark
    convention `analyze_results.resolve_reference` already assumes.
    """
    values = [inc.objective for inc in result.incumbents]
    for earlier, later in itertools.pairwise(values):
        if later < earlier:
            return "min"
        if later > earlier:
            return "max"
    return "min"


def improving_offers(members: list, sense: str) -> set[int]:
    """The ids of the offers that moved this dispatch's best objective.

    An accepted offer is not necessarily an improvement: `SolutionPool` keeps
    a top-`kPoolCapacity`, so a heuristic that keeps beating its own *worst*
    entry accepts indefinitely without the incumbent moving — measured on
    `egout`, FPR earns 40+ acceptances against 4 incumbent improvements.
    Calibrating on acceptances therefore reads the pool's admission policy
    rather than the heuristic's productivity, and it does so in both
    directions at once: it inflates the knee (the last acceptance lands near
    the clock, so "enough budget" becomes "the whole cap") while distorting
    the gaps.

    So the trajectory is taken over offers that strictly improved the best
    objective seen so far *within the dispatch*.  The first accepted offer
    counts as improving: what the pool held when the dispatch opened is not
    in the trace, so the alternative is to discard the opening solution of
    every dispatch, which is worse than over-counting by at most one.
    """
    best: float | None = None
    out: set[int] = set()
    for sample in members:
        if not sample.accepted:
            continue
        # `_field_of` and not attribute access: this reads the *parser's*
        # sample when parse_highs_log offers one, where the field is
        # `objective`, and this module's own adapter type when it does not,
        # where the wire name `obj` is kept.  The alias table is the one
        # place that pairing lives.
        obj = float(_field_of(sample, "obj"))
        if best is None:
            best, _ = obj, out.add(id(sample))
            continue
        margin = 1e-9 * max(1.0, abs(best))
        better = obj < best - margin if sense == "min" else obj > best + margin
        if better:
            best = obj
            out.add(id(sample))
    return out


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
    sense = objective_sense(result)
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
        improving = improving_offers(members, sense)
        for sample in members:
            if sample.worker == OFF_SLOT_WORKER:
                off_slot += int(id(sample) in improving)
                continue
            seen[sample.worker].append(sample.effort_at)
            if id(sample) in improving:
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
    # `mip_heuristic_<name>_effort` as the run was given it, per heuristic.
    # Absent for a heuristic the `.opts` did not set, which is the shipped
    # default rather than a known value — this records what the run was
    # *told*, and inferring the rest would make an unrecorded configuration
    # look recorded.
    efforts: dict[str, float] = field(default_factory=dict)


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


def run_efforts(opts_path: str) -> dict[str, float]:
    """The per-heuristic effort multipliers a run's `.opts` set."""
    if not os.path.isfile(opts_path):
        return {}
    options = read_options_file(Path(opts_path))
    out: dict[str, float] = {}
    for name in PRESOLVE_HEURISTICS:
        raw = options.get(f"mip_heuristic_{name}_effort")
        if raw is None:
            continue
        try:
            out[name] = float(raw)
        except ValueError:
            continue
    return out


def effort_budget(nnz: int, effort: float) -> int:
    """`heuristic_effort_budget` from `src/heuristic_common.h`, in Python.

    Duplicated deliberately: the alternative is inferring the budget from
    the measured effort, which is exactly the quantity being checked
    against it.  If the C++ formula moves, this must move with it.
    """
    if effort <= 0.0:
        return 0
    return int(float(nnz << EFFORT_BASE_SHIFT) * (effort / EFFORT_ANCHOR))


@dataclass
class BudgetCheck:
    """Whether any dispatch was stopped by its effort budget.

    The calibration probe runs at an effort the budget cannot reach, so that
    the wall clock is the single stopping rule and the trajectory measures
    the heuristic rather than the setting being derived from it.  That is a
    property of the *tree*, not of the launcher that wrote it — an arm run
    at a shipped default, or a model large enough to make even a huge budget
    reachable, produces a truncated yield curve that looks exactly like a
    converged one.  So it is checked here rather than assumed.

    `unknown` is dispatches with no recorded effort or no nonzero count;
    they are neither evidence for nor against.
    """

    dispatches: int = 0
    unknown: int = 0
    bound: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return self.dispatches - self.unknown

    @property
    def problems(self) -> list[str]:
        if not self.bound:
            return []
        worst = sorted(self.bound, key=lambda b: -b[2])[:MAX_LISTED]
        listed = ", ".join(
            f"{heur} on {inst} ({frac:.0%})" for heur, inst, frac in worst
        )
        return [
            (
                f"{len(self.bound)} of {self.checked} traced dispatch(es) "
                f"reached {BUDGET_BOUND_FRACTION:.0%} of their effort budget, "
                "so the budget and not the clock stopped them and their yield "
                f"curves are truncated: {listed}"
            )
        ]


def check_budget_headroom(
    views: list[DispatchView], efforts: dict[tuple[str, str, int], dict[str, float]]
) -> BudgetCheck:
    """Audit traced dispatches against the budget they were given."""
    check = BudgetCheck()
    for view in views:
        check.dispatches += 1
        effort = efforts.get((view.instance, view.config, view.seed), {}).get(view.name)
        if effort is None or view.nnz is None or view.total_effort is None:
            check.unknown += 1
            continue
        budget = effort_budget(view.nnz, effort)
        if BUDGET_IS_PER_WORKER[view.name] and view.workers:
            budget *= view.workers
        if budget <= 0:
            check.unknown += 1
            continue
        fraction = view.total_effort / budget
        if fraction >= BUDGET_BOUND_FRACTION:
            check.bound.append((view.name, view.instance, fraction))
    return check


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
        # Only the reference instances are ever looked at, so only they are
        # read: this tree carries a trace and is gigabytes.
        loaded = load_results(
            results_dir, configs, config_dirs=config_dirs, instances=set(instances)
        )
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
                        efforts=run_efforts(path[: -len(".log")] + ".opts"),
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

# Which corroborating signals a run carried.  These are *availability*
# labels, not decision paths: the verdict below is the same predicate at
# every log level (see `classify_run`).
EVIDENCE_HEURSOL = "heursol"
EVIDENCE_HEUR = "heur"
EVIDENCE_SOURCE = "source"

REASON_NO_ACCEPTANCE = "no-acceptance"
REASON_UNREACHED = "unreached"
REASON_TRIVIAL_ONLY = "trivial-only"
REASON_PRODUCED_NOT_IMPROVED = "produced-not-improved"

# Width of the reason column, derived so a new reason cannot silently run
# into the count beside it.
_REASON_WIDTH = (
    max(
        len(r)
        for r in (
            REASON_NO_ACCEPTANCE,
            REASON_UNREACHED,
            REASON_TRIVIAL_ONLY,
            REASON_PRODUCED_NOT_IMPROVED,
        )
    )
    + 2
)


@dataclass(frozen=True)
class RunVerdict:
    """What one run says about whether a presolve screen can see the model."""

    config: str
    seed: int
    # Which corroborating signal the log carried, strongest first.
    evidence: str
    # The verdict: a chain-sourced display row exists.
    informative: bool
    killed: bool
    # A solution exists and none of it is ours.
    trivial_only: bool
    # The pool accepted at least one chain offer.  None when the log carries
    # no trace to say — a run without `--dev-log`.
    produced: bool | None
    accepted: int | None

    @property
    def disagrees(self) -> bool:
        """The chain produced, and none of it reached the display."""
        return bool(self.produced) and not self.informative


def classify_run(run: ProbeRun) -> RunVerdict:
    """Did the presolve chain produce a solution that became the incumbent?

    **The verdict is the source test, at every log level.**  An instance is
    informative when some run shows a display row whose source is one of
    `CHAIN_SOURCES`.  The pool-acceptance signal — `[HeurSol] accepted=1`,
    equivalently `[Heur] found=1` — is read alongside it as `produced`, but
    it does not decide membership.  Two reasons, and the second is the one
    that forces the choice:

    1. **It is the predicate the tuning objective actually scores.**  #107
       ranks candidates on the presolve-exit primal bound, so two candidates
       differ on an instance only if one of them produces a solution that
       *becomes the reported incumbent*.  A pool acceptance that never beat
       HiGHS's own trivial bound leaves the score identical for every
       candidate, which is the definition of a constant instance.
       `supportcase10` is the real case: FJ reports `found=1` while the only
       display row is `u` at 70, so no candidate can be told apart there on
       what the chain did.
    2. **It is the only predicate available at both log levels.**  `[Heur]`
       and `[HeurSol]` need `log_dev_level=3`, and the campaign runs the
       filtering pass *without* `--dev-log` and the trajectory pass *with*
       it.  Deciding membership on acceptance would make the same solve
       classify differently in the two passes — a reproducibility hazard in
       the one artifact that gets pinned by digest into a tuning-set header.
       The source test reads display rows, which every level prints.
    3. **A killed run has incumbents and no ledger.**  `[Heur]` is written
       when a dispatch *ends*, so a run the probe's per-run cap SIGKILLs
       mid-dispatch carries none — while every incumbent row printed before
       the kill survives.  On the pilot `fj` tree this is not a corner case:
       `neos-4532248-waihi` and `nursesched-medium-hint03` were both killed
       at 210 s with `J`-sourced rows and no `[Heur]` line at all, so an
       acceptance-based predicate scores them as having produced nothing on
       instances where FJ demonstrably produced the incumbent.  The probe
       *needs* that cap, so this shape is built into the data it collects.

    What acceptance still buys, when it is there: it splits the hard tier's
    *reason* (see `informative_set`) and it flags the disagreement, so a
    heuristic that produces without ever improving is visible rather than
    silently filed under "found nothing".  That refinement is
    instrumentation-dependent; membership is not.
    """
    result = run.result
    ours = any(inc.source in CHAIN_SOURCES for inc in result.incumbents)
    trivial_only = bool(result.incumbents) and not ours

    chain = [s for s in run.heursols if s.name in PRESOLVE_HEURISTICS]
    presolve = [
        s
        for s in result.heuristic_samples
        if s.phase == "presolve" and s.name in PRESOLVE_HEURISTICS
    ]
    if chain:
        accepted: int | None = sum(1 for s in chain if s.accepted)
        produced: bool | None = bool(accepted)
        evidence = EVIDENCE_HEURSOL
    elif presolve:
        accepted = None
        produced = any(s.found for s in presolve)
        evidence = EVIDENCE_HEUR
    else:
        accepted = None
        produced = None
        evidence = EVIDENCE_SOURCE

    return RunVerdict(
        config=run.config,
        seed=run.seed,
        evidence=evidence,
        informative=ours,
        killed=result.killed,
        trivial_only=trivial_only,
        produced=produced,
        accepted=accepted,
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
    # instance -> the `config/seed` runs whose pool accepted a chain solution
    # that never reached the display.  The two signals disagreeing is a fact
    # about the heuristic, not a defect, and is reported rather than resolved
    # silently.
    disagreements: dict[str, list[str]] = field(default_factory=dict)

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

    An excluded instance carries one of four reasons, in this order of
    precedence:

    * *produced-not-improved* — the chain accepted a solution somewhere and
      none of them ever became the incumbent.  The most specific thing that
      can be said, and a real datum about the heuristic rather than about
      the instance.  Needs the trace, so a no-`--dev-log` tree reports the
      same instances under *trivial-only* or *no-acceptance* instead; the
      **membership is identical either way**, only the label is coarser.
    * *unreached* — every run was killed before the chain reported.
    * *trivial-only* — a solution exists everywhere it was looked for, and
      none of it is ours: HiGHS's own pre-chain heuristics did it.
    * *no-acceptance* — nothing was produced at all.
    """
    informative: list[str] = []
    excluded: list[str] = []
    verdicts: dict[str, list[RunVerdict]] = {}
    reasons: dict[str, str] = {}
    details: dict[str, str] = {}
    disagreements: dict[str, list[str]] = {}
    evidence_counts: dict[str, int] = defaultdict(int)

    for instance in sorted(runs):
        vs = [classify_run(run) for run in runs[instance]]
        verdicts[instance] = vs
        for v in vs:
            evidence_counts[v.evidence] += 1
        clashing = [f"{v.config}/seed{v.seed}" for v in vs if v.disagrees]
        if clashing:
            disagreements[instance] = clashing
        if any(v.informative for v in vs):
            informative.append(instance)
            continue
        excluded.append(instance)
        killed = sum(1 for v in vs if v.killed)
        trivial = sum(1 for v in vs if v.trivial_only)
        produced = sum(1 for v in vs if v.produced)
        if produced:
            reasons[instance] = REASON_PRODUCED_NOT_IMPROVED
            details[instance] = (
                f"{produced} of {len(vs)} run(s) had a chain solution accepted, "
                "none of which ever became the incumbent"
            )
        elif killed == len(vs):
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
        disagreements=disagreements,
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


@dataclass
class EffortKnee:
    """Where one heuristic stops producing, expressed as its own option.

    The `[HeurSol]` trace stamps every accepted solution with the charged
    effort at which it arrived, so one clock-bound dispatch is a whole
    cumulative-yield curve rather than a point on one.  The *knee* is the
    budget that would have sufficed to reach that dispatch's last
    acceptance: spend beyond it bought nothing on that instance.  Read as a
    high quantile over dispatches, it is a measured answer to "how much
    budget should this heuristic get?" — which the shipped vector, inherited
    from a retired envelope's weights, has never had.

    `barren` counts dispatches that never accepted.  They have no knee — the
    budget they would have needed is unknown and larger than what they
    spent — so they are excluded and reported rather than folded in as zero,
    which would drag the quantile down by exactly the instances the
    heuristic is worst at.
    """

    name: str
    options: list[float] = field(default_factory=list)
    censored: list[Observation] = field(default_factory=list)
    barren: int = 0
    still_improving: int = 0

    def quantile(self, p: float) -> float | None:
        """The events-only quantile — a *lower* bound on the knee.

        It sees only dispatches that produced, so it answers "how much
        budget did the ones that worked need?" and says nothing about the
        ones that did not.
        """
        return quantile(sorted(self.options), p) if self.options else None

    def censored_quantile(self, p: float) -> float | None:
        """The censoring-aware quantile, barren dispatches included.

        A barren dispatch is not a knee of zero and not a missing
        observation: it is a *right-censored* one.  The budget it would have
        needed is unknown and strictly greater than what it spent, which is
        exactly the shape Kaplan-Meier is for, and the same treatment
        `HeuristicTrajectory.censored_p95` already gives the improvement-free
        intervals that never ended.  Excluding them biases the answer
        low — they are the instances the heuristic is worst at — and folding
        them in as zero biases it lower still.

        `None` when the censoring never reaches `1 - p`, i.e. when no finite
        budget is implied by this data.
        """
        events = [
            Observation(v, f"{self.name}#{i}", True) for i, v in enumerate(self.options)
        ]
        return km_quantile(events + self.censored, p)


def effort_option_for(
    charged: int, nnz: int, workers: int | None, per_worker: bool
) -> float:
    """Invert `heuristic_effort_budget`: charged effort -> option value.

    FJ's option sizes one worker's allowance and the other three size a
    whole dispatch, so a dispatch total has to be divided by the worker
    count for FJ alone — the same asymmetry `BUDGET_IS_PER_WORKER` carries
    for the headroom check.
    """
    scale = workers if (per_worker and workers and workers > 0) else 1
    return (charged / scale) * EFFORT_ANCHOR / float(nnz << EFFORT_BASE_SHIFT)


def still_improving_at_the_end(view: DispatchView) -> bool:
    """Whether this dispatch was cut off mid-search rather than finished.

    Every dispatch of the probe ends on the clock, so "it stopped improving"
    is never something the run *reports* — it has to be inferred, and the
    dispatch's own rhythm is the only scale available to infer it against.
    If the silence since the last improvement is shorter than this
    dispatch's typical wait between improvements, nothing has been
    established: the next improvement may simply not have arrived yet.

    Such a dispatch's observed knee is a *lower* bound — the true one is
    larger — so counting it as a completed observation drags the upper
    quantiles down onto the cap.  It is right-censored, exactly like a
    barren dispatch and for the same reason.
    """
    if view.stale is None or not view.gaps:
        return False
    workers = view.workers if view.workers and view.workers > 0 else 1
    return view.stale / workers <= statistics.median(view.gaps)


def summarise_knees(views: list[DispatchView]) -> dict[str, EffortKnee]:
    """One yield knee per heuristic, pooled over every traced dispatch."""
    out = {name: EffortKnee(name=name) for name in PRESOLVE_HEURISTICS}
    for view in views:
        knee = out[view.name]
        if view.nnz is None or view.nnz <= 0:
            continue
        if view.productive <= 0:
            knee.barren += 1
            # The reverse knee: what it spent without ever producing, which
            # is a lower bound on what it would have needed.
            if view.total_effort:
                knee.censored.append(
                    Observation(
                        effort_option_for(
                            view.total_effort,
                            view.nnz,
                            view.workers,
                            BUDGET_IS_PER_WORKER[view.name],
                        ),
                        f"{view.name}#{view.instance}",
                        False,
                    )
                )
            continue
        option = effort_option_for(
            view.productive, view.nnz, view.workers, BUDGET_IS_PER_WORKER[view.name]
        )
        if still_improving_at_the_end(view):
            knee.still_improving += 1
            knee.censored.append(
                Observation(option, f"{view.name}#{view.instance}", False)
            )
            continue
        knee.options.append(option)
    return out


def effort_suggestions(knees: dict[str, EffortKnee]) -> list[str]:
    """The proposed effort vector, beside the vector that ships today."""
    lines = []
    for name in PRESOLVE_HEURISTICS:
        knee = knees[name]
        option = f"mip_heuristic_{name}_effort"
        value = knee.quantile(KNEE_QUANTILE)
        if value is None:
            # Not the same as "it produced nothing": every dispatch may have
            # produced and none of them have *finished* producing, which is
            # what a heuristic that improves rarely and late looks like at a
            # 30 s cap.  Its knee is simply not identified by this data, and
            # saying so is the result.
            why = (
                "no dispatch produced"
                if not knee.still_improving
                else f"none finished improving ({knee.still_improving} were still "
                f"improving at the cap, {knee.barren} barren)"
            )
            lines.append(f"  {option:<33} not identified: {why}")
            continue
        shipped = SHIPPED_EFFORT[name]
        censored = knee.censored_quantile(KNEE_QUANTILE)
        top = "unbnd" if censored is None else f"{censored:.4f}"
        lines.append(
            f"  {option:<33} {value:>9.4f} .. {top:<10} (shipped {shipped:.4f}; "
            f"p50 {knee.quantile(0.5):.4f}; {len(knee.options)} finished, "
            f"{knee.still_improving} still improving, {knee.barren} barren)"
        )
    return lines


def derived_defaults(
    trajectories: dict[str, HeuristicTrajectory],
    knees: dict[str, EffortKnee],
    workers: int | None,
    tree: ProbeTree,
    traced_runs: int,
    quality: QualityScan,
) -> dict:
    """The per-heuristic parameter vector this probe derives, as data.

    The report renders these for a human; this is the same numbers in a form
    a run, a search or a diff can consume, with enough provenance attached
    that a value can never be read without the worker count and the tree it
    came from.  Both are load-bearing: an effort value is only valid at the
    worker count it was measured at (FJ's option is per worker and the other
    three are divided by N), and a stall value is in that heuristic's own
    effort unit, which is not comparable across heuristics.

    `stall` is the *lower* end of the reported range — the events-only p95,
    i.e. the largest improvement-free interval that actually ended in an
    acceptance on 95 % of them.  The censoring-aware upper end is carried
    alongside as `stall_max` rather than chosen: it is frequently unbounded,
    which is a statement about the data, not a default anyone can ship.
    """
    out: dict = {
        "source_tree": tree.root,
        "provenance": {
            "configs": list(tree.configs),
            "instances_analysed": len(tree.runs),
            "runs_traced": traced_runs,
            "workers_observed": workers,
            "knee_quantile": KNEE_QUANTILE,
            "stall_quantile": STALL_QUANTILE,
        },
        "heuristics": {},
    }
    for name in PRESOLVE_HEURISTICS:
        knee, trajectory = knees[name], trajectories[name]
        low, high = trajectory.stall_range(workers)
        effort = knee.quantile(KNEE_QUANTILE)
        out["heuristics"][name] = {
            "effort": None if effort is None else round(effort, 6),
            "effort_shipped": SHIPPED_EFFORT[name],
            "effort_max": (
                None
                if knee.censored_quantile(KNEE_QUANTILE) is None
                else round(knee.censored_quantile(KNEE_QUANTILE), 6)
            ),
            "effort_p50": (
                None if knee.quantile(0.5) is None else round(knee.quantile(0.5), 6)
            ),
            "effort_scope": (
                "per_worker" if BUDGET_IS_PER_WORKER[name] else "per_dispatch"
            ),
            "stall": low,
            "stall_max": high,
            "stall_scales_with_workers": STALL_SCALES_WITH_WORKERS[name],
            "dispatches_finished": len(knee.options),
            "dispatches_still_improving": knee.still_improving,
            "dispatches_barren": knee.barren,
            "median_gap_to_best_known": (
                round(statistics.median(quality.gaps[name]), 6)
                if quality.gaps.get(name)
                else None
            ),
            "stale_fraction": (
                None
                if math.isnan(trajectory.stale_fraction)
                else round(trajectory.stale_fraction, 4)
            ),
        }
    return out


# A reference this close to zero makes the capped primal gap
# (`|obj - ref| / max(|ref|, 1)`) saturate: the denominator is pinned at 1,
# so every candidate that is not near-exact scores the same 1.0 and the
# instance ranks on cost alone.  Counted and reported rather than dropped —
# it is a property of the instance, and #107 needs to know what fraction of
# its objective is decided that way.
SATURATION_REFERENCE = 1.0

# Reference tags that carry no usable objective: an instance MIPLIB records
# as infeasible or unbounded would otherwise fall back to the best observed
# primal, which is a self-referential zero gap.
UNUSABLE_REFERENCE_TAGS = frozenset({"=inf=", "=unbd=", "=unkn="})


@dataclass
class QualityScan:
    """How good the solutions were, not merely that there were solutions.

    The informative set answers "can a presolve screen see this instance at
    all".  This answers "and how good is what it found", against the best
    known objective — which is what #107's objective actually scores, so it
    is the axis on which one heuristic's budget is worth more than
    another's.

    Gaps are `run_target.primal_gap` on `run_target.presolve_objective`: the same
    two functions the tuning search uses, deliberately reused rather than
    restated, so a candidate cannot score differently here than there.
    """

    gaps: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    scored_instances: int = 0
    saturating: list[str] = field(default_factory=list)
    unusable_reference: list[str] = field(default_factory=list)


def score_quality(
    runs: dict[str, list[ProbeRun]], solu_path: str | None
) -> QualityScan:
    """Gap-to-best-known per heuristic, over the instances that carry one."""
    scan = QualityScan()
    refs = parse_solu_file(solu_path) if solu_path and os.path.isfile(solu_path) else {}
    for instance in sorted(runs):
        tag, published = refs.get(instance, (None, None))
        if tag in UNUSABLE_REFERENCE_TAGS:
            scan.unusable_reference.append(instance)
            continue
        objectives = {
            run.config: presolve_objective(run.result) for run in runs[instance]
        }
        observed = [o for o in objectives.values() if o is not None]
        reference = resolve_reference(published, observed)
        if reference is None:
            scan.unusable_reference.append(instance)
            continue
        if abs(reference) <= SATURATION_REFERENCE:
            scan.saturating.append(instance)
        scored = {
            config: primal_gap(objective, reference)
            for config, objective in objectives.items()
            if objective is not None
        }
        if not scored:
            continue
        scan.scored_instances += 1
        for config, gap in scored.items():
            scan.gaps[config].append(gap)
        best = min(scored.values())
        # A tie is a win for everyone that reached it: these are single-arm
        # runs of one heuristic each, and "nobody wins" would understate a
        # heuristic that matched the best on every instance.
        for config, gap in scored.items():
            if gap <= best:
                scan.wins[config] += 1
    return scan


def quality_rows(scan: QualityScan) -> list[str]:
    """The per-heuristic quality table."""
    header = (
        f"{'heur':<10} {'scored':>7} {'median gap':>11} {'mean gap':>9} "
        f"{'exact':>6} {'best-of':>8}"
    )
    rows = [header]
    for name in PRESOLVE_HEURISTICS:
        gaps = scan.gaps.get(name, [])
        if not gaps:
            rows.append(f"{name:<10} {0:>7}")
            continue
        exact = sum(1 for g in gaps if g <= 0.0)
        rows.append(
            f"{name:<10} {len(gaps):>7} {statistics.median(gaps):>11.4f} "
            f"{statistics.fmean(gaps):>9.4f} {exact:>6} {scan.wins.get(name, 0):>8}"
        )
    return rows


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
        f"{'heur':<10}{'disp':>6}{'improve':>8}{'productive':>12}{'stale':>12}"
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
            f"{name:<10}{t.dispatches:>6}{t.accepts:>8}"
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
            "#   rule             a chain-sourced incumbent (display source"
            " A/M/G/J) in ANY run"
        ),
        (
            "#                    (union over configs).  Not `the pool accepted"
            " something`: the"
        ),
        (
            "#                    tuning objective scores the presolve-exit primal"
            " bound, so a"
        ),
        (
            "#                    solution that never became the incumbent leaves"
            " every candidate"
        ),
        (
            "#                    with the same score.  HiGHS's own pre-chain"
            " heuristics do not"
        ),
        (
            "#                    count either.  The rule reads display rows, which"
            " every log"
        ),
        (
            "#                    level prints, so a --dev-log run and a plain one"
            " of the same"
        ),
        ("#                    solve yield the same set."),
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
            "#   rule             no chain-sourced incumbent in ANY run"
            " (union over configs).  A"
        ),
        (
            "#                    pool acceptance that never became the"
            " incumbent does not count:"
        ),
        (
            "#                    the tuning objective scores the"
            " presolve-exit primal bound, so"
        ),
        ("#                    such an instance is a constant for every candidate."),
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
        "#   produced-not-improved",
        "#                    the chain had a solution accepted and it never",
        "#                    became the incumbent.  Needs log_dev_level=3; a",
        "#                    tree without it reports the same instances under",
        "#                    trivial-only or no-acceptance.  The membership is",
        "#                    identical either way, only the label is coarser.",
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
            "  NOTE: no [HeurSol] trace in this tree.  The informative set is "
            "unaffected — it reads display rows, which every log level prints "
            "— but the trajectories need one, and a hard-tier instance cannot "
            "be told apart as produced-not-improved.  Rerun with --dev-log."
        )
    if scan.disagreements:
        runs = sum(len(v) for v in scan.disagreements.values())
        notes.append(
            f"  NOTE: on {len(scan.disagreements)} instance(s) ({runs} run(s)) "
            "the pool accepted a chain solution that never became the "
            "incumbent, so [Heur] found=1 and the display disagree.  The "
            "verdict follows the display; see produced-not-improved."
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
    budget: BudgetCheck,
    knees: dict[str, EffortKnee],
    quality: QualityScan,
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
        (
            f"  budget check   {budget.checked - len(budget.bound)}/"
            f"{budget.checked} traced dispatch(es) clock-bound, "
            f"{budget.unknown} unrecorded"
        ),
        "  signals        "
        + ", ".join(
            f"{tier}={scan.evidence_counts.get(tier, 0)}"
            for tier in (EVIDENCE_HEURSOL, EVIDENCE_HEUR, EVIDENCE_SOURCE)
        ),
    ]
    lines += _header_notes(tree, check, scan)
    for problem in budget.problems:
        lines.append(f"  WARNING: {problem}")

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
    for reason in (
        REASON_NO_ACCEPTANCE,
        REASON_TRIVIAL_ONLY,
        REASON_PRODUCED_NOT_IMPROVED,
        REASON_UNREACHED,
    ):
        lines.append(f"  {reason:<{_REASON_WIDTH}}{by_reason.get(reason, 0)}")
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
            f"  NOTE: {off_slot} improving offer(s) came from worker -1 (off any "
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
        (
            f"Solution quality against best known ({quality.scored_instances} "
            "instance(s) with a usable reference):"
        ),
        *["  " + row for row in quality_rows(quality)],
        (
            f"  {len(quality.saturating)} instance(s) have |reference| <= "
            f"{SATURATION_REFERENCE:g}, where the capped gap saturates and the "
            "objective ranks on cost alone"
        ),
    ]
    if quality.unusable_reference:
        lines.append(
            f"  {len(quality.unusable_reference)} instance(s) carry no usable "
            "reference and are excluded from the table"
        )
    lines += [
        "",
        (f"Proposed stall ranges for #107 (per nnz, at {worker_note} worker(s)):"),
        *stall_suggestions(trajectories, workers),
        "",
        (
            f"Proposed effort vector for #107 (p{KNEE_QUANTILE:.0%} yield knee, "
            f"at {worker_note} worker(s)):"
        ),
        *effort_suggestions(knees),
        (
            "  low: the budget that reaches the last improvement on "
            f"{KNEE_QUANTILE:.0%} of the dispatches that *finished* improving.  "
            "high: the same quantile with the rest carried as right-censored -- "
            "a dispatch still improving when the cap fired, and a barren one, "
            "both have a knee larger than what they were seen to spend."
        ),
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
        "--solu",
        default=os.path.join(BENCH_DIR, "miplib2017-v36.solu"),
        metavar="FILE",
        help=(
            "MIPLIB .solu reference objectives for the gap-to-best-known "
            "table (default: the bundled v36 copy)"
        ),
    )
    parser.add_argument(
        "--defaults-output",
        metavar="FILE",
        help=(
            "write the derived per-heuristic effort and stall values here as "
            "JSON, with the worker count and tree they are only valid for"
        ),
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
    knees = summarise_knees(views)
    quality = score_quality(tree.runs, args.solu)
    workers = observed_workers(views)
    budget_check = check_budget_headroom(
        views,
        {
            (run.instance, run.config, run.seed): run.efforts
            for runs in tree.runs.values()
            for run in runs
        },
    )

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
        if args.defaults_output:
            _write(
                args.defaults_output,
                json.dumps(
                    derived_defaults(
                        trajectories, knees, workers, tree, traced_runs, quality
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
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
            budget_check,
            knees,
            quality,
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
