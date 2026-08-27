"""Parse HiGHS MIP solver log output to extract benchmark metrics."""

from __future__ import annotations

import itertools
import math
import re
import warnings
from dataclasses import dataclass, field


@dataclass
class Incumbent:
    """A single incumbent solution update."""

    time: float
    objective: float
    source: str  # Single character: H, J, B, T, L, R, etc.
    nodes: int
    dual_bound: float = float("-inf")  # Dual bound at time of incumbent


@dataclass
class SequentialSample:
    """A single [Sequential] per-heuristic wall-clock observation.

    Emitted by `EffortLedger::book` in `src/effort_ledger.cpp`: once per
    presolve-chain heuristic per solve, and once per dive-time `fpr_lp`
    dispatch.  The four presolve heuristics and `fpr_lp` draw from separate
    budgets, so a consumer comparing rates should say which it means.
    """

    heuristic: str  # fj, fpr, local_mip, scylla, fpr_lp
    effort: int
    wall_ms: float
    effort_per_ms: float


@dataclass
class HeuristicSample:
    """A single `[Heur]` per-heuristic observation.

    Emitted by `EffortLedger::book` alongside the legacy `[Sequential]`
    line, once per presolve-chain heuristic per solve and once per
    dive-time `fpr_lp` dispatch.  Carries what `[Sequential]` cannot:
    *when* the heuristic ran, on the solver's own clock; which side of the
    patch boundary it ran on (`phase`); and whether it produced anything
    (`found`).
    """

    name: str  # fj, fpr, local_mip, scylla, fpr_lp
    phase: str  # presolve | dive
    start_s: float
    end_s: float
    effort: int
    wall_ms: float
    effort_per_ms: float
    found: bool
    # Constraint-matrix nonzeros of the post-presolve MIP — the denominator
    # `mip_heuristic_<name>_patience` is expressed in.  None for a log written
    # before #106 added the field.
    #
    # It is on this line, and not read from HiGHS's own model header,
    # because a *presolve-only* log has no model header: HiGHS prints
    # `MIP <name> has <r> rows; <c> cols; <n> nonzeros;` on entry to branch
    # and bound, and `mip_heuristic_presolve_only` exits before that.  The
    # two size figures such a log does carry are both the wrong matrix
    # (`Nonzeros : N` is the original model, `<r> rows, <c> cols,
    # <n> nonzeros` the post-presolve LP).  See `SolveResult.num_nonzeros`,
    # which stays and is the fallback on full-solve logs.
    nnz: int | None = None


@dataclass
class HeurSolSample:
    """A single `[HeurSol]` per-offered-solution observation.

    Emitted by `IncumbentSink::offer` in `src/incumbent_sink.cpp`, once per
    solution any heuristic worker offers the shared pool — accepted or not.
    `[Heur]` is one line per *dispatch* and therefore cannot show what
    happens inside one; this is the line the patience calibration
    (#106 / #107) reads.

    `effort_at` is the offering worker's own charged effort at the moment of
    the offer, in that heuristic's own effort unit, and is monotone
    non-decreasing within a `(name, dispatch, worker)` triple — the C++ side
    carries a retired worker's charge into its replacement so a rebuild does
    not restart the count.  Units are *not* comparable across heuristics.
    """

    name: str  # fj, fpr, local_mip, scylla, fpr_lp
    dispatch: int  # process-global; unique per dispatch, not dense per solve
    worker: int  # worker slot index; -1 for an offer made off any slot
    effort_at: int
    wall_ms: float  # since the dispatch started; may be negative (see below)
    objective: float
    accepted: bool
    # Index into `SolveResult.heuristic_samples` of the `[Heur]` line that
    # closed this dispatch, or None when the log ended before it (a killed
    # run).  Bound during parsing, from the `[Heur]` line itself: a
    # dispatch's `[HeurSol]` lines all precede it and follow the previous
    # `[Heur]` for the same name.
    #
    # **This is the supported way to relate an offer to its dispatch**, and
    # `SolveResult.dispatch_traces()` is the supported way to consume it.
    # Do not re-derive the binding by zipping `heursol_samples` against
    # `heuristic_samples`: a dispatch that offered nothing emits a `[Heur]`
    # line and no `[HeurSol]` at all, which shifts every later pairing, and
    # `fpr_lp` dispatches repeatedly within one solve.  Two implementations
    # of this seam is how it silently diverges.
    heur_index: int | None = None


# Whether a per-worker `effort_at` gap must be multiplied by the worker
# count to be comparable with the `mip_heuristic_<name>_patience` option value.
#
# The option is one number, but the scope it is armed at differs per
# heuristic and always has (`src/mode_dispatch.cpp`, `make_budget`):
#
#   fj        — the option is *per worker*, and `worker_stale` works out to
#               `nnz * per_nnz`.  A per-worker gap is already in the
#               option's unit.
#   fpr       — whole-dispatch option, `worker_stale = nnz * per_nnz / N`.
#               A per-worker gap is N times smaller than the option value.
#   local_mip — same as fpr.
#   scylla    — its worker gate is armed with the *dispatch* value directly
#               (`budget.stale`, not `worker_stale`), because its per-worker
#               counter is charged the PDLP cost already divided by N.  So a
#               per-worker gap is again in the option's unit.
#
# Getting this wrong is not cosmetic: at the probe's 16 workers, reading a
# p90 straight off an unscaled fpr/local_mip gap ships a patience default 16x
# tighter than intended — the direction this codebase repeatedly flags as
# the one that costs solutions.
#
# `fpr_lp` is deliberately absent: it has no patience option (it keeps the
# pre-#111 `worker_budget >> 2` and its own attempt counters), so there is
# no value for a quantile of its gaps to be offered as, and
# `normalized_gaps` refuses rather than inventing a scope.
_GAP_SCALES_WITH_WORKERS = {
    "fj": False,
    "fpr": True,
    "local_mip": True,
    "scylla": False,
}

# Whether a heuristic's per-worker `effort_at` counter is in the same unit
# as the `[Heur] effort` total, i.e. whether `total - productive` is a
# meaningful "effort spent after the last acceptance".
#
# Scylla's is not.  `[Heur] effort` sums `attempt.effort`, which takes the
# *full* PDLP cost, while the per-worker counter `effort_at` reports takes
# it divided by the worker count (`src/scylla_worker.cpp`, the `local_effort`
# / `actual_effort` split).  The difference is not a rounding error: on a
# measured `gt2` dispatch where *every* offer was accepted, `total -
# productive` still came out at 90% of the dispatch, and on `lseu` at 86%.
# A calibration reading that would conclude Scylla wastes nearly everything
# and set a very tight `mip_heuristic_scylla_patience`.  Scaling by N would not
# fix it either — only the PDLP part of the counter is amortised, the FPR
# rounding part is charged in full — so the number is withheld with its
# reason instead.  `productive_effort` and the gaps stay valid for Scylla:
# both are differences *within* that one counter.
_EFFORT_AT_IN_DISPATCH_UNITS = {
    "fj": True,
    "fpr": True,
    "local_mip": True,
    "scylla": False,
    "fpr_lp": True,
}


@dataclass
class DispatchTrace:
    """All `[HeurSol]` lines of one `(name, dispatch)`, plus its totals.

    Produced by `SolveResult.dispatch_traces()`, which is the supported way
    to bind `[HeurSol]` lines to the `[Heur]` line that closed their
    dispatch — do not re-derive that binding from log order; see
    `HeurSolSample.heur_index`.

    This is the unit #107 calibrates patience on: patience answers "how much
    improvement-free effort is enough before this is going nowhere?", and
    the answer is a high quantile (p90-p95) of
    `normalized_gaps()`, which is already in the option's own unit.
    """

    name: str
    dispatch: int
    samples: list[HeurSolSample]
    # `[Heur] effort` for this dispatch — the effort actually charged,
    # summed over every worker.  None when the matching `[Heur]` line is
    # absent (truncated log).
    total_effort: int | None = None
    # Constraint-matrix nonzeros of the post-presolve MIP, for
    # `normalized_gaps`.  Resolved by `SolveResult.dispatch_traces`, which
    # prefers the `[Heur] nnz=` field and falls back to the model header;
    # see `nnz_missing_reason` for what None means.
    nnz: int | None = None
    # Worker count for this solve, from HiGHS's `Thread count N` line.
    # Needed to put an fpr/local_mip gap into its option's scope.
    workers: int | None = None
    # Why `nnz` is None, phrased in terms of the log shape, so a refusal
    # says what it was given rather than only what it wanted.
    nnz_missing_reason: str | None = None

    @property
    def accepted_samples(self) -> list[HeurSolSample]:
        """Offers the pool took, in emission order."""
        return [s for s in self.samples if s.accepted]

    @property
    def productive_effort(self) -> int:
        """Charged effort at the last accepted solution, summed over workers.

        Per worker because `effort_at` is per worker: each worker's value at
        its own last accepted offer, summed.  Zero when nothing was
        accepted.  In the heuristic's own effort unit — for Scylla that is
        the amortised one, which is why `stale_effort` is withheld there.
        """
        last: dict[int, int] = {}
        for s in self.accepted_samples:
            last[s.worker] = s.effort_at
        return sum(last.values())

    @property
    def stale_effort_unavailable_reason(self) -> str | None:
        """Why `stale_effort` is None, or None when it is available."""
        if self.total_effort is None:
            return (
                f"no [Heur] line closed dispatch {self.name}/{self.dispatch} "
                "(truncated log?), so the dispatch total is unknown"
            )
        if not _EFFORT_AT_IN_DISPATCH_UNITS.get(self.name, False):
            return (
                f"{self.name}'s per-worker effort counter is not in the same unit as "
                "its [Heur] effort total (the PDLP cost is amortised by the worker "
                "count in one and not the other), so total - productive is not a "
                "measure of improvement-free effort"
            )
        return None

    @property
    def stale_effort(self) -> int | None:
        """Charged effort spent after the last acceptance — the gate's target.

        `total_effort - productive_effort`, floored at zero.  None when the
        subtraction is not meaningful; `stale_effort_unavailable_reason`
        says why.  The floor covers a small residual mismatch that is
        genuine on the heuristics where the quantity *is* meaningful:
        LocalMIP charges its cold-start construction sweep to the dispatch
        total but to no worker's counter.
        """
        if self.stale_effort_unavailable_reason is not None:
            return None
        assert self.total_effort is not None
        return max(self.total_effort - self.productive_effort, 0)

    def acceptance_gaps(self, *, include_first: bool = True) -> list[int]:
        """Effort between consecutive accepted offers, per worker.

        Raw, in the offering worker's own counter — see `normalized_gaps`
        for the form that is comparable with a patience option.  Gaps are taken
        *within* a worker, since `effort_at` is that worker's own counter and
        the sequences of different workers interleave in the log.  With
        `include_first` (the default) each worker's first acceptance also
        contributes the effort it spent getting there, which is a genuine
        improvement-free interval and the one a patience gate would have cut
        first.
        """
        per_worker: dict[int, list[int]] = {}
        for s in self.accepted_samples:
            per_worker.setdefault(s.worker, []).append(s.effort_at)
        gaps: list[int] = []
        for values in per_worker.values():
            if include_first:
                gaps.append(values[0])
            gaps.extend(b - a for a, b in itertools.pairwise(values))
        return gaps

    @property
    def gap_scale(self) -> int:
        """Multiplier putting an `acceptance_gaps` value in the option's scope.

        `1` or the worker count; see `_GAP_SCALES_WITH_WORKERS`.  Raises when
        the heuristic has no patience option, or when it needs a worker count
        the log did not carry.
        """
        scales = _GAP_SCALES_WITH_WORKERS.get(self.name)
        if scales is None:
            raise ValueError(
                f"{self.name} has no mip_heuristic_*_patience option, so its gaps have "
                "no option scope to be expressed in"
            )
        if not scales:
            return 1
        if not self.workers:
            raise ValueError(
                f"{self.name}'s patience option is whole-dispatch scoped, so a per-worker "
                "gap must be scaled by the worker count, and the log carried no "
                "'Thread count N (of M threads)' line to read it from"
            )
        return self.workers

    def normalized_gaps(
        self, nnz: int | None = None, *, include_first: bool = True
    ) -> list[float]:
        """`acceptance_gaps`, in the unit `mip_heuristic_<name>_patience` uses.

        Effort units per constraint-matrix nonzero, at the option's own
        scope, so a quantile of this list is directly a candidate value for
        the option.  Two corrections are applied to the raw gaps: division
        by the nonzero count, and multiplication by `gap_scale`.

        Refuses rather than returning a number that is wrong by a factor.
        """
        n = self.nnz if nnz is None else nnz
        if not n:
            raise ValueError(
                f"nonzero count unknown for dispatch {self.name}/{self.dispatch}: "
                f"{self.nnz_missing_reason or 'no source in this log'}"
            )
        scale = self.gap_scale
        return [
            g * scale / n for g in self.acceptance_gaps(include_first=include_first)
        ]


@dataclass
class SolveResult:
    """Parsed result from a HiGHS MIP solve."""

    status: str = ""
    primal_bound: float = float("inf")
    dual_bound: float = float("-inf")
    gap: float = float("inf")
    pd_integral: float = float("inf")
    solve_time: float = 0.0
    nodes: int = 0
    lp_iterations: int = 0
    # Model stats parsed from `MIP <name> has R rows; C cols; NZ nonzeros;
    # I integer variables (B binary)`; None means the line was absent.
    num_rows: int | None = None
    num_cols: int | None = None
    num_nonzeros: int | None = None
    num_integer: int | None = None
    num_binary: int | None = None
    # Parsed from `Thread count N (of M threads). Using K max workers.`, which
    # HiGHS prints once per MIP solve.  `thread_count` is the size of HiGHS's
    # thread pool and therefore **the** worker count our presolve heuristics
    # run at (`ExecutionContext::num_workers` is `highs::parallel::
    # num_threads()`); `max_workers` is B&B's parallel-search cap, a different
    # number.  None means the line was absent.  This is the only record of the
    # effective worker count a benchmark run leaves behind: the harness
    # deliberately does not pin `threads`, so it is a property of the run host
    # rather than of any options file.
    thread_count: int | None = None
    hardware_threads: int | None = None
    max_workers: int | None = None
    # True when `bench/run_benchmark.py` had to SIGKILL the solver because it
    # blew past the runner's grace window.  HiGHS only checks the clock between
    # its own work units, so a single long simplex solve at the root can carry
    # a run well past `time_limit` without ever returning to look; the runner
    # kills it and keeps whatever had been streamed to stdout by then.  Such a
    # log is *truncated, not invalid*: it has no Solving report block, so
    # `status` / `primal_bound` / `gap` stay at their defaults, but every
    # incumbent line printed before the kill is present, and those are what
    # T1st and `primal_integral` are computed from.  Distinguishing this from a
    # clean solve that genuinely found nothing is the point of the flag.
    killed: bool = False
    killed_after: float | None = None
    incumbents: list[Incumbent] = field(default_factory=list)
    sequential_samples: list[SequentialSample] = field(default_factory=list)
    heuristic_samples: list[HeuristicSample] = field(default_factory=list)
    # Both None for a log produced before issue #95, or by any run below
    # log_dev_level=3.
    # One entry per offered solution (#106); empty for the same reasons, and
    # for a run in which no heuristic offered anything.
    heursol_samples: list[HeurSolSample] = field(default_factory=list)

    @property
    def category(self) -> str | None:
        """Local-MIP §6.1.1 category: BP / IP / MBP / MIP / None.

        BP  — all variables binary.
        IP  — all variables integer (no continuous), some non-binary integer.
        MBP — all integers are binary AND some continuous variables exist.
        MIP — some non-binary integer AND some continuous.
        """
        if self.num_cols is None or self.num_integer is None or self.num_binary is None:
            return None
        cont = self.num_cols - self.num_integer
        gen_int = self.num_integer - self.num_binary
        if cont == 0 and gen_int == 0 and self.num_binary > 0:
            return "BP"
        if cont == 0 and gen_int > 0:
            return "IP"
        if cont > 0 and gen_int == 0 and self.num_binary > 0:
            return "MBP"
        if cont > 0 and gen_int > 0:
            return "MIP"
        return None

    def dispatch_traces(self) -> list[DispatchTrace]:
        """Group the `[HeurSol]` lines into one `DispatchTrace` per dispatch.

        **The supported way to relate offers to their dispatch** — see
        `HeurSolSample.heur_index`; do not re-derive the binding by zipping
        the two sample lists.

        Order of first appearance.  Dispatch ids are process-global, so they
        are neither zero-based nor dense within a solve and nothing here
        assumes otherwise — grouping is on the `(name, dispatch)` pair.

        The nonzero count each trace needs for `normalized_gaps` is resolved
        here, preferring the dispatch's own `[Heur] nnz=` field over the
        model header.  The preference is not cosmetic: a `[HeurSol]` line
        exists only at `log_dev_level=3`, and at that level HiGHS prints a
        `MIP / Rows / Cols / Nonzeros` block instead of the one-line
        `MIP <name> has <r> rows; <c> cols; <n> nonzeros;` header
        `_MODEL_HEADER_RE` reads — so on exactly the logs that carry a
        trace, `num_nonzeros` is None.  The block would be the wrong number
        anyway (it is the *original* model, and the patience options are
        denominated in the post-presolve MIP matrix), which is why the field
        is emitted from C++ rather than recovered by a better regex.  The
        header stays as the fallback for a log from a pre-#106 binary.
        """
        traces: dict[tuple[str, int], DispatchTrace] = {}
        for sample in self.heursol_samples:
            key = (sample.name, sample.dispatch)
            trace = traces.get(key)
            if trace is None:
                closing = None
                if sample.heur_index is not None and sample.heur_index < len(
                    self.heuristic_samples
                ):
                    closing = self.heuristic_samples[sample.heur_index]
                nnz = closing.nnz if closing is not None else None
                reason = None
                if nnz is None:
                    nnz = self.num_nonzeros
                if nnz is None:
                    reason = (
                        "the closing [Heur] line carries no nnz= field "
                        "(log written by a binary predating issue #106) and the log has "
                        "no 'MIP <name> has <r> rows; <c> cols; <n> nonzeros;' header "
                        "(log_dev_level=3 prints a Rows/Cols/Nonzeros block instead, and "
                        "that block is the pre-presolve model, not the matrix the patience "
                        "options are denominated in)"
                    )
                trace = DispatchTrace(
                    name=sample.name,
                    dispatch=sample.dispatch,
                    samples=[],
                    total_effort=None if closing is None else closing.effort,
                    nnz=nnz,
                    workers=self.thread_count,
                    nnz_missing_reason=reason,
                )
                traces[key] = trace
            trace.samples.append(sample)
        return list(traces.values())

    @property
    def time_to_first_feasible(self) -> float | None:
        """Time when the first feasible solution was found."""
        if self.incumbents:
            return self.incumbents[0].time
        return None

    @property
    def time_to_best(self) -> float | None:
        """Time when the last (best) incumbent was recorded."""
        if self.incumbents:
            return self.incumbents[-1].time
        return None

    def primal_gap_at(
        self, time_cutoff: float, best_known: float | None = None
    ) -> float | None:
        """Primal gap at a given time cutoff, capped at 1.0.

        If best_known is provided, gap = (obj - best_known) / max(|best_known|, 1).
        Otherwise, uses dual bound at the time of the incumbent.
        Capped at 1.0 so that "no solution" (sentinel 1.0) is never beaten by
        a found-but-terrible solution on instances where |ref| < 1 (e.g.
        scheduling problems with optimal=0 and large violation counts).
        """
        # Find the last incumbent at or before the cutoff
        last_inc = None
        for inc in self.incumbents:
            if inc.time <= time_cutoff:
                last_inc = inc
            else:
                break
        if last_inc is None:
            return None  # No feasible solution by cutoff
        ref = best_known if best_known is not None else last_inc.dual_bound
        if not math.isfinite(ref):
            return None  # dual bound not yet finite; no meaningful gap
        denom = max(abs(ref), 1.0)
        return min(abs(last_inc.objective - ref) / denom, 1.0)

    def primal_gap_curve(
        self, best_known: float | None = None
    ) -> list[tuple[float, float]]:
        """Return (time, gap) points for primal integral computation, gap capped at 1.0."""
        points = []
        for inc in self.incumbents:
            ref = best_known if best_known is not None else inc.dual_bound
            if not math.isfinite(ref):
                continue  # dual bound not yet finite; integral stays at 1.0
            denom = max(abs(ref), 1.0)
            gap = min(abs(inc.objective - ref) / denom, 1.0)
            points.append((inc.time, gap))
        return points

    def primal_integral(
        self, time_limit: float, best_known: float | None = None
    ) -> float:
        """Compute primal integral (area under primal gap curve).

        Uses the P-D integral from HiGHS if available, otherwise computes
        from incumbent updates.
        """
        if not self.incumbents:
            return float(time_limit)  # gap held at 1.0 for entire duration
        curve = self.primal_gap_curve(best_known)
        integral = 0.0
        # Before first feasible: gap is effectively infinite, but we cap at 1.0
        prev_time = 0.0
        prev_gap = 1.0  # No solution = 100% gap
        for t, g in curve:
            # Ignore anything past the horizon the integral is measured over.
            # A clean solve stops at its own limit so this never bites, but a
            # killed run (`killed`) keeps streaming until the runner's grace
            # window expires and can carry incumbents well beyond it.  Without
            # the break those later points are integrated in *and* leave
            # `prev_time > time_limit`, so the remainder term below goes
            # negative and the instance scores better than a solve that found
            # the same solution inside the window.
            if t >= time_limit:
                break
            integral += prev_gap * (t - prev_time)
            prev_time = t
            prev_gap = g
        # Remainder until time_limit
        integral += prev_gap * (time_limit - prev_time)
        return integral


# Regex for MIP log data lines.
# Source char (or space) at position 0, then fields separated by whitespace.
# Format: Src  Proc. InQueue |  Leaves   Expl. | BestBound  BestSol  Gap | Cuts InLp Confl. | LpIters Time
_LOG_LINE_RE = re.compile(
    r"^[ ]?([A-Za-z ])"  # source code (pos 0 or 1, e.g. "H " or " B")
    r"\s+([\d.]+[kMG]?)"  # nodes processed
    r"\s+([\d.]+[kMG]?)"  # nodes in queue
    r"\s+([\d.]+[kMG]?)"  # leaves
    r"\s+([\d.]+)%"  # explored %
    r"\s+(\S+)"  # best bound
    r"\s+(\S+)"  # best solution
    r"\s+(\S+)"  # gap
    r"\s+(\d+)"  # cuts
    r"\s+(\d+)"  # in lp
    r"\s+(\d+)"  # conflicts
    r"\s+([\d.]+[kMG]?)"  # lp iters
    r"\s+([\d.]+)s"  # time
)

# Solving report patterns
_STATUS_RE = re.compile(r"^\s+Status\s+(.+)$")
_PRIMAL_RE = re.compile(r"^\s+Primal bound\s+(.+)$")
_DUAL_RE = re.compile(r"^\s+Dual bound\s+(.+)$")
_GAP_RE = re.compile(r"^\s+Gap\s+(.+)$")
_PD_RE = re.compile(r"^\s+P-D integral\s+(.+)$")
_TIMING_RE = re.compile(r"^\s+Timing\s+([\d.]+)$")
_NODES_RE = re.compile(r"^\s+Nodes\s+(\d+)$")
_LPITERS_RE = re.compile(r"^\s+LP iterations\s+(\d+)$")

# Runner-injected marker, written by `record_partial` / `record_failure` in
# `bench/run_benchmark.py` when the solver had to be killed.  Anchored at
# column 0 so it cannot collide with HiGHS's own indented report lines.  Logs
# written before the runner kept partial output consist of this single line and
# nothing else; both shapes parse.
_KILLED_RE = re.compile(r"^TIMEOUT: process killed after ([\d.]+)s")

# [Sequential] per-heuristic effort line emitted from
# src/effort_ledger.cpp `EffortLedger::book` (issue #71):
#   [Sequential] heur=fpr effort=12345 wall_ms=67.8 effort_per_ms=182
# There is one line per heuristic per solve.
# `wall_ms` takes an optional sign here, as it does on `[Heur]`: the ledger
# times against HiGHS's own solver clock, which bottoms out in
# `high_resolution_clock` (== non-monotonic `system_clock` on libstdc++),
# so a wall-clock step can produce a negative window.  Rare, but a pattern
# without the sign drops the sample instead of surfacing the artefact.
_SEQUENTIAL_RE = re.compile(
    r"^\s*\[Sequential\] heur=(\S+) effort=(\d+) wall_ms=(-?[\d.]+) effort_per_ms=([\d.]+)"
)


def _tagged_fields(line: str, tag: str) -> dict[str, str] | None:
    """`key=value` tokens of a `tag`-prefixed line, or None if it is not one.

    Shared by `[Heur]` and `[HeurSol]`.  Neither is parsed positionally:
    both gained a field mid-issue (`[Heur]` gained `nnz`, `[HeurSol]` gained
    `worker`), and a positional pattern turns the next such addition into a
    silent parse failure across every archived log — including the release
    archive, which re-derives its tables from logs it did not produce.
    `[Sequential]` stays positional: it is frozen for the external tooling
    that parses it.
    """
    stripped = line.strip()
    if not stripped.startswith(tag):
        return None
    fields: dict[str, str] = {}
    for token in stripped[len(tag) :].split():
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    return fields


# Per-heuristic instrumentation at log_dev_level=3, emitted next to the
# legacy `[Sequential]` line by `EffortLedger::book`:
#   [Heur] name=fj phase=presolve start_s=0.412 end_s=1.077 effort=8388608 \
#          wall_ms=665.2 effort_per_ms=12610.1 found=1 nnz=2831
# `wall_ms` may be negative: the solver clock is not monotonic, so a
# negative sample is surfaced rather than silently skipped.  `nnz` is
# optional, being absent from logs written before #106 added it.
_HEUR_TAG = "[Heur]"
_HEUR_KEYS = (
    "name",
    "phase",
    "start_s",
    "end_s",
    "effort",
    "wall_ms",
    "effort_per_ms",
    "found",
)


def _parse_heur(line: str) -> HeuristicSample | None:
    """Parse one `[Heur]` line, or None if it is not one / is malformed."""
    fields = _tagged_fields(line, _HEUR_TAG)
    if fields is None or not all(k in fields for k in _HEUR_KEYS):
        return None
    try:
        nnz = fields.get("nnz")
        return HeuristicSample(
            name=fields["name"],
            phase=fields["phase"],
            start_s=float(fields["start_s"]),
            end_s=float(fields["end_s"]),
            effort=int(fields["effort"]),
            wall_ms=float(fields["wall_ms"]),
            effort_per_ms=float(fields["effort_per_ms"]),
            found=fields["found"] != "0",
            nnz=None if nnz is None else int(nnz),
        )
    except ValueError:
        return None


# Per-offered-solution instrumentation at log_dev_level=3, emitted by
# `IncumbentSink::offer`:
#   [HeurSol] name=fpr dispatch=2 worker=3 effort_at=91238 wall_ms=12.4 \
#             obj=778.45908999999983 accepted=1
#
# Parsed as a `key=value` dict rather than a positional pattern, unlike
# `[Heur]` and `[Sequential]` above.  The line already gained a field once
# (`worker`, added during #106 while three tracks were consuming it), and a
# positional pattern turns the next such addition into a silent parse
# failure in every archived log — including the release archive, which
# re-derives its tables from logs it did not produce.  Unknown keys are
# ignored and missing known keys drop the line.
_HEURSOL_PREFIX = "[HeurSol]"
_HEURSOL_KEYS = (
    "name",
    "dispatch",
    "worker",
    "effort_at",
    "wall_ms",
    "obj",
    "accepted",
)


def _parse_heursol(line: str) -> HeurSolSample | None:
    """Parse one `[HeurSol]` line, or None if it is not one / is malformed."""
    fields = _tagged_fields(line, _HEURSOL_PREFIX)
    if fields is None or not all(k in fields for k in _HEURSOL_KEYS):
        return None
    try:
        return HeurSolSample(
            name=fields["name"],
            dispatch=int(fields["dispatch"]),
            worker=int(fields["worker"]),
            effort_at=int(fields["effort_at"]),
            # Signed for the same reason `[Heur] wall_ms` is: the solver
            # clock is not monotonic, so a window can come out negative.
            wall_ms=float(fields["wall_ms"]),
            objective=float(fields["obj"]),
            accepted=fields["accepted"] != "0",
        )
    except ValueError:
        return None


# Worker counts, from HiGHS's own "Solving MIP model with:" block
# (`HighsMipSolverData.cpp`):
#   Thread count 16 (of 32 threads). Using 8 max workers. Parallel search on
_THREADS_RE = re.compile(
    r"^\s+Thread count\s+(\d+)\s+\(of\s+(\d+)\s+threads\)\.\s+"
    r"Using\s+(\d+)\s+max workers"
)

# Model header emitted by HiGHS right after reading the MPS, e.g.
#   MIP fhnw-sq2 has 91 rows; 650 cols; 1968 nonzeros; 650 integer variables (625 binary)
# Used to classify instances into BP / IP / MBP / MIP (Local-MIP §6.1.1).
_MODEL_HEADER_RE = re.compile(
    r"^MIP\s+\S+\s+has\s+(\d+)\s+rows;\s+(\d+)\s+cols;\s+(\d+)\s+nonzeros;"
    r"\s+(\d+)\s+integer variables\s*\((\d+)\s*binary\)"
)


def _parse_compact_int(s: str) -> int:
    """Parse HiGHS compact integer format (e.g., '1.2k', '3.4M')."""
    s = s.strip()
    multipliers = {"k": 1_000, "M": 1_000_000, "G": 1_000_000_000}
    if s and s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(float(s))


def _parse_float_or_inf(s: str) -> float:
    """Parse a float, handling '-inf', 'inf', 'Large'."""
    s = s.strip()
    if s == "-inf" or s == "-1e+999":
        return float("-inf")
    if s == "inf" or s == "1e+999" or s == "Large":
        return float("inf")
    try:
        return float(s)
    except ValueError:
        return float("inf")


# Source codes that indicate an incumbent update (new feasible solution)
_INCUMBENT_SOURCES = set("ABCDFGHIJLMPRSTUXYZzlup")


def parse_log(log_text: str) -> SolveResult:
    """Parse HiGHS stdout log text and return structured result."""
    result = SolveResult()
    # `[HeurSol]` indices not yet bound to a `[Heur]` line, per heuristic
    # name.  A dispatch's offers all precede the `[Heur]` line that closes
    # it and follow the previous one for that name, so binding on that line
    # is exact — and, unlike zipping the two lists, it survives a dispatch
    # that offered nothing (which emits `[Heur]` and no `[HeurSol]` at all).
    pending_heursol: dict[str, list[int]] = {}

    for line in log_text.splitlines():
        # Try MIP log data line
        m = _LOG_LINE_RE.match(line)
        if m:
            src = m.group(1)
            nodes = _parse_compact_int(m.group(2))
            best_bound_str = m.group(6)
            best_sol_str = m.group(7)
            time_s = float(m.group(13))

            best_bound = _parse_float_or_inf(best_bound_str)
            best_sol = _parse_float_or_inf(best_sol_str)

            src_stripped = src.strip()
            # Space-source lines are periodic status updates, not events — but
            # when the model is solved by presolve (empty B&B, no heuristic ever
            # fires), the first and only log line has a space source while
            # BestSol already holds the presolve-found optimal.  Allow a
            # space-source line to seed the first incumbent so that these
            # instances are not misclassified as infeasible.
            is_event = src_stripped in _INCUMBENT_SOURCES
            is_presolve_seed = (
                not src_stripped
                and not result.incumbents
                and best_sol not in (float("inf"), float("-inf"))
            )
            # Record if objective improved (or first entry)
            if (is_event or is_presolve_seed) and best_sol not in (
                float("inf"),
                float("-inf"),
            ):
                prev_obj = (
                    result.incumbents[-1].objective if result.incumbents else None
                )
                if prev_obj is None or best_sol != prev_obj:
                    result.incumbents.append(
                        Incumbent(
                            time=time_s,
                            objective=best_sol,
                            source=src_stripped or "P",  # 'P' = presolve
                            nodes=nodes,
                            dual_bound=best_bound,
                        )
                    )
            continue

        # Runner marker.  Checked before the report patterns because a killed
        # run has no report block for those to match.
        m = _KILLED_RE.match(line)
        if m:
            result.killed = True
            result.killed_after = float(m.group(1))
            continue

        # Solving report lines
        m = _STATUS_RE.match(line)
        if m:
            result.status = m.group(1).strip()
            continue
        m = _PRIMAL_RE.match(line)
        if m:
            result.primal_bound = _parse_float_or_inf(m.group(1))
            continue
        m = _DUAL_RE.match(line)
        if m:
            result.dual_bound = _parse_float_or_inf(m.group(1))
            continue
        m = _GAP_RE.match(line)
        if m:
            gap_str = m.group(1).strip()
            # Parse "0% (tolerance: 0.01%)" or "5.85%" or "inf" or "Large"
            # Extract leading number before first '%'
            gap_match = re.match(r"([\d.]+)%", gap_str)
            if gap_match:
                result.gap = float(gap_match.group(1)) / 100.0
            elif "Large" in gap_str or gap_str == "inf":
                result.gap = float("inf")
            else:
                result.gap = _parse_float_or_inf(gap_str)
            continue
        m = _PD_RE.match(line)
        if m:
            result.pd_integral = float(m.group(1))
            continue
        m = _TIMING_RE.match(line)
        if m:
            result.solve_time = float(m.group(1))
            continue
        m = _NODES_RE.match(line)
        if m:
            result.nodes = int(m.group(1))
            continue
        m = _LPITERS_RE.match(line)
        if m:
            result.lp_iterations = int(m.group(1))
            continue

        # Sequential per-heuristic calibration line
        m = _SEQUENTIAL_RE.match(line)
        if m:
            result.sequential_samples.append(
                SequentialSample(
                    heuristic=m.group(1),
                    effort=int(m.group(2)),
                    wall_ms=float(m.group(3)),
                    effort_per_ms=float(m.group(4)),
                )
            )
            continue

        # Per-offered-solution instrumentation (key=value, not positional).
        heursol = _parse_heursol(line)
        if heursol is not None:
            pending_heursol.setdefault(heursol.name, []).append(
                len(result.heursol_samples)
            )
            result.heursol_samples.append(heursol)
            continue

        # Per-heuristic instrumentation (key=value, not positional).
        heur = _parse_heur(line)
        if heur is not None:
            heur_index = len(result.heuristic_samples)
            for i in pending_heursol.pop(heur.name, []):
                result.heursol_samples[i].heur_index = heur_index
            result.heuristic_samples.append(heur)
            continue

        # Worker counts (first occurrence wins; one block per solve).
        if result.thread_count is None:
            m = _THREADS_RE.match(line)
            if m:
                result.thread_count = int(m.group(1))
                result.hardware_threads = int(m.group(2))
                result.max_workers = int(m.group(3))
                continue

        # Model header line (first occurrence wins; HiGHS prints it once).
        if result.num_rows is None:
            m = _MODEL_HEADER_RE.match(line)
            if m:
                result.num_rows = int(m.group(1))
                result.num_cols = int(m.group(2))
                result.num_nonzeros = int(m.group(3))
                result.num_integer = int(m.group(4))
                result.num_binary = int(m.group(5))
                continue

    # A killed run never printed its Solving report, so `status` is empty here.
    # Leaving it empty makes the run indistinguishable from an unparsed log in
    # every status tally; naming it keeps the kill visible in reports.  Guarded
    # rather than unconditional so a log that somehow carries both a report and
    # a marker keeps HiGHS's own word for what happened.
    if result.killed and not result.status:
        result.status = "Killed (timeout)"

    # Sanity check: if the Solving report says there is a finite primal bound
    # but no incumbents were recorded, a source code is missing from
    # _INCUMBENT_SOURCES.  Raise early so the gap is caught in tests rather
    # than silently producing wrong T1st / primal_integral values.
    if math.isfinite(result.primal_bound) and not result.incumbents:
        warnings.warn(
            f"primal_bound={result.primal_bound} but incumbents is empty — "
            "a source code may be missing from _INCUMBENT_SOURCES",
            stacklevel=2,
        )

    return result


def parse_log_file(path: str) -> SolveResult:
    """Parse a HiGHS log file."""
    with open(path) as f:
        return parse_log(f.read())
