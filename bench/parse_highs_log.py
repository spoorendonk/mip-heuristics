"""Parse HiGHS MIP solver log output to extract benchmark metrics."""

from __future__ import annotations

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
    dispatch.  Used by `bench/check_effort_drift.py` to calibrate
    `kWeight*` (see issue #71), which considers only the four presolve
    heuristics.
    """

    heuristic: str  # fj, fpr, local_mip, scylla, fpr_lp
    effort: int
    wall_ms: float
    effort_per_ms: float


@dataclass
class HeuristicSample:
    """A single `[Heur]` cannibalization observation (issue #95).

    Emitted by `EffortLedger::book` alongside the legacy `[Sequential]`
    line, once per presolve-chain heuristic per solve and once per
    dive-time `fpr_lp` dispatch.  Carries what `[Sequential]` cannot:
    *when* the heuristic ran, on the solver's own clock, so its window can
    be placed against `[Root] lp_time_s`; which side of the patch boundary
    it ran on (`phase`); and whether it produced anything (`found`).
    """

    name: str  # fj, fpr, local_mip, scylla, fpr_lp
    phase: str  # presolve | dive
    start_s: float
    end_s: float
    effort: int
    wall_ms: float
    effort_per_ms: float
    found: bool


@dataclass
class NativeCounters:
    """The `[Native]` line: HiGHS's own heuristic activity for one solve.

    `rens` / `rins` / `rcfix` count invocations of upstream's RENS, RINS
    and root-reduced-cost heuristics across the whole solve (root node and
    B&B dive alike).  `rens_root` is the root-site subset of `rens`: the
    root gate is the one a presolve-found incumbent closes, so a
    suppressed root RENS is the cannibalization signal, and the merged
    total can hold steady while it vanishes.

    `heur_lp_iters` and `total_lp_iters` are upstream's own fields, but
    they are **shared**, not purely native: `EffortLedger::charge_dive`
    adds to both so `fpr_lp` competes with RENS/RINS for one envelope.
    `fpr_lp_lp_iters` is exactly what our dive heuristic contributed to
    each of them — subtract it from either to get HiGHS's own LP work
    (`native_heur_lp_iters`, `native_total_lp_iters`).  Without that,
    comparing `suite=off` against `suite=all` reads our self-charge as a
    jump in native heuristic activity (on flugpl seed 1: 169 vs 1294 with
    identical rens/rins/rcfix counts), which is the exact confound this
    line exists to eliminate.
    """

    rens: int
    rins: int
    rcfix: int
    heur_lp_iters: int
    total_lp_iters: int
    # Defaults for direct construction only: `_NATIVE_RE` requires every
    # field, so a `[Native]` line without these two does not parse at all.
    rens_root: int = 0
    fpr_lp_lp_iters: int = 0

    @property
    def native_heur_lp_iters(self) -> int:
        """`heur_lp_iters` with our own dive heuristic's charge removed."""
        return self.heur_lp_iters - self.fpr_lp_lp_iters

    @property
    def native_total_lp_iters(self) -> int:
        """`total_lp_iters` with our own dive heuristic's charge removed.

        `charge_dive` adds the same value to both upstream counters, so
        this subtraction is the companion to `native_heur_lp_iters`.
        """
        return self.total_lp_iters - self.fpr_lp_lp_iters


@dataclass
class RootTiming:
    """The `[Root]` line: when the root LP started, and what preceded it.

    `lp_time_s` is elapsed solve seconds at the start of the first root LP
    solve, or negative when the root LP was never reached (presolve solved
    the model, or a limit fired first).  `presolve_heur_s` is the wall time
    the custom presolve chain spent before it.

    On an instance that restarts, HiGHS re-runs both the presolve chain and
    the root node, so `presolve_heur_s` accumulates over every restart while
    `lp_time_s` pins the first root LP.  `presolve_heur_s > lp_time_s` is
    therefore expected there, not a contradiction.
    """

    lp_time_s: float
    presolve_heur_s: float


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
    incumbents: list[Incumbent] = field(default_factory=list)
    sequential_samples: list[SequentialSample] = field(default_factory=list)
    heuristic_samples: list[HeuristicSample] = field(default_factory=list)
    # Both None for a log produced before issue #95, or by any run below
    # log_dev_level=3.
    native: NativeCounters | None = None
    root: RootTiming | None = None

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

    @property
    def time_to_first_feasible(self) -> float | None:
        """Time when the first feasible solution was found."""
        if self.incumbents:
            return self.incumbents[0].time
        return None

    @property
    def time_to_root_lp(self) -> float | None:
        """Elapsed seconds when the first root LP solve started.

        None when the log carries no `[Root]` line, or when the root LP was
        never reached (sentinel negative value).
        """
        if self.root is None or self.root.lp_time_s < 0.0:
            return None
        return self.root.lp_time_s

    @property
    def heuristic_wall_fraction(self) -> float | None:
        """Share of total solve wall time spent inside our heuristics.

        The cannibalization headline number: the sum of every `[Heur]`
        window (presolve chain and B&B dive alike) over the solve time
        HiGHS reports.

        Absent `[Heur]` lines mean two different things and must not be
        conflated: a `suite=off` baseline ran no heuristics and its true
        value is **0.0**, while a log predating issue #95 (or below
        log_dev_level=3) simply cannot say.  The presence of a `[Native]`
        line identifies the former — it is emitted unconditionally on any
        instrumented run, `suite=off` included — so the baseline row keeps
        a real number instead of being silently dropped by whatever
        filters None.  None still means "unknown": no instrumentation, or
        no `Timing` line to divide by.
        """
        if self.solve_time <= 0.0:
            return None
        if not self.heuristic_samples:
            return 0.0 if self.native is not None else None
        total_ms = sum(h.wall_ms for h in self.heuristic_samples)
        return total_ms / 1000.0 / self.solve_time

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

# [Sequential] per-heuristic calibration line emitted from
# src/effort_ledger.cpp `EffortLedger::book` (issue #71):
#   [Sequential] heur=fpr effort=12345 wall_ms=67.8 effort_per_ms=182
# There is one line per heuristic per solve, feeding
# `bench/check_effort_drift.py` to calibrate `kWeight*`.
# `wall_ms` takes an optional sign here and in `_HEUR_RE`: the ledger
# times against HiGHS's own solver clock, which bottoms out in
# `high_resolution_clock` (== non-monotonic `system_clock` on libstdc++),
# so a wall-clock step can produce a negative window.  Rare, but a pattern
# without the sign drops the sample instead of surfacing the artefact.
_SEQUENTIAL_RE = re.compile(
    r"^\s*\[Sequential\] heur=(\S+) effort=(\d+) wall_ms=(-?[\d.]+) effort_per_ms=([\d.]+)"
)

# Cannibalization instrumentation, all three from issue #95 and all three
# at log_dev_level=3.  `[Heur]` is emitted next to `[Sequential]` by
# `EffortLedger::book`; `[Native]` and `[Root]` once per solve by
# `heuristics::log_solve_summary` in src/mode_dispatch.cpp:
#   [Heur] name=fj phase=presolve start_s=0.412 end_s=1.077 effort=8388608 \
#          wall_ms=665.2 effort_per_ms=12610.1 found=1
#   [Native] rens=3 rens_root=1 rins=7 rcfix=1 heur_lp_iters=48211 \
#            total_lp_iters=193044 fpr_lp_lp_iters=1125
#   [Root] lp_time_s=1.402 presolve_heur_s=2.118
# `lp_time_s` takes an optional sign: -1 is the "root LP never reached"
# sentinel, and a pattern without it would silently skip the line.
_HEUR_RE = re.compile(
    r"^\s*\[Heur\] name=(\S+) phase=(\S+) start_s=([\d.]+) end_s=([\d.]+) "
    r"effort=(\d+) wall_ms=(-?[\d.]+) effort_per_ms=([\d.]+) found=(\d+)"
)
_NATIVE_RE = re.compile(
    r"^\s*\[Native\] rens=(\d+) rens_root=(\d+) rins=(\d+) rcfix=(\d+) "
    r"heur_lp_iters=(-?\d+) total_lp_iters=(-?\d+) fpr_lp_lp_iters=(-?\d+)"
)
_ROOT_RE = re.compile(
    r"^\s*\[Root\] lp_time_s=(-?[\d.]+) presolve_heur_s=(-?[\d.]+)"
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
            if is_event or is_presolve_seed:
                # Record if objective improved (or first entry)
                if best_sol != float("inf") and best_sol != float("-inf"):
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

        # Cannibalization instrumentation (issue #95).
        m = _HEUR_RE.match(line)
        if m:
            result.heuristic_samples.append(
                HeuristicSample(
                    name=m.group(1),
                    phase=m.group(2),
                    start_s=float(m.group(3)),
                    end_s=float(m.group(4)),
                    effort=int(m.group(5)),
                    wall_ms=float(m.group(6)),
                    effort_per_ms=float(m.group(7)),
                    found=m.group(8) != "0",
                )
            )
            continue
        m = _NATIVE_RE.match(line)
        if m:
            # Last occurrence wins.  One per solve in practice, but a log
            # concatenating several runs should report the final state
            # rather than the first.
            result.native = NativeCounters(
                rens=int(m.group(1)),
                rens_root=int(m.group(2)),
                rins=int(m.group(3)),
                rcfix=int(m.group(4)),
                heur_lp_iters=int(m.group(5)),
                total_lp_iters=int(m.group(6)),
                fpr_lp_lp_iters=int(m.group(7)),
            )
            continue
        m = _ROOT_RE.match(line)
        if m:
            result.root = RootTiming(
                lp_time_s=float(m.group(1)),
                presolve_heur_s=float(m.group(2)),
            )
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
