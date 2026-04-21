"""Parse HiGHS MIP solver log output to extract benchmark metrics."""

from __future__ import annotations

import re
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
class EffortSample:
    """A single portfolio arm effort/wall-clock observation."""

    arm: str
    effort: int
    wall_ms: float
    effort_per_ms: float
    reward: int | None = None  # None for legacy logs without a reward= field
    # Log tag identifying which bandit emitted the sample. "Portfolio" is the
    # presolve bandit (src/portfolio.cpp); "FprLpPortfolio" is the B&B-dive
    # bandit (src/fpr_lp.cpp). Defaults to "Portfolio" for backward
    # compatibility with historical logs that only had one bandit.
    portfolio_tag: str = "Portfolio"


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
    incumbents: list[Incumbent] = field(default_factory=list)
    effort_samples: list[EffortSample] = field(default_factory=list)

    @property
    def time_to_first_feasible(self) -> float | None:
        """Time when the first feasible solution was found."""
        if self.incumbents:
            return self.incumbents[0].time
        return None

    def primal_gap_at(
        self, time_cutoff: float, best_known: float | None = None
    ) -> float | None:
        """Primal gap at a given time cutoff.

        If best_known is provided, gap = (obj - best_known) / max(|best_known|, 1).
        Otherwise, uses dual bound at the time of the incumbent.
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
        denom = max(abs(ref), 1.0)
        return abs(last_inc.objective - ref) / denom

    def primal_gap_curve(
        self, best_known: float | None = None
    ) -> list[tuple[float, float]]:
        """Return (time, gap) points for primal integral computation."""
        points = []
        for inc in self.incumbents:
            ref = best_known if best_known is not None else inc.dual_bound
            denom = max(abs(ref), 1.0)
            gap = abs(inc.objective - ref) / denom
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
            return float("inf")
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

# Portfolio effort calibration line, emitted by both bandits via the shared
# log_bandit_arm helper in src/bandit_runner.h:
#   [Portfolio] arm=FprDfsLocks2 effort=123456 wall_ms=45.2 effort_per_ms=2731 reward=3
#   [FprLpPortfolio] arm=lp_center effort=5678 wall_ms=10.2 effort_per_ms=557 reward=1
# `[Portfolio]` is the presolve bandit (src/portfolio.cpp); `[FprLpPortfolio]`
# is the B&B-dive bandit (src/fpr_lp.cpp). Both tags share an identical line
# layout so a single regex captures them and a separate tag group records
# which bandit emitted the sample.
# The `reward=<N>` suffix was added in the bandit-dispatch consolidation
# (issue #68); it is optional to keep the parser back-compatible with
# historical logs predating that change.
_EFFORT_RE = re.compile(
    r"^\s*\[(Portfolio|FprLpPortfolio)\] arm=(\S+) effort=(\d+) wall_ms=([\d.]+) effort_per_ms=([\d.]+)"
    r"(?: reward=(\d+))?"
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
_INCUMBENT_SOURCES = set("BCFHIJLPRSTUXYZzlup")


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

            if src.strip() and src.strip() in _INCUMBENT_SOURCES:
                # This line has a new incumbent — only record if objective improved
                if best_sol != float("inf") and best_sol != float("-inf"):
                    prev_obj = (
                        result.incumbents[-1].objective if result.incumbents else None
                    )
                    if prev_obj is None or best_sol != prev_obj:
                        result.incumbents.append(
                            Incumbent(
                                time=time_s,
                                objective=best_sol,
                                source=src.strip(),
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

        # Portfolio effort calibration line
        m = _EFFORT_RE.match(line)
        if m:
            reward = int(m.group(6)) if m.group(6) is not None else None
            result.effort_samples.append(
                EffortSample(
                    arm=m.group(2),
                    effort=int(m.group(3)),
                    wall_ms=float(m.group(4)),
                    effort_per_ms=float(m.group(5)),
                    reward=reward,
                    portfolio_tag=m.group(1),
                )
            )
            continue

    return result


def parse_log_file(path: str) -> SolveResult:
    """Parse a HiGHS log file."""
    with open(path) as f:
        return parse_log(f.read())
