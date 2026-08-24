#!/usr/bin/env python3
"""Evaluate one 8-parameter presolve heuristic configuration on one instance.

`bench/run_benchmark.py` is config-**name** based: a name maps to one
`mip_heuristic_suite` value and one results directory, every heuristic runs at
its shipped default, and the output is a tree to be analysed later.  A
configurator needs the opposite shape — a *parameter vector* in, a *single
scalar* out, one instance at a time — so this module is that shape and nothing
else.  It is the single definition of what a configuration means for the whole
tuning stage (issue #106, deliverable 5; driven by #107), which is why it is
tracked tooling with tests rather than a scratch script.

The eight parameters
--------------------
Four efforts (`fj`, `fpr`, `local_mip`, `scylla`), each a double in [0, 1]
sizing that heuristic's presolve budget, and four stall thresholds, each a
non-negative integer in effort units per matrix nonzero.  **Effort 0 means the
heuristic does not run**, so the fifteen non-empty heuristic subsets plus `off`
are exactly the zero-patterns of the four efforts; inclusion is not a separate
dimension.  `suite_value` performs that reduction, naming in `mip_heuristic_suite`
exactly the heuristics whose effort is strictly positive and emitting `off` when
none is.  **That mapping is the only place the reduction is true**, and not
merely for tidiness: `mip_heuristic_fpr_effort = 0` is *not* equivalent to
omitting `fpr` from the suite, because the suite value is what gates the
dive-time `fpr_lp` (through `heuristics::effective_flags`) while the effort
option is never read by `fpr_lp` at all — it draws from the separate
`mip_heuristic_effort` LP-iteration envelope.  A presolve-only screen cannot see
that difference; #107's full-limit confirmation of the same configuration can,
and would be measuring a heuristic the parameter vector said was off.

A stall threshold of 0 means "no staleness gate at all", not "give up
immediately" — see `docs/PARAMETERS.md` and the option records in
`third_party/highs_patch/apply_patch.cmake`.

Sign convention
---------------
irace minimises what the target runner prints, so this prints a **cost**:

    cost = gap + lambda * tau

where `gap` is the primal gap of the presolve-exit incumbent (capped at 1, and
replaced by `--no-solution-penalty` when the run found nothing) and `tau` is the
heuristics' own wall time in **seconds**.  Lower is better.  #107 states the
objective as `gap_improvement - lambda * tau`, to be maximised, with
`gap_improvement` measured against a no-solution baseline of `gap = 1`; the two
are the same ranking:

    cost = 1 - (gap_improvement - lambda * tau),   gap_improvement = 1 - gap

The additive 1 is identical for every configuration on a given instance, so it
cancels out of every paired comparison irace makes.  Note that `gap_improvement`
is therefore **not** floored at 0: `--no-solution-penalty` exceeds the gap cap by
construction (see `check_penalty_dominates`), so a run that found nothing records
a negative improvement — which is the point, since at the baseline value of 1.0
finding nothing was the cheapest outcome in the space.  `lambda` is a parameter of
this runner (`--lambda`), not a constant: #107 runs the search at two or three
values around `g(0)/T` ~ 0.0017 per second at the campaign's 600 s limit, and
keeps the resulting family of configurations.

Cost is the **heuristics' own** `[Heur] wall_ms`, summed over the presolve
chain — not total presolve wall time and not total solve time.  HiGHS's own
presolve dominates on large models and is not ours to spend.  That line is
emitted only at `log_dev_level=3`, which this runner therefore sets; the same
level makes HiGHS's FeasibilityJump log once per weight bump, so `tau` carries
some logging overhead in the FJ window.  That overhead is present for every
configuration that runs FJ and is part of the measurement, not a defect, but it
means `tau` is not directly comparable to a `[Heur] wall_ms` taken from a
headline-timing run.

Quality is the primal gap against `bench/miplib2017-v36.solu`, resolved through
`analyze_results.resolve_reference` so a run that beats the published objective
is not punished for it.  An instance the solution file marks `=inf=` / `=unbd=`
carries no usable reference and is **refused** (exit 2) rather than scored: the
gap would fall back to the run's own primal and enter the objective as a
self-referential zero.  Such an instance must be kept out of the training list.
An instance with a usable reference but no solution is scored by an explicit
penalty and never dropped.

Reproducibility, and what is *not* reproducible
-----------------------------------------------
Every run writes `<run-dir>/<instance>/<tag>.opts` (the exact options file the
solver was given), `<tag>.log.gz` (its full output, compressed because
`log_dev_level=3` runs to millions of lines) and `<tag>.json` (the parsed
scoring record).  `tag` defaults to a hash of the parameter vector, instance and
seed, so the same evaluation re-runs into the same three files, and any recorded
run can be replayed from its `.opts` alone.

The *definition* of a run is therefore reproducible.  Its **score is not**, and
the gap is not small.  At unpinned `threads` the presolve chain runs N workers
racing to submit, and which one wins the pool varies between runs of an
identical options file: five consecutive presolve-only runs of flugpl at
`random_seed = 0` gave 1202400 / 1240500 / 1240500 / 1201500 / 1201500, a 3.2%
spread on the exact quantity the quality metric scores.  Pinned to `threads = 1`
the same five runs gave 1240500 every time.  (Measured by Track A of #106
against the patched build; it is structural, not a flugpl quirk.)

`threads` is nevertheless left unpinned by default, deliberately, and the
reasoning matters more than the default:

* That spread is **not measurement error, it is the variance of the thing being
  configured**.  The deployed solver runs multi-worker, so the quantity worth
  maximising is the expected quality at the deployment worker count.  A
  configuration that only looks good when one particular worker wins the race is
  genuinely worse than one that produces a good solution reliably, and averaging
  over evaluations is how a racing configurator sees that difference.
* Pinning does not merely narrow the distribution, **it moves it**.  Five
  repeats of the shipped defaults on flugpl at `random_seed = 0`, scored by this
  runner: unpinned gave costs 0.0332 / 0.0333 / 0.0333 / **0.0016** / 0.0333 —
  one run in five reached the optimum — while `threads = 1` gave 0.03253 /
  0.03254 / 0.03253 / 0.03255 / 0.03253, reliably and never the optimum.  The
  single-worker regime is *steadier and worse*: it does not sample the case the
  multi-worker chain occasionally wins.  Suppressing that is the wrong trade for
  a feasibility campaign whose headline capability is cracking an instance
  nothing cracked before (#113), because the configuration that does it would be
  scored as though it never could.  (One instance, one seed, five repeats —
  enough to show the effect exists, not to size it.)
* Pinning `threads = 1` would estimate a **different expectation** precisely,
  rather than the right one noisily — and precision about the wrong quantity
  does not improve with budget.  It also changes what the parameter vector
  *means*, though not in the blunt way an earlier version of this note claimed.
  Track A worked the algebra out of `make_budget` and measured it (the table and
  the numbers are in `docs/PARAMETERS.md`, "These options do not mean the same
  thing at every worker count"): for the three whole-dispatch heuristics both
  aggregates — total spend and the runner-level stall gate — are **N-invariant**,
  and only the per-worker slicing moves; FJ is the exact mirror, per-worker
  quantities invariant and the dispatch total growing with the pool.  So the
  cross-N distortion is not a uniform rescale but a **reallocation between
  heuristics**: on p0548 at the shipped defaults, charged presolve effort goes
  16.9M -> 18.3M (1.08x) for LocalMIP and 500k -> 6.0M (12x) for FJ between N=1
  and N=8, moving FJ's share against LocalMIP from 1:34 to 1:3 at identical
  option values.  Racing might partly absorb a uniform rescale; a reallocation
  between heuristics it cannot, and nothing in the score reveals it.  That is
  the bias, and it is why the search runs at the deployment worker count.
* One quantity *does* transfer, which is what makes the stall ranges in
  `bench/irace/parameters.txt` meaningful across machines: the inert-region
  boundary `stall >= 81920 * effort` is N-independent for all four heuristics
  (the N factors cancel for FJ and are absent for the other three).  "This
  configuration's gate is inert" therefore means the same thing at every worker
  count, even though the effort it is inert relative to does not.
* #113 draws the same line for the same reason: trajectory characterisation runs
  at `threads=1` because the effort timeline has to be reproducible, while the
  instance-filtering pass runs at the normal worker count "since that is the
  regime the search runs in".  This runner is the second case.
* It is also the project's standing rule (see CLAUDE.md's benchmarking note):
  pinning `threads` collapses each heuristic to a single worker, which is the
  reproducible configuration and the wrong one for measuring what the parallel
  chain does.

So the lever against run-to-run noise here is **more (instance, seed) pairs**,
not fewer workers — sigma falls as 1/sqrt(instances x seeds), which is exactly
why #113 sizes the tuning set at 75-100 instead of 25.  `--threads 1` remains
available and is the right setting for a bit-reproducible re-derivation or a
trajectory trace; a run that used it records `threads` in its `.opts` and its
observed worker count in the `.json`, so the two regimes can never be silently
mixed in one results tree.

Because of that reallocation effect, **the worker count is part of the result,
not part of the environment**.  A vector tuned at one worker count and deployed
at another is not a configuration with some noise on it, it is a configuration
for a different machine, and nothing downstream of the score can tell.  Every
`.json` therefore carries `thread_count`, read from HiGHS's own
`Thread count N (of M threads)` line — the same quantity `bench/make_archive.py`
derives into `run.workers_observed` and warns about when a tree mixes two values,
deliberately the same channel rather than a second one.  A run whose log carries
no such line warns, because an unattributable stamp is worse than a wrong one:
it cannot even be checked.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_results import (
    CONTRADICTED_REFERENCE_TAGS,
    USABLE_REFERENCE_TAGS,
    parse_solu_file,
    resolve_reference,
)
from make_archive import PATCH_MARKER
from parse_highs_log import SolveResult, parse_log
from run_benchmark import (
    find_ignored_config_warning,
    find_instance_file,
    resolve_data_dir,
    write_options_file,
)

# Chain order, as `kChain` in `src/mode_dispatch.cpp` spells it.  The suite
# value lists heuristics in this order so one subset has exactly one spelling,
# matching `run_benchmark.CONFIG_SUITES`.
HEURISTICS: tuple[str, ...] = ("fj", "fpr", "local_mip", "scylla")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOLU = os.path.join(BENCH_DIR, "miplib2017-v36.solu")

# Gap charged to a run that produced no solution at all.
#
# It must be **strictly greater than the worst score any found solution can
# reach**, which is `gap_cap + lambda * tau_max` = 1.0 + lambda x the per-run
# time cap.  At 1.0 — the old value, chosen to match the project's gap sentinel
# — that inequality fails by exactly the cost term: a configuration that found
# a bad solution and spent time on it scored *worse* than one that found
# nothing, so the search was rewarded for producing nothing at all.  1.1 is the
# floor at lambda = 1/600 and a 60 s cap; 2.0 leaves room for the larger lambda
# values #107 sweeps and for a longer cap, and `check_penalty_dominates` checks
# the inequality at run time rather than trusting this comment to stay true.
DEFAULT_NO_SOLUTION_PENALTY = 2.0

# Cost weight, per second of heuristic wall time.  Derived, not fitted: the
# primal integral is int g(t) dt, so spending tau extra seconds shifts the
# trajectory right by tau at a cost of about tau * g(0), i.e. ~g(0)/T per second
# — roughly 0.17% of the integral per extra presolve second at the campaign's
# 600 s limit.  #107 sweeps two or three values around it.
DEFAULT_LAMBDA = 1.0 / 600.0

# Per-run solver time limit.  A cap is mandatory rather than advisory: HiGHS's
# own presolve is unbounded on some models, and one PLATO instance spends an
# entire 60 s limit inside it, so an uncapped screen spends its budget on
# something no parameter here controls.
DEFAULT_TIME_LIMIT = 60.0

EXIT_REFUSED = 2


class Refusal(Exception):
    """The evaluation cannot yield a meaningful number, and says why.

    Raised instead of returning a cost, because every alternative is worse: a
    sentinel cost enters the race as data, and a silently dropped instance
    breaks the pairing racing depends on.
    """


@dataclass(frozen=True)
class Parameters:
    """The eight-dimensional configuration point.

    `efforts` and `stalls` are keyed by `HEURISTICS`; every key must be present
    so that the vector a run recorded is complete rather than partly implied by
    whatever the binary's defaults happened to be that week.
    """

    efforts: dict[str, float]
    stalls: dict[str, int]

    def __post_init__(self) -> None:
        for name in HEURISTICS:
            if name not in self.efforts:
                raise ValueError(f"missing effort for {name!r}")
            if name not in self.stalls:
                raise ValueError(f"missing stall threshold for {name!r}")
        for name, value in self.efforts.items():
            if name not in HEURISTICS:
                raise ValueError(f"unknown heuristic {name!r}")
            if not (0.0 <= value <= 1.0) or not math.isfinite(value):
                raise ValueError(f"{name} effort {value} outside [0.0, 1.0]")
        for name, value in self.stalls.items():
            if name not in HEURISTICS:
                raise ValueError(f"unknown heuristic {name!r}")
            if int(value) != value or value < 0:
                raise ValueError(f"{name} stall {value} is not a non-negative integer")

    @property
    def enabled(self) -> tuple[str, ...]:
        """Heuristics with strictly positive effort, in chain order."""
        return tuple(h for h in HEURISTICS if self.efforts[h] > 0.0)


def suite_value(params: Parameters) -> str:
    """The `mip_heuristic_suite` value implied by the zero-pattern.

    This is where "effort 0 means the heuristic does not run" becomes true in
    the sense #107 assumes, and it exists because the effort option alone does
    not make it true: `mip_heuristic_fpr_effort = 0` still leaves `fpr` named in
    the suite, and the suite value — not the effort — is what
    `heuristics::effective_flags` reads to gate the dive-time `fpr_lp`, which
    draws from the separate `mip_heuristic_effort` envelope and never reads the
    presolve effort at all.  Screening presolve-only hides that; the full-limit
    confirmation in #107 does not.

    `off` when no effort is positive, and that exact string is not cosmetic: the
    patch compares `mip_heuristic_suite == "off"` verbatim in two places, so a
    value that selected nothing without being that string would be a
    heuristic-free run that is *not* the vanilla-equivalent one.  An empty value
    or a trailing comma is worse still — an unrecognised token, which the
    dispatcher warns about and then fails *open* to all four heuristics.
    """
    enabled = params.enabled
    return ",".join(enabled) if enabled else "off"


def _format_effort(value: float) -> str:
    """Effort as HiGHS should read it, and as a diff should show it.

    `%.10g` keeps every digit irace can generate (it rounds candidates to
    `digits` decimals, 4 by default) without printing 0.30000000000000004.
    """
    return f"{value:.10g}"


def solver_options(
    params: Parameters,
    seed: int,
    *,
    threads: int | None = None,
    presolve_only: bool = True,
) -> dict[str, str]:
    """The full options file for one evaluation, in a stable order.

    All eight parameters are written even when a heuristic is disabled: the
    options file is the record of which point in the space was evaluated, and a
    record that omits the zeros cannot be told apart from one taken before the
    parameter existed.

    `threads` is left unset by default.  Pinning it collapses each heuristic to
    a single worker, which is the reproducible configuration (`threads=1` plus a
    fixed `random_seed`) but not the regime the search runs in; #113's
    trajectory traces want it, the screen does not.  The module docstring has
    the measured cost of that choice and why it is still the right one.
    """
    options: dict[str, str] = {"mip_heuristic_suite": suite_value(params)}
    # Effort 0 for FJ has to disable *both* FeasibilityJump call sites, and this
    # is the only option that reaches the other one.  At `suite=off` the patch
    # hands HiGHS's own standalone FJ back — deliberately, so that value is a
    # vanilla-equivalent ablation — and that native call site emits no `[Heur]`
    # line, because the patch's `heuristic_effort_used +=` inside HiGHS's
    # `feasibilityJump()` logs nothing.  The all-zero vector therefore banked
    # real FJ quality at tau = 0: free quality, zero measured cost, and `off`
    # beating configurations that found objectives 28x better (markshare2: 375
    # against 10512, scoring 1.000120 against 1.000000).  A search reachable
    # from the initial uniform sample with probability 1/16 would have reported
    # "disable all four heuristics" as its winner.
    #
    # `false` gates the native site at `off` and ours everywhere else, so it is
    # a no-op except at that corner — which is what keeps `off` a *scorable*
    # point of the space rather than one that has to be excluded from it.
    options["mip_heuristic_run_feasibility_jump"] = (
        "true" if params.efforts["fj"] > 0.0 else "false"
    )
    for name in HEURISTICS:
        options[f"mip_heuristic_{name}_effort"] = _format_effort(params.efforts[name])
    for name in HEURISTICS:
        options[f"mip_heuristic_{name}_stall"] = str(int(params.stalls[name]))
    if presolve_only:
        options["mip_heuristic_presolve_only"] = "true"
    # The `[Heur]` line the cost metric reads is `kVerbose`.  Without this the
    # run is silently free.
    options["log_dev_level"] = "3"
    options["random_seed"] = str(seed)
    if threads is not None:
        options["threads"] = str(threads)
    return options


def strip_instance_token(token: str) -> str:
    """One instance-file line reduced to the token it names.

    Removes anything from the first `#` onward, then surrounding whitespace.
    `bench/instances_tuning.txt` is generated with a `#` header block *and* a
    trailing `# <vanilla time-to-first-feasible>` on every instance line, so a
    line arrives as `comp21-2idx               # 0.60s`.

    This exists so that nothing in this stage depends on irace stripping those
    comments itself.  It plausibly does — but irace is not installed on the
    machine this was written on, so that is an unexecuted assumption guarding a
    ten-hour campaign: were it false, `os.path.basename` would yield `0.60s`,
    every one of the search's thousands of evaluations would refuse, and it
    would be discovered at hour ten.  Stripping here costs two lines and is
    correct either way, which is strictly better than a comment-free copy of
    the list that can drift from the list itself.

    A token that is empty or comment-only is a `Refusal`, not an empty name:
    the whole point is that a malformed instance line must be loud.
    """
    stripped = token.split("#", 1)[0].strip()
    if not stripped:
        raise Refusal(
            f"instance token {token!r} is empty or contains only a comment; "
            "an instance file line must name an instance"
        )
    return stripped


def instance_name(token: str) -> str:
    """The MIPLIB name behind an instance token.

    irace's instance file may hold bare names or full paths, with or without a
    trailing comment; the reference lookup needs the name either way.
    """
    base = os.path.basename(strip_instance_token(token))
    for ext in (".mps.gz", ".lp.gz", ".mps", ".lp", ".gz"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def reference_objective(name: str, solu_refs: dict[str, tuple[str, float | None]]):
    """Published reference objective for `name`, or a `Refusal`.

    Two refusals, both of which would otherwise produce a number that looks
    exactly like a real one:

    * `=inf=` / `=unbd=` — the solution file asserts no finite objective
      exists, so a primal gap against it is a category error.  #106 and #107
      both require these instances excluded.
    * no usable value at all — with a single run there are no other configs to
      build a virtual best from, so `resolve_reference` would hand back this
      run's own primal and every configuration would score gap 0.

    Neither is reachable from `bench/instances_plato.txt`: all 233 entries carry
    a published objective in the bundled solution file.  The refusals exist so
    that an instance list which grows past it fails loudly at the first
    evaluation rather than quietly contributing a constant.
    """
    entry = solu_refs.get(name)
    if entry is None:
        raise Refusal(
            f"{name}: no entry in the solution file, so no reference objective; "
            "remove it from the instance list or supply a --solu file that "
            "covers it"
        )
    tag, value = entry
    if tag in CONTRADICTED_REFERENCE_TAGS:
        raise Refusal(
            f"{name}: solution file tags it {tag}, which asserts no finite "
            "optimal objective exists — no primal gap against it is meaningful; "
            "remove it from the instance list"
        )
    if tag not in USABLE_REFERENCE_TAGS or value is None:
        raise Refusal(
            f"{name}: solution file entry {tag} carries no usable objective; "
            "remove it from the instance list or supply a --solu file that "
            "covers it"
        )
    return value


def presolve_objective(result: SolveResult) -> float | None:
    """The objective the run exited presolve with, or None if it found nothing.

    Prefers the Solving report's primal bound, which is HiGHS's own final word.
    A run the harness had to kill never printed that report, so the incumbent
    trajectory is the fallback — the same reasoning that makes a truncated log
    scorable in `run_benchmark.py`.
    """
    if math.isfinite(result.primal_bound):
        return result.primal_bound
    if result.incumbents:
        return result.incumbents[-1].objective
    return None


def primal_gap(objective: float, reference: float) -> float:
    """Relative primal gap, capped at 1.0.

    Same formula as `SolveResult.primal_gap_at`, which cannot be reused here:
    it reads the incumbent trajectory, and a presolve-only run is scored on its
    exit objective, which the Solving report carries even when no display line
    was ever printed.
    """
    denom = max(abs(reference), 1.0)
    return min(abs(objective - reference) / denom, 1.0)


def heuristic_wall_ms(result: SolveResult) -> float:
    """Total `[Heur] wall_ms` over the presolve chain, in milliseconds.

    Presolve-phase samples only.  A presolve-only run reaches no dive, so the
    filter is a no-op there and a guard everywhere else — `fpr_lp`'s dive
    dispatch draws from a different budget and is not what this cost axis
    prices.

    Individual samples are floored at zero: the ledger times against HiGHS's own
    solver clock, which is not monotonic, so a window can come out negative.
    Summing such a sample as-is would hand a configuration a discount.
    """
    return sum(
        max(s.wall_ms, 0.0) for s in result.heuristic_samples if s.phase == "presolve"
    )


def scalar_cost(gap: float, tau_s: float, cost_weight: float) -> float:
    """The minimised scalar: `gap + lambda * tau`.

    See the module docstring for why this is the negation of #107's
    `gap_improvement - lambda * tau` up to a per-instance constant.
    """
    return gap + cost_weight * tau_s


@dataclass
class Evaluation:
    """Everything one evaluation produced, as written to `<tag>.json`."""

    instance: str
    seed: int
    tag: str
    suite: str
    efforts: dict[str, float]
    stalls: dict[str, int]
    cost: float
    gap: float
    gap_improvement: float
    objective: float | None
    reference: float
    heuristic_wall_ms: float
    tau_s: float
    cost_weight: float
    no_solution: bool
    solve_time: float
    status: str
    killed: bool
    thread_count: int | None
    heuristics_traced: list[str] = field(default_factory=list)
    trace_missing: bool = False
    chain_truncated: list[str] = field(default_factory=list)
    log_path: str = ""
    opts_path: str = ""


def run_tag(
    params: Parameters,
    name: str,
    seed: int,
    *,
    presolve_only: bool = True,
    time_limit: float = DEFAULT_TIME_LIMIT,
    threads: int | None = None,
    cost_weight: float = DEFAULT_LAMBDA,
    no_solution_penalty: float = DEFAULT_NO_SOLUTION_PENALTY,
) -> str:
    """A short, deterministic name for one (parameters, instance, seed) point.

    Deterministic so that re-running an evaluation overwrites its own artifacts
    rather than accumulating near-duplicates, and so a `.json` record can be
    located from the parameter vector alone.  irace can override it with
    `--tag` to keep its own configuration and instance ids in the file names.
    """
    payload = "|".join(
        [name, str(seed), suite_value(params)]
        + [f"{h}={_format_effort(params.efforts[h])}" for h in HEURISTICS]
        + [f"{h}_stall={int(params.stalls[h])}" for h in HEURISTICS]
        # Only when it differs from the screen, so the tags a search produces
        # stay stable; a full-solve diagnostic of the same vector is a
        # different measurement and must not overwrite the screening run.
        + ([] if presolve_only else ["full-solve"])
        # Everything else that changes either the run or its score.  The first
        # two change what the solver did — the same vector at a 30 s and a 60 s
        # cap are different measurements, and so are one at N=1 and one
        # unpinned.  The second two change only the number, but the `.json`
        # records that number, so sharing a tag across two lambdas means the
        # second scoring silently overwrites the first.  #107 sweeps lambda by
        # construction, so that collision is the common case, not the corner.
        + [
            f"tl={time_limit:g}",
            f"threads={'auto' if threads is None else threads}",
            f"lambda={cost_weight:.10g}",
            f"penalty={no_solution_penalty:.10g}",
        ]
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def invoke_solver(
    binary: str,
    instance_file: str,
    opts_path: str,
    time_limit: float,
) -> tuple[str, int | None]:
    """Run the solver once.  Returns (combined output, exit code or None).

    None means the process had to be killed.  HiGHS checks its clock between
    work units, so a single long presolve or simplex solve can carry a run past
    `--time_limit` without ever returning to look; the grace window below is
    what bounds that, and whatever the solver streamed before the kill is kept
    and scored, exactly as `run_benchmark.py` does.
    """
    cmd = [
        binary,
        instance_file,
        "--time_limit",
        str(time_limit),
        "--options_file",
        opts_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=time_limit * 1.5 + 60,
        )
        output = proc.stdout
        if proc.stderr:
            output += "\n--- stderr ---\n" + proc.stderr
        return output, proc.returncode
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):  # defensive: text=True should preclude it
            partial = partial.decode(errors="replace")
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if stderr:
            partial += "\n--- stderr ---\n" + stderr
        kill_after = time_limit * 1.5 + 60
        partial += f"\n--- runner ---\nTIMEOUT: process killed after {kill_after}s\n"
        return partial, None


def score_output(
    output: str,
    params: Parameters,
    name: str,
    seed: int,
    reference: float,
    **kwargs,
) -> Evaluation:
    """`score_result` from raw output, for callers that have not parsed it.

    Split from the subprocess call so the whole scoring path — the metric, the
    penalty, the trace check — is testable against a synthetic log with no
    binary and no instance data.
    """
    return score_result(parse_log(output), params, name, seed, reference, **kwargs)


def score_result(
    result: SolveResult,
    params: Parameters,
    name: str,
    seed: int,
    reference: float,
    *,
    cost_weight: float,
    no_solution_penalty: float,
    tag: str,
    require_trace: bool,
) -> Evaluation:
    """Turn one parsed run into the scalar and the record behind it.

    Takes the parsed result rather than the text because a `log_dev_level=3` log
    runs to millions of lines on a large model, and parsing it twice per
    evaluation is minutes across a search.
    """
    objective = presolve_objective(result)
    no_solution = objective is None
    gap = no_solution_penalty if no_solution else primal_gap(objective, reference)
    wall_ms = heuristic_wall_ms(result)
    tau_s = wall_ms / 1000.0
    traced = sorted({s.name for s in result.heuristic_samples if s.phase == "presolve"})
    suite = suite_value(params)
    # A run that names heuristics but carries no `[Heur]` line has produced a
    # cost of zero for work it actually did, which is the cheapest possible
    # configuration and would win a race outright.  The realistic causes are an
    # unpatched binary, a `log_dev_level` that did not take, or a HiGHS bump
    # that moved the line — all campaign-wide, none visible in the scalar.
    #
    # It is only advisory here, because two legitimate runs look the same: one
    # killed inside HiGHS's own presolve, and one whose time limit expired
    # before the chain was reached.  Making it fatal by default would let a
    # single slow instance abort a ten-hour search.  `--require-trace` is the
    # pre-flight: run one evaluation with it on a cheap instance before
    # launching the configurator.
    trace_missing = suite != "off" and not traced and not result.killed
    if trace_missing:
        message = (
            f"no [Heur] presolve sample in the log for suite={suite!r}: the cost "
            "axis is reading zero heuristic time.  Check that the binary is "
            "patched, that log_dev_level=3 took effect, and that the [Heur] "
            "line still has the shape parse_highs_log expects"
        )
        if require_trace:
            raise Refusal(f"{name}: {message}")
        print(f"Warning: {name}: {message}", file=sys.stderr)
    # Enabled heuristics that never got a dispatch.  `run_sequential` runs the
    # chain in order — FJ, FPR, LocalMIP, Scylla — against a shared solver time
    # limit, so a generous head starves the tail and the parameter vector stops
    # describing the run: Scylla's effort and stall are recorded but never
    # exercised.  The deficit is *correlated* with the other three efforts, so
    # a search that could not see it would read the tail as "does not matter"
    # rather than as "did not run".
    #
    # Recorded, not fixed.  The search mostly lives below effort 1.0 and a
    # truncating configuration already pays for its own greed through tau; what
    # was missing was any way to measure how often it happens.  Partial absence
    # is the truncation signature — total absence is `trace_missing` above, a
    # different failure (no instrumentation at all), and the two stay distinct.
    chain_truncated = sorted(set(params.enabled) - set(traced)) if traced else []
    # The worker count is part of the result, not part of the environment: the
    # same vector reallocates effort between heuristics as N changes (see the
    # module docstring), so a run nobody can attribute to a worker count cannot
    # be compared to anything.  A stamp that is merely *wrong* would at least be
    # checkable against `make_archive.py`'s `workers_observed`; a missing one is
    # not, which is why its absence is said out loud rather than left as a null
    # in the record.
    if result.thread_count is None and not result.killed:
        print(
            f"Warning: {name}: no 'Thread count N (of M threads)' line in the "
            "log, so this run records no worker count and cannot be checked for "
            "regime mixing against the rest of the tree",
            file=sys.stderr,
        )
    return Evaluation(
        instance=name,
        seed=seed,
        tag=tag,
        suite=suite,
        efforts=dict(params.efforts),
        stalls=dict(params.stalls),
        cost=scalar_cost(gap, tau_s, cost_weight),
        gap=gap,
        gap_improvement=1.0 - gap,
        objective=objective,
        reference=reference,
        heuristic_wall_ms=wall_ms,
        tau_s=tau_s,
        cost_weight=cost_weight,
        no_solution=no_solution,
        solve_time=result.solve_time,
        status=result.status,
        killed=result.killed,
        thread_count=result.thread_count,
        heuristics_traced=traced,
        trace_missing=trace_missing,
        chain_truncated=chain_truncated,
    )


def check_penalty_dominates(
    no_solution_penalty: float, cost_weight: float, time_limit: float
) -> None:
    """Warn if "found nothing" can score better than "found something bad".

    The scalar is `gap + lambda * tau`, `gap` is capped at 1, and `tau` cannot
    exceed the per-run time cap, so every scorable found solution lands at or
    below `1 + lambda * time_limit`.  A no-solution penalty at or below that
    bound inverts the objective: producing nothing becomes the cheapest outcome
    available, and a racing configurator converges on the configuration that
    does least.  That is not hypothetical — it is what a penalty of 1.0 did.

    A warning rather than a refusal, because the three quantities are all
    campaign knobs and someone may deliberately want a lenient penalty for a
    diagnostic run; but loud, because nothing downstream of the scalar can see
    the inversion.
    """
    worst_found = 1.0 + cost_weight * time_limit
    if no_solution_penalty <= worst_found:
        print(
            f"Warning: --no-solution-penalty {no_solution_penalty:g} does not "
            f"exceed the worst score a found solution can reach "
            f"({worst_found:g} = gap cap 1 + lambda {cost_weight:g} x time limit "
            f"{time_limit:g}), so a run that finds nothing scores better than a "
            "run that finds something bad and the search is inverted",
            file=sys.stderr,
        )


def check_run_usable(
    output: str, returncode: int | None, name: str, log_path: str = ""
) -> None:
    """Refuse a run whose output cannot mean what its parameters say.

    Each of these exits 0-or-close and produces an ordinary-looking log, so
    nothing downstream can tell the resulting number apart from a good one:

    * a non-`kOk`/`kWarning` exit — which is what an unknown or out-of-range
      option in the options file gives, i.e. exactly what a binary predating
      the options this runner writes will do;
    * a missing patch marker — patched and unpatched builds of the same tag
      have identical version banners, so this is the only way to tell them
      apart, and an unpatched binary ignores every parameter here;
    * HiGHS's own warning that it ignored the suite value and failed open to
      all four heuristics.

    A killed run (`returncode is None`) is not a failure: it is a truncated but
    real measurement, and it is scored.

    `log_path` is named in each message because the log is already on disk by
    the time this runs, and it is gzipped — the diagnosis is a `zcat` away only
    if the message says where.
    """
    where = f"; see {log_path}" if log_path else ""
    if returncode is not None and returncode not in (0, 1):
        raise Refusal(
            f"{name}: solver exited {returncode} without solving; an unknown or "
            "out-of-range option in the options file is the usual cause — check "
            "the binary is built from a tree carrying mip_heuristic_presolve_only "
            f"and the four mip_heuristic_*_stall options{where}"
        )
    if PATCH_MARKER not in output:
        raise Refusal(
            f"{name}: the log carries no {PATCH_MARKER!r} line, so this binary is "
            f"not a patched build and ignored every parameter it was given{where}"
        )
    ignored = find_ignored_config_warning(output)
    if ignored is not None:
        raise Refusal(f"{name}: solver ignored its configuration: {ignored}{where}")


def evaluate(
    params: Parameters,
    instance: str,
    seed: int,
    *,
    binary: str,
    data_dir: str | None,
    time_limit: float,
    run_dir: str,
    solu_path: str,
    cost_weight: float,
    no_solution_penalty: float,
    threads: int | None,
    tag: str | None,
    require_trace: bool,
    presolve_only: bool = True,
) -> Evaluation:
    """Run one instance at one parameter vector and score it."""
    # Once, up front: the token is used twice — for the reference lookup and as
    # a candidate path — and a trailing comment has to be gone for both.  The
    # path branch would otherwise survive by accident (a commented path does not
    # exist, so it falls through to the name-based search) which is exactly the
    # kind of accident that stops holding when someone reorders the branch.
    instance = strip_instance_token(instance)
    name = instance_name(instance)
    check_penalty_dominates(no_solution_penalty, cost_weight, time_limit)
    solu_refs = parse_solu_file(solu_path)
    published = reference_objective(name, solu_refs)

    if os.path.exists(instance) and not os.path.isdir(instance):
        instance_file: str | None = instance
    else:
        instance_file = find_instance_file(name, resolve_data_dir(data_dir))
    if instance_file is None:
        raise Refusal(
            f"{name}: no .mps/.mps.gz found; pass --data-dir or set $MIPLIB_DIR"
        )

    tag = tag or run_tag(
        params,
        name,
        seed,
        presolve_only=presolve_only,
        time_limit=time_limit,
        threads=threads,
        cost_weight=cost_weight,
        no_solution_penalty=no_solution_penalty,
    )
    out_dir = os.path.join(run_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    opts_path = os.path.join(out_dir, f"{tag}.opts")
    log_path = os.path.join(out_dir, f"{tag}.log.gz")
    write_options_file(
        solver_options(params, seed, threads=threads, presolve_only=presolve_only),
        opts_path,
    )

    output, returncode = invoke_solver(binary, instance_file, opts_path, time_limit)
    # Compressed rather than plain, and whole rather than filtered.  The trace
    # this scores on is `log_dev_level=3`, which HiGHS's own FeasibilityJump
    # inflates by up to 750x — a single log runs to millions of lines, and a
    # few-thousand-evaluation search would otherwise leave tens of gigabytes
    # behind.  Compressing keeps every line rather than deciding now which ones
    # a later question will need.
    with gzip.open(log_path, "wt") as f:
        f.write(output)

    check_run_usable(output, returncode, name, log_path)

    result = parse_log(output)
    observed = [result.primal_bound] if math.isfinite(result.primal_bound) else []
    # Virtual best: an observed primal that beats the published objective
    # becomes the reference, so a configuration is never punished for finding
    # something better than the library knew about.  With one run there is at
    # most one observation, which makes this a min of two numbers rather than
    # the cross-config resolution `analyze_results` does — same rule, less data.
    reference = resolve_reference(published, observed)
    assert reference is not None  # published is not None, so neither is this

    evaluation = score_result(
        result,
        params,
        name,
        seed,
        reference,
        cost_weight=cost_weight,
        no_solution_penalty=no_solution_penalty,
        tag=tag,
        require_trace=require_trace,
    )
    evaluation.log_path = log_path
    evaluation.opts_path = opts_path
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(asdict(evaluation), f, indent=2, sort_keys=True)
        f.write("\n")
    return evaluation


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI.

    Switch spellings are a contract with `bench/irace/parameters.txt`, whose
    `switch` column has to reproduce them exactly.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate one presolve heuristic configuration on one "
        "instance and print the scalar cost to minimise.",
        epilog="Prints one number to stdout: gap + lambda * heuristic seconds. "
        "Lower is better.",
    )
    parser.add_argument(
        "--instance",
        required=True,
        help="instance name (resolved against the MIPLIB directory) or a path",
    )
    parser.add_argument("--seed", type=int, required=True, help="HiGHS random_seed")
    for name in HEURISTICS:
        switch = name.replace("_", "-")
        parser.add_argument(
            f"--{switch}-effort",
            type=float,
            default=0.0,
            metavar="F",
            help=f"{name} effort in [0,1]; 0 (the default) means {name} does "
            "not run and is not named in the suite",
        )
    for name in HEURISTICS:
        switch = name.replace("_", "-")
        parser.add_argument(
            f"--{switch}-stall",
            type=int,
            default=0,
            metavar="N",
            help=f"{name} stall threshold in effort units per nonzero; "
            "0 (the default) means no staleness gate at all",
        )
    for name in HEURISTICS:
        switch = name.replace("_", "-")
        parser.add_argument(
            f"--{switch}-enabled",
            type=int,
            choices=(0, 1),
            default=None,
            metavar="B",
            help=f"0 forces {name}'s effort to 0.  Exists so a configurator can "
            "sample inclusion as a discrete dimension and make its effort "
            "conditional on it; the semantics are still effort 0 = off",
        )
    parser.add_argument(
        "--lambda",
        dest="cost_weight",
        type=float,
        default=DEFAULT_LAMBDA,
        metavar="L",
        help="cost weight per second of heuristic wall time "
        f"(default {DEFAULT_LAMBDA:.6g} = 1/600, i.e. g(0)/T at the campaign limit)",
    )
    parser.add_argument(
        "--no-solution-penalty",
        type=float,
        default=DEFAULT_NO_SOLUTION_PENALTY,
        metavar="P",
        help="gap charged when the run found nothing (default "
        f"{DEFAULT_NO_SOLUTION_PENALTY}; must exceed 1 + lambda x the time "
        "limit, or finding nothing outscores finding something bad)",
    )
    parser.add_argument(
        "--binary", default="build/bin/highs", help="patched HiGHS binary"
    )
    parser.add_argument(
        "--data-dir", default=None, help="MIPLIB directory (see run_benchmark.py)"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        metavar="S",
        help=f"per-run solver time limit in seconds (default {DEFAULT_TIME_LIMIT})",
    )
    parser.add_argument(
        "--run-dir",
        default="target-runs",
        help="where the .opts / .log / .json of each evaluation are kept",
    )
    parser.add_argument("--tag", default=None, help="name for this run's artifacts")
    parser.add_argument("--solu", default=DEFAULT_SOLU, help="MIPLIB .solu file")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="pin HiGHS's thread count.  Unset by default: pinning it collapses "
        "each heuristic to one worker, which makes a run bit-reproducible (a "
        "3.2%% objective spread on flugpl disappears) but changes what the "
        "parameter vector means, since three of the four effort budgets and the "
        "per-worker stall threshold are divided by the worker count.  Use it "
        "for a trajectory trace or an exact re-derivation, not for the search",
    )
    parser.add_argument(
        "--require-trace",
        action="store_true",
        help="refuse a run that produced no [Heur] line when the suite names a "
        "heuristic, instead of warning.  The pre-flight check: run one "
        "evaluation with it before launching a search",
    )
    parser.add_argument(
        "--no-presolve-only",
        dest="presolve_only",
        action="store_false",
        help="do not set mip_heuristic_presolve_only; for diagnosing a run "
        "against a binary that predates the option",
    )
    return parser


def parameters_from_args(args: argparse.Namespace) -> Parameters:
    """The eight numbers, with `--<h>-enabled 0` applied as effort 0.

    The enable switches are sampling machinery, not a second semantics: a
    configurator needs a discrete dimension to reach "off" at all, because a
    continuous effort sampled in [0,1] essentially never lands exactly on 0.
    They collapse into the effort vector here, so everything downstream — the
    suite value, the options file, the recorded vector — sees only the eight
    numbers.
    """
    efforts: dict[str, float] = {}
    stalls: dict[str, int] = {}
    for name in HEURISTICS:
        effort = getattr(args, f"{name}_effort")
        enabled = getattr(args, f"{name}_enabled")
        efforts[name] = 0.0 if enabled == 0 else effort
        stalls[name] = getattr(args, f"{name}_stall")
    return Parameters(efforts=efforts, stalls=stalls)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        params = parameters_from_args(args)
        evaluation = evaluate(
            params,
            args.instance,
            args.seed,
            binary=args.binary,
            data_dir=args.data_dir,
            time_limit=args.time_limit,
            run_dir=args.run_dir,
            solu_path=args.solu,
            cost_weight=args.cost_weight,
            no_solution_penalty=args.no_solution_penalty,
            threads=args.threads,
            tag=args.tag,
            require_trace=args.require_trace,
            presolve_only=args.presolve_only,
        )
    except (Refusal, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"{evaluation.cost:.10g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
