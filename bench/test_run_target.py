"""Unit tests for bench/run_target.py.

Everything runs against synthesised logs and a fake solver: no built binary, no
MIPLIB, no `.solu` beyond the two lines a test writes itself.  The runner is the
single definition of what a configuration means for the whole tuning stage, so
what it means has to be checkable anywhere.

The one thing these tests cannot cover is `mip_heuristic_presolve_only` itself,
which does not exist yet (issue #106, Track A).  Its absence is confined to a
single option write: `test_options_declare_presolve_only` pins that the option
is requested, and `check_run_usable` turns a binary that rejects it into a
refusal naming the option rather than a plausible-looking number.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_archive import PATCH_MARKER
from run_target import (
    DEFAULT_LAMBDA,
    DEFAULT_NO_SOLUTION_PENALTY,
    DEFAULT_TIME_LIMIT,
    HEURISTICS,
    Parameters,
    Refusal,
    build_arg_parser,
    check_penalty_dominates,
    check_run_usable,
    heuristic_wall_ms,
    instance_name,
    main,
    parameters_from_args,
    presolve_objective,
    primal_gap,
    reference_objective,
    run_tag,
    scalar_cost,
    score_output,
    solver_options,
    strip_instance_token,
    suite_value,
)

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
IRACE_DIR = os.path.join(BENCH_DIR, "irace")

_HEADER = (
    "Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol"
    "              Gap | Cuts InLp Confl. | LpIters     Time\n"
)


def params(**kwargs) -> Parameters:
    """A parameter vector, defaulting to all-zero (i.e. `off`)."""
    efforts = {h: float(kwargs.get(f"{h}_effort", 0.0)) for h in HEURISTICS}
    stalls = {h: int(kwargs.get(f"{h}_stall", 0)) for h in HEURISTICS}
    return Parameters(efforts=efforts, stalls=stalls)


def heur_line(name: str, wall_ms: float, phase: str = "presolve") -> str:
    return (
        f"[Heur] name={name} phase={phase} start_s=0.010 end_s=0.410 "
        f"effort=123456 wall_ms={wall_ms} effort_per_ms=308.6 found=1\n"
    )


def solver_log(
    *,
    objective: float | None = 100.0,
    heur: str = "",
    timing: float = 12.0,
    marker: bool = True,
    killed: bool = False,
    workers: bool = True,
) -> str:
    """A HiGHS log of the shape a presolve-only run produces.

    Not invented: Track A measured this shape against the built binary.  The
    exit is `kSolutionLimit` -> `Status Solution limit reached`, and because the
    root LP never runs, `Dual bound` is -inf and `Gap` is inf on *every*
    presolve-only run.  A run that found nothing still prints the report, with
    `Primal bound inf`.  The fixture carries all three so the scoring path is
    exercised against what the campaign will actually see rather than against a
    full-solve log with the heuristic lines pasted in.
    """
    out = "Running HiGHS 1.15.1 (git hash: 04024d70): Copyright (c) 2026\n"
    if marker:
        out += PATCH_MARKER + " (custom MIP presolve heuristics)\n"
    if workers:
        out += "  Thread count 16 (of 32 threads). Using 8 max workers.\n"
    out += _HEADER
    if objective is not None:
        out += (
            f"H       0       0         0   0.00%          0        {objective}"
            "              Large      0      0      0       0.0   0.4s\n"
        )
    out += heur
    if not killed:
        out += "Solving report\n  Status            Solution limit reached\n"
        out += f"  Primal bound      {'inf' if objective is None else objective}\n"
        out += "  Dual bound        -inf\n  Gap               inf\n"
        out += "  Nodes             0\n  LP iterations     0\n"
        out += f"  Timing            {timing}\n"
    else:
        out += "\n--- runner ---\nTIMEOUT: process killed after 150s\n"
    return out


# --- the zero-pattern -> suite mapping -------------------------------------
#
# The orchestrator's ruling, and the reason this mapping is not cosmetic:
# `mip_heuristic_fpr_effort=0` is not equivalent to omitting `fpr` from the
# suite, because only the suite value gates the dive-time `fpr_lp`.


def test_all_zero_efforts_are_exactly_off():
    """Not "", not ",,,", not "off," — the patch compares == "off" verbatim."""
    assert suite_value(params()) == "off"


def test_zero_effort_heuristic_is_not_named():
    value = suite_value(params(fj_effort=0.1, local_mip_effort=0.2))
    assert value == "fj,local_mip"
    assert "fpr" not in value and "scylla" not in value


def test_suite_lists_in_chain_order_whatever_the_order_asked():
    """One subset, one spelling — matching run_benchmark's `+` config names."""
    assert suite_value(params(scylla_effort=0.1, fj_effort=0.1)) == "fj,scylla"


def test_suite_has_no_empty_token():
    """An empty value or a trailing comma fails *open* to all four heuristics."""
    for kwargs in ({}, {"fpr_effort": 0.5}, {h + "_effort": 0.5 for h in HEURISTICS}):
        value = suite_value(params(**kwargs))
        assert value
        assert not value.startswith(",") and not value.endswith(",")
        assert ",," not in value


def test_every_subset_is_expressible():
    """All sixteen zero-patterns, and all sixteen distinct."""
    seen = set()
    for mask in range(16):
        kwargs = {
            f"{h}_effort": 0.5 for i, h in enumerate(HEURISTICS) if mask & (1 << i)
        }
        seen.add(suite_value(params(**kwargs)))
    assert len(seen) == 16
    assert "off" in seen and "fj,fpr,local_mip,scylla" in seen


def test_tiny_positive_effort_still_counts_as_on():
    """Strictly positive, not "big enough": the cutoff is a parameter, not here."""
    assert suite_value(params(scylla_effort=1e-9)) == "scylla"


# --- the parameter vector ---------------------------------------------------


def test_effort_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"fpr effort 1.5 outside"):
        params(fpr_effort=1.5)


def test_negative_stall_rejected():
    with pytest.raises(ValueError, match="not a non-negative integer"):
        params(fj_stall=-1)


def test_incomplete_vector_rejected():
    """A recorded vector that omits a heuristic cannot be told apart from one
    taken before the parameter existed."""
    with pytest.raises(ValueError, match="missing effort for 'scylla'"):
        Parameters(
            efforts={h: 0.1 for h in HEURISTICS[:-1]}, stalls={h: 1 for h in HEURISTICS}
        )


# --- the options file -------------------------------------------------------


def test_options_declare_presolve_only():
    """The one thing not exercised end-to-end: Track A's option does not exist
    yet, so this pins the request itself."""
    options = solver_options(params(fj_effort=0.1), seed=3)
    assert options["mip_heuristic_presolve_only"] == "true"


def test_options_request_the_trace_level():
    """`[Heur]` is kVerbose; without level 3 the cost axis silently reads 0."""
    assert solver_options(params(fj_effort=0.1), seed=3)["log_dev_level"] == "3"


def test_options_write_all_eight_even_when_disabled():
    options = solver_options(params(fj_effort=0.1, fpr_stall=2048), seed=3)
    for h in HEURISTICS:
        assert f"mip_heuristic_{h}_effort" in options
        assert f"mip_heuristic_{h}_stall" in options
    assert options["mip_heuristic_fpr_effort"] == "0"
    assert options["mip_heuristic_fpr_stall"] == "2048"
    assert options["mip_heuristic_suite"] == "fj"


def test_zero_fj_effort_gates_highs_own_feasibility_jump():
    """The other FJ call site.  At `suite=off` the patch restores HiGHS's
    standalone FeasibilityJump, which emits no `[Heur]` line — so the all-zero
    vector banked real FJ quality at tau = 0 and `off` outscored configurations
    that found objectives 28x better.  Effort 0 has to reach both sites."""
    off = solver_options(params(), seed=1)
    assert off["mip_heuristic_run_feasibility_jump"] == "false"
    without_fj = solver_options(params(fpr_effort=0.5), seed=1)
    assert without_fj["mip_heuristic_run_feasibility_jump"] == "false"
    with_fj = solver_options(params(fj_effort=0.0125), seed=1)
    assert with_fj["mip_heuristic_run_feasibility_jump"] == "true"


def test_options_do_not_pin_threads_by_default():
    """Pinning collapses each heuristic to one worker — the reproducible
    configuration, not the regime the search runs in."""
    assert "threads" not in solver_options(params(fj_effort=0.1), seed=1)
    assert solver_options(params(fj_effort=0.1), seed=1, threads=1)["threads"] == "1"


def test_options_carry_the_seed():
    assert solver_options(params(), seed=7)["random_seed"] == "7"


def test_effort_formatting_is_not_binary_noise():
    options = solver_options(params(local_mip_effort=0.1 + 0.2), seed=0)
    assert options["mip_heuristic_local_mip_effort"] == "0.3"


def test_option_order_is_stable():
    a = list(solver_options(params(fj_effort=0.5), seed=1))
    b = list(solver_options(params(fj_effort=0.5), seed=1))
    assert a == b


# --- the reference objective ------------------------------------------------

_REFS = {
    "good": ("=opt=", 100.0),
    "unbounded": ("=unbd=", None),
    "infeasible": ("=inf=", None),
    "unknown": ("=unkn=", None),
}


def test_published_reference_is_used():
    assert reference_objective("good", _REFS) == 100.0


def test_infeasible_tag_is_refused():
    """A gap against `=inf=` is a category error, not a small one."""
    with pytest.raises(Refusal, match=r"=inf=.*no finite"):
        reference_objective("infeasible", _REFS)


def test_unbounded_tag_is_refused():
    with pytest.raises(Refusal, match=r"=unbd="):
        reference_objective("unbounded", _REFS)


def test_unknown_reference_is_refused_not_scored_zero():
    """With one run there is no virtual best to fall back on: `resolve_reference`
    would hand back this run's own primal and every config would score gap 0."""
    with pytest.raises(Refusal, match="no usable objective"):
        reference_objective("unknown", _REFS)


def test_absent_instance_is_refused():
    with pytest.raises(Refusal, match="no entry in the solution file"):
        reference_objective("nowhere", _REFS)


def test_plato_list_carries_no_refusal():
    """The refusals above are unreachable from the campaign's own instance list;
    they exist for a list that grows past it."""
    from analyze_results import parse_solu_file
    from run_benchmark import load_instances

    refs = parse_solu_file(os.path.join(BENCH_DIR, "miplib2017-v36.solu"))
    for inst in load_instances(os.path.join(BENCH_DIR, "instances_plato.txt")):
        assert reference_objective(inst, refs) is not None


# --- the cost metric --------------------------------------------------------


def test_cost_metric_is_heur_wall_ms_not_solve_time():
    """The whole point of the axis: HiGHS's own presolve dominates on large
    models and is not ours to spend."""
    log = solver_log(heur=heur_line("fj", 400.0) + heur_line("fpr", 100.0), timing=57.0)
    ev = score_output(
        log,
        params(fj_effort=0.1, fpr_effort=0.1),
        "good",
        1,
        100.0,
        cost_weight=DEFAULT_LAMBDA,
        no_solution_penalty=1.0,
        tag="t",
        require_trace=False,
    )
    assert ev.heuristic_wall_ms == pytest.approx(500.0)
    assert ev.tau_s == pytest.approx(0.5)
    assert ev.solve_time == pytest.approx(57.0)
    assert ev.cost == pytest.approx(0.0 + DEFAULT_LAMBDA * 0.5)


def test_dive_samples_are_not_on_the_cost_axis():
    """`fpr_lp` draws from a different budget; this axis prices the chain."""
    from parse_highs_log import parse_log

    log = solver_log(heur=heur_line("fj", 400.0) + heur_line("fpr_lp", 900.0, "dive"))
    assert heuristic_wall_ms(parse_log(log)) == pytest.approx(400.0)


def test_negative_wall_ms_is_floored():
    """The ledger's clock is not monotonic; a negative window must not become a
    discount."""
    from parse_highs_log import parse_log

    log = solver_log(heur=heur_line("fj", -5.0) + heur_line("fpr", 10.0))
    assert heuristic_wall_ms(parse_log(log)) == pytest.approx(10.0)


# --- the no-solution penalty ------------------------------------------------


def _score(
    log: str,
    *,
    penalty: float = DEFAULT_NO_SOLUTION_PENALTY,
    weight: float = 0.0,
    **kwargs,
):
    return score_output(
        log,
        kwargs.pop("params", params(fj_effort=0.1)),
        "good",
        1,
        100.0,
        cost_weight=weight,
        no_solution_penalty=penalty,
        tag="t",
        require_trace=kwargs.pop("require_trace", False),
    )


def test_no_solution_is_penalised_not_dropped():
    ev = _score(solver_log(objective=None, heur=heur_line("fj", 200.0)))
    assert ev.no_solution is True
    assert ev.objective is None
    assert ev.gap == DEFAULT_NO_SOLUTION_PENALTY
    assert ev.cost == pytest.approx(DEFAULT_NO_SOLUTION_PENALTY)


def test_no_solution_penalty_is_explicit():
    ev = _score(solver_log(objective=None, heur=heur_line("fj", 1.0)), penalty=3.0)
    assert ev.cost == pytest.approx(3.0)


def test_finding_nothing_never_beats_finding_something_bad():
    """The property the old version of this test only appeared to check.

    It asserted `gap == 1.0` and `no_solution is False`, never touched `cost`,
    and ran at the helper's default `weight=0.0` — the single lambda at which a
    penalty equal to the gap cap is safe.  So it passed while the shipped
    configuration was inverted: at lambda = 1/600 a run that found a terrible
    solution and spent 60 s scored 1.1 against the 1.0 charged for finding
    nothing, and the search preferred nothing.  Asserted here at the shipped
    lambda and the largest tau the default time cap allows.
    """
    worst_found = _score(
        solver_log(objective=1e9, heur=heur_line("fj", 1000 * DEFAULT_TIME_LIMIT)),
        weight=DEFAULT_LAMBDA,
    )
    nothing_found = _score(
        solver_log(objective=None, heur=heur_line("fj", 0.0)),
        weight=DEFAULT_LAMBDA,
    )
    assert worst_found.gap == 1.0  # the cap, so this is as bad as found gets
    assert worst_found.no_solution is False
    assert worst_found.cost < nothing_found.cost


def test_penalty_dominance_is_checked_not_assumed(capsys):
    """The shipped triple is safe, and a hostile one says so out loud."""
    check_penalty_dominates(
        DEFAULT_NO_SOLUTION_PENALTY, DEFAULT_LAMBDA, DEFAULT_TIME_LIMIT
    )
    assert capsys.readouterr().err == ""
    check_penalty_dominates(1.0, DEFAULT_LAMBDA, DEFAULT_TIME_LIMIT)
    assert "search is inverted" in capsys.readouterr().err


def test_penalty_dominance_tracks_lambda_and_the_time_cap(capsys):
    """It is the triple that has to hold, not the constant: a big enough lambda
    or a long enough cap breaks any fixed penalty.  Asserted on the output, not
    just called — an assertion-free test is the defect the test above exists to
    stop shipping."""
    check_penalty_dominates(2.0, 1.0, 60.0)  # worst found = 1 + 60 = 61 > 2
    assert "search is inverted" in capsys.readouterr().err
    check_penalty_dominates(2.0, DEFAULT_LAMBDA, 600.0)  # 1 + 1 = 2, not >
    assert "search is inverted" in capsys.readouterr().err


# --- the scalar and its sign ------------------------------------------------


def test_lower_cost_is_better():
    good = _score(solver_log(objective=101.0, heur=heur_line("fj", 1.0)))
    bad = _score(solver_log(objective=150.0, heur=heur_line("fj", 1.0)))
    assert good.cost < bad.cost


def test_more_heuristic_time_costs_more():
    cheap = _score(solver_log(objective=110.0, heur=heur_line("fj", 10.0)), weight=1.0)
    dear = _score(solver_log(objective=110.0, heur=heur_line("fj", 5000.0)), weight=1.0)
    assert dear.cost > cheap.cost
    assert dear.cost - cheap.cost == pytest.approx(4.99)


def test_cost_is_the_negated_issue_objective():
    """#107 states `gap_improvement - lambda*tau`, maximised; this prints
    `gap + lambda*tau`, minimised.  They differ by the constant 1."""
    ev = _score(solver_log(objective=120.0, heur=heur_line("fj", 250.0)), weight=0.5)
    assert ev.gap_improvement == pytest.approx(1.0 - ev.gap)
    assert ev.cost == pytest.approx(1.0 - (ev.gap_improvement - 0.5 * ev.tau_s))


def test_scalar_cost_is_a_plain_sum():
    assert scalar_cost(0.25, 2.0, 0.1) == pytest.approx(0.45)


def test_primal_gap_is_relative_and_capped():
    assert primal_gap(110.0, 100.0) == pytest.approx(0.1)
    assert primal_gap(0.5, 0.0) == pytest.approx(0.5)  # denominator floored at 1
    assert primal_gap(-1e6, 100.0) == 1.0


# --- what the objective is read from ----------------------------------------


def test_objective_prefers_the_solving_report():
    from parse_highs_log import parse_log

    assert presolve_objective(parse_log(solver_log(objective=42.0))) == pytest.approx(
        42.0
    )


def test_killed_run_is_scored_from_its_incumbents():
    """A truncated log never printed a report, but every incumbent it printed
    before the kill is real measured data."""
    from parse_highs_log import parse_log

    result = parse_log(solver_log(objective=42.0, killed=True))
    assert result.killed
    assert presolve_objective(result) == pytest.approx(42.0)


# --- the presolve-only exit path, as Track A measured it --------------------
#
# `mip_heuristic_presolve_only` exits with `HighsModelStatus::kSolutionLimit`
# (kOk and kInfeasible are both rewritten by `cleanupSolve`, which would report
# a presolve-exit run as a proven optimum).  That maps to `HighsStatus::kWarning`
# and CLI exit 1, and it prints a complete Solving report.  All three facts are
# load-bearing here and none of them is ours to change, so they are pinned.


def test_presolve_only_exit_code_is_accepted():
    """kSolutionLimit -> kWarning -> exit 1, which must not read as a rejected
    option."""
    check_run_usable(solver_log(heur=heur_line("fj", 10.0)), 1, "good")


def test_presolve_only_status_is_recorded():
    ev = _score(solver_log(heur=heur_line("fj", 10.0)))
    assert ev.status == "Solution limit reached"


def test_missing_dual_bound_does_not_reach_the_score():
    """The root LP never runs, so every presolve-only run reports `Dual bound
    -inf` and `Gap inf`.  Quality is the primal gap against the `.solu`
    reference, which is why that is harmless — but only as long as nothing on
    this path reads the dual side."""
    from parse_highs_log import parse_log

    result = parse_log(solver_log(objective=110.0))
    assert result.dual_bound == float("-inf")
    assert result.gap == float("inf")
    ev = _score(solver_log(objective=110.0, heur=heur_line("fj", 10.0)))
    assert ev.gap == pytest.approx(0.1)  # |110 - 100| / 100, not the dual side


def test_no_solution_shape_is_an_infinite_primal_bound():
    """How a presolve-only run that found nothing actually looks: the report is
    printed, with `Primal bound inf` rather than no line at all."""
    log = solver_log(objective=None, heur=heur_line("fj", 10.0))
    assert "Primal bound      inf" in log
    ev = _score(log)
    assert ev.no_solution is True
    assert ev.cost == pytest.approx(DEFAULT_NO_SOLUTION_PENALTY)


# --- runs that cannot mean what their parameters say ------------------------


def test_rejected_option_is_refused():
    """What a binary predating `mip_heuristic_presolve_only` does: two ERROR
    lines and exit 255."""
    with pytest.raises(Refusal, match="mip_heuristic_presolve_only"):
        check_run_usable(solver_log(), 255, "good")


def test_unpatched_binary_is_refused():
    """Patched and unpatched builds of the same tag have identical banners."""
    with pytest.raises(Refusal, match="not a patched build"):
        check_run_usable(solver_log(marker=False), 0, "good")


def test_ignored_suite_value_is_refused():
    """HiGHS accepts an unknown suite *value* and fails open to all four."""
    log = solver_log() + "Unknown mip_heuristic_suite value 'fj,of'\n"
    with pytest.raises(Refusal, match="ignored its configuration"):
        check_run_usable(log, 0, "good")


def test_time_limit_exit_is_not_a_failure():
    check_run_usable(solver_log(), 1, "good")  # kWarning is the normal outcome


def test_killed_run_is_not_a_failure():
    check_run_usable(solver_log(killed=True), None, "good")


# --- the missing-trace guard ------------------------------------------------


def test_missing_trace_warns_by_default(capsys):
    ev = _score(solver_log(heur=""))
    assert ev.trace_missing is True
    assert "no [Heur] presolve sample" in capsys.readouterr().err


def test_missing_trace_is_fatal_under_require_trace():
    """The pre-flight: one evaluation with this flag before a ten-hour search."""
    with pytest.raises(Refusal, match="cost axis is reading zero"):
        _score(solver_log(heur=""), require_trace=True)


def test_off_suite_needs_no_trace():
    ev = _score(solver_log(heur=""), params=params(), require_trace=True)
    assert ev.trace_missing is False
    assert ev.suite == "off"


def test_killed_run_needs_no_trace():
    """An instance whose limit expired inside HiGHS's own presolve is a real
    measurement, not a misconfiguration — and must not abort the search."""
    ev = _score(solver_log(killed=True, heur=""), require_trace=True)
    assert ev.trace_missing is False


# --- chain truncation -------------------------------------------------------


def test_chain_truncation_is_recorded():
    """A generous head starves the tail: the chain runs FJ, FPR, LocalMIP,
    Scylla against a shared time limit, so Scylla's parameters can be recorded
    without ever being exercised."""
    ev = _score(
        solver_log(heur=heur_line("fj", 20000.0) + heur_line("fpr", 4000.0)),
        params=params(
            fj_effort=1.0, fpr_effort=1.0, local_mip_effort=1.0, scylla_effort=1.0
        ),
    )
    assert ev.heuristics_traced == ["fj", "fpr"]
    assert ev.chain_truncated == ["local_mip", "scylla"]
    assert ev.trace_missing is False  # a different failure, kept distinct


def test_untruncated_chain_records_nothing():
    ev = _score(
        solver_log(heur=heur_line("fj", 10.0) + heur_line("local_mip", 10.0)),
        params=params(fj_effort=0.1, local_mip_effort=0.1),
    )
    assert ev.chain_truncated == []


def test_total_absence_is_trace_missing_not_truncation(capsys):
    """Partial absence is truncation; total absence is missing instrumentation.
    Reporting the second as the first would hide a campaign-wide failure inside
    a field that reads as an ordinary scheduling artefact."""
    ev = _score(solver_log(heur=""), params=params(fj_effort=0.1, scylla_effort=0.1))
    assert ev.trace_missing is True
    assert ev.chain_truncated == []
    capsys.readouterr()


# --- the worker-count stamp -------------------------------------------------
#
# The same vector does not mean the same thing at every worker count: Track A's
# algebra out of `make_budget` shows the whole-dispatch aggregates are
# N-invariant while the per-worker slicing moves, so the cross-N effect is a
# reallocation *between* heuristics (measured on p0548: LocalMIP 1.08x, FJ 12x
# from N=1 to N=8 at identical options).  That makes the worker count part of
# the result, and an unattributable run worse than a mis-attributed one.


def test_worker_count_is_recorded():
    ev = _score(solver_log(heur=heur_line("fj", 10.0)))
    assert ev.thread_count == 16


def test_missing_worker_count_warns(capsys):
    ev = _score(solver_log(heur=heur_line("fj", 10.0), workers=False))
    assert ev.thread_count is None
    assert "records no worker count" in capsys.readouterr().err


def test_killed_run_does_not_warn_about_the_worker_count(capsys):
    """A truncated log may have been cut before the line was printed; that is
    the kill, not a regime that cannot be attributed."""
    _score(solver_log(killed=True, heur="", workers=False))
    assert "records no worker count" not in capsys.readouterr().err


# --- reproducibility --------------------------------------------------------


def test_tag_is_deterministic_and_parameter_sensitive():
    a = run_tag(params(fj_effort=0.1, fj_stall=256), "egout", 3)
    assert a == run_tag(params(fj_effort=0.1, fj_stall=256), "egout", 3)
    assert a != run_tag(params(fj_effort=0.1, fj_stall=512), "egout", 3)
    assert a != run_tag(params(fj_effort=0.2, fj_stall=256), "egout", 3)
    assert a != run_tag(params(fj_effort=0.1, fj_stall=256), "egout", 4)
    assert a != run_tag(params(fj_effort=0.1, fj_stall=256), "flugpl", 3)


def test_tag_separates_runs_that_are_not_the_same_measurement():
    """Two runs of one vector at different caps are different measurements, and
    two scorings of one run at different lambdas are different numbers — but the
    `.json` records the number, so a shared tag silently overwrites the first.
    #107 sweeps lambda by construction, so that collision is the common case."""
    base = params(fj_effort=0.1, fj_stall=256)
    tag = run_tag(base, "mad", 3)
    assert tag != run_tag(base, "mad", 3, time_limit=30.0)
    assert tag != run_tag(base, "mad", 3, threads=1)
    assert tag != run_tag(base, "mad", 3, cost_weight=3 * DEFAULT_LAMBDA)
    assert tag != run_tag(base, "mad", 3, no_solution_penalty=3.0)
    assert tag == run_tag(base, "mad", 3)  # still deterministic


def test_end_to_end_tag_follows_the_time_limit(campaign):
    """The collision that mattered: same vector, two caps, one set of files."""
    assert main(_cli(campaign, "--fj-effort", "0.1", "--time-limit", "30")) == 0
    assert main(_cli(campaign, "--fj-effort", "0.1", "--time-limit", "45")) == 0
    opts = list((campaign / "runs" / "toy").glob("*.opts"))
    assert len(opts) == 2


def test_instance_name_strips_paths_and_extensions():
    assert instance_name("egout") == "egout"
    assert instance_name("/data/miplib/egout.mps.gz") == "egout"
    assert instance_name(" egout.mps \n") == "egout"


# --- instance tokens carrying comments --------------------------------------
#
# `bench/instances_tuning.txt` is generated with a `#` header block and a
# trailing `# <vanilla time-to-first-feasible>` on every instance line.  irace
# very likely strips those, but that is not executable here, and if it were
# false every evaluation in a ten-hour campaign would refuse with the instance
# named `0.60s`.  So the runner does not depend on it.


def test_instance_name_drops_a_trailing_comment():
    """The exact shape `bench/instances_tuning.txt` uses, spaces and all."""
    assert instance_name("comp21-2idx               # 0.60s") == "comp21-2idx"


def test_instance_name_drops_a_comment_after_a_path():
    token = "/data/miplib/comp21-2idx.mps.gz    # 0.60s"
    assert instance_name(token) == "comp21-2idx"


def test_stripping_an_instance_token_is_idempotent():
    """`target-runner` strips before `run_target.py` sees the token, so the
    second strip has to be a no-op rather than a second bite at the string."""
    once = strip_instance_token("comp21-2idx               # 0.60s")
    assert once == "comp21-2idx"
    assert strip_instance_token(once) == once


def test_bare_and_path_tokens_are_untouched():
    assert strip_instance_token("mad") == "mad"
    assert strip_instance_token("/data/miplib/mad.mps.gz") == "/data/miplib/mad.mps.gz"


@pytest.mark.parametrize("token", ["", "   ", "# 0.60s", "   # just a comment"])
def test_comment_only_instance_token_is_refused(token):
    """Loud, not an empty instance name: an empty name would look up an empty
    reference and refuse somewhere far less legible, or resolve to a directory."""
    with pytest.raises(Refusal, match="empty or contains only a comment"):
        instance_name(token)


def test_every_line_of_the_tuning_list_survives_stripping():
    """Against the real file, not a fixture of what it is assumed to look like."""
    with open(os.path.join(BENCH_DIR, "instances_tuning.txt")) as f:
        lines = f.read().splitlines()
    names = [
        instance_name(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert names, "the tuning list is empty; the comment filter is wrong"
    assert "comp21-2idx" in names
    for name in names:
        assert "#" not in name and name == name.strip()
        assert not re.fullmatch(r"[\d.]+s", name), f"{name!r} is a timing comment"


# --- the CLI ----------------------------------------------------------------


def test_enabled_switch_forces_effort_zero():
    """`--<h>-enabled 0` is sampling machinery, not a second semantics: it
    collapses into the effort vector before anything else sees it."""
    args = build_arg_parser().parse_args(
        ["--instance", "x", "--seed", "1", "--fj-effort", "0.5", "--fj-enabled", "0"]
    )
    assert parameters_from_args(args).efforts["fj"] == 0.0
    assert suite_value(parameters_from_args(args)) == "off"


def test_omitted_effort_defaults_to_off():
    args = build_arg_parser().parse_args(["--instance", "x", "--seed", "1"])
    assert suite_value(parameters_from_args(args)) == "off"


def test_lambda_defaults_to_the_derived_weight():
    args = build_arg_parser().parse_args(["--instance", "x", "--seed", "1"])
    assert args.cost_weight == pytest.approx(1.0 / 600.0)


# --- the irace files --------------------------------------------------------
#
# The parameter file's `switch` column is a contract with the CLI above.  Drift
# between them is silent: irace would pass a switch argparse rejects, and every
# evaluation would die with a usage message hours into a search.

_PARAM_RE = re.compile(r'^(\w+)\s+"(--[\w-]+ ?)"\s+([\w,]+)\s+\(([^)]*)\)(.*)$')


def parsed_parameter_file() -> list[tuple[str, str, str, str, str]]:
    entries = []
    with open(os.path.join(IRACE_DIR, "parameters.txt")) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = _PARAM_RE.match(line)
            assert m, f"unparseable parameter line: {line!r}"
            entries.append(m.groups())
    return entries


def test_parameter_file_covers_all_eight_dimensions_plus_inclusion():
    names = {e[0] for e in parsed_parameter_file()}
    assert names == (
        set(HEURISTICS)
        | {f"{h}_effort" for h in HEURISTICS}
        | {f"{h}_stall" for h in HEURISTICS}
    )


def test_every_irace_switch_is_a_runner_switch():
    parser = build_arg_parser()
    for name, switch, _type, _domain, _cond in parsed_parameter_file():
        args = parser.parse_args(
            ["--instance", "x", "--seed", "1", switch.strip(), "1"]
        )
        dest = switch.strip().lstrip("-").replace("-", "_")
        assert getattr(args, dest) is not None, f"{name} -> {switch} did not bind"


def test_effort_and_stall_are_conditional_on_inclusion():
    """A parameter with no effect is one irace's model should not have to
    learn."""
    for name, _switch, _type, _domain, cond in parsed_parameter_file():
        if name in HEURISTICS:
            assert cond.strip() == ""
        else:
            heuristic = name.rsplit("_", 1)[0]
            assert cond.strip() == f'| {heuristic} == "1"'


def test_stall_range_reaches_the_inert_region():
    """The gate is clamped to the budget, so `stall >= 81920 * effort` is
    inert; the top of the range has to reach that at effort 1.0."""
    for name, _switch, _type, domain, _cond in parsed_parameter_file():
        if name.endswith("_stall"):
            low, high = (float(v) for v in domain.split(","))
            assert low >= 1  # log sampling needs a positive lower bound
            assert high >= 81920


def test_effort_range_clears_fjs_granularity_floor():
    """FJ's charged effort moves in steps of CALLBACK_EFFORT = 500000, so its
    option is a no-op whenever `nnz * value < 6.1`."""
    for name, _switch, type_, domain, _cond in parsed_parameter_file():
        if name.endswith("_effort"):
            low, high = (float(v) for v in domain.split(","))
            assert type_ == "r,log"
            assert low >= 0.001
            assert high == 1.0


def test_scenario_points_at_the_tracked_files():
    with open(os.path.join(IRACE_DIR, "scenario.txt")) as f:
        text = f.read()
    assert 'parameterFile = "./parameters.txt"' in text
    assert 'targetRunner = "./target-runner"' in text
    # Non-deterministic runs: the same seed on the same instance is not the
    # same run, and irace has to be told so to re-evaluate with new seeds.
    assert "deterministic = 0" in text


def test_target_runner_is_executable():
    mode = os.stat(os.path.join(IRACE_DIR, "target-runner")).st_mode
    assert mode & stat.S_IXUSR


_ARGV_DUMPER = """#!/bin/sh
# Stands in for the Python interpreter and prints one argument per line.
# `echo` cannot serve here: the assertions below are about argument
# *boundaries*, and echo joins them with spaces so a token that still carries
# its `# 0.60s` comment is indistinguishable from a stripped one.
shift  # the run_target.py path
printf '%s\\n' "$@"
"""


@pytest.fixture
def argv_dumper(tmp_path):
    path = tmp_path / "argv-dump"
    path.write_text(_ARGV_DUMPER)
    path.chmod(0o755)
    return str(path)


def _run_target_runner(argv: list[str], **env):
    # `**env` last so a caller can override the default stand-in interpreter.
    environment = dict(os.environ, **{"RUN_TARGET_PYTHON": "echo", **env})
    return subprocess.run(
        [os.path.join(IRACE_DIR, "target-runner"), *argv],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_target_runner_translates_iraces_positional_call():
    proc = _run_target_runner(
        ["7", "3", "1234", "egout", "--fj-effort", "0.5", "--fj-stall", "256"]
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.split()
    assert "--instance" in out and out[out.index("--instance") + 1] == "egout"
    assert out[out.index("--seed") + 1] == "1234"
    assert out[out.index("--tag") + 1] == "c7-i3-s1234"
    assert out[out.index("--fj-effort") + 1] == "0.5"
    assert out[out.index("--fj-stall") + 1] == "256"


def test_target_runner_refuses_a_capping_bound():
    """A bound consumed as a parameter value shifts every switch by one and
    produces a plausible cost for a configuration nobody chose."""
    proc = _run_target_runner(["7", "3", "1234", "egout", "60", "--fj-effort", "0.5"])
    assert proc.returncode == 1
    assert "capping" in proc.stderr


def test_target_runner_refuses_a_short_call():
    proc = _run_target_runner(["7", "3", "1234"])
    assert proc.returncode == 1


def test_target_runner_strips_a_trailing_comment_from_the_instance(argv_dumper):
    """The wrapper can be the first thing to see the token, so it strips too.
    The shape is the one every line of `bench/instances_tuning.txt` has.

    One argument per line, not `echo` plus `.split()`: the latter reads green
    against a wrapper that does not strip at all, because joining the argv on
    spaces makes `comp21-2idx  # 0.60s` and `comp21-2idx` indistinguishable
    once split again.  Verified by mutation — this assertion fails when the
    `%%#*` expansion is removed, and the split-based one did not.
    """
    proc = _run_target_runner(
        ["7", "3", "1234", "comp21-2idx               # 0.60s", "--fj-effort", "0.5"],
        RUN_TARGET_PYTHON=argv_dumper,
    )
    assert proc.returncode == 0, proc.stderr
    argv = proc.stdout.splitlines()
    assert argv[argv.index("--instance") + 1] == "comp21-2idx"
    # The comment must not have shifted the switches along.
    assert argv[argv.index("--fj-effort") + 1] == "0.5"


def test_target_runner_refuses_a_comment_only_instance():
    proc = _run_target_runner(["7", "3", "1234", "# 0.60s", "--fj-effort", "0.5"])
    assert proc.returncode == 1
    assert "comment only" in proc.stderr


# --- end to end against a fake solver ---------------------------------------


FAKE_SOLVER = """#!/bin/sh
echo "$@" > "$ARGS_FILE"
cat "$LOG_FILE"
"""


@pytest.fixture
def campaign(tmp_path):
    """A fake solver, an instance file and a solution file."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "toy.mps").write_text("NAME toy\nENDATA\n")
    (tmp_path / "toy.solu").write_text("=opt=  toy   100\n=inf=  hopeless\n")
    binary = tmp_path / "fake-highs"
    binary.write_text(FAKE_SOLVER)
    binary.chmod(0o755)
    (tmp_path / "canned.log").write_text(
        solver_log(
            objective=110.0, heur=heur_line("fj", 400.0) + heur_line("fpr", 100.0)
        )
    )
    os.environ["ARGS_FILE"] = str(tmp_path / "argv.txt")
    os.environ["LOG_FILE"] = str(tmp_path / "canned.log")
    yield tmp_path
    del os.environ["ARGS_FILE"], os.environ["LOG_FILE"]


def _cli(campaign, *extra: str) -> list[str]:
    return [
        "--instance",
        "toy",
        "--seed",
        "5",
        "--binary",
        str(campaign / "fake-highs"),
        "--data-dir",
        str(campaign / "data"),
        "--solu",
        str(campaign / "toy.solu"),
        "--run-dir",
        str(campaign / "runs"),
        *extra,
    ]


def test_end_to_end_prints_one_number(campaign, capsys):
    code = main(_cli(campaign, "--fj-effort", "0.0125", "--fj-stall", "256"))
    assert code == 0
    out = capsys.readouterr().out.strip()
    assert len(out.splitlines()) == 1
    # gap = |110 - 100| / 100 = 0.1; tau = 0.5 s at the default lambda.
    assert float(out) == pytest.approx(0.1 + DEFAULT_LAMBDA * 0.5)


def test_end_to_end_keeps_opts_log_and_record(campaign):
    assert main(_cli(campaign, "--fpr-effort", "0.5", "--fpr-stall", "2048")) == 0
    run_dir = campaign / "runs" / "toy"
    tag = run_tag(params(fpr_effort=0.5, fpr_stall=2048), "toy", 5)
    opts = (run_dir / f"{tag}.opts").read_text()
    assert "mip_heuristic_suite = fpr\n" in opts
    assert "mip_heuristic_fpr_stall = 2048\n" in opts
    assert "mip_heuristic_presolve_only = true\n" in opts
    assert "random_seed = 5\n" in opts
    # Compressed, because `log_dev_level=3` runs to millions of lines and a
    # few-thousand-evaluation search would otherwise leave gigabytes behind.
    with gzip.open(run_dir / f"{tag}.log.gz", "rt") as f:
        assert PATCH_MARKER in f.read()
    record = json.loads((run_dir / f"{tag}.json").read_text())
    assert record["suite"] == "fpr"
    assert record["efforts"]["fpr"] == 0.5
    assert record["stalls"]["fpr"] == 2048
    assert record["thread_count"] == 16
    assert record["heuristics_traced"] == ["fj", "fpr"]


def test_beating_the_published_objective_scores_zero_gap(campaign, capsys):
    """The virtual-best branch of `resolve_reference`: an observed primal that
    beats the `.solu` value becomes the reference, so a configuration is never
    punished for finding something better than the library knew about."""
    (campaign / "canned.log").write_text(
        solver_log(objective=90.0, heur=heur_line("fj", 400.0))
    )
    assert main(_cli(campaign, "--fj-effort", "0.1")) == 0
    # Not |90 - 100| / 100 = 0.1: the reference moves to 90 and the gap is 0.
    assert float(capsys.readouterr().out) == pytest.approx(DEFAULT_LAMBDA * 0.4)
    tag = run_tag(params(fj_effort=0.1), "toy", 5)
    record = json.loads((campaign / "runs" / "toy" / f"{tag}.json").read_text())
    assert record["reference"] == 90.0
    assert record["gap"] == 0.0


def test_no_presolve_only_reaches_the_options_file(campaign):
    """The escape hatch has to actually reach the options file — and land in
    its own artifacts, since a full solve of the same vector is a different
    measurement that must not overwrite the screening run."""
    assert main(_cli(campaign, "--fj-effort", "0.1", "--no-presolve-only")) == 0
    screen_tag = run_tag(params(fj_effort=0.1), "toy", 5)
    full_tag = run_tag(params(fj_effort=0.1), "toy", 5, presolve_only=False)
    assert full_tag != screen_tag
    opts = (campaign / "runs" / "toy" / f"{full_tag}.opts").read_text()
    assert "presolve_only" not in opts
    assert not (campaign / "runs" / "toy" / f"{screen_tag}.opts").exists()


def test_end_to_end_passes_the_time_limit_on_the_command_line(campaign):
    """HiGHS takes `--time_limit` on the command line, so it is in no .opts —
    the same reason `make_archive.py` requires it as an argument."""
    assert main(_cli(campaign, "--fj-effort", "0.1", "--time-limit", "30")) == 0
    argv = (campaign / "argv.txt").read_text().split()
    assert argv[argv.index("--time_limit") + 1] == "30.0"
    assert argv[argv.index("--options_file") + 1].endswith(".opts")


def test_end_to_end_is_rerunnable_into_the_same_files(campaign, capsys):
    first = _cli(campaign, "--fj-effort", "0.1", "--fj-stall", "256")
    assert main(first) == 0
    a = capsys.readouterr().out
    before = sorted(os.listdir(campaign / "runs" / "toy"))
    assert main(first) == 0
    assert capsys.readouterr().out == a
    assert sorted(os.listdir(campaign / "runs" / "toy")) == before


def test_end_to_end_refuses_an_excluded_instance(campaign, capsys):
    code = main(
        [
            "--instance",
            "hopeless",
            "--seed",
            "5",
            "--binary",
            str(campaign / "fake-highs"),
            "--data-dir",
            str(campaign / "data"),
            "--solu",
            str(campaign / "toy.solu"),
            "--run-dir",
            str(campaign / "runs"),
            "--fj-effort",
            "0.1",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing for a configurator to read as a cost
    assert "=inf=" in captured.err


def test_end_to_end_off_configuration_scores_the_penalty(campaign, capsys):
    """All four at zero is `off`: it must find nothing, and be *charged* for it.

    Also the end-to-end guard on the inversion that made `off` a plausible
    winner — at `suite=off` the patch restores HiGHS's own FeasibilityJump call
    site, which emits no `[Heur]` line, so without the flag below this run banks
    real FJ quality at tau = 0.
    """
    (campaign / "canned.log").write_text(solver_log(objective=None))
    assert main(_cli(campaign)) == 0
    assert float(capsys.readouterr().out) == pytest.approx(DEFAULT_NO_SOLUTION_PENALTY)
    opts_dir = campaign / "runs" / "toy"
    tag = run_tag(params(), "toy", 5)
    opts = (opts_dir / f"{tag}.opts").read_text()
    assert "mip_heuristic_suite = off\n" in opts
    assert "mip_heuristic_run_feasibility_jump = false\n" in opts
