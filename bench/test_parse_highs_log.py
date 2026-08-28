"""Smoke tests for parse_highs_log."""

import pytest
from parse_highs_log import parse_log


def test_empty_log_returns_default_result():
    result = parse_log("")
    assert result.status == ""
    assert result.incumbents == []


def test_sequential_lines_parse_into_sequential_samples():
    """`[Sequential]` lines carry per-heuristic effort (issue #71)."""
    log = (
        "[Sequential] heur=fj effort=1000 wall_ms=5.0 effort_per_ms=200\n"
        "[Sequential] heur=fpr effort=2500 wall_ms=50.0 effort_per_ms=50\n"
        "[Sequential] heur=local_mip effort=3000 wall_ms=90.0 effort_per_ms=33\n"
        "[Sequential] heur=scylla effort=4000 wall_ms=800.0 effort_per_ms=5\n"
    )
    result = parse_log(log)
    assert len(result.sequential_samples) == 4

    names = [s.heuristic for s in result.sequential_samples]
    assert names == ["fj", "fpr", "local_mip", "scylla"]
    scylla = result.sequential_samples[-1]
    assert scylla.effort == 4000
    assert scylla.wall_ms == 800.0
    assert scylla.effort_per_ms == 5.0


def test_time_to_best_returns_last_incumbent_time():
    """time_to_best is the time of the last incumbent update, for SGM T_best."""
    from parse_highs_log import Incumbent, SolveResult

    r = SolveResult()
    assert r.time_to_best is None
    r.incumbents.append(Incumbent(time=1.5, objective=10.0, source="H", nodes=0))
    assert r.time_to_best == 1.5
    r.incumbents.append(Incumbent(time=7.2, objective=8.0, source="B", nodes=3))
    assert r.time_to_first_feasible == 1.5
    assert r.time_to_best == 7.2


def test_model_header_sets_dimensions_and_category():
    """Category classifier follows Local-MIP §6.1.1 (BP/IP/MBP/MIP)."""
    # BP: all 650 integer, all binary, no continuous
    bp_log = "MIP ex-bp has 91 rows; 500 cols; 1968 nonzeros; 500 integer variables (500 binary)\n"
    r = parse_log(bp_log)
    assert r.num_rows == 91 and r.num_cols == 500 and r.num_binary == 500
    assert r.category == "BP"

    # IP: all integer, not all binary, no continuous
    ip_log = "MIP ex-ip has 50 rows; 100 cols; 200 nonzeros; 100 integer variables (20 binary)\n"
    assert parse_log(ip_log).category == "IP"

    # MBP: binary + continuous, no general integer
    mbp_log = "MIP ex-mbp has 50 rows; 100 cols; 200 nonzeros; 60 integer variables (60 binary)\n"
    assert parse_log(mbp_log).category == "MBP"

    # MIP: general integer + continuous
    mip_log = "MIP ex-mip has 50 rows; 100 cols; 200 nonzeros; 60 integer variables (40 binary)\n"
    assert parse_log(mip_log).category == "MIP"

    # Missing header -> no category
    assert parse_log("").category is None


def test_custom_source_codes_recorded():
    """A, D, M, G source codes (FPR, fpr_lp, LocalMIP, Scylla) must be
    captured in incumbents.  They were missing from _INCUMBENT_SOURCES before
    the fix and caused LocalMIP incumbents to be silently dropped."""
    log = (
        "Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap\n"
        "M       0       0         0   0.00%          0              10              Large      0      0      0       0.0   1.2s\n"
        "A       1       0         1  50.00%          0               8                 0%      0      0      0       1.0   2.5s\n"
        "D       2       0         2  80.00%          5               7                 0%      0      0      0       2.0   3.1s\n"
        "G       3       0         3 100.00%          6               6                 0%      0      0      0       3.0   4.0s\n"
    )
    result = parse_log(log)
    assert len(result.incumbents) == 4
    sources = [inc.source for inc in result.incumbents]
    assert sources == ["M", "A", "D", "G"]
    assert result.incumbents[0].objective == 10.0
    assert result.incumbents[3].objective == 6.0


def test_presolve_optimal_space_source_recorded():
    """When presolve solves the model to optimality (empty B&B), the single
    log line has a space source.  It must be recorded as incumbent 'P' so
    the instance is not misclassified as infeasible."""
    log = (
        "         0       0         0   0.00%   81              81                 0.00%        0      0      0          0   3.2s\n"
        "  Status            Optimal\n"
        "  Primal bound      81\n"
        "  Dual bound        81\n"
        "  Timing            3.2\n"
        "  Nodes             0\n"
    )
    result = parse_log(log)
    assert len(result.incumbents) == 1
    assert result.incumbents[0].source == "P"
    assert result.incumbents[0].objective == 81.0
    assert result.time_to_first_feasible == 3.2
    assert result.primal_bound == 81.0


def test_primal_bound_matches_best_incumbent():
    """Consistency invariant: if incumbents are recorded, the last objective
    must equal primal_bound (within float tolerance)."""
    import math

    log = (
        "H       0       0         0   0.00%          0              20              Large      0      0      0       0.0   1.0s\n"
        "L       5       0         5  50.00%          8              12                 0%      0      0      0      10.0   5.0s\n"
        "  Status            Time limit reached\n"
        "  Primal bound      12\n"
        "  Dual bound        8\n"
        "  Timing            5.0\n"
        "  Nodes             5\n"
    )
    result = parse_log(log)
    assert result.incumbents
    best_inc = result.incumbents[-1].objective
    assert math.isclose(best_inc, result.primal_bound, rel_tol=1e-6)


def test_primal_gap_at_returns_none_when_dual_bound_infinite():
    """primal_gap_at must return None (not NaN) when no best_known is provided
    and the incumbent's dual_bound is ±inf (typical at the root node before
    any B&B bound is computed).  Before the fix, abs(obj - (-inf)) / inf
    produced NaN, which propagated silently into SGM calculations."""
    import math

    # The first incumbent line has BestBound = '-inf' (root, no LP solved yet).
    log = (
        "J       0       0         0   0.00%   -inf            50.0              Large      0      0      0       0.0   1.0s\n"
        "  Status            Time limit reached\n"
        "  Primal bound      50.0\n"
        "  Dual bound        10.0\n"
        "  Timing            1.0\n"
    )
    result = parse_log(log)
    assert len(result.incumbents) == 1
    assert result.incumbents[0].dual_bound == float("-inf")

    # Without best_known, gap should be None (not nan) because ref is -inf.
    gap = result.primal_gap_at(1.0)
    assert gap is None, f"Expected None but got {gap}"

    # With best_known provided, gap is computable and finite.
    gap_known = result.primal_gap_at(1.0, best_known=40.0)
    assert gap_known is not None
    assert math.isfinite(gap_known)

    # primal_integral without best_known should also be finite (skips -inf points).
    pi = result.primal_integral(1.0)
    assert math.isfinite(pi), f"Expected finite primal_integral but got {pi}"


def test_heur_line_accepts_negative_wall_ms():
    """The ledger times against HiGHS's solver clock, which is not
    monotonic (high_resolution_clock == system_clock on libstdc++).  A
    wall-clock step can yield a negative window; the sample must surface
    rather than being silently dropped by the pattern."""
    log = (
        "[Sequential] heur=fpr effort=100 wall_ms=-3.0 effort_per_ms=0.000\n"
        "[Heur] name=fpr phase=presolve start_s=1.0 end_s=0.997 effort=100 "
        "wall_ms=-3.0 effort_per_ms=0.000 found=0\n"
    )
    result = parse_log(log)
    assert len(result.sequential_samples) == 1
    assert result.sequential_samples[0].wall_ms == -3.0
    assert len(result.heuristic_samples) == 1
    assert result.heuristic_samples[0].wall_ms == -3.0


def test_sequential_and_heur_lines_coexist():
    """Both tags are emitted for the same observation; `[Sequential]` is
    the one external tooling parses and must keep parsing."""
    log = (
        "[Sequential] heur=scylla effort=4000 wall_ms=800.0 effort_per_ms=5.000\n"
        "[Heur] name=scylla phase=presolve start_s=1.0 end_s=1.8 effort=4000 "
        "wall_ms=800.0 effort_per_ms=5.000 found=1\n"
    )
    result = parse_log(log)
    assert len(result.sequential_samples) == 1
    assert result.sequential_samples[0].heuristic == "scylla"
    assert result.sequential_samples[0].effort == 4000
    assert len(result.heuristic_samples) == 1
    assert result.heuristic_samples[0].name == "scylla"


def test_sequential_zero_effort_line_parses():
    """Zero-effort [Sequential] lines (e.g. local_mip skipping a cold
    solve) are emitted so a human reader sees the skip; the drift script
    filters them before aggregation. The parser must accept them."""
    log = "[Sequential] heur=local_mip effort=0 wall_ms=0.1 effort_per_ms=0.000\n"
    result = parse_log(log)
    assert len(result.sequential_samples) == 1
    sample = result.sequential_samples[0]
    assert sample.heuristic == "local_mip"
    assert sample.effort == 0
    assert sample.effort_per_ms == 0.0


def test_worker_counts_are_read_off_the_solving_block():
    """The effective worker count is a property of the run host, not of any
    options file — the harness deliberately does not pin `threads` — so this
    line is the only record a benchmark run leaves of what it ran at."""
    log = (
        "Solving MIP model with:\n"
        "   31 rows\n"
        "   42 cols (28 binary, 0 integer, 0 implied int., 14 continuous, 0 domain fixed)\n"
        "   91 nonzeros\n"
        "   Thread count 16 (of 32 threads). Using 8 max workers. Parallel search on\n"
    )
    result = parse_log(log)
    # `thread_count` is HiGHS's pool size, which is what our presolve
    # heuristics run at; `max_workers` is B&B's parallel-search cap.
    assert result.thread_count == 16
    assert result.hardware_threads == 32
    assert result.max_workers == 8


def test_worker_counts_are_none_when_the_line_is_absent():
    result = parse_log("Solving report\n  Status            Optimal\n")
    assert result.thread_count is None
    assert result.hardware_threads is None
    assert result.max_workers is None


def test_a_single_worker_run_parses():
    """`Parallel search off` is the reproducibility setting, not a defect."""
    log = (
        "   Thread count 1 (of 12 threads). Using 1 max workers. Parallel search off\n"
    )
    result = parse_log(log)
    assert (result.thread_count, result.hardware_threads, result.max_workers) == (
        1,
        12,
        1,
    )


def _mip_line(time_s: float, obj: float) -> str:
    """One 'T'-source MIP log line: a new incumbent `obj` found at `time_s`."""
    return (
        f" T       0       0         0   0.00%   0               {obj}"
        f"              inf        0      0      0         0     {time_s}s\n"
    )


def test_a_killed_run_is_flagged_and_named():
    """The runner's marker is what separates a truncated run from an empty one.

    Without it a killed run parses as a clean solve that found nothing: same
    empty `status`, same infinite `primal_bound`. The metrics are identical
    either way -- but the two are different claims and reports say so.
    """
    result = parse_log(
        "Running HiGHS 1.15.1 (git hash: x): c\n"
        + _mip_line(10.0, 12.0)
        + "\n--- runner ---\nTIMEOUT: process killed after 1020.0s\n"
    )
    assert result.killed
    assert result.killed_after == 1020.0
    assert result.status == "Killed (timeout)"
    # The incumbent printed before the kill survives: it is real measured data.
    assert len(result.incumbents) == 1
    assert result.time_to_first_feasible == 10.0


def test_a_bare_timeout_stub_still_parses_as_killed():
    """Logs predating the runner keeping partial output are this one line."""
    result = parse_log("TIMEOUT: process killed after 1020.0s\n")
    assert result.killed
    assert result.killed_after == 1020.0
    assert result.status == "Killed (timeout)"
    assert result.incumbents == []


def test_a_clean_run_is_not_flagged_as_killed():
    result = parse_log("Solving report\n  Status            Optimal\n")
    assert not result.killed
    assert result.killed_after is None
    assert result.status == "Optimal"


def test_a_real_status_outranks_the_kill_marker():
    """Belt and braces: HiGHS's own word wins if a log carries both."""
    result = parse_log(
        "  Status            Time limit reached\n"
        "TIMEOUT: process killed after 1020.0s\n"
    )
    assert result.killed
    assert result.status == "Time limit reached"


def test_primal_integral_ignores_incumbents_past_the_time_limit():
    """A killed run keeps printing past the horizon the integral is measured over.

    Integrating those later points in also drags `prev_time` beyond the limit,
    making the remainder term negative -- so the run would score *better* than
    an identical one whose solution landed inside the window.
    """
    killed = parse_log(
        "Running HiGHS 1.15.1 (git hash: x): c\n"
        + _mip_line(10.0, 12.0)
        + _mip_line(300.0, 11.0)
        + _mip_line(900.0, 10.0)  # after the 600s horizon: must not count
        + "\nTIMEOUT: process killed after 1020.0s\n"
    )
    truncated = parse_log(
        "Running HiGHS 1.15.1 (git hash: x): c\n"
        + _mip_line(10.0, 12.0)
        + _mip_line(300.0, 11.0)
    )
    # 1.0*10 (no solution yet) + 0.2*290 + 0.1*300
    assert killed.primal_integral(600.0, 10.0) == pytest.approx(98.0)
    assert killed.primal_integral(600.0, 10.0) == truncated.primal_integral(600.0, 10.0)


# ---------------------------------------------------------------------------
# `[HeurSol]` — the per-offered-solution trace (#106)
# ---------------------------------------------------------------------------


def _heursol(name, dispatch, worker, effort_at, wall_ms, obj, accepted):
    return (
        f"[HeurSol] name={name} dispatch={dispatch} worker={worker} "
        f"effort_at={effort_at} wall_ms={wall_ms} obj={obj} accepted={accepted}\n"
    )


def _heur(name, effort, found=1, phase="presolve", nnz=100, abandoned=None):
    """One `[Heur]` line.

    `abandoned=None` omits the field, which is the *legacy* form: every log
    written before #119 — including the whole #113 probe tree — looks like
    this.  That is the default on purpose, so the cases that are not about
    the field keep exercising the shape most archived logs have.
    """
    line = (
        f"[Heur] name={name} phase={phase} start_s=0.100 end_s=0.200 "
        f"effort={effort} wall_ms=100.0 effort_per_ms=1.0 found={found}"
    )
    if nnz is not None:
        line += f" nnz={nnz}"
    if abandoned is not None:
        line += f" abandoned_setup={abandoned}"
    return line + "\n"


def _threads(n=1):
    return f"   Thread count {n} (of 32 threads). Using {n} max workers. Parallel search off\n"


# A real `log_dev_level=3` solve of `lseu.mps` by this branch's binary,
# trimmed to the lines the parser reads.  It is the shape that matters and
# the one a synthetic fixture keeps getting wrong: the trace exists **only**
# at dev level 3, and at dev level 3 HiGHS prints the `Nonzeros :` block
# instead of the one-line `MIP <name> has ... nonzeros;` header that
# `_MODEL_HEADER_RE` reads.  So `num_nonzeros` is None here by construction,
# `Nonzeros : 309` is the pre-presolve model, and `[Heur] nnz=242` — the
# post-presolve MIP matrix the patience options are denominated in — is the
# only correct source in the file.
_REAL_DEV_LOG = """Running HiGHS 1.15.1 (git hash: n/a)
mip-heuristics patch active
Coefficient ranges:
  Matrix [1e+00, 5e+02]
MIP        : lseu
Rows       : 28
Cols       : 89
Nonzeros   : 309
Integer    : 89 (89 binary)
   Thread count 16 (of 32 threads). Using 1 max workers. Parallel search off
[HeurSol] name=fj dispatch=1 worker=3 effort_at=500012 wall_ms=5.2 obj=1284 accepted=1
[HeurSol] name=fj dispatch=1 worker=3 effort_at=1000024 wall_ms=9.1 obj=1178 accepted=1
[Heur] name=fj phase=presolve start_s=0.002 end_s=0.289 effort=9272245 wall_ms=287.1 \
effort_per_ms=32297.795 found=1 nnz=242
[HeurSol] name=fpr dispatch=2 worker=0 effort_at=4356 wall_ms=1.1 obj=1150 accepted=1
[Heur] name=fpr phase=presolve start_s=0.289 end_s=0.297 effort=747887 wall_ms=7.5 \
effort_per_ms=99662.127 found=1 nnz=242
"""


def test_real_dev_log_fixture_has_no_model_header():
    """The premise of the `[Heur] nnz=` field, pinned so it cannot rot.

    If HiGHS ever printed the one-line header at dev level 3 as well, this
    would fail and the fallback ordering below could be revisited.
    """
    result = parse_log(_REAL_DEV_LOG)
    assert result.num_nonzeros is None
    assert result.heursol_samples, "fixture must carry a trace"


def test_normalized_gaps_work_on_a_real_dev_log():
    """The case that matters: `[HeurSol]` and `num_nonzeros` are mutually
    exclusive, so before the `nnz=` field this raised on every log that had
    a trace at all."""
    traces = {t.name: t for t in parse_log(_REAL_DEV_LOG).dispatch_traces()}
    assert traces["fj"].nnz == 242  # not 309, which is the pre-presolve model
    # fj's patience option is per-worker scoped, so the gaps need no scaling.
    assert traces["fj"].gap_scale == 1
    assert traces["fj"].normalized_gaps() == [500012 / 242, 500012 / 242]
    # fpr's is whole-dispatch scoped, so they are scaled by the 16 workers.
    assert traces["fpr"].gap_scale == 16
    assert traces["fpr"].normalized_gaps() == [4356 * 16 / 242]


def test_heur_lines_are_parsed_as_key_value():
    """Field order must not matter and an unknown key must be ignored — the
    line gained `nnz` mid-issue and archived logs must survive the next one."""
    log = (
        "[Heur] found=1 nnz=242 effort_per_ms=1.0 wall_ms=100.0 effort=500 "
        "end_s=0.2 start_s=0.1 phase=presolve name=fpr lane=7\n"
    )
    (sample,) = parse_log(log).heuristic_samples
    assert (sample.name, sample.phase, sample.effort, sample.nnz) == (
        "fpr",
        "presolve",
        500,
        242,
    )
    assert sample.found is True


def test_heur_line_without_nnz_still_parses():
    """Logs written before #106 carry no `nnz=`; they must not be dropped."""
    (sample,) = parse_log(_heur("fj", 400, nnz=None)).heuristic_samples
    assert sample.effort == 400
    assert sample.nnz is None


def test_heur_line_carries_abandoned_setup():
    """The #119 field, in both of its values.

    The pair is the whole point: same `effort`, same `found`, different
    field.  Before #119 these two lines were byte-identical, and the #113
    calibration binned both as barren.
    """
    (bailed,) = parse_log(_heur("fpr", 0, found=0, abandoned=1)).heuristic_samples
    (ran,) = parse_log(_heur("fpr", 0, found=0, abandoned=0)).heuristic_samples
    assert bailed.abandoned_setup is True
    assert ran.abandoned_setup is False
    assert (bailed.effort, bailed.found) == (ran.effort, ran.found)


def test_heur_line_without_abandoned_setup_parses_as_unknown():
    """Logs written before #119 carry no `abandoned_setup=`.

    They must not be dropped, and the missing field must read as `None`
    rather than `False`: on such a log the distinction is unobservable, not
    known-absent.  Consumers that only ask "did this dispatch search" get
    the pre-#119 answer from `None` being falsy, which is what makes an
    archived tree classify exactly as it did before.
    """
    (sample,) = parse_log(_heur("fj", 400)).heuristic_samples
    assert sample.effort == 400
    assert sample.abandoned_setup is None
    assert not sample.abandoned_setup


def test_heur_line_missing_a_required_key_is_dropped():
    log = "[Heur] name=fj phase=presolve start_s=0.1 end_s=0.2 effort=5 wall_ms=1.0\n"
    assert parse_log(log).heuristic_samples == []


def test_heur_line_accepts_negative_wall_ms_via_key_value():
    (sample,) = parse_log(
        "[Heur] name=fpr phase=presolve start_s=0.2 end_s=0.1 effort=5 "
        "wall_ms=-3.0 effort_per_ms=0.0 found=0 nnz=9\n"
    ).heuristic_samples
    assert sample.wall_ms == -3.0


def test_heursol_lines_parse_into_heursol_samples():
    log = (
        _heursol("fpr", 7, 0, 100, 1.5, 778.4590899999998, 1)
        + _heursol("fpr", 7, 1, 250, 2.5, 800.0, 0)
        + _heur("fpr", 900)
    )
    result = parse_log(log)
    assert len(result.heursol_samples) == 2

    first, second = result.heursol_samples
    assert first.name == "fpr"
    assert first.dispatch == 7
    assert first.worker == 0
    assert first.effort_at == 100
    assert first.wall_ms == 1.5
    assert first.objective == 778.4590899999998
    assert first.accepted is True
    assert second.accepted is False


def test_heursol_is_parsed_as_key_value_not_positionally():
    log = (
        "[HeurSol] accepted=1 obj=1.5 wall_ms=2.0 effort_at=42 "
        "worker=3 dispatch=9 name=scylla lane=17\n"
    )
    (sample,) = parse_log(log).heursol_samples
    assert (sample.name, sample.dispatch, sample.worker, sample.effort_at) == (
        "scylla",
        9,
        3,
        42,
    )
    assert sample.accepted is True


def test_heursol_line_missing_a_key_is_dropped():
    log = "[HeurSol] name=fj dispatch=1 worker=0 effort_at=5 wall_ms=1.0 obj=2.0\n"
    assert parse_log(log).heursol_samples == []


def test_heursol_accepts_negative_wall_ms_and_scientific_objective():
    """Same non-monotonic solver clock `[Heur] wall_ms` is signed for."""
    log = _heursol("local_mip", 3, 2, 7, -0.4, "-1.2345e+06", 0)
    (sample,) = parse_log(log).heursol_samples
    assert sample.wall_ms == -0.4
    assert sample.objective == -1234500.0


def test_heursol_worker_minus_one_is_accepted():
    """LocalMIP's cold-start publish runs off any worker slot."""
    log = _heursol("local_mip", 3, -1, 900, 0.5, 12.0, 1)
    (sample,) = parse_log(log).heursol_samples
    assert sample.worker == -1


def test_dispatch_traces_group_by_name_and_dispatch():
    """Ids are process-global: neither zero-based nor dense within a solve."""
    log = (
        _heursol("fpr", 41, 0, 10, 1.0, 5.0, 1)
        + _heursol("fpr", 41, 1, 20, 1.0, 4.0, 1)
        + _heur("fpr", 500)
        + _heursol("scylla", 44, 0, 30, 1.0, 3.0, 1)
        + _heur("scylla", 700)
    )
    traces = parse_log(log).dispatch_traces()
    assert [(t.name, t.dispatch, t.total_effort) for t in traces] == [
        ("fpr", 41, 500),
        ("scylla", 44, 700),
    ]
    assert len(traces[0].samples) == 2


def test_dispatch_trace_binds_to_its_own_heur_line_across_a_silent_dispatch():
    """A dispatch that offered nothing emits `[Heur]` and no `[HeurSol]`.

    Zipping the two lists positionally would then bind every later trace to
    the wrong total, so the parser binds on the `[Heur]` line itself: a
    dispatch's offers all precede it and follow the previous one for the
    same name.
    """
    log = (
        _heur("fj", 111, found=0)  # ran, offered nothing
        + _heursol("fpr", 2, 0, 10, 1.0, 5.0, 1)
        + _heur("fpr", 222)
        + _heursol("fpr_lp", 5, 0, 30, 1.0, 3.0, 1)
        + _heur("fpr_lp", 333, phase="dive")
        + _heursol("fpr_lp", 6, 0, 40, 1.0, 2.0, 1)
        + _heur("fpr_lp", 444, phase="dive")
    )
    traces = {
        (t.name, t.dispatch): t.total_effort for t in parse_log(log).dispatch_traces()
    }
    assert traces == {("fpr", 2): 222, ("fpr_lp", 5): 333, ("fpr_lp", 6): 444}


def test_dispatch_trace_without_a_heur_line_has_no_total():
    """A killed run is truncated mid-dispatch; the trace survives, the total does not."""
    log = _heursol("fpr", 2, 0, 10, 1.0, 5.0, 1)
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.total_effort is None
    assert trace.stale_effort is None
    assert "truncated log" in trace.stale_effort_unavailable_reason


def test_productive_and_stale_effort_sum_over_workers():
    """Productive effort is each worker's charge at *its* last acceptance."""
    log = (
        _heursol("local_mip", 3, 0, 100, 1.0, 9.0, 1)
        + _heursol("local_mip", 3, 1, 150, 1.0, 8.0, 1)
        + _heursol("local_mip", 3, 0, 400, 1.0, 7.0, 1)
        + _heursol("local_mip", 3, 1, 900, 1.0, 6.0, 0)  # refused: not productive
        + _heur("local_mip", 2000)
    )
    (trace,) = parse_log(log).dispatch_traces()
    assert len(trace.accepted_samples) == 3
    assert trace.productive_effort == 400 + 150
    assert trace.stale_effort == 2000 - 550


def test_stale_effort_floors_at_zero():
    """LocalMIP charges its cold-start construction sweep to the dispatch
    total but to no worker's counter, so a small mismatch is genuine."""
    log = _heursol("local_mip", 3, 0, 5000, 1.0, 9.0, 1) + _heur("local_mip", 1000)
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.stale_effort == 0


def test_stale_effort_is_withheld_for_scylla():
    """`[Heur] effort` takes the full PDLP cost; the per-worker counter
    `effort_at` reports takes it divided by the worker count.  Measured on a
    `gt2` dispatch where *every* offer was accepted, the subtraction still
    called 90% of the dispatch improvement-free — a number a calibration
    would act on, so it is withheld rather than floored."""
    log = _heursol("scylla", 3, 0, 100, 1.0, 9.0, 1) + _heur("scylla", 900_000)
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.stale_effort is None
    assert "not in the same unit" in trace.stale_effort_unavailable_reason
    # The quantities that *are* differences within that one counter survive.
    assert trace.productive_effort == 100
    assert trace.normalized_gaps(nnz=10) == [10.0]


def test_no_acceptance_makes_the_whole_dispatch_stale():
    log = _heursol("fj", 1, 0, 100, 1.0, 9.0, 0) + _heur("fj", 800, found=0)
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.productive_effort == 0
    assert trace.stale_effort == 800


def test_acceptance_gaps_are_taken_within_a_worker():
    """Two workers interleave in the log; their counters must not be subtracted
    from one another."""
    log = (
        _heursol("fpr", 4, 0, 100, 1.0, 9.0, 1)
        + _heursol("fpr", 4, 1, 30, 1.0, 8.0, 1)
        + _heursol("fpr", 4, 0, 250, 1.0, 7.0, 1)
        + _heursol("fpr", 4, 1, 90, 1.0, 6.0, 1)
        + _heur("fpr", 1000)
    )
    (trace,) = parse_log(log).dispatch_traces()
    assert sorted(trace.acceptance_gaps()) == sorted([100, 150, 30, 60])
    assert sorted(trace.acceptance_gaps(include_first=False)) == sorted([150, 60])


def test_gap_scale_follows_the_option_scope_not_the_counter():
    """`fj_patience` and `scylla_patience` arm a worker gate at their face
    value; `fpr_patience` and `local_mip_patience` are whole-dispatch and are
    divided by
    the worker count into `worker_stale`.  Reading a p90 off an unscaled
    fpr gap at the probe's 16 workers would ship a default 16x too tight."""
    body = _threads(16)
    for name in ("fj", "fpr", "local_mip", "scylla"):
        body += _heursol(name, 10, 0, 320, 1.0, 1.0, 1) + _heur(name, 999, nnz=32)
    traces = {t.name: t for t in parse_log(body).dispatch_traces()}
    assert traces["fj"].gap_scale == 1
    assert traces["scylla"].gap_scale == 1
    assert traces["fpr"].gap_scale == 16
    assert traces["local_mip"].gap_scale == 16
    assert traces["fj"].normalized_gaps() == [10.0]
    assert traces["fpr"].normalized_gaps() == [160.0]


def test_gap_scale_refuses_fpr_lp():
    """The dive heuristic has no patience option, so its gaps have no scope."""
    log = (
        _threads(4)
        + _heursol("fpr_lp", 5, 0, 10, 1.0, 1.0, 1)
        + _heur("fpr_lp", 20, phase="dive")
    )
    (trace,) = parse_log(log).dispatch_traces()
    with pytest.raises(ValueError, match=r"no mip_heuristic_.*_patience option"):
        trace.normalized_gaps()


def test_gap_scale_refuses_a_dispatch_scoped_option_without_a_worker_count():
    """Wrong by a factor of N is worse than absent."""
    log = _heursol("fpr", 4, 0, 100, 1.0, 9.0, 1) + _heur("fpr", 1000)
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.workers is None
    with pytest.raises(ValueError, match="Thread count"):
        trace.normalized_gaps()
    # The per-worker-scoped heuristics are unaffected by a missing count.
    log = _heursol("fj", 4, 0, 100, 1.0, 9.0, 1) + _heur("fj", 1000, nnz=10)
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.normalized_gaps() == [10.0]


def test_nnz_prefers_the_heur_field_over_the_model_header():
    """The header is the pre-presolve model; the field is the matrix the
    heuristics search and the patience options are denominated in."""
    log = (
        "MIP lseu has 28 rows; 89 cols; 309 nonzeros; 89 integer variables (89 binary)\n"
        + _threads(1)
        + _heursol("fj", 4, 0, 242, 1.0, 9.0, 1)
        + _heur("fj", 1000, nnz=242)
    )
    result = parse_log(log)
    assert result.num_nonzeros == 309
    (trace,) = result.dispatch_traces()
    assert trace.nnz == 242
    assert trace.normalized_gaps() == [1.0]
    assert trace.normalized_gaps(309) == [242 / 309]


def test_nnz_falls_back_to_the_model_header_for_a_pre_106_log():
    log = (
        "MIP lseu has 28 rows; 89 cols; 309 nonzeros; 89 integer variables (89 binary)\n"
        + _threads(1)
        + _heursol("fj", 4, 0, 309, 1.0, 9.0, 1)
        + _heur("fj", 1000, nnz=None)
    )
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.nnz == 309
    assert trace.normalized_gaps() == [1.0]


def test_normalized_gaps_refuse_and_name_the_log_shape_when_nnz_is_unavailable():
    log = (
        _threads(1) + _heursol("fj", 4, 0, 10, 1.0, 9.0, 1) + _heur("fj", 100, nnz=None)
    )
    (trace,) = parse_log(log).dispatch_traces()
    assert trace.nnz is None
    with pytest.raises(ValueError) as excinfo:
        trace.normalized_gaps()
    message = str(excinfo.value)
    assert "no nnz= field" in message
    assert "log_dev_level=3" in message


def test_heursol_absent_from_a_log_without_dev_level_3():
    """The line is `kVerbose`, like `[Heur]` and `[Sequential]`."""
    log = "Solving report\n  Status            Optimal\n"
    result = parse_log(log)
    assert result.heursol_samples == []
    assert result.dispatch_traces() == []


def test_presolve_only_log_shape_parses_its_traces():
    """`mip_heuristic_presolve_only` exits after the chain, before the root LP.

    The run leaves a Solving report with a finite primal bound, `Nodes 0`
    and a free-form status, and returns a non-zero exit code — none of which
    the parser keys on.  It is the log shape the tuning target runner
    scores, so the traces have to survive it.
    """
    log = (
        " H       0       0         0   0.00%   inf             1.5e+01 "
        "            inf        0      0      0         0     0.0s\n"
        + _threads(1)
        + _heursol("fj", 1, 0, 100, 1.0, 15.0, 1)
        + _heur("fj", 400, nnz=50)
        + _heursol("fpr", 2, 0, 50, 1.0, 15.0, 1)
        + _heur("fpr", 900, nnz=50)
        + "Solving report\n"
        "  Status            Solution limit reached\n"
        "  Primal bound      15\n"
        "  Dual bound        -inf\n"
        "  Gap               inf\n"
        "  Nodes             0\n"
        "  LP iterations     0\n"
    )
    result = parse_log(log)
    assert result.status == "Solution limit reached"
    assert result.primal_bound == 15.0
    assert result.nodes == 0
    assert [
        (t.name, t.total_effort, t.productive_effort) for t in result.dispatch_traces()
    ] == [
        ("fj", 400, 100),
        ("fpr", 900, 50),
    ]
    # And the calibration quantity is reachable, which is the whole point.
    assert result.dispatch_traces()[0].normalized_gaps() == [2.0]
