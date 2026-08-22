"""Smoke tests for parse_highs_log."""

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
