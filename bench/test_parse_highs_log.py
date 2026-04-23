"""Smoke tests for parse_highs_log."""

from parse_highs_log import parse_log


def test_empty_log_returns_default_result():
    result = parse_log("")
    assert result.status == ""
    assert result.incumbents == []


def test_portfolio_and_fpr_lp_portfolio_effort_lines_both_parse():
    """Both [Portfolio] and [FprLpPortfolio] effort lines are captured.

    The presolve bandit (src/portfolio.cpp) emits `[Portfolio]` while the
    B&B-dive bandit (src/fpr_lp.cpp) emits `[FprLpPortfolio]`. Both share a
    line layout via the shared `log_bandit_arm` helper in
    src/bandit_runner.h; the parser must tag each sample with its origin.
    """
    log = (
        "[Portfolio] arm=FPR effort=1234 wall_ms=2.5 effort_per_ms=494 reward=2\n"
        "[FprLpPortfolio] arm=lp_center effort=5678 wall_ms=10.2 effort_per_ms=557 reward=1\n"
    )
    result = parse_log(log)
    assert len(result.effort_samples) == 2

    presolve, dive = result.effort_samples
    assert presolve.portfolio_tag == "Portfolio"
    assert presolve.arm == "FPR"
    assert presolve.effort == 1234
    assert presolve.wall_ms == 2.5
    assert presolve.effort_per_ms == 494.0
    assert presolve.reward == 2

    assert dive.portfolio_tag == "FprLpPortfolio"
    assert dive.arm == "lp_center"
    assert dive.effort == 5678
    assert dive.wall_ms == 10.2
    assert dive.effort_per_ms == 557.0
    assert dive.reward == 1


def test_effort_line_without_reward_is_backward_compatible():
    """Historical logs predate the `reward=` suffix (issue #68)."""
    log = "[Portfolio] arm=LocalMIP effort=42 wall_ms=1.0 effort_per_ms=42\n"
    result = parse_log(log)
    assert len(result.effort_samples) == 1
    sample = result.effort_samples[0]
    assert sample.arm == "LocalMIP"
    assert sample.reward is None
    assert sample.portfolio_tag == "Portfolio"


def test_sequential_lines_parse_into_sequential_samples():
    """`[Sequential]` lines feed kWeight* calibration (issue #71).

    They share no regex with `[Portfolio]` lines (different field names,
    no reward, no arm) and must not leak into `effort_samples`.
    """
    log = (
        "[Sequential] heur=fj effort=1000 wall_ms=5.0 effort_per_ms=200\n"
        "[Sequential] heur=fpr effort=2500 wall_ms=50.0 effort_per_ms=50\n"
        "[Sequential] heur=local_mip effort=3000 wall_ms=90.0 effort_per_ms=33\n"
        "[Sequential] heur=scylla effort=4000 wall_ms=800.0 effort_per_ms=5\n"
    )
    result = parse_log(log)
    assert result.effort_samples == []
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
