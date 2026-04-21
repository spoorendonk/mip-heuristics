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
