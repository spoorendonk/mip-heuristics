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
