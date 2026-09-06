"""Tests for the #107 selection analysis.

The pre-registered rule is a decision procedure, and the whole point of
writing it down before the search ran was that it not be applied by judgement
afterwards.  These pin the parts that could quietly drift: the reduction from
irace's 16-parameter encoding to the eight numbers, the tie-break, and the
cross-lambda stability verdict.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_irace import (
    Configuration,
    LambdaResult,
    NotFinished,
    collapse,
    read_elites,
    select,
    stability,
)


def _row(**kw):
    """An irace elite row, defaulting to everything enabled and gated."""
    row = {
        "fj": "1",
        "fpr": "1",
        "local_mip": "1",
        "scylla": "1",
        "fj_effort": "0.5",
        "fpr_effort": "12.0",
        "local_mip_effort": "14.0",
        "scylla_effort": "3.0",
        "fj_gated": "1",
        "fpr_gated": "1",
        "local_mip_gated": "1",
        "scylla_gated": "1",
        "fj_patience": "0.12",
        "fpr_patience": "3.0",
        "local_mip_patience": "3.4",
        "scylla_patience": "0.7",
        "irace_id": "1",
    }
    row.update(kw)
    return row


def test_collapse_reduces_to_the_eight_numbers():
    c = collapse(_row())
    assert c.efforts["fpr"] == 12.0
    assert c.patiences["scylla"] == 0.7
    assert c.suite == "fj,fpr,local_mip,scylla"


def test_enabled_zero_means_the_heuristic_does_not_run():
    """`<h> == 0` is how a configurator reaches effort 0; the sampled effort
    beside it is meaningless and must not survive the reduction."""
    c = collapse(_row(fj="0", fj_effort="0.5"))
    assert c.efforts["fj"] == 0.0
    assert "fj" not in c.enabled
    assert c.suite == "fpr,local_mip,scylla"


def test_gated_zero_means_no_staleness_gate():
    """`<h>_gated == 0` selects patience 0, which is *no gate at all* rather
    than a very small one — a distinct configuration (#113)."""
    c = collapse(_row(fpr_gated="0", fpr_patience="3.0"))
    assert c.patiences["fpr"] == 0.0
    # It does not disable the heuristic: effort is untouched.
    assert c.efforts["fpr"] == 12.0


def test_na_is_read_as_zero():
    """irace writes NA for a conditional parameter whose condition did not
    hold, which means the same thing as the switch being off."""
    c = collapse(_row(scylla="0", scylla_effort="NA", scylla_patience="NA"))
    assert c.efforts["scylla"] == 0.0
    assert c.patiences["scylla"] == 0.0


def test_tie_break_prefers_fewer_heuristics():
    """The pre-registered rule: when survivors are statistically
    indistinguishable, take the simpler one.  irace's own ordering is not the
    tie-break, so a two-heuristic survivor beats a four-heuristic one even
    when irace listed the latter first."""
    four = collapse(_row())
    two = collapse(_row(local_mip="0", scylla="0"))
    result = select([four, two], "1_600")
    assert result.selected is two
    assert result.tie is True
    assert result.notes


def test_tie_break_then_prefers_lower_total_effort():
    a = collapse(_row(local_mip="0", scylla="0", fpr_effort="12.0"))
    b = collapse(_row(local_mip="0", scylla="0", fpr_effort="80.0"))
    assert select([b, a], "1_600").selected is a


def test_single_survivor_is_not_reported_as_a_tie():
    result = select([collapse(_row())], "1_600")
    assert result.tie is False
    assert result.notes == []


def _result(tag: str, config: Configuration) -> LambdaResult:
    return LambdaResult(tag=tag, survivors=[config], selected=config)


def test_stability_across_the_lambda_sweep():
    same = collapse(_row())
    ok, message = stability([_result(t, same) for t in ("1_1200", "1_600", "1_300")])
    assert ok
    assert "stable" in message


def test_instability_is_reported_not_resolved():
    """If the three lambdas disagree, that is the result.  Nothing here may
    quietly pick the middle one."""
    four = collapse(_row())
    two = collapse(_row(local_mip="0", scylla="0"))
    ok, message = stability(
        [_result("1_1200", four), _result("1_600", two), _result("1_300", four)]
    )
    assert not ok
    assert "UNSTABLE" in message
    assert "1_600" in message


def test_an_in_progress_search_is_reported_not_crashed(tmp_path, monkeypatch):
    """`getFinalElites` indexes the last entry of `allElites`, which is empty
    until an iteration completes, so an in-progress search errors in R.

    The state file exists and grows from the first experiment, so "the file
    is there" is not evidence of a result — this has to be distinguishable
    from a broken run, since the analysis gets pointed at partial state while
    a search is still going.
    """

    class _Out:
        returncode = 1
        stdout = ""
        stderr = "Error in x[[length(x)]] : attempt to select less than one element"

    monkeypatch.setattr("analyze_irace.subprocess.run", lambda *a, **k: _Out())
    with pytest.raises(NotFinished, match="no iteration has completed"):
        read_elites(tmp_path / "irace.Rdata")


def test_other_r_failures_are_not_swallowed(tmp_path, monkeypatch):
    class _Out:
        returncode = 1
        stdout = ""
        stderr = "Error: cannot open file"

    monkeypatch.setattr("analyze_irace.subprocess.run", lambda *a, **k: _Out())
    with pytest.raises(RuntimeError, match="cannot open file"):
        read_elites(tmp_path / "irace.Rdata")
