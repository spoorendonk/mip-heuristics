"""Unit tests for bench/check_effort_drift.py."""

from __future__ import annotations

import os
import subprocess
import sys


SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_effort_drift.py")


def _make_log(heur_rates: dict[str, float]) -> str:
    """Build a minimal log body with one [Sequential] line per heuristic.

    `heur_rates` maps heuristic name → effort_per_ms. effort is pinned at
    10_000; wall_ms is derived so the line parses back to the requested
    rate without floating-point surprises.
    """
    lines = []
    for heur, rate in heur_rates.items():
        effort = 10_000
        wall_ms = effort / rate
        lines.append(
            f"[Sequential] heur={heur} effort={effort} wall_ms={wall_ms:.1f} effort_per_ms={rate:.0f}"
        )
    return "\n".join(lines) + "\n"


def _write_log(tmpdir: str, name: str, body: str) -> None:
    with open(os.path.join(tmpdir, name), "w") as f:
        f.write(body)


def _run(root: str, max_drift: float = 3.0) -> subprocess.CompletedProcess[str]:
    # Use `sys.executable` so the test picks up whichever interpreter is
    # driving pytest — no hard-coded `.venv/bin/python` path, which would
    # break on CI or a fresh clone without the local venv.
    return subprocess.run(
        [sys.executable, SCRIPT, root, "--max-drift", str(max_drift)],
        capture_output=True,
        text=True,
    )


def test_drift_within_threshold_passes(tmp_path):
    # Four heuristics all within 2× of each other — passes at 3× limit.
    rates = {"fj": 100.0, "fpr": 200.0, "local_mip": 150.0, "scylla": 80.0}
    _write_log(str(tmp_path), "a.log", _make_log(rates))
    res = _run(str(tmp_path))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK: drift within threshold." in res.stdout


def test_drift_exceeds_threshold_fails(tmp_path):
    # 20× spread between fj and scylla — must fail at default 3× limit.
    rates = {"fj": 1000.0, "fpr": 200.0, "local_mip": 150.0, "scylla": 50.0}
    _write_log(str(tmp_path), "a.log", _make_log(rates))
    res = _run(str(tmp_path))
    assert res.returncode == 2, res.stdout + res.stderr
    assert "FAIL" in res.stderr


def test_no_samples_returns_failure(tmp_path):
    _write_log(str(tmp_path), "empty.log", "this log has no sequential lines\n")
    res = _run(str(tmp_path))
    assert res.returncode == 1
    assert "No [Sequential] samples" in res.stderr


def test_missing_heuristic_prints_warning(tmp_path):
    # Only fj samples present → warning listing fpr/local_mip/scylla.
    rates = {"fj": 100.0}
    _write_log(str(tmp_path), "a.log", _make_log(rates))
    res = _run(str(tmp_path), max_drift=100.0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARNING" in res.stderr
    for missing in ("fpr", "local_mip", "scylla"):
        assert missing in res.stderr


def test_unknown_reference_arg_errors(tmp_path):
    rates = {"fj": 100.0, "fpr": 200.0, "local_mip": 150.0, "scylla": 80.0}
    _write_log(str(tmp_path), "a.log", _make_log(rates))
    res = subprocess.run(
        [sys.executable, SCRIPT, str(tmp_path), "--reference", "typo_name"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "ERROR" in res.stderr
    assert "typo_name" in res.stderr


def test_suggested_weights_are_proportional_to_effort_per_ms(tmp_path):
    # Weights are proportional to `effort_per_ms` (not its reciprocal) so
    # that equal-weight allocations yield equal wall-clock spend:
    #   wall_i = share_i / r_i, and share_i = budget * w_i / sum(w),
    #   so w_i ∝ r_i ⇒ wall_i = const.
    # Slowest-per-effort heuristic anchors w=1; faster ones scale up.
    rates = {"fj": 50.0, "fpr": 1000.0, "local_mip": 1000.0, "scylla": 1000.0}
    _write_log(str(tmp_path), "a.log", _make_log(rates))
    res = _run(str(tmp_path), max_drift=100.0)
    assert res.returncode == 0, res.stdout + res.stderr

    # Parse the tabular output — the suggested_w column is the rightmost.
    rows = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if not parts or parts[0] in {"heuristic", "Drift", "OK:", "FAIL:"}:
            continue
        rows[parts[0]] = float(parts[-1])

    # fj is the reference (slowest-per-effort heuristic) → w=1.
    assert abs(rows["fj"] - 1.0) < 1e-6
    # scylla is 20× faster per effort → w = 20.
    assert abs(rows["scylla"] - 20.0) < 1e-3
