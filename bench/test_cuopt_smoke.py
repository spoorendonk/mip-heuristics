"""The smoke script's plan, resume test and ratio table (#238)."""

from __future__ import annotations

from pathlib import Path

import cuopt_smoke
import pytest
from cuopt_smoke import (
    ARMS,
    BASELINE,
    CONFIG,
    PERSONA_INSTANCES,
    TIME_LIMIT,
    Arm,
    build_parser,
    noise_seed,
    pending,
    pin_binary,
    plan,
    report_rows,
    run,
    solve_command,
    t_first,
)
from parse_highs_log import Incumbent, SolveResult


def solved(first: float | None) -> SolveResult:
    result = SolveResult()
    if first is not None:
        result.incumbents = [Incumbent(first, 1.0, "J", 0)]
    return result


def by_arm(solves: list) -> dict[str, list]:
    arms: dict[str, list] = {}
    for solve in solves:
        arms.setdefault(solve.arm.name, []).append(solve)
    return arms


def test_only_main_runs_the_noise_seed():
    arms = by_arm(plan(["a", "b"], [0], (*ARMS[:1], Arm("x"))))
    assert [s.seed for s in arms[BASELINE]] == [0, 1, 0, 1]
    assert [s.seed for s in arms["x"]] == [0, 0]


def test_personas_run_only_where_a_persona_fires():
    fires = next(iter(sorted(PERSONA_INSTANCES)))
    arms = by_arm(plan([fires, "no-persona"], [0], ARMS))
    assert [s.instance for s in arms["personas"]] == [fires]
    assert len(arms["ruin"]) == 2


def test_pending_skips_a_nonempty_log_and_keeps_an_empty_one(tmp_path: Path):
    arm = Arm("x")
    done, empty, missing = plan(["done", "empty", "missing"], [0], (arm,))
    for solve, text in ((done, "log"), (empty, "")):
        path = tmp_path / "x" / CONFIG / "seed0" / f"{solve.instance}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert pending([done, empty, missing], tmp_path) == [empty, missing]


def test_a_solve_names_its_options_and_one_instance(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SMOKE_WRAP", "xk lock --")
    args = build_parser().parse_args(["run", "--output", str(tmp_path)])
    solve = plan(["a", "b"], [3], (Arm("x", ("key=1",)),))[1]
    command = solve_command(solve, args)
    assert command[:3] == ["xk", "lock", "--"]
    listed = Path(command[command.index("--instances") + 1]).read_text()
    assert listed == "b\n"
    assert command[command.index("--seeds") + 1] == "3"
    assert command[command.index("--output") + 1] == str(tmp_path / "x")
    assert command[-2:] == ["--extra-options", "key=1"]


def test_a_run_without_an_incumbent_scores_the_limit():
    assert t_first(solved(None), "a") == TIME_LIMIT
    assert t_first(solved(2.5), "a") == 2.5


def test_a_late_incumbent_does_not_rank_below_none():
    assert t_first(solved(TIME_LIMIT * 2), "a") == TIME_LIMIT


def test_a_tree_keeps_the_binary_it_started_with(tmp_path: Path):
    first = tmp_path / "first"
    first.write_text("build 1")
    out = tmp_path / "out"
    pinned = pin_binary(str(first), out)
    first.write_text("build 2")
    assert Path(pinned).read_text() == "build 1"
    with pytest.raises(ValueError, match="another binary"):
        pin_binary(str(first), out)
    assert pin_binary(pinned, out) == pinned


def test_a_solve_that_left_no_log_fails_the_run(tmp_path: Path, monkeypatch):
    # run_benchmark exits 0 after a failed solve, so the exit code cannot tell.
    monkeypatch.setattr(cuopt_smoke.subprocess, "call", lambda _command: 0)
    binary = tmp_path / "highs"
    binary.write_text("build")
    args = build_parser().parse_args(
        ["run", "--output", str(tmp_path / "out"), "--binary", str(binary)]
    )
    assert run(plan(["a"], [0], (Arm("x"),)), args) == 4


def test_the_table_pairs_arms_on_their_seed_and_the_floor_on_the_noise_seed():
    instances = ["a", "b", "c"]
    results = {
        BASELINE: {
            0: {name: solved(3.0) for name in instances},
            noise_seed([0]): {name: solved(1.0) for name in instances},
        },
        "x": {0: {"a": solved(7.0), "b": solved(7.0), "c": solved(None)}},
    }

    def integral(r: SolveResult, _instance: str) -> float:
        return r.incumbents[0].time if r.incumbents else 2.0

    rows, notes = report_rows(results, [0], instances, integral)

    assert not notes
    assert len(rows) == 2
    floor, arm = rows
    assert floor.startswith("| noise floor (main seed 1) | 3 | ")
    assert "3 vs 3" in floor
    # T1st: (1 + 1) / (3 + 1) on all three instances of the floor.
    assert " 0.500 | 0.500 | " in floor
    # The arm is against seed 0 only; c has no incumbent, so 2 runs of 3 found one.
    assert arm.startswith("| x | 3 | ")
    assert "2 vs 3" in arm


def test_a_pairing_too_small_to_compare_says_so():
    results = {BASELINE: {0: {"a": solved(1.0)}}, "x": {0: {"a": solved(1.0)}}}
    rows, _notes = report_rows(results, [0], ["a"], lambda r, i: 1.0)
    assert rows == [
        "| x | 1 | n/a | n/a | n/a | n/a | 1 vs 1 | n/a | n/a | n/a | n/a |"
    ]


def test_a_planned_run_without_a_pair_is_named_and_an_unplanned_one_is_not():
    outside = "no-persona"
    fires = min(PERSONA_INSTANCES)
    results = {
        BASELINE: {0: {outside: solved(1.0), fires: solved(1.0)}},
        "personas": {0: {"other": solved(1.0)}},
        "ruin": {0: {outside: solved(1.0)}},
    }
    _rows, notes = report_rows(results, [0], [outside, fires], lambda r, i: 1.0)
    assert notes == [
        f"personas: no paired run on {fires}",
        f"ruin: no paired run on {fires}",
    ]
