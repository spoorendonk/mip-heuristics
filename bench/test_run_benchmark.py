"""Tests for run_benchmark's config table, budget sweep and plan resolution."""

import os
import stat
import sys
from pathlib import Path

import pytest
from run_benchmark import (
    BUDGET_SUFFIX,
    CONFIG_SUITES,
    DEFAULT_BUDGET_SWEEP,
    SWEEP_EXEMPT,
    build_base_options,
    build_plan,
    config_options,
    expand_configs,
    find_ignored_config_warning,
    main,
    parse_budget,
    run_single,
    split_config,
    write_log,
    write_options_file,
)

PATCHED = "/build/bin/highs"
EXTERNAL = "/usr/bin/highs"


# --- the raise on an unknown name ------------------------------------------


def test_unknown_config_raises():
    """The silent-failure fix: `patchd` used to return {} and run defaults."""
    with pytest.raises(ValueError, match="unknown config 'patchd'"):
        config_options("patchd")


def test_unknown_config_message_lists_the_known_ones():
    with pytest.raises(ValueError) as exc:
        config_options("nope")
    for name in CONFIG_SUITES:
        assert name in str(exc.value)


def test_unknown_config_raises_from_the_sweep_too():
    with pytest.raises(ValueError, match="unknown config 'patchd'"):
        expand_configs(["patchd"], ["0.30"])


def test_unknown_config_raises_through_build_plan():
    with pytest.raises(ValueError, match="unknown config"):
        build_plan("all_opp", PATCHED, PATCHED)


# --- the seven-config table ------------------------------------------------


@pytest.mark.parametrize(
    "config,suite",
    [
        ("off", "off"),
        ("fj", "fj"),
        ("fpr", "fpr"),
        ("local_mip", "local_mip"),
        ("scylla", "scylla"),
        ("all", "all"),
    ],
)
def test_config_selects_its_suite_value(config, suite):
    assert config_options(config) == {"mip_heuristic_suite": suite}


def test_patched_is_an_alias_for_all():
    """Alias, not rename: `patched` now composes FJ+FPR+LocalMIP+*Scylla*."""
    assert config_options("patched") == config_options("all")
    assert config_options("patched")["mip_heuristic_suite"] == "all"


def test_vanilla_on_the_patched_binary_is_suite_off():
    assert config_options("vanilla") == {"mip_heuristic_suite": "off"}


def test_vanilla_on_an_external_binary_sets_nothing():
    """An unpatched binary has no mip_heuristic_* options to set."""
    assert config_options("vanilla", external_vanilla=True) == {}


def test_external_vanilla_does_not_leak_into_other_configs():
    assert config_options("fpr", external_vanilla=True) == {"mip_heuristic_suite": "fpr"}


# --- budget suffix parsing -------------------------------------------------


def test_split_config_round_trips():
    assert split_config("fpr") == ("fpr", None)
    assert split_config(f"fpr{BUDGET_SUFFIX}0.30") == ("fpr", "0.30")


def test_budget_is_kept_verbatim():
    """`0.30` must name fpr@e0.30, not the fpr@e0.3 a float round-trip gives."""
    assert parse_budget("0.30") == "0.30"
    assert parse_budget("1.00") == "1.00"


@pytest.mark.parametrize("bad", ["abc", "", "1.5", "-0.1"])
def test_bad_budget_raises(bad):
    with pytest.raises(ValueError):
        parse_budget(bad)


def test_swept_config_carries_the_effort_option():
    assert config_options(f"local_mip{BUDGET_SUFFIX}0.60") == {
        "mip_heuristic_suite": "local_mip",
        "mip_heuristic_presolve_effort": "0.60",
    }


def test_swept_config_with_a_bad_budget_raises():
    with pytest.raises(ValueError, match="not a number"):
        config_options(f"fpr{BUDGET_SUFFIX}high")


# --- sweep expansion -------------------------------------------------------


def test_no_sweep_leaves_configs_untouched():
    assert expand_configs(["patched", "vanilla"], []) == (["patched", "vanilla"], [])


def test_sweep_crosses_configs_with_budgets():
    names, notices = expand_configs(["fpr", "scylla"], ["0.05", "0.30"])
    assert names == [
        f"fpr{BUDGET_SUFFIX}0.05",
        f"fpr{BUDGET_SUFFIX}0.30",
        f"scylla{BUDGET_SUFFIX}0.05",
        f"scylla{BUDGET_SUFFIX}0.30",
    ]
    assert notices == []


def test_default_sweep_is_five_budgets_per_config():
    """`fpr`, not `fj`: FJ's budget is fixed, so it is a sweep anchor row."""
    names, _ = expand_configs(["fpr"], list(DEFAULT_BUDGET_SWEEP))
    assert len(names) == len(DEFAULT_BUDGET_SWEEP) == 5
    assert names[0] == f"fpr{BUDGET_SUFFIX}{DEFAULT_BUDGET_SWEEP[0]}"


def test_default_sweep_contains_the_shipped_default():
    """A sweep always has the shipped configuration as one of its rows."""
    assert "0.30" in DEFAULT_BUDGET_SWEEP


def test_expansion_rejects_an_already_suffixed_config():
    with pytest.raises(ValueError, match="already carries"):
        expand_configs([f"fpr{BUDGET_SUFFIX}0.30"], ["0.05"])


def test_expansion_rejects_a_bad_budget_before_any_run():
    with pytest.raises(ValueError):
        expand_configs(["fpr"], ["0.05", "banana"])


# --- vanilla / off interaction with the sweep ------------------------------


@pytest.mark.parametrize("config", sorted(SWEEP_EXEMPT))
def test_exempt_configs_pass_through_the_sweep_once(config):
    """Neither runs a presolve heuristic, so N budgets would be N identical runs."""
    names, notices = expand_configs([config], ["0.05", "0.30", "1.00"])
    assert names == [config]
    assert len(notices) == 1
    assert config in notices[0]


def test_sweep_keeps_the_anchor_alongside_swept_configs():
    names, notices = expand_configs(["vanilla", "fpr"], ["0.05", "0.30"])
    assert names == ["vanilla", f"fpr{BUDGET_SUFFIX}0.05", f"fpr{BUDGET_SUFFIX}0.30"]
    assert len(notices) == 1


@pytest.mark.parametrize("config", sorted(SWEEP_EXEMPT))
def test_explicit_suffix_on_an_exempt_config_raises(config):
    """`vanilla@e0.30` is a directory name that would mean nothing."""
    with pytest.raises(ValueError, match="mip_heuristic_presolve_effort") as exc:
        config_options(f"{config}{BUDGET_SUFFIX}0.30")
    # The message says *why* this config is exempt, not just that it is.
    assert SWEEP_EXEMPT[config] in str(exc.value)


# --- plan resolution: base name decided once -------------------------------


def test_vanilla_takes_the_external_binary():
    plan = build_plan("vanilla", PATCHED, EXTERNAL)
    assert plan.binary == EXTERNAL
    assert plan.options == {}


def test_vanilla_takes_the_patched_binary_when_no_external_one_is_given():
    plan = build_plan("vanilla", PATCHED, PATCHED)
    assert plan.binary == PATCHED
    assert plan.options == {"mip_heuristic_suite": "off"}


def test_every_non_vanilla_config_takes_the_patched_binary():
    for config in CONFIG_SUITES:
        if config == "vanilla":
            continue
        assert build_plan(config, PATCHED, EXTERNAL).binary == PATCHED


def test_swept_plan_keeps_the_base_name_and_the_directory_name():
    plan = build_plan(f"scylla{BUDGET_SUFFIX}0.15", PATCHED, EXTERNAL)
    assert plan.base == "scylla"
    assert plan.name == f"scylla{BUDGET_SUFFIX}0.15"
    assert plan.binary == PATCHED
    assert plan.options == {
        "mip_heuristic_suite": "scylla",
        "mip_heuristic_presolve_effort": "0.15",
    }


def test_swept_names_are_usable_as_directory_names():
    names, _ = expand_configs(["fpr"], list(DEFAULT_BUDGET_SWEEP))
    for name in names:
        assert "/" not in name and not name.startswith(".")


# --- base options: what the harness declines to set by default -------------


def test_base_options_are_empty_by_default():
    """No `threads` (collapses parallelism) and no `log_dev_level` (not free)."""
    assert build_base_options(None, False, []) == {}


def test_threads_is_only_set_when_asked_for():
    assert build_base_options(4, False, []) == {"threads": "4"}


def test_dev_log_sets_level_three():
    """Level 3 is what makes [Heur]/[Native]/[Root]/[Sequential] visible."""
    assert build_base_options(None, True, []) == {"log_dev_level": "3"}


def test_extra_options_are_parsed_and_stripped():
    assert build_base_options(None, False, [" mip_heuristic_effort = 0.10 "]) == {
        "mip_heuristic_effort": "0.10"
    }


def test_malformed_extra_option_is_skipped_not_crashed(capsys):
    assert build_base_options(None, False, ["nonsense"]) == {}
    assert "malformed" in capsys.readouterr().err


# --- a run that never solved is not a result -------------------------------


def _fake_binary(tmp_path: Path, exit_code: int) -> str:
    """A stand-in for `highs` that prints a banner and exits `exit_code`."""
    path = tmp_path / "fake_highs"
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('Running HiGHS')\n"
        f"sys.exit({exit_code})\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _run(tmp_path: Path, exit_code: int) -> tuple[bool, Path]:
    out = tmp_path / "results"
    _, _, _, ok = run_single(
        _fake_binary(tmp_path, exit_code), "model.mps", "model", "fpr", 0, 1.0, str(out)
    )
    return ok, out / "fpr" / "seed0"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_a_solved_run_writes_the_log(tmp_path: Path, exit_code: int):
    """0 = kOk, 1 = kWarning — "Time limit reached", the normal outcome."""
    ok, seed_dir = _run(tmp_path, exit_code)
    assert ok
    assert (seed_dir / "model.log").read_text().startswith("Running HiGHS")
    assert not (seed_dir / "model.log.err").exists()


def test_a_run_that_never_solved_is_parked_as_err(tmp_path: Path, capsys):
    """255 = kError: e.g. an options file HiGHS rejected.

    `analyze_results.py` globs `*.log` and `--skip-existing` treats a
    non-empty one as done, so writing this into `model.log` would both score
    as a legitimately infeasible instance and cement itself across a resume.
    """
    ok, seed_dir = _run(tmp_path, 255)
    assert not ok
    assert not (seed_dir / "model.log").exists()
    assert "exited 255 without solving" in (seed_dir / "model.log.err").read_text()
    assert "without solving" in capsys.readouterr().err


def test_a_failure_clears_a_previous_runs_log(tmp_path: Path):
    """Otherwise a resumed run's stale success masks the new failure."""
    seed_dir = tmp_path / "results" / "fpr" / "seed0"
    seed_dir.mkdir(parents=True)
    (seed_dir / "model.log").write_text("stale success from an earlier run\n")
    ok, _ = _run(tmp_path, 255)
    assert not ok
    assert not (seed_dir / "model.log").exists()


def test_options_file_is_written_beside_the_log(tmp_path: Path):
    """The .opts file is the record of what a result was actually run at."""
    _run(tmp_path, 0)
    opts = (tmp_path / "results" / "fpr" / "seed0" / "model.opts").read_text()
    assert "random_seed = 0" in opts


def test_write_options_file_round_trips(tmp_path: Path):
    path = tmp_path / "o.opts"
    write_options_file({"mip_heuristic_suite": "fpr", "random_seed": "3"}, str(path))
    assert path.read_text() == "mip_heuristic_suite = fpr\nrandom_seed = 3\n"
    assert os.path.exists(path)


def test_a_successful_retry_clears_a_stale_err(tmp_path: Path):
    """Exactly one of .log / .err describes what is in the tree now.

    Otherwise a --skip-existing resume over a partially failed campaign
    leaves .err files whose instances have since succeeded, and .err degrades
    from "failed" to "failed at some point".
    """
    log = tmp_path / "model.log"
    (tmp_path / "model.log.err").write_text("failure from an earlier attempt\n")
    write_log(str(log), "a good run\n")
    assert log.read_text() == "a good run\n"
    assert not (tmp_path / "model.log.err").exists()


# --- a run that solved but ignored its configuration -----------------------


def test_the_fail_open_warning_is_detected():
    """HiGHS accepts an unknown suite *value* and runs all four anyway.

    Verbatim text from src/mode_dispatch.cpp's run_presolve.
    """
    output = (
        "Running HiGHS 1.15.1\n"
        'WARNING: Unknown mip_heuristic_suite value "of"; running all heuristics.\n'
        "  Status            Optimal\n"
    )
    assert "Unknown mip_heuristic_suite" in find_ignored_config_warning(output)


def test_the_fj_taken_away_warning_is_detected():
    """`suite=fj` + run_feasibility_jump=false measures vanilla-minus-FJ."""
    output = (
        'WARNING: mip_heuristic_suite="fj" selects only FeasibilityJump, which '
        "mip_heuristic_run_feasibility_jump=false disables; no heuristic will "
        "run. Use mip_heuristic_suite=off for a vanilla-equivalent run.\n"
    )
    assert find_ignored_config_warning(output) is not None


def test_an_ordinary_log_trips_nothing():
    assert find_ignored_config_warning("Running HiGHS\n  Status  Optimal\n") is None


def test_a_run_that_ignored_its_config_is_parked_as_err(tmp_path: Path, capsys):
    """Exit 0 and a complete log, so only this check can catch it."""
    binary = tmp_path / "fake_highs"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('Running HiGHS')\n"
        'print(\'WARNING: Unknown mip_heuristic_suite value "of"; '
        "running all heuristics.')\n"
        "sys.exit(0)\n"
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "results"
    _, _, _, ok = run_single(
        str(binary), "model.mps", "model", "off", 0, 1.0, str(out)
    )
    assert not ok
    seed_dir = out / "off" / "seed0"
    assert not (seed_dir / "model.log").exists()
    assert "ignored its configuration" in (seed_dir / "model.log.err").read_text()
    assert "ignored its configuration" in capsys.readouterr().err


# --- two names, one configuration ------------------------------------------


def test_budgets_differing_only_as_strings_are_one_configuration():
    """`0.3` and `0.30` are two directories but one number to HiGHS."""
    a = build_plan(f"fpr{BUDGET_SUFFIX}0.3", PATCHED, PATCHED)
    b = build_plan(f"fpr{BUDGET_SUFFIX}0.30", PATCHED, PATCHED)
    assert a.name != b.name  # distinct output directories
    assert a.identity == b.identity  # identical solver behaviour


def test_genuinely_different_budgets_are_different_configurations():
    a = build_plan(f"fpr{BUDGET_SUFFIX}0.30", PATCHED, PATCHED)
    b = build_plan(f"fpr{BUDGET_SUFFIX}0.60", PATCHED, PATCHED)
    assert a.identity != b.identity


def test_identity_separates_configs_that_differ_only_by_binary():
    assert (
        build_plan("vanilla", PATCHED, EXTERNAL).identity
        != build_plan("off", PATCHED, EXTERNAL).identity
    )


def test_aliases_share_one_identity():
    assert build_plan("patched", PATCHED, PATCHED).identity == (
        build_plan("all", PATCHED, PATCHED).identity
    )


# --- main()'s own guards ---------------------------------------------------


def _main(tmp_path: Path, monkeypatch, *argv: str) -> None:
    """Run main() over an empty instance list — every guard, no solves."""
    instances = tmp_path / "none.txt"
    instances.write_text("")
    monkeypatch.setattr(
        sys, "argv",
        ["run_benchmark.py", "--instances", str(instances), "--binary", sys.executable,
         "--data-dir", str(tmp_path), "--output", str(tmp_path / "out"), *argv],
    )
    main()


def test_main_exits_2_on_an_unknown_config(tmp_path: Path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _main(tmp_path, monkeypatch, "--configs", "patchd")
    assert exc.value.code == 2
    assert "unknown config 'patchd'" in capsys.readouterr().err


def test_main_exits_2_on_a_duplicate_config(tmp_path: Path, monkeypatch, capsys):
    """One config name is one output directory, so a repeat is one run twice."""
    with pytest.raises(SystemExit) as exc:
        _main(tmp_path, monkeypatch, "--configs", "fpr", "fpr")
    assert exc.value.code == 2
    assert "duplicate config" in capsys.readouterr().err


def test_main_exits_2_on_an_explicitly_suffixed_exempt_config(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _main(tmp_path, monkeypatch, "--configs", f"vanilla{BUDGET_SUFFIX}0.30")
    assert exc.value.code == 2


def test_main_warns_that_vanilla_and_off_are_the_same_run(tmp_path, monkeypatch, capsys):
    """Without --vanilla-binary they are one configuration under two names."""
    _main(tmp_path, monkeypatch, "--configs", "vanilla", "off")
    err = capsys.readouterr().err
    assert "identical" in err and "duplicated work" in err


def test_main_warns_on_string_duplicate_budgets(tmp_path, monkeypatch, capsys):
    _main(tmp_path, monkeypatch, "--configs", "fpr", "--budget-sweep", "0.3", "0.30")
    assert "identical" in capsys.readouterr().err


def test_main_warns_when_a_config_overrides_an_extra_option(tmp_path, monkeypatch, capsys):
    _main(tmp_path, monkeypatch, "--configs", "fpr",
          "--extra-options", "mip_heuristic_suite=scylla")
    assert "is overridden by config 'fpr'" in capsys.readouterr().err


def test_main_warns_that_an_extra_random_seed_is_ignored(tmp_path, monkeypatch, capsys):
    """The seed names the output directory, so --seeds has to win."""
    _main(tmp_path, monkeypatch, "--configs", "fpr", "--extra-options", "random_seed=7")
    assert "random_seed" in capsys.readouterr().err


def test_main_reports_the_instrumentation_mode(tmp_path, monkeypatch, capsys):
    _main(tmp_path, monkeypatch, "--configs", "fpr")
    assert "Instrumentation: off" in capsys.readouterr().out
    _main(tmp_path, monkeypatch, "--configs", "fpr", "--dev-log")
    assert "log_dev_level=3" in capsys.readouterr().out


def test_main_accepts_the_documented_readme_sweep(tmp_path, monkeypatch, capsys):
    """The README reproduce block must not warn about duplicated work."""
    _main(tmp_path, monkeypatch,
          "--configs", "off", "fj", "fpr", "local_mip", "scylla", "all",
          "--budget-sweep", "0.05", "0.15", "0.30", "0.60", "1.00")
    captured = capsys.readouterr()
    assert "identical" not in captured.err
    assert "off@e" not in captured.out and "fj@e" not in captured.out
    assert "fpr@e0.05" in captured.out and "all@e1.00" in captured.out
