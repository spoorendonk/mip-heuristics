"""Tests for run_benchmark's config table, budget sweep and plan resolution."""

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import run_benchmark
from run_benchmark import (
    BUDGET_SUFFIX,
    CHAIN_EFFORT_OPTIONS,
    CONFIG_SUITES,
    CONFIG_SWEEP_OPTIONS,
    DEFAULT_BUDGET_SWEEP,
    MIPLIB_MIN_INSTANCES,
    MIPLIB_SEARCH_PATH,
    build_arg_parser,
    build_base_options,
    build_plan,
    config_options,
    expand_configs,
    find_ignored_config_warning,
    main,
    parse_budget,
    resolve_data_dir,
    run_single,
    split_config,
    sweep_options_for_suite,
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


def test_vanilla_on_the_patched_binary_is_suite_off():
    assert config_options("vanilla") == {"mip_heuristic_suite": "off"}


def test_vanilla_on_an_external_binary_sets_nothing():
    """An unpatched binary has no mip_heuristic_* options to set."""
    assert config_options("vanilla", external_vanilla=True) == {}


def test_external_vanilla_does_not_leak_into_other_configs():
    assert config_options("fpr", external_vanilla=True) == {
        "mip_heuristic_suite": "fpr"
    }


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


def test_swept_config_carries_its_own_effort_option():
    assert config_options(f"local_mip{BUDGET_SUFFIX}0.60") == {
        "mip_heuristic_suite": "local_mip",
        "mip_heuristic_local_mip_effort": "0.60",
    }


def test_swept_all_carries_every_effort_option():
    """No shared envelope left, so `all@e<V>` is every heuristic at V."""
    assert config_options(f"all{BUDGET_SUFFIX}0.60") == {
        "mip_heuristic_suite": "all",
        "mip_heuristic_fj_effort": "0.60",
        "mip_heuristic_fpr_effort": "0.60",
        "mip_heuristic_local_mip_effort": "0.60",
        "mip_heuristic_scylla_effort": "0.60",
    }


def test_fj_is_swept_now_that_it_has_an_option():
    """The point of #110: FJ's budget was unreachable from any option."""
    assert config_options(f"fj{BUDGET_SUFFIX}0.05") == {
        "mip_heuristic_suite": "fj",
        "mip_heuristic_fj_effort": "0.05",
    }


def test_every_config_has_a_sweep_entry():
    """The sweep's reach is derived from this table, so it must be total."""
    assert set(CONFIG_SWEEP_OPTIONS) == set(CONFIG_SUITES)


def test_swept_config_with_a_bad_budget_raises():
    with pytest.raises(ValueError, match="not a number"):
        config_options(f"fpr{BUDGET_SUFFIX}high")


# --- sweep expansion -------------------------------------------------------


def test_no_sweep_leaves_configs_untouched():
    assert expand_configs(["all", "vanilla"], []) == (["all", "vanilla"], [])


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
    names, _ = expand_configs(["fpr"], list(DEFAULT_BUDGET_SWEEP))
    assert len(names) == len(DEFAULT_BUDGET_SWEEP) == 5
    assert names[0] == f"fpr{BUDGET_SUFFIX}{DEFAULT_BUDGET_SWEEP[0]}"


def test_default_sweep_spans_the_option_range():
    """Ascending, inside [0, 1], and wide: the shipped defaults (0.0125 ..
    0.1821) are per-heuristic since #110, so no single value is "the
    default" and the sweep's job is coverage rather than an anchor row."""
    values = [float(b) for b in DEFAULT_BUDGET_SWEEP]
    assert values == sorted(values)
    assert values[0] >= 0.0 and values[-1] <= 1.0
    assert values[-1] / values[0] >= 10.0


def test_expansion_rejects_an_already_suffixed_config():
    with pytest.raises(ValueError, match="already carries"):
        expand_configs([f"fpr{BUDGET_SUFFIX}0.30"], ["0.05"])


def test_expansion_rejects_a_bad_budget_before_any_run():
    with pytest.raises(ValueError):
        expand_configs(["fpr"], ["0.05", "banana"])


# --- subset configs (#112) -------------------------------------------------

# The four heuristics of the presolve chain, in chain order.
CHAIN = ("fj", "fpr", "local_mip", "scylla")


def test_the_local_chain_matches_the_effort_option_table():
    """This module's CHAIN is a cross-check only while it agrees with the
    module under test; a fifth heuristic must reach both."""
    assert tuple(CHAIN_EFFORT_OPTIONS) == CHAIN


def test_every_subset_of_the_chain_is_a_config():
    """#107 sweeps all fifteen non-empty subsets plus `off`."""
    expected = set()
    for mask in range(1, 1 << len(CHAIN)):
        members = [name for bit, name in enumerate(CHAIN) if mask & (1 << bit)]
        expected.add("all" if len(members) == len(CHAIN) else "+".join(members))
    assert expected <= set(CONFIG_SUITES)


def test_config_names_join_with_plus_and_suite_values_with_commas():
    """A comma in a config name is a results-tree path and a LaTeX label."""
    for name, suite in CONFIG_SUITES.items():
        assert "," not in name
        if "+" in name:
            assert name.split("+") == suite.split(",")
        else:
            assert "," not in suite


def test_subset_configs_map_to_a_comma_separated_suite_value():
    assert config_options("fj+fpr") == {"mip_heuristic_suite": "fj,fpr"}
    assert config_options("fj+fpr+local_mip") == {
        "mip_heuristic_suite": "fj,fpr,local_mip"
    }


def test_a_subset_name_is_not_an_alias_for_a_reordering():
    """One subset, one spelling: `fpr+fj` names no config."""
    with pytest.raises(ValueError, match="unknown config"):
        config_options("fpr+fj")


def test_the_budget_suffix_still_parses_on_a_subset_name():
    """`split_config` partitions on `@e`, which a `+` name does not contain."""
    assert split_config(f"fj+fpr{BUDGET_SUFFIX}0.30") == ("fj+fpr", "0.30")
    plan = build_plan(f"fj+fpr{BUDGET_SUFFIX}0.30", PATCHED, EXTERNAL)
    assert plan.base == "fj+fpr"
    assert plan.name == f"fj+fpr{BUDGET_SUFFIX}0.30"
    assert plan.options == {
        "mip_heuristic_suite": "fj,fpr",
        "mip_heuristic_fj_effort": "0.30",
        "mip_heuristic_fpr_effort": "0.30",
    }


def test_a_subset_sweeps_exactly_its_own_heuristics_effort_options():
    """The #110 x #112 join: a subset config moves the effort options of the
    heuristics it enables and no others, so `fj+scylla@e0.60` says nothing
    about FPR's or LocalMIP's budget."""
    for name, suite in CONFIG_SUITES.items():
        if "+" not in name:
            continue
        expected = tuple(
            CHAIN_EFFORT_OPTIONS[h] for h in CHAIN if h in suite.split(",")
        )
        assert CONFIG_SWEEP_OPTIONS[name] == expected
        assert config_options(f"{name}{BUDGET_SUFFIX}0.60") == {
            "mip_heuristic_suite": suite,
            **{option: "0.60" for option in expected},
        }


def test_no_subset_config_is_left_unswept():
    """Every subset enables at least one heuristic, so every one is swept —
    the `SWEEP_EXEMPT` entry that used to cover `fj` went with the shared
    envelope, FJ having gained an option of its own (#110)."""
    for name in CONFIG_SUITES:
        if "+" in name:
            assert CONFIG_SWEEP_OPTIONS[name]


def test_sweep_options_are_derived_and_reject_an_unknown_heuristic():
    """The drift guard: a config added to CONFIG_SUITES naming a heuristic
    with no effort option raises on import rather than sweeping a subset of
    what it runs and labelling the tree as if it had swept all of it."""
    with pytest.raises(ValueError, match="names no heuristic"):
        sweep_options_for_suite("fj,walksat")


def test_sweep_options_are_listed_in_chain_order():
    """One subset, one option tuple: order comes from the chain, not from the
    order the tokens happen to appear in the suite value."""
    assert sweep_options_for_suite("scylla,fj") == (
        "mip_heuristic_fj_effort",
        "mip_heuristic_scylla_effort",
    )


# --- vanilla / off interaction with the sweep ------------------------------


UNSWEPT = sorted(c for c, options in CONFIG_SWEEP_OPTIONS.items() if not options)


def test_only_the_heuristic_free_configs_are_unswept():
    assert UNSWEPT == ["off", "vanilla"]


@pytest.mark.parametrize("config", UNSWEPT)
def test_unswept_configs_pass_through_the_sweep_once(config):
    """Neither runs a presolve heuristic, so N budgets would be N identical runs."""
    names, notices = expand_configs([config], ["0.05", "0.30", "1.00"])
    assert names == [config]
    assert len(notices) == 1
    assert config in notices[0]


def test_sweep_keeps_the_anchor_alongside_swept_configs():
    names, notices = expand_configs(["vanilla", "fpr"], ["0.05", "0.30"])
    assert names == ["vanilla", f"fpr{BUDGET_SUFFIX}0.05", f"fpr{BUDGET_SUFFIX}0.30"]
    assert len(notices) == 1


@pytest.mark.parametrize("config", UNSWEPT)
def test_explicit_suffix_on_an_unswept_config_raises(config):
    """`vanilla@e0.30` is a directory name that would mean nothing."""
    with pytest.raises(ValueError, match="effort option") as exc:
        config_options(f"{config}{BUDGET_SUFFIX}0.30")
    # The message says *why* this config is not swept, not just that it is not.
    assert "runs no presolve heuristic" in str(exc.value)


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
        "mip_heuristic_scylla_effort": "0.15",
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
    """Level 3 is what makes [Heur]/[Sequential] visible."""
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


# --- a killed run keeps what it printed ------------------------------------


def _hanging_binary(tmp_path: Path, preamble: str) -> str:
    """A stand-in for `highs` that prints `preamble`, flushes, then hangs."""
    path = tmp_path / "hanging_highs"
    path.write_text(
        f"#!{sys.executable}\n"
        "import time\n"
        f"print({preamble!r}, flush=True)\n"
        "time.sleep(300)\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _run_until_killed(tmp_path: Path, monkeypatch, preamble: str) -> Path:
    """Run the hanging binary with the grace window shrunk to a test-sized one."""
    real_run = subprocess.run

    def quick(cmd, **kwargs):
        return real_run(cmd, **{**kwargs, "timeout": 2.0})

    monkeypatch.setattr(run_benchmark.subprocess, "run", quick)
    out = tmp_path / "results"
    _, _, _, ok = run_single(
        _hanging_binary(tmp_path, preamble),
        "model.mps",
        "model",
        "fpr",
        0,
        1.0,
        str(out),
    )
    assert not ok, "a killed run is not a success, whatever it managed to print"
    return out / "fpr" / "seed0"


def test_a_killed_run_keeps_its_partial_output(tmp_path: Path, monkeypatch):
    """HiGHS checks its clock between work units, so one long solve can run over.

    The output up to the kill is real measured data -- and the headline metrics
    read only the incumbent lines, never the Solving report the run never
    reached -- so it is kept as a `.log` and analysed like any other run.
    """
    seed_dir = _run_until_killed(tmp_path, monkeypatch, "Running HiGHS 1.15.1")
    log = seed_dir / "model.log"
    assert log.exists()
    assert not (seed_dir / "model.log.err").exists()
    text = log.read_text()
    assert "Running HiGHS 1.15.1" in text
    assert "TIMEOUT: process killed after" in text


def test_a_killed_run_is_not_retried_on_resume(tmp_path: Path, monkeypatch):
    """Re-running reproduces the same hang, so `.err`'s retry semantics are wrong.

    A non-empty `.log` is what `should_skip` looks for; the marker inside it is
    what keeps the kill visible in the analysis.
    """
    seed_dir = _run_until_killed(tmp_path, monkeypatch, "Running HiGHS 1.15.1")
    assert (seed_dir / "model.log").stat().st_size > 0


def test_a_hang_before_any_output_is_parked_as_err(tmp_path: Path, monkeypatch):
    """No banner means no run to measure -- that is a harness fault, so retry it."""
    seed_dir = _run_until_killed(tmp_path, monkeypatch, "")
    assert not (seed_dir / "model.log").exists()
    err = (seed_dir / "model.log.err").read_text()
    assert "before printing anything parseable" in err


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
    _, _, _, ok = run_single(str(binary), "model.mps", "model", "off", 0, 1.0, str(out))
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


# --- main()'s own guards ---------------------------------------------------


def _main(tmp_path: Path, monkeypatch, *argv: str) -> None:
    """Run main() over an empty instance list — every guard, no solves."""
    instances = tmp_path / "none.txt"
    instances.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark.py",
            "--instances",
            str(instances),
            "--binary",
            sys.executable,
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            *argv,
        ],
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


def test_main_warns_that_vanilla_and_off_are_the_same_run(
    tmp_path, monkeypatch, capsys
):
    """Without --vanilla-binary they are one configuration under two names."""
    _main(tmp_path, monkeypatch, "--configs", "vanilla", "off")
    err = capsys.readouterr().err
    assert "identical" in err and "duplicated work" in err


def test_main_warns_on_string_duplicate_budgets(tmp_path, monkeypatch, capsys):
    _main(tmp_path, monkeypatch, "--configs", "fpr", "--budget-sweep", "0.3", "0.30")
    assert "identical" in capsys.readouterr().err


def test_main_warns_when_a_config_overrides_an_extra_option(
    tmp_path, monkeypatch, capsys
):
    _main(
        tmp_path,
        monkeypatch,
        "--configs",
        "fpr",
        "--extra-options",
        "mip_heuristic_suite=scylla",
    )
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
    _main(
        tmp_path,
        monkeypatch,
        "--configs",
        "off",
        "fj",
        "fpr",
        "local_mip",
        "scylla",
        "all",
        "--budget-sweep",
        "0.05",
        "0.15",
        "0.30",
        "0.60",
        "1.00",
    )
    captured = capsys.readouterr()
    assert "identical" not in captured.err
    # `off` is the only anchor row left: FJ has had its own effort option
    # since #110, so it is swept like the other three.
    assert "off@e" not in captured.out
    assert "fj@e0.05" in captured.out
    assert "fpr@e0.05" in captured.out and "all@e1.00" in captured.out


def test_extra_options_override_of_dev_log_warns(capsys):
    """`--extra-options log_dev_level=1` silently cancels `--dev-log`.

    The run header still announces instrumentation and every solve succeeds,
    so without a warning the omission surfaces only when the finished tree
    turns out to carry no instrumentation — after the campaign has been
    paid for.
    """
    opts = build_base_options(None, True, ["log_dev_level=1"])
    assert opts["log_dev_level"] == "1"  # the override still wins
    err = capsys.readouterr().err
    assert "overrides --dev-log" in err
    assert "no per-heuristic instrumentation" in err


def test_extra_options_log_dev_level_is_quiet_without_dev_log(capsys):
    """Setting the level by hand, without `--dev-log`, is not a collision."""
    opts = build_base_options(None, False, ["log_dev_level=1"])
    assert opts["log_dev_level"] == "1"
    assert "overrides --dev-log" not in capsys.readouterr().err


# --- MIPLIB data-dir resolution --------------------------------------------
#
# The collection is 3.5 GB and shared by every checkout on the machine, so the
# thing worth pinning is that resolution never silently re-downloads it and
# never resolves an explicitly named directory to a different one.


def _populate(d: Path, count: int) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (d / f"inst{i}.mps.gz").write_bytes(b"")
    return d


def test_explicit_data_dir_wins_even_when_empty(tmp_path, monkeypatch):
    """An explicit --data-dir must not resolve elsewhere.

    A typo'd path silently reading some other collection is worse than an
    empty one: the run completes and reports on instances nobody asked for.
    """
    populated = _populate(tmp_path / "shared", MIPLIB_MIN_INSTANCES + 1)
    monkeypatch.setattr("run_benchmark.MIPLIB_SEARCH_PATH", (str(populated),))
    empty = tmp_path / "typo"
    assert resolve_data_dir(str(empty)) == str(empty)


def test_search_path_finds_a_populated_directory(tmp_path, monkeypatch):
    empty = tmp_path / "a"
    empty.mkdir()
    populated = _populate(tmp_path / "b", MIPLIB_MIN_INSTANCES + 1)
    monkeypatch.delenv("MIPLIB_DIR", raising=False)
    monkeypatch.setattr(
        "run_benchmark.MIPLIB_SEARCH_PATH", (str(empty), str(populated))
    )
    assert resolve_data_dir(None) == str(populated)


def test_a_sparse_directory_does_not_count_as_a_collection(tmp_path, monkeypatch):
    """Guards against a stray .mps.gz making an unrelated directory win."""
    sparse = _populate(tmp_path / "sparse", 3)
    populated = _populate(tmp_path / "full", MIPLIB_MIN_INSTANCES + 1)
    monkeypatch.delenv("MIPLIB_DIR", raising=False)
    monkeypatch.setattr(
        "run_benchmark.MIPLIB_SEARCH_PATH", (str(sparse), str(populated))
    )
    assert resolve_data_dir(None) == str(populated)


def test_env_var_outranks_the_search_path(tmp_path, monkeypatch):
    env_dir = _populate(tmp_path / "env", MIPLIB_MIN_INSTANCES + 1)
    default = _populate(tmp_path / "default", MIPLIB_MIN_INSTANCES + 1)
    monkeypatch.setenv("MIPLIB_DIR", str(env_dir))
    monkeypatch.setattr("run_benchmark.MIPLIB_SEARCH_PATH", (str(default),))
    assert resolve_data_dir(None) == str(env_dir)


def test_falls_back_to_the_head_of_the_search_path(tmp_path, monkeypatch):
    """With nothing populated the diagnostic still names a concrete directory."""
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.delenv("MIPLIB_DIR", raising=False)
    monkeypatch.setattr("run_benchmark.MIPLIB_SEARCH_PATH", (str(a), str(b)))
    assert resolve_data_dir(None) == str(a)


def test_persistent_location_precedes_tmp():
    """/tmp does not survive a reboot, so it must never outrank ~/data.

    Inverting these two is how a machine re-downloads 3.5 GB after every
    restart, which is the whole point of the search path.
    """
    assert MIPLIB_SEARCH_PATH.index(os.path.expanduser("~/data/miplib")) < (
        MIPLIB_SEARCH_PATH.index("/tmp/miplib")
    )


def test_tmp_stays_in_the_search_path():
    """Checkouts that already populated /tmp/miplib must keep working."""
    assert "/tmp/miplib" in MIPLIB_SEARCH_PATH


def test_unreadable_candidate_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A candidate we cannot read counts as absent.

    /tmp/miplib is probed for every user, so on a shared machine it may be
    someone else's mode-700 directory.  Raising there would abort exactly the
    run that still needed to locate a collection.
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    populated = _populate(tmp_path / "ok", MIPLIB_MIN_INSTANCES + 1)
    locked.chmod(0o000)
    monkeypatch.delenv("MIPLIB_DIR", raising=False)
    monkeypatch.setattr(
        "run_benchmark.MIPLIB_SEARCH_PATH", (str(locked), str(populated))
    )
    try:
        assert resolve_data_dir(None) == str(populated)
    finally:
        locked.chmod(0o755)


def test_empty_explicit_data_dir_falls_through_to_the_search(tmp_path, monkeypatch):
    """`--data-dir ""` is an unset wrapper variable, not a request for cwd."""
    populated = _populate(tmp_path / "shared", MIPLIB_MIN_INSTANCES + 1)
    monkeypatch.delenv("MIPLIB_DIR", raising=False)
    monkeypatch.setattr("run_benchmark.MIPLIB_SEARCH_PATH", (str(populated),))
    assert resolve_data_dir("") == str(populated)


@pytest.mark.parametrize(
    ("count", "is_collection"),
    [(MIPLIB_MIN_INSTANCES, False), (MIPLIB_MIN_INSTANCES + 1, True)],
)
def test_the_threshold_boundary_is_exclusive(
    tmp_path, monkeypatch, count, is_collection
):
    """Pins `> MIN` rather than `>= MIN`, so the two implementations cannot drift.

    The second candidate is populated so the two outcomes are distinguishable:
    were it absent, a rejected first candidate would still be returned as the
    head-of-path fallback and the assertion could not tell the cases apart.
    """
    candidate = _populate(tmp_path / "cand", count)
    other = _populate(tmp_path / "other", MIPLIB_MIN_INSTANCES + 1)
    monkeypatch.delenv("MIPLIB_DIR", raising=False)
    monkeypatch.setattr(
        "run_benchmark.MIPLIB_SEARCH_PATH", (str(candidate), str(other))
    )
    expected = str(candidate) if is_collection else str(other)
    assert resolve_data_dir(None) == expected


def test_data_dir_default_is_resolved_not_hardcoded():
    """The regression guard for the bug this whole change fixes.

    Reintroducing `default="/tmp/miplib"` on the argparse line is invisible to
    every other test here: `resolve_data_dir` would receive it as an explicit
    value and hand it straight back, while the search-path constants stay
    correct.  The default must be None so resolution actually runs.
    """
    parser = build_arg_parser()
    assert parser.get_default("data_dir") is None


# --- the bash/Python contract ----------------------------------------------
#
# The search path and the threshold are defined twice, once per language.
# Drift is silent and expensive -- the downloader writes one directory and the
# benchmark reads another -- so it is asserted here rather than in prose.
# Same reflex as `bench/check_docs_refs.py` and the pre-push hook parsing
# CLAUDE.md's fenced blocks: cross-artifact claims get a test.

DOWNLOAD_SCRIPT = Path(__file__).resolve().parent / "download_miplib.sh"


def test_shell_threshold_matches_python():
    text = DOWNLOAD_SCRIPT.read_text()
    m = re.search(r"^MIN_INSTANCES=(\d+)", text, re.MULTILINE)
    assert m, "MIN_INSTANCES not found in download_miplib.sh"
    assert int(m.group(1)) == MIPLIB_MIN_INSTANCES


def _shell_default_candidates() -> list[str]:
    """Every directory the shell script appends to its default candidate list.

    `findall`, not `search`: adding a fourth candidate as its own
    `CANDIDATES+=(...)` line is the most natural way to extend the list, and
    matching only the first line would let exactly the drift this test exists
    to catch pass green.  `$MIPLIB_DIR` is dropped because Python holds it in
    the environment rather than in MIPLIB_SEARCH_PATH.
    """
    text = DOWNLOAD_SCRIPT.read_text()
    groups = re.findall(r"^\s*CANDIDATES\+=\(([^)]*)\)", text, re.MULTILINE)
    assert groups, "no CANDIDATES+=(...) line found in download_miplib.sh"
    home = os.path.expanduser("~")
    dirs = []
    for group in groups:
        for raw in group.split():
            d = raw.strip('"').replace("${HOME}", home).replace("$HOME", home)
            if "MIPLIB_DIR" in d:
                continue
            dirs.append(d)
    return dirs


def test_shell_search_path_matches_python():
    """Same directories, same order, in both implementations."""
    assert _shell_default_candidates() == list(MIPLIB_SEARCH_PATH)


def test_shell_threshold_comparison_is_exclusive():
    """Pins the shell's `-gt`, not just its number.

    test_shell_threshold_matches_python pins 200 and
    test_the_threshold_boundary_is_exclusive pins Python's `>`; without this,
    flipping the shell to `-ge` drifts silently and a directory of exactly 200
    files becomes a collection to bash but not to Python.
    """
    text = DOWNLOAD_SCRIPT.read_text()
    assert re.search(r'-gt\s+"\$MIN_INSTANCES"', text), (
        "download_miplib.sh must compare with -gt to match Python's `>`"
    )


def test_shell_and_python_agree_on_the_env_override():
    assert "MIPLIB_DIR" in DOWNLOAD_SCRIPT.read_text()
