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
    BANNER_RE,
    CONFIG_SUITES,
    HIGHS_TAG_RE,
    KNOWN_CONFIGS,
    MIPLIB_MIN_INSTANCES,
    MIPLIB_SEARCH_PATH,
    PATCH_MARKER,
    build_arg_parser,
    build_base_options,
    build_plan,
    check_known_options,
    check_vanilla_binary,
    config_options,
    expected_highs_version,
    find_ignored_config_warning,
    main,
    probe_binary,
    resolve_data_dir,
    run_single,
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
    for name in KNOWN_CONFIGS:
        assert name in str(exc.value)


def test_unknown_config_raises_through_build_plan():
    with pytest.raises(ValueError, match="unknown config"):
        build_plan("all_opp", PATCHED, EXTERNAL)


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


def test_vanilla_is_not_a_suite_value():
    """#147: `vanilla` names a binary, so it maps to no suite at all.

    It used to map to `off`, which is what made a `vanilla` tree without
    `--vanilla-binary` an ablation on the patched binary.
    """
    assert "vanilla" not in CONFIG_SUITES
    assert "vanilla" in KNOWN_CONFIGS
    # An unpatched binary has no `mip_heuristic_*` option to set.
    assert config_options("vanilla") == {}


def test_vanilla_does_not_leak_into_other_configs():
    assert config_options("fpr") == {"mip_heuristic_suite": "fpr"}


# The four heuristics of the presolve chain, in chain order.
CHAIN = ("fj", "fpr", "local_mip", "scylla")


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


def test_vanilla_takes_the_external_binary():
    plan = build_plan("vanilla", PATCHED, EXTERNAL)
    assert plan.binary == EXTERNAL
    assert plan.options == {}


def test_vanilla_without_an_external_binary_raises():
    """#147: no fallback to the patched binary, and the message says why."""
    with pytest.raises(ValueError, match="requires --vanilla-binary") as exc:
        build_plan("vanilla", PATCHED, None)
    assert "ablation" in str(exc.value)


def test_every_other_config_takes_the_patched_binary_without_a_vanilla_one():
    for config in CONFIG_SUITES:
        assert build_plan(config, PATCHED, None).binary == PATCHED


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
    """`suite=fj` + run_feasibility_jump=false runs no FeasibilityJump."""
    output = (
        'WARNING: mip_heuristic_suite="fj" selects only FeasibilityJump, which '
        "mip_heuristic_run_feasibility_jump=false disables; no heuristic will "
        "run. Use mip_heuristic_suite=off to run HiGHS's own FeasibilityJump "
        "instead.\n"
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


def test_identity_separates_configs_that_differ_only_by_binary():
    assert (
        build_plan("vanilla", PATCHED, EXTERNAL).identity
        != build_plan("off", PATCHED, EXTERNAL).identity
    )


# --- the --vanilla-binary probe --------------------------------------------
#
# Every case here runs against a stand-in binary rather than a real HiGHS: the
# probe reads a banner, and a shell script prints one for a tenth of a second
# where building an unpatched HiGHS is a checkout and a compile.  What the
# stand-in has to reproduce faithfully is the *shape* of a no-argument
# invocation of the real CLI: the complaint, the banner, the marker on a
# patched build only, and a non-zero exit — which is why the probe must not
# read the exit status.


def fake_highs(
    tmp_path: Path,
    name: str,
    *,
    patched: bool,
    version: str = "1.15.1",
    banner: bool = True,
    githash: str = "04024d701f",
    unknown_options: tuple[str, ...] = (),
) -> Path:
    """A stand-in `highs` that answers a no-argument probe like the real one.

    With arguments it appends to `<path>.solves`, so a test can assert that a
    refused run never reached a solve.

    `githash` exists because a HiGHS configured outside a git repository
    prints the literal `n/a` there.  `unknown_options` names options this
    stand-in does not have: given an `--options_file` mentioning one, it
    answers the way HiGHS does — the `getOptionIndex` line and exit 255,
    without solving.
    """
    path = tmp_path / name
    marker = (
        f'echo "{PATCH_MARKER} (custom MIP presolve heuristics)"' if patched else ":"
    )
    banner_line = (
        f'echo "Running HiGHS {version} (git hash: {githash}): Copyright (c) 2026"'
        if banner
        else ":"
    )
    reject = "".join(
        f'if [ -n "$OPTS" ] && grep -q "^{opt} " "$OPTS"; then\n'
        f"  echo 'ERROR:   getOptionIndex: Option \"{opt}\" is unknown'\n"
        "  exit 255\n"
        "fi\n"
        for opt in unknown_options
    )
    path.write_text(
        "#!/bin/sh\n"
        "if [ $# -eq 0 ]; then\n"
        '  echo "Please specify filename in .mps|.lp|.ems format."\n'
        f"  {banner_line}\n"
        f"  {marker}\n"
        "  exit 255\n"
        "fi\n"
        'ARGS="$*"\n'
        # MODEL is what makes `.solves` mean "reached a solve".  A model-free
        # `--options_file` run is what the option probe issues and what real
        # HiGHS answers with "Please specify filename" — recording it here
        # would make every `assert not …solves.exists()` pass or fail on
        # whether the probe happened to run, not on whether a solve did.
        "OPTS=\nMODEL=\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    --options_file) OPTS="$2"; shift ;;\n'
        "    --*) if [ $# -gt 1 ]; then shift; fi ;;\n"
        '    *) MODEL="$1" ;;\n'
        "  esac\n"
        "  shift\n"
        "done\n"
        f"{reject}"
        f"{banner_line}\n"
        f"{marker}\n"
        'if [ -z "$MODEL" ]; then\n'
        '  echo "Please specify filename in .mps|.lp|.ems format."\n'
        "  exit 255\n"
        "fi\n"
        f'echo "$ARGS" >> "{path}.solves"\n'
        'echo "  Status            Optimal"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_expected_highs_version_matches_the_tag_this_tree_builds():
    """One definition of the tag: cmake/FetchHiGHS.cmake, read at run time."""
    assert expected_highs_version() == "1.15.1"


def test_the_two_bench_modules_agree_on_the_shared_constants():
    """`make_archive.py` duplicates three constants; drift between them is silent.

    It cannot import them: `make_archive.py` is copied into a release archive
    and run there by `REGENERATE.sh` with no `run_benchmark.py` beside it. So
    the two copies are pinned against each other here instead — the same
    treatment the MIPLIB search path gets across bash and Python. The tag
    regex is the one that bit: it captures `v1.15.1` and the banner prints
    `1.15.1`, so a copy that quietly absorbed the `v` into the pattern would
    make `check_vanilla_binary` reject every genuine binary.
    """
    import make_archive

    assert make_archive.PATCH_MARKER == PATCH_MARKER
    assert make_archive._BANNER_RE.pattern == BANNER_RE.pattern
    assert make_archive.HIGHS_TAG_RE.pattern == HIGHS_TAG_RE.pattern
    assert HIGHS_TAG_RE.search("GIT_TAG        v1.15.1").group(1) == "v1.15.1"


def test_the_probe_reads_the_banner_and_the_marker(tmp_path: Path):
    patched = probe_binary(str(fake_highs(tmp_path, "patched", patched=True)))
    assert patched.version == "1.15.1"
    assert patched.githash == "04024d701f"
    assert patched.patched
    unpatched = probe_binary(str(fake_highs(tmp_path, "unpatched", patched=False)))
    assert unpatched.version == "1.15.1"
    assert not unpatched.patched


def test_the_probe_accepts_an_unpatched_binary_of_the_right_tag(tmp_path: Path):
    check_vanilla_binary(str(fake_highs(tmp_path, "unpatched", patched=False)))


def test_the_probe_refuses_a_patched_binary(tmp_path: Path):
    with pytest.raises(ValueError, match="patched"):
        check_vanilla_binary(str(fake_highs(tmp_path, "patched", patched=True)))


def test_the_probe_refuses_another_tag(tmp_path: Path):
    binary = fake_highs(tmp_path, "old", patched=False, version="1.7.2")
    with pytest.raises(ValueError, match=r"1\.7\.2"):
        check_vanilla_binary(str(binary))


def test_the_probe_refuses_a_binary_that_prints_no_banner(tmp_path: Path):
    """A binary that cannot be identified is not one that may be assumed good."""
    binary = fake_highs(tmp_path, "mystery", patched=False, banner=False)
    with pytest.raises(ValueError, match="no HiGHS banner"):
        check_vanilla_binary(str(binary))


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


def _main_with_one_instance(
    tmp_path: Path,
    monkeypatch,
    *argv: str,
    patched_unknown: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    """Run main() over a list with one real instance file in the data dir.

    Returns `(output tree, patched binary)`, so a caller can assert on what a
    refused run left behind — which is the checkable form of "before any
    solve".  `patched_unknown` names options the *patched* stand-in does not
    have, for the typo case.
    """
    instances = tmp_path / "one.txt"
    instances.write_text("model\n")
    (tmp_path / "model.mps.gz").write_bytes(b"")
    binary = fake_highs(
        tmp_path, "patched", patched=True, unknown_options=patched_unknown
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark.py",
            "--instances",
            str(instances),
            "--binary",
            str(binary),
            "--data-dir",
            str(tmp_path),
            "--output",
            str(out),
            "--time-limit",
            "1",
            *argv,
        ],
    )
    main()
    return out, binary


def test_a_refused_vanilla_binary_stops_the_run_before_any_solve(
    tmp_path: Path, monkeypatch, capsys
):
    vanilla = fake_highs(tmp_path, "also-patched", patched=True)
    with pytest.raises(SystemExit) as exc:
        _main_with_one_instance(
            tmp_path,
            monkeypatch,
            "--configs",
            "vanilla",
            "all",
            "--vanilla-binary",
            str(vanilla),
        )
    assert exc.value.code == 2
    assert PATCH_MARKER in capsys.readouterr().err
    # Neither binary was handed a model, and no tree was written: the refusal
    # prevents a bad results tree rather than describing one afterwards.
    assert not Path(f"{vanilla}.solves").exists()
    assert not (tmp_path / "patched.solves").exists()
    out = tmp_path / "out"
    assert not out.exists() or not list(out.rglob("*.log"))


def test_the_probe_accepts_a_binary_built_outside_a_git_repository():
    """HiGHS prints `git hash: n/a` with no `.git`, and that is still a banner.

    A distro package, a conda build, or one built from the release source
    archive all reach `set(GITHASH "n/a")` in HiGHS's own CMakeLists. The
    banner regex used to require `\\w+` there, so it did not match at all and
    `check_vanilla_binary` refused the binary for "printing no HiGHS banner"
    — while quoting that banner back in the same message. Only the version
    decides anything; the hash is recorded, never compared.
    """
    banner = "Running HiGHS 1.15.1 (git hash: n/a): Copyright (c) 2026"
    match = BANNER_RE.search(banner)
    assert match is not None
    assert match.group(1) == "1.15.1"
    assert match.group(2) == "n/a"


def test_a_vanilla_binary_without_a_git_hash_is_accepted(tmp_path: Path):
    probe = check_vanilla_binary(
        str(fake_highs(tmp_path, "unpatched", patched=False, githash="n/a"))
    )
    assert probe.version == "1.15.1"
    assert probe.githash == "n/a"


def test_the_option_check_passes_options_the_vanilla_binary_has(tmp_path: Path):
    """No `--extra-options`, or only upstream's own, is not a refusal.

    `mip_heuristic_effort` and `mip_heuristic_run_feasibility_jump` are
    upstream's, so a prefix rule over `mip_heuristic_*` would refuse a legal
    sweep. The binary is asked instead.
    """
    vanilla = fake_highs(
        tmp_path,
        "unpatched",
        patched=False,
        unknown_options=("mip_heuristic_suite", "mip_heuristic_fpr_effort"),
    )
    check_known_options(str(vanilla), {}, unpatched=True)
    check_known_options(
        str(vanilla), {"mip_heuristic_effort": "0.05", "threads": "4"}, unpatched=True
    )


def test_the_option_check_names_every_option_the_vanilla_binary_lacks(tmp_path: Path):
    """All of them, not just the first.

    HiGHS stops reading an options file at the first unknown key, so a single
    probing run names one offender and the operator relaunches the campaign
    once per option. The check asks one key at a time for that reason.
    """
    vanilla = fake_highs(
        tmp_path,
        "unpatched",
        patched=False,
        unknown_options=("mip_heuristic_suite", "mip_heuristic_fpr_effort"),
    )
    with pytest.raises(ValueError) as exc:
        check_known_options(
            str(vanilla),
            {
                "mip_heuristic_fpr_effort": "1.0",
                "mip_heuristic_suite": "all",
                "threads": "4",
            },
            unpatched=True,
        )
    assert "mip_heuristic_fpr_effort" in str(exc.value)
    assert "mip_heuristic_suite" in str(exc.value)
    assert "threads" not in str(exc.value)


def test_a_probe_is_not_recorded_as_a_solve(tmp_path: Path):
    """`.solves` must mean "reached a solve", or every refusal test is vacuous.

    The option probe runs `highs --options_file <tmp>` with no model, which
    real HiGHS answers with "Please specify filename" and no solve. A
    stand-in that logged it would make `assert not …solves.exists()` pass or
    fail on whether a probe ran rather than on whether a solve did.
    """
    binary = fake_highs(tmp_path, "unpatched", patched=False)
    check_known_options(str(binary), {"threads": "4"}, unpatched=True)
    assert not Path(f"{binary}.solves").exists()


def test_a_typod_extra_option_stops_the_run_on_the_patched_arm(
    tmp_path: Path, monkeypatch, capsys
):
    """The same campaign-cannot-advance failure, on the larger arm.

    `mip_heuristic_fpr_effrot` is a typo the patched binary rejects, so every
    instance of every patched config would exit 255 without solving. The
    check runs against every distinct binary the run will use, not only the
    vanilla one.
    """
    with pytest.raises(SystemExit) as exc:
        _main_with_one_instance(
            tmp_path,
            monkeypatch,
            "--configs",
            "all",
            "--extra-options",
            "mip_heuristic_fpr_effrot=1.0",
            patched_unknown=("mip_heuristic_fpr_effrot",),
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "mip_heuristic_fpr_effrot" in err
    # The typo branch of the message, not the unpatched-baseline one.
    assert "Check the spelling" in err
    assert not (tmp_path / "patched.solves").exists()


def test_an_extra_option_the_vanilla_binary_lacks_stops_the_run(
    tmp_path: Path, monkeypatch, capsys
):
    """The failure this closes: every vanilla instance exits 255 at solve time.

    `--extra-options mip_heuristic_fpr_effort=1.0` over the default `vanilla
    all` pair is the documented sweep invocation, and since #147 the vanilla
    arm is always an unpatched binary that has no such option. Each instance
    would land in `<inst>.log.err`, the vanilla arm would never advance, and
    `run_plato.sh next` would relaunch a campaign that cannot finish.
    """
    vanilla = fake_highs(
        tmp_path,
        "unpatched",
        patched=False,
        unknown_options=("mip_heuristic_fpr_effort",),
    )
    with pytest.raises(SystemExit) as exc:
        _main_with_one_instance(
            tmp_path,
            monkeypatch,
            "--configs",
            "vanilla",
            "all",
            "--vanilla-binary",
            str(vanilla),
            "--extra-options",
            "mip_heuristic_fpr_effort=1.0",
        )
    assert exc.value.code == 2
    assert "mip_heuristic_fpr_effort" in capsys.readouterr().err
    # Refused before any solve, on either binary — the point is to prevent a
    # half-empty tree rather than to explain one.
    assert not Path(f"{vanilla}.solves").exists()
    assert not (tmp_path / "patched.solves").exists()


def test_a_run_whose_vanilla_binary_passes_the_probe_solves(
    tmp_path: Path, monkeypatch
):
    """The accepting half: an unpatched stand-in reaches the solve loop."""
    vanilla = fake_highs(tmp_path, "unpatched", patched=False)
    out, _ = _main_with_one_instance(
        tmp_path,
        monkeypatch,
        "--configs",
        "vanilla",
        "all",
        "--vanilla-binary",
        str(vanilla),
    )
    assert Path(f"{vanilla}.solves").exists()
    assert (out / "vanilla" / "seed0" / "model.log").exists()
    assert (out / "all" / "seed0" / "model.log").exists()


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


def test_an_empty_vanilla_binary_counts_as_absent(tmp_path, monkeypatch, capsys):
    """`--vanilla-binary ""` is a wrapper passing an unset variable through.

    `bench/run_plato.sh` spells the flag unconditionally in its invocation and
    lets the value be empty when no unpatched binary was found, so this is
    load-bearing in both directions: a config list without `vanilla` must run
    normally, and one with it must get the message explaining the design
    rather than a "not found" about a blank path (`os.path.exists("")` is
    False, so an `is not None` check here would produce exactly that).
    """
    _main(tmp_path, monkeypatch, "--configs", "off", "--vanilla-binary", "")
    out = capsys.readouterr().out
    assert "Vanilla binary" not in out

    with pytest.raises(SystemExit) as exc:
        _main(tmp_path, monkeypatch, "--configs", "vanilla", "--vanilla-binary", "")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "requires --vanilla-binary" in err
    assert "not found" not in err


def test_main_exits_2_when_vanilla_has_no_binary(tmp_path, monkeypatch, capsys):
    """#147: the flag is required, and the refusal says what `off` is instead."""
    with pytest.raises(SystemExit) as exc:
        _main(tmp_path, monkeypatch, "--configs", "vanilla")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--vanilla-binary" in err
    assert "ablation" in err


def test_main_runs_vanilla_and_off_as_two_different_configurations(
    tmp_path, monkeypatch, capsys
):
    """They differ by binary now, so neither is a duplicate of the other."""
    _main(
        tmp_path,
        monkeypatch,
        "--configs",
        "vanilla",
        "off",
        "--vanilla-binary",
        str(fake_highs(tmp_path, "unpatched", patched=False)),
    )
    assert "duplicated work" not in capsys.readouterr().err


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
