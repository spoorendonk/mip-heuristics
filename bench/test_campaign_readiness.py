"""Smoke tests: the bench scripts are ready to run the campaign (#105-#108).

The campaign issues commit to a pipeline before any of it is run for days on
end: `run_benchmark.py` produces a results tree, `make_tuning_set.py` derives
the tuning subset from the baseline arm of that tree, and
`analyze_results.py` turns the trees into the tables the acceptance criteria
name.  Every failure mode of that pipeline is expensive in exactly the same
way — it costs a night of benchmark machine time and is discovered the
morning after.

So these are readiness tests, not unit tests: each one runs the real script
end to end and asserts a property one of #105-#108 depends on — the options a
run is handed, the tree it writes, the chunk boundary and resume, and the
tables that come back out.  The unit-level behaviour of each script is
covered by its own `test_*.py`; what is asserted here is that the pieces
compose into the four stages.

Two stand-ins keep it cheap and hermetic:

* a fake `highs` (`FAKE_HIGHS`) that records the argv and options file it was
  handed and prints a HiGHS-shaped log — this is what makes "the parameters
  are properly passed" checkable without a solver, and lets a run be made to
  crash or to ignore its configuration on demand;
* synthetic `.mps` files, which the runner only ever stats.

The last section runs the *real* binary when a build is present, because the
fake cannot answer the other half of the question: whether HiGHS accepts the
options the campaign sets and emits the lines the parsers read.  It skips
when `build/bin/highs` is absent, which is the state of the Python-only CI
job.

`run_plato.sh` is covered as the chunked launcher all four stages use: it
takes its configs, seeds, instance list and output tree from the environment,
so a stage is an environment rather than a separate launcher.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_results import CUSTOM_SOURCE_LABELS
from parse_highs_log import parse_log
from run_benchmark import (
    CHAIN_EFFORT_OPTIONS,
    CONFIG_SUITES,
    MIPLIB_MIN_INSTANCES,
    build_arg_parser,
)

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
RUN_BENCHMARK = BENCH / "run_benchmark.py"
ANALYZE_RESULTS = BENCH / "analyze_results.py"
MAKE_TUNING_SET = BENCH / "make_tuning_set.py"
RUN_PLATO = BENCH / "run_plato.sh"
APPLY_PATCH = REPO / "third_party" / "highs_patch" / "apply_patch.cmake"

# The chain, in dispatch order.  `CONFIG_SUITES` keys spell subsets with `+`
# in this order, so the sixteen mix-selection configs of #107 are exactly
# `off` plus the fifteen non-empty subsets.
CHAIN = ("fj", "fpr", "local_mip", "scylla")

# A stand-in HiGHS.  It writes one JSON line per invocation to
# $FAKE_HIGHS_RECORD — that record is the evidence for every "the parameters
# reach the binary" assertion — and prints a log shaped like the real one:
# banner, patch marker, model header, source legend, incumbent lines, the
# `[Heur]`/`[Sequential]` instrumentation when (and only when)
# log_dev_level=3, and a solving report.
#
# Its behaviour varies with the options it is given, so a tree is not
# uniform: more heuristics in the suite means an earlier first solution and a
# better objective, and a swept effort option scales the charged effort.  A
# ladder or a mix comparison therefore has a signal to read, which is what
# makes the table assertions meaningful rather than shape-only.
FAKE_HIGHS = r'''#!/usr/bin/env python3
"""A HiGHS stand-in for bench/test_campaign_readiness.py."""

import json
import os
import sys
import time
import zlib

CHAIN = ("fj", "fpr", "local_mip", "scylla")
SOURCE_CHAR = {"fj": "J", "fpr": "A", "local_mip": "M", "scylla": "G"}
# Time to first feasible, cycled over the instance names so a tree spans the
# strata make_tuning_set.py splits on (<1s, 1-10s, 10-100s, 100-600s, never).
T1ST = (0.5, 5.0, 50.0, 300.0, None)


def env_list(name):
    return [s for s in os.environ.get(name, "").split(",") if s]


argv = sys.argv[1:]
instance = os.path.basename(argv[0]).split(".")[0]
opts_path = None
time_limit = "0"
for i, arg in enumerate(argv):
    if arg == "--options_file":
        opts_path = argv[i + 1]
    elif arg == "--time_limit":
        time_limit = argv[i + 1]

options = {}
if opts_path:
    with open(opts_path) as handle:
        for line in handle:
            key, sep, value = line.partition("=")
            if sep:
                options[key.strip()] = value.strip()

record = os.environ.get("FAKE_HIGHS_RECORD")
if record:
    with open(record, "a") as handle:
        handle.write(
            json.dumps(
                {
                    "binary": sys.argv[0],
                    "instance": instance,
                    "time_limit": time_limit,
                    "options_file": opts_path,
                    "options": options,
                }
            )
            + "\n"
        )

time.sleep(float(os.environ.get("FAKE_HIGHS_SLEEP", "0")))

if instance in env_list("FAKE_HIGHS_FAIL"):
    print("ERROR: could not read the model")
    sys.exit(255)

out = ["Running HiGHS 1.15.1 (git hash: 04024d701f): Copyright (c) 2026 HiGHS"]
# The patch marker is the *only* thing separating a patched build from an
# unpatched one of the same tag, so a stand-in named `*-unpatched` omits it.
if "unpatched" not in os.path.basename(sys.argv[0]):
    out.append("mip-heuristics patch active (custom MIP presolve heuristics)")

suite = options.get("mip_heuristic_suite", "")
tokens = [t for t in (s.strip() for s in suite.split(",")) if t]
if suite == "all":
    tokens = list(CHAIN)
if suite in ("", "off"):
    tokens = []
if instance in env_list("FAKE_HIGHS_IGNORE_CONFIG"):
    out.append(f'WARNING: Unknown mip_heuristic_suite value "{suite}" - running all')
    tokens = list(CHAIN)

seed = int(options.get("random_seed", "0"))
digest = zlib.crc32(instance.encode())
t1st = T1ST[digest % len(T1ST)]
objective = 100.0 + digest % 50
if tokens and t1st is not None:
    # More heuristics: sooner, and better.  Gives the mix tables a signal.
    t1st = round(t1st / (1 + len(tokens)), 3)
    objective -= len(tokens) + 0.25 * seed

out.append(
    f"MIP {instance} has 31 rows; 42 cols; 91 nonzeros; "
    "28 integer variables (28 binary)"
)
# The effective worker count, which no options file records because the
# harness pins no thread count.
out.append(
    "Solving MIP model with:\n"
    "   Thread count 16 (of 32 threads). Using 8 max workers. Parallel search on"
)
out.append(
    "Src: B => Branching; H => Heuristic; T => Evaluate node; "
    "A => FPR; D => FPR LP; M => Local MIP; G => Scylla; J => FJ"
)
out.append(
    "Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol"
    "              Gap | Cuts InLp Confl. | LpIters     Time"
)


def incumbent(source, obj, seconds):
    return (
        f"{source}       0       0         0   0.00%         50"
        f"              {obj}              Large      0      0      0"
        f"       0.0   {seconds}s"
    )


if t1st is not None:
    source = SOURCE_CHAR[tokens[0]] if tokens else "H"
    out.append(incumbent(source, objective, t1st))
    out.append(incumbent(source, objective - 1, round(t1st + 1.0, 3)))

if options.get("log_dev_level") == "3":
    start = 0.001
    for name in CHAIN:
        if name not in tokens:
            continue
        effort_option = f"mip_heuristic_{name}_effort"
        # Charged effort tracks the option, so a #106 ladder has a response.
        effort = int(1_000_000 * float(options.get(effort_option, "0.05")) / 0.05)
        wall = 10.0
        rate = effort / wall
        out.append(
            f"[Sequential] heur={name} effort={effort} wall_ms={wall} "
            f"effort_per_ms={rate:.3f}"
        )
        out.append(
            f"[Heur] name={name} phase=presolve start_s={start:.3f} "
            f"end_s={start + wall / 1000:.3f} effort={effort} wall_ms={wall} "
            f"effort_per_ms={rate:.3f} found=1"
        )
        start += wall / 1000

out.append("Solving report")
out.append(f"  Model             {instance}")
out.append("  Status            Time limit reached")
if t1st is not None:
    out.append(f"  Primal bound      {objective - 1}")
out.append("  Dual bound        50")
out.append("  Gap               50%")
out.append("  P-D integral      1.234")
out.append(f"  Timing            {time_limit}")
out.append("  Nodes             1")
out.append("  LP iterations     39")
print("\n".join(out))
'''


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fake_highs(tmp_path: Path, name: str = "highs") -> Path:
    """Write the stand-in binary; two copies stand in for two builds."""
    path = tmp_path / name
    path.write_text(FAKE_HIGHS)
    path.chmod(0o755)
    return path


def instance_dir(tmp_path: Path, names: list[str]) -> Path:
    """A MIPLIB-shaped data directory.  The runner only stats these."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for name in names:
        (data / f"{name}.mps").write_text("NAME\nENDATA\n")
    return data


def miplib_dir(tmp_path: Path, names: list[str]) -> Path:
    """A data directory that passes the runner's own collection test.

    `resolve_data_dir` accepts `$MIPLIB_DIR` only when it holds more than
    `MIPLIB_MIN_INSTANCES` `.mps.gz` files — a sparse directory is not a
    collection — so a run that reaches its instances through the environment
    rather than `--data-dir` has to look like one.
    """
    data = tmp_path / "miplib"
    data.mkdir(exist_ok=True)
    for i in range(MIPLIB_MIN_INSTANCES + 1 - len(names)):
        (data / f"pad{i:04d}.mps.gz").write_bytes(b"")
    for name in names:
        (data / f"{name}.mps.gz").write_bytes(b"")
    return data


def instance_list(path: Path, names: list[str]) -> Path:
    path.write_text("".join(f"{n}\n" for n in names))
    return path


def run_script(script: Path, *args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
        cwd=str(REPO),
    )


def run_benchmark(*args: str, env: dict[str, str] | None = None):
    result = run_script(RUN_BENCHMARK, *args, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def analyze(*args: str):
    result = run_script(ANALYZE_RESULTS, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def config_of(record: dict) -> str:
    """The config directory a recorded run wrote into."""
    return Path(record["options_file"]).parent.parent.name


def seed_of(record: dict) -> int:
    return int(Path(record["options_file"]).parent.name.removeprefix("seed"))


def logs_under(tree: Path, config: str, seed: int = 0) -> list[Path]:
    return sorted((tree / config / f"seed{seed}").glob("*.log"))


def make_run(tmp_path: Path, names: list[str]):
    """Bundle the four paths every runner invocation needs."""
    data = instance_dir(tmp_path, names)
    listing = instance_list(tmp_path / "instances.txt", names)
    binary = fake_highs(tmp_path)
    record = tmp_path / "record.jsonl"
    return data, listing, binary, record


# ---------------------------------------------------------------------------
# the harness contract: what every stage relies on being passed through
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def basic_tree(tmp_path_factory):
    """One `all` + `off` tree at two seeds, reused by the contract tests."""
    tmp_path = tmp_path_factory.mktemp("basic")
    names = [f"inst{i:02d}" for i in range(3)]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        "all",
        "off",
        "--seeds",
        "0",
        "1",
        "--time-limit",
        "60",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )
    return tree, records(record), names


def test_the_runner_writes_the_tree_layout_every_downstream_script_reads(basic_tree):
    tree, _, names = basic_tree
    for config in ("all", "off"):
        for seed in (0, 1):
            seed_dir = tree / config / f"seed{seed}"
            assert sorted(p.stem for p in seed_dir.glob("*.log")) == names
            # The `.opts` beside each log is what make_archive.py reads back
            # as "what this run was actually asked to do".
            assert sorted(p.stem for p in seed_dir.glob("*.opts")) == names
            assert not list(seed_dir.glob("*.log.err"))


def test_every_run_is_handed_its_suite_value_and_the_seed_that_names_its_directory(
    basic_tree,
):
    _, recorded, names = basic_tree
    assert len(recorded) == 2 * 2 * len(names)
    for record in recorded:
        options = record["options"]
        assert options["mip_heuristic_suite"] == CONFIG_SUITES[config_of(record)]
        # A seed pinned anywhere but --seeds would make the directory a lie.
        assert options["random_seed"] == str(seed_of(record))


def test_the_time_limit_reaches_the_binary(basic_tree):
    _, recorded, _ = basic_tree
    assert {r["time_limit"] for r in recorded} == {"60.0"}


def test_the_harness_pins_no_thread_count(basic_tree):
    # #105 and #108 both require it: HiGHS derives the worker count from the
    # host, and pinning it collapses the parallelism being measured.
    _, recorded, _ = basic_tree
    assert not any("threads" in r["options"] for r in recorded)


def test_headline_runs_carry_no_developer_logging(basic_tree):
    tree, recorded, _ = basic_tree
    assert not any("log_dev_level" in r["options"] for r in recorded)
    for log in logs_under(tree, "all"):
        assert parse_log(log.read_text()).heuristic_samples == []


def test_the_logs_a_headline_run_writes_are_parseable(basic_tree):
    tree, _, _ = basic_tree
    # Warnings raise: `parse_log` warns when a solving report proves a
    # solution was found but no incumbent line was recognised, which is what
    # a HiGHS bump adding a source code looks like from here.  A campaign
    # that ignores it bins those instances as never-feasible.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parsed = [parse_log(p.read_text()) for p in logs_under(tree, "all")]
    assert all(r.status for r in parsed)
    assert any(r.incumbents for r in parsed)


def test_dev_log_turns_the_instrumentation_on_and_it_parses(tmp_path):
    names = ["inst00"]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        "all",
        "--time-limit",
        "10",
        "--dev-log",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )
    assert all(r["options"]["log_dev_level"] == "3" for r in records(record))
    parsed = parse_log((tree / "all" / "seed0" / "inst00.log").read_text())
    assert [s.name for s in parsed.heuristic_samples] == list(CHAIN)
    assert {s.phase for s in parsed.heuristic_samples} == {"presolve"}
    assert [s.heuristic for s in parsed.sequential_samples] == list(CHAIN)


def test_an_external_vanilla_binary_is_handed_no_heuristic_options(tmp_path):
    # #105's baseline: a separately built unpatched binary has no
    # mip_heuristic_* options at all, so writing one would fail the run.
    names = ["inst00"]
    data, listing, binary, record = make_run(tmp_path, names)
    unpatched = fake_highs(tmp_path, "highs-unpatched")
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--vanilla-binary",
        str(unpatched),
        "--output",
        str(tree),
        "--configs",
        "vanilla",
        "all",
        "--time-limit",
        "10",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )
    by_config = {config_of(r): r for r in records(record)}
    assert by_config["vanilla"]["binary"] == str(unpatched)
    assert set(by_config["vanilla"]["options"]) == {"random_seed"}
    assert by_config["all"]["binary"] == str(binary)


def test_a_chunk_stops_launching_and_a_resume_finishes_the_tree_exactly_once(tmp_path):
    # The campaign rule (#109): every stage runs in overnight chunks, with
    # `--wall-time-budget` stopping the launches and `--skip-existing`
    # resuming.  Resume is per (config, instance, seed) — so no run repeats.
    names = [f"inst{i:02d}" for i in range(6)]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    common = [
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        "all",
        "--time-limit",
        "10",
    ]
    env = {"FAKE_HIGHS_RECORD": str(record), "FAKE_HIGHS_SLEEP": "0.2"}
    first = run_benchmark(
        *common, "--wall-time-budget", "0.4", "--skip-existing", env=env
    )
    assert "Time budget reached" in first.stdout
    done_first = len(logs_under(tree, "all"))
    assert 0 < done_first < len(names)

    second = run_benchmark(*common, "--skip-existing", env=env)
    assert len(logs_under(tree, "all")) == len(names)
    assert second.stdout.count("SKIP") == done_first
    # Every instance solved once across the two chunks: the resume neither
    # dropped an instance nor paid for one twice.
    assert sorted(r["instance"] for r in records(record)) == names


def test_a_crashed_run_is_parked_beside_the_log_and_retried_on_resume(tmp_path):
    names = ["inst00", "inst01"]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    common = [
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        "all",
        "--time-limit",
        "10",
        "--skip-existing",
    ]
    run_benchmark(
        *common, env={"FAKE_HIGHS_RECORD": str(record), "FAKE_HIGHS_FAIL": "inst01"}
    )
    seed_dir = tree / "all" / "seed0"
    assert (seed_dir / "inst01.log.err").exists()
    assert not (seed_dir / "inst01.log").exists()

    run_benchmark(*common, env={"FAKE_HIGHS_RECORD": str(record)})
    assert (seed_dir / "inst01.log").exists()
    # Exactly one of the two files describes the tree, so the retry clears it.
    assert not (seed_dir / "inst01.log.err").exists()


def test_a_run_that_ignored_its_suite_value_is_not_recorded_as_a_result(tmp_path):
    # "No instance recorded as a non-solving or misconfigured run" is an
    # acceptance criterion of all four stages, and this is the case that
    # exits 0 with an ordinary-looking log.
    names = ["inst00"]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    result = run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        "off",
        "--time-limit",
        "10",
        env={
            "FAKE_HIGHS_RECORD": str(record),
            "FAKE_HIGHS_IGNORE_CONFIG": "inst00",
        },
    )
    assert "ignored its configuration" in result.stderr
    assert (tree / "off" / "seed0" / "inst00.log.err").exists()
    assert not (tree / "off" / "seed0" / "inst00.log").exists()


# ---------------------------------------------------------------------------
# #105 — vanilla baseline, stratification input, difficulty profile
# ---------------------------------------------------------------------------


def test_the_plato_list_is_the_full_233_instance_benchmark():
    from run_benchmark import load_instances

    names = load_instances(str(BENCH / "instances_plato.txt"))
    assert len(names) == 233
    assert len(set(names)) == 233


def test_a_vanilla_tree_feeds_the_stratified_subset_extractor(tmp_path):
    # #105's tree must be "consumable by the stratified-subset extraction tool
    # without modification" — i.e. the layout the runner writes is the layout
    # make_tuning_set.py scans, with no reshaping in between.
    names = [f"inst{i:02d}" for i in range(20)]
    data, listing, binary, record = make_run(tmp_path, names)
    unpatched = fake_highs(tmp_path, "highs-unpatched")
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--vanilla-binary",
        str(unpatched),
        "--output",
        str(tree),
        "--configs",
        "vanilla",
        "--time-limit",
        "600",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )

    subset = tmp_path / "tuning.txt"
    result = run_script(
        MAKE_TUNING_SET,
        str(tree),
        "--instances",
        str(listing),
        "--size",
        "8",
        "--seed",
        "0",
        "--output",
        str(subset),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    chosen = [
        line.split("#", 1)[0].strip()
        for line in subset.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    assert len(chosen) == 8
    assert set(chosen) <= set(names)


def test_the_baseline_profile_counts_every_instance_including_never_feasible(tmp_path):
    # The third purpose of #105's run: the feasibility-difficulty profile,
    # "counts of instances by vanilla time-to-first-feasible, including a
    # never-feasible bucket".  make_tuning_set.py prints it while stratifying.
    names = [f"inst{i:02d}" for i in range(20)]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        "vanilla",
        "--time-limit",
        "600",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )
    result = run_script(
        MAKE_TUNING_SET,
        str(tree),
        "--instances",
        str(listing),
        "--size",
        "8",
        "--output",
        os.devnull,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line.split() for line in result.stderr.splitlines() if line.split()]
    # One row per stratum, a `never` bucket among them, and a TOTAL covering
    # every instance of the reference list — the profile, not a sample of it.
    assert "never" in {row[0] for row in rows}
    total = next(row for row in rows if row[0] == "TOTAL")
    assert int(total[1]) == len(names)


# ---------------------------------------------------------------------------
# #106 — per-heuristic budget ladders
# ---------------------------------------------------------------------------

# A log-spaced ladder stepping by 2x, extending below every shipped default
# (the smallest is fj's 0.0125) as the issue requires.
LADDER = ("0.003125", "0.00625", "0.0125", "0.025", "0.05")


@pytest.fixture(scope="module")
def ladder_tree(tmp_path_factory):
    """One heuristic's ladder, run alone, with the instrumentation on."""
    tmp_path = tmp_path_factory.mktemp("ladder")
    names = [f"inst{i:02d}" for i in range(2)]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        "fpr",
        "--budget-sweep",
        *LADDER,
        "--seeds",
        "0",
        "1",
        "--time-limit",
        "60",
        "--dev-log",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )
    return tree, records(record), names


def test_a_budget_ladder_writes_one_tree_per_point(ladder_tree):
    tree, _, names = ladder_tree
    for value in LADDER:
        for seed in (0, 1):
            assert (
                sorted(p.stem for p in logs_under(tree, f"fpr@e{value}", seed)) == names
            )


def test_a_ladder_point_moves_that_heuristics_option_and_no_other(ladder_tree):
    # Coordinate-wise is the whole search strategy: one heuristic's option
    # sweeps while it runs alone.  A leaked second effort option would make
    # the curve unattributable.
    _, recorded, _ = ladder_tree
    for record in recorded:
        options = record["options"]
        value = config_of(record).partition("@e")[2]
        assert options["mip_heuristic_suite"] == "fpr"
        assert options["mip_heuristic_fpr_effort"] == value
        others = set(CHAIN_EFFORT_OPTIONS.values()) - {"mip_heuristic_fpr_effort"}
        assert not (others & set(options))


def test_the_charged_effort_per_heuristic_is_readable_off_the_ladder(ladder_tree):
    # #106 selects on a curve and has to show the stall thresholds binding at
    # the top of the ladder — both read charged effort per heuristic out of
    # the `[Heur]` lines, per ladder point.
    tree, _, _ = ladder_tree
    charged = {}
    for value in LADDER:
        efforts = [
            sample.effort
            for log in logs_under(tree, f"fpr@e{value}")
            for sample in parse_log(log.read_text()).heuristic_samples
            if sample.name == "fpr"
        ]
        assert efforts
        charged[value] = sum(efforts)
    assert len(set(charged.values())) > 1, charged


def test_the_ladder_is_analysable_under_its_directory_names(ladder_tree):
    tree, _, _ = ladder_tree
    configs = [f"fpr@e{v}" for v in LADDER]
    out = analyze(
        str(tree), "--ablation", "--configs", *configs, "--time-limit", "60"
    ).stdout
    assert "## Ablation summary" in out
    assert "PLATO SGM" in out
    for config in configs:
        assert config in out


# ---------------------------------------------------------------------------
# #107 — heuristic mix selection
# ---------------------------------------------------------------------------

# `off` plus the fifteen non-empty subsets of the chain: the sixteen
# configurations #107 compares.
MIX_CONFIGS = ["off"] + [
    name
    for name, suite in CONFIG_SUITES.items()
    if name != "vanilla" and suite != "off"
]


def test_all_sixteen_mixes_are_expressible_and_distinct():
    assert len(MIX_CONFIGS) == 16
    suites = [CONFIG_SUITES[c] for c in MIX_CONFIGS]
    assert len(set(suites)) == 16
    for name in MIX_CONFIGS:
        if name in ("off", "all"):
            continue
        # A subset name lists its heuristics in chain order, so one subset has
        # exactly one spelling and one results directory.
        assert name.split("+") == CONFIG_SUITES[name].split(",")


@pytest.fixture(scope="module")
def mix_tree(tmp_path_factory):
    """All sixteen mixes, two instances, two seeds — #107's shape."""
    tmp_path = tmp_path_factory.mktemp("mix")
    names = [f"inst{i:02d}" for i in range(2)]
    data, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--output",
        str(tree),
        "--configs",
        *MIX_CONFIGS,
        "--seeds",
        "0",
        "1",
        "--time-limit",
        "60",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )
    return tree, records(record), names


def test_every_mix_runs_at_two_seeds_and_is_handed_its_own_suite_value(mix_tree):
    tree, recorded, names = mix_tree
    for config in MIX_CONFIGS:
        for seed in (0, 1):
            assert sorted(p.stem for p in logs_under(tree, config, seed)) == names
    delivered = {config_of(r): r["options"]["mip_heuristic_suite"] for r in recorded}
    assert delivered == {c: CONFIG_SUITES[c] for c in MIX_CONFIGS}


def test_the_mix_table_ranks_every_configuration_on_the_pre_registered_metric(mix_tree):
    # Pre-registered: primal-integral SGM at the run's time limit, which is
    # the `PLATO SGM` column of the ablation table.
    tree, _, _ = mix_tree
    out = analyze(
        str(tree), "--ablation", "--configs", *MIX_CONFIGS, "--time-limit", "60"
    ).stdout
    assert "PLATO SGM" in out
    rows = {line.split()[0] for line in out.splitlines() if line.split()}
    assert set(MIX_CONFIGS) <= rows


def test_the_best_of_ceiling_row_comes_off_the_mix_tree(mix_tree):
    # #108 reports the offline best-of row over #107's configs, computed from
    # that tree with no extra runs.
    tree, _, _ = mix_tree
    participants = ["fj", "fpr", "local_mip", "scylla"]
    out = analyze(
        str(tree),
        "--ablation",
        "--configs",
        *MIX_CONFIGS,
        "--time-limit",
        "60",
        "--oracle",
        *participants,
    ).stdout
    assert "oracle" in out


# ---------------------------------------------------------------------------
# #108 — headline run against the vanilla baseline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def headline_tree(tmp_path_factory):
    """The selected configuration at three seeds, plus the vanilla arm."""
    tmp_path = tmp_path_factory.mktemp("headline")
    names = [f"inst{i:02d}" for i in range(8)]
    data, listing, binary, record = make_run(tmp_path, names)
    unpatched = fake_highs(tmp_path, "highs-unpatched")
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(data),
        "--binary",
        str(binary),
        "--vanilla-binary",
        str(unpatched),
        "--output",
        str(tree),
        "--configs",
        "fj+fpr+local_mip",
        "vanilla",
        "--seeds",
        "0",
        "1",
        "2",
        "--time-limit",
        "600",
        env={"FAKE_HIGHS_RECORD": str(record)},
    )
    return tree, records(record), names, listing


def test_a_headline_pass_runs_three_seeds_against_a_separate_vanilla_binary(
    headline_tree,
):
    tree, _, names, _ = headline_tree
    for seed in (0, 1, 2):
        assert (
            sorted(p.stem for p in logs_under(tree, "fj+fpr+local_mip", seed)) == names
        )
    assert sorted(p.stem for p in logs_under(tree, "vanilla", 0)) == names


def test_which_binary_produced_a_config_is_readable_off_the_logs(headline_tree):
    # #108 requires binary provenance recorded with the tree, and says where
    # it has to come from: patched and unpatched builds of the same tag print
    # identical version banners, so the `mip-heuristics patch active` marker
    # is the only thing that separates them.  This is what make_archive.py
    # classifies each config on.
    tree, _, _, _ = headline_tree
    assert all(
        "mip-heuristics patch active" in log.read_text()
        for log in logs_under(tree, "fj+fpr+local_mip")
    )
    assert not any(
        "mip-heuristics patch active" in log.read_text()
        for log in logs_under(tree, "vanilla")
    )


def test_every_run_records_the_worker_count_it_ran_at(headline_tree):
    # Every stage has to record the effective worker count, and the harness
    # deliberately does not pin one — so it has to come out of the run.  HiGHS
    # prints it per solve, `parse_highs_log` reads it, and `make_archive.py`
    # carries it into MANIFEST.json / PROVENANCE.md as `workers_observed`.
    tree, _, _, _ = headline_tree
    for config in ("fj+fpr+local_mip", "vanilla"):
        for log in logs_under(tree, config):
            parsed = parse_log(log.read_text())
            assert parsed.thread_count == 16
            assert parsed.hardware_threads == 32


def test_the_headline_reports_the_plato_metric_against_the_baseline(headline_tree):
    tree, _, names, _ = headline_tree
    out = analyze(
        str(tree),
        "--configs",
        "fj+fpr+local_mip",
        "vanilla",
        "--time-limit",
        "600",
        "--baseline",
        "--summary",
    ).stdout
    assert "## PLATO Headline Metrics" in out
    assert "SGM ratio" in out
    assert f"({len(names)} instances" in out


def test_the_held_out_complement_is_the_same_tree_with_two_filters(headline_tree):
    # #108's secondary comparison: the 208 instances not used for tuning, as
    # `--instances plato --exclude-instances tuning` rather than a third list
    # that can drift out of step with the other two.
    tree, _, names, listing = headline_tree
    tuning = instance_list(tree.parent / "tuning.txt", names[:3])
    full = analyze(
        str(tree),
        "--configs",
        "fj+fpr+local_mip",
        "vanilla",
        "--time-limit",
        "600",
        "--baseline",
        "--summary",
        "--instances",
        str(listing),
    ).stdout
    complement = analyze(
        str(tree),
        "--configs",
        "fj+fpr+local_mip",
        "vanilla",
        "--time-limit",
        "600",
        "--baseline",
        "--summary",
        "--instances",
        str(listing),
        "--exclude-instances",
        str(tuning),
    ).stdout
    assert f"## PLATO Headline Metrics ({len(names)} instances" in full
    assert f"## PLATO Headline Metrics ({len(names) - 3} instances" in complement


def test_attribution_needs_no_developer_logging(headline_tree):
    # The per-heuristic attribution table reads incumbent source chars out of
    # an ordinary solve log, so the headline run stays uninstrumented.
    tree, recorded, _, _ = headline_tree
    assert not any("log_dev_level" in r["options"] for r in recorded)
    out = analyze(
        str(tree),
        "--configs",
        "fj+fpr+local_mip",
        "vanilla",
        "--time-limit",
        "600",
        "--attribution",
    ).stdout
    assert "## Heuristic attribution" in out
    assert "#First" in out and "#Best" in out
    assert "FJ" in out


def test_the_attribution_labels_match_the_source_codes_the_patch_defines():
    # The table is only as good as this map: the patch assigns one log char
    # per custom solution source, and analyze_results.py has to name the same
    # set.  A char added on one side and not the other silently files that
    # heuristic's solutions under `HiGHS/other`.
    text = APPLY_PATCH.read_text()
    block = text[text.index("kSolutionSourceFPR)") :]
    block = block[: block.index("kSolutionSourceCleanup)")]
    chars = set(re.findall(r'if \(code\) return \\"(\w)\\"', block))
    assert chars == set(CUSTOM_SOURCE_LABELS)


# ---------------------------------------------------------------------------
# run_plato.sh — the chunked launcher #105 uses
# ---------------------------------------------------------------------------


def plato_python_flags() -> set[str]:
    """The long flags run_plato.sh hands to run_benchmark.py."""
    text = RUN_PLATO.read_text()
    invocation = text[text.index("python3 bench/run_benchmark.py") :]
    invocation = invocation[: invocation.index("\n\n")]
    return set(re.findall(r"(--[a-z-]+)", invocation))


def test_run_plato_passes_only_flags_the_runner_defines():
    defined = {
        option
        for action in build_arg_parser()._actions
        for option in action.option_strings
    }
    assert plato_python_flags() <= defined


def test_run_plato_resumes_and_does_not_pin_threads():
    flags = plato_python_flags()
    assert {"--skip-existing", "--wall-time-budget"} <= flags
    assert "--threads" not in flags
    assert "--dev-log" not in flags


def test_run_plato_leaves_the_machine_idle_by_the_end_of_its_window(tmp_path):
    # `--wall-time-budget` stops *launching*; the instance already running
    # still gets its full limit.  So a chunk sized at the whole window can
    # overrun it by one time limit, which is 10 minutes of a morning at the
    # 600 s headline limit.  #109 states the rule: budget = window - limit.
    #
    # A `python3` that records its argv instead of running it.  Written in
    # `sh` rather than Python: it is first on PATH, so a `#!/usr/bin/env
    # python3` shebang would resolve back to the shim itself.
    shim = tmp_path / "python3"
    argv_log = tmp_path / "argv.txt"
    shim.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{argv_log}"\n')
    shim.chmod(0o755)
    binary = fake_highs(tmp_path)
    result = subprocess.run(
        ["bash", str(RUN_PLATO), "next", "2"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PLATO_BINARY": str(binary),
            "PLATO_VANILLA_BINARY": str(binary),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    invocation = argv_log.read_text()
    assert "--wall-time-budget 6600" in invocation  # 2h window, 600s limit
    assert "--skip-existing" in invocation
    assert "--time-limit 600" in invocation


def test_run_plato_runs_every_config_at_every_seed_and_reports_completion(tmp_path):
    # #106-#108 need more than the #105 shape, and they get it from the
    # environment rather than from a second launcher: configs, seeds, instance
    # list and output tree are all overridable, and `count_done` counts an
    # instance only once *every* seed has it.
    names = [f"inst{i:02d}" for i in range(3)]
    # Through `$MIPLIB_DIR`, not `--data-dir`: run_plato.sh does not pass one,
    # so this exercises the resolution a campaign machine actually uses.
    _, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "plato"
    result = subprocess.run(
        ["bash", str(RUN_PLATO), "next", "1"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env={
            **os.environ,
            "FAKE_HIGHS_RECORD": str(record),
            "MIPLIB_DIR": str(miplib_dir(tmp_path, names)),
            "PLATO_INSTANCES": str(listing),
            "PLATO_OUTPUT": str(tree),
            "PLATO_CONFIGS": "fj+fpr+local_mip vanilla",
            "PLATO_SEEDS": "0 1 2",
            "PLATO_BINARY": str(binary),
            "PLATO_VANILLA_BINARY": str(fake_highs(tmp_path, "highs-unpatched")),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for config in ("fj+fpr+local_mip", "vanilla"):
        for seed in (0, 1, 2):
            assert sorted(p.stem for p in logs_under(tree, config, seed)) == names
    assert "STATUS  : COMPLETE" in result.stdout
    # Completion is judged over every seed, so it cannot be reached by one.
    assert f"paired  : {len(names)} / {len(names)}" in result.stdout


def test_run_plato_sizes_its_chunk_from_the_reduced_limit_too(tmp_path):
    # The tuning stages run at a reduced limit, and the budget rule is
    # relative to it: 2h window at a 60s limit launches for 7140s, not 6600.
    shim = tmp_path / "python3"
    argv_log = tmp_path / "argv.txt"
    shim.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{argv_log}"\n')
    shim.chmod(0o755)
    binary = fake_highs(tmp_path)
    result = subprocess.run(
        ["bash", str(RUN_PLATO), "next", "2"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PLATO_BINARY": str(binary),
            "PLATO_VANILLA_BINARY": str(binary),
            "PLATO_TIME_LIMIT": "60",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    invocation = argv_log.read_text()
    assert "--time-limit 60" in invocation
    assert "--wall-time-budget 7140" in invocation


def test_run_plato_runs_a_budget_ladder_as_config_names(tmp_path):
    # #106's ladder is one config per point: `fpr@e<V>` is a legal config name
    # on its own, so a ladder needs no sweep flag and each point is its own
    # results directory that `status` can count.
    names = ["inst00"]
    _, listing, binary, record = make_run(tmp_path, names)
    tree = tmp_path / "ladder"
    ladder = ["fpr@e0.0055", "fpr@e0.0884", "fpr@e0.3536"]
    result = subprocess.run(
        ["bash", str(RUN_PLATO), "next", "1"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env={
            **os.environ,
            "FAKE_HIGHS_RECORD": str(record),
            "MIPLIB_DIR": str(miplib_dir(tmp_path, names)),
            "PLATO_INSTANCES": str(listing),
            "PLATO_OUTPUT": str(tree),
            "PLATO_CONFIGS": " ".join(ladder),
            "PLATO_SEEDS": "0 1",
            "PLATO_TIME_LIMIT": "60",
            "PLATO_BINARY": str(binary),
            "PLATO_VANILLA_BINARY": str(binary),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    delivered = {
        config_of(r): r["options"].get("mip_heuristic_fpr_effort")
        for r in records(record)
    }
    assert delivered == {name: name.partition("@e")[2] for name in ladder}
    assert "STATUS  : COMPLETE" in result.stdout


def test_run_plato_status_reads_a_tree_without_running_anything():
    result = subprocess.run(
        ["bash", str(RUN_PLATO), "status"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PLATO mipfeas progress" in result.stdout


# ---------------------------------------------------------------------------
# the real binary — what a stand-in cannot answer
# ---------------------------------------------------------------------------

HIGHS_BINARY = REPO / "build" / "bin" / "highs"
HIGHS_INSTANCES = REPO / "build" / "_deps" / "highs-src" / "check" / "instances"

requires_build = pytest.mark.skipif(
    not (HIGHS_BINARY.exists() and (HIGHS_INSTANCES / "egout.mps").exists()),
    reason="no local build: configure and build first (cmake -B build ...)",
)


def real_run(tmp_path: Path, configs: list[str], *extra: str) -> Path:
    """Run the real binary on one tiny instance through the real runner."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    listing = instance_list(tmp_path / "instances.txt", ["egout"])
    tree = tmp_path / "results"
    run_benchmark(
        "--instances",
        str(listing),
        "--data-dir",
        str(HIGHS_INSTANCES),
        "--binary",
        str(HIGHS_BINARY),
        "--output",
        str(tree),
        "--configs",
        *configs,
        "--time-limit",
        "5",
        *extra,
    )
    return tree


@requires_build
def test_the_real_binary_accepts_every_campaign_configuration(tmp_path):
    # The fake cannot answer this: whether HiGHS takes the sixteen suite
    # values #107 sweeps.  An unknown value is accepted by setOptionValue and
    # caught only at solve time, where the dispatcher fails open to all four
    # heuristics — which the runner turns into a `.log.err` rather than a
    # results row.  A tree with no `.err` file is the proof.
    tree = real_run(tmp_path, MIX_CONFIGS)
    for config in MIX_CONFIGS:
        seed_dir = tree / config / "seed0"
        assert (seed_dir / "egout.log").exists(), config
        assert not list(seed_dir.glob("*.log.err")), config
        assert "mip-heuristics patch active" in (seed_dir / "egout.log").read_text()


@requires_build
def test_the_effective_worker_count_is_recoverable_from_a_real_log(tmp_path):
    # The stand-in prints this line because HiGHS does; this is the test that
    # HiGHS still does, and that the pattern `parse_highs_log` and
    # `make_archive.py` match has not drifted from what the solver emits.
    tree = real_run(tmp_path, ["all"])
    parsed = parse_log((tree / "all" / "seed0" / "egout.log").read_text())
    assert parsed.thread_count, "HiGHS no longer reports its thread count"
    assert 0 < parsed.thread_count <= parsed.hardware_threads
    # HiGHS's pool size is what our presolve heuristics run at
    # (`ExecutionContext::num_workers`); `max_workers` is B&B's own cap.
    assert parsed.max_workers is not None


@requires_build
def test_the_real_binary_takes_every_per_heuristic_effort_option(tmp_path):
    # One ladder point per heuristic, each config sweeping its own option:
    # #106's inner loop, on the binary that has to accept those four names.
    tree = real_run(tmp_path, list(CHAIN), "--budget-sweep", "0.05")
    for name in CHAIN:
        seed_dir = tree / f"{name}@e0.05" / "seed0"
        assert (seed_dir / "egout.log").exists(), name
        assert not list(seed_dir.glob("*.log.err")), name
        opts = (seed_dir / "egout.opts").read_text()
        assert f"{CHAIN_EFFORT_OPTIONS[name]} = 0.05" in opts


@requires_build
def test_the_real_instrumentation_appears_only_with_dev_log(tmp_path):
    plain = parse_log(
        (
            real_run(tmp_path / "plain", ["all"]) / "all" / "seed0" / "egout.log"
        ).read_text()
    )
    assert plain.heuristic_samples == []
    assert plain.sequential_samples == []

    dev = parse_log(
        (
            real_run(tmp_path / "dev", ["all"], "--dev-log")
            / "all"
            / "seed0"
            / "egout.log"
        ).read_text()
    )
    assert dev.heuristic_samples, "log_dev_level=3 produced no [Heur] lines"
    assert {s.name for s in dev.heuristic_samples} <= {*CHAIN, "fpr_lp"}
    assert {s.phase for s in dev.heuristic_samples} <= {"presolve", "dive"}
    assert {s.heuristic for s in dev.sequential_samples} <= {*CHAIN, "fpr_lp"}


@requires_build
def test_the_real_effort_option_moves_the_charged_effort(tmp_path):
    # The measurement #106 reads: charged effort has to respond to the option
    # before a ladder over it means anything.  Pinned to one thread so the
    # two points are comparable rather than racing.
    tree = real_run(
        tmp_path,
        ["fpr"],
        "--budget-sweep",
        "0.0125",
        "1.00",
        "--dev-log",
        "--extra-options",
        "threads=1",
    )

    def charged(value: str) -> int:
        log = tree / f"fpr@e{value}" / "seed0" / "egout.log"
        return sum(
            s.effort
            for s in parse_log(log.read_text()).heuristic_samples
            if s.name == "fpr"
        )

    low, high = charged("0.0125"), charged("1.00")
    assert low > 0
    assert high > 2 * low, (low, high)
