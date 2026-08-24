"""Unit tests for bench/make_tuning_set.py.

Everything runs against logs synthesised in a tmp dir: no results tree, no
MIPLIB, no solver.  The point of the script is that a subset is a derived,
checkable artifact, so its own inputs have to be constructible anywhere.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_tuning_set import (
    DEFAULT_BOUNDARIES,
    aggregate_time,
    allocate,
    assign_stratum,
    parse_boundaries,
    stratum_labels,
)
from run_benchmark import load_instances

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_tuning_set.py")

# A HiGHS MIP log line, as `parse_highs_log._LOG_LINE_RE` expects it:
# source char, node counts, explored %, bounds, gap, cut/lp/conflict counts,
# LP iterations, elapsed seconds.
_HEADER = (
    "Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol"
    "              Gap | Cuts InLp Confl. | LpIters     Time\n"
)
_REPORT = (
    "Solving report\n"
    "  Status            Time limit reached\n"
    "  Primal bound      10\n"
    "  Nodes             1\n"
)


def feasible_log(seconds: float) -> str:
    """A run whose first incumbent lands at `seconds`."""
    return (
        _HEADER + f"H       0       0         0   0.00%          0              10"
        f"              Large      0      0      0       0.0   {seconds}s\n" + _REPORT
    )


def infeasible_log() -> str:
    """A finished run that never found a feasible solution."""
    return _HEADER + "Solving report\n  Status            Time limit reached\n"


def solved_without_incumbent_log() -> str:
    """A solved run whose incumbent line carries an unrecognised source code.

    `Q` is not in `parse_highs_log._INCUMBENT_SOURCES`, so the incumbent is
    dropped and `time_to_first_feasible` comes back None even though the
    report proves a solution was found — what a HiGHS bump that adds a
    source code looks like from here.
    """
    return (
        _HEADER + "Q       0       0         0   0.00%          0              10"
        "              Large      0      0      0       0.0   1.2s\n"
        "Solving report\n"
        "  Status            Optimal\n"
        "  Primal bound      10\n"
    )


def truncated_log() -> str:
    """A killed run: banner only, no solving report and no incumbent."""
    return "Running HiGHS 1.15.1\nmip-heuristics patch active\n"


def write_tree(
    tmp_path,
    times: dict[str, object],
    config: str = "vanilla",
    seeds: tuple[int, ...] = (0,),
    name: str = "results",
) -> str:
    """Materialise `<tmp>/<name>/<config>/seed<N>/<instance>.log`.

    A `times` value is either one entry for every seed or a per-seed dict.
    `None` means never feasible, `"truncated"` a killed run, `"sourcegap"` a
    solve whose incumbent source code the parser does not recognise, and
    `"err"` a failed run parked as `<instance>.log.err` the way
    run_benchmark.py does.
    """
    root = os.path.join(str(tmp_path), name)
    for seed in seeds:
        seed_dir = os.path.join(root, config, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        for instance, value in times.items():
            per_seed = value.get(seed, "absent") if isinstance(value, dict) else value
            if per_seed == "absent":
                continue
            path = os.path.join(seed_dir, f"{instance}.log")
            if per_seed == "err":
                with open(path + ".err", "w") as f:
                    f.write("HiGHS exited 1\n")
                continue
            if per_seed == "truncated":
                body = truncated_log()
            elif per_seed == "sourcegap":
                body = solved_without_incumbent_log()
            elif per_seed is None:
                body = infeasible_log()
            else:
                body = feasible_log(float(per_seed))
            with open(path, "w") as f:
                f.write(body)
    return root


def write_reference(tmp_path, names, filename: str = "reference.txt") -> str:
    path = os.path.join(str(tmp_path), filename)
    with open(path, "w") as f:
        f.write("# reference list\n")
        f.write("".join(f"{n}\n" for n in names))
    return path


def run(*args: str, hash_seed: str | None = None) -> subprocess.CompletedProcess[str]:
    # `sys.executable` so the test follows whichever interpreter drives
    # pytest, rather than a hard-coded .venv path that CI does not have.
    env = dict(os.environ)
    if hash_seed is not None:
        # Varying PYTHONHASHSEED perturbs set and dict iteration order across
        # processes; the draw must not move with it.
        env["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


# ---------------------------------------------------------------------------
# Stratum assignment
# ---------------------------------------------------------------------------


def test_labels_are_derived_from_the_boundaries():
    assert stratum_labels(DEFAULT_BOUNDARIES) == [
        "<1s",
        "1-10s",
        "10-100s",
        "100-600s",
        ">=600s",
        "never",
    ]
    assert stratum_labels((0.5, 2.0)) == ["<0.5s", "0.5-2s", ">=2s", "never"]


def test_strata_are_half_open_on_the_right():
    b = DEFAULT_BOUNDARIES
    assert assign_stratum(0.0, b) == "<1s"
    assert assign_stratum(0.999, b) == "<1s"
    # A boundary value belongs to the stratum above it.
    assert assign_stratum(1.0, b) == "1-10s"
    assert assign_stratum(9.99, b) == "1-10s"
    assert assign_stratum(10.0, b) == "10-100s"
    assert assign_stratum(599.9, b) == "100-600s"


def test_never_feasible_is_distinct_from_the_overflow_bucket():
    b = DEFAULT_BOUNDARIES
    # Found a solution at or past the time limit: hard, but not never.
    assert assign_stratum(600.0, b) == ">=600s"
    assert assign_stratum(612.5, b) == ">=600s"
    assert assign_stratum(None, b) == "never"
    assert assign_stratum(float("inf"), b) == "never"


def test_parse_boundaries_rejects_unusable_splits():
    assert parse_boundaries("1,10, 100 ") == (1.0, 10.0, 100.0)
    # `inf` used to pass every check here and then raise OverflowError out
    # of the label formatter, which is a traceback where every other bad
    # boundary is a clean exit 1.
    for bad in ("", "1,1", "10,1", "0,10", "-5", "1,x", "inf", "1e400", "nan"):
        try:
            parse_boundaries(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_aggregate_time_takes_the_lower_middle_across_seeds():
    inf = float("inf")
    # One never-feasible seed out of two must not make the instance never.
    assert aggregate_time([2.0, inf]) == 2.0
    # A majority of never-feasible seeds still lands in the never bucket.
    assert math.isinf(aggregate_time([2.0, inf, inf]))
    assert aggregate_time([5.0, 1.0, 3.0]) == 3.0


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def test_allocation_is_proportional_and_sums_to_size():
    counts = [120, 60, 30, 20, 0, 3]
    alloc = allocate(counts, 40, min_per_stratum=0)
    assert sum(alloc) == 40
    assert all(a <= n for a, n in zip(alloc, counts))
    assert alloc[4] == 0  # empty stratum draws nothing
    # Ordering follows the populations.
    assert alloc[0] > alloc[1] > alloc[2] > alloc[3]


def test_largest_remainder_breaks_ties_by_stratum_order():
    # Three equal strata, four seats: floors give 1 each and the leftover
    # seat goes to the first stratum, deterministically.
    assert allocate([10, 10, 10], 4, min_per_stratum=0) == [2, 1, 1]
    # No remainder to hand out at all.
    assert allocate([10, 10, 10], 3, min_per_stratum=0) == [1, 1, 1]


def test_min_per_stratum_keeps_a_tiny_stratum_alive():
    # Also the float-artifact regression case.  The three remainders are a
    # mathematical tie (4/12 each), but as floats the third sorts first, so a
    # float implementation returns [0, 0, 4] and breaks the stated
    # "ties by stratum order" rule.
    counts = [1, 1, 10]
    # Strict proportionality drops the two singleton strata entirely...
    assert allocate(counts, 4, min_per_stratum=0) == [1, 0, 3]
    # ... which is exactly what the floor exists to prevent.
    alloc = allocate(counts, 4, min_per_stratum=1)
    assert alloc == [1, 1, 2]
    assert sum(alloc) == 4


def test_allocation_never_exceeds_a_stratum_population():
    counts = [1, 9]
    alloc = allocate(counts, 5, min_per_stratum=3)
    assert alloc == [1, 4]
    assert sum(alloc) == 5


def test_allocation_properties_hold_over_a_sweep():
    """The bound `alloc <= population` is a property, not a clamp.

    `allocate` does no capping and makes a single pass over the remainder
    order, both justified by an argument in the comments rather than by
    defensive code.  This sweeps every small shape to keep that argument
    honest.
    """
    for counts in itertools.product(range(4), repeat=4):
        total = sum(counts)
        nonempty = sum(1 for n in counts if n > 0)
        for min_per in range(3):
            reserved = sum(min(min_per, n) for n in counts)
            for size in range(total + 1):
                if size < reserved:
                    continue
                alloc = allocate(list(counts), size, min_per)
                assert sum(alloc) == size, (counts, size, min_per, alloc)
                assert all(a <= n for a, n in zip(alloc, counts))
                assert all(a >= 0 for a in alloc)
                # Every non-empty stratum keeps its reserved seat.
                assert all(a >= min(min_per, n) for a, n in zip(alloc, counts) if n > 0)
                assert all(a == 0 for a, n in zip(alloc, counts) if n == 0)
                if nonempty and size == total:
                    assert list(alloc) == list(counts)


def test_allocation_refuses_what_it_cannot_satisfy():
    for counts, size, min_per in (
        ([2, 2], 5, 0),  # more draws than instances
        ([5, 5, 5], 2, 1),  # too few draws to floor every stratum
        ([5, 5], -1, 0),  # nonsense size
    ):
        try:
            allocate(counts, size, min_per)
        except ValueError:
            continue
        raise AssertionError(f"accepted counts={counts} size={size} min={min_per}")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _spread(n_per_stratum: int = 4) -> dict[str, object]:
    """Instances spread over every default stratum, including never."""
    times: dict[str, object] = {}
    for i in range(n_per_stratum):
        times[f"imm{i}"] = 0.1 + i
        times[f"fast{i}"] = 2.0 + i
        times[f"mod{i}"] = 20.0 + i
        times[f"hard{i}"] = 200.0 + i
        times[f"never{i}"] = None
    # `imm0` is the only one below 1s; keep the rest of the immediate names
    # honest by pinning them under the boundary.
    for i in range(n_per_stratum):
        times[f"imm{i}"] = 0.1 + 0.1 * i
    return times


def test_end_to_end_produces_a_stratified_list(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))
    out = os.path.join(str(tmp_path), "subset.txt")

    res = run(tree, "--instances", ref, "--size", "10", "--seed", "0", "--output", out)
    assert res.returncode == 0, res.stdout + res.stderr

    with open(out) as f:
        text = f.read()
    picked = load_instances(out)
    assert len(picked) == 10
    assert set(picked) <= set(times)
    # Every stratum is represented: 20 instances, 5 non-empty strata, 10 draws.
    for label in ("<1s", "1-10s", "10-100s", "100-600s", "never"):
        assert f"# {label} (" in text
    # The report on stderr shows both distributions.
    assert "stratum" in res.stderr and "TOTAL" in res.stderr


def test_never_feasible_instances_are_sampled_not_dropped(tmp_path):
    # Half the set never becomes feasible; the bucket must be a stratum, not
    # a filter.
    times: dict[str, object] = {f"feas{i}": 1.5 for i in range(6)}
    times.update({f"none{i}": None for i in range(6)})
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))
    out = os.path.join(str(tmp_path), "subset.txt")

    res = run(tree, "--instances", ref, "--size", "6", "--output", out)
    assert res.returncode == 0, res.stdout + res.stderr

    with open(out) as f:
        text = f.read()
    picked = load_instances(out)
    assert len(picked) == 6
    assert sum(1 for name in picked if name.startswith("none")) == 3
    assert "# never (3 of 6)" in text
    # Each entry records the value it was binned on.
    assert "# never" in text and "# 1.50s" in text


def test_header_records_source_seed_boundaries_and_counts(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times, seeds=(0, 1))
    ref = write_reference(tmp_path, sorted(times))
    out = os.path.join(str(tmp_path), "subset.txt")

    res = run(tree, "--instances", ref, "--size", "10", "--seed", "7", "--output", out)
    assert res.returncode == 0, res.stdout + res.stderr
    with open(out) as f:
        text = f.read()

    assert f"source_tree      {tree}" in text
    assert "config           vanilla" in text
    assert "config_seeds     0, 1" in text
    assert f"reference_list   {ref} (20 instances)" in text
    assert "sample_seed      7" in text
    assert "sample_size      10" in text
    assert "boundaries_s     1, 10, 100, 600" in text
    assert "min_per_stratum  1" in text
    # Per-stratum counts of the full set beside the sample.
    assert "#   <1s              4" in text
    assert "#   TOTAL           20" in text
    # And the command that reproduces the file.
    assert "make_tuning_set.py" in text and "--seed 7" in text
    # No timestamp: a generation date would break byte-identity.
    assert "202" not in text.split("\n\n")[0].replace(tree, "").replace(ref, "")


def test_same_tree_and_seed_are_byte_identical(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    first = run(tree, "--instances", ref, "--size", "8", "--seed", "3")
    second = run(tree, "--instances", ref, "--size", "8", "--seed", "3")
    assert first.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert first.stdout  # the list goes to stdout, the report to stderr
    assert "stratum" not in first.stdout.split("\n")[0]

    other = run(tree, "--instances", ref, "--size", "8", "--seed", "4")
    assert other.returncode == 0, other.stderr
    assert other.stdout != first.stdout


def test_draw_ignores_reference_list_order(tmp_path):
    """The draw must be a function of the tree, not of the input's order.

    `test_same_tree_and_seed_are_byte_identical` runs one tree twice in one
    environment, so it cannot see this: deleting the `sorted()` in either
    `sample_stratum` or `build_selection` leaves it green while the drawn set
    moves as soon as the reference list is reordered.  Rewriting the *same*
    reference path keeps the header identical, so the whole file is comparable
    byte for byte.
    """
    times = _spread(6)
    tree = write_tree(tmp_path, times)
    names = sorted(times)
    ref = write_reference(tmp_path, names)

    first = run(tree, "--instances", ref, "--size", "12", "--seed", "5")
    assert first.returncode == 0, first.stderr

    shuffled = list(names)
    random.Random(11).shuffle(shuffled)
    assert shuffled != names
    write_reference(tmp_path, shuffled)  # same path, different order

    # Vary PYTHONHASHSEED at the same time: set and dict iteration order must
    # not reach the draw either.
    second = run(
        tree, "--instances", ref, "--size", "12", "--seed", "5", hash_seed="12345"
    )
    assert second.returncode == 0, second.stderr
    assert second.stdout == first.stdout


def test_draw_ignores_on_disk_creation_order(tmp_path):
    """Nor may it depend on the order the logs were written in.

    This holds today because `analyze_results.load_results` sorts its globs,
    so creation order never reaches this script — the test pins the property
    end to end rather than the mechanism that currently provides it.
    """
    times = _spread(6)
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    first = run(tree, "--instances", ref, "--size", "12", "--seed", "5")
    assert first.returncode == 0, first.stderr

    # Same tree path, same contents, opposite creation order.
    shutil.rmtree(tree)
    write_tree(tmp_path, dict(reversed(list(times.items()))))

    second = run(tree, "--instances", ref, "--size", "12", "--seed", "5")
    assert second.returncode == 0, second.stderr
    assert second.stdout == first.stdout


def test_boundaries_are_configurable_and_recorded(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    res = run(tree, "--instances", ref, "--size", "6", "--boundaries", "5,50")
    assert res.returncode == 0, res.stderr
    assert "boundaries_s     5, 50" in res.stdout
    assert "# <5s (" in res.stdout
    assert "--boundaries 5,50" in res.stdout


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_missing_instance_refuses_rather_than_sampling_a_partial_set(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, [*sorted(times), "absent_instance"])

    res = run(tree, "--instances", ref, "--size", "5")
    assert res.returncode == 2
    assert "absent_instance" in res.stderr
    assert "no usable log" in res.stderr
    assert res.stdout == ""


def test_failed_run_is_reported_as_such(tmp_path):
    times = _spread()
    times["broken"] = "err"
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    res = run(tree, "--instances", ref, "--size", "5")
    assert res.returncode == 2
    assert "broken" in res.stderr
    assert ".log.err" in res.stderr


def test_truncated_log_is_refused_not_binned_as_never_feasible(tmp_path):
    # The trap this guards: a killed run parses into an empty result that is
    # indistinguishable from a genuine never-feasible solve.
    times = _spread()
    times["killed"] = "truncated"
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    res = run(tree, "--instances", ref, "--size", "5")
    assert res.returncode == 2
    assert "killed" in res.stderr
    assert "truncated" in res.stderr
    assert res.stdout == ""


def test_solved_log_without_an_incumbent_line_is_refused(tmp_path):
    """The core risk: a solved instance binned as never-feasible.

    An unrecognised incumbent source code leaves `time_to_first_feasible`
    None on a log that reports `Status Optimal` and a finite primal bound.
    Bins as `never` if it is accepted — and `never` is normally the smallest
    stratum, so `--min-per-stratum` then reserves the misfiled instance a
    seat.  `parse_highs_log` already warns about exactly this condition.
    """
    times = _spread()
    times["sourcegap"] = "sourcegap"
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    res = run(tree, "--instances", ref, "--size", "5")
    assert res.returncode == 2
    assert "sourcegap" in res.stderr
    assert "_INCUMBENT_SOURCES" in res.stderr
    assert res.stdout == ""


def test_partial_seed_coverage_refuses_until_allowed(tmp_path):
    times: dict[str, object] = dict(_spread())
    times["patchy"] = {0: 3.0}  # present for seed 0, absent for seed 1
    tree = write_tree(tmp_path, times, seeds=(0, 1))
    ref = write_reference(tmp_path, sorted(times))

    res = run(tree, "--instances", ref, "--size", "5")
    assert res.returncode == 2
    assert "patchy" in res.stderr
    assert "only some" in res.stderr
    assert "--allow-incomplete-seeds" in res.stderr

    res = run(tree, "--instances", ref, "--size", "5", "--allow-incomplete-seeds")
    assert res.returncode == 0, res.stderr
    assert "partial_seeds    allowed for 1 instance(s)" in res.stdout
    assert "--allow-incomplete-seeds" in res.stdout


def test_empty_tree_and_bad_arguments_are_usage_errors(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    assert run(os.path.join(str(tmp_path), "nope"), "--instances", ref).returncode == 1
    assert (
        run(tree, "--instances", os.path.join(str(tmp_path), "nope.txt")).returncode
        == 1
    )
    assert run(tree, "--instances", ref, "--boundaries", "10,1").returncode == 1
    assert run(tree, "--instances", ref, "--config", "ghost").returncode == 1
    # More draws than the tree holds is a refusal, not a truncated sample.
    res = run(tree, "--instances", ref, "--size", "999")
    assert res.returncode == 1
    assert "999" in res.stderr


# ---------------------------------------------------------------------------
# Tree shapes
# ---------------------------------------------------------------------------


def test_config_is_auto_detected_and_overridable(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times, config="vanilla")
    write_tree(tmp_path, {name: 42.0 for name in times}, config="all")
    ref = write_reference(tmp_path, sorted(times))

    # Several configs: the vanilla arm wins, since stratifying on a patched
    # arm would measure the thing under test.
    res = run(tree, "--instances", ref, "--size", "5")
    assert res.returncode == 0, res.stderr
    assert "config           vanilla" in res.stdout

    res = run(tree, "--instances", ref, "--size", "5", "--config", "all")
    assert res.returncode == 0, res.stderr
    assert "config           all" in res.stdout
    assert "# 42.00s" in res.stdout


def test_a_config_directory_can_be_passed_directly(tmp_path):
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))

    res = run(os.path.join(tree, "vanilla"), "--instances", ref, "--size", "5")
    assert res.returncode == 0, res.stderr
    assert "config           vanilla" in res.stdout


def test_instances_outside_the_reference_list_are_ignored_with_a_note(tmp_path):
    times = _spread()
    times["stranger"] = 4.0
    tree = write_tree(tmp_path, times)
    ref = write_reference(
        tmp_path, sorted(name for name in times if name != "stranger")
    )

    res = run(tree, "--instances", ref, "--size", "5")
    assert res.returncode == 0, res.stderr
    assert "1 instance(s) in the tree are not in" in res.stderr
    assert "stranger" not in res.stdout


# ---------------------------------------------------------------------------
# The informative candidate pool (issue #113)
# ---------------------------------------------------------------------------


def test_informative_pool_restricts_only_the_candidates(tmp_path):
    """#113 samples from what a presolve-only screen can see.

    The stratification, the aggregation and the coverage contract are
    unchanged; only which instances may be *drawn* narrows.  The stratum
    table therefore reports the pool's populations, not the tree's.
    """
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))
    pool_names = ["imm0", "imm1", "never0", "never1"]
    pool = write_reference(tmp_path, pool_names, filename="informative.txt")
    out = os.path.join(str(tmp_path), "subset.txt")

    res = run(
        tree,
        "--instances",
        ref,
        "--informative-instances",
        pool,
        "--size",
        "3",
        "--output",
        out,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    picked = load_instances(out)
    assert len(picked) == 3
    assert set(picked) <= set(pool_names)
    # Only the two strata the pool populates exist; the tree's other 16
    # instances are not candidates and are not counted.
    with open(out) as f:
        text = f.read()
    assert "# <1s (" in text and "# never (" in text
    assert "# 10-100s (" not in text
    assert "informative_set" in text and "sha256:" in text
    assert "--informative-instances" in text


def test_informative_pool_does_not_relax_the_coverage_refusal(tmp_path):
    """The tree must still cover the whole reference list.

    The instances a campaign failed to run are not a random subset of it, so
    a hole outside the pool is still a refusal — the pool narrows candidates,
    it does not narrow what the tree has to have run.
    """
    times = _spread()
    times["mod0"] = {0: "absent"}
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))
    pool = write_reference(tmp_path, ["imm0", "never0"], filename="informative.txt")

    res = run(tree, "--instances", ref, "--informative-instances", pool, "--size", "2")
    assert res.returncode == 2
    assert "mod0" in res.stderr


def test_informative_pool_must_be_a_subset_of_the_reference_list(tmp_path):
    """A pool naming unknown instances is a mismatched pair, not a filter."""
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))
    pool = write_reference(tmp_path, ["imm0", "ghost"], filename="informative.txt")

    res = run(tree, "--instances", ref, "--informative-instances", pool, "--size", "1")
    assert res.returncode == 1
    assert "ghost" in res.stderr

    missing = run(
        tree,
        "--instances",
        ref,
        "--informative-instances",
        os.path.join(str(tmp_path), "nope.txt"),
        "--size",
        "1",
    )
    assert missing.returncode == 1


def test_informative_pool_digest_pins_the_file_not_just_its_path(tmp_path):
    """The header records which *version* of the pool produced the list."""
    times = _spread()
    tree = write_tree(tmp_path, times)
    ref = write_reference(tmp_path, sorted(times))
    pool = write_reference(tmp_path, ["imm0", "imm1", "never0"], filename="inf.txt")
    out = os.path.join(str(tmp_path), "subset.txt")

    def emit() -> str:
        res = run(
            tree,
            "--instances",
            ref,
            "--informative-instances",
            pool,
            "--size",
            "2",
            "--output",
            out,
        )
        assert res.returncode == 0, res.stderr
        with open(out) as f:
            return f.read()

    first = emit()
    assert emit() == first  # byte-identical on a rerun
    with open(pool, "a") as f:
        f.write("# a comment, no new names\n")
    second = emit()
    assert second != first
    # The names it draws are unchanged; only the recorded digest moved.
    assert load_instances(out) == [
        line.split("#")[0].strip()
        for line in first.splitlines()
        if line and not line.startswith("#")
    ]
