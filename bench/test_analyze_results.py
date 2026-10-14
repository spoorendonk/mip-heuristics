"""Unit tests for analyze_results helpers."""

from __future__ import annotations

import csv
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import shutil
import subprocess
from pathlib import Path

from analyze_results import (
    USABLE_REFERENCE_TAGS,
    _config_metrics,
    aggregate_results,
    build_best_known,
    build_oracle_config,
    classify_reference_status,
    contradicted_reference_instances,
    count_first,
    count_wins,
    filter_results,
    get_common_instances,
    heuristic_attribution,
    latex_ablation_table,
    load_results,
    parse_solu_file,
    read_instance_list,
    source_label,
)
from parse_highs_log import (
    KILLED_STATUS,
    Incumbent,
    SolveResult,
    parse_log,
)


def _result(t_first: float | None) -> SolveResult:
    r = SolveResult()
    if t_first is not None:
        r.incumbents.append(Incumbent(time=t_first, objective=1.0, source="H", nodes=0))
    return r


def test_count_first_strict_winner_takes_full_credit():
    agg = {
        "patched": {"a": _result(2.0), "b": _result(1.0)},
        "vanilla": {"a": _result(10.0), "b": _result(30.0)},
    }
    firsts = count_first(agg, ["patched", "vanilla"], ["a", "b"])
    assert firsts == {"patched": 2.0, "vanilla": 0.0}


def test_count_first_ties_within_tolerance_split_credit():
    # patched at 2.00, vanilla at 2.05 → within 0.1s tolerance → each gets 0.5.
    agg = {
        "patched": {"x": _result(2.00)},
        "vanilla": {"x": _result(2.05)},
    }
    firsts = count_first(agg, ["patched", "vanilla"], ["x"])
    assert firsts["patched"] == 0.5
    assert firsts["vanilla"] == 0.5


def test_count_first_skips_instance_with_no_feasible():
    agg = {
        "patched": {"x": _result(None)},
        "vanilla": {"x": _result(None)},
    }
    firsts = count_first(agg, ["patched", "vanilla"], ["x"])
    assert firsts == {"patched": 0.0, "vanilla": 0.0}


def test_count_first_credits_only_feasible_finders_when_others_miss():
    agg = {
        "patched": {"x": _result(5.0)},
        "vanilla": {"x": _result(None)},
    }
    firsts = count_first(agg, ["patched", "vanilla"], ["x"])
    assert firsts == {"patched": 1.0, "vanilla": 0.0}


# ── heuristic attribution ─────────────────────────────────────────────────────


def _result_sources(sources: list[str]) -> SolveResult:
    """Build a SolveResult with one incumbent per source char (improving)."""
    r = SolveResult()
    for i, s in enumerate(sources, start=1):
        r.incumbents.append(
            Incumbent(time=float(i), objective=1.0 / i, source=s, nodes=0)
        )
    return r


def test_source_label_maps_custom_and_buckets_builtin():
    # Custom display chars from the patched-build Src legend.
    assert source_label("A") == "FPR"
    assert source_label("D") == "FPR_LP"
    assert source_label("M") == "LocalMIP"
    assert source_label("G") == "Scylla"
    assert source_label("J") == "FJ"
    # Anything else (HiGHS built-ins, presolve 'P', empty) is bucketed.
    assert source_label("B") == "HiGHS/other"
    assert source_label("P") == "HiGHS/other"
    assert source_label("") == "HiGHS/other"


def test_heuristic_attribution_first_and_best():
    agg = {
        "c": {
            "i1": _result_sources(["J", "A", "M"]),  # first FJ, best LocalMIP
            "i2": _result_sources(["A"]),  # first & best FPR
            "i3": SolveResult(),  # infeasible — skipped
            "i4": _result_sources(["B", "B"]),  # HiGHS/other first & best
        }
    }
    attr = heuristic_attribution(agg, "c", ["i1", "i2", "i3", "i4"])
    assert attr["n_feasible"] == 3
    assert attr["first"] == {"FJ": 1, "FPR": 1, "HiGHS/other": 1}
    assert attr["best"] == {"LocalMIP": 1, "FPR": 1, "HiGHS/other": 1}


def test_heuristic_attribution_totals_equal_feasible():
    agg = {"c": {f"i{i}": _result_sources(["J"]) for i in range(5)}}
    attr = heuristic_attribution(agg, "c", [f"i{i}" for i in range(5)])
    assert attr["n_feasible"] == 5
    assert sum(attr["first"].values()) == 5
    assert sum(attr["best"].values()) == 5


# ── ablation LaTeX table ──────────────────────────────────────────────────────


def test_latex_ablation_table_escapes_and_rows():
    metrics = {
        "all_opp": {
            "feasible": 213.0,
            "sgm_t1st": 3.3,
            "sgm_gap": 0.02,
            "sgm_pi": 22.0,
            "mipfeas_sgm": 17.3,
        },
        "loo_no_fj": {
            "feasible": 200.0,
            "sgm_t1st": 4.0,
            "sgm_gap": 0.03,
            "sgm_pi": 25.0,
            "mipfeas_sgm": 19.0,
        },
    }
    tex = latex_ablation_table(["all_opp", "loo_no_fj"], metrics, 233, 100.0)
    assert r"\begin{tabular}{lrrrrr}" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"loo\_no\_fj" in tex  # underscore escaped for LaTeX
    assert "213" in tex  # #Feas rendered as int
    assert r"all\_opp" in tex


# ── load_results NAME=DIR override ────────────────────────────────────────────


def test_load_results_config_dir_override(tmp_path: Path):
    # A config can be loaded from an explicit directory (used to pull the
    # ablation anchors from bench/results/mipfeas).  Build a tiny seed0 tree.
    real_log = Path(__file__).with_name("results") / "mipfeas" / "patched" / "seed0"
    seed_dir = tmp_path / "anchor" / "seed0"
    seed_dir.mkdir(parents=True)
    sample = next(real_log.glob("*.log"), None)
    if sample is None:
        # No committed logs in this checkout — synthesize a minimal one.
        (seed_dir / "toy.log").write_text(
            "      Status      Time limit reached\n      Primal bound inf\n"
        )
        inst_name = "toy"
    else:
        shutil.copy(sample, seed_dir / sample.name)
        inst_name = sample.stem
    loaded = load_results(
        str(tmp_path / "does_not_exist"),
        ["anchor"],
        {"anchor": str(tmp_path / "anchor")},
    )
    assert "anchor" in loaded
    assert 0 in loaded["anchor"]
    assert inst_name in loaded["anchor"][0]


# ── config directory names ───────────────────────────────────────────────────


def _tiny_tree(root: Path, config: str) -> None:
    seed_dir = root / config / "seed0"
    seed_dir.mkdir(parents=True)
    (seed_dir / "toy.log").write_text(
        "      Status      Time limit reached\n      Primal bound inf\n"
    )


def test_load_results_reads_subset_config_directories(tmp_path: Path):
    """Subset config names join with `+`, and that is a directory name."""
    configs = ["fj+fpr", "fpr+local_mip+scylla", "vanilla"]
    for config in configs:
        _tiny_tree(tmp_path, config)
    loaded = load_results(str(tmp_path), configs)
    assert sorted(loaded) == sorted(configs)
    for config in configs:
        assert "toy" in loaded[config][0]


def test_latex_ablation_table_escapes_underscores_in_config_names():
    """`+` is an ordinary character in LaTeX text mode; `_` still is not."""
    metrics = {
        "fpr+local_mip": {
            "feasible": 5.0,
            "sgm_t1st": 1.0,
            "sgm_gap": 0.01,
            "sgm_pi": 2.0,
            "mipfeas_sgm": 1.5,
        },
    }
    tex = latex_ablation_table(["fpr+local_mip"], metrics, 5, 60.0)
    assert r"fpr+local\_mip" in tex


# ── Instance filtering, the config oracle, and the reference guard (#104) ─────


def _feasible_log(*, objective: float, found_at: float, solve_time: float) -> str:
    """A minimal solved-instance log: one incumbent plus the solving report.

    The incumbent line is what `primal_integral` integrates, so the objective
    and the time it was found are the two knobs a test needs to make one
    config beat another under the headline metric.
    """
    return (
        f"H       0       0         0   0.00%          0"
        f"              {objective}              Large"
        f"      0      0      0       0.0   {found_at}s\n"
        "      Status            Optimal\n"
        f"      Primal bound      {objective}\n"
        "      Nodes             17\n"
        f"      Timing            {solve_time:.2f}\n"
    )


def _write_seeded_tree(root: Path, tree: dict[str, dict[int, dict[str, str]]]) -> None:
    """{config: {seed: {instance: log}}} -> <root>/<config>/seed<N>/<inst>.log."""
    for config, seeds in tree.items():
        for seed, logs in seeds.items():
            seed_dir = root / config / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for inst, text in logs.items():
                (seed_dir / f"{inst}.log").write_text(text)


# --- instance list parsing and include/exclude filtering ---------------------


def test_read_instance_list_ignores_comments_blanks_and_duplicates(tmp_path: Path):
    path = tmp_path / "list.txt"
    path.write_text(
        "# a header comment\n"
        "\n"
        "alpha\n"
        "beta   # trailing comment\n"
        "alpha\n"
        "   gamma   \n"
        "   \n"
    )
    assert read_instance_list(str(path)) == ["alpha", "beta", "gamma"]


def _two_config_tree(tmp_path: Path, instances: list[str]):
    tree = {
        cfg: {
            0: {
                i: _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0)
                for i in instances
            }
        }
        for cfg in ("patched", "vanilla")
    }
    _write_seeded_tree(tmp_path, tree)
    return load_results(str(tmp_path), ["patched", "vanilla"])


def test_include_list_restricts_every_config(tmp_path: Path):
    results = _two_config_tree(tmp_path, ["a", "b", "c"])
    filtered, filt = filter_results(results, include=["a", "c"])

    assert filt.kept == ["a", "c"]
    assert filt.dropped == ["b"]
    for cfg in ("patched", "vanilla"):
        assert sorted(filtered[cfg][0]) == ["a", "c"]
    # The original tree is not mutated — callers may still need the full set.
    assert sorted(results["patched"][0]) == ["a", "b", "c"]


def test_exclude_list_removes_named_instances(tmp_path: Path):
    results = _two_config_tree(tmp_path, ["a", "b", "c"])
    filtered, filt = filter_results(results, exclude=["b"])

    assert filt.kept == ["a", "c"]
    assert sorted(filtered["vanilla"][0]) == ["a", "c"]


def test_include_then_exclude_expresses_the_held_out_complement(tmp_path: Path):
    """The whole point of having both: the complement needs no third file.

    `--instances mipfeas --exclude-instances tuning` is the held-out set, and it
    cannot drift out of sync with the tuning list the way a materialised
    complement file would.
    """
    results = _two_config_tree(tmp_path, ["a", "b", "c", "d"])
    filtered, filt = filter_results(results, include=["a", "b", "c"], exclude=["b"])

    assert filt.kept == ["a", "c"]
    assert "d" in filt.dropped and "b" in filt.dropped
    assert sorted(filtered["patched"][0]) == ["a", "c"]


def test_filter_reports_names_absent_from_the_tree(tmp_path: Path):
    """A typo in a list file otherwise restricts the report to silence."""
    results = _two_config_tree(tmp_path, ["a", "b"])
    _, filt = filter_results(results, include=["a", "typo"], exclude=["alsotypo"])

    assert filt.unknown_included == ["typo"]
    assert filt.unknown_excluded == ["alsotypo"]
    assert filt.kept == ["a"]


# --- the config oracle ------------------------------------------------------


# Every synthetic log below reports dual bound 0 against objective 10, so with
# no reference objective the primal gap saturates at 1.0 and every config
# integrates to the same area.  Passing an explicit reference is what makes the
# metric discriminate — and it is what the real pipeline does, via
# `build_best_known`.
_REFS = {"i1": 10.0, "i2": 10.0}


def _split_decision_tree(tmp_path: Path) -> None:
    """fpr wins i1, scylla wins i2 — a decision no single config can make."""
    _write_seeded_tree(
        tmp_path,
        {
            "fpr": {
                0: {
                    "i1": _feasible_log(objective=10.0, found_at=0.5, solve_time=5.0),
                    "i2": _feasible_log(objective=10.0, found_at=4.5, solve_time=5.0),
                }
            },
            "scylla": {
                0: {
                    "i1": _feasible_log(objective=10.0, found_at=4.5, solve_time=5.0),
                    "i2": _feasible_log(objective=10.0, found_at=0.5, solve_time=5.0),
                }
            },
        },
    )


def test_oracle_selects_the_better_config_per_instance(tmp_path: Path):
    """Selection is per instance, so a split decision must produce a split row."""
    _split_decision_tree(tmp_path)
    results = load_results(str(tmp_path), ["fpr", "scylla"])
    tree, report = build_oracle_config(results, ["fpr", "scylla"], _REFS, 5.0)

    assert report.formed
    assert report.row_picks == {"i1": "fpr", "i2": "scylla"}
    assert report.pick_counts == {"fpr": 1, "scylla": 1}
    # The oracle row carries the winning config's actual result object.
    assert tree[0]["i1"] is results["fpr"][0]["i1"]
    assert tree[0]["i2"] is results["scylla"][0]["i2"]


def test_oracle_aggregates_after_selection_not_on_the_aggregate(tmp_path: Path):
    """Per-instance selection must strictly beat picking one config outright.

    If selection happened on the aggregate, the oracle would merely equal
    whichever single config had the better total.
    """
    _split_decision_tree(tmp_path)
    results = load_results(str(tmp_path), ["fpr", "scylla"])
    tree, _ = build_oracle_config(results, ["fpr", "scylla"], _REFS, 5.0)

    def total_pi(rows):
        return sum(r.primal_integral(5.0, _REFS[i]) for i, r in rows.items())

    oracle_pi = total_pi(tree[0])
    assert oracle_pi < total_pi(results["fpr"][0])
    assert oracle_pi < total_pi(results["scylla"][0])


def test_oracle_drops_instances_missing_from_any_participant(tmp_path: Path):
    """Incompleteness is reported, never silently compared across two sets."""
    log = _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0)
    other = _feasible_log(objective=10.0, found_at=2.0, solve_time=5.0)
    _write_seeded_tree(
        tmp_path,
        {
            "fpr": {0: {"shared": log, "only_fpr": log}},
            "scylla": {0: {"shared": other}},
        },
    )
    results = load_results(str(tmp_path), ["fpr", "scylla"])
    _, report = build_oracle_config(results, ["fpr", "scylla"], {}, 5.0)

    assert report.instances == ["shared"]
    assert report.dropped == ["only_fpr"]


def test_oracle_drops_an_instance_missing_from_one_seed_only(tmp_path: Path):
    """Present in both configs, but not at every shared seed — still dropped.

    Otherwise the oracle would score that instance over fewer seeds than the
    rest of the row: the same different-instance-sets bug in a different hat.
    """
    good = _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0)
    _write_seeded_tree(
        tmp_path,
        {
            "fpr": {0: {"i1": good, "i2": good}, 1: {"i1": good, "i2": good}},
            # i2 never ran at seed 1
            "scylla": {0: {"i1": good, "i2": good}, 1: {"i1": good}},
        },
    )
    results = load_results(str(tmp_path), ["fpr", "scylla"])
    _, report = build_oracle_config(results, ["fpr", "scylla"], {}, 5.0)

    assert report.seeds == [0, 1]
    assert report.instances == ["i1"]
    assert report.dropped == ["i2"]


def test_oracle_selects_within_a_seed_never_across_seeds(tmp_path: Path):
    """The stated rule: per-seed selection, aggregate after.

    `fpr` is superb at seed 0 and dreadful at seed 1; `scylla` is mediocre at
    both.  An oracle allowed to pick the seed would report fpr's seed-0 result
    everywhere — a number no selector could achieve, since nobody chooses the
    RNG outcome.  The honest oracle must take scylla at seed 1.
    """
    _write_seeded_tree(
        tmp_path,
        {
            "fpr": {
                0: {"i1": _feasible_log(objective=10.0, found_at=0.1, solve_time=5.0)},
                1: {"i1": _feasible_log(objective=10.0, found_at=4.9, solve_time=5.0)},
            },
            "scylla": {
                0: {"i1": _feasible_log(objective=10.0, found_at=2.0, solve_time=5.0)},
                1: {"i1": _feasible_log(objective=10.0, found_at=2.0, solve_time=5.0)},
            },
        },
    )
    results = load_results(str(tmp_path), ["fpr", "scylla"])
    tree, report = build_oracle_config(results, ["fpr", "scylla"], _REFS, 5.0)

    # The per-seed winner is still computed and reported, as a diagnostic of
    # how stable the choice is across seeds.
    assert report.seed_picks[(0, "i1")] == "fpr"
    assert report.seed_picks[(1, "i1")] == "scylla"

    # But the ROW must not be fpr's lucky seed-0 run.  fpr is superb at seed 0
    # and dreadful at seed 1, so its seed-collapsed row is the dreadful one and
    # scylla's steady row wins.  Every seed of the oracle carries that row.
    for s in (0, 1):
        assert tree[s]["i1"].time_to_first_feasible == 2.0
    assert report.row_picks["i1"] == "scylla"


def test_oracle_reports_a_participant_absent_from_the_tree(tmp_path: Path):
    log = _feasible_log(objective=1.0, found_at=1.0, solve_time=5.0)
    _write_seeded_tree(tmp_path, {"fpr": {0: {"i1": log}}})
    results = load_results(str(tmp_path), ["fpr"])
    _, report = build_oracle_config(results, ["fpr", "nosuch"], {}, 5.0)

    assert report.missing_participants == ["nosuch"]
    assert report.participants == ["fpr"]


def test_oracle_ties_break_deterministically_by_argument_order(tmp_path: Path):
    same = _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0)
    _write_seeded_tree(tmp_path, {"a": {0: {"i1": same}}, "b": {0: {"i1": same}}})
    results = load_results(str(tmp_path), ["a", "b"])

    _, first = build_oracle_config(results, ["a", "b"], _REFS, 5.0)
    _, second = build_oracle_config(results, ["b", "a"], _REFS, 5.0)
    assert first.row_picks["i1"] == "a"
    assert second.row_picks["i1"] == "b"


# --- the unusable-reference guard -------------------------------------------


def test_reference_status_separates_contradicted_from_merely_unpublished():
    """`=inf=` is a contradiction; a missing entry is just missing.

    The distinction matters: an unpublished reference falls back to the
    virtual best (or to the benchmark's gap=1.0 convention) and is sound, whereas a
    reference the file says cannot exist makes the gap self-referential.
    """
    solu = {
        "solved": ("=opt=", 5.0),
        "bestknown": ("=best=", 7.0),
        "feasible": ("=fea=", 9.0),
        "claimed_infeasible": ("=inf=", None),
        "unbounded": ("=unbd=", None),
        "unknown": ("=unkn=", None),
    }
    instances = [*solu, "not_in_file"]
    status = classify_reference_status(instances, solu)

    assert status["solved"] == "published"
    assert status["bestknown"] == "published"
    assert status["feasible"] == "published"
    assert status["claimed_infeasible"] == "contradicted"
    assert status["unbounded"] == "contradicted"
    assert status["unknown"] == "unpublished"
    assert status["not_in_file"] == "unpublished"

    assert contradicted_reference_instances(instances, solu) == [
        "claimed_infeasible",
        "unbounded",
    ]


def test_no_solu_file_contradicts_nothing():
    """An absent solution file must not empty the benchmark."""
    assert contradicted_reference_instances(["a", "b"], {}) == []


def test_bundled_mipfeas_set_has_a_usable_reference_for_every_instance():
    """Regression for the supportcase22 anomaly (issue #104).

    The bundled solution file and the mipfeas list are two halves of one claim —
    233 instances with a reference objective each.  When they disagree the SGM
    silently changes meaning, so the agreement is asserted rather than assumed.
    """
    bench = os.path.dirname(os.path.abspath(__file__))
    instances = read_instance_list(os.path.join(bench, "instances_mipfeas.txt"))
    assert len(instances) == 233

    solu = parse_solu_file(os.path.join(bench, "miplib2017-v36.solu"))
    status = classify_reference_status(instances, solu)
    assert [i for i in instances if status[i] != "published"] == []

    # supportcase22 is the instance that motivated this check: the previous
    # bundled file marked it `=inf=`.  Pin the property that matters — it has
    # a usable reference — not the exact tag and value, so a legitimate future
    # refresh that proves 110 optimal (`=opt=`) does not fail a correct update.
    tag, value = solu["supportcase22"]
    assert tag in USABLE_REFERENCE_TAGS
    assert value is not None


# --- end-to-end through the CLI ---------------------------------------------


def _run_cli(tmp_path: Path, *args: str):
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_results.py"
    )
    return subprocess.run(
        [sys.executable, script, str(tmp_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _plain_tree(tmp_path: Path) -> None:
    logs = {
        "a": _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0),
        "b": _feasible_log(objective=20.0, found_at=2.0, solve_time=5.0),
        "c": _feasible_log(objective=30.0, found_at=3.0, solve_time=5.0),
    }
    _write_seeded_tree(
        tmp_path, {"patched": {0: dict(logs)}, "vanilla": {0: dict(logs)}}
    )


def test_cli_include_and_exclude_restrict_the_reported_counts(tmp_path: Path):
    _plain_tree(tmp_path)
    keep = tmp_path / "keep.txt"
    keep.write_text("a\nb\n")
    drop = tmp_path / "drop.txt"
    drop.write_text("b\n")

    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--instances",
        str(keep),
        "--exclude-instances",
        str(drop),
        "--time-limit",
        "5",
    )
    assert res.returncode == 0, res.stderr
    assert "## Instance selection" in res.stdout
    assert "1 instances retained" in res.stdout
    # Every table states the count it covers, and it is the restricted one.
    assert "## Paper Metrics (1 instances" in res.stdout
    assert "## Per-instance comparison: patched vs vanilla (1 instances)" in res.stdout
    assert "3 instances" not in res.stdout


def test_cli_reports_the_instance_count_in_every_table(tmp_path: Path):
    """A restricted run must not be mistakable for a full one in any table."""
    _plain_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--baseline",
        "--attribution",
        "--time-limit",
        "5",
    )
    assert res.returncode == 0, res.stderr
    for heading in (
        "## Per-instance comparison: patched vs vanilla (3 instances)",
        "## Paper Metrics (3 instances",
        "### Category breakdown (#Feas / #Win) (3 instances)",
        "## Heuristic attribution (3 instances)",
        "## mipfeas Headline Metrics (3 instances",
    ):
        assert heading in res.stdout, f"missing count in: {heading}"


def test_cli_oracle_row_appears_under_the_headline_metric(tmp_path: Path):
    _split_decision_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "fpr",
        "scylla",
        "--ablation",
        "--oracle",
        "fpr",
        "scylla",
        "--time-limit",
        "5",
    )
    assert res.returncode == 0, res.stderr
    assert "## Config oracle" in res.stdout
    assert "Row 'oracle' = best of: fpr, scylla" in res.stdout
    assert "2 instances" in res.stdout
    assert "0 dropped." in res.stdout

    # The oracle is a row of the ablation table, under the same columns, and
    # its headline SGM beats both participants — the ceiling is above them.
    ablation = res.stdout.split("## Ablation summary", 1)[1]
    mipfeas_sgm = {
        ln.split()[0]: float(ln.split()[-1])
        for ln in ablation.splitlines()
        if ln.split() and ln.split()[0] in ("fpr", "scylla", "oracle")
    }
    assert set(mipfeas_sgm) == {"fpr", "scylla", "oracle"}
    assert mipfeas_sgm["oracle"] < mipfeas_sgm["fpr"]
    assert mipfeas_sgm["oracle"] < mipfeas_sgm["scylla"]


def test_cli_oracle_states_its_seed_rule_in_its_output(tmp_path: Path):
    """The issue requires the multi-seed rule be stated, not just implemented."""
    _split_decision_tree(tmp_path)
    res = _run_cli(
        tmp_path, "--configs", "fpr", "scylla", "--summary", "--oracle", "fpr", "scylla"
    )
    assert res.returncode == 0, res.stderr
    assert "never sees an individual seed" in res.stdout
    assert "more pick a lucky seed" in res.stdout
    assert "ceiling" in res.stdout


def test_cli_oracle_name_collision_is_refused(tmp_path: Path):
    _plain_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--oracle-name",
        "patched",
        "--oracle",
        "patched",
        "vanilla",
    )
    assert res.returncode == 1
    assert "collides with a real config" in res.stderr


def test_cli_refuses_to_fold_in_an_instance_with_no_valid_reference(tmp_path: Path):
    """The supportcase22 class of bug: `=inf=` must not enter the SGM quietly."""
    _plain_tree(tmp_path)
    solu = tmp_path / "refs.solu"
    solu.write_text("=opt=  a  10.0\n=opt=  c  30.0\n=inf=  b\n")

    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--solu",
        str(solu),
        "--baseline",
        "--time-limit",
        "5",
    )
    assert res.returncode == 0, res.stderr
    assert "## Unusable reference objectives" in res.stdout
    assert "Excluded: b" in res.stdout
    # ... and it really is out of the aggregates, not merely mentioned.
    assert "## Paper Metrics (2 instances" in res.stdout
    assert "## mipfeas Headline Metrics (2 instances" in res.stdout


# --- the ceiling invariant, which is what the whole feature promises --------


def _mipfeas_sgm_per_config(results, configs, oracle_name=None, time_limit=5.0):
    """mipfeas headline SGM per config, through the real reporting pipeline."""
    common = get_common_instances(results, configs)
    best_known = build_best_known(results, configs, common, {}, time_limit)
    all_configs = list(configs)
    if oracle_name is not None:
        tree, report = build_oracle_config(
            results,
            configs,
            best_known,
            time_limit,
            name=oracle_name,
            instances=common,
        )
        assert report.formed, report.refused
        results = {**results, oracle_name: tree}
        all_configs = [*configs, oracle_name]
        common = get_common_instances(results, all_configs)
    agg = aggregate_results(results, all_configs, time_limit)
    return {
        c: _config_metrics(results, agg, c, common, time_limit, best_known)[
            "mipfeas_sgm"
        ]
        for c in all_configs
    }


def test_oracle_is_a_ceiling_on_the_headline_metric_with_multiple_seeds(tmp_path: Path):
    """`oracle <= min(participants)` on mipfeas SGM. Two seeds, the failing shape.

    This exact tree was found by search against the first implementation,
    which selected per (instance, seed) on primal integral and then let
    `aggregate_results` collapse the picks by *median primal bound*.  Mixing
    the two criteria threw the per-seed wins away and produced an oracle SGM
    of 3.44 against a best participant of 2.82 — a "ceiling" below the thing
    it was meant to bound.
    """
    tree = {
        "A": {
            0: {"i0": (10.0, 0.31), "i1": (4.0, 4.26)},
            1: {"i0": (10.0, 3.04), "i1": (4.0, 1.59)},
        },
        "B": {
            0: {"i0": (20.0, 0.96), "i1": (5.0, 1.48)},
            1: {"i0": (4.0, 1.47), "i1": (10.0, 2.96)},
        },
    }
    _write_seeded_tree(
        tmp_path,
        {
            c: {
                s: {
                    i: _feasible_log(objective=o, found_at=t, solve_time=5.0)
                    for i, (o, t) in insts.items()
                }
                for s, insts in seeds.items()
            }
            for c, seeds in tree.items()
        },
    )
    results = load_results(str(tmp_path), ["A", "B"])
    sgm = _mipfeas_sgm_per_config(results, ["A", "B"], oracle_name="oracle")

    assert sgm["oracle"] <= min(sgm["A"], sgm["B"]) + 1e-9, sgm


def test_oracle_ceiling_holds_across_randomised_multi_seed_trees(tmp_path: Path):
    """Property test: the ceiling must hold for every tree, not a lucky one.

    The defect it guards against showed up in roughly 2% of random two-seed
    trees, so a single hand-built case is not enough coverage to keep it out.
    """
    rng = random.Random(20260820)
    for trial in range(40):
        root = tmp_path / f"t{trial}"
        n_seeds = rng.choice([2, 3])
        spec = {
            c: {
                s: {
                    i: _feasible_log(
                        objective=float(rng.choice([4, 5, 10, 20, 100])),
                        found_at=round(rng.uniform(0.05, 4.95), 2),
                        solve_time=5.0,
                    )
                    for i in ("i0", "i1", "i2")
                }
                for s in range(n_seeds)
            }
            for c in ("A", "B", "C")
        }
        _write_seeded_tree(root, spec)
        results = load_results(str(root), ["A", "B", "C"])
        sgm = _mipfeas_sgm_per_config(results, ["A", "B", "C"], oracle_name="oracle")
        best = min(sgm[c] for c in ("A", "B", "C"))
        assert sgm["oracle"] <= best + 1e-9, (trial, sgm)


def test_oracle_row_dominates_participants_instance_by_instance(tmp_path: Path):
    """The ceiling is per instance, which is what makes the SGM one follow."""
    _split_decision_tree(tmp_path)
    results = load_results(str(tmp_path), ["fpr", "scylla"])
    common = get_common_instances(results, ["fpr", "scylla"])
    tree, _ = build_oracle_config(
        results, ["fpr", "scylla"], _REFS, 5.0, instances=common
    )
    agg = aggregate_results(results, ["fpr", "scylla"], 5.0)
    for inst in common:
        oracle_pi = tree[0][inst].primal_integral(5.0, _REFS[inst])
        for c in ("fpr", "scylla"):
            assert oracle_pi <= agg[c][inst].primal_integral(5.0, _REFS[inst]) + 1e-12


# --- oracle guard rails -----------------------------------------------------


def test_oracle_refuses_a_single_participant(tmp_path: Path):
    """An oracle over one config is that config relabelled — worse than nothing."""
    log = _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0)
    _write_seeded_tree(tmp_path, {"patched": {0: {"i1": log}}})
    results = load_results(str(tmp_path), ["patched"])
    tree, report = build_oracle_config(results, ["patched", "absent"], _REFS, 5.0)

    assert not report.formed
    assert tree == {}
    assert report.refused is not None
    assert "at least 2" in report.refused


def test_oracle_drops_an_instance_a_non_participant_config_lacks(tmp_path: Path):
    """MUST-FIX regression: an instance outside the tables' common set.

    `best_known` is built over the cross-config common set, so an instance a
    NON-participant config never ran has no reference; scoring it made
    `primal_integral` fall back to the dual bound and the tie-break handed it
    to whichever participant was named first.  Worse, the oracle reported it
    as covered while the table underneath silently dropped it.
    """
    fast = _feasible_log(objective=10.0, found_at=0.5, solve_time=5.0)
    slow = _feasible_log(objective=10.0, found_at=4.5, solve_time=5.0)
    _write_seeded_tree(
        tmp_path,
        {
            "fpr": {0: {"i1": fast, "i2": slow}},
            "scylla": {0: {"i1": slow, "i2": fast}},
            "combined": {0: {"i1": fast}},  # never ran i2
        },
    )
    results = load_results(str(tmp_path), ["fpr", "scylla", "combined"])
    common = get_common_instances(results, ["fpr", "scylla", "combined"])
    assert common == ["i1"]
    best_known = build_best_known(
        results, ["fpr", "scylla", "combined"], common, {}, 5.0
    )

    _, report = build_oracle_config(
        results, ["fpr", "scylla"], best_known, 5.0, instances=common
    )
    # i2 is reported as dropped, and reported under the reason that applies.
    assert report.instances == ["i1"]
    assert report.dropped == ["i2"]
    assert report.dropped_not_common == ["i2"]
    assert report.dropped_incomplete == []
    # ... and it is NOT mis-credited to the first-named participant.
    assert report.pick_counts["fpr"] + report.pick_counts["scylla"] == 1


def test_oracle_row_does_not_move_head_to_head_counts(tmp_path: Path):
    """Additive reporting: the oracle ties with the run it copied, so leaving
    it in the field would halve that config's #First credit."""
    _split_decision_tree(tmp_path)
    results = load_results(str(tmp_path), ["fpr", "scylla"])
    common = get_common_instances(results, ["fpr", "scylla"])
    tree, _ = build_oracle_config(
        results, ["fpr", "scylla"], _REFS, 5.0, instances=common
    )
    results = {**results, "oracle": tree}
    configs = ["fpr", "scylla", "oracle"]
    agg = aggregate_results(results, configs, 5.0)

    without = count_first(agg, ["fpr", "scylla"], common)
    with_oracle = count_first(agg, configs, common, synthetic={"oracle"})
    assert with_oracle["fpr"] == without["fpr"] == 1.0
    assert with_oracle["scylla"] == without["scylla"] == 1.0
    assert with_oracle["oracle"] == 0.0

    wins_without = count_wins(agg, ["fpr", "scylla"], common)
    wins_with = count_wins(agg, configs, common, synthetic={"oracle"})
    assert wins_with["fpr"] == wins_without["fpr"]
    assert wins_with["scylla"] == wins_without["scylla"]
    assert wins_with["oracle"] == 0


def test_cli_oracle_is_excluded_from_head_to_head_columns(tmp_path: Path):
    _split_decision_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "fpr",
        "scylla",
        "--oracle",
        "fpr",
        "scylla",
        "--time-limit",
        "5",
    )
    assert res.returncode == 0, res.stderr
    for row in ("#Win (best obj)", "#First (fastest T1st)"):
        line = next(ln for ln in res.stdout.splitlines() if ln.startswith(row))
        assert line.split()[-1] == "-", line


def _killed_log(*, incumbents: list[tuple[float, float]], killed_after: float) -> str:
    """A log the harness cut short: incumbent lines, then its kill marker."""
    lines = [
        f"H       0       0         0   0.00%          0"
        f"              {objective}              Large"
        f"      0      0      0       0.0   {found_at}s\n"
        for found_at, objective in incumbents
    ]
    return "".join(lines) + f"TIMEOUT: process killed after {killed_after:.0f}s\n"


def _killed(incumbents: list[tuple[float, float]]) -> SolveResult:
    r = SolveResult(killed=True, status=KILLED_STATUS)
    for t, obj in incumbents:
        r.incumbents.append(Incumbent(time=t, objective=obj, source="H", nodes=0))
    return r


def test_primal_at_reads_a_killed_run_off_its_last_incumbent_at_the_limit():
    # #175: a killed run has no Solving report, so `primal_bound` stays inf.
    r = _killed([(10.0, 70.0), (99.5, 50.0), (100.5, 45.0), (105.0, 40.0)])
    assert r.primal_bound == float("inf")
    assert r.primal_at(100.0) == 45.0  # within the grace past the limit
    assert r.primal_at(50.0) == 70.0
    assert r.primal_at(5.0) == float("inf")
    assert r.primal_at(None) == 40.0
    assert _killed([]).primal_at(100.0) == float("inf")


def test_primal_at_reads_a_finished_run_off_its_report():
    r = _result(1.0)  # one incumbent at objective 1.0
    r.primal_bound = 0.5
    assert r.primal_at(100.0) == 0.5
    assert r.primal_at(None) == 0.5


def test_primal_at_keeps_the_report_of_a_run_killed_after_it():
    # A report followed by a kill marker: HiGHS's own word stands, even
    # against an incumbent the cutoff would read differently.
    r = parse_log(
        _feasible_log(objective=60.0, found_at=40.0, solve_time=100.0)
        + "TIMEOUT: process killed after 120s\n"
    )
    assert r.killed
    assert r.status != KILLED_STATUS
    assert r.primal_at(100.0) == 60.0
    assert r.primal_at(10.0) == 60.0  # the incumbent at 40 s is past this cutoff


def test_count_wins_credits_a_killed_run_by_its_incumbents():
    # The issue's fixture: vanilla killed at 50 against a finished 60 wins,
    # killed at 302 against a finished 302 ties.
    finished = SolveResult(primal_bound=60.0)
    tie = SolveResult(primal_bound=302.0)
    agg = {
        "all": {"30n20b8": finished, "air05": tie},
        "vanilla": {
            "30n20b8": _killed([(20.0, 80.0), (90.0, 50.0)]),
            "air05": _killed([(30.0, 302.0)]),
        },
    }
    configs = ["all", "vanilla"]
    insts = ["30n20b8", "air05"]
    assert count_wins(agg, configs, insts, time_limit=100.0) == {"all": 0, "vanilla": 1}
    # An incumbent past the limit and its grace does not count: 50 arrives
    # too late, so vanilla holds 80 and the finished 60 wins.
    assert count_wins(agg, configs, insts, time_limit=80.0) == {"all": 1, "vanilla": 0}


def test_cli_win_counts_read_killed_runs_off_their_incumbents(tmp_path: Path):
    _write_seeded_tree(
        tmp_path,
        {
            "all": {
                0: {
                    "30n20b8": _feasible_log(
                        objective=60.0, found_at=40.0, solve_time=100.0
                    ),
                    "air05": _feasible_log(
                        objective=302.0, found_at=40.0, solve_time=100.0
                    ),
                }
            },
            "vanilla": {
                0: {
                    "30n20b8": _killed_log(
                        incumbents=[(90.0, 50.0)], killed_after=120.0
                    ),
                    "air05": _killed_log(
                        incumbents=[(30.0, 302.0)], killed_after=120.0
                    ),
                }
            },
        },
    )
    res = _run_cli(tmp_path, "--configs", "all", "vanilla", "--time-limit", "100")
    assert res.returncode == 0, res.stderr
    line = next(
        ln for ln in res.stdout.splitlines() if ln.startswith("#Win (best obj)")
    )
    assert line.split()[-2:] == ["0", "1"], line


def test_cli_missing_instance_list_is_a_clean_error(tmp_path: Path):
    _plain_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--instances",
        str(tmp_path / "nope.txt"),
    )
    assert res.returncode == 1
    assert "cannot read instance list" in res.stderr
    assert "Traceback" not in res.stderr


# --- the published formula and the paired statistics -------------------------


def test_paired_reproduces_the_hand_arithmetic():
    import math

    from analyze_results import paired

    # log ratios: log 0.5, 0, log 2 -> mean 0, sample sd = log 2.
    st = paired([1.0, 2.0, 4.0], [2.0, 2.0, 2.0], shift=0.0)
    assert st.n == 3
    assert st.ratio == pytest.approx(1.0)
    assert st.sd == pytest.approx(math.log(2.0))
    assert st.t == 0.0
    assert st.p == pytest.approx(1.0)
    se = math.log(2.0) / math.sqrt(3.0)
    # t_{2, 0.975} = 4.302652729911275; the normal interval is kept beside it.
    assert st.ci_hi == pytest.approx(math.exp(4.302652729911275 * se))
    assert st.ci_lo == pytest.approx(1.0 / st.ci_hi)
    assert st.normal_ci_hi == pytest.approx(math.exp(1.959963984540054 * se))
    assert st.normal_ci_lo == pytest.approx(1.0 / st.normal_ci_hi)
    assert st.normal_p == pytest.approx(1.0)
    assert st.median_ratio == pytest.approx(1.0)
    assert (st.better, st.tied, st.worse) == (1, 1, 1)
    assert st.pct == pytest.approx(0.0)


def test_paired_shift_keeps_a_zero_metric_finite():
    from analyze_results import paired

    # An instance solved at t = 0 has primal integral exactly 0; the shift is
    # what makes its log ratio defined.
    st = paired([0.0, 1.0], [1.0, 1.0], shift=0.001)
    assert st.ratio > 0.0 and st.ratio < 1.0
    assert (st.better, st.tied, st.worse) == (1, 1, 0)


def test_sign_test_is_the_normal_approximation_without_continuity_correction():
    from analyze_results import sign_test

    assert sign_test(57, 61) == pytest.approx(0.7127, abs=1e-3)
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 10) == pytest.approx(1.0)


def test_min_detectable_decrease_is_read_on_the_improvement_side():
    import math

    from analyze_results import min_detectable_decrease

    d = (1.959963984540054 + 0.8416212335729143) * 0.8 / math.sqrt(143)
    assert min_detectable_decrease(0.8, 143) == pytest.approx(1.0 - math.exp(-d))
    assert min_detectable_decrease(0.8, 143) < math.exp(d) - 1.0


def test_bundled_benchmark_reference_covers_every_mipfeas_instance():
    """The benchmark's z* file and the mipfeas list are two halves of one claim."""
    from analyze_results import parse_mipfeas_z

    bench = os.path.dirname(os.path.abspath(__file__))
    z = parse_mipfeas_z(os.path.join(bench, "mipfeas_optimal_objective.csv"))
    instances = read_instance_list(os.path.join(bench, "instances_mipfeas.txt"))
    assert len(instances) == 233
    assert [i for i in instances if i not in z] == []
    assert all(isinstance(v, float) for v in z.values())


def test_apply_reference_formula_prefers_the_benchmark_z_under_mipfeas():
    from analyze_results import apply_reference_formula, formula, set_formula

    before = formula()
    try:
        set_formula("mipfeas")
        assert apply_reference_formula({"a": 5.0, "b": None}, {"a": 7.0}) == {
            "a": 7.0,
            "b": None,
        }
        set_formula("campaign")
        assert apply_reference_formula({"a": 5.0}, {"a": 7.0}) == {"a": 5.0}
        with pytest.raises(ValueError):
            set_formula("neither")
    finally:
        set_formula(before)


def _write_z(tmp_path: Path, values: dict[str, float]) -> Path:
    z = tmp_path / "z.csv"
    lines = ['"InstanceInst.","ObjectiveObje."']
    lines += [f'"{k}","{v}"' for k, v in values.items()]
    z.write_text("\n".join(lines) + "\n")
    return z


def test_cli_paired_prints_one_block_per_pair(tmp_path: Path):
    _plain_tree(tmp_path)
    z = _write_z(tmp_path, {"a": 10.0, "b": 20.0, "c": 30.0})
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--paired",
        "--time-limit",
        "5",
        "--mipfeas-z",
        str(z),
    )
    assert res.returncode == 0, res.stderr
    assert (
        "## Paired per-instance log-ratio (3 instances, 5 s, shift=0.001, formula=mipfeas)"
        in res.stdout
    )
    # Identical logs in both arms: ratio 1, every instance tied, on both metrics.
    lines = [
        line
        for line in res.stdout.splitlines()
        if "patched/vanilla" in line or "final gap" in line
    ]
    assert len(lines) == 2
    assert "ratio=1.000" in lines[0] and "better/tied/worse 0/3/0" in lines[0]
    assert "n=  3" in lines[1] and "(both feasible)" in lines[1]
    assert "## Paper Metrics" not in res.stdout


def test_cli_mipfeas_formula_is_the_default_and_needs_the_reference_file(
    tmp_path: Path,
):
    _plain_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--time-limit",
        "5",
        "--mipfeas-z",
        str(tmp_path / "absent.csv"),
    )
    assert res.returncode == 1
    assert "reference objectives" in res.stderr

    z = _write_z(tmp_path, {"a": 10.0, "b": 20.0, "c": 30.0})
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--time-limit",
        "5",
        "--baseline",
        "--mipfeas-z",
        str(z),
    )
    assert res.returncode == 0, res.stderr
    assert "formula=mipfeas" in res.stdout
    assert "SGM T1st (s=1, none=T)" in res.stdout
    assert "SGM PrimalInt (s=0.001)" in res.stdout


def test_cli_campaign_formula_restores_the_campaign_reading(tmp_path: Path):
    _plain_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--time-limit",
        "5",
        "--baseline",
        "--formula",
        "campaign",
    )
    assert res.returncode == 0, res.stderr
    assert "formula=campaign" in res.stdout
    assert "SGM T1st (s=1)  " in res.stdout
    assert "SGM PrimalInt (s=1)" in res.stdout


def test_cli_exclude_instances_is_repeatable(tmp_path: Path):
    _plain_tree(tmp_path)
    z = _write_z(tmp_path, {"a": 10.0, "b": 20.0, "c": 30.0})
    drop_b = tmp_path / "drop_b.txt"
    drop_b.write_text("b\n")
    drop_c = tmp_path / "drop_c.txt"
    drop_c.write_text("c\n")
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--time-limit",
        "5",
        "--mipfeas-z",
        str(z),
        "--exclude-instances",
        str(drop_b),
        "--exclude-instances",
        str(drop_c),
    )
    assert res.returncode == 0, res.stderr
    assert "1 instances retained" in res.stdout
    assert "## Paper Metrics (1 instances" in res.stdout


def test_aggregate_results_picks_the_median_seed_at_the_limit():
    # #185: seed 0 finished at 100, seeds 1 and 2 were killed with last
    # incumbents 50 and 60.  By `primal_bound` they sort [100, inf, inf] and
    # seed 1 (50) is picked; at the limit they are [50, 60, 100], median 60.
    finished = SolveResult(primal_bound=100.0)
    killed_50 = _killed([(10.0, 80.0), (90.0, 50.0)])
    killed_60 = _killed([(30.0, 60.0)])
    results = {
        "cfg": {0: {"inst": finished}, 1: {"inst": killed_50}, 2: {"inst": killed_60}}
    }
    agg = aggregate_results(results, ["cfg"], 100.0)
    assert agg["cfg"]["inst"] is killed_60


def test_aggregate_results_reads_a_killed_seed_at_the_limit_only():
    # An incumbent past the limit does not count: seed 1's 10 at 150 s is
    # after a 100 s limit, so at the limit it holds 90 and is the worst seed.
    a = SolveResult(primal_bound=40.0)
    b = _killed([(20.0, 90.0), (150.0, 10.0)])
    c = SolveResult(primal_bound=70.0)
    results = {"cfg": {0: {"inst": a}, 1: {"inst": b}, 2: {"inst": c}}}
    assert aggregate_results(results, ["cfg"], 100.0)["cfg"]["inst"] is c


def test_build_best_known_takes_a_killed_runs_incumbent_at_the_limit():
    # #186: a killed run has `primal_bound` inf, so its 50 never entered the
    # observed reference and the finished run's 100 did.
    finished = SolveResult(primal_bound=100.0)
    killed = _killed([(10.0, 80.0), (90.0, 50.0), (150.0, 5.0)])
    results = {"a": {0: {"inst": finished}}, "b": {0: {"inst": killed}}}
    refs = build_best_known(results, ["a", "b"], ["inst"], {}, 100.0)
    # The 5 found at 150 s is past a 100 s limit and does not count.
    assert refs == {"inst": 50.0}
    # A published reference the killed run beat is improved by it too.
    solu = {"inst": ("=best=", 70.0)}
    assert build_best_known(results, ["a", "b"], ["inst"], solu, 100.0) == {
        "inst": 50.0
    }


# --- opt. found, opt. proven, time SGM and the per-run CSV (#176) -----------


def _report_log(
    *, incumbents: list[tuple[float, float]], status: str, primal: float, timing: float
) -> str:
    """Incumbent lines, then a Solving report with this status and timing."""
    lines = [
        f"H       0       0         0   0.00%          0"
        f"              {objective}              Large"
        f"      0      0      0       0.0   {found_at}s\n"
        for found_at, objective in incumbents
    ]
    return "".join(lines) + (
        f"      Status            {status}\n"
        f"      Primal bound      {primal}\n"
        "      Nodes             17\n"
        f"      Timing            {timing:.2f}\n"
    )


def _opt_tree(tmp_path: Path) -> Path:
    """Five runs at a 100 s limit, one per case the three new columns separate.

    a  proven at 40 s on z*             found, proven, time 40
    b  killed, its incumbent is z*      found,          time T (parser says 0)
    c  Optimal, but only at 101 s       found,          time T
    d  Optimal at 20 s, 40.001 vs 40    proven, not found (penalty 2.5e-5), 20
    e  time limit reached at 60 vs 50   neither,        time T
    """
    logs = {
        "a": _report_log(
            incumbents=[(5.0, 10.0)], status="Optimal", primal=10.0, timing=40.0
        ),
        "b": _killed_log(incumbents=[(30.0, 20.0)], killed_after=170.0),
        "c": _report_log(
            incumbents=[(50.0, 30.0)], status="Optimal", primal=30.0, timing=101.0
        ),
        "d": _report_log(
            incumbents=[(10.0, 40.001)], status="Optimal", primal=40.001, timing=20.0
        ),
        "e": _report_log(
            incumbents=[(10.0, 60.0)],
            status="Time limit reached",
            primal=60.0,
            timing=100.02,
        ),
    }
    _write_seeded_tree(tmp_path, {"cfg": {0: logs}})
    return _write_z(tmp_path, {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0, "e": 50.0})


def test_summary_counts_opt_found_opt_proven_and_the_time_sgm(tmp_path: Path):
    z = _opt_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "cfg",
        "--summary",
        "--time-limit",
        "100",
        "--mipfeas-z",
        str(z),
    )
    assert res.returncode == 0, res.stderr
    block = res.stdout.split("## mipfeas Headline Metrics", 1)[1]
    assert "#OptFound" in block and "#OptProven" in block and "SGM(Time)" in block
    row = next(ln.split() for ln in block.splitlines() if ln.startswith("cfg "))
    _, _sgm_pi, feas, found, proven, sgm_t, n = row
    assert (feas, found, proven, n) == ("5", "3", "2", "5")
    times = [40.0, 100.0, 100.0, 20.0, 100.0]
    expected = math.exp(sum(math.log(t + 1.0) for t in times) / len(times)) - 1.0
    assert sgm_t == f"{expected:.1f}"
    # Scoring the killed run at the parser's 0 s would give a different number.
    at_zero = [40.0, 0.0, 100.0, 20.0, 100.0]
    wrong = math.exp(sum(math.log(t + 1.0) for t in at_zero) / 5) - 1.0
    assert sgm_t != f"{wrong:.1f}"


def test_csv_writes_one_row_per_run(tmp_path: Path):
    z = _opt_tree(tmp_path)
    out = tmp_path / "runs.csv"
    res = _run_cli(
        tmp_path,
        "--configs",
        "cfg",
        "--summary",
        "--time-limit",
        "100",
        "--mipfeas-z",
        str(z),
        "--csv",
        str(out),
    )
    assert res.returncode == 0, res.stderr
    with open(out, newline="") as f:
        rows = {r["instance"]: r for r in csv.DictReader(f)}
    assert sorted(rows) == ["a", "b", "c", "d", "e"]
    assert {r["config"] for r in rows.values()} == {"cfg"}
    assert {r["seed"] for r in rows.values()} == {"0"}
    flags = {k: (r["opt_found"], r["opt_proven"]) for k, r in rows.items()}
    assert flags == {
        "a": ("1", "1"),
        "b": ("1", "0"),
        "c": ("1", "0"),
        "d": ("0", "1"),
        "e": ("0", "0"),
    }
    assert {k: float(r["time"]) for k, r in rows.items()} == {
        "a": 40.0,
        "b": 100.0,
        "c": 100.0,
        "d": 20.0,
        "e": 100.0,
    }
    # A killed run carries its incumbent at the limit, not the parser's inf.
    assert float(rows["b"]["primal_bound"]) == 20.0
    assert rows["b"]["killed"] == "1" and rows["b"]["status"] == KILLED_STATUS
    assert float(rows["b"]["t_first"]) == 30.0
    assert float(rows["d"]["reference"]) == 40.0
    assert float(rows["a"]["primal_integral"]) == pytest.approx(0.1)  # 2 for 5 of 100 s
    assert rows["e"]["feasible"] == "1"


def test_csv_is_not_written_unless_asked(tmp_path: Path):
    z = _opt_tree(tmp_path)
    res = _run_cli(
        tmp_path,
        "--configs",
        "cfg",
        "--summary",
        "--time-limit",
        "100",
        "--mipfeas-z",
        str(z),
    )
    assert res.returncode == 0, res.stderr
    assert "Wrote" not in res.stderr
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["z.csv"]


def test_scored_time_gives_only_a_solved_run_its_own_time():
    from analyze_results import scored_time

    def run(status: str, t: float) -> SolveResult:
        return SolveResult(status=status, solve_time=t)

    assert scored_time(run("Optimal", 3.0), 100.0) == 3.0
    assert scored_time(run("Optimal", 140.0), 100.0) == 100.0
    # Stopped early without solving: not a fast solve.
    for status in ("Memory limit reached", "Solution limit reached", "Interrupted"):
        assert scored_time(run(status, 3.0), 100.0) == 100.0
    assert scored_time(run("Time limit reached", 100.02), 100.0) == 100.0
    assert scored_time(_killed([(1.0, 1.0)]), 100.0) == 100.0
    assert scored_time(SolveResult(), 100.0) == 100.0


def test_csv_leaves_a_non_finite_value_empty(tmp_path: Path):
    from analyze_results import write_runs_csv

    out = tmp_path / "runs.csv"
    results = {"cfg": {0: {"inst": SolveResult(status="Time limit reached")}}}
    assert write_runs_csv(str(out), results, ["cfg"], ["inst"], 100.0, {}) == 1
    with open(out, newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["primal_bound"] == "" and row["reference"] == ""
    assert row["t_first"] == "" and row["feasible"] == "0"


# --- t interval, median ratio and seed pairing (#179) ------------------------


def _from_d(d: list[float]) -> tuple[list[float], list[float]]:
    """Arms whose log ratios at shift 1 are exactly `d`: b = 0, a = e^d - 1."""
    return [math.exp(x) - 1.0 for x in d], [0.0] * len(d)


def test_t_quantile_and_tail_match_published_values():
    from analyze_results import t_ppf, t_sf

    for df, q975 in (
        (1, 12.706204736174707),
        (2, 4.302652729911275),
        (4, 2.7764451051977987),
        (9, 2.2621571627409915),
        (30, 2.0422724563012373),
    ):
        assert t_ppf(0.975, df) == pytest.approx(q975, rel=1e-9)
        assert t_sf(q975, df) == pytest.approx(0.025, rel=1e-8)
    assert t_ppf(0.025, 4) == pytest.approx(-2.7764451051977987, rel=1e-9)
    assert t_sf(0.0, 7) == pytest.approx(0.5)


def test_paired_constant_d_gives_a_degenerate_interval_at_exp_d():
    from analyze_results import paired

    a, b = _from_d([0.2] * 6)
    st = paired(a, b, shift=1.0)
    assert st.sd == pytest.approx(0.0, abs=1e-12)
    for v in (st.ratio, st.median_ratio, st.ci_lo, st.ci_hi):
        assert v == pytest.approx(math.exp(0.2))


def test_paired_mixed_sign_d_gives_the_hand_computed_t_interval():
    from analyze_results import paired

    d = [0.1, -0.2, 0.3, -0.1, 0.05]
    st = paired(*_from_d(d), shift=1.0)
    mean = sum(d) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 4)
    half = 2.7764451051977987 * sd / math.sqrt(5)  # t_{4, 0.975}
    assert st.ci_lo == pytest.approx(math.exp(mean - half))
    assert st.ci_hi == pytest.approx(math.exp(mean + half))
    assert st.ci_lo < 1.0 < st.ci_hi
    # Wider than the normal interval, and a larger p-value.
    assert st.ci_lo < st.normal_ci_lo and st.normal_ci_hi < st.ci_hi
    assert st.p > st.normal_p


def test_paired_median_ratio_of_an_odd_and_an_even_n():
    from analyze_results import paired

    odd = paired(*_from_d([0.5, -0.1, 0.0, 2.0, 0.1]), shift=1.0)
    assert odd.median_ratio == pytest.approx(math.exp(0.1))
    even = paired(*_from_d([0.5, -0.1, 2.0, 0.1]), shift=1.0)
    assert even.median_ratio == pytest.approx(math.exp((0.1 + 0.5) / 2))
    # One outlier moves the mean ratio, not the median.
    assert odd.ratio > odd.median_ratio


def test_seed_paired_values_take_the_sgm_over_the_common_seeds():
    from analyze_results import seed_paired_values, shifted_geomean

    def run(v: float) -> SolveResult:
        return SolveResult(primal_bound=v)

    results = {
        "x": {0: {"i": run(1.0), "j": run(4.0)}, 1: {"i": run(3.0), "j": run(2.0)}},
        "y": {0: {"i": run(2.0), "j": run(5.0)}, 1: {"i": run(2.0)}},
    }

    def score(r: SolveResult, inst: str) -> float | None:
        return r.primal_bound

    sp = seed_paired_values(results, "x", "y", ["i", "j"], score, 1.0)
    assert sp.instances == ["i", "j"]
    assert sp.a[0] == pytest.approx(shifted_geomean([1.0, 3.0], 1.0))
    assert sp.b[0] == pytest.approx(2.0)
    # j: y has no seed-1 run, so it pairs on seed 0 alone, and says so.
    assert sp.a[1] == pytest.approx(4.0) and sp.b[1] == pytest.approx(5.0)
    assert sp.differing == {"j": ([0, 1], [0])}
    assert sp.unpaired == []
    # With equal seed sets the paired ratio is the ratio of the SGMs over runs.
    eq = seed_paired_values(results, "x", "y", ["i"], score, 1.0)
    assert eq.differing == {}
    assert (eq.a[0] + 1.0) / (eq.b[0] + 1.0) == pytest.approx(
        (shifted_geomean([1.0, 3.0], 1.0) + 1.0)
        / (shifted_geomean([2.0, 2.0], 1.0) + 1.0)
    )
    # A score of None on either side drops the seed; none left drops the instance.
    gone = seed_paired_values(results, "x", "y", ["i"], lambda r, i: None, 1.0)
    assert gone.instances == [] and gone.unpaired == ["i"]


def test_cli_paired_seeds_pairs_every_seed_and_names_unequal_seed_sets(
    tmp_path: Path,
):
    fast = _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0)
    slow = _feasible_log(objective=10.0, found_at=3.0, solve_time=5.0)
    _write_seeded_tree(
        tmp_path,
        {
            "x": {0: {"a": fast, "b": fast}, 1: {"a": slow, "b": fast}},
            "y": {0: {"a": slow, "b": slow}, 1: {"a": slow}},
        },
    )
    z = _write_z(tmp_path, {"a": 10.0, "b": 10.0})
    res = _run_cli(
        tmp_path,
        "--configs",
        "x",
        "y",
        "--paired-seeds",
        "--time-limit",
        "5",
        "--mipfeas-z",
        str(z),
    )
    assert res.returncode == 0, res.stderr
    assert "formula=mipfeas, seed-paired)" in res.stdout
    assert "seed sets differ on 1 instance(s)" in res.stdout
    assert "b: x {0,1}, y {0}" in res.stdout
    line = next(ln for ln in res.stdout.splitlines() if "primal integral" in ln)
    assert "n=  2" in line and "median=" in line
    # mipfeas integral of one z* incumbent at t: 2t/5.  a: x seeds {0.4, 1.2}
    # against y {1.2, 1.2}; b: x seed 0 only, 0.4 against 1.2.
    s = 0.001
    da = (math.log(0.4 + s) + math.log(1.2 + s)) / 2 - math.log(1.2 + s)
    db = math.log(0.4 + s) - math.log(1.2 + s)
    ratio = math.exp((da + db) / 2)
    assert f"ratio={ratio:.3f}" in line


def test_cli_paired_normal_prints_the_line_format_paired_txt_was_written_in(
    tmp_path: Path,
):
    _plain_tree(tmp_path)
    z = _write_z(tmp_path, {"a": 10.0, "b": 20.0, "c": 30.0})
    common = ("--configs", "patched", "vanilla", "--time-limit", "5")
    normal = _run_cli(tmp_path, *common, "--paired-normal", "--mipfeas-z", str(z))
    assert normal.returncode == 0, normal.stderr
    assert "Intervals are\nnormal, exp(mean +- 1.96*se)" in normal.stdout
    line = next(ln for ln in normal.stdout.splitlines() if "primal integral" in ln)
    assert "median=" not in line and "ratio=1.000 CI=[" in line
    t = _run_cli(tmp_path, *common, "--paired", "--mipfeas-z", str(z))
    assert "intervals and p are t_{n-1}" in t.stdout


def test_cli_paired_says_so_when_too_few_instances_pair(tmp_path: Path):
    _plain_tree(tmp_path)
    keep = tmp_path / "keep.txt"
    keep.write_text("a\n")
    res = _run_cli(
        tmp_path,
        "--configs",
        "patched",
        "vanilla",
        "--paired",
        "--time-limit",
        "5",
        "--instances",
        str(keep),
    )
    assert res.returncode == 0, res.stderr
    assert "primal integral  n=1, not compared" in res.stdout


def test_cli_paired_seeds_caps_the_unequal_seed_notices(tmp_path: Path):
    log = _feasible_log(objective=10.0, found_at=1.0, solve_time=5.0)
    names = [f"i{k:02d}" for k in range(13)]
    _write_seeded_tree(
        tmp_path,
        {
            "x": {0: dict.fromkeys(names, log), 1: dict.fromkeys(names, log)},
            "y": {0: dict.fromkeys(names, log)},
        },
    )
    res = _run_cli(
        tmp_path, "--configs", "x", "y", "--paired-seeds", "--time-limit", "5"
    )
    assert res.returncode == 0, res.stderr
    assert "seed sets differ on 13 instance(s)" in res.stdout
    assert res.stdout.count(": x {0,1}, y {0}") == 10
    assert "... and 3 more" in res.stdout
