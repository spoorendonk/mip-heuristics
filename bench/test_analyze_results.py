"""Unit tests for analyze_results helpers."""

from __future__ import annotations

import os
import sys

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
    Incumbent,
    SolveResult,
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
            "plato_sgm": 17.3,
        },
        "loo_no_fj": {
            "feasible": 200.0,
            "sgm_t1st": 4.0,
            "sgm_gap": 0.03,
            "sgm_pi": 25.0,
            "plato_sgm": 19.0,
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
    # ablation anchors from bench/results/plato).  Build a tiny seed0 tree.
    real_log = Path(__file__).with_name("results") / "plato" / "patched" / "seed0"
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
            "plato_sgm": 1.5,
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

    `--instances plato --exclude-instances tuning` is the held-out set, and it
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
    virtual best (or to PLATO's gap=1.0 convention) and is sound, whereas a
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


def test_bundled_plato_set_has_a_usable_reference_for_every_instance():
    """Regression for the supportcase22 anomaly (issue #104).

    The bundled solution file and the PLATO list are two halves of one claim —
    233 instances with a reference objective each.  When they disagree the SGM
    silently changes meaning, so the agreement is asserted rather than assumed.
    """
    bench = os.path.dirname(os.path.abspath(__file__))
    instances = read_instance_list(os.path.join(bench, "instances_plato.txt"))
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
        "## PLATO Headline Metrics (3 instances",
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
    plato_sgm = {
        ln.split()[0]: float(ln.split()[-1])
        for ln in ablation.splitlines()
        if ln.split() and ln.split()[0] in ("fpr", "scylla", "oracle")
    }
    assert set(plato_sgm) == {"fpr", "scylla", "oracle"}
    assert plato_sgm["oracle"] < plato_sgm["fpr"]
    assert plato_sgm["oracle"] < plato_sgm["scylla"]


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
    assert "## PLATO Headline Metrics (2 instances" in res.stdout


# --- the ceiling invariant, which is what the whole feature promises --------


def _plato_sgm_per_config(results, configs, oracle_name=None, time_limit=5.0):
    """PLATO headline SGM per config, through the real reporting pipeline."""
    common = get_common_instances(results, configs)
    best_known = build_best_known(results, configs, common, {})
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
    agg = aggregate_results(results, all_configs)
    return {
        c: _config_metrics(results, agg, c, common, time_limit, best_known)["plato_sgm"]
        for c in all_configs
    }


def test_oracle_is_a_ceiling_on_the_headline_metric_with_multiple_seeds(tmp_path: Path):
    """`oracle <= min(participants)` on PLATO SGM. Two seeds, the failing shape.

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
    sgm = _plato_sgm_per_config(results, ["A", "B"], oracle_name="oracle")

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
        sgm = _plato_sgm_per_config(results, ["A", "B", "C"], oracle_name="oracle")
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
    agg = aggregate_results(results, ["fpr", "scylla"])
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
    best_known = build_best_known(results, ["fpr", "scylla", "combined"], common, {})

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
    agg = aggregate_results(results, configs)

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
