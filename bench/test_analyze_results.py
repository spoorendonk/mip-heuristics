"""Unit tests for analyze_results helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
from pathlib import Path

from analyze_results import (
    count_first,
    heuristic_attribution,
    latex_ablation_table,
    load_results,
    source_label,
)
from parse_highs_log import Incumbent, SolveResult


def _result(t_first: float | None) -> SolveResult:
    r = SolveResult()
    if t_first is not None:
        r.incumbents.append(
            Incumbent(time=t_first, objective=1.0, source="H", nodes=0)
        )
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
            "i2": _result_sources(["A"]),            # first & best FPR
            "i3": SolveResult(),                     # infeasible — skipped
            "i4": _result_sources(["B", "B"]),       # HiGHS/other first & best
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
        "all_opp": {"feasible": 213.0, "sgm_t1st": 3.3, "sgm_gap": 0.02,
                    "sgm_pi": 22.0, "plato_sgm": 17.3},
        "loo_no_fj": {"feasible": 200.0, "sgm_t1st": 4.0, "sgm_gap": 0.03,
                      "sgm_pi": 25.0, "plato_sgm": 19.0},
    }
    tex = latex_ablation_table(["all_opp", "loo_no_fj"], metrics, 233, 100.0)
    assert r"\begin{tabular}{lrrrrr}" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"loo\_no\_fj" in tex           # underscore escaped for LaTeX
    assert "213" in tex                    # #Feas rendered as int
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
            "      Status      Time limit reached\n"
            "      Primal bound inf\n"
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
