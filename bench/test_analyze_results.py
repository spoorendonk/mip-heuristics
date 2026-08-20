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
    CANNIBALIZATION_CATEGORIES,
    USABLE_REFERENCE_TAGS,
    _config_metrics,
    aggregate_results,
    build_best_known,
    build_oracle_config,
    classify_cannibalization,
    classify_reference_status,
    contradicted_reference_instances,
    count_first,
    count_wins,
    filter_results,
    get_common_instances,
    heuristic_attribution,
    heuristic_wall_seconds,
    is_instrumented,
    latex_ablation_table,
    load_results,
    native_call_total,
    native_lp_share,
    parse_solu_file,
    pick_baseline_config,
    presolve_span_seconds,
    print_cannibalization_tables,
    read_instance_list,
    source_label,
)
from parse_highs_log import (
    HeuristicSample,
    Incumbent,
    NativeCounters,
    RootTiming,
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


# ── budget-sweep directory names (`<config>@e<effort>`) ───────────────────────


def _tiny_tree(root: Path, config: str) -> None:
    seed_dir = root / config / "seed0"
    seed_dir.mkdir(parents=True)
    (seed_dir / "toy.log").write_text(
        "      Status      Time limit reached\n      Primal bound inf\n"
    )


def test_load_results_reads_budget_sweep_directories(tmp_path: Path):
    """run_benchmark's --budget-sweep names directories `<config>@e<V>`.

    The whole point of that naming is that sweep output is analysable with no
    new analysis code, so the default `results_dir/<config>` path has to take
    the `@` verbatim.
    """
    configs = ["fpr@e0.05", "fpr@e0.30", "vanilla"]
    for config in configs:
        _tiny_tree(tmp_path, config)
    loaded = load_results(str(tmp_path), configs)
    assert sorted(loaded) == sorted(configs)
    for config in configs:
        assert "toy" in loaded[config][0]


def test_latex_ablation_table_renders_budget_sweep_config_names():
    """`@` is an ordinary character in LaTeX text mode; `_` still is not."""
    metrics = {
        "local_mip@e0.30": {
            "feasible": 5.0,
            "sgm_t1st": 1.0,
            "sgm_gap": 0.01,
            "sgm_pi": 2.0,
            "plato_sgm": 1.5,
        },
    }
    tex = latex_ablation_table(["local_mip@e0.30"], metrics, 5, 60.0)
    assert r"local\_mip@e0.30" in tex


# ── cannibalization tables (issue #100, records from #95) ─────────────────────


def _heur(
    name: str, phase: str, start: float, wall_ms: float, found: bool = False
) -> HeuristicSample:
    """One `[Heur]` sample; effort and rate are irrelevant to these tables."""
    return HeuristicSample(
        name=name,
        phase=phase,
        start_s=start,
        end_s=start + wall_ms / 1000.0,
        effort=1000,
        wall_ms=wall_ms,
        effort_per_ms=1000.0 / max(wall_ms, 1e-9),
        found=found,
    )


def _instrumented(
    *,
    solve_time: float = 10.0,
    samples: list[HeuristicSample] | None = None,
    rens: int = 2,
    rens_root: int = 1,
    rins: int = 3,
    rcfix: int = 1,
    heur_lp: int = 1000,
    tot_lp: int = 10000,
    ours: int = 0,
    lp_time: float = 1.0,
    span: float = 0.0,
) -> SolveResult:
    """A SolveResult carrying all three issue-#95 records."""
    r = SolveResult(status="Optimal", solve_time=solve_time)
    r.heuristic_samples = list(samples or [])
    r.native = NativeCounters(
        rens=rens,
        rens_root=rens_root,
        rins=rins,
        rcfix=rcfix,
        heur_lp_iters=heur_lp,
        total_lp_iters=tot_lp,
        fpr_lp_lp_iters=ours,
    )
    r.root = RootTiming(lp_time_s=lp_time, presolve_heur_s=span)
    return r


def _uninstrumented(solve_time: float = 10.0) -> SolveResult:
    """A pre-#95 log: solved fine, carries none of the counters."""
    return SolveResult(status="Optimal", solve_time=solve_time)


def test_instrumented_baseline_is_a_real_zero_not_a_missing_value():
    """A `suite=off` row dispatched nothing; that is 0.0, not unknown.

    Only a log predating the instrumentation is unknown.  The distinction is
    what keeps the baseline row — the reference for every other row — out of
    any filter that drops None.
    """
    base = _instrumented()
    assert is_instrumented(base)
    assert base.heuristic_samples == []
    assert heuristic_wall_seconds(base) == 0.0
    assert base.heuristic_wall_fraction == 0.0

    old = _uninstrumented()
    assert not is_instrumented(old)
    assert heuristic_wall_seconds(old) is None
    assert old.heuristic_wall_fraction is None


def test_heuristic_wall_seconds_splits_presolve_from_dive():
    r = _instrumented(
        samples=[
            _heur("fj", "presolve", 0.1, 400.0),
            _heur("fpr", "presolve", 0.5, 600.0),
            _heur("fpr_lp", "dive", 3.0, 250.0),
        ]
    )
    assert heuristic_wall_seconds(r) == 1.25
    assert heuristic_wall_seconds(r, "presolve") == 1.0
    assert heuristic_wall_seconds(r, "dive") == 0.25


def test_native_call_total_sums_only_upstream_call_sites():
    assert native_call_total(_instrumented(rens=2, rins=3, rcfix=1)) == 6
    assert native_call_total(_uninstrumented()) is None


def test_root_site_rens_suppression_alone_is_internal_budget():
    """The merged rens total can hold steady while the root call vanishes.

    Both rows call RENS twice; the patched one never calls it at the root,
    which is the gate a presolve-found incumbent closes.  Collapsing the two
    counters would classify this instance as neutral.
    """
    base = _instrumented(rens=2, rens_root=1)
    row = _instrumented(rens=2, rens_root=0)
    v = classify_cannibalization(row, base)
    assert v.category == "internal-budget"
    assert v.internal and not v.wall
    assert any("rens_root 1->0" in e for e in v.evidence)


def test_wall_clock_category_when_budgets_are_preserved():
    """Same native activity, but a third of the solve went into heuristics."""
    base = _instrumented(solve_time=10.0)
    row = _instrumented(solve_time=10.0, samples=[_heur("fj", "presolve", 0.1, 3300.0)])
    v = classify_cannibalization(row, base)
    assert v.category == "wall-clock"
    assert v.wall and not v.internal


def test_both_categories_when_budget_and_time_are_taken():
    base = _instrumented(solve_time=10.0, rens=3, rens_root=1, rins=4)
    row = _instrumented(
        solve_time=10.0,
        rens=1,
        rens_root=0,
        rins=1,
        samples=[_heur("scylla", "presolve", 0.1, 4000.0)],
    )
    v = classify_cannibalization(row, base)
    assert v.category == "both"
    assert v.internal and v.wall


def test_neutral_when_nothing_moved():
    base = _instrumented(solve_time=10.0)
    row = _instrumented(solve_time=10.0, samples=[_heur("fj", "presolve", 0.1, 50.0)])
    v = classify_cannibalization(row, base)
    assert v.category == "neutral"
    assert v.evidence == ()


def test_merged_call_total_needs_a_margin_but_root_rens_does_not():
    """The merged total is a rate; rens_root is a gate.

    A one-call dip out of forty is thread-interleaving noise on counters the
    B&B dive increments concurrently, and reporting it as budget taken would
    inflate the epic's headline count.  A vanished root call is never noise.
    """
    base = _instrumented(rens=20, rens_root=1, rins=20, rcfix=0)
    drift = _instrumented(rens=20, rens_root=1, rins=19, rcfix=0)
    assert classify_cannibalization(drift, base).category == "neutral"

    real = _instrumented(rens=15, rens_root=1, rins=15, rcfix=0)
    assert classify_cannibalization(real, base).category == "internal-budget"

    gate = _instrumented(rens=20, rens_root=0, rins=20, rcfix=0)
    assert classify_cannibalization(gate, base).category == "internal-budget"


def test_thresholds_are_overridable_for_a_sensitivity_check():
    """The CLI does not expose them; the keyword arguments are the way in."""
    base = _instrumented(solve_time=10.0)
    row = _instrumented(solve_time=10.0, samples=[_heur("fj", "presolve", 0.1, 300.0)])
    assert classify_cannibalization(row, base).category == "neutral"
    assert (
        classify_cannibalization(row, base, wall_fraction=0.01).category == "wall-clock"
    )


def test_heuristic_wall_time_is_reported_from_heur_lines_alone():
    """`[Native]` comes from cleanupSolve; a truncated log can lack it.

    Keying the wall-time helper on `[Native]` alone printed Heur_s='-' next to
    a populated HeurFrac on the same row — self-contradictory.
    """
    r = SolveResult(status="Time limit", solve_time=10.0)
    r.heuristic_samples = [_heur("fj", "presolve", 0.1, 3000.0)]
    assert not is_instrumented(r)
    assert heuristic_wall_seconds(r) == 3.0
    assert r.heuristic_wall_fraction == 0.3


def test_native_lp_share_is_the_gate_quantity():
    """NatShare is what upstream's moreHeuristicsAllowed() tests."""
    r = _instrumented(heur_lp=1500, tot_lp=11000, ours=500)
    assert native_lp_share(r) == 1000 / 10500
    assert native_lp_share(_instrumented(heur_lp=0, tot_lp=0)) is None
    assert native_lp_share(_uninstrumented()) is None


def test_more_native_activity_is_not_cannibalization():
    """Only decreases count; a config that lets HiGHS do more is not a cost."""
    base = _instrumented(rens=1, rens_root=0, rins=1, heur_lp=100)
    row = _instrumented(rens=5, rens_root=2, rins=6, heur_lp=9000)
    assert classify_cannibalization(row, base).category == "neutral"


def test_native_lp_iterations_are_compared_with_our_charge_removed():
    """Reading the shared counters raw bills our dive work as upstream's.

    Both rows do the same native LP work (1000 iterations); the patched row's
    raw `heur_lp_iters` is inflated by fpr_lp's own charge.  Subtracting it —
    which is what `native_heur_lp_iters` does — leaves no signal.
    """
    base = _instrumented(heur_lp=1000, tot_lp=10000, ours=0)
    row = _instrumented(heur_lp=6000, tot_lp=15000, ours=5000)
    assert row.native.native_heur_lp_iters == 1000
    assert classify_cannibalization(row, base).category == "neutral"

    # A genuine drop in native LP work does register.
    starved = _instrumented(heur_lp=5100, tot_lp=14100, ours=5000)
    v = classify_cannibalization(starved, base)
    assert v.category == "internal-budget"
    assert any(
        "native heur LP share" in e and "(1000->100 iters)" in e for e in v.evidence
    )


def test_faster_solve_that_raises_the_native_share_is_not_internal_budget():
    """The limb tests a share, because upstream's gate tests a ratio.

    `moreHeuristicsAllowed` compares `heuristic_lp_iterations` against
    `total_lp_iterations * effort`, so what matters is how close a config
    drove HiGHS to its own gate — not how many iterations it did.  A config
    that simply solves faster cuts native heuristic LP iterations *and* the
    total together; if the total falls further the share rises, HiGHS's
    heuristics were relatively more active, and nothing was starved.  An
    absolute test labels that `internal-budget` and inflates the headline.
    """
    base = _instrumented(heur_lp=1000, tot_lp=10000, ours=0)  # share 0.100
    faster = _instrumented(heur_lp=800, tot_lp=5000, ours=0)  # share 0.160
    assert faster.native.native_heur_lp_iters < base.native.native_heur_lp_iters
    v = classify_cannibalization(faster, base)
    assert not any("native heur LP" in e for e in v.evidence)


def test_root_lp_delay_needs_both_a_relative_and_an_absolute_margin():
    """10% of a 0.02 s root LP is noise; 0.05 s on a 300 s one is too."""
    tiny_base = _instrumented(lp_time=0.02)
    tiny_row = _instrumented(lp_time=0.03)  # +50% but only +0.01 s
    assert classify_cannibalization(tiny_row, tiny_base).category == "neutral"

    big_base = _instrumented(lp_time=300.0)
    big_row = _instrumented(lp_time=300.2)  # +0.2 s but only +0.07%
    assert classify_cannibalization(big_row, big_base).category == "neutral"

    real_base = _instrumented(lp_time=2.0)
    real_row = _instrumented(lp_time=5.0)
    v = classify_cannibalization(real_row, real_base)
    assert v.category == "wall-clock"
    assert any("root LP +3.00s" in e for e in v.evidence)


def test_root_lp_never_reached_yields_no_delay_signal():
    """`lp_time_s=-1` is the sentinel; the parser reports None, not t=0."""
    base = _instrumented(lp_time=-1.0)
    row = _instrumented(lp_time=-1.0)
    assert base.time_to_root_lp is None
    assert classify_cannibalization(row, base).category == "neutral"


def test_restarting_instance_span_may_exceed_root_lp_timestamp():
    """The chain span accumulates over restarts while the timestamp pins the
    first root LP.  Expected, so it must not be corrected or flagged."""
    row = _instrumented(lp_time=1.0, span=4.5, solve_time=20.0)
    assert presolve_span_seconds(row) == 4.5
    assert row.time_to_root_lp == 1.0
    v = classify_cannibalization(row, _instrumented(lp_time=1.0, span=0.0))
    assert v.category == "neutral"


def test_uninstrumented_row_and_uninstrumented_baseline_are_distinguished():
    inst = _instrumented()
    assert (
        classify_cannibalization(_uninstrumented(), inst).category == "not-instrumented"
    )
    assert classify_cannibalization(inst, _uninstrumented()).category == "no-baseline"
    assert classify_cannibalization(inst, None).category == "no-baseline"
    assert classify_cannibalization(inst, inst, is_baseline=True).category == "baseline"
    # An uninstrumented baseline row is labelled by what it carries, not by
    # its role: it cannot serve as a reference either.
    old = _uninstrumented()
    assert (
        classify_cannibalization(old, old, is_baseline=True).category
        == "not-instrumented"
    )


def test_pick_baseline_prefers_the_instrumented_zero_heuristic_config():
    agg = {
        "patched": {"a": _instrumented(samples=[_heur("fj", "presolve", 0.1, 5.0)])},
        # Name deliberately absent from CANNIBALIZATION_BASELINE_NAMES so the
        # structural test (instrumented, no [Heur] lines) has to find it on its
        # own rather than the name fallback answering by accident.
        "zeroheur": {"a": _instrumented()},
    }
    assert pick_baseline_config(agg, ["patched", "zeroheur"]) == "zeroheur"


def test_pick_baseline_refuses_a_heuristic_running_config_with_a_baseline_name():
    """A config that dispatched heuristics is a subject, never a reference.

    Nothing downstream would catch it — the uninstrumented-baseline warning
    does not fire on an instrumented row — so every other row would be
    silently compared against a patched reference.  A config rename (#96)
    lands on exactly this path.
    """
    agg = {
        "patched": {"a": _instrumented(samples=[_heur("fj", "presolve", 0.1, 6.0)])},
        "vanilla": {"a": _instrumented(samples=[_heur("fj", "presolve", 0.1, 5.0)])},
    }
    assert pick_baseline_config(agg, ["patched", "vanilla"]) is None


def test_uninstrumented_rows_are_not_evidence_of_dispatching_nothing():
    """A pre-#95 row has no [Heur] lines because it has no lines at all."""
    agg = {
        "mixed": {
            "a": _instrumented(samples=[_heur("fj", "presolve", 0.1, 5.0)]),
            "b": _uninstrumented(),
        },
    }
    assert pick_baseline_config(agg, ["mixed"]) is None


def test_pick_baseline_falls_back_to_a_known_name_when_uninstrumented():
    """An externally built unpatched `vanilla` emits no instrumentation.

    It is still the right row to name, so the report can say the comparison
    needs a patched suite=off run instead of silently promoting a patched
    config to its own reference.
    """
    agg = {
        "patched": {"a": _instrumented(samples=[_heur("fj", "presolve", 0.1, 5.0)])},
        "vanilla": {"a": _uninstrumented()},
    }
    assert pick_baseline_config(agg, ["patched", "vanilla"]) == "vanilla"


def test_pick_baseline_honours_an_explicit_name_and_rejects_an_unknown_one():
    agg = {"a_cfg": {"i": _instrumented()}, "b_cfg": {"i": _instrumented()}}
    assert pick_baseline_config(agg, ["a_cfg", "b_cfg"], "b_cfg") == "b_cfg"
    assert pick_baseline_config(agg, ["a_cfg", "b_cfg"], "nope") is None


def test_pick_baseline_returns_none_when_nothing_qualifies():
    agg = {"cfg_x": {"i": _instrumented(samples=[_heur("fj", "presolve", 0.1, 5.0)])}}
    assert pick_baseline_config(agg, ["cfg_x"]) is None


# ── end-to-end from synthetic logs ────────────────────────────────────────────


def _synth_log(
    *,
    solve_time: float,
    samples: list[HeuristicSample] = (),
    native: tuple[int, int, int, int, int, int, int] | None = None,
    root: tuple[float, float] | None = None,
) -> str:
    """Render a HiGHS log carrying the issue-#95 lines verbatim.

    Field order and spelling match `heuristics::log_solve_summary` and
    `EffortLedger::book`; the parser's regexes are the contract under test.
    """
    lines = []
    for s in samples:
        lines.append(
            f"[Heur] name={s.name} phase={s.phase} start_s={s.start_s:.3f} "
            f"end_s={s.end_s:.3f} effort={s.effort} wall_ms={s.wall_ms:.1f} "
            f"effort_per_ms={s.effort_per_ms:.1f} found={int(s.found)}"
        )
    if native is not None:
        rens, rens_root, rins, rcfix, heur_lp, tot_lp, ours = native
        lines.append(
            f"[Native] rens={rens} rens_root={rens_root} rins={rins} "
            f"rcfix={rcfix} heur_lp_iters={heur_lp} total_lp_iters={tot_lp} "
            f"fpr_lp_lp_iters={ours}"
        )
    if root is not None:
        lines.append(f"[Root] lp_time_s={root[0]:.3f} presolve_heur_s={root[1]:.3f}")
    lines += [
        "      Status            Optimal",
        "      Nodes             17",
        f"      Timing            {solve_time:.2f}",
    ]
    return "\n".join(lines) + "\n"


def _write_tree(root_dir: Path, tree: dict[str, dict[str, str]]) -> None:
    """{config: {instance: log_text}} -> <root>/<config>/seed0/<instance>.log."""
    for config, logs in tree.items():
        seed_dir = root_dir / config / "seed0"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for inst, text in logs.items():
            (seed_dir / f"{inst}.log").write_text(text)


def _render(
    tmp_path: Path, tree: dict[str, dict[str, str]], capsys, baseline: str | None = None
) -> str:
    """Write a results tree, run the report over it, return the output."""
    _write_tree(tmp_path, tree)
    configs = list(tree)
    results = load_results(str(tmp_path), configs)
    print_cannibalization_tables(
        results, aggregate_results(results, configs), configs, baseline
    )
    return capsys.readouterr().out


def _block(out: str, start: str, end: str | None = None) -> str:
    """The slice of the report between two headings."""
    part = out.split(start, 1)[1]
    return part.split(end, 1)[0] if end else part


def _cells(block: str, instance: str, config: str) -> list[str]:
    """Whitespace-split cells of one instance x config row.

    Asserting by column index rather than by substring is what pins a column
    to its own value: a grep over the whole line passes when two columns are
    wired to the same source.
    """
    for line in block.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0] == instance and parts[1] == config:
            return parts
    raise AssertionError(f"no row for {instance}/{config} in:\n{block}")


def test_cannibalization_tables_render_from_an_instrumented_tree(tmp_path, capsys):
    """Both tables, their aggregates and the classification, no scripting."""
    starved = _synth_log(
        solve_time=10.0,
        samples=[
            _heur("fj", "presolve", 0.1, 900.0, found=True),
            _heur("scylla", "presolve", 1.0, 3000.0),
            _heur("fpr_lp", "dive", 6.0, 500.0),
        ],
        # 1300 shared heuristic LP iterations of which 900 are fpr_lp's own
        # charge, so HiGHS itself did 400 against the baseline's 2000.
        native=(1, 0, 1, 1, 1300, 12000, 900),
        root=(4.5, 3.9),
    )
    quiet = _synth_log(
        solve_time=10.0,
        samples=[_heur("fj", "presolve", 0.1, 40.0)],
        native=(3, 1, 4, 1, 2000, 11000, 0),
        # Span deliberately unlike every other numeric cell in the row, so the
        # assertion on it can only be satisfied by the Span_s column.
        root=(1.05, 0.31),
    )
    base_log = _synth_log(
        solve_time=10.0, native=(3, 1, 4, 1, 2000, 11000, 0), root=(1.0, 0.0)
    )
    out = _render(
        tmp_path,
        {
            "patched": {"starved": starved, "quiet": quiet},
            "vanilla": {"starved": base_log, "quiet": base_log},
        },
        capsys,
    )

    assert "## Cannibalization" in out
    assert "Baseline config: vanilla" in out
    assert "### Internal budget" in out
    assert "### Wall clock" in out
    assert "### Classification counts" in out

    internal = _block(out, "### Internal budget", "#### Aggregate")
    wall = _block(_block(out, "### Wall clock"), "Class:", "#### Aggregate")

    # Internal budget: counters, our dive charge subtracted from both shared
    # LP counters (1300-900, 12000-900), the native share of those two, and
    # the delta against the baseline's own native figure (400-2000).
    row = _cells(internal, "starved", "patched")
    assert row[2:8] == ["1", "0", "1", "1", "400", "11100"]
    assert row[8] == f"{400 / 11100:.4f}"
    assert row[9] == "900"
    assert row[10] == "-1600"
    assert _cells(internal, "starved", "vanilla")[10] == "-"  # no delta on itself

    # Wall clock: Heur_s, Dive_s, HeurFrac, Troot_s, dTroot_s, Span_s, Class.
    row = _cells(wall, "starved", "patched")
    assert row[2] == "4.40" and row[3] == "0.50"  # 4.4 s total, 0.5 s dive
    assert row[4] == "0.4400"
    assert row[5] == "4.50" and row[6] == "3.50"
    assert row[7] == "3.90"
    assert row[8] == "both"
    evidence = " ".join(row[9:])
    assert "rens_root 1->0" in evidence
    assert "native heur LP share" in evidence
    assert "(2000->400 iters)" in evidence
    assert "root LP +3.50s" in evidence

    quiet_row = _cells(wall, "quiet", "patched")
    assert quiet_row[2] == "0.04"  # Heur_s
    assert quiet_row[7] == "0.31"  # Span_s, distinct from every other cell
    assert quiet_row[8] == "neutral"

    # The baseline config is a row in its own right, with real zeros.
    base_row = _cells(wall, "starved", "vanilla")
    assert base_row[2] == "0" and base_row[3] == "0" and base_row[4] == "0"
    assert base_row[6] == "-"  # no delta against itself
    assert base_row[8] == "baseline"


def test_classification_counts_account_for_every_instance(tmp_path, capsys):
    """No instance may silently fall out of the classification."""
    inst = _synth_log(
        solve_time=6.0, native=(2, 1, 2, 0, 500, 5000, 0), root=(0.5, 0.0)
    )
    old = "      Status            Optimal\n      Timing            6.00\n"
    out = _render(
        tmp_path,
        {
            "patched": {"i1": inst, "i2": inst, "i3": old},
            "vanilla": {"i1": inst, "i2": inst, "i3": inst},
        },
        capsys,
    )

    counts = _block(out, "### Classification counts")
    for cfg in ("patched", "vanilla"):
        row = next(ln for ln in counts.splitlines() if ln.startswith(cfg))
        assert sum(int(v) for v in row.split()[1:]) == 3
    header = next(ln for ln in counts.splitlines() if ln.startswith("Config"))
    assert header.split()[1:] == list(CANNIBALIZATION_CATEGORIES)


def test_root_rens_lost_counts_suppression_and_abstains_without_a_baseline(
    tmp_path, capsys
):
    """The root-gate headline must never print 0 for want of a comparison.

    A hard zero in that column reads as "no root RENS was suppressed", which
    is the opposite of "nothing was checked" — every other baseline-relative
    cell renders '-' in that state.
    """
    base_log = _synth_log(
        solve_time=6.0, native=(2, 1, 2, 0, 500, 5000, 0), root=(0.5, 0.0)
    )
    suppressed = _synth_log(
        solve_time=6.0,
        samples=[_heur("fpr", "presolve", 0.1, 100.0, found=True)],
        native=(2, 0, 2, 0, 500, 5000, 0),
        root=(0.52, 0.11),
    )
    out = _render(
        tmp_path, {"patched": {"i1": suppressed}, "vanilla": {"i1": base_log}}, capsys
    )
    agg = _block(out, "### Internal budget", "### Wall clock").split("#### Aggregate")[
        1
    ]
    assert (
        next(ln for ln in agg.splitlines() if ln.startswith("patched")).split()[-1]
        == "1"
    )
    assert (
        next(ln for ln in agg.splitlines() if ln.startswith("vanilla")).split()[-1]
        == "-"
    )

    # Same config with no baseline in the tree: nothing was compared, so the
    # column abstains rather than reporting zero suppressions.
    out = _render(tmp_path / "solo", {"patched": {"i1": suppressed}}, capsys)
    agg = _block(out, "### Internal budget", "### Wall clock").split("#### Aggregate")[
        1
    ]
    assert (
        next(ln for ln in agg.splitlines() if ln.startswith("patched")).split()[-1]
        == "-"
    )


def test_root_lp_column_reports_how_many_instances_stand_behind_it(tmp_path, capsys):
    """An instance whose root LP was never reached leaves that config's SGM.

    The bias runs the wrong way for the epic's claim — the most delayed
    instances are the ones that vanish — so the count must be visible.
    """
    reached = _synth_log(
        solve_time=6.0, native=(2, 1, 2, 0, 500, 5000, 0), root=(1.0, 0.0)
    )
    never = _synth_log(
        solve_time=6.0,
        samples=[_heur("fj", "presolve", 0.1, 50.0)],
        native=(2, 1, 2, 0, 500, 5000, 0),
        root=(-1.0, 0.05),
    )
    out = _render(
        tmp_path,
        {
            "patched": {"i1": reached, "i2": never},
            "vanilla": {"i1": reached, "i2": reached},
        },
        capsys,
    )
    wall_agg = _block(_block(out, "### Wall clock"), "#### Aggregate", "#Root =")
    patched = next(ln for ln in wall_agg.splitlines() if ln.startswith("patched"))
    vanilla = next(ln for ln in wall_agg.splitlines() if ln.startswith("vanilla"))
    assert patched.split()[1] == "2" and patched.split()[6] == "1"  # #Instr, #Root
    assert vanilla.split()[1] == "2" and vanilla.split()[6] == "2"
    # The unreached row is visible as such, not as t=0.
    assert (
        _cells(
            _block(_block(out, "### Wall clock"), "Class:", "#### Aggregate"),
            "i2",
            "patched",
        )[5]
        == "-"
    )


def test_negative_heuristic_window_is_shown_and_kept_out_of_the_aggregate(
    tmp_path, capsys
):
    """HiGHS's solver clock is not monotonic, so a window can come out negative.

    `shifted_geomean` clamps anything <= -1 s and would drag the whole SGM to
    ~-1 with no indication; the per-instance row still shows the artefact,
    which is the reason both bench regexes accept the sign.
    """
    # start_s is an absolute solver-clock reading, so it stays positive even
    # when the window it ends is negative — the parser's end_s pattern is
    # unsigned for exactly that reason.
    artefact = _synth_log(
        solve_time=6.0,
        samples=[_heur("fj", "presolve", 5.0, -3000.0)],
        native=(2, 1, 2, 0, 500, 5000, 0),
        root=(0.5, 0.0),
    )
    normal = _synth_log(
        solve_time=6.0,
        samples=[_heur("fj", "presolve", 0.1, 1000.0)],
        native=(2, 1, 2, 0, 500, 5000, 0),
        root=(0.5, 0.0),
    )
    out = _render(tmp_path, {"patched": {"i1": artefact, "i2": normal}}, capsys)
    rows = _block(_block(out, "### Wall clock"), "Class:", "#### Aggregate")
    assert _cells(rows, "i1", "patched")[2] == "-3.00"
    assert "negative [Heur] window(s) excluded" in out
    wall_agg = _block(_block(out, "### Wall clock"), "#### Aggregate", "#Root =")
    patched = next(ln for ln in wall_agg.splitlines() if ln.startswith("patched"))
    assert patched.split()[2] == "1.00"  # SGM over the surviving sample only


def test_mixed_tree_reports_per_row_rather_than_dropping_instances(tmp_path, capsys):
    """A tree extended after #95 landed carries both kinds of log."""
    inst = _synth_log(
        solve_time=6.0, native=(2, 1, 2, 0, 500, 5000, 0), root=(0.5, 0.0)
    )
    old = "      Status            Optimal\n      Timing            6.00\n"
    out = _render(
        tmp_path,
        {"patched": {"new": inst, "old": old}, "vanilla": {"new": inst, "old": old}},
        capsys,
    )
    internal = _block(out, "### Internal budget", "#### Aggregate")
    assert _cells(internal, "old", "patched")[2:] == ["-"] * 9
    assert _cells(internal, "new", "patched")[2] == "2"
    agg = _block(out, "### Internal budget", "### Wall clock").split("#### Aggregate")[
        1
    ]
    assert (
        next(ln for ln in agg.splitlines() if ln.startswith("patched")).split()[1]
        == "1"
    )


def test_cli_renders_the_tables_end_to_end(tmp_path):
    """Issue #100 asks for this with no ad-hoc scripting — drive the CLI.

    Uses `sys.executable` rather than a hard-coded interpreter path so it runs
    under whatever is driving pytest, matching test_check_effort_drift.py.
    """
    inst = _synth_log(
        solve_time=6.0,
        samples=[_heur("fj", "presolve", 0.1, 3000.0)],
        native=(2, 1, 2, 0, 500, 5000, 0),
        root=(0.9, 0.6),
    )
    base = _synth_log(
        solve_time=6.0, native=(2, 1, 2, 0, 500, 5000, 0), root=(0.5, 0.0)
    )
    _write_tree(tmp_path, {"patched": {"i1": inst}, "vanilla": {"i1": base}})
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analyze_results.py"
    )
    res = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--configs",
            "patched",
            "vanilla",
            "--summary",
            "--cannibalization",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "## Cannibalization" in res.stdout
    assert "### Internal budget" in res.stdout
    assert "### Wall clock" in res.stdout
    assert "wall-clock" in res.stdout

    # ... and the same tables under --ablation, the other call site.
    res = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--configs",
            "patched",
            "vanilla",
            "--ablation",
            "--cannibalization",
            "--cannibalization-baseline",
            "vanilla",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "Baseline config: vanilla" in res.stdout


def test_cannibalization_baseline_row_survives_aggregation(tmp_path, capsys):
    """The baseline's 0.0 heuristic wall time must reach the aggregate row.

    It is the reference every other row is read against; an aggregation that
    filtered None would drop it and leave the table without its zero line.
    """
    base_log = _synth_log(
        solve_time=8.0, native=(2, 1, 2, 0, 500, 5000, 0), root=(0.5, 0.0)
    )
    patched_log = _synth_log(
        solve_time=8.0,
        samples=[_heur("local_mip", "presolve", 0.1, 200.0)],
        native=(2, 1, 2, 0, 500, 5000, 0),
        root=(0.52, 0.21),
    )
    _write_tree(tmp_path, {"vanilla": {"i1": base_log}, "patched": {"i1": patched_log}})
    configs = ["patched", "vanilla"]
    results = load_results(str(tmp_path), configs)
    print_cannibalization_tables(results, aggregate_results(results, configs), configs)
    out = capsys.readouterr().out

    agg_block = out.split("#### Aggregate (SGM shift=1")[1]
    vanilla_agg = next(ln for ln in agg_block.splitlines() if ln.startswith("vanilla"))
    # #Instr=1 and a genuine zero for the heuristic-time column.
    assert vanilla_agg.split()[1] == "1"
    assert vanilla_agg.split()[2] == "0"
    assert "baseline" in out


def test_cannibalization_degrades_on_a_tree_without_instrumentation(tmp_path, capsys):
    """A results tree recorded before issue #95 still analyses."""
    old = (
        "      Status            Optimal\n"
        "      Nodes             3\n"
        "      Timing            5.00\n"
    )
    _write_tree(tmp_path, {"patched": {"i1": old}, "vanilla": {"i1": old}})
    configs = ["patched", "vanilla"]
    results = load_results(str(tmp_path), configs)
    print_cannibalization_tables(results, aggregate_results(results, configs), configs)
    out = capsys.readouterr().out

    assert "not instrumented" in out
    # No table, and no fabricated zeros.
    assert "### Internal budget" not in out
    assert "0.0000" not in out


def test_cannibalization_warns_when_only_the_baseline_lacks_instrumentation(
    tmp_path, capsys
):
    """An externally built unpatched vanilla binary emits no counters at all."""
    patched_log = _synth_log(
        solve_time=6.0,
        samples=[_heur("fj", "presolve", 0.1, 100.0)],
        native=(2, 1, 2, 0, 500, 5000, 0),
        root=(0.6, 0.11),
    )
    _write_tree(
        tmp_path,
        {
            "patched": {"i1": patched_log},
            "vanilla": {
                "i1": "      Status            Optimal\n      Timing            6.00\n"
            },
        },
    )
    configs = ["patched", "vanilla"]
    results = load_results(str(tmp_path), configs)
    print_cannibalization_tables(results, aggregate_results(results, configs), configs)
    out = capsys.readouterr().out

    assert "carries no instrumentation" in out
    assert "no-baseline" in out
    # The patched row's own numbers are still reported.
    assert "### Internal budget" in out


def test_cannibalization_reports_an_unknown_requested_baseline(tmp_path, capsys):
    log = _synth_log(solve_time=6.0, native=(1, 1, 1, 0, 100, 1000, 0), root=(0.3, 0.0))
    _write_tree(tmp_path, {"patched": {"i1": log}})
    results = load_results(str(tmp_path), ["patched"])
    print_cannibalization_tables(
        results, aggregate_results(results, ["patched"]), ["patched"], "nosuch"
    )
    out = capsys.readouterr().out
    assert "is not in this results tree" in out
    assert "no-baseline" in out


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
