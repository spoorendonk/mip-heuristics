"""Unit tests for analyze_results helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
from pathlib import Path

from analyze_results import (
    aggregate_results,
    classify_cannibalization,
    count_first,
    heuristic_attribution,
    heuristic_wall_seconds,
    is_instrumented,
    latex_ablation_table,
    load_results,
    native_call_total,
    pick_baseline_config,
    presolve_span_seconds,
    print_cannibalization_tables,
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
        "local_mip@e0.30": {"feasible": 5.0, "sgm_t1st": 1.0, "sgm_gap": 0.01,
                            "sgm_pi": 2.0, "plato_sgm": 1.5},
    }
    tex = latex_ablation_table(["local_mip@e0.30"], metrics, 5, 60.0)
    assert r"local\_mip@e0.30" in tex
# ── cannibalization tables (issue #100, records from #95) ─────────────────────


def _heur(name: str, phase: str, start: float, wall_ms: float,
          found: bool = False) -> HeuristicSample:
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
        samples=[_heur("fj", "presolve", 0.1, 400.0),
                 _heur("fpr", "presolve", 0.5, 600.0),
                 _heur("fpr_lp", "dive", 3.0, 250.0)]
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
    row = _instrumented(
        solve_time=10.0, samples=[_heur("fj", "presolve", 0.1, 3300.0)]
    )
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
    assert any("native heur LP iters 1000->100" in e for e in v.evidence)


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
    assert classify_cannibalization(_uninstrumented(), inst).category == "not-instrumented"
    assert classify_cannibalization(inst, _uninstrumented()).category == "no-baseline"
    assert classify_cannibalization(inst, None).category == "no-baseline"
    assert classify_cannibalization(inst, inst, is_baseline=True).category == "baseline"
    # An uninstrumented baseline row is labelled by what it carries, not by
    # its role: it cannot serve as a reference either.
    old = _uninstrumented()
    assert classify_cannibalization(old, old, is_baseline=True).category == "not-instrumented"


def test_pick_baseline_prefers_the_instrumented_zero_heuristic_config():
    agg = {
        "patched": {"a": _instrumented(samples=[_heur("fj", "presolve", 0.1, 5.0)])},
        "suite_off": {"a": _instrumented()},
    }
    assert pick_baseline_config(agg, ["patched", "suite_off"]) == "suite_off"


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


def test_cannibalization_tables_render_from_an_instrumented_tree(tmp_path, capsys):
    """Both tables, their aggregates and the classification, no scripting."""
    starved = _synth_log(
        solve_time=10.0,
        samples=[_heur("fj", "presolve", 0.1, 900.0, found=True),
                 _heur("scylla", "presolve", 1.0, 3000.0),
                 _heur("fpr_lp", "dive", 6.0, 500.0)],
        # 1300 shared heuristic LP iterations of which 900 are fpr_lp's own
        # charge, so HiGHS itself did 400 against the baseline's 2000.
        native=(1, 0, 1, 1, 1300, 12000, 900),
        root=(4.5, 3.9),
    )
    quiet = _synth_log(
        solve_time=10.0,
        samples=[_heur("fj", "presolve", 0.1, 40.0)],
        native=(3, 1, 4, 1, 2000, 11000, 0),
        root=(1.05, 0.04),
    )
    base_log = _synth_log(
        solve_time=10.0, native=(3, 1, 4, 1, 2000, 11000, 0), root=(1.0, 0.0)
    )
    _write_tree(tmp_path, {
        "patched": {"starved": starved, "quiet": quiet},
        "vanilla": {"starved": base_log, "quiet": base_log},
    })

    configs = ["patched", "vanilla"]
    results = load_results(str(tmp_path), configs)
    agg = aggregate_results(results, configs)
    print_cannibalization_tables(results, agg, configs)
    out = capsys.readouterr().out

    assert "## Cannibalization" in out
    assert "Baseline config: vanilla" in out
    assert "### Internal budget" in out
    assert "### Wall clock" in out
    assert "### Classification counts" in out

    lines = out.splitlines()
    # Native counters land in the internal-budget row, with our dive charge
    # subtracted from both shared LP counters (1300-900, 12000-900) and the
    # delta taken against the baseline's own native figure (400-2000).
    internal_row = next(
        ln for ln in lines
        if ln.startswith("starved") and "patched" in ln and "11100" in ln
    )
    assert " 400 " in internal_row
    assert "-1600" in internal_row
    # ... and the wall-clock row classifies the instance from both signals.
    starved_patched = next(
        ln for ln in lines
        if ln.startswith("starved") and "patched" in ln and "both" in ln
    )
    assert "root LP +3.50s" in starved_patched
    assert "rens_root 1->0" in starved_patched
    assert "native heur LP iters 2000->400" in starved_patched

    quiet_patched = next(
        ln for ln in lines
        if ln.startswith("quiet") and "patched" in ln and "neutral" in ln
    )
    assert "0.04" in quiet_patched  # presolve span still reported

    # The baseline config is a row in its own right.
    assert any(ln.startswith("starved") and "vanilla" in ln and "baseline" in ln
               for ln in lines)


def test_cannibalization_baseline_row_survives_aggregation(tmp_path, capsys):
    """The baseline's 0.0 heuristic wall time must reach the aggregate row.

    It is the reference every other row is read against; an aggregation that
    filtered None would drop it and leave the table without its zero line.
    """
    base_log = _synth_log(solve_time=8.0, native=(2, 1, 2, 0, 500, 5000, 0),
                          root=(0.5, 0.0))
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
    old = ("      Status            Optimal\n"
           "      Nodes             3\n"
           "      Timing            5.00\n")
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
    _write_tree(tmp_path, {
        "patched": {"i1": patched_log},
        "vanilla": {"i1": "      Status            Optimal\n"
                          "      Timing            6.00\n"},
    })
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
