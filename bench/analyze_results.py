#!/usr/bin/env python3
"""Analyze benchmark results: compute metrics, generate tables and plots."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_highs_log import SolveResult, parse_log_file


def parse_solu_file(path: str) -> dict[str, tuple[str, float | None]]:
    """Parse a MIPLIB .solu file into {instance: (tag, value)}.

    Tags: "=opt=", "=best=", "=unkn=", "=fea=".  Value is None for =unkn=.
    """
    refs: dict[str, tuple[str, float | None]] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or not parts[0].startswith("="):
                continue
            tag, name = parts[0], parts[1]
            val: float | None = None
            if len(parts) >= 3:
                try:
                    val = float(parts[2])
                except ValueError:
                    val = None
            refs[name] = (tag, val)
    return refs


def resolve_reference(
    solu_value: float | None,
    observed_primals: list[float],
    sense: str = "min",
) -> float | None:
    """Pick reference objective for primal-gap computation.

    When observed primals beat the published =best=, use the virtual best
    instead so we don't punish configs that found better solutions.
    `sense` is "min" (default, MIPLIB convention) or "max".

    NOTE: `build_best_known` below only calls this with sense='min'.  That
    matches the current MIPLIB benchmark subsets (head-to-head instances
    are all minimization); adding a maximization instance to a benchmark
    requires threading sense detection from the HiGHS log or .mps file.
    The assumption is guarded here so a future caller can opt in.
    """
    finite = [p for p in observed_primals if math.isfinite(p)]
    if solu_value is None:
        return (min(finite) if sense == "min" else max(finite)) if finite else None
    if not finite:
        return solu_value
    return min(solu_value, min(finite)) if sense == "min" else max(solu_value, max(finite))


def build_best_known(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    configs: list[str],
    instances: list[str],
    solu_refs: dict[str, tuple[str, float | None]],
) -> dict[str, float | None]:
    """Build {instance: reference_objective} from .solu + observed primals."""
    refs: dict[str, float | None] = {}
    for inst in instances:
        observed: list[float] = []
        for c in configs:
            for seed_data in results.get(c, {}).values():
                r = seed_data.get(inst)
                if r and r.primal_bound != float("inf"):
                    observed.append(r.primal_bound)
        solu_value = None
        if inst in solu_refs:
            tag, val = solu_refs[inst]
            # =opt= is proven optimal; =best= is best known (may be beatable);
            # =fea= is a feasible objective that is NOT optimal.  We still
            # treat =fea= as a reference because `resolve_reference` takes
            # min(solu_value, min(observed)) — any config that finds better
            # raises the virtual best — so a gap computed against a =fea=
            # reference is a pessimistic lower bound on the true gap.  Skip
            # =unkn= / =inf= / =unbd=.
            if tag in ("=opt=", "=best=", "=fea=") and val is not None:
                solu_value = val
        refs[inst] = resolve_reference(solu_value, observed)
    return refs


def load_results(
    results_dir: str,
    configs: list[str],
    config_dirs: dict[str, str] | None = None,
) -> dict[str, dict[int, dict[str, SolveResult]]]:
    """Load all parsed results.

    Returns {config: {seed: {instance: SolveResult}}}.
    Supports both seed-aware (results/{config}/seed{N}/*.log) and
    legacy flat (results/{config}/*.log, treated as seed 0) layouts.

    `config_dirs` optionally maps a config name to an explicit directory,
    overriding the default `results_dir/<config>`.  This lets the ablation
    analysis pull the all_opp / vanilla anchors from `bench/results/plato`
    while the ablation configs resolve under `bench/results/ablation`.
    """
    config_dirs = config_dirs or {}
    results: dict[str, dict[int, dict[str, SolveResult]]] = {}
    for config in configs:
        config_dir = config_dirs.get(config, os.path.join(results_dir, config))
        if not os.path.isdir(config_dir):
            print(f"Warning: config directory not found: {config_dir}", file=sys.stderr)
            continue
        results[config] = {}

        # Check for seed subdirectories
        seed_dirs = sorted(Path(config_dir).glob("seed*"))
        if seed_dirs:
            for sd in seed_dirs:
                if not sd.is_dir():
                    continue
                seed_num = int(sd.name.removeprefix("seed"))
                results[config][seed_num] = {}
                for log_file in sorted(sd.glob("*.log")):
                    name = log_file.stem
                    results[config][seed_num][name] = parse_log_file(str(log_file))
        else:
            # Legacy flat layout: treat as seed 0
            results[config][0] = {}
            for log_file in sorted(Path(config_dir).glob("*.log")):
                name = log_file.stem
                results[config][0][name] = parse_log_file(str(log_file))
    return results


def get_seeds(results: dict[str, dict[int, dict[str, SolveResult]]]) -> list[int]:
    """Get sorted list of all seeds across configs."""
    seeds: set[int] = set()
    for config_data in results.values():
        seeds.update(config_data.keys())
    return sorted(seeds)


def get_common_instances(
    results: dict[str, dict[int, dict[str, SolveResult]]], configs: list[str]
) -> list[str]:
    """Get instances present in all configs and at least one seed."""
    per_config: list[set[str]] = []
    for config in configs:
        if config not in results:
            continue
        inst_set: set[str] = set()
        for seed_data in results[config].values():
            inst_set.update(seed_data.keys())
        per_config.append(inst_set)
    if not per_config:
        return []
    return sorted(set.intersection(*per_config))


def aggregate_results(
    results: dict[str, dict[int, dict[str, SolveResult]]], configs: list[str]
) -> dict[str, dict[str, SolveResult]]:
    """Aggregate across seeds using median for each metric.

    Returns {config: {instance: SolveResult}} with median values.
    For incumbents, uses the seed with the median primal_bound (upper-middle for even N).
    """
    aggregated: dict[str, dict[str, SolveResult]] = {}
    for config in configs:
        if config not in results:
            continue
        aggregated[config] = {}
        instances = get_common_instances(results, [config])
        seeds = sorted(results[config].keys())

        for inst in instances:
            seed_results = [
                results[config][s][inst]
                for s in seeds
                if inst in results[config][s]
            ]
            if not seed_results:
                continue
            if len(seed_results) == 1:
                aggregated[config][inst] = seed_results[0]
                continue

            # Pick the median-performing seed based on primal_bound
            # (lower is better for minimization; use the middle one)
            by_obj = sorted(seed_results, key=lambda r: r.primal_bound)
            median_r = by_obj[len(by_obj) // 2]

            aggregated[config][inst] = median_r
    return aggregated


def shifted_geomean(values: list[float], shift: float = 1.0) -> float:
    """Shifted geometric mean: exp(mean(log(x + shift))) - shift."""
    if not values:
        return float("nan")
    log_sum = sum(math.log(max(v + shift, 1e-12)) for v in values)
    return math.exp(log_sum / len(values)) - shift


def format_float(v: float | None, width: int = 10, prec: int = 4) -> str:
    if v is None:
        return "-".rjust(width)
    if v == float("inf"):
        return "inf".rjust(width)
    if abs(v) < 1e-8:
        return "0".rjust(width)
    return f"{v:.{prec}f}".rjust(width)


def format_int(v: int | None, width: int = 8, signed: bool = False) -> str:
    """Right-align an integer, rendering a missing value as '-'."""
    if v is None:
        return "-".rjust(width)
    return (f"{v:+d}" if signed else str(v)).rjust(width)


def count_feasible(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    config: str,
    instances: list[str],
) -> dict[str, int]:
    """Count #Feas: instances with at least one feasible solution across all seeds.

    Returns {"per_seed": {seed: count}, "any": count_any_seed}.
    """
    seeds = sorted(results.get(config, {}).keys())
    per_seed = {}
    any_seed_count = 0

    for inst in instances:
        found_any = False
        for s in seeds:
            r = results.get(config, {}).get(s, {}).get(inst)
            if r and r.incumbents:
                per_seed[s] = per_seed.get(s, 0) + 1
                found_any = True
        if found_any:
            any_seed_count += 1

    return {"per_seed": per_seed, "any": any_seed_count}


def count_first(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
    tie_tol: float = 0.1,
) -> dict[str, float]:
    """Count #First: instances where config finds first feasible strictly
    earliest (all others later by > tie_tol seconds). Ties within tie_tol
    split the credit evenly across the tied configs. Returns fractional
    counts per config.
    """
    firsts: dict[str, float] = {c: 0.0 for c in configs}
    for inst in instances:
        times: dict[str, float] = {}
        for c in configs:
            r = agg_results.get(c, {}).get(inst)
            if r and r.time_to_first_feasible is not None:
                times[c] = r.time_to_first_feasible
        if not times:
            continue
        best = min(times.values())
        leaders = [c for c, t in times.items() if t <= best + tie_tol]
        share = 1.0 / len(leaders)
        for c in leaders:
            firsts[c] += share
    return firsts


def count_wins(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
) -> dict[str, int]:
    """Count #Win: instances where one config finds a strictly better primal bound.

    Ties (all best-bound holders within 1e-6 of each other) are not credited
    to any config. Returns {config: win_count}.
    """
    wins = {c: 0 for c in configs}
    for inst in instances:
        bounds = {}
        for c in configs:
            r = agg_results.get(c, {}).get(inst)
            if r and r.primal_bound != float("inf"):
                bounds[c] = r.primal_bound
        if not bounds:
            continue
        best = min(bounds.values())
        winners = [c for c, b in bounds.items() if abs(b - best) < 1e-6]
        if len(winners) == 1:
            wins[winners[0]] += 1
    return wins


# Incumbent source-char → heuristic label.  In the PATCHED build the custom
# heuristics emit these display chars (see the "Src:" legend printed in every
# patched log): A=FPR, D=FPR_LP, M=LocalMIP, G=Scylla, J=FJ.  HiGHS's built-in
# Feasibility-Jump dispatch is patched off, so a 'J' in a patched log is OUR FJ.
# Every other char is a HiGHS built-in source (branching, rounding, sub-MIP, the
# trivial rounders, the presolve 'P' seed, …) and is bucketed as "HiGHS/other".
CUSTOM_SOURCE_LABELS: dict[str, str] = {
    "A": "FPR",
    "D": "FPR_LP",
    "M": "LocalMIP",
    "G": "Scylla",
    "J": "FJ",
}

# Order used when printing attribution rows (custom heuristics first, then the
# built-in bucket).
ATTRIBUTION_ORDER: list[str] = ["FPR", "FPR_LP", "LocalMIP", "Scylla", "FJ", "HiGHS/other"]


def source_label(src: str) -> str:
    """Map an incumbent source char to a heuristic label (see CUSTOM_SOURCE_LABELS)."""
    return CUSTOM_SOURCE_LABELS.get(src, "HiGHS/other")


def heuristic_attribution(
    agg_results: dict[str, dict[str, SolveResult]],
    config: str,
    instances: list[str],
) -> dict[str, object]:
    """Attribute, for one config, which heuristic found the first feasible
    incumbent and which produced the best (final) incumbent, per instance.

    Returns {"first": {label: count}, "best": {label: count}, "n_feasible": int},
    where labels come from `source_label`.  Only instances with at least one
    recorded incumbent are counted, so the per-dict totals equal n_feasible.
    """
    first: dict[str, int] = {}
    best: dict[str, int] = {}
    n_feasible = 0
    for inst in instances:
        r = agg_results.get(config, {}).get(inst)
        if not r or not r.incumbents:
            continue
        n_feasible += 1
        f = source_label(r.incumbents[0].source)
        b = source_label(r.incumbents[-1].source)
        first[f] = first.get(f, 0) + 1
        best[b] = best.get(b, 0) + 1
    return {"first": first, "best": best, "n_feasible": n_feasible}


def print_attribution(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
) -> None:
    """Print, per config, the first-feasible and best-held heuristic breakdown.

    Reads only the already-parsed incumbent sources — no solves.  Most
    meaningful for the multi-heuristic configs (e.g. all_opp); single-heuristic
    or vanilla configs simply attribute everything to one source.
    """
    instances = get_common_instances(results, configs)
    print(f"\n## Heuristic attribution ({len(instances)} instances)\n")
    print("Columns: #First = found the first feasible incumbent; "
          "#Best = held the best incumbent at termination.\n")

    for c in configs:
        attr = heuristic_attribution(agg_results, c, instances)
        first = attr["first"]  # type: ignore[assignment]
        best = attr["best"]    # type: ignore[assignment]
        n_feas = attr["n_feasible"]
        print(f"### {c}  ({n_feas} feasible)")
        print(f"{'Heuristic':<14} {'#First':>8} {'#Best':>8}")
        print("-" * 32)
        labels = [lbl for lbl in ATTRIBUTION_ORDER if first.get(lbl) or best.get(lbl)]  # type: ignore[union-attr]
        for lbl in labels:
            print(f"{lbl:<14} {first.get(lbl, 0):>8} {best.get(lbl, 0):>8}")  # type: ignore[union-attr]
        print()


def print_comparison_table(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_cutoffs: list[float] | None = None,
    best_known: dict[str, float | None] | None = None,
    time_limit: float = 600.0,
) -> None:
    """Print per-instance comparison table using seed-aggregated values."""
    if time_cutoffs is None:
        time_cutoffs = [10.0, 60.0, 600.0]

    if len(configs) < 2:
        print("Need at least 2 configs for comparison")
        return

    c1, c2 = configs[0], configs[1]
    instances = sorted(set(agg_results.get(c1, {}).keys()) & set(agg_results.get(c2, {}).keys()))

    if not instances:
        print("No common instances found between configs")
        return

    # Determine which time cutoffs are relevant
    max_solve_time = max(
        max((r.solve_time for r in agg_results[c1].values()), default=0),
        max((r.solve_time for r in agg_results[c2].values()), default=0),
    )
    active_cutoffs = [tc for tc in time_cutoffs if tc <= max_solve_time + 1]

    # Header
    print(f"\n{'Instance':<25} ", end="")
    print(f"{'T1st(' + c1 + ')':<10} {'T1st(' + c2 + ')':<10} ", end="")
    for tc in active_cutoffs:
        print(f"{'Gap@' + str(int(tc)) + '(' + c1[:3] + ')':<12} {'Gap@' + str(int(tc)) + '(' + c2[:3] + ')':<12} ", end="")
    print(f"{'PD(' + c1[:3] + ')':<12} {'PD(' + c2[:3] + ')':<12} {'Status(' + c1[:3] + ')':<15} {'Status(' + c2[:3] + ')':<15}")
    print("-" * 180)

    # Per-instance rows
    wins = {"t1st": 0, "gap": {tc: 0 for tc in time_cutoffs}, "pd": 0}
    losses = {"t1st": 0, "gap": {tc: 0 for tc in time_cutoffs}, "pd": 0}
    ties = {"t1st": 0, "gap": {tc: 0 for tc in time_cutoffs}, "pd": 0}

    t1st_vals = {c1: [], c2: []}
    gap_vals = {tc: {c1: [], c2: []} for tc in time_cutoffs}
    pd_vals = {c1: [], c2: []}
    n_both = n_c1_only = n_c2_only = n_neither = 0

    for inst in instances:
        r1, r2 = agg_results[c1][inst], agg_results[c2][inst]

        print(f"{inst:<25} ", end="")

        # Time to first feasible
        t1 = r1.time_to_first_feasible
        t2 = r2.time_to_first_feasible
        print(f"{format_float(t1, 10, 2)} {format_float(t2, 10, 2)} ", end="")
        if t1 is not None or t2 is not None:
            tl = time_limit
            if t1 is not None:
                t1st_vals[c1].append(min(t1, tl))
            if t2 is not None:
                t1st_vals[c2].append(min(t2, tl))
            if t1 is not None and t2 is not None:
                if t1 < t2 - 0.01:
                    wins["t1st"] += 1
                elif t2 < t1 - 0.01:
                    losses["t1st"] += 1
                else:
                    ties["t1st"] += 1
            elif t1 is not None:
                wins["t1st"] += 1
            else:
                losses["t1st"] += 1

        # Track coverage
        if t1 is not None and t2 is not None:
            n_both += 1
        elif t1 is not None:
            n_c1_only += 1
        elif t2 is not None:
            n_c2_only += 1
        else:
            n_neither += 1

        # Gap at cutoffs — treat no solution as gap=1.0; count when at least
        # one config found a solution, skip only when neither did.
        ref = best_known.get(inst) if best_known else None
        for tc in active_cutoffs:
            g1 = r1.primal_gap_at(tc, ref)
            g2 = r2.primal_gap_at(tc, ref)
            print(f"{format_float(g1, 12, 6)} {format_float(g2, 12, 6)} ", end="")
            if g1 is not None or g2 is not None:
                g1c = g1 if g1 is not None else 1.0
                g2c = g2 if g2 is not None else 1.0
                gap_vals[tc][c1].append(g1c)
                gap_vals[tc][c2].append(g2c)
                if g1c < g2c - 1e-6:
                    wins["gap"][tc] += 1
                elif g2c < g1c - 1e-6:
                    losses["gap"][tc] += 1
                else:
                    ties["gap"][tc] += 1

        # P-D integral (from HiGHS)
        pd1 = r1.pd_integral if r1.pd_integral != float("inf") else None
        pd2 = r2.pd_integral if r2.pd_integral != float("inf") else None
        print(f"{format_float(pd1, 12, 4)} {format_float(pd2, 12, 4)} ", end="")
        if pd1 is not None and pd2 is not None:
            pd_vals[c1].append(pd1)
            pd_vals[c2].append(pd2)
            if pd1 < pd2 - 1e-6:
                wins["pd"] += 1
            elif pd2 < pd1 - 1e-6:
                losses["pd"] += 1
            else:
                ties["pd"] += 1

        print(f"{r1.status:<15} {r2.status:<15}")

    # Summary
    print("-" * 180)
    print(f"\n## Summary: {c1} vs {c2} ({len(instances)} instances)\n")
    print(f"Coverage: {n_both} both solved  |  "
          f"{n_c1_only} {c1}-only  |  {n_c2_only} {c2}-only  |  {n_neither} neither\n")

    print(f"{'Metric':<25} {c1 + ' wins':<12} {c2 + ' wins':<12} {'Ties':<8} "
          f"{'SGM(' + c1[:3] + ')':<12} {'SGM(' + c2[:3] + ')':<12}")
    print("-" * 80)

    print(f"{'Time to 1st feasible':<25} {wins['t1st']:<12} {losses['t1st']:<12} {ties['t1st']:<8} "
          f"{format_float(shifted_geomean(t1st_vals[c1], 1.0), 12, 4)} "
          f"{format_float(shifted_geomean(t1st_vals[c2], 1.0), 12, 4)}")

    for tc in time_cutoffs:
        if gap_vals[tc][c1]:
            label = f"Gap @ {int(tc)}s"
            print(f"{label:<25} {wins['gap'][tc]:<12} {losses['gap'][tc]:<12} {ties['gap'][tc]:<8} "
                  f"{format_float(shifted_geomean(gap_vals[tc][c1], 0.001), 12, 6)} "
                  f"{format_float(shifted_geomean(gap_vals[tc][c2], 0.001), 12, 6)}")

    print(f"{'P-D integral':<25} {wins['pd']:<12} {losses['pd']:<12} {ties['pd']:<8} "
          f"{format_float(shifted_geomean(pd_vals[c1], 1.0), 12, 4)} "
          f"{format_float(shifted_geomean(pd_vals[c2], 1.0), 12, 4)}")


def _categorize_instances(
    agg_results: dict[str, dict[str, SolveResult]],
    instances: list[str],
) -> dict[str, str]:
    """Return {instance: category} using the first config that parsed model stats."""
    cats: dict[str, str] = {}
    for inst in instances:
        for cfg_results in agg_results.values():
            r = cfg_results.get(inst)
            if r and r.category is not None:
                cats[inst] = r.category
                break
    return cats


def _print_category_breakdown(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
) -> None:
    """Local-MIP Table 1 style: #Feas / #Win per category × per time cutoff."""
    cats = _categorize_instances(agg_results, instances)
    ordered = ["BP", "IP", "MBP", "MIP"]
    buckets = {c: [i for i in instances if cats.get(i) == c] for c in ordered}
    uncls = [i for i in instances if i not in cats]
    if uncls:
        buckets["?"] = uncls

    print("\n### Category breakdown (#Feas / #Win)\n")
    header = f"{'Category':<8} {'#Inst':>6}"
    for c in configs:
        header += f"  {'Feas(' + c[:4] + ')':>11} {'Win(' + c[:4] + ')':>11}"
    print(header)
    print("-" * len(header))

    for cat, insts_in_cat in buckets.items():
        if not insts_in_cat:
            continue
        feas = {c: count_feasible(results, c, insts_in_cat)["any"] for c in configs}
        wins = count_wins(agg_results, configs, insts_in_cat)
        row = f"{cat:<8} {len(insts_in_cat):>6}"
        for c in configs:
            row += f"  {feas[c]:>11} {wins[c]:>11}"
        print(row)


def print_paper_metrics(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None = None,
) -> None:
    """Print paper-standard metrics: #Feas, #Win, SGM of T1st, SGM of gap@cutoff."""
    instances = get_common_instances(results, configs)
    seeds = get_seeds(results)

    print(f"\n## Paper Metrics ({len(instances)} instances, {len(seeds)} seed(s))\n")

    # --- #Feas per seed and aggregated ---
    print(f"{'#Feas':<25}", end="")
    for c in configs:
        print(f" {c:<12}", end="")
    print()
    print("-" * (25 + 13 * len(configs)))

    feas_data = {c: count_feasible(results, c, instances) for c in configs}
    for s in seeds:
        print(f"  seed {s:<19}", end="")
        for c in configs:
            count = feas_data[c]["per_seed"].get(s, 0)
            print(f" {count:<12}", end="")
        print()
    if len(seeds) > 1:
        print(f"  {'any seed':<21}", end="")
        for c in configs:
            print(f" {feas_data[c]['any']:<12}", end="")
        print()

    # --- #Win (on aggregated) ---
    win_counts = count_wins(agg_results, configs, instances)
    print(f"\n{'#Win (best obj)':<25}", end="")
    for c in configs:
        print(f" {win_counts[c]:<12}", end="")
    print()

    # --- #First (fastest to feasible, tie-split within 0.1s) ---
    first_counts = count_first(agg_results, configs, instances)
    print(f"{'#First (fastest T1st)':<25}", end="")
    for c in configs:
        print(f" {first_counts[c]:<12.1f}", end="")
    print()

    # --- SGM of time-to-first-feasible (shift=1s, matching FJ/FPR) ---
    # Clamp at time_limit: HiGHS occasionally reports an incumbent found
    # fractionally after the wall-clock limit (node completing mid-timeout).
    print(f"\n{'SGM T1st (s=1)':<25}", end="")
    for c in configs:
        t1st = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r and r.time_to_first_feasible is not None:
                t1st.append(min(r.time_to_first_feasible, time_limit))
        print(f" {format_float(shifted_geomean(t1st, 1.0), 12, 4)}", end="")
    print()

    # --- SGM of time-to-best incumbent (shift=1s) ---
    print(f"{'SGM Tbest (s=1)':<25}", end="")
    for c in configs:
        tbest = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r and r.time_to_best is not None:
                tbest.append(min(r.time_to_best, time_limit))
        print(f" {format_float(shifted_geomean(tbest, 1.0), 12, 4)}", end="")
    print()

    # --- SGM of primal gap at cutoff (shift=0.001, matching PLATO) ---
    # Infeasible instances contribute gap=1.0 so all instances are counted
    # (matching Mittelmann's PLATO methodology).
    print(f"{'SGM Gap@' + str(int(time_limit)) + 's (s=0.001)':<25}", end="")
    for c in configs:
        gaps = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r:
                ref = best_known.get(inst) if best_known else None
                g = r.primal_gap_at(time_limit, ref)
                gaps.append(g if g is not None else 1.0)
        print(f" {format_float(shifted_geomean(gaps, 0.001), 12, 6)}", end="")
    print()

    # --- SGM of primal integral (shift=1.0) ---
    print(f"{'SGM Primal Integral':<25}", end="")
    for c in configs:
        pis = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r:
                ref = best_known.get(inst) if best_known else None
                pi = r.primal_integral(time_limit, ref)
                pis.append(pi)
        print(f" {format_float(shifted_geomean(pis, 1.0), 12, 4)}", end="")
    print()

    # --- SGM of HiGHS-reported primal-dual integral (shift=1.0) ---
    print(f"{'SGM P-D Integral':<25}", end="")
    for c in configs:
        pdis = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r and math.isfinite(r.pd_integral):
                pdis.append(r.pd_integral)
        print(f" {format_float(shifted_geomean(pdis, 1.0), 12, 4)}", end="")
    print()

    # --- Category breakdown (Local-MIP Table 1 style) ---
    if len(configs) >= 1:
        _print_category_breakdown(
            results, agg_results, configs, instances, time_limit, best_known
        )

    # --- Reference coverage ---
    if best_known is not None:
        covered = sum(1 for inst in instances if best_known.get(inst) is not None)
        print(f"\n(reference objective available for {covered}/{len(instances)} instances)")


def generate_survival_plot(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    output_path: str,
    gap_threshold: float = 0.01,
) -> None:
    """Generate survival plot: fraction of instances solved to gap% over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping survival plot", file=sys.stderr)
        return

    instances = sorted(set.intersection(*(set(agg_results[c].keys()) for c in configs)))
    if not instances:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for config in configs:
        # For each instance, find time when gap <= threshold
        solve_times = []
        for inst in instances:
            r = agg_results[config][inst]
            # Find first incumbent where gap <= threshold
            found = False
            for inc in r.incumbents:
                denom = max(abs(inc.dual_bound), 1.0)
                gap = abs(inc.objective - inc.dual_bound) / denom
                if gap <= gap_threshold:
                    solve_times.append(inc.time)
                    found = True
                    break
            if not found:
                # Check final status — may have reached gap after last logged incumbent
                if r.gap <= gap_threshold:
                    solve_times.append(r.solve_time)
                else:
                    solve_times.append(float("inf"))

        # Sort and create survival curve
        solve_times.sort()
        n = len(solve_times)
        times = [0.0]
        fractions = [0.0]
        solved = 0
        for t in solve_times:
            if t == float("inf"):
                break
            solved += 1
            times.append(t)
            fractions.append(solved / n)

        # Extend to max time
        max_time = max(r.solve_time for r in agg_results[config].values())
        times.append(max_time)
        fractions.append(fractions[-1])

        ax.step(times, fractions, where="post", label=config, linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Fraction solved to {gap_threshold*100:.0f}% gap")
    ax.set_title(f"Survival Plot (gap threshold = {gap_threshold*100:.0f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Survival plot saved to {output_path}")


def print_plato_summary(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
) -> None:
    """Print PLATO-style headline metrics: primal-integral SGM ratio and feasibility counts.

    The PLATO benchmark uses primal integral (area under primal-gap curve, 600s
    window, shift=0.001) as the primary metric with shifted geometric mean across
    all 233 instances, and counts the number of instances where a feasible solution
    was found within the time limit.
    """
    instances = get_common_instances(results, configs)
    if not instances:
        return

    # PLATO shift matches Mittelmann's published methodology (sh=0.001)
    plato_shift = 0.001

    print(f"\n## PLATO Headline Metrics ({len(instances)} instances, {time_limit:.0f}s, SGM shift={plato_shift})\n")

    pi_per_config: dict[str, list[float]] = {}
    feas_per_config: dict[str, int] = {}
    for c in configs:
        pis = []
        feas_count = 0
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r:
                ref = best_known.get(inst) if best_known else None
                pi = r.primal_integral(time_limit, ref)
                pis.append(pi)
                if r.incumbents:
                    feas_count += 1
        pi_per_config[c] = pis
        feas_per_config[c] = feas_count

    # Print per-config row
    print(f"{'Config':<20} {'SGM(PrimalIntegral)':<22} {'#Feasible':>10} {'#Instances':>12}")
    print("-" * 68)
    for c in configs:
        sgm = shifted_geomean(pi_per_config[c], plato_shift)
        print(f"{c:<20} {sgm:<22.6f} {feas_per_config[c]:>10} {len(instances):>12}")

    # Print pairwise ratios
    if len(configs) >= 2:
        print()
        c1, c2 = configs[0], configs[1]
        sgm1 = shifted_geomean(pi_per_config[c1], plato_shift)
        sgm2 = shifted_geomean(pi_per_config[c2], plato_shift)
        if sgm2 > 0 and math.isfinite(sgm2) and math.isfinite(sgm1):
            ratio = sgm1 / sgm2
            winner = c1 if ratio < 1.0 else c2
            print(f"SGM ratio {c1}/{c2}: {ratio:.4f}  (lower is better; winner: {winner})")
        feas1, feas2 = feas_per_config[c1], feas_per_config[c2]
        print(f"Feasible delta {c1}-{c2}: {feas1 - feas2:+d}  ({c1}={feas1}, {c2}={feas2})")


def _config_metrics(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    config: str,
    instances: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
) -> dict[str, float]:
    """Compute the one-row-per-config ablation metrics for a single config.

    Returns #Feasible (any seed) plus shifted-geometric-mean of T1st (shift=1),
    primal gap @time_limit (shift=0.001), primal integral (shift=1), and the
    PLATO headline (primal integral, shift=0.001).  All reuse the same helpers
    as the headline tables so an ablation row at a horizon T is directly
    comparable to the anchors analyzed at the same T.
    """
    feas = count_feasible(results, config, instances)["any"]
    t1st: list[float] = []
    gaps: list[float] = []
    pis: list[float] = []
    for inst in instances:
        r = agg_results.get(config, {}).get(inst)
        if not r:
            continue
        if r.time_to_first_feasible is not None:
            t1st.append(min(r.time_to_first_feasible, time_limit))
        ref = best_known.get(inst) if best_known else None
        g = r.primal_gap_at(time_limit, ref)
        gaps.append(g if g is not None else 1.0)
        pis.append(r.primal_integral(time_limit, ref))
    return {
        "feasible": float(feas),
        "sgm_t1st": shifted_geomean(t1st, 1.0),
        "sgm_gap": shifted_geomean(gaps, 0.001),
        "sgm_pi": shifted_geomean(pis, 1.0),
        "plato_sgm": shifted_geomean(pis, 0.001),
    }


def print_ablation_summary(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
    latex_path: str | None = None,
) -> None:
    """Print a one-row-per-config ablation table (and optionally write LaTeX).

    Unlike `print_comparison_table` (pairwise), this lists every config on its
    own row, the shape needed for the per-component ablation in the paper.
    """
    instances = get_common_instances(results, configs)
    metrics = {c: _config_metrics(results, agg_results, c, instances, time_limit, best_known)
               for c in configs}

    print(f"\n## Ablation summary ({len(instances)} instances, {time_limit:.0f}s horizon)\n")
    header = (f"{'Config':<22} {'#Feas':>6} {'SGM T1st':>10} "
              f"{'SGM Gap':>10} {'SGM PI':>10} {'PLATO SGM':>11}")
    print(header)
    print("-" * len(header))
    for c in configs:
        m = metrics[c]
        print(f"{c:<22} {int(m['feasible']):>6} {m['sgm_t1st']:>10.4f} "
              f"{m['sgm_gap']:>10.6f} {m['sgm_pi']:>10.4f} {m['plato_sgm']:>11.4f}")

    if latex_path:
        with open(latex_path, "w") as f:
            f.write(latex_ablation_table(configs, metrics, len(instances), time_limit))
        print(f"\nLaTeX ablation table written to {latex_path}")


def latex_ablation_table(
    configs: list[str],
    metrics: dict[str, dict[str, float]],
    n_instances: int,
    time_limit: float,
) -> str:
    """Render the ablation metrics as a booktabs LaTeX table (string).

    Config labels are emitted with underscores escaped for LaTeX.  Kept here
    (rather than in the paper repo) so the table regenerates directly from the
    committed logs, matching the cptp-paper convention.
    """
    def esc(s: str) -> str:
        return s.replace("_", r"\_")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Per-component ablation on {n_instances} PLATO instances "
        rf"at a {time_limit:.0f}\,s horizon (single seed). SGM = shifted "
        r"geometric mean; PI = primal integral; PLATO SGM is the headline "
        r"primal-integral SGM (shift $0.001$). Lower is better except \#Feas.}",
        r"\label{tbl:ablation}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Config & \#Feas & SGM T1st & SGM Gap & SGM PI & PLATO SGM \\",
        r"\midrule",
    ]
    for c in configs:
        m = metrics[c]
        lines.append(
            f"{esc(c)} & {int(m['feasible'])} & {m['sgm_t1st']:.4f} & "
            f"{m['sgm_gap']:.6f} & {m['sgm_pi']:.4f} & {m['plato_sgm']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ── Cannibalization analysis (epic #88; instrumentation from issue #95) ───────
#
# The closeout epic's central empirical question is whether running our
# presolve heuristics starves HiGHS's own RENS/RINS and delays the root LP.
# It separates two kinds of cost, and the tables below keep them apart:
#
#   internal-budget — our heuristics consume the counters HiGHS itself reads
#                     when deciding whether its own heuristics may run, so
#                     RENS/RINS/root-reduced-cost fire less often.  This is an
#                     integration confound, to be eliminated.
#   wall-clock      — those budgets are preserved, but the solve still pays
#                     for the heuristics in end-to-end time (root LP pushed
#                     later, a large share of the solve spent inside them).
#                     This is a real cost and stays in the results.
#
# Every judgement is made against one baseline config — the vanilla-equivalent
# row (a patched binary at `mip_heuristic_suite=off`, which is instrumented and
# dispatches no custom heuristic).  A config compared against nothing cannot be
# classified, and says so rather than guessing.

# Wall-clock half of the classification.  Blunt on purpose: these labels are
# triage for the closeout tables, not a statistical test.
CANNIBALIZATION_WALL_FRACTION = 0.05  # >= 5% of the solve spent in our heuristics
CANNIBALIZATION_ROOT_DELAY_REL = 0.10  # root LP >= 10% later than the baseline's,
CANNIBALIZATION_ROOT_DELAY_ABS = 0.05  # ... and at least this many seconds later
# Internal-budget half.  Call counts are small integers and are compared
# strictly; LP-iteration counts drift for benign reasons (a different incumbent
# changes how long a sub-MIP runs), so they need a margin.
CANNIBALIZATION_LP_DROP_REL = 0.10

# Config names tried, in order, when no baseline is named on the command line
# and the structural test (below) is inconclusive.
CANNIBALIZATION_BASELINE_NAMES: tuple[str, ...] = (
    "vanilla",
    "off",
    "suite_off",
    "baseline",
)

# Printed in this order in the classification counts.
CANNIBALIZATION_CATEGORIES: tuple[str, ...] = (
    "baseline",
    "neutral",
    "wall-clock",
    "internal-budget",
    "both",
    "no-baseline",
    "not-instrumented",
)


@dataclass(frozen=True)
class CannibalizationVerdict:
    """One (instance, config) classification plus the evidence behind it.

    `category` is one of `CANNIBALIZATION_CATEGORIES`:

    baseline         this row *is* the baseline; it is the reference, not a
                     subject.
    neutral          native heuristic activity preserved and no material
                     wall-clock cost.
    wall-clock       native activity preserved, but the solve paid in time —
                     root LP delayed and/or a material share of it spent
                     inside our heuristics.
    internal-budget  HiGHS's own heuristics ran less: fewer calls (root-site
                     RENS counts on its own) or materially fewer heuristic LP
                     iterations of their own.
    both             both signals fired.
    no-baseline      this row is instrumented but the baseline row is missing
                     or not instrumented, so nothing can be compared.
    not-instrumented this log predates issue #95 (or ran below
                     log_dev_level=3) and carries none of the counters.
    """

    category: str
    internal: bool
    wall: bool
    evidence: tuple[str, ...] = ()


def is_instrumented(r: SolveResult | None) -> bool:
    """True when this log carries the issue-#95 cannibalization lines.

    Keyed on the `[Native]` line, the same marker
    `SolveResult.heuristic_wall_fraction` uses.  `[Native]` is emitted once per
    solve on every instrumented run, `mip_heuristic_suite=off` included, so its
    presence — not the presence of `[Heur]` lines — is what separates "ran no
    custom heuristics" from "cannot say".
    """
    return r is not None and r.native is not None


def heuristic_wall_seconds(r: SolveResult | None, phase: str | None = None) -> float | None:
    """Wall seconds spent inside our heuristics, optionally one phase only.

    Returns **0.0**, not None, for an instrumented run that dispatched no
    custom heuristic: the baseline row's true value is zero and it must survive
    any aggregation that filters None, since it is the reference for every
    other row.  None means the log is not instrumented.
    """
    if r is None or not is_instrumented(r):
        return None
    return sum(h.wall_ms for h in r.heuristic_samples
               if phase is None or h.phase == phase) / 1000.0


def native_call_total(r: SolveResult | None) -> int | None:
    """RENS + RINS + root-reduced-cost calls, i.e. HiGHS's own heuristic calls.

    These three counters are purely native — they are incremented at upstream
    call sites only — unlike the LP-iteration counters, which are shared.
    """
    if r is None or r.native is None:
        return None
    return r.native.rens + r.native.rins + r.native.rcfix


def presolve_span_seconds(r: SolveResult | None) -> float | None:
    """`[Root] presolve_heur_s`: the custom presolve chain's total span.

    On a restarting instance HiGHS re-runs the chain, so this accumulates over
    restarts while `time_to_root_lp` pins the *first* root LP.  A span larger
    than the root-LP timestamp is expected there and is neither corrected nor
    flagged.
    """
    if r is None or r.root is None:
        return None
    return r.root.presolve_heur_s


def _config_dispatches_no_heuristics(
    agg_results: dict[str, dict[str, SolveResult]], config: str
) -> bool:
    """True when this config is instrumented and never dispatched a heuristic.

    The structural fingerprint of the vanilla-equivalent row: `[Native]` lines
    present (so it came from a patched binary at log_dev_level=3) and not one
    `[Heur]` line anywhere.
    """
    rows = list(agg_results.get(config, {}).values())
    if not any(is_instrumented(r) for r in rows):
        return False
    return all(not r.heuristic_samples for r in rows)


def pick_baseline_config(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    explicit: str | None = None,
) -> str | None:
    """Choose the config every other row is compared against.

    An explicit name wins outright (None if it is not in the tree, which the
    caller reports).  Otherwise prefer a config that is instrumented *and*
    dispatched nothing — the vanilla-equivalent `suite=off` row — breaking ties
    by `CANNIBALIZATION_BASELINE_NAMES`.  Failing that, fall back to those
    names alone, which is what an externally built unpatched `vanilla` binary
    hits: its logs carry no instrumentation, so the rows come out
    `no-baseline`, with the reason stated rather than silently omitted.
    """
    if explicit is not None:
        return explicit if explicit in configs else None
    candidates = [c for c in configs if _config_dispatches_no_heuristics(agg_results, c)]
    for name in CANNIBALIZATION_BASELINE_NAMES:
        if name in candidates:
            return name
    if candidates:
        return candidates[0]
    for name in CANNIBALIZATION_BASELINE_NAMES:
        if name in configs:
            return name
    return None


def classify_cannibalization(
    row: SolveResult | None,
    base: SolveResult | None,
    is_baseline: bool = False,
    wall_fraction: float = CANNIBALIZATION_WALL_FRACTION,
    root_delay_rel: float = CANNIBALIZATION_ROOT_DELAY_REL,
    root_delay_abs: float = CANNIBALIZATION_ROOT_DELAY_ABS,
    lp_drop_rel: float = CANNIBALIZATION_LP_DROP_REL,
) -> CannibalizationVerdict:
    """Classify one (instance, config) row against the baseline row."""
    if is_baseline:
        return CannibalizationVerdict("baseline", False, False)
    rn = row.native if row is not None else None
    bn = base.native if base is not None else None
    if row is None or rn is None:
        return CannibalizationVerdict("not-instrumented", False, False)
    if base is None or bn is None:
        return CannibalizationVerdict("no-baseline", False, False)

    internal: list[str] = []
    # Root-site RENS on its own.  The root gate is the one a presolve-found
    # incumbent closes, and the merged rens total can hold steady while the
    # root call vanishes, so this is checked before and separately from it.
    if rn.rens_root < bn.rens_root:
        internal.append(f"rens_root {bn.rens_root}->{rn.rens_root}")
    row_calls, base_calls = native_call_total(row), native_call_total(base)
    if row_calls is not None and base_calls is not None and row_calls < base_calls:
        internal.append(f"native calls {base_calls}->{row_calls}")
    # Native LP iterations, with our dive heuristic's own charge subtracted
    # from both sides — the raw counters are shared and reading them raw bills
    # our work as upstream's.
    if bn.native_heur_lp_iters > 0 and (
        rn.native_heur_lp_iters < bn.native_heur_lp_iters * (1.0 - lp_drop_rel)
    ):
        internal.append(
            f"native heur LP iters {bn.native_heur_lp_iters}->{rn.native_heur_lp_iters}"
        )

    wall: list[str] = []
    frac = row.heuristic_wall_fraction
    if frac is not None and frac >= wall_fraction:
        wall.append(f"heur wall {frac * 100:.1f}% of solve")
    t_row, t_base = row.time_to_root_lp, base.time_to_root_lp
    if t_row is not None and t_base is not None:
        delay = t_row - t_base
        # Both a relative and an absolute margin must be cleared: 10% of a
        # 0.02 s root LP is noise, and an 0.05 s delay on a 300 s one is too.
        if delay > max(root_delay_rel * t_base, root_delay_abs):
            wall.append(f"root LP +{delay:.2f}s")

    if internal and wall:
        category = "both"
    elif internal:
        category = "internal-budget"
    elif wall:
        category = "wall-clock"
    else:
        category = "neutral"
    return CannibalizationVerdict(
        category, bool(internal), bool(wall), tuple(internal + wall)
    )


def _median_or_none(values: list[float]) -> float | None:
    """Median of the observed values, None when nothing was observed."""
    return statistics.median(values) if values else None


def _print_internal_budget_table(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
    baseline: str | None,
) -> None:
    """Per instance x config: HiGHS's own heuristic calls and LP iterations."""
    print(f"\n### Internal budget ({len(instances)} instances)\n")
    print("RENSroot is the root-node subset of RENS and is never folded into it: "
          "the root gate is\nthe one a presolve-found incumbent closes, so a "
          "suppressed root call is the signal even\nwhen the merged total holds "
          "steady.  NatHeurLP / NatTotLP are the shared LP-iteration\ncounters "
          "with our own dive heuristic's charge (OurLP) subtracted; dNatHeurLP "
          "is NatHeurLP\nagainst the baseline row.\n")

    header = (f"{'Instance':<24} {'Config':<14} {'RENS':>6} {'RENSroot':>9} "
              f"{'RINS':>6} {'RCfix':>6} {'NatHeurLP':>11} {'NatTotLP':>11} "
              f"{'OurLP':>9} {'dNatHeurLP':>11}")
    print(header)
    print("-" * len(header))

    for inst in instances:
        base_r = agg_results.get(baseline, {}).get(inst) if baseline else None
        for c in configs:
            r = agg_results.get(c, {}).get(inst)
            n = r.native if r is not None else None
            base_n = base_r.native if base_r is not None else None
            delta = None
            if n is not None and base_n is not None and c != baseline:
                delta = n.native_heur_lp_iters - base_n.native_heur_lp_iters
            print(f"{inst:<24} {c:<14} "
                  f"{format_int(n.rens if n else None, 6)} "
                  f"{format_int(n.rens_root if n else None, 9)} "
                  f"{format_int(n.rins if n else None, 6)} "
                  f"{format_int(n.rcfix if n else None, 6)} "
                  f"{format_int(n.native_heur_lp_iters if n else None, 11)} "
                  f"{format_int(n.native_total_lp_iters if n else None, 11)} "
                  f"{format_int(n.fpr_lp_lp_iters if n else None, 9)} "
                  f"{format_int(delta, 11, signed=True)}")

    print("\n#### Aggregate (median over instrumented instances)\n")
    agg_header = (f"{'Config':<14} {'#Instr':>7} {'RENS':>6} {'RENSroot':>9} "
                  f"{'RINS':>6} {'RCfix':>6} {'NatHeurLP':>11} {'NatTotLP':>11} "
                  f"{'OurLP':>9} {'RootRENSlost':>13}")
    print(agg_header)
    print("-" * len(agg_header))
    for c in configs:
        cols: dict[str, list[float]] = {k: [] for k in
                                        ("rens", "rens_root", "rins", "rcfix",
                                         "heur", "tot", "ours")}
        n_instr = 0
        lost = 0
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            n = r.native if r is not None else None
            if n is None:
                continue
            n_instr += 1
            cols["rens"].append(n.rens)
            cols["rens_root"].append(n.rens_root)
            cols["rins"].append(n.rins)
            cols["rcfix"].append(n.rcfix)
            cols["heur"].append(n.native_heur_lp_iters)
            cols["tot"].append(n.native_total_lp_iters)
            cols["ours"].append(n.fpr_lp_lp_iters)
            base_r = agg_results.get(baseline, {}).get(inst) if baseline else None
            base_n = base_r.native if base_r is not None else None
            if (c != baseline and base_n is not None
                    and base_n.rens_root > 0 and n.rens_root == 0):
                lost += 1
        print(f"{c:<14} {n_instr:>7} "
              f"{format_float(_median_or_none(cols['rens']), 6, 1)} "
              f"{format_float(_median_or_none(cols['rens_root']), 9, 1)} "
              f"{format_float(_median_or_none(cols['rins']), 6, 1)} "
              f"{format_float(_median_or_none(cols['rcfix']), 6, 1)} "
              f"{format_float(_median_or_none(cols['heur']), 11, 1)} "
              f"{format_float(_median_or_none(cols['tot']), 11, 1)} "
              f"{format_float(_median_or_none(cols['ours']), 9, 1)} "
              f"{format_int(None if c == baseline else lost, 13)}")
    print("\nRootRENSlost = instances where the baseline called root RENS and "
          "this config did not.")


def _print_wall_clock_table(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
    baseline: str | None,
    verdicts: dict[tuple[str, str], CannibalizationVerdict],
) -> None:
    """Per instance x config: heuristic wall time, root-LP delay, class."""
    print(f"\n### Wall clock ({len(instances)} instances)\n")
    print("Heur_s is the sum of every [Heur] window (0.0 — not missing — for an "
          "instrumented run\nthat dispatched none); Dive_s is the fpr_lp part of "
          "it.  Span_s is the presolve chain's\nfull span, which includes the "
          "shared setup and accumulates across restarts, so it may\nexceed "
          "Troot_s on a restarting instance.\n")

    header = (f"{'Instance':<24} {'Config':<14} {'Heur_s':>8} {'Dive_s':>8} "
              f"{'HeurFrac':>9} {'Troot_s':>9} {'dTroot_s':>9} {'Span_s':>8} "
              f"{'Class':<16} Evidence")
    print(header)
    print("-" * len(header))

    for inst in instances:
        base_r = agg_results.get(baseline, {}).get(inst) if baseline else None
        t_base = base_r.time_to_root_lp if base_r is not None else None
        for c in configs:
            r = agg_results.get(c, {}).get(inst)
            t_root = r.time_to_root_lp if r is not None else None
            d_root = (t_root - t_base
                      if t_root is not None and t_base is not None and c != baseline
                      else None)
            v = verdicts[(inst, c)]
            print(f"{inst:<24} {c:<14} "
                  f"{format_float(heuristic_wall_seconds(r), 8, 2)} "
                  f"{format_float(heuristic_wall_seconds(r, 'dive'), 8, 2)} "
                  f"{format_float(r.heuristic_wall_fraction if r else None, 9, 4)} "
                  f"{format_float(t_root, 9, 2)} "
                  f"{format_float(d_root, 9, 2)} "
                  f"{format_float(presolve_span_seconds(r), 8, 2)} "
                  f"{v.category:<16} {'; '.join(v.evidence)}")

    print("\n#### Aggregate (SGM shift=1 for seconds, median for HeurFrac)\n")
    agg_header = (f"{'Config':<14} {'#Instr':>7} {'Heur_s':>8} {'Dive_s':>8} "
                  f"{'HeurFrac':>9} {'Troot_s':>9} {'Span_s':>8}")
    print(agg_header)
    print("-" * len(agg_header))
    for c in configs:
        heur: list[float] = []
        dive: list[float] = []
        fracs: list[float] = []
        troot: list[float] = []
        span: list[float] = []
        n_instr = 0
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if not is_instrumented(r):
                continue
            n_instr += 1
            # heuristic_wall_seconds is 0.0 rather than None here, so the
            # baseline config keeps a real row instead of an empty one.
            h = heuristic_wall_seconds(r)
            if h is not None:
                heur.append(h)
            d = heuristic_wall_seconds(r, "dive")
            if d is not None:
                dive.append(d)
            f = r.heuristic_wall_fraction if r is not None else None
            if f is not None:
                fracs.append(f)
            t = r.time_to_root_lp if r is not None else None
            if t is not None:
                troot.append(t)
            s = presolve_span_seconds(r)
            if s is not None:
                span.append(s)
        print(f"{c:<14} {n_instr:>7} "
              f"{format_float(shifted_geomean(heur, 1.0) if heur else None, 8, 2)} "
              f"{format_float(shifted_geomean(dive, 1.0) if dive else None, 8, 2)} "
              f"{format_float(_median_or_none(fracs), 9, 4)} "
              f"{format_float(shifted_geomean(troot, 1.0) if troot else None, 9, 2)} "
              f"{format_float(shifted_geomean(span, 1.0) if span else None, 8, 2)}")


def _print_classification_counts(
    configs: list[str],
    instances: list[str],
    verdicts: dict[tuple[str, str], CannibalizationVerdict],
) -> None:
    """Per config, how many instances landed in each cannibalization category."""
    print("\n### Classification counts\n")
    header = f"{'Config':<14}" + "".join(f" {cat:>17}" for cat in CANNIBALIZATION_CATEGORIES)
    print(header)
    print("-" * len(header))
    for c in configs:
        counts = {cat: 0 for cat in CANNIBALIZATION_CATEGORIES}
        for inst in instances:
            counts[verdicts[(inst, c)].category] += 1
        print(f"{c:<14}" + "".join(f" {counts[cat]:>17}" for cat in CANNIBALIZATION_CATEGORIES))


def print_cannibalization_tables(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    baseline_config: str | None = None,
) -> None:
    """Print the internal-budget and wall-clock cannibalization tables.

    Reads only the issue-#95 records already parsed out of the logs — no
    solves.  A results tree predating that instrumentation carries none of
    them, and this reports itself unavailable rather than printing zeros.
    """
    instances = get_common_instances(results, configs)
    print("\n## Cannibalization\n")

    if not instances:
        print("(no common instances across the requested configs)")
        return

    any_instrumented = any(
        is_instrumented(agg_results.get(c, {}).get(inst))
        for c in configs for inst in instances
    )
    if not any_instrumented:
        print("(not instrumented: no [Native] / [Heur] / [Root] lines in these "
              "logs.\n These come from issue #95 and need log_dev_level=3 on a "
              "patched binary;\n a results tree recorded before that carries "
              "none of them.)")
        return

    requested = baseline_config
    baseline = pick_baseline_config(agg_results, configs, baseline_config)
    if requested is not None and baseline is None:
        print(f"(requested baseline config '{requested}' is not in this results "
              "tree — rows cannot be compared)")
    elif baseline is None:
        print("(no vanilla-equivalent baseline config found — pass "
              "--cannibalization-baseline NAME.\n Rows are reported but not "
              "classified.)")
    else:
        print(f"Baseline config: {baseline}")
        if not any(is_instrumented(agg_results.get(baseline, {}).get(i)) for i in instances):
            print(f"WARNING: baseline config '{baseline}' carries no "
                  "instrumentation.  An externally built\n         unpatched "
                  "binary emits none; the comparison needs the patched binary "
                  "at\n         mip_heuristic_suite=off instead.")

    verdicts: dict[tuple[str, str], CannibalizationVerdict] = {}
    for inst in instances:
        base_r = agg_results.get(baseline, {}).get(inst) if baseline else None
        for c in configs:
            verdicts[(inst, c)] = classify_cannibalization(
                agg_results.get(c, {}).get(inst), base_r, is_baseline=(c == baseline)
            )

    _print_internal_budget_table(agg_results, configs, instances, baseline)
    _print_wall_clock_table(agg_results, configs, instances, baseline, verdicts)
    _print_classification_counts(configs, instances, verdicts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HiGHS benchmark results")
    parser.add_argument("results_dir", help="Directory with config subdirectories of log files")
    parser.add_argument("--configs", nargs="+", default=["patched", "vanilla"],
                        help="Configs to compare (default: patched vanilla). An entry of the "
                             "form NAME=DIR loads that config from an explicit directory instead "
                             "of results_dir/NAME (used to pull ablation anchors from "
                             "bench/results/plato).")
    parser.add_argument("--plot", default=None, help="Path to save survival plot (e.g., bench/survival.png)")
    parser.add_argument("--gap-threshold", type=float, default=0.01,
                        help="Gap threshold for survival plot (default: 0.01 = 1%%)")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="Time limit used in the benchmark (for gap@cutoff metric)")
    parser.add_argument("--solu", default=os.path.join(os.path.dirname(__file__),
                                                       "miplib2017-v22.solu"),
                        help="MIPLIB .solu file with reference objectives")
    parser.add_argument(
        "--baseline", action="store_true",
        help=(
            "Print PLATO headline metrics: primal-integral SGM (shift=0.001, "
            "matching Mittelmann's published methodology) and feasibility counts. "
            "Appended after the standard paper metrics table."
        ),
    )
    parser.add_argument(
        "--summary", action="store_true",
        help=(
            "Print only the SGM/paper-metrics summary and PLATO headline; "
            "skip the per-instance comparison table. Implies --baseline."
        ),
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help=(
            "Print a one-row-per-config ablation table (every config on its own "
            "row) instead of the pairwise comparison/paper-metrics tables. Use "
            "for the per-component ablation across many configs."
        ),
    )
    parser.add_argument(
        "--attribution", action="store_true",
        help=(
            "Print the per-config heuristic attribution: which heuristic found "
            "the first feasible incumbent and which held the best at termination "
            "(from the recorded incumbent source chars; no solves)."
        ),
    )
    parser.add_argument(
        "--cannibalization", action="store_true",
        help=(
            "Print the internal-budget and wall-clock cannibalization tables "
            "(epic #88): HiGHS's own RENS/RINS/root-reduced-cost calls and its "
            "share of the shared LP-iteration counters, the heuristic wall-time "
            "and root-LP delay, and a per-instance classification. Needs logs "
            "recorded at log_dev_level=3 on a patched binary (issue #95)."
        ),
    )
    parser.add_argument(
        "--cannibalization-baseline", default=None, metavar="CONFIG",
        help=(
            "Config every other row is compared against in --cannibalization "
            "(default: auto-detect the vanilla-equivalent config — instrumented "
            "and dispatching no custom heuristic)."
        ),
    )
    parser.add_argument(
        "--latex", default=None, metavar="PATH",
        help="With --ablation, also write the ablation table as LaTeX to PATH.",
    )
    args = parser.parse_args()

    # Parse NAME=DIR config overrides so ablation anchors can be loaded from a
    # different results directory than the positional results_dir.
    config_dirs: dict[str, str] = {}
    config_names: list[str] = []
    for entry in args.configs:
        if "=" in entry:
            name, path = entry.split("=", 1)
            name, path = name.strip(), path.strip()
            config_dirs[name] = path
            config_names.append(name)
        else:
            config_names.append(entry)

    results = load_results(args.results_dir, config_names, config_dirs)
    if not results:
        print("No results found", file=sys.stderr)
        sys.exit(1)

    active_configs = [c for c in config_names if c in results]
    agg_results = aggregate_results(results, active_configs)

    solu_refs: dict[str, tuple[str, float | None]] = {}
    if args.solu and os.path.exists(args.solu):
        solu_refs = parse_solu_file(args.solu)

    common = get_common_instances(results, active_configs)
    best_known = build_best_known(results, active_configs, common, solu_refs)

    if args.ablation:
        # Per-component ablation: one row per config, plus optional attribution.
        print_ablation_summary(results, agg_results, active_configs, args.time_limit,
                               best_known, latex_path=args.latex)
        if args.attribution:
            print_attribution(results, agg_results, active_configs)
        if args.cannibalization:
            print_cannibalization_tables(results, agg_results, active_configs,
                                         args.cannibalization_baseline)
        if args.plot:
            generate_survival_plot(agg_results, active_configs, args.plot, args.gap_threshold)
        return

    if not args.summary:
        print_comparison_table(agg_results, active_configs, best_known=best_known, time_limit=args.time_limit)
    print_paper_metrics(results, agg_results, active_configs, args.time_limit,
                        best_known=best_known)

    if args.attribution:
        print_attribution(results, agg_results, active_configs)

    if args.baseline or args.summary:
        print_plato_summary(results, agg_results, active_configs, args.time_limit, best_known)

    if args.cannibalization:
        print_cannibalization_tables(results, agg_results, active_configs,
                                     args.cannibalization_baseline)

    if args.plot:
        generate_survival_plot(agg_results, active_configs, args.plot, args.gap_threshold)


if __name__ == "__main__":
    main()
