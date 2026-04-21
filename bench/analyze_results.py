#!/usr/bin/env python3
"""Analyze benchmark results: compute metrics, generate tables and plots."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
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
    results_dir: str, configs: list[str]
) -> dict[str, dict[int, dict[str, SolveResult]]]:
    """Load all parsed results.

    Returns {config: {seed: {instance: SolveResult}}}.
    Supports both seed-aware (results/{config}/seed{N}/*.log) and
    legacy flat (results/{config}/*.log, treated as seed 0) layouts.
    """
    results: dict[str, dict[int, dict[str, SolveResult]]] = {}
    for config in configs:
        config_dir = os.path.join(results_dir, config)
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
    For incumbents, uses the seed with the median time_to_first_feasible.
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


def count_wins(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
) -> dict[str, int]:
    """Count #Win: instances where config finds the best primal bound.

    Returns {config: win_count}.
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
        for c, b in bounds.items():
            if abs(b - best) < 1e-6:
                wins[c] += 1
    return wins


def print_comparison_table(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_cutoffs: list[float] | None = None,
    best_known: dict[str, float | None] | None = None,
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

    for inst in instances:
        r1, r2 = agg_results[c1][inst], agg_results[c2][inst]

        print(f"{inst:<25} ", end="")

        # Time to first feasible
        t1 = r1.time_to_first_feasible
        t2 = r2.time_to_first_feasible
        print(f"{format_float(t1, 10, 2)} {format_float(t2, 10, 2)} ", end="")
        if t1 is not None and t2 is not None:
            t1st_vals[c1].append(t1)
            t1st_vals[c2].append(t2)
            if t1 < t2 - 0.01:
                wins["t1st"] += 1
            elif t2 < t1 - 0.01:
                losses["t1st"] += 1
            else:
                ties["t1st"] += 1

        # Gap at cutoffs
        ref = best_known.get(inst) if best_known else None
        for tc in active_cutoffs:
            g1 = r1.primal_gap_at(tc, ref)
            g2 = r2.primal_gap_at(tc, ref)
            print(f"{format_float(g1, 12, 6)} {format_float(g2, 12, 6)} ", end="")
            if g1 is not None and g2 is not None:
                gap_vals[tc][c1].append(g1)
                gap_vals[tc][c2].append(g2)
                if g1 < g2 - 1e-6:
                    wins["gap"][tc] += 1
                elif g2 < g1 - 1e-6:
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
                  f"{format_float(shifted_geomean(gap_vals[tc][c1], 0.01), 12, 6)} "
                  f"{format_float(shifted_geomean(gap_vals[tc][c2], 0.01), 12, 6)}")

    print(f"{'P-D integral':<25} {wins['pd']:<12} {losses['pd']:<12} {ties['pd']:<8} "
          f"{format_float(shifted_geomean(pd_vals[c1], 1.0), 12, 4)} "
          f"{format_float(shifted_geomean(pd_vals[c2], 1.0), 12, 4)}")


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

    # --- SGM of time-to-first-feasible (shift=1s, matching FJ/FPR) ---
    print(f"\n{'SGM T1st (s=1)':<25}", end="")
    for c in configs:
        t1st = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r and r.time_to_first_feasible is not None:
                t1st.append(r.time_to_first_feasible)
        print(f" {format_float(shifted_geomean(t1st, 1.0), 12, 4)}", end="")
    print()

    # --- SGM of primal gap at cutoff (shift=0.01, matching FPR/Scylla) ---
    print(f"{'SGM Gap@' + str(int(time_limit)) + 's (s=0.01)':<25}", end="")
    for c in configs:
        gaps = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r:
                ref = best_known.get(inst) if best_known else None
                g = r.primal_gap_at(time_limit, ref)
                if g is not None:
                    gaps.append(g)
        print(f" {format_float(shifted_geomean(gaps, 0.01), 12, 6)}", end="")
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
                if math.isfinite(pi):
                    pis.append(pi)
        print(f" {format_float(shifted_geomean(pis, 1.0), 12, 4)}", end="")
    print()

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


def print_effort_calibration(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    configs: list[str],
) -> None:
    """Print per-arm effort calibration metrics from [Portfolio] log lines.

    For each arm, computes median and IQR of effort_per_ms across all
    instances and seeds to assess whether effort units correlate with
    wall-clock time consistently across arms.
    """
    # Collect all effort samples across configs, seeds, and instances
    per_arm: dict[str, list[float]] = {}
    total_samples = 0
    for config in configs:
        for seed_data in results.get(config, {}).values():
            for solve_result in seed_data.values():
                for sample in solve_result.effort_samples:
                    per_arm.setdefault(sample.arm, []).append(sample.effort_per_ms)
                    total_samples += 1

    if not per_arm:
        return

    print(f"\n## Effort Calibration ({total_samples} samples)\n")
    print(f"{'Arm':<25} {'Samples':>8} {'Median e/ms':>12} {'P25 e/ms':>10} {'P75 e/ms':>10} {'Rel. med.':>10}")
    print("-" * 80)

    summaries: list[tuple[str, int, float, float, float]] = []
    for arm_name in sorted(per_arm.keys()):
        vals = sorted(per_arm[arm_name])
        n = len(vals)
        median = statistics.median(vals)
        if n >= 2:
            q1, _, q3 = statistics.quantiles(vals, n=4)
            p25, p75 = q1, q3
        else:
            p25, p75 = vals[0], vals[-1]
        summaries.append((arm_name, n, median, p25, p75))

    # Relative median: each arm's median normalized to the fastest arm (highest e/ms)
    max_median = max(s[2] for s in summaries) if summaries else 1.0

    for arm_name, n, median, p25, p75 in summaries:
        rel_median = median / max_median if max_median > 0 else 0.0
        print(f"{arm_name:<25} {n:>8} {median:>12.0f} {p25:>10.0f} {p75:>10.0f} {rel_median:>10.2f}")

    # Cross-arm calibration summary
    medians = [s[2] for s in summaries if s[2] > 0]
    if len(medians) >= 2:
        ratio = max(medians) / min(medians)
        print(f"\nMax/min median effort_per_ms ratio: {ratio:.1f}x")
        if ratio > 5.0:
            print("WARNING: >5x spread suggests effort units are poorly calibrated across arms.")
        else:
            print("Effort units appear reasonably calibrated across arms.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HiGHS benchmark results")
    parser.add_argument("results_dir", help="Directory with config subdirectories of log files")
    parser.add_argument("--configs", nargs="+", default=["patched", "vanilla"],
                        help="Configs to compare (default: patched vanilla)")
    parser.add_argument("--plot", default=None, help="Path to save survival plot (e.g., bench/survival.png)")
    parser.add_argument("--gap-threshold", type=float, default=0.01,
                        help="Gap threshold for survival plot (default: 0.01 = 1%%)")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="Time limit used in the benchmark (for gap@cutoff metric)")
    parser.add_argument("--solu", default=os.path.join(os.path.dirname(__file__),
                                                       "miplib2017-v22.solu"),
                        help="MIPLIB .solu file with reference objectives")
    args = parser.parse_args()

    results = load_results(args.results_dir, args.configs)
    if not results:
        print("No results found", file=sys.stderr)
        sys.exit(1)

    active_configs = [c for c in args.configs if c in results]
    agg_results = aggregate_results(results, active_configs)

    solu_refs: dict[str, tuple[str, float | None]] = {}
    if args.solu and os.path.exists(args.solu):
        solu_refs = parse_solu_file(args.solu)

    common = get_common_instances(results, active_configs)
    best_known = build_best_known(results, active_configs, common, solu_refs)

    print_comparison_table(agg_results, active_configs, best_known=best_known)
    print_paper_metrics(results, agg_results, active_configs, args.time_limit,
                        best_known=best_known)
    print_effort_calibration(results, active_configs)

    if args.plot:
        generate_survival_plot(agg_results, active_configs, args.plot, args.gap_threshold)


if __name__ == "__main__":
    main()
