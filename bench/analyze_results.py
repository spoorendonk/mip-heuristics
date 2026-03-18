#!/usr/bin/env python3
"""Analyze benchmark results: compute metrics, generate tables and plots."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_highs_log import SolveResult, parse_log_file


def load_results(results_dir: str, configs: list[str]) -> dict[str, dict[str, SolveResult]]:
    """Load all parsed results. Returns {config: {instance: SolveResult}}."""
    results: dict[str, dict[str, SolveResult]] = {}
    for config in configs:
        config_dir = os.path.join(results_dir, config)
        if not os.path.isdir(config_dir):
            print(f"Warning: config directory not found: {config_dir}", file=sys.stderr)
            continue
        results[config] = {}
        for log_file in sorted(Path(config_dir).glob("*.log")):
            name = log_file.stem
            results[config][name] = parse_log_file(str(log_file))
    return results


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


def print_comparison_table(
    results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_cutoffs: list[float] | None = None,
) -> None:
    """Print per-instance comparison table."""
    if time_cutoffs is None:
        time_cutoffs = [10.0, 60.0, 600.0]

    if len(configs) < 2:
        print("Need at least 2 configs for comparison")
        return

    c1, c2 = configs[0], configs[1]
    instances = sorted(set(results.get(c1, {}).keys()) & set(results.get(c2, {}).keys()))

    if not instances:
        print("No common instances found between configs")
        return

    # Determine which time cutoffs are relevant (at least one instance has solve_time >= cutoff,
    # or has incumbents, meaning gap@T is meaningful)
    max_solve_time = max(
        max((r.solve_time for r in results[c1].values()), default=0),
        max((r.solve_time for r in results[c2].values()), default=0),
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
        r1, r2 = results[c1][inst], results[c2][inst]

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
        for tc in active_cutoffs:
            g1 = r1.primal_gap_at(tc)
            g2 = r2.primal_gap_at(tc)
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


def generate_survival_plot(
    results: dict[str, dict[str, SolveResult]],
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

    instances = sorted(set.intersection(*(set(results[c].keys()) for c in configs)))
    if not instances:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for config in configs:
        # For each instance, find time when gap <= threshold
        solve_times = []
        for inst in instances:
            r = results[config][inst]
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
        max_time = max(r.solve_time for r in results[config].values())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HiGHS benchmark results")
    parser.add_argument("results_dir", help="Directory with config subdirectories of log files")
    parser.add_argument("--configs", nargs="+", default=["patched", "vanilla"],
                        help="Configs to compare (default: patched vanilla)")
    parser.add_argument("--plot", default=None, help="Path to save survival plot (e.g., bench/survival.png)")
    parser.add_argument("--gap-threshold", type=float, default=0.01,
                        help="Gap threshold for survival plot (default: 0.01 = 1%%)")
    args = parser.parse_args()

    results = load_results(args.results_dir, args.configs)
    if not results:
        print("No results found", file=sys.stderr)
        sys.exit(1)

    print_comparison_table(results, args.configs)

    if args.plot:
        generate_survival_plot(results, args.configs, args.plot, args.gap_threshold)


if __name__ == "__main__":
    main()
