#!/usr/bin/env python3
"""Correctness check across the 2x2 heuristic execution matrix.

Runs the HiGHS binary on small test instances (shipped with HiGHS in
check/instances/) for each of the four mode combinations:

  seq/det   — portfolio=false, opportunistic=false
  seq/opp   — portfolio=false, opportunistic=true
  port/det  — portfolio=true,  opportunistic=false
  port/opp  — portfolio=true,  opportunistic=true

Checks that each solve finds the known-optimal objective within a
tolerance.  Reports a per-instance x per-mode pass/fail table.

Usage:
  python bench/correctness_check.py                      # defaults
  python bench/correctness_check.py --binary ./build/bin/highs
  python bench/correctness_check.py --seeds 0 1 2        # multi-seed
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

# Instances that ship with HiGHS check/instances/ and their known optima.
INSTANCES: list[tuple[str, float, float]] = [
    # (name.mps, known_optimal, relative_tolerance)
    ("flugpl.mps", 1201500.0, 1e-6),
    ("egout.mps", 568.1007, 1e-3),
    ("bell5.mps", 8966406.49, 1e-2),
    ("lseu.mps", 1120.0, 1e-3),
    ("gt2.mps", 21166.0, 1e-3),
    ("p0548.mps", 8691.0, 1e-3),
]

MODES = {
    "seq/det": {"mip_heuristic_portfolio": "false", "mip_heuristic_opportunistic": "false"},
    "seq/opp": {"mip_heuristic_portfolio": "false", "mip_heuristic_opportunistic": "true"},
    "port/det": {"mip_heuristic_portfolio": "true", "mip_heuristic_opportunistic": "false"},
    "port/opp": {"mip_heuristic_portfolio": "true", "mip_heuristic_opportunistic": "true"},
}


@dataclass
class RunResult:
    mode: str
    instance: str
    seed: int
    objective: float | None
    status: str
    time_s: float
    passed: bool
    error: str | None = None


def find_instances_dir(binary: str) -> str | None:
    """Try to locate check/instances/ relative to the build tree."""
    # The binary is typically at build/bin/highs.
    # HiGHS source is at build/_deps/highs-src/check/instances/
    # or build/_deps/highs-src/highs/check/instances/ (v1.13+ layout).
    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(binary)))
    candidates = [
        os.path.join(build_dir, "_deps", "highs-src", "check", "instances"),
        os.path.join(build_dir, "_deps", "highs-src", "highs", "check", "instances"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def write_options_file(options: dict[str, str], path: str) -> None:
    """Write a HiGHS options file."""
    with open(path, "w") as f:
        for k, v in options.items():
            f.write(f"{k} = {v}\n")


def run_solve(
    binary: str,
    instance_path: str,
    mode_opts: dict[str, str],
    seed: int,
    time_limit: float,
    tmp_dir: str,
) -> tuple[float | None, str, float]:
    """Run HiGHS and extract objective, status, and solve time.

    Returns (objective_or_None, status_string, solve_time).
    Custom heuristic options are passed via an options file (HiGHS
    doesn't expose them as CLI flags).
    """
    all_opts = {**mode_opts, "random_seed": str(seed)}
    opts_path = os.path.join(tmp_dir, "run.opts")
    write_options_file(all_opts, opts_path)

    cmd = [binary, instance_path, "--time_limit", str(time_limit), "--options_file", opts_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit * 2 + 30)
    output = result.stdout + "\n" + result.stderr

    # Parse objective from "Primal bound" in the solving report
    obj = None
    status = "UNKNOWN"
    time_s = 0.0

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Status"):
            status = stripped.split(None, 1)[-1] if len(stripped.split(None, 1)) > 1 else status
        if stripped.startswith("Primal bound"):
            try:
                obj = float(stripped.split()[-1])
            except (ValueError, IndexError):
                pass
        if stripped.startswith("Timing"):
            try:
                time_s = float(stripped.split()[-1])
            except (ValueError, IndexError):
                pass

    return obj, status, time_s


def check_objective(obj: float | None, known_opt: float, tol: float) -> bool:
    """Check if objective is within relative tolerance of known optimum."""
    if obj is None:
        return False
    denom = max(abs(known_opt), 1.0)
    return abs(obj - known_opt) / denom <= tol


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness check across 2x2 mode matrix")
    parser.add_argument("--binary", default="./build/bin/highs", help="Path to HiGHS binary")
    parser.add_argument("--instances-dir", default=None,
                        help="Path to check/instances/ dir (auto-detected from binary path)")
    parser.add_argument("--time-limit", type=float, default=30,
                        help="Time limit per solve (seconds, default 30)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Random seeds to run (default: 0)")
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()),
                        choices=list(MODES.keys()),
                        help="Modes to test (default: all four)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-solve details")
    args = parser.parse_args()

    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        print(f"Error: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    instances_dir = args.instances_dir or find_instances_dir(binary)
    if instances_dir is None or not os.path.isdir(instances_dir):
        print(f"Error: instances dir not found. Pass --instances-dir explicitly.", file=sys.stderr)
        sys.exit(1)

    # Verify all instances exist
    available = []
    for name, opt, tol in INSTANCES:
        path = os.path.join(instances_dir, name)
        if os.path.exists(path):
            available.append((name, opt, tol, path))
        else:
            print(f"Warning: {name} not found in {instances_dir}, skipping", file=sys.stderr)
    if not available:
        print("Error: no instances found", file=sys.stderr)
        sys.exit(1)

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="correctness_check_")

    results: list[RunResult] = []
    total = len(args.modes) * len(args.seeds) * len(available)
    done = 0

    for mode in args.modes:
        mode_opts = MODES[mode]
        for seed in args.seeds:
            for name, opt, tol, path in available:
                done += 1
                try:
                    obj, status, time_s = run_solve(
                        binary, path, mode_opts, seed, args.time_limit, tmp_dir)
                    passed = check_objective(obj, opt, tol)
                    r = RunResult(mode, name, seed, obj, status, time_s, passed)
                except subprocess.TimeoutExpired:
                    r = RunResult(mode, name, seed, None, "TIMEOUT", args.time_limit, False,
                                  error="process timeout")
                except Exception as e:
                    r = RunResult(mode, name, seed, None, "ERROR", 0.0, False, error=str(e))

                results.append(r)
                if args.verbose:
                    mark = "PASS" if r.passed else "FAIL"
                    obj_str = f"{r.objective:.4f}" if r.objective is not None else "N/A"
                    print(f"  [{done}/{total}] {mark}  {mode:10s}  seed={seed}  "
                          f"{name:20s}  obj={obj_str:>14s}  ref={opt:.4f}  {r.time_s:.1f}s")

    # Summary table
    print()
    print("=" * 80)
    print("CORRECTNESS CHECK SUMMARY")
    print("=" * 80)

    # Header: instance names
    inst_names = [name for name, _, _, _ in available]
    col_w = max(len(n) for n in inst_names) + 2
    header = f"{'Mode':>10s} {'seed':>4s}"
    for name in inst_names:
        header += f"  {name:>{col_w}s}"
    print(header)
    print("-" * len(header))

    pass_count = 0
    fail_count = 0

    for mode in args.modes:
        for seed in args.seeds:
            row = f"{mode:>10s} {seed:>4d}"
            for name in inst_names:
                matching = [r for r in results if r.mode == mode and r.seed == seed
                            and r.instance == name]
                if matching:
                    r = matching[0]
                    if r.passed:
                        row += f"  {'PASS':>{col_w}s}"
                        pass_count += 1
                    else:
                        obj_str = f"{r.objective:.0f}" if r.objective is not None else "N/A"
                        row += f"  {obj_str:>{col_w}s}"
                        fail_count += 1
                else:
                    row += f"  {'???':>{col_w}s}"
            print(row)

    print("-" * len(header))
    total_checks = pass_count + fail_count
    print(f"\n{pass_count}/{total_checks} passed, {fail_count} failed")

    if fail_count > 0:
        print("\nFailed runs:")
        for r in results:
            if not r.passed:
                obj_str = f"{r.objective:.4f}" if r.objective is not None else "N/A"
                ref = next(opt for name, opt, _, _ in available if name == r.instance)
                print(f"  {r.mode:10s} seed={r.seed} {r.instance:20s} "
                      f"got={obj_str} expected={ref:.4f} status={r.status}")
                if r.error:
                    print(f"    error: {r.error}")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
