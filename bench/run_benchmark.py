#!/usr/bin/env python3
"""Run patched vs vanilla HiGHS on MIPLIB instances."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


# Default vanilla options: disable all custom heuristics
VANILLA_OPTIONS = {
    "mip_heuristic_portfolio": "false",
    "mip_heuristic_run_fpr": "false",
    "mip_heuristic_run_local_mip": "false",
    "mip_heuristic_run_scylla_fpr": "false",
}


def load_instances(path: str) -> list[str]:
    """Load instance names from a file (one per line, # comments)."""
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                instances.append(line)
    return instances


def find_instance_file(name: str, data_dir: str) -> str | None:
    """Find instance file, trying .mps.gz and .mps extensions."""
    for ext in [".mps.gz", ".mps"]:
        p = os.path.join(data_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def write_options_file(options: dict[str, str], path: str) -> None:
    """Write a HiGHS options file."""
    with open(path, "w") as f:
        for k, v in options.items():
            f.write(f"{k} = {v}\n")


def run_single(
    binary: str,
    instance_file: str,
    instance_name: str,
    config: str,
    time_limit: float,
    output_dir: str,
    extra_options: dict[str, str] | None = None,
) -> tuple[str, str, bool]:
    """Run HiGHS on a single instance with given config.

    Returns (instance_name, config, success).
    """
    config_dir = os.path.join(output_dir, config)
    os.makedirs(config_dir, exist_ok=True)
    log_path = os.path.join(config_dir, f"{instance_name}.log")

    cmd = [binary, instance_file, "--time_limit", str(time_limit)]

    if extra_options:
        opts_path = os.path.join(config_dir, f"{instance_name}.opts")
        write_options_file(extra_options, opts_path)
        cmd.extend(["--options_file", opts_path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit * 1.5 + 120,  # generous timeout beyond HiGHS limit
        )
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        with open(log_path, "w") as f:
            f.write(output)
        return (instance_name, config, True)
    except subprocess.TimeoutExpired:
        with open(log_path, "w") as f:
            f.write(f"TIMEOUT: process killed after {time_limit * 1.5 + 120}s\n")
        return (instance_name, config, False)
    except Exception as e:
        with open(log_path, "w") as f:
            f.write(f"ERROR: {e}\n")
        return (instance_name, config, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patched vs vanilla HiGHS benchmark")
    parser.add_argument("--instances", required=True, help="File with instance names")
    parser.add_argument("--binary", default="./build/bin/highs", help="Path to HiGHS binary")
    parser.add_argument("--data-dir", default="/tmp/miplib", help="Directory with .mps.gz files")
    parser.add_argument("--time-limit", type=float, default=60, help="Time limit per instance (seconds)")
    parser.add_argument("--output", default="bench/results", help="Output directory for logs")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Parallel jobs (per config, not across configs)")
    parser.add_argument("--configs", nargs="+", default=["patched", "vanilla"],
                        help="Configs to run (default: patched vanilla)")
    args = parser.parse_args()

    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        print(f"Error: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    instances = load_instances(args.instances)
    print(f"Loaded {len(instances)} instances from {args.instances}")

    # Check all instances exist
    missing = []
    instance_files = {}
    for name in instances:
        f = find_instance_file(name, args.data_dir)
        if f is None:
            missing.append(name)
        else:
            instance_files[name] = f
    if missing:
        print(f"Warning: {len(missing)} instances not found in {args.data_dir}:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        instances = [n for n in instances if n in instance_files]

    os.makedirs(args.output, exist_ok=True)

    # Run each config sequentially (to avoid resource contention),
    # but parallelise instances within each config
    for config in args.configs:
        extra_opts = VANILLA_OPTIONS if config == "vanilla" else None
        print(f"\n{'='*60}")
        print(f"Running config: {config} ({len(instances)} instances, {args.time_limit}s limit)")
        print(f"{'='*60}")

        futures = {}
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for name in instances:
                fut = executor.submit(
                    run_single,
                    binary,
                    instance_files[name],
                    name,
                    config,
                    args.time_limit,
                    args.output,
                    extra_opts,
                )
                futures[fut] = name

            done = 0
            for fut in as_completed(futures):
                name, cfg, success = fut.result()
                done += 1
                status = "OK" if success else "FAIL"
                print(f"  [{done}/{len(instances)}] {name}: {status}")

    print(f"\nResults written to {args.output}/")
    print(f"Run: python bench/analyze_results.py {args.output}")


if __name__ == "__main__":
    main()
