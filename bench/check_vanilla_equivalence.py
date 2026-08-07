#!/usr/bin/env python3
"""Verify that the patched binary at `mip_heuristic_suite=off` matches vanilla.

`suite=off` is the patch-overhead row of the closeout benchmark matrix: it
is only meaningful if it behaves exactly like an unpatched HiGHS of the same
tag.  This script proves that instead of assuming it, by running both
binaries on the same instances and comparing what HiGHS itself reports.

It is deliberately *not* a ctest: it needs an unpatched HiGHS binary at the
pinned tag, which no build tree produces (the patch rewrites the source in
place) and no CI runner is guaranteed to have.  Point `--vanilla-binary` at
a system install of the same version — `highs --version` on both must agree
before any comparison means anything, and the script checks that first.

Hard gates, per instance and seed:

  * identical solver status and primal bound;
  * identical node count;
  * identical total and heuristic LP iterations.  The heuristic count is
    what `moreHeuristicsAllowed()` reads, so equality there is the proof
    that the patch does not cannibalize the RENS/RINS budget on the `off`
    path.  HiGHS does not print RENS/RINS *invocation* counts, but it
    cannot invoke them differently while leaving node count, total LP
    iterations and the solution-source display lines all identical;
  * empty normalized log diff — everything HiGHS printed, minus the
    wall-clock columns and the patch's own marker line.

Solve time is reported and compared against `--time-tolerance` as a soft
signal; it is noise-dominated on instances this size and never fails the
run on its own unless `--strict-time` is passed.

Two differences are known and accepted rather than fixed:

  * the `mip-heuristics patch active` marker line, which is the only way to
    tell a patched binary from an unpatched one (the version and githash
    banners are identical).  Normalized away;
  * one `heuristic_effort_used += fj_last_effort` store per FJ callback
    inside stock `feasibilityJump()`.  No control-flow change; invisible in
    the log.

Usage:
  python bench/check_vanilla_equivalence.py --vanilla-binary /usr/local/bin/highs
  python bench/check_vanilla_equivalence.py --vanilla-binary ... --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

# Same instance set as bench/correctness_check.py; they ship with HiGHS.
INSTANCES = [
    "flugpl.mps",
    "egout.mps",
    "bell5.mps",
    "lseu.mps",
    "gt2.mps",
    "p0548.mps",
]

# The one option that puts the patched binary on the vanilla path.
VANILLA_EQUIVALENT_OPTIONS = {"mip_heuristic_suite": "off"}

# Lines dropped before diffing: wall-clock measurements, the patch's
# self-identification marker, and HiGHS's echo of the options it was given
# (the patched run is handed `mip_heuristic_suite=off` and the vanilla run
# has no such option, so that one line always differs by construction —
# it says nothing about solver behaviour).
_VOLATILE_LINE = re.compile(
    r"^\s*(?:"
    r"Timing\b"                       # "Timing            0.09"
    r"|P-D integral\b"                # objective integral, time-weighted
    r"|mip-heuristics patch active"   # the accepted marker
    r"|Set option\b"                  # options-file echo
    r"|[\d.]+\s*\((?:Presolve|Solve|Postsolve)\)\s*$"  # Timing continuation lines
    r")"
)

# A MIP display line, identified by its trailing elapsed-time column
# ("... 338     0.1s").  Everything left of that column is work done, not
# time spent, so only the last token is stripped.
_DISPLAY_LINE = re.compile(r"^.*\s[\d.]+s$")

# Substitutions applied to surviving lines.  Both blank out a value while
# keeping the line itself in the diff, because the rest of the line carries
# signal the comparison wants:
#   * the profiling block's `time [calls] = 0.02 [27]` — the call count is
#     exactly the kind of thing this check exists to compare, so only the
#     seconds are masked;
#   * the banner's git hash, whose *length* depends on how the tree was
#     fetched (shallow clone vs FetchContent) rather than on the commit.
#     `same_build` has already established the two are the same commit.
_MASKS = (
    (re.compile(r"(time \[calls\] = )[\d.]+"), r"\1<t>"),
    (re.compile(r"(git hash: )\w+"), r"\1<hash>"),
)


@dataclass
class Metrics:
    status: str | None = None
    primal_bound: str | None = None
    nodes: str | None = None
    lp_iterations: str | None = None
    heuristic_lp_iterations: str | None = None
    time_s: float = 0.0

    def comparable(self) -> dict[str, str | None]:
        """The fields that must match exactly (time is a soft signal)."""
        return {
            "status": self.status,
            "primal bound": self.primal_bound,
            "nodes": self.nodes,
            "LP iterations": self.lp_iterations,
            "heuristic LP iterations": self.heuristic_lp_iterations,
        }


def parse_metrics(output: str) -> Metrics:
    """Pull the solving-report fields out of a HiGHS CLI log.

    The LP-iteration block is positional: a bare `<n> (heuristics)` line
    follows the `LP iterations <n> (total)` line, so the suffix is what
    identifies it rather than any label of its own.
    """
    m = Metrics()
    for line in output.splitlines():
        s = line.strip()
        if match := re.fullmatch(r"Status\s+(\S.*)", s):
            m.status = match.group(1)
        elif match := re.fullmatch(r"Primal bound\s+(\S+)", s):
            m.primal_bound = match.group(1)
        elif match := re.fullmatch(r"Nodes\s+(\d+)", s):
            # Anchored on the digit: HiGHS also prints a display-table
            # header starting with "Nodes".
            m.nodes = match.group(1)
        elif match := re.fullmatch(r"LP iterations\s+(\d+)", s):
            m.lp_iterations = match.group(1)
        elif match := re.fullmatch(r"(\d+)\s+\(heuristics\)", s):
            m.heuristic_lp_iterations = match.group(1)
        elif match := re.fullmatch(r"Timing\s+([\d.]+)", s):
            m.time_s = float(match.group(1))
    return m


def normalize_log(output: str) -> list[str]:
    """Strip wall-clock content so two runs of identical work compare equal."""
    lines = []
    for line in output.splitlines():
        if _VOLATILE_LINE.match(line):
            continue
        if _DISPLAY_LINE.match(line):
            # Drop the trailing time column; everything left of it is work.
            line = line.rsplit(None, 1)[0]
        for pattern, replacement in _MASKS:
            line = pattern.sub(replacement, line)
        lines.append(line.rstrip())
    return lines


@dataclass
class Comparison:
    instance: str
    seed: int
    failures: list[str] = field(default_factory=list)
    patched_time: float = 0.0
    vanilla_time: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.failures


def compare_runs(instance: str, seed: int, patched: str, vanilla: str,
                 time_tolerance: float, strict_time: bool) -> Comparison:
    """Compare two raw HiGHS logs.  Pure — the tests drive this directly."""
    pm, vm = parse_metrics(patched), parse_metrics(vanilla)
    cmp = Comparison(instance, seed, patched_time=pm.time_s, vanilla_time=vm.time_s)

    vanilla_fields = vm.comparable()
    for name, patched_value in pm.comparable().items():
        vanilla_value = vanilla_fields[name]
        if patched_value != vanilla_value:
            cmp.failures.append(f"{name}: patched={patched_value!r} vanilla={vanilla_value!r}")

    diff = list(difflib.unified_diff(normalize_log(vanilla), normalize_log(patched),
                                     fromfile="vanilla", tofile="patched", lineterm="", n=1))
    if diff:
        cmp.failures.append("normalized log differs:\n    " + "\n    ".join(diff[:40]))

    slower = pm.time_s > max(vm.time_s, 0.01) * time_tolerance
    if slower and strict_time:
        cmp.failures.append(f"solve time {pm.time_s:.2f}s vs vanilla {vm.time_s:.2f}s "
                            f"exceeds {time_tolerance}x")
    return cmp


def find_instances_dir(binary: str) -> str | None:
    """Locate check/instances/ relative to the build tree holding `binary`."""
    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(binary)))
    for c in (os.path.join(build_dir, "_deps", "highs-src", "check", "instances"),
              os.path.join(build_dir, "_deps", "highs-src", "highs", "check", "instances")):
        if os.path.isdir(c):
            return c
    return None


def version_of(binary: str) -> tuple[str, str]:
    """`(version, githash)` from `--version`, e.g. ("1.15.1", "04024d701f")."""
    out = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout
    version = re.search(r"HiGHS version (\S+)", out)
    githash = re.search(r"Githash (\w+)", out)
    return (version.group(1) if version else "?", githash.group(1) if githash else "")


def same_build(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """Whether two `version_of` results identify the same HiGHS commit.

    The githash is compared by prefix: how many characters HiGHS prints
    depends on how the tree was fetched (a shallow `git clone` and a
    FetchContent checkout of the same tag report different lengths), so
    requiring equality would reject a legitimate pairing.
    """
    if a[0] != b[0]:
        return False
    short, long = sorted((a[1], b[1]), key=len)
    return bool(short) and long.startswith(short)


def run_solve(binary: str, instance_path: str, options: dict[str, str], seed: int,
              time_limit: float, tmp_dir: str) -> str:
    opts_path = os.path.join(tmp_dir, "run.opts")
    with open(opts_path, "w") as f:
        for k, v in {**options, "random_seed": str(seed)}.items():
            f.write(f"{k} = {v}\n")
    cmd = [binary, instance_path, "--time_limit", str(time_limit), "--options_file", opts_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit * 2 + 30)
    return r.stdout + "\n" + r.stderr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare patched suite=off against an unpatched HiGHS binary")
    parser.add_argument("--patched-binary", default="./build/bin/highs")
    parser.add_argument("--vanilla-binary", required=True,
                        help="Unpatched HiGHS of the SAME version (e.g. /usr/local/bin/highs)")
    parser.add_argument("--instances-dir", default=None,
                        help="Path to check/instances/ (auto-detected from the patched binary)")
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--time-tolerance", type=float, default=1.5,
                        help="Patched/vanilla solve-time ratio worth reporting (default 1.5)")
    parser.add_argument("--strict-time", action="store_true",
                        help="Fail the run when the time ratio is exceeded, not just report it")
    args = parser.parse_args()

    patched = os.path.abspath(args.patched_binary)
    vanilla = os.path.abspath(args.vanilla_binary)
    for b in (patched, vanilla):
        if not os.path.exists(b):
            sys.exit(f"Error: binary not found: {b}")

    pv, vv = version_of(patched), version_of(vanilla)
    if not same_build(pv, vv):
        sys.exit(f"Error: version mismatch, comparison would be meaningless.\n"
                 f"  patched: {pv[0]} githash {pv[1]}\n  vanilla: {vv[0]} githash {vv[1]}")

    instances_dir = args.instances_dir or find_instances_dir(patched)
    if not instances_dir or not os.path.isdir(instances_dir):
        sys.exit("Error: instances dir not found. Pass --instances-dir explicitly.")

    tmp_dir = tempfile.mkdtemp(prefix="vanilla_equivalence_")
    comparisons: list[Comparison] = []

    for name in INSTANCES:
        path = os.path.join(instances_dir, name)
        if not os.path.exists(path):
            print(f"Warning: {name} not found in {instances_dir}, skipping", file=sys.stderr)
            continue
        for seed in args.seeds:
            p_log = run_solve(patched, path, VANILLA_EQUIVALENT_OPTIONS, seed,
                              args.time_limit, tmp_dir)
            v_log = run_solve(vanilla, path, {}, seed, args.time_limit, tmp_dir)
            c = compare_runs(name, seed, p_log, v_log, args.time_tolerance, args.strict_time)
            comparisons.append(c)
            mark = "PASS" if c.passed else "FAIL"
            print(f"  {mark}  {name:14s} seed={seed}  "
                  f"patched={c.patched_time:.2f}s vanilla={c.vanilla_time:.2f}s")

    failed = [c for c in comparisons if not c.passed]
    print(f"\n{len(comparisons) - len(failed)}/{len(comparisons)} equivalent")
    for c in failed:
        print(f"\n{c.instance} seed={c.seed}:")
        for f in c.failures:
            print(f"  {f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
