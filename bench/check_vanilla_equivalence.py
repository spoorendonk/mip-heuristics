#!/usr/bin/env python3
"""Verify that injecting our heuristics does not perturb HiGHS itself.

The configuration compared is the **pure patch-overhead** one: FeasibilityJump
disabled on both sides (`mip_heuristic_run_feasibility_jump=false`, upstream's
own option), plus `mip_heuristic_suite=off` on the patched side so none of our
four presolve heuristics and no `fpr_lp` runs.  What that proves is what is
worth proving: the patch's call sites, its extra option records and its marker
line do not move HiGHS's presolve, its B&B or its LP path by a single node or
iteration.

It does **not** prove that `suite=off` equals vanilla, and it is not a licence
to use `off` as a baseline.  `off` is the "our four presolve heuristics
disabled" ablation on the patched binary; it hands HiGHS's own standalone
FeasibilityJump call site back, but that call site drives *our* copy of
FeasibilityJump, which the patch modifies (the per-bump `kVerbose` log line is
gone, and #139 corrected two upstream defects in the search itself).  FJ is therefore
excluded from the comparison rather than compared and forgiven.  A benchmark
baseline is always a separately built unpatched binary — that is what
`bench/run_benchmark.py --vanilla-binary` takes, and it refuses a patched one.

It is deliberately *not* a ctest: it needs an unpatched HiGHS binary at the
pinned tag, which no build tree produces (the patch rewrites the source in
place) and no CI runner is guaranteed to have.  Point `--vanilla-binary` at
a system install of the same version — the script refuses a binary carrying
the patch marker, and `highs --version` on both must agree, before any
comparison means anything.

Hard gates, per instance and seed:

  * identical solver status and primal bound;
  * identical node count;
  * identical total and heuristic LP iterations.  The heuristic count is
    what `moreHeuristicsAllowed()` reads, so equality there is the proof
    that the patch does not consume the RENS/RINS budget when nothing of
    ours runs.  HiGHS does not print RENS/RINS *invocation* counts, but it
    cannot invoke them differently while leaving node count, total LP
    iterations and the solution-source display lines all identical;
  * empty normalized log diff — everything HiGHS printed, minus the parts
    that cannot match by construction: the display table's elapsed-time
    column, the timing block, the P-D integral, the profiling block's
    seconds (its call counts are kept and compared), the options-file echo,
    the width of the banner's git hash, and the patch's marker line.

Solve time is compared against `--time-tolerance` and reported inline when
exceeded; it is noise-dominated on instances this size, so it only fails
the run when `--strict-time` is passed.

Two differences are known and accepted rather than fixed:

  * the `mip-heuristics patch active` marker line, which is the only way to
    tell a patched binary from an unpatched one (the version and githash
    banners are otherwise identical).  Normalized away;
  * the per-heuristic instrumentation lines, which only exist on the patched
    side.  Only visible at `log_dev_level=3`, which this script does not set,
    but normalized away regardless.

(The `heuristic_effort_used += fj_last_effort` store the patch adds inside
stock `feasibilityJump()` used to be a third: it is unreachable here, since
neither side runs FeasibilityJump at all.  `tests/test_native_fj.cpp` is what
pins that store, and the call site it hangs off.)

Known and accepted: with FeasibilityJump disabled on both sides, **nothing in
this project compares FJ's search behaviour against upstream's** — this check
excludes it and `tests/test_native_fj.cpp` covers presence and accounting
only.  That is the deliberate trade, not an oversight: FJ is precisely what
the patch changes on purpose, so there is no behaviour to hold equal, and an
equivalence claim that quietly included it would be the false one.

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

from run_benchmark import PATCH_MARKER, probe_binary

# Same instance set as bench/correctness_check.py; they ship with HiGHS.
INSTANCES = [
    "flugpl.mps",
    "egout.mps",
    "bell5.mps",
    "lseu.mps",
    "gt2.mps",
    "p0548.mps",
]

# The pure patch-overhead configuration, per side.
#
# `mip_heuristic_run_feasibility_jump` is upstream's own option, so both
# binaries accept it and both lose FeasibilityJump — the one component the
# patch deliberately changes, and therefore the one that cannot be part of an
# equivalence claim.  `mip_heuristic_suite=off` exists only on the patched
# side and takes our four presolve heuristics and `fpr_lp` out with it.  What
# is left running is HiGHS's own solver on both sides.
PATCHED_OPTIONS = {
    "mip_heuristic_suite": "off",
    "mip_heuristic_run_feasibility_jump": "false",
}
VANILLA_OPTIONS = {"mip_heuristic_run_feasibility_jump": "false"}

# Lines dropped before diffing: wall-clock measurements, the patch's
# self-identification marker, HiGHS's echo of the options it was given (both
# runs are handed `mip_heuristic_run_feasibility_jump=false`, but only the
# patched one also gets `mip_heuristic_suite=off`, so the echo differs by
# construction and says nothing about solver behaviour — the mask is blanket
# over `Set option`, so it holds however the two option sets diverge), and the
# two once-per-solve instrumentation lines from issue #95.
#
# Those two exist only on the patched side; an unpatched binary cannot print
# them, so they are a structural difference rather than a behavioural one.
# They are invisible below `log_dev_level=3` and this script does not raise
# it, so the masks are inert today — which is exactly why they belong here
# now, before someone adds `log_dev_level` to the run and gets an unexplained
# diff instead of a green proof.  `[Heur]` and `[Sequential]` need no entry:
# `run_sequential` returns before the ledger exists at `suite=off`.
_VOLATILE_LINE = re.compile(
    r"^\s*(?:"
    r"Timing\b"  # "Timing            0.09"
    r"|P-D integral\b"  # objective integral, time-weighted
    r"|mip-heuristics patch active"  # the accepted marker
    r"|Set option\b"  # options-file echo
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
    warnings: list[str] = field(default_factory=list)
    patched_time: float = 0.0
    vanilla_time: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.failures


def compare_runs(
    instance: str,
    seed: int,
    patched: str,
    vanilla: str,
    time_tolerance: float,
    strict_time: bool,
) -> Comparison:
    """Compare two raw HiGHS logs.  Pure — the tests drive this directly."""
    pm, vm = parse_metrics(patched), parse_metrics(vanilla)
    cmp = Comparison(instance, seed, patched_time=pm.time_s, vanilla_time=vm.time_s)

    # A log with no solving report parses to all-None, and two all-None logs
    # compare equal — so without this the script would certify equivalence
    # for a pair of runs that both produced nothing at all.
    for label, m in (("patched", pm), ("vanilla", vm)):
        missing = [k for k, v in m.comparable().items() if v is None]
        if missing:
            cmp.failures.append(
                f"{label} log has no parseable solving report "
                f"(missing: {', '.join(missing)})"
            )

    vanilla_fields = vm.comparable()
    for name, patched_value in pm.comparable().items():
        vanilla_value = vanilla_fields[name]
        if patched_value != vanilla_value:
            cmp.failures.append(
                f"{name}: patched={patched_value!r} vanilla={vanilla_value!r}"
            )

    diff = list(
        difflib.unified_diff(
            normalize_log(vanilla),
            normalize_log(patched),
            fromfile="vanilla",
            tofile="patched",
            lineterm="",
            n=1,
        )
    )
    if diff:
        cmp.failures.append("normalized log differs:\n    " + "\n    ".join(diff[:40]))

    if pm.time_s > max(vm.time_s, 0.01) * time_tolerance:
        message = (
            f"solve time {pm.time_s:.2f}s vs vanilla {vm.time_s:.2f}s "
            f"exceeds {time_tolerance}x"
        )
        (cmp.failures if strict_time else cmp.warnings).append(message)
    return cmp


def find_instances_dir(binary: str) -> str | None:
    """Locate check/instances/ relative to the build tree holding `binary`."""
    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(binary)))
    for c in (
        os.path.join(build_dir, "_deps", "highs-src", "check", "instances"),
        os.path.join(build_dir, "_deps", "highs-src", "highs", "check", "instances"),
    ):
        if os.path.isdir(c):
            return c
    return None


# What HiGHS prints when it was configured outside a git repository: its own
# CMakeLists falls back to `set(GITHASH "n/a")`.  A distro package, a conda
# build, or one built from the release *source archive* all report it — and
# that is exactly the kind of binary this script asks you to point
# `--vanilla-binary` at.  It is the absence of a hash, not a hash.
NO_GITHASH = "n/a"


def version_of(binary: str) -> tuple[str, str]:
    """`(version, githash)` from `--version`, e.g. ("1.15.1", "04024d701f").

    The hash group stops at the trailing period HiGHS prints after it, rather
    than at the first non-word character: `\\w+` would take `n/a` down to `n`
    and turn "no hash" into a hash that matches nothing.
    """
    out = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=False
    ).stdout
    version = re.search(r"HiGHS version (\S+)", out)
    githash = re.search(r"Githash ([^.\s]+)", out)
    return (version.group(1) if version else "?", githash.group(1) if githash else "")


def same_build(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """Whether two `version_of` results identify the same HiGHS commit.

    The githash is compared by prefix: how many characters HiGHS prints
    depends on how the tree was fetched (a shallow `git clone` and a
    FetchContent checkout of the same tag report different lengths), so
    requiring equality would reject a legitimate pairing.

    A side reporting `n/a` has no hash to compare, so the version decides
    alone.  Refusing there would reject every unpatched HiGHS that was not
    built from a git checkout — the common shape of a system install, and the
    one this script's own docstring recommends.  It is a weaker check on that
    pairing, and the caller says so rather than pretending otherwise.
    """
    if a[0] != b[0]:
        return False
    if NO_GITHASH in (a[1], b[1]):
        return True
    short, long = sorted((a[1], b[1]), key=len)
    return bool(short) and long.startswith(short)


def run_solve(
    binary: str,
    instance_path: str,
    options: dict[str, str],
    seed: int,
    time_limit: float,
    tmp_dir: str,
) -> str:
    opts_path = os.path.join(tmp_dir, "run.opts")
    with open(opts_path, "w") as f:
        f.writelines(
            f"{k} = {v}\n" for k, v in {**options, "random_seed": str(seed)}.items()
        )
    cmd = [
        binary,
        instance_path,
        "--time_limit",
        str(time_limit),
        "--options_file",
        opts_path,
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=time_limit * 2 + 30,
        )
    except subprocess.TimeoutExpired:
        # Returned rather than raised: a hung run is one FAIL row, not a
        # traceback that abandons the instances after it.
        return f"{binary} timed out after {time_limit * 2 + 30}s"
    return r.stdout + "\n" + r.stderr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the patched binary with FeasibilityJump and our "
        "heuristics disabled against an unpatched HiGHS with FeasibilityJump "
        "disabled — the pure patch-overhead configuration"
    )
    parser.add_argument("--patched-binary", default="./build/bin/highs")
    parser.add_argument(
        "--vanilla-binary",
        required=True,
        help="Unpatched HiGHS of the SAME version (e.g. /usr/local/bin/highs)",
    )
    parser.add_argument(
        "--instances-dir",
        default=None,
        help="Path to check/instances/ (auto-detected from the patched binary)",
    )
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=1.5,
        help="Patched/vanilla solve-time ratio worth reporting (default 1.5)",
    )
    parser.add_argument(
        "--strict-time",
        action="store_true",
        help="Fail the run when the time ratio is exceeded, not just report it",
    )
    args = parser.parse_args()

    patched = os.path.abspath(args.patched_binary)
    vanilla = os.path.abspath(args.vanilla_binary)
    for b in (patched, vanilla):
        if not os.path.exists(b):
            sys.exit(f"Error: binary not found: {b}")

    # The same refusal `bench/run_benchmark.py` makes, and for the same
    # reason: two patched binaries compare equal trivially, so a check run
    # against one would report a green it never earned.  `probe_binary` is
    # imported rather than re-implemented — there is one definition of "is
    # this binary patched?".
    if probe_binary(vanilla).patched:
        sys.exit(
            f"Error: --vanilla-binary {vanilla} prints '{PATCH_MARKER}' — it is "
            "the patched binary. Build an unpatched HiGHS from the same tag; "
            "no setting on the patched build stands in for one."
        )

    pv, vv = version_of(patched), version_of(vanilla)
    if not same_build(pv, vv):
        sys.exit(
            f"Error: version mismatch, comparison would be meaningless.\n"
            f"  patched: {pv[0]} githash {pv[1]}\n  vanilla: {vv[0]} githash {vv[1]}"
        )
    if NO_GITHASH in (pv[1], vv[1]):
        # Say which check actually ran.  A binary built outside a git checkout
        # reports no hash, so the pairing rests on the version string alone —
        # which cannot tell two builds of the same release apart.
        print(
            f"Note: one binary reports githash {NO_GITHASH} (built outside a git "
            f"checkout), so the builds were matched on version {pv[0]} alone."
        )

    instances_dir = args.instances_dir or find_instances_dir(patched)
    if not instances_dir or not os.path.isdir(instances_dir):
        sys.exit("Error: instances dir not found. Pass --instances-dir explicitly.")

    tmp_dir = tempfile.mkdtemp(prefix="vanilla_equivalence_")
    comparisons: list[Comparison] = []

    for name in INSTANCES:
        path = os.path.join(instances_dir, name)
        if not os.path.exists(path):
            print(
                f"Warning: {name} not found in {instances_dir}, skipping",
                file=sys.stderr,
            )
            continue
        for seed in args.seeds:
            p_log = run_solve(
                patched,
                path,
                PATCHED_OPTIONS,
                seed,
                args.time_limit,
                tmp_dir,
            )
            v_log = run_solve(
                vanilla, path, VANILLA_OPTIONS, seed, args.time_limit, tmp_dir
            )
            c = compare_runs(
                name, seed, p_log, v_log, args.time_tolerance, args.strict_time
            )
            comparisons.append(c)
            mark = "PASS" if c.passed else "FAIL"
            print(
                f"  {mark}  {name:14s} seed={seed}  "
                f"patched={c.patched_time:.2f}s vanilla={c.vanilla_time:.2f}s"
                + ("  [" + "; ".join(c.warnings) + "]" if c.warnings else "")
            )

    if not comparisons:
        sys.exit(
            f"Error: compared nothing — none of {INSTANCES} was found in "
            f"{instances_dir}. Pass --instances-dir explicitly."
        )

    failed = [c for c in comparisons if not c.passed]
    print(f"\n{len(comparisons) - len(failed)}/{len(comparisons)} equivalent")
    for c in failed:
        print(f"\n{c.instance} seed={c.seed}:")
        for f in c.failures:
            print(f"  {f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
