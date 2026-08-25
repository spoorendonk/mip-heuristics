#!/usr/bin/env python3
"""Run patched vs vanilla HiGHS on MIPLIB instances.

One config per `mip_heuristic_suite` value, so a per-heuristic ablation is a
config list rather than a hand-written options file.  Output is
`<output>/<config>/seed<N>/<instance>.log` — those directory names are
exactly what `analyze_results.py --configs` takes, so a run is analysable
with no new
analysis code.

Everything goes through `--options_file`: HiGHS's CLI11 parser takes only its
own fixed flag set, and an unknown `--mip_heuristic_*` is a parse error that
exits without solving.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

# Benchmark configs.  Every one of them is a value of `mip_heuristic_suite`
# (#93), so the table is a name -> suite-value map rather than a bag of
# per-config option dicts.
#
# `vanilla` maps to `off` because on the *patched* binary `suite=off` hands
# HiGHS's standalone FeasibilityJump call site back and disables every custom
# heuristic — the presolve chain (FJ/FPR/LocalMIP/Scylla) and the B&B-dive
# `fpr_lp` alike — so it is vanilla-equivalent rather than vanilla-minus-FJ.
# `bench/check_vanilla_equivalence.py` proves that against a separately built
# unpatched binary.  With `--vanilla-binary` the config resolves to no options
# at all, since an unpatched binary has no `mip_heuristic_*` options to set.
# No effort pin is needed either way: the effort-option split reverted
# `mip_heuristic_effort` to upstream's 0.05 default (vanilla semantics), and
# the per-heuristic effort options are irrelevant with the presolve chain off.
#
# The ten subset configs are the pairs and triples the mix-selection stage
# (#107) sweeps alongside the four singletons, `all` and `off` — sixteen
# rows, one per subset of the chain.  They exist because `mip_heuristic_suite`
# takes a comma-separated list (#112); the config *name* joins with `+`
# instead, because the name is a results-tree directory and a column label in
# generated LaTeX, and a comma in either is a needless escaping problem.
#
# Names list heuristics in chain order (FJ -> FPR -> LocalMIP -> Scylla) so
# one subset has one spelling; `fpr+fj` is not a config even though the suite
# value it would map to is legal.
#
# The recorded PLATO table in README.md was measured at `all_opp` — FJ + FPR
# + LocalMIP with Scylla deliberately excluded, because PDLP solves are
# expensive enough to hurt wall-clock on general instances.  That is
# `fj+fpr+local_mip` below.  Expressible is not reproducible: the binary
# those numbers came from predates the runner cleanup, so do not compare a
# fresh run against the recorded `all_opp` row.
CONFIG_SUITES: dict[str, str] = {
    "vanilla": "off",
    "off": "off",
    "fj": "fj",
    "fpr": "fpr",
    "local_mip": "local_mip",
    "scylla": "scylla",
    "fj+fpr": "fj,fpr",
    "fj+local_mip": "fj,local_mip",
    "fj+scylla": "fj,scylla",
    "fpr+local_mip": "fpr,local_mip",
    "fpr+scylla": "fpr,scylla",
    "local_mip+scylla": "local_mip,scylla",
    "fj+fpr+local_mip": "fj,fpr,local_mip",
    "fj+fpr+scylla": "fj,fpr,scylla",
    "fj+local_mip+scylla": "fj,local_mip,scylla",
    "fpr+local_mip+scylla": "fpr,local_mip,scylla",
    "all": "all",
}


@dataclass(frozen=True)
class ConfigPlan:
    """One resolved config: what to run it with, and where it lands.

    `build_plan` is the only place a config name's two consequences — which
    binary, which options — are chosen together.  Choosing them at separate
    use sites is how `vanilla` would silently pick the patched binary while
    its options came from the vanilla branch.
    """

    name: str  # directory name under --output, e.g. `fpr`
    base: str  # entry in CONFIG_SUITES, e.g. `fpr`
    binary: str
    options: dict[str, str]

    @property
    def identity(self) -> tuple[str, tuple[tuple[str, str | float], ...]]:
        """What the solver actually sees, for "is this the same run?" checks.

        Not the name and not the raw options: two names can map to the same
        options, which would otherwise produce identical trees with nothing to
        flag it.  Numeric option values are compared as floats.
        """
        normalized: list[tuple[str, str | float]] = []
        for key, value in sorted(self.options.items()):
            try:
                normalized.append((key, float(value)))
            except ValueError:
                normalized.append((key, value))
        return (self.binary, tuple(normalized))


def resolve_config(config: str) -> str:
    """The config's `CONFIG_SUITES` key, with the unknown-name raise.

    That raise is the point of this module's config surface: the old
    implementation returned `{}` for anything it did not recognise, so a
    mistyped `--configs patchd` produced a fully populated, plausible-looking,
    completely meaningless results tree that nothing downstream noticed.
    """
    if config not in CONFIG_SUITES:
        known = ", ".join(sorted(CONFIG_SUITES))
        raise ValueError(f"unknown config {config!r}; known configs: {known}")
    return config


def config_options(config: str, *, external_vanilla: bool = False) -> dict[str, str]:
    """HiGHS options for one config name.

    Raises ValueError on an unknown name (see `resolve_config`).
    `external_vanilla` says the `vanilla` config runs on a separately built
    unpatched binary, which has no `mip_heuristic_*` options at all.
    """
    base = resolve_config(config)
    if base == "vanilla" and external_vanilla:
        return {}
    return {"mip_heuristic_suite": CONFIG_SUITES[base]}


def build_base_options(
    threads: int | None, dev_log: bool, extra_options: list[str] | None
) -> dict[str, str]:
    """Options applied to every config, from the flags that are not per-config.

    Empty by default, and that emptiness is load-bearing twice over: no
    `threads` (forcing `threads=1` collapses each heuristic to a single worker
    and invalidates a throughput benchmark) and no `log_dev_level` (level 3
    costs up to 4.4x wall time — see `--dev-log`).
    """
    base: dict[str, str] = {}
    if threads is not None:
        base["threads"] = str(threads)
    if dev_log:
        base["log_dev_level"] = "3"
    for kv in extra_options or []:
        if "=" not in kv:
            print(
                f"Warning: ignoring malformed --extra-options entry (no '='): {kv!r}",
                file=sys.stderr,
            )
            continue
        key, value = kv.split("=", 1)
        key, value = key.strip(), value.strip()
        # `--dev-log` is the reason the whole tree is analysable, and an
        # override here cancels it silently: the run header still announces
        # instrumentation, every solve completes, and the omission only
        # surfaces hours later when the tree turns out to carry no
        # instrumentation.  Same failure family as the `random_seed`
        # collision below it.
        if dev_log and key == "log_dev_level" and value != "3":
            print(
                f"Warning: --extra-options {key}={value} overrides --dev-log; "
                "the [Heur]/[Sequential] lines will not be emitted, so this "
                "tree will carry no per-heuristic instrumentation",
                file=sys.stderr,
            )
        base[key] = value
    return base


def build_plan(config: str, patched_binary: str, vanilla_binary: str) -> ConfigPlan:
    """Resolve a config name to its binary and options, in one place.

    Both the binary choice and the options come off the same resolved name, so
    a config cannot pick one branch's binary while taking the other branch's
    options.
    """
    base = resolve_config(config)
    external_vanilla = vanilla_binary != patched_binary
    options = config_options(config, external_vanilla=external_vanilla)
    binary = vanilla_binary if base == "vanilla" else patched_binary
    return ConfigPlan(name=config, base=base, binary=binary, options=options)


def load_instances(path: str) -> list[str]:
    """Load instance names from a file (one per line; `#` starts a comment,
    whether at the start of the line or inline after the name)."""
    instances = []
    with open(path) as f:
        for line in f:
            name = line.split("#", 1)[0].strip()
            if name:
                instances.append(name)
    return instances


# Where a MIPLIB collection may already live, most-preferred first.  Kept in
# step with the same list in `bench/download_miplib.sh` — the two are one
# contract, and a benchmark that cannot find what the downloader just wrote is
# the failure this list exists to prevent.  `~/data/miplib` precedes
# `/tmp/miplib` because /tmp does not survive a reboot, but /tmp stays in the
# list so checkouts that already populated it keep working.
MIPLIB_SEARCH_PATH: tuple[str, ...] = (
    os.path.expanduser("~/data/miplib"),
    "/tmp/miplib",
)

# A directory counts as a collection above this many instances, matching
# MIN_INSTANCES in download_miplib.sh.
MIPLIB_MIN_INSTANCES = 200


def resolve_data_dir(explicit: str | None) -> str:
    """Pick the MIPLIB directory to read instances from.

    An explicit `--data-dir` wins outright, even when it names an empty
    directory: asking for a specific directory must not silently resolve to a
    different one, or a typo'd path reads a different instance set than the one
    named and the run reports on instances nobody asked for.  An explicit
    *empty string* is treated as absent, matching download_miplib.sh -- that is
    a wrapper passing an unset variable through, not a request for the cwd.

    With no explicit value, `$MIPLIB_DIR` then MIPLIB_SEARCH_PATH are probed
    and the first populated one wins.  Falls back to the head of the search
    path so the "not found" diagnostic names a concrete directory.
    """
    if explicit:
        return explicit

    candidates = []
    env_dir = os.environ.get("MIPLIB_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.extend(MIPLIB_SEARCH_PATH)

    for d in candidates:
        if not os.path.isdir(d):
            continue
        found = 0
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.name.endswith(".mps.gz"):
                        found += 1
                        if found > MIPLIB_MIN_INSTANCES:
                            return d
        except OSError:
            # Unreadable candidate counts as absent, not fatal: /tmp/miplib is
            # probed for every user and may be another user's mode-700
            # directory on a shared box.  Crashing there would abort exactly
            # the run that still needed to find a collection.
            continue
    return candidates[0]


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
        f.writelines(f"{k} = {v}\n" for k, v in options.items())


# HiGHS warnings that mean the solve did not do what its config name says.
# These exit 0 with a complete, ordinary-looking log, so nothing below the
# runner can tell the resulting tree apart from a good one.
#
# The first is the important one: HiGHS validates option *names* but not
# string option *values*, so `mip_heuristic_suite=of` is accepted by
# `setOptionValue` and caught only at solve time, where `run_presolve`
# deliberately fails open to all four heuristics with a `kWarning`.  An
# `off/` directory would then hold runs that actually executed `all`.  The
# realistic route there is not a typo but a HiGHS tag bump renaming the
# values — silently rejected options are a recurring failure in this project
# (see the "Bumping the HiGHS tag" note in CLAUDE.md).
#
# The second is the same class from the other side: `suite=fj` with
# `mip_heuristic_run_feasibility_jump=false` asks for FJ and then takes it
# away, so an "FJ isolated" row would measure vanilla-minus-FJ.
#
# These strings are a contract with `run_presolve` in `src/mode_dispatch.cpp`,
# which carries the matching note.  Both ends are pinned by the
# `[bench-contract]` case in `tests/test_smoke.cpp`, which asserts them
# against the running binary's own output — so a reword there fails the C++
# suite rather than silently switching this detection off.
CONFIG_IGNORED_WARNINGS = (
    "Unknown mip_heuristic_suite value",
    "no heuristic will run",
)


def find_ignored_config_warning(output: str) -> str | None:
    """Return the line of `output` saying the run ignored its configuration."""
    for line in output.splitlines():
        if any(marker in line for marker in CONFIG_IGNORED_WARNINGS):
            return line.strip()
    return None


def record_failure(log_path: str, output: str) -> None:
    """Park a failed run's output *beside* the log rather than as the log.

    `should_skip` treats a non-empty `<instance>.log` as done and
    `analyze_results.py` globs `*.log`, so writing a crash, a timeout or a
    rejected options file into that name does two silent things: it cements
    the failure across a `--skip-existing` resume, and it scores as a
    legitimately infeasible instance (no incumbents, `primal_bound=inf`).
    `<instance>.log.err` is matched by neither, so the run is retried on
    resume and the evidence is still on disk.
    """
    with open(log_path + ".err", "w") as f:
        f.write(output)
    if os.path.exists(log_path):
        os.remove(log_path)


def write_log(log_path: str, output: str) -> None:
    """Record a successful run, clearing any `.err` from an earlier attempt.

    The counterpart to `record_failure`, and it has to clear the `.err` for
    the same reason that clears the `.log`: exactly one of the two files
    describes what is in the tree now.  Without this, a `--skip-existing`
    resume over a partially failed campaign leaves `.err` files whose
    instances have since succeeded, so `.err` degrades from "this run failed"
    to "this run failed at some point", which is not a property anyone can
    filter on.
    """
    with open(log_path, "w") as f:
        f.write(output)
    err_path = log_path + ".err"
    if os.path.exists(err_path):
        os.remove(err_path)


def record_partial(log_path: str, output: str, kill_after: float) -> None:
    """Keep a killed run's streamed output as a real log, with the kill marked.

    HiGHS checks the clock between work units, so one long simplex solve at the
    root can carry a run past `time_limit` without it ever returning to look;
    the runner then kills it.  That run is not a harness failure and re-running
    it reproduces the same hang, so `record_failure`'s retry-on-resume
    behaviour is wrong for it — but so is discarding the output, which is what
    this used to do.  Everything the solver printed before the kill is real
    measured data, and the campaign's headline metrics (T1st, primal integral)
    are computed from those incumbent lines alone, not from the Solving report
    the run never reached.

    So it goes in as a `.log`: analysed like any other run, skipped on resume,
    and tagged with the marker `parse_highs_log` turns into `killed`.  What is
    lost is only the trailing summary — and, because stdout is a pipe and
    therefore block-buffered, whatever sat in the last unflushed block.
    """
    write_log(
        log_path,
        output + f"\n--- runner ---\nTIMEOUT: process killed after {kill_after}s\n",
    )


def run_single(
    binary: str,
    instance_file: str,
    instance_name: str,
    config: str,
    seed: int,
    time_limit: float,
    output_dir: str,
    extra_options: dict[str, str] | None = None,
) -> tuple[str, str, int, bool]:
    """Run HiGHS on a single instance with given config and seed.

    Returns (instance_name, config, seed, success).
    """
    seed_dir = os.path.join(output_dir, config, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    log_path = os.path.join(seed_dir, f"{instance_name}.log")

    # Build options: start with extra_options, then add random_seed
    options = dict(extra_options) if extra_options is not None else {}
    options["random_seed"] = str(seed)

    opts_path = os.path.join(seed_dir, f"{instance_name}.opts")
    write_options_file(options, opts_path)

    cmd = [
        binary,
        instance_file,
        "--time_limit",
        str(time_limit),
        "--options_file",
        opts_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=time_limit * 1.5 + 120,  # generous timeout beyond HiGHS limit
        )
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        # HiGHS's CLI exits `int(HighsStatus)`: 0 = kOk, 1 = kWarning — which
        # is what "Time limit reached" gives, the normal benchmark outcome —
        # and 255 = kError, meaning it never solved.  An unknown or
        # out-of-range option in the .opts file lands in that last bucket:
        # two ERROR lines, the banner, exit 255.  Treating that as a completed
        # run is the same silent-failure family as the old `config_opts_for`
        # returning `{}`, one layer down.
        if result.returncode not in (0, 1):
            record_failure(
                log_path,
                output + f"\n--- runner ---\n"
                f"{binary} exited {result.returncode} without solving\n",
            )
            print(
                f"Error: {binary} exited {result.returncode} on {instance_name} "
                f"({config}, seed {seed}) without solving; see {log_path}.err",
                file=sys.stderr,
            )
            return (instance_name, config, seed, False)
        # A run that solved fine but ignored the configuration it was given is
        # worse than one that failed: the exit code is 0 and the log is
        # complete, so the tree is indistinguishable from a good one while the
        # directory name says something the binary did not do.
        ignored = find_ignored_config_warning(output)
        if ignored is not None:
            record_failure(
                log_path,
                output + f"\n--- runner ---\n"
                f"{binary} ignored its configuration: {ignored}\n",
            )
            print(
                f"Error: {binary} ignored its configuration on {instance_name} "
                f"({config}, seed {seed}): {ignored}; see {log_path}.err",
                file=sys.stderr,
            )
            return (instance_name, config, seed, False)
        write_log(log_path, output)
        return (instance_name, config, seed, True)
    except subprocess.TimeoutExpired as exc:
        # On POSIX, `subprocess.run` has already drained into the exception
        # whatever the child streamed before the kill (it documents this where
        # it re-raises).  Binding `exc` is the whole difference between keeping
        # that and losing it.
        partial = exc.stdout or ""
        if isinstance(partial, bytes):  # defensive: text=True should preclude it
            partial = partial.decode(errors="replace")
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if stderr:
            partial += "\n--- stderr ---\n" + stderr
        kill_after = time_limit * 1.5 + 120
        # The banner is the test for "the solver ran and printed something we
        # can measure".  Without it there is no run to keep -- a binary that
        # hung before its first write is a harness problem, and that is exactly
        # the case `record_failure` exists to retry.
        if "Running HiGHS" in partial:
            record_partial(log_path, partial, kill_after)
        else:
            record_failure(
                log_path,
                partial + f"\n--- runner ---\n"
                f"TIMEOUT: process killed after {kill_after}s "
                f"before printing anything parseable\n",
            )
        return (instance_name, config, seed, False)
    except Exception as e:  # noqa: BLE001 - any failure becomes one FAIL row, not a dead campaign
        record_failure(log_path, f"ERROR: {e}\n")
        return (instance_name, config, seed, False)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Split out of `main` so the defaults are assertable.  `--data-dir`'s
    default in particular is load-bearing: it must stay None so that
    `resolve_data_dir` actually runs, and a hardcoded path there is invisible
    to any test that calls `resolve_data_dir` directly.
    """
    parser = argparse.ArgumentParser(
        description="Run patched vs vanilla HiGHS benchmark"
    )
    parser.add_argument("--instances", required=True, help="File with instance names")
    parser.add_argument(
        "--binary", default="./build/bin/highs", help="Path to patched HiGHS binary"
    )
    parser.add_argument(
        "--vanilla-binary",
        default=None,
        metavar="PATH",
        help="Separate binary for the vanilla config (e.g. system HiGHS). "
        "When set, vanilla runs with no custom options — just time limit "
        "and seed — since the external binary has no mip_heuristic_* options.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with .mps.gz files. Default: $MIPLIB_DIR, then "
        + ", then ".join(MIPLIB_SEARCH_PATH)
        + " — the first one holding a collection wins.",
    )
    parser.add_argument(
        "--time-limit", type=float, default=60, help="Time limit per instance (seconds)"
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default="bench/results",
        help="Output directory for logs (default: bench/results)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds to run (default: 0)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all", "vanilla"],
        help=(
            "Configs to run (default: all vanilla). One of: "
            + ", ".join(sorted(CONFIG_SUITES))
            + ". Each selects a mip_heuristic_suite value, with `+` in a name "
            "standing for the `,` in that value; `vanilla` runs the external "
            "--vanilla-binary when one is given. An unknown name is an error, "
            "not a default-option run."
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N instances (for chunked runs, default: 0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Run at most N instances (for chunked runs, default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip instances whose log file already exists (safe to resume)",
    )
    parser.add_argument(
        "--wall-time-budget",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Stop launching new instances after SECONDS of wall time "
        "(current instance still finishes). Use with --skip-existing "
        "to resume later.",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Run instance→config loop order (vanilla+patched per instance) "
        "rather than config→instance. Gives paired results sooner.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Number of solver threads — OMIT unless you specifically need to cap. "
            "Forcing threads=1 collapses each heuristic to a single worker and "
            "silently hides the parallelism the patched heuristics are built for "
            "(see CLAUDE.md benchmarking note).  It is the right setting only when "
            "reproducibility is the point, not throughput."
        ),
    )
    parser.add_argument(
        "--extra-options",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Extra options appended to all config options, "
        "e.g. mip_heuristic_fpr_effort=0.10",
    )
    parser.add_argument(
        "--dev-log",
        action="store_true",
        help=(
            "Set log_dev_level=3, which is what makes the [Heur] / [Sequential] "
            "instrumentation visible to parse_highs_log.py. "
            "OFF by default because it is not free: HiGHS's own FeasibilityJump "
            "logs one line per weight bump at exactly that level, from every "
            "parallel FJ worker, with an fflush each. Measured on five bundled "
            "instances at a 10 s limit that is 97-750x the log volume and up to "
            "4.4x the total solve wall time. The cost is concentrated in the FJ phase, so it "
            "lands on the very numbers the per-heuristic analysis reads, and "
            "asymmetrically: fj's effort_per_ms is depressed by its own logging "
            "while the other three barely log at all, so a --dev-log run's rates "
            "are not comparable with a plain run's. Use it for attribution runs, "
            "not for headline timings."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)

    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        print(f"Error: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    vanilla_binary = binary  # default: same binary at mip_heuristic_suite=off
    if args.vanilla_binary is not None:
        vanilla_binary = os.path.abspath(args.vanilla_binary)
        if not os.path.exists(vanilla_binary):
            print(f"Error: vanilla binary not found: {vanilla_binary}", file=sys.stderr)
            sys.exit(1)
        # `build_plan` decides externality by comparing resolved paths, not by
        # whether the flag was given — run_plato.sh falls back to --binary when
        # `which highs` finds nothing, and this is the line a reader checks to
        # confirm which baseline they measured.
        if vanilla_binary == binary:
            print(
                f"Vanilla binary : {vanilla_binary} (same path as --binary — "
                "vanilla runs the patched binary at mip_heuristic_suite=off)"
            )
        else:
            print(f"Vanilla binary : {vanilla_binary} (external — no custom options)")
    print(f"Patched binary : {binary}")

    # Resolve configs before anything else runs: an unknown name must fail
    # here, not after producing hours of default-option results.
    try:
        config_names = list(args.configs)
        for config in config_names:
            resolve_config(config)
        dupes = sorted({n for n in config_names if config_names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"duplicate config(s) {', '.join(dupes)} — a config name is one "
                "output directory, so a repeat is the same run twice"
            )
        plans = [build_plan(c, binary, vanilla_binary) for c in config_names]
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    # Distinct names can still resolve to one configuration — `vanilla` is
    # `off` unless an external binary was given.  That is "N identical runs
    # under N names", the thing the unknown-name raise exists to prevent, so warn
    # rather than silently burning the compute.  A warning, not an error:
    # run_plato.sh legitimately reaches the vanilla==off case when `which
    # highs` finds nothing.
    seen: dict[tuple[str, tuple[tuple[str, str | float], ...]], str] = {}
    for plan in plans:
        key = plan.identity
        if key in seen:
            print(
                f"Warning: config {plan.name!r} is identical to {seen[key]!r} "
                "(same binary, same options) — duplicated work, not a second "
                "data point",
                file=sys.stderr,
            )
        else:
            seen[key] = plan.name
    print(f"Configs        : {' '.join(p.name for p in plans)}")
    # State the instrumentation decision in the header of every run.  Without
    # it, `--dev-log` is a flag you discover you needed after the campaign:
    # `SolveResult.heuristic_wall_fraction` is `None` on a plain run — not 0.0,
    # which is reserved for an instrumented `suite=off` — so the attribution
    # tables come out empty rather than wrong, hours later.
    base_opts = build_base_options(args.threads, args.dev_log, args.extra_options)

    # Keyed on the option that will actually be written, not on `--dev-log`:
    # `--extra-options log_dev_level=1` overrides the flag, and the header is
    # the record that gets captured (the collision warning goes to stderr, so
    # under `run_benchmark.py > run.log` the two land in different streams and
    # the header is the half that survives).
    print(
        "Instrumentation: "
        + (
            "log_dev_level=3 ([Heur]/[Sequential]) — attribution run, timings inflated"
            if base_opts.get("log_dev_level") == "3"
            else "off — headline timings; pass --dev-log for the attribution tables"
        )
    )

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
        print(
            f"Warning: {len(missing)} instances not found in {args.data_dir}:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        print(
            "Populate a shared collection with `bash bench/download_miplib.sh` "
            "(writes ~/data/miplib, reused by every checkout), or point "
            "--data-dir / $MIPLIB_DIR at an existing one.",
            file=sys.stderr,
        )
        instances = [n for n in instances if n in instance_files]

    os.makedirs(args.output, exist_ok=True)

    # Apply chunk slicing
    if args.start:
        instances = instances[args.start :]
    if args.count is not None:
        instances = instances[: args.count]

    total_runs = len(plans) * len(args.seeds) * len(instances)
    done = 0
    budget_exhausted = False
    run_start = time.time()

    # `run_single` writes random_seed last, from --seeds, because the seed is
    # part of the output path (`seed<N>/`) and an --extra-options pin would
    # make the directory name a lie.  Say so: silently dropping the flag is
    # the same "your option did nothing" failure this changeset exists to fix.
    if "random_seed" in base_opts:
        print(
            f"Warning: --extra-options random_seed={base_opts['random_seed']!r} is "
            f"ignored — the seed comes from --seeds ({' '.join(map(str, args.seeds))}) "
            "and names the output directory",
            file=sys.stderr,
        )
    # Config options win over base options, so any --extra-options pin of a key
    # a config also sets is silently discarded on that config.
    for plan in plans:
        for key in sorted(set(base_opts) & set(plan.options)):
            print(
                f"Warning: --extra-options {key}={base_opts[key]!r} is overridden "
                f"by config {plan.name!r} ({key}={plan.options[key]!r})",
                file=sys.stderr,
            )

    def should_skip(config: str, name: str, seed: int) -> bool:
        if not args.skip_existing:
            return False
        seed_dir = os.path.join(args.output, config, f"seed{seed}")
        log_path = os.path.join(seed_dir, f"{name}.log")
        return os.path.exists(log_path) and os.path.getsize(log_path) > 0

    def check_budget() -> bool:
        """Return True if wall-time budget is exhausted."""
        if args.wall_time_budget is None:
            return False
        return (time.time() - run_start) >= args.wall_time_budget

    def run_one(plan: ConfigPlan, name: str, seed: int) -> None:
        nonlocal done, budget_exhausted
        if budget_exhausted:
            return
        if should_skip(plan.name, name, seed):
            done += 1
            print(f"[{done}/{total_runs}] SKIP {name} ({plan.name}) — log exists")
            return
        if check_budget():
            budget_exhausted = True
            elapsed = time.time() - run_start
            print(
                f"\nTime budget reached ({elapsed / 3600:.1f}h elapsed). "
                f"Re-run with same command to continue."
            )
            return
        extra_opts = {**base_opts, **plan.options}
        _, _, _, success = run_single(
            plan.binary,
            instance_files[name],
            name,
            plan.name,
            seed,
            args.time_limit,
            args.output,
            extra_opts,
        )
        done += 1
        status = "OK" if success else "FAIL"
        elapsed = time.time() - run_start
        print(
            f"[{done}/{total_runs}] {name} ({plan.name}) {status}  "
            f"[{elapsed / 3600:.1f}h elapsed]"
        )

    if args.interleave:
        # instance → seed → config: gives paired vanilla+patched results sooner
        for name in instances:
            for seed in args.seeds:
                for plan in plans:
                    run_one(plan, name, seed)
                    if budget_exhausted:
                        break
                if budget_exhausted:
                    break
            if budget_exhausted:
                break
    else:
        # config → seed → instance: runs all of one config before the next
        for plan in plans:
            for seed in args.seeds:
                print(f"\n{'=' * 60}")
                print(
                    f"Config: {plan.name}, seed: {seed} "
                    f"({len(instances)} instances, {args.time_limit}s limit)"
                )
                print(f"{'=' * 60}")
                for name in instances:
                    run_one(plan, name, seed)
                    if budget_exhausted:
                        break
                if budget_exhausted:
                    break
            if budget_exhausted:
                break

    elapsed_total = time.time() - run_start
    print(
        f"\nDone. {done} runs in {elapsed_total / 3600:.1f}h. Results in {args.output}/"
    )
    # The config names are the directory names, so the analysis command is
    # mechanical — print it rather than making the reader reconstruct it.
    # Two configs is the pairwise/PLATO shape README.md and run_plato.sh
    # document; three or more is the one-row-per-config ablation a sweep
    # produces.  Suggesting --ablation for a `patched vanilla` run would
    # contradict the command run_plato.sh prints seconds later.
    if done:
        # A presolve-only tree has no dual side at all — `Dual bound -inf`,
        # zero nodes, zero LP iterations — so every gap and primal-integral
        # column analyze_results.py prints for it is meaningless.  It reads
        # with analyze_presolve_probe.py instead, and the suggestion is where
        # a reader looks for which one to run.
        if base_opts.get("mip_heuristic_presolve_only", "").lower() == "true":
            print(
                "\nAnalyze with:\n"
                f"  python3 bench/analyze_presolve_probe.py {args.output}"
            )
        else:
            mode = "--ablation" if len(plans) > 2 else "--baseline --summary"
            print(
                "\nAnalyze with:\n"
                f"  python3 bench/analyze_results.py {args.output} {mode} "
                f"--configs {' '.join(p.name for p in plans)} "
                f"--time-limit {args.time_limit:g}"
            )


if __name__ == "__main__":
    main()
