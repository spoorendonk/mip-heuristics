#!/usr/bin/env python3
"""Run patched vs vanilla HiGHS on MIPLIB instances.

One config per `mip_heuristic_suite` value, so a per-heuristic ablation is a
config list rather than a hand-written options file — plus `vanilla`, which is
not a suite value at all but the separately built unpatched binary that
`--vanilla-binary` names (required for that config, and probed before the first
solve).  Output is `<output>/<config>/seed<N>/<instance>.log` — those directory
names are exactly what `analyze_results.py --configs` takes, so a run is
analysable with no new analysis code.

Everything goes through `--options_file`: HiGHS's CLI11 parser takes only its
own fixed flag set, and an unknown `--mip_heuristic_*` is a parse error that
exits without solving.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

# Benchmark configs.  Every one of them is a value of `mip_heuristic_suite`
# (#93), so the table is a name -> suite-value map rather than a bag of
# per-config option dicts.
#
# `vanilla` is deliberately **not** in this table, and that absence is the
# whole of #147.  It used to map to `off`, so a `vanilla` run without
# `--vanilla-binary` was the patched binary at `mip_heuristic_suite=off` —
# an ablation of our four presolve heuristics, filed under the name of a
# baseline.  `off` is not a vanilla proxy: the binary around it is still the
# patched one.  The baseline is a separately built unpatched binary, which
# has no `mip_heuristic_*` options at all, so `vanilla` names a *binary*
# rather than a suite value and carries no options of its own.
#
# No effort pin is needed on the patched side either: the effort-option split
# reverted `mip_heuristic_effort` to upstream's 0.05 default (vanilla
# semantics), so a patched run's B&B heuristic budget already matches.
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

# The one config that is a binary rather than a suite value: it runs the
# separately built unpatched HiGHS named by `--vanilla-binary`, with no
# `mip_heuristic_*` option at all.
VANILLA_CONFIG = "vanilla"

# Every legal `--configs` name.  `vanilla` is a name without a suite value,
# so the two have to be joined here rather than looked up in one table.
KNOWN_CONFIGS: tuple[str, ...] = (VANILLA_CONFIG, *CONFIG_SUITES)

# Three constants this module shares with `bench/make_archive.py`, spelled out
# in both rather than imported from one.  That duplication is deliberate and
# one-directional: `make_archive.py` is copied *into* a release archive and run
# there by `REGENERATE.sh`, with no checkout and no `run_benchmark.py` beside
# it, so it cannot import this module.  The three must stay byte-identical;
# `test_the_two_bench_modules_agree_on_the_shared_constants` in
# `bench/test_run_benchmark.py` fails if they drift.
#
# What a patched binary prints right after its version banner, and the only
# thing separating it from an unpatched build of the same tag (the banner
# itself is identical).  `apply_patch.cmake` injects it into `highsLogHeader`.
PATCH_MARKER = "mip-heuristics patch active"

# The banner both builds print.  Not the `--version` form, which is a
# different line and — decisively — carries no marker, because `--version`
# never reaches `highsLogHeader`.
#
# The hash group is `[^)]+`, not `\w+`: HiGHS sets `GITHASH` to the literal
# `n/a` when it is configured outside a git repository (its own
# `CMakeLists.txt`, the `else()` branch of the `git describe` block), and `/`
# is not a word character.  A HiGHS installed from a distro package or built
# from a release *source archive* prints `git hash: n/a` — so a `\w+` pattern
# does not match at all, `version` comes back None, and `check_vanilla_binary`
# refuses a perfectly good baseline with "printed no HiGHS banner", quoting
# the banner it just printed back at the reader.  Only the version decides
# anything here; the hash is captured for the manifest and read by no check.
BANNER_RE = re.compile(r"Running HiGHS (\S+) \(git hash: ([^)]+)\)")

# The HiGHS tag this checkout builds against, read from the one place that
# defines it — a constant here would be a second definition to keep in step
# with a tag bump.  It captures the tag *as written* (`v1.15.1`), which is what
# `make_archive.py` records in a manifest; the banner prints a bare version, so
# `expected_highs_version` strips the `v` at the point of comparison rather than
# in the pattern.  A regex that differed there by one optional character would
# be the worst kind of duplicate: same name, same file parsed, different value.
HIGHS_TAG_RE = re.compile(r"GIT_TAG\s+(\S+)")
FETCH_HIGHS_CMAKE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cmake",
    "FetchHiGHS.cmake",
)


@dataclass(frozen=True)
class ConfigPlan:
    """One resolved config: what to run it with, and where it lands.

    `build_plan` is the only place a config name's two consequences — which
    binary, which options — are chosen together.  Choosing them at separate
    use sites is how `vanilla` would silently pick the patched binary while
    its options came from the vanilla branch.
    """

    name: str  # directory name under --output, e.g. `fpr`
    base: str  # entry in KNOWN_CONFIGS, e.g. `fpr`
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
    """The config's `KNOWN_CONFIGS` name, with the unknown-name raise.

    That raise is the point of this module's config surface: the old
    implementation returned `{}` for anything it did not recognise, so a
    mistyped `--configs patchd` produced a fully populated, plausible-looking,
    completely meaningless results tree that nothing downstream noticed.
    """
    if config not in KNOWN_CONFIGS:
        known = ", ".join(sorted(KNOWN_CONFIGS))
        raise ValueError(f"unknown config {config!r}; known configs: {known}")
    return config


def config_options(config: str) -> dict[str, str]:
    """HiGHS options for one config name.

    Raises ValueError on an unknown name (see `resolve_config`).  `vanilla`
    is the empty one: it always runs the separately built unpatched binary,
    which has no `mip_heuristic_*` options to set.
    """
    base = resolve_config(config)
    if base == VANILLA_CONFIG:
        return {}
    return {"mip_heuristic_suite": CONFIG_SUITES[base]}


def existing_log(output: str, config: str, name: str, seed: int) -> bool:
    """True when this run already has a non-empty log in the tree.

    One definition, because `--skip-existing` and `--count` have to agree on
    what "already done" means: a chunk that counted an instance as pending
    and then skipped every one of its runs would burn the chunk on nothing.
    """
    log_path = os.path.join(output, config, f"seed{seed}", f"{name}.log")
    return os.path.exists(log_path) and os.path.getsize(log_path) > 0


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


def build_plan(
    config: str, patched_binary: str, vanilla_binary: str | None
) -> ConfigPlan:
    """Resolve a config name to its binary and options, in one place.

    Both the binary choice and the options come off the same resolved name, so
    a config cannot pick one branch's binary while taking the other branch's
    options.

    `vanilla_binary` is `None` when `--vanilla-binary` was not given, and the
    `vanilla` config then raises rather than falling back to the patched
    binary (#147).  The fallback used to be silent: it produced a `vanilla/`
    tree holding the patched binary at `mip_heuristic_suite=off`, which is an
    ablation of our four presolve heuristics and not a baseline.
    """
    base = resolve_config(config)
    binary = patched_binary
    if base == VANILLA_CONFIG:
        if vanilla_binary is None:
            raise ValueError(
                "config 'vanilla' requires --vanilla-binary pointing at a "
                "separately built unpatched HiGHS of the same tag. There is no "
                "fallback: mip_heuristic_suite=off on the patched binary is the "
                "'our four presolve heuristics disabled' ablation, not a vanilla "
                "baseline — use the config named 'off' if that is what you want"
            )
        binary = vanilla_binary
    return ConfigPlan(
        name=config, base=base, binary=binary, options=config_options(config)
    )


@dataclass(frozen=True)
class BinaryProbe:
    """What one HiGHS binary says about itself, without solving anything."""

    version: str | None  # `1.15.1`, or None when no banner was printed
    githash: str | None
    patched: bool  # carries the `mip-heuristics patch active` marker
    output: str  # everything it printed, for the refusal message


def probe_binary(path: str) -> BinaryProbe:
    """Run `path` with no arguments and read the banner it prints.

    No model, so no solve: HiGHS complains that no filename was given, prints
    its log header, and exits non-zero.  The exit status is therefore ignored
    on purpose.  The header is the only place the patch marker appears —
    `--version` bypasses `highsLogHeader` entirely and prints neither it nor
    this banner form.
    """
    try:
        result = subprocess.run(
            [path],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        output = result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"<could not run {path}: {exc}>"
    match = BANNER_RE.search(output)
    return BinaryProbe(
        version=match.group(1) if match else None,
        githash=match.group(2) if match else None,
        patched=PATCH_MARKER in output,
        output=output,
    )


def expected_highs_version() -> str:
    """The HiGHS version this checkout builds against, from FetchHiGHS.cmake.

    Raises rather than defaulting: the vanilla binary's tag has to be checked
    against something, and a missing tag file means the check would silently
    become "any version will do" — the failure family #147 exists to close.

    The `v` of the tag is stripped here, not in `HIGHS_TAG_RE`: the banner
    prints `Running HiGHS 1.15.1`, while a manifest records the tag `v1.15.1`,
    and the pattern is shared with `make_archive.py`, which wants the latter.
    """
    try:
        with open(FETCH_HIGHS_CMAKE) as f:
            text = f.read()
    except OSError as exc:
        raise ValueError(
            f"cannot read the HiGHS tag from {FETCH_HIGHS_CMAKE} ({exc}); "
            "run run_benchmark.py from a checkout, since the vanilla binary's "
            "version has to be checked against the tag this tree builds"
        ) from exc
    match = HIGHS_TAG_RE.search(text)
    if not match:
        raise ValueError(f"no GIT_TAG found in {FETCH_HIGHS_CMAKE}")
    return match.group(1).removeprefix("v")


def check_vanilla_binary(path: str) -> BinaryProbe:
    """Refuse a `--vanilla-binary` that is not an unpatched HiGHS of our tag.

    Called before the first solve, because the point is to prevent a bad
    results tree rather than to describe one afterwards.  Two ways to get it
    wrong, both silent until now: pointing the flag at the patched build
    (which then runs `suite=off`, an ablation), and pointing it at a system
    HiGHS of a different version (which is not comparable at all).

    Returns the accepted probe so the caller can report what it verified
    without running the binary — or re-reading the tag — a second time.
    """
    expected = expected_highs_version()
    probe = probe_binary(path)
    if probe.patched:
        raise ValueError(
            f"--vanilla-binary {path} is a *patched* binary: it prints "
            f"'{PATCH_MARKER}'. The baseline must be a separately built "
            "unpatched HiGHS; the patched binary at mip_heuristic_suite=off "
            "is an ablation of our heuristics, not vanilla"
        )
    if probe.version is None:
        raise ValueError(
            f"--vanilla-binary {path} printed no HiGHS banner, so it cannot "
            f"be identified as an unpatched HiGHS {expected}. It printed:\n"
            f"{probe.output.strip()[:500]}"
        )
    if probe.version != expected:
        raise ValueError(
            f"--vanilla-binary {path} is HiGHS {probe.version}, but this tree "
            f"builds against {expected} (cmake/FetchHiGHS.cmake). A baseline "
            "from a different tag is not comparable"
        )
    return probe


# How HiGHS names an option it does not have, while *reading* an options file —
# before it needs a model, which is what makes the check below cost one
# model-free run.
_UNKNOWN_OPTION_RE = re.compile(r'Option "([^"]+)" is unknown')


def check_known_options(path: str, options: dict[str, str], *, unpatched: bool) -> None:
    """Refuse `--extra-options` the binary at `path` has no option for.

    HiGHS exits 255 without solving on an unknown option name, so an
    `--extra-options` key the binary lacks fails *every instance of every
    config that runs it*: each lands in `<inst>.log.err`, that arm never
    advances, and `run_plato.sh next` relaunches a campaign that cannot
    finish.  Loud, but only per instance and only once the run is under way.

    Two ways in, and both are checked because both cost the same campaign.
    A typo (`mip_heuristic_fpr_effrot`) breaks the *patched* arm, which is
    usually the larger one.  A patched-only option breaks the *vanilla* arm,
    which since #147 is always a separately built unpatched binary with none
    of the ten options the patch adds — and that is the documented sweep
    invocation (`--extra-options mip_heuristic_fpr_effort=1.0` over the
    default `vanilla all` config pair).  `unpatched` only picks which of the
    two the message explains.

    The question is asked of the binary rather than answered from a list
    here.  Of the seventeen `mip_heuristic_*` names a patched build carries,
    **seven are upstream's own** and legal on both binaries —
    `mip_heuristic_effort` and the six `mip_heuristic_run_*` switches
    (`feasibility_jump`, `rens`, `rins`, `root_reduced_cost`, `shifting`,
    `zi_round`) — so a prefix rule would refuse a valid sweep, and a
    hardcoded list of the other ten would need editing on every option change
    and would be wrong silently when it wasn't.  The binary already knows.

    Not the same check as `check_vanilla_binary`: that one identifies the
    binary, this one is about the options *this run* pairs it with, so a
    caller with no `--extra-options` never reaches a refusal.
    """
    unknown: list[str] = []
    for key, value in sorted(options.items()):
        # One key per invocation, not all of them in one file: HiGHS stops
        # reading the file at the first unknown option, so a single run names
        # only the first offender and the operator fixes them one campaign
        # launch at a time.  Each run is model-free and costs milliseconds.
        with tempfile.NamedTemporaryFile("w", suffix=".opts", delete=False) as f:
            opts_path = f.name
        try:
            write_options_file({key: value}, opts_path)
            result = subprocess.run(
                [path, "--options_file", opts_path],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=60,
            )
            output = result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
            raise ValueError(
                f"could not run {path} to check its options: {exc}"
            ) from exc
        finally:
            os.unlink(opts_path)
        # Only "unknown", never "cannot read value": an out-of-range value is
        # a mistake on an option that does exist, it fails identically on both
        # binaries, and refusing it here would be this check overreaching into
        # HiGHS's own validation.
        if _UNKNOWN_OPTION_RE.search(output):
            unknown.append(key)
    if not unknown:
        return
    why = (
        "That binary is the unpatched vanilla baseline, so the options the "
        "patch adds do not exist there, and --extra-options applies to every "
        "config. Drop `vanilla` from --configs, or move the option to a "
        "patched-only run"
        if unpatched
        else "Check the spelling: --extra-options is passed through verbatim, "
        "and HiGHS exits 255 without solving on an unknown key, so every run "
        "of every config on this binary would fail"
    )
    raise ValueError(
        f"--extra-options {', '.join(unknown)} — {path} has no such option. {why}"
    )


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
# away, so an "FJ isolated" row would run no FeasibilityJump at all.
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
        help="Separately built UNPATCHED HiGHS of the same tag. Required by "
        "the `vanilla` config and used by nothing else: vanilla runs with no "
        "custom options at all — just time limit and seed — since an "
        "unpatched binary has no mip_heuristic_* options. The binary is "
        "probed before the first solve and refused if it carries the patch "
        "marker or reports a different version. An empty value counts as "
        "absent, so a wrapper may pass an unset variable through.",
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
            + ", ".join(sorted(KNOWN_CONFIGS))
            + ". Each selects a mip_heuristic_suite value, with `+` in a name "
            "standing for the `,` in that value; `vanilla` selects no suite "
            "value at all and requires --vanilla-binary. An unknown name is "
            "an error, not a default-option run."
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
            "OFF by default because it is still not free, though far cheaper "
            "than it was: HiGHS's own FeasibilityJump logged one fflushed line "
            "per weight bump per worker at exactly this level, which was 99.8% "
            "of a traced run's volume and most of its cost, and apply_patch.cmake "
            "now removes it (453 MB -> 8.7 MB, and 21.6x more FJ search, on a "
            "30 s run of 50v-10). What is left is FJ's periodic table plus our "
            "own two lines. A --dev-log run's rates are still not a plain run's, "
            "so use it for attribution, not for headline timings."
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

    # An empty string counts as absent, the same way `--data-dir ''` does:
    # that is a wrapper passing an unset variable through, not a path.
    vanilla_binary = None
    if args.vanilla_binary:
        vanilla_binary = os.path.abspath(args.vanilla_binary)
        if not os.path.exists(vanilla_binary):
            print(f"Error: vanilla binary not found: {vanilla_binary}", file=sys.stderr)
            sys.exit(1)
        # Just the path here, under its own label: whether it really is an
        # unpatched build of this tag is what the probe below decides, and a
        # header claiming it ahead of the check is a claim the run has not
        # earned yet.  The probe prints the verdict on the `Vanilla binary`
        # line, so the two are not two lines with one label.
        print(f"Vanilla path   : {vanilla_binary}")
    print(f"Patched binary : {binary}")

    # Resolve configs before anything else runs: an unknown name, or a
    # `vanilla` with no unpatched binary behind it, must fail here rather than
    # after producing hours of results under a name it did not honour.
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
        # Before the first solve, not after the tree exists: the probe is here
        # to prevent a mislabelled baseline rather than to describe one.  The
        # path comes off the plan rather than from `vanilla_binary`, which is
        # `str | None`: `build_plan` has already refused a vanilla plan without
        # one, so the plan's binary records that reasoning where a cast would
        # have hidden it.
        vanilla_plan = next((p for p in plans if p.base == VANILLA_CONFIG), None)
        if vanilla_plan is not None:
            probe = check_vanilla_binary(vanilla_plan.binary)
            print(
                f"Vanilla binary : probed — unpatched HiGHS {probe.version}, "
                "no custom options"
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    # Distinct names resolving to one configuration is "N identical runs under
    # N names", the thing the unknown-name raise exists to prevent, so warn
    # rather than silently burning the compute.  No pair of names does that
    # today — `vanilla` is a separate binary since #147 and every other name
    # is its own suite value — so this guards a future table edit that gives
    # one suite value two names.
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

    # The second half of the pre-flight, and it has to be here rather than
    # beside the binary probe above: it needs `base_opts`, which is built from
    # `--extra-options` after the plans are resolved.  Every distinct binary
    # this run will use, not just the vanilla one — a typo'd key fails every
    # instance of every *patched* config in exactly the same way, on the arm
    # that is usually larger.  Deduplicated because the plans share binaries;
    # each key costs one model-free run.
    try:
        for path in sorted({p.binary for p in plans}):
            check_known_options(
                path,
                base_opts,
                unpatched=vanilla_plan is not None and path == vanilla_plan.binary,
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

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
        # `--count` bounds the *work*, not the list.  Sliced off the raw list
        # it composes with `--skip-existing` to do nothing at all: a resumed
        # tree's first N instances are exactly the ones already finished, so
        # the chunk skips N runs and exits, and every later chunk repeats it.
        # A count-bounded chunk is how a campaign borrows the machine for a
        # bounded time without holding it for the whole tree, so it has to
        # advance.
        if args.skip_existing:
            instances = [
                name
                for name in instances
                if any(
                    not existing_log(args.output, plan.name, name, seed)
                    for plan in plans
                    for seed in args.seeds
                )
            ]
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
        return existing_log(args.output, config, name, seed)

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
