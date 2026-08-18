#!/usr/bin/env python3
"""Package a benchmark results tree as a self-contained, verifiable archive.

The archive is what a release points at: raw logs, the exact options file each
run was given, the generated tables, and enough provenance that a reader can
tell which binary produced which row.  `verify` re-derives every table from the
archived logs alone and diffs it against the archived copy, so "regenerable" is
a checked property rather than a claim in a README.

Layout::

    <archive>/
      MANIFEST.json      machine-readable provenance, table index, checksums
      PROVENANCE.md      the same provenance, rendered for a human
      REGENERATE.sh      one-line wrapper around `make_archive.py verify .`
      bench/             analyze_results.py, parse_highs_log.py, the .solu file
                         and this script — the archive regenerates without the
                         repository
      results/           the results tree verbatim: <config>/seed<N>/<inst>.log
                         plus the <inst>.opts each run was given
      tables/            generated table output, one file per recorded command

Provenance this derives from the tree rather than taking on trust:

* **Which binary.** Patched and unpatched builds of the same HiGHS tag print
  identical version and githash banners; only the `mip-heuristics patch active`
  marker line separates them, and it is printed by a *solve*, not by
  `--version`.  Every log is checked for it, so a config's binary is a fact
  read off the logs.
* **Which baseline.** "vanilla-equivalent setting on the patched binary" and
  "separately built unpatched binary" are different claims — the first rests on
  `bench/check_vanilla_equivalence.py`, the second on nothing but the build.
  The marker line tells them apart and the manifest states which one this
  archive supports.
* **Instrumentation state.** `log_dev_level=3` costs 97-750x the log volume and
  1.1-4.4x the wall time, concentrated in the FeasibilityJump phase, so
  attribution runs and headline-timing runs are different runs.  Both the
  requested state (the options file) and the observed state (the `[Heur]` /
  `[Native]` / `[Root]` tags in the log) are recorded, and a disagreement is a
  warning rather than a silent mis-labelling.
* **Thread count.** Several ratios in this project are strongly
  thread-count-dependent — the same binary on the same instances gives
  `local_mip:scylla = 4.68` at 16 workers and `2.81` at 6 — so a tree with no
  recorded thread count cannot be interpreted.  The benchmark harness
  deliberately does *not* set `threads`, which means the effective count is the
  run machine's core count; that is why an unset `threads` raises a warning
  naming `--machine-note`.

Usage::

    bench/make_archive.py build bench/results/plato --output dist/v1.0-archive \\
        --time-limit 600 --configs off all --tar
    bench/make_archive.py verify dist/v1.0-archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent

# Copied into `<archive>/bench/` so the archive regenerates its tables with no
# checkout present.  `analyze_results.py` imports `parse_highs_log` from its own
# directory and defaults `--solu` to the `.solu` beside it, so the three have to
# travel together; this script joins them because `verify` is what REGENERATE.sh
# runs.
ARCHIVE_BENCH_FILES = (
    "analyze_results.py",
    "parse_highs_log.py",
    "miplib2017-v22.solu",
    "make_archive.py",
)

# The only thing that distinguishes a patched binary from an unpatched build of
# the same tag.  Inserted into HighsIO.cpp by
# `third_party/highs_patch/apply_patch.cmake`; the string is duplicated in
# `bench/check_vanilla_equivalence.py`, which normalises it away.
PATCH_MARKER = "mip-heuristics patch active"

_BANNER_RE = re.compile(r"Running HiGHS (\S+) \(git hash: (\w+)\)")

# `log_dev_level=3` tags, from `src/effort_ledger.cpp` and
# `heuristics::log_solve_summary`.  Their presence is the observed
# instrumentation state, as against the `log_dev_level` the options file asked
# for.
INSTRUMENTATION_TAGS = ("[Heur] ", "[Native] ", "[Root] ", "[Sequential] ")

# Provenance sources outside the results tree.
HIGHS_TAG_RE = re.compile(r"GIT_TAG\s+(\S+)")
PATCH_VERSION_RE = re.compile(r'set\(PATCH_VERSION\s+"([^"]+)"\)')
TOOL_PIN_RE = re.compile(r"\b(clang-format|clang-tidy|ruff)==(\S+)")

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_VERSION = 1

# Config names that stand for "no custom heuristic", most-preferred first.
# Same order `analyze_results.py` auto-detects a cannibalization baseline in,
# and the order `discover_configs` sorts an auto-discovered tree into.
BASELINE_NAMES = ("vanilla", "off", "suite_off", "baseline")

DEFAULT_REPOSITORY = "https://github.com/spoorendonk/mip-heuristics"


# ── provenance records ───────────────────────────────────────────────────────


@dataclass
class ConfigProvenance:
    """What one `<config>/` directory of a results tree actually contains."""

    name: str
    binary: str  # patched | unpatched | unknown | mixed
    highs_version: str | None
    highs_git_hash: str | None
    seeds: list[int]
    instances: list[str]
    runs: int
    failed_runs: list[str]
    options: dict[str, str]
    option_variants: list[dict[str, str]]
    runs_without_options: int
    runs_without_banner: int
    instrumentation_requested: bool
    instrumentation_observed: bool


@dataclass
class TableSpec:
    """One published table: the command that makes it and where it landed."""

    name: str
    argv: list[str]
    path: str = ""
    sha256: str = ""


@dataclass
class Manifest:
    manifest_version: int
    created_utc: str
    archive_name: str
    note: str
    source: dict[str, object]
    machine: dict[str, object]
    run: dict[str, object]
    baseline: dict[str, object]
    configs: list[ConfigProvenance]
    tables: list[TableSpec]
    warnings: list[str]
    files: dict[str, str] = field(default_factory=dict)


# ── log inspection ───────────────────────────────────────────────────────────


def inspect_log(path: Path) -> tuple[bool, str | None, str | None, bool]:
    """Read one log for (patched, highs_version, git_hash, instrumented).

    Streams rather than slurping: a `--dev-log` tree runs to megabytes per log
    and a PLATO campaign has hundreds of them.  The banner is on the first
    line, so the loop stops as soon as the instrumentation question is also
    settled.
    """
    patched = False
    version: str | None = None
    git_hash: str | None = None
    instrumented = False
    with path.open(errors="replace") as handle:
        for line in handle:
            if version is None:
                m = _BANNER_RE.search(line)
                if m:
                    version, git_hash = m.group(1), m.group(2)
                    continue
            if not patched and PATCH_MARKER in line:
                patched = True
                continue
            if not instrumented and any(t in line for t in INSTRUMENTATION_TAGS):
                instrumented = True
            if patched and instrumented and version is not None:
                break
    return patched, version, git_hash, instrumented


def read_options_file(path: Path) -> dict[str, str]:
    """Parse a HiGHS options file (`key = value` per line) into a dict."""
    options: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            options[key.strip()] = value.strip()
    return options


def discover_configs(results_dir: Path) -> list[str]:
    """Config directory names under `results_dir`, baseline first.

    A config directory is one holding `seed<N>/` subdirectories — the shape
    `bench/run_benchmark.py` writes.  Anything else in the tree (a stray
    `tables/`, an editor backup) is skipped rather than archived as a config
    with zero runs.

    The order is not cosmetic: it is the order `analyze_results.py` receives
    `--configs` in, which decides column order in every generated table.  A
    baseline config sorts first so an auto-discovered archive reads
    baseline-then-variants rather than alphabetically.
    """
    found = []
    for child in sorted(results_dir.iterdir()):
        if child.is_dir() and any(
            g.is_dir() and g.name.startswith("seed") for g in child.iterdir()
        ):
            found.append(child.name)
    rank = {name: i for i, name in enumerate(BASELINE_NAMES)}
    return sorted(found, key=lambda n: (rank.get(n, len(rank)), n))


def _seed_number(seed_dir: Path) -> int | None:
    suffix = seed_dir.name[len("seed") :]
    return int(suffix) if suffix.isdigit() else None


def collect_config(results_dir: Path, name: str) -> ConfigProvenance:
    """Derive one config's provenance from its logs and options files."""
    config_dir = results_dir / name
    seeds: list[int] = []
    instances: set[str] = set()
    failed: list[str] = []
    runs = 0
    binaries: set[str] = set()
    versions: set[str] = set()
    hashes: set[str] = set()
    observed = False
    missing_options = 0
    missing_banner = 0
    variants: list[dict[str, str]] = []

    for seed_dir in sorted(config_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed = _seed_number(seed_dir)
        if seed is None:
            continue
        seeds.append(seed)
        for err in sorted(seed_dir.glob("*.log.err")):
            failed.append(f"{seed_dir.name}/{err.name}")
        for log in sorted(seed_dir.glob("*.log")):
            runs += 1
            instances.add(log.stem)
            patched, version, git_hash, instrumented = inspect_log(log)
            if version is None:
                # No banner at all: a truncated or hand-assembled log. Absence
                # of the marker is what "unpatched" means, so a log that proves
                # nothing either way must not be counted as the *stronger*
                # baseline claim by default — it is counted here instead.
                missing_banner += 1
            else:
                binaries.add("patched" if patched else "unpatched")
                versions.add(version)
            if git_hash:
                hashes.add(git_hash)
            observed = observed or instrumented
            opts = seed_dir / f"{log.stem}.opts"
            if not opts.is_file():
                missing_options += 1
            else:
                # `random_seed` is per-run by construction (it names the
                # directory), so it is not part of "did every run of this
                # config get the same options?".
                recorded = {
                    k: v
                    for k, v in read_options_file(opts).items()
                    if k != "random_seed"
                }
                if recorded not in variants:
                    variants.append(recorded)

    if len(binaries) > 1:
        binary = "mixed"
    elif binaries:
        binary = binaries.pop()
    else:
        binary = "unknown"

    options = variants[0] if len(variants) == 1 else {}
    requested = any(v.get("log_dev_level") == "3" for v in variants)

    return ConfigProvenance(
        name=name,
        binary=binary,
        highs_version=min(versions) if len(versions) == 1 else None,
        highs_git_hash=min(hashes) if len(hashes) == 1 else None,
        seeds=sorted(set(seeds)),
        instances=sorted(instances),
        runs=runs,
        failed_runs=failed,
        options=options,
        option_variants=variants if len(variants) > 1 else [],
        runs_without_options=missing_options,
        runs_without_banner=missing_banner,
        instrumentation_requested=requested,
        instrumentation_observed=observed,
    )


# ── repository and machine provenance ────────────────────────────────────────


def _git(repo_root: Path, *args: str) -> str | None:
    """Run a read-only git command in `repo_root`, or None if it cannot."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def source_provenance(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Repository state, the pinned HiGHS tag, and the lint tool pins.

    The HiGHS tag and `PATCH_VERSION` together identify the solver a binary was
    built from: the tag says which upstream tree, the patch version says which
    revision of our inserted text it carries.  Neither is visible in a log —
    the banner shows only the upstream tag — so they have to come from the
    checkout that produced the binary.
    """
    fetch = repo_root / "cmake" / "FetchHiGHS.cmake"
    patch = repo_root / "third_party" / "highs_patch" / "apply_patch.cmake"
    workflow = repo_root / ".github" / "workflows" / "ci.yml"

    highs_tag = None
    if fetch.is_file():
        m = HIGHS_TAG_RE.search(fetch.read_text())
        highs_tag = m.group(1) if m else None

    patch_version = None
    if patch.is_file():
        m = PATCH_VERSION_RE.search(patch.read_text())
        patch_version = m.group(1) if m else None

    tool_pins: dict[str, str] = {}
    if workflow.is_file():
        for tool, version in TOOL_PIN_RE.findall(workflow.read_text()):
            tool_pins[tool] = version

    status = _git(repo_root, "status", "--porcelain")
    return {
        "repository": _git(repo_root, "remote", "get-url", "origin")
        or DEFAULT_REPOSITORY,
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "describe": _git(repo_root, "describe", "--tags", "--always", "--dirty"),
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "highs_tag": highs_tag,
        "patch_version": patch_version,
        "tool_pins": tool_pins,
    }


def _first_field(path: str, key: str) -> str | None:
    try:
        with open(path, errors="replace") as handle:
            for line in handle:
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def machine_provenance(note: str) -> dict[str, object]:
    """Describe the host, flagged as auto-detected on the *archiving* machine.

    Nothing in a HiGHS log identifies the machine it ran on, so this is an
    assumption whenever the archive is built somewhere other than where the
    campaign ran.  `source` says so out loud and `--machine-note` is how the
    archiver corrects it.
    """
    mem = _first_field("/proc/meminfo", "MemTotal")
    memory_gb = None
    if mem:
        parts = mem.split()
        if parts and parts[0].isdigit():
            memory_gb = round(int(parts[0]) / (1024 * 1024), 1)
    return {
        "source": "auto-detected on the archive host"
        if not note
        else "auto-detected on the archive host, annotated with --machine-note",
        "hostname": socket.gethostname(),
        "cpu_model": _first_field("/proc/cpuinfo", "model name"),
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "note": note,
    }


# ── baseline classification ──────────────────────────────────────────────────

BASELINE_CLAIMS = {
    "unpatched": (
        "separately built unpatched binary",
        (
            "The baseline logs carry no `mip-heuristics patch active` marker, so "
            "they came from a build without the patch. This is the stronger claim: "
            "it rests on the second build, not on an equivalence argument."
        ),
    ),
    "patched": (
        "vanilla-equivalent setting on the patched binary",
        (
            "The baseline logs carry the `mip-heuristics patch active` marker, so "
            "the baseline is `mip_heuristic_suite=off` on the patched build. That "
            "is vanilla-equivalent rather than vanilla, and the equivalence is "
            "what `bench/check_vanilla_equivalence.py` proves — cite that check, "
            "not the build, when reporting these rows."
        ),
    ),
}


def classify_baseline(configs: list[ConfigProvenance]) -> dict[str, object]:
    """Name the baseline config and state exactly which claim it supports."""
    by_name = {c.name: c for c in configs}
    for candidate in BASELINE_NAMES:
        if candidate in by_name:
            config = by_name[candidate]
            claim, evidence = BASELINE_CLAIMS.get(
                config.binary,
                (
                    "indeterminate",
                    (
                        "The baseline logs disagree about the patch marker, so the "
                        "binary behind them cannot be established."
                    ),
                ),
            )
            return {
                "config": config.name,
                "binary": config.binary,
                "claim": claim,
                "evidence": evidence,
            }
    return {
        "config": None,
        "binary": None,
        "claim": "none",
        "evidence": (
            "No config named "
            + ", ".join(BASELINE_NAMES)
            + " is present, so the archive carries no baseline row."
        ),
    }


# ── table specs ──────────────────────────────────────────────────────────────


def default_table_specs(
    configs: list[str], time_limit: float, instrumented: bool
) -> list[TableSpec]:
    """The table set a tree of this shape supports.

    Two configs is the pairwise/PLATO shape; three or more is the ablation
    shape, the same split `bench/run_benchmark.py` prints at the end of a run.
    The cannibalization table is offered only for an instrumented tree, because
    without `log_dev_level=3` every row classifies as `not-instrumented` — an
    empty table, archived as though it said something.
    """
    base = ["--configs", *configs, "--time-limit", f"{time_limit:g}"]
    ablation = len(configs) > 2
    headline = ["--ablation"] if ablation else ["--baseline"]
    # `--summary` implies `--baseline` and drops the per-instance table, which
    # would otherwise be repeated in full in every extra table file.
    compact = ["--ablation"] if ablation else ["--summary"]

    specs = [
        TableSpec(name="summary", argv=[*base, *headline]),
        TableSpec(name="attribution", argv=[*base, *compact, "--attribution"]),
    ]
    if instrumented:
        specs.append(
            TableSpec(
                name="cannibalization", argv=[*base, *compact, "--cannibalization"]
            )
        )
    return specs


def parse_table_flag(value: str) -> TableSpec:
    """Parse `--table 'name=--ablation --time-limit 600'` into a spec."""
    name, sep, args = value.partition("=")
    name = name.strip()
    if not sep or not name:
        raise ValueError(f"--table {value!r} must be NAME=ARGS, e.g. 'sgm=--summary'")
    argv = shlex.split(args)
    if not argv:
        raise ValueError(f"--table {value!r} has no arguments after '='")
    return TableSpec(name=name, argv=argv)


def render_table(archive: Path, spec: TableSpec, out_dir: Path) -> tuple[Path, str]:
    """Run one table command against the archive's own copies.

    cwd is the archive root and the script is the archived one, so the argv in
    the manifest is literally the command a reader runs — there is no second,
    subtly different invocation recorded anywhere.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{spec.name}.txt"
    cmd = [sys.executable, "bench/analyze_results.py", "results", *spec.argv]
    proc = subprocess.run(cmd, cwd=archive, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"table {spec.name!r} failed ({' '.join(shlex.quote(c) for c in cmd)}) "
            f"with exit {proc.returncode}:\n{proc.stderr.strip()}"
        )
    target.write_text(proc.stdout)
    return target, sha256_file(target)


# ── checksums ────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_tree(archive: Path) -> dict[str, str]:
    """sha256 of every archived file except the manifest itself.

    The manifest is excluded because it is what carries the checksums; hashing
    it into itself is not a thing that can be done, and hashing it separately
    would only record that the file has not changed since it was written.
    """
    sums: dict[str, str] = {}
    for path in sorted(archive.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(archive).as_posix()
        if rel == MANIFEST_NAME:
            continue
        sums[rel] = sha256_file(path)
    return sums


# ── PROVENANCE.md ────────────────────────────────────────────────────────────


def _yes_no(value: object) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def render_provenance(manifest: Manifest) -> str:
    """Render the manifest as the document a reader actually reads."""
    src = manifest.source
    machine = manifest.machine
    run = manifest.run
    lines: list[str] = [
        f"# Provenance — {manifest.archive_name}",
        "",
        (
            "Generated by `bench/make_archive.py`. `MANIFEST.json` carries the same "
            "facts in machine-readable form, plus a sha256 of every archived file."
        ),
        "",
    ]
    if manifest.note:
        lines += [f"> {manifest.note}", ""]

    lines += [
        "## Source",
        "",
        "| | |",
        "|---|---|",
        f"| Repository | {src['repository']} |",
        f"| Commit | `{src['commit']}` |",
        f"| Describe | `{src['describe']}` |",
        f"| Working tree clean | {_yes_no(not src['dirty'] if src['dirty'] is not None else None)} |",
        f"| Upstream solver | HiGHS `{src['highs_tag']}` (fetched and patched at configure time) |",
        f"| Patch version | `{src['patch_version']}` (`PATCH_VERSION` in `third_party/highs_patch/apply_patch.cmake`) |",
        f"| Lint tool pins | {', '.join(f'`{k}=={v}`' for k, v in sorted(src['tool_pins'].items())) or 'none recorded'} |",
        "",
        (
            "A patched and an unpatched build of the same tag print **identical** "
            "version and githash banners. The only thing that separates them is the "
            f"`{PATCH_MARKER}` line, printed by a solve rather than by `--version`, "
            "and every log below was checked for it."
        ),
        "",
        "## Machine",
        "",
        "| | |",
        "|---|---|",
        f"| Source | {machine['source']} |",
        f"| Host | {machine['hostname']} |",
        f"| CPU | {machine['cpu_model']} |",
        f"| Cores visible | {machine['cpu_count']} |",
        f"| Memory | {machine['memory_gb']} GB |",
        f"| Platform | {machine['platform']} |",
        f"| Kernel | {machine['kernel']} |",
    ]
    if machine["note"]:
        lines.append(f"| Note | {machine['note']} |")
    lines += [
        "",
        "## Run",
        "",
        "| | |",
        "|---|---|",
        f"| Time limit | {run['time_limit_s']:g} s per instance |",
        f"| `threads` option | {run['threads_option'] or 'unset — HiGHS default'} |",
        f"| Seeds | {', '.join(str(s) for s in run['seeds']) or 'none'} |",
        f"| Instances | {run['instances']} |",
        f"| Instrumented (`log_dev_level=3`) | {_yes_no(run['instrumented'])} |",
        "",
        (
            "Thread count is load-bearing, not decoration: throughput ratios in this "
            "project do not cancel across worker counts — the same binary on the "
            "same instances gives `local_mip:scylla = 4.68` at 16 workers and "
            "`2.81` at 6."
        ),
        "",
        (
            "Instrumentation state is equally load-bearing. `log_dev_level=3` costs "
            "97-750x the log volume and 1.1-4.4x the wall time, concentrated in the "
            "FeasibilityJump phase, so an attribution run and a headline-timing run "
            "are different runs and their timings are not comparable."
        ),
        "",
        "## Baseline",
        "",
        f"- Config: `{manifest.baseline['config']}`",
        f"- Binary: {manifest.baseline['binary']}",
        f"- Claim: **{manifest.baseline['claim']}**",
        "",
        str(manifest.baseline["evidence"]),
        "",
        "## Configurations",
        "",
        "| Config | Binary | HiGHS banner | Runs | Seeds | Instances | Instrumented | Options |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for config in manifest.configs:
        banner = (
            f"{config.highs_version} / {config.highs_git_hash}"
            if config.highs_version
            else "mixed or absent"
        )
        opts = ", ".join(f"`{k}={v}`" for k, v in sorted(config.options.items())) or (
            "varies — see MANIFEST.json" if config.option_variants else "none"
        )
        lines.append(
            f"| `{config.name}` | {config.binary} | {banner} | {config.runs} | "
            f"{', '.join(str(s) for s in config.seeds)} | {len(config.instances)} | "
            f"{_yes_no(config.instrumentation_observed)} | {opts} |"
        )

    lines += [
        "",
        (
            "Every run's exact options file is archived verbatim beside its log as "
            "`results/<config>/seed<N>/<instance>.opts`; the table above is a "
            "summary of them, not the record."
        ),
        "",
        "## Tables",
        "",
        (
            "Each was produced by running the archived `bench/analyze_results.py` "
            "against `results/`, from the archive root:"
        ),
        "",
    ]
    for spec in manifest.tables:
        rendered = " ".join(shlex.quote(a) for a in spec.argv)
        lines += [
            f"- `{spec.path}`",
            "  ```",
            f"  python3 bench/analyze_results.py results {rendered}",
            "  ```",
        ]
    lines += [
        "",
        (
            "`./REGENERATE.sh` re-runs all of them and diffs the output against the "
            "archived copies, verifying the checksums on the way."
        ),
        "",
    ]

    if manifest.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in manifest.warnings]
        lines.append("")
    return "\n".join(lines)


REGENERATE_SH = """#!/usr/bin/env bash
# Regenerate every table in this archive from the archived logs, and verify
# the archived checksums.  Delegates to the archived copy of make_archive.py
# so there is exactly one implementation of "regenerate and diff".
#
#   ./REGENERATE.sh              # regenerate into a temporary directory
#   ./REGENERATE.sh out          # regenerate into ./out and keep it
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -gt 0 ]; then
    exec "${PYTHON:-python3}" bench/make_archive.py verify . --write "$1"
fi
exec "${PYTHON:-python3}" bench/make_archive.py verify .
"""


# ── build ────────────────────────────────────────────────────────────────────


def copy_results(results_dir: Path, archive: Path, configs: list[str]) -> None:
    """Copy the selected configs verbatim, logs and options files alike."""
    for name in configs:
        shutil.copytree(results_dir / name, archive / "results" / name)


def build_archive(
    results_dir: Path,
    archive: Path,
    *,
    configs: list[str],
    time_limit: float,
    note: str,
    machine_note: str,
    extra_tables: list[TableSpec],
    repo_root: Path = REPO_ROOT,
    bench_dir: Path = BENCH_DIR,
) -> Manifest:
    """Assemble the archive and return its manifest. Raises on a failed table."""
    if archive.exists():
        raise FileExistsError(
            f"{archive} already exists — archives are written once so a "
            f"half-updated one cannot be mistaken for a complete one"
        )

    provenance = [collect_config(results_dir, name) for name in configs]
    warnings: list[str] = []

    for config in provenance:
        if config.binary == "mixed":
            raise ValueError(
                f"config {config.name!r} mixes patched and unpatched logs — the "
                f"patch marker is present in some and absent in others, so the "
                f"binary behind those rows cannot be established"
            )
        if config.binary == "unknown":
            warnings.append(
                f"config {config.name!r} has no log carrying a HiGHS banner, "
                f"so none of it can be attributed to a binary — check whether "
                f"the directory is empty or every run failed (.log.err)"
            )
        if config.runs_without_banner:
            warnings.append(
                f"config {config.name!r} has {config.runs_without_banner} "
                f"log(s) with no HiGHS banner; they prove nothing about which "
                f"binary ran and are excluded from that determination"
            )
        if config.runs_without_options:
            warnings.append(
                f"config {config.name!r} has {config.runs_without_options} "
                f"run(s) with no `.opts` file beside the log, so the options "
                f"those runs were given are not in the archive — an empty "
                f"Options column does not distinguish that from a run that was "
                f"given no options"
            )
        if config.failed_runs:
            warnings.append(
                f"config {config.name!r} has {len(config.failed_runs)} failed "
                f"run(s) (.log.err); they are archived as evidence and are not "
                f"part of any table"
            )
        if config.instrumentation_requested != config.instrumentation_observed:
            warnings.append(
                f"config {config.name!r} asked for log_dev_level="
                f"{'3' if config.instrumentation_requested else 'off'} but its "
                f"logs {'do not carry' if config.instrumentation_requested else 'carry'} "
                f"the [Heur]/[Native]/[Root] tags"
            )
        if config.option_variants:
            warnings.append(
                f"config {config.name!r} was not run with one option set — "
                f"{len(config.option_variants)} distinct sets appear across its "
                f"runs; see MANIFEST.json"
            )

    instrumented = all(c.instrumentation_observed for c in provenance) and bool(
        provenance
    )
    # Read from the option *variants* too, not only from `options`: a config
    # whose runs disagree about anything at all reports `options == {}`, and a
    # false "threads unset" is worse than a missing one — PROVENANCE.md states
    # it as a fact, and thread count is the field the whole document calls
    # load-bearing.
    threads = next(
        (
            opts["threads"]
            for c in provenance
            for opts in ([c.options] if c.options else c.option_variants)
            if opts.get("threads")
        ),
        None,
    )
    if threads is None:
        warnings.append(
            "no `threads` in any options file, so HiGHS used its default and "
            "the effective worker count is the *run* machine's core count — "
            "which is recoverable from the machine block only if the archive "
            "was built on that machine"
        )
    if not machine_note:
        warnings.append(
            "no --machine-note: the machine block describes the archive host, "
            "which is only the benchmark host if they are the same machine"
        )

    seeds = sorted({s for c in provenance for s in c.seeds})
    instances = sorted({i for c in provenance for i in c.instances})

    # Every cheap precondition is settled before the copy: a campaign tree runs
    # to gigabytes, and a `--table` name typo should not cost that copy and
    # then leave a partial archive that the write-once guard blocks the retry
    # on.
    specs = default_table_specs(configs, time_limit, instrumented)
    known = {s.name for s in specs}
    for extra in extra_tables:
        if extra.name in known:
            raise ValueError(
                f"--table {extra.name!r} collides with a default table of the "
                f"same name; pick another name"
            )
        known.add(extra.name)
        specs.append(extra)

    archive.mkdir(parents=True)
    copy_results(results_dir, archive, configs)
    (archive / "bench").mkdir()
    for name in ARCHIVE_BENCH_FILES:
        shutil.copy2(bench_dir / name, archive / "bench" / name)

    try:
        for spec in specs:
            target, digest = render_table(archive, spec, archive / "tables")
            spec.path = target.relative_to(archive).as_posix()
            spec.sha256 = digest
    except RuntimeError:
        # A half-built archive is a by-product, not output; leaving it turns
        # the write-once guard into an obstacle on the retry.
        shutil.rmtree(archive, ignore_errors=True)
        raise

    regenerate = archive / "REGENERATE.sh"
    regenerate.write_text(REGENERATE_SH)
    regenerate.chmod(0o755)

    manifest = Manifest(
        manifest_version=MANIFEST_VERSION,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        archive_name=archive.name,
        note=note,
        source=source_provenance(repo_root),
        machine=machine_provenance(machine_note),
        run={
            "time_limit_s": time_limit,
            "threads_option": threads,
            "seeds": seeds,
            "instances": len(instances),
            "instrumented": instrumented,
        },
        baseline=classify_baseline(provenance),
        configs=provenance,
        tables=specs,
        warnings=warnings,
    )

    (archive / "PROVENANCE.md").write_text(render_provenance(manifest))
    # Checksums last: PROVENANCE.md and REGENERATE.sh are archived files too,
    # and an archive whose own documents are outside the checksum set is one
    # where the provenance can be edited without `verify` noticing.
    manifest.files = checksum_tree(archive)
    write_manifest(archive, manifest)
    return manifest


def write_manifest(archive: Path, manifest: Manifest) -> None:
    payload = asdict(manifest)
    (archive / MANIFEST_NAME).write_text(json.dumps(payload, indent=2) + "\n")


def make_tarball(archive: Path) -> Path:
    """Write `<archive>.tar.gz` beside the archive directory.

    Appends to the whole name rather than going through `with_suffix`, which
    treats the `.0` of a version-numbered directory as the suffix to replace.
    A single tarball is also the only shape Zenodo accepts for a results tree:
    a record takes at most 100 files, and a campaign has thousands.
    """
    tar_path = Path(str(archive) + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(archive, arcname=archive.name)
    return tar_path


# ── verify ───────────────────────────────────────────────────────────────────


def verify_archive(archive: Path, write: Path | None) -> list[str]:
    """Recompute checksums and re-derive every table. Returns problem lines."""
    manifest_path = archive / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"{MANIFEST_NAME} is missing — this is not an archive directory"]
    payload = json.loads(manifest_path.read_text())
    problems: list[str] = []

    recorded: dict[str, str] = payload.get("files", {})
    actual = checksum_tree(archive)
    # Anything regenerated into the archive by an earlier `verify --write` is a
    # by-product, not archive content; excluding it keeps a verify run from
    # making the next one fail.
    extra_roots = {write.name} if write is not None else set()
    for rel, digest in sorted(recorded.items()):
        if rel not in actual:
            problems.append(f"missing file: {rel}")
        elif actual[rel] != digest:
            problems.append(f"checksum mismatch: {rel}")
    for rel in sorted(set(actual) - set(recorded)):
        if rel.split("/", 1)[0] in extra_roots:
            continue
        problems.append(f"unrecorded file present: {rel}")

    out_dir = write
    with tempfile.TemporaryDirectory() as tmp:
        if out_dir is None:
            out_dir = Path(tmp)
        for entry in payload.get("tables", []):
            spec = TableSpec(name=entry["name"], argv=list(entry["argv"]))
            archived = archive / entry["path"]
            if not archived.is_file():
                problems.append(f"table {spec.name}: archived output is missing")
                continue
            try:
                regenerated, digest = render_table(archive, spec, out_dir)
            except RuntimeError as exc:
                problems.append(f"table {spec.name}: {exc}")
                continue
            if digest != entry["sha256"]:
                problems.append(
                    f"table {spec.name}: regenerated output differs from the "
                    f"archived copy ({archived} vs {regenerated})"
                )
    return problems


# ── CLI ──────────────────────────────────────────────────────────────────────


def _add_build_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build", help="package a results tree as a verifiable release archive"
    )
    p.add_argument("results", help="results tree, e.g. bench/results/plato")
    p.add_argument("--output", required=True, help="archive directory to create")
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="config directories to archive (default: every one found). The "
        "order is the order analyze_results.py receives them, so it decides "
        "column order in the generated tables.",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        required=True,
        help="per-instance time limit the campaign ran at. Required: it is a "
        "command-line argument to HiGHS rather than an options-file entry, so "
        "it is not recoverable from the archived .opts files.",
    )
    p.add_argument("--note", default="", help="one-line note about this archive")
    p.add_argument(
        "--machine-note",
        default="",
        help="describe the machine the campaign ran on. The machine block is "
        "auto-detected on the archive host, which is an assumption whenever "
        "the archive is not built where the runs happened.",
    )
    p.add_argument(
        "--table",
        action="append",
        default=[],
        metavar="NAME=ARGS",
        help="extra table beyond the defaults, e.g. "
        "--table 'sgm=--summary --configs off all --time-limit 600'. "
        "Repeatable.",
    )
    p.add_argument(
        "--tar", action="store_true", help="also write <output>.tar.gz beside it"
    )


def _add_verify_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "verify", help="recompute checksums and re-derive every archived table"
    )
    p.add_argument("archive", help="archive directory to verify")
    p.add_argument(
        "--write",
        default=None,
        metavar="DIR",
        help="keep the regenerated tables in DIR instead of a temporary "
        "directory (relative paths resolve inside the archive)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    _add_build_parser(sub)
    _add_verify_parser(sub)
    args = parser.parse_args(argv)

    if args.command == "verify":
        archive = Path(args.archive).resolve()
        write = (archive / args.write).resolve() if args.write else None
        problems = verify_archive(archive, write)
        if problems:
            print(f"{archive}: {len(problems)} problem(s)", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{archive}: checksums match and every table regenerates identically")
        return 0

    results_dir = Path(args.results).resolve()
    if not results_dir.is_dir():
        print(f"Error: no such results tree: {results_dir}", file=sys.stderr)
        return 2
    available = discover_configs(results_dir)
    configs = args.configs if args.configs is not None else available
    missing = [c for c in configs if c not in available]
    if missing:
        print(
            f"Error: config(s) {', '.join(missing)} are not in {results_dir} "
            f"(found: {', '.join(available) or 'none'})",
            file=sys.stderr,
        )
        return 2
    if not configs:
        print(f"Error: {results_dir} holds no config directories", file=sys.stderr)
        return 2

    try:
        extra = [parse_table_flag(t) for t in args.table]
        manifest = build_archive(
            results_dir,
            Path(args.output).resolve(),
            configs=configs,
            time_limit=args.time_limit,
            note=args.note,
            machine_note=args.machine_note,
            extra_tables=extra,
        )
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    archive = Path(args.output).resolve()
    print(f"Archive        : {archive}")
    print(f"Configs        : {' '.join(c.name for c in manifest.configs)}")
    print(
        f"Baseline       : {manifest.baseline['config']} — {manifest.baseline['claim']}"
    )
    print(f"Tables         : {' '.join(t.name for t in manifest.tables)}")
    print(f"Files checksum'd: {len(manifest.files)}")
    for warning in manifest.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if args.tar:
        print(f"Tarball        : {make_tarball(archive)}")
    print(f"\nVerify with:\n  {archive}/REGENERATE.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
