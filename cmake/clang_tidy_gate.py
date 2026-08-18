#!/usr/bin/env python3
"""clang-tidy gate for the first-party sources.

Runs clang-tidy over every `src/` and `tests/` translation unit and fails when
anything is reported against a first-party file.  Headers are covered through
the `HeaderFilterRegex` in the checked-in `.clang-tidy`.

Why this wrapper exists rather than a plain `clang-tidy ...; echo $?`:

clang-tidy's own exit status is unusable on this project.  HiGHS's
`mip/HighsMipWorker.h` defines `~HighsMipWorker()` inline, and that body calls
`reset()` on a `std::unique_ptr<HighsSearch>` whose pointee is only
forward-declared in the same header.  GCC accepts it and the project builds
clean; clang rejects it, so every translation unit that reaches
`HighsMipSolverData.h` — which is all of ours — produces one
`clang-diagnostic-error` and clang-tidy exits non-zero no matter how clean our
code is.  It cannot be fixed from outside: `HighsSearch.h` includes
`HighsMipWorker.h`, so no include order makes the type complete before that
inline destructor body is parsed.

Clang recovers from it rather than bailing (the same TU still emits its full
warning set), so the analysis is usable — but the exit code is not.  This gate
therefore judges the diagnostics itself, and separately fails if any
`clang-diagnostic-error` appears that is *not* the known upstream one, so a new
upstream parse error surfaces instead of hiding behind an expected failure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import subprocess
import sys

# The single upstream parse error described above.  Matched on the message so
# that a different error in the same header still fails the gate.
KNOWN_UPSTREAM_ERROR = "invalid application of 'sizeof' to an incomplete type 'HighsSearch'"

DIAG = re.compile(r"^(?P<path>/\S+?):(?P<line>\d+):(?P<col>\d+): "
                  r"(?P<sev>warning|error): (?P<msg>.*?) \[(?P<check>[\w.,-]+)\]$")


def run_one(clang_tidy: str, build_dir: str, source: pathlib.Path) -> str:
    proc = subprocess.run(
        [clang_tidy, "-p", build_dir, "--quiet", str(source)],
        capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clang-tidy", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    root = pathlib.Path(args.source_dir).resolve()
    sources = sorted(root.glob("src/*.cpp")) + sorted(root.glob("tests/*.cpp"))
    if not sources:
        print(f"clang-tidy gate: no first-party translation units under {root}", file=sys.stderr)
        return 1

    version = subprocess.run([args.clang_tidy, "--version"], capture_output=True,
                             text=True, check=False).stdout.strip().replace("\n", " ")

    jobs = args.jobs or None
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        outputs = list(pool.map(lambda s: run_one(args.clang_tidy, args.build_dir, s), sources))

    findings: dict[tuple[str, str, str, str], str] = {}
    unexpected_errors: set[str] = set()

    for text in outputs:
        for raw in text.splitlines():
            m = DIAG.match(raw)
            if not m:
                continue
            path = m.group("path")
            try:
                rel = str(pathlib.Path(path).resolve().relative_to(root))
            except ValueError:
                rel = path
            first_party = rel.startswith("src/") or rel.startswith("tests/")

            if m.group("check") == "clang-diagnostic-error":
                if KNOWN_UPSTREAM_ERROR not in m.group("msg"):
                    unexpected_errors.add(raw.strip())
                if not first_party:
                    continue

            if not first_party:
                continue
            key = (rel, m.group("line"), m.group("col"), m.group("check"))
            findings.setdefault(key, m.group("msg"))

    if unexpected_errors:
        print("clang-tidy gate: unexpected compiler error(s) — the analysis below ran on a "
              "degraded parse and cannot be trusted:", file=sys.stderr)
        for e in sorted(unexpected_errors):
            print(f"  {e}", file=sys.stderr)

    if findings:
        print(f"clang-tidy gate: {len(findings)} finding(s) in first-party sources "
              f"({version}):", file=sys.stderr)
        for (rel, line, col, check), msg in sorted(findings.items()):
            print(f"  {rel}:{line}:{col}: {msg} [{check}]", file=sys.stderr)

    if findings or unexpected_errors:
        return 1

    print(f"clang-tidy clean: {len(sources)} translation units ({version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
