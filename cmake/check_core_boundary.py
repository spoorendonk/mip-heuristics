#!/usr/bin/env python3
"""Check that `mip_heuristics_core` needs no `HighsMipSolver` (#170).

The core is meant to run FJ and LocalMIP on a model its caller owns, with no
MIP solve anywhere, while the HiGHS adapter (`mip_heuristics`) is the only
code that reaches into the solver.  Two checks here, beside a third the build
makes (`mip_heuristics_core_standalone_tests` links the core against a
libhighs without the MIP solver or the adapter, so it does not link at all if
the core needs either):

1. **Includes, transitively.**  Every header a core translation unit reaches,
   as the compiler itself lists it (`-M`, with the unit's own flags from
   `compile_commands.json`), is a core header, a HiGHS header outside `mip/`,
   the header-only `mip/feasibilityjump.hh`, or a system header.  This is what
   catches an inline dependency — code that only reads a `HighsMipSolver`
   field compiles to plain offset arithmetic and leaves no symbol behind — and
   one reached through another header (`presolve/HPresolve.h` pulls in half of
   `mip/`).  Every core header must be reached by some core unit, or its own
   includes would go unchecked.

2. **Symbols, transitively.**  A simulated whole-archive link of the core
   against `libhighs.a` pulls in no member compiled from `highs/mip/*.cpp`
   and no adapter object, however indirectly (`Highs::run` lives in
   `Highs.cpp` and reaches the MIP solver from there).  The link test finds
   the same thing; this names the chain.

Registered as the ctest `core_boundary`; run standalone with the same
arguments CMakeLists.txt passes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys

_ALLOWED_MIP_HEADERS = {"feasibilityjump.hh"}
# Strong definitions only: a weak or vague-linkage symbol (an inline function
# or template instance, `W`/`V`/`u`) is emitted by every member that uses it,
# so finding one in a `mip/` member says nothing about who owns it.
_STRONG = set("TDBRG")
_WEAK = set("WVu")


def dependencies(entry: dict) -> list[pathlib.Path]:
    """Every file the unit's preprocessor reads, via its own compile flags."""
    args = entry["arguments"] if "arguments" in entry else shlex.split(entry["command"])
    kept: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "-o":
            skip = True
            continue
        if arg == "-c":
            continue
        kept.append(arg)
    out = subprocess.run(
        [*kept, "-M"],
        cwd=entry["directory"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tokens = out.replace("\\\n", " ").split(":", 1)[1].split()
    return [(pathlib.Path(entry["directory"]) / t).resolve() for t in tokens]


def include_violations(
    compile_commands: pathlib.Path,
    core_sources: list[pathlib.Path],
    src_dir: pathlib.Path,
    mip_dir: pathlib.Path,
) -> list[str]:
    core = {p.resolve() for p in core_sources}
    src_dir, mip_dir = src_dir.resolve(), mip_dir.resolve()
    units = [
        e
        for e in json.loads(compile_commands.read_text())
        if pathlib.Path(e["file"]).resolve() in core
    ]
    problems = []
    reached: set[pathlib.Path] = set()
    for entry in units:
        unit = pathlib.Path(entry["file"]).name
        for dep in dependencies(entry):
            reached.add(dep)
            if dep.parent == src_dir and dep not in core:
                problems.append(f"{unit}: reaches adapter header {dep.name}")
            elif dep.parent == mip_dir and dep.name not in _ALLOWED_MIP_HEADERS:
                problems.append(f"{unit}: reaches HiGHS MIP header mip/{dep.name}")
    for header in sorted(p for p in core if p.suffix == ".h" and p not in reached):
        problems.append(
            f"{header.name}: included by no core unit, so its includes are unchecked"
        )
    return problems


def nm_lines(nm: str, *args: str) -> list[str]:
    out = subprocess.run([nm, *args], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def archive_symbols(
    nm: str, lib: str
) -> dict[str, tuple[set[str], set[str], set[str]]]:
    """Per member, in archive order: (strong defs, weak defs, undefined refs)."""
    members: dict[str, tuple[set[str], set[str], set[str]]] = {}
    # `-A`: every line is `archive:member:value type name`, or
    # `archive:member: type name` for an undefined one.
    for line in nm_lines(nm, "-A", lib):
        head, _, rest = line.rpartition(":")
        member = head.rsplit(":", 1)[-1]
        fields = rest.split()
        if len(fields) < 2:
            continue
        kind, name = fields[-2], fields[-1]
        strong, weak, undefined = members.setdefault(member, (set(), set(), set()))
        if kind in _STRONG:
            strong.add(name)
        elif kind in _WEAK:
            weak.add(name)
        elif kind == "U":
            undefined.add(name)
    return members


def runtime_symbols(nm: str, lib: str) -> set[str]:
    """What the shared C++ runtime exports, version suffixes dropped."""
    return {
        fields[-1].split("@", 1)[0]
        for fields in (
            line.split() for line in nm_lines(nm, "-D", "--defined-only", lib)
        )
        if len(fields) >= 2
    }


def symbol_violations(
    nm: str,
    core_lib: str,
    highs_lib: str,
    runtime_lib: str,
    forbidden_members: set[str],
    verbose: bool = False,
) -> list[str]:
    """Every libhighs member a whole-archive link of the core pulls in.

    Simulates the linker over `libhighs.a`: start from every symbol the core
    archive leaves undefined — all of its members, as `--whole-archive` links
    them, not only the ones some test happens to reach — pull the member
    defining each (a strong definition first, else a weak one, as an archive
    member is extracted for either), and follow that member's own undefined
    symbols in turn.  What the shared C++ runtime exports counts as defined,
    and a weak definition comes from an allowed member when one has it: a
    template instantiation or inline function, such as `std::string`'s, is
    no dependency on whichever member happens to carry a copy, since a link
    without the forbidden members still finds it.  Transitive on purpose: a core call into `Highs::run`
    lands in `Highs.cpp`, which is not a `mip/` member, and reaches the MIP
    solver only from there.
    """
    core = archive_symbols(nm, core_lib)
    defined = set().union(*(s | w for s, w, _ in core.values()))
    defined |= runtime_symbols(nm, runtime_lib)
    highs = archive_symbols(nm, highs_lib)
    provider: dict[str, str] = {}
    for member, (_, weak, _) in sorted(
        highs.items(), key=lambda item: item[0] in forbidden_members
    ):
        for name in weak:
            provider.setdefault(name, member)
    for member, (strong, _, _) in highs.items():
        for name in strong:
            provider[name] = member
    # Why each pulled member was pulled: (the member that needed it, symbol).
    reason: dict[str, tuple[str, str]] = {}
    pending = [
        (core_member, name)
        for core_member, (_, _, undefined) in core.items()
        for name in sorted(undefined)
    ]
    while pending:
        needer, name = pending.pop()
        if name in defined or name not in provider:
            continue
        member = provider[name]
        if member in reason:
            continue
        reason[member] = (needer, name)
        strong, weak, undefined = highs[member]
        defined |= strong | weak
        pending.extend((member, sym) for sym in sorted(undefined))
    # One line per core member and symbol that starts a path into a
    # forbidden member; the full chains, demangled, after all of them.
    starts: dict[tuple[str, str], list[str]] = {}
    chains = []
    for member in sorted(m for m in reason if m in forbidden_members):
        chain, cur = [], member
        while cur in reason:
            needer, name = reason[cur]
            chain.append((needer, name))
            cur = needer
        starts.setdefault(chain[-1], []).append(member)
        chains.append((member, chain))
    names = demangle({name for _, chain in chains for _, name in chain})
    problems = [
        f"core member {core_member} needs {names[name]}, which pulls in "
        f"{len(members)} MIP-solver or adapter member(s): {', '.join(members)}"
        for (core_member, name), members in sorted(starts.items())
    ]
    if chains and verbose:
        problems.append("chains (pulled member <- what needed it):")
        problems += [
            f"  {member} <- "
            + " <- ".join(f"{needer} needs {names[name]}" for needer, name in chain)
            for member, chain in chains
        ]
    return problems


def demangle(symbols: set[str]) -> dict[str, str]:
    """`c++filt` over `symbols`; mangled names back if it is not there."""
    ordered = sorted(symbols)
    try:
        out = subprocess.run(
            ["c++filt"],
            input="\n".join(ordered),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {s: s for s in ordered}
    return (
        dict(zip(ordered, out, strict=True))
        if len(out) == len(ordered)
        else {s: s for s in ordered}
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nm", required=True)
    ap.add_argument("--core-lib", required=True)
    ap.add_argument("--highs-lib", required=True)
    ap.add_argument("--cxx-runtime", required=True, help="the shared libstdc++")
    ap.add_argument("--highs-mip-dir", required=True, type=pathlib.Path)
    ap.add_argument("--src-dir", required=True, type=pathlib.Path)
    ap.add_argument("--compile-commands", required=True, type=pathlib.Path)
    ap.add_argument("--core-sources", nargs="+", required=True, type=pathlib.Path)
    ap.add_argument("--adapter-sources", nargs="+", required=True, type=pathlib.Path)
    ap.add_argument(
        "--chains",
        action="store_true",
        help="print every symbol chain after the summary",
    )
    args = ap.parse_args()

    forbidden = {f"{p.name}.o" for p in args.highs_mip_dir.glob("*.cpp")}
    forbidden |= {f"{p.name}.o" for p in args.adapter_sources}
    problems = include_violations(
        args.compile_commands, args.core_sources, args.src_dir, args.highs_mip_dir
    )
    problems += symbol_violations(
        args.nm,
        args.core_lib,
        args.highs_lib,
        args.cxx_runtime,
        forbidden,
        verbose=args.chains,
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"core boundary: {len(args.core_sources)} files, no MIP-solver dependency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
