#!/usr/bin/env python3
"""Verify that docs/PARAMETERS.md still describes constants that exist.

The parameter reference documents ~50 tuning constants, each as a
`### \\`kName\\`` heading followed by a `- **File**: \\`src/x.h\\`` line.  Two
kinds of rot set in silently:

  1. The constant is renamed or deleted, and the entry documents something
     that is not in the tree any more (`kWeightFj` survived this way, with a
     default value that could not be read from anywhere).
  2. The file is renamed or deleted.

Entries used to carry `(line N)` too, which drifted on essentially every
refactor and is why line numbers were dropped — a symbol name is stable and
greppable, a line number is neither.  Don't reintroduce them.

Run standalone (`python bench/check_docs_refs.py`) or via the bench tests.
Exits non-zero and prints one line per problem when anything is stale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "PARAMETERS.md"

# `### `kFoo`` or `### `kFoo`, `kBar`` — a heading naming one or more constants.
# Headings with no backticked symbol (prose headings such as "### FeasibilityJump
# budget — not weight-apportioned") are still checked for a live file path.
_HEADING_RE = re.compile(r"^### (.+)$")
_FILE_RE = re.compile(r"^- \*\*File\*\*: `([^`]+)`(?:\s*\((.*)\))?\s*$")
_BACKTICK_RE = re.compile(r"`([^`]+)`")

# Scope hints that describe a location rather than name a symbol.
_NOT_SYMBOLS = {"anonymous namespace"}

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def _strip_comments(src: str) -> str:
    """Drop C++ comments so a symbol that survives only in prose doesn't count.

    Constants are routinely *discussed* in comments near their use sites, so
    searching raw text lets a renamed constant keep passing: renaming
    `kPoolCapacity` still leaves several comments naming it. Approximate —
    it does not model string literals containing `//` — but the parameter
    constants this checks are plain numeric definitions, never string content.
    """
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", src))


def _symbols(text: str) -> list[str]:
    """Backticked identifiers in `text`, minus prose hints and qualifiers."""
    out = []
    for tok in _BACKTICK_RE.findall(text):
        tok = tok.strip()
        if tok in _NOT_SYMBOLS or not tok:
            continue
        # `Foo::bar` -> check the trailing member; `field of X` handled by caller.
        tok = tok.split("::")[-1]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            out.append(tok)
    return out


def check(doc_path: Path = DOC_PATH, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return a list of human-readable problems; empty means clean."""
    problems: list[str] = []
    lines = doc_path.read_text().splitlines()

    heading: str | None = None
    heading_lineno = 0
    for i, line in enumerate(lines, 1):
        m = _HEADING_RE.match(line)
        if m:
            heading, heading_lineno = m.group(1), i
            continue

        m = _FILE_RE.match(line)
        if not m:
            continue

        rel_path, scope = m.group(1), m.group(2) or ""
        src = repo_root / rel_path
        if not src.is_file():
            problems.append(f"{doc_path.name}:{i}: file does not exist: {rel_path}")
            continue

        body = _strip_comments(src.read_text())
        wanted = _symbols(heading or "") + _symbols(scope)
        for sym in wanted:
            if not re.search(rf"\b{re.escape(sym)}\b", body):
                problems.append(
                    f"{doc_path.name}:{heading_lineno}: `{sym}` documented under "
                    f"'{heading}' but not found in {rel_path}"
                )

    # A line number in a **File**: reference is the failure mode this script
    # exists to prevent; flag any that creep back in.
    for i, line in enumerate(lines, 1):
        if line.startswith("- **File**:") and re.search(r"\blines?\s+\d+", line):
            problems.append(
                f"{doc_path.name}:{i}: **File**: reference carries a line number; "
                f"name the symbol instead (line numbers drift silently)"
            )

    return problems


def main() -> int:
    problems = check()
    if problems:
        print(f"{len(problems)} stale reference(s) in {DOC_PATH.relative_to(REPO_ROOT)}:\n")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"{DOC_PATH.relative_to(REPO_ROOT)}: all references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
