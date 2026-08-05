"""Tests for check_docs_refs."""

from pathlib import Path

from check_docs_refs import check


def _write(tmp_path: Path, doc: str, sources: dict[str, str]) -> tuple[Path, Path]:
    root = tmp_path
    (root / "src").mkdir(parents=True, exist_ok=True)
    for name, body in sources.items():
        (root / name).write_text(body)
    doc_path = root / "PARAMETERS.md"
    doc_path.write_text(doc)
    return doc_path, root


def test_live_reference_is_clean(tmp_path):
    doc, root = _write(
        tmp_path,
        "### `kAlpha` — blending factor\n\n- **File**: `src/scylla.h`\n",
        {"src/scylla.h": "inline constexpr double kAlpha = 0.9;\n"},
    )
    assert check(doc, root) == []


def test_deleted_constant_is_reported(tmp_path):
    """The kWeightFj failure mode: entry outlives the constant it documents."""
    doc, root = _write(
        tmp_path,
        "### `kWeightFj` — FeasibilityJump budget weight\n\n"
        "- **File**: `src/mode_dispatch.cpp`\n",
        {"src/mode_dispatch.cpp": "constexpr double kWeightFpr = 2.43;\n"},
    )
    problems = check(doc, root)
    assert len(problems) == 1
    assert "kWeightFj" in problems[0]
    assert "not found in src/mode_dispatch.cpp" in problems[0]


def test_symbol_surviving_only_in_comments_is_reported(tmp_path):
    """A renamed constant usually keeps being named in nearby comments.

    Searching raw text would let those mentions satisfy the check, which is
    exactly how a rename slips through.
    """
    doc, root = _write(
        tmp_path,
        "### `kPoolCapacity` — pool size\n\n- **File**: `src/solution_pool.h`\n",
        {
            "src/solution_pool.h": (
                "inline constexpr int kPoolSize = 10;\n"
                "// copies the whole pool including up to kPoolCapacity - 1 entries\n"
                "/* historical: kPoolCapacity was the old name */\n"
            )
        },
    )
    problems = check(doc, root)
    assert len(problems) == 1
    assert "kPoolCapacity" in problems[0]


def test_missing_file_is_reported(tmp_path):
    doc, root = _write(
        tmp_path,
        "### `kGone` — something\n\n- **File**: `src/deleted.h`\n",
        {},
    )
    problems = check(doc, root)
    assert len(problems) == 1
    assert "file does not exist: src/deleted.h" in problems[0]


def test_line_number_in_reference_is_reported(tmp_path):
    """Line numbers drift silently, so they are rejected outright."""
    doc, root = _write(
        tmp_path,
        "### `kAlpha` — blending factor\n\n- **File**: `src/scylla.h` (line 12)\n",
        {"src/scylla.h": "inline constexpr double kAlpha = 0.9;\n"},
    )
    problems = check(doc, root)
    assert len(problems) == 1
    assert "carries a line number" in problems[0]


def test_scope_hint_symbols_are_checked_and_prose_hints_ignored(tmp_path):
    """`(anonymous namespace)` is prose; a backticked hint names a real symbol."""
    doc, root = _write(
        tmp_path,
        "### `kCap` — cap\n\n- **File**: `src/a.cpp` (anonymous namespace)\n\n"
        "### `kOther` — other\n\n- **File**: `src/a.cpp` (`helper_fn`)\n",
        {"src/a.cpp": "constexpr int kCap = 1;\nconstexpr int kOther = 2;\n"},
    )
    problems = check(doc, root)
    assert len(problems) == 1
    assert "`helper_fn`" in problems[0]


def test_prose_heading_without_symbol_still_checks_the_file(tmp_path):
    doc, root = _write(
        tmp_path,
        "### FeasibilityJump budget — not weight-apportioned\n\n"
        "- **File**: `src/nope.cpp`\n",
        {},
    )
    problems = check(doc, root)
    assert len(problems) == 1
    assert "file does not exist" in problems[0]


def test_real_parameters_doc_has_no_stale_references():
    """Guards the checked-in docs/PARAMETERS.md itself."""
    assert check() == []
