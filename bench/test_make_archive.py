"""Tests for bench/make_archive.py.

The interesting cases are all provenance: the archive's whole value is that a
reader can tell which binary, which baseline claim, which instrumentation state
and which thread count a row came from.  Each of those is derived from log text
or options files, so each gets a fixture that differs only in the one line that
decides it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_archive import (
    BASELINE_NAMES,
    build_archive,
    classify_baseline,
    collect_config,
    default_table_specs,
    discover_configs,
    inspect_log,
    machine_provenance,
    make_tarball,
    parse_table_flag,
    read_options_file,
    render_provenance,
    source_provenance,
    verify_archive,
)

BANNER = (
    "Running HiGHS 1.15.1 (git hash: 04024d701f): Copyright (c) 2026 under MIT "
    "licence terms\n"
    "Includes third-party software components, see THIRD_PARTY_NOTICES.md\n"
)
MARKER = "mip-heuristics patch active (custom MIP presolve heuristics)\n"

# Enough of a HiGHS MIP log for parse_highs_log to produce a solved result with
# a real incumbent, so the archived tables render actual numbers rather than a
# table of `nan` that no edit could ever move.
BODY = """ H       0       0         0   0.00%   -inf            {first}          Large  0 0 0    10     0.1s
 B       1       0         0   0.00%   100             100              0%     0 0 0    42     0.2s
Solving report
  Status            Optimal
  Primal bound      100
  Dual bound        100
  Gap               0% (tolerance: 0.01%)
  P-D integral      0.1
  Solution status   feasible
  Timing            0.05 (total)
  Nodes             3
  LP iterations     42 (total)
"""

INSTRUMENTATION = (
    "[Sequential] heur=fpr effort=100 wall_ms=1.0 effort_per_ms=100.0\n"
    "[Heur] name=fpr phase=presolve start_s=0.001 end_s=0.002 effort=100 "
    "wall_ms=1.0 effort_per_ms=100.0 found=0\n"
    "[Native] rens=0 rens_root=0 rins=0 rcfix=0 heur_lp_iters=0 "
    "total_lp_iters=42 fpr_lp_lp_iters=0\n"
    "[Root] lp_time_s=0.010 presolve_heur_s=0.002\n"
)


def write_run(
    seed_dir: Path,
    instance: str,
    *,
    patched: bool = True,
    instrumented: bool = True,
    first: int = 120,
    options: dict[str, str] | None = None,
) -> None:
    """Write one `<instance>.log` + `<instance>.opts` pair."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    log = BANNER + (MARKER if patched else "") + BODY.format(first=first)
    if instrumented:
        log += INSTRUMENTATION
    (seed_dir / f"{instance}.log").write_text(log)
    opts = dict(options or {})
    opts.setdefault("random_seed", seed_dir.name.removeprefix("seed"))
    (seed_dir / f"{instance}.opts").write_text(
        "".join(f"{k} = {v}\n" for k, v in opts.items())
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A two-config results tree: patched `off` baseline plus patched `all`."""
    root = tmp_path / "results"
    for config, suite in (("off", "off"), ("all", "all")):
        for seed in (0, 1):
            for instance in ("egout", "flugpl"):
                write_run(
                    root / config / f"seed{seed}",
                    instance,
                    options={"log_dev_level": "3", "mip_heuristic_suite": suite},
                )
    return root


# ── log inspection ───────────────────────────────────────────────────────────


def test_inspect_log_reads_the_marker_and_the_banner(tmp_path: Path):
    write_run(tmp_path, "egout")
    patched, version, git_hash, instrumented = inspect_log(tmp_path / "egout.log")
    assert patched
    assert version == "1.15.1"
    assert git_hash == "04024d701f"
    assert instrumented


def test_inspect_log_separates_binaries_that_share_a_banner(tmp_path: Path):
    """The banner is identical between builds; only the marker differs."""
    write_run(tmp_path / "p", "egout", patched=True)
    write_run(tmp_path / "u", "egout", patched=False)
    patched, p_version, p_hash, _ = inspect_log(tmp_path / "p" / "egout.log")
    unpatched, u_version, u_hash, _ = inspect_log(tmp_path / "u" / "egout.log")
    assert (p_version, p_hash) == (u_version, u_hash)
    assert patched and not unpatched


def test_inspect_log_reports_an_uninstrumented_run(tmp_path: Path):
    write_run(tmp_path, "egout", instrumented=False)
    *_, instrumented = inspect_log(tmp_path / "egout.log")
    assert not instrumented


def test_read_options_file_round_trips(tmp_path: Path):
    (tmp_path / "a.opts").write_text("threads = 16\nmip_heuristic_suite = all\n")
    assert read_options_file(tmp_path / "a.opts") == {
        "threads": "16",
        "mip_heuristic_suite": "all",
    }


# ── config discovery and collection ──────────────────────────────────────────


def test_discover_configs_skips_directories_without_seeds(tree: Path):
    (tree / "tables").mkdir()
    (tree / "tables" / "summary.txt").write_text("not a config\n")
    assert discover_configs(tree) == ["off", "all"]


def test_discover_configs_sorts_the_baseline_first(tmp_path: Path):
    """Config order is table column order, so it is not cosmetic."""
    root = tmp_path / "r"
    for config in ("scylla", "off", "all"):
        write_run(root / config / "seed0", "egout")
    assert discover_configs(root) == ["off", "all", "scylla"]


def test_collect_config_summarises_a_uniform_config(tree: Path):
    config = collect_config(tree, "all")
    assert config.binary == "patched"
    assert config.seeds == [0, 1]
    assert config.instances == ["egout", "flugpl"]
    assert config.runs == 4
    assert config.options == {"log_dev_level": "3", "mip_heuristic_suite": "all"}
    assert config.option_variants == []
    assert config.instrumentation_requested
    assert config.instrumentation_observed


def test_collect_config_records_disagreeing_option_sets(tmp_path: Path):
    root = tmp_path / "r"
    write_run(root / "all" / "seed0", "egout", options={"mip_heuristic_suite": "all"})
    write_run(
        root / "all" / "seed0",
        "flugpl",
        options={"mip_heuristic_suite": "all", "threads": "16"},
    )
    config = collect_config(root, "all")
    assert config.options == {}
    assert len(config.option_variants) == 2


def test_collect_config_ignores_the_seed_when_comparing_options(tmp_path: Path):
    """random_seed names the directory, so it is per-run by construction."""
    root = tmp_path / "r"
    for seed in (0, 1):
        write_run(root / "all" / f"seed{seed}", "egout", options={"threads": "8"})
    config = collect_config(root, "all")
    assert config.option_variants == []
    assert config.options == {"threads": "8"}


def test_collect_config_counts_failed_runs(tmp_path: Path):
    root = tmp_path / "r"
    write_run(root / "all" / "seed0", "egout")
    (root / "all" / "seed0" / "flugpl.log.err").write_text("TIMEOUT\n")
    config = collect_config(root, "all")
    assert config.runs == 1
    assert config.failed_runs == ["seed0/flugpl.log.err"]


def test_collect_config_counts_runs_with_no_options_file(tmp_path: Path):
    """ "No options archived" and "given no options" render identically."""
    root = tmp_path / "r"
    write_run(root / "all" / "seed0", "egout", options={"threads": "8"})
    (root / "all" / "seed0" / "egout.opts").unlink()
    config = collect_config(root, "all")
    assert config.runs_without_options == 1
    assert config.options == {}


def test_collect_config_excludes_a_bannerless_log_from_the_binary_call(
    tmp_path: Path,
):
    """A log proving nothing must not count as the *stronger* claim."""
    root = tmp_path / "r"
    write_run(root / "vanilla" / "seed0", "egout")
    log = root / "vanilla" / "seed0" / "egout.log"
    log.write_text("truncated before the banner\n")
    config = collect_config(root, "vanilla")
    assert config.runs_without_banner == 1
    assert config.binary == "unknown"
    assert classify_baseline([config])["claim"] == "indeterminate"


def test_collect_config_flags_an_instrumentation_disagreement(tmp_path: Path):
    """`--dev-log` cancelled by an override is a silently mis-labelled tree."""
    root = tmp_path / "r"
    write_run(
        root / "all" / "seed0",
        "egout",
        instrumented=False,
        options={"log_dev_level": "3"},
    )
    config = collect_config(root, "all")
    assert config.instrumentation_requested
    assert not config.instrumentation_observed


# ── baseline classification ──────────────────────────────────────────────────


def test_baseline_on_a_patched_binary_is_the_weaker_claim(tree: Path):
    configs = [collect_config(tree, c) for c in ("off", "all")]
    baseline = classify_baseline(configs)
    assert baseline["config"] == "off"
    assert baseline["claim"] == "vanilla-equivalent setting on the patched binary"
    assert "check_vanilla_equivalence.py" in baseline["evidence"]


def test_baseline_on_an_unpatched_binary_is_the_stronger_claim(tmp_path: Path):
    root = tmp_path / "r"
    write_run(root / "vanilla" / "seed0", "egout", patched=False)
    write_run(root / "all" / "seed0", "egout")
    configs = [collect_config(root, c) for c in ("vanilla", "all")]
    baseline = classify_baseline(configs)
    assert baseline["config"] == "vanilla"
    assert baseline["claim"] == "separately built unpatched binary"


def test_baseline_names_match_the_analyzer_that_renders_the_table():
    """A drift here makes PROVENANCE.md name a different baseline than the
    archived `--cannibalization` table computes against — a disagreement
    between the archive's prose and the archive's own numbers."""
    import analyze_results

    assert BASELINE_NAMES == analyze_results.CANNIBALIZATION_BASELINE_NAMES


def test_baseline_absent_when_no_config_stands_for_one(tmp_path: Path):
    root = tmp_path / "r"
    write_run(root / "all" / "seed0", "egout")
    baseline = classify_baseline([collect_config(root, "all")])
    assert baseline["config"] is None
    assert baseline["claim"] == "none"


# ── table specs ──────────────────────────────────────────────────────────────


def test_default_tables_use_the_pairwise_shape_for_two_configs():
    specs = default_table_specs(["off", "all"], 600, instrumented=False)
    assert [s.name for s in specs] == ["summary", "attribution"]
    assert "--baseline" in specs[0].argv
    assert "--ablation" not in specs[0].argv


def test_default_tables_use_the_ablation_shape_for_three():
    specs = default_table_specs(["off", "fpr", "all"], 60, instrumented=False)
    assert all("--ablation" in s.argv for s in specs)


def test_cannibalization_table_only_offered_for_an_instrumented_tree():
    plain = default_table_specs(["off", "all"], 600, instrumented=False)
    instrumented = default_table_specs(["off", "all"], 600, instrumented=True)
    assert "cannibalization" not in [s.name for s in plain]
    assert "cannibalization" in [s.name for s in instrumented]


def test_parse_table_flag_splits_name_from_args():
    spec = parse_table_flag("sgm=--summary --configs off all")
    assert spec.name == "sgm"
    assert spec.argv == ["--summary", "--configs", "off", "all"]


@pytest.mark.parametrize("value", ["no-equals-sign", "=--summary", "empty="])
def test_parse_table_flag_rejects_malformed_input(value: str):
    with pytest.raises(ValueError, match="--table"):
        parse_table_flag(value)


# ── build and verify ─────────────────────────────────────────────────────────


def build(tree: Path, out: Path, **kwargs):
    defaults = {
        "configs": discover_configs(tree),
        "time_limit": 10.0,
        "note": "",
        "machine_note": "bench box",
        "extra_tables": [],
    }
    defaults.update(kwargs)
    return build_archive(tree, out, **defaults)


def test_build_produces_a_self_contained_archive(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    manifest = build(tree, archive)

    assert (archive / "MANIFEST.json").is_file()
    assert (archive / "PROVENANCE.md").is_file()
    assert (archive / "REGENERATE.sh").is_file()
    # The analysis scripts travel with the archive, so no checkout is needed.
    for name in ("analyze_results.py", "parse_highs_log.py", "make_archive.py"):
        assert (archive / "bench" / name).is_file()
    # Logs *and* the options file each run was given.
    assert (archive / "results" / "all" / "seed0" / "egout.log").is_file()
    assert (archive / "results" / "all" / "seed0" / "egout.opts").is_file()
    assert [t.name for t in manifest.tables] == [
        "summary",
        "attribution",
        "cannibalization",
    ]
    for spec in manifest.tables:
        assert (archive / spec.path).is_file()


def test_build_refuses_to_overwrite(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    build(tree, archive)
    with pytest.raises(FileExistsError):
        build(tree, archive)


def test_build_rejects_a_config_mixing_patched_and_unpatched_logs(tmp_path: Path):
    root = tmp_path / "r"
    write_run(root / "all" / "seed0", "egout", patched=True)
    write_run(root / "all" / "seed0", "flugpl", patched=False)
    with pytest.raises(ValueError, match="mixes patched and unpatched"):
        build(root, tmp_path / "arch", configs=["all"])


def test_build_warns_when_threads_is_unset(tree: Path, tmp_path: Path):
    manifest = build(tree, tmp_path / "arch")
    assert manifest.run["threads_option"] is None
    assert any("threads" in w for w in manifest.warnings)


def test_build_records_a_pinned_thread_count(tmp_path: Path):
    root = tmp_path / "r"
    for config in ("off", "all"):
        write_run(root / config / "seed0", "egout", options={"threads": "16"})
    manifest = build(root, tmp_path / "arch")
    assert manifest.run["threads_option"] == "16"
    assert not any("threads" in w for w in manifest.warnings)


def test_build_warns_without_a_machine_note(tree: Path, tmp_path: Path):
    manifest = build(tree, tmp_path / "arch", machine_note="")
    assert any("--machine-note" in w for w in manifest.warnings)


def test_build_warns_on_failed_runs(tree: Path, tmp_path: Path):
    (tree / "all" / "seed0" / "p0548.log.err").write_text("TIMEOUT\n")
    manifest = build(tree, tmp_path / "arch")
    assert any("failed" in w for w in manifest.warnings)
    # Archived as evidence, but not visible to the table commands.
    assert (tmp_path / "arch" / "results" / "all" / "seed0" / "p0548.log.err").is_file()


def test_build_rejects_a_custom_table_colliding_with_a_default(
    tree: Path, tmp_path: Path
):
    with pytest.raises(ValueError, match="collides"):
        build(
            tree,
            tmp_path / "arch",
            extra_tables=[parse_table_flag("summary=--summary")],
        )


def test_build_renders_a_custom_table(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    spec = parse_table_flag("pairwise=--configs off all --time-limit 10 --summary")
    manifest = build(tree, archive, extra_tables=[spec])
    assert "pairwise" in [t.name for t in manifest.tables]
    assert (archive / "tables" / "pairwise.txt").read_text().strip()


def test_build_fails_loudly_on_a_broken_table_command(tree: Path, tmp_path: Path):
    with pytest.raises(RuntimeError, match="not-a-config"):
        build(
            tree,
            tmp_path / "arch",
            extra_tables=[parse_table_flag("bad=--configs not-a-config")],
        )


def test_manifest_checksums_cover_every_archived_file(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    manifest = build(tree, archive)
    on_disk = {
        p.relative_to(archive).as_posix()
        for p in archive.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json"
    }
    assert set(manifest.files) == on_disk
    # PROVENANCE.md is inside the checksum set: provenance that can be edited
    # without `verify` noticing is not provenance.
    assert "PROVENANCE.md" in manifest.files


def test_tarball_keeps_a_version_numbered_name_intact(tree: Path, tmp_path: Path):
    """`with_suffix` would treat the `.0` of `v1.0.0` as the suffix."""
    archive = tmp_path / "mip-heuristics-v1.0.0-archive"
    build(tree, archive)
    tar = make_tarball(archive)
    assert tar.name == "mip-heuristics-v1.0.0-archive.tar.gz"
    with tarfile.open(tar) as handle:
        names = handle.getnames()
    assert f"{archive.name}/MANIFEST.json" in names


def test_verify_passes_on_a_fresh_archive(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    build(tree, archive)
    assert verify_archive(archive, None) == []


def test_verify_detects_an_edited_log(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    build(tree, archive)
    # Both seeds, because `aggregate_results` takes a per-instance median over
    # them: editing one seed of two leaves the median row untouched, so a
    # single-seed edit would test the checksum only.
    for seed in (0, 1):
        log = archive / "results" / "all" / f"seed{seed}" / "egout.log"
        log.write_text(
            log.read_text().replace("Primal bound      100", "Primal bound      50")
        )
    problems = verify_archive(archive, None)
    assert any("checksum mismatch" in p for p in problems)
    # The two mechanisms are independent: the checksum says the bytes moved,
    # and re-deriving the table from the tampered logs says the *published
    # number* moved with them.
    assert any("differs from the" in p for p in problems)


def test_verify_detects_an_edited_provenance_document(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    build(tree, archive)
    (archive / "PROVENANCE.md").write_text("nothing to see here\n")
    assert any("PROVENANCE.md" in p for p in verify_archive(archive, None))


def test_verify_detects_a_smuggled_in_file(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    build(tree, archive)
    (archive / "results" / "all" / "seed0" / "extra.log").write_text(BANNER + BODY)
    assert any("unrecorded file" in p for p in verify_archive(archive, None))


def test_verify_detects_a_missing_file(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    build(tree, archive)
    (archive / "tables" / "summary.txt").unlink()
    problems = verify_archive(archive, None)
    assert any("missing file" in p for p in problems)


def test_verify_reports_a_non_archive_directory(tmp_path: Path):
    assert verify_archive(tmp_path, None) == [
        "MANIFEST.json is missing — this is not an archive directory"
    ]


def test_verify_write_dir_is_not_mistaken_for_archive_content(
    tree: Path, tmp_path: Path
):
    """A kept regeneration must not make the next verify fail."""
    archive = tmp_path / "arch"
    build(tree, archive)
    out = archive / "regen"
    assert verify_archive(archive, out) == []
    assert (out / "summary.txt").is_file()
    assert verify_archive(archive, out) == []


# ── rendered documents ───────────────────────────────────────────────────────


def test_provenance_document_states_the_facts_a_reader_needs(
    tree: Path, tmp_path: Path
):
    manifest = build(tree, tmp_path / "arch", note="closeout campaign")
    text = render_provenance(manifest)
    assert "closeout campaign" in text
    assert "HiGHS `v1.15.1`" in text
    assert "mip-heuristics patch active" in text
    assert "vanilla-equivalent setting on the patched binary" in text
    assert "Instrumented (`log_dev_level=3`) | yes" in text
    assert "clang-format==22.1.8" in text
    # Every table's exact regeneration command is in the document.
    for spec in manifest.tables:
        assert spec.path in text


def test_manifest_json_is_loadable_and_carries_the_table_argv(
    tree: Path, tmp_path: Path
):
    archive = tmp_path / "arch"
    build(tree, archive)
    payload = json.loads((archive / "MANIFEST.json").read_text())
    assert payload["manifest_version"] == 1
    assert payload["source"]["highs_tag"] == "v1.15.1"
    assert payload["baseline"]["config"] == "off"
    names = {t["name"]: t for t in payload["tables"]}
    assert "--cannibalization" in names["cannibalization"]["argv"]


def test_source_provenance_reads_the_pins_from_the_repository():
    src = source_provenance()
    assert src["highs_tag"] == "v1.15.1"
    assert src["patch_version"] and src["patch_version"].isdigit()
    assert src["tool_pins"]["clang-format"] == "22.1.8"
    assert "ruff" in src["tool_pins"]


def test_machine_provenance_says_where_it_came_from():
    plain = machine_provenance("")
    annotated = machine_provenance("16-core bench box")
    assert "archive host" in plain["source"]
    assert "--machine-note" in annotated["source"]
    assert annotated["note"] == "16-core bench box"
    assert plain["cpu_count"]


# ── the shipped entry points ─────────────────────────────────────────────────


def test_regenerate_script_runs_the_archived_verifier(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    build(tree, archive)
    proc = subprocess.run(
        [str(archive / "REGENERATE.sh")],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable},
    )
    assert proc.returncode == 0, proc.stderr
    assert "regenerates identically" in proc.stdout


def test_cli_build_then_verify(tree: Path, tmp_path: Path):
    archive = tmp_path / "arch"
    script = str(Path(__file__).resolve().parent / "make_archive.py")
    built = subprocess.run(
        [
            sys.executable,
            script,
            "build",
            str(tree),
            "--output",
            str(archive),
            "--time-limit",
            "10",
            "--machine-note",
            "bench box",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    verified = subprocess.run(
        [sys.executable, script, "verify", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr


def test_cli_rejects_an_unknown_config(tree: Path, tmp_path: Path):
    script = str(Path(__file__).resolve().parent / "make_archive.py")
    proc = subprocess.run(
        [
            sys.executable,
            script,
            "build",
            str(tree),
            "--output",
            str(tmp_path / "arch"),
            "--time-limit",
            "10",
            "--configs",
            "patchd",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "not in" in proc.stderr
