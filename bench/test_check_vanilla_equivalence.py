"""Tests for check_vanilla_equivalence's parsing and comparison."""

from check_vanilla_equivalence import compare_runs, normalize_log, parse_metrics

# A trimmed but format-faithful HiGHS CLI log: banner, source-key legend,
# two display lines, and the solving report.  `%MARKER%` is the line the
# patch inserts to identify itself — present in a patched log and absent
# (not blank) in a vanilla one — and `{t0}`/`{t1}`/`{timing}` are the
# wall-clock fields that must not make two runs of identical work compare
# unequal.
LOG = """Running HiGHS 1.15.1 (git hash: abc1234): Copyright (c) 2025 HiGHS
%MARKER%
Src: B => Branching; C => Central rounding;

        Nodes      |    B&B Tree     |
Src  Proc. InQueue |  Leaves   Expl. | LpIters     Time

         0       0         0   0.00%         0     {t0}s
 B      18       2         3  26.56%       338     {t1}s

Solving report
  Status            Optimal
  Primal bound      1201500
  Nodes             89
  P-D integral      0.000532021661948
  Timing            {timing}
                    0.00 (Presolve)
                    {timing} (Solve)
  LP iterations     858
                    311 (strong br.)
                    289 (heuristics)
"""


def render(marker: str = "", t0: str = "0.0", t1: str = "0.1", timing: str = "0.09") -> str:
    body = LOG.format(t0=t0, t1=t1, timing=timing)
    return body.replace("%MARKER%\n", f"{marker}\n" if marker else "")


def test_parse_metrics_reads_the_solving_report():
    m = parse_metrics(render())
    assert m.status == "Optimal"
    assert m.primal_bound == "1201500"
    assert m.nodes == "89"
    assert m.lp_iterations == "858"
    assert m.heuristic_lp_iterations == "289"
    assert m.time_s == 0.09


def test_parse_metrics_ignores_the_display_table_header():
    """The table header line also begins with "Nodes"; only the report counts."""
    assert parse_metrics("        Nodes      |    B&B Tree     |").nodes is None


def test_normalize_log_drops_the_patch_marker_and_wall_clock():
    patched = render(marker="mip-heuristics patch active (custom MIP presolve heuristics)",
                     t0="0.1", t1="0.4", timing="0.31")
    assert normalize_log(patched) == normalize_log(render())


def test_normalize_log_drops_the_options_echo():
    """The patched run is given an option the vanilla run cannot have."""
    patched = render().replace("Solving report",
                               'Set option mip_heuristic_suite to "off"\nSolving report')
    assert normalize_log(patched) == normalize_log(render())


def test_normalize_log_drops_the_issue_95_instrumentation_lines():
    """`[Native]` / `[Root]` are emitted at suite=off too — that run is the
    reference the patched rows are compared against — and an unpatched
    binary cannot print them.  Only visible at log_dev_level=3, which this
    script does not set, so the mask must hold before someone raises it."""
    patched = render().replace(
        "Solving report",
        "[Native] rens=1 rens_root=1 rins=1 rcfix=1 heur_lp_iters=697 "
        "total_lp_iters=2125 fpr_lp_lp_iters=0\n"
        "[Root] lp_time_s=0.039 presolve_heur_s=0.000\n"
        "Solving report",
    )
    assert normalize_log(patched) == normalize_log(render())


def test_normalize_log_masks_profiling_seconds_but_keeps_call_counts():
    """The call count is signal; the seconds beside it are not."""
    masked = normalize_log("      subMIP time [calls] = 0.02 [27]")
    assert masked == normalize_log("      subMIP time [calls] = 0.01 [27]")
    assert masked != normalize_log("      subMIP time [calls] = 0.02 [26]")


def test_normalize_log_masks_the_banner_githash_length():
    """Shallow clone and FetchContent print the same commit at different widths."""
    assert (normalize_log("Running HiGHS 1.15.1 (git hash: 04024d7): Copyright") ==
            normalize_log("Running HiGHS 1.15.1 (git hash: 04024d701f): Copyright"))


def test_identical_work_compares_equal_despite_timing():
    c = compare_runs("flugpl.mps", 0,
                     patched=render(marker="mip-heuristics patch active", timing="0.31"),
                     vanilla=render(timing="0.09"),
                     time_tolerance=1.5, strict_time=False)
    assert c.passed, c.failures
    assert c.patched_time == 0.31


def test_strict_time_flags_a_slower_patched_run():
    c = compare_runs("flugpl.mps", 0, patched=render(timing="0.90"),
                     vanilla=render(timing="0.09"), time_tolerance=1.5, strict_time=True)
    assert not c.passed
    assert any("solve time" in f for f in c.failures)


def test_a_differing_node_count_fails():
    c = compare_runs("flugpl.mps", 0,
                     patched=render().replace("Nodes             89", "Nodes             90"),
                     vanilla=render(), time_tolerance=1.5, strict_time=False)
    assert not c.passed
    assert any(f.startswith("nodes:") for f in c.failures)


def test_a_differing_heuristic_lp_iteration_count_fails():
    """The RENS/RINS budget-cannibalization signal — the point of the check."""
    c = compare_runs("flugpl.mps", 0,
                     patched=render().replace("289 (heuristics)", "412 (heuristics)"),
                     vanilla=render(), time_tolerance=1.5, strict_time=False)
    assert not c.passed
    assert any(f.startswith("heuristic LP iterations:") for f in c.failures)


def test_two_unparseable_logs_do_not_certify_equivalence():
    """Two logs with no solving report are equal but prove nothing."""
    c = compare_runs("flugpl.mps", 0, patched="crashed\n", vanilla="crashed\n",
                     time_tolerance=1.5, strict_time=False)
    assert not c.passed
    assert any("no parseable solving report" in f for f in c.failures)


def test_a_slow_run_warns_without_strict_time():
    c = compare_runs("flugpl.mps", 0, patched=render(timing="0.90"),
                     vanilla=render(timing="0.09"), time_tolerance=1.5, strict_time=False)
    assert c.passed
    assert any("solve time" in w for w in c.warnings)


def test_an_extra_log_line_fails():
    c = compare_runs("flugpl.mps", 0,
                     patched=render() + "\n[Sequential] heur=fpr effort=1 wall_ms=1\n",
                     vanilla=render(), time_tolerance=1.5, strict_time=False)
    assert not c.passed
    assert any("normalized log differs" in f for f in c.failures)
