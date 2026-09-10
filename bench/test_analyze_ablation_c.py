"""Ablation C's reader (#165) — the verdict, which is the part that decides."""

import io
import math
from pathlib import Path

import pytest
from analyze_ablation_c import (
    MIN_DISPATCHES_FOR_A_NULL,
    Paired,
    count_one,
    paired,
    report,
)

# A `[Heur]` dive line and an incumbent line, in the shapes the parser wants.
HEUR = (
    "[Heur] name=fpr_lp phase=dive start_s=1.0 end_s=1.1 effort=10 "
    "wall_ms=100.0 effort_per_ms=0.1 found={found} nnz=2831 abandoned_setup={bail}"
)
# An ordinary MIP display row: the leading character is the solution source.
INCUMBENT = (
    "{src}     0       0         0   0.00%   -inf            "
    "12.5               inf        0      0      0         0     0.1s"
)


def write_log(tmp_path: Path, name: str, *lines: str, dev_log: bool = True) -> Path:
    d = tmp_path / "seed0"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.log"
    p.write_text("\n".join(lines) + "\n")
    # The run's own `.opts`, which is where tracing is read from.
    (d / f"{name}.opts").write_text(
        ("log_dev_level = 3\n" if dev_log else "") + "mip_heuristic_suite = fpr_lp\n"
    )
    return p


def verdict(rows) -> str:
    buf = io.StringIO()
    report(rows, out=buf)
    return buf.getvalue()


def test_dispatches_and_yields_are_counted_separately(tmp_path):
    """A dispatch is a `[Heur]` line; a yield is an accepted `D` incumbent."""
    p = write_log(
        tmp_path,
        "inst",
        HEUR.format(found=1, bail=0),
        HEUR.format(found=0, bail=0),
        INCUMBENT.format(src="D"),
        INCUMBENT.format(src="J"),
    )
    c = count_one(p)
    assert c.dispatches == 2
    assert c.yields == 1  # the J-sourced incumbent is FJ's, not fpr_lp's
    assert c.traced


def test_a_setup_bail_is_neither_a_search_nor_an_absence(tmp_path):
    """#119: `abandoned_setup=1` is a third state and is reported as its own."""
    p = write_log(tmp_path, "inst", HEUR.format(found=0, bail=1))
    c = count_one(p)
    assert (c.dispatches, c.abandoned) == (1, 1)


def test_plentiful_dispatches_with_no_yield_is_a_decisive_null(tmp_path):
    rows = [
        count_one(write_log(tmp_path, f"i{i}", *[HEUR.format(found=0, bail=0)] * 5))
        for i in range(MIN_DISPATCHES_FOR_A_NULL)
    ]
    assert "Decisive null" in verdict(rows)


def test_too_few_dispatches_is_not_a_null(tmp_path):
    """The failure this exists to prevent: reading a too-short run as evidence."""
    rows = [count_one(write_log(tmp_path, "i0", HEUR.format(found=0, bail=0)))]
    out = verdict(rows)
    assert "inconclusive" in out
    assert "Decisive null" not in out


def test_one_yield_beats_any_number_of_barren_dispatches(tmp_path):
    rows = [
        count_one(
            write_log(
                tmp_path,
                f"i{i}",
                *[HEUR.format(found=0, bail=0)] * 5,
                *([INCUMBENT.format(src="D")] if i == 0 else []),
            )
        )
        for i in range(MIN_DISPATCHES_FOR_A_NULL)
    ]
    assert "Not a null" in verdict(rows)


def test_an_untraced_tree_is_unreadable_rather_than_null(tmp_path):
    """Trap 1 in #165: absence of the line meant absence of tracing, not of the dive.

    An untraced run yields no `[Heur]` lines whatever the dive did, so a
    dispatch count of zero there must never be reported as a null.
    """
    rows = [
        count_one(write_log(tmp_path, "i0", INCUMBENT.format(src="J"), dev_log=False))
    ]
    out = verdict(rows)
    assert "unreadable" in out
    assert "PLATO_DEV_LOG=1" in out
    assert "Decisive null" not in out


def test_a_traced_run_with_no_dispatch_is_data_not_an_absence(tmp_path):
    """The mirror of the trap above, which the first version of this file hit.

    At `suite=fpr_lp` the only heuristic that can emit a `[Heur]` line is the
    one under test, so a traced run with none did not dispatch -- it is a zero
    in the fire rate, not an unreadable run.  Inferring tracing from the lines
    themselves discards exactly the instances where the answer is no.
    """
    r = count_one(write_log(tmp_path, "i0", INCUMBENT.format(src="J")))
    assert r.traced is True
    assert r.dispatches == 0
    assert "untraced" not in verdict([r])


def test_a_killed_run_is_a_lower_bound_not_a_zero(tmp_path):
    """The harness SIGKILLs an overrunning solve, so its log stops mid-solve."""
    fired = [
        count_one(write_log(tmp_path, f"i{i}", *[HEUR.format(found=0, bail=0)] * 5))
        for i in range(3)
    ]
    cut = count_one(
        write_log(tmp_path, "killed", "TIMEOUT: process killed after 300.0s")
    )
    assert cut.killed is True
    out = verdict([*fired, cut])
    # Denominator is the three observable runs, not four.
    assert "fired on 3/3 observable" in out
    assert "killed -- lower bound" in out


@pytest.mark.parametrize("src", ["D", "J"])
def test_yields_are_readable_without_tracing(tmp_path, src):
    """The source character is in the ordinary log, which is why C1 need not trace."""
    c = count_one(write_log(tmp_path, "i0", INCUMBENT.format(src=src), dev_log=False))
    assert c.traced is False
    assert c.yields == (1 if src == "D" else 0)


# ── the paired contribution half ────────────────────────────────────────────


def test_paired_is_a_log_ratio_of_the_arm_over_the_control():
    """An arm twice the control's integral on every instance reads as 2.0.

    Zero variance is a legitimate input rather than a crash: every instance
    moved by the same factor.  It has no finite t, and the interval has zero
    width, so it still reads as separated.
    """
    p = paired([math.log(2.0)] * 5)
    assert p is not None
    assert p.ratio == pytest.approx(2.0)
    assert p.t == math.inf
    assert p.separated


def test_separated_means_the_interval_excludes_no_effect():
    tight = paired([math.log(1.3)] * 20 + [math.log(1.31)] * 20)
    wide = paired([math.log(0.4), math.log(3.0)] * 10)
    assert tight is not None and wide is not None
    assert tight.separated
    assert not wide.separated  # same centre, interval spans 1.0


def test_power_arithmetic_matches_the_pre_registered_formula():
    """n = 8 sd^2 / ln(1+delta)^2, the formula Ablation B's write-up uses."""
    p = paired([math.log(1.1), math.log(0.9)] * 25)
    assert p is not None
    n = p.n_for(1.2)
    assert n == math.ceil(8 * p.sd**2 / math.log(1.2) ** 2)
    # Round trip: `n_for` ceils, so that n resolves *at least* what was asked.
    assert Paired(n, 1.2, 1.0, 1.4, 0.0, p.sd).detectable() <= 0.2
    assert Paired(n - 1, 1.2, 1.0, 1.4, 0.0, p.sd).detectable() > 0.2


def test_a_single_instance_yields_no_interval():
    """One paired observation has no spread, so it must not be summarised."""
    assert paired([math.log(1.5)]) is None
