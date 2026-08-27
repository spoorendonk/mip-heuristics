"""Unit tests for bench/analyze_presolve_probe.py.

The fixtures are built from the shapes real probe logs actually have, not
from a guessed one.  That distinction is not academic: an earlier version of
this file fabricated a log carrying both a one-line model header and a
`[HeurSol]` trace, which no run ever produces, and three live defects passed
a full green suite behind it.  Every builder below is annotated with the real
log it mirrors.

Two shapes matter most:

* A run **without** `--dev-log` prints the one-line `MIP <name> has ...
  nonzeros` header and no trace.
* A run **with** `--dev-log` prints the trace and, at that level, a *block*
  model header instead — so `SolveResult.num_nonzeros` is None on precisely
  the runs that carry the trace.  That is issue F1, owned by the parser; the
  tests below pin it as a diagnostic rather than working around it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_presolve_probe import (
    CHAIN_SOURCES,
    PRESOLVE_HEURISTICS,
    REASON_NO_ACCEPTANCE,
    REASON_PRODUCED_NOT_IMPROVED,
    REASON_TRIVIAL_ONLY,
    REASON_UNREACHED,
    AdapterError,
    DispatchView,
    HeuristicTrajectory,
    Observation,
    ProbeRun,
    WorkerSeries,
    classify_run,
    dispatch_views,
    heursol_from_lines,
    heursol_samples,
    improving_offers,
    informative_set,
    km_quantile,
    objective_sense,
    parse_heursol_line,
    parse_quantiles,
    parser_supports_heursol,
    quantile,
    summarise_traces,
)
from parse_highs_log import SolveResult, parse_log
from run_benchmark import load_instances

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "analyze_presolve_probe.py"
)

# ---------------------------------------------------------------------------
# Fixtures, transcribed from the pilot probe tree
# ---------------------------------------------------------------------------

# Lines 1-3 of every run; the marker is what tells a patched binary from an
# unpatched one, since the version and githash banners are identical.
_BANNER = [
    (
        "Running HiGHS 1.15.1 (git hash: 04024d701f): Copyright (c) 2026 under"
        " MIT licence terms"
    ),
    (
        "Includes third-party software components, see THIRD_PARTY_NOTICES.md"
        " for full details"
    ),
]
_MARKER = (
    "mip-heuristics patch active (custom MIP presolve heuristics;"
    " spoorendonk/mip-heuristics)"
)


# The one-line model header, printed at the default log level only.
def _model_oneline(nnz: int) -> str:
    return (
        f"MIP probe has 165684 rows; 14770 cols; {nnz} nonzeros;"
        " 14770 integer variables (14770 binary)"
    )


# What `log_dev_level=3` prints instead: a block whose `Nonzeros` is the
# *original* matrix, not the post-presolve one the stall options are
# expressed against.  `parse_highs_log` matches neither, so `num_nonzeros`
# comes out None.
_MODEL_BLOCK = [
    " MIP      : probe",
    "Rows      : 51",
    "Cols      : 220",
    "Nonzeros  : 2808",
    "Integer   : 200 (200 binary)",
]

_LEGEND = [
    (
        "Src: B => Branching; C => Central rounding; F => Feasibility pump;"
        " H => Heuristic;"
    ),
    (
        "     l => Trivial lower; p => Trivial point; u => Trivial upper;"
        " z => Trivial zero;"
    ),
    "     A => FPR; D => FPR LP; M => Local MIP; G => Scylla; J => FJ",
]
_TABLE_HEADER = (
    "Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol"
    "              Gap |   Cuts   InLp Confl. | LpIters     Time"
)


def _solving_block(threads: int) -> list[str]:
    return [
        "Solving MIP model with:",
        "   105209 rows",
        (
            "   8955 cols (8955 binary, 0 integer, 0 implied int.,"
            " 0 continuous, 0 domain fixed)"
        ),
        "   361283 nonzeros",
        (
            f"   Thread count {threads} (of 32 threads). Using 1 max workers."
            " Parallel search off"
        ),
    ]


def display_row(source: str, objective: float, seconds: float = 4.1) -> str:
    """One incumbent row, exactly as the pilot logs print it."""
    return (
        f" {source}       0       0         0   0.00%   -inf            "
        f"{objective}                 Large        0      0      0         0"
        f"     {seconds}s"
    )


def heur_line(
    name: str,
    effort: int,
    found: int = 1,
    phase: str = "presolve",
    wall: float = 100.0,
) -> str:
    rate = effort / wall if wall else 0.0
    return (
        f"[Heur] name={name} phase={phase} start_s=0.100 end_s=0.200 "
        f"effort={effort} wall_ms={wall} effort_per_ms={rate:.1f} found={found}"
    )


def heursol_line(
    name: str,
    dispatch: int,
    worker: int | None,
    effort_at: int,
    accepted: int = 1,
    obj: float = 10.0,
    wall: float = 1.0,
) -> str:
    worker_field = "" if worker is None else f"worker={worker} "
    return (
        f"[HeurSol] name={name} dispatch={dispatch} {worker_field}"
        f"effort_at={effort_at} wall_ms={wall} obj={obj} accepted={accepted}"
    )


def probe_log(
    *,
    rows: tuple[tuple[str, float], ...] = (),
    heur: tuple[str, ...] = (),
    heursol: tuple[str, ...] = (),
    dev_log: bool = False,
    patched: bool = True,
    threads: int = 16,
    nnz: int = 555082,
    status: str = "Solution limit reached",
    killed: bool = False,
) -> str:
    """One presolve-only probe run's stdout.

    `dev_log` switches the model header from the one-line form to the block
    form, which is what actually happens at `log_dev_level=3` and is why a
    traced run has no parseable nonzero count today.
    """
    lines = list(_BANNER)
    if patched:
        lines.insert(2, _MARKER)
    lines.append("Set option mip_heuristic_presolve_only to true")
    lines += _MODEL_BLOCK if dev_log else [_model_oneline(nnz)]
    lines += _solving_block(threads)
    lines += _LEGEND
    lines.append(_TABLE_HEADER)
    lines.append("")
    for source, objective in rows:
        lines.append(display_row(source, objective))
    lines += list(heursol)
    lines += list(heur)
    if killed:
        lines.append("TIMEOUT: process killed after 61.0s")
        return "\n".join(lines) + "\n"
    lines.append("Solving report")
    lines.append("  Model             probe")
    lines.append(f"  Status            {status}")
    if rows:
        lines.append(f"  Primal bound      {rows[-1][1]}")
    lines.append("  Nodes             0")
    return "\n".join(lines) + "\n"


PROBE_OPTS = (
    "mip_heuristic_suite = all\n"
    "mip_heuristic_fj_effort = 1.0\n"
    "mip_heuristic_fj_stall = 0\n"
    "mip_heuristic_presolve_only = true\n"
    "random_seed = 0\n"
)
FULL_SOLVE_OPTS = "mip_heuristic_suite = all\nrandom_seed = 0\n"


def write_tree(
    tmp_path,
    logs: dict[str, dict[str, str]],
    name: str = "probe",
    opts: str = PROBE_OPTS,
) -> str:
    """Materialise `<tmp>/<name>/<config>/seed<N>/<instance>.{log,opts}`.

    The `.opts` beside each log is not decoration: it is how the probe check
    knows the run was presolve-only, and `run_benchmark.py` writes it.
    """
    root = os.path.join(str(tmp_path), name)
    for config, entries in logs.items():
        for key, text in entries.items():
            instance, _, seed = key.partition("@")
            seed_dir = os.path.join(root, config, f"seed{seed or 0}")
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, f"{instance}.log"), "w") as f:
                f.write(text)
            if opts is not None:
                with open(os.path.join(seed_dir, f"{instance}.opts"), "w") as f:
                    f.write(opts)
    return root


def write_reference(tmp_path, names, filename: str = "reference.txt") -> str:
    path = os.path.join(str(tmp_path), filename)
    with open(path, "w") as f:
        f.write("# reference list\n")
        f.write("".join(f"{n}\n" for n in names))
    return path


def run(*args: str, hash_seed: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def make_run(text: str, config: str = "all", seed: int = 0, instance: str = "inst"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = parse_log(text)
    return ProbeRun(
        config=config,
        seed=seed,
        instance=instance,
        result=result,
        heursols=heursol_from_lines(text.splitlines()),
        presolve_only=True,
        patched=True,
    )


# ---------------------------------------------------------------------------
# A stand-in for the parser's post-merge grouping API
# ---------------------------------------------------------------------------


class _StandInTrace:
    """What `parse_highs_log.DispatchTrace` exposes to this module."""

    def __init__(self, name, dispatch, total_effort, nnz, samples):
        self.name = name
        self.dispatch = dispatch
        self.total_effort = total_effort
        self.nnz = nnz
        self.samples = samples


class _StandInResult:
    """A `SolveResult` that groups dispatches itself, as Track B's does.

    Only the members this module touches.  Dispatch ids are deliberately
    process-global values — large, non-dense, shared across names — because
    that is what the real counter produces and what the analysis must not
    assume away.
    """

    def __init__(self, traces, thread_count=16):
        self._traces = traces
        self.thread_count = thread_count
        self.killed = False
        self.status = "Solution limit reached"
        self.primal_bound = 10.0
        self.incumbents = []
        self.heuristic_samples = []
        self.heursol_samples = [s for t in traces for s in t.samples]

    def dispatch_traces(self):
        return list(self._traces)


def sample(name, dispatch, worker, effort_at, accepted=True, obj=10.0):
    """One parsed `[HeurSol]` line.

    `obj` defaults to a constant, so a fixture that wants a *series* has to
    say what each offer's objective was: since #113 the trajectory is built
    from offers that improved the dispatch's best, and identical objectives
    are one improvement followed by repeats.
    """
    return parse_heursol_line(
        heursol_line(name, dispatch, worker, effort_at, int(accepted), obj=obj)
    )


# ---------------------------------------------------------------------------
# The [HeurSol] adapter
# ---------------------------------------------------------------------------


def test_adapter_parses_key_value_fields_in_any_order():
    line = (
        "[HeurSol] accepted=1 obj=-3.5 worker=2 name=local_mip effort_at=4096 "
        "dispatch=1 wall_ms=-0.5 pool_rank=3"
    )
    parsed = parse_heursol_line(line)
    assert parsed is not None
    assert parsed.name == "local_mip"
    assert parsed.dispatch == 1
    assert parsed.worker == 2
    assert parsed.effort_at == 4096
    assert parsed.wall_ms == -0.5  # the solver clock is not monotonic
    assert parsed.obj == -3.5
    assert parsed.accepted is True


def test_adapter_reads_the_off_slot_worker_verbatim():
    """`worker=-1` is a real value, not a sentinel to normalise away."""
    parsed = parse_heursol_line(heursol_line("local_mip", 0, -1, 0))
    assert parsed is not None and parsed.worker == -1


def test_adapter_refuses_a_line_missing_a_contract_field():
    for bad in (
        "[HeurSol] name=fj dispatch=0 worker=0 wall_ms=1 obj=3 accepted=1",
        "[HeurSol] name=fj dispatch=0 worker=0 effort_at=x wall_ms=1 obj=3 accepted=1",
    ):
        try:
            parse_heursol_line(bad)
        except AdapterError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_adapter_ignores_lines_that_are_not_heursol():
    assert parse_heursol_line(heur_line("fj", 100)) is None
    assert parse_heursol_line("  Status  Optimal") is None


def test_fallback_binds_each_offer_to_the_heur_line_that_closes_it():
    """`heur_index` is derived, not read: the line does not carry it.

    A dispatch's `[HeurSol]` lines precede the `[Heur]` line for the same
    name and follow the previous one, so binding is positional.  Offers still
    unbound at end of file keep None — what a killed run leaves behind.
    """
    lines = [
        heursol_line("fj", 100, 0, 10),
        heursol_line("fpr", 101, 0, 20),
        heur_line("fj", 1000),  # heuristic_samples[0]
        heur_line("fpr", 2000),  # heuristic_samples[1]
        heursol_line("scylla", 103, 0, 30),  # never closed
    ]
    got = heursol_from_lines(lines)
    assert [(s.name, s.heur_index) for s in got] == [
        ("fj", 0),
        ("fpr", 1),
        ("scylla", None),
    ]


class _AliasSample:
    """A parser sample type spelled with the alternative field names."""

    def __init__(self):
        self.heuristic = "fpr"
        self.dispatch = 77
        self.worker = 1
        self.effort = 512
        self.wall_ms = 2.0
        self.objective = 7.5
        self.accepted = True


class _AliasResult:
    def __init__(self):
        self.heursol_samples = [_AliasSample()]


def test_adapter_prefers_the_parser_over_the_text_fallback():
    parsed = heursol_samples(_AliasResult())  # type: ignore[arg-type]
    assert [(s.name, s.dispatch, s.effort_at) for s in parsed] == [("fpr", 77, 512)]


def test_adapter_falls_back_to_the_log_only_when_it_has_to():
    if parser_supports_heursol():
        assert heursol_samples(SolveResult()) == []
    else:
        assert heursol_samples(SolveResult()) == []
        text = [heursol_line("fj", 0, 0, 100), heur_line("fj", 1)]
        assert [s.name for s in heursol_from_lines(text)] == ["fj"]


# ---------------------------------------------------------------------------
# Dispatch grouping
# ---------------------------------------------------------------------------


def test_process_global_dispatch_ids_do_not_drop_heuristics():
    """The regression that made three of four heuristics vanish.

    `dispatch` is a process-global counter, so at `suite=all` the chain takes
    ids 0..3 while each name has exactly one `[Heur]` line.  Anything that
    treats the id as an index into that name's `[Heur]` list keeps only the
    heuristic that ran first.
    """
    text = probe_log(
        heursol=tuple(
            heursol_line(name, index, 0, 100 * (index + 1))
            for index, name in enumerate(PRESOLVE_HEURISTICS)
        ),
        heur=tuple(heur_line(name, 1000) for name in PRESOLVE_HEURISTICS),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert notes == []
    assert sorted(v.name for v in views) == sorted(PRESOLVE_HEURISTICS)
    assert all(v.accepts == 1 for v in views)


def _run_parts(text: str):
    run_obj = make_run(text)
    return run_obj.result, run_obj.heursols


def test_grouping_delegates_to_the_parser_when_it_offers_it():
    """Process-global, non-dense ids must survive untouched."""
    traces = [
        _StandInTrace(
            "fpr",
            9182,
            4000,
            1000,
            [
                sample("fpr", 9182, 0, 100, obj=10.0),
                sample("fpr", 9182, 1, 250, obj=9.0),
            ],
        ),
        _StandInTrace("scylla", 9183, 2000, 1000, [sample("scylla", 9183, 0, 800)]),
    ]
    views, notes = dispatch_views("i", "all", 0, _StandInResult(traces), [])
    assert notes == []
    assert {(v.name, v.dispatch) for v in views} == {("fpr", 9182), ("scylla", 9183)}
    fpr = next(v for v in views if v.name == "fpr")
    assert fpr.productive == 350  # summed over workers
    assert fpr.stale == 3650


def test_a_devlog_run_without_a_nonzero_count_is_a_diagnostic():
    """Pins issue F1 rather than working around it.

    At `log_dev_level=3` HiGHS prints a block model header the parser does
    not match, so `num_nonzeros` is None on exactly the runs that carry the
    trace.  The nonzero count belongs on the `[Heur]` line; until it is
    there, the dispatch is skipped with a reason instead of being normalised
    by an invented number.
    """
    text = probe_log(
        dev_log=True,
        heursol=(heursol_line("fj", 0, 0, 100),),
        heur=(heur_line("fj", 1000),),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert views == []
    assert any("nonzero count" in n for n in notes)


def test_gaps_are_taken_within_one_worker_series():
    """Two workers interleaved in the log must not be differenced together.

    Objectives descend across the four offers so that every one of them is an
    improvement: since #113 the series is built from offers that moved the
    dispatch's best objective, not from every accepted offer, so a fixture
    that leaves them all at one value has a one-offer series and no gaps.
    """
    text = probe_log(
        heursol=(
            heursol_line("fpr", 0, 0, 100, obj=10.0),
            heursol_line("fpr", 0, 1, 50, obj=9.0),
            heursol_line("fpr", 0, 0, 300, obj=8.0),
            heursol_line("fpr", 0, 1, 400, obj=7.0),
        ),
        heur=(heur_line("fpr", 1000),),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert notes == []
    assert sorted(views[0].gaps) == [50, 100, 200, 350]


def test_the_off_slot_worker_is_kept_out_of_the_gap_distribution():
    """`worker=-1` is LocalMIP's cold-start publish, not a worker's interval.

    It is a real accepted solution, so it counts for the informative set; it
    is not an improvement-free interval any gate could have cut, so pooling
    it would put a whole construction sweep into the quantile.
    """
    text = probe_log(
        heursol=(
            heursol_line("local_mip", 0, -1, 0, obj=10.0),
            heursol_line("local_mip", 0, 0, 400, obj=9.0),
        ),
        heur=(heur_line("local_mip", 1000),),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert notes == []
    assert views[0].gaps == [400]
    assert views[0].off_slot_accepts == 1
    assert views[0].productive == 400
    # It is production, so it counts toward `produced`; whether the instance
    # is informative is the display's call, as everywhere else.
    assert classify_run(make_run(text)).produced


def test_non_monotone_effort_drops_the_dispatch_rather_than_clipping():
    text = probe_log(
        heursol=(
            heursol_line("fj", 0, 0, 500),
            heursol_line("fj", 0, 0, 100),
        ),
        heur=(heur_line("fj", 1000),),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert views == []
    assert any("non-monotone" in n for n in notes)


def test_a_dispatch_that_never_offered_is_still_counted():
    """It emits no `[HeurSol]` line, and it is what the gate exists to cut.

    Recovering it from `[Heur]` alone is the difference between "stopped
    producing" and "never produced"; leaving it out is what makes an
    events-only quantile too tight.
    """
    text = probe_log(heur=(heur_line("local_mip", 1000, found=0),))
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert notes == []
    assert len(views) == 1
    assert views[0].name == "local_mip"
    assert views[0].stale == 1000 and views[0].productive == 0
    assert summarise_traces(views)["local_mip"].stale_fraction == 1.0


def test_found_without_an_accepted_offer_is_a_disagreement():
    """Only when the log has a trace at all; otherwise it is just unobserved."""
    traced = probe_log(
        heursol=(heursol_line("fpr", 0, 0, 100),),
        heur=(heur_line("fpr", 1000), heur_line("local_mip", 1000, found=1)),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(traced))
    assert [v.name for v in views] == ["fpr"]
    assert any("no accepted [HeurSol] offer" in n for n in notes)

    untraced = probe_log(heur=(heur_line("local_mip", 1000, found=1),))
    views, notes = dispatch_views("i", "all", 0, *_run_parts(untraced))
    assert views == []
    assert any("no [HeurSol] lines at all" in n for n in notes)


def test_a_barren_dispatch_beside_a_productive_one_is_not_double_counted():
    text = probe_log(
        heursol=(heursol_line("fpr", 0, 0, 100),),
        heur=(heur_line("fpr", 1000), heur_line("scylla", 500, found=0)),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert notes == []
    assert {v.name: v.accepts for v in views} == {"fpr": 1, "scylla": 0}
    assert {v.name: v.stale for v in views} == {"fpr": 900, "scylla": 500}


def test_a_truncated_chain_keeps_its_gaps_without_a_total():
    """The third real shape: SIGKILLed mid-dispatch, after some offers.

    The `[Heur]` line closing the dispatch never printed, so `heur_index`
    stays None and the dispatch has no total.  Its completed gaps are still
    real observations; what it cannot contribute is a productive/stale split,
    and inventing one from the offers alone would understate the spend.
    """
    text = probe_log(
        killed=True,
        heursol=(
            heursol_line("fpr", 0, 0, 100, obj=10.0),
            heursol_line("fpr", 0, 0, 350, obj=9.0),
        ),
    )
    views, notes = dispatch_views("i", "all", 0, *_run_parts(text))
    assert notes == []
    assert len(views) == 1
    assert views[0].total_effort is None
    assert views[0].stale is None
    assert views[0].gaps == [100, 250]

    summary = summarise_traces(views)["fpr"]
    assert summary.unknown_total == 1
    assert summary.total_effort == 0 and summary.productive_effort == 0
    assert summary.gaps_per_nnz  # the gaps still count
    assert summary.censored_per_nnz == []


def test_a_truncated_chain_is_reported_as_such(tmp_path):
    logs = {
        "all": {
            "cut": probe_log(killed=True, heursol=(heursol_line("fpr", 0, 0, 100),))
        }
    }
    tree = write_tree(tmp_path, logs, name="truncated")
    ref = write_reference(tmp_path, ["cut"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "no [Heur] line (a killed run)" in res.stdout


def test_single_worker_flag_filters_out_multiworker_logs():
    text = probe_log(
        threads=16,
        heursol=(heursol_line("fj", 0, 3, 100),),
        heur=(heur_line("fj", 1000),),
    )
    assert dispatch_views("i", "all", 0, *_run_parts(text))[0] != []
    views, notes = dispatch_views(
        "i", "all", 0, *_run_parts(text), single_worker_only=True
    )
    assert views == []
    assert any("thread count 16" in n for n in notes)


# ---------------------------------------------------------------------------
# The informative set
# ---------------------------------------------------------------------------


def test_chain_sources_are_the_ones_the_patch_assigns():
    assert set("AMGJ") == CHAIN_SOURCES  # FPR / LocalMIP / Scylla / FJ
    assert "D" not in CHAIN_SOURCES  # fpr_lp is dive-time
    assert not CHAIN_SOURCES & set("lpuzXYTB")  # HiGHS's own


def test_a_trivial_upper_solution_is_not_chain_evidence():
    """The real `supportcase10` shape, and the primary path of the filter.

    HiGHS's trivial heuristics run inside `runSetup()`, before the chain, so
    a presolve-only run can report a solution none of our heuristics found.
    An instance solved only that way is a constant in every comparison the
    search makes, which is what the hard tier is for.
    """
    verdict = classify_run(make_run(probe_log(rows=(("u", 70),))))
    assert verdict.evidence == "source"
    assert not verdict.informative
    assert verdict.trivial_only

    scan = informative_set({"supportcase10": [make_run(probe_log(rows=(("u", 70),)))]})
    assert scan.excluded == ["supportcase10"]
    assert scan.reasons["supportcase10"] == REASON_TRIVIAL_ONLY
    assert "trivial" in scan.details["supportcase10"]


def test_a_chain_sourced_solution_is_evidence():
    """The real `mad` shape: eight `A` rows from FPR."""
    text = probe_log(rows=(("A", 5.12), ("A", 3.77), ("A", 1.48)))
    verdict = classify_run(make_run(text))
    assert verdict.evidence == "source" and verdict.informative
    assert not verdict.trivial_only


def test_heursol_counts_acceptance_without_deciding_membership():
    """`produced` and `informative` are different questions.

    The pool accepting an offer is production; the offer becoming the
    reported incumbent is what a candidate configuration can be scored on.
    """
    offers = (
        heursol_line("fj", 0, 0, 100, accepted=0),
        heursol_line("fj", 0, 0, 200, accepted=1),
    )
    ledger = (heur_line("fj", 1000),)

    unimproving = classify_run(make_run(probe_log(heursol=offers, heur=ledger)))
    assert unimproving.evidence == "heursol" and unimproving.accepted == 1
    assert unimproving.produced
    assert not unimproving.informative
    assert unimproving.disagrees

    improving = classify_run(
        make_run(probe_log(rows=(("J", 5.0),), heursol=offers, heur=ledger))
    )
    assert improving.produced and improving.informative
    assert not improving.disagrees


def test_fpr_lp_offers_are_not_presolve_evidence():
    text = probe_log(
        heursol=(heursol_line("fpr_lp", 0, 0, 100, accepted=1),),
        heur=(heur_line("fpr_lp", 1000, phase="dive"),),
    )
    verdict = classify_run(make_run(text))
    assert verdict.evidence == "source"
    assert not verdict.informative


def test_heur_found_is_read_as_production_not_as_membership():
    verdict = classify_run(make_run(probe_log(heur=(heur_line("fj", 1000, found=1),))))
    assert verdict.evidence == "heur"
    assert verdict.produced and not verdict.informative


def test_the_verdict_does_not_move_with_the_log_level():
    """The reproducibility hazard the source test exists to remove.

    The filtering pass runs without `--dev-log` and the trajectory pass with
    it.  A predicate only one of them can evaluate would let the same solve
    land in different informative sets.
    """
    for rows, expected in ((("A", 5.0),), True), ((("u", 70.0),), False):
        plain = classify_run(make_run(probe_log(rows=rows)))
        traced = classify_run(
            make_run(
                probe_log(
                    rows=rows,
                    heursol=(heursol_line("fj", 0, 0, 200),),
                    heur=(heur_line("fj", 1000),),
                )
            )
        )
        assert plain.informative is expected
        assert traced.informative is expected


def test_a_killed_run_can_still_show_the_chain_produced_the_incumbent():
    """`[Heur]` is written when a dispatch ends, so a kill leaves none.

    The probe needs a per-run cap, so this shape is built into the data it
    collects: on the pilot `fj` tree, `neos-4532248-waihi` and
    `nursesched-medium-hint03` were killed at 210 s with `J`-sourced rows
    and no `[Heur]` line, which an acceptance-based predicate would score as
    having produced nothing.
    """
    verdict = classify_run(make_run(probe_log(rows=(("J", 77.4),), killed=True)))
    assert verdict.informative
    assert verdict.produced is None  # no ledger to say
    assert verdict.killed

    scan = informative_set(
        {"neos-4532248-waihi": [make_run(probe_log(rows=(("J", 77.4),), killed=True))]}
    )
    assert scan.informative == ["neos-4532248-waihi"]


def test_produced_but_never_improved_is_its_own_reason():
    """A heuristic that produces and never improves is a datum about it.

    Distinct from trivial-only, where our chain produced nothing at all.
    """
    run_obj = make_run(
        probe_log(
            rows=(("u", 70.0),),
            heursol=(heursol_line("fj", 0, 0, 200),),
            heur=(heur_line("fj", 1000),),
        ),
        config="all",
    )
    scan = informative_set({"supportcase10": [run_obj]})
    assert scan.excluded == ["supportcase10"]
    assert scan.reasons["supportcase10"] == REASON_PRODUCED_NOT_IMPROVED
    assert scan.disagreements["supportcase10"] == ["all/seed0"]


def test_a_tree_without_the_trace_keeps_the_membership_and_loses_the_label():
    """Same instances excluded; only the reason is coarser."""
    traced = make_run(
        probe_log(
            rows=(("u", 70.0),),
            heursol=(heursol_line("fj", 0, 0, 200),),
            heur=(heur_line("fj", 1000),),
        )
    )
    plain = make_run(probe_log(rows=(("u", 70.0),)))
    with_trace = informative_set({"x": [traced]})
    without = informative_set({"x": [plain]})
    assert with_trace.excluded == without.excluded == ["x"]
    assert with_trace.reasons["x"] == REASON_PRODUCED_NOT_IMPROVED
    assert without.reasons["x"] == REASON_TRIVIAL_ONLY
    assert not without.disagreements


def test_informative_filter_is_a_union_over_configurations():
    """One config's failure must not exclude an instance.

    Cracking a previously-unsolved instance is the headline capability of a
    feasibility campaign, so an instance only *some* candidate can solve has
    to stay in the set.
    """
    barren = make_run(probe_log(), config="a")
    cracked = make_run(probe_log(rows=(("A", 10),)), config="b")
    both = informative_set({"x": [barren, cracked]})
    assert both.informative == ["x"] and both.excluded == []
    alone = informative_set({"x": [barren]})
    assert alone.excluded == ["x"]
    assert alone.reasons["x"] == REASON_NO_ACCEPTANCE


def test_killed_runs_are_unreached_not_never_feasible():
    killed = make_run(probe_log(killed=True), config="a")
    scan = informative_set({"ns1760995": [killed, killed]})
    assert scan.reasons["ns1760995"] == REASON_UNREACHED
    assert "killed" in scan.details["ns1760995"]


def test_a_clean_barren_run_outranks_a_killed_one():
    killed = make_run(probe_log(killed=True), config="a")
    clean = make_run(probe_log(), config="b")
    scan = informative_set({"x": [killed, clean]})
    assert scan.reasons["x"] == REASON_NO_ACCEPTANCE
    assert scan.details["x"] == "1 of 2 run(s) killed"


# ---------------------------------------------------------------------------
# Trajectories and the stall estimate
# ---------------------------------------------------------------------------


def test_quantile_is_type_seven():
    assert math.isnan(quantile([], 0.5))
    assert quantile([5.0], 0.9) == 5.0
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert math.isclose(quantile([1.0, 2.0, 3.0, 4.0], 0.9), 3.7)


def test_km_quantile_matches_the_empirical_one_without_censoring():
    values = [float(v) for v in range(1, 21)]
    events = [Observation(v, "i", True) for v in values]
    # With no censoring the KM survival is the empirical one: it steps to
    # 0.05 at the 19th of 20 observations, so the p95 is the smallest value
    # whose survival has reached 0.05.  The interpolated type-7 quantile of
    # the same sample is 19.05; the step estimator reports an observed value
    # rather than one between two of them, which is the point of using it.
    assert km_quantile(events, 0.95) == 19.0
    assert km_quantile(events, 0.5) == 10.0
    assert km_quantile([], 0.95) is None


def test_km_quantile_is_none_when_the_censoring_hides_the_tail():
    """ "Not identifiable" is a real answer and beats extrapolating one."""
    observations = [Observation(1.0, "i", True)] + [
        Observation(2.0, "i", False) for _ in range(50)
    ]
    assert km_quantile(observations, 0.95) is None


def test_barren_dispatches_raise_the_stall_estimate():
    """The bias the censored view exists to remove.

    An events-only p95 is computed conditional on the heuristic producing
    again, so dispatches that stopped producing -- the ones the gate exists
    to cut -- are invisible to it, and the estimate comes out too tight.
    """
    productive = DispatchView(
        instance="easy",
        config="all",
        seed=0,
        name="fpr",
        dispatch=1,
        nnz=100,
        total_effort=400,
        workers=1,
        series=(WorkerSeries(0, (100, 200, 300, 400)),),
    )
    barren = [
        DispatchView(
            instance=f"hard{i}",
            config="all",
            seed=0,
            name="fpr",
            dispatch=10 + i,
            nnz=100,
            total_effort=5000,
            workers=1,
            series=(),
        )
        for i in range(5)
    ]
    events_only = summarise_traces([productive])["fpr"]
    with_barren = summarise_traces([productive, *barren])["fpr"]
    assert events_only.events_p95() == with_barren.events_p95()
    # The censored view either lifts the estimate or declares it unbounded.
    high = with_barren.censored_p95()
    assert high is None or high > events_only.events_p95()


def test_each_instance_carries_equal_weight():
    """One easy instance with many acceptances must not own the quantile."""
    loud = [
        DispatchView(
            instance="easy",
            config="all",
            seed=0,
            name="fj",
            dispatch=d,
            nnz=100,
            total_effort=100,
            workers=1,
            series=(WorkerSeries(0, (100,)),),
        )
        for d in range(50)
    ]
    quiet = DispatchView(
        instance="hard",
        config="all",
        seed=0,
        name="fj",
        dispatch=99,
        nnz=100,
        total_effort=100_000,
        workers=1,
        series=(WorkerSeries(0, (90_000,)),),
    )
    trajectory = summarise_traces([*loud, quiet])["fj"]
    weights = {}
    for o in trajectory.weighted():
        weights[o.instance] = weights.get(o.instance, 0.0) + o.weight
    assert math.isclose(weights["easy"], 1.0)
    assert math.isclose(weights["hard"], 1.0)


def test_stall_range_scales_only_for_dispatch_scoped_options():
    """`fpr`/`local_mip` are divided by N on the way to the per-worker gate.

    The measured gaps are raw effort per nonzero and the option is a multiple
    of `nnz << 10` since #116, so the base divides out of every expectation
    here -- 40 effort units per nonzero is `40 / 1024` of a base budget.
    """
    base = float(1 << 10)
    gaps = [Observation(10.0, "i", True) for _ in range(20)]
    for name, expected in (
        ("fpr", 40 / base),
        ("local_mip", 40 / base),
        ("fj", 10 / base),
        ("scylla", 10 / base),
    ):
        t = HeuristicTrajectory(name=name, observations=list(gaps))
        assert t.stall_range(4)[0] == expected
    assert HeuristicTrajectory(name="fj").stall_range(4) == (None, None)


def test_summarise_normalises_by_nonzeros():
    view = DispatchView(
        instance="i",
        config="c",
        seed=0,
        name="fpr",
        dispatch=0,
        nnz=100,
        total_effort=1000,
        workers=1,
        series=(WorkerSeries(worker=0, accepted_efforts=(200, 500)),),
    )
    summary = summarise_traces([view])["fpr"]
    assert summary.gaps_per_nnz == [2.0, 3.0]
    assert summary.censored_per_nnz == [5.0]
    assert set(summarise_traces([])) == set(PRESOLVE_HEURISTICS)


def test_stale_effort_is_split_across_workers_before_pooling():
    """Gaps are per worker, so the censored interval must be too."""
    view = DispatchView(
        instance="i",
        config="c",
        seed=0,
        name="scylla",
        dispatch=0,
        nnz=10,
        total_effort=800,
        workers=4,
        series=(),
    )
    summary = summarise_traces([view])["scylla"]
    assert summary.censored_per_nnz == [20.0] * 4  # 800 / (10 nnz * 4 workers)


def test_parse_quantiles_rejects_unusable_values():
    assert parse_quantiles("0.5, 0.9 ") == (0.5, 0.9)
    for bad in ("", "0", "1.5", "-0.1", "0.9,0.5", "0.5,0.5", "x", "nan"):
        try:
            parse_quantiles(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _tree(tmp_path):
    """Two configs over five instances, with every interesting log shape."""
    traced = probe_log(
        rows=(("A", 12.0),),
        heursol=(
            heursol_line("fpr", 0, 0, 100),
            heursol_line("fpr", 0, 1, 250),
            heursol_line("fpr", 0, 0, 400),
        ),
        heur=(heur_line("fpr", 1000), heur_line("local_mip", 2000, found=0)),
    )
    barren = probe_log()
    # The chain accepted a solution that never beat HiGHS's trivial upper
    # bound: production without a signal any candidate can be scored on.
    unimproving = probe_log(
        rows=(("u", 70),),
        heursol=(heursol_line("fj", 0, 0, 200),),
        heur=(heur_line("fj", 1000),),
    )
    logs = {
        "a": {
            "easy": traced,
            "onlyb": barren,
            "barren": barren,
            "ns1760995": probe_log(killed=True),
            "trivial": probe_log(rows=(("u", 70),)),
            "unimproving": unimproving,
        },
        "b": {
            "easy": traced,
            "onlyb": probe_log(rows=(("M", 1.0),)),
            "barren": barren,
            "ns1760995": probe_log(killed=True),
            "trivial": probe_log(rows=(("u", 70),)),
            "unimproving": unimproving,
        },
    }
    tree = write_tree(tmp_path, logs)
    ref = write_reference(tmp_path, sorted(logs["a"]))
    return tree, ref


def test_end_to_end_splits_the_set_and_reports_trajectories(tmp_path):
    tree, ref = _tree(tmp_path)
    informative = os.path.join(str(tmp_path), "informative.txt")
    hard = os.path.join(str(tmp_path), "hard.txt")
    res = run(
        tree,
        "--instances",
        ref,
        "--informative-output",
        informative,
        "--hard-tier-output",
        hard,
    )
    assert res.returncode == 0, res.stdout + res.stderr

    assert load_instances(informative) == ["easy", "onlyb"]
    assert load_instances(hard) == [
        "barren",
        "ns1760995",
        "trivial",
        "unimproving",
    ]

    with open(hard) as f:
        hard_text = f.read()
    assert "did any configuration crack it" in hard_text
    for reason in (
        "unreached",
        "trivial-only",
        "no-acceptance",
        "produced-not-improved",
    ):
        assert reason in hard_text
    # The rule the set was drawn under is stated where the set lives.
    with open(informative) as f:
        assert "never became the incumbent" in f.read()

    assert "ns1760995" in res.stdout and "trivial" in res.stdout
    assert "Informative set: 2 of 6" in res.stdout
    # The two signals disagreeing is reported, not silently resolved.
    assert "found=1 and the display disagree" in res.stdout
    assert "stall_lo" in res.stdout and "stall_hi" in res.stdout
    assert "mip_heuristic_fpr_stall" in res.stdout


def test_outputs_are_byte_identical_across_runs(tmp_path):
    tree, ref = _tree(tmp_path)
    informative = os.path.join(str(tmp_path), "informative.txt")
    hard = os.path.join(str(tmp_path), "hard.txt")
    outputs = []
    for hash_seed in ("0", "12345"):
        res = run(
            tree,
            "--instances",
            ref,
            "--informative-output",
            informative,
            "--hard-tier-output",
            hard,
            "--hard-tier-size",
            "1",
            hash_seed=hash_seed,
        )
        assert res.returncode == 0, res.stdout + res.stderr
        with open(informative, "rb") as f:
            first = f.read()
        with open(hard, "rb") as f:
            outputs.append((first, f.read()))
    assert outputs[0] == outputs[1]


def test_hard_tier_size_samples_deterministically(tmp_path):
    tree, ref = _tree(tmp_path)
    hard = os.path.join(str(tmp_path), "hard.txt")
    res = run(
        tree, "--instances", ref, "--hard-tier-output", hard, "--hard-tier-size", "1"
    )
    assert res.returncode == 0, res.stderr
    assert len(load_instances(hard)) == 1

    bad = run(
        tree, "--instances", ref, "--hard-tier-output", hard, "--hard-tier-size", "9"
    )
    assert bad.returncode == 1 and "hard-tier-size" in bad.stderr


def test_narrowing_to_one_config_warns_about_the_union(tmp_path):
    tree, ref = _tree(tmp_path)
    informative = os.path.join(str(tmp_path), "informative.txt")
    res = run(
        tree, "--instances", ref, "--configs", "a", "--informative-output", informative
    )
    assert res.returncode == 0, res.stderr
    assert load_instances(informative) == ["easy"]
    assert "union over configurations" in res.stdout


def test_a_full_solve_tree_is_refused(tmp_path):
    """The validation mistake this guard exists to prevent.

    Pointed at a full-solve vanilla tree every run falls to the weakest
    evidence tier, "informative" degrades to "the solver found something
    inside its time limit", and the emitted list gets pinned by digest into
    a tuning-set header as though it meant something.
    """
    logs = {"vanilla": {"easy": probe_log(rows=(("A", 1.0),))}}
    tree = write_tree(tmp_path, logs, name="full", opts=FULL_SOLVE_OPTS)
    ref = write_reference(tmp_path, ["easy"])

    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert "presolve_only" in res.stderr

    allowed = run(tree, "--instances", ref, "--allow-non-probe")
    assert allowed.returncode == 0, allowed.stderr
    assert "WARNING" in allowed.stdout


def test_an_unpatched_tree_is_refused(tmp_path):
    """Without the marker the binary ran none of the chain."""
    logs = {"vanilla": {"easy": probe_log(rows=(("H", 1.0),), patched=False)}}
    tree = write_tree(tmp_path, logs, name="unpatched")
    ref = write_reference(tmp_path, ["easy"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert "patch active" in res.stderr


def test_a_missing_opts_file_is_reported_not_assumed(tmp_path):
    logs = {"all": {"easy": probe_log(rows=(("A", 1.0),))}}
    tree = write_tree(tmp_path, logs, name="noopts", opts=None)
    ref = write_reference(tmp_path, ["easy"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert "no .opts" in res.stderr


def test_a_tree_missing_the_reference_list_is_a_refusal(tmp_path):
    tree, _ = _tree(tmp_path)
    ref = write_reference(tmp_path, ["easy", "absent"], filename="wide.txt")
    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert "absent" in res.stderr

    allowed = run(tree, "--instances", ref, "--allow-missing")
    assert allowed.returncode == 0, allowed.stderr
    assert "Informative set: 1 of 1" in allowed.stdout


def test_a_log_with_no_evidence_at_all_is_a_refusal(tmp_path):
    logs = {
        "a": {"easy": probe_log(rows=(("A", 1.0),)), "void": "Running HiGHS 1.15.1\n"}
    }
    tree = write_tree(tmp_path, logs, name="broken")
    ref = write_reference(tmp_path, ["easy", "void"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert "void" in res.stderr and "TIMEOUT" in res.stderr


def test_an_err_file_is_named_as_a_failed_run(tmp_path):
    logs = {"a": {"easy": probe_log(rows=(("A", 1.0),))}}
    tree = write_tree(tmp_path, logs, name="errtree")
    with open(os.path.join(tree, "a", "seed0", "broken.log.err"), "w") as f:
        f.write("HiGHS exited 1\n")
    ref = write_reference(tmp_path, ["easy", "broken"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert ".log.err" in res.stderr


def test_a_tree_that_mixes_worker_counts_says_so(tmp_path):
    def traced(threads: int) -> str:
        return probe_log(
            threads=threads,
            rows=(("A", 1.0),),
            heursol=(heursol_line("fpr", 0, 0, 100),),
            heur=(heur_line("fpr", 1000),),
        )

    logs = {"a": {"easy": traced(2)}, "b": {"easy": traced(16)}}
    tree = write_tree(tmp_path, logs, name="mixed")
    ref = write_reference(tmp_path, ["easy"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "mixes machines" in res.stdout

    same = write_tree(tmp_path, {"a": {"easy": traced(2)}}, name="same")
    res = run(same, "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "mixes machines" not in res.stdout


def test_a_single_config_directory_is_its_own_config(tmp_path):
    logs = {"probeconf": {"easy": probe_log(rows=(("A", 1.0),))}}
    root = write_tree(tmp_path, logs, name="solo")
    ref = write_reference(tmp_path, ["easy"])
    res = run(os.path.join(root, "probeconf"), "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "Informative set: 1 of 1" in res.stdout


# ---------------------------------------------------------------------------
# The budget headroom check (#113)
# ---------------------------------------------------------------------------
#
# The calibration probe runs at an effort whose budget cannot be reached, so
# that the wall clock is the single stopping rule and the trace measures the
# heuristic rather than the setting being derived from it.  A tree that
# violates that produces a truncated yield curve which looks exactly like a
# converged one, so it is checked rather than assumed.


def _traced_run(effort_charged: int, nnz: int = 4096) -> str:
    return probe_log(
        rows=(("A", 1.0),),
        nnz=nnz,
        heursol=(heursol_line("fpr", 0, 0, effort_charged // 2),),
        heur=(heur_line("fpr", effort_charged),),
    )


def _opts(effort: float) -> str:
    return (
        "mip_heuristic_suite = fpr\n"
        f"mip_heuristic_fpr_effort = {effort}\n"
        "mip_heuristic_fpr_stall = 0\n"
        "mip_heuristic_presolve_only = true\n"
        "random_seed = 0\n"
    )


def test_a_clock_bound_dispatch_passes_the_budget_check(tmp_path):
    # nnz 4096 at effort 1e4 is a budget of 3.4e12; a dispatch charging 1e6
    # stopped for some other reason, which is what the probe wants.
    tree = write_tree(
        tmp_path,
        {"fpr": {"easy": _traced_run(1_000_000)}},
        name="free",
        opts=_opts(1e4),
    )
    ref = write_reference(tmp_path, ["easy"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "budget check   1/1 traced dispatch(es) clock-bound" in res.stdout
    assert "effort budget" not in res.stdout


def test_a_budget_bound_dispatch_is_a_warning_naming_it(tmp_path):
    # Same charged effort against the shipped default: nnz 4096 at effort
    # 0.0884 is a budget of 2.97e7... so charge past it.
    tree = write_tree(
        tmp_path,
        {"fpr": {"easy": _traced_run(30_000_000)}},
        name="bound",
        opts=_opts(0.0884),
    )
    ref = write_reference(tmp_path, ["easy"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "budget check   0/1 traced dispatch(es) clock-bound" in res.stdout
    assert "reached 95% of their effort budget" in res.stdout
    # Naming it is the point: which heuristic on which instance is what says
    # whether the tree is unusable or one arm of it is.
    assert "fpr on easy" in res.stdout


def test_an_unrecorded_effort_is_neither_evidence_nor_a_warning(tmp_path):
    # A `.opts` that never set the option records the shipped default, not a
    # known value.  Inferring it would make an unrecorded configuration look
    # recorded — and the inference would have to come from the very effort
    # the check is comparing against.
    tree = write_tree(
        tmp_path,
        {"fpr": {"easy": _traced_run(30_000_000)}},
        name="silent",
        opts="mip_heuristic_presolve_only = true\nrandom_seed = 0\n",
    )
    ref = write_reference(tmp_path, ["easy"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "budget check   0/0 traced dispatch(es) clock-bound, 1 unrecorded" in (
        res.stdout
    )
    assert "effort budget" not in res.stdout


def test_fj_budget_is_read_per_worker(tmp_path):
    # FJ's option sizes one *worker's* allowance, so a dispatch's budget is
    # the option times the worker count.  Read as a dispatch budget, this run
    # would be 8x over and reported as budget-bound.
    charged = 4 * (4096 << 12) * 8  # comfortably inside the budget below
    log = probe_log(
        threads=8,
        nnz=4096,
        rows=(("J", 1.0),),
        heursol=(heursol_line("fj", 0, 0, charged // 2),),
        heur=(heur_line("fj", charged),),
    )
    opts = (
        "mip_heuristic_suite = fj\n"
        "mip_heuristic_fj_effort = 20.0\n"  # 20 base budgets, per worker
        "mip_heuristic_fj_stall = 0\n"
        "mip_heuristic_presolve_only = true\n"
        "random_seed = 0\n"
    )
    tree = write_tree(tmp_path, {"fj": {"easy": log}}, name="perworker", opts=opts)
    ref = write_reference(tmp_path, ["easy"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "budget check   1/1 traced dispatch(es) clock-bound" in res.stdout


# ---------------------------------------------------------------------------
# Smoke: the pieces the report is assembled from still fit together (#113)
# ---------------------------------------------------------------------------
#
# Deliberately shallow.  These do not check the statistics — that is what the
# cases above are for — they check the joins that broke while this was being
# built and that a full run only reveals four minutes in: reading a field
# through the parser's spelling rather than ours, and the CLI actually
# producing the artifacts the campaign consumes.


def test_the_improvement_filter_reads_either_sample_shape():
    """`obj` in our own adapter type, `objective` in the parser's.

    Reading it by direct attribute access works against whichever type
    happens to be in the fixtures and raises against the other, and only on
    a tree big enough to have taken the other branch.
    """
    ours = [
        parse_heursol_line(heursol_line("fpr", 0, 0, 100, obj=10.0)),
        parse_heursol_line(heursol_line("fpr", 0, 0, 200, obj=9.0)),
    ]
    assert improving_offers(ours, "min") == {id(ours[0]), id(ours[1])}

    theirs = [_AliasSample(), _AliasSample()]
    theirs[0].objective, theirs[1].objective = 10.0, 9.0
    assert improving_offers(theirs, "min") == {id(theirs[0]), id(theirs[1])}


def test_only_offers_that_move_the_best_objective_count():
    # The third is accepted, but for the pool's top-K, not for beating best.
    offers = [
        parse_heursol_line(heursol_line("fpr", 0, 0, effort, obj=obj))
        for effort, obj in ((100, 10.0), (200, 9.0), (300, 9.5))
    ]
    improving = improving_offers(offers, "min")
    assert improving == {id(offers[0]), id(offers[1])}
    # ... and the sense decides which way "better" runs: read as a
    # maximisation, the same descending sequence improves exactly once.
    assert improving_offers(offers, "max") == {id(offers[0])}


def test_the_sense_comes_off_the_incumbent_trajectory():
    falling = probe_log(rows=(("A", 10.0), ("A", 4.0)))
    rising = probe_log(rows=(("A", 4.0), ("A", 10.0)))
    assert objective_sense(parse_log(falling)) == "min"
    assert objective_sense(parse_log(rising)) == "max"
    # One row says nothing; MIPLIB's convention is the fallback.
    assert objective_sense(parse_log(probe_log(rows=(("A", 4.0),)))) == "min"


def test_the_cli_writes_the_artifacts_the_campaign_consumes(tmp_path):
    logs = {
        "fpr": {
            "easy": probe_log(
                rows=(("A", 1.0),),
                heursol=(heursol_line("fpr", 0, 0, 100, obj=1.0),),
                heur=(heur_line("fpr", 1000),),
            )
        }
    }
    tree = write_tree(tmp_path, logs, name="smoke")
    ref = write_reference(tmp_path, ["easy"])
    defaults = os.path.join(str(tmp_path), "defaults.json")
    report = os.path.join(str(tmp_path), "report.txt")
    res = run(
        tree,
        "--instances",
        ref,
        "--defaults-output",
        defaults,
        "--report-output",
        report,
    )
    assert res.returncode == 0, res.stderr

    data = json.loads(Path(defaults).read_text())
    assert set(data) == {"source_tree", "provenance", "heuristics"}
    assert set(data["heuristics"]) == set(PRESOLVE_HEURISTICS)
    fpr = data["heuristics"]["fpr"]
    # The four numbers #107 is handed, and the scope they are only valid in.
    for key in ("effort", "stall", "effort_shipped", "median_gap_to_best_known"):
        assert key in fpr, key
    assert data["provenance"]["workers_observed"] is not None

    text = Path(report).read_text()
    for section in (
        "Informative set:",
        "Solution quality against best known",
        "Proposed stall ranges",
        "Proposed effort vector",
    ):
        assert section in text, section
