"""Unit tests for bench/analyze_presolve_probe.py.

Everything runs against logs synthesised in a tmp dir: no probe, no MIPLIB,
no solver.  The `[HeurSol]` lines are written to the frozen #106 contract, so
these tests are also the record of what this module's adapter expects of
`bench/parse_highs_log.py` — which is another track's file and may not carry
the sample type yet.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_presolve_probe import (
    PRESOLVE_HEURISTICS,
    REASON_NO_ACCEPTANCE,
    REASON_UNREACHED,
    AdapterError,
    DispatchTrace,
    HeuristicTrajectory,
    ProbeRun,
    WorkerSeries,
    classify_run,
    dispatch_traces,
    heursol_from_text,
    heursol_samples,
    informative_set,
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

_TABLE_HEADER = (
    "Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol"
    "              Gap | Cuts InLp Confl. | LpIters     Time\n"
)


# ---------------------------------------------------------------------------
# Synthetic logs
# ---------------------------------------------------------------------------


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
    nnz: int = 1000,
    threads: int = 1,
    heur: tuple[str, ...] = (),
    heursol: tuple[str, ...] = (),
    incumbent: float | None = None,
    primal: float | None = 10.0,
    status: str = "Optimal",
    killed: bool = False,
    model_header: bool = True,
) -> str:
    """One presolve-only probe run's stdout.

    `primal` with no `incumbent` is the shape a presolve-only exit produces
    when it never prints a display-table row: the Solving report is the only
    place the solution appears.
    """
    lines = ["Running HiGHS 1.15.1", "mip-heuristics patch active"]
    if model_header:
        lines.append(
            f"MIP probe has 10 rows; 20 cols; {nnz} nonzeros; "
            "20 integer variables (20 binary)"
        )
    lines.append("Solving MIP model with:")
    lines.append(
        f"   Thread count {threads} (of 32 threads). Using {threads} max workers."
    )
    if incumbent is not None:
        lines.append(_TABLE_HEADER.rstrip("\n"))
        lines.append(
            f"H       0       0         0   0.00%          0              "
            f"{incumbent}              Large      0      0      0       0.0   1.5s"
        )
    lines += list(heursol)
    lines += list(heur)
    if killed:
        lines.append("TIMEOUT: process killed after 61.0s")
        return "\n".join(lines) + "\n"
    lines.append("Solving report")
    lines.append(f"  Status            {status}")
    if primal is not None:
        lines.append(f"  Primal bound      {primal}")
    lines.append("  Nodes             0")
    return "\n".join(lines) + "\n"


def write_tree(tmp_path, logs: dict[str, dict[str, str]], name: str = "probe") -> str:
    """Materialise `<tmp>/<name>/<config>/seed<N>/<instance>.log`.

    `logs` maps config -> instance -> log text, one seed (`seed0`) unless the
    instance key carries a `@<seed>` suffix.
    """
    root = os.path.join(str(tmp_path), name)
    for config, entries in logs.items():
        for key, text in entries.items():
            instance, _, seed = key.partition("@")
            seed_dir = os.path.join(root, config, f"seed{seed or 0}")
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, f"{instance}.log"), "w") as f:
                f.write(text)
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
    # A finite primal bound with no incumbent line is the normal shape of a
    # presolve-only exit, and `parse_log` warns on it because in a full-solve
    # tree it means a missing source code.  `load_probe_tree` tallies those
    # warnings rather than echoing them; here they are simply noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = parse_log(text)
    return ProbeRun(
        config=config,
        seed=seed,
        instance=instance,
        result=result,
        heursols=heursol_from_text(text),
    )


# ---------------------------------------------------------------------------
# The [HeurSol] adapter
# ---------------------------------------------------------------------------


def test_adapter_parses_key_value_fields_in_any_order():
    """Order-insensitive, and an added field must not break the adapter.

    The contract requires `key=value` parsing rather than a positional regex
    precisely so a later field is additive.
    """
    line = (
        "[HeurSol] accepted=1 obj=-3.5 worker=2 name=local_mip effort_at=4096 "
        "dispatch=1 wall_ms=-0.5 pool_rank=3"
    )
    sample = parse_heursol_line(line)
    assert sample is not None
    assert sample.name == "local_mip"
    assert sample.dispatch == 1
    assert sample.worker == 2
    assert sample.effort_at == 4096
    assert sample.wall_ms == -0.5  # the solver clock is not monotonic
    assert sample.obj == -3.5
    assert sample.accepted is True


def test_adapter_reads_a_worker_less_line_as_worker_none():
    """A build predating the amended contract must not silently become worker 0.

    Merging every worker into one series is exactly the interleaving artefact
    `worker=` was added to remove.
    """
    sample = parse_heursol_line(heursol_line("fj", 0, None, 100))
    assert sample is not None and sample.worker is None


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


class _StandInSample:
    """A parser sample type spelled with the alternative field names."""

    def __init__(self, heuristic, dispatch, worker, effort, wall_ms, objective, ok):
        self.heuristic = heuristic
        self.dispatch = dispatch
        self.worker = worker
        self.effort = effort
        self.wall_ms = wall_ms
        self.objective = objective
        self.accepted = ok


class _StandInResult:
    heursol_samples: list[_StandInSample]

    def __init__(self, samples):
        self.heursol_samples = samples


def test_adapter_prefers_the_parser_over_the_text_fallback():
    """Once `parse_highs_log` carries the samples, the text is not consulted."""
    stand_in = _StandInResult([_StandInSample("fpr", 0, 1, 512, 2.0, 7.5, True)])
    text = heursol_line("scylla", 9, 9, 999)
    samples = heursol_samples(stand_in, text)  # type: ignore[arg-type]
    assert [s.name for s in samples] == ["fpr"]
    assert samples[0].effort_at == 512 and samples[0].worker == 1


def test_adapter_falls_back_to_the_log_text_only_when_it_has_to():
    text = heursol_line("fj", 0, 0, 100)
    if parser_supports_heursol():
        # Track B has landed: an empty SolveResult means the log had none.
        assert heursol_samples(SolveResult(), text) == []
    else:
        assert [s.name for s in heursol_samples(SolveResult(), text)] == ["fj"]


# ---------------------------------------------------------------------------
# The informative set
# ---------------------------------------------------------------------------


def test_heursol_evidence_counts_accepted_presolve_offers():
    text = probe_log(
        heursol=(
            heursol_line("fj", 0, 0, 100, accepted=0),
            heursol_line("fj", 0, 0, 200, accepted=1),
        ),
        heur=(heur_line("fj", 1000),),
    )
    verdict = classify_run(make_run(text))
    assert verdict.evidence == "heursol"
    assert verdict.accepted == 1
    assert verdict.informative


def test_fpr_lp_offers_are_not_presolve_evidence():
    """`fpr_lp` runs on the far side of a root LP a probe never reaches."""
    text = probe_log(
        heursol=(heursol_line("fpr_lp", 0, 0, 100, accepted=1),),
        heur=(heur_line("fpr_lp", 1000, phase="dive"),),
        primal=None,
        status="Time limit reached",
    )
    verdict = classify_run(make_run(text))
    assert verdict.evidence == "bound"
    assert not verdict.informative


def test_evidence_falls_back_to_the_primal_bound_without_dev_log():
    """The filtering pass runs without `--dev-log`; it must still classify.

    A presolve-only run exits before the root LP, so a finite primal bound in
    the Solving report is a solution the chain produced — even when the exit
    path printed no display-table row for it to be read from.
    """
    verdict = classify_run(make_run(probe_log(primal=10.0)))
    assert verdict.evidence == "bound"
    assert verdict.informative

    barren = classify_run(make_run(probe_log(primal=None, status="Time limit reached")))
    assert not barren.informative


def test_heur_evidence_is_used_when_the_trace_is_absent():
    text = probe_log(heur=(heur_line("fj", 1000, found=1),), primal=None)
    verdict = classify_run(make_run(text))
    assert verdict.evidence == "heur"
    assert verdict.informative


def test_informative_filter_is_a_union_over_configurations():
    """The whole point: one config's failure must not exclude an instance.

    Cracking a previously-unsolved instance is the headline capability of a
    feasibility campaign, so an instance only *some* candidate can solve has
    to stay in the set.
    """
    barren = make_run(probe_log(primal=None, status="Time limit reached"), config="a")
    cracked = make_run(probe_log(primal=10.0), config="b")
    both = informative_set({"x": [barren, cracked]})
    assert both.informative == ["x"] and both.excluded == []
    # The single-config case is the narrowing special case, not the default.
    alone = informative_set({"x": [barren]})
    assert alone.excluded == ["x"]
    assert alone.reasons["x"] == REASON_NO_ACCEPTANCE


def test_killed_runs_are_unreached_not_never_feasible():
    """A killed probe log is a legitimate result, not missing data.

    `ns1760995` spends the entire 600 s limit inside HiGHS's own presolve, so
    the screen never looks at the model.  That is a different fact from "the
    heuristics ran and found nothing", and the hard tier records which.
    """
    killed = make_run(probe_log(killed=True, primal=None), config="a")
    scan = informative_set({"ns1760995": [killed, killed]})
    assert scan.excluded == ["ns1760995"]
    assert scan.reasons["ns1760995"] == REASON_UNREACHED
    assert "killed" in scan.details["ns1760995"]


def test_a_clean_barren_run_outranks_a_killed_one():
    killed = make_run(probe_log(killed=True, primal=None), config="a")
    clean = make_run(probe_log(primal=None, status="Time limit reached"), config="b")
    scan = informative_set({"x": [killed, clean]})
    assert scan.reasons["x"] == REASON_NO_ACCEPTANCE
    assert scan.details["x"] == "1 of 2 run(s) killed"


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


def test_quantile_is_type_seven():
    assert quantile([], 0.5) != quantile([], 0.5)  # nan
    assert quantile([5.0], 0.9) == 5.0
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert math.isclose(quantile([1.0, 2.0, 3.0, 4.0], 0.9), 3.7)


def test_gaps_are_taken_within_one_worker_series():
    """Two workers interleaved in the log must not be differenced together.

    Worker 0 accepts at 100 and 300, worker 1 at 50 and 400.  Per worker the
    gaps are 100/200 and 50/350; pooling the raw sequence would invent gaps
    of -50, 250, 100.
    """
    text = probe_log(
        threads=2,
        heursol=(
            heursol_line("fpr", 0, 0, 100),
            heursol_line("fpr", 0, 1, 50),
            heursol_line("fpr", 0, 0, 300),
            heursol_line("fpr", 0, 1, 400),
        ),
        heur=(heur_line("fpr", 1000),),
    )
    traces, notes = dispatch_traces(make_run(text))
    assert notes == []
    assert len(traces) == 1
    assert sorted(traces[0].gaps) == [50, 100, 200, 350]


def test_productive_effort_sums_over_workers():
    """Per the contract: `effort_at` is per worker, so a dispatch sums them."""
    text = probe_log(
        threads=2,
        heursol=(
            heursol_line("scylla", 0, 0, 300),
            heursol_line("scylla", 0, 1, 400),
        ),
        heur=(heur_line("scylla", 1000),),
    )
    traces, _ = dispatch_traces(make_run(text))
    assert traces[0].productive == 700
    assert traces[0].stale == 300


def test_non_monotone_effort_drops_the_dispatch_rather_than_clipping():
    """A contract violation is a data error, surfaced, never repaired.

    Dropping the negative difference would bias p90-p95 downward, and a stall
    threshold set too tight costs solutions.
    """
    text = probe_log(
        heursol=(
            heursol_line("fj", 0, 0, 500),
            heursol_line("fj", 0, 0, 100),
        ),
        heur=(heur_line("fj", 1000),),
    )
    traces, notes = dispatch_traces(make_run(text))
    assert traces == []
    assert any("non-monotone" in n for n in notes)


def test_a_worker_less_trace_disables_the_trajectory_pass():
    text = probe_log(
        heursol=(heursol_line("fj", 0, None, 100),),
        heur=(heur_line("fj", 1000),),
    )
    traces, notes = dispatch_traces(make_run(text))
    assert traces == []
    assert any("worker=" in n for n in notes)


def test_a_dispatch_id_outside_the_heur_range_is_a_diagnostic():
    """The 0-based per-name counter is an assumption, so it is checked."""
    text = probe_log(
        heursol=(heursol_line("fpr", 7, 0, 100),),
        heur=(heur_line("fpr", 1000),),
    )
    traces, notes = dispatch_traces(make_run(text))
    assert traces == []
    assert any("0-based" in n for n in notes)


def test_found_without_an_accepted_offer_is_a_disagreement():
    text = probe_log(heur=(heur_line("local_mip", 1000, found=1),))
    traces, notes = dispatch_traces(make_run(text))
    assert traces == []
    assert any("found=1" in n for n in notes)


def test_a_barren_dispatch_is_wholly_stale():
    text = probe_log(heur=(heur_line("local_mip", 1000, found=0),), primal=None)
    traces, notes = dispatch_traces(make_run(text))
    assert notes == []
    assert traces[0].stale == 1000 and traces[0].productive == 0
    summary = summarise_traces(traces)
    assert summary["local_mip"].stale_fraction == 1.0


def test_repeated_dispatches_are_kept_apart():
    """A full solve re-enters the chain for sub-MIPs; #113 sees only the root."""
    text = probe_log(
        heursol=(
            heursol_line("fj", 0, 0, 100),
            heursol_line("fj", 1, 0, 250),
        ),
        heur=(heur_line("fj", 1000), heur_line("fj", 2000)),
    )
    traces, notes = dispatch_traces(make_run(text))
    assert notes == []
    assert [(t.dispatch, t.productive, t.stale) for t in traces] == [
        (0, 100, 900),
        (1, 250, 1750),
    ]


def test_single_worker_flag_filters_out_multiworker_logs():
    text = probe_log(
        threads=16,
        heursol=(heursol_line("fj", 0, 3, 100),),
        heur=(heur_line("fj", 1000),),
    )
    assert dispatch_traces(make_run(text))[0] != []
    traces, notes = dispatch_traces(make_run(text), single_worker_only=True)
    assert traces == []
    assert any("thread count 16" in n for n in notes)


def test_a_log_without_a_model_header_cannot_be_normalised():
    text = probe_log(
        model_header=False,
        heursol=(heursol_line("fj", 0, 0, 100),),
        heur=(heur_line("fj", 1000),),
    )
    traces, notes = dispatch_traces(make_run(text))
    assert traces == []
    assert any("nnz" in n for n in notes)


def test_stall_suggestion_scales_only_for_dispatch_scoped_options():
    """The option units differ per heuristic; the suggestion must follow.

    `fpr` and `local_mip` take a whole-dispatch threshold that `make_budget`
    divides by N, so a per-worker gap has to be multiplied back up.  `fj`'s
    option is per-worker already, and `scylla` takes the dispatch-level value
    as its worker threshold.
    """
    gaps = [10.0] * 10
    assert HeuristicTrajectory(name="fpr", gaps_per_nnz=gaps).suggested_stall(4) == 40
    assert (
        HeuristicTrajectory(name="local_mip", gaps_per_nnz=gaps).suggested_stall(4)
        == 40
    )
    assert HeuristicTrajectory(name="fj", gaps_per_nnz=gaps).suggested_stall(4) == 10
    assert (
        HeuristicTrajectory(name="scylla", gaps_per_nnz=gaps).suggested_stall(4) == 10
    )
    assert HeuristicTrajectory(name="fj").suggested_stall(4) is None


def test_gaps_are_normalised_by_nonzeros():
    trace = DispatchTrace(
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
    summary = summarise_traces([trace])["fpr"]
    assert summary.gaps_per_nnz == [2.0, 3.0]
    assert summary.tails_per_nnz == [5.0]
    assert set(summarise_traces([])) == set(PRESOLVE_HEURISTICS)


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
        threads=2,
        heursol=(
            heursol_line("fpr", 0, 0, 100),
            heursol_line("fpr", 0, 1, 250),
            heursol_line("fpr", 0, 0, 400),
        ),
        heur=(heur_line("fpr", 1000), heur_line("local_mip", 2000, found=0)),
    )
    barren = probe_log(primal=None, status="Time limit reached")
    logs = {
        "a": {
            "easy": traced,
            "onlyb": barren,
            "barren": barren,
            "ns1760995": probe_log(killed=True, primal=None),
            "bound": probe_log(primal=42.0),
        },
        "b": {
            "easy": traced,
            "onlyb": probe_log(primal=1.0),
            "barren": barren,
            "ns1760995": probe_log(killed=True, primal=None),
            "bound": probe_log(primal=42.0),
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

    assert load_instances(informative) == ["bound", "easy", "onlyb"]
    assert load_instances(hard) == ["barren", "ns1760995"]

    with open(hard) as f:
        hard_text = f.read()
    # The tier states the question it is scored on, and why it is separate.
    assert "did any configuration crack it" in hard_text
    assert "unreached" in hard_text and "no-acceptance" in hard_text

    # The excluded instances are listed in the report itself, not only in the
    # file: that list is a result.
    assert "ns1760995" in res.stdout and "barren" in res.stdout
    assert "Informative set: 3 of 5" in res.stdout
    # Trajectories: fpr saw 3 acceptances, local_mip none.
    assert "stall_p95" in res.stdout
    assert "mip_heuristic_fpr_stall" in res.stdout
    assert "mip_heuristic_local_mip_stall" in res.stdout


def test_outputs_are_byte_identical_across_runs(tmp_path):
    """Same tree plus same arguments regenerate both lists byte for byte."""
    tree, ref = _tree(tmp_path)
    # The same output paths on both runs: they appear in the regeneration
    # command the header records, so varying them would vary the bytes for a
    # reason that has nothing to do with determinism.
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
            second = f.read()
        outputs.append((first, second))
    assert outputs[0] == outputs[1]
    # No timestamp, so nothing in the header can drift between reruns.
    assert b"20" + b"26-" not in outputs[0][0]


def test_hard_tier_size_samples_deterministically(tmp_path):
    tree, ref = _tree(tmp_path)
    hard = os.path.join(str(tmp_path), "hard.txt")
    res = run(
        tree, "--instances", ref, "--hard-tier-output", hard, "--hard-tier-size", "1"
    )
    assert res.returncode == 0, res.stderr
    picked = load_instances(hard)
    assert len(picked) == 1 and picked[0] in ("barren", "ns1760995")

    bad = run(
        tree, "--instances", ref, "--hard-tier-output", hard, "--hard-tier-size", "9"
    )
    assert bad.returncode == 1 and "hard-tier-size" in bad.stderr


def test_narrowing_to_one_config_warns_about_the_union(tmp_path):
    tree, ref = _tree(tmp_path)
    informative = os.path.join(str(tmp_path), "informative.txt")
    res = run(
        tree,
        "--instances",
        ref,
        "--configs",
        "a",
        "--informative-output",
        informative,
    )
    assert res.returncode == 0, res.stderr
    # `onlyb` is informative in config b alone, so narrowing drops it.
    assert load_instances(informative) == ["bound", "easy"]
    assert "union over configurations" in res.stdout


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
    """A log with no report, no incumbent and no TIMEOUT marker never ran."""
    logs = {"a": {"easy": probe_log(primal=10.0), "void": "Running HiGHS 1.15.1\n"}}
    tree = write_tree(tmp_path, logs, name="broken")
    ref = write_reference(tmp_path, ["easy", "void"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert "void" in res.stderr and "TIMEOUT" in res.stderr


def test_an_err_file_is_named_as_a_failed_run(tmp_path):
    logs = {"a": {"easy": probe_log(primal=10.0)}}
    tree = write_tree(tmp_path, logs, name="errtree")
    with open(os.path.join(tree, "a", "seed0", "broken.log.err"), "w") as f:
        f.write("HiGHS exited 1\n")
    ref = write_reference(tmp_path, ["easy", "broken"])
    res = run(tree, "--instances", ref)
    assert res.returncode == 2
    assert ".log.err" in res.stderr


def test_a_tree_that_mixes_worker_counts_says_so(tmp_path):
    """A stall suggestion is only valid at the worker count it was measured at.

    Two machines in one tree therefore make the single summary number a
    fiction, and the report must not present it as one.
    """

    def traced(threads: int) -> str:
        return probe_log(
            threads=threads,
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
    logs = {"probeconf": {"easy": probe_log(primal=10.0)}}
    root = write_tree(tmp_path, logs, name="solo")
    ref = write_reference(tmp_path, ["easy"])
    res = run(os.path.join(root, "probeconf"), "--instances", ref)
    assert res.returncode == 0, res.stderr
    assert "Informative set: 1 of 1" in res.stdout
