# bench/ — the benchmark harness and the calibration campaign

Everything here reads or writes a **results tree**: `<config>/seed<N>/<instance>.log`
plus the `.opts` each run was given. That layout is the only contract between
the scripts; there are no aliases and no second spelling of a config name.

Results trees are **gitignored** (`bench/results*`). They are large, they are
per-machine, and the way to move one is `make_archive.py`, which packages the
logs with a `REGENERATE.sh` that re-derives every table and diffs it — so
"regenerable from the archive alone" is checked rather than claimed.

## Running things

| script | what it does |
|---|---|
| `run_benchmark.py` | runs instances × configs × seeds into a results tree. Resumes with `--skip-existing`; bounds a chunk by hours (`--wall-time-budget`) or by pending work (`--count`). The `vanilla` config is a second, unpatched binary (`--vanilla-binary`, required and probed) — not a setting on the patched one |
| `run_plato.sh` | the chunked launcher every campaign stage uses. A stage is an *environment*, not a separate script — configs, seeds, instance list, output tree, extra options |
| `run_presolve_probe.sh` | `run_plato.sh` with the calibration probe's environment (issue #113). Modes: `preprobe`, `budget`, `serial` |
| `run_target.py` | scores **one parameter vector** on one instance set — the inner loop of the #107 tuning search |
| `download_miplib.sh` | fetches MIPLIB2017 once per machine (3.5 GB, outside every checkout) |

## Reading things

| script | what it produces |
|---|---|
| `parse_highs_log.py` | one `SolveResult` per log: incumbents, bounds, timings, `[Heur]`/`[HeurSol]` traces. Every other reader goes through it |
| `analyze_results.py` | the headline tables — SGM, primal integral, wins, oracle rows, instance filters |
| `analyze_presolve_probe.py` | the calibration probe: informative set, hard tier, effort trajectories, gap to best known, and the derived parameter vector |
| `make_tuning_set.py` | a stratified tuning subset, sampled from a results tree on time-to-first-feasible |
| `derive_from_probe.sh` | **probe tree → every artifact, one command** (see below), written into `ablation_effort/` |
| `check_vanilla_equivalence.py` | proves the patch does not perturb HiGHS: `suite=off` plus FeasibilityJump disabled, against a separately built unpatched binary with FeasibilityJump disabled |
| `make_archive.py` | the release archive, with derived provenance |
| `check_docs_refs.py` | fails the suite if `docs/PARAMETERS.md` names a constant that no longer exists |

## The calibration chain

```
run_presolve_probe.sh preprobe next <hours>     # the measurement
derive_from_probe.sh                            # everything downstream
```

The probe runs each heuristic **alone**, presolve-only, at a budget that cannot
bind with the staleness gate off — so the wall clock is the single stopping
rule, the same one for every heuristic on every instance. `derive_from_probe.sh`
then produces, all from those logs:

* `informative.txt` — instances where the chain produced the reported incumbent
* `hard_tier.txt` — the complement, each with its reason
* `report.txt` — counts, the budget-headroom check, trajectories, quality
* `defaults.json` — the derived per-heuristic effort and patience
* `instances_tuning.txt` — the stratified tuning subset

Every artifact carries the command that regenerates it and a digest of its
inputs, and none carries a timestamp: same trees plus same seed reproduce them
byte for byte.

They land in **`bench/ablation_effort/`**, which is tracked — `bench/results*`
is not, and the numbers behind a shipped default should not live only on the
machine that ran the probe. That directory's README carries the findings: what
the shipped effort and patience defaults are, how they were derived, and the
caveats that travel with them.

## Things that will bite you

* **Never pass `--threads`** (or set `threads=` in an `.opts`) unless
  reproducibility is the point. It collapses each heuristic to one worker, and
  for a tuning run it *moves* the objective's distribution rather than
  narrowing it — FJ's budget is per worker while the other three are per
  dispatch, so changing the count reallocates budget between heuristics.
* **`--dev-log` is a different run.** It sets `log_dev_level=3`, which is what
  makes `[Heur]`/`[HeurSol]` visible — and what makes the log big. Attribution
  runs and headline-timing runs are not the same runs.
* **A killed run is evidence, not a lost run.** The harness SIGKILLs a solve
  that outruns its limit (HiGHS checks its clock between work units), keeps the
  partial log with a `TIMEOUT:` marker, and the parsers report it as `killed`.
  Such logs still score: T1st and the primal integral read incumbent lines, not
  the report the run never reached.
* **`[Heur]` is written when a dispatch *ends*.** A killed run therefore has
  incumbent rows and no ledger, which is why probe membership follows the
  incumbent and never the trace.
* **Zero effort has two causes, and the line says which.** A dispatch whose
  sequential setup found the deadline already passed never searched (#117),
  and used to book an `effort=0 found=0` line indistinguishable from one that
  searched and produced nothing. `abandoned_setup=<0|1>` (#119) separates
  them, so the three shapes a consumer must tell apart stay apart: no
  `[Heur]` line is a killed run, `abandoned_setup=1` is a bail, and the field
  absent or `0` is a dispatch that ran. Absent means a log written before
  #119 — `HeuristicSample.abandoned_setup` is `None` there, and
  `analyze_presolve_probe.py` treats that as "ran", which is what makes an
  archived tree classify exactly as it did before.
* **A config name carries no budget.** Every heuristic runs at its shipped
  default; moving one for a run goes through `--extra-options`.

## Reader-facing docs

`docs/REPRODUCIBILITY.md` (what is reproducible, and the exact protocols),
`docs/PARAMETERS.md` (every tunable, and where its default came from),
`docs/RELEASE.md` (cutting a release), `CONTRIBUTING.md` (build, test, lint).
