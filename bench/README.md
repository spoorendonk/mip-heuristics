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
| `run_mipfeas.sh` | the chunked launcher every campaign stage uses. A stage is an *environment*, not a separate script — configs, seeds, instance list, output tree, extra options |
| `run_presolve_probe.sh` | `run_mipfeas.sh` with the calibration probe's environment (issue #113). Modes: `preprobe`, `budget`, `serial` |
| `run_finalists.sh` | the #107 finalists at the campaign limit — `heldout`, `confirm`. `FINALISTS_ONLY` names one arm for a stage that owns one rather than the set |
| `run_headline.sh` | the headline: the shipped configuration at **default options** over the full 233 at 600 s, against the vanilla arm in the same tree. One seed by default -- three is what the stage asks for and is the deliverable, one is the gate, and the script says why. `until 08:00` bounds a window by wall-clock time; `report` prints the three pre-registered tables |
| `run_ablation_c.sh` | `fpr_lp`'s two stages (#165): `capability` (does it fire, and yield, given every advantage) then `contribution` (the paired campaign metric) |
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
| `compare_finalists.sh` | assembles the per-arm trees into the view `analyze_results.py` wants, then scores them |
| `analyze_ablation_c.py` | `fpr_lp`'s two readings: `capability` (dispatches, setup bails, accepted yields, and which of three verdicts) and `contribution` (the paired log-ratio, partitioned by the capability run's labels, with the never-fired instances as a null control) |
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
machine that ran the probe.

## Where the results live

Four tracked directories, one per stage of the closeout campaign. Each carries
its own README with the findings, the caveats, and the commands that regenerate
every file in it from a results tree.

| directory | what it settles |
|---|---|
| `ablation_effort/` | per-heuristic effort and patience, measured with each heuristic **alone** |
| `ablation_search/` | the joint search over mix, effort and patience — and the reversal that corrected its first reading |
| `ablation_fprlp/` | whether `fpr_lp` earns a share of upstream's RENS/RINS envelope (it does not) |
| `headline/` | the shipped configuration against vanilla over the full 233, plus the generated provenance record |

**The results trees themselves are not published** — `bench/results*` is
gitignored and runs to several GB. Every number in those four READMEs is
readable without them, and regenerable from them. The runs are 16-worker and
non-deterministic by design, so a re-run reproduces the *result*, never the
logs; `docs/REPRODUCIBILITY.md` states that contract and the stage-by-stage
recipe.

`make_archive.py` packages a results tree with its provenance and a
`REGENERATE.sh` that re-derives every table from the archived logs and diffs
it. That is run as a release step — proving the tables come from the logs is a
check on our own arithmetic — but the archive stays in the gitignored `dist/`
and is not deposited. See `docs/RELEASE.md`.

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
