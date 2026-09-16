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
| `analyze_presolve_probe.py` | the calibration probe: informative set and its excluded complement, effort trajectories, gap to best known, and the derived parameter vector |
| `make_tuning_set.py` | a stratified tuning subset, sampled from a results tree on time-to-first-feasible |
| `derive_from_probe.sh` | **probe tree → every artifact, one command** (see below), written into `ablation_effort/` |
| `compare_finalists.sh` | assembles the per-arm trees into the view `analyze_results.py` wants, then scores them |
| `analyze_ablation_c.py` | `fpr_lp`'s two readings: `capability` (dispatches, setup bails, accepted yields, and which of three verdicts) and `contribution` (the paired log-ratio, partitioned by the capability run's labels, with the never-fired instances as a null control) |
| `check_vanilla_equivalence.py` | proves the patch does not perturb HiGHS: every heuristic's effort zeroed plus FeasibilityJump disabled, against a separately built unpatched binary with FeasibilityJump disabled |
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

Four tracked directories, one per stage of the campaign. Each carries its own
README with the findings, the caveats, and the commands that regenerate every
file in it from a results tree.

| directory | what it settles |
|---|---|
| `ablation_effort/` | per-heuristic effort and patience, measured with each heuristic **alone** |
| `ablation_search/` | the joint search over mix, effort and patience, and the finalist confirmation that selected the shipped vector |
| `ablation_fprlp/` | whether `fpr_lp` earns a share of upstream's RENS/RINS envelope (it does not) |
| `headline/` | the shipped configuration against vanilla over the full 233 |

**The results trees themselves are not published** — `bench/results*` is
gitignored and runs to several GB. Every number in those four READMEs is
readable without them, and regenerable from them. The runs are 16-worker and
non-deterministic by design, so a re-run reproduces the *result*, never the
logs; `docs/REPRODUCIBILITY.md` states that contract and the stage-by-stage
recipe.

`make_archive.py` packages a results tree with its provenance and a
`REGENERATE.sh` that re-derives every table from the archived logs and diffs
it. Proving the tables come from the logs is a check on our own arithmetic, so
it is worth running whether or not anyone else sees the result — the archive
stays in the gitignored `dist/`.

## Reader-facing docs

`docs/REPRODUCIBILITY.md` (what is reproducible, and the exact protocols),
`docs/PARAMETERS.md` (every tunable, and where its default came from),
`docs/README.md` (the source papers), `CONTRIBUTING.md` (build, test, lint).
