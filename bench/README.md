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

## Where the MIPLIB collection lives

The collection is a 3.5 GB download (~7.3 GB extracted), so it is stored **once per machine, outside any checkout**, and located by a search path rather than a fixed path. `bench/download_miplib.sh` and `run_benchmark.py --data-dir` share it and probe in this order:

1. an explicit `--data-dir` / `DEST_DIR` argument — wins outright, even when the directory is empty, so a name is never silently resolved to some other directory. The two artifacts then differ on what that means: `run_benchmark.py` reads nothing and reports the instances as missing, whereas `download_miplib.sh` treats it as the destination and **downloads 3.5 GB into it** — so check a `DEST_DIR` before passing it
2. `$MIPLIB_DIR`
3. `~/data/miplib`
4. `/tmp/miplib`

The first directory holding more than 200 `.mps.gz` files wins. Only when none does is anything downloaded, and a fresh download lands in the *first* candidate — `~/data/miplib` normally, or `$MIPLIB_DIR` when that is set. `/tmp` is probed so an existing copy is reused instead of refetched, but it is never a download destination, because a collection there does not survive a reboot; the script says so and prints the `mv` that relocates it. The script prints the resolved directory on stdout and everything else on stderr, so `DATA_DIR=$(bash bench/download_miplib.sh)` works.

## Configs

`bench/run_benchmark.py` has one config per selectable subset of the five heuristics and no aliases: `off`, `all`, the five singletons, and every subset between them (`fj+fpr`, `fj+fpr+local_mip+fpr_lp`, …) — thirty-one non-empty subsets in all — plus `vanilla`, which is not a subset at all but the separately built unpatched binary that `--vanilla-binary` names. A config *zeroes* the effort option of every heuristic its name does not list and sets nothing for the ones it does, so a named heuristic runs at the binary's shipped default. Names join with `+`, because the name is a results-tree directory and a table label; they list heuristics in selection order — the presolve chain in dispatch order, then `fpr_lp` — so one subset has exactly one spelling.

```bash
bash bench/download_miplib.sh                       # once per machine; see above
python3 bench/run_benchmark.py \
    --instances bench/instances_tuning.txt \
    --output bench/results/sweep \
    --configs off fj fpr local_mip scylla all \
    --time-limit 600 --seeds 0 1 2 --skip-existing
python3 bench/analyze_results.py bench/results/sweep --ablation --time-limit 600 \
    --configs off fj fpr local_mip scylla all
```

`run_benchmark.py` prints the matching `analyze_results.py` command when it finishes, so the config list does not have to be retyped. A config name carries **no budget of its own**: to move a heuristic's effort or patience off its default, pass `--extra-options mip_heuristic_fpr_effort=12.0`. That is also how `scylla` and `fpr_lp` are enabled at all, since both ship at `0` — naming them in a config keeps them unzeroed but does not raise them.

`off` is the reference row for an ablation sweep, not `vanilla`: the sweep asks what each heuristic adds on this binary, and `off` is that binary with none of ours enabled. `vanilla` is a different question and a different binary — it requires `--vanilla-binary /path/to/unpatched/highs`, built from the same HiGHS tag, and the run is refused before its first solve if that binary carries the `mip-heuristics patch active` marker or reports another version.

Two things the harness deliberately does not do by default:

- **No `threads=`.** Forcing `threads=1` collapses each heuristic to a single worker. It is the right setting for reproducibility and the wrong one for a throughput benchmark, so `--threads` exists but has no default.
- **No `log_dev_level=3`.** Pass `--dev-log` to turn on the `[Heur]` / `[HeurSol]` instrumentation that `bench/parse_highs_log.py` reads for per-heuristic attribution. At level 3 a run pays FJ's periodic table plus our own two lines — a few per cent at campaign time limits, and proportionally more on very short solves where a fixed logging cost dominates. Attribution runs and headline-timing runs are therefore different runs.

## Tuning subsets

Tuning on the hard instances alone would over-allocate: presolve effort buys feasibility where feasibility is hard and is pure overhead where branch-and-bound has an incumbent in the first second, and that overhead delays the root LP. `bench/make_tuning_set.py` derives a subset that spans the spectrum instead, stratified on **vanilla time-to-first-feasible** — the axis the primal integral responds to — read out of an existing vanilla results tree:

```bash
python3 bench/make_tuning_set.py bench/results/mipfeas/vanilla --config vanilla \
    --instances bench/instances_mipfeas.txt \
    --informative-instances bench/ablation_effort/informative.txt \
    --size 90 --seed 0 --output bench/instances_tuning.txt
```

`--informative-instances` narrows the candidate pool to the instances a presolve-only probe produced an accepted solution on in any configuration; the rest carry no ranking signal for a presolve search. The list goes to `--output` (stdout by default) and the stratum table to stderr, so the distribution of the full set and of the sample are visible side by side. Strata are half-open intervals split at `--boundaries` (default `1,10,100,600`) plus a `never` bucket; a solution found at or past the time limit lands in `>=600s` rather than being filed as never-feasible. Draws are allocated proportionally by largest remainder in integer arithmetic, with one seat reserved per non-empty stratum (`--min-per-stratum`) so a small stratum is not rounded out of the set it was stratified for. The header records all of that — source tree, config, seeds, boundaries, allocation rule, and the per-stratum counts of both the full set and the sample — and carries no timestamp, because the same tree and `--seed` must regenerate the file byte for byte.

It **refuses** (exit 2) a tree that does not cover the reference list for every seed of the chosen config, naming what is absent and whether the run failed (`.log.err`), was truncated, or reported a primal bound with no incumbent line. The last two both parse into a result with no incumbents, which is indistinguishable from a genuine never-feasible run and would otherwise be binned as one — and `never` is normally the smallest stratum, so `--min-per-stratum` would then reserve the misfiled instance a seat. The instances a campaign failed to run are not a random subset of it, so sampling around them biases the subset in exactly the direction the stratification measures.

`bench/instances_small.txt` is a separate 25-instance list stratified on *optimality* solve time; it is the set a small-instance budget sweep runs on, and it is not the tuning set.

## Instance subsets and the config oracle

Any report restricts to an instance list, or excludes one, without re-running a solve:

```bash
# headline over the full mipfeas set
python3 bench/analyze_results.py bench/results/mipfeas --configs all vanilla \
    --time-limit 600 --summary

# the same comparison over the held-out complement of the tuning set
python3 bench/analyze_results.py bench/results/mipfeas --configs all vanilla \
    --time-limit 600 --summary \
    --instances bench/instances_mipfeas.txt --exclude-instances bench/instances_tuning.txt
```

`bench/instances_tuning.txt` is the 90-instance tuning list and is entirely inside the mipfeas 233, so that second command is the held-out complement: exactly 143 instances, the set the headline number is read on.

`--instances` applies first, then `--exclude-instances`, so the complement never has to exist as a third file that can drift out of sync with the tuning list it is defined against. Both are applied to the loaded tree before aggregation, so **every table reports the instance count it actually covers** and a restricted run cannot be misread as a full one.

`--oracle A B C` adds a best-of-those-configs row — the ceiling any per-instance selection mechanism could reach, which is what makes a negative result about a *selector* separable from a negative result about *selection*:

```bash
python3 bench/analyze_results.py bench/results/sweep --ablation --time-limit 600 \
    --configs fpr local_mip scylla all --oracle fpr local_mip scylla
```

Selection is per instance, on the headline metric (primal integral at `--time-limit`), among exactly the seed-collapsed rows the tables already show for each participant. That is what makes the row a genuine **ceiling** — its headline SGM is less than or equal to every participant's, instance by instance — and it is guaranteed by construction rather than hoped for. The oracle never sees an individual seed, so it can no more pick a lucky run than a real selector could. Per-seed winners are still reported, as a diagnostic of how stable the choice is, but they do not build the row.

The oracle is **additive**: it gets its own row and moves no existing one. It is held out of the head-to-head `#Win` / `#First` columns (it is a copy of the participant it selected and would otherwise tie with it, halving that config's credit). Instances absent from any participant at any shared seed, or outside the common set the tables cover, are dropped and counted. At least two participants are required — an oracle over one config is that config relabelled. Rename the row with `--oracle-name` if a real config is already called `oracle`.

This is unrelated to the *virtual best* inside the same script, which is reference-objective handling — when an observed primal beats the published `.solu` value, that observed value becomes the reference so a config is not punished for finding something better.

**Reference objectives** come from `bench/miplib2017-v36.solu` (upstream MIPLIB 2017, retrieved 2026-08-20). An instance the file marks `=inf=` or `=unbd=` has no finite objective to measure a gap against, so the script excludes it from every table and says so, rather than folding a self-referential gap into a 233-instance SGM.

## Reader-facing docs

`docs/REPRODUCIBILITY.md` (what is reproducible, and the exact protocols),
`docs/PARAMETERS.md` (every tunable, and where its default came from),
`docs/README.md` (the source papers), `README.md` (build, test, lint).
