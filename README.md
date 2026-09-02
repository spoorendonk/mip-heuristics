# mip-heuristics

A unified open-source reference implementation and empirical evaluation of four modern primal heuristics — FeasibilityJump, FPR, LocalMIP and Scylla — inside one solver. All four are integrated into [HiGHS](https://github.com/ERGO-Code/HiGHS) v1.15.1 via a patched build, behind a common integration interface with shared budgeting and solution submission, so they can be measured against each other under identical conditions. See [Heuristics](#heuristics) for algorithmic details and paper references.

The contribution is the open implementations and the comparable measurements, not a solver configuration that beats HiGHS: the combined patched solver gives only a small aggregate improvement over vanilla on the PLATO `mipfeas` benchmark, and the honest end-to-end finding is that additional heuristics may not compensate for the solver progress they displace. See [Benchmarks](#benchmarks) for the numbers and their provenance.

**Documentation**: [`CONTRIBUTING.md`](CONTRIBUTING.md) (build, lint, review bar) · [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) (what is reproducible, and the PLATO protocol) · [`docs/RELEASE.md`](docs/RELEASE.md) (release process, artifact archive, DOI wiring) · [`docs/PARAMETERS.md`](docs/PARAMETERS.md) (every tunable constant) · [`docs/README.md`](docs/README.md) (source papers).

## Quick Start

**Prerequisites**: CMake 3.25+, GCC 13+ or Clang 17+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)          # first build fetches and compiles HiGHS
./build/bin/highs model.mps                        # mip_heuristic_suite defaults to `all`

# The custom options are not CLI flags — HiGHS's command line takes only its
# own fixed set, and rejects an unknown `--flag` without solving.  Pass them
# through an options file:
printf 'mip_heuristic_suite = fpr\n' > run.opts
./build/bin/highs --options_file run.opts model.mps
```

Full PLATO benchmark against vanilla HiGHS (requires MIPLIB instances and a separately built unpatched HiGHS of the same tag, ~77h total):

```bash
bash bench/download_miplib.sh
export PLATO_VANILLA_BINARY=/path/to/unpatched/highs   # not the patched build
bench/run_plato.sh next 24    # run in chunks; resumes safely
bench/run_plato.sh status     # check progress
python3 bench/analyze_results.py bench/results/plato --configs all vanilla --time-limit 600 --baseline
```

## Heuristics

**FPR (Fix, Propagate, and Repair)** — LP-free DFS tree search that fixes integer variables one at a time, propagates bounds at each node, repairs the partial assignment whenever propagation refutes a node — shifting whole variable domains rather than values, since most variables are not yet fixed — and backtracks on infeasibility. After the DFS, WalkSAT and RepairSearch repair any violations that remain in the completed solution. The presolve variant (Class 1) runs multiple strategy configurations in parallel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)). The full backtracking+WalkSAT+RepairSearch pipeline is not present in HiGHS, SCIP, or CBC.

**fpr_lp (LP-guided FPR, Classes 2–3)** — Uses the root LP solution to seed the DFS fixing order and initial values (paper Classes 2, 3a, 3b). Dispatched during the B&B dive (after RENS/RINS), not presolve. Workers are bound to distinct LP arm configurations; excess workers wrap with distinct seeds. Shares the FPR rounding kernel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)) (Classes 2–3).

**LocalMIP** — Weighted tabu local search with constraint-violation tracking, lifting moves, and multi-start backtracking. Finds improving moves by solving small MIP subproblems over the neighborhood. Based on Lin, Zou, Cai, "An Efficient Local Search Solver for Mixed Integer Programming," CP 2024, Article 19 ([doi:10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19)). Not in HiGHS or SCIP; cuOpt has a GPU variant citing the same paper. This is a CPU/HiGHS implementation with parallel multistart.

**Scylla** — PDLP-based feasibility pump: alternates approximate LP solves (PDLP) with FPR rounding, progressive objective blending, and cycling perturbation. N independent pump chains share one mutex-guarded PDLP instance; workers that lose the lock round against the most-recent stale snapshot to stay productive. Based on Mexi et al., *OR Proceedings 2023* ([doi:10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9)); same concept as cuOpt (arXiv:2510.20499). This is a CPU/HiGHS reference implementation — no novelty claim, but it is the only publicly available CPU implementation.

**FeasibilityJump** — LP-free Lagrangian heuristic. Thin wrapper around HiGHS's built-in FJ implementation, routed through our parallel infrastructure for effort budgeting and shared solution-pool integration. Based on Luteberget, Sartor, *Mathematical Programming Computation* 15, 365–388, 2023 ([doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8)). Note: at any `mip_heuristic_suite` value other than `off`, HiGHS's internal FJ dispatch is disabled and FJ runs through our infrastructure instead. Upstream's `mip_heuristic_run_feasibility_jump` (default true) still switches FJ off entirely.

Reference PDFs are in `docs/`.

## Execution Modes

The heuristics always run as the fixed chain FJ → FPR → LocalMIP → Scylla, each with its own effort budget (`mip_heuristic_fj_effort`, `mip_heuristic_fpr_effort`, `mip_heuristic_local_mip_effort`, `mip_heuristic_scylla_effort`) and its own patience (`mip_heuristic_<name>_patience`, the improvement-free effort it tolerates before giving up) so one can be tuned without moving the others. Both are multiples of `nnz << 10`, vanilla HiGHS's own single-thread FeasibilityJump limit, so `effort = 1.0` is one vanilla FJ budget and `patience < effort` reads on its face. Each heuristic parallelises the same way: continuous workers that self-terminate, with no epoch barrier and no bit-identical guarantee across runs.

For a reproducible run, set `threads=1` together with a fixed `random_seed`. That is the project's reproducibility contract — a single worker per heuristic, deterministic within one binary. It is not a separate mode and needs no extra option. It is also *not* the benchmark configuration: one worker per heuristic removes the contention Scylla is built around. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) has the full contract, including what is deliberately not reproducible and why.

One caveat for library embedders (not CLI users): HiGHS's task executor is a process-global singleton, initialised by the first `run()` in the process. A later solve that asks for a *different* thread count fails outright rather than silently using the old one, so pinning `threads=1` on a second `Highs` instance returns an error unless you first call `Highs::resetGlobalScheduler(true)`.

`mip_heuristic_suite` is the single option that selects which heuristics run (default `all`). Its value is either one of two aliases, or a comma-separated list of heuristic names:

| Value | Heuristics | Notes |
|--------|-----------|-------|
| `off` | none of ours | the ablation: HiGHS's own pipeline, native FeasibilityJump included |
| `fj` | FJ | isolate FeasibilityJump |
| `fpr` | FPR (+ `fpr_lp`) | isolate FPR for ablation |
| `local_mip` | LocalMIP | isolate LocalMIP |
| `scylla` | Scylla | PDLP pump only |
| `fj,fpr` | FJ+FPR (+ `fpr_lp`) | any comma-separated subset — all fifteen are expressible |
| `all` | FJ+FPR+LocalMIP+Scylla (+ `fpr_lp`) | **default** |

Order within a list is irrelevant (`fpr,fj` is `fj,fpr`), whitespace around a name is ignored, and repeating a name is harmless. `off` and `all` are aliases for the whole value, not names inside a list: `fj,off` is rejected, because `off` means "none of ours, HiGHS's own FeasibilityJump call site handed back" rather than merely "no heuristic".

An unrecognised value warns and falls back to `all` rather than silently disabling everything. The warning names the offending token, since one typo inside an otherwise valid list quietly promotes the run to all four heuristics; `bench/run_benchmark.py` greps for that warning and discards the run rather than filing it under the configuration it did not honour.

**`suite=off` is an ablation, not a vanilla baseline.** It disables our four presolve heuristics and `fpr_lp`, and hands HiGHS's standalone FeasibilityJump call site back — but the binary around it is still the patched one, including our copy of FeasibilityJump. Use it to measure what the chain contributes on this binary; a vanilla comparison needs a separately built unpatched HiGHS (`bench/run_benchmark.py --vanilla-binary`, which refuses a binary carrying the patch marker).

Put `mip_heuristic_run_feasibility_jump = false` in the options file alongside `mip_heuristic_suite = off` for the pure patch-overhead configuration, with no heuristic at all. That is the configuration `bench/check_vanilla_equivalence.py` compares against an unpatched binary with FeasibilityJump likewise disabled, and it *requires* the two to agree: same objective, same node count, same total and heuristic LP iterations, and an empty log diff once wall-clock content is normalized away (the timing block, the P-D integral, the profiling seconds, the git-hash width, the options-file echo and the `mip-heuristics patch active` marker). It is a gate, not a recorded result — it needs a second binary, so it cannot run in CI and every release re-runs it (`docs/RELEASE.md`). What it establishes when green is that injecting the heuristics does not perturb HiGHS's presolve, B&B or LP path — not that any setting of ours reproduces vanilla.

**`fpr_lp` follows the FPR bit.** It runs at B&B dive time on the same continuous workers, and is gated on the same flag as presolve FPR — so it runs at `suite=fpr` and `suite=all`, and is *disabled* at `suite=local_mip` and `suite=scylla` as well as at `off`. That is deliberate: a per-heuristic attribution run must not leave a second FPR variant running at dive time. It does mean a dive-time result under `suite=local_mip` cannot be attributed to `fpr_lp`.

`mip_heuristic_run_feasibility_jump` is upstream's own option and keeps its meaning: setting it false disables FeasibilityJump at every suite value, ours and HiGHS's alike.

## Benchmarks

### PLATO mipfeas — 233 instances, 600s time limit

Full PLATO mipfeas benchmark (233 MIPLIB 2017 instances, 600s per instance, system HiGHS as vanilla baseline). Configuration: the then-current `mip_heuristic_preset=all_opp` — FJ + FPR + LocalMIP, Scylla deliberately excluded.

| Metric | Patched (`all_opp`) | Vanilla HiGHS |
|---|---|---|
| #Feasible | **213** | 208 |
| #Win (strict, best primal obj at 600s) | **59** | 41 |
| SGM Time-to-first-feasible (s=1) | **3.6s** | 3.8s |
| SGM Gap@600s (s=0.001) | 0.00699 | **0.00638** |
| SGM Primal Integral (s=1) | **33.25** | 33.57 |
| SGM P-D Integral | 26.3 | **23.9** |
| PLATO headline SGM (s=0.001) | **26.0** | 26.8 |

> **Provenance.** This row cannot be reproduced on `HEAD`, by design. The configuration is expressible again — `all_opp` was FJ + FPR + LocalMIP without Scylla, which the single-valued `mip_heuristic_suite` (#93) could not name and `mip_heuristic_suite = fj,fpr,local_mip` (#112) now does — but the binary is gone: the numbers predate the #92 runner cleanup, which altered several things they depend on — workers no longer stop their peers on retiring, LocalMIP's cold start is primed once per dispatch rather than per worker, FJ's charge against the then-shared presolve envelope is floored, and two of the three budget weights were rescaled (that envelope and its weights have since been replaced by a per-heuristic effort option each, #110). The closeout benchmark campaign re-measures on the final tree; treat the row as the last full-campaign result, not as a claim about `HEAD`.

#### Findings

**PLATO headline (SGM primal integral, lower is better): 26.0 vs 26.8 — patched wins** (ratio 0.970). Patched also finds more feasible solutions (213 vs 208) and wins more head-to-head matchups by final objective (59 vs 41 strict wins).

**SGM T1st**: patched 3.6s vs vanilla 3.8s — patched finds its first feasible solution faster on average, despite heuristics running after presolve via our dispatch infrastructure. Vanilla finds a first solution sooner on more individual instances (#First 117.5 vs 97.5) because HiGHS's trivial heuristics fire before the LP; patched wins the SGM average because our heuristics find solutions on harder instances where vanilla fails.

**SGM Gap@600s** (0.00699 vs 0.00638) and **P-D Integral** (26.3 vs 23.9) favour vanilla — vanilla spends more time in B&B, tightening the dual bound, while our presolve heuristics consume budget before the root LP.

**SGM Primal Integral** (33.25 vs 33.57) favours patched narrowly. All SGM computations treat instances with no solution as gap=1.0 / PI=time-limit across the full 233-instance set.

**Summary**: patched wins the PLATO headline metric (−3%), finds more feasible solutions (+5), and wins more decisive head-to-head matchups. Vanilla is better on dual-bound-weighted metrics due to more B&B time.

**To reproduce:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
bash bench/download_miplib.sh
bench/run_plato.sh next 24   # run in chunks; resumes safely — repeat until 233/233
python3 bench/analyze_results.py bench/results/plato --configs all vanilla --time-limit 600 --baseline --summary
```

Results land in `bench/results/plato/`. The vanilla binary has no default and is never searched for on PATH: set `PLATO_VANILLA_BINARY=/path/to/unpatched/highs`, or drop `vanilla` from `PLATO_CONFIGS`. What a chunked run does is environment, not a second launcher — `PLATO_CONFIGS`, `PLATO_SEEDS`, `PLATO_INSTANCES` and `PLATO_OUTPUT` — see `docs/REPRODUCIBILITY.md`.

### Where the MIPLIB collection lives

The collection is a 3.5 GB download (~7.3 GB extracted), so it is stored **once per machine, outside any checkout**, and located by a search path rather than a fixed path. `bench/download_miplib.sh` and `run_benchmark.py --data-dir` share it and probe in this order:

1. an explicit `--data-dir` / `DEST_DIR` argument — wins outright, even when the directory is empty, so a name is never silently resolved to some other directory. The two artifacts then differ on what that means: `run_benchmark.py` reads nothing and reports the instances as missing, whereas `download_miplib.sh` treats it as the destination and **downloads 3.5 GB into it** — so check a `DEST_DIR` before passing it
2. `$MIPLIB_DIR`
3. `~/data/miplib`
4. `/tmp/miplib`

The first directory holding more than 200 `.mps.gz` files wins. Only when none does is anything downloaded, and a fresh download lands in the *first* candidate — `~/data/miplib` normally, or `$MIPLIB_DIR` when that is set. `/tmp` is probed so an existing copy is reused instead of refetched, but it is never a download destination, because a collection there does not survive a reboot; the script says so and prints the `mv` that relocates it. The script prints the resolved directory on stdout and everything else on stderr, so `DATA_DIR=$(bash bench/download_miplib.sh)` works.

### Tuning subset

Tuning on the hard instances alone would over-allocate: presolve effort buys feasibility where feasibility is hard and is pure overhead where branch-and-bound has an incumbent in the first second, and that overhead delays the root LP. `bench/make_tuning_set.py` derives a subset that spans the spectrum instead, stratified on **vanilla time-to-first-feasible** — the axis the primal integral responds to — read out of an existing vanilla results tree:

```bash
python3 bench/make_tuning_set.py bench/results/plato --config vanilla \
    --instances bench/instances_plato.txt --size 40 --seed 0 \
    --output bench/instances_tuning.txt
```

The list goes to `--output` (stdout by default) and the stratum table to stderr, so the distribution of the full set and of the sample are visible side by side. Strata are half-open intervals split at `--boundaries` (default `1,10,100,600`) plus a `never` bucket; a solution found at or past the time limit lands in `>=600s` rather than being filed as never-feasible. Draws are allocated proportionally by largest remainder, with one seat reserved per non-empty stratum (`--min-per-stratum`) so a small stratum is not rounded out of the set it was stratified for. The header records all of that — source tree, config, seeds, boundaries, allocation rule, and the per-stratum counts of both the full set and the sample — and carries no timestamp, because the same tree and `--seed` must regenerate the file byte for byte.

It **refuses** (exit 2) a tree that does not cover the reference list for every seed of the chosen config, naming what is absent and whether the run failed (`.log.err`), was truncated, or reported a primal bound with no incumbent line. The last two both parse into a result with no incumbents, which is indistinguishable from a genuine never-feasible run and would otherwise be binned as one — and `never` is normally the smallest stratum, so `--min-per-stratum` would then reserve the misfiled instance a seat. The instances a campaign failed to run are not a random subset of it, so sampling around them biases the subset in exactly the direction the stratification measures.

This does not replace `bench/instances_small.txt`, which is stratified on *optimality* solve time and is the recorded input of the retired budget-weight calibration; it stays the small-instance set a budget sweep runs on.

### Per-heuristic ablation and budget sweep

`bench/run_benchmark.py` has one config per `mip_heuristic_suite` value and no aliases: `off`, `all`, the four singletons, and the ten pairs and triples between them (`fj+fpr`, `fj+fpr+local_mip`, …), plus `vanilla`, which is not a suite value at all but the separately built unpatched binary that `--vanilla-binary` names. Config names join with `+` where the option value uses `,`, because the name is a results-tree directory and a table label; they list heuristics in chain order, so one subset has exactly one spelling.

```bash
bash bench/download_miplib.sh                       # once per machine; see above
python3 bench/run_benchmark.py \
    --instances bench/instances_small.txt \
    --output bench/results/sweep \
    --configs off fj fpr local_mip scylla all \
    --time-limit 600 --seeds 0 1 2 --skip-existing
python3 bench/analyze_results.py bench/results/sweep --ablation --time-limit 600 \
    --configs off fj fpr local_mip scylla all
```

`run_benchmark.py` prints the matching `analyze_results.py` command when it finishes, so the config list does not have to be retyped. To move a heuristic's effort or patience option off its default, pass `--extra-options mip_heuristic_fpr_effort=12.0`; a config name is exactly a `mip_heuristic_suite` value and carries no budget of its own.

`off` is the reference row here, not `vanilla`: an ablation sweep asks what each heuristic adds on this binary, and `off` is that binary with none of ours enabled. `vanilla` is a different question and a different binary — it requires `--vanilla-binary /path/to/unpatched/highs`, built from the same HiGHS tag, and the run is refused before its first solve if that binary carries the `mip-heuristics patch active` marker or reports another version. There is no fallback: the patched binary at `mip_heuristic_suite=off` is the ablation, and it is not a stand-in for an unpatched build.

Every config runs each of its heuristics at that heuristic's own shipped default budget. To move one, pass it through `--extra-options`; the four effort options are independent, so raising one does not lower the others.

Two things the harness deliberately does not do by default:

- **No `threads=`.** Forcing `threads=1` collapses each heuristic to a single worker. It is the right setting for reproducibility and the wrong one for a throughput benchmark, so `--threads` exists but has no default.
- **No `log_dev_level=3`.** Pass `--dev-log` to turn on the `[Heur]` / `[Sequential]` instrumentation that `bench/parse_highs_log.py` reads for the per-heuristic budget analysis. It is not free: HiGHS's own FeasibilityJump logs one line per weight bump at exactly that level, from every parallel FJ worker, with an `fflush` each. Measured on five bundled instances at a 10 s limit that is 97–750x the log volume (bell5: 16 KB → 3.5 MB) and 1.1–4.4x the total solve wall time (egout: 0.048 s → 0.212 s), concentrated in the FJ phase — i.e. in the number the analysis is reading. Use `--dev-log` for attribution runs and leave it off for headline timings.

### Instance subsets and the config oracle

Any report restricts to an instance list, or excludes one, without re-running a solve:

```bash
# headline over the full PLATO set
python3 bench/analyze_results.py bench/results/plato --configs all vanilla \
    --time-limit 600 --summary

# the same comparison over the held-out complement of the tuning set
python3 bench/analyze_results.py bench/results/plato --configs all vanilla \
    --time-limit 600 --summary \
    --instances bench/instances_plato.txt --exclude-instances bench/instances_small.txt
```

`bench/instances_small.txt` is the 25-instance tuning list and is entirely inside the PLATO 233, so that second command is the held-out complement: exactly 208 instances.

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

## Build Options

| Flag | Default | Description |
|------|---------|-------------|
| `-DCMAKE_BUILD_TYPE=Release` | — | Optimized build. The heuristics are unusable at `-O0`. |
| `-DMIP_HEURISTICS_REQUIRE_LINT=ON` | `OFF` | Turn a missing or wrong-major-version clang tool into a **configure failure** instead of a warning. CI sets it; so should you — the default failure mode is a gate that silently checks nothing. |
| `-DMIP_HEURISTICS_INSTRUMENT=OFF` | `ON` | Compile out the LocalMIP warm-start branch counters. They are consumed by two tests, so leave them on unless you are measuring their overhead in a production build. |
| `-DMIP_HEURISTICS_CUDA=ON` | `OFF` | Enable cuPDLP GPU backend for Scylla. Requires `CUDA_HOME` exported; **fails the configure** rather than falling back to CPU, because GPU vs CPU is a compile-time `#ifdef` in HiGHS and a silent fallback would be indistinguishable at the command line. Build into a separate tree and verify with `ldd build-gpu/bin/highs \| grep cudart`. |

## Testing

```bash
# Once per checkout — `.venv/bin` is the exact path the lint gates search.
python3 -m venv .venv
.venv/bin/pip install clang-format==22.1.8 clang-tidy==22.1.8 pytest

ctest --test-dir build --output-on-failure -j$(nproc)                     # everything
ctest --test-dir build -LE lint --output-on-failure                       # fast loop
ctest --test-dir build -R "execution-mode: flugpl objective" --output-on-failure
./build/mip_heuristics_tests "[mode-matrix]"                              # Catch2 tag
```

Catch2 v3. Characterization tests verify known-optimal objectives against MIPLIB instances bundled with HiGHS.

`clang-format` and `clang-tidy` run over `src/` and `tests/` as ctest tests labelled `lint`, adding roughly 30 s to a full run — hence `ctest -LE lint` while iterating. Without the venv above they are **not registered at all** and `ctest` reports green having linted nothing, which is what `-DMIP_HEURISTICS_REQUIRE_LINT=ON` exists to prevent. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the tool-version contract.

## License

[MIT](LICENSE)
