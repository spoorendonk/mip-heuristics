# mip-heuristics

[![CI](https://github.com/spoorendonk/mip-heuristics/actions/workflows/ci.yml/badge.svg)](https://github.com/spoorendonk/mip-heuristics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A unified open-source reference implementation and empirical evaluation of four modern primal heuristics — FeasibilityJump, FPR, LocalMIP and Scylla — inside one solver, plus `fpr_lp`, FPR's LP-guided variant. All of them are integrated into [HiGHS](https://github.com/ERGO-Code/HiGHS) v1.15.1 via a patched build, behind a common integration interface with shared budgeting and solution submission, so they can be measured against each other under identical conditions.

The contribution is the open implementations and the comparable measurements, not a solver configuration that beats HiGHS: on instances never used for tuning the patched solver improves the `mipfeas` primal-integral SGM by 16.4% — it reaches a good solution *sooner*, while final solution quality is level.

This repository has no paper of its own; it implements four published ones, listed under [References](#references).

**Documentation**: [`CONTRIBUTING.md`](CONTRIBUTING.md) (build, lint, review bar) · [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) (what is reproducible, and the `mipfeas` protocol) · [`docs/PARAMETERS.md`](docs/PARAMETERS.md) (every tunable constant) · [`bench/README.md`](bench/README.md) (the harness and the campaign) · [`docs/README.md`](docs/README.md) (source papers).

## Key Features

- **Four presolve heuristics behind one runner contract** — FeasibilityJump, FPR, LocalMIP and Scylla share effort budgeting, a solution pool, a wall-clock deadline and continuous parallel workers, so a comparison between them is a comparison of the algorithms.
- **A fifth at branch-and-bound dive time** — `fpr_lp`, FPR's LP-guided variant (paper Classes 2–3), drawing from upstream's own RENS/RINS envelope and charging back what it spends.
- **Compiled into HiGHS, not bolted on** — a patched build of v1.15.1. At default options a patched binary matches vanilla's B&B heuristic budget exactly, and `bench/check_vanilla_equivalence.py` proves the injection perturbs neither presolve, B&B nor the LP path.
- **Paper-faithful, with the deviations written down** — every departure is recorded beside the code and in [`docs/PARAMETERS.md`](docs/PARAMETERS.md). Two upstream FeasibilityJump defects are corrected so the search matches its paper.
- **One budget unit** — effort and patience are both multiples of `nnz << 10`, vanilla HiGHS's own single-thread FJ limit, so `effort = 1.0` means one vanilla FJ budget and `patience < effort` reads on its face.
- **Selection *is* the budget** — a heuristic runs iff its effort option is above zero, so all thirty-two subsets of the five are a zero-pattern of five doubles. No selector option, no rebuild.
- **Defaults that were measured** — searched over eight parameters and confirmed on a held-out instance set, not chosen by hand. Scylla and `fpr_lp` ship disabled because the measurement said so.
- **A presolve-only mode** — exits before the root LP, ~25x cheaper than a full solve, and the only way to measure the chain without branch-and-bound diluting it.

## Quick Start

**Prerequisites**: CMake 3.25+, GCC 13+ or Clang 17+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)          # first build fetches and compiles HiGHS
./build/bin/highs model.mps             # the shipped heuristic defaults
```

The custom options are **not CLI flags** — HiGHS's command line takes only its own fixed set and rejects an unknown `--flag` without solving. Pass them through an options file:

```bash
printf 'mip_heuristic_fj_effort = 0\nmip_heuristic_local_mip_effort = 0\n' > run.opts
./build/bin/highs --options_file run.opts model.mps
```

A patched binary says so on the third line of its log header, which is the only thing that distinguishes it from an unpatched build of the same tag:

```
Running HiGHS 1.15.1 (git hash: 04024d701f): Copyright (c) 2026 under MIT licence terms
Includes third-party software components, see THIRD_PARTY_NOTICES.md for full details
mip-heuristics patch active (custom MIP presolve heuristics; spoorendonk/mip-heuristics)
```

## Heuristics

**FeasibilityJump** — LP-free Lagrangian heuristic. HiGHS's own FJ implementation, routed through our parallel infrastructure for effort budgeting and shared solution-pool integration, with two defects in `feasibilityjump.hh` corrected by `third_party/highs_patch/apply_patch.cmake` so that the search matches the paper:

  * the jump value for a row with a **negative coefficient**. Forming the row's valid range divides both endpoints of the row's bound interval by the variable's coefficient, which reverses the interval when that coefficient is negative; without a swap the range comes out empty and the row contributes neither a critical value nor a slope, where eq. (5)/(6) give the critical value an explicit negative-coefficient case and Algorithm 1 accumulates the pre-bound slope by that sign. The correction is one conditional swap, placed before the integer ceil/floor. Damage concentrates on general integers and continuous columns; binaries are barely affected, since the jump is the opposite bound either way.
  * the **sign of the objective term** in the move score. Sect. 2.6 makes the score a minimised sum and every other term is improvement-positive, so the term is negated at both sites that spell it.

  Pinned by `tests/test_fj_jump_value.cpp` and `tests/test_fj_objective_sign.cpp`, and by a post-check inside the patch script that refuses a half-applied patch. Based on Luteberget, Sartor, *Mathematical Programming Computation* 15, 365–388, 2023 ([doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8)).

**FPR (Fix, Propagate, and Repair)** — LP-free DFS tree search that fixes integer variables one at a time, propagates bounds at each node, repairs the partial assignment whenever a fixing or its propagation leaves a constraint unsatisfiable — shifting whole variable domains rather than values, since most variables are not yet fixed — and backtracks on infeasibility. After the DFS, WalkSAT and RepairSearch repair any violations that remain in the completed solution. The presolve variant (paper Class 1) runs multiple strategy configurations in parallel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)). The full backtracking + WalkSAT + RepairSearch pipeline is not present in HiGHS, SCIP, or CBC.

**fpr_lp (LP-guided FPR, paper Classes 2–3)** — uses the root LP solution to seed the DFS fixing order and initial values (Classes 2, 3a, 3b). Dispatched during the B&B dive, after RENS/RINS, rather than in presolve. Workers are bound to distinct LP arm configurations; excess workers wrap the arm list with distinct seeds. Shares the FPR rounding kernel. Same citation as FPR.

**LocalMIP** — weighted tabu local search with constraint-violation tracking, lifting moves, and multi-start backtracking. Based on Lin, Zou, Cai, "An Efficient Local Search Solver for Mixed Integer Programming," CP 2024, Article 19 ([doi:10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19)). Not in HiGHS or SCIP; cuOpt has a GPU variant citing the same paper. This is a CPU/HiGHS implementation with parallel multistart.

**Scylla** — PDLP-based feasibility pump: alternates approximate LP solves (PDLP) with FPR rounding, progressive objective blending, and cycling perturbation. N independent pump chains share one mutex-guarded PDLP instance; workers that lose the lock round against the most-recent stale snapshot to stay productive. Based on Mexi et al., *OR Proceedings 2023* ([doi:10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9)); same concept as cuOpt (arXiv:2510.20499). This is a CPU/HiGHS reference implementation — no novelty claim, but it is the only publicly available CPU implementation.

Reference PDFs are in `docs/`.

## Configuration

The four presolve heuristics always run as the fixed chain FJ → FPR → LocalMIP → Scylla, each with its own **effort** budget and its own **patience** — the improvement-free effort it tolerates before giving up — so one can be tuned without moving the others. Both are multiples of `nnz << 10`, vanilla HiGHS's own single-thread FeasibilityJump limit, so `effort = 1.0` is exactly one vanilla FJ budget and `patience < effort` reads on its face. Each heuristic parallelises the same way: continuous workers that self-terminate, with no barrier between them and no bit-identical guarantee across runs.

| heuristic | effort option | default | patience option | default | |
|---|---|---|---|---|---|
| `fj` | `mip_heuristic_fj_effort` | 0.3317 | `mip_heuristic_fj_patience` | 0.0 | no staleness gate — `0` means no gate at all, not "give up at once" |
| `fpr` | `mip_heuristic_fpr_effort` | 3.1610 | `mip_heuristic_fpr_patience` | 0.3372 | |
| `local_mip` | `mip_heuristic_local_mip_effort` | 3.2865 | `mip_heuristic_local_mip_patience` | 3.1943 | above its own ceiling, so the clamp `effort/4` = 0.8216 applies |
| `scylla` | `mip_heuristic_scylla_effort` | **0.0** | `mip_heuristic_scylla_patience` | 0.0 | **disabled** |
| `fpr_lp` | `mip_heuristic_fpr_lp_effort` | **0.0** | — | — | **disabled**; a share of a different envelope, see below |

**What ships is three of the five.** The defaults are `B'-mix-cheapest`, the configuration the joint search selected and held-out validation confirmed ([stage 3](#reproducing)).

**Scylla ships disabled, and that is a measurement.** It produced zero accepted incumbents across ~380 runs on 233 instances while dispatching on every solve, and it is absent from all 14 survivors of the configuration search. The implementation is maintained and tested; it is the *budget* that is zero, so raising the option re-enables the heuristic with no rebuild.

**`fpr_lp` ships disabled for a different reason**, and its option is in a different unit. It runs at B&B dive time and draws from upstream's RENS/RINS LP-iteration envelope (`mip_heuristic_effort`), charging back what it spends — so its budget is zero-sum against those two rather than an independent allowance, and `mip_heuristic_fpr_lp_effort` is its **share of that envelope**. A call is sized at `share x min(remaining headroom, per-call cap)`, so `1.0` takes the whole slice and a share above `1.0` grows the budget without bound. Overdrawing is self-correcting: the charge-back depletes the counters the headroom is read from, so the next call finds none and skips. `0` returns above every read and write of those counters, which is what makes an `fpr_lp` ablation leave RENS and RINS exactly as they were. It ships at `0` because given the whole envelope it produces accepted incumbents, and inside the shipped chain it produces none ([stage 4](#reproducing)).

### Selecting heuristics

**A heuristic runs iff its effort option is above zero.** There is no separate selector: the effort option is both the budget and the switch, and a value at or below zero skips the heuristic entirely — no dispatch, no setup, no `[Heur]` trace line. Every subset of the five is therefore expressible as a zero-pattern:

| To run | Set |
|--------|-----|
| the shipped default | nothing — `fj`, `fpr` and `local_mip` are on, `scylla` and `fpr_lp` ship at `0` |
| FJ alone | `fpr`, `local_mip` efforts to `0` |
| presolve FPR alone | `fj`, `local_mip` efforts to `0` |
| Scylla alone | `fj`, `fpr`, `local_mip` to `0`, and `mip_heuristic_scylla_effort` above `0` |
| `fpr_lp` alone | the four presolve efforts to `0`, and `mip_heuristic_fpr_lp_effort` above `0` |
| nothing of ours | all five to `0` |

Because `scylla` and `fpr_lp` ship at `0`, enabling either takes a positive value rather than merely leaving it alone. `bench/run_benchmark.py` wraps this as config *names* — `fj+fpr`, `all`, `off` — which zero the efforts a name does not list; see [`bench/README.md`](bench/README.md).

`mip_heuristic_run_feasibility_jump` is upstream's own switch and keeps its meaning: `false` disables FeasibilityJump entirely, the same thing `mip_heuristic_fj_effort = 0` does. The one configuration that contradicts itself — a positive `mip_heuristic_fj_effort` with `mip_heuristic_run_feasibility_jump = false` and nothing else enabled — warns, because it asks for FJ and then takes it away, leaving a run with no heuristic at all while being spelled like an "FJ isolated" row. `bench/run_benchmark.py` greps for that warning and discards the run rather than filing it under a configuration it did not honour.

**Zeroing all five is an ablation, not a vanilla baseline.** It disables our four presolve heuristics and `fpr_lp`, but the binary around it is still the patched one. It also runs **no FeasibilityJump at all**: upstream's standalone FJ call site never fires on a patched build, so FJ is ours or it is absent. Use it to measure what the chain contributes on this binary; a vanilla comparison needs a separately built unpatched HiGHS (`bench/run_benchmark.py --vanilla-binary`, which refuses a binary carrying the patch marker).

Put `mip_heuristic_run_feasibility_jump = false` in the options file alongside the five zeros for the **pure patch-overhead** configuration. That is what `bench/check_vanilla_equivalence.py` compares against an unpatched binary with FeasibilityJump likewise disabled, and it *requires* the two to agree: same objective, same node count, same total and heuristic LP iterations, and an empty log diff once wall-clock content is normalised away. It is a gate, not a recorded result — it needs a second binary, so it cannot run in CI and every release re-runs it. What it establishes when green is that injecting the heuristics does not perturb HiGHS's presolve, B&B or LP path.

### Other options

`mip_heuristic_presolve_only` (bool, default `false`) exits the solve after the presolve chain and before the root LP, keeping the incumbent. It is roughly 25x cheaper than a full solve and is the only way to measure the chain without B&B diluting it, which is what makes the tuning search affordable. Such a run reports `Solution limit reached`, `Nodes 0`, `LP iterations 0`, `Dual bound -inf` and CLI exit 1, so a presolve-only tree is not comparable to a full-solve tree on any dual-side metric.

`mip_heuristic_effort` keeps **vanilla semantics and its vanilla default of 0.05**: it is upstream's B&B dive-heuristic envelope, which gates RENS/RINS and `fpr_lp` alike. A patched binary at default options matches vanilla's B&B heuristic budget exactly.

### Reproducible runs

Set `threads = 1` together with a fixed `random_seed`. That is the project's reproducibility contract — a single worker per heuristic, deterministic within one binary. It is not a separate mode and needs no extra option. It is also *not* the benchmark configuration: one worker per heuristic removes the contention Scylla is built around, and it reallocates budget between heuristics, since FJ's option sizes one worker's allowance where the other three size a whole dispatch. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) has the full contract, including what is deliberately not reproducible and why.

One caveat for library embedders (not CLI users): HiGHS's task executor is a process-global singleton, initialised by the first `run()` in the process. A later solve that asks for a *different* thread count fails outright rather than silently using the old one, so pinning `threads=1` on a second `Highs` instance returns an error unless you first call `Highs::resetGlobalScheduler(true)`.

## Results

233 instances, 600 s, one seed, 16 workers, CPU build. `mipfeas` is a MIP
feasibility benchmark — 233 instances from the MIPLIB 2017 benchmark collection
with the known-infeasible ones excluded, scored by the primal integral —
introduced by Bussieck and Dirkse in
[*Expanding the Focus*](https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/)
(GAMS, 17 March 2026). Full write-up and the paired statistics:
[`bench/headline/`](bench/headline/).

| Metric | Patched | Vanilla |
|---|---|---|
| **SGM primal integral, all 233** | **19.18** | 26.57 |
| **SGM primal integral, held-out 143** | **22.89** | 27.36 |
| #Feasible | **214** | 211 |

Paired per instance, on the log-ratio of the primal integral:

| set | n | ratio | 95% CI | p | better / tied / worse |
|---|---|---|---|---|---|
| all 233 | 233 | 0.722 | [0.633, 0.823] | <0.001 | 107 / 32 / 94 |
| **held-out 143** | **143** | **0.836** | **[0.732, 0.956]** | **0.009** | 57 / 25 / 61 |
| tuning 90 | 90 | 0.572 | [0.441, 0.741] | <0.001 | 50 / 7 / 33 |

**The held-out number is the result: 16.4% better on instances never used for
tuning.** The tuning set shows 43% — the selection bias the split exists to
quantify, and a factor of nearly three. Reporting the tuning figure as the
headline would overstate the effect by that much.

### Three qualifications that belong with it

**The win is magnitude, not frequency.** 57 better against 61 worse on
held-out; a sign test finds nothing (p = 0.71). The heuristics do not win more
often — they win *bigger*: `comp07-2idx` 600 → 7.8 and `sorrell3` 162 → 2.3,
against `fast0507` 14.4 → 248.

**Final solution quality is level.** Paired final primal gap at 600 s: better on
45, tied on 130, worse on 34. HiGHS's own machinery still holds **188 of 214**
final answers. The claim is that HiGHS reaches a good solution *sooner*, not
that it reaches a better one — which is what the primal integral measures and
what a feasibility heuristic should claim.

**Low power with significance implies overstatement.** At the measured paired sd
the held-out n=143 resolves about 17.5%, and the observed effect is 16.4% — both
read on the same side of the ratio. Quote the interval, not the point.

### Where it comes from

Partitioning the 233 instances by which heuristic produced the patched arm's
*first* incumbent:

| first incumbent from | n | ratio vs vanilla | 95% CI |
|---|---|---|---|
| **FeasibilityJump** | **112** | **0.563** | **[0.451, 0.702]** |
| FPR | 25 | 0.732 | [0.409, 1.309] |
| HiGHS/other | 68 | 0.956 | [0.722, 1.267] |
| **no incumbent found** | **20** | **1.005** | **[0.995, 1.014]** |

The effect is concentrated where FeasibilityJump gets there first and is absent
everywhere else, with the 20 instances where nothing is found acting as an
internal control: the two arms are identical to within 1%, so the pairing is
tight and 0.563 is an effect rather than noise.

That does **not** make this a result about parallelism. Our FJ differs from
vanilla's in three confounded ways — 16 opportunistic workers against one call,
a per-worker budget totalling ~5x vanilla's single allowance, and the two
corrected upstream defects ([#159](https://github.com/spoorendonk/mip-heuristics/issues/159),
[#160](https://github.com/spoorendonk/mip-heuristics/issues/160)) — and vanilla
runs its own FJ, so the comparison already includes FJ-vs-FJ.

Which heuristic produced which solution, across the same 233:

| | patched #First | #Best | vanilla #First | #Best |
|---|---|---|---|---|
| FJ | 114 | 20 | 97 | 10 |
| FPR | 16 | 2 | — | — |
| LocalMIP | 5 | 4 | — | — |
| HiGHS/other | 79 | **188** | 114 | **201** |

Ours find the first feasible solution on 135 of 214 instances against vanilla's
97 of 211, and hold the final best on 26 against 10.

## Build Options

| Flag | Default | Description |
|------|---------|-------------|
| `-DCMAKE_BUILD_TYPE=Release` | — | Optimized build. The heuristics are unusable at `-O0`. |
| `-DMIP_HEURISTICS_REQUIRE_LINT=ON` | `OFF` | Turn a missing or wrong-major-version clang tool into a **configure failure** instead of a warning. CI sets it; so should you — the default failure mode is a gate that silently checks nothing. |
| `-DMIP_HEURISTICS_INSTRUMENT=OFF` | `ON` | Compile out the LocalMIP warm-start branch counters. They are consumed by two tests, so leave them on unless you are measuring their overhead in a production build. |
| `-DMIP_HEURISTICS_CUDA=ON` | `OFF` | Enable cuPDLP GPU backend for Scylla. Requires `CUDA_HOME` exported; **fails the configure** rather than falling back to CPU, because GPU vs CPU is a compile-time `#ifdef` in HiGHS and a silent fallback would be indistinguishable at the command line. Build into a separate tree and verify with `ldd build-gpu/bin/highs \| grep cudart`. |

## Tests

```bash
# Once per checkout — `.venv/bin` is the exact path the lint gates search.
python3 -m venv .venv
.venv/bin/pip install clang-format==22.1.8 clang-tidy==22.1.8 ruff==0.16.3 pytest

ctest --test-dir build --output-on-failure -j$(nproc)                     # everything
ctest --test-dir build -LE lint --output-on-failure                       # fast loop
ctest --test-dir build -R "execution-mode: flugpl objective" --output-on-failure
./build/mip_heuristics_tests "[mode-matrix]"                              # Catch2 tag
```

Catch2 v3. Characterization tests verify known-optimal objectives against MIPLIB instances bundled with HiGHS.

`clang-format` and `clang-tidy` run over `src/` and `tests/` as ctest tests labelled `lint`, adding roughly 30 s to a full run — hence `ctest -LE lint` while iterating. Without the venv above they are **not registered at all** and `ctest` reports green having linted nothing, which is what `-DMIP_HEURISTICS_REQUIRE_LINT=ON` exists to prevent. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the tool-version contract.

## Reproducing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
bash bench/download_miplib.sh                  # once per machine, 3.5 GB, outside every checkout
export MIPFEAS_VANILLA_BINARY=/path/to/unpatched/highs   # a second binary; never the patched one
bench/run_mipfeas.sh next 24                   # run in chunks; resumes safely — repeat until 233/233
bench/run_mipfeas.sh status                    # progress and estimated time remaining
python3 bench/analyze_results.py bench/results/mipfeas --configs all vanilla \
    --time-limit 600 --baseline --summary
```

The full campaign is roughly 77 hours (233 instances × 600 s × 2 configs, run
interleaved so partial results are always paired). A stage is an *environment*
rather than a separate script — `MIPFEAS_CONFIGS`, `MIPFEAS_SEEDS`,
`MIPFEAS_INSTANCES`, `MIPFEAS_OUTPUT`.

**The vanilla baseline is always a separately built unpatched binary** of the tag
in `cmake/FetchHiGHS.cmake`. It has no default and is never searched for on
`PATH`, and the harness refuses one that carries the patch marker or reports
another version. The patched binary with every heuristic zeroed is an *ablation*
and never a stand-in.

Five stages produced the numbers above. Each is a tracked launcher plus a tracked
reader; each results tree is gitignored (~5 GB in total) and each stage's derived
artifacts are committed, so every number is readable without the logs and
regenerable from them. All stages ran on one 16-core / 32-thread machine at
HiGHS's default worker count (**16 workers**), CPU build, one seed.

| # | stage | what it runs | what it decides | write-up |
|---|---|---|---|---|
| 1 | vanilla baseline | unpatched HiGHS v1.15.1, 233 instances, 600 s | the control for stage 5, and the time-to-first-feasible axis stage 2 stratifies on | — |
| 2 | presolve probe | each presolve heuristic **alone**, presolve-only, 30 s, 233 instances | a starting effort and patience per heuristic, and the 90-instance tuning set | [`bench/ablation_effort/`](bench/ablation_effort/) |
| 3 | joint search | irace over 8 parameters on the 90 tuning instances, presolve-only, 60 s; then finalists at 600 s | the shipped mix and budgets — `B'-mix-cheapest` | [`bench/ablation_search/`](bench/ablation_search/) |
| 4 | `fpr_lp` ablation | capability at 120 s, then contribution at 600 s, 49 instances | `mip_heuristic_fpr_lp_effort = 0` | [`bench/ablation_fprlp/`](bench/ablation_fprlp/) |
| 5 | headline | 233 instances, 600 s: shipped defaults, the previous four-heuristic vector, and vanilla | the reported result | [`bench/headline/`](bench/headline/) |

Two findings from those stages are worth carrying out of them. **Scylla was not
retired by the campaign metric** but by per-heuristic attribution — zero incumbent
improvements against FJ's 24763 and LocalMIP's 4651 over a 233-instance tree; an
objective that scores outcomes cannot retire a component. And the differences
among the surviving mix arms on the tuning set are **unresolvable with this
benchmark at any affordable n**, which is why the selection rests on the
out-of-sample comparison rather than on the search's own ranking.

[`bench/README.md`](bench/README.md) is the harness: every script, the config
naming and its zeroing semantics, where the MIPLIB collection lives, instance
filters and the config oracle, and how a tuning subset is derived.
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) is the contract: what is
reproducible, what deliberately is not, and the stage-by-stage recipe. The runs
are 16-worker and therefore non-deterministic by design, so a re-run reproduces
the *result*, not the logs.

## Citing

[`CITATION.cff`](CITATION.cff) carries the citation metadata in machine-readable
form — GitHub's "Cite this repository" button reads it.

**There is no tagged release and no DOI yet**, and this repository has no paper
of its own. Until the first release mints one
([#166](https://github.com/spoorendonk/mip-heuristics/issues/166)), cite a commit
rather than `main`: the benchmark tables move with the code.

What there *is* to cite is the work these implementations are of — cite the
source paper for the heuristic you are referring to, from the list below. The
`mipfeas` benchmark, its instance set and its metric are Bussieck and Dirkse's
([*Expanding the Focus*](https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/),
GAMS, 2026).

## References

1. **Salvagnin, Roberti, Fischetti (2025)** — *A fix-propagate-repair heuristic for mixed integer programming*. Mathematical Programming Computation, 17:111–139. DOI: [10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5) — FPR and `fpr_lp`.

2. **Lin, Zou, Cai (2024)** — *An Efficient Local Search Solver for Mixed Integer Programming*. Proc. CP 2024, LIPIcs vol. 307, Article 19, pp. 19:1–19:19. DOI: [10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19) — LocalMIP.

3. **Mexi, Besançon, Bolusani, Chmiela, Hoen, Gleixner (2023)** — *Scylla: a matrix-free fix-propagate-and-project heuristic for mixed-integer optimization*. OR Proceedings 2023, 57–63. DOI: [10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9) — Scylla.

4. **Luteberget, Sartor (2023)** — *Feasibility Jump: an LP-free Lagrangian MIP heuristic*. Mathematical Programming Computation, 15:365–388. DOI: [10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8) — FeasibilityJump.

All four are open access under CC BY 4.0 and the PDFs are in
[`docs/`](docs/README.md), which maps each paper to the source files that
implement it.

## Related Projects

- [**HiGHS**](https://github.com/ERGO-Code/HiGHS) — the solver these heuristics are compiled into, via a patched build of v1.15.1
- [**cuOpt**](https://github.com/NVIDIA/cuopt) — NVIDIA's GPU solver, with variants of LocalMIP and of the Scylla pump (arXiv:2510.20499)

## License

[MIT](LICENSE)
