# mip-heuristics

A unified open-source reference implementation and empirical evaluation of four modern primal heuristics — FeasibilityJump, FPR, LocalMIP and Scylla — inside one solver, plus `fpr_lp`, FPR's LP-guided variant. All of them are integrated into [HiGHS](https://github.com/ERGO-Code/HiGHS) v1.15.1 via a patched build, behind a common integration interface with shared budgeting and solution submission, so they can be measured against each other under identical conditions. See [Heuristics](#heuristics) for algorithmic details and paper references.

The contribution is the open implementations and the comparable measurements, not a solver configuration that beats HiGHS: on instances never used for tuning the patched solver improves the `mipfeas` primal-integral SGM by 16.4% — it reaches a good solution *sooner*, while final solution quality is level. See [Evaluation](#evaluation) for the numbers, the experiment that produced them, and their provenance.

**Documentation**: [`CONTRIBUTING.md`](CONTRIBUTING.md) (build, lint, review bar) · [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) (what is reproducible, and the `mipfeas` protocol) · [`docs/PARAMETERS.md`](docs/PARAMETERS.md) (every tunable constant) · [`docs/README.md`](docs/README.md) (source papers).

## Quick Start

**Prerequisites**: CMake 3.25+, GCC 13+ or Clang 17+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)          # first build fetches and compiles HiGHS
./build/bin/highs model.mps             # the shipped heuristic defaults

# The custom options are not CLI flags — HiGHS's command line takes only its
# own fixed set, and rejects an unknown `--flag` without solving.  Pass them
# through an options file:
printf 'mip_heuristic_fj_effort = 0\nmip_heuristic_local_mip_effort = 0\n' > run.opts
./build/bin/highs --options_file run.opts model.mps
```

Full `mipfeas` benchmark against vanilla HiGHS (requires MIPLIB instances and a separately built unpatched HiGHS of the same tag, ~77 h total):

```bash
bash bench/download_miplib.sh
export MIPFEAS_VANILLA_BINARY=/path/to/unpatched/highs   # not the patched build
bench/run_mipfeas.sh next 24    # run in chunks; resumes safely
bench/run_mipfeas.sh status     # check progress
python3 bench/analyze_results.py bench/results/mipfeas --configs all vanilla --time-limit 600 --baseline
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

**What ships is three of the five.** The defaults are `B'-mix-cheapest`, the configuration the joint search selected and held-out validation confirmed ([stage 3](#the-experiment)).

**Scylla ships disabled, and that is a measurement.** It produced zero accepted incumbents across ~380 runs on 233 instances while dispatching on every solve, and it is absent from all 14 survivors of the configuration search. The implementation is maintained and tested; it is the *budget* that is zero, so raising the option re-enables the heuristic with no rebuild.

**`fpr_lp` ships disabled for a different reason**, and its option is in a different unit. It runs at B&B dive time and draws from upstream's RENS/RINS LP-iteration envelope (`mip_heuristic_effort`), charging back what it spends — so its budget is zero-sum against those two rather than an independent allowance, and `mip_heuristic_fpr_lp_effort` is its **share of that envelope**. A call is sized at `share x min(remaining headroom, per-call cap)`, so `1.0` takes the whole slice and a share above `1.0` grows the budget without bound. Overdrawing is self-correcting: the charge-back depletes the counters the headroom is read from, so the next call finds none and skips. `0` returns above every read and write of those counters, which is what makes an `fpr_lp` ablation leave RENS and RINS exactly as they were. It ships at `0` because given the whole envelope it produces accepted incumbents, and inside the shipped chain it produces none ([stage 4](#the-experiment)).

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

Because `scylla` and `fpr_lp` ship at `0`, enabling either takes a positive value rather than merely leaving it alone. `bench/run_benchmark.py` wraps this as config *names* — `fj+fpr`, `all`, `off` — which zero the efforts a name does not list; see [Running the benchmarks](#running-the-benchmarks).

`mip_heuristic_run_feasibility_jump` is upstream's own switch and keeps its meaning: `false` disables FeasibilityJump entirely, the same thing `mip_heuristic_fj_effort = 0` does. The one configuration that contradicts itself — a positive `mip_heuristic_fj_effort` with `mip_heuristic_run_feasibility_jump = false` and nothing else enabled — warns, because it asks for FJ and then takes it away, leaving a run with no heuristic at all while being spelled like an "FJ isolated" row. `bench/run_benchmark.py` greps for that warning and discards the run rather than filing it under a configuration it did not honour.

**Zeroing all five is an ablation, not a vanilla baseline.** It disables our four presolve heuristics and `fpr_lp`, but the binary around it is still the patched one. It also runs **no FeasibilityJump at all**: upstream's standalone FJ call site never fires on a patched build, so FJ is ours or it is absent. Use it to measure what the chain contributes on this binary; a vanilla comparison needs a separately built unpatched HiGHS (`bench/run_benchmark.py --vanilla-binary`, which refuses a binary carrying the patch marker).

Put `mip_heuristic_run_feasibility_jump = false` in the options file alongside the five zeros for the **pure patch-overhead** configuration. That is what `bench/check_vanilla_equivalence.py` compares against an unpatched binary with FeasibilityJump likewise disabled, and it *requires* the two to agree: same objective, same node count, same total and heuristic LP iterations, and an empty log diff once wall-clock content is normalised away. It is a gate, not a recorded result — it needs a second binary, so it cannot run in CI and every release re-runs it. What it establishes when green is that injecting the heuristics does not perturb HiGHS's presolve, B&B or LP path.

### Other options

`mip_heuristic_presolve_only` (bool, default `false`) exits the solve after the presolve chain and before the root LP, keeping the incumbent. It is roughly 25x cheaper than a full solve and is the only way to measure the chain without B&B diluting it, which is what makes the tuning search affordable. Such a run reports `Solution limit reached`, `Nodes 0`, `LP iterations 0`, `Dual bound -inf` and CLI exit 1, so a presolve-only tree is not comparable to a full-solve tree on any dual-side metric.

`mip_heuristic_effort` keeps **vanilla semantics and its vanilla default of 0.05**: it is upstream's B&B dive-heuristic envelope, which gates RENS/RINS and `fpr_lp` alike. A patched binary at default options matches vanilla's B&B heuristic budget exactly.

### Reproducible runs

Set `threads = 1` together with a fixed `random_seed`. That is the project's reproducibility contract — a single worker per heuristic, deterministic within one binary. It is not a separate mode and needs no extra option. It is also *not* the benchmark configuration: one worker per heuristic removes the contention Scylla is built around, and it reallocates budget between heuristics, since FJ's option sizes one worker's allowance where the other three size a whole dispatch. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) has the full contract, including what is deliberately not reproducible and why.

One caveat for library embedders (not CLI users): HiGHS's task executor is a process-global singleton, initialised by the first `run()` in the process. A later solve that asks for a *different* thread count fails outright rather than silently using the old one, so pinning `threads=1` on a second `Highs` instance returns an error unless you first call `Highs::resetGlobalScheduler(true)`.

## Evaluation

**What `mipfeas` is.** A MIP feasibility benchmark: 233 instances selected from
the MIPLIB 2017 benchmark collection with the known-infeasible ones excluded,
scored by the primal integral. It is a broad collaboration — Bussieck and Dirkse
(GAMS), Mittelmann, ZIB and NVIDIA — introduced in [*Expanding the Focus:
Introducing the mipfeas Benchmark*](https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/)
(GAMS, 17 March 2026), which is the source for the instance set and the metric
used throughout this repository. Results are published on Hans Mittelmann's
server at Arizona State, [plato.asu.edu](https://plato.asu.edu/bench.html) —
*PLATO* is that host's name, not a name for the benchmark.

### The experiment

Five stages. Each is a tracked launcher plus a tracked reader; each stage's
results tree is gitignored (~5 GB in total) and each stage's derived artifacts
are committed, so every number below is readable without the logs and
regenerable from them. All stages ran on one 16-core / 32-thread machine at
HiGHS's default worker count (**16 workers**), CPU build, one seed.

| # | stage | what it runs | what it decides | write-up |
|---|---|---|---|---|
| 1 | vanilla baseline | unpatched HiGHS v1.15.1, 233 instances, 600 s | the control for stage 5, and the time-to-first-feasible axis stage 2 stratifies on | — |
| 2 | presolve probe | each presolve heuristic **alone**, presolve-only, 30 s, 233 instances | a starting effort and patience per heuristic, and the 90-instance tuning set | [`bench/ablation_effort/`](bench/ablation_effort/) |
| 3 | joint search | irace over 8 parameters on the 90 tuning instances, presolve-only, 60 s; then finalists at 600 s | the shipped mix and budgets — `B'-mix-cheapest` | [`bench/ablation_search/`](bench/ablation_search/) |
| 4 | `fpr_lp` ablation | capability at 120 s, then contribution at 600 s, 49 instances | `mip_heuristic_fpr_lp_effort = 0` | [`bench/ablation_fprlp/`](bench/ablation_fprlp/) |
| 5 | headline | 233 instances, 600 s: shipped defaults, the previous four-heuristic vector, and vanilla | the reported result | [`bench/headline/`](bench/headline/) |

**Stage 1 — vanilla baseline.** A separately built *unpatched* HiGHS of the tag in `cmake/FetchHiGHS.cmake`, over all 233 `mipfeas` instances at the benchmark's 600 s limit. Two binaries is the design: the patched binary with every heuristic zeroed is an ablation and never a stand-in.

**Stage 2 — the presolve probe.** Each of the four presolve heuristics runs **alone** — not as the chain, since `run_sequential` is sequential and at an unbindable budget the first heuristic would take the entire cap — with `mip_heuristic_presolve_only = true`, its effort option at the `1e6` ceiling so the budget **cannot bind**, and its patience at `0` so there is **no staleness gate at all**. The wall clock is then the single stopping rule, identical for every heuristic on every instance: a 30 s per-run cap, 233 instances, one seed, 932 runs in 8.0 h. Three things come out of it:

  * **effort** = the median charged effort at a dispatch's *last incumbent improvement*, over the dispatches that finished improving. A barren dispatch has no knee and poses a cost question rather than a budget one; a dispatch still improving when the cap fired is right-censored and would bias the estimate upward.
  * **patience** = `min(p95 of the inter-improvement gaps, effort / 4)`. The p95 is a retention claim, the clamp a cost claim — and the clamp binds on all four, so each derived patience is a quarter of the effort beside it. The divisor is `kPatienceCeilingDivisor = 4`, upstream's own ratio: HiGHS's FeasibilityJump pairs `kMaxTotalEffort = nnz << 10` with `kMaxEffortSinceLastImprovement = nnz << 8`.
  * **the instance sets.** 191 of the 233 instances are *informative* — a presolve heuristic produced the reported incumbent on at least one arm — and `bench/instances_tuning.txt` is 90 instances sampled from those, stratified on **stage 1's vanilla time-to-first-feasible** rather than on anything the probe measured, which is what keeps the tuning set from being conditioned on the outcome it will be used to compare. 90 rather than a smaller set because the winner of a search over `n` configurations is biased upward by roughly `sigma * sqrt(2 ln n)`.

  The derived vector — fj 0.5665, fpr 12.2559, local_mip 13.9607, scylla 3.0680 — is a **search centre for stage 3, not a shipped default**: 97–99% of every heuristic's effort lands after its last incumbent improvement, and the report brackets each median against a censoring-aware upper end (e.g. fpr 12.26 .. 93.71).

**Stage 3 — the joint search.** irace over eight real parameters (four efforts, four patiences), plus two discrete switches per heuristic so a log-sampled real can reach exactly `0` in both senses. Domains are per-heuristic and centred on stage 2's knees. The objective is fixed ahead of the runs in `bench/irace/README.md`: `cost = gap + lambda * tau`, where `gap` is the primal gap of the presolve-exit incumbent (capped at 1, penalty 2.0 when nothing is found) and `tau` is the heuristics' own wall time. 3000 experiments per `lambda`, `lambda` in {1/1200, 1/600, 1/300}, on the 90 tuning instances, presolve-only at 60 s, seed 1. A second, constrained search forces all four heuristics on.

  Scylla is absent from all 14 free-search survivors at every cost weight. Five finalist arms were then scored at the campaign metric — primal-integral SGM at 600 s — on a 49-instance confirmation subset of the tuning set, and the surviving candidates on a **48-instance held-out set disjoint from the tuning 90**. The tie-break (fewer heuristics, then lower total effort) selects `B'-mix-cheapest`: fj 0.3317, fpr 3.161, local_mip 3.2865, scylla 0. Held-out, paired against the four-heuristic vector, `B'` is **12.6% better** (ratio 0.874, 95% CI [0.782, 0.977], 31 win / 7 tie / 10 loss) at under a quarter of its total effort — 6.78 against 29.85.

  A methodological finding travels with it. At the measured paired sd of 0.44–0.57, n=48 resolves a 20.6% difference at 80% power and the whole 233-instance campaign at three seeds would resolve 5.1%, while the differences among the mix arms on the **tuning-set** confirmation are 0–8% — not merely unresolved there but **unresolvable with this benchmark at any affordable n**. What separates `B'` is the out-of-sample comparison above, and what retires Scylla is not the metric at all but per-heuristic attribution: **zero** incumbent improvements against FJ's 24763 and LocalMIP's 4651 over a 233-instance tree. An objective that scores outcomes cannot retire a component.

**Stage 4 — the `fpr_lp` ablation.** `fpr_lp` charges its work back to the counters `moreHeuristicsAllowed()` reads, so its budget is zero-sum against RENS and RINS and shipping it unmeasured is not neutral in either direction. Two stages on the 49-instance confirmation set:

  * **capability** — `fpr_lp` alone, owning the whole envelope (`mip_heuristic_run_rens = false`, `mip_heuristic_run_rins = false`, `mip_heuristic_effort = 1.0`) at a per-call budget that cannot bind, 120 s. It reaches dive nodes on 38 of 48 observable instances (79%) and produces 60 accepted incumbents on 10 of 49. The output is a **count**, so it needs no power calculation and a zero would have been decisive at any n.
  * **contribution** — the shipped configuration with and without `fpr_lp` at share 1.0, paired on identical instances at the 600 s campaign limit with RENS/RINS live. **Zero accepted incumbents across all 49 runs**; nothing separates in either direction (ratio 0.955, 95% CI [0.87, 1.05]). The ten instances where the dive never fires are a null control and come back at 0.996, CI [0.99, 1.00], which is what licenses reading the rest.

  The verdict rests on the zero, not on the metric: at n=49 the observed 4.5% would need n=403 to resolve. `fpr_lp` is **dominated by RENS/RINS inside the shared envelope** — a larger share buys more of nothing at their expense and a smaller one converges on `0`, so the lever the data points at is a gate, not a budget.

**Stage 5 — the headline.** All 233 instances at 600 s: the shipped binary at **default options** (no `--extra-options` at all, so this measures what a user gets), the previous four-heuristic vector including Scylla, and the stage 1 vanilla arm. `mip_heuristic_effort` is at upstream's default on both sides, so the only difference between patched and vanilla is what runs during presolve.

### The result

233 instances, 600 s, one seed, 16 workers, CPU build. Full write-up and the
paired statistics: [`bench/headline/`](bench/headline/).

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

#### Three qualifications that belong with it

**The win is magnitude, not frequency.** 57 better against 61 worse on
held-out; a sign test finds nothing (p = 0.71). The heuristics do not win more
often — they win *bigger*: `comp07-2idx` 600 → 7.8 and `sorrell3` 162 → 2.3,
against `fast0507` 14.4 → 248.

**Final solution quality is level.** Paired final primal gap at 600 s: better
on 45, tied on 130, worse on 34. HiGHS's own machinery still holds **188 of
214** final answers. The claim is that HiGHS reaches a good solution *sooner*,
not that it reaches a better one — which is what the primal integral measures
and what a feasibility heuristic should claim.

**Low power with significance implies overstatement.** At the measured paired
sd the held-out n=143 resolves about 17.5%, and the observed effect is 16.4% —
both read on the same side of the ratio. Quote the interval, not the point.

#### Where it comes from

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

#### Attribution

| | patched #First | #Best | vanilla #First | #Best |
|---|---|---|---|---|
| FJ | 114 | 20 | 97 | 10 |
| FPR | 16 | 2 | — | — |
| LocalMIP | 5 | 4 | — | — |
| HiGHS/other | 79 | **188** | 114 | **201** |

Ours find the first feasible solution on 135 of 214 instances against vanilla's
97 of 211, and hold the final best on 26 against 10.

> **Reproducing.** [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) has the
> stage-by-stage recipe. The runs are 16-worker and therefore non-deterministic
> by design, so a re-run reproduces the *result*, not the logs. The campaign's
> logs are not published; the aggregated tables and the paired statistics are
> tracked in [`bench/headline/`](bench/headline/).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
bash bench/download_miplib.sh
bench/run_mipfeas.sh next 24   # run in chunks; resumes safely — repeat until 233/233
python3 bench/analyze_results.py bench/results/mipfeas --configs all vanilla --time-limit 600 --baseline --summary
```

Results land in `bench/results/mipfeas/`. The vanilla binary has no default and is never searched for on PATH: set `MIPFEAS_VANILLA_BINARY=/path/to/unpatched/highs`, or drop `vanilla` from `MIPFEAS_CONFIGS`. What a chunked run does is environment, not a second launcher — `MIPFEAS_CONFIGS`, `MIPFEAS_SEEDS`, `MIPFEAS_INSTANCES` and `MIPFEAS_OUTPUT` — see `docs/REPRODUCIBILITY.md`.

## Running the benchmarks

### Where the MIPLIB collection lives

The collection is a 3.5 GB download (~7.3 GB extracted), so it is stored **once per machine, outside any checkout**, and located by a search path rather than a fixed path. `bench/download_miplib.sh` and `run_benchmark.py --data-dir` share it and probe in this order:

1. an explicit `--data-dir` / `DEST_DIR` argument — wins outright, even when the directory is empty, so a name is never silently resolved to some other directory. The two artifacts then differ on what that means: `run_benchmark.py` reads nothing and reports the instances as missing, whereas `download_miplib.sh` treats it as the destination and **downloads 3.5 GB into it** — so check a `DEST_DIR` before passing it
2. `$MIPLIB_DIR`
3. `~/data/miplib`
4. `/tmp/miplib`

The first directory holding more than 200 `.mps.gz` files wins. Only when none does is anything downloaded, and a fresh download lands in the *first* candidate — `~/data/miplib` normally, or `$MIPLIB_DIR` when that is set. `/tmp` is probed so an existing copy is reused instead of refetched, but it is never a download destination, because a collection there does not survive a reboot; the script says so and prints the `mv` that relocates it. The script prints the resolved directory on stdout and everything else on stderr, so `DATA_DIR=$(bash bench/download_miplib.sh)` works.

### Configs

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

### Tuning subsets

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

### Instance subsets and the config oracle

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
.venv/bin/pip install clang-format==22.1.8 clang-tidy==22.1.8 ruff==0.16.3 pytest

ctest --test-dir build --output-on-failure -j$(nproc)                     # everything
ctest --test-dir build -LE lint --output-on-failure                       # fast loop
ctest --test-dir build -R "execution-mode: flugpl objective" --output-on-failure
./build/mip_heuristics_tests "[mode-matrix]"                              # Catch2 tag
```

Catch2 v3. Characterization tests verify known-optimal objectives against MIPLIB instances bundled with HiGHS.

`clang-format` and `clang-tidy` run over `src/` and `tests/` as ctest tests labelled `lint`, adding roughly 30 s to a full run — hence `ctest -LE lint` while iterating. Without the venv above they are **not registered at all** and `ctest` reports green having linted nothing, which is what `-DMIP_HEURISTICS_REQUIRE_LINT=ON` exists to prevent. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the tool-version contract.

## License

[MIT](LICENSE)
