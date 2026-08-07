# mip-heuristics

A complete MIP primal heuristics suite integrated into [HiGHS](https://github.com/ERGO-Code/HiGHS) v1.15.1 via a patched build. Makes FJ, FPR, LocalMIP, and Scylla (PDLP-based feasibility pump) available natively within HiGHS as a research and experimentation platform. See [Heuristics](#heuristics) for algorithmic details and paper references.

## Quick Start

**Prerequisites**: CMake 3.25+, GCC 13+ or Clang 17+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)          # first build ~5 min (fetches HiGHS)
./build/bin/highs --mip_heuristic_preset all_opp model.mps
```

Full PLATO benchmark against vanilla HiGHS (requires MIPLIB instances, ~77h total):

```bash
bash bench/download_miplib.sh
bench/run_plato.sh next 24    # run in chunks; resumes safely
bench/run_plato.sh status     # check progress
python3 bench/analyze_results.py bench/results/plato --configs patched vanilla --time-limit 600 --baseline
```

## Heuristics

**FPR (Fix, Propagate, and Repair)** — LP-free DFS tree search that fixes integer variables one at a time, propagates bounds at each node, and backtracks on infeasibility. After the DFS, WalkSAT and RepairSearch repair any remaining constraint violations. The presolve variant (Class 1) runs multiple strategy configurations in parallel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)). The full backtracking+WalkSAT+RepairSearch pipeline is not present in HiGHS, SCIP, or CBC.

**fpr_lp (LP-guided FPR, Classes 2–3)** — Uses the root LP solution to seed the DFS fixing order and initial values (paper Classes 2, 3a, 3b). Dispatched during the B&B dive (after RENS/RINS), not presolve. Workers are bound to distinct LP arm configurations; excess workers wrap with distinct seeds. Shares the FPR rounding kernel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)) (Classes 2–3).

**LocalMIP** — Weighted tabu local search with constraint-violation tracking, lifting moves, and multi-start backtracking. Finds improving moves by solving small MIP subproblems over the neighborhood. Based on Lin, Zou, Cai, "An Efficient Local Search Solver for Mixed Integer Programming," CP 2024, Article 19 ([doi:10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19)). Not in HiGHS or SCIP; cuOpt has a GPU variant citing the same paper. This is a CPU/HiGHS implementation with parallel multistart.

**Scylla** — PDLP-based feasibility pump: alternates approximate LP solves (PDLP) with FPR rounding, progressive objective blending, and cycling perturbation. N independent pump chains share one mutex-guarded PDLP instance; workers that lose the lock round against the most-recent stale snapshot to stay productive. Based on Mexi et al., *OR Proceedings 2023* ([doi:10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9)); same concept as cuOpt (arXiv:2510.20499). This is a CPU/HiGHS reference implementation — no novelty claim, but it is the only publicly available CPU implementation.

**FeasibilityJump** — LP-free Lagrangian heuristic. Thin wrapper around HiGHS's built-in FJ implementation, routed through our parallel infrastructure for effort budgeting and shared solution-pool integration. Based on Luteberget, Sartor, *Mathematical Programming Computation* 15, 365–388, 2023 ([doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8)). Note: `mip_heuristic_run_feasibility_jump` (default true in our patch) disables HiGHS's internal FJ dispatch and routes it through our infrastructure.

Reference PDFs are in `docs/`.

## Execution Modes

The heuristics always run as the fixed chain FJ → FPR → LocalMIP → Scylla with a weighted effort budget. Each heuristic parallelises the same way: continuous workers that self-terminate, with no epoch barrier and no bit-identical guarantee across runs.

For a reproducible run, set `threads=1` together with a fixed `random_seed`. That is the project's reproducibility contract — a single worker per heuristic, deterministic within one binary. It is not a separate mode and needs no extra option.

One caveat for library embedders (not CLI users): HiGHS's task executor is a process-global singleton, initialised by the first `run()` in the process. A later solve that asks for a *different* thread count fails outright rather than silently using the old one, so pinning `threads=1` on a second `Highs` instance returns an error unless you first call `Highs::resetGlobalScheduler(true)`.

The `mip_heuristic_preset` option sets the per-heuristic enable flags:

| Preset | Heuristics | Notes |
|--------|-----------|-------|
| `off` | none | disables all custom heuristics |
| `fpr` | FPR | isolate FPR for ablation |
| `all_opp` | FJ+FPR+LocalMIP | **recommended** |
| `scylla` | Scylla | PDLP pump only |

`all_opp` is named for the continuous ("opportunistic") runner that used to be one of two parallel modes; since the epoch-gated mode was removed it is the only one. The name is kept because it labels the recorded benchmark results below. Its former sibling `all_det` named the removed mode and no longer exists.

There is no standalone `local_mip` preset; to run LocalMIP alone, use individual flags (e.g. `--mip_heuristic_run_fpr=false --mip_heuristic_run_scylla=false --mip_heuristic_run_feasibility_jump=false`).

When no preset is set, individual flags are used. The default with no flags set: FPR + LocalMIP + Scylla + FJ all enabled. `all_opp` is the recommended preset for most use cases.

`fpr_lp` runs at B&B dive time on the same continuous workers.

## Benchmarks

### PLATO mipfeas — 233 instances, 600s time limit

Full PLATO mipfeas benchmark (233 MIPLIB 2017 instances, 600s per instance, system HiGHS as vanilla baseline). Default preset: `all_opp` (FJ+FPR+LocalMIP).

| Metric | Patched (`all_opp`) | Vanilla HiGHS |
|---|---|---|
| #Feasible | **213** | 208 |
| #Win (strict, best primal obj at 600s) | **59** | 41 |
| SGM Time-to-first-feasible (s=1) | **3.6s** | 3.8s |
| SGM Gap@600s (s=0.001) | 0.00699 | **0.00638** |
| SGM Primal Integral (s=1) | **33.25** | 33.57 |
| SGM P-D Integral | 26.3 | **23.9** |
| PLATO headline SGM (s=0.001) | **26.0** | 26.8 |

> **Provenance.** These numbers were measured before the #92 runner cleanup. That changeset altered several things these figures depend on — workers no longer stop their peers on retiring, LocalMIP's cold start is primed once per dispatch rather than per worker, FJ's charge against the presolve envelope is floored, and two of the three `kWeight*` constants were rescaled — so a build of the current tree is not the binary that produced this table. The closeout benchmark campaign re-measures on the final tree; treat the row as the last full-campaign result, not as a claim about `HEAD`.

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
python3 bench/analyze_results.py bench/results/plato --configs patched vanilla --time-limit 600 --baseline --summary
```

Results land in `bench/results/plato/`. Vanilla binary defaults to system HiGHS (`which highs`); override with `PLATO_VANILLA_BINARY=/path/to/highs`.

## Build Options

| Flag | Description |
|------|-------------|
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build (default) |
| `-DMIP_HEURISTICS_CUDA=ON` | Enable cuPDLP GPU backend for Scylla. Requires `CUDA_HOME` exported; **fails the configure** rather than falling back to CPU. Verify with `ldd build-gpu/bin/highs \| grep cudart` |

## Testing

```bash
cd build && ctest --output-on-failure
cd build && ctest -R "execution-mode: flugpl objective" --output-on-failure     # single test
cd build && ./mip_heuristics_tests "[mode-matrix]"                            # Catch2 tag
```

Catch2 v3. Characterization tests verify known-optimal objectives against MIPLIB instances bundled with HiGHS.

## License

[MIT](LICENSE)
