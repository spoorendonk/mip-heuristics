# mip-heuristics

A complete MIP primal heuristics suite integrated into [HiGHS](https://github.com/ERGO-Code/HiGHS) v1.15.1 via a patched build. Makes FJ, FPR, LocalMIP, and Scylla (PDLP-based feasibility pump) available natively within HiGHS as a research and experimentation platform. See [Heuristics](#heuristics) for algorithmic details and paper references.

## Quick Start

**Prerequisites**: CMake 3.25+, GCC 13+ or Clang 17+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)          # first build ~5 min (fetches HiGHS)
./build/bin/highs --mip_heuristic_suite all model.mps
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

**FeasibilityJump** — LP-free Lagrangian heuristic. Thin wrapper around HiGHS's built-in FJ implementation, routed through our parallel infrastructure for effort budgeting and shared solution-pool integration. Based on Luteberget, Sartor, *Mathematical Programming Computation* 15, 365–388, 2023 ([doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8)). Note: at any `mip_heuristic_suite` value other than `off`, HiGHS's internal FJ dispatch is disabled and FJ runs through our infrastructure instead. Upstream's `mip_heuristic_run_feasibility_jump` (default true) still switches FJ off entirely.

Reference PDFs are in `docs/`.

## Execution Modes

The heuristics always run as the fixed chain FJ → FPR → LocalMIP → Scylla with a weighted effort budget. Each heuristic parallelises the same way: continuous workers that self-terminate, with no epoch barrier and no bit-identical guarantee across runs.

For a reproducible run, set `threads=1` together with a fixed `random_seed`. That is the project's reproducibility contract — a single worker per heuristic, deterministic within one binary. It is not a separate mode and needs no extra option.

One caveat for library embedders (not CLI users): HiGHS's task executor is a process-global singleton, initialised by the first `run()` in the process. A later solve that asks for a *different* thread count fails outright rather than silently using the old one, so pinning `threads=1` on a second `Highs` instance returns an error unless you first call `Highs::resetGlobalScheduler(true)`.

`mip_heuristic_suite` is the single option that selects which heuristics run (default `all`):

| Value | Heuristics | Notes |
|--------|-----------|-------|
| `off` | none | vanilla-equivalent: HiGHS's own pipeline, native FeasibilityJump included |
| `fj` | FJ | isolate FeasibilityJump |
| `fpr` | FPR (+ `fpr_lp`) | isolate FPR for ablation |
| `local_mip` | LocalMIP | isolate LocalMIP |
| `scylla` | Scylla | PDLP pump only |
| `all` | FJ+FPR+LocalMIP+Scylla (+ `fpr_lp`) | **default** |

An unrecognised value warns and falls back to `all` rather than silently disabling everything.

**`suite=off` is a true vanilla ablation on one binary.** The patch hands HiGHS's standalone FeasibilityJump call site back at `off`, so an `off` run is what an unpatched build of the same tag does. `bench/check_vanilla_equivalence.py` verifies that against an unpatched binary — identical objective, node count and heuristic LP iterations, and an empty log diff apart from the `mip-heuristics patch active` marker. Add `--mip_heuristic_run_feasibility_jump=false` on top of `off` for the pure patch-overhead configuration, with no heuristics at all.

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

> **Provenance.** Two reasons this row cannot be reproduced on `HEAD`, both by design. First, the configuration is gone: `all_opp` was FJ + FPR + LocalMIP without Scylla, and the single-valued `mip_heuristic_suite` (#93) cannot express that combination — the closest value, `all`, adds Scylla. Second, the binary is gone: the numbers predate the #92 runner cleanup, which altered several things they depend on — workers no longer stop their peers on retiring, LocalMIP's cold start is primed once per dispatch rather than per worker, FJ's charge against the presolve envelope is floored, and two of the three `kWeight*` constants were rescaled. The closeout benchmark campaign re-measures on the final tree; treat the row as the last full-campaign result, not as a claim about `HEAD`.

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
