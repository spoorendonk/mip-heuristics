# mip-heuristics

A complete MIP primal heuristics suite integrated into [HiGHS](https://github.com/ERGO-Code/HiGHS) v1.14.0 via a patched build. Makes FJ, FPR, LocalMIP, Scylla (PDLP-based feasibility pump), and a Thompson-sampling adaptive portfolio available natively within HiGHS as a research and experimentation platform. See [Heuristics](#heuristics) for algorithmic details and paper references.

## Quick Start

**Prerequisites**: CMake 3.25+, GCC 13+ or Clang 17+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)          # first build ~5 min (fetches HiGHS)
./build/bin/highs --mip_heuristic_preset all_opp model.mps
```

Five-instance benchmark against vanilla HiGHS (requires MIPLIB instances):

```bash
bash bench/download_miplib.sh
python3 bench/run_benchmark.py \
  --instances bench/instances_small.txt \
  --binary ./build/bin/highs \
  --data-dir /tmp/miplib \
  --time-limit 300
python3 bench/analyze_results.py bench/results
```

## Heuristics

**FPR (Fix, Propagate, and Repair)** — LP-free DFS tree search that fixes integer variables one at a time, propagates bounds at each node, and backtracks on infeasibility. After the DFS, WalkSAT and RepairSearch repair any remaining constraint violations. The presolve variant (Class 1) runs multiple strategy configurations in parallel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)). The full backtracking+WalkSAT+RepairSearch pipeline is not present in HiGHS, SCIP, or CBC.

**fpr_lp (LP-guided FPR, Classes 2–3)** — Uses the root LP solution to seed the DFS fixing order and initial values (paper Classes 2, 3a, 3b). Dispatched during the B&B dive (after RENS/RINS), not presolve. Workers are bound to distinct LP arm configurations; excess workers wrap with distinct seeds. Shares the FPR rounding kernel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)) (Classes 2–3).

**LocalMIP** — Weighted tabu local search with constraint-violation tracking, lifting moves, and multi-start backtracking. Finds improving moves by solving small MIP subproblems over the neighborhood. Based on Lin, Zou, Cai, "An Efficient Local Search Solver for Mixed Integer Programming," CP 2024, Article 19 ([doi:10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19)). Not in HiGHS or SCIP; cuOpt has a GPU variant citing the same paper. This is a CPU/HiGHS implementation with epoch-gated parallel multistart.

**Scylla** — PDLP-based feasibility pump: alternates approximate LP solves (PDLP) with FPR rounding, progressive objective blending, and cycling perturbation. N independent pump chains share one mutex-guarded PDLP instance; workers that lose the lock round against the most-recent stale snapshot to stay productive. Based on Mexi et al., *OR Proceedings 2023* ([doi:10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9)); same concept as cuOpt (arXiv:2510.20499). This is a CPU/HiGHS reference implementation — no novelty claim, but it is the only publicly available CPU implementation.

**FeasibilityJump** — LP-free Lagrangian heuristic. Thin wrapper around HiGHS's built-in FJ implementation, routed through our parallel infrastructure for effort budgeting and portfolio integration. Based on Luteberget, Sartor, *Mathematical Programming Computation* 15, 365–388, 2023 ([doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8)). Note: `mip_heuristic_run_feasibility_jump` (default true in our patch) disables HiGHS's internal FJ dispatch and routes it through our infrastructure.

**Thompson portfolio** — Beta-Bernoulli bandit that adaptively selects arms (FPR, LocalMIP, FJ, Scylla) based on feasibility success rates. Experimental; adaptive heuristic selection of this kind is not present in HiGHS, SCIP, or cuOpt. Based on Russo, Van Roy, Kazerouni, Osband, Wen, "A tutorial on Thompson sampling," *Foundations and Trends in Machine Learning* 11(1):1–96, 2018 ([doi:10.1561/2200000070](https://doi.org/10.1561/2200000070)).

Reference PDFs are in `docs/`.

## Execution Modes

Two orthogonal flags form a 2×2 dispatch matrix:

| | `opportunistic=false` (epoch-gated, deterministic) | `opportunistic=true` (continuous parallel) |
|---|---|---|
| `portfolio=false` (sequential) | **seq/det**: FJ → FPR → LocalMIP → Scylla with N synchronized workers per epoch | **seq/opp**: same sequence, continuous workers per heuristic |
| `portfolio=true` (bandit) | **port/det**: Thompson bandit, epoch-synchronized workers | **port/opp**: Thompson bandit, continuous workers |

The `mip_heuristic_preset` option sets both flags and the per-heuristic enable flags at once:

| Preset | Heuristics | Mode | Notes |
|--------|-----------|------|-------|
| `off` | none | — | disables all custom heuristics |
| `fpr` | FPR | seq/det | isolate FPR for ablation |
| `all_det` | FJ+FPR+LocalMIP | seq/det | deterministic, reproducible |
| `all_opp` | FJ+FPR+LocalMIP | seq/opp | **recommended** |
| `scylla` | Scylla | seq/opp | PDLP pump only |
| `portfolio` | all | port/opp | experimental adaptive selection |

There is no standalone `local_mip` preset; to run LocalMIP alone, use individual flags (e.g. `--mip_heuristic_run_fpr=false --mip_heuristic_run_scylla=false --mip_heuristic_run_feasibility_jump=false`).

When no preset is set, individual flags are used. The default with no flags set: FPR + LocalMIP + Scylla + FJ all enabled in seq/det mode. `all_opp` is the recommended preset for most use cases.

`fpr_lp` runs at B&B dive time and is not affected by the `portfolio` flag; only `mip_heuristic_opportunistic` selects between its epoch-gated and continuous variants.

## Benchmarks

### PLATO mipfeas — 233 instances, 600s time limit

Full PLATO mipfeas benchmark (233 MIPLIB 2017 instances, 600s per instance, system HiGHS as vanilla baseline). Default preset: `all_opp` (seq/opp mode, FJ+FPR+LocalMIP).

| Metric | Patched (`all_opp`) | Vanilla HiGHS |
|---|---|---|
| #Feasible | 197 | **207** |
| #Win (best primal obj at 600s) | **179** | 154 |
| #Gap@600s wins (191 mutually-solved) | **46** | 34 |
| SGM Time-to-first-feasible (s=1) | 10.8s | **3.8s** |
| SGM Gap@600s (s=0.01) | 0.0148 | **0.0127** |
| SGM Primal Integral (s=0.001) | 48.4 | **42.2** |
| SGM P-D Integral | 25.9 | **23.9** |
| PLATO headline SGM (s=0.001) | 40.0 | **33.0** |

#### Findings

Instance breakdown across 233 total: **191** solved by both configs, **16** by vanilla only, **6** by patched only, **20** by neither.

**Head-to-head on 191 mutually-solved instances** — patched wins Gap@600s (gap to best-known) on 46, vanilla on 34, 111 ties. This is the most direct comparison: same instance, both found solutions.

**#Win (best primal obj) — 179 vs 154** — counts who found the better primal bound across all 213 instances where at least one config found a solution (191 + 16 + 6). Ties within 1e-6 are credited to both configs simultaneously, so 179 + 154 = 333 = 213 + 120 ties. Better primal bound = better gap to best-known (same denominator), so this is equivalent to a gap-based winner count.

**SGM metrics** (Gap@600s, Primal Integral, PLATO) favour vanilla. Note these are each computed over the per-config feasible set — 197 instances for patched, 207 for vanilla — so the populations differ. Two factors compound against patched: (1) the presolve heuristics run before B&B, so every instance accumulates gap area during the ~7s overhead before the first node; (2) the 16 instances vanilla solves but patched does not are absent from patched's SGM but present in vanilla's.

Patched wins more head-to-head quality matchups on instances both solve. Vanilla leads on aggregate metrics and feasibility coverage.

**To reproduce:**

```bash
# 1. Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. Download MIPLIB 2017 instances (~2 GB) to /tmp/miplib
bash bench/download_miplib.sh

# 3. Run benchmark (233 instances × 2 configs × 600s ≈ 77h total wall time)
bench/run_plato.sh next 24        # run in 24h chunks; resumes safely
bench/run_plato.sh status         # check progress
# Repeat until status shows 233/233

# 4. Analyze
python3 bench/analyze_results.py bench/results/plato --configs patched vanilla --time-limit 600 --baseline
```

Results land in `bench/results/plato/`. The vanilla binary defaults to system HiGHS (`which highs`); override with `PLATO_VANILLA_BINARY=/path/to/highs`.

## Build Options

| Flag | Description |
|------|-------------|
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build (default) |
| `-DMIP_HEURISTICS_CUDA=ON` | Enable cuPDLP GPU backend for Scylla; falls back to CPU if no CUDA compiler found |

## Testing

```bash
cd build && ctest --output-on-failure
cd build && ctest -R "Portfolio: flugpl" --output-on-failure   # single test
cd build && ./mip_heuristics_tests "[portfolio]"               # Catch2 tag
```

Catch2 v3. Characterization tests verify known-optimal objectives against MIPLIB instances bundled with HiGHS.

## License

[MIT](LICENSE)
