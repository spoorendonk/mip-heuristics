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
| #Feasible | **213** | 208 |
| #Win (best primal obj at 600s) | **179** | 154 |
| #Gap@600s wins (215 with ≥1 solution) | **53** | 42 |
| SGM Time-to-first-feasible (s=1) | 5.2s | **3.8s** |
| SGM Gap@600s (s=0.01) | 0.0255 | **0.0240** |
| SGM Primal Integral (s=1) | 58.3 | **55.8** |
| SGM P-D Integral | 25.9 | **23.9** |
| PLATO headline SGM (s=0.001) | 46.3 | **44.6** |

#### Findings

Instance breakdown across 233 total: **206** solved by both configs, **2** by vanilla only, **7** by patched only, **18** by neither.

**Head-to-head Gap@600s across 215 instances (≥1 solution found)** — patched 53, vanilla 42, 120 ties. Infeasible-for-one-side instances are counted with gap=1.0. On the 206 mutually-solved instances: patched 46, vanilla 40, 120 ties — patched wins the majority of decisive matchups.

**#Win (best primal obj) — 179 vs 154** — counts who found the better primal bound across all 215 instances where at least one config found a solution (206+7+2). Ties within 1e-6 are credited to both configs simultaneously: 179+154=333=215+118 ties. On the 97 decisive (non-tie) instances, patched wins 61 and vanilla wins 36.

**Heuristic attribution in patched**: LocalMIP finds the first feasible solution on **123/233 instances**; RINS (Sub-MIP) on 31; FJ on 10. FPR contributes first on 2 instances; fpr_lp on 3 incumbents total. At termination, RINS holds the best solution on 149 instances (vs 106 for vanilla), with LocalMIP holding best on 17. The new heuristics are additive — patched finds more solutions (213 vs 208) with LocalMIP doing the heavy lifting.

**SGM metrics** (Gap@600s, Primal Integral, PLATO) favour vanilla narrowly. All SGM computations treat infeasible instances as gap=1.0 / PI=time-limit, so both configs compete on the full 233-instance set. The remaining PLATO ratio is 1.04 (46.3 vs 44.6) — the main driver is patched's slower time-to-first-feasible (~5s vs ~4s SGM) due to presolve heuristics running before B&B.

**Summary**: patched leads on feasibility (213 vs 208), head-to-head Gap@600s (53–42), and #Win (179 vs 154). Vanilla leads narrowly on aggregate time-integrated SGM metrics due to the presolve overhead. On the 206 instances both solve, P-D integral favours patched (33.9 vs 36.7).

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
