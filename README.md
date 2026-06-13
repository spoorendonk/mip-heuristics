# mip-heuristics

A complete MIP primal heuristics suite integrated into [HiGHS](https://github.com/ERGO-Code/HiGHS) v1.14.0 via a patched build. The project makes FJ, FPR (Salvagnin et al. 2025), LocalMIP (Lin–Zou–Cai CP 2024), a PDLP-based feasibility pump, and a Thompson-sampling adaptive portfolio available natively within HiGHS — without requiring GPU infrastructure — as a research and experimentation platform. GPU-based implementations of several of these algorithms exist in NVIDIA cuOpt (arXiv:2510.20499); this project provides CPU reference implementations integrated into HiGHS's parallel B&B infrastructure.

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
python bench/run_benchmark.py \
  --instances bench/instances_small.txt \
  --binary ./build/bin/highs \
  --data-dir /tmp/miplib \
  --time-limit 300
python bench/analyze_results.py bench/results
```

## Heuristics

**FPR (Fix, Propagate, and Repair)** — LP-free DFS tree search that fixes integer variables one at a time, propagates bounds at each node, and backtracks on infeasibility. After the DFS, WalkSAT and RepairSearch repair any remaining constraint violations. The presolve variant (Class 1) runs multiple strategy configurations in parallel. Based on Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025 ([doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)). The full backtracking+WalkSAT+RepairSearch pipeline is not present in HiGHS, SCIP, or CBC.

**fpr_lp (LP-guided FPR, Classes 2–3)** — Uses the root LP solution to seed the DFS fixing order and initial values (paper Classes 2, 3a, 3b). Dispatched during the B&B dive (after RENS/RINS), not presolve. Workers are bound to distinct LP arm configurations; excess workers wrap with distinct seeds. Shares the FPR rounding kernel.

**LocalMIP** — Weighted tabu local search with constraint-violation tracking, lifting moves, and multi-start backtracking. Finds improving moves by solving small MIP subproblems over the neighborhood. Based on Lin, Zou, Cai, "An Efficient Local Search Solver for Mixed Integer Programming," CP 2024, Article 19 ([doi:10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19)). Not in HiGHS or SCIP; cuOpt has a GPU variant citing the same paper. This is a CPU/HiGHS implementation with epoch-gated parallel multistart.

**Scylla** — PDLP-based feasibility pump: alternates approximate LP solves (PDLP) with FPR rounding, progressive objective blending, and cycling perturbation. N independent pump chains share one mutex-guarded PDLP instance; workers that lose the lock round against the most-recent stale snapshot to stay productive. Based on Mexi et al., *OR Proceedings 2023* ([doi:10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9)); same concept as cuOpt (arXiv:2510.20499). This is a CPU/HiGHS reference implementation — no novelty claim, but it is the only publicly available CPU implementation.

**FeasibilityJump** — LP-free Lagrangian heuristic. Thin wrapper around HiGHS's built-in FJ implementation, routed through our parallel infrastructure for effort budgeting and portfolio integration. Based on Luteberget, Sartor, *Mathematical Programming Computation* 15, 365–388, 2023 ([doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8)). Note: `mip_heuristic_run_feasibility_jump` (default true in our patch) disables HiGHS's internal FJ dispatch and routes it through our infrastructure.

**Thompson portfolio** — Beta-Bernoulli bandit that adaptively selects arms (FPR, LocalMIP, FJ, Scylla) based on feasibility success rates. Experimental; adaptive heuristic selection of this kind is not present in HiGHS, SCIP, or cuOpt.

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

Individual flags (`mip_heuristic_run_fpr`, `mip_heuristic_run_local_mip`, `mip_heuristic_run_scylla`, `mip_heuristic_run_feasibility_jump`, `mip_heuristic_portfolio`, `mip_heuristic_opportunistic`) take effect when no preset is set.

`fpr_lp` runs at B&B dive time and is not affected by the `portfolio` flag; only `mip_heuristic_opportunistic` selects between its epoch-gated and continuous variants.

## Benchmarks

Stage 1 results on 25 hard MIPLIB 2017 instances (`bench/instances_hard25.txt`), 300s time limit:

- Patched HiGHS (`all_opp`): **35.6% primal integral improvement** over vanilla HiGHS.
- Vanilla HiGHS is faster to first feasible solution; patched HiGHS wins on solution quality over time.

PLATO mipfeas 233-instance results are in progress (`bench/instances_bench.txt`).

## Tunable Parameters

~57 `constexpr` parameters control per-heuristic behavior (WalkSAT step counts, RepairSearch node limits, bandit priors, PDLP epsilon schedules, etc.). See [`docs/PARAMETERS.md`](docs/PARAMETERS.md) for the full annotated list with defaults and suggested ranges.

Key solver-level options:

| Option | Default | Description |
|--------|---------|-------------|
| `mip_heuristic_preset` | `""` | Convenience preset; overrides individual flags |
| `mip_heuristic_effort` | `0.30` | Normalized wall-clock budget fraction for all heuristics |

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

## Citation

```bibtex
@software{mip-heuristics,
  title  = {mip-heuristics: A complete MIP primal heuristics suite for HiGHS},
  author = {Spoorendonk, Simon},
  year   = {2026},
  url    = {https://github.com/spoorendonk/mip-heuristics},
  note   = {Zenodo DOI pending}
}
```

## License

[MIT](LICENSE)
