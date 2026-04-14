# mip-heuristics

Custom MIP primal heuristics compiled directly into a patched build of [HiGHS](https://github.com/ERGO-Code/HiGHS).

## Overview

This project implements primal heuristics for finding feasible solutions to mixed-integer programs. The heuristics are compiled as a static object library whose objects are injected into the HiGHS `highs` library target. HiGHS v1.13.1 is fetched via CMake FetchContent and patched at build time. The result is a single `highs` binary with the custom heuristics running alongside HiGHS's built-in ones during presolve and branch-and-bound.

## Heuristics

- **FPR (Fix-Propagate-Repair)** -- Fix-and-propagate framework with multiple variable-ranking and value-selection strategies, plus WalkSAT-based repair search. LP-free and LP-based variants run in separate phases. Based on Salvagnin et al. [1].
- **LocalMIP** -- Tabu-based neighborhood search with constraint-violation tracking, lifting moves, and backtracking multiple starts. Based on Lin et al. [2].
- **Scylla** -- Matrix-free fix-propagate-and-project using PDLP as an approximate LP oracle, with objective perturbation and solution-pool crossover restarts. Based on Mexi et al. [3].
- **FeasibilityJump** -- LP-free Lagrangian heuristic. Wraps HiGHS's built-in implementation with effort budgeting and portfolio integration. Based on Luteberget and Sartor [4].
- **Portfolio** -- Thompson sampling (Beta-Bernoulli) bandit that adaptively selects among FPR, LocalMIP, and FeasibilityJump arms. Has deterministic and opportunistic (parallel) modes.

Reference papers are in `docs/`.

## Building

**Prerequisites**: CMake 3.25+, a C++23 compiler (GCC 13+ or Clang 17+).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

First build is slow (~5 min) because HiGHS is fetched and built via FetchContent.

## Usage

Run the patched HiGHS binary as usual:

```bash
./build/bin/highs model.mps
```

Enable portfolio mode (Thompson sampling arm selection):

```bash
./build/bin/highs --mip_heuristic_portfolio true model.mps
```

### Custom options

| Option | Default | Description |
|--------|---------|-------------|
| `mip_heuristic_run_fpr` | `false` | Enable FPR heuristic |
| `mip_heuristic_run_local_mip` | `false` | Enable LocalMIP heuristic |
| `mip_heuristic_run_scylla` | `false` | Enable Scylla heuristic |
| `mip_heuristic_run_feasibility_jump` | `false` | Enable Feasibility Jump (standalone or portfolio arm) |
| `mip_heuristic_local_mip_parallel` | `false` | Run LocalMIP workers in parallel |
| `mip_heuristic_portfolio` | `false` | Adaptive Thompson-sampling portfolio over arms |
| `mip_heuristic_opportunistic` | `false` | Use continuous (opportunistic) parallelism instead of epoch-gated deterministic parallelism; combines with `mip_heuristic_portfolio` to form the 2×2 execution matrix |

## Benchmarking

The `bench/` directory has scripts for comparing patched vs vanilla HiGHS on MIPLIB instances:

```bash
# Download MIPLIB instances
bash bench/download_miplib.sh

# Run comparison
python bench/run_benchmark.py \
  --instances bench/instances_small.txt \
  --binary ./build/bin/highs \
  --data-dir /tmp/miplib \
  --time-limit 60

# Analyze results
python bench/analyze_results.py bench/results
```

## Tests

Catch2 v3 characterization tests against MIPLIB instances bundled with HiGHS:

```bash
cd build && ctest --output-on-failure
```

## References

1. D. Salvagnin, R. Roberti, M. Fischetti. *A fix-propagate-repair heuristic for mixed integer programming.* Mathematical Programming Computation 17, 111--139, 2025. [doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5)
2. P. Lin, M. Zou, S. Cai. *An Efficient Local Search Solver for Mixed Integer Programming.* In Proc. CP 2024, Article 19, pp. 19:1--19:19. [doi:10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19)
3. G. Mexi, M. Besançon, S. Bolusani, A. Chmiela, A. Hoen, A. Gleixner. *Scylla: a matrix-free fix-propagate-and-project heuristic for mixed-integer optimization.* In Operations Research Proceedings 2023, pp. 57--63. [doi:10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9)
4. B. Luteberget, G. Sartor. *Feasibility Jump: an LP-free Lagrangian MIP heuristic.* Mathematical Programming Computation 15, 365--388, 2023. [doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8)

## License

[MIT](LICENSE)
