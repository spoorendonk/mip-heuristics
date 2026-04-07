# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.devkit/standards/cpp.md

## Project Overview

Custom MIP (Mixed-Integer Programming) heuristics integrated into the HiGHS solver via a patched fork. The heuristics run during HiGHS's presolve phase and are compiled as object files linked directly into the `highs` library target.

## Build Commands

```bash
# Configure (from repo root)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Run a single test by name
cd build && ctest -R "Portfolio: flugpl" --output-on-failure

# Run tests matching a Catch2 tag
cd build && ./mip_heuristics_tests "[portfolio]"
```

First build is slow (~5 min) because it fetches and builds HiGHS via FetchContent.

## Architecture

**Integration model**: Heuristics are compiled as a static object library (`mip_heuristics`) whose objects are injected into the HiGHS `highs` target. The HiGHS source is fetched at build time (v1.13.1) with patches applied from `third_party/highs_patch/`. Heuristics access HiGHS internals directly via `HighsMipSolver&`.

**Heuristic entry points** — each has a standalone `run()` that HiGHS calls during presolve:
- `fpr` — Feasibility Pump with Rounding. `fpr_core` contains the shared single-attempt logic.
- `local_mip` — neighborhood-search local MIP solver.
- `scylla` — parallel Scylla restarts with LP-guided scoring.
- `portfolio` — adaptive bandit (Thompson sampling) that selects among FPR, LocalMIP, and FeasibilityJump arms. Has deterministic and opportunistic (parallel) modes.

**Shared utilities** (`src/`):
- `heuristic_common.h` — `HeuristicResult`, `CscMatrix`, row violation, clamping, deadline helpers.
- `thompson_sampler` — Beta-Bernoulli Thompson Sampling bandit (thread-safe option).
- `solution_pool` — thread-safe top-K solution pool with crossover restarts.

**HiGHS options** added by the patch:
- `mip_heuristic_run_fpr`, `mip_heuristic_run_local_mip`, `mip_heuristic_run_scylla` — enable/disable individual heuristics.
- `mip_heuristic_scylla_parallel` — run Scylla pump chains in parallel (independent of portfolio mode).
- `mip_heuristic_portfolio`, `mip_heuristic_portfolio_opportunistic` — enable portfolio mode / parallel opportunistic mode.
- `mip_heuristic_run_feasibility_jump` — enable FJ arm in portfolio.

**Testing**: Catch2 v3. Tests use `.mps` instances from HiGHS's own `check/instances/` directory (path injected via `INSTANCES_DIR` compile definition). Characterization tests verify known-optimal objectives.

**Benchmarking**: `bench/` has scripts for MIPLIB benchmarks — `run_benchmark.py` runs instances, `analyze_results.py` parses results.
