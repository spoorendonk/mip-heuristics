# Plan: New repo — compile custom primal heuristics into HiGHS

## Context

The standalone mip-heuristics library can't compete with HiGHS on upper bounds due to lack of LP guidance. The algorithms (FPR, LocalMIP, ScyllaFPR) are the real value. This plan creates a new repo where HiGHS is the vessel: we compile our heuristics into HiGHS, producing the standard `highs` binary with extra primal heuristics. The code is structured so the algorithms are easy to read and repurpose — HiGHS maintainers can adopt what they want.

**HiGHS landscape**: Mark Turner has draft branches for `local-mip` (WalkSAT-style 1-opt) and `mt/fix-and-propagate` (one-shot fix+propagate, no repair). Neither is merged or a PR. Our implementations are more mature: FPR adds WalkSAT repair; LocalMIP adds weight decay, restarts, aspiration, lift moves. We build independently and let benchmarks showcase quality.

**Not porting**: Diving (redundant inside HiGHS B&B). FJ is used via the existing HiGHS implementation, wrapped as a portfolio arm via `feasibilityJumpCapture` patch.

## Project structure

```
mip-heuristics/
  plan.md                        # this plan
  CMakeLists.txt                 # fetch HiGHS, build patched binary
  cmake/
    FetchHiGHS.cmake             # FetchContent + PATCH_COMMAND
  third_party/highs_patch/
    apply_patch.cmake            # idempotent string_replace patches
  src/                           # OUR CODE — readable, self-contained algorithms
    fpr.h / fpr.cpp              # Fix-Propagate-Repair with WalkSAT
    local_mip.h / local_mip.cpp  # LocalMIP tabu search
    scylla_fpr.h / scylla_fpr.cpp # LP-guided FPR variant
    heuristic_common.h           # shared types/utilities (HeuristicResult, CscMatrix)
    adaptive/
      portfolio.h / portfolio.cpp  # adaptive portfolio orchestrator
      thompson_sampler.h / .cpp    # Beta-Bernoulli Thompson Sampling bandit
      solution_pool.h / .cpp       # thread-safe top-K solution pool
  tests/
    test_basic.cpp
```

Target: HiGHS **v1.13.1** (same as cptp, proven patch compatibility).

**Design principle**: `src/` contains the algorithms in readable form. `third_party/highs_patch/` is the minimal glue. Someone reading `src/fpr.cpp` should understand the algorithm without reading HiGHS internals.

## Patch design (apply_patch.cmake)

Idempotent `string(FIND ...)` + `string(REPLACE ...)` on `HighsMipSolver.cpp` and `HighsOptions.h`. No `HighsUserHeuristic` dispatch layer — the patch calls algorithm entry points directly, gated by `mip_heuristic_run_*` options.

### Patch points (2 insertions in HighsMipSolver.cpp)

| Location | After | Insert |
|---|---|---|
| Pre-root-node | `feasibilityJump()` block closing `}` | Portfolio mode: `portfolio::run_presolve(*this)`. Sequential: `fpr::run(); local_mip::run()`. FJ skipped when portfolio is on (runs as arm). |
| B&B dive | RINS/RENS block closing `}` | Portfolio mode: `portfolio::run_lp_based(*this)`. Sequential: `scylla_fpr::run()`. Standalone RINS/RENS guarded when portfolio is on. |

Includes added at top: `fpr.h`, `local_mip.h`, `scylla_fpr.h`, `adaptive/portfolio.h`.

Our OBJECT library objects are injected into the `highs` target via `target_sources` — no need to touch `cmake/sources.cmake`.

## Data bridge

Algorithms access HiGHS data directly (no copying):

| What | HiGHS accessor |
|---|---|
| Row-major constraints | `mipdata_->ARstart_/ARindex_/ARvalue_` |
| Column-major constraints | `model_->a_matrix_.start_/index_/value_` |
| Variable bounds | `mipdata_->domain.col_lower_/col_upper_` |
| Variable types | `model_->integrality_` |
| Objective | `model_->col_cost_`, `model_->sense_` |
| Row bounds | `model_->row_lower_/row_upper_` |
| Incumbent | `mipdata_->incumbent` |
| LP solution (Scylla) | `mipdata_->lp.getLpSolver().getSolution().col_value` |
| Inject solution | `mipdata_->addIncumbent(sol, obj, source)` |

`heuristic_common.h` provides `HeuristicResult`, `CscMatrix`, `build_csc`, and `is_integer` helpers.

## Implementation steps

### ~~Step 1: Skeleton~~ ✅ Done (PR #1)

CMake + FetchHiGHS v1.13.1 + patch wiring + no-op algorithm stubs. Build pipeline verified: `highs` binary builds, Catch2 smoke test passes.

### ~~Step 2: Port FPR~~ ✅ Done (PR #2)

Full FPR algorithm (~600 LOC) ported into `src/fpr.cpp`. Zero-copy constraint access via AR arrays + local CSC column view. Three phases: rank by degree×|cost| → greedy fix with worklist propagation + snapshot backtracking → WalkSAT repair with O(1) violated-set tracking. Finds `H` solutions on flugpl and neos-911970 at pre-root-node.

### ~~Step 3: Port LocalMIP~~ ✅ Done (PR #3)

Tabu search with breakthrough scoring, lift moves, weight smoothing, aspiration, BMS sampling. ~600 LOC core in `src/local_mip.cpp`. Runs after FPR when incumbent is available.

### ~~Step 4: Port ScyllaFPR~~ ✅ Done (PR #4)

LP-guided FPR variant in `src/scylla_fpr.cpp`. Ranks by LP fractionality instead of degree. Runs during B&B dives when LP solution is available.

### ~~Step 5: Shared FPR core~~ ✅ Done (PR #6)

Extracted common FPR logic into shared core, eliminating duplication between FPR and ScyllaFPR.

### ~~Step 6: Code quality~~ ✅ Done (PRs #5, #7)

Added `.clang-format` and `.clang-tidy` configs. Fixed all clang-tidy warnings.

### ~~Step 7: HiGHS option parameters~~ ✅ Done (PR #8)

Registered `mip_heuristic_run_fpr`, `mip_heuristic_run_local_mip`, `mip_heuristic_run_scylla_fpr` as standard HiGHS boolean options (default `true`). Call sites gated with `if` guards.

### ~~Step 8: Adaptive portfolio~~ ✅ Done (PR #11)

Thompson Sampling bandit with FPR, LocalMIP, FJ arms (presolve) and ScyllaFPR, RINS/RENS arms (LP-based). Thread-safe solution pool with crossover/copy restarts. Two execution modes: deterministic (epoch-based, `for_each` parallel) and opportunistic (free-running workers, atomic budgets). New options: `mip_heuristic_portfolio`, `mip_heuristic_portfolio_opportunistic`. FJ wrapped via `feasibilityJumpCapture` patch.

### Step 9: Benchmark
- Patched HiGHS vs vanilla on MIPLIB subset.

### Future (not in first pass)
- Pseudocost diving — skip, redundant inside HiGHS B&B

## Verification

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
./build/bin/highs test.mps
# verify heuristics fire (addIncumbent prints source on new incumbent)
ctest --test-dir build -j$(nproc)
```

## Key reference files

- `~/code/my/cptp/cmake/FetchHiGHS.cmake` — FetchContent pattern
- `~/code/my/cptp/third_party/highs_patch/apply_patch.cmake` — patch script pattern
- `~/code/my/cptp/build/_deps/highs-src/highs/mip/HighsMipSolver.cpp` — patch target
- `~/code/my/mip-heuristics-old/src/heuristic/fpr/fpr.cpp` — FPR to port
- `~/code/my/mip-heuristics-old/src/heuristic/local_mip/local_mip.cpp` — LocalMIP to port
- `~/code/my/mip-heuristics-old/src/heuristic/scylla/scylla_fpr.cpp` — ScyllaFPR to port
