# Plan: New repo — compile custom primal heuristics into HiGHS

## Context

The standalone mip-heuristics library can't compete with HiGHS on upper bounds due to lack of LP guidance. The algorithms (FPR, LocalMIP, ScyllaFPR) are the real value. This plan creates a new repo where HiGHS is the vessel: we compile our heuristics into HiGHS, producing the standard `highs` binary with extra primal heuristics. The code is structured so the algorithms are easy to read and repurpose — HiGHS maintainers can adopt what they want.

**HiGHS landscape**: Mark Turner has draft branches for `local-mip` (WalkSAT-style 1-opt) and `mt/fix-and-propagate` (one-shot fix+propagate, no repair). Neither is merged or a PR. Our implementations are more mature: FPR adds WalkSAT repair; LocalMIP adds weight decay, restarts, aspiration, lift moves. We build independently and let benchmarks showcase quality.

**Not porting**: FJ (HiGHS already has it, core identical), Diving (redundant inside HiGHS B&B), Thompson Sampling portfolio (less valuable inside B&B).

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
    common.h                     # shared types/utilities (constraint iteration helpers)
  tests/
    test_basic.cpp
```

Target: HiGHS **v1.13.1** (same as cptp, proven patch compatibility).

**Design principle**: `src/` contains the algorithms in readable form. `third_party/highs_patch/` is the minimal glue. Someone reading `src/fpr.cpp` should understand the algorithm without reading HiGHS internals.

## Patch design (apply_patch.cmake)

Idempotent `string(FIND ...)` + `string(REPLACE ...)` on `HighsMipSolver.cpp`. No `HighsUserHeuristic` dispatch layer — the patch calls algorithm entry points directly.

### Patch points (2 insertions in HighsMipSolver.cpp)

| Location | After | Insert |
|---|---|---|
| Pre-root-node | `feasibilityJump()` block closing `}` | `fpr::run(*this); local_mip::run(*this);` |
| B&B dive | RINS/RENS block closing `}` | `scylla_fpr::run(*this);` |

Includes added at top: `fpr.h`, `local_mip.h`, `scylla_fpr.h`.

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

`common.h` provides lightweight iteration helpers for clean algorithm code.

## Implementation steps

### ~~Step 1: Skeleton~~ ✅ Done (PR #1)

CMake + FetchHiGHS v1.13.1 + patch wiring + no-op algorithm stubs. Build pipeline verified: `highs` binary builds, Catch2 smoke test passes.

### Step 2: Port FPR (Fix-Propagate-Repair)
- **Source**: `mip-heuristics-old/src/heuristic/fpr/fpr.cpp` (~400 LOC core)
- **Algorithm**: Rank variables by degree×objective → greedy fix → propagate bounds → WalkSAT repair
- **LP-free**: Yes. Runs at root before B&B starts.
- **Key differentiator vs HiGHS `mt/fix-and-propagate`**: WalkSAT repair phase. HiGHS branch gives up on infeasibility; we repair.
- **Why first**: Smallest, validates entire patch infrastructure end-to-end.

### Step 3: Port LocalMIP
- **Source**: `mip-heuristics-old/src/heuristic/local_mip/local_mip.cpp` (~600 LOC core)
- **Algorithm**: Tabu search with breakthrough scoring, lift moves, weight smoothing, aspiration
- **LP-free**: Yes. Needs incumbent to start (naturally runs after FPR finds one).
- **Key differentiators vs HiGHS `local-mip` branch**: Weight decay, restarts, aspiration criterion, longer tabu tenure, lift moves (vs one-opt), BMS sampling.
- **Why second**: Strongest algorithm, but depends on incumbent from Step 2.

### Step 4: Port ScyllaFPR
- **Source**: `mip-heuristics-old/src/heuristic/scylla/scylla_fpr.cpp` (~100 LOC wrapper)
- **Algorithm**: FPR but ranks by LP fractionality instead of degree.
- **LP-dependent**: Yes. Runs during B&B dives when LP solution is available.
- **Why third**: Trivial once FPR exists (ranking-strategy swap). Demonstrates LP-guided variant.

### Step 5: Benchmark
- Patched HiGHS vs vanilla on MIPLIB subset.

### Future (not in first pass)
- FJ improvements (pool-based restarts, configurable stall detection) as small upstream PR
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
