# Plan: New repo — compile custom primal heuristics into HiGHS

## Context

The standalone mip-heuristics library can't compete with HiGHS on upper bounds due to lack of LP guidance. The algorithms (FPR, LocalMIP, ScyllaFPR) are the real value. This plan creates a new repo where HiGHS is the vessel: we compile our heuristics into HiGHS, producing the standard `highs` binary with extra primal heuristics. The code is structured so the algorithms are easy to read and repurpose — HiGHS maintainers can adopt what they want.

**HiGHS landscape**: Mark Turner has draft branches for `local-mip` (WalkSAT-style 1-opt) and `mt/fix-and-propagate` (one-shot fix+propagate, no repair). Neither is merged or a PR. Our implementations are more mature: FPR adds WalkSAT repair; LocalMIP adds weight decay, restarts, aspiration, lift moves. We build independently and let benchmarks showcase quality.

**Not porting**: FJ (HiGHS already has it, core identical), Diving (redundant inside HiGHS B&B), Thompson Sampling portfolio (less valuable inside B&B).

## Step 0: Repo transition (manual)

- Rename `spoorendonk/mip-heuristics` → `spoorendonk/mip-heuristics-old` on GitHub
- Rename `~/code/my/mip-heuristics` → `~/code/my/mip-heuristics-old` locally
- Create new `spoorendonk/mip-heuristics` on GitHub and clone to `~/code/my/mip-heuristics`

The old repo stays as reference during porting.

## Project structure

```
mip-heuristics/
  plan.md                        # this plan
  CMakeLists.txt                 # fetch HiGHS, build patched binary
  cmake/
    FetchHiGHS.cmake             # FetchContent + PATCH_COMMAND
  third_party/highs_patch/
    apply_patch.cmake             # idempotent string_replace patches (~150 lines)
    HighsUserHeuristic.h          # injected into HiGHS src/mip/ — thin dispatch layer
    HighsUserHeuristic.cpp        # calls into our src/ implementations
  src/                            # OUR CODE — readable, self-contained algorithms
    fpr.h / fpr.cpp               # Fix-Propagate-Repair with WalkSAT
    local_mip.h / local_mip.cpp  # LocalMIP tabu search
    scylla_fpr.h / scylla_fpr.cpp # LP-guided FPR variant
    common.h                      # shared types/utilities (constraint iteration helpers)
  tests/
    test_basic.cpp
```

Target: HiGHS **v1.13.1** (same as cptp, proven patch compatibility).

**Design principle**: `src/` contains the algorithms in readable form. `third_party/highs_patch/` is the minimal glue. Someone reading `src/fpr.cpp` should understand the algorithm without reading HiGHS internals.

## Patch design (apply_patch.cmake)

Follows cptp pattern: idempotent `string(FIND ...)` + `string(REPLACE ...)`.

### Files patched (3 files, ~30 lines of diff)

| HiGHS file | Change | Purpose |
|---|---|---|
| `cmake/sources.cmake` | Add `HighsUserHeuristic.h/cpp` | Register injected files in build |
| `HighsMipSolverData.cpp` ~L2053 | Add `HighsUserHeuristic::runAtRoot(mipsolver)` after shifting+flush | Run FPR/LocalMIP at root node |
| `HighsMipSolver.cpp` ~L299 | Add `HighsUserHeuristic::runAtNode(mipsolver)` after RINS/RENS+flush | Run ScyllaFPR during B&B dives |

### HighsUserHeuristic (thin dispatch layer)

```cpp
class HighsUserHeuristic {
public:
    static void runAtRoot(HighsMipSolver& mipsolver);   // FPR, LocalMIP
    static void runAtNode(HighsMipSolver& mipsolver);   // ScyllaFPR (needs LP sol)
};
```

Guards: skip when `mipsolver.submip`.

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

## Heuristics — stepwise porting plan

### Step 1: FPR (Fix-Propagate-Repair)
- **Source**: `mip-heuristics-old/src/heuristic/fpr/fpr.cpp` (~400 LOC core)
- **Algorithm**: Rank variables by degree×objective → greedy fix → propagate bounds → WalkSAT repair
- **LP-free**: Yes. Runs at root before B&B starts.
- **Key differentiator vs HiGHS `mt/fix-and-propagate`**: WalkSAT repair phase. HiGHS branch gives up on infeasibility; we repair.
- **Why first**: Smallest, validates entire patch infrastructure.

### Step 2: LocalMIP
- **Source**: `mip-heuristics-old/src/heuristic/local_mip/local_mip.cpp` (~600 LOC core)
- **Algorithm**: Tabu search with breakthrough scoring, lift moves, weight smoothing, aspiration
- **LP-free**: Yes. Needs incumbent to start (naturally runs after FPR finds one).
- **Key differentiators vs HiGHS `local-mip` branch**: Weight decay, restarts, aspiration criterion, longer tabu tenure, lift moves (vs one-opt), BMS sampling.
- **Why second**: Strongest algorithm, but depends on incumbent from Step 1.

### Step 3: ScyllaFPR
- **Source**: `mip-heuristics-old/src/heuristic/scylla/scylla_fpr.cpp` (~100 LOC wrapper)
- **Algorithm**: FPR but ranks by LP fractionality instead of degree.
- **LP-dependent**: Yes. Runs during B&B dives when LP solution is available.
- **Why third**: Trivial once FPR exists (ranking-strategy swap). Demonstrates LP-guided variant.

### Future (not in first pass)
- FJ improvements (pool-based restarts, configurable stall detection) as small upstream PR
- Pseudocost diving — skip, redundant inside HiGHS B&B

## Build system

```cmake
cmake_minimum_required(VERSION 3.25)
project(mip-heuristics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

include(cmake/FetchHiGHS.cmake)

add_library(mip_heuristics_impl OBJECT
    src/fpr.cpp
    src/local_mip.cpp
    src/scylla_fpr.cpp
)
target_include_directories(mip_heuristics_impl PRIVATE ${highs_SOURCE_DIR}/highs src)

target_sources(highs PRIVATE $<TARGET_OBJECTS:mip_heuristics_impl>)
target_include_directories(highs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
```

Output: standard `highs` binary at `build/_deps/highs-build/bin/highs`.

## Implementation sequence

1. **Skeleton** — CMake, FetchHiGHS.cmake, empty HighsUserHeuristic (no-op). Verify `highs` builds and runs. Commit plan.md with skeleton.
2. **Patch wiring** — apply_patch.cmake with root + node dispatch hooks. Verify patch applies, `highs` still works.
3. **Port FPR** — Rewrite to use HiGHS arrays. Wire into `runAtRoot()`. Test on small MPS.
4. **Port LocalMIP** — Wire into `runAtRoot()` (after FPR). Test similarly.
5. **Port ScyllaFPR** — Wire into `runAtNode()`. Test with LP-guided ranking.
6. **Benchmark** — Patched HiGHS vs vanilla on MIPLIB subset.

## Verification

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
./build/_deps/highs-build/bin/highs test.mps
# verify heuristics fire (addIncumbent prints source on new incumbent)
```

## Key reference files

- `~/code/my/cptp/cmake/FetchHiGHS.cmake` — FetchContent pattern
- `~/code/my/cptp/third_party/highs_patch/apply_patch.cmake` — patch script pattern
- `~/code/my/cptp/build/_deps/highs-src/highs/mip/HighsMipSolver.cpp:260-306` — B&B dive dispatch
- `~/code/my/cptp/build/_deps/highs-src/highs/mip/HighsMipSolverData.cpp:2045-2055` — root dispatch
- `~/code/my/mip-heuristics-old/src/heuristic/fpr/fpr.cpp` — FPR to port
- `~/code/my/mip-heuristics-old/src/heuristic/local_mip/local_mip.cpp` — LocalMIP to port
- `~/code/my/mip-heuristics-old/src/heuristic/scylla/scylla_fpr.cpp` — ScyllaFPR to port
