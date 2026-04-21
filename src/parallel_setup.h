#pragma once

#include "heuristic_common.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Shared scaffold for every heuristic's `run_parallel_*` entry point.
//
// Before this helper, each of FJ / FPR / LocalMIP / Scylla re-derived
// the same ~10-line block of constants (csc, num_workers, base_seed,
// worker_budget, epoch_budget, default_run_cap, stale_budget).  The
// duplicated boilerplate was a drift risk: unifying it in one place is
// a pure refactor — per-field semantics are unchanged.
//
// Not consumed by:
//   - `portfolio.cpp` — has a `PresolveSetup` that owns additional
//     bandit arms, priors, incumbent snapshot, and an optional
//     `ContestedPdlp` for the Scylla arm.  Shape does not match.
//   - `fpr_lp.cpp` — has an `LpFprSetup` that owns LP references,
//     a reduced-cost vector, and a shared `ContestedPdlp`.  Shape
//     does not match either.
//
// Ownership notes (post-Wave-4 #72 + Wave-5 #69):
//   - The `SolutionPool` is NOT owned here.  Pool ownership lives in
//     `mode_dispatch::run_sequential` for the sequential-mode path and
//     is passed through each heuristic's `run_parallel(..., SolutionPool&)`.
//   - `total_effort`, `effort_since_improvement`, `finished` live in
//     `EpochWorkerBase` (in `epoch_runner.h`) for each worker that needs
//     them.  `ParallelSetup::stale_budget` is the *derived* value each
//     worker's base struct receives on construction (for the three
//     heuristics that honour it — FJ overrides internally).
struct ParallelSetup {
    const HighsLp &model;
    HighsMipSolverData *mipdata;
    CscMatrix csc;
    size_t N;                // num_threads
    uint32_t base_seed;      // seeded from `random_seed` via heuristic_base_seed
    size_t worker_budget;    // max_effort / N (floor division)
    size_t epoch_budget;     // worker_budget / kEpochsPerWorker (min 1)
    size_t default_run_cap;  // max_effort / (N * 10) (min 1) — opportunistic attempt cap
    size_t stale_budget;     // max_effort / 4 — generic staleness ceiling

    ParallelSetup(HighsMipSolver &mipsolver, size_t max_effort);
};

// Shared tuning constant.  Each worker in the deterministic epoch loop
// takes ~`kEpochsPerWorker` turns inside the total budget; smaller values
// synchronize more often (finer improvement broadcast) at the cost of
// more per-epoch overhead.  Historically FJ used 20 while FPR, LocalMIP,
// and Scylla used 10; the divergence was undocumented drift and this
// value is standardised here.  Pending a formal MIPLIB benchmark (see
// issue #71 for effort-unit normalisation), we keep the majority value.
inline constexpr size_t kEpochsPerWorker = 10;

inline ParallelSetup::ParallelSetup(HighsMipSolver &mipsolver, size_t max_effort)
    : model(*mipsolver.model_),
      mipdata(mipsolver.mipdata_.get()),
      csc(build_csc(model.num_col_, model.num_row_, mipdata->ARstart_, mipdata->ARindex_,
                    mipdata->ARvalue_)),
      N(static_cast<size_t>(std::max(1, highs::parallel::num_threads()))),
      base_seed(heuristic_base_seed(mipsolver.options_mip_->random_seed)),
      worker_budget(max_effort / N),
      epoch_budget(std::max<size_t>(worker_budget / kEpochsPerWorker, 1)),
      default_run_cap(std::max<size_t>(max_effort / (N * 10), 1)),
      stale_budget(max_effort >> 2) {}
