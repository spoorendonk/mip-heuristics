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
//   - `epoch_budget` is a *method*, not a field: each caller passes the
//     epochs-per-worker constant it wants (FJ uses `kEpochsPerWorkerFj`,
//     the rest use `kEpochsPerWorker`).  Keeps the derivation centralised
//     while allowing the FJ/other-heuristics divergence to stay explicit
//     at each call site.  See the constant docstrings below.
struct ParallelSetup {
    const HighsLp &model;
    HighsMipSolverData *mipdata;
    CscMatrix csc;
    size_t N;                // num_threads
    uint32_t base_seed;      // seeded from `random_seed` via heuristic_base_seed
    size_t worker_budget;    // max_effort / N (floor division)
    size_t default_run_cap;  // max_effort / (N * 10) (min 1) — opportunistic attempt cap
    size_t stale_budget;     // max_effort / 4 — generic staleness ceiling

    ParallelSetup(HighsMipSolver &mipsolver, size_t max_effort);

    // Deterministic per-epoch effort slice for a given epochs-per-worker
    // cadence: worker_budget / epochs, floored at 1.  Callers pass the
    // heuristic-specific constant (see below) rather than reading a field,
    // because FJ and the other three heuristics disagree on the cadence.
    [[nodiscard]] size_t epoch_budget(size_t epochs) const {
        return std::max<size_t>(worker_budget / epochs, 1);
    }
};

// Shared tuning constant for FPR, LocalMIP, and Scylla: each worker in the
// deterministic epoch loop takes ~`kEpochsPerWorker` turns inside the total
// budget; smaller values synchronize more often (finer improvement
// broadcast) at the cost of more per-epoch overhead.
inline constexpr size_t kEpochsPerWorker = 10;

// FJ uses 20 epochs per worker historically.  Halving it to the unified
// 10 in the Wave-6 refactor was unvalidated — FJ's synchronization
// cadence matters for pool-crossover behaviour and a change could regress
// on FJ-dominant instances.  Kept at 20 until a formal MIPLIB benchmark
// validates a unified value; see issue #71 for effort-unit normalisation.
inline constexpr size_t kEpochsPerWorkerFj = 20;

inline ParallelSetup::ParallelSetup(HighsMipSolver &mipsolver, size_t max_effort)
    : model(*mipsolver.model_),
      mipdata(mipsolver.mipdata_.get()),
      csc(build_csc(model.num_col_, model.num_row_, mipdata->ARstart_, mipdata->ARindex_,
                    mipdata->ARvalue_)),
      N(static_cast<size_t>(std::max(1, highs::parallel::num_threads()))),
      base_seed(heuristic_base_seed(mipsolver.options_mip_->random_seed)),
      worker_budget(max_effort / N),
      default_run_cap(std::max<size_t>(max_effort / (N * 10), 1)),
      stale_budget(max_effort >> 2) {}
