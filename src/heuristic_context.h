#pragma once

#include "heuristic_common.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Common execution scaffold for the four presolve heuristics (issue #94).
//
// Before this header, each of FJ / FPR / LocalMIP / Scylla re-derived the
// same ~10-line block of constants for itself (`ParallelSetup`: csc,
// num_workers, base_seed, worker_budget, default_run_cap, stale_budget),
// and the model sizes on top of that.  The three structs below carve that
// single struct along its actual seams — what the heuristic *searches*,
// what it may *spend*, and how it *runs* — so a heuristic's entry point can
// take exactly the parts it needs, and `mode_dispatch` can build the
// expensive parts once for the whole chain.
//
// Not consumed by `fpr_lp.cpp`: it has an `LpFprSetup` that owns LP
// references, a reduced-cost vector and a shared `ContestedPdlp`.  Shape
// does not match.
//
// Ownership: the `SolutionPool` is not here — it is owned by
// `mode_dispatch::run_sequential` and threaded through each heuristic's
// entry point.  Per-worker effort/staleness bookkeeping lives in
// `WorkerBudgetState` (worker_base.h); `HeuristicBudget`
// holds the *derived* values each worker's base struct receives on
// construction (the three heuristics that honour it — FJ's per-worker
// allowance comes from its own fixed total instead, see mode_dispatch.cpp).

// Read-only view of the model a heuristic searches.  Cheap to copy: every
// member is a non-owning pointer or a derived size.  The pointees are owned
// by the `HeuristicContext` this view was carved from (the CSC) or by the
// solver (model, mipdata), and outlive the whole heuristic chain.
struct ProblemView {
    const HighsLp *model = nullptr;
    const HighsMipSolverData *mipdata = nullptr;
    const CscMatrix *csc = nullptr;

    // Derived sizes, previously recomputed at every call site that wanted
    // one of them.
    HighsInt ncol = 0;
    HighsInt nrow = 0;
    size_t nnz = 0;

    // A model with no columns or no rows: every heuristic declines it.
    bool degenerate() const { return ncol == 0 || nrow == 0; }
};

// One heuristic's slice of the presolve effort envelope.  `total` used to
// travel separately as a bare `max_effort` parameter while the other three
// were fields of `ParallelSetup`; they are one thing and now travel as one.
struct HeuristicBudget {
    size_t total = 0;        // whole-dispatch ceiling, summed over workers
    size_t per_worker = 0;   // total / N (floor division)
    size_t attempt_cap = 0;  // per-attempt cap: max(total / (N * 10), 1)
    size_t stale = 0;        // total / 4 — generic staleness ceiling
};

// How a heuristic runs: the worker count, the RNG base seed, and the one
// termination predicate.
//
// Historical note on `attempt_cap`: the deleted epoch-gated runner gave FJ a
// separate cadence (`kEpochsPerWorkerFj = 20` against 10 for the rest), on
// the grounds that "FJ's synchronization cadence matters for pool-crossover
// behaviour and a change could regress on FJ-dominant instances".  That only
// ever applied to the epoch runner — the continuous runner has always used
// this cap for FJ too — so #92 removed a constant, not a behaviour.  The
// concern was never benchmarked; recorded here so it is not lost with the
// constant, but no cadence changed for any surviving execution path.
struct ExecutionContext {
    HighsMipSolver &mipsolver;
    size_t num_workers;   // highs::parallel::num_threads(), at least 1
    uint32_t base_seed;   // seeded from `random_seed` via heuristic_base_seed
    double time_limit;

    // The single "should we stop?" predicate.  Three hand-rolled copies of
    // it existed before this struct.
    //
    // Not thread-safe for concurrent callers: `terminatorTerminated()`
    // writes `mipsolver.termination_status_` when a terminator is attached.
    // `mode_dispatch` calls this between heuristics, with every parallel
    // region already joined, and inside a parallel region the worker
    // holding `ContinuousLoopState`'s claimable poller seat calls it on
    // everyone's behalf.  One pre-existing exception: FPR's multi-attempt
    // inner loop (`FprWorker::run_attempt`) polls it directly from its own
    // worker thread so a 32-attempt fill cannot outrun a solver timeout.
    // That is only a race when a terminator is attached — the write above
    // is skipped otherwise — and predates this struct; folding the three
    // hand-rolled copies into one method is what makes it visible.
    bool terminated() const {
        return mipsolver.mipdata_->terminatorTerminated() ||
               mipsolver.timer_.read() >= time_limit;
    }
};

// Owns the per-dispatch derived state (the CSC transpose) that `ProblemView`
// points at, and hands out the three views above.
//
// One instance covers a whole FJ -> FPR -> LocalMIP -> Scylla chain: the
// row-major buffers the CSC is built from are written by
// `HighsMipSolverData::runSetup()` before any heuristic dispatch and are not
// touched again while the chain runs, so a single snapshot is valid for all
// four.  (Each heuristic used to build its own identical copy.)
class HeuristicContext {
public:
    explicit HeuristicContext(HighsMipSolver &mipsolver);

    // Non-copyable: `problem_.csc` points into `csc_`.
    HeuristicContext(const HeuristicContext &) = delete;
    HeuristicContext &operator=(const HeuristicContext &) = delete;

    const ProblemView &problem() const { return problem_; }
    ExecutionContext &exec() { return exec_; }

    // Derive one heuristic's budget from its share of the envelope.
    HeuristicBudget budget(size_t total) const {
        const size_t n = exec_.num_workers;
        return HeuristicBudget{total, total / n, std::max<size_t>(total / (n * 10), 1),
                               total >> 2};
    }

private:
    CscMatrix csc_;
    ProblemView problem_;
    ExecutionContext exec_;
};

inline HeuristicContext::HeuristicContext(HighsMipSolver &mipsolver)
    : csc_(build_csc(mipsolver.model_->num_col_, mipsolver.model_->num_row_,
                     mipsolver.mipdata_->ARstart_, mipsolver.mipdata_->ARindex_,
                     mipsolver.mipdata_->ARvalue_)),
      problem_{mipsolver.model_, mipsolver.mipdata_.get(), &csc_, mipsolver.model_->num_col_,
               mipsolver.model_->num_row_, mipsolver.mipdata_->ARindex_.size()},
      exec_{mipsolver, static_cast<size_t>(std::max(1, highs::parallel::num_threads())),
            heuristic_base_seed(mipsolver.options_mip_->random_seed),
            mipsolver.options_mip_->time_limit} {}
