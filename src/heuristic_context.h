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
#include <vector>

// Common execution scaffold for the four presolve heuristics (issue #94).
//
// Before this header, each of FJ / FPR / LocalMIP / Scylla re-derived the
// same ~10-line block of constants for itself (`ParallelSetup`: csc,
// num_workers, base_seed, worker_budget, default_run_cap, stale_budget),
// and the model sizes on top of that.  The three structs below carve that
// single struct along its actual seams — what the heuristic *searches*,
// what it may *spend*, and how it *runs* — so a heuristic's entry point can
// take exactly the parts it needs, and `mode_dispatch` can build the
// expensive part (the CSC transpose) once for the whole chain.
//
// `fpr_lp.cpp` takes only `make_exec` / `make_budget`: it runs on the same
// continuous parallel runner but keeps its own `LpFprSetup` for the LP
// references, reduced costs and shared `ContestedPdlp` that the presolve
// heuristics have no equivalent of.
//
// Ownership: solution submission is not here — it lives in `IncumbentSink`
// (incumbent_sink.h), owned by `mode_dispatch::run_sequential` and threaded
// through each heuristic's entry point.  Per-worker effort/staleness
// bookkeeping lives in
// `WorkerBudgetState` (worker_base.h); `HeuristicBudget`
// holds the *derived* values each worker's base struct receives on
// construction (the three heuristics that honour it — FJ's per-worker
// allowance comes from its own fixed total instead, see mode_dispatch.cpp).

// The uniform runner contract every presolve heuristic implements:
//
//     size_t <ns>::run(const ProblemView &problem, const HeuristicBudget &budget,
//                      ExecutionContext &exec, IncumbentSink &sink);
//
// `mode_dispatch::run_sequential` owns all four arguments — including the
// source tag the sink attributes this heuristic's solutions with — and books
// the returned effort through `EffortLedger`, the single point of effort
// accounting.  No heuristic self-books.  The per-heuristic headers describe
// only what their own runner does differently.
//
// `ExecutionContext` is passed by non-const reference to match the entry
// signature, but it is immutable for the duration of a dispatch: it is
// shared by every worker of every heuristic in the chain, so caching
// mutable per-dispatch state on it would be an unsynchronised shared
// write.  Both of its methods are `const`.

// Read-only view of the model a heuristic searches.  Every member but the
// incumbent snapshot is a non-owning pointer or a derived size; the pointees
// are owned by the caller that built it (the CSC — `run_sequential`'s local)
// or by the solver (model, mipdata), and outlive the whole heuristic chain.
// Built once per dispatch and passed by const reference — the snapshot makes
// it no longer trivially cheap to copy.
struct ProblemView {
    const HighsLp *model = nullptr;
    const HighsMipSolverData *mipdata = nullptr;
    const CscMatrix *csc = nullptr;

    // Derived sizes, previously recomputed at every call site that wanted
    // one of them.
    HighsInt ncol = 0;
    HighsInt nrow = 0;
    size_t nnz = 0;

    // Snapshot of `HighsMipSolverData::incumbent`, copied once per dispatch
    // on the dispatching thread (issue #98).  Workers read *this*, never
    // `mipdata->incumbent`: submission is immediate, so a peer worker's
    // accepted solution runs `addIncumbent`, whose whole-vector assignment
    // (`incumbent = sol;`) rewrites the live buffer under a concurrent
    // reader — element-wise while the sizes match, reallocating out from
    // under it on the empty-to-sized transition.  Empty when the solver had
    // no incumbent at dispatch time.  `fpr_lp` keeps the equivalent copy in
    // its own `LpFprSetup`.
    //
    // The snapshot is the *floor* a worker starts from, not necessarily what
    // it gets: both readers consult the shared pool first, which holds the
    // seeded incumbent plus everything accepted since (`IncumbentSink` seeds
    // it at construction and every accept goes through it, under its own
    // lock).  So dropping the live read costs neither of them a warm start —
    // LocalMIP's `resolve_worker_start` only reaches this copy with an empty
    // pool, i.e. when nothing has been submitted and the live incumbent
    // still equals it, and `fj.cpp` resolves pool-first for the same reason.
    // Nothing may read this *instead* of the pool: an `FjWorker` rebuilt
    // mid-dispatch would then silently lose a peer's find.
    std::vector<double> incumbent;

    // Per-column snapshot of `HighsDomain::isBinary`, taken with the
    // incumbent and for the same reason (issue #99).  `addIncumbent` runs
    // `getDomain().propagate()` and `redcostfixing.propagateRootRedcost`,
    // both of which tighten the root domain's bound vectors element-wise,
    // while workers classify columns from those same vectors.  Unlike the
    // incumbent this can never dangle — the bound vectors are sized once at
    // setup — so it is a torn read rather than a use-after-free, but it is
    // still a race, and a column's classification flipping mid-dispatch is
    // not something any worker is written to expect.
    //
    // `uint8_t` rather than `std::vector<bool>`: workers index this from
    // hot loops, and the bit-packed specialisation's proxy references are
    // both slower and not safely readable from several threads at once.
    std::vector<uint8_t> binary;

    // A model with no columns or no rows: every heuristic declines it.
    bool degenerate() const { return ncol == 0 || nrow == 0; }
};

// Snapshot `HighsDomain::isBinary` for every column.  Must run on the
// dispatching thread, before any parallel region — see `ProblemView::binary`.
inline std::vector<uint8_t> build_binary_mask(const HighsMipSolver &mipsolver) {
    const HighsInt ncol = mipsolver.model_->num_col_;
    const HighsDomain &domain = mipsolver.mipdata_->getDomain();
    std::vector<uint8_t> mask(static_cast<size_t>(ncol), 0);
    for (HighsInt j = 0; j < ncol; ++j) {
        mask[j] = domain.isBinary(j) ? 1 : 0;
    }
    return mask;
}

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

    // Deterministic seed for worker `w`.  The runner seeds its own per-worker
    // `Rng` with this, and heuristics that pre-construct their workers seed
    // them with it too — three hand-written copies of the expression before
    // it lived here, which is three chances for one of them to drift.
    uint32_t worker_seed(int w) const {
        return base_seed + static_cast<uint32_t>(w) * kSeedStride;
    }
};

// Derive one dispatch's execution parameters.  Shared by `run_sequential`
// and by `fpr_lp`, which runs on the same continuous parallel runner from a
// setup of its own shape.
inline ExecutionContext make_exec(HighsMipSolver &mipsolver) {
    return ExecutionContext{mipsolver,
                            static_cast<size_t>(std::max(1, highs::parallel::num_threads())),
                            heuristic_base_seed(mipsolver.options_mip_->random_seed),
                            mipsolver.options_mip_->time_limit};
}

// Split a heuristic's slice of the effort envelope into the per-worker,
// per-attempt and staleness ceilings its workers and the runner use.
inline HeuristicBudget make_budget(size_t total, size_t num_workers) {
    return HeuristicBudget{total, total / num_workers,
                           std::max<size_t>(total / (num_workers * 10), 1), total >> 2};
}

// Build the CSC transpose into caller-owned `csc` and return a view over it
// together with the model pointers, derived sizes and the incumbent
// snapshot.
//
// `csc` must outlive every use of the returned view.  One call covers a
// whole FJ -> FPR -> LocalMIP -> Scylla chain: the row-major buffers the
// transpose is built from are written by `HighsMipSolverData::runSetup()`
// before any heuristic dispatch and are not touched again while the chain
// runs, so a single snapshot is valid for all four.  (Each heuristic used
// to build its own identical copy.)  Must be called on the dispatching
// thread, before any parallel region — see `ProblemView::incumbent`.
inline ProblemView make_problem(HighsMipSolver &mipsolver, CscMatrix &csc) {
    const HighsLp *model = mipsolver.model_;
    HighsMipSolverData *mipdata = mipsolver.mipdata_.get();
    csc = build_csc(model->num_col_, model->num_row_, mipdata->ARstart_, mipdata->ARindex_,
                    mipdata->ARvalue_);
    return ProblemView{model,
                       mipdata,
                       &csc,
                       model->num_col_,
                       model->num_row_,
                       mipdata->ARindex_.size(),
                       mipdata->incumbent,
                       build_binary_mask(mipsolver)};
}
