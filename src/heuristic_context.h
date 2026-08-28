#pragma once

#include "deadline.h"
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
// construction.  All four heuristics' budgets come from their own
// `mip_heuristic_<name>_effort` option; FJ's sizes `per_worker` and lets
// `total` scale with the pool, the other three size `total` and are
// divided across it (see mode_dispatch.cpp).

// The uniform runner contract every presolve heuristic implements:
//
//     DispatchOutcome <ns>::run(const ProblemView &problem, const HeuristicBudget &budget,
//                               ExecutionContext &exec, IncumbentSink &sink);
//
// `mode_dispatch::run_sequential` owns all four arguments — including the
// source tag the sink attributes this heuristic's solutions with — and books
// the returned outcome through `EffortLedger`, the single point of effort
// accounting.  No heuristic self-books.  The per-heuristic headers describe
// only what their own runner does differently.
//
// `ExecutionContext` is passed by non-const reference to match the entry
// signature, but it is immutable for the duration of a dispatch: it is
// shared by every worker of every heuristic in the chain, so caching
// mutable per-dispatch state on it would be an unsynchronised shared
// write.  Both of its methods are `const`.

// What one dispatch of a heuristic did, as its runner reports it (issue
// #119).  Used to be a bare `size_t` effort count.
//
// The second field exists because zero effort has two causes that a log
// cannot tell apart.  A heuristic that searched and produced nothing and
// one that never searched at all — because #117 made the *sequential*
// setup abandon its work on an already-passed deadline — both booked
// `effort=0 found=0`, and the #113 calibration bins the second as
// "barren", which is the population its patience estimate rests on.  A
// setup bail is not a barren dispatch: it is a cost of setup, and it
// happens precisely on the large, hard instances the calibration is most
// sensitive to.
//
// Why the *return type* rather than a narrower channel.  The flag has to
// travel from a bail site deep inside one heuristic to `run_sequential`,
// which is the only caller and the only booker.  The two alternatives were
// a mutable field on `ExecutionContext` — narrower in signatures, but that
// object is shared by every worker of every heuristic in the chain and is
// documented immutable for exactly that reason, and it would need a reset
// per heuristic that nothing forces anyone to write — and a flag on
// `IncumbentSink`, which is the per-heuristic mutable channel
// `run_and_charge` already reads across the call but has nothing to do
// with solution submission.  Both smuggle per-dispatch state into an
// object that outlives the dispatch; a return value cannot leak into the
// next heuristic because there is no state to forget to clear.
//
// `[[nodiscard]]`, for the reason `IncumbentSink::offer` is: a dropped
// outcome loses both the effort — which `run_sequential` books into
// `heuristic_effort_used` and nothing else can recover — and the bail flag
// this issue exists to carry.  `-Werror=unused-result` on `mip_heuristics`
// and `mip_heuristics_tests` is what makes the attribute a build failure
// rather than a warning that scrolls past.  It costs a caller that wants
// one field nothing: `run(...).effort` is a *use* of the return value, so
// only a call whose whole result is thrown away trips it, and the single
// such caller (a test driving warm-start counters) spells the discard.
struct [[nodiscard]] DispatchOutcome {
    // Effort charged, in this heuristic's own unit.  `run_sequential`
    // books exactly this into `heuristic_effort_used`.
    size_t effort = 0;

    // This dispatch abandoned its sequential setup on the wall-clock
    // deadline and never searched (issue #117's bail, made visible).  It
    // implies `effort == 0`, but the converse is what this field exists to
    // deny.
    //
    // Scope: *only* the deadline bails in `fpr::precompute_var_orders` and
    // `scylla::precompute_config_var_orders` set it, plus the dive-time
    // equivalent.  A heuristic declining for another reason — a degenerate
    // model, a zero budget, a `ContestedPdlp` that failed to initialise —
    // reports a plain zero, because none of those is a dispatch the clock
    // cut short.
    bool abandoned_setup = false;

    // The bail, spelled once so the two sites cannot disagree.
    static DispatchOutcome abandoned() { return {.effort = 0, .abandoned_setup = true}; }
};

// Read-only view of the model a heuristic searches.  Every member but the
// incumbent snapshot is a non-owning pointer or a derived size; the pointees
// are owned by the caller that built it (the CSC — `run_sequential`'s local)
// or by the solver (model, mipdata), and outlive the whole heuristic chain.
// Built once per dispatch and passed by const reference — the snapshot makes
// it no longer trivially cheap to copy.
struct ProblemView {
    const HighsLp* model = nullptr;
    const HighsMipSolverData* mipdata = nullptr;
    const CscMatrix* csc = nullptr;

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
    // Scope: taken once for the *whole* FJ -> FPR -> LocalMIP -> Scylla
    // chain, so a column that root propagation fixes after FJ's first
    // incumbent is still classified with its pre-FJ value by the other
    // three.  Deliberate, and cheap: workers enforce bounds from
    // `model->col_lower_/col_upper_`, never from `HighsDomain`, so the
    // classification was already decoupled from the bounds they respect.
    //
    // `uint8_t` rather than `std::vector<bool>`: workers index this from
    // hot loops, and the bit-packed specialisation costs a shift and mask
    // per read plus a proxy object.  Concurrent *reads* of a frozen
    // `vector<bool>` would be well-defined — the hazard is concurrent
    // read/write of neighbouring bits, which cannot arise for a mask built
    // before the parallel region — so this is a throughput choice, not a
    // correctness one.
    std::vector<uint8_t> binary;

    // A model with no columns or no rows: every heuristic declines it.
    [[nodiscard]] bool degenerate() const { return ncol == 0 || nrow == 0; }
};

// Snapshot `HighsDomain::isBinary` for every column.  Must run on the
// dispatching thread, before any parallel region — see `ProblemView::binary`.
inline std::vector<uint8_t> build_binary_mask(const HighsMipSolver& mipsolver) {
    const HighsInt ncol = mipsolver.model_->num_col_;
    const HighsDomain& domain = mipsolver.mipdata_->getDomain();
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

    // Runner-level staleness ceiling: the dispatch stops once
    // `ContinuousLoopState::effort_since_improvement` — summed over every
    // worker — crosses this.  Absolute and instance-scaled since issue
    // #111 (`patience_threshold(nnz, <this heuristic's patience option>,
    // total)`, sized by the caller from `mip_heuristic_<name>_patience`),
    // not the `total / 4` it used to be.  A quarter of the budget is not a
    // patience criterion: it says "I have spent a quarter of what I was
    // given", which is true at every budget and therefore bounds nothing.
    // A quarter of the budget is what the *clamp* is (#116), which is a
    // different job — see `kPatienceCeilingDivisor`.
    size_t stale = 0;

    // Per-worker share of the same ceiling.  The runner's counter
    // aggregates N workers, so one worker's share is `stale / N`; that
    // relation is not new, it is what `total / 4` per worker against
    // `total / 4` per dispatch already meant, and what FJ's `nnz << 8`
    // against a `N * nnz << 8` runner gate already was at the shipped
    // default.  Scylla is the documented exception — see scylla_worker.cpp.
    size_t worker_stale = 0;

    // This heuristic was handed nothing, and must therefore do nothing
    // (issue #106).
    //
    // `mip_heuristic_<name>_effort = 0` sizes `total` to zero, and #107
    // spells "this heuristic is excluded from the configuration" exactly
    // that way — a zero-pattern of four continuous parameters rather than a
    // separate discrete subset dimension, which is what keeps its search
    // tractable.  That reduction needs a zero budget to be worth exactly
    // what omitting the heuristic is worth, so every entry point checks
    // this alongside `ProblemView::degenerate()` and returns before any
    // setup.
    //
    // For the *presolve chain* that makes the two indistinguishable.  It
    // does not extend to the whole solve for `fpr`: omitting `fpr` from
    // `mip_heuristic_suite` also disables the dive-time `fpr_lp`, through
    // `heuristics::effective_flags`, while a zero presolve effort does not
    // — `fpr_lp` draws from upstream's `mip_heuristic_effort` envelope and
    // never reads this option.  That is a property of the two option
    // surfaces, not something this check could repair; #107's target runner
    // derives the suite value from the zero-pattern, which is where they
    // are reconciled.  `run_opportunistic_loop` already declined a zero total,
    // but three of the four heuristics do real work before they reach it —
    // Scylla builds a `ContestedPdlp` (a whole `Highs` LP copy), the
    // per-config variable orders and N workers; FPR precomputes its
    // variable orders — and that work is neither free nor charged, so it
    // was invisible in the effort total while still costing wall time.
    [[nodiscard]] bool disabled() const { return total == 0; }
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
    HighsMipSolver& mipsolver;
    size_t num_workers;  // highs::parallel::num_threads(), at least 1
    uint32_t base_seed;  // seeded from `random_seed` via heuristic_base_seed
    double time_limit;

    // The wall-clock half of "should we stop?", and the only half a worker
    // thread may call (issue #114).
    //
    // `HighsTimer::read()` is `const` and, for the solve clock, writes
    // nothing — it reads `clock_start`/`clock_time` and calls
    // `getWallTime()`.  Nothing starts or stops that clock while a
    // heuristic dispatch is in flight, so concurrent readers are safe.
    // That is what separates this from `terminated()` below, and it is why
    // every presolve heuristic can poll the deadline from inside its own
    // inner loop without a poller seat.
    //
    // Deliberately the solver's own clock rather than a `steady_clock`
    // snapshot: a second origin would no longer agree with the `[Heur]
    // start_s`/`end_s` the ledger emits, which the tests and
    // `bench/parse_highs_log.py` both read against this same limit.
    // `HighsTimer` bottoms out in `high_resolution_clock` and is therefore
    // not monotonic (see `effort_ledger.h`); that risk is pre-existing and
    // is the price of one shared origin.
    [[nodiscard]] bool past_deadline() const { return deadline().expired(); }

    // The same deadline as a standalone pollable value, for the layers
    // below a heuristic's runner: `fpr_core`'s DFS and the repair search
    // under it are shared by three callers and have no `ExecutionContext`
    // (issue #117).  One definition rather than two — a sub-algorithm
    // that stopped against a different clock or a different limit than
    // its runner would be the drift this method exists to prevent.
    [[nodiscard]] Deadline deadline() const { return make_deadline(mipsolver.timer_, time_limit); }

    // The full "should we stop?" predicate.  Three hand-rolled copies of
    // it existed before this struct.
    //
    // Not thread-safe for concurrent callers: `terminatorTerminated()`
    // writes `mipsolver.termination_status_` when a terminator is attached.
    // That write — not the clock read — is the whole reason this one needs
    // a single caller.  `mode_dispatch` calls it between heuristics, with
    // every parallel region already joined, and inside a parallel region
    // the worker holding `ContinuousLoopState`'s claimable poller seat
    // calls it on everyone's behalf.
    //
    // There is no longer an exception: FPR's multi-attempt inner loop used
    // to poll this directly from its own worker thread, which was a race
    // whenever a terminator was attached.  It polls `past_deadline()`
    // instead (issue #114), which is the half it actually needed, so the
    // seat is now the only route to the terminator.
    [[nodiscard]] bool terminated() const {
        return mipsolver.mipdata_->terminatorTerminated() || past_deadline();
    }

    // Deterministic seed for worker `w`.  The runner seeds its own per-worker
    // `Rng` with this, and heuristics that pre-construct their workers seed
    // them with it too — three hand-written copies of the expression before
    // it lived here, which is three chances for one of them to drift.
    [[nodiscard]] uint32_t worker_seed(int w) const {
        return base_seed + (static_cast<uint32_t>(w) * kSeedStride);
    }
};

// Derive one dispatch's execution parameters.  Shared by `run_sequential`
// and by `fpr_lp`, which runs on the same continuous parallel runner from a
// setup of its own shape.
inline ExecutionContext make_exec(HighsMipSolver& mipsolver) {
    return ExecutionContext{mipsolver,
                            static_cast<size_t>(std::max(1, highs::parallel::num_threads())),
                            heuristic_base_seed(mipsolver.options_mip_->random_seed),
                            mipsolver.options_mip_->time_limit};
}

// The dispatch's deadline, for a callee that was handed the solver but not
// the `ExecutionContext` built from it — `fpr_core`'s attempt lifecycle, on
// behalf of all three of its callers (issue #117).  Reads the same option
// `make_exec` copies into `ExecutionContext::time_limit`, so deriving it
// here rather than threading the context through cannot drift: there is one
// option and one clock.
inline Deadline deadline_of(const HighsMipSolver& mipsolver) {
    return make_deadline(mipsolver.timer_, mipsolver.options_mip_->time_limit);
}

// Split a heuristic's slice of the effort envelope into the per-worker,
// per-attempt and staleness ceilings its workers and the runner use.
//
// `stale` is passed in rather than derived here (issue #111): it is the
// one quantity in this struct that must *not* be a function of `total`,
// and every caller knows the model's `nnz` and its heuristic's
// patience value.  See `patience_threshold` in heuristic_common.h.
inline HeuristicBudget make_budget(size_t total, size_t num_workers, size_t stale) {
    // A zero total is "this heuristic is excluded" (issue #106), so every
    // derived ceiling is zero too and `disabled()` holds.  Spelling it out
    // rather than letting the expressions below run: `attempt_cap` floors
    // at 1, so a zero budget used to license one attempt — the whole of
    // Scylla's, which charges a full PDLP solve and does not stop for
    // `attempt_cap` once started — and `stale` arrives here *unclamped*,
    // because `patience_threshold` special-cases a zero budget by skipping
    // the clamp.  Both are ceilings that only make sense above a budget
    // that exists.
    if (total == 0) {
        return HeuristicBudget{};
    }
    // Designated initialisers for the same reason `make_problem` below
    // gives them: this aggregate is five `size_t` members in a row, so a
    // mis-ordered addition converts silently between them.  #111 appended
    // the fifth.
    return HeuristicBudget{.total = total,
                           .per_worker = total / num_workers,
                           .attempt_cap = std::max<size_t>(total / (num_workers * 10), 1),
                           .stale = stale,
                           .worker_stale = std::max<size_t>(stale / num_workers, 1)};
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
inline ProblemView make_problem(HighsMipSolver& mipsolver, CscMatrix& csc) {
    const HighsLp* model = mipsolver.model_;
    HighsMipSolverData* mipdata = mipsolver.mipdata_.get();
    csc = build_csc(model->num_col_, model->num_row_, mipdata->ARstart_, mipdata->ARindex_,
                    mipdata->ARvalue_);
    // Designated initialisers: two snapshots have been appended to this
    // aggregate in as many issues, and three of the members in the middle
    // are a positional `HighsInt, HighsInt, size_t` run that a mis-ordered
    // addition would silently convert between.
    return ProblemView{.model = model,
                       .mipdata = mipdata,
                       .csc = &csc,
                       .ncol = model->num_col_,
                       .nrow = model->num_row_,
                       .nnz = mipdata->ARindex_.size(),
                       .incumbent = mipdata->incumbent,
                       .binary = build_binary_mask(mipsolver)};
}
