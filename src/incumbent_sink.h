#pragma once

#include "solution_sink.h"
#include "worker_base.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

class HighsMipSolver;

// The `SolutionSink` a presolve chain or an `fpr_lp` dive hands its
// workers: every solution the pool accepts is traced, and submitted to
// HiGHS at once unless HiGHS could do nothing with it (`reaches_highs`).
//
// Owns the mutex that serialises HiGHS's non-thread-safe `trySolution`
// and the `[HeurSol]` dispatch context; the pool, the tag and the offer
// verdicts are `SolutionSink`'s (#170).  Before this class the pool +
// mutex + accept wiring was written out twice, verbatim, in
// `mode_dispatch.cpp` and `fpr_lp.cpp`, and every worker hard-coded the
// source constant of its own heuristic at its `try_add` call.
//
// Submission is immediate rather than batched: `on_accept` runs as soon as
// the pool takes a solution, so a HiGHS incumbent timestamp reflects find
// time rather than end-of-dispatch flush time.
class IncumbentSink final : public SolutionSink {
public:
    // Constructs the pool, seeds it from the current incumbent, and opens
    // a dispatch.  `source` tags everything offered until `set_source`
    // says otherwise.
    IncumbentSink(HighsMipSolver& mipsolver, int source);

    // Retarget the attribution tag for subsequent offers.  Legal only
    // between heuristics, on the dispatching thread, with every parallel
    // region joined — `mode_dispatch::run_sequential` is the sole caller,
    // and that is the same invariant which lets it book effort without
    // synchronisation.
    //
    // This is also *the* dispatch boundary, and `[HeurSol]` uses it as one
    // (#106): a retarget happens exactly once per presolve-chain dispatch,
    // immediately before it, and the only other way an offer can reach a
    // new source tag is a freshly constructed sink — which is what `fpr_lp`
    // does, one per dive dispatch.  So construction and retarget together
    // enumerate every dispatch, and both take the next `dispatch` id.
    void set_source(int source) { begin_dispatch(source); }

    // Trace id of the dispatch currently being attributed.  Drawn from a
    // process-global counter, so `(name, dispatch)` identifies one dispatch
    // uniquely across the whole process — not merely within one solve, which
    // a per-sink counter could not manage: `fpr_lp` builds a new sink per
    // dive, and a per-sink counter would hand every dive the same id.
    [[nodiscard]] uint64_t dispatch_id() const { return dispatch_id_; }

private:
    // Take the next process-global dispatch id, remember the heuristic name
    // the tag maps to, and stamp the dispatch's start on the solver clock.
    void begin_dispatch(int source);

    // Submit to HiGHS if it could change anything there, then emit the
    // `[HeurSol]` line.
    void on_accept(double objective, const std::vector<double>& solution, int source,
                   const WorkerTrace& trace, size_t effort_at) override;

    // Whether `trySolution` could act on `solution` (#243).  Under a flood
    // most accepted offers enter the pool's top-K without beating HiGHS's
    // incumbent, and for those `trySolution` is an O(nnz) feasibility pass,
    // under `highs_mtx_`, that ends in an `addIncumbent` doing nothing: it
    // stores only `solobj < upper_bound`, and otherwise acts only to run
    // the MIP solution callback or to fill an empty incumbent.  So the call
    // is skipped exactly when neither can happen and the objective, computed
    // as `trySolution` computes it, is not below `highs_bound_`.  Which
    // solutions reach HiGHS's incumbent does not change: `highs_bound_` is
    // never below HiGHS's bound, only stale above it, and a stale value just
    // lets a call through.
    //
    // Not the pool's `improved_best`: that clears a margin, against a
    // watermark in the worker's objective space, so an offer improving
    // HiGHS's incumbent by less than the margin would be lost.
    [[nodiscard]] bool reaches_highs(const std::vector<double>& solution) const;

    // Copy HiGHS's bound into `highs_bound_`: on the dispatching thread with
    // every parallel region joined, or under `highs_mtx_`, the two places
    // `upper_bound` may be read.
    void sync_highs_bound();

    // Emit one `[HeurSol]` line.  `const` and lock-free by construction —
    // see the definition for the threading argument.
    void trace_offer(const WorkerTrace& trace, size_t effort_at, double objective) const;

    HighsMipSolver& mipsolver_;
    // Serialises `trySolution`: `HighsMipSolverData::addIncumbent` is not
    // thread-safe and `on_accept` runs on whichever worker thread produced
    // the solution.
    std::mutex highs_mtx_;
    // HiGHS's `upper_bound` as of the last `sync_highs_bound`, or +inf while
    // HiGHS has no incumbent.  Written only under `highs_mtx_` or at a
    // dispatch boundary; read lock-free by `reaches_highs`.  `upper_bound`
    // only falls between syncs, so a relaxed read is at worst too high.
    std::atomic<double> highs_bound_{std::numeric_limits<double>::infinity()};
    // The MIP solution callback sees every `trySolution` that is feasible,
    // improving or not, so none may be skipped.  Set with the bound.
    bool callback_sees_all_ = true;

    // `[HeurSol]` dispatch context.  Written only by `begin_dispatch`, i.e.
    // at construction and at `set_source`, both of which run on the
    // dispatching thread with every parallel region joined — the same
    // invariant `set_source` and `EffortLedger` already rely on.  Workers
    // only ever read them, so no synchronisation is needed and none is
    // added: a log mutex here would serialise every offer a second time,
    // on top of the pool's own lock.
    const char* dispatch_name_ = "unknown";
    uint64_t dispatch_id_ = 0;
    double dispatch_start_s_ = 0.0;
};
