#pragma once

#include "solution_sink.h"
#include "worker_base.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

class HighsMipSolver;

// The `SolutionSink` a presolve chain or an `fpr_lp` dive hands its
// workers: every solution the pool accepts is submitted to HiGHS at once,
// and traced.
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

    // Submit to HiGHS, then emit the `[HeurSol]` line.
    void on_accept(double objective, const std::vector<double>& solution, int source,
                   const WorkerTrace& trace, size_t effort_at) override;

    // Emit one `[HeurSol]` line.  `const` and lock-free by construction —
    // see the definition for the threading argument.
    void trace_offer(const WorkerTrace& trace, size_t effort_at, double objective) const;

    HighsMipSolver& mipsolver_;
    // Serialises `trySolution`: `HighsMipSolverData::addIncumbent` is not
    // thread-safe and `on_accept` runs on whichever worker thread produced
    // the solution.
    std::mutex highs_mtx_;

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
