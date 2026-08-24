#pragma once

#include "rng.h"
#include "solution_pool.h"
#include "worker_base.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

class HighsMipSolver;

// The one place a heuristic worker hands a solution back to the solver.
//
// Owns the shared `SolutionPool`, the mutex that serialises HiGHS's
// non-thread-safe `trySolution`, and the `kSolutionSource*` tag the
// running heuristic's entries are attributed with.  Before this class the
// pool + mutex + on_accept wiring was written out twice, verbatim, in
// `mode_dispatch.cpp` and `fpr_lp.cpp`, and every worker hard-coded the
// source constant of its own heuristic at its `try_add` call.
//
// Submission is immediate rather than batched: the accept callback runs
// as soon as the pool takes a solution, so a HiGHS incumbent timestamp
// reflects find time rather than end-of-dispatch flush time.
class IncumbentSink {
public:
    // Constructs the pool, seeds it from the current incumbent, and wires
    // the accept callback.  `source` tags everything offered until
    // `set_source` says otherwise.
    IncumbentSink(HighsMipSolver& mipsolver, int source);

    IncumbentSink(const IncumbentSink&) = delete;
    IncumbentSink& operator=(const IncumbentSink&) = delete;

    // Offer a candidate solution.  Returns true if the pool accepted it,
    // in which case HiGHS has already been told, from inside this call.
    // Safe to call concurrently from any worker.
    //
    // `[[nodiscard]]` since issue #111.  This return value is the
    // project's one definition of "the heuristic produced something" —
    // `accepted()` counts it, and the `found` field of the `[Heur]` line
    // is that counter moving.  Every presolve worker used to drop it and
    // substitute a worker-local notion ("I beat my own best"), which
    // resets to nothing on rebuild, so the staleness counters the stall
    // gates read were cleared by solutions the pool had refused: on
    // `fpr/flugpl` at one worker, 2,785,359 effort against a 69,632
    // ceiling with exactly one accepted incumbent, i.e. 39 ceilings'
    // worth of free resets.  A discarded verdict is now a compile error.
    // The two deliberate discards are spelled `static_cast<void>` with a
    // reason at the call site.
    //
    // The bool is computed inside `SolutionPool`'s own lock and returned
    // by value, so reading it adds no shared state (#98/#99).
    //
    // `effort_at` is the offering worker's own charged effort at the moment
    // of the offer — the counter that worker's *own* stall gate reads, so a
    // difference between two `effort_at` values is directly comparable with
    // `HeuristicBudget::worker_stale` (#106).  Every worker keeps such a
    // counter already; none of them is recomputed or redefined for this,
    // and Scylla's stays the amortised (PDLP cost ÷ N) one its gate uses.
    // It is *not* monotone across a dispatch: FJ, LocalMIP and Scylla all
    // rebuild a retired worker in place and a rebuild starts a fresh
    // counter at zero, so the per-dispatch sequence is sawtooth.
    //
    // `trace` names the worker slot the offer comes from and carries the
    // charge of that slot's retired occupants, so `trace.at(effort_at)` is
    // monotone across rebuilds; see `WorkerTrace` in worker_base.h.
    //
    // Emits the `[HeurSol]` trace line (see incumbent_sink.cpp).
    [[nodiscard]] bool offer(double objective, const std::vector<double>& solution,
                             const WorkerTrace& trace, size_t effort_at);

    // Number of offers the pool has accepted since construction.  The
    // `found` field of the `[Heur]` instrumentation line (issue #95) is
    // this counter moving across one heuristic's dispatch; the sink is
    // the only place that knows, because a worker's return value is its
    // effort and nothing else.  Relaxed loads are enough: the dispatching
    // thread reads it either side of a joined parallel region, so the
    // join already provides the ordering.
    //
    // "Accepted by the pool", not "improved the incumbent": the pool also
    // admits a solution within `kDiversityObjTolerance` of the best when
    // it is structurally diverse.  `found=1` therefore means the heuristic
    // produced a feasible solution worth keeping, which is what the
    // `found` field of the `[Heur]` line reports.
    size_t accepted() const { return accepted_.load(std::memory_order_relaxed); }

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

    // Restart material for a worker beginning a fresh attempt.  Both are
    // thread-safe (the pool takes its own lock).
    bool get_restart(Rng& rng, std::vector<double>& out) { return pool_.get_restart(rng, out); }
    bool copy_best(std::vector<double>& out) { return pool_.copy_best(out); }

private:
    // Take the next process-global dispatch id, remember the heuristic name
    // the tag maps to, and stamp the dispatch's start on the solver clock.
    void begin_dispatch(int source);

    // Emit one `[HeurSol]` line.  `const` and lock-free by construction —
    // see the definition for the threading argument.
    void trace_offer(const WorkerTrace& trace, size_t effort_at, double objective,
                     bool accepted) const;

    HighsMipSolver& mipsolver_;
    SolutionPool pool_;
    // Serialises `trySolution`: `HighsMipSolverData::addIncumbent` is not
    // thread-safe and the accept callback fires on whichever worker
    // thread produced the solution.
    std::mutex highs_mtx_;
    int source_;
    std::atomic<size_t> accepted_{0};

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
