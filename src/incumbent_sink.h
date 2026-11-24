#pragma once

#include "solution_sink.h"
#include "worker_base.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <vector>

class HighsMipSolver;

// The objective `HighsMipSolverData::trySolution` computes for a point, and
// the scale its rounding is measured against.
struct HighsObjective {
    double value = 0.0;
    // `sum |c_i x_i|`.
    double magnitude = 0.0;
};

// `trySolution`'s own objective over `cost`, term for term and in the same
// order: each product taken in `HighsCDouble`, as HiGHS does since v1.16.0.
// `cost.size()` must equal `solution.size()`.
[[nodiscard]] HighsObjective highs_objective(std::span<const double> cost,
                                             const std::vector<double>& solution);

// Whether a point of `objective` could become HiGHS's incumbent while its
// bound is `bound` (`highs_bound`), i.e. whether `addIncumbent` could store
// it.  Not promised bit for bit against `trySolution`: HiGHS's sources are
// built without this project's `-ffp-contract=off`, so the two builds may
// round the compensated arithmetic differently.  Any such difference is a
// few ulps of `magnitude` at most; refusing only past `kSlack` times it
// (some 4500 ulps) keeps every point HiGHS could store.  The negated `>=`
// lets a NaN through, as `trySolution` would.
[[nodiscard]] inline bool may_improve(const HighsObjective& objective, double bound) {
    constexpr double kSlack = 1e-12;
    return !(objective.value >= bound + (kSlack * objective.magnitude));
}

// HiGHS's `upper_bound`, or +inf while it has no incumbent.  Reads `mipdata`:
// on the solving thread, or under a lock that serialises `trySolution`.
[[nodiscard]] double highs_bound(const HighsMipSolver& mipsolver);

// Whether the MIP solution callback sees every `trySolution` that is
// feasible, improving or not, so that no call may be skipped.
[[nodiscard]] bool callback_sees_all(const HighsMipSolver& mipsolver);

// What every `SolutionSink` inside a HiGHS solve shares: the pool seeded
// from the incumbent, the source tag's `[HeurSol]` dispatch context, and the
// trace line itself.  Where an accepted solution goes is the subclass's:
// `IncumbentSink` submits it to HiGHS at once, `LongLivedSink`
// (long_lived_owner.h) queues it for the solving thread.
class HighsSink : public SolutionSink {
public:
    // Trace id of the dispatch currently being attributed.  Drawn from a
    // process-global counter, so `(name, dispatch)` identifies one dispatch
    // uniquely across the whole process — not merely within one solve, which
    // a per-sink counter could not manage: `fpr_lp` builds a new sink per
    // dive, and a per-sink counter would hand every dive the same id.
    [[nodiscard]] uint64_t dispatch_id() const { return dispatch_id_; }

protected:
    // Constructs the pool, seeds it from the current incumbent, and opens
    // a dispatch.  `source` tags everything offered until `begin_dispatch`
    // says otherwise.  On the solving thread, with no worker running: it
    // reads the live `mipdata->incumbent`.
    HighsSink(HighsMipSolver& mipsolver, int source);

    // Take the next process-global dispatch id, remember the heuristic name
    // the tag maps to, stamp the dispatch's start on the solver clock, and
    // read `callback_sees_all`.  On the solving thread, every parallel
    // region joined.
    void begin_dispatch(int source);

    // Emit one `[HeurSol]` line.  `const` and lock-free by construction —
    // see the definition for the threading argument.
    void trace_offer(const WorkerTrace& trace, size_t effort_at, double objective) const;

    HighsMipSolver& mipsolver_;
    // `callback_sees_all` as of the last `begin_dispatch`.
    bool callback_sees_all_ = true;

private:
    // `[HeurSol]` dispatch context.  Written only by `begin_dispatch`, which
    // runs on the solving thread with every parallel region joined — the
    // same invariant `IncumbentSink::set_source` and `EffortLedger` rely
    // on.  Workers only ever read them, so no synchronisation is needed and
    // none is added: a log mutex here would serialise every offer a second
    // time, on top of the pool's own lock.
    const char* dispatch_name_ = "unknown";
    uint64_t dispatch_id_ = 0;
    double dispatch_start_s_ = 0.0;
};

// The `SolutionSink` a presolve chain or an `fpr_lp` dive hands its
// workers: every solution the pool accepts is traced, and submitted to
// HiGHS at once unless HiGHS could do nothing with it (`reaches_highs`).
//
// Owns the mutex that serialises HiGHS's non-thread-safe `trySolution`;
// the pool, the tag and the offer verdicts are `SolutionSink`'s (#170), the
// seeding and the `[HeurSol]` context `HighsSink`'s.
//
// Calling `trySolution` from a worker thread is legal only because no
// search runs beside these workers: B&B waits for a chain dispatch and for
// an `fpr_lp` dive.  A heuristic that runs beside B&B uses `LongLivedSink`.
//
// Submission is immediate rather than batched: `on_accept` runs as soon as
// the pool takes a solution, so a HiGHS incumbent timestamp reflects find
// time rather than end-of-dispatch flush time.
class IncumbentSink final : public HighsSink {
public:
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
    void set_source(int source) {
        begin_dispatch(source);
        sync_highs_bound();
    }

private:
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
    // is skipped only when neither can happen and `may_improve` says no
    // against `highs_bound_`.  Which solutions reach HiGHS's incumbent does
    // not change: `highs_bound_` is never below HiGHS's bound, only stale
    // above it, and a stale value just lets a call through.  This mirrors
    // `addIncumbent`'s decision, so a HiGHS tag bump re-reads it
    // (docs/ARCHITECTURE.md, bumping the tag).
    //
    // Not the pool's `improved_best`: that clears a margin, against a
    // watermark in the worker's objective space, so an offer improving
    // HiGHS's incumbent by less than the margin would be lost.
    [[nodiscard]] bool reaches_highs(const std::vector<double>& solution) const;

    // Copy HiGHS's bound into `highs_bound_`: on the dispatching thread with
    // every parallel region joined, or under `highs_mtx_`, the two places
    // `upper_bound` may be read.
    void sync_highs_bound();

    // Serialises `trySolution`: `HighsMipSolverData::addIncumbent` is not
    // thread-safe and `on_accept` runs on whichever worker thread produced
    // the solution.
    std::mutex highs_mtx_;
    // `highs_bound` as of the last `sync_highs_bound`.  Written only under
    // `highs_mtx_` or at a dispatch boundary; read lock-free by
    // `reaches_highs`.  `upper_bound` only falls between syncs, so a relaxed
    // read is at worst too high.
    std::atomic<double> highs_bound_{std::numeric_limits<double>::infinity()};
};
