#pragma once

#include "rng.h"
#include "solution_pool.h"
#include "worker_base.h"

#include <atomic>
#include <cstddef>
#include <vector>

class HighsLp;

// The one place a heuristic worker hands a solution back.
//
// Owns the shared `SolutionPool` and the tag its entries are attributed
// with, and counts what the pool accepted.  What happens to an accepted
// solution beyond the pool is `on_accept`'s, which does nothing here: the
// HiGHS adapter (`IncumbentSink`) submits it to the solver there, and a
// caller running the workers on a model it owns (#170) overrides it to
// take the solutions wherever it wants them.
class SolutionSink {
public:
    // A pool for `model`: its sense decides what "better" means, and its
    // integrality is the mask diversity-aware admission measures Hamming
    // distance on.  `source` tags everything offered.
    SolutionSink(const HighsLp& model, int source);
    virtual ~SolutionSink() = default;

    SolutionSink(const SolutionSink&) = delete;
    SolutionSink& operator=(const SolutionSink&) = delete;

    // What one offer did, in the two senses that matter (#116).
    //
    // `accepted` is the pool's admission verdict, unchanged since #111.
    // `improved_incumbent` is whether the offer moved the best objective
    // the solve knows — decided by `SolutionPool` against the best it held
    // before the insertion, which is the same thing while a presolve
    // dispatch runs: the sink seeds the pool from the incumbent and every
    // solution that reaches HiGHS goes through the pool first, so nothing
    // can lower the incumbent behind the pool's back.  Deriving it there
    // rather than from a solver field is also what keeps it off a worker
    // thread, where reading `mipdata` races `addIncumbent` (#98/#99).
    struct OfferResult {
        bool accepted = false;
        bool improved_incumbent = false;
    };

    // Offer a candidate solution.  When the pool accepts it, `on_accept`
    // has already run, from inside this call.  Safe to call concurrently
    // from any worker.
    //
    // Two verdicts, because there are two questions and #111 answered the
    // wrong one with the right mechanism.  Every presolve worker used to
    // drop the pool's answer and substitute a worker-local notion ("I beat
    // my own best"), which resets to nothing on rebuild, so the staleness
    // counters the patience gates read were cleared by solutions the pool
    // had refused: on `fpr/flugpl` at one worker, 2,785,359 effort against
    // a 69,632 ceiling with exactly one accepted incumbent, i.e. 39
    // ceilings' worth of free resets.  #111 pointed the gates at
    // `accepted`, which fixed the refusals but left the pool's *admission
    // policy* driving them — it keeps a top-K, so a heuristic beating its
    // own worst entry resets staleness forever.  #113's probe put a number
    // on the difference: 233 instances, presolve-only, 30 s, 16 workers,
    // FPR earns ~3.3 M acceptances against 590 incumbent improvements,
    // Scylla 367,801 against 374.  Five orders of magnitude, so a patience
    // calibrated on improvements cannot be spent against a gate that
    // resets on acceptances.
    //
    // So `OfferResult::improved_incumbent` is what every staleness gate
    // reads (#116), and `accepted` is what `accepted()`, `[Heur] found`
    // and `[HeurSol] accepted` keep reporting — production, in the sense
    // of "a feasible solution worth keeping", is still the pool's call and
    // external tooling reads it as that.
    //
    // `[[nodiscard]]` since #111, and returning a struct rather than a
    // bool is deliberate: it makes every gate site name which fact it
    // reads instead of inheriting whichever one `offer` happened to mean.
    // The two deliberate discards are spelled `static_cast<void>` with a
    // reason at the call site.
    //
    // Both flags are computed inside `SolutionPool`'s own lock — which is
    // where the pre-offer best is race-free — and returned by value, so
    // reading them adds no shared state (#98/#99).
    //
    // `effort_at` is the offering worker's own charged effort at the moment
    // of the offer — the counter that worker's *own* patience gate reads, so a
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
    [[nodiscard]] OfferResult offer(double objective, const std::vector<double>& solution,
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

    // Restart material for a worker beginning a fresh attempt.  Both are
    // thread-safe (the pool takes its own lock).
    bool get_restart(Rng& rng, std::vector<double>& out) { return pool_.get_restart(rng, out); }
    bool copy_best(std::vector<double>& out) { return pool_.copy_best(out); }

protected:
    // Called once per accepted offer, on the offering worker's thread,
    // after the pool lock is released and before `offer` returns — so
    // concurrently from every worker of a dispatch, and it must be
    // thread-safe.  `source` is the tag the entry was stored with.
    virtual void on_accept(double objective, const std::vector<double>& solution, int source,
                           const WorkerTrace& trace, size_t effort_at);

    SolutionPool pool_;
    // Tag for everything offered.  Written only between parallel regions
    // (`IncumbentSink::set_source`).
    int source_;

private:
    std::atomic<size_t> accepted_{0};
};
