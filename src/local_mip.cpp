#include "local_mip.h"

#include "heuristic_common.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "local_mip_construction.h"
#include "local_mip_worker.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "rng.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <utility>
#include <vector>

namespace local_mip {

using local_mip_detail::construct_initial_solution;
using local_mip_detail::construction_effort_cap;
using local_mip_detail::LocalMipWorker;
using local_mip_detail::perturb_solution;

namespace {

// Test-only branch counters for `resolve_worker_start`.  Atomic so the
// opportunistic runner can increment from concurrent workers without
// racing.  Reset and read via the API in `local_mip.h`.
std::atomic<int64_t> g_pool_count{0};
std::atomic<int64_t> g_incumbent_count{0};
std::atomic<int64_t> g_construction_count{0};

// The wall-clock poll counters (#162).  Same contract as the three above.
std::atomic<int64_t> g_deadline_polls{0};
std::atomic<int64_t> g_deadline_poll_effort{0};

}  // namespace

void reset_warm_start_counters() {
    if constexpr (kInstrumented) {
        g_pool_count.store(0, std::memory_order_relaxed);
        g_incumbent_count.store(0, std::memory_order_relaxed);
        g_construction_count.store(0, std::memory_order_relaxed);
    }
}

void reset_deadline_poll_counters() {
    if constexpr (kInstrumented) {
        g_deadline_polls.store(0, std::memory_order_relaxed);
        g_deadline_poll_effort.store(0, std::memory_order_relaxed);
    }
}

DeadlinePollCounters deadline_poll_counters() {
    if constexpr (kInstrumented) {
        return {g_deadline_polls.load(std::memory_order_relaxed),
                g_deadline_poll_effort.load(std::memory_order_relaxed)};
    }
    return {0, 0};
}

// The invariant is about the whole attempt: the poll count has to cover
// the effort the attempt charged, including whatever it spent after its
// last poll.  Hence two calls rather than one — a poll, and an attempt's
// closing charge.
void note_deadline_poll() {
    if constexpr (kInstrumented) {
        g_deadline_polls.fetch_add(1, std::memory_order_relaxed);
    }
}

void note_attempt_effort(int64_t effort_charged) {
    if constexpr (kInstrumented) {
        g_deadline_poll_effort.fetch_add(effort_charged, std::memory_order_relaxed);
    }
}

WarmStartCounters warm_start_counters() {
    if constexpr (kInstrumented) {
        return {g_pool_count.load(std::memory_order_relaxed),
                g_incumbent_count.load(std::memory_order_relaxed),
                g_construction_count.load(std::memory_order_relaxed)};
    }
    return {0, 0, 0};
}

namespace {

// Helper to bump a warm-start counter only when instrumentation is
// enabled.  Compiles to a no-op in production builds.
inline void bump_counter(std::atomic<int64_t>& counter) {
    if constexpr (kInstrumented) {
        counter.fetch_add(1, std::memory_order_relaxed);
    } else {
        (void)counter;
    }
}

// Row + integer feasibility check for a candidate solution. Used once
// per cold-construct branch so a feasible construction lands in the
// shared pool with the LocalMIP source tag; if infeasible the caller
// still uses it as the search's starting point (paper's intended
// behaviour).
bool is_solution_feasible(const HighsMipSolver& mipsolver, const std::vector<double>& solution) {
    const auto* model = mipsolver.model_;
    const auto* mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    const double feastol = mipdata->feastol;
    const double inttol = mipdata->epsilon;
    if (std::cmp_not_equal(solution.size(), ncol)) {
        return false;
    }
    // Integer feasibility.
    for (HighsInt j = 0; j < ncol; ++j) {
        if (model->integrality_[j] == HighsVarType::kInteger ||
            model->integrality_[j] == HighsVarType::kImplicitInteger) {
            if (std::abs(solution[j] - std::round(solution[j])) > inttol) {
                return false;
            }
        }
        if (solution[j] < model->col_lower_[j] - feastol ||
            solution[j] > model->col_upper_[j] + feastol) {
            return false;
        }
    }
    // Row feasibility — walk HiGHS's ARstart_/ARindex_/ARvalue_ once.
    for (HighsInt i = 0; i < nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = mipdata->ARstart_[i]; k < mipdata->ARstart_[i + 1]; ++k) {
            lhs += mipdata->ARvalue_[k] * solution[mipdata->ARindex_[k]];
        }
        if (lhs < model->row_lower_[i] - feastol || lhs > model->row_upper_[i] + feastol) {
            return false;
        }
    }
    return true;
}

double compute_solution_objective(const HighsMipSolver& mipsolver,
                                  const std::vector<double>& solution) {
    const auto* model = mipsolver.model_;
    double obj = model->offset_;
    for (HighsInt j = 0; j < model->num_col_; ++j) {
        obj += model->col_cost_[j] * solution[j];
    }
    return obj;
}

// Resolve the starting point for a worker with the paper's cold-start
// fallback (issue #75):
//
//   1. Prefer the pool's best if one exists (an earlier heuristic in
//      the same presolve chain or another worker may have landed one).
//   2. Else prefer `incumbent` if non-empty (warm start).  That is the
//      dispatch's snapshot (`ProblemView::incumbent`), never the live
//      `mipdata->incumbent`, which a peer worker's accepted solution can
//      reallocate mid-read (issue #98).  Reaching this branch means the
//      pool is empty, so nothing has been submitted and the snapshot is
//      still what the solver holds.
//   3. Else run the paper's construction phase
//      (`construct_initial_solution`), capped at
//      `construction_effort_cap(max_effort)`, with a per-worker
//      seeded RNG so cold-start diversity matches the existing
//      perturbation-based diversity of workers 1..N-1.
//
// The callers read the returned vector via `.data()` and feed it
// straight into `LocalMipWorker`'s `initial_solution` pointer — the
// worker's constructor clamps and rounds defensively, so the
// construction result being mildly infeasible is fine.
//
// NOTE: The pool-first branch is the cold/warm boundary that issue
// #74 is expected to further refine (pool-aware warm-start).  This
// file's contract with #74: return the pool's best when
// `snap.has_solution` is true, never fall through to incumbent in
// that case.  That matches #75's out-of-scope note.
// `cold_start_cache` lets a caller amortise cold-start construction across
// N workers: the first worker that hits the construction branch writes
// its result into the cache; subsequent workers re-use the same base
// vector (they'll perturb it downstream, so diversity survives).  Pass
// null to disable caching.  Flagged by review R3 — N full constructions
// per presolve dispatch was wasteful on big MIPs.
// `effort_out` accumulates construction effort the call paid (0 if
// the function returned via the pool or incumbent branches, or via the
// cold-start cache hit).  Callers add it to
// `mipdata->heuristic_effort_used` (R1-3 round-3 review).
std::vector<double> resolve_worker_start(HighsMipSolver& mipsolver, const CscMatrix& csc,
                                         IncumbentSink& sink, const WorkerTrace& trace,
                                         const std::vector<double>& incumbent, size_t max_effort,
                                         uint32_t seed,
                                         std::vector<double>* cold_start_cache = nullptr,
                                         size_t* effort_out = nullptr) {
    // `copy_best` takes the pool lock once and copies only the top
    // entry's solution vector.  Previous versions used
    // `sorted_entries()` which copies up to kPoolCapacity entries
    // (each sized `ncol`) just to read entry 0 — round-2 reviewers R1,
    // R2, R3 all flagged the waste on big MIPs.
    std::vector<double> start;
    if (sink.copy_best(start)) {
        bump_counter(g_pool_count);
        return start;
    }
    if (!incumbent.empty()) {
        bump_counter(g_incumbent_count);
        return incumbent;
    }
    // Cold start: neither the pool nor the incumbent has a solution.
    // Re-use a cached construction if one was produced earlier in this
    // dispatch; otherwise run the paper's construction phase and cache
    // it for subsequent workers.
    if (cold_start_cache != nullptr && !cold_start_cache->empty()) {
        bump_counter(g_construction_count);
        return *cold_start_cache;
    }
    bump_counter(g_construction_count);
    Rng rng(seed);
    std::vector<double> constructed;
    size_t construction_effort = construct_initial_solution(
        mipsolver, csc, rng, construction_effort_cap(max_effort), constructed);
    if (effort_out != nullptr) {
        *effort_out += construction_effort;
    }
    // If the construction happens to land on a feasible integer point,
    // publish it to the shared pool so downstream heuristics (and
    // HiGHS's own incumbent path) pick it up.  Tag as `LocalMIP`
    // (source char 'M') — we don't mint a new `Construction` source
    // tag because that would require an upstream HiGHS patch.
    // Infeasible constructions are the paper's intended input to the
    // search phase and are not inserted.
    if (!constructed.empty() && is_solution_feasible(mipsolver, constructed)) {
        double obj = compute_solution_objective(mipsolver, constructed);
        // Discarded on purpose: this is a publish, not a worker's attempt
        // verdict.  `resolve_worker_start` runs at *worker construction*
        // — on the dispatching thread for the prime, and on task threads
        // from the runner's `MakeState` callback with peers already
        // running — which is before the `LocalMipWorker` and its
        // `WorkerBudgetState` exist, and `MakeState` yields no
        // `AttemptResult` for `note_staleness` either.  So there is no
        // staleness counter for either half of the verdict to feed, at
        // either level.
        // `effort_at` is the construction sweep this call has just paid.
        // The `LocalMipWorker` the caller builds for this slot takes the
        // same number as its `WorkerTrace::effort_base`, so the slot's
        // `effort_at` continues from here rather than restarting at the
        // worker's own zeroed `ctx_.effort` (#106).
        static_cast<void>(sink.offer(obj, constructed, trace, trace.at(construction_effort)));
    }
    if (cold_start_cache != nullptr) {
        *cold_start_cache = constructed;
    }
    return constructed;
}

// Carry a retiring worker's charge into its slot's trace base, so the
// slot's `[HeurSol] effort_at` keeps rising across a rebuild instead of
// restarting at the replacement's zeroed `ctx_.effort` (#106).  A free
// function rather than two lines inside the runner callback: `run` is
// already at the cognitive-complexity threshold, and this is a
// self-contained responsibility that has nothing to do with restart
// selection.
void retire_trace(const std::unique_ptr<LocalMipWorker>& worker, WorkerTrace& trace) {
    if (worker) {
        trace.effort_base = worker->traced_effort();
    }
}

}  // namespace

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, IncumbentSink& sink) {
    // Issue #75: the old `mipdata->incumbent.empty()` early-return is
    // gone.  Cold-start is now handled by `resolve_worker_start` which
    // runs the paper's construction phase when neither pool nor
    // incumbent has a solution.  The sibling issue #74 handles the
    // warm-start-with-pool path; this function stays neutral on that
    // (pool-first lookup in `resolve_worker_start` already covers it).
    if (problem.degenerate() || budget.disabled()) {
        return {};
    }

    HighsMipSolver& mipsolver = exec.mipsolver;
    const HighsInt ncol = problem.ncol;

    struct LmState {
        std::unique_ptr<LocalMipWorker> worker;
        // Trace-only slot identity, carried across rebuilds (#106).  Its
        // `effort_base` also absorbs the cold-start construction sweeps,
        // which are charged to the dispatch total but not to any worker's
        // `ctx_.effort`.
        WorkerTrace trace;
    };

    // Cold-start cache shared across all workers of this dispatch: the
    // first worker that falls through to the construction branch pays
    // the full O(nnz) cost, peers re-use the cached base vector and
    // diverge via perturbation.  `std::mutex`-protected because the
    // runner's MakeState callback runs on multiple task threads
    // concurrently.
    std::mutex cold_start_cache_mu;
    std::vector<double> cold_start_cache;

    // Per-thread-safe accumulator for cold-start construction effort.
    // R1-3 round-3 review: the construction sweep is wall-time-visible
    // and must be booked into `mipdata->heuristic_effort_used`.  Use
    // `std::atomic<size_t>` so concurrent MakeState/Run callbacks can
    // accumulate without holding the cold-start mutex.
    std::atomic<size_t> construction_effort{0};

    // Prime the cache on this thread, before any worker starts.
    //
    // The cache was written for the epoch runner, which resolved every
    // worker's start sequentially *before* the parallel region — so the
    // first worker paid the O(nnz) construction and the rest hit the
    // cache, one construction per dispatch.  The continuous runner enters
    // MakeState on all N workers at once, so every one of them finds the
    // cache empty and constructs: the mutex de-duplicates the write, not
    // the work.  That is N× the cold-start cost on exactly the instances
    // where it is most expensive, and it inflates the effort LocalMIP
    // reports.  Priming here restores the
    // one-per-dispatch property.
    //
    // Returns via the pool or incumbent branch (cheaply, leaving the cache
    // empty) whenever either can seed a start, so this only constructs
    // when the workers would have had to anyway.
    // Guarded on `max_effort`: `run_opportunistic_loop` returns immediately
    // at a zero budget, so an unconditional prime would make a
    // no-search dispatch report non-zero effort where it used to report 0.
    if (budget.total > 0) {
        size_t primed_effort = 0;
        // `worker = -1`: the prime runs on the dispatching thread, before
        // any worker slot exists, so its publish belongs to no slot.
        resolve_worker_start(mipsolver, *problem.csc, sink, WorkerTrace{-1, 0}, problem.incumbent,
                             budget.per_worker, exec.base_seed, &cold_start_cache, &primed_effort);
        construction_effort.fetch_add(primed_effort, std::memory_order_relaxed);
    }

    size_t total_effort = run_opportunistic_loop(
        exec, budget,
        [&](int worker_idx, Rng& rng) -> LmState {
            auto seed = static_cast<uint32_t>(rng());
            std::vector<double> local_cache;
            {
                std::scoped_lock lock(cold_start_cache_mu);
                local_cache = cold_start_cache;  // cheap if empty, one copy if warm
            }
            size_t my_construction_effort = 0;
            std::vector<double> start = resolve_worker_start(
                mipsolver, *problem.csc, sink, WorkerTrace{worker_idx, 0}, problem.incumbent,
                budget.per_worker, seed, &local_cache, &my_construction_effort);
            if (my_construction_effort > 0) {
                construction_effort.fetch_add(my_construction_effort, std::memory_order_relaxed);
            }
            if (!local_cache.empty()) {
                // R1/R2/R3 round-3 review: drop the lock-free outer
                // `cold_start_cache.empty()` check — `std::vector::empty()`
                // reads the size member which races concurrent writers
                // under the mutex (textbook DCL UB on a non-atomic
                // compound type).  The single locked check below is
                // cheap; MakeState fires N times per dispatch, not per
                // attempt.
                std::scoped_lock lock(cold_start_cache_mu);
                if (cold_start_cache.empty()) {
                    cold_start_cache = local_cache;
                }
            }
            if (worker_idx != 0) {
                perturb_solution(start, problem.binary.data(), problem.model->integrality_,
                                 problem.model->col_lower_, problem.model->col_upper_, ncol, rng);
            }
            // The construction this slot just paid seeds its trace base, so
            // the worker's zeroed `ctx_.effort` continues the slot's count
            // instead of restarting below the value already emitted.
            const WorkerTrace trace{worker_idx, my_construction_effort};
            return LmState{
                std::make_unique<LocalMipWorker>(mipsolver, exec, *problem.csc, sink,
                                                 budget.per_worker, budget.worker_stale, seed,
                                                 start.data(), problem.binary.data(), trace),
                trace};
        },
        [&](LmState& state, Rng& rng, size_t run_cap) -> AttemptResult {
            if (!state.worker || state.worker->finished()) {
                // Restart from pool, incumbent, or fresh construction
                // (cold-start), with fresh perturbation.
                // Retire the outgoing occupant's charge into the slot's
                // trace base before it is destroyed (#106).
                retire_trace(state.worker, state.trace);
                std::vector<double> restart_sol;
                if (!sink.get_restart(rng, restart_sol)) {
                    // Snapshot, not `problem.mipdata->incumbent`: this runs
                    // on a worker thread while peers submit (issue #98).
                    // The pool is empty on this branch, so no submission has
                    // happened and the snapshot is current.
                    if (!problem.incumbent.empty()) {
                        restart_sol = problem.incumbent;
                    } else {
                        // note (R2-9 / R3-6 round-4 review): cold-start
                        // construction is booked into the *global*
                        // accountant only, not the runner's per-attempt
                        // budget cap.  Intentional: the per-attempt cap
                        // paces wall spend, the outer global budget is
                        // what bounds the heuristic.  The effort here is
                        // booked into `construction_effort` and added to
                        // `mipdata->heuristic_effort_used` after the
                        // opportunistic loop returns; it does not
                        // participate in the inner per-iteration budget
                        // checks.  Bounded by
                        // `construction_effort_cap(worker_budget)` per
                        // restart so total construction work scales
                        // with the outer budget.
                        auto cseed = static_cast<uint32_t>(rng());
                        Rng construct_rng(cseed);
                        size_t my_construction_effort = construct_initial_solution(
                            mipsolver, *problem.csc, construct_rng,
                            construction_effort_cap(budget.per_worker), restart_sol);
                        construction_effort.fetch_add(my_construction_effort,
                                                      std::memory_order_relaxed);
                        state.trace.effort_base += my_construction_effort;
                    }
                }
                perturb_solution(restart_sol, problem.binary.data(), problem.model->integrality_,
                                 problem.model->col_lower_, problem.model->col_upper_, ncol, rng);
                auto seed = static_cast<uint32_t>(rng());
                state.worker = std::make_unique<LocalMipWorker>(
                    mipsolver, exec, *problem.csc, sink, budget.per_worker, budget.worker_stale,
                    seed, restart_sol.data(), problem.binary.data(), state.trace);
            }
            return state.worker->run_attempt(run_cap);
        });

    // The caller books this through `EffortLedger` (issue #79), which is
    // the single point of effort accounting.  No `abandoned_setup`: what
    // LocalMIP does before the runner — the cold-start prime — is charged
    // work rather than an unwatched sequential setup, and it carries no
    // deadline bail to report (issue #119).
    return {.effort = total_effort + construction_effort.load(std::memory_order_relaxed)};
}

}  // namespace local_mip
