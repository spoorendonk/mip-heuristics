#include "fj.h"

#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "mip/HighsMipSolver.h"
#include "opportunistic_runner.h"

#include <memory>
#include <utility>
#include <vector>

namespace fj {

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, IncumbentSink& sink) {
    if (problem.degenerate() || budget.disabled()) {
        return {};
    }

    HighsMipSolver& mipsolver = exec.mipsolver;
    const auto random_seed_opp = static_cast<uint32_t>(mipsolver.options_mip_->random_seed);

    struct FjState {
        std::unique_ptr<FjWorker> worker;
        uint32_t initial_seed = 0;
        bool first_creation = true;
        // Trace-only slot identity, carried across rebuilds (#106).
        WorkerTrace trace;
    };

    // No setup to abandon, so this dispatch can only ever report a plain
    // effort count.  MakeState below builds no worker at all — it returns
    // a null slot plus three scalars — and the `FjWorker` is constructed
    // lazily in the RunAttempt callback, which `run_opportunistic_loop`
    // reaches only *after* `should_stop`, whose first act is the deadline
    // poll.  So an already-expired dispatch never constructs one: FJ has
    // nothing ahead of that gate to give up on, where FPR and Scylla
    // precompute variable orders ahead of it and therefore can (#117).
    // This does not touch the narrower standing caveat that once
    // construction has begun nothing bounds it — the deadline is polled
    // between attempts, not inside `FjWorker`'s constructor.
    return {.effort = run_opportunistic_loop(
                exec, budget,
                [random_seed_opp](int worker_idx, Rng& /*rng*/) -> FjState {
                    // Pin first attempt to random_seed + w; worker 0 matches vanilla FJ's seed.
                    return FjState{nullptr, random_seed_opp + static_cast<uint32_t>(worker_idx),
                                   true, WorkerTrace{worker_idx, 0}};
                },
                [&](FjState& state, Rng& rng, size_t run_cap) -> AttemptResult {
                    if (!state.worker || state.worker->finished()) {
                        uint32_t seed;
                        if (state.first_creation) {
                            seed = state.initial_seed;
                            state.first_creation = false;
                        } else {
                            seed = static_cast<uint32_t>(rng());
                        }
                        // Pool first, dispatch snapshot second — the same order
                        // LocalMIP's `resolve_worker_start` uses, and the reason
                        // dropping the live `mipdata->incumbent` read (issue #98)
                        // costs FJ nothing.  The runner rebuilds a stalled worker
                        // inside the parallel region, and that rebuild used to
                        // warm-start from whatever a peer had just found; the pool
                        // holds every such solution (`IncumbentSink` seeds it from
                        // the incumbent and every accept goes through it) and
                        // `copy_best` takes its own lock, so this reads the same
                        // material without racing a concurrent `addIncumbent`.
                        std::vector<double> start;
                        if (!sink.copy_best(start)) {
                            start = problem.incumbent;
                        }
                        // Carry the outgoing worker's charge into the replacement's
                        // trace base, so the `[HeurSol] effort_at` of this slot keeps
                        // rising instead of restarting with the fresh
                        // `WorkerBudgetState` (#106).  Nothing about what the budget
                        // counts changes: `base_` still starts at zero.
                        if (state.worker) {
                            state.trace.effort_base = state.worker->traced_effort();
                        }
                        // `budget.worker_stale` is this worker's share of the
                        // dispatch's absolute patience ceiling (issue #111) — the
                        // `nnz << 8` FJ used to compute from its own copy of the
                        // matrix, now sized once alongside every other
                        // heuristic's.
                        state.worker = std::make_unique<FjWorker>(
                            mipsolver, exec, sink, budget.per_worker, budget.worker_stale, seed,
                            std::move(start), state.trace);
                    }
                    return state.worker->run_attempt(run_cap);
                })};
}

}  // namespace fj
