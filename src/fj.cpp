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

size_t run(const ProblemView& problem, const HeuristicBudget& budget, ExecutionContext& exec,
           IncumbentSink& sink) {
    if (problem.degenerate()) {
        return 0;
    }

    HighsMipSolver& mipsolver = exec.mipsolver;
    const auto random_seed_opp = static_cast<uint32_t>(mipsolver.options_mip_->random_seed);

    struct FjState {
        std::unique_ptr<FjWorker> worker;
        uint32_t initial_seed = 0;
        bool first_creation = true;
    };

    return run_opportunistic_loop(
        exec, budget,
        [random_seed_opp](int worker_idx, Rng& /*rng*/) -> FjState {
            // Pin first attempt to random_seed + w; worker 0 matches vanilla FJ's seed.
            return FjState{nullptr, random_seed_opp + static_cast<uint32_t>(worker_idx), true};
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
                // `budget.worker_stale` is this worker's share of the
                // dispatch's absolute stall ceiling (issue #111) — the
                // `nnz << 8` FJ used to compute from its own copy of the
                // matrix, now sized once alongside every other
                // heuristic's.
                state.worker =
                    std::make_unique<FjWorker>(mipsolver, sink, budget.per_worker,
                                               budget.worker_stale, seed, std::move(start));
            }
            return state.worker->run_attempt(run_cap);
        });
}

}  // namespace fj
