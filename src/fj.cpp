#include "fj.h"

#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "mip/HighsMipSolver.h"
#include "opportunistic_runner.h"

#include <memory>

namespace fj {

size_t run(const ProblemView &problem, const HeuristicBudget &budget, ExecutionContext &exec,
           IncumbentSink &sink) {
    if (problem.degenerate()) {
        return 0;
    }

    HighsMipSolver &mipsolver = exec.mipsolver;
    const uint32_t random_seed_opp = static_cast<uint32_t>(mipsolver.options_mip_->random_seed);

    struct FjState {
        std::unique_ptr<FjWorker> worker;
        uint32_t initial_seed = 0;
        bool first_creation = true;
    };

    return run_opportunistic_loop(
        exec, budget,
        [random_seed_opp](int worker_idx, Rng & /*rng*/) -> FjState {
            // Pin first attempt to random_seed + w; worker 0 matches vanilla FJ's seed.
            return FjState{nullptr, random_seed_opp + static_cast<uint32_t>(worker_idx), true};
        },
        [&](FjState &state, Rng &rng, size_t run_cap) -> AttemptResult {
            if (!state.worker || state.worker->finished()) {
                uint32_t seed;
                if (state.first_creation) {
                    seed = state.initial_seed;
                    state.first_creation = false;
                } else {
                    seed = static_cast<uint32_t>(rng());
                }
                state.worker =
                    std::make_unique<FjWorker>(mipsolver, sink, budget.per_worker, seed);
            }
            return state.worker->run_attempt(run_cap);
        });
}

}  // namespace fj
