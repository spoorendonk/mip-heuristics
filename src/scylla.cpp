#include "scylla.h"

#include "contested_pdlp.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "io/HighsIO.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "scylla_worker.h"
#include "solution_pool.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>

namespace scylla {

namespace {

// Emit the PDLP/FPR overlap metrics that issue #76 asks for as the
// acceptance signal.  Sum per-worker fresh / stale counters before
// the workers are destroyed so operators running with log_dev_level=3
// can see whether the stale-snapshot path actually kept peer workers
// busy during a held mutex.
void log_overlap_ratio(const HighsLogOptions &log_options,
                       const std::vector<std::unique_ptr<ScyllaWorker>> &workers,
                       std::uint64_t extra_fresh, std::uint64_t extra_stale) {
    std::uint64_t fresh = extra_fresh;
    std::uint64_t stale = extra_stale;
    for (const auto &w : workers) {
        if (!w) {
            continue;
        }
        fresh += w->fresh_solves();
        stale += w->stale_rounds();
    }
    const std::uint64_t total = fresh + stale;
    const double ratio = total == 0 ? 0.0 : static_cast<double>(stale) / static_cast<double>(total);
    // `kVerbose` matches the existing `[Sequential]` lines emitted from
    // `mode_dispatch::log_sequential`; operators setting
    // `log_dev_level=3` expect to see this alongside them.
    highsLogDev(
        log_options, HighsLogType::kVerbose, "[ScyllaOverlap] fresh=%llu stale=%llu ratio=%.3f\n",
        static_cast<unsigned long long>(fresh), static_cast<unsigned long long>(stale), ratio);
}

HighsInt compute_pdlp_iter_cap(size_t max_effort, size_t nnz_lp) {
    if (nnz_lp == 0) {
        return 100;
    }
    auto cap = static_cast<HighsInt>((max_effort >> 2) / nnz_lp);
    return cap < 100 ? 100 : cap;
}

size_t run_parallel_workers(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    HeuristicContext ctx(mipsolver);
    const ProblemView &pv = ctx.problem();
    const ExecutionContext &exec = ctx.exec();
    const HeuristicBudget budget = ctx.budget(max_effort);

    const HighsInt pdlp_iter_cap = compute_pdlp_iter_cap(budget.total, pv.nnz);
    ContestedPdlp pdlp(mipsolver, pdlp_iter_cap);
    if (!pdlp.initialized()) {
        return 0;
    }

    std::atomic<uint64_t> improvement_gen{0};

    // Pre-construct workers outside the parallel region so MakeState
    // can hand them back by index without racing on std::make_unique.
    const int N = static_cast<int>(exec.num_workers);
    std::vector<std::unique_ptr<ScyllaWorker>> workers;
    workers.reserve(exec.num_workers);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = exec.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<ScyllaWorker>(mipsolver, pdlp, *pv.csc, pool,
                                                         budget.total, seed, w, N,
                                                         &improvement_gen));
    }

    struct ScyllaOppState {
        int worker_idx;
    };

    // Retired-worker counters so `log_overlap_ratio` can include the
    // contributions of workers that finished and were replaced mid-run
    // — flagged by R3 in the round-2 review.  `std::atomic` because
    // workers are rebuilt from the runner's worker-pinned callback,
    // which may run on different task threads.
    std::atomic<std::uint64_t> retired_fresh{0};
    std::atomic<std::uint64_t> retired_stale{0};

    size_t total_effort = run_opportunistic_loop(
        mipsolver, N, budget.total, budget.stale, budget.attempt_cap, exec.base_seed,
        [](int worker_idx, Rng & /*rng*/) -> ScyllaOppState { return ScyllaOppState{worker_idx}; },
        [&](ScyllaOppState &state, Rng &rng, size_t run_cap) -> AttemptResult {
            auto &worker = workers[state.worker_idx];
            auto rebuild = [&]() {
                // Harvest the retired worker's overlap counters before
                // the rebuild drops its destructor on the floor.
                retired_fresh.fetch_add(worker->fresh_solves(), std::memory_order_relaxed);
                retired_stale.fetch_add(worker->stale_rounds(), std::memory_order_relaxed);
                // Rebuild stale worker with a fresh seed so the runner
                // doesn't lose parallelism over time (mirrors the fpr_lp
                // path).  `pdlp` is shared, so warm-start etc. are
                // reinitialized from scratch but the underlying LP stays.
                uint32_t new_seed = static_cast<uint32_t>(rng());
                worker =
                    std::make_unique<ScyllaWorker>(mipsolver, pdlp, *pv.csc, pool, budget.total,
                                                   new_seed, state.worker_idx, N, &improvement_gen);
            };
            if (worker->finished()) {
                rebuild();
            }
            auto attempt = worker->run_attempt(run_cap);
            if (attempt.effort == 0 && worker->finished()) {
                // Finished *and* no measurable effort in the same call — the
                // nominal-1 guard below does not cover this, and returning 0
                // would retire the chain slot for the rest of the dispatch.
                // Rebuild and take the attempt now.
                rebuild();
                attempt = worker->run_attempt(run_cap);
            }
            // Report a nominal 1 unit when the chain is still alive but the
            // attempt produced no measurable effort (e.g. a PDLP stall that has
            // not yet hit kMaxPdlpStalls). Prevents run_opportunistic_loop's
            // zero-effort guard from retiring a live chain.
            if (attempt.effort == 0 && !worker->finished()) {
                attempt.effort = 1;
            }
            return attempt;
        });

    log_overlap_ratio(mipsolver.options_mip_->log_options, workers,
                      retired_fresh.load(std::memory_order_relaxed),
                      retired_stale.load(std::memory_order_relaxed));
    return total_effort;
}

}  // namespace

size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    const auto *model = mipsolver.model_;
    if (model->num_col_ == 0 || model->num_row_ == 0) {
        return 0;
    }
    return run_parallel_workers(mipsolver, pool, max_effort);
}

}  // namespace scylla
