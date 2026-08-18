#include "scylla.h"

#include "contested_pdlp.h"
#include "fpr_var_order.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "io/HighsIO.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "scylla_worker.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>

namespace scylla {

namespace {

// Variable orders for every entry of `kFprConfigs`, computed on the
// dispatching thread before any parallel region.
//
// `ScyllaWorker`'s constructor used to compute its own, which was fine for
// the initial workers (built sequentially below) but not for the rebuild
// in `run_opportunistic_loop`'s callback: that runs on a task thread, so a
// rebuilt worker read the live root domain through `bucket_by_type` while
// a peer's accepted solution was propagating it (issue #99), and for the
// `kTypecl` strategies also called `HighsCliqueTable::cliquePartition`,
// which mutates and reallocates the clique table that `addIncumbent`'s
// `extractObjCliques` is writing at the same time.  `fpr::run` has always
// precomputed for exactly this reason; Scylla now does too.
//
// Behaviour-identical: the per-config seed is `base_seed + config_index`,
// independent of the worker seed, so a worker rebuilt with a fresh seed
// computed the same order it now looks up.  It also drops n redundant
// computations of the same `kNumFprConfigs` orders at construction.
std::vector<std::vector<HighsInt>> precompute_config_var_orders(HighsMipSolver& mipsolver) {
    std::vector<std::vector<HighsInt>> orders(kNumFprConfigs);
    const uint32_t base = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    for (int i = 0; i < kNumFprConfigs; ++i) {
        Rng rng(base + static_cast<uint32_t>(i));
        orders[i] = compute_var_order(mipsolver, kFprConfigs[i].strat.var_strategy, rng, nullptr);
    }
    return orders;
}

// Emit the PDLP/FPR overlap metrics that issue #76 asks for as the
// acceptance signal.  Sum per-worker fresh / stale counters before
// the workers are destroyed so operators running with log_dev_level=3
// can see whether the stale-snapshot path actually kept peer workers
// busy during a held mutex.
void log_overlap_ratio(const HighsLogOptions& log_options,
                       const std::vector<std::unique_ptr<ScyllaWorker>>& workers,
                       std::uint64_t extra_fresh, std::uint64_t extra_stale) {
    std::uint64_t fresh = extra_fresh;
    std::uint64_t stale = extra_stale;
    for (const auto& w : workers) {
        if (!w) {
            continue;
        }
        fresh += w->fresh_solves();
        stale += w->stale_rounds();
    }
    const std::uint64_t total = fresh + stale;
    const double ratio = total == 0 ? 0.0 : static_cast<double>(stale) / static_cast<double>(total);
    // `kVerbose` matches the existing `[Sequential]` lines emitted from
    // `EffortLedger::book` (src/effort_ledger.cpp); operators setting
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

}  // namespace

size_t run(const ProblemView& problem, const HeuristicBudget& budget, ExecutionContext& exec,
           IncumbentSink& sink) {
    if (problem.degenerate()) {
        return 0;
    }

    HighsMipSolver& mipsolver = exec.mipsolver;
    const HighsInt pdlp_iter_cap = compute_pdlp_iter_cap(budget.total, problem.nnz);
    ContestedPdlp pdlp(mipsolver, pdlp_iter_cap);
    if (!pdlp.initialized()) {
        return 0;
    }

    std::atomic<uint64_t> improvement_gen{0};

    // Sequential: `compute_var_order` reaches `cliquePartition` and the live
    // root domain, neither of which is safe from a worker thread (#99).
    const std::vector<std::vector<HighsInt>> var_orders = precompute_config_var_orders(mipsolver);

    // Pre-construct workers outside the parallel region so MakeState
    // can hand them back by index without racing on std::make_unique.
    const int n = static_cast<int>(exec.num_workers);
    std::vector<std::unique_ptr<ScyllaWorker>> workers;
    workers.reserve(exec.num_workers);
    for (int w = 0; w < n; ++w) {
        uint32_t seed = exec.worker_seed(w);
        workers.push_back(std::make_unique<ScyllaWorker>(
            mipsolver, pdlp, *problem.csc, sink, problem.binary.data(), var_orders, budget.total,
            seed, w, n, &improvement_gen));
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
        exec, budget,
        [](int worker_idx, Rng& /*rng*/) -> ScyllaOppState { return ScyllaOppState{worker_idx}; },
        [&](ScyllaOppState& state, Rng& rng, size_t run_cap) -> AttemptResult {
            auto& worker = workers[state.worker_idx];
            auto attempt = attempt_with_rebuild(worker, run_cap, [&]() {
                // Harvest the retired worker's overlap counters before
                // the rebuild drops its destructor on the floor.
                retired_fresh.fetch_add(worker->fresh_solves(), std::memory_order_relaxed);
                retired_stale.fetch_add(worker->stale_rounds(), std::memory_order_relaxed);
                // Rebuild stale worker with a fresh seed so the runner
                // doesn't lose parallelism over time (mirrors the fpr_lp
                // path).  `pdlp` is shared, so warm-start etc. are
                // reinitialized from scratch but the underlying LP stays.
                auto new_seed = static_cast<uint32_t>(rng());
                worker = std::make_unique<ScyllaWorker>(
                    mipsolver, pdlp, *problem.csc, sink, problem.binary.data(), var_orders,
                    budget.total, new_seed, state.worker_idx, n, &improvement_gen);
            });
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

}  // namespace scylla
