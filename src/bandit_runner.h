#pragma once

#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>

// Shared infrastructure for the opportunistic Thompson-sampling bandit
// loop used by `portfolio::run_presolve_opportunistic` (presolve-time
// multi-arm bandit over FJ / FPR / LocalMIP / Scylla) and
// `fpr_lp::run_portfolio_opportunistic` (dive-time bandit over the 10
// LP-dependent FPR arms).  Both variants share the same control-flow
// skeleton (N workers sample arms, apply effort-proportional pulls,
// update bandit, stop on staleness or budget exhaustion); the only
// per-caller differences are how a single arm pull executes and what
// tag the per-pull log line uses.

// Compare two objective values under the model's optimization sense.
inline bool objective_better(bool minimize, double lhs, double rhs) {
    constexpr double kTol = 1e-9;
    return minimize ? lhs < rhs - kTol : lhs > rhs + kTol;
}

// Reward in {0, 1, 2, 3} for a Thompson update given the pool delta.
// 0 = no feasible; 1 = feasible but not best; 2 = first feasible ever;
// 3 = new global best.
inline int compute_reward(SolutionPool::Snapshot before, SolutionPool::Snapshot after,
                          const HeuristicResult &result, bool minimize) {
    if (!result.found_feasible) {
        return 0;
    }
    if (!before.has_solution) {
        return after.has_solution &&
                       !objective_better(minimize, after.best_objective, result.objective)
                   ? 2
                   : 1;
    }
    bool improved = after.has_solution &&
                    objective_better(minimize, after.best_objective, before.best_objective) &&
                    !objective_better(minimize, after.best_objective, result.objective);
    return improved ? 3 : 1;
}

// Effort-proportional budget cap for a single opportunistic arm pull.
// Each pull gets at most kBudgetCapMultiplier * avg_effort for that arm.
// First pull (no history) uses total_budget / (num_arms * 10).
inline constexpr double kBudgetCapMultiplier = 2.5;

inline size_t compute_budget_cap(const ThompsonSampler &bandit, int arm, size_t total_budget,
                                 int num_arms) {
    auto stats = bandit.stats(arm);
    if (stats.pulls > 0 && stats.avg_effort > 0.0) {
        return static_cast<size_t>(kBudgetCapMultiplier * stats.avg_effort);
    }
    return total_budget / static_cast<size_t>(std::max(num_arms * 10, 1));
}

// Generic opportunistic Thompson-sampling bandit loop.
//
// `make_run_arm(worker_idx)` returns a per-worker callable of signature
//   HeuristicResult(int arm, std::mt19937 &rng, int attempt, size_t arm_budget)
// Returning a fresh callable per worker lets callers thread per-worker
// state (e.g. a persistent ScyllaWorker) through the closure.
//
// `log_arm(arm, effort, reward, wall_ms)` emits the per-pull log line;
// callers use their own preferred tag and arm-name mapping.
//
// Returns aggregate effort consumed across workers.
template <typename MakeRunArm, typename LogArm>
size_t run_bandit_opportunistic_loop(HighsMipSolver &mipsolver, ThompsonSampler &bandit,
                                     SolutionPool &pool, int num_workers, int num_arms,
                                     size_t budget, size_t stale_budget, uint32_t base_seed,
                                     bool minimize, MakeRunArm make_run_arm, LogArm log_arm) {
    auto *mipdata = mipsolver.mipdata_.get();
    const double time_limit = mipsolver.options_mip_->time_limit;

    std::atomic<size_t> total_effort{0};
    std::atomic<size_t> effort_since_improvement{0};
    std::atomic<bool> stop{false};

    highs::parallel::for_each(
        0, static_cast<HighsInt>(num_workers),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * kSeedStride);
                auto run_arm = make_run_arm(static_cast<int>(w));
                int attempt_counter = 0;

                while (!stop.load(std::memory_order_relaxed)) {
                    // Worker 0 polls the terminator / timer every 8 attempts.
                    // These calls are not guaranteed thread-safe for concurrent
                    // callers.  Other workers observe `stop` atomically.
                    if (w == 0 && attempt_counter % 8 == 0) {
                        if (mipdata->terminatorTerminated() ||
                            mipsolver.timer_.read() >= time_limit) {
                            stop.store(true, std::memory_order_relaxed);
                        }
                    }
                    if (stop.load(std::memory_order_relaxed)) {
                        break;
                    }

                    int arm = bandit.select_effort_aware(rng);
                    auto before = pool.snapshot();

                    size_t remaining =
                        budget - std::min(budget, total_effort.load(std::memory_order_relaxed));
                    size_t arm_budget =
                        std::min(compute_budget_cap(bandit, arm, budget, num_arms), remaining);
                    if (arm_budget == 0) {
                        stop.store(true, std::memory_order_relaxed);
                        break;
                    }

                    int attempt = attempt_counter++;

                    auto t0 = std::chrono::steady_clock::now();
                    HeuristicResult result = run_arm(arm, rng, attempt, arm_budget);
                    auto t1 = std::chrono::steady_clock::now();

                    auto after = pool.snapshot();
                    int reward = compute_reward(before, after, result, minimize);
                    bandit.update(arm, reward);
                    bandit.record_effort(arm, result.effort);

                    if (result.effort > 0) {
                        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                        log_arm(arm, result.effort, reward, wall_ms);
                    }

                    if (reward >= 2) {
                        effort_since_improvement.store(0, std::memory_order_relaxed);
                    } else {
                        effort_since_improvement.fetch_add(result.effort,
                                                           std::memory_order_relaxed);
                    }

                    if (effort_since_improvement.load(std::memory_order_relaxed) >= stale_budget) {
                        stop.store(true, std::memory_order_relaxed);
                    }

                    // Zero-effort return: guard against spinning on an arm
                    // that made no progress.
                    if (result.effort == 0) {
                        break;
                    }

                    size_t new_total = total_effort.fetch_add(result.effort) + result.effort;
                    if (new_total >= budget) {
                        stop.store(true, std::memory_order_relaxed);
                    }
                }
            }
        },
        1);

    return total_effort.load(std::memory_order_relaxed);
}
