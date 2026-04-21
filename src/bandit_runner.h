#pragma once

#include "continuous_loop.h"
#include "heuristic_common.h"
#include "io/HighsIO.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

// Shared infrastructure for Thompson-sampling bandit dispatch.
//
// Two complementary primitives live here:
//
//   - `run_bandit_opportunistic_loop` — continuous N-worker loop used by
//     `portfolio::run_presolve_opportunistic` (presolve-time multi-arm
//     bandit over FJ / FPR / LocalMIP / Scylla) and
//     `fpr_lp::run_portfolio_opportunistic` (dive-time bandit over the
//     10 LP-dependent FPR arms).
//
//   - `BanditWorker` concept + `make_bandit_restart_callback` — the
//     epoch-gated deterministic counterpart, used by
//     `portfolio::run_presolve` (det branch) and
//     `fpr_lp::run_portfolio_deterministic`.  Both portfolios spawn one
//     worker per thread, each worker runs whichever arm the bandit
//     currently assigned, and at each epoch boundary the restart
//     callback computes reward, updates the bandit, and reassigns the
//     arm for the next epoch.
//
// All four call sites share the same "select arm, snapshot pool, run
// arm, snapshot pool, compute reward, update bandit" skeleton; the
// primitives here factor that out so the portfolios only provide what
// genuinely differs (per-arm run logic, arm-name mapping, and the log
// tag).

// Unified per-arm log line used by both the deterministic and
// opportunistic bandit paths.  Format:
//   [<tag>] arm=<name> effort=<N> wall_ms=<X.X> effort_per_ms=<X> reward=<R>
// The `[Portfolio]` tag is parsed by `bench/parse_highs_log.py`; the
// parser accepts the trailing `reward=<N>` as an optional field so
// historical logs without a reward suffix still parse.
inline void log_bandit_arm(const HighsLogOptions &log_options, const char *tag,
                           const char *arm_name, size_t effort, double wall_ms, int reward) {
    double effort_per_ms = wall_ms > 0.0 ? static_cast<double>(effort) / wall_ms : 0.0;
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[%s] arm=%s effort=%zu wall_ms=%.1f effort_per_ms=%.0f reward=%d\n", tag, arm_name,
                effort, wall_ms, effort_per_ms, reward);
}

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
// First pull (no history) uses total_budget / (num_arms * 10), clamped by
// kFirstPullEffortCap so a single cold-start arm cannot burn more than
// ~100-200 ms wall time on large instances (effort_per_ms ≈ 100k-200k
// across FPR/LocalMIP arms, so 20M effort ≈ 100-200 ms).  Without this
// clamp, an arm like FprRepairSearchLocks on a 9k-nnz instance at
// mip_heuristic_effort=0.30 gets a ~150 M effort first-pull budget and
// burns ~1.4 s before the bandit has any observation to throttle it.
// The repair_search loop only honours the budget after the effort-counter
// fix in repair_search.cpp:287 (it was previously ignoring PropEngine
// work), so this cap is the other half of the T1st-regression fix.
inline constexpr double kBudgetCapMultiplier = 2.5;
inline constexpr size_t kFirstPullEffortCap = 20'000'000;

inline size_t compute_budget_cap(const ThompsonSampler &bandit, int arm, size_t total_budget,
                                 int num_arms) {
    auto stats = bandit.stats(arm);
    if (stats.pulls > 0 && stats.avg_effort > 0.0) {
        return static_cast<size_t>(kBudgetCapMultiplier * stats.avg_effort);
    }
    const size_t proportional = total_budget / static_cast<size_t>(std::max(num_arms * 10, 1));
    return std::min(proportional, kFirstPullEffortCap);
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
[[nodiscard]] size_t run_bandit_opportunistic_loop(HighsMipSolver &mipsolver,
                                                   ThompsonSampler &bandit, SolutionPool &pool,
                                                   int num_workers, int num_arms, size_t budget,
                                                   size_t stale_budget, uint32_t base_seed,
                                                   bool minimize, MakeRunArm make_run_arm,
                                                   LogArm log_arm) {
    ContinuousLoopState loop;

    highs::parallel::for_each(
        0, static_cast<HighsInt>(num_workers),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * kSeedStride);
                auto run_arm = make_run_arm(static_cast<int>(w));
                int attempt_counter = 0;

                while (!loop.stopped()) {
                    if (w == 0 && attempt_counter % 8 == 0) {
                        loop.poll_termination(mipsolver);
                    }
                    if (loop.stopped()) {
                        break;
                    }

                    int arm = bandit.select_effort_aware(rng);
                    auto before = pool.snapshot();

                    size_t remaining =
                        budget -
                        std::min(budget, loop.total_effort.load(std::memory_order_relaxed));
                    size_t arm_budget =
                        std::min(compute_budget_cap(bandit, arm, budget, num_arms), remaining);
                    if (arm_budget == 0) {
                        loop.request_stop();
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

                    // Order preserved from pre-factor inline code:
                    //   1. note_staleness — bandit bails on stale budget *before*
                    //      the zero-effort check, so a zero-effort attempt still
                    //      accumulates staleness for peer workers.
                    //   2. zero-effort guard — breaks *this* worker's loop
                    //      without bumping total_effort.  Request global stop
                    //      so peers don't keep running past a solver timeout
                    //      between worker-0's next polling tick.
                    //   3. add_effort — only non-zero effort grows the budget.
                    loop.note_staleness(result.effort, /*improved=*/reward >= 2, stale_budget);
                    if (result.effort == 0) {
                        loop.request_stop();
                        break;
                    }
                    loop.add_effort(result.effort, budget);
                }
            }
        },
        1);

    return loop.total_effort.load(std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Deterministic bandit dispatch (epoch-gated)
// -----------------------------------------------------------------------------
//
// Counterpart to `run_bandit_opportunistic_loop` for the deterministic
// portfolio path.  The caller still drives `run_epoch_loop` directly
// (because worker construction and per-worker state differ), but the
// restart callback — "compute reward from pre/post snapshot, update
// bandit, pick next arm, reassign" — is the same for both portfolios
// and lives here.
//
// Workers cooperating with this callback must expose the portfolio-
// side contract below in addition to the `EpochWorker` concept.

template <typename T>
concept BanditWorker = requires(T w, int arm, SolutionPool::Snapshot snap) {
    { w.assigned_arm() } -> std::convertible_to<int>;
    { w.last_result() } -> std::convertible_to<const HeuristicResult &>;
    { w.last_wall_ms() } -> std::convertible_to<double>;
    { w.pre_snapshot() } -> std::convertible_to<SolutionPool::Snapshot>;
    { w.set_pre_snapshot(snap) } -> std::same_as<void>;
    { w.assign_arm(arm) } -> std::same_as<void>;
};

// Build the epoch-boundary restart callback shared by
// `portfolio::run_presolve` (det) and
// `fpr_lp::run_portfolio_deterministic`.
//
// `arm_name_fn(int arm)` returns a C-string name for logging.
// `tag` is the log tag ("Portfolio" or "FprLpPortfolio").  Logging
// only fires when the prior epoch actually ran (i.e. the worker had
// a previously assigned arm); this avoids a spurious zero-effort log
// line for workers whose initial arm assignment comes from the
// callback itself.
template <BanditWorker W, typename ArmNameFn>
auto make_bandit_restart_callback(ThompsonSampler &bandit, SolutionPool &pool,
                                  std::vector<std::unique_ptr<W>> &workers,
                                  std::vector<std::mt19937> &bandit_rngs, bool minimize,
                                  const HighsLogOptions &log_options, const char *tag,
                                  ArmNameFn arm_name_fn) {
    // The returned lambda outlives this factory's stack frame, so
    // captures must not bind to local parameter aliases.  The explicit
    // capture list below binds reference parameters (`bandit`, `pool`,
    // `workers`, `bandit_rngs`, `log_options`) to their caller-owned
    // referents by reference, and captures value-typed parameters
    // (`minimize`, `tag`) by value; `arm_name_fn` is moved in via init
    // capture.  Do not switch this to default-capture `[&]`: that would
    // capture `minimize` and `tag` by reference to the local parameter
    // slots, which dangle once this factory returns.
    return [&bandit, &pool, &workers, &bandit_rngs, &log_options, minimize, tag,
            arm_name_fn = std::move(arm_name_fn)](int w) mutable {
        auto &worker = *workers[w];
        const int prev_arm = worker.assigned_arm();
        if (prev_arm >= 0) {
            auto after_snap = pool.snapshot();
            const auto &result = worker.last_result();
            int reward = compute_reward(worker.pre_snapshot(), after_snap, result, minimize);
            bandit.update(prev_arm, reward);
            bandit.record_effort(prev_arm, result.effort);
            if (result.effort > 0) {
                log_bandit_arm(log_options, tag, arm_name_fn(prev_arm), result.effort,
                               worker.last_wall_ms(), reward);
            }
        }
        worker.set_pre_snapshot(pool.snapshot());
        int next_arm = bandit.select_effort_aware(bandit_rngs[w]);
        worker.assign_arm(next_arm);
    };
}
