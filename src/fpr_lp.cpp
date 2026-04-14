#include "fpr_lp.h"

#include "epoch_runner.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "io/HighsIO.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace fpr_lp {

namespace {

// ---------------------------------------------------------------------------
// LP-dependent arms (paper Section 6.3, Classes 2 and 3)
// ---------------------------------------------------------------------------
//
// Each arm is a (strategy, framework mode) pair bound to a reference LP
// solution:
//   Class 2  — zero-obj LP strategies — analytic center  (ac_ptr)
//   Class 3a — zerolp configs         — zero-obj vertex  (zv_ptr)
//   Class 3b — lp/cliques2 configs    — full-obj LP      (lp_ptr)

constexpr NamedConfig kClass2Configs[] = {
    {kStratZerocore, FrameworkMode::kDfs},
    {kStratZerocore, FrameworkMode::kDive},
    {kStratZerocore, FrameworkMode::kDiveprop},
    {kStratCliques, FrameworkMode::kDfs},  // paper: "if predominant clique
                                           // structure"; run unconditionally,
                                           // degrades gracefully on non-clique
                                           // models
};
constexpr int kNumClass2 = static_cast<int>(std::size(kClass2Configs));

constexpr NamedConfig kClass3aConfigs[] = {
    {kStratZerolp, FrameworkMode::kDfs},
    {kStratZerolp, FrameworkMode::kDiveprop},
};
constexpr int kNumClass3a = static_cast<int>(std::size(kClass3aConfigs));

constexpr NamedConfig kClass3bConfigs[] = {
    {kStratCliques2, FrameworkMode::kDiveprop},
    {kStratLp, FrameworkMode::kDfs},
    {kStratLp, FrameworkMode::kDive},
    {kStratLp, FrameworkMode::kDiveprop},
};
constexpr int kNumClass3b = static_cast<int>(std::size(kClass3bConfigs));

constexpr int kNumLpArms = kNumClass2 + kNumClass3a + kNumClass3b;

constexpr const char *kLpArmNames[] = {
    "ZerocoreDfs",       // Class 2
    "ZerocoreDive",      // Class 2
    "ZerocoreDiveprop",  // Class 2
    "CliquesDfs",        // Class 2
    "ZerolpDfs",         // Class 3a
    "ZerolpDiveprop",    // Class 3a
    "Cliques2Diveprop",  // Class 3b
    "LpDfs",             // Class 3b
    "LpDive",            // Class 3b
    "LpDiveprop",        // Class 3b
};
static_assert(std::size(kLpArmNames) == kNumLpArms, "kLpArmNames must match total LP arm count");

// An arm binds a NamedConfig to the LP reference pointer it requires.
struct LpArm {
    const NamedConfig *config;
    const double *lp_ref;
};

// Max workers for parallel LP-FPR modes.
constexpr int kMaxLpFprWorkers = 8;

// Effort-proportional budget cap for opportunistic arm pulls.
// Mirrors portfolio.cpp::kBudgetCapMultiplier.
constexpr double kBudgetCapMultiplier = 2.5;

// ---------------------------------------------------------------------------
// Shared setup: LP references, CSC matrix, precomputed var_orders
// ---------------------------------------------------------------------------

using VarOrderTable = std::vector<std::vector<HighsInt>>;

struct LpFprSetup {
    // Combined arm table (Class 2 + 3a + 3b, in that order).  Owning.
    std::vector<LpArm> arms;

    // Per-arm variable orderings, precomputed sequentially before any
    // parallel region to avoid races on HighsCliqueTable::cliquePartition.
    VarOrderTable var_orders;

    // CSC matrix of the model — built once, shared read-only.
    CscMatrix csc;

    // LP reference vectors.  Owned here so raw pointers stored in `arms`
    // remain valid for the lifetime of the setup.  Each may be empty if
    // the corresponding LP computation failed; in that case the pointer
    // fallback is the full-obj LP solution.
    std::vector<double> analytic_center;  // ac_ptr source (Class 2)
    std::vector<double> zero_vertex;      // zv_ptr source (Class 3a)

    // Incumbent hint (copy — snapshot taken before dispatch to keep the
    // pointer stable while mipdata->incumbent may be mutated by HiGHS).
    std::vector<double> incumbent_snapshot;

    size_t budget = 0;
    size_t stale_budget = 0;
    bool minimize = true;
};

// Build the shared LP-FPR setup.  Returns nullopt when the model is
// empty or the LP relaxation is not at an optimal scaled state (the
// caller should skip LP-FPR entirely in that case).
std::optional<LpFprSetup> build_setup(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return std::nullopt;
    }

    auto lp_status = mipdata->lp.getStatus();
    if (!HighsLpRelaxation::scaledOptimal(lp_status)) {
        return std::nullopt;
    }

    LpFprSetup s;
    s.minimize = (model->sense_ == ObjSense::kMinimize);
    s.budget = max_effort;
    s.stale_budget = max_effort >> 2;

    s.csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    s.incumbent_snapshot = mipdata->incumbent;

    // Full-obj LP solution — direct reference to the solver's col_value
    // vector (stable while we run because we do not trigger further LP
    // solves during LP-FPR).
    const auto &lp_sol = mipdata->lp.getLpSolver().getSolution().col_value;
    const double *lp_ptr = lp_sol.data();

    // Zero-obj analytic center (for Class 2 zerocore strategies).
    s.analytic_center = compute_analytic_center(mipsolver, /*use_objective=*/false);
    const double *ac_ptr = s.analytic_center.empty() ? lp_ptr : s.analytic_center.data();

    // Zero-obj LP vertex (for Class 3a zerolp strategies).
    s.zero_vertex = compute_zero_obj_vertex(mipsolver);
    const double *zv_ptr = s.zero_vertex.empty() ? lp_ptr : s.zero_vertex.data();

    s.arms.reserve(kNumLpArms);
    for (int i = 0; i < kNumClass2; ++i) {
        s.arms.push_back({&kClass2Configs[i], ac_ptr});
    }
    for (int i = 0; i < kNumClass3a; ++i) {
        s.arms.push_back({&kClass3aConfigs[i], zv_ptr});
    }
    for (int i = 0; i < kNumClass3b; ++i) {
        s.arms.push_back({&kClass3bConfigs[i], lp_ptr});
    }

    // Precompute var_orders sequentially — required before any parallel
    // region because clique-based var_strategies call
    // HighsCliqueTable::cliquePartition which mutates internal state.
    s.var_orders.resize(kNumLpArms);
    for (int i = 0; i < kNumLpArms; ++i) {
        std::mt19937 rng(kBaseSeedOffset + static_cast<uint32_t>(i) + 200);
        s.var_orders[i] = compute_var_order(mipsolver, s.arms[i].config->strat.var_strategy, rng,
                                            s.arms[i].lp_ref);
    }

    return s;
}

// ---------------------------------------------------------------------------
// Bandit reward helpers (shared with the portfolio paths)
// ---------------------------------------------------------------------------

bool objective_better(bool minimize, double lhs, double rhs) {
    constexpr double kTol = 1e-9;
    return minimize ? lhs < rhs - kTol : lhs > rhs + kTol;
}

int compute_reward(SolutionPool::Snapshot before, SolutionPool::Snapshot after,
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

size_t compute_opportunistic_budget_cap(const ThompsonSampler &bandit, int arm, size_t total_budget,
                                        int num_arms) {
    auto stats = bandit.stats(arm);
    if (stats.pulls > 0 && stats.avg_effort > 0.0) {
        return static_cast<size_t>(kBudgetCapMultiplier * stats.avg_effort);
    }
    return total_budget / static_cast<size_t>(std::max(num_arms * 10, 1));
}

}  // namespace

// ---------------------------------------------------------------------------
// LpFprWorker: EpochWorker that runs one LP-dependent FPR arm at a time
// ---------------------------------------------------------------------------

class LpFprWorker {
public:
    LpFprWorker(HighsMipSolver &mipsolver, const LpFprSetup &setup, SolutionPool &pool, int arm_idx,
                uint32_t seed, bool one_shot = false)
        : mipsolver_(mipsolver),
          setup_(setup),
          pool_(pool),
          arm_idx_(arm_idx),
          one_shot_(one_shot),
          rng_(seed) {}

    EpochResult run_epoch(size_t epoch_budget) {
        EpochResult epoch{};

        // Persistent (non-portfolio) mode only: after K stale epochs
        // randomize to another LP arm from the full 10-element pool.
        // var_orders are precomputed for every arm so the switch is
        // race-free.  In one-shot (portfolio) mode the bandit picks
        // the next arm via assign_arm() — never touch arm_idx_ here.
        if (!one_shot_ && epochs_without_improvement_ >= kStaleEpochThreshold) {
            randomize_arm();
            epochs_without_improvement_ = 0;
        }

        std::vector<double> initial_solution;
        const double *init_ptr = nullptr;
        if (pool_.get_restart(rng_, initial_solution)) {
            init_ptr = initial_solution.data();
        }

        const LpArm &arm = setup_.arms[arm_idx_];
        const auto &var_order = setup_.var_orders[arm_idx_];

        FprConfig cfg{};
        cfg.max_effort = epoch_budget;
        cfg.hint = setup_.incumbent_snapshot.empty() ? nullptr : setup_.incumbent_snapshot.data();
        cfg.scores = nullptr;
        cfg.cont_fallback = nullptr;
        cfg.csc = &setup_.csc;
        cfg.mode = arm.config->mode;
        cfg.strategy = &arm.config->strat;
        cfg.lp_ref = arm.lp_ref;
        cfg.precomputed_var_order = var_order.data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order.size());

        last_result_ = fpr_attempt(mipsolver_, cfg, rng_, attempt_idx_, init_ptr);
        ++attempt_idx_;

        epoch.effort = last_result_.effort;

        if (last_result_.found_feasible) {
            pool_.try_add(last_result_.objective, last_result_.solution);
            epoch.found_improvement = true;
            epochs_without_improvement_ = 0;
        } else {
            ++epochs_without_improvement_;
            if (epochs_without_improvement_ >= kHardStaleThreshold) {
                finished_ = true;
            }
        }

        // One-shot portfolio mode: mark finished so run_epoch_loop's
        // restart callback fires and the bandit can reassign the next
        // arm.  Mirrors PortfolioWorker in portfolio.cpp.
        if (one_shot_) {
            finished_ = true;
        }

        return epoch;
    }

    bool finished() const { return finished_; }

    void reset_staleness() { epochs_without_improvement_ = 0; }

    // --- Portfolio-specific interface (used by restart callback) ---

    void assign_arm(int new_arm_idx) {
        arm_idx_ = new_arm_idx;
        finished_ = false;
        epochs_without_improvement_ = 0;
        last_result_ = {};
    }

    int arm_idx() const { return arm_idx_; }
    const HeuristicResult &last_result() const { return last_result_; }
    void set_pre_snapshot(SolutionPool::Snapshot snap) { pre_snap_ = snap; }
    SolutionPool::Snapshot pre_snapshot() const { return pre_snap_; }

private:
    void randomize_arm() { arm_idx_ = std::uniform_int_distribution<int>(0, kNumLpArms - 1)(rng_); }

    HighsMipSolver &mipsolver_;
    const LpFprSetup &setup_;
    SolutionPool &pool_;

    int arm_idx_;
    int attempt_idx_ = 0;
    int epochs_without_improvement_ = 0;
    bool finished_ = false;
    bool one_shot_ = false;

    HeuristicResult last_result_{};
    SolutionPool::Snapshot pre_snap_{};

    std::mt19937 rng_;

    // Hard stale threshold for LP-FPR workers in opportunistic mode.
    // Mirrors FprWorker::kHardStaleThreshold so the worker signals
    // "finished" to trigger replacement instead of spinning.
    static constexpr int kHardStaleThreshold = 15;

    // Number of stale epochs before a worker randomizes its arm.
    // Mirrors fpr.cpp's kStaleEpochThreshold.
    static constexpr int kStaleEpochThreshold = 3;
};

static_assert(EpochWorker<LpFprWorker>, "LpFprWorker must satisfy EpochWorker concept");

namespace {

// ---------------------------------------------------------------------------
// Worker count for LP-FPR parallel modes
// ---------------------------------------------------------------------------

int compute_worker_count(const HighsMipSolver &mipsolver) {
    const auto *model = mipsolver.model_;
    // Each LP-FPR worker carries a var_order vector (ncol HighsInt) plus
    // working FPR state; mirror fpr.cpp's estimate.
    const size_t per_worker_mem =
        static_cast<size_t>(model->num_col_) * (sizeof(HighsInt) + sizeof(double));
    const int mem_cap = max_workers_for_memory(per_worker_mem);
    return std::min({highs::parallel::num_threads(), kMaxLpFprWorkers, mem_cap});
}

uint32_t base_seed_for(const HighsMipSolver &mipsolver) {
    return static_cast<uint32_t>(mipsolver.mipdata_->numImprovingSols + kBaseSeedOffset);
}

// ---------------------------------------------------------------------------
// 4 entry points
// ---------------------------------------------------------------------------

void run_sequential_deterministic(HighsMipSolver &mipsolver, const LpFprSetup &setup,
                                  SolutionPool &pool) {
    // In seq/det we spawn one worker per LP arm (up to kNumLpArms) and
    // let run_epoch_loop drive them.  The number of threads caps total
    // parallelism; excess arms still get worker slots because each
    // worker is lightweight and runs sequentially inside its slot.
    auto *mipdata = mipsolver.mipdata_.get();
    const int N = std::min(kNumLpArms, compute_worker_count(mipsolver));
    if (N <= 0) {
        return;
    }

    const uint32_t base_seed = base_seed_for(mipsolver);

    std::vector<std::unique_ptr<LpFprWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<LpFprWorker>(mipsolver, setup, pool, w, seed));
    }

    constexpr int kEpochsPerWorker = 10;
    const size_t per_worker = setup.budget / static_cast<size_t>(N);
    const size_t epoch_budget = std::max<size_t>(per_worker / kEpochsPerWorker, 1);

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, setup.budget, epoch_budget,
        [](int) { /* LpFprWorkers rarely hit hard stale in det mode */ }, setup.stale_budget);

    mipdata->heuristic_effort_used += total_effort;
}

void run_sequential_opportunistic(HighsMipSolver &mipsolver, const LpFprSetup &setup,
                                  SolutionPool &pool) {
    auto *mipdata = mipsolver.mipdata_.get();
    const int N = compute_worker_count(mipsolver);
    if (N <= 0) {
        return;
    }

    const uint32_t base_seed = base_seed_for(mipsolver);
    const size_t default_run_cap =
        std::max<size_t>(setup.budget / (static_cast<size_t>(N) * 10), 1);

    // Per-worker lightweight state: just the LpFprWorker instance.
    struct LpFprOppState {
        std::unique_ptr<LpFprWorker> worker;
    };

    size_t total_effort = run_opportunistic_loop(
        mipsolver, N, setup.budget, setup.stale_budget, default_run_cap, base_seed,
        [&](int worker_idx, std::mt19937 & /*rng*/) -> LpFprOppState {
            // Initial arm is worker_idx modulo the arm pool.
            int arm = worker_idx % kNumLpArms;
            uint32_t seed = base_seed + static_cast<uint32_t>(worker_idx) * kSeedStride;
            return LpFprOppState{std::make_unique<LpFprWorker>(mipsolver, setup, pool, arm, seed)};
        },
        [&](LpFprOppState &state, std::mt19937 &rng, size_t run_cap) -> HeuristicResult {
            if (state.worker->finished()) {
                int arm = std::uniform_int_distribution<int>(0, kNumLpArms - 1)(rng);
                uint32_t seed = static_cast<uint32_t>(rng());
                state.worker = std::make_unique<LpFprWorker>(mipsolver, setup, pool, arm, seed);
            }
            auto epoch = state.worker->run_epoch(run_cap);
            HeuristicResult result;
            result.effort = epoch.effort;
            if (epoch.found_improvement) {
                result.found_feasible = true;
                result.objective = pool.snapshot().best_objective;
            }
            return result;
        });

    mipdata->heuristic_effort_used += total_effort;
}

void run_portfolio_deterministic(HighsMipSolver &mipsolver, const LpFprSetup &setup,
                                 SolutionPool &pool) {
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsLogOptions &log_options = mipsolver.options_mip_->log_options;
    const int N = compute_worker_count(mipsolver);
    if (N <= 0) {
        return;
    }
    const int num_arms = kNumLpArms;

    // Uniform priors (α=1): let the bandit learn from scratch.
    std::vector<double> priors(num_arms, 1.0);
    ThompsonSampler bandit(num_arms, priors.data(), /*use_mutex=*/false);

    const uint32_t base_seed = base_seed_for(mipsolver);

    // Separate RNGs for bandit selection in the restart callback (sequential).
    std::vector<std::mt19937> bandit_rngs(N);
    for (int w = 0; w < N; ++w) {
        bandit_rngs[w].seed(base_seed + static_cast<uint32_t>(w) * kSeedStride + 1);
    }

    std::vector<std::unique_ptr<LpFprWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<LpFprWorker>(mipsolver, setup, pool, /*arm_idx=*/0, seed,
                                                        /*one_shot=*/true));
    }

    // Seed each worker with an initial pre-snapshot and arm so the first
    // bandit reward is computed against a valid baseline.  The restart
    // callback then updates the bandit and reassigns on every subsequent
    // epoch boundary.
    for (int w = 0; w < N; ++w) {
        workers[w]->set_pre_snapshot(pool.snapshot());
        int arm = bandit.select_effort_aware(bandit_rngs[w]);
        workers[w]->assign_arm(arm);
    }

    constexpr int kEpochsPerWorker = 20;
    const size_t epoch_budget =
        std::max<size_t>(setup.budget / (static_cast<size_t>(N) * kEpochsPerWorker), 1);

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, setup.budget, epoch_budget,
        [&](int w) {
            auto &worker = *workers[w];
            // Update bandit from the just-finished epoch.
            auto after_snap = pool.snapshot();
            int reward = compute_reward(worker.pre_snapshot(), after_snap, worker.last_result(),
                                        setup.minimize);
            bandit.update(worker.arm_idx(), reward);
            bandit.record_effort(worker.arm_idx(), worker.last_result().effort);

            highsLogDev(log_options, HighsLogType::kVerbose,
                        "[FprLpPortfolio] arm=%s effort=%zu reward=%d\n",
                        kLpArmNames[worker.arm_idx()], worker.last_result().effort, reward);

            worker.set_pre_snapshot(pool.snapshot());
            int next_arm = bandit.select_effort_aware(bandit_rngs[w]);
            worker.assign_arm(next_arm);
        },
        setup.stale_budget);

    mipdata->heuristic_effort_used += total_effort;
}

void run_portfolio_opportunistic(HighsMipSolver &mipsolver, const LpFprSetup &setup,
                                 SolutionPool &pool) {
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsLogOptions &log_options = mipsolver.options_mip_->log_options;
    const int N = compute_worker_count(mipsolver);
    if (N <= 0) {
        return;
    }
    const int num_arms = kNumLpArms;

    std::vector<double> priors(num_arms, 1.0);
    ThompsonSampler bandit(num_arms, priors.data(), /*use_mutex=*/true);

    const uint32_t base_seed = base_seed_for(mipsolver);
    const double time_limit = mipsolver.options_mip_->time_limit;
    const bool minimize = setup.minimize;
    const size_t budget = setup.budget;
    const size_t stale_budget = setup.stale_budget;

    std::atomic<size_t> total_effort{0};
    std::atomic<size_t> effort_since_improvement{0};
    std::atomic<bool> stop{false};

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * kSeedStride);
                int attempt_counter = 0;

                while (!stop.load(std::memory_order_relaxed)) {
                    // Worker 0 polls termination every 8 attempts (timer
                    // and terminator are not thread-safe for concurrent
                    // callers).  Mirrors opportunistic_runner.h.
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

                    size_t current = total_effort.load(std::memory_order_relaxed);
                    size_t remaining = budget - std::min(budget, current);
                    size_t arm_budget = std::min(
                        compute_opportunistic_budget_cap(bandit, arm, budget, num_arms), remaining);
                    if (arm_budget == 0) {
                        stop.store(true, std::memory_order_relaxed);
                        break;
                    }

                    std::vector<double> restart;
                    pool.get_restart(rng, restart);
                    const double *restart_ptr = restart.empty() ? nullptr : restart.data();

                    const LpArm &lp_arm = setup.arms[arm];
                    const auto &var_order = setup.var_orders[arm];

                    FprConfig cfg{};
                    cfg.max_effort = arm_budget;
                    cfg.hint = setup.incumbent_snapshot.empty() ? nullptr
                                                                : setup.incumbent_snapshot.data();
                    cfg.scores = nullptr;
                    cfg.cont_fallback = nullptr;
                    cfg.csc = &setup.csc;
                    cfg.mode = lp_arm.config->mode;
                    cfg.strategy = &lp_arm.config->strat;
                    cfg.lp_ref = lp_arm.lp_ref;
                    cfg.precomputed_var_order = var_order.data();
                    cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order.size());

                    auto t0 = std::chrono::steady_clock::now();
                    auto result = fpr_attempt(mipsolver, cfg, rng, attempt_counter++, restart_ptr);
                    auto t1 = std::chrono::steady_clock::now();

                    if (result.found_feasible) {
                        pool.try_add(result.objective, result.solution);
                    }

                    auto after = pool.snapshot();
                    int reward = compute_reward(before, after, result, minimize);
                    bandit.update(arm, reward);
                    bandit.record_effort(arm, result.effort);

                    if (result.effort > 0) {
                        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                        highsLogDev(log_options, HighsLogType::kVerbose,
                                    "[FprLpPortfolio] arm=%s effort=%zu reward=%d wall_ms=%.1f\n",
                                    kLpArmNames[arm], result.effort, reward, wall_ms);
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

                    // Guard against zero-effort returns (prevents spin).
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

    mipdata->heuristic_effort_used += total_effort.load(std::memory_order_relaxed);
}

}  // namespace

void run(HighsMipSolver &mipsolver, size_t max_effort) {
    auto setup_opt = build_setup(mipsolver, max_effort);
    if (!setup_opt) {
        return;
    }
    auto &setup = *setup_opt;

    SolutionPool pool(kPoolCapacity, setup.minimize);
    seed_pool(pool, mipsolver);

    const auto *options = mipsolver.options_mip_;
    const bool portfolio = options->mip_heuristic_portfolio;
    const bool opportunistic = options->mip_heuristic_opportunistic;

    if (portfolio) {
        if (opportunistic) {
            run_portfolio_opportunistic(mipsolver, setup, pool);
        } else {
            run_portfolio_deterministic(mipsolver, setup, pool);
        }
    } else {
        if (opportunistic) {
            run_sequential_opportunistic(mipsolver, setup, pool);
        } else {
            run_sequential_deterministic(mipsolver, setup, pool);
        }
    }

    // Submit best solutions to solver (sequential).
    auto *mipdata = mipsolver.mipdata_.get();
    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceFPR);
    }
}

}  // namespace fpr_lp
