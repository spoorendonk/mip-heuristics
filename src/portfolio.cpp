#include "portfolio.h"

#include "contested_pdlp.h"
#include "epoch_runner.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "io/HighsIO.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "scylla_worker.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace portfolio {

namespace {

// Arm indices for presolve portfolio (used as arm_type).
// FPR arms 0-5 correspond to the paper's 6 LP-free configs.
enum PresolveArm {
    kArmFprDfsBadobjcl = 0,
    kArmFprDfsLocks2,
    kArmFprDiveLocks2,
    kArmFprDfsrepLocks,
    kArmFprDfsrepBadobjcl,
    kArmFprDivepropRandom,
    kArmFprRepairSearchLocks,
    kArmLocalMIP,
    kArmFJ,
    kArmScylla,
};

// Strategy configs for each FPR arm (matching fpr.cpp's kLpFreeConfigs)
struct FprArmConfig {
    int arm_id;
    FprStrategyConfig strat;
    FrameworkMode mode;
};
constexpr FprArmConfig kFprArms[] = {
    {kArmFprDfsBadobjcl, kStratBadobjcl, FrameworkMode::kDfs},
    {kArmFprDfsLocks2, kStratLocks2, FrameworkMode::kDfs},
    {kArmFprDiveLocks2, kStratLocks2, FrameworkMode::kDive},
    {kArmFprDfsrepLocks, kStratLocks, FrameworkMode::kDfsrep},
    {kArmFprDfsrepBadobjcl, kStratBadobjcl, FrameworkMode::kDfsrep},
    {kArmFprDivepropRandom, kStratRandom, FrameworkMode::kDiveprop},
    {kArmFprRepairSearchLocks, kStratLocks, FrameworkMode::kRepairSearch},
};
constexpr int kNumFprArms = static_cast<int>(std::size(kFprArms));

// Returns the FprArmConfig for the given arm_type, or nullptr if not FPR.
// Arm IDs are contiguous 0..kNumFprArms-1, so direct index suffices.
const FprArmConfig *find_fpr_arm(int arm_type) {
    if (arm_type >= 0 && arm_type < kNumFprArms) {
        return &kFprArms[arm_type];
    }
    return nullptr;
}

// Uniform priors (α=1, β=1): let the bandit learn from scratch.
constexpr double kFjAlpha = 1.0;
constexpr double kFprArmAlpha = 1.0;
constexpr double kLocalMipAlpha = 1.0;
constexpr double kScyllaAlpha = 1.0;

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
        // First feasible ever
        return after.has_solution &&
                       !objective_better(minimize, after.best_objective, result.objective)
                   ? 2
                   : 1;
    }
    // Had solution before — check if we improved global best
    bool improved = after.has_solution &&
                    objective_better(minimize, after.best_objective, before.best_objective) &&
                    !objective_better(minimize, after.best_objective, result.objective);
    return improved ? 3 : 1;
}

constexpr const char *kArmNames[] = {
    "FprDfsBadobjcl",        // kArmFprDfsBadobjcl
    "FprDfsLocks2",          // kArmFprDfsLocks2
    "FprDiveLocks2",         // kArmFprDiveLocks2
    "FprDfsrepLocks",        // kArmFprDfsrepLocks
    "FprDfsrepBadobjcl",     // kArmFprDfsrepBadobjcl
    "FprDivepropRandom",     // kArmFprDivepropRandom
    "FprRepairSearchLocks",  // kArmFprRepairSearchLocks
    "LocalMIP",              // kArmLocalMIP
    "FJ",                    // kArmFJ
    "Scylla",                // kArmScylla
};
static_assert(std::size(kArmNames) == kArmScylla + 1, "kArmNames must match PresolveArm enum");

const char *arm_name(int arm_type) {
    if (arm_type >= 0 && arm_type < static_cast<int>(std::size(kArmNames))) {
        return kArmNames[arm_type];
    }
    return "Unknown";
}

void log_arm_effort(const HighsLogOptions &log_options, int arm_type, size_t effort,
                    double wall_ms) {
    double effort_per_ms = wall_ms > 0.0 ? static_cast<double>(effort) / wall_ms : 0.0;
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[Portfolio] arm=%s effort=%zu wall_ms=%.1f "
                "effort_per_ms=%.0f\n",
                arm_name(arm_type), effort, wall_ms, effort_per_ms);
}

// Pre-computed variable orders for FPR arms (avoids cliquePartition data race).
// Indexed by FprArmConfig index (0..kNumFprArms-1), not by arm_type enum.
using FprVarOrders = std::vector<std::vector<HighsInt>>;

FprVarOrders precompute_fpr_var_orders(const HighsMipSolver &mipsolver);

// Setup state shared between deterministic and opportunistic portfolio modes.
struct PresolveSetup {
    std::vector<int> enabled_arms;
    std::vector<double> priors;
    CscMatrix csc;
    FprVarOrders fpr_var_orders;
    std::vector<double> incumbent_snapshot;
    size_t budget;
    size_t stale_budget;
    bool minimize;
    // Shared Scylla PDLP solver; non-null when the Scylla arm is
    // enabled and the instance has non-trivial structure.  All
    // PortfolioWorkers that build a ScyllaWorker route their PDLP solves
    // through this instance's internal mutex, so at most one PDLP
    // solve is in flight at a time across the whole portfolio.
    std::unique_ptr<ContestedPdlp> scylla_pdlp;
};

std::optional<PresolveSetup> build_presolve_setup(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const auto *options = mipsolver.options_mip_;
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return std::nullopt;
    }

    PresolveSetup s;
    s.minimize = (model->sense_ == ObjSense::kMinimize);
    s.budget = max_effort;
    s.stale_budget = max_effort >> 2;

    if (options->mip_heuristic_run_feasibility_jump) {
        s.enabled_arms.push_back(kArmFJ);
        s.priors.push_back(kFjAlpha);
    }
    if (options->mip_heuristic_run_fpr) {
        for (int i = 0; i < kNumFprArms; ++i) {
            s.enabled_arms.push_back(kFprArms[i].arm_id);
            s.priors.push_back(kFprArmAlpha);
        }
    }
    if (options->mip_heuristic_run_local_mip) {
        s.enabled_arms.push_back(kArmLocalMIP);
        s.priors.push_back(kLocalMipAlpha);
    }
    // Scylla arm: attempt to build the shared ContestedPdlp first.  If
    // the instance lacks the structure PDLP needs (no rows or zero
    // nonzeros) the arm is silently omitted from the portfolio so that
    // no bandit pull can ever land on a no-op path.
    if (options->mip_heuristic_run_scylla) {
        const size_t nnz_lp = mipdata->ARindex_.size();
        HighsInt pdlp_iter_cap = 100;
        if (nnz_lp > 0) {
            auto cap = static_cast<HighsInt>((max_effort >> 2) / nnz_lp);
            pdlp_iter_cap = cap < 100 ? 100 : cap;
        }
        auto pdlp = std::make_unique<ContestedPdlp>(mipsolver, pdlp_iter_cap);
        if (pdlp->initialized()) {
            s.scylla_pdlp = std::move(pdlp);
            s.enabled_arms.push_back(kArmScylla);
            s.priors.push_back(kScyllaAlpha);
        }
    }
    if (s.enabled_arms.empty()) {
        return std::nullopt;
    }

    s.csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    if (options->mip_heuristic_run_fpr) {
        s.fpr_var_orders = precompute_fpr_var_orders(mipsolver);
    }

    s.incumbent_snapshot = mipdata->incumbent;
    return s;
}

FprVarOrders precompute_fpr_var_orders(const HighsMipSolver &mipsolver) {
    FprVarOrders orders(kNumFprArms);
    for (int i = 0; i < kNumFprArms; ++i) {
        std::mt19937 rng(42 + static_cast<uint32_t>(i));
        orders[i] = compute_var_order(mipsolver, kFprArms[i].strat.var_strategy, rng, nullptr);
    }
    return orders;
}

HeuristicResult run_presolve_arm(HighsMipSolver &mipsolver, int arm_type, std::mt19937 &rng,
                                 int attempt_idx, const double *restart_sol, const CscMatrix &csc,
                                 const std::vector<double> &incumbent_snapshot, size_t max_effort,
                                 const FprVarOrders &fpr_var_orders) {
    if (mipsolver.mipdata_->terminatorTerminated()) {
        return {};
    }
    if (mipsolver.timer_.read() >= mipsolver.options_mip_->time_limit) {
        return {};
    }
    // Check if this is an FPR config arm
    const FprArmConfig *fpr_arm = find_fpr_arm(arm_type);
    if (fpr_arm) {
        // Find the index into kFprArms for this arm
        int fpr_idx = static_cast<int>(fpr_arm - kFprArms);

        FprConfig cfg{};
        cfg.max_effort = max_effort;
        cfg.hint = incumbent_snapshot.empty() ? nullptr : incumbent_snapshot.data();
        cfg.scores = nullptr;
        cfg.cont_fallback = nullptr;
        cfg.csc = &csc;
        cfg.mode = fpr_arm->mode;
        cfg.strategy = &fpr_arm->strat;
        cfg.lp_ref = nullptr;
        // Use pre-computed var order to avoid cliquePartition data race
        cfg.precomputed_var_order = fpr_var_orders[fpr_idx].data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(fpr_var_orders[fpr_idx].size());
        return fpr_attempt(mipsolver, cfg, rng, attempt_idx, restart_sol);
    }

    switch (arm_type) {
        case kArmLocalMIP: {
            const double *init = restart_sol;
            if (!init && !incumbent_snapshot.empty()) {
                init = incumbent_snapshot.data();
            }
            uint32_t seed = static_cast<uint32_t>(rng());
            return local_mip::worker(mipsolver, csc, seed, init, max_effort);
        }
        case kArmFJ: {
            auto *mipdata = mipsolver.mipdata_.get();
            const HighsInt ncol = mipsolver.model_->num_col_;
            HeuristicResult result;
            std::vector<double> captured_sol;
            double captured_obj = 0.0;

            // Prefer pool restart over static incumbent snapshot.
            // restart_sol is a raw pointer from the pool, so we must copy into a
            // vector to satisfy the FJ signature (const vector<double>*).
            const std::vector<double> *hint = nullptr;
            std::vector<double> restart_vec;
            if (restart_sol) {
                restart_vec.assign(restart_sol, restart_sol + ncol);
                hint = &restart_vec;
            } else if (!incumbent_snapshot.empty()) {
                hint = &incumbent_snapshot;
            }

            size_t fj_effort = 0;
            mipdata->feasibilityJumpCapture(captured_sol, captured_obj, fj_effort, max_effort,
                                            hint);
            if (!captured_sol.empty()) {
                result.found_feasible = true;
                result.solution = std::move(captured_sol);
                result.objective = captured_obj;
            }
            result.effort = fj_effort;
            return result;
        }
        default:
            return {};
    }
}

// ── PortfolioWorker: EpochWorker wrapper for bandit-selected arms ──
//
// Each worker runs whichever arm the bandit assigns.  Non-Scylla arms
// are stateless (one invocation per epoch, then finished=true triggers
// reassignment).  Scylla arms delegate to a persistent ScyllaWorker that
// preserves PDLP warm-start state across epochs.
class PortfolioWorker {
public:
    PortfolioWorker(HighsMipSolver &mipsolver, const PresolveSetup &setup, SolutionPool &pool,
                    uint32_t seed, int worker_idx)
        : mipsolver_(mipsolver), setup_(setup), pool_(pool), rng_(seed), worker_idx_(worker_idx) {}

    EpochResult run_epoch(size_t epoch_budget) {
        if (finished_) {
            return {};
        }

        const int arm_type = setup_.enabled_arms[assigned_arm_];
        EpochResult epoch{};

        if (arm_type == kArmScylla) {
            if (!pump_ || pump_->finished()) {
                pump_ = std::make_unique<ScyllaWorker>(
                    mipsolver_, *setup_.scylla_pdlp, setup_.csc, pool_, setup_.budget,
                    static_cast<uint32_t>(rng_()), worker_idx_, highs::parallel::num_threads());
            }
            auto t0 = std::chrono::steady_clock::now();
            epoch = pump_->run_epoch(epoch_budget);
            auto t1 = std::chrono::steady_clock::now();
            finished_ = pump_->finished();
            // Accumulate HeuristicResult across Scylla epochs for reward computation.
            // (last_result_ is reset in assign_arm(), not here.)
            last_result_.effort += epoch.effort;
            if (epoch.found_improvement) {
                last_result_.found_feasible = true;
                last_result_.objective = pool_.snapshot().best_objective;
            }
            if (epoch.effort > 0) {
                double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                log_arm_effort(mipsolver_.options_mip_->log_options, arm_type, epoch.effort,
                               wall_ms);
            }
        } else {
            std::vector<double> restart;
            pool_.get_restart(rng_, restart);
            const double *restart_ptr = restart.empty() ? nullptr : restart.data();

            auto t0 = std::chrono::steady_clock::now();
            last_result_ = run_presolve_arm(mipsolver_, arm_type, rng_, attempt_counter_++,
                                            restart_ptr, setup_.csc, setup_.incumbent_snapshot,
                                            epoch_budget, setup_.fpr_var_orders);
            auto t1 = std::chrono::steady_clock::now();

            epoch.effort = last_result_.effort;
            if (last_result_.found_feasible) {
                pool_.try_add(last_result_.objective, last_result_.solution);
                epoch.found_improvement = true;
            }

            if (last_result_.effort > 0) {
                double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                log_arm_effort(mipsolver_.options_mip_->log_options, arm_type, last_result_.effort,
                               wall_ms);
            }

            // Non-Scylla arms: mark finished so restart callback reassigns.
            finished_ = true;
        }

        return epoch;
    }

    bool finished() const { return finished_; }

    void reset_staleness() {
        if (pump_) {
            pump_->reset_staleness();
        }
    }

    // --- Portfolio-specific interface (used by restart callback) ---

    void assign_arm(int bandit_arm_idx) {
        assigned_arm_ = bandit_arm_idx;
        finished_ = false;
        last_result_ = {};
    }

    void set_pre_snapshot(SolutionPool::Snapshot snap) { pre_snap_ = snap; }

    int assigned_arm() const { return assigned_arm_; }
    const HeuristicResult &last_result() const { return last_result_; }
    SolutionPool::Snapshot pre_snapshot() const { return pre_snap_; }

private:
    HighsMipSolver &mipsolver_;
    const PresolveSetup &setup_;
    SolutionPool &pool_;
    std::mt19937 rng_;
    int worker_idx_ = 0;

    int assigned_arm_ = -1;
    int attempt_counter_ = 0;
    bool finished_ = true;  // starts finished → restart fires on epoch 0

    HeuristicResult last_result_{};
    SolutionPool::Snapshot pre_snap_{};

    std::unique_ptr<ScyllaWorker> pump_;  // lazy-init, persists across reassignments
};

static_assert(EpochWorker<PortfolioWorker>, "PortfolioWorker must satisfy EpochWorker concept");

// Effort-proportional budget cap for opportunistic arm pulls.
// Each pull gets at most kBudgetCapMultiplier * avg_effort for that arm.
// First pull (no history) uses total_budget / (num_arms * 10).
constexpr double kBudgetCapMultiplier = 2.5;

size_t compute_budget_cap(const ThompsonSampler &bandit, int arm, size_t total_budget,
                          int num_arms) {
    auto stats = bandit.stats(arm);
    if (stats.pulls > 0 && stats.avg_effort > 0.0) {
        return static_cast<size_t>(kBudgetCapMultiplier * stats.avg_effort);
    }
    return total_budget / static_cast<size_t>(std::max(num_arms * 10, 1));
}

void run_presolve_opportunistic(HighsMipSolver &mipsolver, const PresolveSetup &setup) {
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsLogOptions &log_options = mipsolver.options_mip_->log_options;
    const int N = highs::parallel::num_threads();
    const int num_arms = static_cast<int>(setup.enabled_arms.size());

    ThompsonSampler bandit(num_arms, setup.priors.data(), true);
    SolutionPool pool(kPoolCapacity, setup.minimize);
    seed_pool(pool, mipsolver);

    const auto &enabled_arms = setup.enabled_arms;
    const auto &incumbent_snapshot = setup.incumbent_snapshot;
    const auto &csc = setup.csc;
    const auto &fpr_var_orders = setup.fpr_var_orders;
    const bool minimize = setup.minimize;
    const size_t budget = setup.budget;
    const size_t stale_budget = setup.stale_budget;

    const double time_limit = mipsolver.options_mip_->time_limit;

    std::atomic<size_t> total_effort{0};
    std::atomic<size_t> effort_since_improvement{0};
    std::atomic<bool> stop{false};

    uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);

    highs::parallel::for_each(
        0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(base_seed + static_cast<uint32_t>(w) * kSeedStride);
                int attempt_counter = 0;
                std::unique_ptr<ScyllaWorker> pump;

                while (!stop.load(std::memory_order_relaxed)) {
                    // Worker 0 periodically checks termination (not thread-safe
                    // to call from multiple workers)
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

                    HeuristicResult result{};
                    auto t0 = std::chrono::steady_clock::now();

                    if (enabled_arms[arm] == kArmScylla) {
                        // Scylla: lazy-init ScyllaWorker, preserves PDLP warm-start.
                        // All workers share the single setup.scylla_pdlp
                        // instance; its mutex serializes PDLP runs.  The arm
                        // is only present when the shared PDLP initialized,
                        // so the dereference is always safe.
                        if (!pump || pump->finished()) {
                            pump = std::make_unique<ScyllaWorker>(
                                mipsolver, *setup.scylla_pdlp, csc, pool, budget,
                                static_cast<uint32_t>(rng()), static_cast<int>(w), N);
                        }
                        auto epoch = pump->run_epoch(arm_budget);
                        result.effort = epoch.effort;
                        if (epoch.found_improvement) {
                            result.found_feasible = true;
                            result.objective = pool.snapshot().best_objective;
                        }
                    } else {
                        std::vector<double> restart;
                        pool.get_restart(rng, restart);
                        const double *restart_ptr = restart.empty() ? nullptr : restart.data();

                        result = run_presolve_arm(mipsolver, enabled_arms[arm], rng,
                                                  attempt_counter++, restart_ptr, csc,
                                                  incumbent_snapshot, arm_budget, fpr_var_orders);

                        if (result.found_feasible) {
                            pool.try_add(result.objective, result.solution);
                        }
                    }

                    auto t1 = std::chrono::steady_clock::now();

                    auto after = pool.snapshot();
                    int reward = compute_reward(before, after, result, minimize);
                    bandit.update(arm, reward);
                    bandit.record_effort(arm, result.effort);

                    if (result.effort > 0) {
                        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                        log_arm_effort(log_options, enabled_arms[arm], result.effort, wall_ms);
                    }

                    if (reward >= 2) {
                        effort_since_improvement.store(0, std::memory_order_relaxed);
                        // Reset staleness on all workers' ScyllaWorkers would require
                        // cross-worker access; each ScyllaWorker tracks its own.
                    } else {
                        effort_since_improvement.fetch_add(result.effort,
                                                           std::memory_order_relaxed);
                    }

                    if (effort_since_improvement.load(std::memory_order_relaxed) >= stale_budget) {
                        stop.store(true, std::memory_order_relaxed);
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

    // Flush pool solutions to HiGHS (sequential, use generic H tag since
    // pool mixes arms)
    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
    }
}

}  // namespace

void run_presolve(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic) {
    auto setup_opt = build_presolve_setup(mipsolver, max_effort);
    if (!setup_opt) {
        return;
    }
    auto &setup = *setup_opt;

    // Dispatch to opportunistic mode if requested.  The opportunistic flag
    // is passed in from mode_dispatch::run_presolve, which reads the new
    // orthogonal `mip_heuristic_opportunistic` option.
    if (opportunistic) {
        run_presolve_opportunistic(mipsolver, setup);
        return;
    }

    auto *mipdata = mipsolver.mipdata_.get();
    const int N = highs::parallel::num_threads();
    const int num_arms = static_cast<int>(setup.enabled_arms.size());
    const bool minimize = setup.minimize;
    const HighsLogOptions &log_options = mipsolver.options_mip_->log_options;

    ThompsonSampler bandit(num_arms, setup.priors.data(), /*use_mutex=*/false);
    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    uint32_t base_seed = static_cast<uint32_t>(mipdata->numImprovingSols + kBaseSeedOffset);

    // Separate RNGs for bandit selection in the restart callback (sequential).
    std::vector<std::mt19937> bandit_rngs(N);
    for (int w = 0; w < N; ++w) {
        bandit_rngs[w].seed(base_seed + static_cast<uint32_t>(w) * kSeedStride + 1);
    }

    // Workers start with finished_=true so the first restart callback assigns
    // their initial arms.
    std::vector<std::unique_ptr<PortfolioWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        workers.push_back(std::make_unique<PortfolioWorker>(mipsolver, setup, pool, seed, w));
    }

    constexpr int kEpochsPerWorker = 20;
    const size_t epoch_budget =
        std::max<size_t>(setup.budget / (static_cast<size_t>(N) * kEpochsPerWorker), 1);

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, setup.budget, epoch_budget,
        [&](int w) {
            auto &worker = *workers[w];
            // Update bandit from previous epoch (skip initial assignment).
            if (worker.assigned_arm() >= 0) {
                auto after_snap = pool.snapshot();
                int reward = compute_reward(worker.pre_snapshot(), after_snap, worker.last_result(),
                                            minimize);
                bandit.update(worker.assigned_arm(), reward);
                bandit.record_effort(worker.assigned_arm(), worker.last_result().effort);
            }
            // Snapshot pool before next epoch (sequential, deterministic).
            worker.set_pre_snapshot(pool.snapshot());
            // Select next arm.
            int arm = bandit.select_effort_aware(bandit_rngs[w]);
            worker.assign_arm(arm);
        },
        setup.stale_budget);

    mipdata->heuristic_effort_used += total_effort;

    // Flush pool solutions to HiGHS (best first).
    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceHeuristic);
    }
}

}  // namespace portfolio
