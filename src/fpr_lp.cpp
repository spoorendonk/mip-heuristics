#include "fpr_lp.h"

#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "io/HighsIO.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "solution_pool.h"
#include "thompson_sampler.h"

#include <random>
#include <vector>

namespace fpr_lp {

namespace {

// Paper Section 6.3, Class 2 — zero-obj LP strategies
constexpr NamedConfig kClass2Configs[] = {
    {kStratZerocore, FrameworkMode::kDfs},
    {kStratZerocore, FrameworkMode::kDive},
    {kStratZerocore, FrameworkMode::kDiveprop},
    {kStratCliques, FrameworkMode::kDfs},  // paper: "if predominant clique
                                           // structure"; run unconditionally,
                                           // degrades gracefully on non-clique
                                           // models
};
constexpr int kNumClass2 = static_cast<int>(sizeof(kClass2Configs) / sizeof(kClass2Configs[0]));

// Paper Section 6.3, Class 3 — full-obj LP strategies.
// Split into two groups by reference solution:
//   3a: zerolp configs use the zero-obj LP vertex (zv_ptr)
//   3b: lp/cliques2 configs use the full-obj LP solution (lp_ptr)
constexpr NamedConfig kClass3aConfigs[] = {
    {kStratZerolp, FrameworkMode::kDfs},
    {kStratZerolp, FrameworkMode::kDiveprop},
};
constexpr int kNumClass3a = static_cast<int>(sizeof(kClass3aConfigs) / sizeof(kClass3aConfigs[0]));

constexpr NamedConfig kClass3bConfigs[] = {
    {kStratCliques2, FrameworkMode::kDiveprop},
    {kStratLp, FrameworkMode::kDfs},
    {kStratLp, FrameworkMode::kDive},
    {kStratLp, FrameworkMode::kDiveprop},
};
constexpr int kNumClass3b = static_cast<int>(sizeof(kClass3bConfigs) / sizeof(kClass3bConfigs[0]));

// ── Portfolio arm descriptor ────────────────────────────────────────────
// Binds a NamedConfig to its LP reference pointer for bandit selection.
struct LpArm {
    const NamedConfig *config;
    const double *lp_ref;  // pointer to the appropriate LP reference solution
};

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

// Run a set of configs in parallel, collecting results into a pool.
// Returns total effort consumed.
size_t run_configs(HighsMipSolver &mipsolver, const CscMatrix &csc, const NamedConfig *configs,
                   int num_configs, const double *hint, const double *lp_ref, SolutionPool &pool,
                   size_t budget) {
    // Pre-compute variable orders sequentially to avoid data races on
    // HighsCliqueTable::cliquePartition (which mutates internal state).
    std::vector<std::vector<HighsInt>> var_orders(num_configs);
    for (int w = 0; w < num_configs; ++w) {
        std::mt19937 rng(42 + static_cast<uint32_t>(w) + 100);
        var_orders[w] = compute_var_order(mipsolver, configs[w].strat.var_strategy, rng, lp_ref);
    }

    std::vector<HeuristicResult> results(num_configs);

    highs::parallel::for_each(
        0, static_cast<HighsInt>(num_configs),
        [&](HighsInt lo, HighsInt hi) {
            for (HighsInt w = lo; w < hi; ++w) {
                std::mt19937 rng(42 + static_cast<uint32_t>(w) + 100);

                FprConfig cfg{};
                cfg.max_effort = budget;
                cfg.hint = hint;
                cfg.scores = nullptr;
                cfg.cont_fallback = nullptr;
                cfg.csc = &csc;
                cfg.mode = configs[w].mode;
                cfg.strategy = &configs[w].strat;
                cfg.lp_ref = lp_ref;
                cfg.precomputed_var_order = var_orders[w].data();
                cfg.precomputed_var_order_size = static_cast<HighsInt>(var_orders[w].size());

                results[w] = fpr_attempt(mipsolver, cfg, rng, 0, nullptr);
            }
        },
        1);

    size_t total_effort = 0;
    for (int w = 0; w < num_configs; ++w) {
        total_effort += results[w].effort;
        if (results[w].found_feasible) {
            pool.try_add(results[w].objective, results[w].solution);
        }
    }
    return total_effort;
}

// ── Portfolio mode: Thompson sampling over LP-dependent configs ──────────

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

void run_portfolio(HighsMipSolver &mipsolver, size_t max_effort, const CscMatrix &csc,
                   const double *hint, const double *ac_ptr, const double *zv_ptr,
                   const double *lp_ptr, SolutionPool &pool) {
    const auto *model = mipsolver.model_;
    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    const HighsLogOptions &log_options = mipsolver.options_mip_->log_options;

    // Build arm descriptors binding each config to its LP reference.
    std::vector<LpArm> arms;
    arms.reserve(kNumLpArms);
    for (int i = 0; i < kNumClass2; ++i) {
        arms.push_back({&kClass2Configs[i], ac_ptr});
    }
    for (int i = 0; i < kNumClass3a; ++i) {
        arms.push_back({&kClass3aConfigs[i], zv_ptr});
    }
    for (int i = 0; i < kNumClass3b; ++i) {
        arms.push_back({&kClass3bConfigs[i], lp_ptr});
    }

    // Pre-compute variable orders to avoid cliquePartition data races.
    const int num_arms = static_cast<int>(arms.size());
    std::vector<std::vector<HighsInt>> var_orders(num_arms);
    for (int i = 0; i < num_arms; ++i) {
        std::mt19937 rng(42 + static_cast<uint32_t>(i) + 200);
        var_orders[i] =
            compute_var_order(mipsolver, arms[i].config->strat.var_strategy, rng, arms[i].lp_ref);
    }

    // Uniform priors — let the bandit learn from scratch.
    std::vector<double> priors(num_arms, 1.0);
    ThompsonSampler bandit(num_arms, priors.data(), /*use_mutex=*/false);

    std::mt19937 rng(42);
    size_t total_effort = 0;
    size_t effort_since_improvement = 0;
    const size_t stale_budget = max_effort >> 2;
    int attempt_counter = 0;

    while (total_effort < max_effort) {
        if (effort_since_improvement > stale_budget) {
            break;
        }

        int arm = bandit.select_effort_aware(rng);
        auto before = pool.snapshot();

        size_t remaining = max_effort - total_effort;

        std::vector<double> restart;
        pool.get_restart(rng, restart);
        const double *restart_ptr = restart.empty() ? nullptr : restart.data();

        FprConfig cfg{};
        cfg.max_effort = remaining;
        cfg.hint = hint;
        cfg.scores = nullptr;
        cfg.cont_fallback = nullptr;
        cfg.csc = &csc;
        cfg.mode = arms[arm].config->mode;
        cfg.strategy = &arms[arm].config->strat;
        cfg.lp_ref = arms[arm].lp_ref;
        cfg.precomputed_var_order = var_orders[arm].data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_orders[arm].size());

        auto result = fpr_attempt(mipsolver, cfg, rng, attempt_counter++, restart_ptr);

        if (result.found_feasible) {
            pool.try_add(result.objective, result.solution);
        }

        auto after = pool.snapshot();
        int reward = compute_reward(before, after, result, minimize);
        bandit.update(arm, reward);
        bandit.record_effort(arm, result.effort);

        total_effort += result.effort;

        if (reward >= 2) {
            effort_since_improvement = 0;
        } else {
            effort_since_improvement += result.effort;
        }

        highsLogDev(log_options, HighsLogType::kVerbose,
                    "[FprLpPortfolio] arm=%s effort=%zu reward=%d\n", kLpArmNames[arm],
                    result.effort, reward);
    }

    mipsolver.mipdata_->heuristic_effort_used += total_effort;
}

}  // namespace

void run(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return;
    }

    // Guard: need an optimal LP relaxation
    auto lp_status = mipdata->lp.getStatus();
    if (!HighsLpRelaxation::scaledOptimal(lp_status)) {
        return;
    }

    const bool minimize = (model->sense_ == ObjSense::kMinimize);
    SolutionPool pool(kPoolCapacity, minimize);

    // Seed pool with incumbent if available
    if (!mipdata->incumbent.empty()) {
        double obj = model->offset_;
        for (HighsInt j = 0; j < ncol; ++j) {
            obj += model->col_cost_[j] * mipdata->incumbent[j];
        }
        pool.try_add(obj, mipdata->incumbent);
    }

    // Build CSC once
    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    const double *hint = mipdata->incumbent.empty() ? nullptr : mipdata->incumbent.data();

    // Get LP solution
    const auto &lp_sol = mipdata->lp.getLpSolver().getSolution().col_value;
    const double *lp_ptr = lp_sol.data();

    // Compute zero-obj analytic center (for zerocore strategies)
    auto analytic_center = compute_analytic_center(mipsolver, false);
    const double *ac_ptr = analytic_center.empty() ? lp_ptr : analytic_center.data();

    // Compute zero-obj LP vertex (for zerolp strategies)
    auto zero_vertex = compute_zero_obj_vertex(mipsolver);
    const double *zv_ptr = zero_vertex.empty() ? lp_ptr : zero_vertex.data();

    // Portfolio mode: bandit selection over all LP-dependent configs.
    // Sequential mode: original class-by-class parallel execution.
    if (mipsolver.options_mip_->mip_heuristic_portfolio) {
        run_portfolio(mipsolver, max_effort, csc, hint, ac_ptr, zv_ptr, lp_ptr, pool);
    } else {
        size_t total_effort = 0;
        size_t remaining = max_effort;

        // Class 2: zero-obj LP strategies (use analytic center / zero vertex)
        auto snap_before_c2 = pool.snapshot();
        total_effort +=
            run_configs(mipsolver, csc, kClass2Configs, kNumClass2, hint, ac_ptr, pool, remaining);
        remaining = (total_effort < max_effort) ? max_effort - total_effort : 0;

        // Check if Class 2 found a new solution — if so, skip Class 3
        // (paper: stop if feasible found between classes).
        // Compare pool snapshot before/after to detect new entries (pool was
        // seeded with incumbent, so non-empty doesn't mean Class 2 found anything).
        auto snap_after_c2 = pool.snapshot();
        bool class2_found_new =
            snap_after_c2.has_solution &&
            (!snap_before_c2.has_solution ||
             (minimize ? snap_after_c2.best_objective < snap_before_c2.best_objective
                       : snap_after_c2.best_objective > snap_before_c2.best_objective));

        if (!class2_found_new && remaining > 0) {
            // Class 3a: zerolp configs use zero-obj LP vertex
            total_effort += run_configs(mipsolver, csc, kClass3aConfigs, kNumClass3a, hint, zv_ptr,
                                        pool, remaining);
            remaining = (total_effort < max_effort) ? max_effort - total_effort : 0;
            // Class 3b: lp/cliques2 configs use full-obj LP solution
            if (remaining > 0) {
                total_effort += run_configs(mipsolver, csc, kClass3bConfigs, kNumClass3b, hint,
                                            lp_ptr, pool, remaining);
            }
        }

        mipdata->heuristic_effort_used += total_effort;
    }

    // Submit best solutions to solver
    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceFPR);
    }
}

}  // namespace fpr_lp
