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

// Test hook counters; see fpr_lp.h.  std::atomic so concurrent entry
// points don't race; relaxed is fine (monotonic, not used for
// synchronization).
std::atomic<size_t> g_seq_det_count{0};
std::atomic<size_t> g_seq_opp_count{0};

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
    const uint32_t base = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    for (int i = 0; i < kNumLpArms; ++i) {
        // +200 offset spaces these seeds away from the presolve-FPR
        // var-order seeds (also derived from the same base) so the two
        // heuristics' RNG streams don't collide on small seed values.
        std::mt19937 rng(base + static_cast<uint32_t>(i) + 200);
        s.var_orders[i] = compute_var_order(mipsolver, s.arms[i].config->strat.var_strategy, rng,
                                            s.arms[i].lp_ref);
    }

    return s;
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

        auto t0 = std::chrono::steady_clock::now();
        last_result_ = fpr_attempt(mipsolver_, cfg, rng_, attempt_idx_, init_ptr);
        auto t1 = std::chrono::steady_clock::now();
        last_wall_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++attempt_idx_;

        epoch.effort = last_result_.effort;

        if (last_result_.found_feasible) {
            pool_.try_add(last_result_.objective, last_result_.solution, kSolutionSourceFprLp);
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
        last_wall_ms_ = 0.0;
    }

    // Portfolio-side accessors (BanditWorker concept).  `assigned_arm()`
    // mirrors `PortfolioWorker`'s API so bandit_runner.h's shared
    // restart callback works against both workers.
    int assigned_arm() const { return arm_idx_; }
    const HeuristicResult &last_result() const { return last_result_; }
    double last_wall_ms() const { return last_wall_ms_; }
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
    double last_wall_ms_ = 0.0;
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

int compute_worker_count(const HighsMipSolver & /*mipsolver*/) {
    return highs::parallel::num_threads();
}

uint32_t base_seed_for(const HighsMipSolver &mipsolver) {
    return heuristic_base_seed(mipsolver.options_mip_->random_seed);
}

// ---------------------------------------------------------------------------
// 4 entry points
// ---------------------------------------------------------------------------

void run_sequential_deterministic(HighsMipSolver &mipsolver, const LpFprSetup &setup,
                                  SolutionPool &pool) {
    // Spawn `num_threads` workers; worker w binds to arm `w % kNumLpArms`.
    // Matches the presolve FPR pattern (src/fpr.cpp) where excess workers
    // wrap around the curated config list with distinct seeds for diversity.
    auto *mipdata = mipsolver.mipdata_.get();
    const int N = compute_worker_count(mipsolver);
    if (N <= 0) {
        return;
    }
    g_seq_det_count.fetch_add(1, std::memory_order_relaxed);

    const uint32_t base_seed = base_seed_for(mipsolver);

    std::vector<std::unique_ptr<LpFprWorker>> workers;
    workers.reserve(N);
    for (int w = 0; w < N; ++w) {
        uint32_t seed = base_seed + static_cast<uint32_t>(w) * kSeedStride;
        int arm = w % kNumLpArms;
        workers.push_back(std::make_unique<LpFprWorker>(mipsolver, setup, pool, arm, seed));
    }

    constexpr int kEpochsPerWorker = 10;
    const size_t per_worker = setup.budget / static_cast<size_t>(N);
    const size_t epoch_budget = std::max<size_t>(per_worker / kEpochsPerWorker, 1);

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, setup.budget, epoch_budget,
        [](int) { /* seq/det: finished workers stay finished; loop exits on all-finished */ },
        setup.stale_budget);

    mipdata->heuristic_effort_used += total_effort;
}

void run_sequential_opportunistic(HighsMipSolver &mipsolver, const LpFprSetup &setup,
                                  SolutionPool &pool) {
    auto *mipdata = mipsolver.mipdata_.get();
    const int N = compute_worker_count(mipsolver);
    if (N <= 0) {
        return;
    }
    g_seq_opp_count.fetch_add(1, std::memory_order_relaxed);

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

}  // namespace

void run(HighsMipSolver &mipsolver, size_t max_effort) {
    auto setup_opt = build_setup(mipsolver, max_effort);
    if (!setup_opt) {
        return;
    }
    auto &setup = *setup_opt;

    SolutionPool pool(kPoolCapacity, setup.minimize);
    seed_pool(pool, mipsolver);

    // fpr_lp is one heuristic family (LP-dependent FPR, Classes 2-3), so
    // it always runs arm-aligned parallel workers — num_threads workers
    // bound to the top-N arms from kClass2/3a/3b, sharing the solution
    // pool.  The mip_heuristic_portfolio flag (a meta-portfolio over
    // different heuristic families) does not apply here; only
    // mip_heuristic_opportunistic picks between epoch-gated and continuous
    // parallelism.
    if (mipsolver.options_mip_->mip_heuristic_opportunistic) {
        run_sequential_opportunistic(mipsolver, setup, pool);
    } else {
        run_sequential_deterministic(mipsolver, setup, pool);
    }

    // Submit best solutions to solver (sequential).  Each entry carries
    // its own per-heuristic source tag so HiGHS logs the correct origin
    // (`D` for the LP-dependent FPR arms) rather than a generic `A`.
    auto *mipdata = mipsolver.mipdata_.get();
    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, entry.source);
    }
}

DispatchCounts dispatch_counts() {
    return {g_seq_det_count.load(std::memory_order_relaxed),
            g_seq_opp_count.load(std::memory_order_relaxed)};
}

void reset_dispatch_counts() {
    g_seq_det_count.store(0, std::memory_order_relaxed);
    g_seq_opp_count.store(0, std::memory_order_relaxed);
}

}  // namespace fpr_lp
