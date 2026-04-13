#pragma once

#include "epoch_runner.h"
#include "heuristic_common.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

class HighsMipSolver;
struct CscMatrix;
class SolutionPool;

// Encapsulates a PDLP pump chain (Mexi et al. 2023, Algorithm 1.1)
// with epoch-based execution.  Each worker owns its own PDLP solver
// instance, warm-start vectors, cycling history, and RNG.
//
// When num_fpr_workers == 1 (default), uses the legacy single-fpr_attempt
// path (strategy=nullptr, cont_fallback=x_bar).  When num_fpr_workers > 1,
// runs M-way parallel FPR rounding with strategy-aware configs.
//
// Satisfies the EpochWorker concept from epoch_runner.h.
class PumpWorker {
public:
    PumpWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
               size_t total_budget, uint32_t seed, int num_fpr_workers = 1);
    ~PumpWorker();

    // Run pump chain iterations until epoch_budget effort is consumed.
    // Sets finished_ when the worker cannot make further progress.
    EpochResult run_epoch(size_t epoch_budget);

    bool finished() const { return finished_; }
    size_t total_effort() const { return total_effort_; }

    // Reset the improvement staleness counter (called at epoch boundary
    // when another worker found an improvement).
    void reset_staleness() { effort_since_improvement_ = 0; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    HighsMipSolver &mipsolver_;
    const CscMatrix &csc_;
    SolutionPool &pool_;
    const size_t total_budget_;
    const uint32_t seed_;
    const int num_fpr_workers_;

    HighsInt ncol_ = 0;
    HighsInt nrow_ = 0;
    double cost_scale_ = 1.0;
    size_t nnz_lp_ = 0;
    size_t stale_budget_ = 0;

    int pdlp_stall_count_ = 0;

    double epsilon_;
    double alpha_K_ = 1.0;
    int K_ = 0;

    size_t total_effort_ = 0;
    size_t effort_since_improvement_ = 0;
    bool finished_ = false;

    std::vector<std::vector<double>> cycle_history_;
    std::vector<double> scores_;
    std::vector<double> modified_cost_;
    std::mt19937 rng_;

    // Multi-worker FPR rounding state (used when num_fpr_workers_ > 1).
    std::vector<std::vector<HighsInt>> var_orders_;
    std::vector<HeuristicResult> rounding_results_;
};

static_assert(EpochWorker<PumpWorker>, "PumpWorker must satisfy EpochWorker concept");
