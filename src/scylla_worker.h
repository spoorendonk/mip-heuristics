#pragma once

#include "epoch_runner.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <vector>

class HighsMipSolver;
class SolutionPool;
class ContestedPdlp;

// Static FPR strategy assignment for Scylla workers.  Worker `w`
// receives `kFprConfigs[w % kNumFprConfigs]`, redistributing the
// rounding-strategy diversity the old intra-iteration parallel
// rounding provided (one strategy per config) across N workers
// rather than sampling per iteration.
inline constexpr NamedConfig kFprConfigs[] = {
    {kStratBadobjcl, FrameworkMode::kDfs},
    {kStratLocks2, FrameworkMode::kDfs},
    {kStratLocks2, FrameworkMode::kDive},
    {kStratLocks, FrameworkMode::kDfsrep},
};
inline constexpr int kNumFprConfigs = static_cast<int>(std::size(kFprConfigs));

// One worker of the Scylla feasibility-pump heuristic (Mexi et al.
// 2023, Algorithm 1.1).  Each worker owns its per-worker state
// (warm-start, α_K decay, ε schedule, cycle history, modified-cost
// buffer, RNG) but shares a single `ContestedPdlp` instance with its
// peers.  The shared solver serializes PDLP runs via a mutex so at
// most one LP solve is in flight at a time; workers parallelize over
// `fpr_attempt` rounding, cycling detection, and objective updates.
//
// FPR rounding uses a single strategy per worker: worker `w` is
// assigned `kFprConfigs[w % kNumFprConfigs]` at construction time.
//
// Satisfies the EpochWorker concept from epoch_runner.h.
class ScyllaWorker {
public:
    ScyllaWorker(HighsMipSolver &mipsolver, ContestedPdlp &pdlp, const CscMatrix &csc,
                 SolutionPool &pool, size_t total_budget, uint32_t seed, int fpr_config_index);

    // Run iterations until epoch_budget effort is consumed.  Sets
    // finished_ when the worker cannot make further progress.
    EpochResult run_epoch(size_t epoch_budget);

    bool finished() const { return finished_; }
    size_t total_effort() const { return total_effort_; }

    // Reset the improvement staleness counter (called at epoch boundary
    // when another worker found an improvement).
    void reset_staleness() { effort_since_improvement_ = 0; }

private:
    HighsMipSolver &mipsolver_;
    ContestedPdlp &pdlp_;
    const CscMatrix &csc_;
    SolutionPool &pool_;
    const size_t total_budget_;

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

    // Per-worker state.
    std::vector<double> warm_start_col_value_;
    std::vector<double> warm_start_row_dual_;
    bool warm_start_valid_ = false;

    std::vector<std::vector<double>> cycle_history_;
    std::vector<double> modified_cost_;
    std::mt19937 rng_;

    // FPR strategy assignment (static, one per worker).
    int fpr_config_index_ = 0;
    std::vector<HighsInt> var_order_;
};

static_assert(EpochWorker<ScyllaWorker>, "ScyllaWorker must satisfy EpochWorker concept");
