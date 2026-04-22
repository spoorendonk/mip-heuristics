#pragma once

#include "epoch_runner.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "util/HighsInt.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

class HighsMipSolver;
class SolutionPool;
class ContestedPdlp;

// FPR strategy assignment for Scylla workers.  Workers `0..kNumFprConfigs-1`
// receive `kFprConfigs[w]` (deterministic round-robin, guaranteeing each
// strategy is covered once when N >= kNumFprConfigs).  Additional workers
// receive a seed-driven pseudo-random choice so redundant workers do not
// cluster on the same strategy — still deterministic per (seed, worker_idx).
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
// FPR rounding uses a single strategy per worker, assigned at
// construction time by `select_fpr_config(worker_idx, seed)`.
//
// Satisfies the EpochWorker concept from epoch_runner.h.
class ScyllaWorker {
public:
    ScyllaWorker(HighsMipSolver &mipsolver, ContestedPdlp &pdlp, const CscMatrix &csc,
                 SolutionPool &pool, size_t total_budget, uint32_t seed, int worker_idx,
                 int num_workers, std::atomic<uint64_t> *improvement_gen = nullptr);

    // Run iterations until epoch_budget effort is consumed.  Sets
    // base_.finished when the worker cannot make further progress.
    EpochResult run_epoch(size_t epoch_budget);

    bool finished() const { return base_.finished; }
    size_t total_effort() const { return base_.total_effort; }

    // Reset the improvement staleness counter (called at epoch boundary
    // when another worker found an improvement).
    void reset_staleness() { base_.reset_staleness(); }

private:
    HighsMipSolver &mipsolver_;
    ContestedPdlp &pdlp_;
    const CscMatrix &csc_;
    SolutionPool &pool_;

    HighsInt ncol_ = 0;
    HighsInt nrow_ = 0;
    double cost_scale_ = 1.0;
    size_t nnz_lp_ = 0;

    // Effort / staleness / finished bookkeeping.  `total_budget` is set
    // in the constructor; `stale_budget` derives from `total_budget >> 2`
    // at init time.
    EpochWorkerBase base_;

    // Number of concurrent ScyllaWorkers sharing the contested PDLP; used
    // to amortize per-iteration effort so each worker charges its fair
    // share of the (serialized) PDLP work rather than the full cost.
    int num_workers_ = 1;

    int pdlp_stall_count_ = 0;

    double epsilon_;
    double alpha_K_ = 1.0;
    int K_ = 0;

    // Per-worker state.
    std::vector<double> warm_start_col_value_;
    std::vector<double> warm_start_row_dual_;
    bool warm_start_valid_ = false;

    std::vector<std::vector<double>> cycle_history_;
    std::vector<double> modified_cost_;
    Rng rng_;

    // FPR strategy assignment (static, one per worker).
    int fpr_config_index_ = 0;
    std::vector<HighsInt> var_order_;

    // Persistent scratch reused across fpr_attempt calls inside run_epoch
    // to avoid per-iteration malloc/free churn on the DFS + WalkSAT path.
    FprScratch fpr_scratch_;

    // Cross-worker improvement broadcast.  When any worker bumps the
    // generation, peers reset their local staleness on the next loop
    // iteration — prevents workers from dying on `base_.stale_budget` while a
    // peer just improved.  Plumbed by every path that can run multiple
    // Scylla workers concurrently: standalone Scylla det + opp, and port/det
    // (via PortfolioWorker, see portfolio.cpp) + port/opp.  The epoch_runner
    // barrier also calls `reset_staleness()` in det modes, but that is
    // coarser than this atomic, which kicks in mid-epoch.  Null only in
    // single-worker contexts (LpFprWorker).
    std::atomic<uint64_t> *improvement_gen_ = nullptr;
    uint64_t last_seen_gen_ = 0;
};

static_assert(EpochWorker<ScyllaWorker>, "ScyllaWorker must satisfy EpochWorker concept");
