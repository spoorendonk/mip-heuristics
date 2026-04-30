#pragma once

#include "contested_pdlp.h"
#include "epoch_runner.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "util/HighsInt.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

class HighsMipSolver;
class SolutionPool;

// Per-worker default cap on consecutive stale-snapshot rounds before the
// worker must force a blocking `solve()` to refresh its view.  Stale
// rounds are cheap (no PDLP work, reuses the latest completed snapshot)
// but a worker that has done too many in a row risks living forever on
// a degenerate cached primal; this nudges it back to the lock at a
// bounded rate.  Issue #76.
//
// The cap is tunable per-worker via `compute_max_stale_rounds(nnz_lp)` —
// on small LPs each PDLP solve is fast so we don't want to stall many
// rounds (any one peer will refresh soon anyway); on large LPs the
// inverse holds because a blocking solve may take seconds.  Reviewer
// R3 flagged the old unconditional `4` as a tuning hole.
inline constexpr int kMaxStaleRoundsDefault = 4;
inline constexpr int kMaxStaleRoundsMin = 2;
inline constexpr int kMaxStaleRoundsMax = 16;

// Size threshold for scaling up the stale cap.  Tuned so a 10k-nnz LP
// (tiny) sticks with the default 4, a 1M-nnz LP (multi-second PDLP
// solve) scales to ~16 (= 4 + 12 extras at 83k each), and the
// `kMaxStaleRoundsMax = 16` ceiling caps anything bigger.  The exact
// ramp is less important than avoiding the one-size-fits-all 4.
// (R2-4 round-3 review: previous 250'000 reached only ~8 at 1M nnz,
// not the documented ~16.)
inline constexpr size_t kNnzPerExtraStaleRound = 83'000;

inline int compute_max_stale_rounds(size_t nnz_lp) {
    const int extra = static_cast<int>(nnz_lp / kNnzPerExtraStaleRound);
    const int cap = kMaxStaleRoundsDefault + extra;
    return cap < kMaxStaleRoundsMin   ? kMaxStaleRoundsMin
           : cap > kMaxStaleRoundsMax ? kMaxStaleRoundsMax
                                      : cap;
}

// Compatibility alias for callers that don't have `nnz_lp_` in scope.
inline constexpr int kMaxStaleRounds = kMaxStaleRoundsDefault;

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

    // Observability for issue #76: how many iterations used a fresh
    // solve (held the mutex) vs rounded against a stale snapshot.
    uint64_t fresh_solves() const { return fresh_solves_; }
    uint64_t stale_rounds() const { return stale_rounds_; }

    // Reset the improvement staleness counter (called at epoch boundary
    // when another worker found an improvement).
    void reset_staleness() { base_.reset_staleness(); }

private:
    // Shared handling of a completed PDLP solve result used by both the
    // blocking (`must_force_fresh`) branch and the non-blocking
    // `try_solve_or_snapshot` fresh branch of `run_epoch`.  Moves from
    // `result.col_value` / `result.row_dual` into warm-start state and
    // updates `pdlp_stall_count_`.  Returns true when the worker must
    // break out of the run loop (error / infeasible / stall cap /
    // empty primal), false to continue with `iters_out` set and
    // `x_bar_ptr` pointing at `warm_start_col_value_`.  Kept on the
    // header so tests can call it independently if needed.
    bool absorb_fresh_solve(ContestedPdlp::SolveResult &result, HighsInt &iters_out,
                            const std::vector<double> *&x_bar_ptr);

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

    // Stale-snapshot overlap bookkeeping (issue #76).  `stale_snapshot_`
    // keeps a `shared_ptr` to the most-recent Snapshot we rounded
    // against (purely for ownership / lifetime — the upstream may
    // replace its atomic slot underneath us at any time).
    // `last_seen_snapshot_gen_` is the *identity* token: we compare
    // generations, not shared_ptr addresses, because freed-and-recycled
    // heap slots can give two distinct Snapshots the same `.get()`
    // value, while monotonic generations cannot collide.  Generation 0
    // is the "have not seen any snapshot yet" sentinel; the first
    // published snapshot is generation 1 (see ContestedPdlp::Snapshot).
    // `consecutive_stale_rounds_` is reset whenever we manage a fresh
    // solve or observe a higher generation; when it hits
    // `max_stale_rounds_` we issue a blocking `solve()` on the next
    // iteration to guarantee forward progress.  The cap is sized at
    // construction from `nnz_lp_` (see `compute_max_stale_rounds`)
    // rather than the historical flat `kMaxStaleRounds = 4`, because
    // PDLP latency scales with LP size and flat-4 is too eager to
    // force-fresh on large MIPs (R3).
    std::shared_ptr<const ContestedPdlp::Snapshot> stale_snapshot_;
    uint64_t last_seen_snapshot_gen_ = 0;
    int consecutive_stale_rounds_ = 0;
    int max_stale_rounds_ = kMaxStaleRoundsDefault;
    // Counters exposed for tests / observability.
    uint64_t fresh_solves_ = 0;
    uint64_t stale_rounds_ = 0;

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
