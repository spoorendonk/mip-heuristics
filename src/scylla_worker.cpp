#include "scylla_worker.h"

#include "contested_pdlp.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "pump_common.h"
#include "solution_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

// Workers 0..kNumFprConfigs-1 cover every FPR config exactly once
// (deterministic round-robin, preserving strategy diversity when
// N >= kNumFprConfigs).  Additional workers draw a pseudo-random
// config from their own seed, avoiding the "16 workers, 4 copies of
// each config" pathology while keeping assignment deterministic per
// (seed, worker_idx).
int select_fpr_config(int worker_idx, uint32_t seed) {
    if (worker_idx >= 0 && worker_idx < kNumFprConfigs) {
        return worker_idx;
    }
    if (worker_idx < 0) {
        return ((worker_idx % kNumFprConfigs) + kNumFprConfigs) % kNumFprConfigs;
    }
    Rng cfg_rng(seed);
    return static_cast<int>(cfg_rng() % static_cast<uint32_t>(kNumFprConfigs));
}

}  // namespace

ScyllaWorker::ScyllaWorker(HighsMipSolver &mipsolver, ContestedPdlp &pdlp, const CscMatrix &csc,
                           SolutionPool &pool, size_t total_budget, uint32_t seed, int worker_idx,
                           int num_workers, std::atomic<uint64_t> *improvement_gen)
    : mipsolver_(mipsolver),
      pdlp_(pdlp),
      csc_(csc),
      pool_(pool),
      num_workers_(std::max(num_workers, 1)),
      epsilon_(pump::kEpsilonInit),
      rng_(seed),
      fpr_config_index_(select_fpr_config(worker_idx, seed)),
      improvement_gen_(improvement_gen) {
    base_.total_budget = total_budget;
    if (!pdlp_.initialized()) {
        base_.finished = true;
        return;
    }
    const auto *model = mipsolver_.model_;
    ncol_ = model->num_col_;
    nrow_ = model->num_row_;
    // Release-safe guard against a 0-column model reaching this
    // worker.  The `pdlp_.initialized()` short-circuit above already
    // catches LPs with `ncol_==0` (ContestedPdlp's constructor refuses
    // to build them), but if a future refactor decouples those checks
    // — or wires Scylla directly to a model whose columns were all
    // fixed out by presolve — the downstream loops `for (j; j < ncol_)`
    // become no-ops while later index ops on `warm_start_col_value_`
    // / `modified_cost_` would be UB.  An `assert` would be a no-op
    // under `NDEBUG` (default for Release builds), so abort
    // unconditionally: corrupt rounding from an undersized model is
    // far worse than a loud crash.
    if (ncol_ == 0) {
        std::fprintf(stderr,
                     "ScyllaWorker: model has 0 columns; refusing to construct (would yield "
                     "UB on later index ops).\n");
        std::abort();
    }

    const auto &orig_cost = model->col_cost_;
    const auto &integrality = model->integrality_;
    HighsInt num_integers = 0;
    double norm_c_sq = 0.0;
    for (HighsInt j = 0; j < ncol_; ++j) {
        if (is_integer(integrality, j)) {
            ++num_integers;
        }
        norm_c_sq += orig_cost[j] * orig_cost[j];
    }
    // `ncol_ == 0` was already aborted above, so `num_integers` is the
    // only remaining graceful-exit condition (a continuous LP with no
    // integer variables — Scylla rounding has nothing to do).
    if (num_integers == 0) {
        base_.finished = true;
        return;
    }

    double norm_c = std::sqrt(norm_c_sq);
    cost_scale_ = (norm_c > 1e-15) ? std::sqrt(num_integers) / norm_c : 1.0;

    nnz_lp_ = pdlp_.nnz_lp();
    if (nnz_lp_ == 0) {
        base_.finished = true;
        return;
    }
    // Size the per-worker stale-rounds cap from the LP (R3): small
    // LPs → tight cap (PDLP is fast, forcing fresh doesn't cost much);
    // large LPs → longer cap (blocking solve is expensive, stay on the
    // snapshot longer).
    max_stale_rounds_ = compute_max_stale_rounds(nnz_lp_);

    base_.stale_budget = base_.total_budget >> 2;
    modified_cost_ = orig_cost;
    cycle_history_.reserve(pump::kCycleWindow);

    // Pre-compute variable order for this worker's static strategy.
    Rng order_rng(heuristic_base_seed(mipsolver_.options_mip_->random_seed) +
                  static_cast<uint32_t>(fpr_config_index_));
    var_order_ = compute_var_order(mipsolver_, kFprConfigs[fpr_config_index_].strat.var_strategy,
                                   order_rng, nullptr);
}

bool ScyllaWorker::absorb_fresh_solve(ContestedPdlp::SolveResult &result, HighsInt &iters_out,
                                      const std::vector<double> *&x_bar_ptr) {
    if (result.status == HighsStatus::kError) {
        base_.finished = true;
        return true;
    }
    if (result.model_status == HighsModelStatus::kInfeasible) {
        base_.finished = true;
        return true;
    }
    iters_out = result.pdlp_iters;
    if (iters_out == 0) {
        ++pdlp_stall_count_;
        if (pdlp_stall_count_ >= pump::kMaxPdlpStalls) {
            base_.finished = true;
            return true;
        }
    } else {
        pdlp_stall_count_ = 0;
    }
    if (result.col_value.empty()) {
        base_.finished = true;
        return true;
    }
    warm_start_col_value_ = std::move(result.col_value);
    warm_start_row_dual_ = std::move(result.row_dual);
    warm_start_valid_ = result.value_valid && result.dual_valid;
    x_bar_ptr = &warm_start_col_value_;
    return false;
}

EpochResult ScyllaWorker::run_epoch(size_t epoch_budget) {
    if (base_.finished) {
        return {};
    }

    const auto *model = mipsolver_.model_;
    auto *mipdata = mipsolver_.mipdata_.get();
    const auto &integrality = model->integrality_;
    const auto &orig_cost = model->col_cost_;
    const double time_limit = mipsolver_.options_mip_->time_limit;

    EpochResult epoch{};

    while (epoch.effort < epoch_budget && base_.total_effort < base_.total_budget) {
        // Wall-clock `time_limit` is enforced by the outer loop between
        // epochs and by the `remaining <= 0` guard below (which also
        // computes PDLP's input time_limit from the same timer read).
        // A redundant worker-level check before that guard would just
        // double the clock_gettime cost per iteration for no extra
        // precision.
        if (improvement_gen_ != nullptr) {
            uint64_t gen = improvement_gen_->load(std::memory_order_relaxed);
            if (gen != last_seen_gen_) {
                last_seen_gen_ = gen;
                base_.effort_since_improvement = 0;
            }
        }
        if (base_.effort_since_improvement > base_.stale_budget) {
            base_.finished = true;
            break;
        }

        // `K_` was historically bumped here (before the solve path).  A
        // stale round uses a peer's x_bar and does not correspond to
        // this worker's paper-defined pump iteration, so we now advance
        // `K_` only on fresh rounds — see the guarded block near the
        // end of this loop where cycle_history_, alpha_K_, and
        // modified_cost_ are also only updated when `fresh`.

        double remaining = time_limit - mipsolver_.timer_.read();
        if (remaining <= 0.0) {
            base_.finished = true;
            break;
        }

        // Issue #76: overlap FPR rounding with PDLP solves.  Try to grab
        // the PDLP mutex non-blockingly.  If successful, run a fresh
        // solve (publishes the snapshot on the way out).  If contended,
        // fall back to the most-recent published snapshot so the worker
        // still produces useful FPR work while another chain holds the
        // mutex.  After kMaxStaleRounds consecutive stale iterations we
        // force a blocking solve() to guarantee forward progress.
        bool fresh = false;
        HighsInt iters_this_round = 0;
        // x_bar_source refers to whichever vector we round against this
        // iteration: either our freshly-updated warm-start (fresh path)
        // or the current stale snapshot's col_value.
        const std::vector<double> *x_bar_ptr = nullptr;

        const bool must_force_fresh = consecutive_stale_rounds_ >= max_stale_rounds_;
        if (must_force_fresh) {
            // Blocking path: either no one is solving (we take the
            // mutex immediately) or we wait behind one worker.  This
            // guarantees every worker eventually sees a fresh solve
            // for its *own* modified_cost_, preventing a pathological
            // "live on a stale snapshot that never matches my objective"
            // regime.
            auto solve_res =
                pdlp_.solve(modified_cost_, warm_start_col_value_, warm_start_row_dual_,
                            warm_start_valid_, epsilon_, remaining);
            if (absorb_fresh_solve(solve_res, iters_this_round, x_bar_ptr)) {
                break;
            }
            fresh = true;
            // `solve()` also publishes a fresh snapshot via
            // `run_locked_with_accounting`; keep stale_snapshot_ in step
            // and update `last_seen_snapshot_gen_` so the identity check
            // in the stale path next round compares against this fresh
            // generation, not a stale one.
            stale_snapshot_ = pdlp_.latest_snapshot();
            if (stale_snapshot_) {
                last_seen_snapshot_gen_ = stale_snapshot_->generation;
            }
        } else {
            auto try_res = pdlp_.try_solve_or_snapshot(modified_cost_, warm_start_col_value_,
                                                       warm_start_row_dual_, warm_start_valid_,
                                                       epsilon_, remaining);
            if (try_res.fresh) {
                if (absorb_fresh_solve(try_res.solve, iters_this_round, x_bar_ptr)) {
                    break;
                }
                fresh = true;
                // Sync stale_snapshot_ to the snapshot this solve just
                // published so the next iteration's "same generation?"
                // check correctly recognises it; otherwise the first
                // stale round after our own fresh solve would round
                // against our own just-published result — wasted FPR
                // work.  Flagged by review R2.
                stale_snapshot_ = pdlp_.latest_snapshot();
                if (stale_snapshot_) {
                    last_seen_snapshot_gen_ = stale_snapshot_->generation;
                }
            } else {
                // Stale path: round against the most-recent published
                // snapshot.  If no snapshot is available yet (cold
                // start before any solve completed) or the snapshot is
                // identical to the one we last consumed, skip this
                // iteration cheaply — nothing new to round.
                auto snap = try_res.stale_snapshot;
                if (!snap || !snap->value_valid || snap->col_value.empty()) {
                    // No cache yet.  Count this as a stale round so we
                    // hit kMaxStaleRounds and force a blocking solve().
                    ++consecutive_stale_rounds_;
                    ++stale_rounds_;
                    // Tiny effort charge so the outer budget loop keeps
                    // moving; mirrors the "nominal 1 unit" convention
                    // used in opportunistic_runner.
                    base_.total_effort += 1;
                    base_.effort_since_improvement += 1;
                    epoch.effort += 1;
                    continue;
                }
                if (snap->generation == last_seen_snapshot_gen_) {
                    // Nothing new since last iteration — count toward
                    // the stale-round cap but don't do duplicate work.
                    // Compare by generation rather than `shared_ptr`
                    // address: a freed Snapshot's heap slot can be
                    // recycled by the allocator, giving two distinct
                    // publications the same `.get()` value, but
                    // generations are strictly monotonic per
                    // `ContestedPdlp` instance and cannot collide.
                    ++consecutive_stale_rounds_;
                    ++stale_rounds_;
                    base_.total_effort += 1;
                    base_.effort_since_improvement += 1;
                    epoch.effort += 1;
                    continue;
                }
                stale_snapshot_ = snap;
                last_seen_snapshot_gen_ = snap->generation;
                fresh = false;
                // Do NOT overwrite this worker's warm_start_ — we only
                // round, we don't update our own PDLP state.  The next
                // blocking solve() will still use the last warm-start
                // this worker personally got back from PDLP.
                x_bar_ptr = &snap->col_value;
            }
        }

        if (fresh) {
            ++fresh_solves_;
            consecutive_stale_rounds_ = 0;
        } else {
            ++stale_rounds_;
            ++consecutive_stale_rounds_;
        }

        // Split PDLP effort accounting:
        //  - epoch.effort gets the FULL cost (iters * nnz) so the outer
        //    run_epoch_loop / run_opportunistic_loop budget check sees
        //    actual aggregate work and stops on time.
        //  - Per-worker counters (base_.total_effort, base_.effort_since_improvement)
        //    get the amortized cost (÷ num_workers) so each worker's
        //    staleness threshold reflects its fair share of the serialized
        //    PDLP pipeline rather than the full cost.
        //
        // Stale rounds contribute zero PDLP effort (no iters were run)
        // but the downstream FPR attempt still consumes effort and is
        // charged below.
        size_t actual_effort = static_cast<size_t>(iters_this_round) * nnz_lp_;
        size_t local_effort = actual_effort / static_cast<size_t>(num_workers_);
        base_.total_effort += local_effort;
        base_.effort_since_improvement += local_effort;
        epoch.effort += actual_effort;

        const auto &x_bar = *x_bar_ptr;

        // Fast path: PDLP solution already MIP-feasible.
        {
            bool mip_feasible = true;
            const double feastol = mipsolver_.options_mip_->mip_feasibility_tolerance;
            for (HighsInt j = 0; j < ncol_; ++j) {
                if (!is_integer(integrality, j)) {
                    continue;
                }
                if (std::abs(x_bar[j] - std::round(x_bar[j])) > feastol) {
                    mip_feasible = false;
                    break;
                }
            }
            if (mip_feasible) {
                for (HighsInt i = 0; i < nrow_; ++i) {
                    double lhs = 0.0;
                    for (HighsInt k = mipdata->ARstart_[i]; k < mipdata->ARstart_[i + 1]; ++k) {
                        lhs += mipdata->ARvalue_[k] * x_bar[mipdata->ARindex_[k]];
                    }
                    if (lhs > model->row_upper_[i] + feastol ||
                        lhs < model->row_lower_[i] - feastol) {
                        mip_feasible = false;
                        break;
                    }
                }
            }
            if (mip_feasible) {
                double obj = model->offset_;
                for (HighsInt j = 0; j < ncol_; ++j) {
                    obj += orig_cost[j] * x_bar[j];
                }
                pool_.try_add(obj, x_bar, kSolutionSourceScylla);
                base_.effort_since_improvement = 0;
                if (improvement_gen_ != nullptr) {
                    improvement_gen_->fetch_add(1, std::memory_order_relaxed);
                }
                epoch.found_improvement = true;
                continue;
            }
        }

        size_t remaining_budget =
            std::min(epoch_budget - std::min(epoch_budget, epoch.effort),
                     base_.total_budget - std::min(base_.total_budget, base_.total_effort));
        if (remaining_budget == 0) {
            break;
        }

        const auto &named = kFprConfigs[fpr_config_index_];
        FprConfig cfg{};
        cfg.max_effort = remaining_budget;
        cfg.hint = x_bar.data();
        cfg.scores = nullptr;
        cfg.cont_fallback = x_bar.data();
        cfg.csc = &csc_;
        cfg.mode = named.mode;
        cfg.strategy = &named.strat;
        cfg.lp_ref = nullptr;
        cfg.precomputed_var_order = var_order_.data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order_.size());
        cfg.scratch = &fpr_scratch_;

        std::vector<double> restart;
        pool_.get_restart(rng_, restart);
        const double *restart_ptr = restart.empty() ? nullptr : restart.data();

        HeuristicResult rounded = fpr_attempt(mipsolver_, cfg, rng_, 0, restart_ptr);

        base_.total_effort += rounded.effort;
        base_.effort_since_improvement += rounded.effort;
        epoch.effort += rounded.effort;

        if (rounded.found_feasible && !rounded.solution.empty()) {
            pool_.try_add(rounded.objective, rounded.solution, kSolutionSourceScylla);
            base_.effort_since_improvement = 0;
            if (improvement_gen_ != nullptr) {
                improvement_gen_->fetch_add(1, std::memory_order_relaxed);
            }
            epoch.found_improvement = true;
        }

        if (rounded.solution.empty()) {
            continue;
        }

        auto &x_hat = rounded.solution;

        // Pump-state advance is guarded by `fresh`: a stale round
        // rounds against a peer worker's x_bar and does not
        // correspond to this chain's paper-defined pump iteration.
        // Decaying `alpha_K_`, bumping `K_`, rewriting
        // `modified_cost_`, and overwriting our cycle slot on stale
        // rounds would drift this chain's schedule away from the
        // paper's semantics.  Effort accounting for the FPR rounding
        // is still charged above regardless, so the outer budget
        // still stops on time.  (R2 flagged this; user-directed in
        // review round 3.)
        if (fresh) {
            if (pump::detect_cycling(cycle_history_, x_hat, integrality, ncol_)) {
                pump::perturb(x_hat, *model, rng_);
            }
            // Cycle-history invariant (Mexi 2023 Algorithm 1.1, line 13):
            // `cycle_history_` is a ring buffer of size at most
            // `kCycleWindow`, indexed implicitly by `K_`.  The slot for
            // iteration `K` is `(K - 1) % kCycleWindow` once the buffer
            // is full; before that we just push_back.  Crucially, both
            // `K_` AND `cycle_history_` are mutated only inside this
            // `fresh` block — stale rounds (from issue #76's overlap
            // refactor and commit 0c29d86's pump-state freeze) skip
            // both, so the invariant
            //
            //     cycle_history_.size() == min(K_, kCycleWindow)
            //
            // holds across stale rounds without any extra bookkeeping.
            //
            // One pre-write check is enough — the slot-write + ++K_
            // mechanics make the post-condition automatic, so the
            // historical post-write assert here was redundant
            // (R2-10 round-4 review).  Release-safe abort matches
            // `b4dd29d` (in_flight) and `231a77e` (ncol_==0) so all
            // three structural invariants share the same defensive
            // style: `assert` is a no-op under `NDEBUG` (default for
            // `-DCMAKE_BUILD_TYPE=Release`), and silently corrupted
            // cycling detection would mask the very pump-state bug
            // this guard exists to catch (R3-8 round-4 review).
            // `std::abort()` skips RAII on sibling worker threads and
            // any in-flight cuPDLP GPU state — the OS reclaims those
            // on process exit, and "die loud" is the correct response
            // to a structural invariant violation (R2-5 round-5).
            const int expected_size = std::min(K_, pump::kCycleWindow);
            if (static_cast<int>(cycle_history_.size()) != expected_size) {
                std::fprintf(stderr,
                             "ScyllaWorker: cycle_history invariant violated "
                             "(size=%zu, expected=%d, K_=%d). Pump state corruption — "
                             "aborting.\n",
                             cycle_history_.size(), expected_size, K_);
                std::abort();
            }
            if (static_cast<int>(cycle_history_.size()) < pump::kCycleWindow) {
                cycle_history_.push_back(x_hat);
            } else {
                cycle_history_[(K_ - 1) % pump::kCycleWindow] = x_hat;
            }

            alpha_K_ *= pump::kAlpha;
            pump::compute_pump_objective(orig_cost, x_hat, x_bar, integrality, model->col_lower_,
                                         model->col_upper_, alpha_K_, cost_scale_, ncol_,
                                         modified_cost_);
            epsilon_ = std::max(pump::kBeta * epsilon_, pump::kEpsilonFloor);
            ++K_;
        }
    }

    return epoch;
}
