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
    if (num_integers == 0 || ncol_ == 0) {
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

    base_.stale_budget = base_.total_budget >> 2;
    modified_cost_ = orig_cost;
    cycle_history_.reserve(pump::kCycleWindow);

    // Pre-compute variable order for this worker's static strategy.
    Rng order_rng(heuristic_base_seed(mipsolver_.options_mip_->random_seed) +
                  static_cast<uint32_t>(fpr_config_index_));
    var_order_ = compute_var_order(mipsolver_, kFprConfigs[fpr_config_index_].strat.var_strategy,
                                   order_rng, nullptr);
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

        ++K_;

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

        const bool must_force_fresh = consecutive_stale_rounds_ >= kMaxStaleRounds;
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

            if (solve_res.status == HighsStatus::kError) {
                base_.finished = true;
                break;
            }
            if (solve_res.model_status == HighsModelStatus::kInfeasible) {
                base_.finished = true;
                break;
            }
            iters_this_round = solve_res.pdlp_iters;
            if (iters_this_round == 0) {
                ++pdlp_stall_count_;
                if (pdlp_stall_count_ >= pump::kMaxPdlpStalls) {
                    base_.finished = true;
                    break;
                }
            } else {
                pdlp_stall_count_ = 0;
            }
            if (solve_res.col_value.empty()) {
                base_.finished = true;
                break;
            }
            warm_start_col_value_ = std::move(solve_res.col_value);
            warm_start_row_dual_ = std::move(solve_res.row_dual);
            warm_start_valid_ = solve_res.value_valid && solve_res.dual_valid;
            fresh = true;
            x_bar_ptr = &warm_start_col_value_;
        } else {
            auto try_res = pdlp_.try_solve_or_snapshot(modified_cost_, warm_start_col_value_,
                                                       warm_start_row_dual_, warm_start_valid_,
                                                       epsilon_, remaining);
            if (try_res.fresh) {
                const auto &solve_res = try_res.solve;
                if (solve_res.status == HighsStatus::kError) {
                    base_.finished = true;
                    break;
                }
                if (solve_res.model_status == HighsModelStatus::kInfeasible) {
                    base_.finished = true;
                    break;
                }
                iters_this_round = solve_res.pdlp_iters;
                if (iters_this_round == 0) {
                    ++pdlp_stall_count_;
                    if (pdlp_stall_count_ >= pump::kMaxPdlpStalls) {
                        base_.finished = true;
                        break;
                    }
                } else {
                    pdlp_stall_count_ = 0;
                }
                if (solve_res.col_value.empty()) {
                    base_.finished = true;
                    break;
                }
                warm_start_col_value_ = solve_res.col_value;
                warm_start_row_dual_ = solve_res.row_dual;
                warm_start_valid_ = solve_res.value_valid && solve_res.dual_valid;
                fresh = true;
                x_bar_ptr = &warm_start_col_value_;
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
                if (snap.get() == stale_snapshot_.get()) {
                    // Nothing new since last iteration — count toward
                    // the stale-round cap but don't do duplicate work.
                    ++consecutive_stale_rounds_;
                    ++stale_rounds_;
                    base_.total_effort += 1;
                    base_.effort_since_improvement += 1;
                    epoch.effort += 1;
                    continue;
                }
                stale_snapshot_ = snap;
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

        if (pump::detect_cycling(cycle_history_, x_hat, integrality, ncol_)) {
            pump::perturb(x_hat, *model, rng_);
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
    }

    return epoch;
}
