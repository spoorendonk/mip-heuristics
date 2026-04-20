#include "local_mip_worker.h"

#include "heuristic_common.h"
#include "local_mip_caches.h"
#include "local_mip_core.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "solution_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace local_mip_detail {

void perturb_solution(std::vector<double> &solution, const HighsMipSolverData &mipdata,
                      const std::vector<HighsVarType> &integrality,
                      const std::vector<double> &col_lb, const std::vector<double> &col_ub,
                      HighsInt ncol, std::mt19937 &rng) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        if (coin(rng) > kPerturbBinaryFraction) {
            continue;
        }
        if (mipdata.domain.isBinary(j)) {
            solution[j] = (solution[j] < 0.5) ? 1.0 : 0.0;
        } else {
            double lo = std::ceil(col_lb[j]);
            double hi = std::floor(col_ub[j]);
            if (hi <= lo) {
                continue;
            }
            auto irange = static_cast<int64_t>(hi - lo);
            int64_t shift = std::uniform_int_distribution<int64_t>(1, irange)(rng);
            double current = std::round(solution[j]);
            solution[j] = lo + std::fmod(current - lo + shift, irange + 1.0);
            solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
        }
    }
}

LocalMipWorker::LocalMipWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
                               size_t total_budget, uint32_t seed, const double *initial_solution,
                               size_t stale_budget)
    : mipsolver_(mipsolver),
      csc_(csc),
      pool_(pool),
      total_budget_(total_budget),
      stale_budget_(stale_budget > 0 ? stale_budget : total_budget >> 2),
      rng_(seed),
      ctx_(mipsolver, csc) {
    const HighsInt ncol = mipsolver.model_->num_col_;
    auto *mipdata = ctx_.mipdata;

    // Precompute variable subsets
    for (HighsInt j = 0; j < ncol; ++j) {
        if (std::abs(ctx_.col_cost[j]) >= kEpsZero) {
            costed_vars_.push_back(j);
        }
        if (mipdata->domain.isBinary(j)) {
            binary_vars_.push_back(j);
        }
    }
    ctx_.lift.costed_vars = &costed_vars_;

    // Initialize solution
    const double *src = initial_solution
                            ? initial_solution
                            : (!mipdata->incumbent.empty() ? mipdata->incumbent.data() : nullptr);
    if (src) {
        for (HighsInt j = 0; j < ncol; ++j) {
            double v = src[j];
            if (ctx_.is_int(j)) {
                v = std::round(v);
            }
            ctx_.solution[j] = std::max(ctx_.col_lb[j], std::min(ctx_.col_ub[j], v));
        }
    } else {
        for (HighsInt j = 0; j < ncol; ++j) {
            double v = std::clamp(0.0, ctx_.col_lb[j], ctx_.col_ub[j]);
            if (ctx_.is_int(j)) {
                v = std::round(v);
            }
            ctx_.solution[j] = v;
        }
    }

    ctx_.rebuild_state();
    best_objective_ = ctx_.minimize ? std::numeric_limits<double>::infinity()
                                    : -std::numeric_limits<double>::infinity();
    best_solution_.resize(ncol);
}

EpochResult LocalMipWorker::run_epoch(size_t epoch_budget) {
    if (finished_) {
        return {};
    }

    const HighsInt ncol = mipsolver_.model_->num_col_;
    const double time_limit = mipsolver_.options_mip_->time_limit;

    EpochResult epoch{};
    size_t effort_start = ctx_.effort;
    size_t effort_at_last_improvement = effort_start;

    while (ctx_.effort - effort_start < epoch_budget &&
           total_effort_ + (ctx_.effort - effort_start) < total_budget_) {
        if (mipsolver_.timer_.read() >= time_limit) {
            finished_ = true;
            break;
        }
        if (effort_since_improvement_ + (ctx_.effort - effort_start) > stale_budget_) {
            finished_ = true;
            break;
        }

        bool feasible_mode = ctx_.violated.empty();

        if (feasible_mode) {
            bool need_full_recheck = ctx_.was_infeasible ||
                                     (ctx_.feasible_recheck_counter % kFeasibleRecheckPeriod == 0);
            ctx_.was_infeasible = false;
            ++ctx_.feasible_recheck_counter;

            bool truly_feasible = true;
            if (need_full_recheck) {
                truly_feasible = ctx_.full_recheck(/*update_sets=*/true, /*early_exit=*/false);
            }
            if (!truly_feasible) {
                ++step_;
                continue;
            }

            double obj = ctx_.current_obj;
            bool improved = false;
            if (!best_feasible_) {
                improved = true;
            } else if (ctx_.minimize) {
                improved = (obj < best_objective_ - ctx_.epsilon);
            } else {
                improved = (obj > best_objective_ + ctx_.epsilon);
            }

            if (improved) {
                if (!need_full_recheck) {
                    if (!ctx_.full_recheck(/*update_sets=*/false,
                                           /*early_exit=*/true)) {
                        ctx_.rebuild_state();
                        ++step_;
                        continue;
                    }
                }
                best_feasible_ = true;
                best_objective_ = obj;
                best_solution_ = ctx_.solution;
                steps_since_improvement_ = 0;
                epoch.found_improvement = true;

                pool_.try_add(obj, ctx_.solution);
                effort_since_improvement_ = 0;
                effort_at_last_improvement = ctx_.effort;
            }

            ctx_.lift.recompute_all(ctx_);
            Candidate lift_best;
            lift_best.score = 0.0;
            {
                HighsInt write = 0;
                for (HighsInt read = 0;
                     read < static_cast<HighsInt>(ctx_.lift.positive_list.size()); ++read) {
                    HighsInt j = ctx_.lift.positive_list[read];
                    if (!ctx_.lift.in_positive[j]) {
                        continue;
                    }
                    ctx_.lift.positive_list[write++] = j;
                    if (ctx_.lift.score[j] <= lift_best.score) {
                        continue;
                    }
                    double lo = ctx_.lift.lo[j], hi = ctx_.lift.hi[j];
                    if (lo > hi) {
                        continue;
                    }
                    double target;
                    if (ctx_.minimize) {
                        target = (ctx_.col_cost[j] > 0) ? lo : hi;
                    } else {
                        target = (ctx_.col_cost[j] > 0) ? hi : lo;
                    }
                    target = ctx_.clamp_and_round(j, target);
                    if (std::abs(target - ctx_.solution[j]) < kEpsZero) {
                        continue;
                    }
                    lift_best = {j, target, ctx_.lift.score[j], 0.0};
                }
                ctx_.lift.positive_list.resize(write);
            }

            if (lift_best.var_idx != -1) {
                ctx_.apply_move_with_tabu(lift_best.var_idx, lift_best.new_val, step_, rng_);
            } else {
                ctx_.update_weights(rng_, /*is_feasible=*/true, best_feasible_, best_objective_);
            }

            ++steps_since_improvement_;
            if (steps_since_improvement_ >= kFeasiblePlateau) {
                finished_ = true;
                break;
            }
        } else {
            Candidate cand = infeasible_step(ctx_, rng_, step_, best_feasible_, best_objective_,
                                             costed_vars_, binary_vars_);

            if (cand.var_idx != -1) {
                ctx_.apply_move_with_tabu(cand.var_idx, cand.new_val, step_, rng_);
            }

            ++steps_since_improvement_;
            if (ctx_.violated.empty()) {
                steps_since_improvement_ = 0;
            }
        }

        // Activity refresh
        if (step_ % kActivityPeriod == 0 && step_ > 0) {
            ctx_.rebuild_state();
        }

        // Restart logic
        if (steps_since_improvement_ >= kRestartInterval) {
            steps_since_improvement_ = 0;
            ++restart_count_;

            if (best_feasible_ && (restart_count_ % 2 == 1)) {
                ctx_.solution = best_solution_;
            } else {
                for (HighsInt j = 0; j < ncol; ++j) {
                    if (ctx_.mipdata->domain.isBinary(j)) {
                        ctx_.solution[j] = (rng_() % 2 == 0) ? 0.0 : 1.0;
                    } else if (ctx_.is_int(j)) {
                        double lo = std::max(ctx_.col_lb[j], -1e8);
                        double hi = std::min(ctx_.col_ub[j], lo + 100.0);
                        ctx_.solution[j] = std::max(
                            ctx_.col_lb[j],
                            std::min(
                                ctx_.col_ub[j],
                                std::round(std::uniform_real_distribution<double>(lo, hi)(rng_))));
                    } else {
                        double lo = ctx_.col_lb[j] > -kHighsInf ? ctx_.col_lb[j] : -1e6;
                        double hi = ctx_.col_ub[j] < kHighsInf ? ctx_.col_ub[j] : lo + 1e6;
                        if (hi > lo) {
                            ctx_.solution[j] = std::uniform_real_distribution<double>(lo, hi)(rng_);
                        } else {
                            ctx_.solution[j] = lo;
                        }
                    }
                }
            }

            ctx_.rebuild_state();
            std::fill(ctx_.tabu_inc_until.begin(), ctx_.tabu_inc_until.end(), 0);
            std::fill(ctx_.tabu_dec_until.begin(), ctx_.tabu_dec_until.end(), 0);
        }

        ++step_;
    }

    size_t epoch_effort = ctx_.effort - effort_start;
    total_effort_ += epoch_effort;
    // Only add effort consumed since the last improvement within this
    // epoch (avoid double-counting when improvement resets the counter).
    effort_since_improvement_ += ctx_.effort - effort_at_last_improvement;
    epoch.effort = epoch_effort;

    return epoch;
}

}  // namespace local_mip_detail
