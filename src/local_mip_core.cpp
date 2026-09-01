#include "local_mip_core.h"

#include "heuristic_common.h"
#include "local_mip_caches.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace local_mip_detail {

namespace {

double compute_objective(const HighsLp* model, const std::vector<double>& solution) {
    double obj = model->offset_;
    for (HighsInt j = 0; j < model->num_col_; ++j) {
        obj += model->col_cost_[j] * solution[j];
    }
    return obj;
}

}  // namespace

// --- WorkerCtx ---

WorkerCtx::WorkerCtx(HighsMipSolver& mipsolver, const CscMatrix& csc_, const uint8_t* binary_)
    : model(mipsolver.model_),
      ar_start(mipsolver.mipdata_->ARstart_),
      ar_index(mipsolver.mipdata_->ARindex_),
      ar_value(mipsolver.mipdata_->ARvalue_),
      col_lb(mipsolver.model_->col_lower_),
      col_ub(mipsolver.model_->col_upper_),
      col_cost(mipsolver.model_->col_cost_),
      row_lo(mipsolver.model_->row_lower_),
      row_hi(mipsolver.model_->row_upper_),
      integrality(mipsolver.model_->integrality_),
      csc(csc_),
      feastol(mipsolver.mipdata_->feastol),
      epsilon(mipsolver.mipdata_->epsilon),
      minimize(mipsolver.model_->sense_ == ObjSense::kMinimize),
      ncol(mipsolver.model_->num_col_),
      nrow(mipsolver.model_->num_row_),
      binary(binary_),
      solution(ncol),
      lhs(nrow),
      weight(nrow, 1),
      violated(nrow),
      satisfied(nrow),
      viol_cache(nrow),
      lift(ncol),
      tabu_inc_until(ncol, 0),
      tabu_dec_until(ncol, 0) {
    batch.reserve(kBmsBudget);
    sampled.reserve(static_cast<size_t>(kBmsConstraints) * 3);
}

void WorkerCtx::update_violated(HighsInt i) {
    double viol = compute_violation(i, lhs[i]);
    bool was = violated.contains(i);
    bool now = (viol > kViolTol);
    if (now && !was) {
        violated.add(i);
        satisfied.remove(i);
    } else if (!now && was) {
        violated.remove(i);
        if (!is_equality(i)) {
            satisfied.add(i);
        }
    }
}

void WorkerCtx::apply_move(HighsInt j, double new_val) {
    double old_val = solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < kEpsZero) {
        return;
    }
    solution[j] = new_val;
    current_obj += col_cost[j] * delta;
    effort += csc.col_start[j + 1] - csc.col_start[j];
    // Only maintain LiftCache during feasible mode; on the
    // infeasible→feasible transition, rebuild_state marks all dirty.
    bool dirty_lift = !was_infeasible && !lift.all_dirty;
    if (dirty_lift) {
        lift.mark_dirty(j);
    }
    for (HighsInt p = csc.col_start[j]; p < csc.col_start[j + 1]; ++p) {
        HighsInt i = csc.col_row[p];
        lhs[i] += csc.col_val[p] * delta;
        update_violated(i);
        if (dirty_lift) {
            for (HighsInt k = ar_start[i]; k < ar_start[i + 1]; ++k) {
                lift.mark_dirty(ar_index[k]);
            }
        }
    }
}

void WorkerCtx::apply_move_with_tabu(HighsInt j, double new_val, HighsInt step, Rng& rng) {
    double delta = new_val - solution[j];
    apply_move(j, new_val);
    HighsInt tabu_len = kTabuBase + static_cast<HighsInt>(rng() % kTabuVar);
    if (delta > 0) {
        tabu_dec_until[j] = step + tabu_len;
    } else {
        tabu_inc_until[j] = step + tabu_len;
    }
}

bool WorkerCtx::full_recheck(bool update_sets, bool early_exit) {
    effort += ar_index.size();
    if (update_sets) {
        violated.clear();
        satisfied.clear();
    }
    bool feasible = true;
    for (HighsInt i = 0; i < nrow; ++i) {
        double l = 0.0;
        for (HighsInt k = ar_start[i]; k < ar_start[i + 1]; ++k) {
            l += ar_value[k] * solution[ar_index[k]];
        }
        lhs[i] = l;
        if (compute_violation(i, l) > kViolTol) {
            feasible = false;
            if (early_exit) {
                break;
            }
            if (update_sets) {
                violated.add(i);
            }
        } else if (update_sets && !is_equality(i)) {
            satisfied.add(i);
        }
    }
    return feasible;
}

void WorkerCtx::rebuild_state() {
    was_infeasible = true;
    feasible_recheck_counter = 0;
    full_recheck(/*update_sets=*/true, /*early_exit=*/false);
    lift.mark_all_dirty();
    current_obj = compute_objective(model, solution);
}

// Cognitive complexity 35 (threshold 25).  Kept whole: LocalMIP's tight-delta rule (Defs 4-5): one
// branch per bound / integrality / direction combination. Decomposing it would move work across a
// worker's inner loop, and the closeout takes no unmeasured performance risk; the standards also
// rank fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
double WorkerCtx::compute_tight_delta(HighsInt i, HighsInt j, double coeff) const {
    if (std::abs(coeff) < kEpsZero) {
        return 0.0;
    }
    double l = lhs[i];
    double gap;
    bool row_violated;
    // The first two branches share an expression form but differ in which
    // bound they measure against.  (This used to carry a NOLINTBEGIN for
    // bugprone-branch-clone; verified against clang-tidy 22 that the check
    // does not fire here, so the suppression was dead and is gone.)
    if (l > row_hi[i] + feastol) {
        gap = l - row_hi[i];  // upper violated
        row_violated = true;
    } else if (l < row_lo[i] - feastol) {
        gap = l - row_lo[i];  // lower violated
        row_violated = true;
    } else {
        // Satisfied: push toward the nearest bound
        double gap_hi = (row_hi[i] < kHighsInf) ? (l - row_hi[i]) : kHighsInf;
        double gap_lo = (row_lo[i] > -kHighsInf) ? (l - row_lo[i]) : kHighsInf;
        gap = (std::abs(gap_hi) <= std::abs(gap_lo)) ? gap_hi : gap_lo;
        row_violated = false;
    }

    double delta = -gap / coeff;

    if (is_equality(i)) {
        // Deliberately keeps the coefficient-sign rule rather than
        // `round_tight_delta` (issue #123): a violated equality row cannot be
        // made satisfied by any rounding direction — both sides of the bound
        // are infeasible — so "round away from zero" has nothing to reach.
        // The satisfied half of this branch is unreachable in practice: both
        // `satisfied.add` sites guard on `!is_equality`, so no equality row
        // enters the set the satisfied-row callers iterate.
        if (is_int(j)) {
            delta = (coeff > 0) ? std::floor(delta) : std::ceil(delta);
        }
        double new_val = solution[j] + delta;
        if (new_val < col_lb[j] || new_val > col_ub[j]) {
            if ((gap > 0 && coeff > 0) || (gap < 0 && coeff < 0)) {
                delta = col_lb[j] - solution[j];
            } else {
                delta = col_ub[j] - solution[j];
            }
        }
    } else {
        // Integer rounding follows the row's state, not the coefficient's
        // sign: away from zero when the row is violated, toward zero when it
        // is satisfied.  See `round_tight_delta` for the rule and the paper
        // citation (issue #123).
        if (is_int(j)) {
            delta = round_tight_delta(delta, row_violated);
        }
        // Clamp to variable bounds (Paper Eq 5 min/max with l_j, u_j).
        double new_val = solution[j] + delta;
        if (new_val < col_lb[j]) {
            delta = col_lb[j] - solution[j];
        } else if (new_val > col_ub[j]) {
            delta = col_ub[j] - solution[j];
        }
    }
    return delta;
}

void WorkerCtx::update_weights(Rng& rng, bool is_feasible, bool best_feasible, double best_obj) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (coin(rng) >= kSmoothProb) {
        // With probability 1 - sp: strengthen
        if (is_feasible) {
            obj_weight += 1;
        } else {
            for (auto ci : violated) {
                weight[ci] += 1;
            }
        }
    } else {
        // With probability sp: smooth (weaken)
        bool obj_better =
            best_feasible && (minimize ? (current_obj < best_obj) : (current_obj > best_obj));
        if (obj_better && obj_weight > 1) {
            obj_weight -= 1;
        }
        for (auto ci : satisfied) {
            if (weight[ci] > 1) {
                weight[ci] -= 1;
            }
        }
    }
}

// --- LiftCache method implementations ---

// Cognitive complexity 43 (threshold 25).  Kept whole: incremental score recomputation for one
// variable; the branch structure is the score definition itself. Decomposing it would move work
// across a worker's inner loop, and the closeout takes no unmeasured performance risk; the
// standards also rank fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void LiftCache::recompute_one(HighsInt j, WorkerCtx& ctx) {
    double old_score = score[j];
    if (std::abs(ctx.col_cost[j]) < kEpsZero) {
        score[j] = 0.0;
        dirty[j] = false;
        if (old_score > 0.0 && in_positive[j]) {
            in_positive[j] = false;
            // lazy removal: stale entries filtered during scan
        }
        return;
    }
    // Compute lift bounds
    ctx.effort += ctx.csc.col_start[j + 1] - ctx.csc.col_start[j];
    double lo_j = ctx.col_lb[j];
    double hi_j = ctx.col_ub[j];
    for (HighsInt p = ctx.csc.col_start[j]; p < ctx.csc.col_start[j + 1]; ++p) {
        HighsInt i = ctx.csc.col_row[p];
        double coeff = ctx.csc.col_val[p];
        if (std::abs(coeff) < kEpsZero) {
            continue;
        }
        double residual = ctx.lhs[i] - (coeff * ctx.solution[j]);
        if (ctx.row_hi[i] < kHighsInf) {
            double bound = (ctx.row_hi[i] - residual) / coeff;
            if (coeff > 0) {
                hi_j = std::min(hi_j, bound);
            } else {
                lo_j = std::max(lo_j, bound);
            }
        }
        if (ctx.row_lo[i] > -kHighsInf) {
            double bound = (ctx.row_lo[i] - residual) / coeff;
            if (coeff > 0) {
                lo_j = std::max(lo_j, bound);
            } else {
                hi_j = std::min(hi_j, bound);
            }
        }
    }
    if (ctx.is_int(j)) {
        lo_j = std::ceil(lo_j - ctx.feastol);
        hi_j = std::floor(hi_j + ctx.feastol);
    }
    lo[j] = lo_j;
    hi[j] = hi_j;

    if (lo_j > hi_j) {
        score[j] = 0.0;
    } else {
        double target;
        if (ctx.minimize) {
            target = (ctx.col_cost[j] > 0) ? lo_j : hi_j;
        } else {
            target = (ctx.col_cost[j] > 0) ? hi_j : lo_j;
        }
        target = ctx.clamp_and_round(j, target);
        if (std::abs(target - ctx.solution[j]) < kEpsZero) {
            score[j] = 0.0;
        } else {
            double obj_delta = ctx.col_cost[j] * (target - ctx.solution[j]);
            if (!ctx.minimize) {
                obj_delta = -obj_delta;
            }
            score[j] = -obj_delta;  // positive = improving
        }
    }
    // Maintain positive-lift list
    if (score[j] > 0.0) {
        if (!in_positive[j]) {
            in_positive[j] = true;
            positive_list.push_back(j);
        }
    } else {
        if (in_positive[j]) {
            in_positive[j] = false;
            // lazy removal: stale entries filtered during scan
        }
    }
    dirty[j] = false;
}

void LiftCache::recompute_all(WorkerCtx& ctx) {
    if (all_dirty) {
        // Only recompute columns with nonzero cost; zero-cost columns
        // always have score=0 and never need lift recomputation.
        if (costed_vars != nullptr) {
            for (HighsInt j : *costed_vars) {
                recompute_one(j, ctx);
            }
        } else {
            for (HighsInt j = 0; j < ctx.ncol; ++j) {
                recompute_one(j, ctx);
            }
        }
        dirty_list.clear();
    } else {
        for (HighsInt j : dirty_list) {
            if (dirty[j]) {
                recompute_one(j, ctx);
            }
        }
        dirty_list.clear();
    }
    all_dirty = false;
}

}  // namespace local_mip_detail
