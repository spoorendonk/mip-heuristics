#include "local_mip.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

namespace {

double compute_objective(const HighsLp* model,
                         const std::vector<double>& solution) {
  double obj = model->offset_;
  for (HighsInt j = 0; j < model->num_col_; ++j)
    obj += model->col_cost_[j] * solution[j];
  return obj;
}

}  // namespace

namespace local_mip {

void run(HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;
  if (mipdata->incumbent.empty()) return;

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                        mipdata->ARvalue_);
  std::mt19937 rng(mipdata->numImprovingSols + 137);

  const double dl = heuristic_deadline(mipsolver.options_mip_->time_limit,
                                       mipsolver.timer_.read());
  auto result = worker(mipsolver, csc, rng, nullptr, dl);
  if (result.found_feasible)
    mipdata->trySolution(result.solution, kSolutionSourceLocalMIP);
}

HeuristicResult worker(HighsMipSolver& mipsolver, const CscMatrix& csc,
                       std::mt19937& rng, const double* initial_solution,
                       double deadline) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const auto& ARstart = mipdata->ARstart_;
  const auto& ARindex = mipdata->ARindex_;
  const auto& ARvalue = mipdata->ARvalue_;
  const auto& col_lb = model->col_lower_;
  const auto& col_ub = model->col_upper_;
  const auto& col_cost = model->col_cost_;
  const auto& row_lo = model->row_lower_;
  const auto& row_hi = model->row_upper_;
  const auto& integrality = model->integrality_;
  const double feastol = mipdata->feastol;
  const bool minimize = (model->sense_ == ObjSense::kMinimize);

  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;

  HeuristicResult result;

  if (ncol == 0 || nrow == 0) return result;

  const auto& col_start = csc.col_start;
  const auto& col_row = csc.col_row;
  const auto& col_val = csc.col_val;

  auto is_integer = [&](HighsInt j) { return ::is_integer(integrality, j); };

  // --- Constants ---
  constexpr double kViolTol = 5e-7;
  constexpr HighsInt kMaxSteps = 500000;
  constexpr HighsInt kRestartInterval = 200000;
  constexpr HighsInt kTermCheckInterval = 1000;
  constexpr HighsInt kActivityPeriod = 100000;
  constexpr double kSmoothProb = 1e-4;
  constexpr HighsInt kBmsConstraints = 12;
  constexpr HighsInt kBmsBudget = 2250;
  constexpr HighsInt kBmsSatCon = 1;
  constexpr HighsInt kBmsSatBudget = 80;
  constexpr HighsInt kEasyBudget = 5;
  constexpr HighsInt kTabuBase = 10;
  constexpr HighsInt kTabuVar = 5;

  // --- Solution and constraint state ---
  std::vector<double> solution(ncol);
  std::vector<double> lhs(nrow);
  std::vector<uint64_t> weight(nrow, 1);

  // Batch violation cache: memoize compute_violation(i, lhs[i]) within a batch
  constexpr double kViolCacheSentinel = -1.0;
  std::vector<double> viol_cache(nrow, kViolCacheSentinel);
  std::vector<HighsInt> viol_cache_used;
  viol_cache_used.reserve(nrow);

  // Violated list with O(1) add/remove
  std::vector<HighsInt> violated;
  std::vector<HighsInt> violated_pos(nrow, -1);
  violated.reserve(nrow);

  auto add_violated = [&](HighsInt i) {
    if (violated_pos[i] != -1) return;
    violated_pos[i] = static_cast<HighsInt>(violated.size());
    violated.push_back(i);
  };
  auto remove_violated = [&](HighsInt i) {
    HighsInt p = violated_pos[i];
    if (p == -1) return;
    HighsInt last = violated.back();
    violated[p] = last;
    violated_pos[last] = p;
    violated.pop_back();
    violated_pos[i] = -1;
  };

  // Satisfied list (non-equality rows only, for sat-MTM diversification)
  std::vector<HighsInt> satisfied;
  std::vector<HighsInt> satisfied_pos(nrow, -1);
  satisfied.reserve(nrow);

  auto is_equality = [&](HighsInt i) -> bool {
    return row_lo[i] == row_hi[i] && row_lo[i] > -kHighsInf;
  };
  auto add_satisfied = [&](HighsInt i) {
    if (is_equality(i)) return;
    if (satisfied_pos[i] != -1) return;
    satisfied_pos[i] = static_cast<HighsInt>(satisfied.size());
    satisfied.push_back(i);
  };
  auto remove_satisfied = [&](HighsInt i) {
    HighsInt p = satisfied_pos[i];
    if (p == -1) return;
    HighsInt last = satisfied.back();
    satisfied[p] = last;
    satisfied_pos[last] = p;
    satisfied.pop_back();
    satisfied_pos[i] = -1;
  };

  auto compute_violation = [&](HighsInt i, double l) -> double {
    return row_violation(l, row_lo[i], row_hi[i]);
  };
  auto is_violated = [&](HighsInt i, double l) -> bool {
    return l > row_hi[i] + feastol || l < row_lo[i] - feastol;
  };

  auto update_violated = [&](HighsInt i) {
    double viol = compute_violation(i, lhs[i]);
    bool was_violated = (violated_pos[i] != -1);
    bool now_violated = (viol > kViolTol);
    if (now_violated && !was_violated) {
      add_violated(i);
      remove_satisfied(i);
    } else if (!now_violated && was_violated) {
      remove_violated(i);
      add_satisfied(i);
    }
  };

  // Directional tabu
  std::vector<HighsInt> tabu_inc_until(ncol, 0);
  std::vector<HighsInt> tabu_dec_until(ncol, 0);

  auto is_tabu = [&](HighsInt j, double delta, HighsInt step) -> bool {
    if (delta > 0 && step < tabu_inc_until[j]) return true;
    if (delta < 0 && step < tabu_dec_until[j]) return true;
    return false;
  };

  // Best solution tracking
  bool best_feasible = false;
  double best_objective = minimize ? std::numeric_limits<double>::infinity()
                                   : -std::numeric_limits<double>::infinity();
  std::vector<double> best_solution;

  // Lift cache
  std::vector<double> lift_lo(ncol), lift_hi(ncol), lift_score(ncol);
  std::vector<bool> lift_dirty(ncol, true);
  std::vector<HighsInt> lift_dirty_list;
  lift_dirty_list.reserve(ncol);
  bool lift_all_dirty = true;

  // Positive-lift list: columns with lift_score > 0 (avoids O(ncol) scan)
  std::vector<HighsInt> lift_positive_list;
  std::vector<bool> lift_in_positive(ncol, false);
  lift_positive_list.reserve(ncol);

  auto clamp_and_round = [&](HighsInt j, double val) -> double {
    return clamp_round(val, col_lb[j], col_ub[j], is_integer(j));
  };

  auto recompute_obj = [&]() -> double {
    return compute_objective(model, solution);
  };

  double current_obj = 0.0;
  bool was_infeasible = true;
  HighsInt feasible_recheck_counter = 0;
  constexpr HighsInt kFeasibleRecheckPeriod = 100;

  // Rebuild all constraint state from scratch
  auto rebuild_state = [&]() {
    was_infeasible = true;
    feasible_recheck_counter = 0;
    violated.clear();
    std::fill(violated_pos.begin(), violated_pos.end(), -1);
    satisfied.clear();
    std::fill(satisfied_pos.begin(), satisfied_pos.end(), -1);
    for (HighsInt i = 0; i < nrow; ++i) {
      double l = 0.0;
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k)
        l += ARvalue[k] * solution[ARindex[k]];
      lhs[i] = l;
      if (compute_violation(i, l) > kViolTol)
        add_violated(i);
      else
        add_satisfied(i);
    }
    lift_all_dirty = true;
    lift_dirty_list.clear();
    std::fill(lift_dirty.begin(), lift_dirty.end(), true);
    lift_positive_list.clear();
    std::fill(lift_in_positive.begin(), lift_in_positive.end(), false);
    current_obj = recompute_obj();
  };

  // Apply a move: update solution, LHS, violated/satisfied lists, lift dirty
  auto apply_move = [&](HighsInt j, double new_val) {
    double old_val = solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < 1e-15) return;
    solution[j] = new_val;
    current_obj += col_cost[j] * delta;
    if (!lift_dirty[j]) { lift_dirty[j] = true; lift_dirty_list.push_back(j); }
    for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
      HighsInt i = col_row[p];
      lhs[i] += col_val[p] * delta;
      update_violated(i);
      // Invalidate lift cache for all variables sharing this row
      if (!lift_all_dirty) {
        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
          HighsInt jj = ARindex[k];
          if (!lift_dirty[jj]) { lift_dirty[jj] = true; lift_dirty_list.push_back(jj); }
        }
      }
    }
  };

  // Tight delta for ranged rows: compute delta that satisfies the binding bound
  auto compute_tight_delta = [&](HighsInt i, HighsInt j,
                                 double coeff) -> double {
    if (std::abs(coeff) < 1e-15) return 0.0;
    double l = lhs[i];
    // Determine gap to the binding bound
    double gap;
    bool row_violated = is_violated(i, l);
    // NOLINTBEGIN(bugprone-branch-clone) — same expression form, different
    // bounds
    if (l > row_hi[i] + feastol)
      gap = l - row_hi[i];  // upper violated
    else if (l < row_lo[i] - feastol)
      gap = l - row_lo[i];  // lower violated
    else if (row_hi[i] < kHighsInf)
      gap = l - row_hi[i];  // satisfied: push toward upper
    else
      gap = l - row_lo[i];  // satisfied: push toward lower
    // NOLINTEND(bugprone-branch-clone)

    double delta = -gap / coeff;

    if (is_equality(i)) {
      // Equality: round for integers
      if (is_integer(j)) delta = std::round(delta);
      // Clamp direction-aware
      double new_val = solution[j] + delta;
      if (new_val < col_lb[j] || new_val > col_ub[j]) {
        if ((gap > 0 && coeff > 0) || (gap < 0 && coeff < 0))
          delta = col_lb[j] - solution[j];
        else
          delta = col_ub[j] - solution[j];
      }
    } else {
      // Inequality: floor/ceil for integers based on coeff sign
      if (is_integer(j)) {
        delta = (coeff > 0) ? std::floor(delta) : std::ceil(delta);
      }
      // Clamp
      double new_val = solution[j] + delta;
      if (new_val < col_lb[j] || new_val > col_ub[j]) {
        if (row_violated) {
          // Push toward violation-reducing bound
          delta = (coeff > 0) ? (col_lb[j] - solution[j])
                              : (col_ub[j] - solution[j]);
        } else {
          // Satisfied: push toward slack-creating bound
          delta = (coeff > 0) ? (col_ub[j] - solution[j])
                              : (col_lb[j] - solution[j]);
        }
      }
    }
    return delta;
  };

  // Combined candidate scores: progress (weighted violation improvement)
  // and bonus (newly satisfied count + small objective term) in one pass
  auto compute_candidate_scores =
      [&](HighsInt j, double new_val) -> std::pair<double, double> {
    double old_val = solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < 1e-15)
      return {-std::numeric_limits<double>::infinity(), 0.0};

    double progress = 0.0;
    double bonus = 0.0;
    for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
      HighsInt i = col_row[p];
      double coeff = col_val[p];
      double old_lhs = lhs[i];
      double new_lhs = old_lhs + coeff * delta;
      double old_viol;
      if (viol_cache[i] >= 0.0) {
        old_viol = viol_cache[i];
      } else {
        old_viol = compute_violation(i, old_lhs);
        viol_cache[i] = old_viol;
        viol_cache_used.push_back(i);
      }
      double new_viol = compute_violation(i, new_lhs);
      progress += static_cast<double>(weight[i]) * (old_viol - new_viol);
      if (old_viol > kViolTol && new_viol <= kViolTol) bonus += 1.0;
    }
    double obj_delta = col_cost[j] * delta;
    if (!minimize) obj_delta = -obj_delta;
    bonus -= 0.001 * obj_delta;
    return {progress, bonus};
  };

  // Aspiration: would applying this move to the current solution beat best?
  auto is_aspiration = [&](HighsInt j, double new_val) -> bool {
    if (!best_feasible) return false;
    double delta = new_val - solution[j];
    double obj_delta = col_cost[j] * delta;
    double new_obj = current_obj + obj_delta;
    return minimize ? (new_obj < best_objective - 1e-9)
                    : (new_obj > best_objective + 1e-9);
  };

  // Breakthrough delta: move toward best objective value.
  // Caller must supply current_obj to avoid redundant O(ncol) recomputation.
  auto compute_breakthrough_delta = [&](HighsInt j,
                                        double cur_obj) -> double {
    double obj_coeff = col_cost[j];
    if (std::abs(obj_coeff) < 1e-15) return 0.0;

    double obj_gap = cur_obj - best_objective;
    if (!minimize) obj_gap = -obj_gap;

    double delta = -obj_gap / obj_coeff;

    if (is_integer(j)) {
      delta = (obj_coeff > 0) ? std::floor(delta) : std::ceil(delta);
    }
    double new_val = solution[j] + delta;
    if (new_val < col_lb[j] || new_val > col_ub[j]) {
      delta = (obj_coeff > 0) ? (col_lb[j] - solution[j])
                              : (col_ub[j] - solution[j]);
    }
    return delta;
  };

  // Candidate struct
  struct Candidate {
    HighsInt var_idx = -1;
    double new_val = 0.0;
    double score = -std::numeric_limits<double>::infinity();
    double bonus = 0.0;
  };

  // Batch candidate buffer
  struct BatchCand {
    HighsInt var_idx;
    double new_val;
  };
  std::vector<BatchCand> batch;
  batch.reserve(kBmsBudget);

  // Select best from batch with tabu/aspiration filtering
  auto select_best_from_batch = [&](HighsInt step,
                                    bool aspiration) -> Candidate {
    Candidate best;
    for (const auto& c : batch) {
      double delta = c.new_val - solution[c.var_idx];
      if (std::abs(delta) < 1e-15) continue;

      if (is_tabu(c.var_idx, delta, step)) {
        if (!(aspiration && is_aspiration(c.var_idx, c.new_val))) continue;
      }

      auto [prog, bon] = compute_candidate_scores(c.var_idx, c.new_val);

      if (prog > best.score + kViolTol) {
        best = {c.var_idx, c.new_val, prog, bon};
      } else if (prog > best.score - kViolTol) {
        if (bon > best.bonus) best = {c.var_idx, c.new_val, prog, bon};
      }
    }
    // Reset viol cache after batch evaluation
    for (HighsInt i : viol_cache_used) viol_cache[i] = kViolCacheSentinel;
    viol_cache_used.clear();
    return best;
  };

  // Append candidate to batch (clamp+round, skip zero-delta)
  auto append_candidate = [&](HighsInt j, double delta) {
    double new_val = clamp_and_round(j, solution[j] + delta);
    if (std::abs(new_val - solution[j]) < 1e-15) return;
    batch.push_back({j, new_val});
  };

  // Sampled constraint buffer for BMS
  struct WeightedCon {
    HighsInt ci;
    uint64_t w;
  };
  std::vector<WeightedCon> sampled;
  sampled.reserve(static_cast<size_t>(kBmsConstraints) * 3);

  // Lift bounds computation (ranged rows)
  auto compute_lift_bounds = [&](HighsInt j) -> std::pair<double, double> {
    double lo = col_lb[j];
    double hi = col_ub[j];
    for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
      HighsInt i = col_row[p];
      double coeff = col_val[p];
      if (std::abs(coeff) < 1e-15) continue;
      double residual = lhs[i] - coeff * solution[j];
      // From row_hi: coeff*x + residual <= row_hi
      if (row_hi[i] < kHighsInf) {
        double bound = (row_hi[i] - residual) / coeff;
        if (coeff > 0)
          hi = std::min(hi, bound);
        else
          lo = std::max(lo, bound);
      }
      // From row_lo: coeff*x + residual >= row_lo
      if (row_lo[i] > -kHighsInf) {
        double bound = (row_lo[i] - residual) / coeff;
        if (coeff > 0)
          lo = std::max(lo, bound);
        else
          hi = std::min(hi, bound);
      }
    }
    if (is_integer(j)) {
      lo = std::ceil(lo - feastol);
      hi = std::floor(hi + feastol);
    }
    return {lo, hi};
  };

  // Recompute lift cache for dirty variables
  auto recompute_one_lift = [&](HighsInt j) {
    double old_score = lift_score[j];
    if (std::abs(col_cost[j]) < 1e-15) {
      lift_score[j] = 0.0;
      lift_dirty[j] = false;
      if (old_score > 0.0 && lift_in_positive[j]) {
        lift_in_positive[j] = false;
        // lazy removal: stale entries filtered during scan
      }
      return;
    }
    auto [lo, hi] = compute_lift_bounds(j);
    lift_lo[j] = lo;
    lift_hi[j] = hi;
    if (lo > hi) {
      lift_score[j] = 0.0;
    } else {
      double target;
      if (minimize)
        target = (col_cost[j] > 0) ? lo : hi;
      else
        target = (col_cost[j] > 0) ? hi : lo;
      target = clamp_and_round(j, target);
      if (std::abs(target - solution[j]) < 1e-15)
        lift_score[j] = 0.0;
      else {
        double obj_delta = col_cost[j] * (target - solution[j]);
        if (!minimize) obj_delta = -obj_delta;
        lift_score[j] = -obj_delta;  // positive = improving
      }
    }
    // Maintain positive-lift list
    if (lift_score[j] > 0.0) {
      if (!lift_in_positive[j]) {
        lift_in_positive[j] = true;
        lift_positive_list.push_back(j);
      }
    } else {
      if (lift_in_positive[j]) {
        lift_in_positive[j] = false;
        // lazy removal: stale entries filtered during scan
      }
    }
    lift_dirty[j] = false;
  };

  auto recompute_lift_cache = [&]() {
    if (lift_all_dirty) {
      for (HighsInt j = 0; j < ncol; ++j) recompute_one_lift(j);
      lift_dirty_list.clear();
    } else {
      for (HighsInt j : lift_dirty_list) {
        if (lift_dirty[j]) recompute_one_lift(j);
      }
      lift_dirty_list.clear();
    }
    lift_all_dirty = false;
  };

  // Weight update with geometric-skip smooth decay
  auto update_weights = [&]() {
    for (auto ci : violated) weight[ci] += 1;
    if (!satisfied.empty()) {
      std::geometric_distribution<HighsInt> skip_dist(kSmoothProb);
      HighsInt idx = skip_dist(rng);
      while (idx < static_cast<HighsInt>(satisfied.size())) {
        HighsInt ci = satisfied[idx];
        if (weight[ci] > 1) weight[ci] -= 1;
        idx += 1 + skip_dist(rng);
      }
    }
  };

  // Precompute variables with nonzero cost for breakthrough moves
  std::vector<HighsInt> costed_vars;
  for (HighsInt j = 0; j < ncol; ++j)
    if (std::abs(col_cost[j]) >= 1e-15) costed_vars.push_back(j);

  // --- Initialize solution ---
  if (initial_solution) {
    for (HighsInt j = 0; j < ncol; ++j) {
      double v = initial_solution[j];
      if (is_integer(j)) v = std::round(v);
      solution[j] = std::max(col_lb[j], std::min(col_ub[j], v));
    }
  } else if (!mipdata->incumbent.empty()) {
    for (HighsInt j = 0; j < ncol; ++j) {
      double v = mipdata->incumbent[j];
      if (is_integer(j)) v = std::round(v);
      solution[j] = std::max(col_lb[j], std::min(col_ub[j], v));
    }
  } else {
    for (HighsInt j = 0; j < ncol; ++j) {
      if (mipdata->domain.isBinary(j)) {
        solution[j] = 0.0;
      } else if (is_integer(j)) {
        double lo = std::max(col_lb[j], -1e8);
        double hi = std::min(col_ub[j], lo + 100.0);
        solution[j] = std::max(
            col_lb[j], std::min(col_ub[j], std::round((lo + hi) * 0.5)));
      } else {
        double val = 0.0;
        if (col_lb[j] > 0.0)
          val = col_lb[j];
        else if (col_ub[j] < 0.0)
          val = col_ub[j];
        solution[j] = std::max(col_lb[j], std::min(col_ub[j], val));
      }
    }
  }

  // Build initial LHS and violated/satisfied lists
  rebuild_state();

  HighsInt steps_since_improvement = 0;
  HighsInt restart_count = 0;
  const double effective_deadline =
      std::min(mipsolver.options_mip_->time_limit, deadline);

  // --- Main loop ---
  for (HighsInt step = 0; step < kMaxSteps; ++step) {
    if (step % kTermCheckInterval == 0 &&
        (mipdata->terminatorTerminated() ||
         mipsolver.timer_.read() >= effective_deadline))
      break;

    bool feasible_mode = violated.empty();

    if (feasible_mode) {
      // Full O(nnz) recheck on infeasible→feasible transition or periodically;
      // otherwise trust incremental lhs[] (O(nrow) check only).
      bool need_full_recheck = was_infeasible ||
                               (feasible_recheck_counter % kFeasibleRecheckPeriod == 0);
      was_infeasible = false;
      ++feasible_recheck_counter;

      bool truly_feasible = true;
      if (need_full_recheck) {
        for (HighsInt i = 0; i < nrow; ++i) {
          double l = 0.0;
          for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k)
            l += ARvalue[k] * solution[ARindex[k]];
          lhs[i] = l;
          if (is_violated(i, l)) {
            truly_feasible = false;
            add_violated(i);
            remove_satisfied(i);
          }
        }
      }
      // When !need_full_recheck, trust incremental state: apply_move's
      // update_violated() already maintains the violated set for every
      // row touched by each move, so no row can become violated without
      // being caught.  The periodic full recheck guards against FP drift.
      if (!truly_feasible) continue;

      // Track best solution
      double obj = current_obj;
      bool improved = false;
      if (!best_feasible)
        improved = true;
      else if (minimize)
        improved = (obj < best_objective - 1e-9);
      else
        improved = (obj > best_objective + 1e-9);

      if (improved) {
        // Full recheck before recording best (guard against FP drift)
        if (!need_full_recheck) {
          bool still_ok = true;
          for (HighsInt i = 0; i < nrow; ++i) {
            double l = 0.0;
            for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k)
              l += ARvalue[k] * solution[ARindex[k]];
            lhs[i] = l;
            if (is_violated(i, l)) { still_ok = false; break; }
          }
          if (!still_ok) { rebuild_state(); continue; }
        }
        best_feasible = true;
        best_objective = obj;
        best_solution = solution;
        steps_since_improvement = 0;
      }

      // Lift move: find variable giving best feasible objective improvement
      recompute_lift_cache();
      Candidate lift_best;
      lift_best.score = 0.0;  // must strictly improve
      // Compact stale entries and find best lift in a single pass
      {
        HighsInt write = 0;
        for (HighsInt read = 0; read < static_cast<HighsInt>(lift_positive_list.size()); ++read) {
          HighsInt j = lift_positive_list[read];
          if (!lift_in_positive[j]) continue;
          lift_positive_list[write++] = j;
          if (lift_score[j] <= lift_best.score) continue;
          double lo = lift_lo[j], hi = lift_hi[j];
          if (lo > hi) continue;
          double target;
          if (minimize)
            target = (col_cost[j] > 0) ? lo : hi;
          else
            target = (col_cost[j] > 0) ? hi : lo;
          target = clamp_and_round(j, target);
          if (std::abs(target - solution[j]) < 1e-15) continue;
          lift_best = {j, target, lift_score[j], 0.0};
        }
        lift_positive_list.resize(write);
      }

      if (lift_best.var_idx != -1) {
        double delta = lift_best.new_val - solution[lift_best.var_idx];
        apply_move(lift_best.var_idx, lift_best.new_val);
        HighsInt tabu_len = kTabuBase + static_cast<HighsInt>(rng() % kTabuVar);
        if (delta > 0)
          tabu_dec_until[lift_best.var_idx] = step + tabu_len;
        else
          tabu_inc_until[lift_best.var_idx] = step + tabu_len;
        continue;
      }

      update_weights();
      ++steps_since_improvement;
    } else {
      // --- Infeasible mode ---
      was_infeasible = true;

      // Phase 1: BMS tight moves from violated constraints
      HighsInt num_to_sample =
          std::min(kBmsConstraints * 3, static_cast<HighsInt>(violated.size()));
      HighsInt num_to_keep =
          std::min(kBmsConstraints, static_cast<HighsInt>(violated.size()));

      sampled.clear();
      if (num_to_sample == static_cast<HighsInt>(violated.size())) {
        for (auto ci : violated) sampled.push_back({ci, weight[ci]});
      } else {
        for (HighsInt s = 0; s < num_to_sample; ++s) {
          HighsInt idx = static_cast<HighsInt>(rng() % violated.size());
          sampled.push_back({violated[idx], weight[violated[idx]]});
        }
      }

      if (static_cast<HighsInt>(sampled.size()) > num_to_keep) {
        std::partial_sort(sampled.begin(), sampled.begin() + num_to_keep,
                          sampled.end(),
                          [](const WeightedCon& a, const WeightedCon& b) {
                            return a.w > b.w;
                          });
        sampled.resize(num_to_keep);
      }

      batch.clear();
      HighsInt budget_remaining = kBmsBudget;

      for (auto& [ci, w] : sampled) {
        (void)w;
        if (budget_remaining <= 0) break;
        for (HighsInt k = ARstart[ci];
             k < ARstart[ci + 1] && budget_remaining > 0; ++k) {
          HighsInt j = ARindex[k];
          --budget_remaining;
          double delta = compute_tight_delta(ci, j, ARvalue[k]);
          append_candidate(j, delta);
        }
      }

      // Phase 2: Breakthrough moves
      if (best_feasible) {
        for (HighsInt j : costed_vars) {
          double delta = compute_breakthrough_delta(j, current_obj);
          append_candidate(j, delta);
        }
      }

      Candidate cand = select_best_from_batch(step, true);

      // Phase 3: Random violated constraint fallback
      if (cand.var_idx == -1 || cand.score < -kViolTol) {
        batch.clear();
        HighsInt ci = violated[rng() % violated.size()];
        for (HighsInt k = ARstart[ci]; k < ARstart[ci + 1]; ++k) {
          HighsInt j = ARindex[k];
          double delta = compute_tight_delta(ci, j, ARvalue[k]);
          double new_val = clamp_and_round(j, solution[j] + delta);
          if (std::abs(new_val - solution[j]) < 1e-15) continue;
          batch.push_back({j, new_val});
        }
        auto fallback = select_best_from_batch(step, false);
        bool better = fallback.score > cand.score + kViolTol ||
                      (fallback.score > cand.score - kViolTol &&
                       fallback.bonus > cand.bonus);
        if (better) cand = fallback;
      }

      // Phase 4: Perturbation (last resort)
      if (cand.var_idx == -1 && !violated.empty()) {
        HighsInt ci = violated[rng() % violated.size()];
        HighsInt row_len = ARstart[ci + 1] - ARstart[ci];
        if (row_len > 0) {
          HighsInt k = ARstart[ci] + static_cast<HighsInt>(rng() % row_len);
          HighsInt j = ARindex[k];
          double new_val;
          if (mipdata->domain.isBinary(j)) {
            new_val = (solution[j] < 0.5) ? 1.0 : 0.0;
          } else if (is_integer(j)) {
            HighsInt dir = (rng() % 2 == 0) ? 1 : -1;
            new_val = clamp_and_round(j, solution[j] + dir);
          } else {
            double range = std::min(col_ub[j], col_lb[j] + 1e6) - col_lb[j];
            double perturbation = std::uniform_real_distribution<double>(
                -0.1 * range, 0.1 * range)(rng);
            new_val = clamp_and_round(j, solution[j] + perturbation);
          }
          if (std::abs(new_val - solution[j]) > 1e-15) {
            auto [prog, bon] = compute_candidate_scores(j, new_val);
            // Clean up viol_cache (compute_candidate_scores populated it
            // outside select_best_from_batch which normally handles cleanup)
            for (HighsInt ii : viol_cache_used) viol_cache[ii] = kViolCacheSentinel;
            viol_cache_used.clear();
            cand = {j, new_val, prog, bon};
          }
        }
      }

      // Phase 5: Sat-MTM diversification
      if (cand.score <= 0 && best_feasible && !satisfied.empty()) {
        batch.clear();
        HighsInt num_sat_sample =
            std::min(kBmsSatCon, static_cast<HighsInt>(satisfied.size()));
        HighsInt sat_budget = kBmsSatBudget;
        for (HighsInt s = 0; s < num_sat_sample && sat_budget > 0; ++s) {
          HighsInt ci = satisfied[rng() % satisfied.size()];
          for (HighsInt k = ARstart[ci]; k < ARstart[ci + 1] && sat_budget > 0;
               ++k) {
            HighsInt j = ARindex[k];
            --sat_budget;
            double delta = compute_tight_delta(ci, j, ARvalue[k]);
            double new_val = clamp_and_round(j, solution[j] + delta);
            if (std::abs(new_val - solution[j]) < 1e-15) continue;
            batch.push_back({j, new_val});
          }
        }
        auto sat_cand = select_best_from_batch(step, false);
        if (sat_cand.var_idx != -1 && sat_cand.score > cand.score)
          cand = sat_cand;
      }

      // Phase 6: Easy moves + weight update
      if (cand.score <= 0) {
        update_weights();
        batch.clear();
        HighsInt num_easy = std::min(kEasyBudget, ncol);
        for (HighsInt s = 0; s < num_easy; ++s) {
          HighsInt j = static_cast<HighsInt>(rng() % ncol);
          // Try: toward 0/nearest bound
          double target;
          if (col_lb[j] > 0)
            target = col_lb[j];
          else if (col_ub[j] < 0)
            target = col_ub[j];
          else
            target = 0.0;
          append_candidate(j, target - solution[j]);
          // Try: toward lower bound
          if (col_lb[j] > -1e15 && col_lb[j] < 0)
            append_candidate(j, col_lb[j] - solution[j]);
          // Try: toward upper bound
          if (col_ub[j] < 1e15 && col_ub[j] > 0)
            append_candidate(j, col_ub[j] - solution[j]);
          // Try: midpoint for continuous
          if (!is_integer(j) && col_lb[j] > -1e15 && col_ub[j] < 1e15)
            append_candidate(j, (col_lb[j] + col_ub[j]) * 0.5 - solution[j]);
        }
        auto easy_cand = select_best_from_batch(step, false);
        if (easy_cand.var_idx != -1 &&
            (cand.var_idx == -1 || easy_cand.score > cand.score))
          cand = easy_cand;
      }

      // Apply move
      if (cand.var_idx != -1) {
        double delta = cand.new_val - solution[cand.var_idx];
        apply_move(cand.var_idx, cand.new_val);
        HighsInt tabu_len = kTabuBase + static_cast<HighsInt>(rng() % kTabuVar);
        if (delta > 0)
          tabu_dec_until[cand.var_idx] = step + tabu_len;
        else
          tabu_inc_until[cand.var_idx] = step + tabu_len;
      }

      ++steps_since_improvement;
      if (violated.empty()) steps_since_improvement = 0;
    }

    // Activity refresh: recompute all LHS to prevent FP drift
    if (step % kActivityPeriod == 0 && step > 0) {
      rebuild_state();
    }

    // Restart logic
    if (steps_since_improvement >= kRestartInterval) {
      steps_since_improvement = 0;
      ++restart_count;

      // Try incumbent on odd restarts
      if (best_feasible && (restart_count % 2 == 1)) {
        solution = best_solution;
      } else {
        // Random restart
        for (HighsInt j = 0; j < ncol; ++j) {
          if (mipdata->domain.isBinary(j)) {
            solution[j] = (rng() % 2 == 0) ? 0.0 : 1.0;
          } else if (is_integer(j)) {
            double lo = std::max(col_lb[j], -1e8);
            double hi = std::min(col_ub[j], lo + 100.0);
            solution[j] = std::max(
                col_lb[j],
                std::min(col_ub[j],
                         std::round(std::uniform_real_distribution<double>(
                             lo, hi)(rng))));
          } else {
            double lo = col_lb[j] > -kHighsInf ? col_lb[j] : -1e6;
            double hi = col_ub[j] < kHighsInf ? col_ub[j] : lo + 1e6;
            if (hi > lo)
              solution[j] = std::uniform_real_distribution<double>(lo, hi)(rng);
            else
              solution[j] = lo;
          }
        }
      }

      rebuild_state();
      std::fill(tabu_inc_until.begin(), tabu_inc_until.end(), 0);
      std::fill(tabu_dec_until.begin(), tabu_dec_until.end(), 0);
    }

    ++result.effort;
  }

  if (best_feasible) {
    result.found_feasible = true;
    result.objective = best_objective;
    result.solution = std::move(best_solution);
  }

  return result;
}

}  // namespace local_mip
