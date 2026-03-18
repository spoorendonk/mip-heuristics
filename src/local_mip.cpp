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

  // Cap LocalMIP at 10% of time limit (min 5s, max 30s)
  const double tl = mipsolver.options_mip_->time_limit;
  const double dl = mipsolver.timer_.read() +
                    std::min(30.0, std::max(5.0, 0.1 * tl));
  auto result = worker(mipsolver, csc, rng, nullptr, dl);
  if (result.found_feasible)
    mipdata->trySolution(result.solution, kSolutionSourceHeuristic);
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

  // Violation helpers (ranged rows)
  auto compute_violation = [&](HighsInt i, double l) -> double {
    return std::max(0.0, l - row_hi[i]) + std::max(0.0, row_lo[i] - l);
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
  bool lift_all_dirty = true;

  // Clamp and round
  auto clamp_and_round = [&](HighsInt j, double val) -> double {
    if (is_integer(j)) val = std::round(val);
    return std::max(col_lb[j], std::min(col_ub[j], val));
  };

  // Compute objective
  auto compute_objective = [&]() -> double {
    double obj = model->offset_;
    for (HighsInt j = 0; j < ncol; ++j) obj += col_cost[j] * solution[j];
    return obj;
  };

  double current_obj = 0.0;

  // Rebuild all constraint state from scratch
  auto rebuild_state = [&]() {
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
    current_obj = compute_objective();
  };

  // Apply a move: update solution, LHS, violated/satisfied lists, lift dirty
  auto apply_move = [&](HighsInt j, double new_val) {
    double old_val = solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < 1e-15) return;
    solution[j] = new_val;
    current_obj += col_cost[j] * delta;
    lift_dirty[j] = true;
    for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
      HighsInt i = col_row[p];
      lhs[i] += col_val[p] * delta;
      update_violated(i);
      // Invalidate lift cache for all variables sharing this row
      if (!lift_all_dirty) {
        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k)
          lift_dirty[ARindex[k]] = true;
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
      double old_viol = compute_violation(i, old_lhs);
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
  auto recompute_lift_cache = [&]() {
    for (HighsInt j = 0; j < ncol; ++j) {
      if (!lift_all_dirty && !lift_dirty[j]) continue;
      if (std::abs(col_cost[j]) < 1e-15) {
        lift_score[j] = 0.0;
        lift_dirty[j] = false;
        continue;
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
      lift_dirty[j] = false;
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

  // --- Main loop ---
  for (HighsInt step = 0; step < kMaxSteps; ++step) {
    if (step % kTermCheckInterval == 0 &&
        (mipdata->terminatorTerminated() ||
         mipsolver.timer_.read() >= std::min(mipsolver.options_mip_->time_limit,
                                             deadline)))
      break;

    bool feasible_mode = violated.empty();

    if (feasible_mode) {
      // Verify feasibility from scratch (prevent FP drift false positives)
      bool truly_feasible = true;
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
        best_feasible = true;
        best_objective = obj;
        best_solution = solution;
        steps_since_improvement = 0;
      }

      // Lift move: find variable giving best feasible objective improvement
      recompute_lift_cache();
      Candidate lift_best;
      lift_best.score = 0.0;  // must strictly improve
      for (HighsInt j = 0; j < ncol; ++j) {
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
