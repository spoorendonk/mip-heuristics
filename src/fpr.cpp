#include "fpr.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

namespace fpr {

void run(HighsMipSolver& mipsolver) {
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
  if (ncol == 0 || nrow == 0) return;
  const HighsInt nnz = static_cast<HighsInt>(ARindex.size());

  // Build column view (CSC) from row-major AR arrays
  std::vector<HighsInt> col_start(ncol + 1, 0);
  for (HighsInt k = 0; k < nnz; ++k) col_start[ARindex[k] + 1]++;
  for (HighsInt j = 0; j < ncol; ++j) col_start[j + 1] += col_start[j];
  std::vector<HighsInt> col_row(nnz);
  std::vector<double> col_val(nnz);
  {
    std::vector<HighsInt> pos(col_start);
    for (HighsInt i = 0; i < nrow; ++i)
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        HighsInt j = ARindex[k];
        col_row[pos[j]] = i;
        col_val[pos[j]] = ARvalue[k];
        pos[j]++;
      }
  }

  auto is_integer = [&](HighsInt j) {
    return integrality[j] != HighsVarType::kContinuous;
  };

  // Clamp a value into finite bounds, falling back to 0 or the finite end
  auto finite_clamp = [](double val, double lo, double hi) -> double {
    if (lo > -kHighsInf && hi < kHighsInf)
      return std::max(lo, std::min(hi, val));
    if (lo > -kHighsInf) return std::max(lo, std::min(lo + 1e6, val));
    if (hi < kHighsInf) return std::min(hi, std::max(hi - 1e6, val));
    return std::max(-1e6, std::min(1e6, val));
  };

  // Phase 1: Rank variables — score = degree * (1 + |cost|)
  std::vector<HighsInt> var_order(ncol);
  std::vector<double> scores(ncol);
  for (HighsInt j = 0; j < ncol; ++j) {
    var_order[j] = j;
    if (!is_integer(j)) {
      scores[j] = -1.0;  // continuous vars sort to end
    } else {
      double degree = static_cast<double>(col_start[j + 1] - col_start[j]);
      scores[j] = degree * (1.0 + std::abs(col_cost[j]));
    }
  }
  std::sort(var_order.begin(), var_order.end(),
            [&](HighsInt a, HighsInt b) { return scores[a] > scores[b]; });

  // Per-variable state for fix & propagate
  struct VarState {
    double lb, ub, val;
    bool fixed;
  };

  std::vector<double> solution(ncol);
  std::vector<double> lhs_cache(nrow);

  const HighsInt repair_budget = std::max(HighsInt{1000}, 20 * ncol);
  constexpr int max_attempts = 10;
  constexpr double greedy_prob = 0.7;

  std::mt19937 rng(mipdata->numImprovingSols + 42);

  // Raw violation for delta evaluation (no tolerance dead zone)
  auto raw_violation = [&](HighsInt i, double lhs) -> double {
    double v = 0.0;
    if (row_hi[i] < kHighsInf) v += std::max(0.0, lhs - row_hi[i]);
    if (row_lo[i] > -kHighsInf) v += std::max(0.0, row_lo[i] - lhs);
    return v;
  };

  // Boolean feasibility check (with tolerance)
  auto is_violated = [&](HighsInt i, double lhs) -> bool {
    if (lhs > row_hi[i] + feastol) return true;
    if (lhs < row_lo[i] - feastol) return true;
    return false;
  };

  // Pre-allocate propagation worklist and flag arrays (reused across calls)
  std::vector<HighsInt> prop_worklist;
  prop_worklist.reserve(nrow);
  std::vector<char> prop_in_wl(nrow);

  // Pre-allocate var state and snapshot vectors (reused across attempts)
  std::vector<VarState> vs(ncol);
  std::vector<VarState> saved_vs(ncol);
  std::vector<double> saved_sol(ncol);

  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    if (mipdata->terminatorTerminated()) return;

    // --- Initialize solution ---
    if (attempt == 0 && !mipdata->incumbent.empty()) {
      // Use incumbent as hint
      for (HighsInt j = 0; j < ncol; ++j) {
        double v = mipdata->incumbent[j];
        if (is_integer(j)) v = std::round(v);
        solution[j] = std::max(col_lb[j], std::min(col_ub[j], v));
      }
    } else if (attempt == 0) {
      // Default: objective-greedy for binaries, midpoint for others
      for (HighsInt j = 0; j < ncol; ++j) {
        if (mipdata->domain.isBinary(j)) {
          solution[j] = 0.0;
        } else if (is_integer(j)) {
          double lo = std::max(col_lb[j], -1e8);
          double hi = std::min(col_ub[j], lo + 100.0);
          solution[j] = std::round((lo + hi) * 0.5);
          solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
        } else {
          solution[j] = finite_clamp(0.0, col_lb[j], col_ub[j]);
        }
      }
    } else {
      // Random init
      for (HighsInt j = 0; j < ncol; ++j) {
        if (mipdata->domain.isBinary(j)) {
          solution[j] = std::uniform_int_distribution<int>(0, 1)(rng);
        } else if (is_integer(j)) {
          double lo = std::max(col_lb[j], -1e8);
          double hi = std::min(col_ub[j], lo + 100.0);
          solution[j] =
              std::round(std::uniform_real_distribution<double>(lo, hi)(rng));
          solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
        } else {
          double lo = finite_clamp(0.0, col_lb[j], col_ub[j]);
          double hi = std::min(col_ub[j], lo + 1e6);
          if (hi < kHighsInf && lo > -kHighsInf && hi > lo)
            solution[j] = std::uniform_real_distribution<double>(lo, hi)(rng);
          else
            solution[j] = lo;
        }
      }
    }

    bool use_hint = (attempt == 0 && !mipdata->incumbent.empty());

    // Shuffle top 30% of ranking for diversity on later attempts
    if (attempt > 0) {
      HighsInt shuffle_len = std::max(HighsInt{1}, ncol * 3 / 10);
      std::shuffle(var_order.begin(), var_order.begin() + shuffle_len, rng);
    }

    // --- Phase 2: Fix & Propagate ---
    for (HighsInt j = 0; j < ncol; ++j) {
      vs[j].lb = col_lb[j];
      vs[j].ub = col_ub[j];
      vs[j].val = 0.0;
      vs[j].fixed = false;
    }

    // choose_fix_value: hint-aware, objective-greedy
    auto choose_fix_value = [&](HighsInt j) -> double {
      double lo = vs[j].lb;
      double hi = vs[j].ub;

      if (use_hint) {
        double hint = solution[j];
        if (is_integer(j)) hint = std::round(hint);
        if (hint >= lo - feastol && hint <= hi + feastol)
          return std::max(lo, std::min(hi, hint));
      }

      // Binary: objective-greedy
      if (mipdata->domain.isBinary(j)) {
        if (minimize)
          return (col_cost[j] >= 0) ? lo : hi;
        else
          return (col_cost[j] >= 0) ? hi : lo;
      }

      // Integer: objective-greedy or midpoint
      if (std::abs(col_cost[j]) < 1e-15) {
        double mid = std::round((lo + hi) * 0.5);
        return std::max(lo, std::min(hi, mid));
      }
      if (minimize)
        return (col_cost[j] > 0) ? lo : hi;
      else
        return (col_cost[j] > 0) ? hi : lo;
    };

    // fix_variable
    auto fix_variable = [&](HighsInt j, double value) -> bool {
      if (value < vs[j].lb - feastol || value > vs[j].ub + feastol)
        return false;
      value = std::max(vs[j].lb, std::min(vs[j].ub, value));
      if (is_integer(j)) value = std::round(value);
      vs[j].fixed = true;
      vs[j].val = value;
      solution[j] = value;
      return true;
    };

    // propagate: worklist-based bound tightening
    auto propagate = [&]() -> bool {
      prop_worklist.clear();
      for (HighsInt i = 0; i < nrow; ++i) {
        prop_worklist.push_back(i);
        prop_in_wl[i] = 1;
      }

      auto enqueue_neighbors = [&](HighsInt j) {
        for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
          HighsInt i = col_row[p];
          if (!prop_in_wl[i]) {
            prop_in_wl[i] = 1;
            prop_worklist.push_back(i);
          }
        }
      };

      while (!prop_worklist.empty()) {
        HighsInt i = prop_worklist.back();
        prop_worklist.pop_back();
        prop_in_wl[i] = 0;

        double fixed_sum = 0.0;
        double min_act = 0.0, max_act = 0.0;
        HighsInt num_unfixed = 0;

        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
          HighsInt j = ARindex[k];
          double a = ARvalue[k];
          if (vs[j].fixed) {
            fixed_sum += a * vs[j].val;
          } else {
            ++num_unfixed;
            if (a > 0) {
              min_act += a * vs[j].lb;
              max_act += a * vs[j].ub;
            } else {
              min_act += a * vs[j].ub;
              max_act += a * vs[j].lb;
            }
          }
        }

        if (num_unfixed == 0) continue;

        bool has_upper = row_hi[i] < kHighsInf;
        bool has_lower = row_lo[i] > -kHighsInf;

        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
          HighsInt j = ARindex[k];
          if (vs[j].fixed) continue;
          double a = ARvalue[k];
          if (std::abs(a) < 1e-15) continue;

          double old_lb = vs[j].lb;
          double old_ub = vs[j].ub;

          double min_others, max_others;
          if (a > 0) {
            min_others = min_act - a * old_lb;
            max_others = max_act - a * old_ub;
          } else {
            min_others = min_act - a * old_ub;
            max_others = max_act - a * old_lb;
          }

          double new_lb = old_lb;
          double new_ub = old_ub;

          // Tighten from row upper bound: fixed_sum + a*x + others <= row_hi
          if (has_upper) {
            double bound = row_hi[i] - fixed_sum - min_others;
            if (a > 0)
              new_ub = std::min(new_ub, bound / a);
            else
              new_lb = std::max(new_lb, bound / a);
          }

          // Tighten from row lower bound: fixed_sum + a*x + others >= row_lo
          if (has_lower) {
            double bound = row_lo[i] - fixed_sum - max_others;
            if (a > 0)
              new_lb = std::max(new_lb, bound / a);
            else
              new_ub = std::min(new_ub, bound / a);
          }

          if (is_integer(j)) {
            new_lb = std::ceil(new_lb - feastol);
            new_ub = std::floor(new_ub + feastol);
          }

          new_lb = std::max(new_lb, col_lb[j]);
          new_ub = std::min(new_ub, col_ub[j]);

          if (new_lb > new_ub + feastol) return false;  // wipeout

          bool changed = false;
          if (new_lb > old_lb + feastol) {
            vs[j].lb = new_lb;
            changed = true;
          }
          if (new_ub < old_ub - feastol) {
            vs[j].ub = new_ub;
            changed = true;
          }

          // Domain collapse — auto-fix
          if (!vs[j].fixed && vs[j].ub - vs[j].lb < feastol) {
            double val = (vs[j].lb + vs[j].ub) * 0.5;
            if (is_integer(j)) val = std::round(val);
            vs[j].fixed = true;
            vs[j].val = val;
            solution[j] = val;
            changed = true;
          }

          if (changed) enqueue_neighbors(j);
        }
      }
      return true;
    };

    // Fix integer variables in ranked order
    bool fix_failed = false;
    for (HighsInt idx = 0; idx < ncol; ++idx) {
      HighsInt j = var_order[idx];
      if (!is_integer(j)) continue;
      if (vs[j].fixed) continue;

      double value = choose_fix_value(j);

      if (!fix_variable(j, value)) {
        // Try opposite bound
        double alt = (value == vs[j].lb) ? vs[j].ub : vs[j].lb;
        if (is_integer(j)) alt = std::round(alt);
        if (!fix_variable(j, alt)) {
          fix_failed = true;
          break;
        }
      }

      // Snapshot before propagation for backtracking
      saved_vs = vs;
      saved_sol = solution;

      if (!propagate()) {
        // Wipeout — restore and try opposite value
        vs = saved_vs;
        solution = saved_sol;
        vs[j].fixed = false;

        double alt;
        if (mipdata->domain.isBinary(j)) {
          alt = (value < 0.5) ? 1.0 : 0.0;
        } else {
          alt = (value == vs[j].lb) ? vs[j].ub : vs[j].lb;
          if (is_integer(j)) alt = std::round(alt);
        }

        if (!fix_variable(j, alt) || !propagate()) {
          fix_failed = true;
          break;
        }
      }
    }

    if (fix_failed) continue;

    // Fix remaining unfixed variables
    for (HighsInt j = 0; j < ncol; ++j) {
      if (vs[j].fixed) continue;
      double lo = vs[j].lb;
      double hi = vs[j].ub;

      if (!is_integer(j)) {
        // Continuous: objective-biased, with infinite-bound safety
        if (std::abs(col_cost[j]) > 1e-15) {
          double preferred = minimize ? (col_cost[j] > 0 ? lo : hi)
                                      : (col_cost[j] > 0 ? hi : lo);
          solution[j] = finite_clamp(preferred, lo, hi);
        } else {
          solution[j] = finite_clamp(0.0, lo, hi);
        }
      } else {
        solution[j] = choose_fix_value(j);
        solution[j] = std::max(lo, std::min(hi, solution[j]));
        if (is_integer(j)) solution[j] = std::round(solution[j]);
      }
      solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
    }

    // --- Compute LHS cache ---
    for (HighsInt i = 0; i < nrow; ++i) {
      double lhs = 0.0;
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k)
        lhs += ARvalue[k] * solution[ARindex[k]];
      lhs_cache[i] = lhs;
    }

    // Check feasibility
    bool feasible = true;
    for (HighsInt i = 0; i < nrow; ++i) {
      if (is_violated(i, lhs_cache[i])) {
        feasible = false;
        break;
      }
    }

    // --- Phase 3: WalkSAT Repair ---
    if (!feasible) {
      // Violated set with O(1) add/remove
      std::vector<HighsInt> violated;
      std::vector<HighsInt> violated_pos(nrow, -1);

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

      for (HighsInt i = 0; i < nrow; ++i)
        if (is_violated(i, lhs_cache[i])) add_violated(i);

      for (HighsInt step = 0; step < repair_budget && !violated.empty();
           ++step) {
        // Pick random violated row
        HighsInt pick = std::uniform_int_distribution<HighsInt>(
            0, static_cast<HighsInt>(violated.size()) - 1)(rng);
        HighsInt i = violated[pick];

        HighsInt row_len = ARstart[i + 1] - ARstart[i];
        if (row_len == 0) continue;

        double ci_lhs = lhs_cache[i];

        // Determine binding direction: which bound is violated?
        double target_rhs;
        if (ci_lhs > row_hi[i] + feastol)
          target_rhs = row_hi[i];  // lhs too high
        else
          target_rhs = row_lo[i];  // lhs too low

        HighsInt best_var = -1;
        double best_delta_viol = std::numeric_limits<double>::infinity();
        double best_new_val = 0.0;

        // Evaluate each variable in this row
        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
          HighsInt j = ARindex[k];
          double a = ARvalue[k];
          if (std::abs(a) < 1e-15) continue;

          double old_val = solution[j];
          double target_delta = (target_rhs - ci_lhs) / a;
          double new_val = old_val + target_delta;

          // Round integers in helpful direction
          if (is_integer(j)) {
            if (ci_lhs > row_hi[i] + feastol) {
              // Need to decrease lhs: if a > 0 decrease x, if a < 0 increase x
              new_val = (a > 0) ? std::floor(new_val + feastol)
                                : std::ceil(new_val - feastol);
            } else {
              // Need to increase lhs
              new_val = (a > 0) ? std::ceil(new_val - feastol)
                                : std::floor(new_val + feastol);
            }
          }
          new_val = std::max(col_lb[j], std::min(col_ub[j], new_val));
          if (std::abs(new_val - old_val) < 1e-15) continue;

          // Evaluate total violation delta across all rows containing j
          // (raw violation without tolerance for smooth greedy ranking)
          double delta_change = new_val - old_val;
          double delta_viol = 0.0;
          for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
            HighsInt i2 = col_row[p];
            double coeff = col_val[p];
            double old_lhs = lhs_cache[i2];
            double new_lhs = old_lhs + coeff * delta_change;
            delta_viol +=
                raw_violation(i2, new_lhs) - raw_violation(i2, old_lhs);
          }

          if (delta_viol < best_delta_viol) {
            best_delta_viol = delta_viol;
            best_var = j;
            best_new_val = new_val;
          }
        }

        if (best_var == -1) continue;

        // Greedy (prob 0.7) or random move
        HighsInt changed_var = -1;
        double delta_change = 0.0;

        double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        if (roll > greedy_prob && row_len > 1) {
          // Random variable from this row
          HighsInt k = ARstart[i] + std::uniform_int_distribution<HighsInt>(
                                        0, row_len - 1)(rng);
          HighsInt j = ARindex[k];
          double a = ARvalue[k];
          if (std::abs(a) < 1e-15) continue;
          double old_val = solution[j];
          double new_val = old_val + (target_rhs - ci_lhs) / a;
          if (is_integer(j)) new_val = std::round(new_val);
          new_val = std::max(col_lb[j], std::min(col_ub[j], new_val));
          if (std::abs(new_val - old_val) > 1e-15) {
            solution[j] = new_val;
            changed_var = j;
            delta_change = new_val - old_val;
          }
        } else {
          double old_val = solution[best_var];
          solution[best_var] = best_new_val;
          changed_var = best_var;
          delta_change = best_new_val - old_val;
        }

        if (changed_var == -1 || std::abs(delta_change) < 1e-15) continue;

        // Update LHS cache and violated set for all rows containing changed_var
        for (HighsInt p = col_start[changed_var];
             p < col_start[changed_var + 1]; ++p) {
          HighsInt i2 = col_row[p];
          lhs_cache[i2] += col_val[p] * delta_change;
          bool was = violated_pos[i2] != -1;
          bool now = is_violated(i2, lhs_cache[i2]);
          if (was && !now)
            remove_violated(i2);
          else if (!was && now)
            add_violated(i2);
        }
      }

      feasible = violated.empty();
    }

    if (!feasible) continue;

    // Verify feasibility from scratch
    bool truly_feasible = true;
    for (HighsInt i = 0; i < nrow; ++i) {
      double lhs = 0.0;
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k)
        lhs += ARvalue[k] * solution[ARindex[k]];
      if (is_violated(i, lhs)) {
        truly_feasible = false;
        break;
      }
    }
    if (!truly_feasible) continue;

    // Submit solution to HiGHS
    if (mipdata->trySolution(solution, kSolutionSourceHeuristic)) return;
  }
}

}  // namespace fpr
