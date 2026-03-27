#include "fpr_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

FprConfig build_default_fpr_config(const HighsMipSolver &mipsolver,
                                   const CscMatrix &csc,
                                   std::vector<double> &scores_buf,
                                   std::vector<double> &cont_fallback_buf) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const auto &integrality = model->integrality_;
  const auto &col_cost = model->col_cost_;
  const HighsInt ncol = model->num_col_;

  // Ranking: degree * (1 + |cost|)
  scores_buf.resize(ncol);
  for (HighsInt j = 0; j < ncol; ++j) {
    if (!is_integer(integrality, j)) {
      scores_buf[j] = -1.0;
    } else {
      double degree =
          static_cast<double>(csc.col_start[j + 1] - csc.col_start[j]);
      scores_buf[j] = degree * (1.0 + std::abs(col_cost[j]));
    }
  }

  cont_fallback_buf.assign(ncol, 0.0);

  const double *hint =
      mipdata->incumbent.empty() ? nullptr : mipdata->incumbent.data();

  FprConfig cfg{};
  cfg.max_effort = 0; // caller must set budget
  cfg.rng_seed_offset = 42;
  cfg.hint = hint;
  cfg.scores = scores_buf.data();
  cfg.cont_fallback = cont_fallback_buf.data();
  cfg.csc = &csc;
  return cfg;
}

size_t fpr_core(HighsMipSolver &mipsolver, const FprConfig &cfg) {
  auto *mipdata = mipsolver.mipdata_.get();
  std::mt19937 rng(cfg.rng_seed_offset);
  size_t cumulative_effort = 0;

  for (int attempt = 0;; ++attempt) {
    if (mipdata->terminatorTerminated()) {
      return cumulative_effort;
    }
    if (cumulative_effort >= cfg.max_effort) {
      return cumulative_effort;
    }
    auto result = fpr_attempt(mipsolver, cfg, rng, attempt, nullptr);
    cumulative_effort += result.effort;
    if (result.found_feasible) {
      if (mipdata->trySolution(result.solution, kSolutionSourceFPR)) {
        return cumulative_effort;
      }
    }
  }
}

HeuristicResult fpr_attempt(HighsMipSolver &mipsolver, const FprConfig &cfg,
                            std::mt19937 &rng, int attempt_idx,
                            const double *initial_solution) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const auto &ARstart = mipdata->ARstart_;
  const auto &ARindex = mipdata->ARindex_;
  const auto &ARvalue = mipdata->ARvalue_;
  const auto &col_lb = model->col_lower_;
  const auto &col_ub = model->col_upper_;
  const auto &col_cost = model->col_cost_;
  const auto &row_lo = model->row_lower_;
  const auto &row_hi = model->row_upper_;
  const auto &integrality = model->integrality_;
  const double feastol = mipdata->feastol;
  const bool minimize = (model->sense_ == ObjSense::kMinimize);

  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) {
    return {};
  }

  // Use caller's CSC if provided, otherwise build our own
  CscMatrix owned_csc;
  if (!cfg.csc) {
    owned_csc = build_csc(ncol, nrow, ARstart, ARindex, ARvalue);
  }
  const auto &csc_ref = cfg.csc ? *cfg.csc : owned_csc;
  const auto &col_start = csc_ref.col_start;
  const auto &col_row = csc_ref.col_row;
  const auto &col_val = csc_ref.col_val;

  auto is_int = [&](HighsInt j) { return is_integer(integrality, j); };

  auto finite_clamp = [](double val, double lo, double hi) -> double {
    if (lo > -kHighsInf && hi < kHighsInf) {
      return std::max(lo, std::min(hi, val));
    }
    if (lo > -kHighsInf) {
      return std::max(lo, std::min(lo + 1e6, val));
    }
    if (hi < kHighsInf) {
      return std::min(hi, std::max(hi - 1e6, val));
    }
    return std::max(-1e6, std::min(1e6, val));
  };

  // Phase 1: Rank variables using caller-provided scores
  std::vector<HighsInt> var_order(ncol);
  for (HighsInt j = 0; j < ncol; ++j) {
    var_order[j] = j;
  }
  std::sort(var_order.begin(), var_order.end(), [&](HighsInt a, HighsInt b) {
    return cfg.scores[a] > cfg.scores[b];
  });

  struct VarState {
    double lb, ub, val;
    bool fixed;
  };

  std::vector<double> solution(ncol);
  std::vector<double> lhs_cache(nrow);

  const HighsInt repair_budget = std::max(HighsInt{1000}, 20 * ncol);
  constexpr double greedy_prob = 0.7;

  auto viol = [&](HighsInt i, double lhs) -> double {
    return row_violation(lhs, row_lo[i], row_hi[i]);
  };

  auto is_violated = [&](HighsInt i, double lhs) -> bool {
    if (lhs > row_hi[i] + feastol) {
      return true;
    }
    if (lhs < row_lo[i] - feastol) {
      return true;
    }
    return false;
  };

  size_t total_prop_work = 0;

  std::vector<HighsInt> prop_worklist;
  prop_worklist.reserve(nrow);
  std::vector<char> prop_in_wl(nrow);

  std::vector<HighsInt> violated;
  std::vector<HighsInt> violated_pos(nrow, -1);
  violated.reserve(nrow);

  std::vector<VarState> vs(ncol);
  std::vector<std::pair<HighsInt, VarState>> vs_undo;
  std::vector<std::pair<HighsInt, double>> sol_undo;
  vs_undo.reserve(ncol);
  sol_undo.reserve(ncol);

  // --- Initialize solution ---
  if (initial_solution) {
    for (HighsInt j = 0; j < ncol; ++j) {
      double v = initial_solution[j];
      if (is_int(j)) {
        v = std::round(v);
      }
      solution[j] = std::max(col_lb[j], std::min(col_ub[j], v));
    }
  } else if (attempt_idx == 0 && cfg.hint) {
    for (HighsInt j = 0; j < ncol; ++j) {
      double v = cfg.hint[j];
      if (is_int(j)) {
        v = std::round(v);
      }
      solution[j] = std::max(col_lb[j], std::min(col_ub[j], v));
    }
  } else if (attempt_idx == 0) {
    for (HighsInt j = 0; j < ncol; ++j) {
      if (mipdata->domain.isBinary(j)) {
        solution[j] = 0.0;
      } else if (is_int(j)) {
        double lo = std::max(col_lb[j], -1e8);
        double hi = std::min(col_ub[j], lo + 100.0);
        solution[j] = std::round((lo + hi) * 0.5);
        solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
      } else {
        solution[j] = finite_clamp(0.0, col_lb[j], col_ub[j]);
      }
    }
  } else {
    for (HighsInt j = 0; j < ncol; ++j) {
      if (mipdata->domain.isBinary(j)) {
        solution[j] = std::uniform_int_distribution<int>(0, 1)(rng);
      } else if (is_int(j)) {
        double lo = std::max(col_lb[j], -1e8);
        double hi = std::min(col_ub[j], lo + 100.0);
        solution[j] =
            std::round(std::uniform_real_distribution<double>(lo, hi)(rng));
        solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
      } else {
        double lo = finite_clamp(0.0, col_lb[j], col_ub[j]);
        double hi = std::min(col_ub[j], lo + 1e6);
        if (hi < kHighsInf && lo > -kHighsInf && hi > lo) {
          solution[j] = std::uniform_real_distribution<double>(lo, hi)(rng);
        } else {
          solution[j] = lo;
        }
      }
    }
  }

  // Shuffle top 30% of ranking for diversity on later attempts
  if (attempt_idx > 0) {
    HighsInt shuffle_len = std::max(HighsInt{1}, ncol * 3 / 10);
    std::shuffle(var_order.begin(), var_order.begin() + shuffle_len, rng);
  }

  // --- Phase 2: Fix & Propagate ---
  vs_undo.clear();
  sol_undo.clear();
  for (HighsInt j = 0; j < ncol; ++j) {
    vs[j].lb = col_lb[j];
    vs[j].ub = col_ub[j];
    vs[j].val = 0.0;
    vs[j].fixed = false;
  }

  // choose_fix_value: hint-aware on attempt 0 only, objective-greedy fallback
  const bool use_hint = (attempt_idx == 0 && cfg.hint != nullptr);
  auto choose_fix_value = [&](HighsInt j) -> double {
    double lo = vs[j].lb;
    double hi = vs[j].ub;

    if (use_hint) {
      double h = cfg.hint[j];
      if (is_int(j)) {
        h = std::round(h);
      }
      if (h >= lo - feastol && h <= hi + feastol) {
        return std::max(lo, std::min(hi, h));
      }
    }

    if (mipdata->domain.isBinary(j)) {
      if (minimize) {
        return (col_cost[j] >= 0) ? lo : hi;
      } else {
        return (col_cost[j] >= 0) ? hi : lo;
      }
    }

    if (std::abs(col_cost[j]) < 1e-15) {
      double mid = std::round((lo + hi) * 0.5);
      return std::max(lo, std::min(hi, mid));
    }
    if (minimize) {
      return (col_cost[j] > 0) ? lo : hi;
    } else {
      return (col_cost[j] > 0) ? hi : lo;
    }
  };

  auto fix_variable = [&](HighsInt j, double value) -> bool {
    if (value < vs[j].lb - feastol || value > vs[j].ub + feastol) {
      return false;
    }
    value = std::max(vs[j].lb, std::min(vs[j].ub, value));
    if (is_int(j)) {
      value = std::round(value);
    }
    vs_undo.push_back({j, vs[j]});
    sol_undo.push_back({j, solution[j]});
    vs[j].fixed = true;
    vs[j].val = value;
    solution[j] = value;
    return true;
  };

  auto propagate = [&](HighsInt fixed_var) -> bool {
    prop_worklist.clear();
    // Seed only the rows containing the just-fixed variable (AC-3).
    // Same fixpoint as seeding all rows, since unaffected rows cannot tighten.
    for (HighsInt p = col_start[fixed_var]; p < col_start[fixed_var + 1]; ++p) {
      HighsInt i = col_row[p];
      if (!prop_in_wl[i]) {
        prop_in_wl[i] = 1;
        prop_worklist.push_back(i);
      }
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

    HighsInt prop_iters = 0;
    const HighsInt prop_budget = 10 * nrow;
    while (!prop_worklist.empty()) {
      if (++prop_iters > prop_budget) {
        total_prop_work += prop_iters;
        return false;
      }
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

      if (num_unfixed == 0) {
        continue;
      }

      bool has_upper = row_hi[i] < kHighsInf;
      bool has_lower = row_lo[i] > -kHighsInf;

      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        HighsInt j = ARindex[k];
        if (vs[j].fixed) {
          continue;
        }
        double a = ARvalue[k];
        if (std::abs(a) < 1e-15) {
          continue;
        }

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

        if (has_upper) {
          double bound = row_hi[i] - fixed_sum - min_others;
          if (a > 0) {
            new_ub = std::min(new_ub, bound / a);
          } else {
            new_lb = std::max(new_lb, bound / a);
          }
        }

        if (has_lower) {
          double bound = row_lo[i] - fixed_sum - max_others;
          if (a > 0) {
            new_lb = std::max(new_lb, bound / a);
          } else {
            new_ub = std::min(new_ub, bound / a);
          }
        }

        if (is_int(j)) {
          new_lb = std::ceil(new_lb - feastol);
          new_ub = std::floor(new_ub + feastol);
        }

        new_lb = std::max(new_lb, col_lb[j]);
        new_ub = std::min(new_ub, col_ub[j]);

        if (new_lb > new_ub + feastol) {
          return false;
        }

        bool changed = false;
        if (new_lb > old_lb + feastol || new_ub < old_ub - feastol) {
          vs_undo.push_back({j, vs[j]});
          sol_undo.push_back({j, solution[j]});
          if (new_lb > old_lb + feastol) {
            vs[j].lb = new_lb;
            changed = true;
          }
          if (new_ub < old_ub - feastol) {
            vs[j].ub = new_ub;
            changed = true;
          }
        }

        if (!vs[j].fixed && vs[j].ub - vs[j].lb < feastol) {
          if (!changed) {
            vs_undo.push_back({j, vs[j]});
            sol_undo.push_back({j, solution[j]});
          }
          double val = (vs[j].lb + vs[j].ub) * 0.5;
          if (is_int(j)) {
            val = std::round(val);
          }
          vs[j].fixed = true;
          vs[j].val = val;
          solution[j] = val;
          changed = true;
        }

        if (changed) {
          enqueue_neighbors(j);
        }
      }
    }
    total_prop_work += prop_iters;
    return true;
  };

  // Fix integer variables in ranked order
  bool fix_failed = false;
  for (HighsInt idx = 0; idx < ncol; ++idx) {
    HighsInt j = var_order[idx];
    if (!is_int(j)) {
      continue;
    }
    if (vs[j].fixed) {
      continue;
    }

    double value = choose_fix_value(j);

    if (!fix_variable(j, value)) {
      double alt = (value == vs[j].lb) ? vs[j].ub : vs[j].lb;
      if (is_int(j)) {
        alt = std::round(alt);
      }
      if (!fix_variable(j, alt)) {
        fix_failed = true;
        break;
      }
    }

    HighsInt vs_mark = static_cast<HighsInt>(vs_undo.size());
    HighsInt sol_mark = static_cast<HighsInt>(sol_undo.size());

    if (!propagate(j)) {
      // Replay undo logs in reverse to restore state
      for (HighsInt u = static_cast<HighsInt>(vs_undo.size()) - 1; u >= vs_mark;
           --u) {
        vs[vs_undo[u].first] = vs_undo[u].second;
      }
      vs_undo.resize(vs_mark);
      for (HighsInt u = static_cast<HighsInt>(sol_undo.size()) - 1;
           u >= sol_mark; --u) {
        solution[sol_undo[u].first] = sol_undo[u].second;
      }
      sol_undo.resize(sol_mark);
      vs[j].fixed = false;

      double alt;
      if (mipdata->domain.isBinary(j)) {
        alt = (value < 0.5) ? 1.0 : 0.0;
      } else {
        alt = (value == vs[j].lb) ? vs[j].ub : vs[j].lb;
        if (is_int(j)) {
          alt = std::round(alt);
        }
      }

      if (!fix_variable(j, alt) || !propagate(j)) {
        fix_failed = true;
        break;
      }
    }
  }

  if (fix_failed) {
    return HeuristicResult::failed(total_prop_work);
  }

  // Fix remaining unfixed variables
  for (HighsInt j = 0; j < ncol; ++j) {
    if (vs[j].fixed) {
      continue;
    }
    double lo = vs[j].lb;
    double hi = vs[j].ub;

    if (!is_int(j)) {
      if (std::abs(col_cost[j]) > 1e-15) {
        bool want_low = (minimize == (col_cost[j] > 0));
        solution[j] = finite_clamp(want_low ? lo : hi, lo, hi);
      } else {
        solution[j] = finite_clamp(cfg.cont_fallback[j], lo, hi);
      }
    } else {
      solution[j] = choose_fix_value(j);
      solution[j] = std::max(lo, std::min(hi, solution[j]));
      if (is_int(j)) {
        solution[j] = std::round(solution[j]);
      }
    }
    solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
  }

  // --- Compute LHS cache ---
  total_prop_work += ARindex.size();
  for (HighsInt i = 0; i < nrow; ++i) {
    double lhs = 0.0;
    for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
      lhs += ARvalue[k] * solution[ARindex[k]];
    }
    lhs_cache[i] = lhs;
  }

  bool feasible = true;
  for (HighsInt i = 0; i < nrow; ++i) {
    if (is_violated(i, lhs_cache[i])) {
      feasible = false;
      break;
    }
  }

  // --- Phase 3: WalkSAT Repair ---
  if (!feasible) {
    for (auto i : violated) {
      violated_pos[i] = -1;
    }
    violated.clear();

    auto add_violated = [&](HighsInt i) {
      if (violated_pos[i] != -1) {
        return;
      }
      violated_pos[i] = static_cast<HighsInt>(violated.size());
      violated.push_back(i);
    };
    auto remove_violated = [&](HighsInt i) {
      HighsInt p = violated_pos[i];
      if (p == -1) {
        return;
      }
      HighsInt last = violated.back();
      violated[p] = last;
      violated_pos[last] = p;
      violated.pop_back();
      violated_pos[i] = -1;
    };

    for (HighsInt i = 0; i < nrow; ++i) {
      if (is_violated(i, lhs_cache[i])) {
        add_violated(i);
      }
    }

    for (HighsInt step = 0; step < repair_budget && !violated.empty(); ++step) {
      if (total_prop_work >= cfg.max_effort) {
        break;
      }
      HighsInt pick = std::uniform_int_distribution<HighsInt>(
          0, static_cast<HighsInt>(violated.size()) - 1)(rng);
      HighsInt i = violated[pick];

      HighsInt row_len = ARstart[i + 1] - ARstart[i];
      if (row_len == 0) {
        continue;
      }

      double ci_lhs = lhs_cache[i];

      double target_rhs;
      if (ci_lhs > row_hi[i] + feastol) {
        target_rhs = row_hi[i];
      } else {
        target_rhs = row_lo[i];
      }

      HighsInt best_var = -1;
      double best_delta_viol = std::numeric_limits<double>::infinity();
      double best_new_val = 0.0;

      // Evaluate all candidate variables in violated row
      total_prop_work += row_len; // scanning row entries
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        HighsInt j = ARindex[k];
        double a = ARvalue[k];
        if (std::abs(a) < 1e-15) {
          continue;
        }

        double old_val = solution[j];
        double target_delta = (target_rhs - ci_lhs) / a;
        double new_val = old_val + target_delta;

        if (is_int(j)) {
          if (ci_lhs > row_hi[i] + feastol) {
            new_val = (a > 0) ? std::floor(new_val + feastol)
                              : std::ceil(new_val - feastol);
          } else {
            new_val = (a > 0) ? std::ceil(new_val - feastol)
                              : std::floor(new_val + feastol);
          }
        }
        new_val = std::max(col_lb[j], std::min(col_ub[j], new_val));
        if (std::abs(new_val - old_val) < 1e-15) {
          continue;
        }

        // Evaluate delta_viol by scanning column
        HighsInt col_deg = col_start[j + 1] - col_start[j];
        total_prop_work += col_deg;
        double delta_change = new_val - old_val;
        double delta_viol = 0.0;
        for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
          HighsInt i2 = col_row[p];
          double coeff = col_val[p];
          double old_lhs = lhs_cache[i2];
          double new_lhs = old_lhs + coeff * delta_change;
          delta_viol += viol(i2, new_lhs) - viol(i2, old_lhs);
        }

        if (delta_viol < best_delta_viol) {
          best_delta_viol = delta_viol;
          best_var = j;
          best_new_val = new_val;
        }
      }

      if (best_var == -1) {
        continue;
      }

      HighsInt changed_var = -1;
      double delta_change = 0.0;

      double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
      if (roll > greedy_prob && row_len > 1) {
        HighsInt k = ARstart[i] + std::uniform_int_distribution<HighsInt>(
                                      0, row_len - 1)(rng);
        HighsInt j = ARindex[k];
        double a = ARvalue[k];
        if (std::abs(a) < 1e-15) {
          continue;
        }
        double old_val = solution[j];
        double new_val = old_val + (target_rhs - ci_lhs) / a;
        if (is_int(j)) {
          new_val = std::round(new_val);
        }
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

      if (changed_var == -1 || std::abs(delta_change) < 1e-15) {
        continue;
      }

      // Apply move: update LHS cache for all rows of changed variable
      total_prop_work += col_start[changed_var + 1] - col_start[changed_var];
      for (HighsInt p = col_start[changed_var]; p < col_start[changed_var + 1];
           ++p) {
        HighsInt i2 = col_row[p];
        lhs_cache[i2] += col_val[p] * delta_change;
        bool was = violated_pos[i2] != -1;
        bool now = is_violated(i2, lhs_cache[i2]);
        if (was && !now) {
          remove_violated(i2);
        } else if (!was && now) {
          add_violated(i2);
        }
      }
    }

    feasible = violated.empty();
  }

  if (!feasible) {
    return HeuristicResult::failed(total_prop_work);
  }

  // Verify feasibility using cached LHS values (O(nrow) vs O(nnz))
  for (HighsInt i = 0; i < nrow; ++i) {
    if (is_violated(i, lhs_cache[i])) {
      return HeuristicResult::failed(total_prop_work);
    }
  }

  double obj = model->offset_;
  for (HighsInt j = 0; j < ncol; ++j) {
    obj += col_cost[j] * solution[j];
  }

  HeuristicResult result;
  result.found_feasible = true;
  result.solution = std::move(solution);
  result.objective = obj;
  result.effort = total_prop_work;
  return result;
}
