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

  // Paper: artificial bounding box [-100000, +100000] for infinite bounds
  auto finite_clamp = [](double val, double lo, double hi) -> double {
    constexpr double kBox = 1e5;
    if (lo > -kHighsInf && hi < kHighsInf) {
      return std::max(lo, std::min(hi, val));
    }
    if (lo > -kHighsInf) {
      return std::max(lo, std::min(lo + kBox, val));
    }
    if (hi < kHighsInf) {
      return std::min(hi, std::max(hi - kBox, val));
    }
    return std::max(-kBox, std::min(kBox, val));
  };

  // Phase 1: Rank variables
  std::vector<HighsInt> var_order;
  if (cfg.precomputed_var_order != nullptr) {
    // Use pre-computed order (avoids data races on cliquePartition)
    var_order.assign(cfg.precomputed_var_order,
                     cfg.precomputed_var_order + cfg.precomputed_var_order_size);
  } else if (cfg.strategy) {
    var_order =
        compute_var_order(mipsolver, cfg.strategy->var_strategy, rng, cfg.lp_ref);
  } else {
    // Legacy: sort by caller-provided scores
    var_order.resize(ncol);
    for (HighsInt j = 0; j < ncol; ++j) {
      var_order[j] = j;
    }
    std::sort(var_order.begin(), var_order.end(), [&](HighsInt a, HighsInt b) {
      return cfg.scores[a] > cfg.scores[b];
    });
  }

  std::vector<double> solution(ncol);
  std::vector<double> lhs_cache(nrow);

  const HighsInt repair_budget = cfg.repair_iterations;

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

  // choose_fix_value: strategy-aware or legacy hint+objective-greedy fallback
  const bool use_hint = (attempt_idx == 0 && cfg.hint != nullptr);
  auto choose_fix_value = [&](HighsInt j) -> double {
    // Strategy-based value selection (paper Table 2)
    if (cfg.strategy) {
      return choose_value(j, vs[j].lb, vs[j].ub, is_int(j), minimize,
                          col_cost[j], cfg.strategy->val_strategy, rng,
                          cfg.lp_ref, &mipsolver, vs.data(), &csc_ref);
    }

    // Legacy behavior
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
      }
      return (col_cost[j] >= 0) ? hi : lo;
    }

    if (std::abs(col_cost[j]) < 1e-15) {
      double mid = std::round((lo + hi) * 0.5);
      return std::max(lo, std::min(hi, mid));
    }
    if (minimize) {
      return (col_cost[j] > 0) ? lo : hi;
    }
    return (col_cost[j] > 0) ? hi : lo;
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

    size_t prop_work = 0;
    const size_t prop_budget = 10 * static_cast<size_t>(ARindex.size());
    while (!prop_worklist.empty()) {
      HighsInt i = prop_worklist.back();
      prop_worklist.pop_back();
      prop_in_wl[i] = 0;
      prop_work += ARstart[i + 1] - ARstart[i];
      if (prop_work > prop_budget) {
        total_prop_work += prop_work;
        return false;
      }

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
    total_prop_work += prop_work;
    return true;
  };

  // --- Phase 2: DFS Fix & Propagate (paper Fig. 1) ---

  // Backtrack undo stacks to given marks
  auto backtrack_to = [&](HighsInt vs_mark, HighsInt sol_mark) {
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
  };

  // Find first unfixed integer variable in var_order
  auto find_next_unfixed_int = [&]() -> HighsInt {
    for (HighsInt idx = 0; idx < ncol; ++idx) {
      HighsInt j = var_order[idx];
      if (is_int(j) && !vs[j].fixed) return j;
    }
    return -1;
  };

  // Compute alternative value for branching
  auto compute_alt = [&](HighsInt j, double preferred) -> double {
    if (mipdata->domain.isBinary(j)) {
      return (preferred < 0.5) ? 1.0 : 0.0;
    }
    double alt = (std::abs(preferred - vs[j].lb) < feastol) ? vs[j].ub
                                                             : vs[j].lb;
    if (is_int(j)) alt = std::round(alt);
    return alt;
  };

  struct DfsNode {
    HighsInt var;
    double val;
    HighsInt vs_mark;
    HighsInt sol_mark;
  };

  const bool do_propagate = mode_propagates(cfg.mode);
  const bool do_backtrack = mode_backtracks(cfg.mode);
  const HighsInt node_limit = ncol + 1;

  std::vector<DfsNode> dfs_stack;
  dfs_stack.reserve(do_backtrack ? 2 * static_cast<size_t>(ncol) : ncol);
  HighsInt nodes_visited = 0;
  bool found_complete = false;

  // Seed the DFS with the first unfixed integer variable
  HighsInt first_var = find_next_unfixed_int();
  if (first_var < 0) {
    // All integer variables already fixed (e.g., by propagation)
    found_complete = true;
  } else {
    double pref = choose_fix_value(first_var);
    double alt = compute_alt(first_var, pref);
    HighsInt vs_mark = static_cast<HighsInt>(vs_undo.size());
    HighsInt sol_mark = static_cast<HighsInt>(sol_undo.size());

    if (do_backtrack) {
      dfs_stack.push_back({first_var, alt, vs_mark, sol_mark});
    }
    dfs_stack.push_back({first_var, pref, vs_mark, sol_mark});
  }

  while (!dfs_stack.empty() && nodes_visited < node_limit && !found_complete) {
    auto node = dfs_stack.back();
    dfs_stack.pop_back();
    ++nodes_visited;

    // Backtrack to parent state
    backtrack_to(node.vs_mark, node.sol_mark);

    // Apply the branching fixing
    if (!fix_variable(node.var, node.val)) {
      continue;  // can't fix, try next node (sibling)
    }

    // Propagate
    if (do_propagate) {
      if (!propagate(node.var)) {
        continue;  // infeasible, try next node (sibling)
      }
    }

    // Find next unfixed integer variable
    HighsInt next_var = find_next_unfixed_int();

    if (next_var < 0) {
      // All integer variables fixed
      found_complete = true;
      break;
    }

    // Branch on next variable: push children to DFS stack
    double pref = choose_fix_value(next_var);
    double alt = compute_alt(next_var, pref);
    HighsInt vs_mark = static_cast<HighsInt>(vs_undo.size());
    HighsInt sol_mark = static_cast<HighsInt>(sol_undo.size());

    if (do_backtrack) {
      dfs_stack.push_back({next_var, alt, vs_mark, sol_mark});
    }
    dfs_stack.push_back({next_var, pref, vs_mark, sol_mark});
  }

  if (!found_complete) {
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
        double fallback = cfg.cont_fallback ? cfg.cont_fallback[j] : 0.0;
        solution[j] = finite_clamp(fallback, lo, hi);
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

  // --- Phase 3: WalkSAT Repair (modes: dfsrep, dive, diveprop) ---
  if (!feasible && mode_repairs(cfg.mode)) {
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

    // Best-state tracking (paper Fig. 4, lines 23-26)
    double best_total_viol = 0.0;
    for (HighsInt i = 0; i < nrow; ++i) {
      best_total_viol += viol(i, lhs_cache[i]);
    }
    std::vector<double> best_solution;
    std::vector<double> best_lhs;
    if (cfg.repair_track_best) {
      best_solution = solution;
      best_lhs = lhs_cache;
    }

    // Candidate struct for WalkSAT selection
    struct Candidate {
      HighsInt var;
      double new_val;
      double damage;  // violation increase only (ignoring improvements)
    };
    std::vector<Candidate> cand;

    for (HighsInt step = 0; step < repair_budget && !violated.empty(); ++step) {
      if (total_prop_work >= cfg.max_effort) {
        break;
      }

      // Pick a violated constraint uniformly at random (Fig. 4, line 4)
      HighsInt pick = std::uniform_int_distribution<HighsInt>(
          0, static_cast<HighsInt>(violated.size()) - 1)(rng);
      HighsInt i = violated[pick];

      HighsInt row_len = ARstart[i + 1] - ARstart[i];
      if (row_len == 0) {
        continue;
      }

      double ci_lhs = lhs_cache[i];
      double ci_viol = viol(i, ci_lhs);

      // Compute best shift for each variable in the constraint (Fig. 4, line 8)
      // Filter: only candidates with s_j != 0 whose shift reduces this
      // constraint's violation (Fig. 4, lines 9-11)
      cand.clear();
      double best_damage = std::numeric_limits<double>::infinity();

      total_prop_work += row_len;
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        HighsInt j = ARindex[k];
        double a = ARvalue[k];
        if (std::abs(a) < 1e-15) {
          continue;
        }

        // ComputeShift: best shift to reduce violation of constraint i
        double target_rhs;
        if (ci_lhs > row_hi[i] + feastol) {
          target_rhs = row_hi[i];
        } else {
          target_rhs = row_lo[i];
        }
        double old_val = solution[j];
        double new_val = old_val + (target_rhs - ci_lhs) / a;

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

        // s_j == 0 check
        if (std::abs(new_val - old_val) < 1e-15) {
          continue;
        }

        // Check if shift reduces violation of the picked constraint
        double delta_change = new_val - old_val;
        double new_ci_lhs = ci_lhs + a * delta_change;
        double new_ci_viol = viol(i, new_ci_lhs);
        if (new_ci_viol >= ci_viol - feastol) {
          continue;  // shift doesn't reduce this constraint's violation
        }

        // Evaluate damage: violation increase across OTHER constraints
        // (paper: only count increases, ignore improvements)
        HighsInt col_deg = col_start[j + 1] - col_start[j];
        total_prop_work += col_deg;
        double damage = 0.0;
        for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
          HighsInt i2 = col_row[p];
          if (i2 == i) continue;  // skip the picked constraint itself
          double coeff = col_val[p];
          double old_lhs = lhs_cache[i2];
          double new_lhs = old_lhs + coeff * delta_change;
          double dv = viol(i2, new_lhs) - viol(i2, old_lhs);
          if (dv > 0) {
            damage += dv;  // only count increases
          }
        }

        best_damage = std::min(best_damage, damage);
        cand.push_back({j, new_val, damage});
      }

      if (cand.empty()) {
        continue;
      }

      // WalkSAT selection (paper Fig. 4, lines 17-21)
      HighsInt chosen_var;
      double chosen_val;
      if (best_damage > feastol &&
          std::uniform_real_distribution<double>(0.0, 1.0)(rng) <
              cfg.repair_noise) {
        // Random walk: pick random from all candidates
        HighsInt idx = std::uniform_int_distribution<HighsInt>(
            0, static_cast<HighsInt>(cand.size()) - 1)(rng);
        chosen_var = cand[idx].var;
        chosen_val = cand[idx].new_val;
      } else {
        // Greedy: filter to candidates with damage == best_damage,
        // pick random among them
        std::vector<HighsInt> best_indices;
        for (HighsInt ci = 0; ci < static_cast<HighsInt>(cand.size()); ++ci) {
          if (cand[ci].damage <= best_damage + feastol) {
            best_indices.push_back(ci);
          }
        }
        HighsInt idx = best_indices[std::uniform_int_distribution<HighsInt>(
            0, static_cast<HighsInt>(best_indices.size()) - 1)(rng)];
        chosen_var = cand[idx].var;
        chosen_val = cand[idx].new_val;
      }

      double old_val = solution[chosen_var];
      double delta_change = chosen_val - old_val;
      solution[chosen_var] = chosen_val;

      // Apply move: update LHS cache for all rows of changed variable
      total_prop_work += col_start[chosen_var + 1] - col_start[chosen_var];
      for (HighsInt p = col_start[chosen_var];
           p < col_start[chosen_var + 1]; ++p) {
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

      // Update best-state tracking (paper Fig. 4, lines 23-26)
      if (cfg.repair_track_best) {
        double curr_total_viol = 0.0;
        for (HighsInt ii = 0; ii < nrow; ++ii) {
          curr_total_viol += viol(ii, lhs_cache[ii]);
        }
        if (curr_total_viol < best_total_viol) {
          best_total_viol = curr_total_viol;
          best_solution = solution;
          best_lhs = lhs_cache;
        }
      }
    }

    // Restore best state if tracking enabled (paper Fig. 4, line 27)
    if (cfg.repair_track_best && !violated.empty()) {
      solution = best_solution;
      lhs_cache = best_lhs;
      // Rebuild violated set from best state
      for (auto vi : violated) {
        violated_pos[vi] = -1;
      }
      violated.clear();
      for (HighsInt i = 0; i < nrow; ++i) {
        if (is_violated(i, lhs_cache[i])) {
          add_violated(i);
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

  // Greedy 1-opt: try shifting each integer variable toward better objective
  // (paper Section 6: "before adding it to the solution pool, we apply a
  //  simple greedy 1-opt step to try to improve its objective")
  for (HighsInt j = 0; j < ncol; ++j) {
    if (!is_int(j)) continue;
    if (std::abs(col_cost[j]) < 1e-15) continue;

    // Determine improvement direction
    double direction;
    if (minimize) {
      direction = (col_cost[j] > 0) ? -1.0 : 1.0;
    } else {
      direction = (col_cost[j] > 0) ? 1.0 : -1.0;
    }
    double new_val = solution[j] + direction;
    new_val = std::max(col_lb[j], std::min(col_ub[j], new_val));
    if (std::abs(new_val - solution[j]) < 1e-15) continue;

    // Check feasibility of the shift across all affected constraints
    double delta = new_val - solution[j];
    bool shift_feasible = true;
    total_prop_work += col_start[j + 1] - col_start[j];
    for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
      HighsInt row = col_row[p];
      double new_lhs = lhs_cache[row] + col_val[p] * delta;
      if (is_violated(row, new_lhs)) {
        shift_feasible = false;
        break;
      }
    }

    if (shift_feasible) {
      // Apply the improving shift
      for (HighsInt p = col_start[j]; p < col_start[j + 1]; ++p) {
        lhs_cache[col_row[p]] += col_val[p] * delta;
      }
      solution[j] = new_val;
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
