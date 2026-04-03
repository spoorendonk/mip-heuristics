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
#include "prop_engine.h"
#include "repair_search.h"
#include "walksat.h"

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

  std::vector<double> lhs_cache(nrow);

  const HighsInt repair_budget = cfg.repair_iterations;

  auto is_violated = [&](HighsInt i, double lhs) -> bool {
    return is_row_violated(lhs, row_lo[i], row_hi[i], feastol);
  };

  // --- Create PropEngine for Phase 2 ---
  PropEngine E(ncol, nrow, ARstart.data(), ARindex.data(), ARvalue.data(),
               csc_ref, col_lb.data(), col_ub.data(), row_lo.data(),
               row_hi.data(), integrality.data(), feastol);

  // --- Initialize solution in E ---
  if (initial_solution) {
    for (HighsInt j = 0; j < ncol; ++j) {
      double v = initial_solution[j];
      if (is_int(j)) {
        v = std::round(v);
      }
      E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], v));
    }
  } else if (attempt_idx == 0 && cfg.hint) {
    for (HighsInt j = 0; j < ncol; ++j) {
      double v = cfg.hint[j];
      if (is_int(j)) {
        v = std::round(v);
      }
      E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], v));
    }
  } else if (attempt_idx == 0) {
    for (HighsInt j = 0; j < ncol; ++j) {
      if (mipdata->domain.isBinary(j)) {
        E.sol(j) = 0.0;
      } else if (is_int(j)) {
        double lo = std::max(col_lb[j], -1e8);
        double hi = std::min(col_ub[j], lo + 100.0);
        E.sol(j) = std::round((lo + hi) * 0.5);
        E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], E.sol(j)));
      } else {
        E.sol(j) = finite_clamp(0.0, col_lb[j], col_ub[j]);
      }
    }
  } else {
    for (HighsInt j = 0; j < ncol; ++j) {
      if (mipdata->domain.isBinary(j)) {
        E.sol(j) = std::uniform_int_distribution<int>(0, 1)(rng);
      } else if (is_int(j)) {
        double lo = std::max(col_lb[j], -1e8);
        double hi = std::min(col_ub[j], lo + 100.0);
        E.sol(j) =
            std::round(std::uniform_real_distribution<double>(lo, hi)(rng));
        E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], E.sol(j)));
      } else {
        double lo = finite_clamp(0.0, col_lb[j], col_ub[j]);
        double hi = std::min(col_ub[j], lo + 1e6);
        if (hi < kHighsInf && lo > -kHighsInf && hi > lo) {
          E.sol(j) = std::uniform_real_distribution<double>(lo, hi)(rng);
        } else {
          E.sol(j) = lo;
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
  // (E already initialized with global bounds via constructor)

  // choose_fix_value: strategy-aware or legacy hint+objective-greedy fallback
  const bool use_hint = (attempt_idx == 0 && cfg.hint != nullptr);
  auto choose_fix_value = [&](HighsInt j) -> double {
    // Strategy-based value selection (paper Table 2)
    if (cfg.strategy) {
      return choose_value(j, E.var(j).lb, E.var(j).ub, is_int(j), minimize,
                          col_cost[j], cfg.strategy->val_strategy, rng,
                          cfg.lp_ref, &mipsolver, &E.var(0), &csc_ref);
    }

    // Legacy behavior
    double lo = E.var(j).lb;
    double hi = E.var(j).ub;

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

  // Paper Section 6: "fix all trivially-roundable variables (if any) to the
  // corresponding bound" before running strategies.
  if (!mipdata->uplocks.empty()) {
    const auto &uplocks = mipdata->uplocks;
    const auto &downlocks = mipdata->downlocks;
    for (HighsInt j = 0; j < ncol; ++j) {
      if (!is_int(j) || E.var(j).fixed) continue;
      if (uplocks[j] == 0 && downlocks[j] != 0) {
        E.fix(j, E.var(j).ub);
      } else if (downlocks[j] == 0 && uplocks[j] != 0) {
        E.fix(j, E.var(j).lb);
      }
    }
  }

  // Paper Section 6: "perform a first round of constraint propagation, until
  // a fixpoint is reached" before starting the DFS.
  for (HighsInt j = 0; j < ncol; ++j) {
    if (E.var(j).fixed) {
      E.seed_worklist(j);
    }
  }
  E.propagate(-1);

  // --- Phase 2: DFS Fix & Propagate (paper Fig. 1) ---

  // Find first unfixed integer variable in var_order starting from hint.
  // Returns {variable, index} or {-1, -1} if none found.
  const auto var_order_size = static_cast<HighsInt>(var_order.size());
  auto find_next_unfixed_int = [&](HighsInt hint) -> std::pair<HighsInt, HighsInt> {
    for (HighsInt idx = hint; idx < var_order_size; ++idx) {
      HighsInt j = var_order[idx];
      if (is_int(j) && !E.var(j).fixed) return {j, idx};
    }
    return {-1, -1};
  };

  // Compute alternative value for branching
  auto compute_alt = [&](HighsInt j, double preferred) -> double {
    if (mipdata->domain.isBinary(j)) {
      return (preferred < 0.5) ? 1.0 : 0.0;
    }
    double alt = (std::abs(preferred - E.var(j).lb) < feastol) ? E.var(j).ub
                                                                : E.var(j).lb;
    if (is_int(j)) alt = std::round(alt);
    return alt;
  };

  struct DfsNode {
    HighsInt var;
    double val;
    HighsInt vs_mark;
    HighsInt sol_mark;
    HighsInt scan_start;  // hint: resume var_order scan from this index
  };

  const bool do_propagate = mode_propagates(cfg.mode);
  const bool do_backtrack = mode_backtracks(cfg.mode);
  const HighsInt node_limit = ncol + 1;

  std::vector<DfsNode> dfs_stack;
  dfs_stack.reserve(do_backtrack ? 2 * static_cast<size_t>(ncol) : ncol);
  HighsInt nodes_visited = 0;
  bool found_complete = false;

  // Seed the DFS with the first unfixed integer variable
  auto [first_var, first_idx] = find_next_unfixed_int(0);
  if (first_var < 0) {
    // All integer variables already fixed (e.g., by propagation)
    found_complete = true;
  } else {
    double pref = choose_fix_value(first_var);
    double alt = compute_alt(first_var, pref);
    HighsInt vs_m = E.vs_mark();
    HighsInt sol_m = E.sol_mark();
    HighsInt next_scan = first_idx + 1;

    if (do_backtrack) {
      dfs_stack.push_back({first_var, alt, vs_m, sol_m, next_scan});
    }
    dfs_stack.push_back({first_var, pref, vs_m, sol_m, next_scan});
  }

  while (!dfs_stack.empty() && nodes_visited < node_limit && !found_complete) {
    auto node = dfs_stack.back();
    dfs_stack.pop_back();
    ++nodes_visited;

    // Backtrack to parent state
    E.backtrack_to(node.vs_mark, node.sol_mark);

    // Apply the branching fixing
    if (!E.fix(node.var, node.val)) {
      continue;  // can't fix, try next node (sibling)
    }

    // Propagate
    if (do_propagate) {
      if (!E.propagate(node.var)) {
        continue;  // infeasible, try next node (sibling)
      }
    }

    // Find next unfixed integer variable (scan from hint, not from 0)
    auto [next_var, next_idx] = find_next_unfixed_int(node.scan_start);

    if (next_var < 0) {
      // All integer variables fixed
      found_complete = true;
      break;
    }

    // Branch on next variable: push children to DFS stack
    double pref = choose_fix_value(next_var);
    double alt = compute_alt(next_var, pref);
    HighsInt vs_m = E.vs_mark();
    HighsInt sol_m = E.sol_mark();
    HighsInt next_scan = next_idx + 1;

    if (do_backtrack) {
      dfs_stack.push_back({next_var, alt, vs_m, sol_m, next_scan});
    }
    dfs_stack.push_back({next_var, pref, vs_m, sol_m, next_scan});
  }

  if (!found_complete) {
    return HeuristicResult::failed(E.effort());
  }

  // Fix remaining unfixed variables (continuous and residual integers)
  for (HighsInt j = 0; j < ncol; ++j) {
    if (E.var(j).fixed) {
      continue;
    }
    double lo = E.var(j).lb;
    double hi = E.var(j).ub;

    if (!is_int(j)) {
      if (std::abs(col_cost[j]) > 1e-15) {
        bool want_low = (minimize == (col_cost[j] > 0));
        E.sol(j) = finite_clamp(want_low ? lo : hi, lo, hi);
      } else {
        double fallback = cfg.cont_fallback ? cfg.cont_fallback[j] : 0.0;
        E.sol(j) = finite_clamp(fallback, lo, hi);
      }
    } else {
      E.sol(j) = choose_fix_value(j);
      E.sol(j) = std::round(std::max(lo, std::min(hi, E.sol(j))));
    }
    E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], E.sol(j)));
  }

  // --- Copy solution out of E for Phase 3 and result ---
  std::vector<double> solution(E.sol_data(), E.sol_data() + ncol);
  size_t total_prop_work = E.effort();

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

  // --- Phase 3: RepairSearch (Fig. 5) or WalkSAT Repair ---
  if (!feasible && cfg.mode == FrameworkMode::kRepairSearch) {
    size_t rs_effort = 0;
    feasible = repair_search(
        E, solution, lhs_cache, col_lb.data(), col_ub.data(), row_lo.data(),
        row_hi.data(), cfg.repair_iterations, cfg.repair_noise,
        cfg.repair_track_best,
        cfg.max_effort > total_prop_work ? cfg.max_effort - total_prop_work : 0,
        rng, rs_effort);
    total_prop_work += rs_effort;
  } else if (!feasible && mode_repairs(cfg.mode)) {
    size_t walk_effort = 0;
    feasible = walksat_repair(
        E, solution, lhs_cache, col_lb.data(), col_ub.data(), repair_budget,
        cfg.repair_noise, cfg.repair_track_best,
        cfg.max_effort > total_prop_work ? cfg.max_effort - total_prop_work : 0,
        rng, walk_effort);
    total_prop_work += walk_effort;
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

  // Greedy 1-opt (paper Section 6)
  greedy_1opt(E, solution, lhs_cache, col_cost.data(), minimize,
              total_prop_work);

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
