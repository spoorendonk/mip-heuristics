#include "repair_search.h"

#include <random>
#include <vector>

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "prop_engine.h"
#include "walksat.h"

namespace {

// SyncChanges(E, R): transfer domain deductions from R to E.
// For each variable where R narrowed the domain beyond E's:
//   - Binary fixed in R: fix in E
//   - Non-binary singleton in R: fix in E
//   - Otherwise: tighten E's bounds to intersection
// Returns false if any transfer causes infeasibility in E.
bool sync_changes(PropEngine& E, const PropEngine& R) {
  bool any_seeded = false;
  for (HighsInt j = 0; j < E.ncol(); ++j) {
    if (E.var(j).fixed) continue;

    const auto& rv = R.var(j);
    const auto& ev = E.var(j);

    // Check if R narrowed this variable's domain
    bool r_tightened_lb = rv.lb > ev.lb + E.feastol();
    bool r_tightened_ub = rv.ub < ev.ub - E.feastol();

    if (!r_tightened_lb && !r_tightened_ub) continue;

    // If R made this a singleton, fix in E
    if (rv.fixed || rv.ub - rv.lb < E.feastol()) {
      double val = rv.fixed ? rv.val : (rv.lb + rv.ub) * 0.5;
      if (!E.fix(j, val)) return false;
      E.seed_worklist(j);
      any_seeded = true;
    } else {
      if (r_tightened_lb) {
        if (!E.tighten_lb(j, rv.lb)) return false;
        any_seeded = true;
      }
      if (r_tightened_ub) {
        if (!E.tighten_ub(j, rv.ub)) return false;
        any_seeded = true;
      }
    }
  }
  // Propagate the synchronized deductions through E
  if (any_seeded) {
    if (!E.propagate(-1)) return false;
  }
  return true;
}

}  // namespace

bool repair_search(PropEngine& E, std::vector<double>& solution,
                   std::vector<double>& lhs_cache, const double* col_lb,
                   const double* col_ub, const double* row_lo,
                   const double* row_hi, HighsInt repair_iterations,
                   double repair_noise, bool repair_track_best,
                   size_t max_effort, std::mt19937& rng,
                   size_t& effort_out) {
  const HighsInt ncol = E.ncol();
  const HighsInt nrow = E.nrow();
  const double feastol = E.feastol();

  auto viol = [&](HighsInt i, double lhs) -> double {
    return row_violation(lhs, row_lo[i], row_hi[i]);
  };

  auto is_violated = [&](HighsInt i, double lhs) -> bool {
    return lhs > row_hi[i] + feastol || lhs < row_lo[i] - feastol;
  };

  // Build violated set
  std::vector<HighsInt> violated;
  std::vector<HighsInt> violated_pos(nrow, -1);
  violated.reserve(nrow);

  auto add_violated = [&](HighsInt i) {
    if (violated_pos[i] != -1) return;
    violated_pos[i] = static_cast<HighsInt>(violated.size());
    violated.push_back(i);
  };
  auto remove_violated = [&](HighsInt i) {
    HighsInt pos = violated_pos[i];
    if (pos == -1) return;
    HighsInt last = violated.back();
    violated[pos] = last;
    violated_pos[last] = pos;
    violated.pop_back();
    violated_pos[i] = -1;
  };

  for (HighsInt i = 0; i < nrow; ++i) {
    if (is_violated(i, lhs_cache[i])) {
      add_violated(i);
    }
  }

  if (violated.empty()) {
    effort_out = 0;
    return true;
  }

  // Best-state tracking
  double best_total_viol = 0.0;
  for (HighsInt i = 0; i < nrow; ++i) {
    best_total_viol += viol(i, lhs_cache[i]);
  }
  std::vector<double> best_solution;
  std::vector<double> best_lhs;
  if (repair_track_best) {
    best_solution = solution;
    best_lhs = lhs_cache;
  }

  // Create secondary engine R from global bounds (shares E's matrix pointers)
  PropEngine R(ncol, nrow, E.ar_start(), E.ar_index(), E.ar_value(),
               E.csc_start(), E.csc_row(), E.csc_val(), col_lb, col_ub, row_lo,
               row_hi, E.integrality(), feastol);

  size_t total_effort = 0;
  size_t prev_r_effort = 0;
  size_t e_effort_baseline = E.effort();

  for (HighsInt step = 0; step < repair_iterations && !violated.empty();
       ++step) {
    if (total_effort >= max_effort) break;

    // Pick a violated constraint uniformly at random
    HighsInt pick = std::uniform_int_distribution<HighsInt>(
        0, static_cast<HighsInt>(violated.size()) - 1)(rng);
    HighsInt row = violated[pick];

    // WalkSAT move selection
    auto move = walksat_select_move(row, solution.data(), lhs_cache.data(),
                                    col_lb, col_ub, E, repair_noise, rng,
                                    total_effort);
    if (move.var < 0) continue;

    HighsInt chosen_var = move.var;
    double chosen_val = move.val;

    // --- RepairSearch: propagate through R, then create disjunction ---

    // Reset R to global bounds and propagate the repair move
    R.reset();
    if (!R.fix(chosen_var, chosen_val) || !R.propagate(chosen_var)) {
      // R detected infeasibility — cannot derive deductions, but still
      // apply the move to the solution (fall through like RepairWalk)
    } else {
      // SyncChanges: transfer R's deductions to E
      HighsInt e_vs_mark = E.vs_mark();
      HighsInt e_sol_mark = E.sol_mark();
      bool sync_ok = sync_changes(E, R);

      if (!sync_ok) {
        // Sync caused infeasibility in E — backtrack and try alternative
        E.backtrack_to(e_vs_mark, e_sol_mark);

        // MoveToDisjunction: create alternative branch
        bool is_binary =
            col_lb[chosen_var] == 0.0 && col_ub[chosen_var] == 1.0 &&
            E.is_int(chosen_var);
        if (is_binary) {
          double alt_val = (chosen_val < 0.5) ? 1.0 : 0.0;
          if (E.fix(chosen_var, alt_val) && E.propagate(chosen_var)) {
            // Alternative branch succeeded in E — continue
          } else {
            E.backtrack_to(e_vs_mark, e_sol_mark);
          }
        }
        // For non-binary, just skip (the main branch failed)
      }
    }
    total_effort += R.effort() - prev_r_effort;
    prev_r_effort = R.effort();

    // Apply the move to solution and lhs_cache (regardless of E's state)
    double old_val = solution[chosen_var];
    double delta_change = chosen_val - old_val;
    solution[chosen_var] = chosen_val;

    total_effort += E.csc_start()[chosen_var + 1] - E.csc_start()[chosen_var];
    for (HighsInt p = E.csc_start()[chosen_var];
         p < E.csc_start()[chosen_var + 1]; ++p) {
      HighsInt i2 = E.csc_row()[p];
      lhs_cache[i2] += E.csc_val()[p] * delta_change;
      bool was = violated_pos[i2] != -1;
      bool now = is_violated(i2, lhs_cache[i2]);
      if (was && !now) {
        remove_violated(i2);
      } else if (!was && now) {
        add_violated(i2);
      }
    }

    // Best-state tracking
    if (repair_track_best) {
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

  // Restore best state if tracking enabled
  if (repair_track_best && !violated.empty()) {
    solution = best_solution;
    lhs_cache = best_lhs;
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

  effort_out = total_effort + (E.effort() - e_effort_baseline);
  return violated.empty();
}
