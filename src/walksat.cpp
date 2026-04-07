#include "walksat.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "heuristic_common.h"
#include "prop_engine.h"

WalkSatMove walksat_select_move(HighsInt row, const double* solution,
                                const double* lhs_cache, const double* col_lb,
                                const double* col_ub, const PropEngine& data,
                                double noise, std::mt19937& rng, size_t& effort,
                                WalkSatScratch& scratch) {
  const HighsInt* ar_start = data.ar_start();
  const HighsInt* ar_index = data.ar_index();
  const double* ar_value = data.ar_value();
  const HighsInt* csc_start = data.csc_start();
  const HighsInt* csc_row = data.csc_row();
  const double* csc_val = data.csc_val();
  const double* row_lo = data.row_lo();
  const double* row_hi = data.row_hi();
  const double feastol = data.feastol();

  auto viol = [&](HighsInt i, double lhs) -> double {
    return row_violation(lhs, row_lo[i], row_hi[i]);
  };

  HighsInt row_len = ar_start[row + 1] - ar_start[row];
  if (row_len == 0) return {};

  double ci_lhs = lhs_cache[row];
  double ci_viol = viol(row, ci_lhs);

  auto& cand = scratch.cand;
  cand.clear();
  double best_damage = std::numeric_limits<double>::infinity();

  effort += row_len;
  for (HighsInt k = ar_start[row]; k < ar_start[row + 1]; ++k) {
    HighsInt j = ar_index[k];
    double a = ar_value[k];
    if (std::abs(a) < 1e-15) continue;

    double target_rhs;
    if (ci_lhs > row_hi[row] + feastol) {
      target_rhs = row_hi[row];
    } else {
      target_rhs = row_lo[row];
    }
    double old_val = solution[j];
    double new_val = old_val + (target_rhs - ci_lhs) / a;

    if (data.is_int(j)) {
      if (ci_lhs > row_hi[row] + feastol) {
        new_val =
            (a > 0) ? std::floor(new_val + feastol) : std::ceil(new_val - feastol);
      } else {
        new_val =
            (a > 0) ? std::ceil(new_val - feastol) : std::floor(new_val + feastol);
      }
    }
    new_val = std::max(col_lb[j], std::min(col_ub[j], new_val));

    if (std::abs(new_val - old_val) < 1e-15) continue;

    double delta_change = new_val - old_val;
    double new_ci_lhs = ci_lhs + a * delta_change;
    double new_ci_viol = viol(row, new_ci_lhs);
    if (new_ci_viol >= ci_viol - feastol) continue;

    HighsInt col_deg = csc_start[j + 1] - csc_start[j];
    effort += col_deg;
    double damage = 0.0;
    for (HighsInt p = csc_start[j]; p < csc_start[j + 1]; ++p) {
      HighsInt i2 = csc_row[p];
      if (i2 == row) continue;
      double coeff = csc_val[p];
      double old_lhs = lhs_cache[i2];
      double new_lhs = old_lhs + coeff * delta_change;
      double dv = viol(i2, new_lhs) - viol(i2, old_lhs);
      if (dv > 0) damage += dv;
    }

    best_damage = std::min(best_damage, damage);
    cand.push_back({j, new_val, damage});
  }

  if (cand.empty()) return {};

  // WalkSAT selection (paper Fig. 4, lines 17-21)
  if (best_damage > feastol &&
      std::uniform_real_distribution<double>(0.0, 1.0)(rng) < noise) {
    HighsInt idx = std::uniform_int_distribution<HighsInt>(
        0, static_cast<HighsInt>(cand.size()) - 1)(rng);
    return {cand[idx].var, cand[idx].new_val};
  }
  auto& best_indices = scratch.best_indices;
  best_indices.clear();
  for (HighsInt ci = 0; ci < static_cast<HighsInt>(cand.size()); ++ci) {
    if (cand[ci].damage <= best_damage + feastol) {
      best_indices.push_back(ci);
    }
  }
  HighsInt idx = best_indices[std::uniform_int_distribution<HighsInt>(
      0, static_cast<HighsInt>(best_indices.size()) - 1)(rng)];
  return {cand[idx].var, cand[idx].new_val};
}

// ---------------------------------------------------------------------------
// WalkSAT repair walk (paper Fig. 4)
// ---------------------------------------------------------------------------

bool walksat_repair(const PropEngine& data, std::vector<double>& solution,
                    std::vector<double>& lhs_cache, const double* col_lb,
                    const double* col_ub, HighsInt max_iterations, double noise,
                    bool track_best, size_t max_effort, std::mt19937& rng,
                    size_t& effort) {
  const HighsInt nrow = data.nrow();
  const double feastol = data.feastol();
  const double* row_lo = data.row_lo();
  const double* row_hi = data.row_hi();
  const HighsInt* csc_start = data.csc_start();
  const HighsInt* csc_row = data.csc_row();
  const double* csc_val = data.csc_val();

  // Violated set
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

  // Initialize violated set and total violation
  double total_viol = 0.0;
  for (HighsInt i = 0; i < nrow; ++i) {
    double v = row_violation(lhs_cache[i], row_lo[i], row_hi[i]);
    total_viol += v;
    if (is_row_violated(lhs_cache[i], row_lo[i], row_hi[i], feastol)) {
      add_violated(i);
    }
  }

  if (violated.empty()) return true;

  // Delta-based best-state tracking: instead of copying full solution/lhs_cache
  // vectors on each improvement (O(ncol + nrow)), we keep an undo log of
  // changed entries and record a mark at each improvement. Restore replays
  // only the entries after the best mark — O(changes_since_best).
  struct UndoEntry {
    HighsInt idx;
    double old_val;
  };
  std::vector<UndoEntry> sol_undo;
  std::vector<UndoEntry> lhs_undo;
  double best_viol = total_viol;
  HighsInt best_sol_mark = 0;
  HighsInt best_lhs_mark = 0;

  WalkSatScratch scratch;

  for (HighsInt step = 0; step < max_iterations && !violated.empty(); ++step) {
    if (effort >= max_effort) break;

    HighsInt pick = std::uniform_int_distribution<HighsInt>(
        0, static_cast<HighsInt>(violated.size()) - 1)(rng);
    HighsInt row = violated[pick];

    auto move = walksat_select_move(row, solution.data(), lhs_cache.data(),
                                    col_lb, col_ub, data, noise, rng, effort,
                                    scratch);
    if (move.var < 0) continue;

    double old_val = solution[move.var];
    double delta = move.val - old_val;
    if (track_best) {
      sol_undo.push_back({move.var, old_val});
    }
    solution[move.var] = move.val;

    // Apply move: update lhs_cache, violated set, and total_viol incrementally
    effort += csc_start[move.var + 1] - csc_start[move.var];
    for (HighsInt p = csc_start[move.var]; p < csc_start[move.var + 1]; ++p) {
      HighsInt i2 = csc_row[p];
      double old_v = row_violation(lhs_cache[i2], row_lo[i2], row_hi[i2]);
      if (track_best) {
        lhs_undo.push_back({i2, lhs_cache[i2]});
      }
      lhs_cache[i2] += csc_val[p] * delta;
      double new_v = row_violation(lhs_cache[i2], row_lo[i2], row_hi[i2]);
      total_viol += new_v - old_v;

      bool was = violated_pos[i2] != -1;
      bool now = is_row_violated(lhs_cache[i2], row_lo[i2], row_hi[i2], feastol);
      if (was && !now) remove_violated(i2);
      else if (!was && now) add_violated(i2);
    }

    // Update best-state tracking (paper Fig. 4, lines 23-26)
    if (track_best && total_viol < best_viol) {
      best_viol = total_viol;
      best_sol_mark = static_cast<HighsInt>(sol_undo.size());
      best_lhs_mark = static_cast<HighsInt>(lhs_undo.size());
    }
  }

  // Restore best state if tracking enabled (paper Fig. 4, line 27)
  if (track_best && !violated.empty()) {
    // Undo changes back to the best mark
    for (HighsInt u = static_cast<HighsInt>(sol_undo.size()) - 1;
         u >= best_sol_mark; --u) {
      solution[sol_undo[u].idx] = sol_undo[u].old_val;
    }
    for (HighsInt u = static_cast<HighsInt>(lhs_undo.size()) - 1;
         u >= best_lhs_mark; --u) {
      lhs_cache[lhs_undo[u].idx] = lhs_undo[u].old_val;
    }
    // Rebuild violated set from restored state
    for (auto vi : violated) violated_pos[vi] = -1;
    violated.clear();
    for (HighsInt i = 0; i < nrow; ++i) {
      if (is_row_violated(lhs_cache[i], row_lo[i], row_hi[i], feastol)) {
        add_violated(i);
      }
    }
  }

  return violated.empty();
}

// ---------------------------------------------------------------------------
// Greedy 1-opt improvement (paper Section 6)
// ---------------------------------------------------------------------------

void greedy_1opt(const PropEngine& data, std::vector<double>& solution,
                 std::vector<double>& lhs_cache, const double* col_cost,
                 bool minimize, size_t& effort) {
  const HighsInt ncol = data.ncol();
  const HighsInt* csc_start = data.csc_start();
  const HighsInt* csc_row = data.csc_row();
  const double* csc_val = data.csc_val();
  const double* col_lb = data.col_lb();
  const double* col_ub = data.col_ub();
  const double* row_lo = data.row_lo();
  const double* row_hi = data.row_hi();
  const double feastol = data.feastol();

  for (HighsInt j = 0; j < ncol; ++j) {
    if (!data.is_int(j)) continue;
    if (std::abs(col_cost[j]) < 1e-15) continue;

    double direction;
    if (minimize) {
      direction = (col_cost[j] > 0) ? -1.0 : 1.0;
    } else {
      direction = (col_cost[j] > 0) ? 1.0 : -1.0;
    }
    double new_val = solution[j] + direction;
    new_val = std::max(col_lb[j], std::min(col_ub[j], new_val));
    if (std::abs(new_val - solution[j]) < 1e-15) continue;

    double delta = new_val - solution[j];
    bool shift_feasible = true;
    effort += csc_start[j + 1] - csc_start[j];
    for (HighsInt p = csc_start[j]; p < csc_start[j + 1]; ++p) {
      HighsInt row = csc_row[p];
      double new_lhs = lhs_cache[row] + csc_val[p] * delta;
      if (is_row_violated(new_lhs, row_lo[row], row_hi[row], feastol)) {
        shift_feasible = false;
        break;
      }
    }

    if (shift_feasible) {
      for (HighsInt p = csc_start[j]; p < csc_start[j + 1]; ++p) {
        lhs_cache[csc_row[p]] += csc_val[p] * delta;
      }
      solution[j] = new_val;
    }
  }
}
