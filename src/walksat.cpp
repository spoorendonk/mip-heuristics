#include "walksat.h"

#include <cmath>
#include <limits>
#include <vector>

#include "heuristic_common.h"
#include "prop_engine.h"

WalkSatMove walksat_select_move(HighsInt row, const double* solution,
                                const double* lhs_cache, const double* col_lb,
                                const double* col_ub, const PropEngine& data,
                                double noise, std::mt19937& rng,
                                size_t& effort) {
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

  struct Candidate {
    HighsInt var;
    double new_val;
    double damage;
  };
  std::vector<Candidate> cand;
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
    // Random walk
    HighsInt idx = std::uniform_int_distribution<HighsInt>(
        0, static_cast<HighsInt>(cand.size()) - 1)(rng);
    return {cand[idx].var, cand[idx].new_val};
  }
  // Greedy: pick random among best-damage candidates
  std::vector<HighsInt> best_indices;
  for (HighsInt ci = 0; ci < static_cast<HighsInt>(cand.size()); ++ci) {
    if (cand[ci].damage <= best_damage + feastol) {
      best_indices.push_back(ci);
    }
  }
  HighsInt idx = best_indices[std::uniform_int_distribution<HighsInt>(
      0, static_cast<HighsInt>(best_indices.size()) - 1)(rng)];
  return {cand[idx].var, cand[idx].new_val};
}
