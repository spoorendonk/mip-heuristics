#include "scylla_fpr.h"

#include <cassert>
#include <cmath>
#include <vector>

#include "fpr_core.h"
#include "heuristic_common.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

namespace scylla_fpr {

void run(HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();

  // Guard: need an optimal LP relaxation
  auto lp_status = mipdata->lp.getStatus();
  if (!HighsLpRelaxation::scaledOptimal(lp_status)) return;

  const auto& lp_sol = mipdata->lp.getLpSolver().getSolution().col_value;
  assert(static_cast<HighsInt>(lp_sol.size()) >= model->num_col_);

  const auto& integrality = model->integrality_;
  const HighsInt ncol = model->num_col_;
  if (ncol == 0) return;
  if (mipdata->terminatorTerminated()) return;

  // Ranking: LP fractionality (most fractional first)
  std::vector<double> scores(ncol);
  for (HighsInt j = 0; j < ncol; ++j) {
    if (!is_integer(integrality, j))
      scores[j] = -1.0;
    else
      scores[j] = std::abs(lp_sol[j] - std::round(lp_sol[j]));
  }

  // Hint: LP solution
  const double* hint = lp_sol.data();

  // Continuous fallback: LP solution for zero-cost vars
  // (lp_sol already has length >= ncol, so we use it directly)
  const double* cont_fallback = lp_sol.data();

  FprConfig cfg{};
  cfg.max_attempts = 1;
  cfg.rng_seed_offset = 137;
  cfg.hint = hint;
  cfg.scores = scores.data();
  cfg.cont_fallback = cont_fallback;
  cfg.csc = nullptr;

  fpr_core(mipsolver, cfg);
}

}  // namespace scylla_fpr
