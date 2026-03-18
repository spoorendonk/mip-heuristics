#include "fpr.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "fpr_core.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

namespace fpr {

void run(HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const auto& integrality = model->integrality_;
  const auto& col_cost = model->col_cost_;
  const HighsInt ncol = model->num_col_;
  if (ncol == 0) return;

  // Ranking: degree * (1 + |cost|)
  auto csc = build_csc(ncol, model->num_row_, mipdata->ARstart_,
                       mipdata->ARindex_, mipdata->ARvalue_);
  std::vector<double> scores(ncol);
  for (HighsInt j = 0; j < ncol; ++j) {
    if (!is_integer(integrality, j)) {
      scores[j] = -1.0;
    } else {
      double degree =
          static_cast<double>(csc.col_start[j + 1] - csc.col_start[j]);
      scores[j] = degree * (1.0 + std::abs(col_cost[j]));
    }
  }

  // Hint: incumbent solution (if available)
  const double* hint =
      mipdata->incumbent.empty() ? nullptr : mipdata->incumbent.data();

  // Continuous fallback: 0.0 for zero-cost vars
  std::vector<double> cont_fallback(ncol, 0.0);

  FprConfig cfg{};
  cfg.max_attempts = 10;
  cfg.rng_seed_offset = 42;
  cfg.hint = hint;
  cfg.scores = scores.data();
  cfg.cont_fallback = cont_fallback.data();
  cfg.csc = &csc;
  // Cap FPR at 10% of time limit (min 5s, max 30s)
  const double tl = mipsolver.options_mip_->time_limit;
  cfg.deadline = mipsolver.timer_.read() +
                 std::min(30.0, std::max(5.0, 0.1 * tl));

  fpr_core(mipsolver, cfg);
}

}  // namespace fpr
