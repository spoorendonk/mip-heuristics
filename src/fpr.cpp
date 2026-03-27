#include "fpr.h"

#include <vector>

#include "fpr_core.h"
#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

namespace fpr {

void run(HighsMipSolver &mipsolver) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  if (ncol == 0) {
    return;
  }

  auto csc = build_csc(ncol, model->num_row_, mipdata->ARstart_,
                       mipdata->ARindex_, mipdata->ARvalue_);

  const size_t nnz = mipdata->ARindex_.size();
  std::vector<double> scores, cont_fallback;
  auto cfg = build_default_fpr_config(mipsolver, csc, scores, cont_fallback);
  cfg.max_effort = nnz << 10;

  fpr_core(mipsolver, cfg);
}

} // namespace fpr
