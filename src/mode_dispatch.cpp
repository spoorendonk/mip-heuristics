#include "mode_dispatch.h"

#include "fj.h"
#include "fpr.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "portfolio.h"
#include "scylla.h"

namespace heuristics {

bool run_presolve(HighsMipSolver &mipsolver, size_t budget) {
  const auto *options = mipsolver.options_mip_;

  if (options->mip_heuristic_portfolio) {
    portfolio::run_presolve(mipsolver, budget);
  } else {
    // Sequential mode: run heuristics in order.
    if (options->mip_heuristic_run_feasibility_jump) {
      if (fj::run(mipsolver, budget)) {
        return true;  // proven infeasible
      }
    }
    if (options->mip_heuristic_run_fpr) {
      fpr::run(mipsolver, budget);
    }
    if (options->mip_heuristic_run_local_mip) {
      local_mip::run(mipsolver, budget);
    }
  }

  // Scylla runs independently (not a portfolio arm yet).
  if (options->mip_heuristic_run_scylla) {
    if (options->mip_heuristic_scylla_parallel) {
      scylla::run_parallel(mipsolver, budget);
    } else {
      scylla::run(mipsolver, budget);
    }
  }

  return false;
}

}  // namespace heuristics
