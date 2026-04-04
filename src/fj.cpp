#include "fj.h"

#include <vector>

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

namespace fj {

bool run(HighsMipSolver &mipsolver, size_t max_effort) {
  auto *mipdata = mipsolver.mipdata_.get();
  std::vector<double> sol;
  double obj = 0.0;
  size_t effort = 0;
  auto status =
      mipdata->feasibilityJumpCapture(sol, obj, effort, max_effort, nullptr);
  if (!sol.empty()) {
    mipdata->trySolution(sol, kSolutionSourceHeuristic);
  }
  mipdata->heuristic_effort_used += effort;
  return status == HighsModelStatus::kInfeasible;
}

} // namespace fj
