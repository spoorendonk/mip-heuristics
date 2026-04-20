#pragma once

#include "epoch_runner.h"
#include "heuristic_common.h"
#include "local_mip_caches.h"
#include "local_mip_core.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "solution_pool.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace local_mip_detail {

inline constexpr double kPerturbBinaryFraction = 0.2;
inline constexpr size_t kEpochsPerWorker = 10;

// Perturb solution: flip ~20% of binary vars, randomly shift general integers.
void perturb_solution(std::vector<double> &solution, const HighsMipSolverData &mipdata,
                      const std::vector<HighsVarType> &integrality,
                      const std::vector<double> &col_lb, const std::vector<double> &col_ub,
                      HighsInt ncol, std::mt19937 &rng);

// EpochWorker wrapping WorkerCtx. Runs weighted local search against the
// supplied SolutionPool, accumulating effort and submitting improving
// solutions to the pool.
class LocalMipWorker {
public:
    LocalMipWorker(HighsMipSolver &mipsolver, const CscMatrix &csc, SolutionPool &pool,
                   size_t total_budget, uint32_t seed, const double *initial_solution,
                   size_t stale_budget = 0);

    EpochResult run_epoch(size_t epoch_budget);

    bool finished() const { return finished_; }

    void reset_staleness() { effort_since_improvement_ = 0; }

private:
    HighsMipSolver &mipsolver_;
    const CscMatrix &csc_;
    SolutionPool &pool_;
    const size_t total_budget_;
    const size_t stale_budget_;
    std::mt19937 rng_;

    WorkerCtx ctx_;
    std::vector<HighsInt> costed_vars_;
    std::vector<HighsInt> binary_vars_;

    bool best_feasible_ = false;
    double best_objective_ = 0.0;
    std::vector<double> best_solution_;

    size_t total_effort_ = 0;
    size_t effort_since_improvement_ = 0;
    HighsInt steps_since_improvement_ = 0;
    HighsInt restart_count_ = 0;
    HighsInt step_ = 0;
    bool finished_ = false;
};

static_assert(EpochWorker<LocalMipWorker>, "LocalMipWorker must satisfy EpochWorker concept");

}  // namespace local_mip_detail
