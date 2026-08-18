#pragma once

#include "heuristic_common.h"
#include "incumbent_sink.h"
#include "local_mip_caches.h"
#include "local_mip_core.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "util/HighsInt.h"
#include "worker_base.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace local_mip_detail {

inline constexpr double kPerturbBinaryFraction = 0.2;

// Perturb solution: flip ~20% of binary vars, randomly shift general integers.
//
// `binary` is the dispatch's `isBinary` snapshot (`ProblemView::binary`),
// not the live root domain: this runs on worker threads while a peer's
// accepted solution propagates that domain (issue #99).  Must be at least
// `ncol` entries.
void perturb_solution(std::vector<double>& solution, const uint8_t* binary,
                      const std::vector<HighsVarType>& integrality,
                      const std::vector<double>& col_lb, const std::vector<double>& col_ub,
                      HighsInt ncol, Rng& rng);

// Worker wrapping WorkerCtx. Runs weighted local search, accumulating
// effort and submitting improving solutions through the shared
// `IncumbentSink`.
class LocalMipWorker {
public:
    // `initial_solution` is the start `resolve_worker_start` produced, or
    // null for a bounds-clamped zero start.  It used to fall back to
    // `mipdata->incumbent` when null — a live read from a worker thread
    // (issue #98), and a dead one: the caller only passes null when its
    // resolved start is empty, which by `resolve_worker_start`'s ordering
    // means the pool and the incumbent were empty too.
    // `binary` is the dispatch's `isBinary` snapshot (`ProblemView::binary`,
    // issue #99); it must outlive the worker.
    LocalMipWorker(HighsMipSolver& mipsolver, const CscMatrix& csc, IncumbentSink& sink,
                   size_t total_budget, uint32_t seed, const double* initial_solution,
                   const uint8_t* binary);

    AttemptResult run_attempt(size_t attempt_budget);

    bool finished() const { return base_.finished; }

private:
    HighsMipSolver& mipsolver_;
    const CscMatrix& csc_;
    IncumbentSink& sink_;
    Rng rng_;

    // Effort / staleness / finished bookkeeping.  `total_budget` and
    // `stale_budget` are set in the constructor.
    WorkerBudgetState base_;

    WorkerCtx ctx_;
    std::vector<HighsInt> costed_vars_;
    std::vector<HighsInt> binary_vars_;

    bool best_feasible_ = false;
    double best_objective_ = 0.0;
    std::vector<double> best_solution_;

    HighsInt steps_since_improvement_ = 0;
    HighsInt restart_count_ = 0;
    HighsInt step_ = 0;

    // Paper-style random-walk diversification (Lin, Zou, Cai §4.1):
    // when a feasible-mode plateau triggers, perturb the solution
    // and continue searching instead of declaring the worker finished.
    // Bounded by `kFeasibleMaxRandomWalks` so we eventually stop on
    // instances where perturbation can't break the plateau.
    HighsInt feasible_random_walks_done_ = 0;
};

}  // namespace local_mip_detail
