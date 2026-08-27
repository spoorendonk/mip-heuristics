#pragma once

#include "heuristic_common.h"
#include "heuristic_context.h"
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
    LocalMipWorker(HighsMipSolver& mipsolver, const ExecutionContext& exec, const CscMatrix& csc,
                   IncumbentSink& sink, size_t total_budget, size_t stale_budget, uint32_t seed,
                   const double* initial_solution, const uint8_t* binary, WorkerTrace trace);

    AttemptResult run_attempt(size_t attempt_budget);

    [[nodiscard]] bool finished() const { return base_.finished; }

    // Monotone charged effort for the `[HeurSol]` trace (#106): this
    // worker's own `WorkerCtx::effort` plus what the slot's retired
    // occupants — and this occupant's own cold-start construction, which is
    // charged outside `ctx_` — already spent.  `local_mip::run` reads it
    // off the outgoing worker to seed the replacement's `WorkerTrace`.
    [[nodiscard]] size_t traced_effort() const { return trace_.at(ctx_.effort); }

private:
    HighsMipSolver& mipsolver_;
    // Read only for `past_deadline()`, every `kTermCheckInterval` steps of
    // the search loop: one attempt at a large effort option runs far past
    // the solve's `time_limit` otherwise (issue #114).
    const ExecutionContext& exec_;
    const CscMatrix& csc_;
    IncumbentSink& sink_;
    Rng rng_;

    // Effort / staleness / finished bookkeeping.  `total_budget` and
    // `stale_budget` are set in the constructor; since issue #111 the
    // patience is the caller's absolute, instance-scaled
    // `HeuristicBudget::worker_stale`, not `total_budget >> 2` — a
    // quarter of an allowance grows with the allowance and so can never
    // stop a heuristic from spending all of it.
    WorkerBudgetState base_;

    // Trace-only slot identity; see `WorkerTrace` in worker_base.h.
    const WorkerTrace trace_;

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
