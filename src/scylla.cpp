#include "scylla.h"

#include "heuristic_common.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "pump_worker.h"
#include "solution_pool.h"

#include <algorithm>

namespace scylla {

void run_parallel(HighsMipSolver &mipsolver, size_t max_effort) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return;
    }

    const bool minimize = (model->sense_ == ObjSense::kMinimize);

    auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    SolutionPool pool(kPoolCapacity, minimize);
    seed_pool(pool, mipsolver);

    const int mem_cap = max_workers_for_memory(estimate_worker_memory_scylla(ncol, 1));
    const int M = std::min({highs::parallel::num_threads(), 4, mem_cap});

    PumpWorker worker(mipsolver, csc, pool, max_effort, kBaseSeedOffset, M);
    while (!worker.finished() && worker.total_effort() < max_effort) {
        worker.run_epoch(max_effort);
    }

    mipdata->heuristic_effort_used += worker.total_effort();

    for (auto &entry : pool.sorted_entries()) {
        mipdata->trySolution(entry.solution, kSolutionSourceScylla);
    }
}

}  // namespace scylla
