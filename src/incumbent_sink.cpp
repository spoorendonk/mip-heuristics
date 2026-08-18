#include "incumbent_sink.h"

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

IncumbentSink::IncumbentSink(HighsMipSolver& mipsolver, int source)
    : pool_(kPoolCapacity, mipsolver.model_->sense_ == ObjSense::kMinimize), source_(source) {
    // Seed first, register second: the seeded incumbent came from HiGHS
    // and re-submitting it here would be pointless work on the accept
    // path.
    seed_pool(pool_, mipsolver);

    auto* mipdata = mipsolver.mipdata_.get();
    pool_.set_on_accept([this, mipdata](const std::vector<double>& sol, int src) {
        std::lock_guard<std::mutex> guard(highs_mtx_);
        mipdata->trySolution(sol, src);
    });
}
