#include "incumbent_sink.h"

#include "io/HighsIO.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <atomic>

namespace {

// Process-global dispatch counter (issue #106).
//
// `(name, dispatch)` has to identify one dispatch uniquely, and a per-sink
// counter cannot do it: `run_sequential` builds one sink for a whole
// presolve chain while `fpr_lp` builds a fresh one per dive dispatch, so a
// per-sink counter would stamp every dive `0`.  A process-global counter
// also survives the nested `HighsMipSolver` instances RENS/RINS run their
// sub-MIPs on, whose presolve chains would otherwise restart the numbering
// inside an enclosing solve's log.
//
// Ids are therefore neither dense nor zero-based within any one solve, and
// nothing may assume they are — `bench/parse_highs_log.py` groups by the
// pair and never by position.
std::atomic<uint64_t> g_next_dispatch_id{0};

// The `kChain` name a solution-source tag belongs to.
//
// A second spelling of the same pairing `mode_dispatch.cpp`'s `kChain`
// already holds — deliberately, because the sink is handed the tag and not
// the name, and `set_source`'s signature is fixed by its caller.  The
// pairing is pinned from the outside instead of from a shared table:
// `tests/test_heursol_trace.cpp` checks that every `[HeurSol] name=` in a
// solve is one the same log's `[Heur] name=` lines announced, which fails
// if either side drifts.
const char* name_for_source(int source) {
    switch (source) {
        case kSolutionSourceFJ:
            return "fj";
        case kSolutionSourceFPR:
            return "fpr";
        case kSolutionSourceLocalMIP:
            return "local_mip";
        case kSolutionSourceScylla:
            return "scylla";
        case kSolutionSourceFprLp:
            return "fpr_lp";
        default:
            // Reachable only for the generic `kSolutionSourceHeuristic` tag
            // a presolve sink is constructed with, which is replaced by
            // `set_source` before any worker runs — so no offer is made
            // under it on any current path.  Named rather than asserted:
            // an unattributed trace line is a better failure than a crash
            // in an observability path.
            return "unknown";
    }
}

}  // namespace

IncumbentSink::IncumbentSink(HighsMipSolver& mipsolver, int source)
    : mipsolver_(mipsolver),
      pool_(kPoolCapacity, mipsolver.model_->sense_ == ObjSense::kMinimize),
      source_(source) {
    // Seed first, register second: the seeded incumbent came from HiGHS
    // and re-submitting it here would be pointless work on the accept
    // path.
    seed_pool(pool_, mipsolver);

    auto* mipdata = mipsolver.mipdata_.get();
    pool_.set_on_accept([this, mipdata](const std::vector<double>& sol, int src) {
        std::scoped_lock guard(highs_mtx_);
        mipdata->trySolution(sol, src);
    });

    // Constructing a sink opens a dispatch: `fpr_lp` builds one per dive
    // and never calls `set_source`, so this is that path's only boundary.
    begin_dispatch(source);
}

void IncumbentSink::begin_dispatch(int source) {
    source_ = source;
    dispatch_name_ = name_for_source(source);
    dispatch_id_ = g_next_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    // Same clock as `EffortLedger::now_s()` — the solver's own
    // `HighsMipSolver::timer_`, not a raw `steady_clock` — so `[HeurSol]
    // wall_ms` shares an origin with `[Heur] start_s` and with HiGHS's own
    // display-line time column.  It is not monotonic (`HighsTimer` bottoms
    // out in `high_resolution_clock`, which libstdc++ aliases to
    // `system_clock`), so `wall_ms` may come out negative; the bench parser
    // accepts the sign rather than dropping the sample.
    dispatch_start_s_ = mipsolver_.timer_.read();
}

bool IncumbentSink::offer(double objective, const std::vector<double>& solution,
                          const WorkerTrace& trace, size_t effort_at) {
    const bool accepted = pool_.try_add(objective, solution, source_);
    if (accepted) {
        accepted_.fetch_add(1, std::memory_order_relaxed);
    }
    trace_offer(trace, effort_at, objective, accepted);
    return accepted;
}

void IncumbentSink::trace_offer(const WorkerTrace& trace, size_t effort_at, double objective,
                                bool accepted) const {
    // Threading, and why there is no mutex here.
    //
    // `offer` is called concurrently by every worker of a dispatch.  This
    // runs *after* `SolutionPool::try_add` has returned, so it is outside
    // the pool's lock and lengthens no critical section: the serialisation
    // an offer already pays is unchanged.  Nothing new is shared either —
    // the dispatch fields are written only on the dispatching thread with
    // every parallel region joined (see the declarations), `effort_at` and
    // `trace` are the caller's own locals, and `HighsTimer::read()` is a
    // `const` read of clock 0, which the runner's termination poller
    // already performs from worker threads.
    //
    // `highsLogDev` has three delivery paths and the line survives all
    // three whole.  To a file or to the console it is one `vfprintf`, and
    // glibc holds that stream's lock for the call, so it cannot interleave
    // with another worker's.  To a user callback — the path
    // `solve_capturing_log` in `tests/test_common.h` installs, and the one
    // this project's own tests read — it is one `vsnprintf` into a
    // *caller-local* buffer followed by a direct call on the offering
    // thread, so the message is still delivered as a unit; what the
    // callback then does with it, including any locking, is the callback's
    // own business (the test harness takes a mutex).  Either way no
    // ordering between workers is promised and none is needed: consumers
    // group by `(name, dispatch, worker)`, never by position.  HiGHS's own
    // FeasibilityJump already logs from parallel workers at this very
    // level, so this path is not new contention — it is the same one, at a
    // far lower rate (once per offer, not once per weight bump).
    const HighsLogOptions& log_options = mipsolver_.options_mip_->log_options;
    // Check the level before touching the clock: `highsLogDev` would drop
    // the line anyway, but the timer read and the formatting would already
    // have been paid on every offer of every run.  `log_dev_level` is an
    // `HighsInt*` into the options record, stable for the solve.
    if (*log_options.log_dev_level < kHighsLogDevLevelVerbose) {
        return;
    }
    const double wall_ms = (mipsolver_.timer_.read() - dispatch_start_s_) * 1000.0;
    // `%.17g` round-trips a double exactly: the objective is what the
    // calibration compares against a reference value, so it must not be
    // rounded on the way out.
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[HeurSol] name=%s dispatch=%llu worker=%d effort_at=%zu wall_ms=%.1f "
                "obj=%.17g accepted=%d\n",
                dispatch_name_, static_cast<unsigned long long>(dispatch_id_), trace.worker,
                effort_at, wall_ms, objective, accepted ? 1 : 0);
}
