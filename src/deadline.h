#pragma once

#include "lp_data/HConst.h"
#include "util/HighsTimer.h"

// The solve's wall-clock deadline, in the form a *sub-algorithm* polls it
// (issue #117).
//
// `ExecutionContext::past_deadline()` (heuristic_context.h) is the same
// question asked by a heuristic's runner, and it is the shape a runner
// wants: one object carrying the worker count, the seed and the limit.
// The DFS in `fpr_core.cpp` and the repair search below it are reached
// from three different callers (`FprWorker`, `ScyllaWorker`, `fpr_lp`) and
// have no `ExecutionContext` in scope, so they take this instead — two
// words, copyable, and free of every HiGHS MIP header.
//
// Thread-safety is the whole reason this can exist: `HighsTimer::read()`
// is `const` and, for the solve clock, writes nothing, so any worker
// thread may poll it without the poller seat.  See `ExecutionContext` for
// the write-free / writing split.
struct Deadline {
    // The solver's own clock (`HighsMipSolver::timer_`), so a poll shares
    // its origin with the `[Heur] start_s`/`end_s` window the ledger
    // emits and with the `time_limit` option itself.
    //
    // Null means "no finite limit", which makes `expired()` a null test
    // rather than a clock read that can never return true — the reason a
    // poll can sit on a per-node cadence in the DFS without a second
    // "is there a limit at all?" flag beside it.
    const HighsTimer* timer = nullptr;
    double limit = 0.0;

    [[nodiscard]] bool expired() const { return timer != nullptr && timer->read() >= limit; }

    // Seconds left, `kHighsInf` when there is no limit, never negative.
    // For sub-solvers that take a time limit rather than a predicate — and
    // a caller handing this to one must reject a zero itself rather than
    // pass it on, because HiGHS does not read `time_limit == 0.0` as any
    // one thing: LP presolve guards its own timeout on `time_limit > 0`
    // (`Highs.cpp`), so 0.0 removes that guard, while simplex
    // (`HEkk.cpp`) and IPM (`ipx/control.cc`) test only `< kHighsInf` and
    // so treat 0.0 as already expired, aborting on their first check.
    // Neither is a limit anyone asked for.
    [[nodiscard]] double remaining() const {
        if (timer == nullptr) {
            return kHighsInf;
        }
        const double left = limit - timer->read();
        return left > 0.0 ? left : 0.0;
    }
};

// A deadline `limit` seconds into `timer`'s clock, collapsing an infinite
// limit to the free-to-poll form above.
inline Deadline make_deadline(const HighsTimer& timer, double limit) {
    return limit < kHighsInf ? Deadline{&timer, limit} : Deadline{};
}
