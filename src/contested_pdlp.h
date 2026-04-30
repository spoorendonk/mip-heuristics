#pragma once

#include "Highs.h"
#include "lp_data/HighsStatus.h"
#include "util/HighsInt.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

class HighsMipSolver;

// Thread-safe wrapper around a single PDLP `Highs` instance shared by N
// Scylla workers.  One mutex guards the entire `changeColsCost →
// setSolution → run → getSolution` critical section so only one PDLP
// solve is in flight at a time.  This eliminates concurrency questions
// around the underlying (possibly GPU-backed cuPDLP) solver and keeps
// memory to a single LP copy + single iterate regardless of N.
//
// Overlap design (issue #76): workers that cannot grab the mutex fall
// back to rounding against the most-recent *completed* PDLP snapshot,
// published via a `std::atomic<std::shared_ptr<const Snapshot>>` slot.
// Readers acquire-load the shared_ptr (libstdc++ implements the
// specialisation with a brief internal spinlock rather than truly
// lock-free, but contention on reads is bounded and far shorter than
// the PDLP solve itself).  Writers serialise through the mutex and
// release-store the new snapshot atomically so stale readers never
// tear.  This lets N-1 workers keep producing useful FPR work while
// one worker is inside the PDLP solve, without breaking the one-
// solve-in-flight invariant (cuPDLP GPU state safety).
//
// Lifetime invariant: the wrapped LP is built once from
// `mipsolver.mipdata_->AR*` at construction time via
// `pump::build_lp_relaxation`.  The LP rows / constraint matrix are
// frozen for the lifetime of the instance — only column costs change
// on each `solve()`.  Safe for presolve-time use (HiGHS internals are
// immutable there); NOT safe for B&B-dive use where node bounds mutate
// between calls.
class ContestedPdlp {
public:
    struct SolveResult {
        std::vector<double> col_value;
        std::vector<double> row_dual;
        HighsInt pdlp_iters = 0;
        HighsStatus status = HighsStatus::kError;
        HighsModelStatus model_status = HighsModelStatus::kNotset;
        bool value_valid = false;
        bool dual_valid = false;
    };

    // Immutable snapshot of a completed PDLP solve.  Workers keep a
    // local `shared_ptr<const Snapshot>` so they can round against stale
    // data while a peer holds the mutex.  The object is never mutated
    // after publication — every completed solve produces a new instance.
    //
    // `generation` is a monotonic counter (per `ContestedPdlp` instance)
    // assigned at publication time; the first published snapshot is
    // generation 1, each subsequent fresh publish increments by one.
    // Use it instead of `shared_ptr` address comparison for "did the
    // upstream snapshot change since I last looked?" — addresses can
    // be recycled if a freed Snapshot is replaced by a new allocation
    // at the same heap slot, but generation numbers are unambiguous.
    struct Snapshot {
        std::vector<double> col_value;
        std::vector<double> row_dual;
        HighsInt pdlp_iters = 0;
        bool value_valid = false;
        bool dual_valid = false;
        uint64_t generation = 0;
    };

    // Outcome of `try_solve_or_snapshot`: either a freshly computed
    // solve (we held the mutex) or a reference to the most recent
    // completed snapshot (someone else was solving).
    struct TrySolveResult {
        bool fresh = false;
        SolveResult solve;
        std::shared_ptr<const Snapshot> stale_snapshot;
    };

    // Builds the shared PDLP Highs instance from the presolved MIP
    // relaxation.  `initialized()==false` when the instance has no
    // rows / no nonzeros; callers should short-circuit.
    ContestedPdlp(HighsMipSolver &mipsolver, HighsInt pdlp_iter_cap);

    virtual ~ContestedPdlp() = default;

    ContestedPdlp(const ContestedPdlp &) = delete;
    ContestedPdlp &operator=(const ContestedPdlp &) = delete;

    bool initialized() const { return initialized_; }
    size_t nnz_lp() const { return nnz_lp_; }
    HighsInt num_col() const { return ncol_; }

    // Solve PDLP with the caller's objective and warm-start.  The mutex
    // is held for the full changeColsCost + setSolution + run +
    // getSolution path; callers block when another chain is active.
    //
    // `warm_start_col_value` / `warm_start_row_dual` may be empty (cold
    // start) but must otherwise have length == ncol/nrow respectively.
    // `epsilon` is passed as `pdlp_optimality_tolerance`.  `time_limit`
    // is a wall-clock cap for this single solve (seconds).
    //
    // On success, publishes the result as the latest Snapshot so that
    // other workers hitting `try_solve_or_snapshot` can round against
    // it concurrently.
    SolveResult solve(const std::vector<double> &modified_cost,
                      const std::vector<double> &warm_start_col_value,
                      const std::vector<double> &warm_start_row_dual, bool warm_start_valid,
                      double epsilon, double time_limit);

    // Non-blocking variant: `try_lock` the PDLP mutex.
    //
    //  - Lock acquired: run a fresh PDLP solve, publish the Snapshot,
    //    release, return `{fresh=true, solve=<result>}`.
    //  - Lock contended: return `{fresh=false, stale_snapshot=<latest>}`
    //    immediately.  The snapshot pointer may be null if no solve
    //    has completed yet (cold caller).
    //
    // Invariant preserved: at most one PDLP solve is in flight at a
    // time (cuPDLP GPU state safety).  Enforced by `try_lock` plus a
    // debug assertion on `in_flight_count_`.
    TrySolveResult try_solve_or_snapshot(const std::vector<double> &modified_cost,
                                         const std::vector<double> &warm_start_col_value,
                                         const std::vector<double> &warm_start_row_dual,
                                         bool warm_start_valid, double epsilon, double time_limit);

    // Latest completed Snapshot (shared ownership) or null if no solve
    // has completed yet.  Read via `std::atomic<std::shared_ptr<>>`
    // acquire-load (libstdc++ uses a brief internal spinlock — not
    // strictly lock-free, but well below PDLP solve latency).  Callers
    // may hold the returned pointer across iterations since a Snapshot
    // is immutable after publication.
    std::shared_ptr<const Snapshot> latest_snapshot() const {
        return snapshot_.load(std::memory_order_acquire);
    }

    // Exposed for tests: peak number of concurrent solves observed.
    // Must always be <= 1 (the one-solve-in-flight invariant).
    int peak_in_flight() const { return peak_in_flight_.load(std::memory_order_relaxed); }

    // Exposed for tests: number of Snapshots published so far.  Bumped
    // once per successful `run_locked_with_accounting`.
    uint64_t snapshot_generation() const {
        return snapshot_generation_.load(std::memory_order_acquire);
    }

protected:
    // Default is the real HiGHS path; tests override with a canned
    // solve (sleep + fake output) to exercise the lock/snapshot
    // plumbing without dragging a full Highs instance in.  Caller
    // (either `solve()` or `try_solve_or_snapshot()`) already holds
    // `mu_` when this runs.
    virtual SolveResult solve_locked(const std::vector<double> &modified_cost,
                                     const std::vector<double> &warm_start_col_value,
                                     const std::vector<double> &warm_start_row_dual,
                                     bool warm_start_valid, double epsilon, double time_limit);

    // Constructor for the test double: does not build the Highs LP.
    // `initialized()` is forced to true so tests can drive the lock /
    // snapshot paths with an overridden `solve_locked`.
    struct ForTesting {};
    explicit ContestedPdlp(ForTesting);

    // Test hook: enter the locked critical section (returns a unique_lock)
    // without running a solve.  Lets tests deterministically simulate
    // "worker A is inside the solve right now" to drive the try_lock
    // contention path.  Protected so only test subclasses can reach it.
    std::unique_lock<std::mutex> acquire_for_test() { return std::unique_lock<std::mutex>(mu_); }

    // Test hook: publish an arbitrary Snapshot without running a solve.
    // Bumps `snapshot_generation_` so tests can verify visibility.
    // CONTRACT: caller MUST hold `mu_` (typically via
    // `acquire_for_test()`); the function does not lock internally so
    // it can be used inside a fixture that already simulates the
    // production publish path's serialisation.
    void publish_snapshot_for_test(Snapshot snap);

private:
    // Wraps `solve_locked` with the in-flight-count tripwire and the
    // snapshot publication.  `mu_` must be held on entry.
    SolveResult run_locked_with_accounting(const std::vector<double> &modified_cost,
                                           const std::vector<double> &warm_start_col_value,
                                           const std::vector<double> &warm_start_row_dual,
                                           bool warm_start_valid, double epsilon,
                                           double time_limit);

    // Publish the result of a just-completed solve as the latest
    // Snapshot.  Only called while `mu_` is held, so publications are
    // serialised; concurrent stale readers see the update via atomic
    // release/acquire.
    void publish_snapshot_locked(const SolveResult &result);

    std::mutex mu_;
    Highs highs_;
    bool initialized_ = false;
    size_t nnz_lp_ = 0;
    HighsInt ncol_ = 0;
    HighsInt nrow_ = 0;

    // Atomic shared_ptr slot.  Written only by the mutex holder,
    // concurrent with lock-free stale readers.  Using the C++20
    // `std::atomic<std::shared_ptr<T>>` partial specialisation rather
    // than the deprecated free-function overloads so the release /
    // acquire happens in one call.
    std::atomic<std::shared_ptr<const Snapshot>> snapshot_;

    std::atomic<int> in_flight_count_{0};
    std::atomic<int> peak_in_flight_{0};
    std::atomic<uint64_t> snapshot_generation_{0};
};
