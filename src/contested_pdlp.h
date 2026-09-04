#pragma once

#include "deadline.h"
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
    ContestedPdlp(HighsMipSolver& mipsolver, HighsInt pdlp_iter_cap);

    virtual ~ContestedPdlp() = default;

    ContestedPdlp(const ContestedPdlp&) = delete;
    ContestedPdlp& operator=(const ContestedPdlp&) = delete;

    bool initialized() const { return initialized_; }
    size_t nnz_lp() const { return nnz_lp_; }
    HighsInt num_col() const { return ncol_; }

    // Solve PDLP with the caller's objective and warm-start.  The mutex
    // is held for the full changeColsCost + setSolution + run +
    // getSolution path; callers block when another chain is active.
    //
    // `warm_start_col_value` / `warm_start_row_dual` may be empty (cold
    // start) but must otherwise have length == ncol/nrow respectively.
    // `epsilon` is the pump's single stopping error.  It is written to
    // `kkt_tolerance`, which cuPDLP-C resolves into all three of its
    // termination tolerances — `D_PRIMAL_TOL`, `D_DUAL_TOL` and
    // `D_GAP_TOL` (#140).  Writing `pdlp_optimality_tolerance` alone, as
    // this used to, relaxed the duality gap while the two feasibilities
    // the paper names stayed at the HiGHS default; writing the three
    // options explicitly, as the first fix did, additionally perturbs LP
    // presolve and so changes the LP being solved.  See `solve_locked`
    // for the evidence behind both of those.
    //
    // This solve's wall-clock cap is *not* a parameter (issue #117): it is
    // the time left on the solve's own deadline, read inside the critical
    // section.  A caller cannot compute it — it computes a value, then
    // blocks here for the length of a peer's whole solve, and what it
    // computed is stale by exactly that wait.  A worker that reached this
    // call after the deadline gets an empty `SolveResult` and no solve at
    // all, which is what retires it in `absorb_fresh_solve`.
    //
    // On success, publishes the result as the latest Snapshot so that
    // other workers hitting `try_solve_or_snapshot` can round against
    // it concurrently.
    SolveResult solve(const std::vector<double>& modified_cost,
                      const std::vector<double>& warm_start_col_value,
                      const std::vector<double>& warm_start_row_dual, bool warm_start_valid,
                      double epsilon);

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
    TrySolveResult try_solve_or_snapshot(const std::vector<double>& modified_cost,
                                         const std::vector<double>& warm_start_col_value,
                                         const std::vector<double>& warm_start_row_dual,
                                         bool warm_start_valid, double epsilon);

    // Latest completed Snapshot (shared ownership) or null if no solve
    // has completed yet.  Read via `std::atomic<std::shared_ptr<>>`
    // acquire-load (libstdc++ uses a brief internal spinlock — not
    // strictly lock-free, but well below PDLP solve latency).  Callers
    // may hold the returned pointer across iterations since a Snapshot
    // is immutable after publication.
    std::shared_ptr<const Snapshot> latest_snapshot() const {
        return snapshot_.load(std::memory_order_acquire);
    }

    // The tolerance options as they currently stand on the wrapped
    // instance.  Exposed for tests, and both halves of it are load-bearing
    // (#140): `kkt` is the one `solve_locked` writes from `epsilon` on
    // every solve, and the other three must stay at their HiGHS defaults.
    // The original reason was that writing *those* perturbs LP presolve and
    // changes the LP the pump is solving; #153 then turned presolve off on
    // this instance, so that reason is history.  What survives it: the
    // three are not what cuPDLP-C's termination check resolves from, and
    // `kkt_tolerance` keeps HiGHS's own KKT accounting consistent with the
    // solve.  None of it is observable from outside the class without a way
    // to read the options back.
    //
    // THREAD CONTRACT: dispatching-thread only, with no solve in flight.
    // This reads `highs_` without taking `mu_`, and `solve_locked` writes
    // the same options under it — matching the explicit contracts on
    // `acquire_for_test` and `publish_snapshot_for_test`.  Call it after a
    // solve has returned and before another is started.
    struct SolveTolerances {
        // Written from `epsilon`; cuPDLP-C resolves it into all three
        // termination parameters.
        double kkt = 0.0;
        // Deliberately *not* written.  Historically because `HPresolve`
        // reads them too, so driving them from `epsilon` presolved a
        // different LP -- moot since #153 turned presolve off here.  They
        // stay unwritten because `kkt_tolerance` is the parameter
        // cuPDLP-C's termination check resolves from.  Expected to equal
        // `kDefaultKktTolerance` at all times.
        double primal_feasibility = 0.0;
        double dual_feasibility = 0.0;
        double pdlp_optimality = 0.0;
        // The per-solve deadline `solve_locked` writes from
        // `Deadline::remaining()` (#152).  Read back for the same reason as
        // `kkt`: nothing outside this class can otherwise see that the
        // solve was given a deadline at all, and deleting the write left
        // the whole suite green on a build that handed cuPDLP-C none —
        // every wall-clock case here is an *upper* bound, which an
        // unlimited sub-solve does not violate, and the one lower bound
        // (#152's ratio) is only made easier by it.
        double time_limit = 0.0;
    };
    SolveTolerances tolerances_for_test() const;

    // The wrapped instance's accumulated `Highs` run time, i.e. exactly
    // the quantity `runPresolve` charges against `options_.time_limit`
    // (#152).  `solve_locked` zeroes the clock before every solve, so
    // this reads back one solve's duration however many have run; without
    // that reset it is their sum, which is what made a dispatch's
    // per-solve limit meet the accumulated total at the halfway point.
    //
    // THREAD CONTRACT: dispatching-thread only, with no solve in flight —
    // as for `tolerances_for_test` above, and for the same reason.
    double run_time_for_test() const;

    // The presolve status the wrapped instance's last `run()` left behind
    // (#153).  `presolve` is written `off` in the constructor, so this must
    // read `HighsPresolveStatus::kNotPresolved` after every solve — the
    // only way, from outside the class, to see that the pump's warm start
    // is reaching cuPDLP-C in the full column space rather than as a
    // prefix truncated onto a reduced LP's columns.
    //
    // `getModelPresolveStatus()` stands after `run()` returns:
    // `clearPresolve` runs at the head of the next `run`/`presolve` or on a
    // model-modifying call, and `changeColsCost` precedes `run` inside
    // `solve_locked`, so this reports the last solve.
    //
    // THREAD CONTRACT: dispatching-thread only, with no solve in flight —
    // as for `tolerances_for_test` and `run_time_for_test` above, and for
    // the same reason.
    HighsPresolveStatus presolve_status_for_test() const;

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
    // `mu_` when this runs, and `time_limit` is the time left on the
    // deadline as of *inside* the lock, not as of the call (#117).
    virtual SolveResult solve_locked(const std::vector<double>& modified_cost,
                                     const std::vector<double>& warm_start_col_value,
                                     const std::vector<double>& warm_start_row_dual,
                                     bool warm_start_valid, double epsilon, double time_limit);

    // Constructor for the test double: does not build the Highs LP.
    // `initialized()` is forced to true so tests can drive the lock /
    // snapshot paths with an overridden `solve_locked`.  `deadline`
    // defaults to one that never expires, which is what a test driving the
    // lock/snapshot plumbing wants; a test of the deadline itself passes
    // its own `HighsTimer` (issue #117).
    struct ForTesting {};
    explicit ContestedPdlp(ForTesting /*unused*/, Deadline deadline = {});

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
    // Wraps `solve_locked` with the in-flight-count tripwire, the
    // deadline's remaining time and the snapshot publication.  `mu_` must
    // be held on entry — which is what makes this, and not either public
    // entry point, the place the remaining time is read (#117).
    SolveResult run_locked_with_accounting(const std::vector<double>& modified_cost,
                                           const std::vector<double>& warm_start_col_value,
                                           const std::vector<double>& warm_start_row_dual,
                                           bool warm_start_valid, double epsilon);

    // Publish the result of a just-completed solve as the latest
    // Snapshot.  Only called while `mu_` is held, so publications are
    // serialised; concurrent stale readers see the update via atomic
    // release/acquire.
    void publish_snapshot_locked(const SolveResult& result);

    std::mutex mu_;
    Highs highs_;
    // The solve's wall-clock deadline, taken from the MIP solver at
    // construction.  Every solve's time limit is derived from it inside
    // the lock; nothing outside this class passes one in (#117).
    Deadline deadline_;
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
