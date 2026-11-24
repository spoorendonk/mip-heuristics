#pragma once

// Long-lived mode's owner, queue and sink (#332).  The plug-in sites HiGHS
// calls are in long_lived.h; this is the adapter side, free to be C++23.
//
// The threading rule this file exists to keep: **only the solving thread
// calls `trySolution` or reads `mipdata` (or `model_`, which a restart
// overwrites) while B&B can run.**  `addIncumbent` asserts
// `!parallelLockActive()` and writes every B&B worker's bounds, so a
// heuristic thread that called it would race the search.  A producer
// therefore touches three things only: the `SolutionQueue`, the owner's
// stop flag and deadline, and a `LongLivedSink` built for it on the
// solving thread.  Everything that reaches HiGHS goes through `drain`.

#include "deadline.h"
#include "incumbent_sink.h"

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

class HighsMipSolver;

namespace heuristics {

// Most solutions the queue holds between two drains.  A poll point comes
// before every B&B node batch, so the queue only has to bridge one batch
// (or one root-node stage); a producer that outruns it loses its worst
// points, never its best.
inline constexpr size_t kLongLivedQueueCapacity = 16;

// Bounded multi-producer queue of solutions in the presolved space, drained
// on the solving thread.
//
// `push` filters against a copy of HiGHS's bound that only the solving
// thread refreshes (`publish`), as `IncumbentSink::reaches_highs` does
// against its own: a stale copy is only ever too high, so it lets a point
// through and never loses one HiGHS could store.  The copy lives under the
// queue's lock with the epoch it belongs to, because a bound is meaningless
// in another epoch's space; `push` takes that lock anyway.
class SolutionQueue {
public:
    struct Entry {
        // `HighsMipSolverData::numRestarts` of the space `x` is in.
        int epoch = 0;
        int source = 0;
        HighsObjective objective;
        std::vector<double> x;
    };

    SolutionQueue();

    // Producer side, any thread.  Refuses a point from an older epoch, or
    // one of the current epoch that cannot improve the published bound
    // (unless the MIP solution callback sees every `trySolution`).  When
    // full, the point replaces the worst entry if it ranks better, else it
    // is refused.  A NaN objective ranks as +inf.  Returns whether it was
    // queued.
    [[nodiscard]] bool push(int epoch, int source, const HighsObjective& objective,
                            const std::vector<double>& x);

    // Solving thread: move every entry into `out` (cleared first).
    void take(std::vector<Entry>& out);

    // Solving thread: the bound, epoch and callback state producers filter
    // against from now on.  Entries of an older epoch still queued are
    // dropped and counted in the return value.
    size_t publish(int epoch, double bound, bool callback_sees_all);

private:
    std::mutex mtx_;
    std::vector<Entry> entries_;
    int epoch_ = 0;
    double bound_ = std::numeric_limits<double>::infinity();
    bool callback_sees_all_ = true;
};

// What `LongLived` (long_lived.h) allocates when the option is on: the
// queue, the producers' stop flag, and the threads it joins at
// `cleanupSolve`.  Constructed, started, drained and stopped on the
// solving thread; producers see only `stop_requested`, `past_deadline`
// and the queue.
//
// No producer calls `terminatorTerminated()`: it writes
// (`ExecutionContext::terminator`, heuristic_context.h).  The solving
// thread's own checks end the solve, and `cleanupSolve` stops the owner.
class LongLivedOwner {
public:
    // Publishes HiGHS's bound to the queue, then calls the start hook if a
    // test set one.  On the solving thread, after setup.
    explicit LongLivedOwner(HighsMipSolver& mipsolver);
    // `stop()`, swallowing a producer's exception: a solve that threw
    // past `cleanupSolve` is already failing.
    ~LongLivedOwner();
    LongLivedOwner(const LongLivedOwner&) = delete;
    LongLivedOwner& operator=(const LongLivedOwner&) = delete;

    // Run `body` on a thread of its own until it returns.  It must return
    // soon after `stop_requested()` turns true.  An exception it throws is
    // kept and rethrown by `rethrow`.  Solving thread, before `stop`.
    void start(std::function<void(LongLivedOwner&)> body);

    // Raise the stop flag and join every thread.  Idempotent.  Solving
    // thread.
    void stop();

    // Rethrow the first exception a producer ended with, if any.  After
    // `stop`.
    void rethrow();

    // Publish the current epoch (dropping older entries), take the queue
    // and hand HiGHS, best first, every entry that may still improve its
    // bound (all of them while the MIP solution callback sees every
    // `trySolution`), each under its own source tag; then republish the
    // bound.  Solving thread.
    void drain();

    // Producer side, any thread.
    [[nodiscard]] bool stop_requested() const { return stop_.load(std::memory_order_acquire); }
    [[nodiscard]] bool past_deadline() const { return deadline_.expired(); }
    [[nodiscard]] SolutionQueue& queue() { return queue_; }

    // Entries handed to `trySolution`, and entries dropped for being from
    // an older epoch.  Any thread.
    [[nodiscard]] size_t submitted() const { return submitted_.load(std::memory_order_relaxed); }
    [[nodiscard]] size_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

    // Test only: called with the solver and the owner at the end of the
    // owner's constructor, on the solving thread, so a test can build a
    // `LongLivedSink` and start a producer before any heuristic of #334
    // exists.  Set it around `Highs::run` on the thread that runs it;
    // empty by default.
    using StartHook = std::function<void(HighsMipSolver&, LongLivedOwner&)>;
    static void set_start_hook(StartHook hook);

private:
    void publish();

    HighsMipSolver& mipsolver_;
    SolutionQueue queue_;
    // `time_limit` on the solver's clock, `const`-readable from any thread.
    Deadline deadline_;
    std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
    std::mutex error_mtx_;
    std::exception_ptr error_;
    // `drain`'s scratch, kept for its capacity.
    std::vector<SolutionQueue::Entry> drained_;
    std::atomic<size_t> submitted_{0};
    std::atomic<size_t> dropped_{0};
};

// The `SolutionSink` a long-lived producer offers to: `on_accept` queues
// the solution for the solving thread instead of calling `trySolution`,
// and traces it as `IncumbentSink` does (`HighsSink::trace_offer`).
//
// Built on the solving thread, which is where it reads the incumbent (the
// pool seed), the objective and the epoch; after that it reads no
// `mipdata` and no `model_`.  It belongs to one epoch: a point it queues
// after a restart is in the old space and is dropped at the drain, so a
// producer that outlives a restart needs a new sink.
//
// Its `[HeurSol]` line goes out on the producer's thread while B&B logs on
// the solving thread; `HighsSink::trace_offer` says why the line still
// arrives whole, and a user log callback already sees concurrent calls
// from a dispatch's workers.
class LongLivedSink final : public HighsSink {
public:
    LongLivedSink(HighsMipSolver& mipsolver, LongLivedOwner& owner, int source);

private:
    void on_accept(double objective, const std::vector<double>& solution, int source,
                   const WorkerTrace& trace, size_t effort_at) override;

    SolutionQueue& queue_;
    // `model_->col_cost_` at construction: the producer may not read
    // `model_`, which a restart overwrites.
    std::vector<double> cost_;
    // `numRestarts` at construction: the space the producer computes in.
    int epoch_;
};

}  // namespace heuristics
