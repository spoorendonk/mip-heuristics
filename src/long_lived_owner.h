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
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "solution_pool.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

class HighsMipSolver;
class HighsOptions;

namespace heuristics {

// One long-lived slot (#356): heuristic `name` running `run` as slot `worker`
// of its kind, offering under `source`, restarting from the shared pool by
// `policy`.
struct LongLivedSlotSpec {
    const char* name = nullptr;
    int source = 0;
    UntilStopped run = nullptr;
    HeuristicBudget budget;
    SolutionPool::RestartPolicy policy = SolutionPool::RestartPolicy::kMixed;
    int worker = 0;
};

// The slots `options` asks for, on a model of `nnz` nonzeros:
// `mip_heuristic_long_lived_threads` of them, dealt round-robin in chain
// order to FJ, LocalMIP, ruin and recombine, those whose effort option is
// above zero (FJ also needs `mip_heuristic_run_feasibility_jump`).  Slot `i`
// of a heuristic is its worker `i`, for its seed.  Each gets one chain
// worker's attempt cap and patience at one worker, and no other ceiling.
// Empty when the option is 0 or none of the four runs.  Defined beside the
// chain table (mode_dispatch.cpp).
[[nodiscard]] std::vector<LongLivedSlotSpec> long_lived_slot_specs(const HighsOptions& options,
                                                                   size_t nnz);

// Why the slots were stopped, for their `[Heur] stopped_by`.
enum class SlotStop : uint8_t { kRestart, kEnd };

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
    // full, the point replaces an entry of an older epoch than its own if
    // there is one (#356), else the worst entry if it ranks better, else it
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
// queue, the solution pool every producer shares (#333), the producers'
// stop flag, and the threads it joins at `cleanupSolve`.  Constructed,
// started, drained and stopped on the solving thread; producers see only
// `stop_requested`, `past_deadline`, the queue and the pool, the last
// through the sinks and restart sources built for them.
//
// The pool belongs to one epoch, as a sink does: a restart changes the
// presolved space, so the first call after one replaces the pool with an
// empty one over the new columns.  A producer still holding the old pool
// keeps it alive and offers into it harmlessly; its queued points are
// dropped at the drain.  Within an epoch the pool holds HiGHS's latest
// incumbent at each drain where the bound moved, whatever found it
// (`drain`; incumbents found between two drains reach it as the last), and
// the chain's final pool (`seed`).
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

    // Raise the stop flag and join every thread.  Idempotent: the first
    // call's time is the one `charge_slots` measures joins from.  Solving
    // thread.
    void stop();

    // Start one thread per spec (#356), each running its heuristic's
    // `run_until_stopped` over one view of the presolved model, built here
    // and shared by every slot, with a `LongLivedSink` and a restart source
    // of its own.  Each slot's `exec` carries the owner's stop flag (on its
    // deadline too, for recombine's bound-prop arm and its FPR DFS), the
    // solve's deadline, no terminator and one worker.  Does nothing when
    // `specs` is empty, the owner is stopped (a restart stops it for good),
    // or the deadline has passed.  Solving thread, once the presolve chain
    // has returned; at most once per owner.
    void start_slots(std::span<const LongLivedSlotSpec> specs);

    // Book every slot's effort, one `[Heur] phase=longlived` line each, and
    // drop the slots.  `stopped_by` is `deadline` when the time limit came
    // before the stop, else `why` (`restart` or `end`), or `done` when the
    // slot returned before either; `join_ms` is how long after the first of
    // the two it returned, 0 for `done`.  Solving thread, after `stop`.
    void charge_slots(SlotStop why);

    // Rethrow the first exception a producer ended with, if any.  After
    // `stop`.
    void rethrow();

    // Admit HiGHS's incumbent to the pool if its bound has fallen since the
    // pool last saw it.  HiGHS does not record which of its components
    // found an incumbent, so B&B, its own heuristics, sub-MIPs and
    // `fpr_lp`'s dives share one origin, `kSolutionSourceHeuristic`, the
    // generic tag `HighsSink` already seeds the incumbent with.  Then publish the current epoch
    // (dropping older entries), take the queue and hand HiGHS, best first,
    // every entry that may still improve its bound (all of them while the
    // MIP solution callback sees every `trySolution`), each under its own
    // source tag.  Then admit the incumbent again if an entry moved it,
    // under that entry's tag: it is in the pool already unless evicted
    // since, and the twin rule keeps a second copy out.  Last, republish the
    // bound.  Solving thread.
    void drain();

    // The current epoch's shared pool, for a `LongLivedSink`.  A restart's
    // fresh pool is fed the incumbent when it is swapped in, so it is never
    // empty while HiGHS has one.  Solving thread.
    [[nodiscard]] const std::shared_ptr<SolutionPool>& pool();

    // A `RestartSource` over the current epoch's pool, by `policy`
    // (`SolutionPool::get_restart`): the best member, or a diverse one.
    // Built on the solving thread; callable from any thread.  Bound to the
    // epoch it was built in: after a restart it still serves that epoch's
    // points, in the old presolved space, so a producer builds a new one
    // with its new sink.
    [[nodiscard]] RestartSource restart_source(SolutionPool::RestartPolicy policy);

    // Copy the chain's final pool into the shared one, each entry under its
    // own tag, then admit the incumbent.  On the solving thread, when the
    // presolve chain returns (`run_sequential`).
    void seed(SolutionSink& chain);

    // One `[LongLivedPool]` dev-verbose line: the epoch, the members and
    // their origins best first, as HiGHS's display-line source codes.
    // Solving thread.
    void log_pool();

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
    // A running slot; long_lived.cpp.
    struct Slot;

    void publish();
    // Replace the pool if HiGHS has restarted since it was built.
    void sync_epoch();
    // Admit `mipdata->incumbent` under `origin` if HiGHS's bound is below
    // the one last fed.
    void feed_incumbent(int origin);

    HighsMipSolver& mipsolver_;
    SolutionQueue queue_;
    // Read and replaced on the solving thread only; producers hold copies.
    std::shared_ptr<SolutionPool> pool_;
    // `numRestarts` of the space `pool_` is in.
    int pool_epoch_;
    // `highs_bound` when the incumbent was last fed, +inf in a new epoch.
    double fed_bound_ = std::numeric_limits<double>::infinity();
    // `seed`'s and `log_pool`'s scratch.
    std::vector<SolutionPool::EntryRef> refs_;
    // `time_limit` on the solver's clock, `const`-readable from any thread.
    Deadline deadline_;
    std::atomic<bool> stop_{false};
    // The solver's clock at the first `stop`.
    double stop_s_ = 0.0;
    // The slots' view and the transpose it reads, built by `start_slots`
    // and valid until a restart overwrites `model_`, which stops the slots
    // first.  The destructor joins every thread before a member goes.
    CscMatrix csc_;
    std::optional<ProblemView> view_;
    std::vector<std::unique_ptr<Slot>> slots_;
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
// It offers into the owner's shared pool under its own tag, so sinks of
// heuristics running at once keep their attributions apart (#333).
//
// Built on the solving thread, which is where it takes the pool and reads
// the objective and the epoch; after that it reads no `mipdata` and no
// `model_`.  It belongs to one epoch: a point it queues after a restart is
// in the old space and is dropped at the drain, so a producer that
// outlives a restart needs a new sink.
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
