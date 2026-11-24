#include "long_lived.h"

#include "effort_ledger.h"
#include "heuristic_common.h"
#include "highs_context.h"
#include "io/HighsIO.h"
#include "long_lived_owner.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace heuristics {

namespace {

// Best-first order of the queue and the drain; a NaN objective ranks last.
double rank(const HighsObjective& objective) {
    return std::isnan(objective.value) ? std::numeric_limits<double>::infinity() : objective.value;
}

LongLivedOwner::StartHook& start_hook() {
    static LongLivedOwner::StartHook hook;
    return hook;
}

}  // namespace

SolutionQueue::SolutionQueue() {
    entries_.reserve(kLongLivedQueueCapacity);
}

bool SolutionQueue::push(int epoch, int source, const HighsObjective& objective,
                         const std::vector<double>& x) {
    const double key = rank(objective);
    std::scoped_lock lock(mtx_);
    if (epoch < epoch_) {
        return false;
    }
    if (!callback_sees_all_ && epoch == epoch_ && !may_improve(objective, bound_)) {
        return false;
    }
    if (entries_.size() < kLongLivedQueueCapacity) {
        entries_.push_back({.epoch = epoch, .source = source, .objective = objective, .x = x});
        return true;
    }
    // An entry of an older epoch than the point's goes first, whatever its
    // rank: the next drain drops it anyway, and objectives of two spaces do
    // not compare.
    auto victim =
        std::ranges::find_if(entries_, [epoch](const Entry& e) { return e.epoch < epoch; });
    if (victim == entries_.end()) {
        victim = std::ranges::max_element(entries_, {},
                                          [](const Entry& e) { return rank(e.objective); });
        if (!(key < rank(victim->objective))) {
            return false;
        }
    }
    victim->epoch = epoch;
    victim->source = source;
    victim->objective = objective;
    victim->x.assign(x.begin(), x.end());
    return true;
}

void SolutionQueue::take(std::vector<Entry>& out) {
    out.clear();
    std::scoped_lock lock(mtx_);
    out.swap(entries_);
    // `out`'s old buffer came back empty; keep the queue's capacity so a
    // push never reallocates the entry array.
    entries_.reserve(kLongLivedQueueCapacity);
}

size_t SolutionQueue::publish(int epoch, double bound, bool callback_sees_all) {
    std::scoped_lock lock(mtx_);
    epoch_ = epoch;
    bound_ = bound;
    callback_sees_all_ = callback_sees_all;
    return std::erase_if(entries_, [epoch](const Entry& e) { return e.epoch < epoch; });
}

// A running slot.  Built on the solving thread; `effort` and `end_s` are
// written on the slot's thread and read after the join.
struct LongLivedOwner::Slot {
    Slot(HighsMipSolver& mipsolver, LongLivedOwner& owner, const LongLivedSlotSpec& slot_spec,
         const Deadline& deadline)
        : spec(slot_spec),
          sink(mipsolver, owner, slot_spec.source),
          source(owner.restart_source(slot_spec.policy)),
          exec{.num_workers = 1,
               .base_seed = heuristic_base_seed(mipsolver.options_mip_->random_seed),
               .deadline = deadline,
               .log_options = mipsolver.options_mip_->log_options,
               // No terminator: `terminatorTerminated` writes, and only the
               // solving thread may.
               .terminator = {},
               .stop = &owner.stop_} {}

    LongLivedSlotSpec spec;
    LongLivedSink sink;
    RestartSource source;
    ExecutionContext exec;
    double start_s = 0.0;
    size_t effort = 0;
    double end_s = 0.0;
};

LongLivedOwner::LongLivedOwner(HighsMipSolver& mipsolver)
    : mipsolver_(mipsolver),
      pool_(make_solution_pool(mipsolver.model_->integrality_)),
      pool_epoch_(mipsolver.mipdata_->numRestarts),
      deadline_(deadline_of(mipsolver)) {
    drained_.reserve(kLongLivedQueueCapacity);
    feed_incumbent(kSolutionSourceHeuristic);
    publish();
    if (start_hook()) {
        // A hook that throws after `start` leaves the destructor unrun:
        // join here, or the joinable thread terminates the test binary.
        try {
            start_hook()(mipsolver_, *this);
        } catch (...) {
            stop();
            throw;
        }
    }
}

LongLivedOwner::~LongLivedOwner() {
    stop();
}

void LongLivedOwner::set_start_hook(StartHook hook) {
    start_hook() = std::move(hook);
}

void LongLivedOwner::start(std::function<void(LongLivedOwner&)> body) {
    threads_.emplace_back([this, body = std::move(body)] {
        try {
            body(*this);
        } catch (...) {
            std::scoped_lock lock(error_mtx_);
            if (!error_) {
                error_ = std::current_exception();
            }
        }
    });
}

void LongLivedOwner::start_slots(std::span<const LongLivedSlotSpec> specs) {
    if (specs.empty() || stop_requested() || deadline_.expired() || view_.has_value()) {
        return;
    }
    view_.emplace(make_problem(mipsolver_, csc_));
    // The stop flag on the deadline too, for the sub-algorithms that poll a
    // `Deadline` and not `exec` (recombine's bound-prop arm, FPR's DFS).
    Deadline deadline = deadline_;
    deadline.stop = &stop_;
    const HighsTimer* timer = &mipsolver_.timer_;
    slots_.reserve(specs.size());
    for (const LongLivedSlotSpec& spec : specs) {
        Slot* slot =
            slots_.emplace_back(std::make_unique<Slot>(mipsolver_, *this, spec, deadline)).get();
        slot->start_s = timer->read();
        start([slot, timer, &problem = *view_](LongLivedOwner& /*owner*/) {
            try {
                slot->effort = slot->spec.run(problem, slot->spec.budget, slot->exec,
                                              slot->spec.worker, slot->source, slot->sink);
            } catch (...) {
                slot->end_s = timer->read();
                throw;
            }
            slot->end_s = timer->read();
        });
    }
}

void LongLivedOwner::charge_slots(SlotStop why) {
    if (slots_.empty()) {
        return;
    }
    const double limit = deadline_.timer != nullptr ? deadline_.limit : kHighsInf;
    // What a slot that was still running had to stop for: the first of the
    // time limit and the owner's stop.
    const double trigger = std::min(stop_s_, limit);
    const char* reason = "end";
    if (limit <= stop_s_) {
        reason = "deadline";
    } else if (why == SlotStop::kRestart) {
        reason = "restart";
    }
    EffortLedger ledger(mipsolver_);
    for (const std::unique_ptr<Slot>& slot : slots_) {
        const bool stopped = slot->end_s >= trigger;
        const double join_ms = stopped ? (slot->end_s - trigger) * 1000.0 : 0.0;
        ledger.charge_long_lived(slot->spec.name, slot->effort, slot->sink.accepted() > 0,
                                 slot->start_s, slot->end_s, stopped ? reason : "done",
                                 slot->sink.dispatch_id(), join_ms);
    }
    slots_.clear();
    view_.reset();
    csc_ = CscMatrix{};
}

void LongLivedOwner::stop() {
    if (!stop_requested()) {
        stop_s_ = mipsolver_.timer_.read();
    }
    stop_.store(true, std::memory_order_release);
    for (std::thread& thread : threads_) {
        thread.join();
    }
    threads_.clear();
}

void LongLivedOwner::rethrow() {
    std::exception_ptr error;
    {
        std::scoped_lock lock(error_mtx_);
        error = std::exchange(error_, nullptr);
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void LongLivedOwner::publish() {
    const size_t stale = queue_.publish(mipsolver_.mipdata_->numRestarts, highs_bound(mipsolver_),
                                        callback_sees_all(mipsolver_));
    dropped_.fetch_add(stale, std::memory_order_relaxed);
}

void LongLivedOwner::sync_epoch() {
    const int epoch = mipsolver_.mipdata_->numRestarts;
    if (epoch == pool_epoch_) {
        return;
    }
    pool_ = make_solution_pool(mipsolver_.model_->integrality_);
    pool_epoch_ = epoch;
    fed_bound_ = std::numeric_limits<double>::infinity();
    // The fresh pool holds the incumbent from the start, as the first
    // epoch's does from the constructor: a sink built over an empty one
    // would count every offer as an improvement, and a restart source would
    // find no best member.
    feed_incumbent(kSolutionSourceHeuristic);
}

void LongLivedOwner::feed_incumbent(int origin) {
    // `highs_bound` is +inf while there is no incumbent, so an empty one is
    // never fed; within an epoch the bound only falls.
    const double bound = highs_bound(mipsolver_);
    if (!(bound < fed_bound_)) {
        return;
    }
    fed_bound_ = bound;
    const std::vector<double>& incumbent = mipsolver_.mipdata_->incumbent;
    static_cast<void>(pool_->try_add(pool_objective(mipsolver_, incumbent),
                                     std::make_shared<const std::vector<double>>(incumbent), origin,
                                     SolutionPool::Admission::kAlways));
}

const std::shared_ptr<SolutionPool>& LongLivedOwner::pool() {
    sync_epoch();
    return pool_;
}

RestartSource LongLivedOwner::restart_source(SolutionPool::RestartPolicy policy) {
    return [pool = pool(), policy](Rng& rng, std::vector<double>& out) {
        return pool->get_restart(rng, out, policy);
    };
}

void LongLivedOwner::seed(SolutionSink& chain) {
    sync_epoch();
    // The chain's entries first, so an incumbent the chain found is their
    // twin and keeps its finder's tag.
    chain.entry_refs(refs_);
    for (const SolutionPool::EntryRef& ref : refs_) {
        static_cast<void>(pool_->try_add(ref.objective, ref.solution, ref.source));
    }
    refs_.clear();
    feed_incumbent(kSolutionSourceHeuristic);
}

void LongLivedOwner::log_pool() {
    const HighsLogOptions& log_options = mipsolver_.options_mip_->log_options;
    if (*log_options.log_dev_level < kHighsLogDevLevelVerbose) {
        return;
    }
    sync_epoch();
    pool_->entry_refs(refs_);
    std::string origins;
    for (const SolutionPool::EntryRef& ref : refs_) {
        origins += mipsolver_.mipdata_->solutionSourceToString(ref.source);
    }
    const size_t members = refs_.size();
    const double best = refs_.empty() ? kHighsInf : refs_.front().objective;
    refs_.clear();
    highsLogDev(log_options, HighsLogType::kVerbose,
                "[LongLivedPool] epoch=%d members=%zu best=%.17g origins=%s\n", pool_epoch_,
                members, best, origins.c_str());
}

void LongLivedOwner::drain() {
    sync_epoch();
    feed_incumbent(kSolutionSourceHeuristic);
    // Publishing first drops every entry from before a restart, and the
    // queue refuses older epochs from then on, so what `take` hands over
    // is all of the current epoch.
    publish();
    queue_.take(drained_);
    HighsMipSolverData& mipdata = *mipsolver_.mipdata_;
    const bool sees_all = callback_sees_all(mipsolver_);
    std::ranges::stable_sort(drained_, {},
                             [](const SolutionQueue::Entry& e) { return rank(e.objective); });
    // The tag of the entry that last lowered HiGHS's bound, if one did:
    // installs are serialised here, so that entry is the incumbent.
    int installed = kSolutionSourceNone;
    for (const SolutionQueue::Entry& entry : drained_) {
        // Against the live bound, which the entries before this one may
        // have lowered: best-first, so a later entry rarely gets through.
        const double bound = highs_bound(mipsolver_);
        if (!sees_all && !may_improve(entry.objective, bound)) {
            continue;
        }
        mipdata.trySolution(entry.x, entry.source);
        submitted_.fetch_add(1, std::memory_order_relaxed);
        if (highs_bound(mipsolver_) < bound) {
            installed = entry.source;
        }
    }
    drained_.clear();
    if (installed != kSolutionSourceNone) {
        feed_incumbent(installed);
    }
    publish();
}

LongLivedSink::LongLivedSink(HighsMipSolver& mipsolver, LongLivedOwner& owner, int source)
    : HighsSink(mipsolver, owner.pool(), source),
      queue_(owner.queue()),
      cost_(mipsolver.model_->col_cost_),
      epoch_(mipsolver.mipdata_->numRestarts) {}

void LongLivedSink::on_accept(double objective, const std::vector<double>& solution, int source,
                              const WorkerTrace& trace, size_t effort_at) {
    // A point of another width cannot be this epoch's, and `trySolution`
    // refuses it before `addIncumbent`, so the callback would not see it
    // either: dropping it here loses nothing.
    if (solution.size() == cost_.size()) {
        static_cast<void>(queue_.push(epoch_, source, highs_objective(cost_, solution), solution));
    }
    trace_offer(trace, effort_at, objective);
}

LongLived::LongLived(HighsMipSolver& mipsolver) : mipsolver_(mipsolver) {
    if (mipsolver.submip || !mipsolver.options_mip_->mip_heuristic_long_lived) {
        return;
    }
    owner_ = std::make_unique<LongLivedOwner>(mipsolver);
    mipsolver.long_lived_ = owner_.get();
}

LongLived::~LongLived() {
    mipsolver_.long_lived_ = nullptr;
}

void drain_long_lived(HighsMipSolver& mipsolver) {
    if (mipsolver.long_lived_ != nullptr) {
        mipsolver.long_lived_->drain();
    }
}

void start_long_lived(HighsMipSolver& mipsolver) {
    LongLivedOwner* owner = mipsolver.long_lived_;
    const HighsOptions& options = *mipsolver.options_mip_;
    if (owner == nullptr || options.mip_heuristic_long_lived_threads <= 0) {
        return;
    }
    owner->start_slots(long_lived_slot_specs(options, mipsolver.mipdata_->ARindex_.size()));
}

void restart_long_lived(HighsMipSolver& mipsolver) {
    LongLivedOwner* owner = mipsolver.long_lived_;
    if (owner == nullptr) {
        return;
    }
    owner->stop();
    owner->charge_slots(SlotStop::kRestart);
}

void stop_long_lived(HighsMipSolver& mipsolver) {
    LongLivedOwner* owner = mipsolver.long_lived_;
    if (owner == nullptr) {
        return;
    }
    owner->stop();
    owner->charge_slots(SlotStop::kEnd);
    owner->drain();
    owner->log_pool();
    owner->rethrow();
}

}  // namespace heuristics
