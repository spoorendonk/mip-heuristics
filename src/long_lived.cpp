#include "long_lived.h"

#include "highs_context.h"
#include "long_lived_owner.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <limits>
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
    auto worst =
        std::ranges::max_element(entries_, {}, [](const Entry& e) { return rank(e.objective); });
    if (!(key < rank(worst->objective))) {
        return false;
    }
    worst->epoch = epoch;
    worst->source = source;
    worst->objective = objective;
    worst->x.assign(x.begin(), x.end());
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

LongLivedOwner::LongLivedOwner(HighsMipSolver& mipsolver)
    : mipsolver_(mipsolver), deadline_(deadline_of(mipsolver)) {
    drained_.reserve(kLongLivedQueueCapacity);
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

void LongLivedOwner::stop() {
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

void LongLivedOwner::drain() {
    // Publishing first drops every entry from before a restart, and the
    // queue refuses older epochs from then on, so what `take` hands over
    // is all of the current epoch.
    publish();
    queue_.take(drained_);
    HighsMipSolverData& mipdata = *mipsolver_.mipdata_;
    const bool sees_all = callback_sees_all(mipsolver_);
    std::ranges::stable_sort(drained_, {},
                             [](const SolutionQueue::Entry& e) { return rank(e.objective); });
    for (const SolutionQueue::Entry& entry : drained_) {
        // Against the live bound, which the entries before this one may
        // have lowered: best-first, so a later entry rarely gets through.
        if (!sees_all && !may_improve(entry.objective, highs_bound(mipsolver_))) {
            continue;
        }
        mipdata.trySolution(entry.x, entry.source);
        submitted_.fetch_add(1, std::memory_order_relaxed);
    }
    drained_.clear();
    publish();
}

LongLivedSink::LongLivedSink(HighsMipSolver& mipsolver, LongLivedOwner& owner, int source)
    : HighsSink(mipsolver, source),
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

void stop_long_lived(HighsMipSolver& mipsolver) {
    LongLivedOwner* owner = mipsolver.long_lived_;
    if (owner == nullptr) {
        return;
    }
    owner->stop();
    owner->drain();
    owner->rethrow();
}

}  // namespace heuristics
