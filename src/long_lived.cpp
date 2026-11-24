#include "long_lived.h"

#include "highs_context.h"
#include "io/HighsIO.h"
#include "long_lived_owner.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
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

void LongLivedOwner::sync_epoch() {
    const int epoch = mipsolver_.mipdata_->numRestarts;
    if (epoch == pool_epoch_) {
        return;
    }
    pool_ = make_solution_pool(mipsolver_.model_->integrality_);
    pool_epoch_ = epoch;
    fed_bound_ = std::numeric_limits<double>::infinity();
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

void stop_long_lived(HighsMipSolver& mipsolver) {
    LongLivedOwner* owner = mipsolver.long_lived_;
    if (owner == nullptr) {
        return;
    }
    owner->stop();
    owner->drain();
    owner->log_pool();
    owner->rethrow();
}

}  // namespace heuristics
