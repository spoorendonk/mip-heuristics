#include "solution_sink.h"

#include "lp_data/HConst.h"

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

std::shared_ptr<SolutionPool> make_solution_pool(const std::vector<HighsVarType>& integrality) {
    auto pool = std::make_shared<SolutionPool>(kPoolCapacity);
    std::vector<bool> int_mask(integrality.size());
    for (size_t j = 0; j < integrality.size(); ++j) {
        int_mask[j] = (integrality[j] != HighsVarType::kContinuous);
    }
    pool->set_integer_mask(std::move(int_mask));
    return pool;
}

SolutionSink::SolutionSink(const std::vector<HighsVarType>& integrality, int source)
    : SolutionSink(make_solution_pool(integrality), source) {}

SolutionSink::SolutionSink(std::shared_ptr<SolutionPool> pool, int source)
    : pool_(std::move(pool)), source_(source) {}

SolutionSink::OfferResult SolutionSink::offer(double objective, const std::vector<double>& solution,
                                              const WorkerTrace& trace, size_t effort_at) {
    const SolutionPool::AddResult added = pool_->try_add(objective, solution, source_);
    // `accepted_` and `on_accept` both stay on the admission verdict: they
    // feed `[Heur] found` and `[HeurSol] accepted`, whose meaning external
    // tooling depends on.  Only the gates read the other flag.
    if (added.accepted) {
        accepted_.fetch_add(1, std::memory_order_relaxed);
        on_accept(objective, solution, source_, trace, effort_at);
    }
    return OfferResult{.accepted = added.accepted, .improved_incumbent = added.improved_best};
}

void SolutionSink::on_accept(double /*objective*/, const std::vector<double>& /*solution*/,
                             int /*source*/, const WorkerTrace& /*trace*/, size_t /*effort_at*/) {}
