#include "solution_sink.h"

#include "lp_data/HConst.h"

#include <atomic>
#include <utility>
#include <vector>

SolutionSink::SolutionSink(const std::vector<HighsVarType>& integrality, int source)
    : pool_(kPoolCapacity), source_(source) {
    std::vector<bool> int_mask(integrality.size());
    for (size_t j = 0; j < integrality.size(); ++j) {
        int_mask[j] = (integrality[j] != HighsVarType::kContinuous);
    }
    pool_.set_integer_mask(std::move(int_mask));
}

SolutionSink::OfferResult SolutionSink::offer(double objective, const std::vector<double>& solution,
                                              const WorkerTrace& trace, size_t effort_at) {
    const SolutionPool::AddResult added = pool_.try_add(objective, solution, source_);
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
