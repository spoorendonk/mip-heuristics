// FPR under the objective cut (#208, #232): with a solution in the pool at
// dispatch, every attempt propagates the objective row bounded by the pool's
// cutoff, so nothing FPR offers lies above it.  One worker, so offers and
// attempt starts are ordered; no clock.  The assertions compare runs on the
// same platform, never a count fixed in advance.

#include "fpr.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "highs_context.h"
#include "parallel/HighsParallel.h"
#include "solution_sink.h"
#include "test_common.h"
#include "worker_base.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <mutex>
#include <vector>

namespace {

constexpr ObjectiveCutParams kCutOn{.on = true, .rel = 1e-3, .abs = 1e-7};
constexpr size_t kBudget = size_t{1} << 22;
// The objective row is checked to `mip_feasibility_tolerance` on its
// activity, as every row is.
constexpr double kRowTol = 1e-6;

// Every accepted offer, in order.
class RecordingSink : public SolutionSink {
public:
    using SolutionSink::SolutionSink;
    std::mutex mtx;
    std::vector<double> objectives;
    std::vector<std::vector<double>> solutions;

private:
    void on_accept(double obj, const std::vector<double>& x, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t /*effort_at*/) override {
        std::scoped_lock guard(mtx);
        objectives.push_back(obj);
        solutions.push_back(x);
    }
};

struct Run {
    std::vector<double> objectives;  // accepted offers, the seed's first if any
    std::vector<std::vector<double>> solutions;
    size_t effort = 0;
};

// One FPR dispatch on a fresh solver, one worker, the pool seeded with
// `seed` when it is non-empty.
Run run_fpr(ObjectiveCutParams cut, const std::vector<double>& seed, double seed_obj) {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "egout.mps");

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    ExecutionContext exec = make_exec(*mipsolver);
    exec.num_workers = 1;
    RecordingSink sink(*problem.integrality, kSolutionSourceHeuristic);
    if (!seed.empty()) {
        REQUIRE(sink.offer(seed_obj, seed, WorkerTrace{0, 0}, 0).accepted);
    }
    HeuristicBudget budget = make_budget(kBudget, exec.num_workers, kBudget);
    budget.objective_cut = cut;

    const DispatchOutcome out = fpr::run(*mipsolver, problem, budget, exec, sink);
    return {sink.objectives, sink.solutions, out.effort};
}

}  // namespace

TEST_CASE("FPR objective cut: no offer above the cutoff in force", "[objective-cut][fpr]") {
    const Run plain = run_fpr({}, {}, 0.0);
    REQUIRE(!plain.objectives.empty());

    // Seed with the worst point the uncut run accepted, so the cut leaves
    // room below it.
    const auto worst = std::ranges::max_element(plain.objectives) - plain.objectives.begin();
    const double seed_obj = plain.objectives[worst];
    const Run cut = run_fpr(kCutOn, plain.solutions[worst], seed_obj);
    REQUIRE(cut.effort > 0);
    REQUIRE(cut.objectives.front() == seed_obj);
    REQUIRE(cut.objectives.size() > 1);  // the loop below checks something

    // One worker: an attempt starts after the previous one's offer, so the
    // cutoff in force for offer k is the target of the best accepted before it.
    double best = seed_obj;
    for (size_t k = 1; k < cut.objectives.size(); ++k) {
        const double cutoff = objective_cut_target(best, kCutOn);
        INFO("offer " << k << ": objective " << cut.objectives[k] << ", cutoff " << cutoff);
        REQUIRE(cut.objectives[k] <= cutoff + kRowTol);
        best = std::min(best, cut.objectives[k]);
    }
}

TEST_CASE("FPR objective cut: off, or on with an empty pool, FPR is unchanged",
          "[objective-cut][fpr]") {
    const Run off = run_fpr({}, {}, 0.0);
    // With the pool empty at dispatch no objective row is built.
    const Run on_empty = run_fpr(kCutOn, {}, 0.0);
    REQUIRE(on_empty.objectives == off.objectives);
    REQUIRE(on_empty.effort == off.effort);
}
