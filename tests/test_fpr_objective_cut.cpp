// FPR under the objective cut (#208, #232): with a solution in the pool at
// dispatch, every attempt propagates the objective row bounded by the pool's
// cutoff, so nothing FPR offers lies above it.  One worker, so offers and
// attempt starts are ordered; no clock.  The assertions compare runs on the
// same platform, never a count fixed in advance.

#include "fpr.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "highs_context.h"
#include "objective_cut_test_util.h"
#include "parallel/HighsParallel.h"
#include "solution_sink.h"
#include "test_common.h"
#include "worker_base.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <vector>

namespace {

constexpr size_t kBudget = size_t{1} << 22;

struct Run {
    std::vector<double> objectives;  // accepted offers, the seed's first if any
    std::vector<std::vector<double>> solutions;
    size_t effort = 0;
};

// One FPR dispatch on a fresh solver, one worker, the pool seeded with
// `seed` when it is non-empty (with `inflate(seed)` when `inflated`).
Run run_fpr(ObjectiveCutParams cut, const std::vector<double>& seed, bool inflated = false) {
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
        const std::vector<double> x = inflated ? inflate(problem, seed) : seed;
        REQUIRE(!x.empty());
        REQUIRE(sink.offer(problem.objective(x), x, WorkerTrace{0, 0}, 0).accepted);
    }
    HeuristicBudget budget = make_budget(kBudget, exec.num_workers, kBudget);
    budget.objective_cut = cut;

    const DispatchOutcome out = fpr::run(*mipsolver, problem, budget, exec, sink);
    return {sink.objectives, sink.solutions, out.effort};
}

}  // namespace

TEST_CASE("FPR objective cut: no offer above the cutoff in force", "[objective-cut][fpr]") {
    const Run plain = run_fpr({}, {});
    REQUIRE(!plain.objectives.empty());

    // Seed with a point far worse than the uncut run's first, so the cutoff
    // leaves room for a find.  A find is expected, not implied: the row still
    // propagates and moves FPR's lock-driven choices, and the uncut run's
    // first point (from platform-specific random streams) may already sit at
    // the inflated one.  The per-offer bound below holds whatever the count,
    // so a platform where either precondition fails skips, never fails.
    const Run cut = run_fpr(kCutOn, plain.solutions.front(), /*inflated=*/true);
    REQUIRE(cut.effort > 0);
    const double seed_obj = cut.objectives.front();
    if (objective_cut_target(seed_obj, kCutOn) <= plain.objectives.front() ||
        cut.objectives.size() < 2) {
        SKIP("no room below the seed, or no find under the cut, on this platform");
    }

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
    // Cut off is main by construction: with no objective row built
    // (`fpr::run` passes the workers none), the worker searches the base
    // view on main's path.  With the pool empty at dispatch no row is built
    // either, so the cut on must match it.
    const Run off = run_fpr({}, {});
    const Run on_empty = run_fpr(kCutOn, {});
    REQUIRE(on_empty.objectives == off.objectives);
    REQUIRE(on_empty.effort == off.effort);
}
