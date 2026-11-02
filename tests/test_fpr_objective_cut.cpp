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
#include <cmath>
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

// `x` with every column whose cost is positive raised to its upper bound,
// when the result stays feasible: a point far worse than `x`, so a cutoff
// below it is loose for anything FPR finds unseeded.  Empty when it is not
// feasible (on egout, a fixed-charge network, raising a charge keeps it so).
std::vector<double> inflate(const ProblemView& problem, std::vector<double> x) {
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        if ((*problem.col_cost)[j] > 0.0 && std::isfinite((*problem.col_upper)[j])) {
            x[j] = (*problem.col_upper)[j];
        }
    }
    for (HighsInt i = 0; i < problem.nrow; ++i) {
        double act = 0.0;
        for (HighsInt k = (*problem.ar_start)[i]; k < (*problem.ar_start)[i + 1]; ++k) {
            act += (*problem.ar_value)[k] * x[(*problem.ar_index)[k]];
        }
        if (act > (*problem.row_upper)[i] + kRowTol || act < (*problem.row_lower)[i] - kRowTol) {
            return {};
        }
    }
    return x;
}

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

    // Seed with a point far worse than the uncut run's first.  Its cutoff
    // is loose for that point, and a row that bounds nothing tightens
    // nothing, so the cut run searches as the uncut run did until it reaches
    // that point, and offers it: the loop checks something on any platform,
    // whatever FPR's random streams do there.
    const Run cut = run_fpr(kCutOn, plain.solutions.front(), /*inflated=*/true);
    REQUIRE(cut.effort > 0);
    const double seed_obj = cut.objectives.front();
    REQUIRE(objective_cut_target(seed_obj, kCutOn) > plain.objectives.front());
    REQUIRE(cut.objectives.size() > 1);

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
