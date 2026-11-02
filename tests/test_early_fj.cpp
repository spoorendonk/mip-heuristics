#include "early_heuristics.h"
#include "Highs.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "test_common.h"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <vector>

// ===================================================================
// FeasibilityJump on the original model while presolve runs (#205).
//
// In a solve the worker stops when presolve returns, so what it finds
// depends on timing.  These cases take that race out: they drive
// `HighsMipSolver::run`'s own sequence by hand (init, `EarlyFj`,
// `runMipPresolve`, `finish`, `runSetup`) and let the worker run to its own
// budget with `finish(true)`, then assert the mechanism — what `finish`
// hands to `solution_` and what `runSetup` makes of it.  The full-solve
// cases at the end assert only what holds whatever the race does.
// ===================================================================

namespace {

// lseu: all binary, so FJ finds a point on the original model within its
// first few callbacks.  Not egout or flugpl: on egout's original model
// (equality rows over continuous columns presolve mostly removes) FJ found
// nothing in 3e7 units, on flugpl nothing at all.
constexpr const char* kInstance = "lseu.mps";
constexpr double kOptimum = 1120.0;
// Ten budgets (~3e6 units): at 1.0 the budget (`nnz << 10`, ~3e5 units) is
// under FJ's first callback (5e5 units, #194), a single short window.
constexpr double kEffort = 10.0;

// `HighsMipSolver::run` up to the point the early worker is started, with
// presolve on (unlike `build_bare_mipsolver`): the solution `finish` hands
// over is in original columns and `runSetup` has a real postsolve stack to
// crush it with.
struct Harness {
    Highs highs;
    HighsCallback cb{&highs};
    // `Highs` hands the MIP solver its profiling clocks, and presolve's
    // probing and enumeration start them unconditionally; declared before
    // `mipsolver` so they outlive it.
    HighsTimer profiling_timer;
    HighsProfiling profiling;
    std::unique_ptr<HighsMipSolver> mipsolver;

    explicit Harness(bool maximise = false) {
        // `HighsMipSolverData::init` reads `parallel::num_threads()`, as at
        // every `build_bare_mipsolver` call site.  At one thread, the
        // `threads` option below: a solve later in the same process refuses
        // a scheduler of another width.
        highs::parallel::initialize_scheduler(1);
        highs.setOptionValue("output_flag", false);
        require_option(highs, "threads", 1);
        require_option(highs, "random_seed", 0);
        require_option(highs, "mip_heuristic_early_fj_effort", kEffort);
        REQUIRE(highs.readModel(instances_dir() + "/" + kInstance) == HighsStatus::kOk);
        if (maximise) {
            // The same problem stated as a maximisation of the negated
            // objective, so its optimum is `-kOptimum`.
            std::vector<double> cost = highs.getLp().col_cost_;
            for (double& c : cost) {
                c = -c;
            }
            const auto ncol = static_cast<HighsInt>(cost.size());
            REQUIRE(highs.changeColsCost(0, ncol - 1, cost.data()) == HighsStatus::kOk);
            REQUIRE(highs.changeObjectiveSense(ObjSense::kMaximize) == HighsStatus::kOk);
        }
        mipsolver = std::make_unique<HighsMipSolver>(cb, highs.getOptions(), highs.getLp(),
                                                     highs.getSolution());
        profiling.multi_threaded = false;
        profiling.initialize(profiling_timer, /*subsolver_profiling=*/false);
        mipsolver->setProfiling(&profiling);
        mipsolver->timer_.start();
        mipsolver->improving_solution_file_ = nullptr;
        mipsolver->mipdata_ = std::make_unique<HighsMipSolverData>(*mipsolver);
        mipsolver->mipdata_->init();
        mipsolver->modelstatus_ = HighsModelStatus::kNotset;
    }

    void presolve() const {
        mipsolver->mipdata_->runMipPresolve(mipsolver->options_mip_->presolve_reduction_limit);
    }

    // Whether `solution_` is feasible by the test `cleanupSolve` applies.
    [[nodiscard]] bool solution_feasible() const {
        const double tol = mipsolver->options_mip_->mip_feasibility_tolerance;
        return mipsolver->solution_objective_ < kHighsInf && mipsolver->bound_violation_ <= tol &&
               mipsolver->row_violation_ <= tol && mipsolver->integrality_violation_ <= tol;
    }
};

}  // namespace

TEST_CASE("early FJ: the option exists and is off by default", "[early-fj]") {
    Highs h;
    h.setOptionValue("output_flag", false);
    double effort = -1.0;
    REQUIRE(h.getOptionValue("mip_heuristic_early_fj_effort", effort) == HighsStatus::kOk);
    REQUIRE(effort == 0.0);
}

TEST_CASE("early FJ: its solution becomes the MIP start presolve crushes", "[early-fj]") {
    Harness harness;
    HighsMipSolver& ms = *harness.mipsolver;
    heuristics::EarlyFj early(ms);
    harness.presolve();
    REQUIRE(ms.modelstatus_ == HighsModelStatus::kNotset);
    early.finish(true);

    // Handed over in original space: original columns, feasible there by
    // HiGHS's own test, and no better than the optimum.
    REQUIRE(harness.solution_feasible());
    REQUIRE(ms.solution_.size() == static_cast<size_t>(ms.orig_model_->num_col_));
    REQUIRE(ms.solution_objective_ >= kOptimum - 1e-3);
    const std::vector<double> handed_over = ms.solution_;
    const double objective = ms.solution_objective_;

    // `runSetup` takes it as a MIP start: crushed into the presolved
    // columns as the incumbent, and its objective as the upper bound in the
    // presolved space.
    ms.mipdata_->runSetup();
    REQUIRE(ms.mipdata_->incumbent.size() == static_cast<size_t>(ms.model_->num_col_));
    REQUIRE(ms.mipdata_->upper_bound == Catch::Approx(objective - ms.model_->offset_));
    // And the original-space copy is untouched: it is the answer
    // `cleanupSolve` reports unless the search finds better.
    REQUIRE(ms.solution_ == handed_over);
    REQUIRE(ms.solution_objective_ == objective);
}

TEST_CASE("early FJ: a maximisation hands over the objective in its own sense", "[early-fj]") {
    Harness harness(/*maximise=*/true);
    HighsMipSolver& ms = *harness.mipsolver;
    heuristics::EarlyFj early(ms);
    harness.presolve();
    early.finish(true);

    REQUIRE(harness.solution_feasible());
    // `solution_objective_` is in the model's own sense, so a maximisation
    // whose optimum is `-kOptimum` reports at most that.
    REQUIRE(ms.solution_objective_ <= -kOptimum + 1e-3);
    const double objective = ms.solution_objective_;
    ms.mipdata_->runSetup();
    // The presolved model is a minimisation: the bound is the negation.
    REQUIRE(ms.mipdata_->upper_bound == Catch::Approx(-objective - ms.model_->offset_));
}

TEST_CASE("early FJ: a better feasible solution already held is kept", "[early-fj]") {
    Harness harness;
    HighsMipSolver& ms = *harness.mipsolver;
    // A stand-in for a user's MIP start, better than any feasible point.
    const std::vector<double> marker(static_cast<size_t>(ms.orig_model_->num_col_), 7.0);
    ms.solution_ = marker;
    ms.solution_objective_ = kOptimum - 1.0;
    ms.bound_violation_ = 0.0;
    ms.row_violation_ = 0.0;
    ms.integrality_violation_ = 0.0;

    heuristics::EarlyFj early(ms);
    harness.presolve();
    early.finish(true);

    REQUIRE(ms.solution_ == marker);
    REQUIRE(ms.solution_objective_ == kOptimum - 1.0);
}

TEST_CASE("early FJ: an infeasible solution already held is replaced", "[early-fj]") {
    Harness harness;
    HighsMipSolver& ms = *harness.mipsolver;
    // An infeasible MIP start with an objective no feasible point reaches:
    // the objective alone must not keep it.
    ms.solution_.assign(static_cast<size_t>(ms.orig_model_->num_col_), 7.0);
    ms.solution_objective_ = kOptimum - 1.0;
    ms.bound_violation_ = 1.0;
    ms.row_violation_ = 0.0;
    ms.integrality_violation_ = 0.0;

    heuristics::EarlyFj early(ms);
    harness.presolve();
    early.finish(true);

    REQUIRE(harness.solution_feasible());
    REQUIRE(ms.solution_objective_ >= kOptimum - 1e-3);
}

TEST_CASE("early FJ: presolve's own verdict keeps what presolve concluded", "[early-fj]") {
    Harness harness;
    HighsMipSolver& ms = *harness.mipsolver;
    heuristics::EarlyFj early(ms);
    harness.presolve();
    // As if presolve had decided the model: nothing is handed over.
    ms.modelstatus_ = HighsModelStatus::kInfeasible;
    early.finish(true);

    REQUIRE(ms.solution_.empty());
    REQUIRE(ms.solution_objective_ == kHighsInf);
}

TEST_CASE("early FJ: a presolve that ran out of time still reports the solution", "[early-fj]") {
    Harness harness;
    HighsMipSolver& ms = *harness.mipsolver;
    heuristics::EarlyFj early(ms);
    harness.presolve();
    // As if the time limit had expired in presolve: `runSetup` never runs,
    // so `finish` sets the bound `cleanupSolve` reports (`upper_bound` plus
    // the presolved model's offset) itself.
    ms.modelstatus_ = HighsModelStatus::kTimeLimit;
    early.finish(true);

    REQUIRE(harness.solution_feasible());
    REQUIRE(ms.mipdata_->upper_bound + ms.model_->offset_ == Catch::Approx(ms.solution_objective_));
}

TEST_CASE("early FJ: never in a sub-MIP", "[early-fj]") {
    Harness harness;
    HighsMipSolver& ms = *harness.mipsolver;
    ms.submip = true;
    heuristics::EarlyFj early(ms);
    harness.presolve();
    early.finish(true);

    REQUIRE(ms.solution_.empty());
    REQUIRE(ms.solution_objective_ == kHighsInf);
}

TEST_CASE("early FJ: a solve books it only when the option is on", "[early-fj]") {
    // Whatever the race with presolve does, the worker is started and
    // joined, so the `[Heur]` line is there — with effort 0 if presolve
    // won outright.
    auto run = [](double effort) {
        return solve_capturing_log(kInstance, [effort](Highs& h) {
            require_option(h, "threads", 1);
            require_option(h, "random_seed", 0);
            require_option(h, "log_dev_level", 3);
            require_option(h, "mip_heuristic_early_fj_effort", effort);
        });
    };
    Highs original;
    original.setOptionValue("output_flag", false);
    REQUIRE(original.readModel(instances_dir() + "/" + kInstance) == HighsStatus::kOk);
    const HighsInt original_nnz = original.getLp().a_matrix_.numNz();
    const auto lines = run(1.0);
    REQUIRE(log_contains(lines, "[Heur] name=early_fj "));
    // The original model's nnz, the unit its budget is in, not the
    // presolved matrix the chain's lines carry.
    const std::string nnz_field = " nnz=" + std::to_string(original_nnz) + " ";
    REQUIRE(std::ranges::any_of(lines, [&](const std::string& line) {
        return line.contains("[Heur] name=early_fj ") && line.contains(nnz_field);
    }));
    REQUIRE_FALSE(log_contains(run(0.0), "[Heur] name=early_fj "));
}

TEST_CASE("early FJ: a solve with it on still reaches the optimum", "[early-fj]") {
    double objective = 0.0;
    solve_capturing_log(
        kInstance,
        [](Highs& h) {
            require_option(h, "threads", 1);
            require_option(h, "random_seed", 0);
            require_option(h, "mip_rel_gap", 0.0);
            require_option(h, "mip_heuristic_early_fj_effort", 1.0);
        },
        [&](Highs& h) {
            REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
            REQUIRE(h.getInfoValue("objective_function_value", objective) == HighsStatus::kOk);
        });
    REQUIRE(objective == Catch::Approx(kOptimum).epsilon(1e-4));
}
