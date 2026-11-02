// The objective cut's shared plumbing (#208, #221): the options, the cutoff
// formula, the zero-objective guard and the sink's best objective.  What
// each heuristic does with the cutoff is tested with that heuristic.

#include "heuristic_context.h"
#include "Highs.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolverData.h"  // for kSolutionSource* constants
#include "solution_sink.h"
#include "test_common.h"
#include "worker_base.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <utility>
#include <vector>

namespace {

// min c.x  s.t.  x0 + x1 >= 1,  x binary.
HighsLp two_binaries(std::vector<double> cost) {
    HighsLp lp;
    lp.num_col_ = 2;
    lp.num_row_ = 1;
    lp.col_cost_ = std::move(cost);
    lp.col_lower_ = {0.0, 0.0};
    lp.col_upper_ = {1.0, 1.0};
    lp.row_lower_ = {1.0};
    lp.row_upper_ = {kHighsInf};
    lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_ = {0, 1, 2};
    lp.a_matrix_.index_ = {0, 0};
    lp.a_matrix_.value_ = {1.0, 1.0};
    return lp;
}

constexpr double kEpsilon = 1e-9;

bool view_has_objective(std::vector<double> cost) {
    const HighsLp lp = two_binaries(std::move(cost));
    ProblemStorage storage;
    const auto made = make_problem(lp, storage, /*feastol=*/1e-6, kEpsilon);
    REQUIRE(made.has_value());
    return has_objective(*made);
}

}  // namespace

TEST_CASE("Objective cut: options registered with their defaults", "[objective_cut]") {
    Highs h;
    bool on = true;
    REQUIRE(h.getOptionValue("mip_heuristic_objective_cut", on) == HighsStatus::kOk);
    REQUIRE_FALSE(on);
    double rel = 0.0;
    REQUIRE(h.getOptionValue("mip_heuristic_objective_cut_rel", rel) == HighsStatus::kOk);
    REQUIRE(rel == 1e-3);
    double abs = 0.0;
    REQUIRE(h.getOptionValue("mip_heuristic_objective_cut_abs", abs) == HighsStatus::kOk);
    REQUIRE(abs == 1e-7);
    // Settable within their bounds, refused below zero.
    require_option(h, "mip_heuristic_objective_cut", true);
    require_option(h, "mip_heuristic_objective_cut_rel", 0.01);
    require_option(h, "mip_heuristic_objective_cut_abs", 1.0);
    REQUIRE(h.setOptionValue("mip_heuristic_objective_cut_rel", -1.0) != HighsStatus::kOk);
    REQUIRE(h.setOptionValue("mip_heuristic_objective_cut_rel", 1.5) != HighsStatus::kOk);
    REQUIRE(h.setOptionValue("mip_heuristic_objective_cut_abs", -1.0) != HighsStatus::kOk);
    REQUIRE(h.setOptionValue("mip_heuristic_objective_cut_abs", 1e21) != HighsStatus::kOk);
}

TEST_CASE("Objective cut: target is best - max(rel*|best|, abs)", "[objective_cut]") {
    const ObjectiveCutParams params{.on = true, .rel = 1e-3, .abs = 1e-7};
    // The relative step dominates away from zero, on either sign.
    REQUIRE(objective_cut_target(1000.0, params) == Catch::Approx(999.0));
    REQUIRE(objective_cut_target(-1000.0, params) == Catch::Approx(-1001.0));
    // The absolute step is the floor near zero.
    REQUIRE(objective_cut_target(0.0, params) == -1e-7);
    REQUIRE(objective_cut_target(1e-6, params) == Catch::Approx(1e-6 - 1e-7));
    // A larger absolute step takes over where it exceeds the relative one.
    const ObjectiveCutParams coarse{.on = true, .rel = 1e-3, .abs = 5.0};
    REQUIRE(objective_cut_target(1000.0, coarse) == Catch::Approx(995.0));
    REQUIRE(objective_cut_target(10000.0, coarse) == Catch::Approx(9990.0));
    // An empty pool (`best_objective()` infinite) cuts nothing.
    REQUIRE(objective_cut_target(kHighsInf, params) == kHighsInf);
    // The cut off cuts nothing, whatever the steps say.
    const ObjectiveCutParams off{.on = false, .rel = 1e-3, .abs = 1e-7};
    REQUIRE(objective_cut_target(1000.0, off) == kHighsInf);
}

TEST_CASE("Objective cut: a zero objective disables the cut", "[objective_cut]") {
    REQUIRE_FALSE(view_has_objective({0.0, 0.0}));
    // Costs at or below the matrix epsilon count as zero.
    REQUIRE_FALSE(view_has_objective({kEpsilon, -kEpsilon}));
    REQUIRE(view_has_objective({0.0, 1.0}));
    REQUIRE(view_has_objective({-2.0, 0.0}));
}

TEST_CASE("Objective cut: sink best_objective follows accepts", "[objective_cut]") {
    const std::vector<HighsVarType> integrality{HighsVarType::kInteger};
    SolutionSink sink(integrality, kSolutionSourceFPR);
    REQUIRE(sink.best_objective() == kHighsInf);
    REQUIRE(sink.offer(5.0, {5.0}, WorkerTrace{}, 0).improved_incumbent);
    REQUIRE(sink.best_objective() == 5.0);
    REQUIRE(sink.offer(3.0, {3.0}, WorkerTrace{}, 0).improved_incumbent);
    REQUIRE(sink.best_objective() == 3.0);
    // A worse offer, accepted or not, leaves the best where it is.
    static_cast<void>(sink.offer(4.0, {4.0}, WorkerTrace{}, 0));
    REQUIRE(sink.best_objective() == 3.0);
}

TEST_CASE("Objective cut: option on leaves the solve unchanged", "[objective_cut]") {
    // No heuristic reads the cut yet (#222-#225), so a solve with it on
    // must still reach the optimum; each heuristic's own tests pin what it
    // does with the cut.
    Highs h;
    require_option(h, "output_flag", false);
    require_option(h, "mip_heuristic_objective_cut", true);
    REQUIRE(h.readModel(instances_dir() + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj = 0.0;
    REQUIRE(h.getInfoValue("objective_function_value", obj) == HighsStatus::kOk);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}
