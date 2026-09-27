// The heuristic core on its own (#170): FJ and LocalMIP run on a model the
// caller owns — read with `Highs::readModel`, with no presolve and no
// `HighsMipSolver` anywhere — through `mip_heuristics_core` alone.  This
// file is `mip_heuristics_core_tests`, which links that target and nothing
// else of ours and includes only core headers.  The hand-built models are
// in test_core_standalone.cpp, which links against a libhighs without the
// MIP solver.

#include "core_test_support.h"
#include "heuristic_context.h"
#include "Highs.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace core_test;

namespace {

const std::string kInstancesDir = INSTANCES_DIR;

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(kInstancesDir + "/" + instance) == HighsStatus::kOk);
}

}  // namespace

TEST_CASE("core: the view over an original model matches the model", "[core]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    REQUIRE(s.problem.model == &lp);
    REQUIRE(s.problem.ncol == lp.num_col_);
    REQUIRE(s.problem.nrow == lp.num_row_);
    REQUIRE(s.problem.nnz == lp.a_matrix_.index_.size());
    REQUIRE(s.problem.ar_start->size() == static_cast<size_t>(lp.num_row_) + 1);
    REQUIRE(s.problem.csc->col_start.size() == static_cast<size_t>(lp.num_col_) + 1);
    REQUIRE(s.problem.uplocks->size() == static_cast<size_t>(lp.num_col_));
    REQUIRE(s.problem.binary.size() == static_cast<size_t>(lp.num_col_));
    REQUIRE(s.problem.incumbent.empty());
}

TEST_CASE("core: FJ and LocalMIP find feasible solutions on an original model", "[core]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    Found found(lp);
    run_both(s, found);

    for (BestSink* sink : {&found.fj, &found.local_mip}) {
        REQUIRE(sink->found);
        REQUIRE(is_feasible(lp, sink->solution));
        REQUIRE(sink->objective == Catch::Approx(original_objective(lp, sink->solution)));
    }
}
