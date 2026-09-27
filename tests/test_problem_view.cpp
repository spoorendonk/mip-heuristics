// The core `make_problem` against the HiGHS adapter's view of the same model
// (#170).  With presolve off, the solver's model is the one `readModel`
// returned, and `runSetup` builds its row-wise copy, lock counts and root
// domain from it — so the two builders must agree on everything they both
// derive: the row-wise matrix, the locks, the column bounds and the binary
// mask.  A mis-transposed matrix, swapped locks or a loosened binary test in
// the core builder shows up here, where no worker outcome would expose it.
//
// On the instances below the agreement is exact.  Elsewhere two legitimate
// differences intervene and the instance is left out rather than the check
// loosened: HiGHS still scales a MIP with presolve off (`scaleMIP`, so its
// coefficients and costs differ: flugpl, bell5, dcmulti, small_mip), and
// `runSetup`'s root propagation tightens the domain the adapter's binary
// mask is read from (p0548, egout, bell5).

#include "heuristic_context.h"
#include "Highs.h"
#include "highs_context.h"
#include "mip/HighsMipSolver.h"
#include "parallel/HighsParallel.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

TEST_CASE("problem view: the core builder matches the adapter's view of the model",
          "[problem-view][core]") {
    highs::parallel::initialize_scheduler();
    for (const char* instance : {"lseu.mps", "gt2.mps", "rgn.mps", "gesa2.mps", "p01.mps"}) {
        INFO(instance);
        Highs solver_side;
        solver_side.setOptionValue("output_flag", false);
        HighsCallback cb(&solver_side);
        auto mipsolver = build_bare_mipsolver(solver_side, cb, instance);
        CscMatrix csc;
        const ProblemView adapter = make_problem(*mipsolver, csc);

        Highs core_side;
        core_side.setOptionValue("output_flag", false);
        REQUIRE(core_side.readModel(kInstancesDir + "/" + instance) == HighsStatus::kOk);
        const HighsOptions& options = core_side.getOptions();
        ProblemStorage storage;
        const auto made =
            make_problem(core_side.getLp(), storage, options.mip_feasibility_tolerance,
                         options.small_matrix_value);
        REQUIRE(made.has_value());
        const ProblemView& core = *made;

        REQUIRE(core.ncol == adapter.ncol);
        REQUIRE(core.nrow == adapter.nrow);
        CHECK(*core.ar_start == *adapter.ar_start);
        CHECK(*core.ar_index == *adapter.ar_index);
        CHECK(*core.ar_value == *adapter.ar_value);
        CHECK(*core.uplocks == *adapter.uplocks);
        CHECK(*core.downlocks == *adapter.downlocks);
        CHECK(*core.col_lower == *adapter.col_lower);
        CHECK(*core.col_upper == *adapter.col_upper);
        CHECK(core.binary == adapter.binary);
        CHECK(*core.col_cost == *adapter.col_cost);
        CHECK(core.offset == adapter.offset);
        CHECK(core.feastol == adapter.feastol);
        CHECK(core.epsilon == adapter.epsilon);
    }
}
