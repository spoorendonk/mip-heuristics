// The core on a libhighs without the MIP solver (#170).
//
// `mip_heuristics_core_standalone_tests` links `libmip_heuristics_core.a`
// against a copy of `libhighs.a` with every member built from
// `highs/mip/*.cpp` and every adapter object removed.  It links at all only
// if nothing the core needs lives in the MIP solver or the adapter, which is
// the `core_boundary` claim checked by the linker; and these cases then run
// FJ and LocalMIP on it.  Nothing here may use the `Highs` class, whose own
// `run` reaches the MIP solver: the model is built by hand.

#include "core_test_support.h"
#include "heuristic_context.h"
#include "io/HighsIO.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace core_test;

namespace {

// `HighsLogOptions`' own defaults are null pointers the logger dereferences.
struct QuietLog {
    bool output_flag = false;
    bool log_to_console = false;
    HighsInt log_dev_level = 0;
    HighsLogOptions options;
    QuietLog() {
        options.output_flag = &output_flag;
        options.log_to_console = &log_to_console;
        options.log_dev_level = &log_dev_level;
    }
};

}  // namespace

TEST_CASE("core: FJ and LocalMIP find feasible solutions on a hand-built model", "[core]") {
    const QuietLog log;
    const HighsLp lp = small_model();
    const Setup s(lp, log.options);
    Found found(lp);
    run_both(s, found);

    for (BestSink* sink : {&found.fj, &found.local_mip}) {
        REQUIRE(sink->found);
        REQUIRE(is_feasible(lp, sink->solution));
        REQUIRE(sink->objective == Catch::Approx(original_objective(lp, sink->solution)));
    }
}
