// FJ's CDCL lane (#237): which slot takes it, and its one attempt on a model
// the caller owns.  Core only.

#include "core_test_support.h"
#include "fj.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "test_paths.h"
#include "worker_base.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <vector>

using namespace core_test;

namespace {

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/" + instance) == HighsStatus::kOk);
}

constexpr size_t kLaneEffort = size_t{1} << 32;

}  // namespace

TEST_CASE("CDCL lane: the last slot, and only with two or more", "[core][fj_cdcl]") {
    CHECK_FALSE(fj::is_cdcl_slot(0, 1));
    CHECK_FALSE(fj::is_cdcl_slot(0, 2));
    CHECK(fj::is_cdcl_slot(1, 2));
    CHECK_FALSE(fj::is_cdcl_slot(2, 4));
    CHECK(fj::is_cdcl_slot(3, 4));
}

TEST_CASE("CDCL lane: offers a feasible point on a pure-binary model, then stands down",
          "[core][fj_cdcl]") {
    Highs highs;
    read(highs, "p0548.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    BestSink sink(*s.problem.integrality, 0);
    const WorkerTrace trace{1, 0};

    const AttemptResult first =
        fj::run_cdcl_lane(s.problem, kLaneEffort, s.exec, sink, trace, /*seed=*/0);
    CHECK(first.effort > 0);
    CHECK(first.found_improvement);
    REQUIRE(sink.found);
    CHECK(is_feasible(lp, sink.solution));

    // A feasible point is known now, so the lane does nothing at all.
    const AttemptResult second =
        fj::run_cdcl_lane(s.problem, kLaneEffort, s.exec, sink, trace, /*seed=*/0);
    CHECK(second.effort == 0);
    CHECK_FALSE(second.found_improvement);
}

TEST_CASE("CDCL lane: a model that is not pure binary costs nothing", "[core][fj_cdcl]") {
    // egout has continuous columns.
    Highs highs;
    read(highs, "egout.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    BestSink sink(*s.problem.integrality, 0);
    const AttemptResult lane =
        fj::run_cdcl_lane(s.problem, kLaneEffort, s.exec, sink, WorkerTrace{1, 0}, /*seed=*/0);
    CHECK(lane.effort == 0);
    CHECK_FALSE(sink.found);
}

TEST_CASE("CDCL lane: stops within its effort cap", "[core][fj_cdcl]") {
    Highs highs;
    read(highs, "p0548.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    BestSink sink(*s.problem.integrality, 0);
    constexpr size_t kCap = 1000;
    const AttemptResult lane =
        fj::run_cdcl_lane(s.problem, kCap, s.exec, sink, WorkerTrace{1, 0}, /*seed=*/0);
    // The encoder and the solver both stop at their next poll past the cap.
    CHECK(lane.effort > 0);
    CHECK_FALSE(sink.found);
}
