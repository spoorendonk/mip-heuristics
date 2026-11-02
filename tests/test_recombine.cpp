// The bound-prop recombiner (#227) in the heuristic core, on a model the
// caller owns: its adaptive limit, the cases with no offspring, and an
// offspring of two FJ points.

#include "core_test_support.h"
#include "deadline.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "recombine.h"
#include "rng.h"
#include "test_paths.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace core_test;

namespace {

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/" + instance) == HighsStatus::kOk);
}

// FJ's first feasible point on `s` from `seed`.
std::vector<double> fj_point(const Setup& s, uint64_t seed) {
    BestSink sink(*s.problem.integrality, 0);
    FjWorker fj(s.problem, s.exec, sink, kBudget, kBudget, seed, /*start=*/{}, WorkerTrace{0, 0});
    run_until_found(fj, sink);
    REQUIRE(sink.found);
    return sink.solution;
}

}  // namespace

TEST_CASE("recombine: the limit on free columns follows cuOpt's adaptation", "[recombine]") {
    using recombine::next_max_vars_from_other;
    // An offspring equal to a parent grows it by 10%, rounded up.
    CHECK(next_max_vars_from_other(200, true) == 220);
    CHECK(next_max_vars_from_other(21, true) == 24);
    // At most kBpMaxVarsFromOther.
    CHECK(next_max_vars_from_other(9500, true) == 10000);
    CHECK(next_max_vars_from_other(10000, true) == 10000);
    // Any other shrinks it by 1%, rounded down.
    CHECK(next_max_vars_from_other(200, false) == 198);
    CHECK(next_max_vars_from_other(1000, false) == 990);
    // At least kBpMinVarsFromOther.
    CHECK(next_max_vars_from_other(21, false) == 20);
    CHECK(next_max_vars_from_other(20, false) == 20);
}

TEST_CASE("recombine: no offspring with nothing free or nothing fixed", "[recombine]") {
    Highs highs;
    read(highs, "lseu.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    recombine::BoundPropRecombiner recombiner(s.problem);
    const std::vector<double> a = fj_point(s, 0);
    const double obj_a = s.problem.objective(a);
    Rng rng(1);
    std::vector<double> out = {1.0};

    SECTION("identical parents leave no column free") {
        const recombine::Offspring o =
            recombiner.recombine(a, obj_a, a, obj_a, kBudget, Deadline{}, rng, out);
        CHECK_FALSE(o.success);
        CHECK_FALSE(o.feasible);
        CHECK(out.empty());
    }
    SECTION("parents differing on every integer, within the limit, leave none fixed") {
        // lseu's integers are all binaries, fewer than the initial limit.
        size_t num_integer = 0;
        std::vector<double> b = a;
        for (HighsInt j = 0; j < lp.num_col_; ++j) {
            if (is_integer(lp.integrality_, j)) {
                b[j] = 1.0 - std::round(a[j]);
                ++num_integer;
            }
        }
        REQUIRE(num_integer > 0);
        REQUIRE(num_integer <= recombine::kBpInitialVarsFromOther);
        const recombine::Offspring o = recombiner.recombine(a, obj_a, b, s.problem.objective(b),
                                                            kBudget, Deadline{}, rng, out);
        CHECK_FALSE(o.success);
        CHECK(out.empty());
    }
    CHECK(recombiner.max_vars_from_other() == recombine::kBpInitialVarsFromOther);
}

TEST_CASE("recombine: an offspring of two FJ points keeps what they agree on", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> a = fj_point(s, 0);
    const std::vector<double> b = fj_point(s, 1);
    REQUIRE(a != b);
    const double obj_a = s.problem.objective(a);
    const double obj_b = s.problem.objective(b);

    recombine::BoundPropRecombiner recombiner(s.problem);
    size_t num_feasible = 0;
    std::vector<double> out;
    for (uint64_t seed = 0; seed < 8; ++seed) {
        Rng rng(seed);
        const recombine::Offspring o =
            recombiner.recombine(a, obj_a, b, obj_b, kBudget, Deadline{}, rng, out);
        REQUIRE(out.size() == static_cast<size_t>(lp.num_col_));
        CHECK(o.effort > 0);
        for (HighsInt j = 0; j < lp.num_col_; ++j) {
            CHECK(out[j] >= lp.col_lower_[j] - kFeasTol);
            CHECK(out[j] <= lp.col_upper_[j] + kFeasTol);
            if (is_integer(lp.integrality_, j)) {
                CHECK(out[j] == std::round(out[j]));
                if (std::round(a[j]) == std::round(b[j])) {
                    CHECK(out[j] == std::round(a[j]));
                }
            }
        }
        if (o.feasible) {
            ++num_feasible;
            CHECK(is_feasible(lp, out));
            CHECK(std::abs(o.objective - s.problem.objective(out)) <=
                  1e-6 * (1 + std::abs(o.objective)));
        }
    }
    CHECK(num_feasible > 0);
}

TEST_CASE("recombine: the same seed gives the same offspring", "[recombine]") {
    Highs highs;
    read(highs, "p0548.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> a = fj_point(s, 0);
    const std::vector<double> b = fj_point(s, 1);
    REQUIRE(a != b);
    const double obj_a = s.problem.objective(a);
    const double obj_b = s.problem.objective(b);

    std::vector<double> first;
    std::vector<double> second;
    {
        recombine::BoundPropRecombiner recombiner(s.problem);
        Rng rng(3);
        static_cast<void>(
            recombiner.recombine(a, obj_a, b, obj_b, kBudget, Deadline{}, rng, first));
    }
    {
        recombine::BoundPropRecombiner recombiner(s.problem);
        Rng rng(3);
        static_cast<void>(
            recombiner.recombine(a, obj_a, b, obj_b, kBudget, Deadline{}, rng, second));
    }
    REQUIRE_FALSE(first.empty());
    CHECK(first == second);
}
