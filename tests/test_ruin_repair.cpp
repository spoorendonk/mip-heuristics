// Ruin-and-repair LNS (#207) in the heuristic core, on a model the caller
// owns: its schedule, its ruin set, FJ's objective cutoff, and both runners.

#include "core_test_support.h"
#include "fj_worker.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "rng.h"
#include "ruin_repair.h"
#include "test_paths.h"

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

using namespace core_test;

namespace {

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/" + instance) == HighsStatus::kOk);
}

// Every objective the sink accepted, in order.
class RecordingSink : public SolutionSink {
public:
    using SolutionSink::SolutionSink;
    std::mutex mtx;
    std::vector<double> objectives;

private:
    void on_accept(double obj, const std::vector<double>& /*x*/, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t /*effort_at*/) override {
        std::scoped_lock guard(mtx);
        objectives.push_back(obj);
    }
};

// FJ's first feasible point on `s`, without a cutoff.
std::vector<double> first_fj_point(const Setup& s) {
    BestSink sink(*s.problem.integrality, 0);
    FjWorker fj(s.problem, s.exec, sink, kBudget, kBudget, /*seed=*/0, /*start=*/{},
                WorkerTrace{0, 0});
    run_until_found(fj, sink);
    REQUIRE(sink.found);
    return sink.solution;
}

}  // namespace

TEST_CASE("ruin_repair: the ruin size follows cuOpt's schedule", "[ruin_repair]") {
    using ruin_repair::ruin_size;
    // Base |I| / 200 clamped to [6, 40].
    CHECK(ruin_size(100, 0) == 6);
    CHECK(ruin_size(2000, 0) == 10);
    CHECK(ruin_size(100000, 0) == 40);
    // Grows by one base every 6 failed repairs, at most 7 bases.
    CHECK(ruin_size(2000, 5) == 10);
    CHECK(ruin_size(2000, 6) == 20);
    CHECK(ruin_size(2000, 36) == 70);
    CHECK(ruin_size(2000, 1000) == 70);
    // Never more than there are columns.
    CHECK(ruin_size(4, 0) == 4);
    CHECK(ruin_size(0, 0) == 0);
}

TEST_CASE("ruin_repair: the repair budget is cuOpt's seconds at the calibrated rate",
          "[ruin_repair]") {
    using ruin_repair::kRepairEffortPerSecond;
    using ruin_repair::repair_budget;
    const auto at = [](double seconds) {
        return static_cast<size_t>(seconds * kRepairEffortPerSecond);
    };
    CHECK(repair_budget(1000, 0) == 1000 + at(0.15));
    CHECK(repair_budget(1000, 10) == 1000 + at(0.15 + (0.02 * 10)));
    // Capped at two seconds.
    CHECK(repair_budget(1000, 1000) == 1000 + at(2.0));
}

TEST_CASE("ruin_repair: the ruin set takes guidance first, repeats nothing, leaves scratch zeroed",
          "[ruin_repair]") {
    const std::vector<HighsInt> integer_cols = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint8_t> chosen(10, 0);
    std::vector<HighsInt> out;
    Rng rng(7);

    SECTION("guidance fills a set no larger than itself") {
        std::vector<HighsInt> guidance = {3, 5, 7, 9};
        ruin_repair::select_ruin_set(guidance, integer_cols, 2, rng, chosen, out);
        REQUIRE(out.size() == 2);
        for (const HighsInt j : out) {
            CHECK(std::ranges::find(guidance, j) != guidance.end());
        }
    }
    SECTION("guidance first, then uniform draws, no repeats") {
        std::vector<HighsInt> guidance = {3, 5, 7};
        ruin_repair::select_ruin_set(guidance, integer_cols, 6, rng, chosen, out);
        REQUIRE(out.size() == 6);
        std::vector<HighsInt> head(out.begin(), out.begin() + 3);
        std::ranges::sort(head);
        CHECK(head == std::vector<HighsInt>{3, 5, 7});
        std::vector<HighsInt> sorted = out;
        std::ranges::sort(sorted);
        CHECK(std::ranges::adjacent_find(sorted) == sorted.end());
    }
    SECTION("a set as large as the integers takes them all") {
        std::vector<HighsInt> guidance;
        ruin_repair::select_ruin_set(guidance, integer_cols, 10, rng, chosen, out);
        std::ranges::sort(out);
        CHECK(out == integer_cols);
    }
    CHECK(std::ranges::all_of(chosen, [](uint8_t c) { return c == 0; }));
}

TEST_CASE("ruin_repair: is_feasible agrees with the model", "[ruin_repair]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    std::vector<double> x = first_fj_point(s);
    REQUIRE(is_feasible(lp, x));
    CHECK(ruin_repair::is_feasible(s.problem, x));
    // Off a bound.
    x[0] = lp.col_upper_[0] + 1.0;
    CHECK_FALSE(ruin_repair::is_feasible(s.problem, x));
    // Wrong length.
    CHECK_FALSE(ruin_repair::is_feasible(s.problem, std::vector<double>(1, 0.0)));
}

TEST_CASE("ruin_repair: under a cutoff FJ offers only points better than it", "[ruin_repair]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> seed = first_fj_point(s);
    const double seed_objective = s.problem.objective(seed);
    const double cutoff = seed_objective - (ruin_repair::kCutoffMarginFeastols * s.problem.feastol);

    RecordingSink sink(*s.problem.integrality, 0);
    FjWorker fj(s.problem, s.exec, sink, kBudget, kBudget, /*seed=*/1, seed, WorkerTrace{0, 0},
                cutoff);
    for (int attempt = 0; attempt < 20 && !fj.finished(); ++attempt) {
        static_cast<void>(fj.run_attempt(kAttemptBudget));
    }
    // FJ satisfies the cutoff row to within feastol, so every point is
    // strictly better than the seed.
    for (const double obj : sink.objectives) {
        CHECK(obj < seed_objective);
        CHECK(obj <= cutoff + s.problem.feastol);
    }
    if (!fj.best_found().empty()) {
        CHECK(is_feasible(lp, fj.best_found()));
        CHECK(fj.best_found_objective() < seed_objective);
    }
}

TEST_CASE("ruin_repair: a dispatch improves a feasible point in the pool", "[ruin_repair]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> seed = first_fj_point(s);
    const double seed_objective = s.problem.objective(seed);

    BestSink sink(*s.problem.integrality, 0);
    static_cast<void>(sink.offer(seed_objective, seed, WorkerTrace{0, 0}, 0));
    REQUIRE(sink.found);
    ExecutionContext exec = s.exec;
    const HeuristicBudget budget = make_budget(kBudget, 1, kBudget);
    const DispatchOutcome outcome = ruin_repair::run(s.problem, budget, exec, sink);

    CHECK(outcome.effort > 0);
    REQUIRE(sink.objective < seed_objective);
    CHECK(is_feasible(lp, sink.solution));
}

TEST_CASE("ruin_repair: a dispatch with nothing to ruin, or no budget, does nothing",
          "[ruin_repair]") {
    Highs highs;
    read(highs, "egout.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    ExecutionContext exec = s.exec;
    BestSink sink(*s.problem.integrality, 0);

    SECTION("an empty pool") {
        const DispatchOutcome outcome =
            ruin_repair::run(s.problem, make_budget(kBudget, 1, kBudget), exec, sink);
        CHECK(outcome.effort == 0);
    }
    SECTION("a zero budget") {
        const std::vector<double> seed = first_fj_point(s);
        static_cast<void>(sink.offer(s.problem.objective(seed), seed, WorkerTrace{0, 0}, 0));
        const DispatchOutcome outcome = ruin_repair::run(s.problem, HeuristicBudget{}, exec, sink);
        CHECK(outcome.effort == 0);
        CHECK(sink.accepted() == 1);
    }
}

TEST_CASE("ruin_repair: a caller's slot without a seed waits and returns at the stop",
          "[ruin_repair][until_stopped]") {
    Highs highs;
    read(highs, "egout.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    BestSink sink(*s.problem.integrality, 0);

    std::thread stopper([&stop] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop.store(true);
    });
    const size_t effort =
        ruin_repair::run_until_stopped(s.problem, make_until_stopped_budget(kBudget, kBudget), exec,
                                       /*worker=*/0, RestartSource{}, sink);
    stopper.join();
    // Nothing to ruin, so nothing was charged and nothing found.
    CHECK(effort == 0);
    CHECK(sink.accepted() == 0);
}

TEST_CASE("ruin_repair: a caller's waiting slot returns at the terminator",
          "[ruin_repair][until_stopped]") {
    Highs highs;
    read(highs, "egout.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    ExecutionContext exec = s.exec;
    // No stop flag: only the terminator can end the wait.
    int polls = 0;
    exec.terminator = [&polls] { return ++polls > 3; };
    BestSink sink(*s.problem.integrality, 0);
    const size_t effort =
        ruin_repair::run_until_stopped(s.problem, make_until_stopped_budget(kBudget, kBudget), exec,
                                       /*worker=*/0, RestartSource{}, sink);
    CHECK(effort == 0);
    CHECK(polls > 3);
}

TEST_CASE("ruin_repair: an infeasible population point leaves a slot ruining its own best",
          "[ruin_repair][until_stopped]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> seed = first_fj_point(s);
    const double seed_objective = s.problem.objective(seed);
    // The lower bounds, which `take_restart` keeps as they are.
    const std::vector<double> bad = lp.col_lower_;
    REQUIRE_FALSE(is_feasible(lp, bad));
    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    BestSink sink(*s.problem.integrality, 0);

    // The seed once, then only the infeasible point, until the first
    // improvement or a fixed number of attempts.
    constexpr int kMaxSourceCalls = 200;
    int calls = 0;
    const RestartSource source = [&](Rng& /*rng*/, std::vector<double>& out) {
        if (sink.found || ++calls > kMaxSourceCalls) {
            stop.store(true);
            return false;
        }
        out = calls == 1 ? seed : bad;
        return true;
    };
    static_cast<void>(ruin_repair::run_until_stopped(
        s.problem, make_until_stopped_budget(kBudget, kBudget), exec, /*worker=*/0, source, sink));
    REQUIRE(sink.found);
    CHECK(sink.objective < seed_objective);
    CHECK(is_feasible(lp, sink.solution));
}

TEST_CASE("ruin_repair: a caller's slot ruins the source's point", "[ruin_repair][until_stopped]") {
    Highs highs;
    read(highs, "egout.mps");
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> seed = first_fj_point(s);
    const double seed_objective = s.problem.objective(seed);
    std::atomic<bool> stop{false};
    ExecutionContext exec = s.exec;
    exec.stop = &stop;
    BestSink sink(*s.problem.integrality, 0);

    // Hands out the seed every attempt, and stops the slot at the first
    // improvement the sink accepts or after a fixed number of attempts.
    constexpr int kMaxSourceCalls = 200;
    int calls = 0;
    const RestartSource source = [&](Rng& /*rng*/, std::vector<double>& out) {
        if (sink.found || ++calls > kMaxSourceCalls) {
            stop.store(true);
            return false;
        }
        out = seed;
        return true;
    };
    const size_t effort = ruin_repair::run_until_stopped(
        s.problem, make_until_stopped_budget(kBudget, kBudget), exec, /*worker=*/0, source, sink);
    CHECK(effort > 0);
    REQUIRE(sink.found);
    CHECK(sink.objective < seed_objective);
    CHECK(is_feasible(lp, sink.solution));
}
