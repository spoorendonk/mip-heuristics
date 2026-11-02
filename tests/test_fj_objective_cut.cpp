// FJ under the objective cut (#208, #222): one worker, a fixed seed, the
// pool's best read back through the sink.  The cut is tightened in place by
// the patched `FeasibilityJumpSolver::setObjectiveCut`, so every case runs
// a single `FjWorker` across its tightenings.

#include "core_test_support.h"
#include "fj_worker.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "test_paths.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <vector>

using namespace core_test;

namespace {

constexpr ObjectiveCutParams kCutOn{.on = true, .rel = 1e-3, .abs = 1e-7};
// Total effort for one worker: enough for several improvements on the
// instance below, small enough to keep the case fast.
constexpr size_t kTotal = size_t{1} << 25;

// Every accepted offer, in order, with the slot's traced effort and the
// point itself: the in-place cut refresh is checked against the model too.
class RecordingSink : public SolutionSink {
public:
    using SolutionSink::SolutionSink;
    std::mutex mtx;
    std::vector<double> objectives;
    std::vector<size_t> efforts;
    std::vector<std::vector<double>> solutions;

private:
    void on_accept(double obj, const std::vector<double>& x, int /*source*/,
                   const WorkerTrace& trace, size_t effort_at) override {
        std::scoped_lock guard(mtx);
        objectives.push_back(obj);
        efforts.push_back(trace.at(effort_at));
        solutions.push_back(x);
    }
};

// One worker until it retires or spends `kTotal`, attempt by attempt.
void run(FjWorker& fj) {
    for (int attempt = 0; attempt < kMaxAttempts && !fj.finished(); ++attempt) {
        static_cast<void>(fj.run_attempt(kAttemptBudget));
    }
}

// FJ's row is satisfied within its violation tolerance, `feastol`, and the
// objective is accumulated in another order than the row's LHS.
bool honours(double objective, double cutoff) {
    return objective <= cutoff + kFeasTol + (1e-9 * std::max(1.0, std::abs(cutoff)));
}

struct Model {
    Highs highs;
    Model() {
        highs.setOptionValue("output_flag", false);
        REQUIRE(highs.readModel(instances_dir() + "/p0548.mps") == HighsStatus::kOk);
    }
};

}  // namespace

TEST_CASE("FJ objective cut: each offer honours the cutoff in force, one worker",
          "[objective_cut][fj]") {
    Model m;
    const Setup s(m.highs.getLp(), m.highs.getOptions().log_options);
    RecordingSink sink(*s.problem.integrality, 0);
    FjWorker fj(s.problem, s.exec, sink, kTotal, kTotal, /*seed=*/0, /*start=*/{},
                WorkerTrace{0, 0}, std::numeric_limits<double>::infinity(), kCutOn);
    run(fj);

    // (b) Several tightenings, all in this one worker: nothing here can
    // rebuild it, and its traced effort rises across them.
    REQUIRE(sink.objectives.size() >= 3);
    REQUIRE(std::ranges::is_sorted(sink.efforts));
    REQUIRE(std::ranges::adjacent_find(sink.efforts) == sink.efforts.end());
    // (a) With one worker the pool's best before an offer is the previous
    // accepted offer, so the cutoff in force is its target.
    for (size_t k = 1; k < sink.objectives.size(); ++k) {
        INFO("offer " << k << ": " << sink.objectives[k] << " after " << sink.objectives[k - 1]);
        REQUIRE(honours(sink.objectives[k], objective_cut_target(sink.objectives[k - 1], kCutOn)));
    }
    // A stale cut row would let FJ report a point that breaks a model row.
    for (const auto& x : sink.solutions) {
        REQUIRE(is_feasible(m.highs.getLp(), x));
    }
}

TEST_CASE("FJ objective cut: a peer's solution is applied on resume", "[objective_cut][fj]") {
    Model m;
    const Setup s(m.highs.getLp(), m.highs.getOptions().log_options);
    // A first worker's best stands in for a peer's find.
    BestSink peer_sink(*s.problem.integrality, 0);
    FjWorker peer(s.problem, s.exec, peer_sink, kTotal, kTotal, /*seed=*/1, /*start=*/{},
                  WorkerTrace{0, 0});
    run(peer);
    REQUIRE(peer_sink.found);

    RecordingSink sink(*s.problem.integrality, 0);
    FjWorker fj(s.problem, s.exec, sink, kTotal, kTotal, /*seed=*/0, /*start=*/{},
                WorkerTrace{0, 0}, std::numeric_limits<double>::infinity(), kCutOn);
    // One attempt with an empty pool, then the peer's find lands between
    // attempts, as a concurrent worker's would.
    static_cast<void>(fj.run_attempt(kAttemptBudget));
    const size_t before = sink.objectives.size();
    REQUIRE(sink.offer(peer_sink.objective, peer_sink.solution, WorkerTrace{1, 0}, 0)
                .improved_incumbent);
    const double cutoff = objective_cut_target(peer_sink.objective, kCutOn);
    run(fj);
    // The worker must still find something under the peer's cutoff, or the
    // loop below would check nothing.
    REQUIRE(sink.objectives.size() > before + 1);
    for (size_t k = before + 1; k < sink.objectives.size(); ++k) {
        INFO("offer " << k << ": " << sink.objectives[k]);
        REQUIRE(honours(sink.objectives[k], cutoff));
        REQUIRE(is_feasible(m.highs.getLp(), sink.solutions[k]));
    }
}

TEST_CASE("FJ objective cut: off, or on with an empty pool, searches as without it",
          "[objective_cut][fj]") {
    Model m;
    const Setup s(m.highs.getLp(), m.highs.getOptions().log_options);
    const auto offers = [&](ObjectiveCutParams cut) {
        RecordingSink sink(*s.problem.integrality, 0);
        FjWorker fj(s.problem, s.exec, sink, kTotal, kTotal, /*seed=*/0, /*start=*/{},
                    WorkerTrace{0, 0}, std::numeric_limits<double>::infinity(), cut);
        run(fj);
        return std::vector<std::vector<double>>{
            sink.objectives, std::vector<double>(sink.efforts.begin(), sink.efforts.end())};
    };
    // (c) Off is the pre-#222 worker, whatever the steps say.
    const auto plain = offers({});
    REQUIRE(offers({.on = false, .rel = 1e-3, .abs = 1e-7}) == plain);
    // On, the row is added at the first finite cutoff, so the search up to
    // and including the first offer is the plain one.
    const auto cut = offers(kCutOn);
    REQUIRE_FALSE(plain[0].empty());
    REQUIRE_FALSE(cut[0].empty());
    REQUIRE(cut[0][0] == plain[0][0]);
    REQUIRE(cut[1][0] == plain[1][0]);
}
