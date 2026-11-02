// LocalMIP under the objective cut (#208, #223): the worker's search targets
// the pool's cutoff instead of only its own best, read from the sink at
// construction and on the attempt loop's deadline-poll cadence.  Fixed seeds,
// no clock.

#include "core_test_support.h"
#include "fj_worker.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "local_mip_worker.h"
#include "rng.h"
#include "test_paths.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

using namespace core_test;
using local_mip_detail::LocalMipWorker;

namespace {

constexpr ObjectiveCutParams kCutOn{.on = true, .rel = 1e-3, .abs = 1e-7};
constexpr size_t kTotal = size_t{1} << 25;
constexpr size_t kShortAttempt = 1024;
constexpr int kShortAttempts = 1 << 17;

struct Model {
    Highs highs;
    Model() {
        highs.setOptionValue("output_flag", false);
        REQUIRE(highs.readModel(instances_dir() + "/p0548.mps") == HighsStatus::kOk);
    }
};

// A peer's find: FJ's best on the model, standing in for what another
// heuristic left in the pool.
struct PeerFind {
    double objective = 0.0;
    std::vector<double> solution;
};

PeerFind peer_find(const Setup& s) {
    BestSink sink(*s.problem.integrality, 0);
    FjWorker fj(s.problem, s.exec, sink, kTotal, kTotal, /*seed=*/1, /*start=*/{},
                WorkerTrace{1, 0});
    for (int k = 0; k < kMaxAttempts && !fj.finished(); ++k) {
        static_cast<void>(fj.run_attempt(kAttemptBudget));
    }
    REQUIRE(sink.found);
    return {sink.objective, sink.solution};
}

// Every accepted offer, in order, with the slot's traced effort.
class RecordingSink : public SolutionSink {
public:
    using SolutionSink::SolutionSink;
    std::mutex mtx;
    std::vector<double> objectives;
    std::vector<size_t> efforts;

private:
    void on_accept(double obj, const std::vector<double>& /*x*/, int /*source*/,
                   const WorkerTrace& trace, size_t effort_at) override {
        std::scoped_lock guard(mtx);
        objectives.push_back(obj);
        efforts.push_back(trace.at(effort_at));
    }
};

LocalMipWorker make_worker(const Setup& s, SolutionSink& sink, ObjectiveCutParams cut) {
    return {s.problem,
            s.exec,
            sink,
            kTotal,
            kTotal,
            /*seed=*/0,
            /*initial_solution=*/nullptr,
            WorkerTrace{0, 0},
            /*stall_on_violations=*/false,
            cut};
}

}  // namespace

TEST_CASE("LocalMIP objective cut: a rebuilt worker targets the pool's cutoff",
          "[objective_cut][local_mip]") {
    Model m;
    const Setup s(m.highs.getLp(), m.highs.getOptions().log_options);
    const PeerFind peer = peer_find(s);
    BestSink sink(*s.problem.integrality, 0);
    REQUIRE(sink.offer(peer.objective, peer.solution, WorkerTrace{1, 0}, 0).improved_incumbent);

    // A worker built over a pool that already holds the peer's find starts
    // with the pool's best minus the step as its target, not infinity.
    const LocalMipWorker on = make_worker(s, sink, kCutOn);
    REQUIRE(on.cut_target_for_test() == objective_cut_target(peer.objective, kCutOn));
    REQUIRE(on.cut_target_for_test() < peer.objective);
    // And that is what the search is handed, from its first step: a target
    // whose `best - epsilon` row is the cut's own, though this worker has
    // found nothing yet.
    REQUIRE(on.search_has_target_for_test());
    REQUIRE(on.search_target_for_test() == on.cut_target_for_test() + on.epsilon_for_test());
    // Off, whatever the steps say, it has no cutoff and no target.
    const LocalMipWorker off = make_worker(s, sink, {.on = false, .rel = 1e-3, .abs = 1e-7});
    REQUIRE(off.cut_target_for_test() == kHighsInf);
    REQUIRE_FALSE(off.search_has_target_for_test());
}

TEST_CASE("LocalMIP objective cut: a tightening keeps the weights, and the attempt picks it up",
          "[objective_cut][local_mip]") {
    Model m;
    const Setup s(m.highs.getLp(), m.highs.getOptions().log_options);
    const PeerFind peer = peer_find(s);

    SECTION("refresh_objective_cut moves the cutoff and touches no weight") {
        BestSink sink(*s.problem.integrality, 0);
        LocalMipWorker worker = make_worker(s, sink, kCutOn);
        REQUIRE(worker.cut_target_for_test() == kHighsInf);
        // Weights away from their initial 1, so "unchanged" is not "reset".
        Rng rng(7);
        worker.randomize_weights(rng);
        static_cast<void>(worker.run_attempt(kAttemptBudget));
        const std::vector<uint64_t> weights = worker.weights_for_test();
        const uint64_t obj_weight = worker.obj_weight_for_test();

        REQUIRE(sink.offer(peer.objective, peer.solution, WorkerTrace{1, 0}, 0).improved_incumbent);
        worker.refresh_objective_cut();
        REQUIRE(worker.cut_target_for_test() == objective_cut_target(peer.objective, kCutOn));
        REQUIRE(worker.weights_for_test() == weights);
        REQUIRE(worker.obj_weight_for_test() == obj_weight);
    }

    SECTION("run_attempt reads a peer's find on its first poll") {
        BestSink sink(*s.problem.integrality, 0);
        LocalMipWorker worker = make_worker(s, sink, kCutOn);
        REQUIRE(worker.cut_target_for_test() == kHighsInf);
        REQUIRE(sink.offer(peer.objective, peer.solution, WorkerTrace{1, 0}, 0).improved_incumbent);
        // The poll comes before the attempt's first step; anything the
        // worker offers after it can only lower the pool's best.
        static_cast<void>(worker.run_attempt(1));
        REQUIRE(worker.cut_target_for_test() <= objective_cut_target(peer.objective, kCutOn));
    }
}

// "Off is main" is not pinned here: an off worker's cutoff is infinite, so
// the search's target reduces to the worker's own best exactly, and the
// deterministic LocalMIP tests elsewhere in the suite run that path.  What
// this pins is that the search reads the cutoff at all, once the worker has
// finds of its own: the same seed from an empty pool makes the same first
// offer either way (nothing to cut against yet), and after the last offer
// the search is handed a target a step below the worker's own best, where
// without the cut it is handed that best.  Whether the lower target changes
// a later step is not asserted: it reaches only `infeasible_step`'s
// breakthrough scores, so whether a run diverges depends on the random
// stream, which differs between standard libraries (it does on libc++ and
// does not on libstdc++).
TEST_CASE("LocalMIP objective cut: the search reads the cutoff", "[objective_cut][local_mip]") {
    Model m;
    const Setup s(m.highs.getLp(), m.highs.getOptions().log_options);
    struct Run {
        std::vector<double> objectives;
        std::vector<size_t> efforts;
        double search_target = 0.0;
        double epsilon = 0.0;
    };
    const auto run = [&](ObjectiveCutParams cut) {
        RecordingSink sink(*s.problem.integrality, 0);
        LocalMipWorker worker = make_worker(s, sink, cut);
        // Short attempts: the cutoff is read at an attempt's first poll,
        // and p0548's improvements come ~2000 effort apart, inside one
        // default attempt, the last of them shortly before it retires.
        for (int k = 0; k < kShortAttempts && !worker.finished(); ++k) {
            static_cast<void>(worker.run_attempt(kShortAttempt));
        }
        REQUIRE(worker.search_has_target_for_test());
        return Run{sink.objectives, sink.efforts, worker.search_target_for_test(),
                   worker.epsilon_for_test()};
    };
    const Run plain = run({});
    const Run cut = run(kCutOn);
    REQUIRE_FALSE(plain.objectives.empty());
    REQUIRE_FALSE(cut.objectives.empty());
    REQUIRE(cut.objectives[0] == plain.objectives[0]);
    REQUIRE(cut.efforts[0] == plain.efforts[0]);
    // Without the cut, the target is the worker's own best.
    REQUIRE(plain.search_target == plain.objectives.back());
    // With it, the cut's own row below the pool's best, which is the
    // worker's own: a step lower.
    REQUIRE(cut.search_target == objective_cut_target(cut.objectives.back(), kCutOn) + cut.epsilon);
    REQUIRE(cut.search_target < cut.objectives.back());
}
