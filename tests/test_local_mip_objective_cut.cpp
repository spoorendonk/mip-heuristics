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

LocalMipWorker make_worker(const Setup& s, SolutionSink& sink, ObjectiveCutParams cut) {
    return {s.problem,
            s.exec,
            sink,
            kTotal,
            kTotal,
            /*seed=*/0,
            nullptr,
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
    // Off, whatever the steps say, it has no cutoff.
    const LocalMipWorker off = make_worker(s, sink, {.on = false, .rel = 1e-3, .abs = 1e-7});
    REQUIRE(off.cut_target_for_test() == kHighsInf);
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

TEST_CASE("LocalMIP objective cut: off searches as without it", "[objective_cut][local_mip]") {
    Model m;
    const Setup s(m.highs.getLp(), m.highs.getOptions().log_options);
    const auto offers = [&](ObjectiveCutParams cut) {
        RecordingSink sink(*s.problem.integrality, 0);
        LocalMipWorker worker = make_worker(s, sink, cut);
        for (int k = 0; k < kMaxAttempts && !worker.finished(); ++k) {
            static_cast<void>(worker.run_attempt(kAttemptBudget));
        }
        return std::vector<std::vector<double>>{
            sink.objectives, std::vector<double>(sink.efforts.begin(), sink.efforts.end())};
    };
    const auto plain = offers({});
    REQUIRE_FALSE(plain[0].empty());
    REQUIRE(offers({.on = false, .rel = 1e-3, .abs = 1e-7}) == plain);
}
