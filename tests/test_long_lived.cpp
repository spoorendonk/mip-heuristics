#include "Highs.h"
#include "long_lived.h"
#include "long_lived_owner.h"
#include "lp_data/HighsCallback.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "test_common.h"

#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using heuristics::LongLivedOwner;
using heuristics::LongLivedSink;

namespace {

// Sets the owner's test-only start hook for one scope.
struct HookGuard {
    explicit HookGuard(std::function<void(HighsMipSolver&, LongLivedOwner&)> hook) {
        LongLivedOwner::set_start_hook(std::move(hook));
    }
    ~HookGuard() { LongLivedOwner::set_start_hook({}); }
    HookGuard(const HookGuard&) = delete;
    HookGuard& operator=(const HookGuard&) = delete;
};

// Options under which nothing but B&B finds a solution: presolve off, so the
// presolved space is the original one and a prior solve's `col_value` is a
// point in it; no restart, so it stays that space; no heuristic of this
// project or of HiGHS's.
void quiet_search(Highs& h) {
    require_option(h, "threads", 1);
    require_option(h, "random_seed", 0);
    require_option(h, "presolve", std::string("off"));
    require_option(h, "mip_allow_restart", false);
    require_option(h, "mip_heuristic_effort", 0.0);
    require_option(h, "mip_heuristic_run_feasibility_jump", false);
    require_option(h, "mip_heuristic_fj_effort", 0.0);
    require_option(h, "mip_heuristic_fpr_effort", 0.0);
    require_option(h, "mip_heuristic_local_mip_effort", 0.0);
}

struct Optimum {
    double objective = 0.0;
    std::vector<double> x;
};

Optimum solve_for_optimum(const char* inst) {
    Highs h;
    h.setOptionValue("output_flag", false);
    quiet_search(h);
    REQUIRE(h.readModel(instances_dir() + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
    return {.objective = h.getInfo().objective_function_value, .x = h.getSolution().col_value};
}

// The first improving solution a solve of `inst` reports, which must be
// clearly worse than its optimum.  Presolve off, so it is a point in the
// space a bare solver on `inst` works in; heuristics on, so the solve
// passes through more than one incumbent.
Optimum first_incumbent(const char* inst, double optimum) {
    std::vector<Optimum> found;
    auto cb = [](int callback_type, const std::string& /*message*/, const HighsCallbackOutput* out,
                 HighsCallbackInput* /*in*/, void* data) {
        if (callback_type == kCallbackMipSolution) {
            static_cast<std::vector<Optimum>*>(data)->push_back(
                {.objective = out->objective_function_value, .x = out->mip_solution});
        }
    };
    Highs h;
    h.setOptionValue("output_flag", false);
    require_option(h, "threads", 1);
    require_option(h, "random_seed", 0);
    require_option(h, "presolve", std::string("off"));
    REQUIRE(h.setCallback(HighsCallbackFunctionType(cb), &found) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackMipSolution) == HighsStatus::kOk);
    REQUIRE(h.readModel(instances_dir() + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE_FALSE(found.empty());
    REQUIRE(found.front().objective > optimum + (1e-6 * std::abs(optimum)));
    return found.front();
}

// A producer's handshake with the test: `go` from the solving thread, `pushed`
// back from the producer.
struct Handshake {
    std::mutex mtx;
    std::condition_variable cv;
    bool go = false;
    bool pushed = false;
};

// Start a producer that, once released, offers `point` through a
// `LongLivedSink` as a long-lived heuristic will, then returns.  From the
// start hook, on the solving thread.
void start_producer(HighsMipSolver& mipsolver, LongLivedOwner& owner, Handshake& hs,
                    const Optimum& point) {
    auto sink = std::make_shared<LongLivedSink>(mipsolver, owner, kSolutionSourceFJ);
    owner.start([&hs, &point, sink](LongLivedOwner& self) {
        {
            std::unique_lock lock(hs.mtx);
            while (!hs.go && !self.stop_requested()) {
                hs.cv.wait_for(lock, std::chrono::milliseconds(1));
            }
            if (!hs.go) {
                return;
            }
        }
        static_cast<void>(sink->offer(point.objective, point.x, WorkerTrace{.worker = 0}, 0));
        {
            std::scoped_lock lock(hs.mtx);
            hs.pushed = true;
        }
        hs.cv.notify_all();
    });
}

// Release the producer and wait until it has pushed.  Solving thread.
void release_producer(Handshake& hs) {
    std::unique_lock lock(hs.mtx);
    hs.go = true;
    hs.cv.notify_all();
    hs.cv.wait(lock, [&] { return hs.pushed; });
}

}  // namespace

TEST_CASE("long-lived: the option exists and is off by default", "[long-lived]") {
    Highs h;
    h.setOptionValue("output_flag", false);
    bool on = true;
    REQUIRE(h.getOptionValue("mip_heuristic_long_lived", on) == HighsStatus::kOk);
    REQUIRE_FALSE(on);
}

TEST_CASE("long-lived: an owner exists only with the option on, once per solve", "[long-lived]") {
    for (const bool on : {false, true}) {
        int owners = 0;
        HookGuard guard([&](HighsMipSolver& mipsolver, LongLivedOwner& /*owner*/) {
            REQUIRE_FALSE(mipsolver.submip);
            ++owners;
        });
        Highs h;
        h.setOptionValue("output_flag", false);
        require_option(h, "threads", 1);
        require_option(h, "mip_heuristic_long_lived", on);
        REQUIRE(h.readModel(instances_dir() + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        REQUIRE(owners == (on ? 1 : 0));
    }
}

TEST_CASE("long-lived: a point pushed while B&B runs is installed at the next poll point",
          "[long-lived]") {
    const char* inst = "bell5.mps";
    const Optimum optimum = solve_for_optimum(inst);

    Handshake hs;
    LongLivedOwner* owner = nullptr;
    // The origin of the first poll point after the push, and whether the
    // owner had submitted by then; -1 while no poll point followed it.
    int next_origin = -1;
    bool submitted_by_next = false;
    std::vector<std::string> lines;

    HookGuard guard([&](HighsMipSolver& mipsolver, LongLivedOwner& o) {
        owner = &o;
        start_producer(mipsolver, o, hs, optimum);
    });

    std::function<void(int, const std::string&, const HighsCallbackOutput*)> on_callback =
        [&](int callback_type, const std::string& message, const HighsCallbackOutput* out) {
            if (callback_type == kCallbackLogging) {
                lines.push_back(message);
                return;
            }
            if (callback_type != kCallbackMipUserSolution) {
                return;
            }
            bool pushed = false;
            {
                std::scoped_lock lock(hs.mtx);
                pushed = hs.pushed;
            }
            if (pushed) {
                if (next_origin < 0) {
                    next_origin = out->external_solution_query_origin;
                    submitted_by_next = owner->submitted() > 0;
                }
                return;
            }
            if (out->external_solution_query_origin != kExternalMipSolutionQueryOriginBeforeDive ||
                !(out->mip_primal_bound >
                  optimum.objective + (1e-6 * std::abs(optimum.objective)))) {
                return;
            }
            release_producer(hs);
        };
    auto trampoline = [](int callback_type, const std::string& message,
                         const HighsCallbackOutput* out, HighsCallbackInput* /*in*/, void* data) {
        (*static_cast<decltype(on_callback)*>(data))(callback_type, message, out);
    };

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);
    quiet_search(h);
    require_option(h, "mip_heuristic_long_lived", true);
    REQUIRE(h.setCallback(HighsCallbackFunctionType(trampoline), &on_callback) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackMipUserSolution) == HighsStatus::kOk);
    REQUIRE(h.readModel(instances_dir() + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    REQUIRE(hs.pushed);
    REQUIRE(next_origin == kExternalMipSolutionQueryOriginBeforeDive);
    REQUIRE(submitted_by_next);
    REQUIRE(source_codes(lines).contains('J'));
    REQUIRE(h.getInfo().objective_function_value == Catch::Approx(optimum.objective));
}

TEST_CASE("long-lived: a point pushed after the last poll point is installed by the final drain",
          "[long-lived]") {
    const char* inst = "flugpl.mps";
    const Optimum optimum = solve_for_optimum(inst);

    std::atomic<bool> pushed_after_stop{false};
    HookGuard guard([&](HighsMipSolver& mipsolver, LongLivedOwner& o) {
        auto sink = std::make_shared<LongLivedSink>(mipsolver, o, kSolutionSourceFJ);
        o.start([&, sink](LongLivedOwner& self) {
            while (!self.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            static_cast<void>(
                sink->offer(optimum.objective, optimum.x, WorkerTrace{.worker = 0}, 0));
            pushed_after_stop = true;
        });
    });

    // `mip_heuristic_presolve_only` ends with `kSolutionLimit`, which
    // `Highs::run` reports as a warning, so not through `solve_capturing_log`.
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    } capture;
    auto log_cb = [](int callback_type, const std::string& message,
                     const HighsCallbackOutput* /*out*/, HighsCallbackInput* /*in*/, void* data) {
        if (callback_type == kCallbackLogging) {
            auto* cap = static_cast<LogCapture*>(data);
            std::scoped_lock lock(cap->mtx);
            cap->lines.push_back(message);
        }
    };
    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);
    quiet_search(h);
    require_option(h, "mip_heuristic_presolve_only", true);
    require_option(h, "mip_heuristic_long_lived", true);
    REQUIRE(h.setCallback(HighsCallbackFunctionType(log_cb), &capture) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(h.readModel(instances_dir() + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kWarning);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kSolutionLimit);
    const double objective = h.getInfo().objective_function_value;
    std::vector<std::string> lines;
    {
        std::scoped_lock lock(capture.mtx);
        lines = capture.lines;
    }

    REQUIRE(pushed_after_stop);
    REQUIRE(objective == Catch::Approx(optimum.objective));
    REQUIRE(source_codes(lines).contains('J'));
}

namespace {

// A bare solver on `inst` and an owner on it, as the solving thread
// would hold them after setup.
struct BareOwner {
    Highs highs;
    HighsCallback cb{&highs};
    std::unique_ptr<HighsMipSolver> mipsolver;
    std::unique_ptr<LongLivedOwner> owner;

    BareOwner(const char* inst, bool callback_sees_all) {
        highs::parallel::initialize_scheduler(1);
        highs.setOptionValue("output_flag", false);
        if (callback_sees_all) {
            cb.user_callback = [this](int callback_type, const std::string& /*message*/,
                                      const HighsCallbackOutput* /*out*/,
                                      HighsCallbackInput* /*in*/, void* /*data*/) {
                if (callback_type == kCallbackMipSolution) {
                    ++solution_callbacks;
                }
            };
            cb.active[kCallbackMipSolution] = 1;
        }
        mipsolver = build_bare_mipsolver(highs, cb, inst);
        owner = std::make_unique<LongLivedOwner>(*mipsolver);
    }

    // Pushes from a producer thread of its own; the objective is taken here,
    // on the solving thread, since a producer may not read `model_`.
    [[nodiscard]] bool push(const Optimum& point, int epoch) const {
        const HighsObjective objective = highs_objective(mipsolver->model_->col_cost_, point.x);
        bool queued = false;
        std::thread producer(
            [&] { queued = owner->queue().push(epoch, kSolutionSourceFJ, objective, point.x); });
        producer.join();
        return queued;
    }

    int solution_callbacks = 0;
};

}  // namespace

// The drain publishes `numRestarts` before it takes the queue, so this is
// the whole of the epoch rule; a restart is what moves the epoch, and the
// test does not write HiGHS's counter to fake one.
TEST_CASE("long-lived: publishing a later epoch drops the older epoch's entries", "[long-lived]") {
    heuristics::SolutionQueue queue;
    const std::vector<double> x{1.0, 0.0};
    const HighsObjective objective{.value = 1.0, .magnitude = 1.0};
    auto push_from_producer = [&](int epoch) {
        bool queued = false;
        std::thread producer([&] { queued = queue.push(epoch, kSolutionSourceFJ, objective, x); });
        producer.join();
        return queued;
    };
    const double inf = std::numeric_limits<double>::infinity();

    REQUIRE(push_from_producer(0));
    REQUIRE(push_from_producer(0));
    REQUIRE(queue.publish(1, inf, false) == 2);
    REQUIRE_FALSE(push_from_producer(0));
    REQUIRE(push_from_producer(1));

    std::vector<heuristics::SolutionQueue::Entry> taken;
    queue.take(taken);
    REQUIRE(taken.size() == 1);
    REQUIRE(taken.front().epoch == 1);
}

TEST_CASE("long-lived: a non-improving entry reaches trySolution only for the solution callback",
          "[long-lived]") {
    const char* inst = "egout.mps";
    const Optimum optimum = solve_for_optimum(inst);
    const Optimum worse = first_incumbent(inst, optimum.objective);

    // Both are queued: the published bound is +inf until the first drain.
    // Best first, the optimum goes in and the worse point no longer
    // improves on it.
    SECTION("callback off: skipped at the drain, then refused by the filter") {
        BareOwner bare(inst, false);
        REQUIRE(bare.push(worse, 0));
        REQUIRE(bare.push(optimum, 0));
        bare.owner->drain();
        REQUIRE(bare.owner->submitted() == 1);
        REQUIRE(highs_bound(*bare.mipsolver) == Catch::Approx(optimum.objective));
        REQUIRE_FALSE(bare.push(worse, 0));
    }

    SECTION("callback on: every entry is submitted and the callback sees each") {
        BareOwner bare(inst, true);
        REQUIRE(bare.push(worse, 0));
        REQUIRE(bare.push(optimum, 0));
        bare.owner->drain();
        REQUIRE(bare.owner->submitted() == 2);
        REQUIRE(bare.solution_callbacks == 2);
        REQUIRE(highs_bound(*bare.mipsolver) == Catch::Approx(optimum.objective));
        REQUIRE(bare.push(worse, 0));
    }
}
