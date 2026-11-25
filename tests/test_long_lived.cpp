#include "Highs.h"
#include "long_lived.h"
#include "long_lived_owner.h"
#include "lp_data/HighsCallback.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "test_common.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <latch>
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

TEST_CASE("long-lived pool: two sinks over one pool keep their own tags while offering at once",
          "[long-lived]") {
    BareOwner bare("egout.mps", false);
    const HighsLp& model = *bare.mipsolver->model_;
    HighsInt marker = -1;
    for (HighsInt j = 0; j < model.num_col_ && marker < 0; ++j) {
        if (model.integrality_[j] != HighsVarType::kContinuous) {
            marker = j;
        }
    }
    REQUIRE(marker >= 0);

    // Each sink offers `kPerSink` points told apart by the integer value
    // at `marker`: no two are twins, and together they fill the pool, so
    // every offer is admitted whatever order the two threads interleave.
    static_assert(kPoolCapacity % 2 == 0, "the two sinks fill the pool exactly");
    constexpr int kPerSink = kPoolCapacity / 2;
    const std::array<int, 2> tags{kSolutionSourceFJ, kSolutionSourceLocalMIP};
    auto tag_of = [&](const std::vector<double>& x) {
        return tags[static_cast<size_t>(x[marker] - 1.0) / kPerSink];
    };
    std::array<std::unique_ptr<LongLivedSink>, 2> sinks;
    for (size_t s = 0; s < 2; ++s) {
        sinks[s] = std::make_unique<LongLivedSink>(*bare.mipsolver, *bare.owner, tags[s]);
    }
    std::latch start(2);
    std::array<int, 2> accepted{0, 0};
    std::vector<std::thread> producers;
    producers.reserve(2);
    for (size_t s = 0; s < 2; ++s) {
        producers.emplace_back([&, s] {
            std::vector<double> x(model.num_col_, 0.0);
            start.arrive_and_wait();
            for (int i = 0; i < kPerSink; ++i) {
                const int id = (static_cast<int>(s) * kPerSink) + i;
                x[marker] = id + 1.0;
                accepted[s] += static_cast<int>(
                    sinks[s]->offer(100.0 - id, x, WorkerTrace{.worker = 0}, 0).accepted);
            }
        });
    }
    for (std::thread& producer : producers) {
        producer.join();
    }
    REQUIRE(accepted == std::array<int, 2>{kPerSink, kPerSink});

    std::vector<SolutionPool::EntryRef> members;
    bare.owner->pool()->entry_refs(members);
    REQUIRE(members.size() == static_cast<size_t>(kPoolCapacity));
    for (const SolutionPool::EntryRef& member : members) {
        REQUIRE(member.source == tag_of(*member.solution));
    }
    // And each accepted point went on to the queue under the same tag.
    std::vector<heuristics::SolutionQueue::Entry> queued;
    bare.owner->queue().take(queued);
    REQUIRE(queued.size() == static_cast<size_t>(kPoolCapacity));
    for (const heuristics::SolutionQueue::Entry& entry : queued) {
        REQUIRE(entry.source == tag_of(entry.x));
    }
}

TEST_CASE("long-lived pool: a B&B incumbent after the chain reaches the pool at the next drain",
          "[long-lived]") {
    const char* inst = "egout.mps";
    const Optimum optimum = solve_for_optimum(inst);
    const Optimum worse = first_incumbent(inst, optimum.objective);
    BareOwner bare(inst, false);
    HighsMipSolver& mipsolver = *bare.mipsolver;

    // The chain's sink, as `run_sequential` holds it when it returns: it
    // found `worse`, which HiGHS took as its incumbent.
    {
        IncumbentSink chain(mipsolver, kSolutionSourceFPR);
        REQUIRE(
            chain.offer(pool_objective(mipsolver, worse.x), worse.x, WorkerTrace{.worker = 0}, 0)
                .accepted);
        bare.owner->seed(chain);
    }
    std::vector<SolutionPool::EntryRef> members;
    bare.owner->pool()->entry_refs(members);
    REQUIRE(members.size() == 1);
    REQUIRE(members.front().source == kSolutionSourceFPR);

    // B&B then finds the optimum, on the solving thread, as a node does.
    mipsolver.mipdata_->trySolution(optimum.x, kSolutionSourceBranching);
    REQUIRE(highs_bound(mipsolver) == Catch::Approx(optimum.objective));
    bare.owner->pool()->entry_refs(members);
    REQUIRE(members.size() == 1);

    bare.owner->drain();
    bare.owner->pool()->entry_refs(members);
    REQUIRE(members.size() == 2);
    REQUIRE(members.front().source == kSolutionSourceHeuristic);
    REQUIRE(*members.front().solution == mipsolver.mipdata_->incumbent);
    REQUIRE(members.back().source == kSolutionSourceFPR);

    // The restart source hands it out, and a producer thread may call it.
    const RestartSource source = bare.owner->restart_source(SolutionPool::RestartPolicy::kBestOnly);
    std::vector<double> restart;
    bool served = false;
    std::thread producer([&] {
        Rng rng(1);
        served = source(rng, restart);
    });
    producer.join();
    REQUIRE(served);
    REQUIRE(restart == mipsolver.mipdata_->incumbent);
}

TEST_CASE("long-lived pool: every drain of a solve leaves HiGHS's incumbent the pool's best",
          "[long-lived]") {
    const char* inst = "bell5.mps";
    HighsMipSolver* solver = nullptr;
    LongLivedOwner* owner = nullptr;
    HookGuard guard([&](HighsMipSolver& mipsolver, LongLivedOwner& o) {
        solver = &mipsolver;
        owner = &o;
    });

    // Recorded in the callback and asserted after `run`: a throw from the
    // callback would unwind through HiGHS.
    int checked = 0;
    int mismatched = 0;
    bool heuristic_led = false;
    int highs_after_heuristic = 0;
    std::vector<std::string> lines;
    std::function<void(int, const std::string&)> on_callback = [&](int callback_type,
                                                                   const std::string& message) {
        if (callback_type == kCallbackLogging) {
            lines.push_back(message);
            return;
        }
        if (callback_type != kCallbackMipUserSolution || solver->mipdata_->incumbent.empty()) {
            return;
        }
        // The callback runs right after the drain, on the solving thread.
        const std::vector<double>& incumbent = solver->mipdata_->incumbent;
        std::vector<double> best;
        Rng rng(0);
        const bool served =
            owner->restart_source(SolutionPool::RestartPolicy::kBestOnly)(rng, best);
        mismatched +=
            static_cast<int>(!served || pool_objective(*solver, best) !=
                                            Catch::Approx(pool_objective(*solver, incumbent)));
        std::vector<SolutionPool::EntryRef> members;
        owner->pool()->entry_refs(members);
        const bool highs_leads = members.front().source == kSolutionSourceHeuristic;
        highs_after_heuristic += static_cast<int>(highs_leads && heuristic_led);
        heuristic_led = heuristic_led || !highs_leads;
        ++checked;
    };
    auto trampoline = [](int callback_type, const std::string& message,
                         const HighsCallbackOutput* /*out*/, HighsCallbackInput* /*in*/,
                         void* data) {
        (*static_cast<decltype(on_callback)*>(data))(callback_type, message);
    };

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);
    require_option(h, "threads", 1);
    require_option(h, "random_seed", 0);
    require_option(h, "presolve", std::string("off"));
    require_option(h, "mip_allow_restart", false);
    require_option(h, "log_dev_level", 3);
    require_option(h, "mip_heuristic_long_lived", true);
    REQUIRE(h.setCallback(HighsCallbackFunctionType(trampoline), &on_callback) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackMipUserSolution) == HighsStatus::kOk);
    REQUIRE(h.readModel(instances_dir() + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    REQUIRE(checked > 0);
    REQUIRE(mismatched == 0);
    // A heuristic's point led the pool at one drain and HiGHS's at a later
    // one: B&B improved on the chain, and that point reached the pool.
    REQUIRE(heuristic_led);
    REQUIRE(highs_after_heuristic > 0);
    // The final line lists the members best first: the chain's seeding left
    // at least one of its own heuristics' points.
    const auto line = std::ranges::find_if(
        lines, [](const std::string& l) { return l.contains("[LongLivedPool] "); });
    REQUIRE(line != lines.end());
    const std::string origins = line->substr(line->find("origins=") + 8);
    REQUIRE(origins.find_first_of("JAM") != std::string::npos);
}

TEST_CASE("long-lived: a full queue gives an older epoch's entry up for a newer one",
          "[long-lived]") {
    heuristics::SolutionQueue queue;
    const std::vector<double> x{1.0, 0.0};
    const HighsObjective good{.value = 1.0, .magnitude = 1.0};
    const HighsObjective worse{.value = 2.0, .magnitude = 1.0};
    for (size_t i = 0; i < heuristics::kLongLivedQueueCapacity; ++i) {
        REQUIRE(queue.push(0, kSolutionSourceFJ, good, x));
    }
    // By rank alone the worse point is refused: every entry beats it.
    REQUIRE_FALSE(queue.push(0, kSolutionSourceFJ, worse, x));
    REQUIRE(queue.push(1, kSolutionSourceFJ, worse, x));

    std::vector<heuristics::SolutionQueue::Entry> taken;
    queue.take(taken);
    REQUIRE(taken.size() == heuristics::kLongLivedQueueCapacity);
    REQUIRE(std::ranges::count_if(taken, [](const auto& e) { return e.epoch == 1; }) == 1);
}

namespace {

// The `[Heur]` lines of long-lived slots (#356) among `lines`.
std::vector<std::string> slot_lines(const std::vector<std::string>& lines) {
    std::vector<std::string> out;
    for (const std::string& line : lines) {
        if (line.contains("[Heur] ") && line.contains(" phase=longlived ")) {
            out.push_back(line);
        }
    }
    return out;
}

// The value of `key` on a `key=value` trace line, empty when it is absent.
std::string field(const std::string& line, const std::string& key) {
    const std::string tag = " " + key + "=";
    const size_t pos = line.find(tag);
    if (pos == std::string::npos) {
        return {};
    }
    const size_t begin = pos + tag.size();
    return line.substr(begin, line.find_first_of(" \n", begin) - begin);
}

size_t count_field(const std::vector<std::string>& lines, const std::string& key,
                   const std::string& value) {
    return static_cast<size_t>(
        std::ranges::count_if(lines, [&](const std::string& l) { return field(l, key) == value; }));
}

void slots_on(Highs& h, int threads) {
    require_option(h, "threads", 1);
    require_option(h, "random_seed", 0);
    require_option(h, "log_dev_level", 3);
    require_option(h, "mip_heuristic_long_lived", true);
    require_option(h, "mip_heuristic_long_lived_threads", threads);
}

}  // namespace

TEST_CASE("long-lived slots: the thread count exists and is 0 by default", "[long-lived]") {
    Highs h;
    h.setOptionValue("output_flag", false);
    HighsInt threads = -1;
    REQUIRE(h.getOptionValue("mip_heuristic_long_lived_threads", threads) == HighsStatus::kOk);
    REQUIRE(threads == 0);
}

TEST_CASE("long-lived slots: they start only when both options ask for them", "[long-lived]") {
    struct Case {
        bool on;
        int threads;
        size_t slots;
    };
    for (const Case c :
         {Case{.on = true, .threads = 0, .slots = 0}, Case{.on = false, .threads = 4, .slots = 0},
          Case{.on = true, .threads = 4, .slots = 4}}) {
        const auto lines = solve_capturing_log("flugpl.mps", [&](Highs& h) {
            slots_on(h, c.threads);
            require_option(h, "mip_heuristic_long_lived", c.on);
        });
        // The trace is on, so a missing slot line is not a missing trace.
        REQUIRE(log_contains(lines, "[Heur] name=fj phase=presolve "));
        REQUIRE(slot_lines(lines).size() == c.slots);
    }
}

TEST_CASE("long-lived slots: one line per thread, booked at the end of the solve", "[long-lived]") {
    // Slots start from the owner, which only the main MIP has ("an owner
    // exists only with the option on, once per solve"); a sub-MIP logs
    // nothing here, so this line count could not show one anyway.
    const auto lines = solve_capturing_log("bell5.mps", [](Highs& h) {
        slots_on(h, 4);
        require_option(h, "mip_allow_restart", false);
    });
    const auto slots = slot_lines(lines);
    REQUIRE(slots.size() == 4);
    // Dealt round-robin over the heuristics whose effort is above zero: FJ
    // and LocalMIP at the defaults.
    REQUIRE(count_field(slots, "name", "fj") == 2);
    REQUIRE(count_field(slots, "name", "local_mip") == 2);
    REQUIRE(count_field(slots, "stopped_by", "end") + count_field(slots, "stopped_by", "done") ==
            4);
    for (const std::string& slot : slots) {
        REQUIRE_FALSE(field(slot, "dispatch").empty());
        REQUIRE_FALSE(field(slot, "join_ms").empty());
    }
}

namespace {

// A slots-on solve of `inst` whose log is captured, with the MIP solution
// callback on: the drain then submits every queued point
// (`callback_sees_all`), so a slot's accepted point is a submitted one.
struct OwnerSolve {
    std::vector<std::string> lines;
    // What the owner had submitted when it logged its pool, after the final
    // drain.
    size_t submitted = 0;
    // What it had submitted as each slot line was logged, after that
    // phase's drains (`restart_long_lived`, `stop_long_lived`).
    std::vector<size_t> submitted_at_slot_line;
    HighsStatus run_status = HighsStatus::kError;
    HighsModelStatus status = HighsModelStatus::kNotset;
    double objective = 0.0;
};

template <typename Configure>
OwnerSolve solve_with_owner(const char* inst, Configure&& configure) {
    LongLivedOwner* owner = nullptr;
    HookGuard guard([&](HighsMipSolver& /*mipsolver*/, LongLivedOwner& o) { owner = &o; });
    std::mutex mtx;
    OwnerSolve out;
    std::function<void(int, const std::string&)> on_callback = [&](int callback_type,
                                                                   const std::string& message) {
        if (callback_type != kCallbackLogging) {
            return;
        }
        std::scoped_lock lock(mtx);
        out.lines.push_back(message);
        // The pool line is logged on the solving thread, after the final drain.
        if (message.contains("[LongLivedPool] ")) {
            out.submitted = owner->submitted();
        }
        if (message.contains("[Heur] ") && message.contains(" phase=longlived ")) {
            out.submitted_at_slot_line.push_back(owner->submitted());
        }
    };
    auto trampoline = [](int callback_type, const std::string& message,
                         const HighsCallbackOutput* /*out*/, HighsCallbackInput* /*in*/,
                         void* data) {
        (*static_cast<decltype(on_callback)*>(data))(callback_type, message);
    };

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);
    configure(h);
    REQUIRE(h.setCallback(HighsCallbackFunctionType(trampoline), &on_callback) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackMipSolution) == HighsStatus::kOk);
    REQUIRE(h.readModel(instances_dir() + "/" + inst) == HighsStatus::kOk);
    out.run_status = h.run();
    out.status = h.getModelStatus();
    out.objective = h.getInfo().objective_function_value;
    std::scoped_lock lock(mtx);
    return out;
}

}  // namespace

TEST_CASE("long-lived slots: the time limit stops every slot, and their points reach HiGHS",
          "[long-lived]") {
    // issue-3166 is far from solved at the limit, so B&B is still running
    // when it passes.
    const OwnerSolve solve = solve_with_owner("issue-3166.mps", [](Highs& h) {
        slots_on(h, 2);
        // The chain cut to FJ's short dispatch, so it ends well inside the
        // limit and the slots (two FJ) start.
        require_option(h, "mip_heuristic_fpr_effort", 0.0);
        require_option(h, "mip_heuristic_local_mip_effort", 0.0);
        require_option(h, "time_limit", 3.0);
    });
    REQUIRE(solve.run_status == HighsStatus::kWarning);
    REQUIRE(solve.status == HighsModelStatus::kTimeLimit);

    const auto slots = slot_lines(solve.lines);
    REQUIRE(slots.size() == 2);
    REQUIRE(count_field(slots, "stopped_by", "deadline") == 2);
    // One-sided: whether a slot finds anything in the time left is not the
    // mechanism; that a found point is submitted is.
    if (count_field(slots, "found", "1") > 0) {
        REQUIRE(solve.submitted > 0);
    }
}

TEST_CASE("long-lived slots: a restart pauses every slot and they resume in the new space",
          "[long-lived]") {
    // lseu restarts at the root; each restart pauses both slots (one line
    // each) and, when the solve goes on, resumes them over the new model
    // (#361), so the solve books one more pair of lines at its end.
    const OwnerSolve solve = solve_with_owner("lseu.mps", [](Highs& h) { slots_on(h, 2); });
    REQUIRE(solve.run_status == HighsStatus::kOk);
    REQUIRE(solve.status == HighsModelStatus::kOptimal);
    REQUIRE(solve.objective == Catch::Approx(1120.0));

    // Every `performRestart` follows one of these two messages.
    const auto restarts =
        static_cast<size_t>(std::ranges::count_if(solve.lines, [](const std::string& l) {
            return l.contains(" inactive integer columns, restarting") ||
                   l.contains("Restarting search from the root node");
        }));
    REQUIRE(restarts >= 1);
    // A restart's presolve can finish the solve itself (once a slot
    // installed the optimum), and then nothing resumes after it.  `runSetup`
    // logs this line only when the solve goes on, which is exactly when the
    // slots resume (no limit is set, so `terminate()` stays false).
    const auto setups = static_cast<size_t>(std::ranges::count_if(
        solve.lines, [](const std::string& l) { return l.contains("Model after restart has "); }));
    REQUIRE((setups == restarts || setups + 1 == restarts));
    const auto slots = slot_lines(solve.lines);
    // A pair per phase: the first, and one per resume.  The phase a restart
    // ends books `restart`, or `done` for a slot that returned on its own
    // first; the phase after the last resume ends with the solve.
    REQUIRE(slots.size() == 2 * (setups + 1));
    for (size_t i = 0; i < slots.size(); ++i) {
        const std::string why = field(slots[i], "stopped_by");
        if (i < 2 * restarts) {
            REQUIRE((why == "restart" || why == "done"));
        } else {
            REQUIRE(why != "restart");
        }
    }
    // The lines of one phase come together, two per phase, each after every
    // drain of its phase and before the next phase starts.  A resumed slot
    // offering into an old epoch's queue would have every point refused:
    // with the callback on, a phase whose slots found a point submits at
    // least one.  One-sided: whether a phase finds anything is not the
    // mechanism.
    REQUIRE(solve.submitted_at_slot_line.size() == slots.size());
    size_t before = 0;
    for (size_t i = 0; i + 1 < slots.size(); i += 2) {
        const size_t after = solve.submitted_at_slot_line[i];
        if (field(slots[i], "found") == "1" || field(slots[i + 1], "found") == "1") {
            REQUIRE(after > before);
        }
        before = after;
    }
    if (setups == 0) {
        SKIP("the only restart's presolve finished the solve: no resume to observe");
    }
}
