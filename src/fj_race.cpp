#include "fj_race.h"

// The adapter carries `MIP_HEURISTICS_HAS_GPU_FJ` in a `MIP_HEURISTICS_CUDA`
// tree only; elsewhere this is an empty translation unit and `kChain` runs
// `fj::run` (mode_dispatch.cpp).
#ifdef MIP_HEURISTICS_HAS_GPU_FJ

#include "fj.h"
#include "fj_personas.h"
#include "gpu/climber.h"
#include "gpu/device_model.h"
#include "gpu/host_model.h"
#include "heuristic_common.h"
#include "io/HighsIO.h"
#include "solution_sink.h"
#include "worker_base.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fj {

namespace {

// The device side of one race, built on the GPU thread, so its setup
// reads nothing the dispatching thread alone may (`mipdata`,
// `terminated()`).  Members
// go in the order they must outlive each other: the batch reads the
// device model, which was copied from the host model.
struct GpuSide {
    std::unique_ptr<gpu_fj::HostModel> host;
    std::unique_ptr<gpu_fj::DeviceModel> device;
    std::unique_ptr<gpu_fj::Climber> batch;
};

// Whether the GPU thread is to stop: the race is over or the deadline past.
bool halted(const ExecutionContext& exec, const std::atomic<bool>& stop) {
    return stop.load(std::memory_order_relaxed) || exec.past_deadline();
}

// The device side after `gpu.host` is built: upload, then the batch.  It
// checks `halted` between steps, so once `fj::run` has returned the join
// waits on at most the step in flight (#283's review); false with `error`
// empty when it stopped so.
bool start_gpu(GpuSide& gpu, const ExecutionContext& exec, const std::atomic<bool>& stop,
               std::string& error) {
    const std::int32_t climbers = gpu_fj::default_climbers(error);
    if (climbers <= 0 || halted(exec, stop)) {
        return false;
    }
    gpu.device = gpu_fj::DeviceModel::upload(gpu.host->pod(), error);
    if (!gpu.device || halted(exec, stop)) {
        return false;
    }
    gpu.batch = gpu_fj::Climber::create(*gpu.device, climbers, error);
    if (!gpu.batch && gpu.device->has_related_table() && !halted(exec, stop)) {
        // `upload` keeps the related-variable table whenever the model and
        // the table fit, but the batch is sized beside both, so a batch
        // that would fit beside the bare model can fail here (#277's
        // review).  Free the table's copy first, then try without it.
        gpu.device.reset();
        error.clear();
        gpu.device = gpu_fj::DeviceModel::upload(gpu.host->pod(), error, 0);
        if (!gpu.device) {
            return false;
        }
        gpu.batch = gpu_fj::Climber::create(*gpu.device, climbers, error);
    }
    return gpu.batch != nullptr;
}

// A climber's start, as `FjWorker` takes one: the pool's best, else the
// dispatch's incumbent, clamped into FJ's bounds; a column with neither
// starts where FJ's own start puts it.
void start_point(const ProblemView& problem, const gpu_fj::GpuFjModel& model, SolutionSink& sink,
                 std::vector<double>& seed_point, std::vector<double>& x) {
    if (!sink.copy_best(seed_point)) {
        seed_point = problem.incumbent;
    }
    const auto ncol = static_cast<size_t>(model.ncol);
    const bool use_seed = seed_point.size() == ncol;
    x.resize(ncol);
    for (size_t j = 0; j < ncol; ++j) {
        x[j] = fj_clamped_start(model.col_lower[j], model.col_upper[j],
                                use_seed ? seed_point[j] : kHighsInf);
    }
}

struct Climb {
    size_t effort = 0;
    std::string error;  // empty unless the GPU side failed
    // Every climber got its start: the GPU raced, whatever stopped it.
    bool raced = false;
};

// The GPU thread's whole run.  It polls only `stop` and the deadline
// (`past_deadline()`, safe on any thread), offers through the sink, and
// returns its effort and any CUDA failure rather than throwing.
class Climbing {
public:
    Climbing(gpu_fj::Climber& batch, const gpu_fj::GpuFjModel& model, const ProblemView& problem,
             const HeuristicBudget& budget, const ExecutionContext& exec, SolutionSink& sink)
        : batch_(batch),
          model_(model),
          problem_(problem),
          budget_(budget),
          exec_(exec),
          sink_(sink),
          count_(static_cast<size_t>(batch.count())),
          traces_(count_),
          seeds_(count_),
          chunks_(count_) {}

    Climb run(const std::atomic<bool>& stop) {
        Climb out;
        if (start_all(stop, out.error)) {
            out.raced = true;
            while (!halted(exec_, stop) && step(out.error)) {
            }
        }
        for (size_t c = 0; c < count_; ++c) {
            out.effort += traces_[c].at(chunks_[c].total_effort);
        }
        return out;
    }

private:
    // False, `error` empty, when halted before every climber has its start:
    // `run` then takes no step.
    bool start_all(const std::atomic<bool>& stop, std::string& error) {
        // CPU worker `w` takes `random_seed + w` (`fj.cpp`'s `make_state`,
        // `random_seed = base_seed - kBaseSeedOffset`); the climbers take
        // the next `count_`, so no climber shares a seeded worker's stream.
        const size_t first_slot = exec_.num_workers;
        const std::uint64_t random_seed = exec_.base_seed - kBaseSeedOffset;
        for (size_t c = 0; c < count_; ++c) {
            traces_[c].worker = static_cast<int>(first_slot + c);
            seeds_[c] = random_seed + first_slot + c;
            if (halted(exec_, stop) || !restart(c, error)) {
                return false;
            }
        }
        return true;
    }

    bool restart(size_t c, std::string& error) {
        start_point(problem_, model_, sink_, seed_point_, start_);
        chunks_[c] = {};
        return batch_.init(static_cast<std::int32_t>(c), start_, seeds_[c], error);
    }

    // One launch: every climber up to its next callback point, then each
    // one's improvement offered and each stalled one restarted.
    bool step(std::string& error) {
        if (!batch_.run(~std::uint64_t{0}, chunks_, error)) {
            return false;
        }
        for (size_t c = 0; c < count_; ++c) {
            const gpu_fj::Climber::Chunk& chunk = chunks_[c];
            if (chunk.improved) {
                if (!batch_.read_best(static_cast<std::int32_t>(c), best_, error)) {
                    return false;
                }
                // Discarded: like `FjWorker` (#191), a climber's patience
                // reads FJ's own `effortSinceLastImprovement`, not the
                // pool's verdict.
                static_cast<void>(sink_.offer(problem_.offset + chunk.objective, best_, traces_[c],
                                              traces_[c].at(chunk.total_effort)));
            }
            // `FjWorker`'s stale test: strictly past the budget.
            if (chunk.finished || chunk.effort_since_improvement > budget_.worker_stale) {
                traces_[c].effort_base += chunk.total_effort;
                seeds_[c] += count_;
                if (!restart(c, error)) {
                    return false;
                }
            }
        }
        return true;
    }

    gpu_fj::Climber& batch_;
    const gpu_fj::GpuFjModel& model_;
    const ProblemView& problem_;
    const HeuristicBudget& budget_;
    const ExecutionContext& exec_;
    SolutionSink& sink_;
    size_t count_;
    std::vector<WorkerTrace> traces_;
    std::vector<std::uint64_t> seeds_;
    std::vector<gpu_fj::Climber::Chunk> chunks_;
    std::vector<double> seed_point_;
    std::vector<double> start_;
    std::vector<double> best_;
};

// The GPU side's failure.  Before every climber started, FJ ran on the CPU
// alone, so the warning is the bench-contract string a CPU build emits
// (`CONFIG_IGNORED_WARNINGS`), which drops the run's row; after, the GPU
// raced and may have offered finds, so the row stands and the warning
// says only that the climbers stopped early.
void warn_gpu_failure(const ExecutionContext& exec, const Climb& climb) {
    if (climb.raced) {
        highsLogUser(exec.log_options, HighsLogType::kWarning,
                     "mip_heuristic_fj_gpu=true but the GPU FJ climbers stopped mid-race (%s).\n",
                     climb.error.c_str());
        return;
    }
    highsLogUser(exec.log_options, HighsLogType::kWarning,
                 "mip_heuristic_fj_gpu=true but the GPU FJ stopped (%s); "
                 "FeasibilityJump runs on the CPU only.\n",
                 climb.error.c_str());
}

// The GPU thread's body: the device setup too runs here, so it never
// delays the CPU workers.  A model whose bounds FJ refuses gets no climber
// and no warning: `FjWorker` refuses the same bounds, so FJ runs nowhere
// and the option changes nothing.
Climb climb_on_gpu(const ProblemView& problem, const HeuristicBudget& budget,
                   const ExecutionContext& exec, SolutionSink& sink,
                   const std::atomic<bool>& stop) {
    Climb out;
    if (halted(exec, stop)) {
        return out;
    }
    GpuSide gpu;
    gpu.host = std::make_unique<gpu_fj::HostModel>(problem);
    if (!gpu.host->legal_bounds() || halted(exec, stop)) {
        return out;
    }
    if (!start_gpu(gpu, exec, stop, out.error)) {
        // A setup cut short by the race's end is no failure to warn about.
        if (halted(exec, stop)) {
            out.error.clear();
        } else if (out.error.empty()) {
            out.error = "device setup failed";
        }
        return out;
    }
    const gpu_fj::GpuFjModel model = gpu.host->pod();
    Climbing climbing(*gpu.batch, model, problem, budget, exec, sink);
    return climbing.run(stop);
}

}  // namespace

DispatchOutcome race(const ProblemView& problem, const HeuristicBudget& budget,
                     ExecutionContext& exec, SolutionSink& sink) {
    if (!budget.fj_gpu || problem.degenerate() || budget.disabled()) {
        return run(problem, budget, exec, sink);
    }
    std::atomic<bool> stop{false};
    Climb climb;
    // Nothing the GPU side throws ends the solve: like a CUDA error, it
    // becomes the warning, and FJ's CPU outcome stands.
    std::thread thread([&] {
        try {
            climb = climb_on_gpu(problem, budget, exec, sink, stop);
        } catch (const std::exception& e) {
            climb.error = e.what();
        } catch (...) {
            climb.error = "an unknown exception";
        }
    });
    DispatchOutcome outcome;
    try {
        outcome = run(problem, budget, exec, sink);
    } catch (...) {
        stop.store(true, std::memory_order_relaxed);
        thread.join();
        throw;
    }
    stop.store(true, std::memory_order_relaxed);
    thread.join();
    if (!climb.error.empty()) {
        warn_gpu_failure(exec, climb);
    }
    // Charged by the dispatcher, on this thread, with the CPU workers'.
    outcome.effort += climb.effort;
    return outcome;
}

}  // namespace fj

#endif  // MIP_HEURISTICS_HAS_GPU_FJ
