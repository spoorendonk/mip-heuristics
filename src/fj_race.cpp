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
#include "io/HighsIO.h"
#include "solution_sink.h"
#include "worker_base.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fj {

namespace {

// The device side of one race, built on the dispatching thread.  Members
// go in the order they must outlive each other: the batch reads the
// device model, which was copied from the host model.
struct GpuSide {
    std::unique_ptr<gpu_fj::HostModel> host;
    std::unique_ptr<gpu_fj::DeviceModel> device;
    std::unique_ptr<gpu_fj::Climber> batch;
};

bool start_gpu(const ProblemView& problem, GpuSide& gpu, std::string& error) {
    gpu.host = std::make_unique<gpu_fj::HostModel>(problem);
    if (!gpu.host->legal_bounds()) {
        error = "a column's bounds are ones FJ refuses";
        return false;
    }
    const std::int32_t climbers = gpu_fj::default_climbers(error);
    if (climbers <= 0) {
        return false;
    }
    gpu.device = gpu_fj::DeviceModel::upload(gpu.host->pod(), error);
    if (!gpu.device) {
        return false;
    }
    gpu.batch = gpu_fj::Climber::create(*gpu.device, climbers, error);
    if (!gpu.batch && gpu.device->has_related_table()) {
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
        const double lower = model.col_lower[j];
        const double upper = model.col_upper[j];
        x[j] = use_seed && std::isfinite(seed_point[j])
                   ? std::max(lower, std::min(upper, seed_point[j]))
                   : fj_bound_start(lower, upper);
    }
}

struct Climb {
    size_t effort = 0;
    std::string error;  // empty unless a CUDA call failed
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
        if (start_all(out.error)) {
            while (!stop.load(std::memory_order_relaxed) && !exec_.past_deadline() &&
                   step(out.error)) {
            }
        }
        for (size_t c = 0; c < count_; ++c) {
            out.effort += traces_[c].at(chunks_[c].total_effort);
        }
        return out;
    }

private:
    bool start_all(std::string& error) {
        // The CPU workers take seeds from `base_seed` up; the climbers take
        // the next `count_`, so no climber shares a CPU worker's stream.
        const size_t first_slot = exec_.num_workers;
        for (size_t c = 0; c < count_; ++c) {
            traces_[c].worker = static_cast<int>(first_slot + c);
            seeds_[c] = std::uint64_t{exec_.base_seed} + first_slot + c;
            if (!restart(c, error)) {
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
            if (chunk.finished || chunk.effort_since_improvement >= budget_.worker_stale) {
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

void warn_cpu_only(const ExecutionContext& exec, const std::string& error) {
    highsLogUser(exec.log_options, HighsLogType::kWarning,
                 "mip_heuristic_fj_gpu=true but the GPU FJ stopped (%s); "
                 "FeasibilityJump runs on the CPU only.\n",
                 error.c_str());
}

}  // namespace

DispatchOutcome race(const ProblemView& problem, const HeuristicBudget& budget,
                     ExecutionContext& exec, SolutionSink& sink) {
    if (!budget.fj_gpu || problem.degenerate() || budget.disabled()) {
        return run(problem, budget, exec, sink);
    }
    GpuSide gpu;
    std::string error;
    if (!start_gpu(problem, gpu, error)) {
        warn_cpu_only(exec, error);
        return run(problem, budget, exec, sink);
    }

    const gpu_fj::GpuFjModel model = gpu.host->pod();
    Climbing climbing(*gpu.batch, model, problem, budget, exec, sink);
    std::atomic<bool> stop{false};
    Climb climb;
    std::exception_ptr failure;
    std::thread thread([&] {
        try {
            climb = climbing.run(stop);
        } catch (...) {
            failure = std::current_exception();
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
    if (failure) {
        std::rethrow_exception(failure);
    }
    if (!climb.error.empty()) {
        warn_cpu_only(exec, climb.error);
    }
    // Charged by the dispatcher, on this thread, with the CPU workers'.
    outcome.effort += climb.effort;
    return outcome;
}

}  // namespace fj

#endif  // MIP_HEURISTICS_HAS_GPU_FJ
