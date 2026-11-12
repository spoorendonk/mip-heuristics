#pragma once

// A device copy of a `GpuFjModel` (#261), owned for the life of a GPU FJ
// run.  Included from `.cu` and host files alike, so it names no CUDA type:
// the device buffers sit behind `Impl`.  Every call reports a CUDA failure
// through `error` and returns false (or null) rather than throwing, so a
// caller can fall back to the CPU workers.

#include "gpu_fj_model.h"

#include <memory>
#include <span>
#include <string>

namespace gpu_fj {

class DeviceModel {
public:
    // Copies `host` to the device, or returns null with `error` set.
    static std::unique_ptr<DeviceModel> upload(const GpuFjModel& host, std::string& error);
    ~DeviceModel();
    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;

    [[nodiscard]] std::int32_t ncol() const;
    [[nodiscard]] std::int32_t nrow() const;

    // The activity `a_i . x` of every row, for `x` of `ncol()` values, into
    // `lhs` of `nrow()` values.  Each row is summed in its row-wise order, as
    // a sequential host loop over `row_start..row_start+1` sums it, so with
    // device FMA contraction off (`--fmad=false`) the result is bitwise the
    // host's.  Not const: it writes the model's device scratch, so one
    // model serves one thread at a time.
    [[nodiscard]] bool row_activity(std::span<const double> x, std::span<double> lhs,
                                    std::string& error);

private:
    struct Impl;
    explicit DeviceModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Seconds to create this process's CUDA context on the current device, or a
// negative value with `error` set.  Measured once, the first call in a
// process; a later call finds the context made and returns its own (small)
// cost.
double init_context_seconds(std::string& error);

}  // namespace gpu_fj
