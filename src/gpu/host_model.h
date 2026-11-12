#pragma once

// The host half of the GPU feasibility jump's model (#261): the arrays a
// `GpuFjModel` points at that a `ProblemView` does not already hold, built
// once per run on the calling thread.  Plain C++ in the core, compiled and
// tested in every tree, so the conversion is gated on a machine with no
// GPU; only the device copy (`device_model.cu`) needs CUDA.

#include "gpu_fj_model.h"

#include <cstdint>
#include <vector>

struct ProblemView;

namespace gpu_fj {

class HostModel {
public:
    // `problem` must outlive this, and the `GpuFjModel` it hands out.
    explicit HostModel(const ProblemView& problem);

    // A view of the problem and of the arrays held here.
    [[nodiscard]] GpuFjModel pod() const { return pod_; }

    // False when `FjWorker` would refuse the model at its first attempt: a
    // column whose bounds, rounded as FJ rounds them, are empty, infinite
    // on the wrong side, or NaN.  `pod()` is still well formed.
    [[nodiscard]] bool legal_bounds() const { return legal_bounds_; }

private:
    std::vector<double> col_lower_;
    std::vector<double> col_upper_;
    std::vector<std::uint8_t> col_is_int_;
    std::vector<std::int32_t> con_row_;
    std::vector<double> con_rhs_;
    std::vector<ConSense> con_sense_;
    GpuFjModel pod_;
    bool legal_bounds_ = true;
};

}  // namespace gpu_fj
