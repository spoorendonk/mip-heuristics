#pragma once

// The model the GPU feasibility jump reads (#210, #261), as raw host
// pointers and sizes.  nvcc compiles every file that includes this, so it
// includes nothing but <cstdint>: the C++23 headers the rest of the
// heuristics use (`heuristic_context.h`, `std::expected`) must never reach
// a `.cu` translation unit.  The host side fills one from a `ProblemView`
// (`gpu_fj_host.h`); `DeviceModel` copies it to the device.
//
// Indices are 32-bit: `FjWorker` refuses a `HIGHSINT64` build, and the
// host side asserts `HighsInt` is 32 bits wide.

#include <cstdint>

namespace gpu_fj {

// One FJ constraint, as `FjWorker` hands it to `addConstraint`: a row with a
// finite lower bound gives a `>=` constraint, one with a finite upper bound
// a `<=` constraint, so a ranged or equality row gives both and a free row
// none.  Its coefficients are the row's, read through `row`.
enum class ConSense : std::uint8_t { kGte = 0, kLte = 1 };

struct GpuFjModel {
    std::int32_t ncol = 0;
    std::int32_t nrow = 0;
    std::int64_t nnz = 0;

    // Row-wise matrix (`ProblemView::ar_*`), `nrow + 1` starts.
    const std::int32_t* row_start = nullptr;
    const std::int32_t* row_index = nullptr;
    const double* row_value = nullptr;
    // Column-wise matrix (`ProblemView::csc`), `ncol + 1` starts.
    const std::int32_t* col_start = nullptr;
    const std::int32_t* col_row = nullptr;
    const double* col_value = nullptr;

    // Per column: bounds as FJ takes them (an integer column's rounded
    // inward by `feastol`, as `FjWorker` rounds them), the minimisation
    // cost, and 1 for an integer column.
    const double* col_lower = nullptr;
    const double* col_upper = nullptr;
    const double* col_cost = nullptr;
    const std::uint8_t* col_is_int = nullptr;

    // The FJ constraints, `ncon` of them, in `FjWorker`'s order.
    std::int32_t ncon = 0;
    const std::int32_t* con_row = nullptr;
    const double* con_rhs = nullptr;
    const ConSense* con_sense = nullptr;
};

}  // namespace gpu_fj
