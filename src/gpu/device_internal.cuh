#pragma once

// What the GPU feasibility jump's `.cu` files share and no host file sees:
// CUDA error reporting, an owning device buffer, and `DeviceModel`'s device
// arrays.

#include "device_model.h"
#include "gpu_fj_model.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

namespace gpu_fj {

// Records a failed CUDA call in `error`, naming the call, and clears the
// thread's last error, so a later launch's `cudaGetLastError` does not
// report this failure as its own.
inline bool ok(cudaError_t status, const char* what, std::string& error) {
    if (status == cudaSuccess) {
        return true;
    }
    error = std::string(what) + ": " + cudaGetErrorString(status);
    static_cast<void>(cudaGetLastError());
    return false;
}

// The free device memory an allocation leaves alone: room for the CUDA
// runtime's own later allocations and for `cudaMalloc`'s rounding.
inline constexpr std::size_t kMemoryReserve = std::size_t{64} << 20U;

// Whether `fixed` bytes plus `count` times `each` fit in the device's free
// memory less `kMemoryReserve`.  If not, or if the query fails, `error` says
// so and names `what`.
inline bool fits(std::size_t fixed, std::size_t each, std::size_t count, const char* what,
                 std::string& error) {
    std::size_t free = 0;
    std::size_t total = 0;
    if (!ok(cudaMemGetInfo(&free, &total), what, error)) {
        return false;
    }
    const std::size_t room = free > kMemoryReserve ? free - kMemoryReserve : 0;
    if (fixed <= room && (count == 0 || each <= (room - fixed) / count)) {
        return true;
    }
    error = std::string(what) + ": needs " + std::to_string(fixed) + " + " + std::to_string(count) +
            " x " + std::to_string(each) + " bytes, " + std::to_string(room) + " free";
    return false;
}

// One device allocation, freed with its owner.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allocates `count` elements and copies them from `host`.  An empty
    // array allocates nothing and leaves the pointer null.
    bool assign(const T* host, std::size_t count, const char* what, std::string& error) {
        return allocate(count, what, error) &&
               (count == 0 ||
                ok(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice), what, error));
    }
    bool allocate(std::size_t count, const char* what, std::string& error) {
        if (count == 0) {
            return true;
        }
        return ok(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), what, error);
    }

    [[nodiscard]] T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

// The model's device arrays, as raw pointers a kernel takes by value.
struct DeviceModelView {
    double equality_tol;
    double violation_tol;
    std::int32_t ncol;
    std::int32_t nrow;
    std::int32_t ncon;
    const std::int32_t* row_start;
    const std::int32_t* row_index;
    const double* row_value;
    const std::int32_t* col_start;
    const std::int32_t* col_row;
    const double* col_value;
    const double* col_lower;
    const double* col_upper;
    const double* col_cost;
    const std::uint8_t* col_is_int;
    const std::int32_t* con_row;
    const double* con_rhs;
    const ConSense* con_sense;
    const std::int32_t* row_con_start;
    // Column `j`'s jump-value breakpoints go in a climber's scratch at
    // `break_start[j] .. break_start[j + 1]`: two per constraint the column
    // is in, plus its two bounds.  `ncol + 1` starts.
    const std::int64_t* break_start;
    // The related-variable table (#277), or null `rel_start` without one.
    // Column `v`'s related columns are `rel_col[rel_start[v] ..
    // rel_start[v + 1])`, in the order a move of `v` first meets them in
    // its constraints' cells (`collect`'s).  Related entry `r`'s cells are
    // `rel_cell[r] .. rel_cell[r + 1]`, in its column's order, each the
    // constraint's place `cell_k` in `v`'s constraint list (as `move`
    // builds it) and the cell's CSC position `cell_p`.
    const std::int32_t* rel_start;
    const std::int32_t* rel_col;
    const std::int32_t* rel_cell;
    const std::int32_t* cell_k;
    const std::int32_t* cell_p;
};

struct DeviceModel::Impl {
    DeviceModelView view{};
    DeviceBuffer<std::int32_t> row_start, row_index, col_start, col_row, con_row, row_con_start;
    DeviceBuffer<std::int32_t> rel_start, rel_col, rel_cell, cell_k, cell_p;
    DeviceBuffer<double> row_value, col_value, col_lower, col_upper, col_cost, con_rhs;
    DeviceBuffer<std::uint8_t> col_is_int;
    DeviceBuffer<ConSense> con_sense;
    DeviceBuffer<std::int64_t> break_start;
    std::int64_t num_breaks = 0;
};

}  // namespace gpu_fj
