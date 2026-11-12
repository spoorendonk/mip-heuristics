#include "device_model.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <span>
#include <string>
#include <utility>

namespace gpu_fj {

namespace {

// Records a failed CUDA call in `error`, naming the call, and clears the
// thread's last error, so a later launch's `cudaGetLastError` does not
// report this failure as its own.
bool ok(cudaError_t status, const char* what, std::string& error) {
    if (status == cudaSuccess) {
        return true;
    }
    error = std::string(what) + ": " + cudaGetErrorString(status);
    static_cast<void>(cudaGetLastError());
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
        if (count == 0) {
            return true;
        }
        return ok(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), what, error) &&
               ok(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice), what, error);
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

constexpr int kRowActivityThreads = 256;

// One thread per row, summing it in row-wise order.
__global__ void row_activity_kernel(std::int32_t nrow, const std::int32_t* row_start,
                                    const std::int32_t* row_index, const double* row_value,
                                    const double* x, double* lhs) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrow) {
        return;
    }
    double sum = 0.0;
    for (std::int32_t p = row_start[row]; p < row_start[row + 1]; ++p) {
        sum += row_value[p] * x[row_index[p]];
    }
    lhs[row] = sum;
}

}  // namespace

struct DeviceModel::Impl {
    std::int32_t ncol = 0;
    std::int32_t nrow = 0;
    DeviceBuffer<std::int32_t> row_start, row_index, col_start, col_row, con_row;
    DeviceBuffer<double> row_value, col_value, col_lower, col_upper, col_cost, con_rhs;
    DeviceBuffer<std::uint8_t> col_is_int;
    DeviceBuffer<ConSense> con_sense;
    // Scratch for `row_activity`.
    DeviceBuffer<double> x, lhs;
};

std::unique_ptr<DeviceModel> DeviceModel::upload(const GpuFjModel& host, std::string& error) {
    auto impl = std::make_unique<Impl>();
    impl->ncol = host.ncol;
    impl->nrow = host.nrow;
    const auto ncol = static_cast<std::size_t>(host.ncol);
    const auto nrow = static_cast<std::size_t>(host.nrow);
    const auto nnz = static_cast<std::size_t>(host.nnz);
    const auto ncon = static_cast<std::size_t>(host.ncon);
    const bool uploaded = impl->row_start.assign(host.row_start, nrow + 1, "row_start", error) &&
                          impl->row_index.assign(host.row_index, nnz, "row_index", error) &&
                          impl->row_value.assign(host.row_value, nnz, "row_value", error) &&
                          impl->col_start.assign(host.col_start, ncol + 1, "col_start", error) &&
                          impl->col_row.assign(host.col_row, nnz, "col_row", error) &&
                          impl->col_value.assign(host.col_value, nnz, "col_value", error) &&
                          impl->col_lower.assign(host.col_lower, ncol, "col_lower", error) &&
                          impl->col_upper.assign(host.col_upper, ncol, "col_upper", error) &&
                          impl->col_cost.assign(host.col_cost, ncol, "col_cost", error) &&
                          impl->col_is_int.assign(host.col_is_int, ncol, "col_is_int", error) &&
                          impl->con_row.assign(host.con_row, ncon, "con_row", error) &&
                          impl->con_rhs.assign(host.con_rhs, ncon, "con_rhs", error) &&
                          impl->con_sense.assign(host.con_sense, ncon, "con_sense", error) &&
                          impl->x.allocate(ncol, "x", error) &&
                          impl->lhs.allocate(nrow, "lhs", error);
    if (!uploaded) {
        return nullptr;
    }
    return std::unique_ptr<DeviceModel>(new DeviceModel(std::move(impl)));
}

DeviceModel::DeviceModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceModel::~DeviceModel() = default;

std::int32_t DeviceModel::ncol() const {
    return impl_->ncol;
}
std::int32_t DeviceModel::nrow() const {
    return impl_->nrow;
}

bool DeviceModel::row_activity(std::span<const double> x, std::span<double> lhs,
                               std::string& error) {
    if (x.size() != static_cast<std::size_t>(impl_->ncol) ||
        lhs.size() != static_cast<std::size_t>(impl_->nrow)) {
        error = "row_activity: x or lhs has the wrong size";
        return false;
    }
    if (impl_->nrow == 0) {
        return true;
    }
    if (!x.empty() &&
        !ok(cudaMemcpy(impl_->x.get(), x.data(), x.size_bytes(), cudaMemcpyHostToDevice),
            "row_activity: copy x", error)) {
        return false;
    }
    const auto blocks = static_cast<unsigned>(
        (std::int64_t{impl_->nrow} + kRowActivityThreads - 1) / kRowActivityThreads);
    row_activity_kernel<<<blocks, kRowActivityThreads>>>(
        impl_->nrow, impl_->row_start.get(), impl_->row_index.get(), impl_->row_value.get(),
        impl_->x.get(), impl_->lhs.get());
    return ok(cudaGetLastError(), "row_activity: launch", error) &&
           ok(cudaMemcpy(lhs.data(), impl_->lhs.get(), lhs.size_bytes(), cudaMemcpyDeviceToHost),
              "row_activity: copy lhs", error);
}

double init_context_seconds(std::string& error) {
    const auto start = std::chrono::steady_clock::now();
    // `cudaFree(nullptr)` is the conventional no-op that forces the runtime
    // to create the primary context.
    if (!ok(cudaFree(nullptr), "init_context", error)) {
        return -1.0;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace gpu_fj
