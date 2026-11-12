#include "device_internal.cuh"
#include "device_model.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <utility>
#include <vector>

namespace gpu_fj {

std::unique_ptr<DeviceModel> DeviceModel::upload(const GpuFjModel& host, std::string& error) {
    auto impl = std::make_unique<Impl>();
    const auto ncol = static_cast<std::size_t>(host.ncol);
    const auto nrow = static_cast<std::size_t>(host.nrow);
    const auto nnz = static_cast<std::size_t>(host.nnz);
    const auto ncon = static_cast<std::size_t>(host.ncon);

    // Each constraint a column is in gives it at most two breakpoints (the
    // ends of its valid range), its bounds one each.
    std::vector<std::int64_t> break_start(ncol + 1, 0);
    for (std::size_t col = 0; col < ncol; ++col) {
        std::int64_t cons = 0;
        for (std::int32_t p = host.col_start[col]; p < host.col_start[col + 1]; ++p) {
            const std::int32_t row = host.col_row[p];
            cons += host.row_con_start[row + 1] - host.row_con_start[row];
        }
        break_start[col + 1] = break_start[col] + (2 * cons) + 2;
    }
    impl->num_breaks = break_start[ncol];

    const bool uploaded =
        impl->row_start.assign(host.row_start, nrow + 1, "row_start", error) &&
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
        impl->row_con_start.assign(host.row_con_start, nrow + 1, "row_con_start", error) &&
        impl->break_start.assign(break_start.data(), ncol + 1, "break_start", error);
    if (!uploaded) {
        return nullptr;
    }
    impl->view = DeviceModelView{
        .equality_tol = host.equality_tol,
        .violation_tol = host.violation_tol,
        .ncol = host.ncol,
        .nrow = host.nrow,
        .ncon = host.ncon,
        .row_start = impl->row_start.get(),
        .row_index = impl->row_index.get(),
        .row_value = impl->row_value.get(),
        .col_start = impl->col_start.get(),
        .col_row = impl->col_row.get(),
        .col_value = impl->col_value.get(),
        .col_lower = impl->col_lower.get(),
        .col_upper = impl->col_upper.get(),
        .col_cost = impl->col_cost.get(),
        .col_is_int = impl->col_is_int.get(),
        .con_row = impl->con_row.get(),
        .con_rhs = impl->con_rhs.get(),
        .con_sense = impl->con_sense.get(),
        .row_con_start = impl->row_con_start.get(),
        .break_start = impl->break_start.get(),
    };
    return std::unique_ptr<DeviceModel>(new DeviceModel(std::move(impl)));
}

DeviceModel::DeviceModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceModel::~DeviceModel() = default;

std::int32_t DeviceModel::ncol() const {
    return impl_->view.ncol;
}
std::int32_t DeviceModel::ncon() const {
    return impl_->view.ncon;
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
