#include "device_internal.cuh"
#include "device_model.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace gpu_fj {

namespace {

// The related-variable table on the host (`DeviceModelView::rel_start`).
struct RelatedTable {
    std::vector<std::int32_t> start, col, cell, cell_k, cell_p;

    [[nodiscard]] std::size_t bytes() const {
        return (start.size() + col.size() + cell.size() + cell_k.size() + cell_p.size()) *
               sizeof(std::int32_t);
    }
};

// `host`'s related-variable table, or false when it would hold more than
// `cap` cells.  A row of length `len` with `cons` constraints gives each of
// its columns `cons * (len - 1)` cells, so the size is known up front.
bool build_related(const GpuFjModel& host, std::int64_t cap, RelatedTable& t) {
    const auto ncol = static_cast<std::size_t>(host.ncol);
    cap = std::min<std::int64_t>(cap, std::numeric_limits<std::int32_t>::max());
    std::int64_t cells = 0;
    for (std::int32_t row = 0; row < host.nrow; ++row) {
        const std::int64_t len = host.row_start[row + 1] - host.row_start[row];
        const std::int64_t cons = host.row_con_start[row + 1] - host.row_con_start[row];
        cells += cons * len * (len - 1);
        if (cells > cap) {
            return false;
        }
    }
    // Each row cell's CSC position.  The CSC is the row storage transposed
    // row by row (`heuristic_common.h`), checked here.
    std::vector<std::int32_t> csc_of(static_cast<std::size_t>(host.nnz));
    std::vector<std::int32_t> cursor(host.col_start, host.col_start + ncol);
    for (std::int32_t row = 0; row < host.nrow; ++row) {
        for (std::int32_t q = host.row_start[row]; q < host.row_start[row + 1]; ++q) {
            const std::int32_t u = host.row_index[q];
            const std::int32_t p = cursor[static_cast<std::size_t>(u)]++;
            if (p >= host.col_start[u + 1] || host.col_row[p] != row) {
                return false;
            }
            csc_of[static_cast<std::size_t>(q)] = p;
        }
    }
    struct Cell {
        std::int32_t slot;  // the column's place in `related`
        std::int32_t p;
        std::int32_t k;
    };
    std::vector<std::int32_t> slot(ncol, -1);
    std::vector<std::int32_t> related;
    std::vector<Cell> found;
    t.start.assign(ncol + 1, 0);
    t.cell.assign(1, 0);
    t.col.reserve(static_cast<std::size_t>(cells));
    t.cell.reserve(static_cast<std::size_t>(cells) + 1);
    t.cell_k.reserve(static_cast<std::size_t>(cells));
    t.cell_p.reserve(static_cast<std::size_t>(cells));
    for (std::int32_t v = 0; v < host.ncol; ++v) {
        related.clear();
        found.clear();
        // `v`'s constraints in `move`'s order, each one's cells in row
        // order: `collect`'s work ids.
        std::int32_t k = 0;
        for (std::int32_t p = host.col_start[v]; p < host.col_start[v + 1]; ++p) {
            const std::int32_t row = host.col_row[p];
            for (std::int32_t con = host.row_con_start[row]; con < host.row_con_start[row + 1];
                 ++con, ++k) {
                for (std::int32_t q = host.row_start[row]; q < host.row_start[row + 1]; ++q) {
                    const std::int32_t u = host.row_index[q];
                    if (u == v) {
                        continue;
                    }
                    std::int32_t& at = slot[static_cast<std::size_t>(u)];
                    if (at < 0) {
                        at = static_cast<std::int32_t>(related.size());
                        related.push_back(u);
                    }
                    found.push_back(Cell{at, csc_of[static_cast<std::size_t>(q)], k});
                }
            }
        }
        // Each related column's cells in its column's order, a row's
        // constraints in turn: `gather`'s order.
        std::ranges::sort(found, [](const Cell& a, const Cell& b) {
            return a.slot != b.slot ? a.slot < b.slot : a.p != b.p ? a.p < b.p : a.k < b.k;
        });
        std::size_t e = 0;
        for (std::size_t i = 0; i < related.size(); ++i) {
            for (; e < found.size() && found[e].slot == static_cast<std::int32_t>(i); ++e) {
                t.cell_k.push_back(found[e].k);
                t.cell_p.push_back(found[e].p);
            }
            t.col.push_back(related[i]);
            t.cell.push_back(static_cast<std::int32_t>(t.cell_k.size()));
            slot[static_cast<std::size_t>(related[i])] = -1;
        }
        t.start[static_cast<std::size_t>(v) + 1] = static_cast<std::int32_t>(t.col.size());
    }
    return true;
}

}  // namespace

std::unique_ptr<DeviceModel> DeviceModel::upload(const GpuFjModel& host, std::string& error,
                                                 std::int64_t max_related_cells) {
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

    // The arrays below: row and column starts and index arrays, their
    // values, the column and constraint data, and the breakpoint starts.
    constexpr std::size_t kInt = sizeof(std::int32_t);
    const std::size_t bytes =
        ((2 * (nrow + 1) + ncol + 1 + (2 * nnz) + ncon) * kInt) + (2 * nnz * sizeof(double)) +
        (ncol * ((3 * sizeof(double)) + sizeof(std::uint8_t))) +
        (ncon * (sizeof(double) + sizeof(ConSense))) + ((ncol + 1) * sizeof(std::int64_t));
    // The table only where it fits beside the model; a model without it
    // still runs.
    RelatedTable related;
    if (host.ncol > 0 && build_related(host, max_related_cells, related) &&
        !fits(bytes + related.bytes(), 0, 0, "upload: related table", error)) {
        related = RelatedTable{};
        error.clear();
    }
    const bool uploaded =
        fits(bytes, 0, 0, "upload: model", error) &&
        impl->rel_start.assign(related.start.data(), related.start.size(), "rel_start", error) &&
        impl->rel_col.assign(related.col.data(), related.col.size(), "rel_col", error) &&
        impl->rel_cell.assign(related.cell.data(), related.cell.size(), "rel_cell", error) &&
        impl->cell_k.assign(related.cell_k.data(), related.cell_k.size(), "cell_k", error) &&
        impl->cell_p.assign(related.cell_p.data(), related.cell_p.size(), "cell_p", error) &&
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
        .rel_start = impl->rel_start.get(),
        .rel_col = impl->rel_col.get(),
        .rel_cell = impl->rel_cell.get(),
        .cell_k = impl->cell_k.get(),
        .cell_p = impl->cell_p.get(),
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
bool DeviceModel::has_related_table() const {
    return impl_->view.rel_start != nullptr;
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
