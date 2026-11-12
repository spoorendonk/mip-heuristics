#include "climber.h"
#include "device_internal.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <span>
#include <string>
#include <utility>

namespace gpu_fj {

namespace {

constexpr int kThreads = 256;

unsigned blocks_for(std::int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

// `Constraint::score`: 0 when `lhs` satisfies the constraint, else minus
// the violation.  `std::max(0., d)` is `0. < d ? d : 0.`, spelled so here,
// so a signed zero comes out as upstream's does.
__device__ double con_score(ConSense sense, double rhs, double lhs) {
    const double d = sense == ConSense::kLte ? lhs - rhs : rhs - lhs;
    return -(0.0 < d ? d : 0.0);
}

// `eq` in upstream's FJ.
__device__ bool eq(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

// A breakpoint of the jump-value scan: a value and the slope change there.
struct Break {
    double value;
    double weight;
};

// `std::pair`'s `operator<`, which `std::sort` orders the breakpoints by.
__device__ bool less(const Break& a, const Break& b) {
    return a.value < b.value || (!(b.value < a.value) && a.weight < b.weight);
}

__device__ void sift_down(Break* a, std::int64_t root, std::int64_t n) {
    while (2 * root + 1 < n) {
        std::int64_t child = 2 * root + 1;
        if (child + 1 < n && less(a[child], a[child + 1])) {
            ++child;
        }
        if (!less(a[root], a[child])) {
            return;
        }
        const Break t = a[root];
        a[root] = a[child];
        a[child] = t;
        root = child;
    }
}

// Ascending in place.  Equal breakpoints are equal in both fields, so the
// sorted sequence is the one `std::sort` gives, whatever the sort's
// stability.
__device__ void heap_sort(Break* a, std::int64_t n) {
    for (std::int64_t i = n / 2; i-- > 0;) {
        sift_down(a, i, n);
    }
    for (std::int64_t end = n; end-- > 1;) {
        const Break t = a[0];
        a[0] = a[end];
        a[end] = t;
        sift_down(a, 0, end);
    }
}

// `resetIncumbent`'s constraint loop: each constraint's LHS, summed in row
// order, and whether it is violated; with `init`'s weight of 1.
__global__ void init_constraints_kernel(DeviceModelView m, const double* x, double* lhs,
                                        double* weight, std::int32_t* violated_flag) {
    const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= m.ncon) {
        return;
    }
    const std::int32_t row = m.con_row[c];
    double sum = 0.0;
    for (std::int32_t p = m.row_start[row]; p < m.row_start[row + 1]; ++p) {
        sum += m.row_value[p] * x[m.row_index[p]];
    }
    lhs[c] = sum;
    weight[c] = 1.0;
    violated_flag[c] = con_score(m.con_sense[c], m.con_rhs[c], sum) < -m.violation_tol ? 1 : 0;
}

// `JumpMove::updateValue` for column `j`, with #139's negative-coefficient
// swap.  A column's cells are its CSC entries, each row's constraints in
// order, which is upstream's constraint order.
__device__ double jump_value(const DeviceModelView& m, std::int32_t j, const double* x,
                             const double* lhs, const double* weight, Break* buf) {
    const double incumbent = x[j];
    const double lower = m.col_lower[j];
    const double upper = m.col_upper[j];
    const bool is_int = m.col_is_int[j] != 0;
    double current_value = lower;  // can be -inf
    double current_slope = 0.0;
    std::int64_t n = 0;

    for (std::int32_t p = m.col_start[j]; p < m.col_start[j + 1]; ++p) {
        const std::int32_t row = m.col_row[p];
        const double coeff = m.col_value[p];
        for (std::int32_t c = m.row_con_start[row]; c < m.row_con_start[row + 1]; ++c) {
            // `FjWorker` splits every equality row, so a constraint has one
            // bound interval, never upstream's three for `Equal`.
            const double rhs = m.con_rhs[c];
            const bool lte = m.con_sense[c] == ConSense::kLte;
            const double bound_first = lte ? -INFINITY : rhs;
            const double bound_second = lte ? rhs : INFINITY;
            const double residual = lhs[c] - (coeff * incumbent);
            double first = (1.0 / coeff) * (bound_first - residual);
            double second = (1.0 / coeff) * (bound_second - residual);
            // #139: a negative coefficient reverses the range.
            if (coeff < 0.0) {
                const double t = first;
                first = second;
                second = t;
            }
            if (is_int) {
                first = ceil(first - m.equality_tol);
                second = floor(second + m.equality_tol);
            }
            if (first > second) {
                continue;
            }
            const double w = weight[c];
            if (first > current_value) {
                current_slope -= w;
                if (first < upper) {
                    buf[n++] = Break{first, w};
                }
            }
            if (second <= current_value) {
                current_slope += w;
            } else if (second < upper) {
                buf[n++] = Break{second, w};
            }
        }
    }
    if (isfinite(lower)) {
        buf[n++] = Break{lower, 0.0};
    }
    if (isfinite(upper)) {
        buf[n++] = Break{upper, 0.0};
    }
    heap_sort(buf, n);

    double current_score = 0.0;
    current_value = n == 0 ? incumbent : buf[0].value;
    double best_score = current_score;
    double best_value = current_value;
    for (std::int64_t i = 0; i < n; ++i) {
        current_score += (buf[i].value - current_value) * current_slope;
        current_slope += buf[i].weight;
        current_value = buf[i].value;
        if (eq(best_value, incumbent, m.equality_tol) ||
            (!eq(current_value, incumbent, m.equality_tol) && current_score < best_score)) {
            best_score = current_score;
            best_value = current_value;
        }
        if (!eq(best_value, incumbent, m.equality_tol) && current_slope >= 0.0) {
            break;
        }
    }
    return best_value;
}

// `resetMoves` for every column: the jump value, then its score, with
// #139's objective term subtracted, and whether the move is good.
__global__ void reset_moves_kernel(DeviceModelView m, double objective_weight, const double* x,
                                   const double* lhs, const double* weight, Break* breaks,
                                   double* jump, double* score, std::int32_t* good_flag) {
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= m.ncol) {
        return;
    }
    const auto col = static_cast<std::int32_t>(j);
    const double value = jump_value(m, col, x, lhs, weight, breaks + m.break_start[col]);
    const double delta = value - x[col];
    double s = 0.0;
    s -= objective_weight * m.col_cost[col] * delta;
    for (std::int32_t p = m.col_start[col]; p < m.col_start[col + 1]; ++p) {
        const std::int32_t row = m.col_row[p];
        const double coeff = m.col_value[p];
        for (std::int32_t c = m.row_con_start[row]; c < m.row_con_start[row + 1]; ++c) {
            const double candidate = lhs[c] + (coeff * delta);
            s += weight[c] * (con_score(m.con_sense[c], m.con_rhs[c], candidate) -
                              con_score(m.con_sense[c], m.con_rhs[c], lhs[c]));
        }
    }
    jump[col] = value;
    score[col] = s;
    good_flag[col] = s > 0.0 ? 1 : 0;
}

// The flagged indices, ascending, into `list`, each one's place into `pos`
// (-1 unflagged), and their number into `*count`: the order upstream's
// sequential push-backs give at `init`.  `offset` is the flags' exclusive
// prefix sum.
__global__ void compact_kernel(std::int64_t n, const std::int32_t* flag, const std::int32_t* offset,
                               std::int32_t* list, std::int32_t* pos, std::int32_t* count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (flag[i] != 0) {
        list[offset[i]] = static_cast<std::int32_t>(i);
        pos[i] = offset[i];
    } else {
        pos[i] = -1;
    }
    if (i == n - 1) {
        *count = offset[i] + flag[i];
    }
}

template <typename T>
bool to_host(std::vector<T>& out, const T* device, std::size_t n, const char* what,
             std::string& error) {
    out.resize(n);
    return n == 0 ||
           ok(cudaMemcpy(out.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost), what, error);
}

}  // namespace

struct Climber::Impl {
    DeviceModelView m{};
    double objective_weight = 0.0;
    DeviceBuffer<double> x, lhs, weight, jump, score;
    DeviceBuffer<std::int32_t> violated_flag, violated, violated_pos, good_flag, good, good_pos;
    DeviceBuffer<std::int32_t> offset, counts;  // counts: violated, good
    DeviceBuffer<Break> breaks;
    DeviceBuffer<unsigned char> scan_temp;
    std::size_t scan_temp_bytes = 0;

    // Compacts `flag` (`n` entries) into `list`, `pos` and `counts[slot]`.
    bool compact(std::int64_t n, std::int32_t* flag, std::int32_t* list, std::int32_t* pos,
                 int slot, const char* what, std::string& error) {
        if (n == 0) {
            return ok(cudaMemset(counts.get() + slot, 0, sizeof(std::int32_t)), what, error);
        }
        std::size_t bytes = scan_temp_bytes;
        return ok(cub::DeviceScan::ExclusiveSum(scan_temp.get(), bytes, flag, offset.get(),
                                                static_cast<int>(n)),
                  what, error) &&
               launched(compact_kernel, n, what, error, n, flag, offset.get(), list, pos,
                        counts.get() + slot);
    }

    template <typename Kernel, typename... Args>
    static bool launched(Kernel kernel, std::int64_t n, const char* what, std::string& error,
                         Args... args) {
        kernel<<<blocks_for(n), kThreads>>>(args...);
        return ok(cudaGetLastError(), what, error);
    }

    bool reset_moves(std::string& error) {
        if (m.ncol == 0) {
            return compact(0, nullptr, nullptr, nullptr, 1, "reset_moves", error);
        }
        return launched(reset_moves_kernel, m.ncol, "reset_moves", error, m, objective_weight,
                        x.get(), lhs.get(), weight.get(), breaks.get(), jump.get(), score.get(),
                        good_flag.get()) &&
               compact(m.ncol, good_flag.get(), good.get(), good_pos.get(), 1, "good set", error);
    }
};

std::unique_ptr<Climber> Climber::create(const DeviceModel& model, std::string& error) {
    auto impl = std::make_unique<Impl>();
    impl->m = model.impl_->view;
    const auto ncol = static_cast<std::size_t>(impl->m.ncol);
    const auto ncon = static_cast<std::size_t>(impl->m.ncon);
    const std::size_t nmax = std::max(ncol, ncon);
    // The scan's scratch, sized for the longer of the two flag arrays.
    if (nmax > 0 && !ok(cub::DeviceScan::ExclusiveSum(
                            nullptr, impl->scan_temp_bytes, static_cast<std::int32_t*>(nullptr),
                            static_cast<std::int32_t*>(nullptr), static_cast<int>(nmax)),
                        "scan size", error)) {
        return nullptr;
    }
    const bool allocated =
        impl->x.allocate(ncol, "x", error) && impl->lhs.allocate(ncon, "lhs", error) &&
        impl->weight.allocate(ncon, "weight", error) && impl->jump.allocate(ncol, "jump", error) &&
        impl->score.allocate(ncol, "score", error) &&
        impl->violated_flag.allocate(ncon, "violated_flag", error) &&
        impl->violated.allocate(ncon, "violated", error) &&
        impl->violated_pos.allocate(ncon, "violated_pos", error) &&
        impl->good_flag.allocate(ncol, "good_flag", error) &&
        impl->good.allocate(ncol, "good", error) &&
        impl->good_pos.allocate(ncol, "good_pos", error) &&
        impl->offset.allocate(nmax, "offset", error) && impl->counts.allocate(2, "counts", error) &&
        impl->breaks.allocate(static_cast<std::size_t>(model.impl_->num_breaks), "breaks", error) &&
        impl->scan_temp.allocate(impl->scan_temp_bytes, "scan_temp", error);
    if (!allocated) {
        return nullptr;
    }
    return std::unique_ptr<Climber>(new Climber(std::move(impl)));
}

Climber::Climber(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Climber::~Climber() = default;

bool Climber::init(std::span<const double> start, std::string& error) {
    Impl& s = *impl_;
    if (start.size() != static_cast<std::size_t>(s.m.ncol)) {
        error = "init: start has the wrong size";
        return false;
    }
    s.objective_weight = 0.0;
    if (!start.empty() &&
        !ok(cudaMemcpy(s.x.get(), start.data(), start.size_bytes(), cudaMemcpyHostToDevice),
            "init: copy start", error)) {
        return false;
    }
    if (s.m.ncon > 0 &&
        !Impl::launched(init_constraints_kernel, s.m.ncon, "init: constraints", error, s.m,
                        s.x.get(), s.lhs.get(), s.weight.get(), s.violated_flag.get())) {
        return false;
    }
    return s.compact(s.m.ncon, s.violated_flag.get(), s.violated.get(), s.violated_pos.get(), 0,
                     "violated set", error) &&
           s.reset_moves(error);
}

bool Climber::reset_moves(double objective_weight, std::string& error) {
    impl_->objective_weight = objective_weight;
    return impl_->reset_moves(error);
}

bool Climber::read(State& state, std::string& error) const {
    const Impl& s = *impl_;
    const auto ncol = static_cast<std::size_t>(s.m.ncol);
    const auto ncon = static_cast<std::size_t>(s.m.ncon);
    std::vector<std::int32_t> counts;
    std::vector<std::int32_t> violated;
    std::vector<std::int32_t> good;
    const bool copied =
        to_host(counts, s.counts.get(), 2, "read: counts", error) &&
        to_host(state.x, s.x.get(), ncol, "read: x", error) &&
        to_host(state.lhs, s.lhs.get(), ncon, "read: lhs", error) &&
        to_host(state.weight, s.weight.get(), ncon, "read: weight", error) &&
        to_host(violated, s.violated.get(), ncon, "read: violated", error) &&
        to_host(state.violated_pos, s.violated_pos.get(), ncon, "read: violated_pos", error) &&
        to_host(state.jump, s.jump.get(), ncol, "read: jump", error) &&
        to_host(state.score, s.score.get(), ncol, "read: score", error) &&
        to_host(good, s.good.get(), ncol, "read: good", error) &&
        to_host(state.good_pos, s.good_pos.get(), ncol, "read: good_pos", error);
    if (!copied) {
        return false;
    }
    violated.resize(static_cast<std::size_t>(counts[0]));
    good.resize(static_cast<std::size_t>(counts[1]));
    state.violated = std::move(violated);
    state.good = std::move(good);
    state.objective_weight = s.objective_weight;
    return true;
}

}  // namespace gpu_fj
