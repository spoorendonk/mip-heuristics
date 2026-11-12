#include "climber.h"
#include "device_internal.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <limits>
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

// Ascending in place.  Breakpoints that compare equal are equal in both
// fields up to the sign of a zero value, so the sorted sequence is the one
// `std::sort` gives, whatever either sort's stability, except that a tie
// of `-0.0` and `+0.0` may come out in either order (and the jump value
// with it, as the zero of the other sign).
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

// `resetMoves` for column `col`: the jump value, then its score, with
// #139's objective term subtracted.
__device__ void reset_move(const DeviceModelView& m, std::int32_t col, double objective_weight,
                           const double* x, const double* lhs, const double* weight, Break* breaks,
                           double* jump, double* score) {
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
}

// `resetMoves` for every column, and whether each move is good.
__global__ void reset_moves_kernel(DeviceModelView m, double objective_weight, const double* x,
                                   const double* lhs, const double* weight, Break* breaks,
                                   double* jump, double* score, std::int32_t* good_flag) {
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= m.ncol) {
        return;
    }
    const auto col = static_cast<std::int32_t>(j);
    reset_move(m, col, objective_weight, x, lhs, weight, breaks, jump, score);
    good_flag[col] = score[col] > 0.0 ? 1 : 0;
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

// ---------------------------------------------------------------------------
// The step loop (#263): upstream's `solve`, `selectScalarVariable`,
// `updateWeights` and `doVariableMove`, on one persistent block.  Thread 0
// takes every decision that draws from the random stream or edits a list
// (the violated and good sets) in upstream's order; the block spreads the
// moved variable's (constraint, cell) work and the neighbours' score
// updates.  A neighbour's score change is gathered over its own column
// against the touched constraints, never summed with float atomics, so a
// climber is deterministic.
//
// Three deviations from upstream, none in what a step computes:
// - Upstream adds each `modifyMove` (or weight-bump) term to a neighbour's
//   score constraint by constraint and calls `updateGoodMoves` after each;
//   here a neighbour's terms are summed over its column first, then added,
//   and its good-set membership is updated once, neighbours in the order
//   their first cell comes in the moved constraints' cells.  So a score can
//   differ from upstream's in rounding, and the good list's order differs,
//   which is why a GPU seed never replays a CPU one's draws.
// - Where upstream recomputes every move (the objective-weight bump and a
//   weight rescale) and adds each variable to the good set in turn, the
//   good set is rebuilt in index order.
// - The random stream is splitmix64, not `std::mt19937`.

constexpr int kStepThreads = 256;
// `FeasibilityJumpSolver`'s constants: `maxMovesToEvaluate`,
// `randomVarProbability`, `randomCellProbability`,
// `kMinRelativeObjectiveImprovement`, the weight decay `FjWorker` leaves at
// its default, and the weight above which every weight is rescaled.
constexpr std::int32_t kMaxMovesToEvaluate = 25;
constexpr double kRandomVarProbability = 0.001;
constexpr double kRandomCellProbability = 0.01;
constexpr double kMinRelativeObjectiveImprovement = 1e-4;
constexpr double kWeightUpdateDecay = 1.0;
constexpr double kRescaleAbove = 1.0e20;
constexpr double kRescaleBy = 1.0e-20;
constexpr std::uint32_t kNoOwner = 0xFFFFFFFFU;

// A climber's scalars: upstream's solver members, plus the set sizes.
struct Scalars {
    double objective_weight;
    double weight_increment;
    double objective;  // `incumbentObjective`
    double best_objective;
    std::uint64_t total_effort;
    std::uint64_t effort_at_last_callback;
    std::uint64_t effort_at_last_improvement;
    std::uint64_t best_violation;
    std::uint64_t bumps;
    std::uint64_t steps;
    std::uint64_t rng;
    std::int32_t n_violated;
    std::int32_t n_good;
    std::int32_t last_var;
    std::int32_t improved;
    std::int32_t finished;
};

// A climber's device arrays, by value into the step kernel.
struct ClimberView {
    double* x;
    double* lhs;
    double* weight;
    double* jump;
    double* score;
    double* best_x;
    std::int32_t* violated;
    std::int32_t* violated_pos;
    std::int32_t* good;
    std::int32_t* good_pos;
    Break* breaks;
    // Step scratch.  `touch[c]` is constraint `c`'s place in the touched
    // list (-1 when untouched); `owner[j]` the first work id that names
    // column `j` (`kNoOwner` otherwise); both are restored after each use.
    std::int32_t* touch;
    std::uint32_t* owner;
    std::int32_t* list;  // the moved variable's constraints
    double* coef;        // its coefficient in each
    double* old_lhs;
    double* new_lhs;
    std::int32_t* work_off;  // each touched constraint's first work id
    std::int32_t* nbr;       // the distinct columns of the touched cells
    Scalars* s;
};

// splitmix64.
__device__ std::uint64_t next(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

// Uniform on [0, 1), as `std::uniform_real_distribution<double>(0., 1.)`.
__device__ double uniform(std::uint64_t& state) {
    return static_cast<double>(next(state) >> 11U) * 0x1.0p-53;
}

__device__ std::int32_t draw(std::uint64_t& state, std::int32_t n) {
    return static_cast<std::int32_t>(next(state) % static_cast<std::uint64_t>(n));
}

__device__ std::int32_t row_len(const DeviceModelView& m, std::int32_t row) {
    return m.row_start[row + 1] - m.row_start[row];
}

__device__ double score_of(const DeviceModelView& m, std::int32_t c, double lhs) {
    return con_score(m.con_sense[c], m.con_rhs[c], lhs);
}

using BlockScan = cub::BlockScan<std::int32_t, kStepThreads>;

// A block-wide exclusive scan over `n` items in tiles: `emit(i, offset,
// value)` for each, offsets running across tiles.  Every thread calls it
// and gets the total.
template <typename Value, typename Emit>
__device__ std::int32_t scan_each(std::int32_t n, BlockScan::TempStorage& tmp, Value value,
                                  Emit emit) {
    std::int32_t carry = 0;
    for (std::int32_t base = 0; base < n; base += kStepThreads) {
        const std::int32_t i = base + static_cast<std::int32_t>(threadIdx.x);
        const std::int32_t v = i < n ? value(i) : 0;
        std::int32_t off = 0;
        std::int32_t total = 0;
        BlockScan(tmp).ExclusiveSum(v, off, total);
        if (i < n) {
            emit(i, carry + off, v);
        }
        __syncthreads();
        carry += total;
    }
    __syncthreads();
    return carry;
}

// The column work id `w` names: cell `w - work_off[k]` of the `k`th of the
// `m` constraints `cons`, found by binary search over `work_off` (cuOpt's
// work-id mapping, at block scale).  Every constraint has a cell, so the
// offsets strictly increase.
__device__ std::int32_t work_col(const DeviceModelView& m, const ClimberView& c,
                                 const std::int32_t* cons, std::int32_t n, std::int32_t w) {
    std::int32_t lo = 0;
    std::int32_t hi = n - 1;
    while (lo < hi) {
        const std::int32_t mid = (lo + hi + 1) / 2;
        if (c.work_off[mid] <= w) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const std::int32_t row = m.con_row[cons[lo]];
    return m.row_index[m.row_start[row] + (w - c.work_off[lo])];
}

// The distinct columns of the cells of constraints `cons[0..n)` (`total`
// cells, `work_off` set), but `skip`, into `nbr` in first-cell order;
// returns their number.
__device__ std::int32_t collect(const DeviceModelView& m, const ClimberView& c,
                                const std::int32_t* cons, std::int32_t n, std::int32_t total,
                                std::int32_t skip, BlockScan::TempStorage& tmp) {
    for (std::int32_t w = static_cast<std::int32_t>(threadIdx.x); w < total; w += kStepThreads) {
        const std::int32_t u = work_col(m, c, cons, n, w);
        if (u != skip) {
            atomicMin(c.owner + u, static_cast<std::uint32_t>(w));
        }
    }
    __syncthreads();
    const std::int32_t count = scan_each(
        total, tmp,
        [&](std::int32_t w) {
            const std::int32_t u = work_col(m, c, cons, n, w);
            return u != skip && c.owner[u] == static_cast<std::uint32_t>(w) ? 1 : 0;
        },
        [&](std::int32_t w, std::int32_t off, std::int32_t first) {
            if (first != 0) {
                c.nbr[off] = work_col(m, c, cons, n, w);
            }
        });
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < count; i += kStepThreads) {
        c.owner[c.nbr[i]] = kNoOwner;
    }
    __syncthreads();
    return count;
}

// `term(k, con, coeff)` summed over column `u`'s cells in touched
// constraints, `k` being the constraint's place in the touched list.
template <typename Term>
__device__ double gather(const DeviceModelView& m, const ClimberView& c, std::int32_t u,
                         Term term) {
    double d = 0.0;
    for (std::int32_t p = m.col_start[u]; p < m.col_start[u + 1]; ++p) {
        const std::int32_t row = m.col_row[p];
        for (std::int32_t con = m.row_con_start[row]; con < m.row_con_start[row + 1]; ++con) {
            const std::int32_t k = c.touch[con];
            if (k >= 0) {
                d += term(k, con, m.col_value[p]);
            }
        }
    }
    return d;
}

// `updateGoodMoves`, thread 0 only.
__device__ void update_good(const ClimberView& c, Scalars& s, std::int32_t u) {
    const bool good = c.score[u] > 0.0;
    const std::int32_t pos = c.good_pos[u];
    if (good && pos == -1) {
        c.good_pos[u] = s.n_good;
        c.good[s.n_good++] = u;
    } else if (!good && pos != -1) {
        const std::int32_t last = c.good[--s.n_good];
        c.good[pos] = last;
        c.good_pos[last] = pos;
        c.good_pos[u] = -1;
    }
}

// `setValue`'s violated-set update for constraint `con` at `lhs`, thread 0
// only.
__device__ void update_violated(const DeviceModelView& m, const ClimberView& c, Scalars& s,
                                std::int32_t con, double lhs) {
    const double cost = score_of(m, con, lhs);
    const std::int32_t pos = c.violated_pos[con];
    if (cost < -m.violation_tol && pos == -1) {
        c.violated_pos[con] = s.n_violated;
        c.violated[s.n_violated++] = con;
    }
    if (cost >= -m.violation_tol && pos != -1) {
        const std::int32_t last = c.violated[--s.n_violated];
        c.violated[pos] = last;
        c.violated_pos[last] = pos;
        c.violated_pos[con] = -1;
    }
}

// The good set from scratch, in index order.
__device__ void rebuild_good(const DeviceModelView& m, const ClimberView& c, Scalars& s,
                             BlockScan::TempStorage& tmp) {
    const std::int32_t n = scan_each(
        m.ncol, tmp, [&](std::int32_t j) { return c.score[j] > 0.0 ? 1 : 0; },
        [&](std::int32_t j, std::int32_t off, std::int32_t good) {
            c.good_pos[j] = good != 0 ? off : -1;
            if (good != 0) {
                c.good[off] = j;
            }
        });
    if (threadIdx.x == 0) {
        s.n_good = n;
    }
    __syncthreads();
}

// `updateWeights`, every thread.
__device__ void update_weights(const DeviceModelView& m, const ClimberView& c, Scalars& s,
                               BlockScan::TempStorage& tmp) {
    __shared__ int rescale;
    const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
    const double inc = s.weight_increment;
    const std::int32_t nv = s.n_violated;
    if (tid == 0) {
        ++s.bumps;
        rescale = 0;
        if (nv == 0) {
            s.objective_weight += inc;
            rescale = s.objective_weight > kRescaleAbove ? 1 : 0;
        }
    }
    __syncthreads();
    std::uint64_t dt = 0;
    if (nv == 0) {
        for (std::int32_t j = tid; j < m.ncol; j += kStepThreads) {
            c.score[j] -= inc * m.col_cost[j] * (c.jump[j] - c.x[j]);
        }
        __syncthreads();
        rebuild_good(m, c, s, tmp);
        dt = static_cast<std::uint64_t>(m.ncol);
    } else {
        for (std::int32_t k = tid; k < nv; k += kStepThreads) {
            const std::int32_t con = c.violated[k];
            const double w = c.weight[con] + inc;
            c.weight[con] = w;
            if (w > kRescaleAbove) {
                rescale = 1;
            }
            c.touch[con] = k;
        }
        __syncthreads();
        const std::int32_t total = scan_each(
            nv, tmp, [&](std::int32_t k) { return row_len(m, m.con_row[c.violated[k]]); },
            [&](std::int32_t k, std::int32_t off, std::int32_t) { c.work_off[k] = off; });
        const std::int32_t count = collect(m, c, c.violated, nv, total, -1, tmp);
        for (std::int32_t i = tid; i < count; i += kStepThreads) {
            const std::int32_t u = c.nbr[i];
            const double shift = c.jump[u] - c.x[u];
            c.score[u] += gather(m, c, u, [&](std::int32_t, std::int32_t con, double coeff) {
                const double lhs = c.lhs[con];
                return inc * (score_of(m, con, lhs + (coeff * shift)) - score_of(m, con, lhs));
            });
        }
        __syncthreads();
        if (tid == 0) {
            for (std::int32_t i = 0; i < count; ++i) {
                update_good(c, s, c.nbr[i]);
            }
        }
        for (std::int32_t k = tid; k < nv; k += kStepThreads) {
            c.touch[c.violated[k]] = -1;
        }
        __syncthreads();
        dt = static_cast<std::uint64_t>(total);
    }
    if (tid == 0) {
        s.weight_increment /= kWeightUpdateDecay;
    }
    __syncthreads();
    if (rescale != 0) {
        if (tid == 0) {
            s.weight_increment *= kRescaleBy;
            s.objective_weight *= kRescaleBy;
        }
        for (std::int32_t con = tid; con < m.ncon; con += kStepThreads) {
            c.weight[con] *= kRescaleBy;
        }
        __syncthreads();
        for (std::int32_t j = tid; j < m.ncol; j += kStepThreads) {
            reset_move(m, j, s.objective_weight, c.x, c.lhs, c.weight, c.breaks, c.jump, c.score);
        }
        __syncthreads();
        rebuild_good(m, c, s, tmp);
        // `constraints.size()`, then each `resetMoves`'s cells.
        const auto cells = static_cast<std::uint64_t>(m.break_start[m.ncol] - (2 * m.ncol)) / 2;
        dt += static_cast<std::uint64_t>(m.ncon) + cells;
    }
    if (tid == 0) {
        s.total_effort += dt;
    }
    __syncthreads();
}

// `selectScalarVariable` while the good set is not empty, thread 0 only.
__device__ std::int32_t select_good(const ClimberView& c, Scalars& s) {
    if (uniform(s.rng) < kRandomVarProbability) {
        return c.good[draw(s.rng, s.n_good)];
    }
    const std::int32_t sample = min(kMaxMovesToEvaluate, s.n_good);
    s.total_effort += static_cast<std::uint64_t>(sample);
    double best_score = -INFINITY;
    std::int32_t best_var = -1;
    for (std::int32_t i = 0; i < sample; ++i) {
        const std::int32_t u = c.good[draw(s.rng, s.n_good)];
        if (c.score[u] > best_score) {
            best_score = c.score[u];
            best_var = u;
        }
    }
    return best_var;
}

// `selectScalarVariable` at a local minimum, after the weights moved,
// thread 0 only.
__device__ std::int32_t select_at_minimum(const DeviceModelView& m, const ClimberView& c,
                                          Scalars& s) {
    if (s.n_violated == 0) {
        return draw(s.rng, m.ncol);
    }
    const std::int32_t row = m.con_row[c.violated[draw(s.rng, s.n_violated)]];
    const std::int32_t start = m.row_start[row];
    if (uniform(s.rng) < kRandomCellProbability) {
        return m.row_index[start + draw(s.rng, row_len(m, row))];
    }
    double best_score = -INFINITY;
    std::int32_t best_var = -1;
    for (std::int32_t p = start; p < m.row_start[row + 1]; ++p) {
        const std::int32_t u = m.row_index[p];
        if (c.score[u] > best_score) {
            best_score = c.score[u];
            best_var = u;
        }
    }
    return best_var;
}

// `doVariableMove(v)`, every thread: `setValue` with each neighbour's
// `modifyMove` and `updateGoodMoves`, then `resetMoves(v)`.
__device__ void move(const DeviceModelView& m, const ClimberView& c, Scalars& s,
                     BlockScan::TempStorage& tmp, std::int32_t v) {
    __shared__ double delta_shared;
    const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
    if (tid == 0) {
        const double value = c.jump[v];
        const double d = value - c.x[v];
        c.x[v] = value;
        s.objective += m.col_cost[v] * d;
        delta_shared = d;
    }
    __syncthreads();
    const double delta = delta_shared;
    const std::int32_t col_start = m.col_start[v];
    // `v`'s constraints, in its cells' order.
    const std::int32_t deg = scan_each(
        m.col_start[v + 1] - col_start, tmp,
        [&](std::int32_t i) {
            const std::int32_t row = m.col_row[col_start + i];
            return m.row_con_start[row + 1] - m.row_con_start[row];
        },
        [&](std::int32_t i, std::int32_t off, std::int32_t n) {
            const std::int32_t row = m.col_row[col_start + i];
            for (std::int32_t t = 0; t < n; ++t) {
                c.list[off + t] = m.row_con_start[row] + t;
                c.coef[off + t] = m.col_value[col_start + i];
            }
        });
    for (std::int32_t k = tid; k < deg; k += kStepThreads) {
        const std::int32_t con = c.list[k];
        const double old_lhs = c.lhs[con];
        const double new_lhs = old_lhs + (c.coef[k] * delta);
        c.lhs[con] = new_lhs;
        c.old_lhs[k] = old_lhs;
        c.new_lhs[k] = new_lhs;
        c.touch[con] = k;
    }
    __syncthreads();
    if (tid == 0) {
        for (std::int32_t k = 0; k < deg; ++k) {
            update_violated(m, c, s, c.list[k], c.new_lhs[k]);
        }
    }
    const std::int32_t total = scan_each(
        deg, tmp, [&](std::int32_t k) { return row_len(m, m.con_row[c.list[k]]); },
        [&](std::int32_t k, std::int32_t off, std::int32_t) { c.work_off[k] = off; });
    const std::int32_t count = collect(m, c, c.list, deg, total, v, tmp);
    for (std::int32_t i = tid; i < count; i += kStepThreads) {
        const std::int32_t u = c.nbr[i];
        const double shift = c.jump[u] - c.x[u];
        c.score[u] += gather(m, c, u, [&](std::int32_t k, std::int32_t con, double coeff) {
            const double w = c.weight[con];
            const double old_lhs = c.old_lhs[k];
            const double new_lhs = c.new_lhs[k];
            const double old_term =
                w * (score_of(m, con, old_lhs + (coeff * shift)) - score_of(m, con, old_lhs));
            const double new_term =
                w * (score_of(m, con, new_lhs + (coeff * shift)) - score_of(m, con, new_lhs));
            return new_term - old_term;
        });
    }
    __syncthreads();
    if (tid == 0) {
        for (std::int32_t i = 0; i < count; ++i) {
            update_good(c, s, c.nbr[i]);
        }
        reset_move(m, v, s.objective_weight, c.x, c.lhs, c.weight, c.breaks, c.jump, c.score);
        update_good(c, s, v);
        // `setValue`'s cells of each constraint, then `resetMoves`'s own.
        s.total_effort += static_cast<std::uint64_t>(total) + static_cast<std::uint64_t>(deg);
        s.last_var = v;
        ++s.steps;
    }
    for (std::int32_t k = tid; k < deg; k += kStepThreads) {
        c.touch[c.list[k]] = -1;
    }
    __syncthreads();
}

enum Control : int { kStep = 0, kStop = 1, kSaveBest = 2 };

// The top of `solve`'s loop, thread 0 only: the effort callback, the
// improvement bookkeeping, and the feasible-improvement callback.
__device__ int top_of_loop(const DeviceModelView& m, Scalars& s, std::uint64_t taken,
                           std::uint64_t max_steps) {
    if (taken >= max_steps) {
        return kStop;
    }
    if (s.total_effort - s.effort_at_last_callback > kCallbackEffort) {
        s.effort_at_last_callback = s.total_effort;
        return kStop;
    }
    if (static_cast<std::uint64_t>(s.n_violated) < s.best_violation) {
        s.effort_at_last_improvement = s.total_effort;
        s.best_violation = static_cast<std::uint64_t>(s.n_violated);
    }
    if (s.n_violated == 0 && s.objective < s.best_objective) {
        const double gain = (s.best_objective - s.objective) / fmax(1.0, fabs(s.objective));
        if (gain > kMinRelativeObjectiveImprovement) {
            s.effort_at_last_improvement = s.total_effort;
            s.best_objective = s.objective;
            s.effort_at_last_callback = s.total_effort;
            s.improved = 1;
            return kSaveBest;
        }
    }
    if (m.ncol == 0) {
        s.finished = 1;
        return kStop;
    }
    return kStep;
}

// `solve`'s loop up to its next callback, or `max_steps` steps.
__global__ void __launch_bounds__(kStepThreads)
    step_kernel(DeviceModelView m, ClimberView c, std::uint64_t max_steps) {
    __shared__ Scalars s;
    __shared__ BlockScan::TempStorage tmp;
    __shared__ int control;
    __shared__ std::int32_t var;
    const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
    if (tid == 0) {
        s = *c.s;
        s.improved = 0;
    }
    __syncthreads();
    for (std::uint64_t taken = 0;; ++taken) {
        if (tid == 0) {
            control = top_of_loop(m, s, taken, max_steps);
        }
        __syncthreads();
        if (control == kSaveBest) {
            for (std::int32_t j = tid; j < m.ncol; j += kStepThreads) {
                c.best_x[j] = c.x[j];
            }
        }
        if (control != kStep) {
            break;
        }
        if (tid == 0) {
            var = s.n_good > 0 ? select_good(c, s) : -1;
        }
        __syncthreads();
        if (var < 0) {
            update_weights(m, c, s, tmp);
            if (tid == 0) {
                var = select_at_minimum(m, c, s);
            }
            __syncthreads();
        }
        move(m, c, s, tmp, var);
    }
    __syncthreads();
    if (tid == 0) {
        *c.s = s;
    }
}

// `init`'s scalars: `resetIncumbent`'s objective, summed in column order,
// and the effort of `nNonzeros` plus each `resetMoves`'s cells.
__global__ void init_scalars_kernel(DeviceModelView m, const double* x, std::uint64_t seed,
                                    std::uint64_t effort, Scalars* s) {
    double objective = 0.0;
    for (std::int32_t j = 0; j < m.ncol; ++j) {
        objective += m.col_cost[j] * x[j];
    }
    *s = Scalars{};
    s->weight_increment = 1.0;
    s->objective = objective;
    s->best_objective = INFINITY;
    s->total_effort = effort;
    s->best_violation = ~std::uint64_t{0};
    s->rng = seed;
    s->last_var = -1;
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
    std::uint64_t cells = 0;  // upstream's `nNonzeros`
    double objective_weight = 0.0;
    DeviceBuffer<double> x, lhs, weight, jump, score, best_x, coef, old_lhs, new_lhs;
    DeviceBuffer<std::int32_t> violated_flag, violated, violated_pos, good_flag, good, good_pos;
    DeviceBuffer<std::int32_t> offset, touch, list, work_off, nbr;
    DeviceBuffer<std::uint32_t> owner;
    DeviceBuffer<Scalars> scalars;
    DeviceBuffer<Break> breaks;
    DeviceBuffer<unsigned char> scan_temp;
    std::size_t scan_temp_bytes = 0;

    // Compacts `flag` (`n` entries) into `list`, `pos` and `*count`.
    bool compact(std::int64_t n, std::int32_t* flag, std::int32_t* list, std::int32_t* pos,
                 std::int32_t* count, const char* what, std::string& error) {
        if (n == 0) {
            return ok(cudaMemset(count, 0, sizeof(std::int32_t)), what, error);
        }
        std::size_t bytes = scan_temp_bytes;
        return ok(cub::DeviceScan::ExclusiveSum(scan_temp.get(), bytes, flag, offset.get(),
                                                static_cast<int>(n)),
                  what, error) &&
               launched(compact_kernel, n, what, error, n, flag, offset.get(), list, pos, count);
    }

    template <typename Kernel, typename... Args>
    static bool launched(Kernel kernel, std::int64_t n, const char* what, std::string& error,
                         Args... args) {
        kernel<<<blocks_for(n), kThreads>>>(args...);
        return ok(cudaGetLastError(), what, error);
    }

    // The objective weight lives in `scalars` on the device; this copy is
    // the host's, set with it.
    bool set_objective_weight(double w, std::string& error) {
        objective_weight = w;
        return ok(cudaMemcpy(&scalars.get()->objective_weight, &w, sizeof(double),
                             cudaMemcpyHostToDevice),
                  "objective weight", error);
    }

    bool reset_moves(std::string& error) {
        std::int32_t* n_good = &scalars.get()->n_good;
        if (m.ncol == 0) {
            return compact(0, nullptr, nullptr, nullptr, n_good, "reset_moves", error);
        }
        return launched(reset_moves_kernel, m.ncol, "reset_moves", error, m, objective_weight,
                        x.get(), lhs.get(), weight.get(), breaks.get(), jump.get(), score.get(),
                        good_flag.get()) &&
               compact(m.ncol, good_flag.get(), good.get(), good_pos.get(), n_good, "good set",
                       error);
    }

    [[nodiscard]] ClimberView view() const {
        return ClimberView{x.get(),       lhs.get(),      weight.get(),   jump.get(),
                           score.get(),   best_x.get(),   violated.get(), violated_pos.get(),
                           good.get(),    good_pos.get(), breaks.get(),   touch.get(),
                           owner.get(),   list.get(),     coef.get(),     old_lhs.get(),
                           new_lhs.get(), work_off.get(), nbr.get(),      scalars.get()};
    }

    bool read_scalars(Scalars& out, std::string& error) const {
        return ok(cudaMemcpy(&out, scalars.get(), sizeof(Scalars), cudaMemcpyDeviceToHost),
                  "read: scalars", error);
    }
};

std::unique_ptr<Climber> Climber::create(const DeviceModel& model, std::string& error) {
    auto impl = std::make_unique<Impl>();
    impl->m = model.impl_->view;
    const auto ncol = static_cast<std::size_t>(impl->m.ncol);
    const auto ncon = static_cast<std::size_t>(impl->m.ncon);
    const std::size_t nmax = std::max(ncol, ncon);
    impl->cells = static_cast<std::uint64_t>(model.impl_->num_breaks - (2 * impl->m.ncol)) / 2;
    // A step's work ids are 32-bit.
    if (impl->cells > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        error = "create: too many cells for 32-bit work ids";
        return nullptr;
    }
    // The scan's scratch, sized for the longer of the two flag arrays.
    if (nmax > 0 && !ok(cub::DeviceScan::ExclusiveSum(
                            nullptr, impl->scan_temp_bytes, static_cast<std::int32_t*>(nullptr),
                            static_cast<std::int32_t*>(nullptr), static_cast<int>(nmax)),
                        "scan size", error)) {
        return nullptr;
    }
    Impl& s = *impl;
    const bool allocated =
        s.x.allocate(ncol, "x", error) && s.lhs.allocate(ncon, "lhs", error) &&
        s.weight.allocate(ncon, "weight", error) && s.jump.allocate(ncol, "jump", error) &&
        s.score.allocate(ncol, "score", error) && s.best_x.allocate(ncol, "best_x", error) &&
        s.violated_flag.allocate(ncon, "violated_flag", error) &&
        s.violated.allocate(ncon, "violated", error) &&
        s.violated_pos.allocate(ncon, "violated_pos", error) &&
        s.good_flag.allocate(ncol, "good_flag", error) && s.good.allocate(ncol, "good", error) &&
        s.good_pos.allocate(ncol, "good_pos", error) && s.offset.allocate(nmax, "offset", error) &&
        s.touch.allocate(ncon, "touch", error) && s.owner.allocate(ncol, "owner", error) &&
        s.list.allocate(ncon, "list", error) && s.coef.allocate(ncon, "coef", error) &&
        s.old_lhs.allocate(ncon, "old_lhs", error) && s.new_lhs.allocate(ncon, "new_lhs", error) &&
        s.work_off.allocate(ncon, "work_off", error) && s.nbr.allocate(ncol, "nbr", error) &&
        s.scalars.allocate(1, "scalars", error) &&
        s.breaks.allocate(static_cast<std::size_t>(model.impl_->num_breaks), "breaks", error) &&
        s.scan_temp.allocate(s.scan_temp_bytes, "scan_temp", error);
    // The step scratch starts, and is left after every step, unmarked:
    // every byte 0xFF is a `touch` of -1 and an `owner` of `kNoOwner`.
    const bool cleared =
        allocated &&
        (ncon == 0 ||
         ok(cudaMemset(s.touch.get(), 0xFF, ncon * sizeof(std::int32_t)), "touch", error)) &&
        (ncol == 0 ||
         ok(cudaMemset(s.owner.get(), 0xFF, ncol * sizeof(std::uint32_t)), "owner", error));
    if (!cleared) {
        return nullptr;
    }
    return std::unique_ptr<Climber>(new Climber(std::move(impl)));
}

Climber::Climber(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Climber::~Climber() = default;

bool Climber::init(std::span<const double> start, std::uint64_t seed, std::string& error) {
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
    init_scalars_kernel<<<1, 1>>>(s.m, s.x.get(), seed, 2 * s.cells, s.scalars.get());
    if (!ok(cudaGetLastError(), "init: scalars", error)) {
        return false;
    }
    if (s.m.ncon > 0 &&
        !Impl::launched(init_constraints_kernel, s.m.ncon, "init: constraints", error, s.m,
                        s.x.get(), s.lhs.get(), s.weight.get(), s.violated_flag.get())) {
        return false;
    }
    return s.compact(s.m.ncon, s.violated_flag.get(), s.violated.get(), s.violated_pos.get(),
                     &s.scalars.get()->n_violated, "violated set", error) &&
           s.reset_moves(error);
}

bool Climber::reset_moves(double objective_weight, std::string& error) {
    return impl_->set_objective_weight(objective_weight, error) && impl_->reset_moves(error);
}

bool Climber::run(std::uint64_t max_steps, Chunk& chunk, std::string& error) {
    Impl& s = *impl_;
    step_kernel<<<1, kStepThreads>>>(s.m, s.view(), max_steps);
    Scalars out{};
    if (!ok(cudaGetLastError(), "run: launch", error) || !s.read_scalars(out, error)) {
        return false;
    }
    s.objective_weight = out.objective_weight;
    chunk.total_effort = out.total_effort;
    chunk.effort_since_improvement = out.total_effort - out.effort_at_last_improvement;
    chunk.improved = out.improved != 0;
    chunk.objective = out.best_objective;
    chunk.finished = out.finished != 0;
    return true;
}

bool Climber::read_best(std::vector<double>& x, std::string& error) const {
    const Impl& s = *impl_;
    Scalars out{};
    if (!s.read_scalars(out, error)) {
        return false;
    }
    if (std::isinf(out.best_objective)) {
        x.clear();
        return true;
    }
    return to_host(x, s.best_x.get(), static_cast<std::size_t>(s.m.ncol), "read: best", error);
}

bool Climber::read(State& state, std::string& error) const {
    const Impl& s = *impl_;
    const auto ncol = static_cast<std::size_t>(s.m.ncol);
    const auto ncon = static_cast<std::size_t>(s.m.ncon);
    Scalars out{};
    std::vector<std::int32_t> violated;
    std::vector<std::int32_t> good;
    const bool copied =
        s.read_scalars(out, error) && to_host(state.x, s.x.get(), ncol, "read: x", error) &&
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
    violated.resize(static_cast<std::size_t>(out.n_violated));
    good.resize(static_cast<std::size_t>(out.n_good));
    state.violated = std::move(violated);
    state.good = std::move(good);
    state.objective_weight = out.objective_weight;
    state.weight_increment = out.weight_increment;
    state.objective = out.objective;
    state.total_effort = out.total_effort;
    state.steps = out.steps;
    state.bumps = out.bumps;
    state.last_var = out.last_var;
    return true;
}

}  // namespace gpu_fj
