#include "climber.h"
#include "device_internal.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
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
            // bound interval, never upstream's two for `Equal`.
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
            // An integer range holding no integer is not skipped: v1.16
            // dropped v1.15's `validRange.first > validRange.second` skip.
            if (is_int) {
                first = ceil(first - m.equality_tol);
                second = floor(second + m.equality_tol);
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

// A cell's term of a `resetMoves` score: constraint `c`'s weighted score
// change when its `lhs` moves by `coeff * delta`.
__device__ double cell_score(const DeviceModelView& m, std::int32_t c, double lhs, double weight,
                             double coeff, double delta) {
    return weight * (con_score(m.con_sense[c], m.con_rhs[c], lhs + (coeff * delta)) -
                     con_score(m.con_sense[c], m.con_rhs[c], lhs));
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
            s += cell_score(m, c, lhs[c], weight[c], coeff, delta);
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
// takes every decision that draws from the random stream, and the moved
// column's jump value (a sort over its breakpoints); the block spreads the
// moved variable's (constraint, cell) work, mapped to work ids tile by
// tile (`map_tile`), the neighbours' score updates, the violated and good
// set edits (`update_set`) and the moved column's score.  A neighbour's
// score change is gathered over its own column against the touched
// constraints in column order, and the moved column's score is a
// fixed-order block reduction: no sum uses float atomics
// atomics, so a climber is deterministic.
//
// Five deviations from upstream, none in what a step computes:
// - Upstream adds each `modifyMove` (or weight-bump) term to a neighbour's
//   score constraint by constraint and calls `updateGoodMoves` after each;
//   here a neighbour's terms are summed over its column first, then added,
//   and its good-set membership is updated once, neighbours in the order
//   their first cell comes in the moved constraints' cells.  So a score can
//   differ from upstream's in rounding.
// - The violated and good sets are edited by the block, removals before
//   additions (`update_set`), not item by item with swap-removes and
//   appends interleaved.  The sets are upstream's, their order is not, so
//   the draws over them differ; with the deviation above, this is why a
//   GPU seed never replays a CPU one's draws.
// - The moved column's `resetMoves` score is a block reduction over its
//   cells, objective term last, not upstream's sum in column order: equal
//   up to rounding.
// - Where a weight rescale recomputes every move and upstream adds each
//   variable to the good set in turn, the good set is rebuilt in index
//   order.
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
    std::int32_t* hole;      // `update_set`'s, max(ncol, ncon) long
    Scalars* s;

    // Climber `k`'s arrays in a batch that lays each array out climber
    // after climber: `ncol`, `ncon`, max(`ncol`, `ncon`) and `nbreaks` long
    // per climber.
    __host__ __device__ ClimberView slot(std::size_t k, std::size_t ncol, std::size_t ncon,
                                         std::size_t nbreaks) const {
        const std::size_t col = k * ncol;
        const std::size_t con = k * ncon;
        return ClimberView{x + col,
                           lhs + con,
                           weight + con,
                           jump + col,
                           score + col,
                           best_x + col,
                           violated + con,
                           violated_pos + con,
                           good + col,
                           good_pos + col,
                           breaks + (k * nbreaks),
                           touch + con,
                           owner + col,
                           list + con,
                           coef + con,
                           old_lhs + con,
                           new_lhs + con,
                           work_off + con,
                           nbr + col,
                           hole + (k * (ncol > ncon ? ncol : ncon)),
                           s + k};
    }
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
using BlockReduce = cub::BlockReduce<double, kStepThreads>;

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

// The constraint each work id of the tile `[base, base + kStepThreads)`
// names, into thread `t`'s `k` for work id `base + t` (cuOpt's
// load-balanced search, at block scale, with no per-work-id binary
// search): each constraint starting in the tile marks its start, and an
// inclusive max-scan carries the marks forward, seeded by `first`, the
// constraint holding work id `base - 1` (0 for the first tile).  Every
// constraint has a cell (`HostModel` gives an empty row none, as
// `addConstraint` does), so the offsets strictly increase and at most
// `kStepThreads` constraints, `first + 1` onward, start in a tile.
// Returns the tile's last constraint, the next tile's `first`, on every
// thread.
__device__ std::int32_t map_tile(const ClimberView& c, std::int32_t n, std::int32_t base,
                                 std::int32_t first, BlockScan::TempStorage& tmp, std::int32_t& k) {
    __shared__ std::int32_t mark[kStepThreads];
    const auto t = static_cast<std::int32_t>(threadIdx.x);
    mark[t] = -1;
    __syncthreads();
    const std::int32_t j = first + 1 + t;
    if (j < n && c.work_off[j] < base + kStepThreads) {
        mark[c.work_off[j] - base] = j;
    }
    __syncthreads();
    const std::int32_t seed = t == 0 ? max(mark[0], first) : mark[t];
    std::int32_t last = 0;
    BlockScan(tmp).InclusiveScan(
        seed, k, [](std::int32_t a, std::int32_t b) { return a > b ? a : b; }, last);
    __syncthreads();
    return last;
}

// `each(w, u)` on every thread for every work id `w` of constraints
// `cons[0..n)` (`total` cells, `work_off` set), tile by tile, `u` being the
// column of the cell `w` names (-1 past `total`).  `each` may run block
// collectives.
template <typename Each>
__device__ void for_tiles(const DeviceModelView& m, const ClimberView& c, const std::int32_t* cons,
                          std::int32_t n, std::int32_t total, BlockScan::TempStorage& tmp,
                          Each each) {
    std::int32_t first = 0;
    for (std::int32_t base = 0; base < total; base += kStepThreads) {
        std::int32_t k = 0;
        first = map_tile(c, n, base, first, tmp, k);
        const std::int32_t w = base + static_cast<std::int32_t>(threadIdx.x);
        std::int32_t u = -1;
        if (w < total) {
            const std::int32_t row = m.con_row[cons[k]];
            u = m.row_index[m.row_start[row] + (w - c.work_off[k])];
        }
        each(w, u);
    }
}

// The distinct columns of the cells of constraints `cons[0..n)` (`total`
// cells, `work_off` set), but `skip`, into `nbr` in first-cell order;
// returns their number.
__device__ std::int32_t collect(const DeviceModelView& m, const ClimberView& c,
                                const std::int32_t* cons, std::int32_t n, std::int32_t total,
                                std::int32_t skip, BlockScan::TempStorage& tmp) {
    for_tiles(m, c, cons, n, total, tmp, [&](std::int32_t w, std::int32_t u) {
        if (u >= 0 && u != skip) {
            atomicMin(c.owner + u, static_cast<std::uint32_t>(w));
        }
    });
    __syncthreads();
    std::int32_t count = 0;
    for_tiles(m, c, cons, n, total, tmp, [&](std::int32_t w, std::int32_t u) {
        const std::int32_t first =
            u >= 0 && u != skip && c.owner[u] == static_cast<std::uint32_t>(w) ? 1 : 0;
        std::int32_t off = 0;
        std::int32_t tile = 0;
        BlockScan(tmp).ExclusiveSum(first, off, tile);
        if (first != 0) {
            c.nbr[count + off] = u;
        }
        __syncthreads();
        count += tile;
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

// The membership edits of set `list` (`pos`, `size`) for the distinct
// `items[0..n)`, every thread: item `i` belongs iff `want(i)`, which reads
// no set.  Upstream (`updateGoodMoves`, `setValue`'s violated-set update)
// edits item by item, swap-removing and appending in turn; here the
// removals go first, each hole below the shrunk size filled by a survivor
// from the tail, holes in item order matched to survivors in position
// order, and then the additions are appended in item order.  The set is
// upstream's, its order is not.  `hole` holds up to `n` entries.
template <typename Want>
__device__ void update_set(std::int32_t* list, std::int32_t* pos, std::int32_t& size,
                           const std::int32_t* items, std::int32_t n, std::int32_t* hole,
                           BlockScan::TempStorage& tmp, Want want) {
    const std::int32_t old_size = size;
    const auto leaves = [&](std::int32_t i) { return pos[items[i]] != -1 && !want(i); };
    const std::int32_t kept =
        old_size - scan_each(
                       n, tmp, [&](std::int32_t i) { return leaves(i) ? 1 : 0; },
                       [](std::int32_t, std::int32_t, std::int32_t) {});
    // The holes, and every leaver unmarked.
    scan_each(
        n, tmp, [&](std::int32_t i) { return leaves(i) && pos[items[i]] < kept ? 1 : 0; },
        [&](std::int32_t i, std::int32_t off, std::int32_t is_hole) {
            if (is_hole != 0) {
                hole[off] = pos[items[i]];
            }
            if (leaves(i)) {
                pos[items[i]] = -1;
            }
        });
    // The tail's survivors, one per hole.
    scan_each(
        old_size - kept, tmp, [&](std::int32_t r) { return pos[list[kept + r]] != -1 ? 1 : 0; },
        [&](std::int32_t r, std::int32_t off, std::int32_t stays) {
            if (stays != 0) {
                const std::int32_t e = list[kept + r];
                list[hole[off]] = e;
                pos[e] = hole[off];
            }
        });
    const std::int32_t added = scan_each(
        n, tmp, [&](std::int32_t i) { return pos[items[i]] == -1 && want(i) ? 1 : 0; },
        [&](std::int32_t i, std::int32_t off, std::int32_t joins) {
            if (joins != 0) {
                list[kept + off] = items[i];
                pos[items[i]] = kept + off;
            }
        });
    if (threadIdx.x == 0) {
        size = kept + added;
    }
    __syncthreads();
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
                               BlockScan::TempStorage& tmp, std::uint64_t cells) {
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
        // Upstream moves only the scores here: the good set, empty at a
        // local minimum, stays empty until a move touches a variable.
        __syncthreads();
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
        update_set(c.good, c.good_pos, s.n_good, c.nbr, count, c.hole, tmp,
                   [&](std::int32_t i) { return c.score[c.nbr[i]] > 0.0; });
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
    __shared__ double jump_shared;
    __shared__ BlockReduce::TempStorage reduce_tmp;
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
    update_set(
        c.violated, c.violated_pos, s.n_violated, c.list, deg, c.hole, tmp,
        [&](std::int32_t k) { return score_of(m, c.list[k], c.new_lhs[k]) < -m.violation_tol; });
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
    update_set(c.good, c.good_pos, s.n_good, c.nbr, count, c.hole, tmp,
               [&](std::int32_t i) { return c.score[c.nbr[i]] > 0.0; });
    // `resetMoves(v)`: the jump value on thread 0 (a sort over its
    // breakpoints), the score over `v`'s cells by the block.
    if (tid == 0) {
        jump_shared = jump_value(m, v, c.x, c.lhs, c.weight, c.breaks + m.break_start[v]);
    }
    __syncthreads();
    const double value = jump_shared;
    const double shift = value - c.x[v];
    double part = 0.0;
    for (std::int32_t k = tid; k < deg; k += kStepThreads) {
        const std::int32_t con = c.list[k];
        part += cell_score(m, con, c.lhs[con], c.weight[con], c.coef[k], shift);
    }
    const double cells_score = BlockReduce(reduce_tmp).Sum(part);
    if (tid == 0) {
        c.jump[v] = value;
        c.score[v] = -(s.objective_weight * m.col_cost[v] * shift) + cells_score;
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

// The step loop's `var` while the good set is empty; a selection that finds
// no variable returns -1.
constexpr std::int32_t kAtMinimum = -2;

// The top of `solve`'s loop, thread 0 only: the effort callback, the
// improvement bookkeeping, and the feasible-improvement callback.
__device__ int top_of_loop(const DeviceModelView& m, Scalars& s, std::uint64_t taken,
                           std::uint64_t max_steps) {
    if (s.finished != 0) {
        return kStop;  // `solve` returned
    }
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

// `solve`'s loop up to its next callback, or `max_steps` steps, for the
// batch's climber `blockIdx.x`.
__global__ void __launch_bounds__(kStepThreads)
    step_kernel(DeviceModelView m, ClimberView batch, std::int64_t nbreaks, std::uint64_t cells,
                std::uint64_t max_steps) {
    const ClimberView c =
        batch.slot(blockIdx.x, static_cast<std::size_t>(m.ncol), static_cast<std::size_t>(m.ncon),
                   static_cast<std::size_t>(nbreaks));
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
            var = s.n_good > 0 ? select_good(c, s) : kAtMinimum;
        }
        __syncthreads();
        if (var == kAtMinimum) {
            update_weights(m, c, s, tmp, cells);
            if (tid == 0) {
                var = select_at_minimum(m, c, s);
            }
            __syncthreads();
        }
        if (var < 0) {
            // v1.16's `selectVariable() == UINT_MAX` break: no score beat
            // -inf (all NaN), and `solve` returns.
            if (tid == 0) {
                s.finished = 1;
            }
            break;
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

// Device bytes a batch's arrays take, as `create` allocates them: each
// climber's, and the `init` scratch the climbers share.
struct BatchBytes {
    std::size_t per_climber;
    std::size_t shared;
};

BatchBytes batch_bytes(std::size_t ncol, std::size_t ncon, std::size_t nbreaks,
                       std::size_t scan_temp_bytes) {
    constexpr std::size_t kInt = sizeof(std::int32_t);
    constexpr std::size_t kDouble = sizeof(double);
    return BatchBytes{
        // x, jump, score, best_x; good, good_pos, nbr, owner.
        .per_climber = (ncol * ((4 * kDouble) + (4 * kInt))) +
                       // lhs, weight, coef, old_lhs, new_lhs; violated,
                       // violated_pos, touch, list, work_off.
                       (ncon * ((5 * kDouble) + (5 * kInt))) +
                       // hole.
                       (std::max(ncol, ncon) * kInt) + (nbreaks * sizeof(Break)) + sizeof(Scalars),
        // violated_flag, good_flag, offset.
        .shared = ((ncon + ncol + std::max(ncol, ncon)) * kInt) + scan_temp_bytes,
    };
}

}  // namespace

std::int32_t default_climbers(std::string& error) {
    int device = 0;
    int sms = 0;
    if (!ok(cudaGetDevice(&device), "default_climbers", error) ||
        !ok(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
            "default_climbers", error)) {
        return 0;
    }
    return sms * kClimbersPerSm;
}

struct Climber::Impl {
    DeviceModelView m{};
    std::uint64_t cells = 0;  // upstream's `nNonzeros`
    std::int64_t nbreaks = 0;
    std::int32_t count = 0;
    // Per climber, on the host: the objective weight, set with the device's
    // copy in `scalars`, and whether `init` has run.
    std::vector<double> objective_weight;
    std::vector<std::uint8_t> initialized;
    std::vector<Scalars> out;  // `run`'s read-back
    // Each climber's arrays, laid out climber after climber
    // (`ClimberView::slot`).
    DeviceBuffer<double> x, lhs, weight, jump, score, best_x, coef, old_lhs, new_lhs;
    DeviceBuffer<std::int32_t> violated, violated_pos, good, good_pos, touch, list, work_off, nbr,
        hole;
    DeviceBuffer<std::uint32_t> owner;
    DeviceBuffer<Scalars> scalars;
    DeviceBuffer<Break> breaks;
    // `init`'s and `reset_moves`' scratch, shared: the host runs them one
    // climber at a time.
    DeviceBuffer<std::int32_t> violated_flag, good_flag, offset;
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

    [[nodiscard]] bool valid(std::int32_t k, const char* what, std::string& error) const {
        if (k >= 0 && k < count) {
            return true;
        }
        error = std::string(what) + ": no climber " + std::to_string(k);
        return false;
    }

    [[nodiscard]] ClimberView batch() const {
        return ClimberView{x.get(),       lhs.get(),      weight.get(),   jump.get(),
                           score.get(),   best_x.get(),   violated.get(), violated_pos.get(),
                           good.get(),    good_pos.get(), breaks.get(),   touch.get(),
                           owner.get(),   list.get(),     coef.get(),     old_lhs.get(),
                           new_lhs.get(), work_off.get(), nbr.get(),      hole.get(),
                           scalars.get()};
    }

    [[nodiscard]] ClimberView view(std::int32_t k) const {
        return batch().slot(static_cast<std::size_t>(k), static_cast<std::size_t>(m.ncol),
                            static_cast<std::size_t>(m.ncon), static_cast<std::size_t>(nbreaks));
    }

    // The objective weight lives in `scalars` on the device; this copy is
    // the host's, set with it.
    bool set_objective_weight(std::int32_t k, double w, std::string& error) {
        objective_weight[static_cast<std::size_t>(k)] = w;
        return ok(
            cudaMemcpy(&view(k).s->objective_weight, &w, sizeof(double), cudaMemcpyHostToDevice),
            "objective weight", error);
    }

    bool reset_moves(std::int32_t k, std::string& error) {
        const ClimberView c = view(k);
        std::int32_t* n_good = &c.s->n_good;
        if (m.ncol == 0) {
            return compact(0, nullptr, nullptr, nullptr, n_good, "reset_moves", error);
        }
        return launched(reset_moves_kernel, m.ncol, "reset_moves", error, m,
                        objective_weight[static_cast<std::size_t>(k)], c.x, c.lhs, c.weight,
                        c.breaks, c.jump, c.score, good_flag.get()) &&
               compact(m.ncol, good_flag.get(), c.good, c.good_pos, n_good, "good set", error);
    }

    bool read_scalars(std::int32_t k, Scalars& s, std::string& error) const {
        return ok(cudaMemcpy(&s, view(k).s, sizeof(Scalars), cudaMemcpyDeviceToHost),
                  "read: scalars", error);
    }

    bool allocate(std::string& error) {
        const auto ncol = static_cast<std::size_t>(m.ncol);
        const auto ncon = static_cast<std::size_t>(m.ncon);
        const auto k = static_cast<std::size_t>(count);
        const std::size_t kcol = k * ncol;
        const std::size_t kcon = k * ncon;
        return x.allocate(kcol, "x", error) && lhs.allocate(kcon, "lhs", error) &&
               weight.allocate(kcon, "weight", error) && jump.allocate(kcol, "jump", error) &&
               score.allocate(kcol, "score", error) && best_x.allocate(kcol, "best_x", error) &&
               violated.allocate(kcon, "violated", error) &&
               violated_pos.allocate(kcon, "violated_pos", error) &&
               good.allocate(kcol, "good", error) && good_pos.allocate(kcol, "good_pos", error) &&
               touch.allocate(kcon, "touch", error) && owner.allocate(kcol, "owner", error) &&
               list.allocate(kcon, "list", error) && coef.allocate(kcon, "coef", error) &&
               old_lhs.allocate(kcon, "old_lhs", error) &&
               new_lhs.allocate(kcon, "new_lhs", error) &&
               work_off.allocate(kcon, "work_off", error) && nbr.allocate(kcol, "nbr", error) &&
               hole.allocate(k * std::max(ncol, ncon), "hole", error) &&
               scalars.allocate(k, "scalars", error) &&
               breaks.allocate(k * static_cast<std::size_t>(nbreaks), "breaks", error) &&
               violated_flag.allocate(ncon, "violated_flag", error) &&
               good_flag.allocate(ncol, "good_flag", error) &&
               offset.allocate(std::max(ncol, ncon), "offset", error) &&
               scan_temp.allocate(scan_temp_bytes, "scan_temp", error);
    }
};

std::unique_ptr<Climber> Climber::create(const DeviceModel& model, std::int32_t climbers,
                                         std::string& error) {
    if (climbers < 1) {
        error = "create: a batch needs at least one climber";
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    Impl& s = *impl;
    s.m = model.impl_->view;
    s.count = climbers;
    s.nbreaks = model.impl_->num_breaks;
    const auto ncol = static_cast<std::size_t>(s.m.ncol);
    const auto ncon = static_cast<std::size_t>(s.m.ncon);
    const std::size_t nmax = std::max(ncol, ncon);
    s.cells = static_cast<std::uint64_t>(s.nbreaks - (2 * s.m.ncol)) / 2;
    // A step's work ids are 32-bit.
    if (s.cells > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        error = "create: too many cells for 32-bit work ids";
        return nullptr;
    }
    // The scan's scratch, sized for the longer of the two flag arrays.
    if (nmax > 0 && !ok(cub::DeviceScan::ExclusiveSum(
                            nullptr, s.scan_temp_bytes, static_cast<std::int32_t*>(nullptr),
                            static_cast<std::int32_t*>(nullptr), static_cast<int>(nmax)),
                        "scan size", error)) {
        return nullptr;
    }
    const BatchBytes bytes =
        batch_bytes(ncol, ncon, static_cast<std::size_t>(s.nbreaks), s.scan_temp_bytes);
    if (!fits(bytes.shared, bytes.per_climber, static_cast<std::size_t>(climbers),
              "create: climbers", error) ||
        !s.allocate(error)) {
        return nullptr;
    }
    // The step scratch starts, and is left after every step, unmarked:
    // every byte 0xFF is a `touch` of -1 and an `owner` of `kNoOwner`.
    const auto k = static_cast<std::size_t>(climbers);
    const bool cleared =
        (ncon == 0 ||
         ok(cudaMemset(s.touch.get(), 0xFF, k * ncon * sizeof(std::int32_t)), "touch", error)) &&
        (ncol == 0 ||
         ok(cudaMemset(s.owner.get(), 0xFF, k * ncol * sizeof(std::uint32_t)), "owner", error));
    if (!cleared) {
        return nullptr;
    }
    s.objective_weight.assign(k, 0.0);
    s.initialized.assign(k, 0);
    s.out.resize(k);
    return std::unique_ptr<Climber>(new Climber(std::move(impl)));
}

Climber::Climber(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Climber::~Climber() = default;

std::int32_t Climber::count() const {
    return impl_->count;
}

bool Climber::init(std::int32_t k, std::span<const double> start, std::uint64_t seed,
                   std::string& error) {
    Impl& s = *impl_;
    if (!s.valid(k, "init", error)) {
        return false;
    }
    if (start.size() != static_cast<std::size_t>(s.m.ncol)) {
        error = "init: start has the wrong size";
        return false;
    }
    const ClimberView c = s.view(k);
    s.initialized[static_cast<std::size_t>(k)] = 0;
    s.objective_weight[static_cast<std::size_t>(k)] = 0.0;
    if (!start.empty() &&
        !ok(cudaMemcpy(c.x, start.data(), start.size_bytes(), cudaMemcpyHostToDevice),
            "init: copy start", error)) {
        return false;
    }
    init_scalars_kernel<<<1, 1>>>(s.m, c.x, seed, 2 * s.cells, c.s);
    if (!ok(cudaGetLastError(), "init: scalars", error)) {
        return false;
    }
    if (s.m.ncon > 0 && !Impl::launched(init_constraints_kernel, s.m.ncon, "init: constraints",
                                        error, s.m, c.x, c.lhs, c.weight, s.violated_flag.get())) {
        return false;
    }
    const bool built = s.compact(s.m.ncon, s.violated_flag.get(), c.violated, c.violated_pos,
                                 &c.s->n_violated, "violated set", error) &&
                       s.reset_moves(k, error);
    s.initialized[static_cast<std::size_t>(k)] = built ? 1 : 0;
    return built;
}

bool Climber::reset_moves(std::int32_t k, double objective_weight, std::string& error) {
    return impl_->valid(k, "reset_moves", error) &&
           impl_->set_objective_weight(k, objective_weight, error) && impl_->reset_moves(k, error);
}

bool Climber::run(std::uint64_t max_steps, std::span<Chunk> chunks, std::string& error) {
    Impl& s = *impl_;
    if (chunks.size() != s.out.size()) {
        error = "run: one chunk per climber";
        return false;
    }
    if (std::ranges::find(s.initialized, std::uint8_t{0}) != s.initialized.end()) {
        error = "run: a climber has had no init";
        return false;
    }
    step_kernel<<<static_cast<unsigned>(s.count), kStepThreads>>>(s.m, s.batch(), s.nbreaks,
                                                                  s.cells, max_steps);
    if (!ok(cudaGetLastError(), "run: launch", error) ||
        !ok(cudaMemcpy(s.out.data(), s.scalars.get(), s.out.size() * sizeof(Scalars),
                       cudaMemcpyDeviceToHost),
            "run: scalars", error)) {
        return false;
    }
    for (std::size_t k = 0; k < chunks.size(); ++k) {
        const Scalars& out = s.out[k];
        Chunk& chunk = chunks[k];
        s.objective_weight[k] = out.objective_weight;
        chunk.total_effort = out.total_effort;
        chunk.effort_since_improvement = out.total_effort - out.effort_at_last_improvement;
        chunk.improved = out.improved != 0;
        chunk.objective = out.best_objective;
        chunk.finished = out.finished != 0;
    }
    return true;
}

bool Climber::read_best(std::int32_t k, std::vector<double>& x, std::string& error) const {
    const Impl& s = *impl_;
    Scalars out{};
    if (!s.valid(k, "read_best", error) || !s.read_scalars(k, out, error)) {
        return false;
    }
    if (std::isinf(out.best_objective)) {
        x.clear();
        return true;
    }
    return to_host(x, s.view(k).best_x, static_cast<std::size_t>(s.m.ncol), "read: best", error);
}

bool Climber::read(std::int32_t k, State& state, std::string& error) const {
    const Impl& s = *impl_;
    if (!s.valid(k, "read", error)) {
        return false;
    }
    const ClimberView c = s.view(k);
    const auto ncol = static_cast<std::size_t>(s.m.ncol);
    const auto ncon = static_cast<std::size_t>(s.m.ncon);
    Scalars out{};
    std::vector<std::int32_t> violated;
    std::vector<std::int32_t> good;
    const bool copied =
        s.read_scalars(k, out, error) && to_host(state.x, c.x, ncol, "read: x", error) &&
        to_host(state.lhs, c.lhs, ncon, "read: lhs", error) &&
        to_host(state.weight, c.weight, ncon, "read: weight", error) &&
        to_host(violated, c.violated, ncon, "read: violated", error) &&
        to_host(state.violated_pos, c.violated_pos, ncon, "read: violated_pos", error) &&
        to_host(state.jump, c.jump, ncol, "read: jump", error) &&
        to_host(state.score, c.score, ncol, "read: score", error) &&
        to_host(good, c.good, ncol, "read: good", error) &&
        to_host(state.good_pos, c.good_pos, ncol, "read: good_pos", error);
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
