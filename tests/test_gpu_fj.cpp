// The GPU feasibility jump's model (#261) and its climbers' initial state
// (#262).  The host half runs in every tree: `HostModel` must hand the
// device the constraints and bounds `FjWorker` hands upstream FJ.  The
// `[gpu]` cases, compiled only in a `MIP_HEURISTICS_CUDA` tree, compare a
// climber's state after `init` with upstream FJ's, recomputed on the host;
// they fail, never skip, on a machine with no usable device.

#include "core_test_support.h"
#include "gpu/host_model.h"
#include "Highs.h"
#include "test_paths.h"

#ifdef MIP_HEURISTICS_HAS_GPU_FJ
#include "fj_personas.h"
#include "fj_worker.h"
#include "gpu/climber.h"
#include "gpu/device_model.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#endif

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using namespace core_test;

namespace {

// Between them: negative coefficients, general integers, equality and
// ranged rows (the `[gpu]` case checks the first two are present).
constexpr std::array<const char*, 6> kInstances = {"egout.mps", "flugpl.mps", "bell5.mps",
                                                   "p0548.mps", "gt2.mps",    "gesa2.mps"};

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/" + instance) == HighsStatus::kOk);
}

}  // namespace

TEST_CASE("gpu_fj: the host model expands rows as FjWorker does", "[gpu_fj]") {
    for (const char* instance : kInstances) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        const Setup s(highs.getLp(), highs.getOptions().log_options);
        const ProblemView& p = s.problem;
        const gpu_fj::HostModel host(p);
        const gpu_fj::GpuFjModel m = host.pod();
        REQUIRE(host.legal_bounds());
        REQUIRE(m.ncol == p.ncol);
        REQUIRE(m.nrow == p.nrow);
        REQUIRE(std::cmp_equal(m.nnz, p.nnz));

        // Every finite bound of a nonempty row is one constraint, `>=`
        // before `<=`, in row order.
        REQUIRE(m.equality_tol == p.epsilon);
        REQUIRE(m.violation_tol == p.feastol);
        std::int32_t con = 0;
        for (HighsInt row = 0; row < p.nrow; ++row) {
            REQUIRE(m.row_con_start[row] == con);
            const bool empty = (*p.ar_start)[row + 1] == (*p.ar_start)[row];
            const double lower = empty ? -kHighsInf : (*p.row_lower)[row];
            const double upper = empty ? kHighsInf : (*p.row_upper)[row];
            if (std::isfinite(lower)) {
                REQUIRE(con < m.ncon);
                REQUIRE(m.con_row[con] == row);
                REQUIRE(m.con_sense[con] == gpu_fj::ConSense::kGte);
                REQUIRE(m.con_rhs[con] == lower);
                ++con;
            }
            if (std::isfinite(upper)) {
                REQUIRE(con < m.ncon);
                REQUIRE(m.con_row[con] == row);
                REQUIRE(m.con_sense[con] == gpu_fj::ConSense::kLte);
                REQUIRE(m.con_rhs[con] == upper);
                ++con;
            }
        }
        REQUIRE(con == m.ncon);
        REQUIRE(m.row_con_start[p.nrow] == con);

        for (HighsInt col = 0; col < p.ncol; ++col) {
            const bool is_int = (*p.integrality)[col] != HighsVarType::kContinuous;
            REQUIRE(m.col_is_int[col] == (is_int ? 1 : 0));
            if (is_int) {
                REQUIRE(m.col_lower[col] == std::ceil((*p.col_lower)[col] - p.feastol));
                REQUIRE(m.col_upper[col] == std::floor((*p.col_upper)[col] + p.feastol));
            } else {
                REQUIRE(m.col_lower[col] == (*p.col_lower)[col]);
                REQUIRE(m.col_upper[col] == (*p.col_upper)[col]);
            }
        }
    }
}

TEST_CASE("gpu_fj: an empty integer domain is not legal for FJ", "[gpu_fj]") {
    Highs highs;
    read(highs, "egout.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    // `make_problem` refuses such a column itself, so the view is edited
    // after it: a caller building its own view could hand one over.
    ProblemView p = s.problem;
    HighsInt col = 0;
    while ((*p.integrality)[col] == HighsVarType::kContinuous) {
        ++col;
    }
    std::vector<double> lower = *p.col_lower;
    std::vector<double> upper = *p.col_upper;
    // Rounded inward, [0.2, 0.8] holds no integer.
    lower[col] = 0.2;
    upper[col] = 0.8;
    p.col_lower = &lower;
    p.col_upper = &upper;
    const gpu_fj::HostModel host(p);
    REQUIRE_FALSE(host.legal_bounds());
}

namespace {

// A one-column-per-entry model: `cols` columns with the given bounds,
// costs and integrality, and `rows` given column-wise as one coefficient
// per (row, column) pair, `0` for none.
HighsLp tiny_lp(const std::vector<double>& col_lower, const std::vector<double>& col_upper,
                const std::vector<double>& cost, const std::vector<double>& row_lower,
                const std::vector<double>& row_upper,
                const std::vector<std::vector<double>>& coeff_by_col, bool integer) {
    HighsLp lp;
    lp.num_col_ = static_cast<HighsInt>(col_lower.size());
    lp.num_row_ = static_cast<HighsInt>(row_lower.size());
    lp.col_lower_ = col_lower;
    lp.col_upper_ = col_upper;
    lp.col_cost_ = cost;
    lp.row_lower_ = row_lower;
    lp.row_upper_ = row_upper;
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_.assign(1, 0);
    for (const std::vector<double>& col : coeff_by_col) {
        for (size_t row = 0; row < col.size(); ++row) {
            if (col[row] != 0.0) {
                lp.a_matrix_.index_.push_back(static_cast<HighsInt>(row));
                lp.a_matrix_.value_.push_back(col[row]);
            }
        }
        lp.a_matrix_.start_.push_back(static_cast<HighsInt>(lp.a_matrix_.index_.size()));
    }
    lp.integrality_.assign(col_lower.size(),
                           integer ? HighsVarType::kInteger : HighsVarType::kContinuous);
    return lp;
}

}  // namespace

TEST_CASE("gpu_fj: an empty row gives no constraint, as upstream stores none", "[gpu_fj]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // Row 0 holds x; row 1 is empty, ranged (`make_problem` refuses an
    // empty row its zero activity violates).
    const HighsLp lp = tiny_lp({0.0}, {10.0}, {1.0}, {1.0, -1.0}, {5.0, 2.0}, {{1.0, 0.0}}, true);
    const Setup s(lp, highs.getOptions().log_options);
    const gpu_fj::HostModel host(s.problem);
    const gpu_fj::GpuFjModel m = host.pod();
    REQUIRE(m.ncon == 2);
    REQUIRE(m.con_row[0] == 0);
    REQUIRE(m.con_row[1] == 0);
    REQUIRE(m.row_con_start[1] == 2);
    REQUIRE(m.row_con_start[2] == 2);
}

#ifdef MIP_HEURISTICS_HAS_GPU_FJ
namespace {

// Upstream FJ's state after `FeasibilityJumpSolver::init`, recomputed on
// the host from the model as `FjWorker` hands it to upstream: each
// constraint's cells from the row-wise matrix, a column's cells in
// constraint order, then `resetIncumbent`, `JumpMove::updateValue` and
// `resetMoves` transliterated from `feasibilityjump.hh`, both #139 fixes
// included.
class Oracle {
public:
    Oracle(const gpu_fj::GpuFjModel& m, std::span<const double> start)
        : m_(m), cells_(static_cast<size_t>(m.ncol)), x_(start.begin(), start.end()) {
        const auto ncon = static_cast<size_t>(m.ncon);
        state_.lhs.assign(ncon, 0.0);
        state_.weight.assign(ncon, 1.0);
        state_.violated_pos.assign(ncon, -1);
        for (std::int32_t c = 0; c < m.ncon; ++c) {
            const std::int32_t row = m.con_row[c];
            double lhs = 0.0;
            for (std::int32_t k = m.row_start[row]; k < m.row_start[row + 1]; ++k) {
                cells_[m.row_index[k]].push_back({c, m.row_value[k]});
                lhs += m.row_value[k] * x_[m.row_index[k]];
            }
            state_.lhs[c] = lhs;
            if (con_score(c, lhs) < -m.violation_tol) {
                state_.violated_pos[c] = static_cast<std::int32_t>(state_.violated.size());
                state_.violated.push_back(c);
            }
        }
        state_.x = x_;
        reset_moves(0.0);
    }

    // `init`'s `resetMoves` loop under objective weight `w`.
    void reset_moves(double w) {
        const auto ncol = static_cast<size_t>(m_.ncol);
        state_.objective_weight = w;
        state_.jump.assign(ncol, 0.0);
        state_.score.assign(ncol, 0.0);
        state_.good.clear();
        state_.good_pos.assign(ncol, -1);
        for (std::int32_t j = 0; j < m_.ncol; ++j) {
            const double value = update_value(j);
            double score = 0.0;
            score -= w * m_.col_cost[j] * (value - x_[j]);
            for (const Cell& cell : cells_[j]) {
                const double lhs = state_.lhs[cell.con];
                const double candidate = lhs + (cell.coeff * (value - x_[j]));
                score += state_.weight[cell.con] *
                         (con_score(cell.con, candidate) - con_score(cell.con, lhs));
            }
            state_.jump[j] = value;
            state_.score[j] = score;
            if (score > 0.0) {
                state_.good_pos[j] = static_cast<std::int32_t>(state_.good.size());
                state_.good.push_back(j);
            }
        }
    }

    [[nodiscard]] const gpu_fj::Climber::State& state() const { return state_; }

private:
    struct Cell {
        std::int32_t con;
        double coeff;
    };

    [[nodiscard]] double con_score(std::int32_t c, double lhs) const {
        const double rhs = m_.con_rhs[c];
        return m_.con_sense[c] == gpu_fj::ConSense::kLte ? -(std::max(0., lhs - rhs))
                                                         : -(std::max(0., rhs - lhs));
    }

    static bool eq(double a, double b, double tol) { return std::fabs(a - b) < tol; }

    [[nodiscard]] double update_value(std::int32_t j) {
        const double tol = m_.equality_tol;
        const double incumbent = x_[j];
        const double lower = m_.col_lower[j];
        const double upper = m_.col_upper[j];
        double current_value = lower;
        double current_slope = 0.0;
        buffer_.clear();
        for (const Cell& cell : cells_[j]) {
            const bool lte = m_.con_sense[cell.con] == gpu_fj::ConSense::kLte;
            const double rhs = m_.con_rhs[cell.con];
            const std::pair<double, double> bound =
                lte ? std::pair{-kHighsInf, rhs} : std::pair{rhs, kHighsInf};
            const double residual = state_.lhs[cell.con] - (cell.coeff * incumbent);
            std::pair<double, double> range = {(1.0 / cell.coeff) * (bound.first - residual),
                                               (1.0 / cell.coeff) * (bound.second - residual)};
            if (cell.coeff < 0.0) {
                std::swap(range.first, range.second);
            }
            if (m_.col_is_int[j] != 0) {
                range = {std::ceil(range.first - tol), std::floor(range.second + tol)};
            }
            if (range.first > range.second) {
                continue;
            }
            const double w = state_.weight[cell.con];
            if (range.first > current_value) {
                current_slope -= w;
                if (range.first < upper) {
                    buffer_.emplace_back(range.first, w);
                }
            }
            if (range.second <= current_value) {
                current_slope += w;
            } else if (range.second < upper) {
                buffer_.emplace_back(range.second, w);
            }
        }
        if (std::isfinite(lower)) {
            buffer_.emplace_back(lower, 0);
        }
        if (std::isfinite(upper)) {
            buffer_.emplace_back(upper, 0);
        }
        std::ranges::sort(buffer_);
        double current_score = 0.0;
        current_value = buffer_.empty() ? incumbent : buffer_[0].first;
        double best_score = current_score;
        double best_value = current_value;
        for (const auto& [value, weight] : buffer_) {
            current_score += (value - current_value) * current_slope;
            current_slope += weight;
            current_value = value;
            if (eq(best_value, incumbent, tol) ||
                (!eq(current_value, incumbent, tol) && current_score < best_score)) {
                best_score = current_score;
                best_value = current_value;
            }
            if (!eq(best_value, incumbent, tol) && current_slope >= 0.) {
                break;
            }
        }
        return best_value;
    }

    const gpu_fj::GpuFjModel& m_;
    std::vector<std::vector<Cell>> cells_;
    std::vector<double> x_;
    std::vector<std::pair<double, double>> buffer_;
    gpu_fj::Climber::State state_;
};

// Relative tolerance 1e-9; `bitwise` counts the values whose bits differ,
// a zero of the other sign included.
void require_close(const std::vector<double>& got, const std::vector<double>& want,
                   size_t& bitwise) {
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        INFO("index " << i << ": " << got[i] << " vs " << want[i]);
        REQUIRE(std::fabs(got[i] - want[i]) <=
                1e-9 * std::max(std::fabs(got[i]), std::fabs(want[i])));
        bitwise +=
            std::bit_cast<std::uint64_t>(got[i]) == std::bit_cast<std::uint64_t>(want[i]) ? 0 : 1;
    }
}

void require_state(const gpu_fj::Climber::State& got, const gpu_fj::Climber::State& want,
                   size_t& bitwise) {
    REQUIRE(got.x == want.x);
    REQUIRE(got.weight == want.weight);
    REQUIRE(got.objective_weight == want.objective_weight);
    require_close(got.lhs, want.lhs, bitwise);
    REQUIRE(got.violated == want.violated);
    REQUIRE(got.violated_pos == want.violated_pos);
    require_close(got.jump, want.jump, bitwise);
    require_close(got.score, want.score, bitwise);
    REQUIRE(got.good == want.good);
    REQUIRE(got.good_pos == want.good_pos);
}

// FJ's own start, `FjWorker`'s with no point to take.
std::vector<double> bound_start(const gpu_fj::GpuFjModel& m) {
    std::vector<double> x(static_cast<size_t>(m.ncol));
    for (std::int32_t j = 0; j < m.ncol; ++j) {
        x[j] = fj_bound_start(m.col_lower[j], m.col_upper[j]);
    }
    return x;
}

// A point with no special structure: inside each column's bounds, off
// their midpoint, integral on integer columns.
std::vector<double> interior_start(const gpu_fj::GpuFjModel& m) {
    std::vector<double> x(static_cast<size_t>(m.ncol));
    for (std::int32_t j = 0; j < m.ncol; ++j) {
        const double up = m.col_upper[j];
        double lo = -10.0;
        if (std::isfinite(m.col_lower[j])) {
            lo = m.col_lower[j];
        } else if (std::isfinite(up)) {
            lo = up - 10.0;
        }
        const double hi = std::isfinite(up) ? up : lo + 10.0;
        const double frac = 0.3 + (0.4 * ((j * 7919) % 13) / 13.0);
        const double v = lo + ((hi - lo) * frac);
        x[j] = m.col_is_int[j] != 0 ? std::clamp(std::round(v), lo, hi) : v;
    }
    return x;
}

// A climber's state after `init(start)` and, if `objective_weight` is
// set, `reset_moves(objective_weight)`.
gpu_fj::Climber::State device_state(const gpu_fj::GpuFjModel& m, std::span<const double> start,
                                    double objective_weight = 0.0) {
    std::string error;
    auto device = gpu_fj::DeviceModel::upload(m, error);
    INFO(error);
    REQUIRE(device != nullptr);
    auto climber = gpu_fj::Climber::create(*device, error);
    REQUIRE(climber != nullptr);
    REQUIRE(climber->init(start, 0, error));
    if (objective_weight != 0.0) {
        REQUIRE(climber->reset_moves(objective_weight, error));
    }
    gpu_fj::Climber::State state;
    REQUIRE(climber->read(state, error));
    return state;
}

}  // namespace

TEST_CASE("gpu_fj: a climber's init state is upstream FJ's", "[gpu]") {
    std::string error;
    const double init = gpu_fj::init_context_seconds(error);
    INFO(error);
    REQUIRE(init >= 0.0);
    WARN("CUDA context init: " << init << " s");

    size_t negative = 0;
    size_t general_int = 0;
    size_t compared = 0;
    size_t bitwise = 0;
    for (const char* instance : kInstances) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        const Setup s(highs.getLp(), highs.getOptions().log_options);
        const gpu_fj::HostModel host(s.problem);
        const gpu_fj::GpuFjModel m = host.pod();
        for (std::int64_t k = 0; k < m.nnz; ++k) {
            negative += m.row_value[k] < 0.0 ? 1 : 0;
        }
        for (std::int32_t j = 0; j < m.ncol; ++j) {
            general_int +=
                m.col_is_int[j] != 0 && (m.col_lower[j] != 0.0 || m.col_upper[j] != 1.0) ? 1 : 0;
        }
        for (const std::vector<double>& start : {bound_start(m), interior_start(m)}) {
            Oracle oracle(m, start);
            require_state(device_state(m, start), oracle.state(), bitwise);
            // The objective term, which `init`'s zero weight leaves out.
            oracle.reset_moves(0.5);
            require_state(device_state(m, start, 0.5), oracle.state(), bitwise);
            compared += 2 * (static_cast<size_t>(m.ncon) + (2 * static_cast<size_t>(m.ncol)));
        }
    }
    REQUIRE(negative > 0);
    REQUIRE(general_int > 0);
    WARN(bitwise << " of " << compared << " values differ from upstream's bitwise");
}

TEST_CASE("gpu_fj: a negative coefficient's valid range is swapped (#139)", "[gpu]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // -x <= -3, x integer in [0, 10], from x = 0: the jump is to 3, the
    // least move that satisfies the row.  Unswapped, the range comes out
    // empty, the row is ignored, and x jumps to its upper bound 10.
    const HighsLp lp = tiny_lp({0.0}, {10.0}, {0.0}, {-kHighsInf}, {-3.0}, {{-1.0}}, true);
    const Setup s(lp, highs.getOptions().log_options);
    const gpu_fj::HostModel host(s.problem);
    const std::vector<double> start = {0.0};
    const gpu_fj::Climber::State state = device_state(host.pod(), start);
    REQUIRE(state.violated == std::vector<std::int32_t>{0});
    REQUIRE(state.jump == std::vector<double>{3.0});
    // The violation of 3 is removed at weight 1.
    REQUIRE(state.score == std::vector<double>{3.0});
    REQUIRE(state.good == std::vector<std::int32_t>{0});
}

TEST_CASE("gpu_fj: the objective term is subtracted from a move's score (#139)", "[gpu]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // Two unconstrained integers in [0, 10] at 5, minimising +x0 - x1: each
    // jumps to its lower bound, so x0's move improves the objective by 5
    // and x1's worsens it by 5.  At objective weight 1 the scores are +5
    // and -5; upstream's added term gives -5 and +5.
    const HighsLp lp = tiny_lp({0.0, 0.0}, {10.0, 10.0}, {1.0, -1.0}, {}, {}, {{}, {}}, true);
    const Setup s(lp, highs.getOptions().log_options);
    const gpu_fj::HostModel host(s.problem);
    const std::vector<double> start = {5.0, 5.0};
    const gpu_fj::Climber::State state = device_state(host.pod(), start, 1.0);
    REQUIRE(state.jump == std::vector<double>{0.0, 0.0});
    REQUIRE(state.score == std::vector<double>{5.0, -5.0});
    REQUIRE(state.good == std::vector<std::int32_t>{0});
}

namespace {

// A climber on `m`, set up with `init(start, seed)`.  `device` must
// outlive it.
std::unique_ptr<gpu_fj::Climber> climber_on(const gpu_fj::DeviceModel& device,
                                            std::span<const double> start, std::uint64_t seed) {
    std::string error;
    auto climber = gpu_fj::Climber::create(device, error);
    INFO(error);
    REQUIRE(climber != nullptr);
    REQUIRE(climber->init(start, seed, error));
    return climber;
}

std::unique_ptr<gpu_fj::DeviceModel> upload(const gpu_fj::GpuFjModel& m) {
    std::string error;
    auto device = gpu_fj::DeviceModel::upload(m, error);
    INFO(error);
    REQUIRE(device != nullptr);
    return device;
}

gpu_fj::Climber::State read_state(const gpu_fj::Climber& climber) {
    std::string error;
    gpu_fj::Climber::State state;
    INFO(error);
    REQUIRE(climber.read(state, error));
    return state;
}

// Upstream's cells: a constraint's are its row's, a column's one per
// constraint of each of its rows.
std::int32_t con_cells(const gpu_fj::GpuFjModel& m, std::int32_t c) {
    const std::int32_t row = m.con_row[c];
    return m.row_start[row + 1] - m.row_start[row];
}

std::int32_t col_cells(const gpu_fj::GpuFjModel& m, std::int32_t j) {
    std::int32_t n = 0;
    for (std::int32_t p = m.col_start[j]; p < m.col_start[j + 1]; ++p) {
        const std::int32_t row = m.col_row[p];
        n += m.row_con_start[row + 1] - m.row_con_start[row];
    }
    return n;
}

double con_score(const gpu_fj::GpuFjModel& m, std::int32_t c, double lhs) {
    const double d =
        m.con_sense[c] == gpu_fj::ConSense::kLte ? lhs - m.con_rhs[c] : m.con_rhs[c] - lhs;
    return -std::max(0.0, d);
}

// Close up to rounding that grows with the weights' size.
void require_near(double got, double want, double scale) {
    INFO(got << " vs " << want << " at scale " << scale);
    REQUIRE(std::fabs(got - want) <= 1e-7 * (1.0 + scale + std::fabs(want)));
}

// A climber's state after any number of steps against a recomputation from
// its point, weights and (stale) jump values: every LHS, the violated set,
// every score, and the good set as the scores' positive ones.
void require_consistent(const gpu_fj::GpuFjModel& m, const gpu_fj::Climber::State& got) {
    const auto ncon = static_cast<size_t>(m.ncon);
    double max_weight = 1.0;
    for (const double w : got.weight) {
        max_weight = std::max(max_weight, w);
    }
    std::vector<double> lhs(ncon, 0.0);
    for (std::int32_t c = 0; c < m.ncon; ++c) {
        const std::int32_t row = m.con_row[c];
        double scale = 0.0;
        for (std::int32_t p = m.row_start[row]; p < m.row_start[row + 1]; ++p) {
            lhs[c] += m.row_value[p] * got.x[m.row_index[p]];
            scale += std::fabs(m.row_value[p] * got.x[m.row_index[p]]);
        }
        require_near(got.lhs[c], lhs[c], scale);
        const bool violated = con_score(m, c, got.lhs[c]) < -m.violation_tol;
        REQUIRE((got.violated_pos[c] != -1) == violated);
        if (violated) {
            REQUIRE(got.violated[got.violated_pos[c]] == c);
        }
    }
    REQUIRE(std::ranges::count_if(got.violated_pos, [](std::int32_t p) { return p != -1; }) ==
            static_cast<std::ptrdiff_t>(got.violated.size()));
    for (std::int32_t j = 0; j < m.ncol; ++j) {
        const double shift = got.jump[j] - got.x[j];
        double score = -got.objective_weight * m.col_cost[j] * shift;
        double scale = std::fabs(score);
        for (std::int32_t p = m.col_start[j]; p < m.col_start[j + 1]; ++p) {
            const std::int32_t row = m.col_row[p];
            for (std::int32_t c = m.row_con_start[row]; c < m.row_con_start[row + 1]; ++c) {
                const double term =
                    got.weight[c] *
                    (con_score(m, c, lhs[c] + (m.col_value[p] * shift)) - con_score(m, c, lhs[c]));
                score += term;
                scale += std::fabs(term) + (max_weight * std::fabs(lhs[c]));
            }
        }
        INFO("column " << j);
        require_near(got.score[j], score, scale);
        REQUIRE((got.good_pos[j] != -1) == (got.score[j] > 0.0));
        if (got.good_pos[j] != -1) {
            REQUIRE(got.good[got.good_pos[j]] == j);
        }
    }
    REQUIRE(std::ranges::count_if(got.good_pos, [](std::int32_t p) { return p != -1; }) ==
            static_cast<std::ptrdiff_t>(got.good.size()));
}

template <typename T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

}  // namespace

TEST_CASE("gpu_fj: the step loop's chunks end at upstream's callback effort", "[gpu]") {
    STATIC_REQUIRE(gpu_fj::kCallbackEffort == kFjCallbackEffort);
}

TEST_CASE("gpu_fj: a climber finds a feasible point on presolved bundled MIPs", "[gpu]") {
    // The instances `fj::run_until_stopped` must solve at SCIP's patience,
    // from FJ's own start, one climber with no restart.
    for (const char* instance : {"egout.mps", "bell5.mps", "lseu.mps", "gt2.mps"}) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        REQUIRE(highs.presolve() == HighsStatus::kOk);
        const HighsLp lp = highs.getPresolvedLp();
        const Setup s(lp, highs.getOptions().log_options);
        const gpu_fj::HostModel host(s.problem);
        const gpu_fj::GpuFjModel m = host.pod();
        const auto device = upload(m);
        const auto climber = climber_on(*device, bound_start(m), 0);
        // SCIP's 128 x nnz (`test_core.cpp`).
        const std::uint64_t cap = 128 * static_cast<std::uint64_t>(m.nnz);
        std::string error;
        gpu_fj::Climber::Chunk chunk;
        do {
            INFO(error);
            REQUIRE(climber->run(~std::uint64_t{0}, chunk, error));
        } while (!chunk.improved && chunk.total_effort < cap);
        REQUIRE(chunk.improved);
        std::vector<double> best;
        REQUIRE(climber->read_best(best, error));
        REQUIRE(is_feasible(lp, best));
        double objective = 0.0;
        for (std::int32_t j = 0; j < m.ncol; ++j) {
            objective += m.col_cost[j] * best[j];
        }
        // `incumbentObjective` is kept incrementally, so up to rounding.
        REQUIRE(std::fabs(objective - chunk.objective) <= 1e-9 * (1.0 + std::fabs(objective)));
    }
}

TEST_CASE("gpu_fj: after its chunks a climber's state matches a recomputation, per seed", "[gpu]") {
    constexpr int kChunks = 4;
    for (const char* instance : kInstances) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        const Setup s(highs.getLp(), highs.getOptions().log_options);
        const gpu_fj::HostModel host(s.problem);
        const gpu_fj::GpuFjModel m = host.pod();
        const auto device = upload(m);
        const std::vector<double> start = interior_start(m);
        const auto first = climber_on(*device, start, 7);
        const auto second = climber_on(*device, start, 7);
        std::string error;
        // Upstream's `effortAtLastCallback` starts at 0, before `init`'s effort.
        std::uint64_t last_effort = 0;
        for (int k = 0; k < kChunks; ++k) {
            gpu_fj::Climber::Chunk a;
            gpu_fj::Climber::Chunk b;
            INFO(error);
            REQUIRE(first->run(~std::uint64_t{0}, a, error));
            REQUIRE(second->run(~std::uint64_t{0}, b, error));
            REQUIRE(a.total_effort == b.total_effort);
            REQUIRE(a.improved == b.improved);
            // A chunk with no new best ends at the first step past one
            // callback period.
            if (!a.improved) {
                REQUIRE(a.total_effort - last_effort > gpu_fj::kCallbackEffort);
            }
            last_effort = a.total_effort;
        }
        const gpu_fj::Climber::State a = read_state(*first);
        const gpu_fj::Climber::State b = read_state(*second);
        REQUIRE(a.steps > 0);
        require_consistent(m, a);
        // The same seed, the same bytes.
        REQUIRE(same_bytes(a.x, b.x));
        REQUIRE(same_bytes(a.lhs, b.lhs));
        REQUIRE(same_bytes(a.weight, b.weight));
        REQUIRE(same_bytes(a.jump, b.jump));
        REQUIRE(same_bytes(a.score, b.score));
        REQUIRE(a.violated == b.violated);
        REQUIRE(a.good == b.good);
        REQUIRE(a.bumps == b.bumps);
        std::vector<double> best_a;
        std::vector<double> best_b;
        REQUIRE(first->read_best(best_a, error));
        REQUIRE(second->read_best(best_b, error));
        REQUIRE(same_bytes(best_a, best_b));
    }
}

TEST_CASE("gpu_fj: a step's effort is upstream's", "[gpu]") {
    constexpr int kSteps = 400;
    std::uint64_t bumps = 0;
    for (const char* instance : {"p0548.mps", "bell5.mps", "flugpl.mps"}) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        const Setup s(highs.getLp(), highs.getOptions().log_options);
        const gpu_fj::HostModel host(s.problem);
        const gpu_fj::GpuFjModel m = host.pod();
        const auto device = upload(m);
        const auto climber = climber_on(*device, bound_start(m), 3);
        gpu_fj::Climber::State pre = read_state(*climber);
        // `init`: `nNonzeros`, then each `resetMoves`'s cells.
        std::uint64_t cells = 0;
        for (std::int32_t c = 0; c < m.ncon; ++c) {
            cells += static_cast<std::uint64_t>(con_cells(m, c));
        }
        REQUIRE(pre.total_effort == 2 * cells);
        std::string error;
        for (int step = 0; step < kSteps; ++step) {
            gpu_fj::Climber::Chunk chunk;
            INFO(error);
            REQUIRE(climber->run(1, chunk, error));
            const gpu_fj::Climber::State post = read_state(*climber);
            if (post.steps == pre.steps) {
                // Stopped at a callback before the step.
                REQUIRE(post.total_effort == pre.total_effort);
                continue;
            }
            REQUIRE(post.steps == pre.steps + 1);
            const std::int32_t v = post.last_var;
            auto want = static_cast<std::uint64_t>(col_cells(m, v));
            for (std::int32_t p = m.col_start[v]; p < m.col_start[v + 1]; ++p) {
                const std::int32_t row = m.col_row[p];
                for (std::int32_t c = m.row_con_start[row]; c < m.row_con_start[row + 1]; ++c) {
                    want += static_cast<std::uint64_t>(con_cells(m, c));
                }
            }
            const std::uint64_t got = post.total_effort - pre.total_effort;
            if (post.bumps != pre.bumps) {
                ++bumps;
                REQUIRE(pre.good.empty());
                auto bump = static_cast<std::uint64_t>(m.ncol);
                if (!pre.violated.empty()) {
                    bump = 0;
                    for (const std::int32_t c : pre.violated) {
                        bump += static_cast<std::uint64_t>(con_cells(m, c));
                    }
                }
                REQUIRE(got == want + bump);
            } else {
                // The sample, unless the step took a random good variable.
                const auto sample =
                    static_cast<std::uint64_t>(std::min<size_t>(25, pre.good.size()));
                REQUIRE((got == want + sample || got == want));
            }
            pre = post;
        }
    }
    REQUIRE(bumps > 0);
}
#endif
