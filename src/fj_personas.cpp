#include "fj_personas.h"

#include "heuristic_common.h"
#include "heuristic_context.h"
#include "structure.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <utility>

// Deviations from cuOpt's starts (`cpu/starts/covering.cpp`,
// `cpu/starts/cardinality.cpp`), all for reproducibility or to read this
// model's tolerances:
// - No wall-clock budget of its own (cuOpt: 0.4 s covering, 0.5 s exact-k).
//   A start is O(rows log rows + passes * nnz log width), and
//   `kPersonaNnzLimit` is cuOpt's own size gate; a clock would make the
//   start, and so the whole FJ trajectory, depend on the machine.  It is
//   charged to FJ's effort and stops at the solve's deadline instead
//   (`StartBudget`, #326), which changes only a run the clock ends.
// - Stable sorts, so ties break by row or entry order on every standard
//   library rather than by `std::sort`'s implementation.
// - `problem.feastol` where cuOpt hard-codes `1e-6` (HiGHS's default
//   `mip_feasibility_tolerance` is that value), and a plain row sum where it
//   uses a compensated one.

void StartBudget::poll() {
    next_poll_ = effort_ + kStartPollEffort;
    expired_ = exec_.past_deadline();
}

namespace {

// A sort's charge: `n log2 n` entries read.
size_t sort_effort(size_t n) {
    return n * static_cast<size_t>(std::bit_width(n));
}

double row_activity(const ProblemView& problem, HighsInt row, const std::vector<double>& x) {
    double sum = 0.0;
    for (HighsInt p = (*problem.ar_start)[row]; p < (*problem.ar_start)[row + 1]; ++p) {
        sum += (*problem.ar_value)[p] * x[(*problem.ar_index)[p]];
    }
    return sum;
}

struct RepairMove {
    double effect;
    HighsInt var;
    double coeff;
    double new_val;
};

// cuOpt's `collect_row_repair_moves`: every integer column of the row that
// can move one step in the direction that shifts the row sum by
// `direction`, by decreasing effect.
void collect_repair_moves(const ProblemView& problem, HighsInt row, double direction,
                          const std::vector<double>& x, std::vector<RepairMove>& out) {
    const double tol = problem.feastol;
    out.clear();
    for (HighsInt p = (*problem.ar_start)[row]; p < (*problem.ar_start)[row + 1]; ++p) {
        const HighsInt var = (*problem.ar_index)[p];
        if (!is_integer(*problem.integrality, var)) {
            continue;
        }
        const double coeff = (*problem.ar_value)[p];
        const double val = x[var];
        const double lb = (*problem.col_lower)[var];
        const double ub = (*problem.col_upper)[var];
        const bool is_bin = problem.binary[var] != 0;
        // Raising the column shifts the sum by `direction * coeff`.
        const double raise = direction * coeff;
        if (raise > 0 && val < ub - tol) {
            const double new_val = is_bin ? 1.0 : std::floor(val) + 1;
            if (new_val > val && new_val <= ub + tol) {
                out.push_back({raise, var, coeff, new_val});
            }
        } else if (raise < 0 && val > lb + tol) {
            const double new_val = is_bin ? 0.0 : std::ceil(val) - 1;
            if (new_val < val && new_val >= lb - tol) {
                out.push_back({-raise, var, coeff, new_val});
            }
        }
    }
    std::ranges::stable_sort(
        out, [](const RepairMove& a, const RepairMove& b) { return a.effect > b.effect; });
}

HighsInt row_width(const ProblemView& problem, HighsInt row) {
    return (*problem.ar_start)[row + 1] - (*problem.ar_start)[row];
}

// The side a row with activity `sum` violates: the direction to move its
// activity in (+1 up, -1 down) and the side to reach, or nothing when the
// row holds.  An equality is driven to its side from either direction.
struct Violation {
    double direction;
    double target;
};

std::optional<Violation> violated_side(double lb, double ub, double sum, double tol) {
    const bool has_lb = std::isfinite(lb);
    const bool has_ub = std::isfinite(ub);
    // `detect_structure`'s equality test, so an exact-k row is an equality
    // here too whatever the feasibility tolerance.
    const bool is_equality = has_lb && has_ub && std::abs(lb - ub) <= kExactKTol;
    if (is_equality && std::abs(sum - lb) > tol) {
        return Violation{sum < lb ? 1.0 : -1.0, lb};
    }
    if (has_lb && sum < lb - tol) {
        return Violation{1.0, lb};
    }
    if (has_ub && sum > ub + tol) {
        return Violation{-1.0, ub};
    }
    return std::nullopt;
}

// One covering pass over `order`: each row violated at `x` repaired by
// cuOpt's greedy one-step moves, equalities driven to their side.  Each row
// is charged its width plus one, so an empty row is charged too; a repaired
// one its width again and the sort of its moves.
void covering_pass(const ProblemView& problem, const std::vector<HighsInt>& order,
                   std::vector<double>& x, std::vector<RepairMove>& candidates,
                   StartBudget& budget) {
    const double tol = problem.feastol;
    for (const HighsInt row : order) {
        const HighsInt width = row_width(problem, row);
        if (budget.charge(static_cast<size_t>(width) + 1)) {
            return;
        }
        if (width == 0) {
            continue;
        }
        double sum = row_activity(problem, row, x);
        const auto violation =
            violated_side((*problem.row_lower)[row], (*problem.row_upper)[row], sum, tol);
        if (!violation) {
            continue;
        }
        const auto [direction, target] = *violation;
        collect_repair_moves(problem, row, direction, x, candidates);
        budget.charge(static_cast<size_t>(width) + sort_effort(candidates.size()));
        for (const RepairMove& m : candidates) {
            if (direction > 0 ? sum >= target - tol : sum <= target + tol) {
                break;
            }
            sum += m.coeff * (m.new_val - x[m.var]);
            x[m.var] = m.new_val;
        }
    }
}

// How many exact-k rows each column appears in. A deadline cut leaves it
// partial; the caller checks `budget.expired()`.
std::vector<HighsInt> exact_k_degrees(const ProblemView& problem, std::span<const ExactKRow> rows,
                                      StartBudget& budget) {
    const auto& start = *problem.ar_start;
    const auto& index = *problem.ar_index;
    std::vector<HighsInt> degree(static_cast<size_t>(problem.ncol), 0);
    for (const ExactKRow& r : rows) {
        if (budget.charge(static_cast<size_t>(row_width(problem, r.row)) + 1)) {
            break;
        }
        for (HighsInt p = start[r.row]; p < start[r.row + 1]; ++p) {
            ++degree[index[p]];
        }
    }
    return degree;
}

// Each column's decided value, -1 if undecided: rows in order, each fixing
// its free columns, lowest degree first, to reach exactly k ones. A deadline
// cut leaves it partial; the caller checks `budget.expired()`.
std::vector<int8_t> exact_k_states(const ProblemView& problem, std::span<const ExactKRow> rows,
                                   const std::vector<HighsInt>& degree, StartBudget& budget) {
    const auto& start = *problem.ar_start;
    const auto& index = *problem.ar_index;
    std::vector<int8_t> state(static_cast<size_t>(problem.ncol), -1);
    std::vector<HighsInt> free_vars;
    for (const ExactKRow& r : rows) {
        if (budget.charge(static_cast<size_t>(row_width(problem, r.row)) + 1)) {
            break;
        }
        HighsInt selected = 0;
        free_vars.clear();
        for (HighsInt p = start[r.row]; p < start[r.row + 1]; ++p) {
            const HighsInt var = index[p];
            selected += state[var] == 1 ? 1 : 0;
            if (state[var] < 0) {
                free_vars.push_back(var);
            }
        }
        const HighsInt needed = r.k - selected;
        if (needed < 0 || std::cmp_less(free_vars.size(), needed)) {
            continue;
        }
        std::ranges::stable_sort(free_vars,
                                 [&](HighsInt a, HighsInt b) { return degree[a] < degree[b]; });
        budget.charge(sort_effort(free_vars.size()));
        for (size_t i = 0; i < free_vars.size(); ++i) {
            state[free_vars[i]] = static_cast<int8_t>(std::cmp_less(i, needed) ? 1 : 0);
        }
    }
    return state;
}

}  // namespace

HighsInt count_violated_rows(const ProblemView& problem, const std::vector<double>& x,
                             StartBudget& budget) {
    HighsInt violated = 0;
    for (HighsInt row = 0; row < problem.nrow; ++row) {
        if (budget.charge(static_cast<size_t>(row_width(problem, row)) + 1)) {
            break;
        }
        violated += is_row_violated(row_activity(problem, row, x), (*problem.row_lower)[row],
                                    (*problem.row_upper)[row], problem.feastol)
                        ? 1
                        : 0;
    }
    return violated;
}

bool greedy_covering_start(const ProblemView& problem, std::vector<double>& x,
                           StartBudget& budget) {
    const HighsInt baseline = count_violated_rows(problem, x, budget);
    if (budget.expired()) {
        return false;
    }
    std::vector<double> candidate = x;
    std::vector<HighsInt> order(static_cast<size_t>(problem.nrow));
    std::ranges::iota(order, HighsInt{0});
    std::ranges::stable_sort(order, [&](HighsInt a, HighsInt b) {
        return row_width(problem, a) < row_width(problem, b);
    });
    budget.charge(sort_effort(order.size()));
    std::vector<RepairMove> candidates;
    for (int pass = 0; pass < kCoveringPasses && !budget.expired(); ++pass) {
        covering_pass(problem, order, candidate, candidates, budget);
    }
    const HighsInt violated = count_violated_rows(problem, candidate, budget);
    if (budget.expired() || violated >= baseline) {
        return false;
    }
    x = std::move(candidate);
    return true;
}

bool exact_k_start(const ProblemView& problem, std::span<const ExactKRow> rows,
                   std::vector<double>& x, StartBudget& budget) {
    if (rows.empty()) {
        return false;
    }
    std::vector<ExactKRow> by_width(rows.begin(), rows.end());
    std::ranges::stable_sort(by_width, [&](const ExactKRow& a, const ExactKRow& b) {
        return row_width(problem, a.row) < row_width(problem, b.row);
    });
    if (budget.charge(sort_effort(by_width.size()))) {
        return false;
    }

    const std::vector<HighsInt> degree = exact_k_degrees(problem, by_width, budget);
    if (budget.expired()) {
        return false;
    }
    const std::vector<int8_t> state = exact_k_states(problem, by_width, degree, budget);
    if (budget.expired()) {
        return false;
    }

    const HighsInt baseline = count_violated_rows(problem, x, budget);
    std::vector<double> candidate = x;
    for (HighsInt var = 0; var < problem.ncol; ++var) {
        if (state[var] >= 0) {
            candidate[var] = state[var];
        }
    }
    budget.charge(static_cast<size_t>(problem.ncol));
    const HighsInt violated = count_violated_rows(problem, candidate, budget);
    if (budget.expired() || violated >= baseline) {
        return false;
    }
    x = std::move(candidate);
    return true;
}

std::vector<FjPersona> assign_personas(const RowStructure& structure, size_t nnz,
                                       size_t num_workers) {
    std::vector<FjPersona> out(num_workers, FjPersona::kSeeded);
    if (nnz > kPersonaNnzLimit) {
        return out;
    }
    std::vector<FjPersona> supported;
    if (structure.lower_bound_rows > 0) {
        supported.push_back(FjPersona::kCovering);
    }
    // A `k = 0` row only re-fixes what the bound start already has at zero.
    if (std::ranges::any_of(structure.exact_k, [](const ExactKRow& r) { return r.k >= 1; })) {
        supported.push_back(FjPersona::kExactK);
    }
    for (size_t i = 0; i < supported.size() && i + 1 < out.size(); ++i) {
        out[i + 1] = supported[i];
    }
    return out;
}

bool lp_rounded_start(const ProblemView& problem, std::span<const double> lp_x, Rng& rng,
                      std::vector<double>& out) {
    const auto ncol = static_cast<size_t>(problem.ncol);
    out.resize(ncol);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (HighsInt col = 0; col < problem.ncol; ++col) {
        const double lower = (*problem.col_lower)[col];
        const double upper = (*problem.col_upper)[col];
        // Not `std::clamp`: a column with no integral value inside its
        // bounds has them crossed once rounded inward, which `std::clamp`
        // does not allow; this form lands on `lower` and fails below.
        // The LP value is tested before the clamp, which would map a NaN
        // onto `upper` (`std::min` keeps its first argument when the
        // comparison is false) and so hide it.
        if (!std::isfinite(lp_x[col])) {
            return false;
        }
        double value = std::max(lower, std::min(upper, lp_x[col]));
        if (is_integer(*problem.integrality, col)) {
            const double fraction = value - std::floor(value);
            value = coin(rng) < fraction ? std::ceil(value) : std::floor(value);
            if (value < lower || value > upper) {
                return false;
            }
        }
        out[col] = value;
    }
    return true;
}

void apply_persona(FjPersona persona, const ProblemView& problem, const RowStructure& structure,
                   std::vector<double>& start, StartBudget& budget) {
    // The zero charge polls the deadline before anything is touched.
    if (persona == FjPersona::kSeeded || budget.charge(0)) {
        return;
    }
    // The point `FjWorker` would start from: the resolved one clamped into
    // the bounds, a column it has no finite value for (or every column, for
    // an empty start) at its bound start.
    start.resize(static_cast<size_t>(problem.ncol), kHighsInf);
    for (HighsInt col = 0; col < problem.ncol; ++col) {
        const double lower = (*problem.col_lower)[col];
        const double upper = (*problem.col_upper)[col];
        start[col] = fj_clamped_start(lower, upper, start[col]);
    }
    budget.charge(static_cast<size_t>(problem.ncol));
    if (persona == FjPersona::kExactK) {
        exact_k_start(problem, structure.exact_k, start, budget);
    }
    greedy_covering_start(problem, start, budget);
}
