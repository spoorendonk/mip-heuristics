#include "fj_personas.h"

#include "heuristic_context.h"
#include "structure.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

// Deviations from cuOpt's starts (`cpu/starts/covering.cpp`,
// `cpu/starts/cardinality.cpp`), all for reproducibility or to read this
// model's tolerances:
// - No wall-clock budget (cuOpt: 0.4 s covering, 0.5 s exact-k).  A start
//   is O(rows log rows + passes * nnz log width), and `kPersonaNnzLimit`
//   is cuOpt's own size gate; a clock would make the start, and so the
//   whole FJ trajectory, depend on the machine.
// - Stable sorts, so ties break by row or entry order on every standard
//   library rather than by `std::sort`'s implementation.
// - `problem.feastol` where cuOpt hard-codes `1e-6` (HiGHS's default
//   `mip_feasibility_tolerance` is that value), and a plain row sum where it
//   uses a compensated one.

namespace {

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
        if ((*problem.integrality)[var] == HighsVarType::kContinuous) {
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

// One covering pass over `order`: each row violated at `x` repaired by
// cuOpt's greedy one-step moves, equalities driven to their side.
void covering_pass(const ProblemView& problem, const std::vector<HighsInt>& order,
                   std::vector<double>& x, std::vector<RepairMove>& candidates) {
    const double tol = problem.feastol;
    for (const HighsInt row : order) {
        if (row_width(problem, row) == 0) {
            continue;
        }
        const double lb = (*problem.row_lower)[row];
        const double ub = (*problem.row_upper)[row];
        const bool has_lb = std::isfinite(lb);
        const bool has_ub = std::isfinite(ub);
        if (!has_lb && !has_ub) {
            continue;
        }
        double sum = row_activity(problem, row, x);
        const bool is_equality = has_lb && has_ub && std::abs(lb - ub) < tol;
        double direction = 0;
        double target = 0;
        if (is_equality && std::abs(sum - lb) > tol) {
            direction = sum < lb ? 1.0 : -1.0;
            target = lb;
        } else if (has_lb && sum < lb - tol) {
            direction = 1;
            target = lb;
        } else if (has_ub && sum > ub + tol) {
            direction = -1;
            target = ub;
        } else {
            continue;
        }
        collect_repair_moves(problem, row, direction, x, candidates);
        for (const RepairMove& m : candidates) {
            if (direction > 0 ? sum >= target - tol : sum <= target + tol) {
                break;
            }
            sum += m.coeff * (m.new_val - x[m.var]);
            x[m.var] = m.new_val;
        }
    }
}

}  // namespace

HighsInt count_violated_rows(const ProblemView& problem, const std::vector<double>& x) {
    HighsInt violated = 0;
    for (HighsInt row = 0; row < problem.nrow; ++row) {
        const double sum = row_activity(problem, row, x);
        if (sum < (*problem.row_lower)[row] - problem.feastol ||
            sum > (*problem.row_upper)[row] + problem.feastol) {
            ++violated;
        }
    }
    return violated;
}

bool greedy_covering_start(const ProblemView& problem, std::vector<double>& x) {
    const HighsInt baseline = count_violated_rows(problem, x);
    std::vector<double> candidate = x;
    std::vector<HighsInt> order(static_cast<size_t>(problem.nrow));
    std::iota(order.begin(), order.end(), HighsInt{0});
    std::ranges::stable_sort(order, [&](HighsInt a, HighsInt b) {
        return row_width(problem, a) < row_width(problem, b);
    });
    std::vector<RepairMove> candidates;
    for (int pass = 0; pass < kCoveringPasses; ++pass) {
        covering_pass(problem, order, candidate, candidates);
    }
    if (count_violated_rows(problem, candidate) >= baseline) {
        return false;
    }
    x = std::move(candidate);
    return true;
}

bool exact_k_start(const ProblemView& problem, std::span<const ExactKRow> rows,
                   std::vector<double>& x) {
    if (rows.empty()) {
        return false;
    }
    std::vector<ExactKRow> by_width(rows.begin(), rows.end());
    std::ranges::stable_sort(by_width, [&](const ExactKRow& a, const ExactKRow& b) {
        return row_width(problem, a.row) < row_width(problem, b.row);
    });

    const auto& start = *problem.ar_start;
    const auto& index = *problem.ar_index;
    std::vector<HighsInt> degree(static_cast<size_t>(problem.ncol), 0);
    for (const ExactKRow& r : by_width) {
        for (HighsInt p = start[r.row]; p < start[r.row + 1]; ++p) {
            ++degree[index[p]];
        }
    }

    // -1 undecided, else the column's value.
    std::vector<int8_t> state(static_cast<size_t>(problem.ncol), -1);
    std::vector<HighsInt> free_vars;
    for (const ExactKRow& r : by_width) {
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
        if (needed < 0 || static_cast<HighsInt>(free_vars.size()) < needed) {
            continue;
        }
        std::ranges::stable_sort(free_vars,
                                 [&](HighsInt a, HighsInt b) { return degree[a] < degree[b]; });
        for (size_t i = 0; i < free_vars.size(); ++i) {
            state[free_vars[i]] = static_cast<int8_t>(std::cmp_less(i, needed) ? 1 : 0);
        }
    }

    const HighsInt baseline = count_violated_rows(problem, x);
    std::vector<double> candidate = x;
    for (HighsInt var = 0; var < problem.ncol; ++var) {
        if (state[var] >= 0) {
            candidate[var] = state[var];
        }
    }
    if (count_violated_rows(problem, candidate) >= baseline) {
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
    if (structure.count_of(RowKind::kCovering) + structure.count_of(RowKind::kPartitioning) +
            structure.count_of(RowKind::kCardinality) >
        0) {
        supported.push_back(FjPersona::kCovering);
    }
    if (!structure.exact_k.empty()) {
        supported.push_back(FjPersona::kExactK);
    }
    for (size_t i = 0; i < supported.size() && i + 1 < out.size(); ++i) {
        out[i + 1] = supported[i];
    }
    return out;
}

void apply_persona(FjPersona persona, const ProblemView& problem, const RowStructure& structure,
                   std::vector<double>& start) {
    if (persona == FjPersona::kSeeded) {
        return;
    }
    // The point `FjWorker` would start from: the resolved one clamped into
    // the bounds, a column it has no finite value for (or every column, for
    // an empty start) at its bound start.
    start.resize(static_cast<size_t>(problem.ncol), kHighsInf);
    for (HighsInt col = 0; col < problem.ncol; ++col) {
        const double lower = (*problem.col_lower)[col];
        const double upper = (*problem.col_upper)[col];
        start[col] = std::isfinite(start[col]) ? std::max(lower, std::min(upper, start[col]))
                                               : fj_bound_start(lower, upper);
    }
    if (persona == FjPersona::kExactK) {
        exact_k_start(problem, structure.exact_k, start);
    }
    greedy_covering_start(problem, start);
}
