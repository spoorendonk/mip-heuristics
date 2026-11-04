#include "line_segment.h"

#include "fj_worker.h"
#include "heuristic_common.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <utility>

namespace line_segment {

Quality evaluate(const ProblemView& problem, const std::vector<double>& x, size_t& effort) {
    const double tol = problem.feastol;
    double violation = 0.0;
    bool feasible = true;
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        const double v = x[j];
        const double bound_violation =
            row_violation(v, (*problem.col_lower)[j], (*problem.col_upper)[j]);
        // A fractional integer column counts as violation too, so a
        // fractional point never ranks above an integral one that violates
        // its rows by less.
        const double integrality_violation =
            is_integer(*problem.integrality, j) ? std::abs(v - std::round(v)) : 0.0;
        violation += bound_violation + integrality_violation;
        feasible = feasible && bound_violation <= tol && integrality_violation <= tol;
    }
    const auto& start = *problem.ar_start;
    const auto& index = *problem.ar_index;
    const auto& value = *problem.ar_value;
    for (HighsInt i = 0; i < problem.nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = start[i]; k < start[i + 1]; ++k) {
            lhs += value[k] * x[index[k]];
        }
        const double lo = (*problem.row_lower)[i];
        const double hi = (*problem.row_upper)[i];
        violation += row_violation(lhs, lo, hi);
        feasible = feasible && !is_row_violated(lhs, lo, hi, tol);
    }
    effort += problem.nnz + static_cast<size_t>(problem.ncol);
    if (feasible) {
        return {.feasible = true, .value = problem.objective(x)};
    }
    return {.feasible = false, .value = violation};
}

std::vector<size_t> middle_first_order(size_t n) {
    std::vector<size_t> order;
    order.reserve(n);
    // Half-open ranges [left, right), consumed front to back: a FIFO over
    // a vector that only grows to at most 2n + 1 entries.
    std::vector<std::pair<size_t, size_t>> queue;
    queue.reserve((2 * n) + 1);
    queue.emplace_back(0, n);
    for (size_t head = 0; head < queue.size(); ++head) {
        const auto [left, right] = queue[head];
        if (left >= right) {
            continue;
        }
        const size_t mid = left + ((right - 1 - left) / 2);
        order.push_back(mid);
        queue.emplace_back(left, mid);
        queue.emplace_back(mid + 1, right);
    }
    return order;
}

namespace {

// The rows the columns of `frac` touch, and every row's activity at `x`.
struct TouchedRows {
    std::vector<double> base;
    std::vector<HighsInt> rows;
};

// The rows of `frac`, or nothing when a row no column of `frac` touches is
// violated at `x`: that row is fixed, and no configuration can be feasible.
std::optional<TouchedRows> touched_rows(const ProblemView& problem, const std::vector<double>& x,
                                        const std::vector<HighsInt>& frac, size_t& effort) {
    const CscMatrix& csc = *problem.csc;
    const auto& start = *problem.ar_start;
    const auto& index = *problem.ar_index;
    const auto& value = *problem.ar_value;
    TouchedRows out{.base = std::vector<double>(problem.nrow, 0.0), .rows = {}};
    for (HighsInt i = 0; i < problem.nrow; ++i) {
        for (HighsInt k = start[i]; k < start[i + 1]; ++k) {
            out.base[i] += value[k] * x[index[k]];
        }
    }
    effort += problem.nnz;
    std::vector<char> touched(problem.nrow, 0);
    for (const HighsInt j : frac) {
        for (HighsInt k = csc.col_start[j]; k < csc.col_start[j + 1]; ++k) {
            const HighsInt i = csc.col_row[k];
            if (touched[i] == 0) {
                touched[i] = 1;
                out.rows.push_back(i);
            }
        }
    }
    for (HighsInt i = 0; i < problem.nrow; ++i) {
        if (touched[i] == 0 && is_row_violated(out.base[i], (*problem.row_lower)[i],
                                               (*problem.row_upper)[i], problem.feastol)) {
            return std::nullopt;
        }
    }
    return out;
}

// Whether the configuration `mask` of `frac` (bit b set: `frac[b]` one
// above its floor) satisfies every touched row.  `lhs` is scratch.
bool config_feasible(const ProblemView& problem, const std::vector<HighsInt>& frac, size_t mask,
                     const TouchedRows& touched, std::vector<double>& lhs, size_t& effort) {
    const CscMatrix& csc = *problem.csc;
    for (const HighsInt i : touched.rows) {
        lhs[i] = touched.base[i];
    }
    for (size_t b = 0; b < frac.size(); ++b) {
        if (((mask >> b) & 1U) == 0) {
            continue;
        }
        const HighsInt j = frac[b];
        for (HighsInt k = csc.col_start[j]; k < csc.col_start[j + 1]; ++k) {
            lhs[csc.col_row[k]] += csc.col_val[k];
            ++effort;
        }
    }
    effort += touched.rows.size();
    return std::ranges::none_of(touched.rows, [&](HighsInt i) {
        return is_row_violated(lhs[i], (*problem.row_lower)[i], (*problem.row_upper)[i],
                               problem.feastol);
    });
}

// Brute force over the up/down configurations of `frac` (each at its floor
// in `x` on entry).  Applies the first that makes every row feasible and
// returns whether there was one.
bool brute_force(const ProblemView& problem, std::vector<double>& x,
                 const std::vector<HighsInt>& frac, size_t& effort) {
    const std::optional<TouchedRows> touched = touched_rows(problem, x, frac, effort);
    if (!touched) {
        return false;
    }
    // The columns whose next integer is above their (fractional) upper
    // bound: a configuration rounding one of them up is out of bounds.
    size_t no_up = 0;
    for (size_t b = 0; b < frac.size(); ++b) {
        if (x[frac[b]] + 1.0 > (*problem.col_upper)[frac[b]] + problem.feastol) {
            no_up |= size_t{1} << b;
        }
    }
    std::vector<double> lhs(problem.nrow);
    const size_t n_configs = size_t{1} << frac.size();
    for (size_t mask = 0; mask < n_configs; ++mask) {
        if ((mask & no_up) != 0 || !config_feasible(problem, frac, mask, *touched, lhs, effort)) {
            continue;
        }
        for (size_t b = 0; b < frac.size(); ++b) {
            if (((mask >> b) & 1U) != 0) {
                x[frac[b]] += 1.0;
            }
        }
        return true;
    }
    return false;
}

// cuOpt's `simple_rounding_kernel`: every column of `frac` rounds in a
// lock-free direction, or none does.
bool lock_rounding(const ProblemView& problem, std::vector<double>& x,
                   const std::vector<HighsInt>& frac) {
    std::vector<double> rounded;
    rounded.reserve(frac.size());
    for (const HighsInt j : frac) {
        const double v = x[j];
        const bool can_up =
            (*problem.uplocks)[j] == 0 && std::ceil(v) <= (*problem.col_upper)[j] + problem.feastol;
        const bool can_down = (*problem.downlocks)[j] == 0 &&
                              std::floor(v) >= (*problem.col_lower)[j] - problem.feastol;
        if (can_up && can_down) {
            rounded.push_back((*problem.col_cost)[j] > 0 ? std::floor(v) : std::ceil(v));
        } else if (can_up) {
            rounded.push_back(std::ceil(v));
        } else if (can_down) {
            rounded.push_back(std::floor(v));
        } else {
            return false;
        }
    }
    for (size_t b = 0; b < frac.size(); ++b) {
        x[frac[b]] = rounded[b];
    }
    return true;
}

// cuOpt's `nearest_rounding_kernel`.
void random_nearest(const ProblemView& problem, std::vector<double>& x,
                    const std::vector<HighsInt>& frac, Rng& rng) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (const HighsInt j : frac) {
        const double w = coin(rng);
        double t = 2.0 * w * (1.0 - w);
        if (w > 0.5) {
            t = 1.0 - t;
        }
        const double lo = std::ceil((*problem.col_lower)[j] - problem.feastol);
        const double hi = std::floor((*problem.col_upper)[j] + problem.feastol);
        x[j] = std::clamp(std::floor(x[j] + t), lo, hi);
    }
}

}  // namespace

size_t round_nearest(const ProblemView& problem, std::vector<double>& x, Rng& rng) {
    std::vector<HighsInt> frac;
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        x[j] = std::clamp(x[j], (*problem.col_lower)[j], (*problem.col_upper)[j]);
        if (!is_integer(*problem.integrality, j)) {
            continue;
        }
        const double r = std::round(x[j]);
        if (std::abs(x[j] - r) <= problem.feastol) {
            x[j] = r;
        } else {
            frac.push_back(j);
        }
    }
    auto effort = static_cast<size_t>(problem.ncol);
    if (frac.empty()) {
        return effort;
    }
    if (frac.size() <= kBruteForceMaxFractional) {
        std::vector<double> floors = x;
        for (const HighsInt j : frac) {
            floors[j] = std::floor(x[j]);
        }
        if (brute_force(problem, floors, frac, effort)) {
            x = std::move(floors);
            return effort;
        }
    }
    effort += frac.size();
    if (!lock_rounding(problem, x, frac)) {
        random_nearest(problem, x, frac, rng);
    }
    return effort;
}

namespace {

// Integer columns where `x` and `y` differ by more than `tol`.
size_t integer_diffs(const ProblemView& problem, const std::vector<double>& x,
                     const std::vector<double>& y) {
    size_t diffs = 0;
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        if (is_integer(*problem.integrality, j) && std::abs(x[j] - y[j]) > problem.feastol) {
            ++diffs;
        }
    }
    return diffs;
}

// The running best of one search, and cuOpt's `save_solution_if_better`.
struct Best {
    const ProblemView& problem;
    const SearchParams& params;
    std::vector<double> point;
    Quality quality;

    void offer(const std::vector<double>& x, const Quality& q) {
        if (!q.better_than(quality)) {
            return;
        }
        if (params.mode == Mode::kRecombiner) {
            const Parents& parents = params.parents;
            const bool counts_anyway =
                q.feasible && (q.better_than(parents.best) || parents.both_infeasible);
            if (!counts_anyway && (integer_diffs(problem, x, *parents.a) == 0 ||
                                   integer_diffs(problem, x, *parents.b) == 0)) {
                return;
            }
        }
        point = x;
        quality = q;
    }
};

}  // namespace

SearchResult search(const ProblemView& problem, const ExecutionContext& exec, SolutionSink& sink,
                    WorkerTrace& trace, Rng& rng, const std::vector<double>& p1,
                    const std::vector<double>& delta, const SearchParams& params) {
    SearchResult result;
    // Everything this search charges itself, kept in the trace as it goes;
    // FJ's runs reach it through `traced_effort()`.
    const auto charge = [&](size_t effort) {
        result.effort += effort;
        trace.effort_base += effort;
    };
    const auto evaluated = [&](const std::vector<double>& x) {
        size_t effort = 0;
        const Quality q = evaluate(problem, x, effort);
        charge(effort);
        return q;
    };
    const auto out_of_budget = [&] {
        return result.effort >= params.total_budget || exec.past_deadline();
    };

    Best best{.problem = problem, .params = params, .point = {}, .quality = {}};
    if (params.mode == Mode::kLocalSearch) {
        best.point = p1;
        best.quality = evaluated(p1);
    }
    std::vector<double> previous = p1;
    std::vector<double> x(problem.ncol);
    for (const size_t index : middle_first_order(params.n_points)) {
        const auto step = static_cast<double>(index + 1);
        for (HighsInt j = 0; j < problem.ncol; ++j) {
            x[j] = p1[j] + (step * delta[j]);
        }
        charge(round_nearest(problem, x, rng));
        best.offer(x, evaluated(x));
        if (out_of_budget()) {
            break;
        }
        const size_t diffs = integer_diffs(problem, x, previous);
        charge(static_cast<size_t>(problem.ncol));
        previous = x;
        if (diffs <= kMinIntegerDiffs) {
            continue;
        }
        // The charge above can spend what was left; FJ gets no budget then.
        if (out_of_budget()) {
            break;
        }
        const size_t fj_budget =
            std::min(params.fj_budget_per_point, params.total_budget - result.effort);
        FjWorker fj(problem, exec, sink, fj_budget, fj_budget, static_cast<uint32_t>(rng()), x,
                    trace);
        const AttemptResult attempt = fj.run_attempt(fj_budget);
        trace.effort_base = fj.traced_effort();
        result.effort += attempt.effort;
        result.improved = result.improved || attempt.found_improvement;
        if (!fj.best_found().empty()) {
            best.offer(fj.best_found(), evaluated(fj.best_found()));
        }
        if (out_of_budget()) {
            break;
        }
    }
    result.point = std::move(best.point);
    result.quality = best.quality;
    return result;
}

}  // namespace line_segment
