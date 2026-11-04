#pragma once

#include "heuristic_context.h"
#include "rng.h"
#include "worker_base.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ExecutionContext;
class SolutionSink;

// Line-segment search (#213, after NVIDIA cuOpt's `line_segment_search_t`
// and its `round_nearest` at f3fe1a5, Apache-2.0; read for technique, no
// code copied): walk the points `p1 + i * delta`, round each, and repair
// the ones that moved with FJ.
namespace line_segment {

// cuOpt's `check_brute_force_rounding` threshold: with at most this many
// fractional integer columns every up/down configuration is tried.
inline constexpr size_t kBruteForceMaxFractional = 8;

// cuOpt's `min_n_integer_diffs`: a rounded point that differs from the
// previous one on at most this many integer columns is not repaired by FJ.
inline constexpr size_t kMinIntegerDiffs = 5;

// A point's quality, compared lexicographically: feasible beats infeasible,
// a feasible point by objective (offset included), an infeasible one by its
// summed row violation.  cuOpt ranks by FJ's weighted violation plus the
// weighted objective, and there are no FJ weights here (a deviation, the
// same as `recombine::Offspring`'s).
struct Quality {
    bool feasible = false;
    double value = kHighsInf;

    [[nodiscard]] bool better_than(const Quality& other) const {
        if (feasible != other.feasible) {
            return feasible;
        }
        return value < other.value;
    }
};

// `x`'s quality.  Adds the effort charged (`nnz` plus `ncol`) to `effort`.
[[nodiscard]] Quality evaluate(const ProblemView& problem, const std::vector<double>& x,
                               size_t& effort);

// cuOpt's `middle_first_iterator_t`: the indices 0..n-1 breadth-first over
// the halving of [0, n-1], the middle first, then the middles of the left
// and the right half, and so on.
[[nodiscard]] std::vector<size_t> middle_first_order(size_t n);

// cuOpt's `round_nearest`, in place, returning the effort charged.  Every
// column is clamped to its bounds and an integer column within `feastol`
// of an integer is snapped to it.  The fractional integer columns are then
// rounded by the first of:
//   1. brute force, when there are at most `kBruteForceMaxFractional`: the
//      first up/down configuration, in mask order, that makes every row
//      feasible (cuOpt keeps whichever of its threads wins; lowest index is
//      a deviation), or nothing when none does;
//   2. lock rounding: a column rounds in a direction it has no lock in and
//      whose integer is within bounds, down when both are open and its cost
//      is positive, else up; when some column can do neither, nothing is
//      rounded here;
//   3. randomized nearest: with w ~ U[0, 1), t = 2w(1 - w), t = 1 - t when
//      w > 0.5, the column takes floor(v + t) within its bounds.
[[nodiscard]] size_t round_nearest(const ProblemView& problem, std::vector<double>& x, Rng& rng);

enum class Mode : uint8_t { kLocalSearch, kRecombiner };

// The parents in recombiner mode: a point must beat `best` while feasible,
// be feasible while both parents are not, or differ from both on the
// integer columns to count (cuOpt's `save_solution_if_better`).
struct Parents {
    const std::vector<double>* a = nullptr;
    const std::vector<double>* b = nullptr;
    Quality best;
    bool both_infeasible = false;
};

struct SearchParams {
    size_t n_points = 0;
    // FJ's total and stale budget per repaired point, capped by what is
    // left of `total_budget`.  cuOpt holds each FJ to 1 s and to an
    // iteration limit and a count of local minima; here an effort budget
    // alone (a deviation), and no FJ run stops sooner than
    // `kFjCallbackEffort`.
    size_t fj_budget_per_point = 0;
    // The whole search's effort, in place of cuOpt's timer, checked where
    // cuOpt checks it: after each rounding and each FJ run.
    size_t total_budget = 0;
    Mode mode = Mode::kLocalSearch;
    Parents parents;
};

struct SearchResult {
    // The best point counted, or empty in recombiner mode when none was.
    std::vector<double> point;
    Quality quality;
    size_t effort = 0;
    // Some FJ run improved the sink's incumbent.
    bool improved = false;
};

// cuOpt's `search_line_segment`: the points `p1 + i * delta`, i = 1..n,
// visited in `middle_first_order`, each rounded by `round_nearest` and
// counted.  A rounding that differs from the previous one (initially `p1`)
// on more than `kMinIntegerDiffs` integer columns is then repaired by a
// fresh `FjWorker` from it, which offers its own finds to `sink`; FJ's best
// feasible point is counted (its infeasible end point is not exposed, a
// deviation).  In local-search mode `p1` is the start and the result is it
// unless a point beats it; in recombiner mode only a point that passes
// `Parents` counts.  cuOpt keeps the best and the best feasible apart and
// returns the latter when there is one; under the lexicographic `Quality`
// the best is that point.  `trace.effort_base` is kept current.
[[nodiscard]] SearchResult search(const ProblemView& problem, const ExecutionContext& exec,
                                  SolutionSink& sink, WorkerTrace& trace, Rng& rng,
                                  const std::vector<double>& p1, const std::vector<double>& delta,
                                  const SearchParams& params);

}  // namespace line_segment
