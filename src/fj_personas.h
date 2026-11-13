#pragma once

#include "rng.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

struct ProblemView;
struct RowStructure;
struct ExactKRow;

// FJ lane personas (#206, cuOpt's CPU FJ portfolio, `portfolio.cpp`
// `apply_lane_diversification`): a slot's first worker starts from a point
// built for the structure the model has, instead of only from another
// seed.  Starts only — the moves are upstream FJ's.  Rebuilt workers
// restart from the pool as before.  Building a start is not charged to FJ's
// effort and polls no deadline: it is a few passes over the rows, capped by
// `kPersonaNnzLimit`.
enum class FjPersona : uint8_t {
    // The plain start: the resolved point, else each column's bound.
    kSeeded,
    // cuOpt's `apply_greedy_covering_start` on the plain start.
    kCovering,
    // cuOpt's `apply_exact_k_start`, then the covering start (cuOpt's lane 8).
    kExactK,
};

// cuOpt's `start_nnz_limit`: no start persona on a larger model.
inline constexpr size_t kPersonaNnzLimit = 8'000'000;
// cuOpt's `max_passes` of its covering start.
inline constexpr int kCoveringPasses = 2;

// The persona of each of `num_workers` slots.  Slot 0 is always seeded: it
// carries vanilla FJ's seed.  Slots 1.. take the personas `structure`
// supports, in the order above — covering when a uniform binary row binds
// from below (`RowStructure::lower_bound_rows`), exact-k when an exact-k
// row with `k >= 1` does — and the rest stay seeded.  All seeded above `kPersonaNnzLimit`.
std::vector<FjPersona> assign_personas(const RowStructure& structure, size_t nnz,
                                       size_t num_workers);

// cuOpt's lane count, the period of its lane assignments.
inline constexpr size_t kExchangeLanePeriod = 16;

// Whether `slot` searches with the cardinality exchange (#218): cuOpt's
// `use_cardinality_exchange` lanes 1, 7, 11 and 15, read modulo its 16
// lanes.  Slot 0 never does: it is vanilla FJ.
constexpr bool is_exchange_slot(size_t slot) {
    const size_t lane = slot % kExchangeLanePeriod;
    return lane == 1 || lane == 7 || lane == 11 || lane == 15;
}

// FJ's own start for a column with no point to take: its lower bound, else
// its upper, else 0.  `FjWorker` and the personas share it.
inline double fj_bound_start(double lower, double upper) {
    if (std::isfinite(lower)) {
        return lower;
    }
    if (std::isfinite(upper)) {
        return upper;
    }
    return 0.0;
}

// A column's FJ start from a point: `value` clamped into the bounds, or the
// bound start when `value` is not finite.  `FjWorker`, the personas and the
// GPU climbers (`fj_race`) share it, so their starts cannot drift apart.
inline double fj_clamped_start(double lower, double upper, double value) {
    return std::isfinite(value) ? std::max(lower, std::min(upper, value))
                                : fj_bound_start(lower, upper);
}

// Rewrite `start` (empty: the bound start) for `persona`.  Each step keeps
// its result only if it leaves fewer violated rows than it found, as
// cuOpt's do, so a feasible start comes back unchanged.  Reads the view
// only; safe on any thread.
void apply_persona(FjPersona persona, const ProblemView& problem, const RowStructure& structure,
                   std::vector<double>& start);

// The two steps, exposed for the tests.  Both return whether they kept
// their result.  `x` must be a full point.
bool greedy_covering_start(const ProblemView& problem, std::vector<double>& x);
bool exact_k_start(const ProblemView& problem, std::span<const ExactKRow> rows,
                   std::vector<double>& x);

// cuOpt's LP start (#217): `cpu/setup/lp.cpp` `apply_lp_rounded_start`,
// whose pump (`cpu/search/fp.cpp` `run_cpu_feasibility_pump`) runs only
// projection 0 at cuOpt's default `lp_pump_projections = 1`.  Each column of
// `lp_x` is clamped into its bounds, and an integer column is then rounded up
// with probability equal to its fractional part, else down.  Returns false,
// with `out` unspecified, when a value is not finite or an integer column has
// no integral value within its bounds: cuOpt then drops the LP start.  `lp_x`
// holds at least `problem.ncol` values.  Reads the view only; safe on any
// thread.  Not gated here: the caller applies `kPersonaNnzLimit`, which is
// also cuOpt's `lp_start_nnz_limit`.
bool lp_rounded_start(const ProblemView& problem, std::span<const double> lp_x, Rng& rng,
                      std::vector<double>& out);

// Rows of `problem` violated at `x` by more than `feastol`.
HighsInt count_violated_rows(const ProblemView& problem, const std::vector<double>& x);
