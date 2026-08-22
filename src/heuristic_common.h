#pragma once

#include "lp_data/HConst.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Variable state used during fix-and-propagate.
struct VarState {
    double lb, ub, val;
    bool fixed;
};

struct HeuristicResult {
    bool found_feasible = false;
    std::vector<double> solution;
    double objective = std::numeric_limits<double>::infinity();
    size_t effort = 0;

    static HeuristicResult failed(size_t e = 0) {
        HeuristicResult r;
        r.effort = e;
        return r;
    }
};

struct CscMatrix {
    std::vector<HighsInt> col_start;
    std::vector<HighsInt> col_row;
    std::vector<double> col_val;
};

inline CscMatrix build_csc(HighsInt ncol, HighsInt nrow, const std::vector<HighsInt>& ar_start,
                           const std::vector<HighsInt>& ar_index,
                           const std::vector<double>& ar_value) {
    const auto nnz = static_cast<HighsInt>(ar_index.size());
    CscMatrix csc;
    csc.col_start.assign(ncol + 1, 0);
    for (HighsInt k = 0; k < nnz; ++k) {
        csc.col_start[ar_index[k] + 1]++;
    }
    for (HighsInt j = 0; j < ncol; ++j) {
        csc.col_start[j + 1] += csc.col_start[j];
    }
    csc.col_row.resize(nnz);
    csc.col_val.resize(nnz);
    {
        std::vector<HighsInt> pos(csc.col_start);
        for (HighsInt i = 0; i < nrow; ++i) {
            for (HighsInt k = ar_start[i]; k < ar_start[i + 1]; ++k) {
                HighsInt j = ar_index[k];
                csc.col_row[pos[j]] = i;
                csc.col_val[pos[j]] = ar_value[k];
                pos[j]++;
            }
        }
    }
    return csc;
}

inline bool is_integer(const std::vector<HighsVarType>& integrality, HighsInt j) {
    return integrality[j] != HighsVarType::kContinuous;
}

// Tolerance hierarchy:
//   feastol  (~1e-6)  — from solver, used for feasibility checks
//   kViolTol (5e-7)   — local_mip local-search violation threshold
//   1e-15             — numerical zero (avoids division/move on zero-delta)

// Row violation: how much lhs exceeds [lo, hi] bounds.
inline double row_violation(double lhs, double lo, double hi) {
    return std::max(0.0, lhs - hi) + std::max(0.0, lo - lhs);
}

// Whether a row is violated beyond the given feasibility tolerance.
inline bool is_row_violated(double lhs, double lo, double hi, double feastol) {
    return lhs > hi + feastol || lhs < lo - feastol;
}

// Clamp value to [lb, ub], rounding if integer.
inline double clamp_round(double val, double lb, double ub, bool integer) {
    if (integer) {
        val = std::round(val);
    }
    return std::max(lb, std::min(ub, val));
}

// Window for clamping the integer perturbation shift range when one or
// both variable bounds are non-finite (kHighsInf, which IS infinity per
// HiGHS HConst.h) or finite-but-huge.  Used by both
// `local_mip_detail::perturb_solution` and `pump::perturb`; sharing the
// constant keeps the two perturbation paths in lock-step (R1-4 / R3-11
// round-5 review).  ±64 around the current value gives the perturbation
// enough room to actually move the variable without overflowing the
// `static_cast<int64_t>(hi - lo)` that drives `uniform_int_distribution`.
constexpr double kInfBoundShiftWindow = 64.0;

// Threshold for treating a finite-but-huge `hi - lo` range as
// effectively unbounded.  `int64_t::max()` is ~9.2e18; a user-supplied
// model with bounds at `±1e18` would produce a `static_cast<int64_t>`
// at or beyond that limit, which is UB even though `std::isfinite`
// returns true (R1-3 round-5 review).  Comparing the double range
// against this threshold catches the case before the cast.
constexpr double kSafeInt64DoubleRange = 1e18;

// Deterministic per-worker RNG seeding.
//
// `kBaseSeedOffset` keeps the final seed non-zero when the user leaves
// `random_seed` at its default of 0 (xoshiro256++'s SplitMix64 seeding
// accepts zero, but a non-zero seed makes the first sampled bits less
// uniform-looking in small RNG probes).
// `kSeedStride` is a large prime that spaces adjacent workers' seeds far
// apart in the SplitMix64 expansion so their draws don't immediately
// correlate.
constexpr uint32_t kBaseSeedOffset = 42;
constexpr uint32_t kSeedStride = 997;

// Base seed for heuristic workers: direct propagation of the user-facing
// `random_seed` option, plus the constant offset above.  Every heuristic
// should derive its per-worker seeds from this so that changing
// `random_seed` observably changes heuristic behaviour.
inline uint32_t heuristic_base_seed(HighsInt random_seed) {
    return static_cast<uint32_t>(random_seed) + kBaseSeedOffset;
}

// Effort budget scaled by an effort fraction.  `nnz << 12` is the
// reference base budget at the anchor effort 0.05 (upstream's
// mip_heuristic_effort default); the formula scales linearly in `effort`.
// Two kinds of call site, on two separate budgets:
//  - the presolve chain (`run_sequential` in mode_dispatch.cpp) passes each
//    heuristic's own `mip_heuristic_<name>_effort`, which sizes a whole
//    dispatch — except FJ's, which sizes one worker's allowance (#110);
//  - fpr_lp::run passes `mip_heuristic_effort` (vanilla default 0.05 →
//    exactly the base budget) as its per-call cap on the shared RENS/RINS
//    LP-iteration headroom.
inline size_t heuristic_effort_budget(size_t nnz, double effort) {
    if (effort <= 0.0) {
        return 0;
    }
    constexpr int kBaseShift = 12;
    constexpr double kEffortAnchor = 0.05;
    double scale = effort / kEffortAnchor;
    return static_cast<size_t>(static_cast<double>(nnz << kBaseShift) * scale);
}

// Absolute, instance-scaled stall threshold (issue #111).
//
// A stall threshold answers "how much improvement-free search is enough
// before this is going nowhere?".  Three of the four presolve heuristics
// used to answer it with a fraction of their own effort budget
// (`total >> 2`), which cannot bound over-budgeting: doubling the budget
// doubled the tolerance, so the gate never fired relatively sooner and
// charged effort tracked the option one-for-one across a 20x sweep.  It
// restated the budget instead of measuring the search.
//
// FeasibilityJump was the exception and is the model: `nnz << 8` step
// units, an absolute quantity that scales with the *instance* and not
// with the allowance.  `per_nnz` is that multiplier — effort units per
// constraint-matrix nonzero, i.e. roughly "this many full sweeps of the
// matrix without an improvement".  Each heuristic names its own, since
// their effort counters are in different units (FJ step units;
// FPR/LocalMIP coefficient accesses; Scylla PDLP iters x nnz).
//
// Clamped to `budget` because a threshold above the allowance can never
// fire, and a heuristic that cannot reach its own gate should report the
// budget as its ceiling rather than a number it will never approach.
// `budget == 0` means "no ceiling known"; the floor of 1 keeps a
// degenerate `nnz == 0` model from producing a threshold that trips
// before any work happens.
inline size_t stall_threshold(size_t nnz, size_t per_nnz, size_t budget) {
    const size_t threshold = std::max<size_t>(nnz * per_nnz, 1);
    return budget == 0 ? threshold : std::min(threshold, budget);
}
