#pragma once

#include "lp_data/HConst.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// Variable state used during fix-and-propagate.
struct VarState {
    double lb, ub, val;
    bool fixed;
};

// `solution` is meaningful whenever it is non-empty, and `found_feasible`
// says only whether it satisfies every row -- the two are independent
// since issue #155.  `objective` is meaningful *only* when
// `found_feasible`; an infeasible point leaves it at infinity, because a
// cost evaluated on a violated assignment is not this heuristic's answer
// to anything and no caller may treat it as one.
//
// The infeasible-but-populated shape exists for Mexi et al.'s Algorithm
// 1.1 line 12 (`x_hat = fix-and-propagate(x_bar)`): the feasibility pump
// is *defined* by being pulled toward the rounded point whether or not it
// is feasible, so `fpr_attempt` has to be able to hand one back.  See
// `fpr_attempt_finish` in fpr_core.cpp for what the point is on each of
// its failure paths.
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

    // A complete integer assignment that violates at least one row.
    // `found_feasible` stays false, so every offer site — all of which
    // gate on it — is unaffected; what changes is that a caller which
    // wants the *direction* the rounding pointed in can now read it.
    // `objective` is left at its default and is *not* the cost of `point`:
    // a cost on an assignment that violates rows is not an answer, and
    // leaving it there means no caller can mistake it for one.  Every
    // *decision* site reads it only under `found_feasible`.  One place
    // reads it regardless — `fpr_lp`'s `kVerbose` trace line, which
    // printed `inf` for a failed attempt before this existed too — so the
    // rule is "no caller acts on it otherwise", not "no caller reads it".
    static HeuristicResult infeasible_point(std::vector<double> point, size_t e) {
        HeuristicResult r;
        r.solution = std::move(point);
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
//   feastol                     (~1e-6) — from the solver; the
//                                         feasibility tolerance for
//                                         row/bound checks.
//   kScoreTol, kViolDeltaTol    (5e-7)  — local_mip's own
//                                         score/violation-magnitude
//                                         comparison epsilons
//                                         (`local_mip_caches.h`), not
//                                         feasibility tolerances.
//   1e-15                               — numerical zero (avoids
//                                         division/move on zero-delta)

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

// The one base both budget parameters are multipliers of: `nnz << 10`,
// vanilla HiGHS's hardcoded single-thread FeasibilityJump limit
// (`HighsFeasibilityJump.cpp`).  Chosen because it is the only figure in
// this arithmetic that upstream itself picked, which makes `effort = 1.0`
// mean "one vanilla FJ budget" rather than a number that needs decoding.
//
// It replaces `nnz << 12` scaled by `effort / 0.05` (#116): the 4096
// existed only so FJ's default came out at `nnz << 10` anyway, and the
// 0.05 was upstream's own `mip_heuristic_effort` default used as an
// anchor.  Two historical constants multiplied to 81,920, which is what a
// reader had to know to compare an effort option against a patience
// threshold.  Now they are the same unit and `patience < effort` is
// legible.
inline constexpr int kBudgetBaseShift = 10;

// Effort budget: `effort` multiples of the base above.
// Two kinds of call site, on two separate budgets:
//  - the presolve chain (`run_sequential` in mode_dispatch.cpp) passes each
//    heuristic's own `mip_heuristic_<name>_effort`, which sizes a whole
//    dispatch — except FJ's, which sizes one worker's allowance (#110);
//  - fpr_lp::run caps itself against upstream's own `mip_heuristic_effort`,
//    which is on a different scale entirely and goes through
//    `vanilla_effort_budget` below.
//
// The product saturates rather than converting out of range.  The option's
// upper bound is `1e6` since #113 — a budget that cannot bind, so that a
// calibration probe measures the heuristic and not the setting derived from
// it — and `double -> size_t` is undefined when the value does not fit, so
// the guard is the same one `saturating_mul` exists for one level down.
inline size_t heuristic_effort_budget(size_t nnz, double effort);

// Upstream's `mip_heuristic_effort` converted to the same budget it always
// produced.  It is *not* one of our four options: it is HiGHS's own B&B
// heuristic knob, whose default is 0.05, and `fpr_lp` caps its per-call
// slice against it.  When our options were multipliers of `nnz << 12`
// anchored at 0.05, that default landed exactly on the base budget; now
// that they are multiples of `nnz << 10`, the same value has to be scaled
// by `(1 << 2) / 0.05` to keep meaning what it meant.  Spelled once, here,
// rather than left as an 80 at the call site (#116).
inline size_t vanilla_effort_budget(size_t nnz, double mip_heuristic_effort) {
    constexpr double kVanillaAnchor = 0.05;
    constexpr double kBaseRatio = 4.0;  // (1 << 12) / (1 << 10)
    return heuristic_effort_budget(nnz, mip_heuristic_effort * kBaseRatio / kVanillaAnchor);
}

inline size_t heuristic_effort_budget(size_t nnz, double effort) {
    if (effort <= 0.0) {
        return 0;
    }
    double budget = static_cast<double>(nnz << kBudgetBaseShift) * effort;
    if (!(budget < static_cast<double>(SIZE_MAX))) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(budget);
}

// `a * b`, saturating at SIZE_MAX instead of wrapping.
//
// Every factor these budgets are built from is now user-supplied: the
// patience multipliers are options with the same wide upper bound as the
// effort ones (#106), and `nnz` is whatever model was loaded, so the
// product overflows on a large instance at the top of the range.  A
// wrapped product is the worst possible failure here — it produces a
// *small* threshold, so the gate fires almost immediately and the
// heuristic silently does nothing, which reads as "this parameter value
// is terrible" to whatever is searching the space.  Saturating gives the
// monotone answer instead: a bigger multiplier never means a tighter
// gate.
[[nodiscard]] constexpr size_t saturating_mul(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return a > SIZE_MAX / b ? SIZE_MAX : a * b;
}

// The largest fraction of a heuristic's own ceiling its patience may be
// (issue #116).
//
// A patience at or above the ceiling fires exactly at exhaustion, which
// is indistinguishable from having no gate at all — and silently so,
// since nothing reports which of the two bounded the dispatch.  That is
// not hypothetical: the p95 inter-improvement gap #113 measured exceeds
// the ceiling on three of the four heuristics, FJ's by a factor of 4,400,
// so an honest value read off improvement counts would leave FJ running
// to its budget on every instance while looking like a tuned parameter.
//
// A quarter is the shape FeasibilityJump has always shipped (`nnz << 8`
// against a `nnz << 10` budget), and where all four shipped defaults
// already sit — 21-28% of their ceilings before this clamp existed, and
// exactly 25% since #113's vector — so applying it moves no default and
// bounds every future one.
inline constexpr size_t kPatienceCeilingDivisor = 4;

// Absolute, instance-scaled patience: the improvement-free effort a
// heuristic tolerates before giving up (issues #111, #116).
//
// It answers "how much improvement-free search is enough before this is
// going nowhere?".  Three of the four presolve heuristics used to answer
// it with a fraction of their own effort budget (`total >> 2`), which
// cannot bound over-budgeting: doubling the budget doubled the tolerance,
// so the gate never fired relatively sooner and charged effort tracked
// the option one-for-one across a 20x sweep.  It restated the budget
// instead of measuring the search.
//
// FeasibilityJump was the exception and is the model: `nnz << 8` step
// units, an absolute quantity that scales with the *instance* and not
// with the allowance.  `per_base` is that multiplier, in the same unit as
// the heuristic's effort option since #116 — multiples of
// `nnz << kBudgetBaseShift` — which is what makes `patience < effort`
// legible without a conversion.  Each heuristic names its own, as a
// `mip_heuristic_<name>_patience` option since #106; the four are read in
// `kChain` (mode_dispatch.cpp) and are **not** comparable across
// heuristics, because only each heuristic knows what its own effort
// counter counts (FJ step units; FPR/LocalMIP coefficient accesses;
// Scylla PDLP iters x nnz).
//
// Clamped to `budget / kPatienceCeilingDivisor` rather than to `budget`,
// so a gate that exists can always fire strictly before exhaustion; see
// that constant.  `budget == 0` means "no ceiling known"; the floor of 1
// keeps a degenerate `nnz == 0` model — or a ceiling smaller than the
// divisor — from producing a threshold that trips before any work
// happens.
//
// **"Strictly before exhaustion" is not unconditional**, because that
// floor deliberately outranks it.  `stale()` is `> stale_budget` while
// `exhausted()` is `>= total_budget`, so a threshold `S` fires at effort
// `S + 1` and precedes exhaustion only while `S <= budget - 2`.  At
// `budget <= 2` the floor pins `S` at 1 and the gate coincides with
// exhaustion (`budget == 2`) or never fires at all (`budget == 1`) — from
// `budget == 3` up it precedes exhaustion again.  That is the right
// trade: a gate that fires with the budget is a far better failure than
// one that trips before any work happens, which is what dropping the
// floor would give.  It is also unreachable in practice — `budget` is
// `effort x (nnz << 10)` (times the worker count for FJ), so two effort
// units needs roughly a one-nonzero model at the bottom of the option's
// range, and `budget == 0` is already excluded upstream by `make_budget`
// as "this heuristic does not run".
//
// `per_base == 0` means **no patience gate at all** (issue #106), not
// "give up immediately".  Since #106 the multiplier is an option, and a
// search of the patience axis needs a point where the gate provably never
// fires — otherwise "how much does this gate cost?" has no zero to
// measure against.  The unbounded value is returned *before* the clamp,
// which is what keeps "no gate" distinguishable from a value the clamp
// merely pushed down onto the ceiling.
[[nodiscard]] inline size_t patience_threshold(size_t nnz, double per_base, size_t budget) {
    if (per_base <= 0.0) {
        return SIZE_MAX;
    }
    const size_t threshold = std::max<size_t>(heuristic_effort_budget(nnz, per_base), 1);
    if (budget == 0) {
        return threshold;
    }
    const size_t ceiling = std::max<size_t>(budget / kPatienceCeilingDivisor, 1);
    return std::min(threshold, ceiling);
}
