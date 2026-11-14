#pragma once

#include "heuristic_common.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace pump {

// Algorithm 1.1 parameters (Mexi et al. 2023, §2).
inline constexpr double kAlpha = 0.9;
inline constexpr double kEpsilonInit = 0.01;
inline constexpr double kBeta = 0.98;
inline constexpr double kEpsilonFloor = 1e-8;
inline constexpr int kCycleWindow = 3;
inline constexpr double kPerturbFraction = 0.2;
inline constexpr double kCycleTol = 0.5;  // integer values differ by >= 1.0
inline constexpr int kMaxPdlpStalls = 3;

// Build LP relaxation from the presolved MIP model (strip integrality).
inline HighsLp build_lp_relaxation(const HighsLp& model, const HighsMipSolverData& mipdata) {
    HighsLp lp;
    lp.num_col_ = model.num_col_;
    lp.num_row_ = model.num_row_;
    lp.col_cost_ = model.col_cost_;
    lp.col_lower_ = model.col_lower_;
    lp.col_upper_ = model.col_upper_;
    lp.row_lower_ = model.row_lower_;
    lp.row_upper_ = model.row_upper_;
    lp.sense_ = model.sense_;
    lp.offset_ = model.offset_;
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.num_col_ = model.num_col_;
    lp.a_matrix_.num_row_ = model.num_row_;
    lp.a_matrix_.start_ = mipdata.ARstart_;
    lp.a_matrix_.index_ = mipdata.ARindex_;
    lp.a_matrix_.value_ = mipdata.ARvalue_;
    return lp;
}

// Compute modified objective (Algorithm 1.1, line 15).
inline void compute_pump_objective(
    const std::vector<double>& orig_cost, const std::vector<double>& x_rounded,
    const std::vector<double>& x_lp, const std::vector<HighsVarType>& integrality,
    const std::vector<double>& col_lb, const std::vector<double>& col_ub, double alpha_K,
    double cost_scale, HighsInt ncol, std::vector<double>& modified_cost) {
    for (HighsInt j = 0; j < ncol; ++j) {
        double scaled_cost = alpha_K * cost_scale * orig_cost[j];
        if (is_integer(integrality, j)) {
            double delta;
            if (col_lb[j] == 0.0 && col_ub[j] == 1.0) {
                delta = 1.0 - (2.0 * x_rounded[j]);
            } else {
                double diff = x_lp[j] - x_rounded[j];
                delta = (diff >= 0.0) ? 1.0 : -1.0;
            }
            modified_cost[j] = scaled_cost + ((1.0 - alpha_K) * delta);
        } else {
            modified_cost[j] = scaled_cost;
        }
    }
}

// Detect cycling: check if x_rounded matches any solution in history.
inline bool detect_cycling(const std::vector<std::vector<double>>& history,
                           const std::vector<double>& x_rounded,
                           const std::vector<HighsVarType>& integrality, HighsInt ncol) {
    for (const auto& prev : history) {
        if (prev.empty()) {
            continue;
        }
        bool match = true;
        for (HighsInt j = 0; j < ncol; ++j) {
            if (!is_integer(integrality, j)) {
                continue;
            }
            if (std::abs(x_rounded[j] - prev[j]) > kCycleTol) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

// Record one fresh-round iterate in the cycle-detection window
// (Algorithm 1.1, line 13).  `completed_rounds` is the number of fresh
// pump rounds already recorded — the caller's iteration counter *before*
// it is incremented for the incoming iterate, since the write precedes
// the increment.  While the buffer is short of `kCycleWindow` entries we
// append, so the iterate of round `r` (1-based) lands at index `r - 1`.
// Once full, the slot to reuse is the oldest entry, which is exactly
// `completed_rounds % kCycleWindow`: index `i` last held the iterate
// written at `completed_rounds - kCycleWindow`.  Writing
// `(completed_rounds - 1) % kCycleWindow` instead clobbers the *newest*
// entry at the first wrap and shortens the detection window for two
// rounds (#126).
inline void record_cycle_entry(std::vector<std::vector<double>>& history, int completed_rounds,
                               const std::vector<double>& x) {
    if (static_cast<int>(history.size()) < kCycleWindow) {
        history.push_back(x);
    } else {
        history[completed_rounds % kCycleWindow] = x;
    }
}

// The integer range a perturbation of a column at value `x` draws from:
// `[lo, lo + irange]`, the column's integer bounds, or a window of
// `kInfBoundShiftWindow` either side of `round(x)` when either bound is
// infinite or the range would overflow `int64_t`.  False when there is
// nothing to draw from: a non-finite `x` (casting NaN to int64_t is UB, and
// `round(x) ± kInfBoundShiftWindow` would carry NaN into the draw) or fewer
// than two integers in range.  Shared by Mexi's `perturb` and the cuOpt
// variant's `cuopt::perturb`.
//
// The finite window: `kHighsInf` (== std::infinity per HiGHS HConst.h) is
// caught by `!std::isfinite`, but adversarial user-supplied bounds at e.g.
// ±1e20 satisfy isfinite yet still overflow int64_t (R1-3 round-5 review).
// `kSafeInt64DoubleRange` catches both cases at once.
inline bool perturb_range(double x, double lb, double ub, double& lo, int64_t& irange) {
    if (!std::isfinite(x)) {
        return false;
    }
    lo = std::ceil(lb);
    double hi = std::floor(ub);
    if (!std::isfinite(lo) || !std::isfinite(hi) || hi - lo > kSafeInt64DoubleRange) {
        const double current = std::round(x);
        lo = current - kInfBoundShiftWindow;
        hi = current + kInfBoundShiftWindow;
    }
    if (hi <= lo) {
        return false;
    }
    irange = static_cast<int64_t>(hi - lo);
    return irange >= 1;
}

// Perturb a rounded solution to break cycling (Algorithm 1.1, line 14).
inline void perturb(std::vector<double>& x, const HighsLp& model, Rng& rng) {
    // `kInfBoundShiftWindow` and `kSafeInt64DoubleRange` are shared
    // with `local_mip_detail::perturb_solution` via heuristic_common.h
    // (R1-4 / R3-11 round-5 review): the two perturbation paths must
    // use the same window so their bound-clamping behaviour stays
    // identical.
    const HighsInt ncol = model.num_col_;
    const auto& integrality = model.integrality_;
    const auto& lb = model.col_lower_;
    const auto& ub = model.col_upper_;

    for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) > kPerturbFraction) {
            continue;
        }
        double lo = 0.0;
        int64_t irange = 0;
        if (!perturb_range(x[j], lb[j], ub[j], lo, irange)) {
            continue;
        }
        const double current = std::round(x[j]);
        int64_t shift = std::uniform_int_distribution<int64_t>(1, irange)(rng);
        x[j] = lo + std::fmod(current - lo + static_cast<double>(shift),
                              static_cast<double>(irange) + 1.0);
        // Final clamp to the original bounds — mirrors the closing
        // step of `local_mip_detail::perturb_solution` so the two
        // patterns stay structurally identical.  Defensive against any
        // fmod / cast roundoff escaping `[lb[j], ub[j]]` (R3-5
        // round-5 review).
        x[j] = std::max(lb[j], std::min(ub[j], x[j]));
    }
}

// cuOpt-style projection (#290, option `mip_heuristic_scylla_cuopt_pump`).
// Technique read from NVIDIA cuOpt's feasibility pump (Apache-2.0; no code
// copied): an exact L1 distance through auxiliary columns, an alpha that
// decays before every projection and resets on an improving offer, and a
// PDLP tolerance read off how integral the last LP point already is.
namespace cuopt {

inline constexpr double kAlphaInit = 0.99;
inline constexpr double kAlphaDecay = 0.9;
// Tolerance schedule: below each integral fraction, the next solve's
// tolerance.  At or above the last threshold the pump uses Scylla's own
// floor `kEpsilonFloor` (cuOpt uses its absolute tolerance there; the
// floor is this pump's tightest solve, a recorded reading).
inline constexpr double kIntegralFracLoose = 0.80;
inline constexpr double kIntegralFracMid = 0.93;
inline constexpr double kIntegralFracTight = 0.97;
inline constexpr double kTolLoose = 0.1;
inline constexpr double kTolMid = 0.01;
inline constexpr double kTolTight = 1e-3;
// The first projection has no LP point to read a fraction from; it counts
// as fraction 0 (cuOpt starts loose too).
inline constexpr double kTolFirst = kTolLoose;

inline double lp_tolerance(double integral_fraction) {
    if (integral_fraction < kIntegralFracLoose) {
        return kTolLoose;
    }
    if (integral_fraction < kIntegralFracMid) {
        return kTolMid;
    }
    if (integral_fraction < kIntegralFracTight) {
        return kTolTight;
    }
    return kEpsilonFloor;
}

// Fraction of the integer columns of `x` that are integral within `feastol`;
// 0 when there are none.
inline double integral_fraction(std::span<const double> x,
                                const std::vector<HighsVarType>& integrality, double feastol) {
    HighsInt num_int = 0;
    HighsInt num_integral = 0;
    for (HighsInt j = 0; std::cmp_less(j, x.size()); ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        ++num_int;
        if (std::abs(x[j] - std::round(x[j])) <= feastol) {
            ++num_integral;
        }
    }
    return num_int == 0 ? 0.0 : static_cast<double>(num_integral) / num_int;
}

// Does integer column `j` get an auxiliary distance column?  Only when its
// rounding can land strictly inside its bounds: a range above 1 or an
// infinite bound.
inline bool needs_aux(double lb, double ub) {
    return !std::isfinite(lb) || !std::isfinite(ub) || ub - lb > 1.0;
}

// The projection's objective over the LP with auxiliary columns
// (`ncol + aux_cols.size()` costs, two row lowers per aux column).
//
// `dist` is the L1 distance to `x_hat` over the integer columns: +1 on a
// column rounded to its lower bound (`x - l`), -1 at its upper bound
// (`u - x`; upper checked first, so a fixed column reads -1), and for an
// interior rounding of an aux-carrying column cost 1 on `d_k` with rows
// `d_k - x_j >= -x_hat_j`, `d_k + x_j >= x_hat_j`.  The constants `-l`, `u`
// only shift the objective, so they are left out of it (the LP's offset
// does not move its argmin).  An aux column whose rounding is at a bound
// gets cost 0 and its rows relaxed (`-inf`).  Continuous columns: 0.
//
// cost = (1-a)/||dist||_2 * dist + a/||c||_2 * c, each term dropped when its
// norm is zero (the scaled term would not be finite).
inline void build_projection(std::span<const double> orig_cost, std::span<const double> x_hat,
                             const std::vector<HighsVarType>& integrality,
                             std::span<const double> lb, std::span<const double> ub,
                             std::span<const HighsInt> aux_cols, double alpha, double feastol,
                             std::vector<double>& cost, std::vector<double>& aux_row_lower) {
    const auto ncol = static_cast<HighsInt>(orig_cost.size());
    const auto naux = static_cast<HighsInt>(aux_cols.size());
    cost.assign(ncol + naux, 0.0);
    aux_row_lower.assign(2 * static_cast<size_t>(naux), -kHighsInf);
    // `cost` holds `dist` until the blend below.
    for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        if (x_hat[j] >= ub[j] - feastol) {
            cost[j] = -1.0;
        } else if (x_hat[j] <= lb[j] + feastol) {
            cost[j] = 1.0;
        }
    }
    for (HighsInt k = 0; k < naux; ++k) {
        const HighsInt j = aux_cols[k];
        if (cost[j] != 0.0 || !std::isfinite(x_hat[j])) {
            continue;
        }
        cost[ncol + k] = 1.0;
        aux_row_lower[2 * static_cast<size_t>(k)] = -x_hat[j];
        aux_row_lower[(2 * static_cast<size_t>(k)) + 1] = x_hat[j];
    }
    double dist_sq = 0.0;
    for (const double d : cost) {
        dist_sq += d * d;
    }
    double c_sq = 0.0;
    for (const double c : orig_cost) {
        c_sq += c * c;
    }
    const double dist_w = dist_sq > 0.0 ? (1.0 - alpha) / std::sqrt(dist_sq) : 0.0;
    const double c_w = c_sq > 0.0 ? alpha / std::sqrt(c_sq) : 0.0;
    for (HighsInt j = 0; j < ncol; ++j) {
        cost[j] = (dist_w * cost[j]) + (c_w * orig_cost[j]);
    }
    for (HighsInt k = 0; k < naux; ++k) {
        cost[ncol + k] *= dist_w;
    }
}

// Warm-start primal for the next projection: the rounding `x_hat`, with
// each aux column at the distance `|x_hat_j - x_bar_j|` to the last LP point
// (0 where that is not finite).
inline void build_warm_primal(std::span<const double> x_hat, std::span<const double> x_bar,
                              std::span<const HighsInt> aux_cols, std::vector<double>& out) {
    out.assign(x_hat.begin(), x_hat.end());
    out.reserve(x_hat.size() + aux_cols.size());
    for (const HighsInt j : aux_cols) {
        const double d = std::abs(x_hat[j] - x_bar[j]);
        out.push_back(std::isfinite(d) ? d : 0.0);
    }
}

// Cycle handling (#295).  cuOpt runs two detectors on each fresh round:
//
// - Assignment cycle: the round's (infeasible) rounding equals one of the
//   last `kAssignmentRing` roundings on every integer column.  Checked only
//   once alpha has nearly stopped moving, `a/kAlphaDecay - a <
//   kAssignmentCheckStep` (before that the objective still changes enough
//   between rounds to pull the LP point off a repeated rounding); the
//   rounding enters the ring every round.
// - Distance cycle: the L1 distance over the integer columns from the new
//   LP point to the rounding its projection was pulled toward has stopped
//   falling: once the last `kDistanceWindow` distances are recorded, a cycle
//   when `avg - cur < kDistanceCycleRatio * avg`, `avg` over those previous
//   distances, not counting `cur` (cuOpt's `check_distance_cycle`).
//
// A cycle restarts the descent: the rounding is perturbed when it is in the
// ring (always after an assignment cycle, which has just put it there), the
// result enters the ring, and the distance history is cleared.
inline constexpr int kAssignmentRing = 30;
inline constexpr double kAssignmentCheckStep = 0.005;
inline constexpr int kDistanceWindow = 70;
inline constexpr double kDistanceCycleRatio = 0.1;
// Each integer column is redrawn with this probability (cuOpt's
// `perturbate`, `change_ratio`).
inline constexpr double kPerturbProbability = 0.1;

inline bool assignment_check_due(double alpha) {
    return (alpha / kAlphaDecay) - alpha < kAssignmentCheckStep;
}

// The ring stores a 64-bit hash of each rounding's integer columns, not the
// rounding: 30 copies of every integer column per worker is too much memory
// on large models, and a false match needs a 2^-64 collision.  cuOpt
// compares the stored vectors.  Values are rounded first (integer columns
// of a rounding are integral; `+ 0.0` folds -0 into 0).
inline uint64_t assignment_hash(std::span<const double> x,
                                const std::vector<HighsVarType>& integrality) {
    uint64_t h = 0;
    for (HighsInt j = 0; std::cmp_less(j, x.size()); ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        // SplitMix64's finaliser (https://prng.di.unimi.it/splitmix64.c)
        // over the running hash xor the value's bits: order-sensitive, and
        // one changed column changes every later step.
        h ^= std::bit_cast<uint64_t>(std::round(x[j]) + 0.0);
        h += 0x9e3779b97f4a7c15ULL;
        h = (h ^ (h >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27U)) * 0x94d049bb133111ebULL;
        h ^= h >> 31U;
    }
    return h;
}

// Fixed-capacity ring, oldest entry overwritten once full.  `ring` is
// reserved to `capacity` by its owner, so a push never allocates.
template <typename T>
inline void ring_push(std::vector<T>& ring, size_t& next, size_t capacity, T value) {
    if (ring.size() < capacity) {
        ring.push_back(value);
    } else {
        ring[next] = value;
    }
    next = (next + 1) % capacity;
}

inline bool in_ring(std::span<const uint64_t> ring, uint64_t h) {
    return std::ranges::find(ring, h) != ring.end();
}

// L1 distance between `a` and `b` over the integer columns.
inline double int_l1_distance(std::span<const double> a, std::span<const double> b,
                              const std::vector<HighsVarType>& integrality) {
    double d = 0.0;
    for (HighsInt j = 0; std::cmp_less(j, a.size()); ++j) {
        if (is_integer(integrality, j)) {
            d += std::abs(a[j] - b[j]);
        }
    }
    return d;
}

// `previous`: the distances recorded so far, `cur` not among them.
inline bool distance_cycle(std::span<const double> previous, double cur) {
    if (std::cmp_less(previous.size(), kDistanceWindow)) {
        return false;
    }
    double sum = 0.0;
    for (const double d : previous) {
        sum += d;
    }
    const double avg = sum / static_cast<double>(previous.size());
    return avg - cur < kDistanceCycleRatio * avg;
}

// cuOpt's perturbation: each integer column with probability
// `kPerturbProbability` to a uniform integer in its bounds, the current
// value included.  Unlike Mexi's `pump::perturb` there is no shift that
// forces a change.  A column with an infinite or huge bound draws from
// `perturb_range`'s finite window around its value (#291's spec), where
// cuOpt sets it to its finite bound.
inline void perturb(std::span<double> x, const std::vector<HighsVarType>& integrality,
                    std::span<const double> lb, std::span<const double> ub, Rng& rng) {
    for (HighsInt j = 0; std::cmp_less(j, x.size()); ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) > kPerturbProbability) {
            continue;
        }
        double lo = 0.0;
        int64_t irange = 0;
        if (!perturb_range(x[j], lb[j], ub[j], lo, irange)) {
            continue;
        }
        x[j] = lo + static_cast<double>(std::uniform_int_distribution<int64_t>(0, irange)(rng));
        x[j] = std::max(lb[j], std::min(ub[j], x[j]));
    }
}

}  // namespace cuopt

}  // namespace pump
