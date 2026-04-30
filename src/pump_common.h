#pragma once

#include "heuristic_common.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsMipSolverData.h"

#include <cmath>
#include <random>
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
inline HighsLp build_lp_relaxation(const HighsLp &model, const HighsMipSolverData &mipdata) {
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
    const std::vector<double> &orig_cost, const std::vector<double> &x_rounded,
    const std::vector<double> &x_lp, const std::vector<HighsVarType> &integrality,
    const std::vector<double> &col_lb, const std::vector<double> &col_ub, double alpha_K,
    double cost_scale, HighsInt ncol, std::vector<double> &modified_cost) {
    for (HighsInt j = 0; j < ncol; ++j) {
        double scaled_cost = alpha_K * cost_scale * orig_cost[j];
        if (is_integer(integrality, j)) {
            double delta;
            if (col_lb[j] == 0.0 && col_ub[j] == 1.0) {
                delta = 1.0 - 2.0 * x_rounded[j];
            } else {
                double diff = x_lp[j] - x_rounded[j];
                delta = (diff >= 0.0) ? 1.0 : -1.0;
            }
            modified_cost[j] = scaled_cost + (1.0 - alpha_K) * delta;
        } else {
            modified_cost[j] = scaled_cost;
        }
    }
}

// Detect cycling: check if x_rounded matches any solution in history.
inline bool detect_cycling(const std::vector<std::vector<double>> &history,
                           const std::vector<double> &x_rounded,
                           const std::vector<HighsVarType> &integrality, HighsInt ncol) {
    for (const auto &prev : history) {
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

// Perturb a rounded solution to break cycling (Algorithm 1.1, line 14).
inline void perturb(std::vector<double> &x, const HighsLp &model, Rng &rng) {
    // Window for clamping the shift range when one or both bounds are
    // non-finite (kHighsInf).  Same UB pattern and same ±64 fix as
    // `local_mip_detail::perturb_solution`: casting
    // `kHighsInf - (-kHighsInf)` (~2e30) to int64_t is undefined
    // behaviour, and a non-finite `current` would propagate NaN through
    // the shift arithmetic.  Tracked as R1-3 round-4 review (and the
    // analogous R2-5 round-3 fix in `src/local_mip_worker.cpp`).  Keep
    // the value at 64.0 for parity with `kInfBoundShiftWindow` there.
    constexpr double kInfBoundShiftWindow = 64.0;
    const HighsInt ncol = model.num_col_;
    const auto &integrality = model.integrality_;
    const auto &lb = model.col_lower_;
    const auto &ub = model.col_upper_;

    for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) > kPerturbFraction) {
            continue;
        }
        // Skip variables whose current value is non-finite (NaN or
        // ±inf): casting NaN to int64_t is UB and `current ±
        // kInfBoundShiftWindow` would propagate NaN through the shift
        // arithmetic below.
        if (!std::isfinite(x[j])) {
            continue;
        }
        double lo = std::ceil(lb[j]);
        double hi = std::floor(ub[j]);
        double current = std::round(x[j]);
        // Clamp `lo`/`hi` to a finite window around the current value
        // when either bound is non-finite.  Without this guard the
        // `static_cast<int64_t>(hi - lo)` below overflows (kHighsInf is
        // ~1e30, far outside int64_t range) and
        // `uniform_int_distribution<int64_t>(1, irange)` would see a
        // garbage upper bound.
        if (!std::isfinite(lo)) {
            lo = current - kInfBoundShiftWindow;
        }
        if (!std::isfinite(hi)) {
            hi = current + kInfBoundShiftWindow;
        }
        if (hi <= lo) {
            continue;
        }
        auto irange = static_cast<int64_t>(hi - lo);
        if (irange < 1) {
            continue;
        }
        int64_t shift = std::uniform_int_distribution<int64_t>(1, irange)(rng);
        x[j] = lo + std::fmod(current - lo + shift, irange + 1.0);
    }
}

}  // namespace pump
