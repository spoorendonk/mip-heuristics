#include "fpr_val_select.h"

#include "heuristic_common.h"

#include <algorithm>
#include <cmath>
#include <random>

// ===================================================================
// Value selection
// ===================================================================

namespace {

double val_up(double ub) {
    return ub;
}

double val_random(double lb, double ub, bool is_int, std::mt19937& rng) {
    double v = std::uniform_real_distribution<double>(lb, ub)(rng);
    if (is_int) {
        v = std::round(v);
    }
    return std::max(lb, std::min(ub, v));
}

double val_goodobj(double lb, double ub, bool minimize, double cost) {
    if (std::abs(cost) < 1e-15) {
        return lb;  // paper Section 4.2: "always pick the lower bound"
    }
    if (minimize) {
        return (cost > 0) ? lb : ub;
    }
    return (cost > 0) ? ub : lb;
}

double val_badobj(double lb, double ub, bool minimize, double cost) {
    if (std::abs(cost) < 1e-15) {
        return lb;  // paper Section 4.2: "always pick the lower bound"
    }
    // Opposite of goodobj
    if (minimize) {
        return (cost > 0) ? ub : lb;
    }
    return (cost > 0) ? lb : ub;
}

// LP-based probabilistic rounding (paper Section 4.2):
// v_j = ceil(x^LP_j) with probability f_j = frac(x^LP_j), else floor(x^LP_j)
double val_lp_based(double lb, double ub, bool is_int, double lp_val, std::mt19937& rng) {
    double clamped = std::max(lb, std::min(ub, lp_val));
    if (!is_int) {
        return clamped;
    }

    double f = clamped - std::floor(clamped);
    double v;
    if (f < 1e-10) {
        v = std::round(clamped);
    } else {
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        v = (u < f) ? std::ceil(clamped) : std::floor(clamped);
    }
    return std::max(lb, std::min(ub, v));
}

// Dynamic locks (paper Section 4.2, "loosedyn"):
// Use precomputed min/max activities to count how many constraints would become
// infeasible if variable goes up vs down. Pick direction with fewer dynamic
// locks.
double val_loosedyn(HighsInt j, double lb, double ub, bool /* is_int */, bool minimize, double cost,
                    const double* row_lo, const double* row_hi, const double* min_act,
                    const double* max_act, const CscMatrix& csc) {
    // Count dynamic up-locks and down-locks for variable j
    HighsInt up_locks = 0;
    HighsInt down_locks = 0;
    const HighsInt total_rows = csc.col_start[j + 1] - csc.col_start[j];

    for (HighsInt r = 0; r < total_rows; ++r) {
        HighsInt p = csc.col_start[j] + r;
        HighsInt i = csc.col_row[p];
        double a = csc.col_val[p];

        // Check if constraint is infeasible or redundant
        bool has_upper = row_hi[i] < kHighsInf;
        bool has_lower = row_lo[i] > -kHighsInf;
        if (has_upper && min_act[i] > row_hi[i]) {
            // Upper bound infeasible — skip
            continue;
        }
        if (has_lower && max_act[i] < row_lo[i]) {
            // Lower bound infeasible — skip
            continue;
        }
        // Paper Section 4.2: "constraints that have already become redundant
        // do not contribute to the lock counters."
        if ((!has_lower || min_act[i] >= row_lo[i]) && (!has_upper || max_act[i] <= row_hi[i])) {
            continue;
        }
        // Active constraint: count locks
        if (has_upper && a > 0) {
            ++up_locks;
        }
        if (has_lower && a < 0) {
            ++up_locks;
        }
        if (has_upper && a < 0) {
            ++down_locks;
        }
        if (has_lower && a > 0) {
            ++down_locks;
        }

        // Early exit: if one direction already wins by more than remaining rows
        HighsInt remaining = total_rows - r - 1;
        if (up_locks > down_locks + remaining) {
            return lb;
        }
        if (down_locks > up_locks + remaining) {
            return ub;
        }
    }

    // Pick direction with fewer locks
    if (up_locks < down_locks) {
        return ub;
    } else if (down_locks < up_locks) {
        return lb;
    }
    // Tie: fall back to objective direction
    return val_goodobj(lb, ub, minimize, cost);
}

}  // namespace

double choose_value(HighsInt j, double lb, double ub, bool is_int, bool minimize, double cost,
                    ValStrategy strategy, std::mt19937& rng, const double* lp_ref,
                    const double* row_lo, const double* row_hi, const double* min_act,
                    const double* max_act, const CscMatrix* csc) {
    double v;
    switch (strategy) {
        case ValStrategy::kUp:
            v = val_up(ub);
            break;
        case ValStrategy::kRandom:
            v = val_random(lb, ub, is_int, rng);
            break;
        case ValStrategy::kGoodobj:
            v = val_goodobj(lb, ub, minimize, cost);
            break;
        case ValStrategy::kBadobj:
            v = val_badobj(lb, ub, minimize, cost);
            break;
        case ValStrategy::kLoosedyn:
            if (min_act && max_act && row_lo && row_hi && csc) {
                v = val_loosedyn(j, lb, ub, is_int, minimize, cost, row_lo, row_hi, min_act,
                                 max_act, *csc);
            } else {
                v = val_goodobj(lb, ub, minimize, cost);
            }
            break;
        case ValStrategy::kZerocore:
        case ValStrategy::kZerolp:
        case ValStrategy::kCore:
        case ValStrategy::kLp:
            if (lp_ref) {
                v = val_lp_based(lb, ub, is_int, lp_ref[j], rng);
            } else {
                v = val_goodobj(lb, ub, minimize, cost);
            }
            break;
        default:
            v = val_goodobj(lb, ub, minimize, cost);
            break;
    }
    if (is_int) {
        v = std::round(v);
    }
    return std::max(lb, std::min(ub, v));
}
