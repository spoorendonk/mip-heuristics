#pragma once

#include "heuristic_common.h"
#include "util/HighsInt.h"

// ---------------------------------------------------------------------------
// Value selection strategies (paper Table 2, Section 4.2)
// ---------------------------------------------------------------------------

enum class ValStrategy {
    kUp,        // always upper bound
    kRandom,    // random between lb and ub
    kGoodobj,   // fix toward objective
    kBadobj,    // fix against objective
    kLoosedyn,  // dynamic locks based on current activities
    kZerocore,  // zero-obj analytic center, fractional rounding
    kZerolp,    // zero-obj LP vertex, fractional rounding
    kCore,      // full-obj analytic center, fractional rounding
    kLp,        // full-obj LP solution, fractional rounding
};

// Choose a fixing value for variable j given the current domain [lb, ub].
// For LP-based strategies, `lp_ref[j]` provides the reference LP value.
// For loosedyn, precomputed row activities and bound arrays are needed.
double choose_value(HighsInt j, double lb, double ub, bool is_int, bool minimize, double cost,
                    ValStrategy strategy, Rng& rng, const double* lp_ref,
                    // loosedyn support: nullable pointers
                    const double* row_lo, const double* row_hi, const double* min_act,
                    const double* max_act, const CscMatrix* csc);
