#pragma once

#include "deadline.h"

#include <cstdint>
#include <vector>

class HighsMipSolver;

// ---------------------------------------------------------------------------
// LP reference solutions
// ---------------------------------------------------------------------------
//
// Both solves take the solve's wall-clock `deadline` and hand the
// sub-solver what is left of it (capped at 30 s), returning empty without
// constructing a `Highs` at all once it has passed: `Deadline::remaining()`
// is 0.0 there and HiGHS reads `time_limit == 0.0` as *no* limit, so
// passing that on would uncap the very solve this bounds.
//
// An empty return therefore means "no reference available" — from an
// expired clock or from a failed solve alike, and the caller cannot tell
// which by looking at it.  That is why `fpr_lp`'s setup polls the deadline
// around these calls rather than testing the vector (issue #118): before
// it did, an expiry here was silently absorbed by the `lp_ptr` fallback and
// the setup went on to build all ten variable orders.

// Solve the LP relaxation without objective using barrier (no crossover)
// to obtain the analytic center. Returns col_value vector.  Adds the LP
// iterations spent (simplex + IPM + crossover) to `lp_iterations` so the
// caller can charge them against the shared B&B heuristic budget.
std::vector<double> compute_analytic_center(const HighsMipSolver& mipsolver, bool use_objective,
                                            const Deadline& deadline, int64_t& lp_iterations);

// Solve the LP relaxation without objective using simplex to obtain a vertex.
// Adds the LP iterations spent to `lp_iterations` as above.
std::vector<double> compute_zero_obj_vertex(const HighsMipSolver& mipsolver,
                                            const Deadline& deadline, int64_t& lp_iterations);
