#pragma once

#include "deadline.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <vector>

class PropEngine;
struct FprScratch;

// SyncChanges(E, R) (paper Fig. 5 line 13, Sect. 5.1): transfer domain
// deductions from the secondary engine R into the primary engine E.
// Exposed here for direct unit testing (issue #125) -- see the doc comment
// on its definition in repair_search.cpp for the full case analysis. Not
// part of the public API used by `fpr_attempt_finish`, which only calls
// `repair_search` below; every production caller reaches this only through
// that function.
bool sync_changes(PropEngine& E, const PropEngine& R);

// Paper Fig. 5: RepairSearch with secondary propagation engine R.
// E: main propagation engine (has partial assignment from Phase 2).
// solution/lhs_cache: current complete assignment (may violate constraints).
// col_lb/col_ub: global column bounds (for initializing R).
// row_lo/row_hi: row bounds.
// `scratch` supplies the reusable buffers (violated set, undo stacks, DFS
// stack, best-state snapshots, nested walksat_select_move scratch).  All
// are cleared/resized at entry so prior contents are discarded — capacity
// persists across calls.
// `deadline` stops the node loop on the solve's wall clock, alongside
// `repair_iterations` and `max_effort` (issue #117): one node is two
// propagation fixpoints, so on a large model the effort gate alone lets
// this run for seconds past a time limit.  A default-constructed
// `Deadline` never expires, which is what a caller with no time limit
// gets from `make_deadline`.
// Returns true if a feasible solution was found (solution modified in-place).
bool repair_search(PropEngine& E, std::vector<double>& solution, std::vector<double>& lhs_cache,
                   const double* col_lb, const double* col_ub, const double* row_lo,
                   const double* row_hi, HighsInt repair_iterations, double repair_noise,
                   bool repair_track_best, size_t max_effort, Rng& rng, size_t& effort_out,
                   FprScratch& scratch, const Deadline& deadline);
