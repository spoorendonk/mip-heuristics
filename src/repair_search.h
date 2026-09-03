#pragma once

#include "deadline.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <cstddef>
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

// Default for `repair_search`'s `progress_threshold` below: how many
// consecutive DFS nodes may pass without improving the best total
// violation before the search abandons the current subtree and jumps to
// the lowest-violation open node (paper Fig. 5 lines 18-19, Sect. 5.1:
// "if we detect that we are not making enough progress in the current
// subtree, we backtrack directly to the most promising open node").
//
// A *parameter* rather than a file-scope constant since issue #130,
// because until #130 it could not be demonstrated to do anything: a
// second, ungated `BacktrackBestOpen` at the foot of every loop
// iteration left the search sitting on the best open node at every step,
// so the gate never had a subtree to abandon.  With that call gone this
// is the only steering the node loop has, and the two ends of its range
// are two different searches -- which is what
// tests/test_repair_search.cpp pins.
inline constexpr HighsInt kRepairProgressThreshold = 10;

// Optional instrumentation for `repair_search` (issue #130).  Nothing in
// `src/` reads it -- the production call site in `fpr_core.cpp` passes
// nullptr -- and it exists because the node loop is otherwise opaque from
// outside: `repair_search` returns one bool and an effort figure, neither
// of which can say whether the stall gate fired.  Counters are *added
// to*, never reset, so a caller reusing one struct across attempts gets a
// total.
struct RepairSearchStats {
    // Times the stall gate fired and jumped to the best open node.
    // Exactly zero when `progress_threshold` is high enough never to be
    // reached, which is half of what the #130 test asserts.
    size_t best_open_jumps = 0;
    // DFS nodes popped from Q.  A run that stops short of
    // `repair_iterations` stopped on feasibility, an empty Q, the effort
    // budget or the clock.
    size_t nodes_visited = 0;
};

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
// `progress_threshold` is the stall gate above; production passes
// `kRepairProgressThreshold`.  `stats`, when non-null, accumulates the
// counters in `RepairSearchStats`.
bool repair_search(PropEngine& E, std::vector<double>& solution, std::vector<double>& lhs_cache,
                   const double* col_lb, const double* col_ub, const double* row_lo,
                   const double* row_hi, HighsInt repair_iterations, HighsInt progress_threshold,
                   double repair_noise, bool repair_track_best, size_t max_effort, Rng& rng,
                   size_t& effort_out, FprScratch& scratch, const Deadline& deadline,
                   RepairSearchStats* stats = nullptr);
