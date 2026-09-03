#pragma once

#include "deadline.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <utility>
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
//
// It is the *default* of `repair_search`'s trailing parameter rather
// than an argument the production call site spells out, so the one
// production value lives in exactly one place and no call site can
// disagree with it.  `stats` below carries no default for the opposite
// reason (#118/#119's precedent): it is a per-caller choice, not a
// tuning constant.  The value itself is pinned by the same test, since
// the defaulted call and a `kRepairProgressThreshold` call agree by
// construction whatever it is.
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
    // Nodes whose *bound* branch -- the non-binary half of
    // `MoveToDisjunction` -- actually moved the incumbent point (issue
    // #131).  Before #131 this was zero on every model: the disjunction
    // was built from the node's current domain rather than from the
    // shifted interval the repair move implies, so both branches
    // re-imposed a bound R already had and `sync_changes` had nothing to
    // transfer.  Binary nodes are not counted here -- they take the
    // `fix` branch, which applies unconditionally.
    size_t bound_branch_moves = 0;
};

// One side of the repair disjunction (paper Sect. 5.1's
// `MoveToDisjunction`), in the form `RepairSearchNode` carries and
// `apply_branch_to_r` replays: `is_fix` selects `PropEngine::fix(val)`,
// otherwise `is_lb` selects `tighten_lb(val)` over `tighten_ub(val)`.
struct RepairBranch {
    HighsInt var;
    double val;
    bool is_fix;
    bool is_lb;
};

// MoveToDisjunction(move, E, R) (paper Sect. 5.1, last paragraph): turn
// the repair move `var: cur_val -> move_val` into the two-branch
// disjunction the repair tree searches over.  Exposed here for direct
// unit testing for the same reason `sync_changes` above is (issue #125):
// it is the paper's own named function, its arithmetic is what issue
// #131 was about, and the outcome of a whole `repair_search` run cannot
// separate its endpoints -- the point ends up clamped to whatever bound
// propagation implies, which is usually the row's, not the branch's.
// Every production caller reaches this through `repair_search` below.
std::pair<RepairBranch, RepairBranch> move_to_disjunction(const PropEngine& E, const PropEngine& R,
                                                          HighsInt var, double cur_val,
                                                          double move_val);

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
// `stats`, when non-null, accumulates the counters in
// `RepairSearchStats`; production passes nullptr.  `progress_threshold`
// is the stall gate above and is defaulted, so production names no value
// -- see `kRepairProgressThreshold`.
bool repair_search(PropEngine& E, std::vector<double>& solution, std::vector<double>& lhs_cache,
                   const double* col_lb, const double* col_ub, const double* row_lo,
                   const double* row_hi, HighsInt repair_iterations, double repair_noise,
                   bool repair_track_best, size_t max_effort, Rng& rng, size_t& effort_out,
                   FprScratch& scratch, const Deadline& deadline, RepairSearchStats* stats,
                   HighsInt progress_threshold = kRepairProgressThreshold);
