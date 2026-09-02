#pragma once

#include "deadline.h"
#include "rng.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <vector>

class PropEngine;

// Scratch buffers for `repair_walk`.  One per worker; not thread-safe.
// Everything is cleared (never freed) at entry so capacity persists across
// the many calls one DFS makes -- `repair_walk` runs at *every* infeasible
// node (paper Fig. 1 line 8), which is the hottest allocation site the
// in-tree repair could have had.
//
// Deliberately its own type rather than a second consumer of
// `WalkSatScratch`: that one is shared between `walksat_repair` and
// `repair_search`, which are alternative *Phase 3* paths at one call site
// and provably never overlap.  `repair_walk` runs inside Phase 2, so it
// would be the first sharer whose non-overlap rests on an argument about
// two different phases rather than on one `if`/`else`.
struct RepairWalkScratch {
    // A candidate repair move: shift column `var`'s whole current domain
    // by `shift`, costing `damage` (summed violation *increase* over the
    // other rows the column appears in; WalkSAT ignores the improvements,
    // paper Sect. 5).
    struct Candidate {
        HighsInt var;
        double shift;
        double damage;
    };
    std::vector<Candidate> cand;
    std::vector<HighsInt> best_indices;

    // Rows whose activity range proves them violated, plus the O(1)
    // membership/position index into it (-1 = absent), sized `nrow`.
    std::vector<HighsInt> violated;
    std::vector<HighsInt> violated_pos;

    // The last `kTabuLength` shifted columns (paper Sect. 5: "a short tabu
    // list of the last 3 shifts in order to avoid short cycles").  Kept as
    // a plain vector because it holds three entries.
    std::vector<HighsInt> tabu;
};

// Whether applying a fixing to column `j` left one of the rows `j` appears
// in unsatisfiable -- the infeasibility half of `Apply(fixing, P)`, paper
// Fig. 1 line 4.
//
// A row is unsatisfiable when the activity range its current domain admits
// is disjoint from the row's bounds: *no* completion satisfies it, so the
// node is refuted.  Only `j`'s own rows are scanned, because only their
// activities moved -- the same incremental reading the paper's shared
// activity structures exist for.
//
// This is what makes `dive` a repair strategy rather than a blind dive.
// With propagation disabled nothing else can report an infeasible node:
// `PropEngine::fix` rejects a value only when it falls outside the
// column's *own* current domain and never looks at a row, so before this
// existed `infeas` was permanently false in `dive` and the repair below
// was unreachable there -- while the paper calls `dive` "an incremental
// repair strategy that constructs a complete solution in a single big
// dive".  It also covers a case propagation misses in every mode:
// `PropEngine::propagate` skips a row with no unfixed columns, so a row
// that a fixing both completes *and* violates is invisible to it.
//
// `E` must have had `init_activities()` called; returns false otherwise.
// `effort` is incremented by the coefficient accesses consumed.
bool any_violated_row_in_column(const PropEngine& E, HighsInt j, size_t& effort);

// RepairWalk on a *partial* assignment (paper Fig. 1 line 8, Sect. 5).
//
// This is the repair the paper's framework is built around, and the reason
// Sect. 5 extends WalkSAT's violation and shift definitions from values to
// domains: at an infeasible node of the fix-and-propagate DFS most columns
// are not fixed, so "the partial assignment at the current node is encoded
// by the current domain".  Concretely:
//
//   - A row's violation is the distance between its activity range
//     `[m_i, M_i]` -- the same min/max activities constraint propagation
//     maintains, which is why `E` must have had `init_activities()` called
//     -- and its bounds.  A row is violated when *no* completion of the
//     current domain can satisfy it.
//   - A repair move is a *shift*: column `j`'s whole current domain
//     `[lb, ub]` translates by `s`, keeping its width.  The paper is
//     explicit that domain *enlargement* is not a legal move, "as it would
//     lead to trivial repair actions where fixings are just undone", and
//     that shifts must therefore be available on non-fixed columns too, so
//     that propagation's tightenings can be moved rather than dropped.
//     The shift is clipped so the translated interval still lies inside
//     the column's *structural* bounds ("clip it so that the variable is
//     still within its global bounds") -- see `PropEngine::shift_domain`.
//     Note the two rules are independent: the clip keeps the interval
//     legal, width preservation is what forbids enlargement.
//
// Every change goes through `E`, so the engine's own undo stacks are what
// makes both the paper's soft restart and the DFS's backtrack work: a
// caller that backtracks past this node undoes the repair with it, and
// nothing outside `E` has to be snapshotted.
//
// `repair_walk` deliberately does **not** propagate.  Paper Sect. 5.1: "it
// is not possible to do constraint propagation from an infeasible state" --
// that limitation is exactly what RepairSearch (Fig. 5) exists to work
// around, on a separate engine.
//
// Returns true when no row is violated any more, i.e. the node was
// repaired.  `effort_out` is the coefficient accesses consumed; the caller
// charges them (`PropEngine::add_effort`) so they land in the same counter
// the DFS budget gate reads.  `max_steps` is the paper's per-call
// iteration limit; the effort valve is *internal*
// (`kRepairWalkBudgetPerNnz`, see its definition) rather than a caller
// argument, because every effort number a caller here could derive is
// either the per-call DFS slice -- already spent by the time propagation
// has refuted the node -- or the attempt budget, which `E.effort()`
// outgrows on a long attempt, silently turning the repair into a no-op
// exactly on the hard instances it exists for.  The deadline is polled
// once per step, and an expiry is **not** signalled outward: like
// `sync_changes`, the only thing this `bool` can mean to its caller is
// "the node is still infeasible", and reading a truncation as a verdict is
// the bug #127/#151 exist to prevent.
bool repair_walk(PropEngine& E, HighsInt max_steps, double noise, Rng& rng, size_t& effort_out,
                 RepairWalkScratch& scratch, const Deadline& deadline);
