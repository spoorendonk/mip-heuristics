#pragma once

#include "rng.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <vector>

class PropEngine;

struct WalkSatMove {
    HighsInt var = -1;
    double val = 0.0;
};

// Pre-allocated scratch buffers to avoid per-call heap allocations.
//
// Used by both walksat_select_move (cand / best_indices) and walksat_repair
// (violated, violated_pos, sol_undo, lhs_undo).  Callers own the scratch
// (typically per-worker) and pass it in so hot paths never allocate.
// walksat_repair clears the repair-specific vectors at entry (clear() retains
// capacity), so reusing across calls is safe.
struct WalkSatScratch {
    struct Candidate {
        HighsInt var;
        double new_val;
        double damage;
    };
    std::vector<Candidate> cand;
    std::vector<HighsInt> best_indices;

    // Used by walksat_repair.
    struct UndoEntry {
        HighsInt idx;
        double old_val;
    };
    std::vector<HighsInt> violated;
    std::vector<HighsInt> violated_pos;  // size=nrow, -1 = absent
    std::vector<UndoEntry> sol_undo;
    std::vector<UndoEntry> lhs_undo;
};

// Select a WalkSAT repair move for the given violated row.
// Uses PropEngine for read-only matrix data, bounds, integrality, feastol.
// solution/lhs_cache are the current (possibly infeasible) assignment.
// col_lb/col_ub are clamping bounds for candidate shifts.
// Returns {-1, 0.0} if no valid candidate found.
// Increments effort by coefficient accesses consumed.
WalkSatMove walksat_select_move(HighsInt row, const double* solution, const double* lhs_cache,
                                const double* col_lb, const double* col_ub, const PropEngine& data,
                                double noise, Rng& rng, size_t& effort, WalkSatScratch& scratch);

// Run flat WalkSAT repair walk (paper Fig. 4) on solution/lhs_cache.
// Returns true if feasible. Modifies solution/lhs_cache in place.
// total_viol is the current total violation (updated incrementally).
// `scratch` holds reusable buffers — walksat_repair clears them at entry so
// any existing contents are discarded.  Pass the same scratch across calls
// to avoid per-call allocations.
//
// `max_iterations` is the paper's per-call step limit; the effort valve is
// *internal* (`kWalkSatBudgetPerNnz`, see its definition in walksat.cpp)
// rather than a caller argument, for the reason `repair_walk` documents at
// length (issue #156, the same defect one call site later): every effort
// number `fpr_core.cpp` could derive here is either the per-call DFS slice,
// already spent by the time the leaf is reached, or the attempt budget
// (`FprConfig::max_effort`), which is not an upper bound on
// `PropEngine::effort()` at all — the DFS gate is the slice and an attempt
// spans calls, so past the crossing the cap arrives as 0 and Phase 3 becomes
// a no-op that still pays its entry scan, on precisely the long attempts on
// the hard models the repair exists for.
//
// `effort` is in-out: the caller's running total is added to, so the valve
// gates the *delta* this call charges rather than the absolute value it was
// handed.  No caller passes a nonzero total today -- `fpr_core.cpp` is the
// only one and passes 0, and `repair_search` reuses `walksat_select_move`
// rather than this function -- so the two forms are behaviourally identical
// as things stand.  The delta form is what keeps the valve per call for a
// caller that does accumulate, which is the property #156 wanted to make
// un-reintroducible.
bool walksat_repair(const PropEngine& data, std::vector<double>& solution,
                    std::vector<double>& lhs_cache, const double* col_lb, const double* col_ub,
                    HighsInt max_iterations, double noise, bool track_best, Rng& rng,
                    size_t& effort, WalkSatScratch& scratch);

// Greedy 1-opt: shift each integer variable by ±1 toward better objective
// if the shift maintains feasibility. Modifies solution/lhs_cache in place.
void greedy_1opt(const PropEngine& data, std::vector<double>& solution,
                 std::vector<double>& lhs_cache, const double* col_cost, bool minimize,
                 size_t& effort);
