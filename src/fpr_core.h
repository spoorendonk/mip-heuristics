#pragma once

#include "fpr_strategies.h"
#include "prop_engine.h"
#include "rng.h"
#include "util/HighsInt.h"
#include "walksat.h"

#include <cstddef>
#include <optional>
#include <vector>

struct CscMatrix;
struct HeuristicResult;
class HighsMipSolver;

// DFS node used by fpr_attempt's Phase 2 fix-and-propagate search.  Declared
// here only so FprScratch can own a reusable stack across calls; never
// constructed outside fpr_core.cpp.
struct FprDfsNode {
    HighsInt var;
    double val;
    HighsInt vs_mark;
    HighsInt sol_mark;
    HighsInt act_mark;
    HighsInt pq_mark;
    HighsInt cursor_reset;
};

// DFS node used by repair_search (paper Fig. 5).  Declared here only so
// FprScratch can own a reusable stack across calls; never constructed
// outside repair_search.cpp.
struct RepairSearchNode {
    HighsInt var;        // variable to branch on (-1 for root)
    double val;          // fix value or bound value
    bool is_fix;         // true = fix(var, val), false = tighten bound
    bool is_lb;          // if !is_fix: true = tighten_lb, false = tighten_ub
    HighsInt e_vs_mark;  // E undo marks
    HighsInt e_sol_mark;
    HighsInt r_vs_mark;  // R undo marks
    HighsInt r_sol_mark;
    HighsInt sol_undo_mark;  // solution undo mark
    HighsInt lhs_undo_mark;  // lhs_cache undo mark
    double violation;        // total violation at parent (for BacktrackBestOpen)
};

// Per-worker scratch buffers for fpr_attempt.  Reused across calls to avoid
// malloc/free churn on the DFS+repair hot path.  fpr_attempt clears (not
// frees) the vectors at entry so capacity persists across attempts.  One
// scratch per worker thread; not thread-safe.
struct FprScratch {
    // Phase 1: variable ordering buffer.
    std::vector<HighsInt> var_order;

    // Phase 2: DFS node stack for fix-and-propagate.
    std::vector<FprDfsNode> dfs_stack;

    // Phase 2 → Phase 3: incremental row LHS cache and the complete
    // solution extracted from PropEngine for the repair step.
    std::vector<double> lhs_cache;
    std::vector<double> solution;

    // Phase 3: WalkSAT / RepairSearch scratch.  `walksat` is shared between
    // walksat_repair and repair_search (they are never called concurrently —
    // repair_search is an alternative Phase 3 to walksat_repair at the same
    // call site in fpr_attempt).  repair_search reuses its violated /
    // violated_pos / sol_undo / lhs_undo fields, and the nested
    // walksat_select_move call inside repair_search uses cand /
    // best_indices.
    WalkSatScratch walksat;

    // Phase 3 RepairSearch: DFS node stack (paper Fig. 5 Q).
    std::vector<RepairSearchNode> repair_dfs_stack;

    // Phase 3 RepairSearch: best-seen solution / lhs snapshots (paper line 17
    // / line 28).  Kept in scratch so the O(ncol)+O(nrow) copies on each
    // improvement reuse previously-allocated capacity across calls.
    std::vector<double> repair_best_solution;
    std::vector<double> repair_best_lhs;

    // Phase 2: reusable primary PropEngine E.  Constructed lazily on the
    // first fpr_attempt call for this worker (problem data is pulled from
    // HighsMipSolver at that point) and reset() on subsequent calls so
    // every internal vector (vs_, solution_, prop_in_wl_, undo stacks,
    // activity state, and the IndexedMinHeap-backed domain PQ) retains
    // its capacity across reuse.  Constructing PropEngine freshly per
    // attempt was the dominant remaining allocation source in
    // fpr_attempt's hot path (PropEngine::PropEngine allocates
    // vs_(ncol), solution_(ncol), prop_in_wl_(nrow) and reserves 4*ncol
    // for two undo vectors).  A single engine per worker amortises those
    // allocations across attempts.
    std::optional<PropEngine> prop_engine;

    // Phase 3 RepairSearch: reusable secondary PropEngine R.
    // repair_search constructs R from the *same* problem data as E (see
    // repair_search.cpp:PropEngine R(...) — every pointer comes from
    // E.ar_*()/E.csc_*()/col_lb/col_ub/...), so R's validity guard
    // reduces to "is the cached R's ncol/nrow consistent with E?".
    // Same capacity-retention rationale as prop_engine above.
    std::optional<PropEngine> repair_prop_engine_r;
};

struct FprConfig {
    size_t max_effort;  // effort budget (coefficient accesses)
    // Per-variable hint for choose_fix_value (nullable; length ncol if non-null).
    // Used only on attempt 0 (papers: FPR uses incumbent, Scylla uses LP sol).
    const double *hint;
    // Ranking scores per variable (length ncol; caller computes).
    // Used only when strategy is null (legacy mode).
    const double *scores;
    // Fallback values for zero-cost continuous vars (length ncol)
    const double *cont_fallback;
    // Optional pre-built CSC matrix (avoids redundant build if caller already has
    // one)
    const CscMatrix *csc;

    // --- Framework mode (paper Section 3) ---
    FrameworkMode mode = FrameworkMode::kDiveprop;

    // --- Strategy (paper Table 3) ---
    // When non-null, uses the paper's variable ranking and value selection.
    // When null, falls back to legacy scores-based ranking + hint/goodobj.
    const FprStrategyConfig *strategy = nullptr;
    // LP reference solution for LP-based strategies (nullable).
    const double *lp_ref = nullptr;

    // --- Pre-computed variable order (avoids data races on cliquePartition) ---
    // When non-null, fpr_attempt uses this order instead of computing one.
    const HighsInt *precomputed_var_order = nullptr;
    HighsInt precomputed_var_order_size = 0;

    // --- Repair parameters (paper: Salvagnin et al. 2025, Section 5) ---
    // Noise parameter p: probability of random walk move (paper default: 0.75).
    // Greedy probability = 1 - repair_noise.
    double repair_noise = 0.75;
    // DFS node limit for RepairSearch mode (paper Fig. 5).  The paper quotes
    // 200; we cap at 50 because RepairSearch's two PropEngine fixpoints per
    // node dominate cost on tight instances (each ~760k coef accesses on
    // 9k-nnz LPs), so 200 nodes can burn ~1.4 s regardless of max_effort
    // (see `bench/FPR_REPAIR_SEARCH_LOCKS.md` for the ticino profile).
    HighsInt repair_iterations = 50;

    // Step limit for flat WalkSAT repair (kDfsrep / kDive / kDiveprop arms).
    // Kept at the paper's 200 because WalkSAT steps are cheap (O(row.degree)
    // per flip) and the RepairSearch blow-up rationale does not apply.
    HighsInt walksat_iterations = 200;
    // Track best total violation during walk and restore at end (paper: yes).
    bool repair_track_best = true;

    // Optional per-worker scratch buffers.  When non-null, fpr_attempt reuses
    // these vectors instead of allocating locals — the intended hot-path use
    // (FPR/FPR_LP/Scylla/Portfolio workers).  When null, local vectors are
    // used (handy for one-shot callers).  Not thread-safe.
    //
    // Lifetime constraint: if `scratch` is non-null and persists across
    // fpr_attempt calls, the problem data pointed to by `csc` and by
    // HighsMipSolver's AR*/bounds/integrality buffers must remain valid
    // for the scratch's lifetime.  fpr_attempt caches a PropEngine inside
    // the scratch that holds observer pointers into those buffers, and
    // the pointer-identity guard only re-emplaces when pointer *values*
    // differ — dangling-pointer comparison is technically indeterminate
    // per the C++ standard (benign on all mainstream implementations).
    // In practice every hot-path caller pairs a stable `csc` with its
    // scratch, so this is a latent-footgun warning rather than a live
    // hazard.
    FprScratch *scratch = nullptr;
};

// Single-attempt variant for portfolio mode. Returns result without submitting.
// Uses provided RNG and attempt index. If initial_solution is non-null, uses it
// as the starting point (overriding cfg.hint). Otherwise falls back to cfg.hint
// on attempt 0, or random initialization on later attempts.
HeuristicResult fpr_attempt(HighsMipSolver &mipsolver, const FprConfig &cfg, Rng &rng,
                            int attempt_idx, const double *initial_solution);
