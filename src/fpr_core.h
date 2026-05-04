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
    HighsInt e_pq_mark;  // E PQ mark (-1 if pq not initialized).  Required
                         // when Phase 2 used a dynamic-var strategy that
                         // ran `E.init_domain_pq()` — without restoring
                         // pq state on backtrack the heap diverges from
                         // vs_ and a subsequent pq_notify erases a var
                         // not in the heap (#77 lifecycle rotation now
                         // exercises this combination).
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

// ---------------------------------------------------------------------------
// Pause/resume lifecycle (issue #77)
// ---------------------------------------------------------------------------
//
// `fpr_attempt` above runs Phase 1 (rank), Phase 2 (DFS fix-and-propagate),
// Phase 2.5 (fill remaining unfixed), and Phase 3 (repair / 1-opt) end to end
// in one call.  On a long DFS subtree this pegs cfg.max_effort and discards
// the subtree's work — peers in the same epoch barrier idle while the slow
// worker burns its slice.  The lifecycle below splits the attempt into three
// callable stages so a worker can pause at the per-epoch gate, return to the
// runner, then resume next epoch with the DFS state intact.
//
// State that must survive across `fpr_attempt_step` calls lives in
// `FprAttemptState` (DFS cursor / counters / phase tag) and in `FprScratch`
// (`dfs_stack` and `prop_engine`).  PropEngine.reset() runs once inside
// `begin`; calling it between `step` invocations corrupts the DFS undo
// stacks — the cardinal correctness invariant of this API.
//
// Determinism: every per-call input is either identical across runs (the
// `cfg` reference and `mipsolver` problem buffers are immutable for the
// attempt's lifetime) or a per-worker piece of state threaded through
// (`Rng &rng` and `FprAttemptState`).  Two runs with identical seeds
// produce bit-identical attempt traces — see `[fpr][resume][determinism]`
// in tests/test_fpr.cpp.
//
// One-shot callers (portfolio, scylla, fpr_lp, tests) keep using
// `fpr_attempt` above — it is a thin wrapper around begin/step/finish.

struct FprAttemptState {
    // Set in `fpr_attempt_begin`; read in `step`/`finish`.
    HighsInt ncol = 0;
    HighsInt nrow = 0;
    int attempt_idx = 0;
    bool dynamic_var = false;
    bool do_propagate = false;
    bool do_backtrack = false;
    HighsInt node_limit = 0;
    HighsInt var_order_size = 0;

    // DFS progress.  `var_order_cursor` and `nodes_visited` advance during
    // `step`; `found_complete` flips true when the DFS hits a leaf with
    // every integer fixed.  `dfs_stack` and `prop_engine` live in
    // `FprScratch` (capacity persists across attempts).
    HighsInt var_order_cursor = 0;
    HighsInt nodes_visited = 0;
    bool found_complete = false;

    // Cumulative effort consumed by this attempt across all begin/step/finish
    // calls.  `step`'s budget gate compares against the engine's effort
    // counter; the worker reads `effort_consumed` deltas to attribute work
    // to the current epoch slice.
    size_t effort_consumed = 0;

    enum class Phase {
        // Attempt has not begun (constructor default) or has been
        // finalized by `fpr_attempt_finish`.  `begin` may be called.
        kIdle,
        // DFS is in progress; another `step` call may resume it.
        kDfs,
        // DFS has terminated (leaf found or stack exhausted); the next
        // call must be `fpr_attempt_finish`.
        kReadyToFinish,
    };
    Phase phase = Phase::kIdle;
};

enum class FprStepResult {
    // Per-call effort budget exhausted; attempt is alive, caller may
    // re-enter `fpr_attempt_step` with more budget next epoch.
    kBudgetGate,
    // DFS terminated (success leaf found or stack exhausted / node_limit
    // hit); caller must call `fpr_attempt_finish` to materialize the
    // verdict.
    kVerdictReady,
};

// Phase 1 + Phase 2 seeding.  Initializes E (PropEngine) once, runs the
// trivially-roundable fixings + initial propagation, and pushes the root
// DFS node.  Sets `state.phase = kDfs` (or `kReadyToFinish` if Phase 1
// already produced a complete fixing).
//
// `cfg.scratch` MUST be non-null; the lifecycle API does not support the
// one-shot `local_scratch` fallback (one-shot callers should keep using
// `fpr_attempt`).
void fpr_attempt_begin(FprAttemptState &state, HighsMipSolver &mipsolver, const FprConfig &cfg,
                       Rng &rng, int attempt_idx, const double *initial_solution);

// Phase 2 DFS resume.  Runs the fix-and-propagate loop until either the
// per-call effort budget is exhausted (returns `kBudgetGate`) or the DFS
// terminates (returns `kVerdictReady`, caller calls `finish`).
//
// `effort_remaining` is the per-call slice; the attempt's overall cap is
// `cfg.max_effort` (which the worker typically sets very high so the slice
// is the only effective gate).  Calling `step` when
// `state.phase != kDfs` is a programming error.
FprStepResult fpr_attempt_step(FprAttemptState &state, HighsMipSolver &mipsolver,
                               const FprConfig &cfg, Rng &rng, size_t effort_remaining);

// Phase 2.5 (fill remaining unfixed) + Phase 3 (repair / 1-opt) + result
// build.  Always runs to verdict in one call (Phase 3 self-throttles via
// `cfg.repair_iterations` / `cfg.walksat_iterations`).  Sets
// `state.phase = kIdle` so the next attempt can call `begin` on the same
// state object.  `state.found_complete == false` shortcuts to a `failed`
// verdict.
HeuristicResult fpr_attempt_finish(FprAttemptState &state, HighsMipSolver &mipsolver,
                                   const FprConfig &cfg, Rng &rng);
