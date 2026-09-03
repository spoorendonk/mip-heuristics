#pragma once

#include "deadline.h"
#include "heuristic_common.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <vector>

// Indexed min-heap keyed by (domain_size, var) — replaces the
// std::set<pair<double,HighsInt>> that PropEngine previously used for its
// domain-size priority queue.  std::set is a node-based rb-tree: clear()
// frees every node and init_domain_pq() then re-heap-allocates one node
// per unfixed integer variable, so reset+reuse paid the full allocation
// cost on every attempt.  The indexed heap is backed by two std::vector
// members whose capacities persist across clear()/reset() — the same
// capacity-retention property the rest of FprScratch now relies on.
//
// Ordering matches the prior std::set<pair<double,HighsInt>>: ascending
// key, with var index as tiebreak.  The std::set tiebreak was load-bearing
// for determinism tests (same-seed → same solve path depends on which
// equal-domain variable is picked); preserving it keeps those tests green.
//
// Cost model: insert / erase / update-key are O(log N).  top is O(1).
// contains is O(1).  clear is O(|heap|) — only touches pos_ slots that
// are actually occupied, not the whole ncol array.
class IndexedMinHeap {
public:
    // Preallocate capacity for `ncol` variables.  Called once per engine;
    // subsequent clear() / reset() cycles reuse the allocation.
    void reserve(HighsInt ncol);

    // Reset to empty.  Only touches pos_ for currently-heaped entries; the
    // pos_ array's capacity (= ncol) is retained.
    void clear();

    [[nodiscard]] bool empty() const { return heap_.empty(); }
    [[nodiscard]] std::size_t size() const { return heap_.size(); }
    [[nodiscard]] bool contains(HighsInt var) const {
        return var >= 0 && static_cast<std::size_t>(var) < pos_.size() && pos_[var] != kNotPresent;
    }

    // Precondition: !empty().
    [[nodiscard]] HighsInt top_var() const { return heap_.front().var; }

    // Precondition: !contains(var).
    void insert(double key, HighsInt var);

    // Precondition: contains(var).
    void erase(HighsInt var);

    // Precondition: contains(var).  Moves `var` to a new key, re-heapifying.
    void update(HighsInt var, double new_key);

private:
    struct Entry {
        double key;
        HighsInt var;
    };

    static constexpr HighsInt kNotPresent = -1;

    // Compare equivalent to std::pair<double, HighsInt> — breaks ties by
    // var index to match the prior std::set ordering.
    static bool entry_less(const Entry& a, const Entry& b) {
        return a.key < b.key || (a.key == b.key && a.var < b.var);
    }

    void sift_up(HighsInt idx);
    void sift_down(HighsInt idx);

    std::vector<Entry> heap_;
    std::vector<HighsInt> pos_;  // pos_[var] = heap index, or kNotPresent.
};

// Outcome of a `PropEngine::propagate` call (issues #127, #151).
//
// A truncated fixpoint (`kBudgetExhausted`, `kDeadlineExpired`) is *not* a
// verdict: it means fewer inferences were drawn than a full AC-3 pass would
// have drawn, which is sound (every tightening actually applied is a valid
// deduction) but incomplete. Only `kInfeasible` is a proof that no
// completion of the current partial assignment satisfies the model. Every
// caller that used to read `propagate()`'s `bool` as "prune this node" must
// distinguish the verdict from the two truncations: pruning on either would
// discard feasible subtrees the engine simply stopped short of exploring.
//
// The two truncations differ only in what the caller should do *next*, and
// that is the whole reason #151 gave the clock its own state rather than
// folding it into `kBudgetExhausted`. Budget exhaustion says "this one
// fixpoint got expensive", and the search is expected to carry on; deadline
// expiry says "the solve is over", so a caller that can stop promptly
// should, and one that cannot must at least not read it as infeasibility.
enum class PropResult {
    kFixpoint,         // Worklist drained; every reachable deduction applied.
    kInfeasible,       // A row's bounds became inconsistent (lb > ub).
    kBudgetExhausted,  // Per-call matrix-access budget hit before the
                       // worklist drained. Sound but incomplete -- treat the
                       // node as propagated-but-unfinished, never pruned.
    kDeadlineExpired,  // The wall-clock deadline passed mid-fixpoint (#151).
                       // Exactly as sound and exactly as non-pruning as
                       // kBudgetExhausted; see `set_deadline` below.
};

// Reusable AC-3 constraint propagation engine.
// Owns its own VarState, solution, and undo stacks.
// References shared read-only problem data (constraint matrix, bounds).
class PropEngine {
public:
    PropEngine(HighsInt ncol, HighsInt nrow, const HighsInt* ar_start, const HighsInt* ar_index,
               const double* ar_value, const HighsInt* csc_start, const HighsInt* csc_row,
               const double* csc_val, const double* col_lb, const double* col_ub,
               const double* row_lo, const double* row_hi, const HighsVarType* integrality,
               double feastol);

    // Convenience: extract CSC pointers from CscMatrix.
    PropEngine(HighsInt ncol, HighsInt nrow, const HighsInt* ar_start, const HighsInt* ar_index,
               const double* ar_value, const CscMatrix& csc, const double* col_lb,
               const double* col_ub, const double* row_lo, const double* row_hi,
               const HighsVarType* integrality, double feastol)
        : PropEngine(ncol, nrow, ar_start, ar_index, ar_value, csc.col_start.data(),
                     csc.col_row.data(), csc.col_val.data(), col_lb, col_ub, row_lo, row_hi,
                     integrality, feastol) {}

    // Fix variable j to value. Returns false if value outside domain.
    bool fix(HighsInt j, double value);

    // Re-fix variable j to value, *overriding* its current domain rather
    // than refining it (issue #125, paper Sect. 5.1's RepairSearch
    // `SyncChanges`).  `fix()` validates against the column's *current*
    // `[lb, ub]` -- correct for a DFS branching decision, but wrong here:
    // syncing a secondary engine's deduction back into the primary one
    // means overriding what the primary engine already decided (flipping
    // an already-fixed binary, or moving an unfixed column to a value
    // outside its currently-propagated range when the two engines'
    // domains are disjoint), so the only bound that should gate it is the
    // column's *structural* one -- `col_lb`/`col_ub`, the original problem
    // bounds -- not whatever this engine's own propagation had narrowed it
    // to.  Narrows `[lb, ub]` to the singleton `[value, value]`, unlike
    // `fix()`, which leaves the pre-fix bounds in place: a column `fix()`
    // committed keeps whatever `[lb, ub]` it had before the call (wide, for
    // a DFS decision-fix that never got its own bounds tightened) and can
    // therefore still show a fixed column with a domain wider than its
    // value -- this asymmetry is not incidental, it is what lets `fix()`
    // accept a later flip of its own commitment inside the same wide
    // bounds.  `refix()`'s narrowing has no such reason to stay wide (its
    // whole point is a fresh, unambiguous commitment) and produces the
    // same *shape* of narrow-fixed column that an auto-fix does via
    // `tighten_lb`/`tighten_ub`, so a binary it re-fixes stops matching
    // `move_to_disjunction`'s `[lb, ub] == [0, 1]` test and takes the
    // point-split branch instead of the flip branch -- see that
    // function's own note and issue #131.  Since #131 that is correct
    // rather than vacuous: the shifted interval is the singleton at the
    // flipped value, so the preferred child forces the flip.  It still
    // spends a node on the other child, which re-imposes the column's
    // current value and which the flip form would not produce.  Returns false
    // only if `value` falls outside the column's structural bounds --
    // callers of `refix()` in `sync_changes` always derive `value` from
    // the secondary engine R's own domain, which is itself always inside
    // the structural bounds, so this should not fail in practice; any
    // deeper inconsistency with the *rest* of the primary engine's state
    // is a job for the propagation that follows, not for this call
    // (paper's own ordering: flip, then let propagation judge).
    bool refix(HighsInt j, double value);

    // Translate variable j's current domain by `delta` -- the paper's
    // repair *shift* (Sect. 5, issue #124).  The interval moves; its width
    // does not.  That is the whole point: Sect. 5 needs shifts on non-fixed
    // columns so a propagated tightening can be moved rather than dropped,
    // and its footnote rules domain *enlargement* out as a repair move
    // outright, "as it would lead to trivial repair actions where fixings
    // are just undone".  So this is neither `fix()` (which refines within
    // the current domain) nor `refix()` (which overrides it with a
    // singleton): it is the one operation that leaves the current domain
    // intact and only relocates it.
    //
    // A fixed column shifts by its value, since its effective domain is the
    // singleton `[val, val]` -- note that `fix()` leaves the pre-fix
    // `[lb, ub]` wide, so the *value*, not the bounds, is what a fixed
    // column's interval is read from.  The shifted result is then narrowed
    // to that singleton, the same shape `refix()` produces (see its comment
    // and issue #131).
    //
    // Returns false, changing nothing, when the shift is not available:
    // either endpoint of the current interval infinite (an unbounded
    // interval cannot be translated -- and its activity contribution would
    // not move if it were), or the translated interval leaving the column's
    // *structural* bounds `[col_lb, col_ub]`.  Callers clip `delta` to the
    // structural bounds themselves, so the second case is a guard rather
    // than a control path.
    //
    // Does **not** seed the propagation worklist, unlike `tighten_lb` /
    // `tighten_ub`: repair runs on a state propagation has already refuted,
    // and the paper is explicit that it may not propagate from there.
    bool shift_domain(HighsInt j, double delta);

    // Tighten lower bound of variable j. Returns false if infeasible.
    bool tighten_lb(HighsInt j, double new_lb);

    // Tighten upper bound of variable j. Returns false if infeasible.
    bool tighten_ub(HighsInt j, double new_ub);

    // AC-3 constraint propagation. If fixed_var >= 0, seeds worklist from
    // that variable's rows. If fixed_var == -1, assumes worklist already seeded.
    // Returns kInfeasible only on a proven bound inconsistency; a per-call
    // matrix-access budget hit returns kBudgetExhausted and an expired
    // wall-clock deadline returns kDeadlineExpired, neither of which is a
    // verdict -- see PropResult above. [[nodiscard]] because the four
    // outcomes are not interchangeable (issues #127, #151): a caller that drops
    // the return would silently treat every call as a fixpoint, the same
    // shape of bug this three-valuing fixed. A caller that genuinely has
    // nothing to decide from the verdict discards it explicitly with
    // `static_cast<void>` and a reason, matching the `-Werror=unused-result`
    // culture this project already applies to `IncumbentSink::offer`.
    [[nodiscard]] PropResult propagate(HighsInt fixed_var = -1);

    // Add rows of variable j to the propagation worklist.
    void seed_worklist(HighsInt j);

    // Restore state to the given undo marks. Also clears the propagation
    // worklist to avoid stale entries referencing now-reverted state.
    // If act_mark >= 0, also restores row activities to that mark.
    void backtrack_to(HighsInt vs_mark, HighsInt sol_mark, HighsInt act_mark = -1,
                      HighsInt pq_mark = -1);

    // Current undo stack sizes (for DFS node marks).
    [[nodiscard]] HighsInt vs_mark() const;
    [[nodiscard]] HighsInt sol_mark() const;

    // Access variable state and solution values.
    VarState& var(HighsInt j) { return vs_[j]; }
    [[nodiscard]] const VarState& var(HighsInt j) const { return vs_[j]; }
    double& sol(HighsInt j) { return solution_[j]; }
    [[nodiscard]] double sol(HighsInt j) const { return solution_[j]; }

    // Direct access to the full solution vector.
    double* sol_data() { return solution_.data(); }
    [[nodiscard]] const double* sol_data() const { return solution_.data(); }

    // Reset all variables to global bounds, nothing fixed.
    void reset();

    // Row activity tracking: per-row min/max activities computed from current
    // variable bounds. Maintained incrementally across fix/tighten/propagate.
    // Must call init_activities() once before use.
    void init_activities();
    [[nodiscard]] double row_min_activity(HighsInt i) const { return min_activity_[i]; }
    [[nodiscard]] double row_max_activity(HighsInt i) const { return max_activity_[i]; }
    [[nodiscard]] const double* min_activity_data() const { return min_activity_.data(); }
    [[nodiscard]] const double* max_activity_data() const { return max_activity_.data(); }
    [[nodiscard]] bool activities_initialized() const { return !min_activity_.empty(); }
    [[nodiscard]] HighsInt act_mark() const;

    // Domain priority queue: unfixed integer variables sorted by domain size.
    // Maintained incrementally across fix/tighten/propagate with undo support.
    void init_domain_pq();
    [[nodiscard]] HighsInt pq_top() const;
    [[nodiscard]] HighsInt pq_mark() const;
    [[nodiscard]] bool pq_initialized() const { return pq_active_; }

    // The wall-clock deadline `propagate()` polls from *inside* its own
    // fixpoint loop (issue #151).  A default-constructed `Deadline` (null
    // timer) means "no limit" and costs the loop nothing: the poll cadence
    // counter is then initialised to SIZE_MAX, so the per-row comparison
    // can never fire and no clock is ever read -- an un-armed engine, which
    // is every engine a unit test builds, runs exactly what it ran before.
    //
    // Set rather than taken by the constructor, because the engine outlives
    // an attempt: `fpr_core.cpp`'s `acquire_engine` caches one `PropEngine`
    // per worker across attempts and `repair_search` caches the secondary
    // engine R the same way, so the arming has to happen on the *reuse*
    // path as much as on the construction path.  The `Deadline` must come
    // from `ExecutionContext::deadline()` or `deadline_of(mipsolver)` --
    // the only two ways to build one -- so a fixpoint cannot stop against a
    // different clock or limit than the runner above it.
    void set_deadline(const Deadline& d) { deadline_ = d; }
    [[nodiscard]] const Deadline& deadline() const { return deadline_; }

    // Accumulated propagation effort (coefficient accesses).
    [[nodiscard]] size_t effort() const { return prop_work_; }
    void add_effort(size_t e) { prop_work_ += e; }

    [[nodiscard]] HighsInt ncol() const { return ncol_; }
    [[nodiscard]] HighsInt nrow() const { return nrow_; }
    [[nodiscard]] double feastol() const { return feastol_; }
    [[nodiscard]] bool is_int(HighsInt j) const {
        return integrality_[j] != HighsVarType::kContinuous;
    }

    // Read-only access to problem data.
    [[nodiscard]] const double* col_lb() const { return col_lb_; }
    [[nodiscard]] const double* col_ub() const { return col_ub_; }
    [[nodiscard]] const double* row_lo() const { return row_lo_; }
    [[nodiscard]] const double* row_hi() const { return row_hi_; }
    [[nodiscard]] const HighsVarType* integrality() const { return integrality_; }
    [[nodiscard]] const HighsInt* ar_start() const { return ar_start_; }
    [[nodiscard]] const HighsInt* ar_index() const { return ar_index_; }
    [[nodiscard]] const double* ar_value() const { return ar_value_; }
    [[nodiscard]] const HighsInt* csc_start() const { return col_start_; }
    [[nodiscard]] const HighsInt* csc_row() const { return col_row_; }
    [[nodiscard]] const double* csc_val() const { return col_val_; }

private:
    // Problem dimensions
    HighsInt ncol_;
    HighsInt nrow_;

    // Shared read-only problem data (not owned)
    const HighsInt* ar_start_;
    const HighsInt* ar_index_;
    const double* ar_value_;
    const HighsInt* col_start_;
    const HighsInt* col_row_;
    const double* col_val_;
    const double* col_lb_;
    const double* col_ub_;
    const double* row_lo_;
    const double* row_hi_;
    const HighsVarType* integrality_;
    double feastol_;
    size_t nnz_;

    // Update row activities for all rows containing variable j, given its
    // old VarState. Call after vs_[j] has been modified.
    void update_activities(HighsInt j, const VarState& old_vs);

    // Owned mutable state
    std::vector<VarState> vs_;
    std::vector<double> solution_;
    std::vector<std::pair<HighsInt, VarState>> vs_undo_;
    std::vector<std::pair<HighsInt, double>> sol_undo_;
    std::vector<HighsInt> prop_worklist_;
    std::vector<char> prop_in_wl_;
    size_t prop_work_ = 0;

    // Null (no finite limit) until `set_deadline` arms it.
    Deadline deadline_;

    // Per-row min/max activities (empty until init_activities() is called)
    std::vector<double> min_activity_;
    std::vector<double> max_activity_;

    // Activity undo: {row, old_min, old_max}
    struct ActUndo {
        HighsInt row;
        double old_min;
        double old_max;
    };
    std::vector<ActUndo> act_undo_;

    // Domain priority queue for dynamic variable selection.
    // See IndexedMinHeap above for the rationale behind the vector-backed
    // replacement of the prior std::set<pair<double,HighsInt>>.
    bool pq_active_ = false;
    IndexedMinHeap domain_pq_;
    struct PqUndo {
        HighsInt var;
        double old_dom;    // domain size before change
        double new_dom;    // domain size after change
        bool was_present;  // was the var in the PQ before this change?
        bool is_present;   // is the var in the PQ after this change?
    };
    std::vector<PqUndo> pq_undo_;
    void pq_notify(HighsInt j, const VarState& old_vs);
};
