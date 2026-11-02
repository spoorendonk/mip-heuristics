#pragma once

#include "heuristic_common.h"
#include "heuristic_context.h"

#include <vector>

// The objective cut as a model row (#208, #224), for the heuristics that read
// the constraint matrix row-wise or column-wise through the view and
// propagate it — FPR's attempt, and Scylla's rounding through it.  The
// objective `col_cost . x` becomes row `nrow`, bounded above by the cutoff
// less the view's offset, so propagation tightens columns against it like
// any other row and the search never fixes its way above the cutoff.
//
// Built once per dispatch on the dispatching thread, and only when the cut is
// on.  Row `nrow` starts at `+inf` (no cut): each worker takes its own
// `ObjectiveRowView` and moves only that copy's bound, so nothing shared is
// written after the parallel region starts.
//
// What the view changes and what it keeps:
// - The row-wise arrays, the CSC and the row bounds hold `nrow + 1` rows; the
//   first `nrow` are the base view's, in order, so row indices keep their
//   meaning.  The objective row's entries are the costs that are exactly
//   nonzero: dropping the ones below `epsilon`, as `has_objective` does,
//   would make the row a relaxation of the cut it stands for.
// - The lock counts are the base view's.  Counting the objective row would
//   give every column with a positive cost an up-lock and every one with a
//   negative cost a down-lock, which changes which columns FPR's
//   trivially-roundable phase rounds; that phase is about the model's rows,
//   not the cut's.
// - `model` still points at the base model, whose `a_matrix_` has `nrow`
//   rows: a reader of `model` (FJ's row-wise copy) must take the base view,
//   never this one.  `degenerate()` is decided on the base view too: a model
//   with no rows has one here.  Everything else — bounds, costs, offset, integrality,
//   tolerances and both snapshots — is the base view's.
class ObjectiveRowStorage {
public:
    explicit ObjectiveRowStorage(const ProblemView& base);
    // `view_` points into this object's own members.
    ObjectiveRowStorage(const ObjectiveRowStorage&) = delete;
    ObjectiveRowStorage& operator=(const ObjectiveRowStorage&) = delete;
    ObjectiveRowStorage(ObjectiveRowStorage&&) = delete;
    ObjectiveRowStorage& operator=(ObjectiveRowStorage&&) = delete;
    ~ObjectiveRowStorage() = default;

    // The view with row `nrow` uncut (`+inf`).
    [[nodiscard]] const ProblemView& view() const { return view_; }

    // The objective row's index: the base view's `nrow`.
    [[nodiscard]] HighsInt objective_row() const { return view_.nrow - 1; }

private:
    std::vector<HighsInt> ar_start_;
    std::vector<HighsInt> ar_index_;
    std::vector<double> ar_value_;
    CscMatrix csc_;
    std::vector<double> row_lower_;
    std::vector<double> row_upper_;
    ProblemView view_;
};

// One worker's copy of `ObjectiveRowStorage::view()`, whose `row_upper`
// points at a vector this object owns, so the worker can move the cutoff
// without touching a peer's.  Everything else is shared with the storage,
// which must outlive it.  The pointer stays the same for the object's life,
// so a `PropEngine` cached on this view (`acquire_engine`'s pointer-identity
// check) is reused across attempts, and the engine reads the bound live:
// `set_cutoff` before an attempt is all a worker does.
class ObjectiveRowView {
public:
    explicit ObjectiveRowView(const ObjectiveRowStorage& storage);
    // `view_` points into this object's own `row_upper_`.
    ObjectiveRowView(const ObjectiveRowView&) = delete;
    ObjectiveRowView& operator=(const ObjectiveRowView&) = delete;
    ObjectiveRowView(ObjectiveRowView&&) = delete;
    ObjectiveRowView& operator=(ObjectiveRowView&&) = delete;
    ~ObjectiveRowView() = default;

    [[nodiscard]] const ProblemView& view() const { return view_; }

    // Bound the objective row by `target`, an objective in the pool's form
    // (offset included; `objective_cut_target`'s result).  A non-finite
    // `target` — `kHighsInf`, the cut off or the pool empty — lifts the cut.
    void set_cutoff(double target);

    // The objective row's upper bound in force: `target - offset`, or
    // `kHighsInf`.
    [[nodiscard]] double row_bound() const { return row_upper_.back(); }

private:
    std::vector<double> row_upper_;
    ProblemView view_;
};
