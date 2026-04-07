#pragma once

#include <vector>

#include "heuristic_common.h"
#include "util/HighsInt.h"

// Reusable AC-3 constraint propagation engine.
// Owns its own VarState, solution, and undo stacks.
// References shared read-only problem data (constraint matrix, bounds).
class PropEngine {
 public:
  PropEngine(HighsInt ncol, HighsInt nrow, const HighsInt* ar_start,
             const HighsInt* ar_index, const double* ar_value,
             const HighsInt* csc_start, const HighsInt* csc_row,
             const double* csc_val, const double* col_lb, const double* col_ub,
             const double* row_lo, const double* row_hi,
             const HighsVarType* integrality, double feastol);

  // Convenience: extract CSC pointers from CscMatrix.
  PropEngine(HighsInt ncol, HighsInt nrow, const HighsInt* ar_start,
             const HighsInt* ar_index, const double* ar_value,
             const CscMatrix& csc, const double* col_lb, const double* col_ub,
             const double* row_lo, const double* row_hi,
             const HighsVarType* integrality, double feastol)
      : PropEngine(ncol, nrow, ar_start, ar_index, ar_value,
                   csc.col_start.data(), csc.col_row.data(), csc.col_val.data(),
                   col_lb, col_ub, row_lo, row_hi, integrality, feastol) {}

  // Fix variable j to value. Returns false if value outside domain.
  bool fix(HighsInt j, double value);

  // Tighten lower bound of variable j. Returns false if infeasible.
  bool tighten_lb(HighsInt j, double new_lb);

  // Tighten upper bound of variable j. Returns false if infeasible.
  bool tighten_ub(HighsInt j, double new_ub);

  // AC-3 constraint propagation. If fixed_var >= 0, seeds worklist from
  // that variable's rows. If fixed_var == -1, assumes worklist already seeded.
  // Returns false on infeasibility or budget exhaustion.
  bool propagate(HighsInt fixed_var = -1);

  // Add rows of variable j to the propagation worklist.
  void seed_worklist(HighsInt j);

  // Restore state to the given undo marks. Also clears the propagation
  // worklist to avoid stale entries referencing now-reverted state.
  // If act_mark >= 0, also restores row activities to that mark.
  void backtrack_to(HighsInt vs_mark, HighsInt sol_mark, HighsInt act_mark = -1);

  // Current undo stack sizes (for DFS node marks).
  HighsInt vs_mark() const;
  HighsInt sol_mark() const;

  // Access variable state and solution values.
  VarState& var(HighsInt j) { return vs_[j]; }
  const VarState& var(HighsInt j) const { return vs_[j]; }
  double& sol(HighsInt j) { return solution_[j]; }
  double sol(HighsInt j) const { return solution_[j]; }

  // Direct access to the full solution vector.
  double* sol_data() { return solution_.data(); }
  const double* sol_data() const { return solution_.data(); }

  // Reset all variables to global bounds, nothing fixed.
  void reset();

  // Row activity tracking: per-row min/max activities computed from current
  // variable bounds. Maintained incrementally across fix/tighten/propagate.
  // Must call init_activities() once before use.
  void init_activities();
  double row_min_activity(HighsInt i) const { return min_activity_[i]; }
  double row_max_activity(HighsInt i) const { return max_activity_[i]; }
  const double* min_activity_data() const { return min_activity_.data(); }
  const double* max_activity_data() const { return max_activity_.data(); }
  bool activities_initialized() const { return !min_activity_.empty(); }
  HighsInt act_mark() const;

  // Accumulated propagation effort (coefficient accesses).
  size_t effort() const { return prop_work_; }
  void add_effort(size_t e) { prop_work_ += e; }

  HighsInt ncol() const { return ncol_; }
  HighsInt nrow() const { return nrow_; }
  double feastol() const { return feastol_; }
  bool is_int(HighsInt j) const {
    return integrality_[j] != HighsVarType::kContinuous;
  }

  // Read-only access to problem data.
  const double* col_lb() const { return col_lb_; }
  const double* col_ub() const { return col_ub_; }
  const double* row_lo() const { return row_lo_; }
  const double* row_hi() const { return row_hi_; }
  const HighsVarType* integrality() const { return integrality_; }
  const HighsInt* ar_start() const { return ar_start_; }
  const HighsInt* ar_index() const { return ar_index_; }
  const double* ar_value() const { return ar_value_; }
  const HighsInt* csc_start() const { return col_start_; }
  const HighsInt* csc_row() const { return col_row_; }
  const double* csc_val() const { return col_val_; }

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
};
