#include "prop_engine.h"

#include <algorithm>
#include <cmath>

#include "lp_data/HConst.h"

PropEngine::PropEngine(HighsInt ncol, HighsInt nrow, const HighsInt* ar_start,
                       const HighsInt* ar_index, const double* ar_value,
                       const HighsInt* csc_start, const HighsInt* csc_row_p,
                       const double* csc_val_p, const double* col_lb,
                       const double* col_ub, const double* row_lo,
                       const double* row_hi, const HighsVarType* integrality,
                       double feastol)
    : ncol_(ncol),
      nrow_(nrow),
      ar_start_(ar_start),
      ar_index_(ar_index),
      ar_value_(ar_value),
      col_start_(csc_start),
      col_row_(csc_row_p),
      col_val_(csc_val_p),
      col_lb_(col_lb),
      col_ub_(col_ub),
      row_lo_(row_lo),
      row_hi_(row_hi),
      integrality_(integrality),
      feastol_(feastol),
      nnz_(static_cast<size_t>(ar_start[nrow])),
      vs_(ncol),
      solution_(ncol, 0.0),
      prop_in_wl_(nrow, 0) {
  vs_undo_.reserve(ncol);
  sol_undo_.reserve(ncol);
  prop_worklist_.reserve(nrow);
  reset();
}

void PropEngine::reset() {
  for (HighsInt j = 0; j < ncol_; ++j) {
    vs_[j].lb = col_lb_[j];
    vs_[j].ub = col_ub_[j];
    vs_[j].val = 0.0;
    vs_[j].fixed = false;
    solution_[j] = 0.0;
  }
  vs_undo_.clear();
  sol_undo_.clear();
  for (HighsInt i : prop_worklist_) {
    prop_in_wl_[i] = 0;
  }
  prop_worklist_.clear();
}

bool PropEngine::fix(HighsInt j, double value) {
  if (value < vs_[j].lb - feastol_ || value > vs_[j].ub + feastol_) {
    return false;
  }
  value = std::max(vs_[j].lb, std::min(vs_[j].ub, value));
  if (is_int(j)) {
    value = std::round(value);
  }
  vs_undo_.push_back({j, vs_[j]});
  sol_undo_.push_back({j, solution_[j]});
  vs_[j].fixed = true;
  vs_[j].val = value;
  solution_[j] = value;
  return true;
}

bool PropEngine::tighten_lb(HighsInt j, double new_lb) {
  if (is_int(j)) {
    new_lb = std::ceil(new_lb - feastol_);
  }
  new_lb = std::max(new_lb, col_lb_[j]);
  if (new_lb > vs_[j].ub + feastol_) {
    return false;
  }
  if (new_lb <= vs_[j].lb + feastol_) {
    return true;  // no tightening
  }
  vs_undo_.push_back({j, vs_[j]});
  sol_undo_.push_back({j, solution_[j]});
  vs_[j].lb = new_lb;
  // Auto-fix if domain becomes singleton
  if (!vs_[j].fixed && vs_[j].ub - vs_[j].lb < feastol_) {
    double val = (vs_[j].lb + vs_[j].ub) * 0.5;
    if (is_int(j)) val = std::round(val);
    vs_[j].fixed = true;
    vs_[j].val = val;
    solution_[j] = val;
  }
  seed_worklist(j);
  return true;
}

bool PropEngine::tighten_ub(HighsInt j, double new_ub) {
  if (is_int(j)) {
    new_ub = std::floor(new_ub + feastol_);
  }
  new_ub = std::min(new_ub, col_ub_[j]);
  if (new_ub < vs_[j].lb - feastol_) {
    return false;
  }
  if (new_ub >= vs_[j].ub - feastol_) {
    return true;  // no tightening
  }
  vs_undo_.push_back({j, vs_[j]});
  sol_undo_.push_back({j, solution_[j]});
  vs_[j].ub = new_ub;
  // Auto-fix if domain becomes singleton
  if (!vs_[j].fixed && vs_[j].ub - vs_[j].lb < feastol_) {
    double val = (vs_[j].lb + vs_[j].ub) * 0.5;
    if (is_int(j)) val = std::round(val);
    vs_[j].fixed = true;
    vs_[j].val = val;
    solution_[j] = val;
  }
  seed_worklist(j);
  return true;
}

void PropEngine::seed_worklist(HighsInt j) {
  for (HighsInt p = col_start_[j]; p < col_start_[j + 1]; ++p) {
    HighsInt i = col_row_[p];
    if (!prop_in_wl_[i]) {
      prop_in_wl_[i] = 1;
      prop_worklist_.push_back(i);
    }
  }
}

bool PropEngine::propagate(HighsInt fixed_var) {
  if (fixed_var >= 0) {
    // Seed only the rows containing the just-fixed variable (AC-3).
    // Clear stale prop_in_wl_ flags before clearing the worklist.
    for (HighsInt wi : prop_worklist_) {
      prop_in_wl_[wi] = 0;
    }
    prop_worklist_.clear();
    for (HighsInt p = col_start_[fixed_var]; p < col_start_[fixed_var + 1];
         ++p) {
      HighsInt i = col_row_[p];
      if (!prop_in_wl_[i]) {
        prop_in_wl_[i] = 1;
        prop_worklist_.push_back(i);
      }
    }
  }
  // When fixed_var == -1, assume prop_worklist is already seeded by caller.

  size_t prop_work = 0;
  const size_t prop_budget = 10 * nnz_;
  while (!prop_worklist_.empty()) {
    HighsInt i = prop_worklist_.back();
    prop_worklist_.pop_back();
    prop_in_wl_[i] = 0;
    prop_work += ar_start_[i + 1] - ar_start_[i];
    if (prop_work > prop_budget) {
      prop_work_ += prop_work;
      // Clear stale worklist markers
      for (HighsInt wi : prop_worklist_) {
        prop_in_wl_[wi] = 0;
      }
      prop_worklist_.clear();
      return false;
    }

    double fixed_sum = 0.0;
    double min_act = 0.0, max_act = 0.0;
    HighsInt num_unfixed = 0;

    for (HighsInt k = ar_start_[i]; k < ar_start_[i + 1]; ++k) {
      HighsInt j = ar_index_[k];
      double a = ar_value_[k];
      if (vs_[j].fixed) {
        fixed_sum += a * vs_[j].val;
      } else {
        ++num_unfixed;
        if (a > 0) {
          min_act += a * vs_[j].lb;
          max_act += a * vs_[j].ub;
        } else {
          min_act += a * vs_[j].ub;
          max_act += a * vs_[j].lb;
        }
      }
    }

    if (num_unfixed == 0) {
      continue;
    }

    bool has_upper = row_hi_[i] < kHighsInf;
    bool has_lower = row_lo_[i] > -kHighsInf;

    for (HighsInt k = ar_start_[i]; k < ar_start_[i + 1]; ++k) {
      HighsInt j = ar_index_[k];
      if (vs_[j].fixed) {
        continue;
      }
      double a = ar_value_[k];
      if (std::abs(a) < 1e-15) {
        continue;
      }

      double old_lb = vs_[j].lb;
      double old_ub = vs_[j].ub;

      double min_others, max_others;
      if (a > 0) {
        min_others = min_act - a * old_lb;
        max_others = max_act - a * old_ub;
      } else {
        min_others = min_act - a * old_ub;
        max_others = max_act - a * old_lb;
      }

      double new_lb = old_lb;
      double new_ub = old_ub;

      if (has_upper) {
        double bound = row_hi_[i] - fixed_sum - min_others;
        if (a > 0) {
          new_ub = std::min(new_ub, bound / a);
        } else {
          new_lb = std::max(new_lb, bound / a);
        }
      }

      if (has_lower) {
        double bound = row_lo_[i] - fixed_sum - max_others;
        if (a > 0) {
          new_lb = std::max(new_lb, bound / a);
        } else {
          new_ub = std::min(new_ub, bound / a);
        }
      }

      if (is_int(j)) {
        new_lb = std::ceil(new_lb - feastol_);
        new_ub = std::floor(new_ub + feastol_);
      }

      new_lb = std::max(new_lb, col_lb_[j]);
      new_ub = std::min(new_ub, col_ub_[j]);

      if (new_lb > new_ub + feastol_) {
        prop_work_ += prop_work;
        // Clear stale worklist markers on infeasibility too
        for (HighsInt wi : prop_worklist_) {
          prop_in_wl_[wi] = 0;
        }
        prop_worklist_.clear();
        return false;
      }

      bool changed = false;
      if (new_lb > old_lb + feastol_ || new_ub < old_ub - feastol_) {
        vs_undo_.push_back({j, vs_[j]});
        sol_undo_.push_back({j, solution_[j]});
        if (new_lb > old_lb + feastol_) {
          vs_[j].lb = new_lb;
          changed = true;
        }
        if (new_ub < old_ub - feastol_) {
          vs_[j].ub = new_ub;
          changed = true;
        }
      }

      if (!vs_[j].fixed && vs_[j].ub - vs_[j].lb < feastol_) {
        if (!changed) {
          vs_undo_.push_back({j, vs_[j]});
          sol_undo_.push_back({j, solution_[j]});
        }
        double val = (vs_[j].lb + vs_[j].ub) * 0.5;
        if (is_int(j)) {
          val = std::round(val);
        }
        vs_[j].fixed = true;
        vs_[j].val = val;
        solution_[j] = val;
        changed = true;
      }

      if (changed) {
        seed_worklist(j);
      }
    }
  }
  prop_work_ += prop_work;
  return true;
}

void PropEngine::backtrack_to(HighsInt vs_mark_target, HighsInt sol_mark_target) {
  for (HighsInt u = static_cast<HighsInt>(vs_undo_.size()) - 1;
       u >= vs_mark_target; --u) {
    vs_[vs_undo_[u].first] = vs_undo_[u].second;
  }
  vs_undo_.resize(vs_mark_target);
  for (HighsInt u = static_cast<HighsInt>(sol_undo_.size()) - 1;
       u >= sol_mark_target; --u) {
    solution_[sol_undo_[u].first] = sol_undo_[u].second;
  }
  sol_undo_.resize(sol_mark_target);
  // Clear stale worklist entries that may reference now-reverted state.
  for (HighsInt wi : prop_worklist_) {
    prop_in_wl_[wi] = 0;
  }
  prop_worklist_.clear();
}

HighsInt PropEngine::vs_mark() const {
  return static_cast<HighsInt>(vs_undo_.size());
}

HighsInt PropEngine::sol_mark() const {
  return static_cast<HighsInt>(sol_undo_.size());
}
