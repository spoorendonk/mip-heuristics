#include "prop_engine.h"

#include "lp_data/HConst.h"

#include <algorithm>
#include <cmath>

PropEngine::PropEngine(HighsInt ncol, HighsInt nrow, const HighsInt* ar_start,
                       const HighsInt* ar_index, const double* ar_value, const HighsInt* csc_start,
                       const HighsInt* csc_row_p, const double* csc_val_p, const double* col_lb,
                       const double* col_ub, const double* row_lo, const double* row_hi,
                       const HighsVarType* integrality, double feastol)
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
    min_activity_.clear();
    max_activity_.clear();
    act_undo_.clear();
    pq_active_ = false;
    domain_pq_.clear();
    pq_undo_.clear();
}

bool PropEngine::fix(HighsInt j, double value) {
    if (value < vs_[j].lb - feastol_ || value > vs_[j].ub + feastol_) {
        return false;
    }
    value = std::max(vs_[j].lb, std::min(vs_[j].ub, value));
    if (is_int(j)) {
        value = std::round(value);
    }
    VarState old_vs = vs_[j];
    vs_undo_.push_back({j, old_vs});
    sol_undo_.push_back({j, solution_[j]});
    vs_[j].fixed = true;
    vs_[j].val = value;
    solution_[j] = value;
    update_activities(j, old_vs);
    pq_notify(j, old_vs);
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
    VarState old_vs = vs_[j];
    vs_undo_.push_back({j, old_vs});
    sol_undo_.push_back({j, solution_[j]});
    vs_[j].lb = new_lb;
    // Auto-fix if domain becomes singleton
    if (!vs_[j].fixed && vs_[j].ub - vs_[j].lb < feastol_) {
        double val = (vs_[j].lb + vs_[j].ub) * 0.5;
        if (is_int(j)) {
            val = std::round(val);
        }
        vs_[j].fixed = true;
        vs_[j].val = val;
        solution_[j] = val;
    }
    update_activities(j, old_vs);
    pq_notify(j, old_vs);
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
    VarState old_vs = vs_[j];
    vs_undo_.push_back({j, old_vs});
    sol_undo_.push_back({j, solution_[j]});
    vs_[j].ub = new_ub;
    // Auto-fix if domain becomes singleton
    if (!vs_[j].fixed && vs_[j].ub - vs_[j].lb < feastol_) {
        double val = (vs_[j].lb + vs_[j].ub) * 0.5;
        if (is_int(j)) {
            val = std::round(val);
        }
        vs_[j].fixed = true;
        vs_[j].val = val;
        solution_[j] = val;
    }
    update_activities(j, old_vs);
    pq_notify(j, old_vs);
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
        for (HighsInt p = col_start_[fixed_var]; p < col_start_[fixed_var + 1]; ++p) {
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
            VarState pre_change_vs = vs_[j];
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
                update_activities(j, pre_change_vs);
                pq_notify(j, pre_change_vs);
                seed_worklist(j);
            }
        }
    }
    prop_work_ += prop_work;
    return true;
}

void PropEngine::backtrack_to(HighsInt vs_mark_target, HighsInt sol_mark_target,
                              HighsInt act_mark_target, HighsInt pq_mark_target) {
    for (HighsInt u = static_cast<HighsInt>(vs_undo_.size()) - 1; u >= vs_mark_target; --u) {
        vs_[vs_undo_[u].first] = vs_undo_[u].second;
    }
    vs_undo_.resize(vs_mark_target);
    for (HighsInt u = static_cast<HighsInt>(sol_undo_.size()) - 1; u >= sol_mark_target; --u) {
        solution_[sol_undo_[u].first] = sol_undo_[u].second;
    }
    sol_undo_.resize(sol_mark_target);
    // Restore row activities
    if (act_mark_target >= 0) {
        for (HighsInt u = static_cast<HighsInt>(act_undo_.size()) - 1; u >= act_mark_target; --u) {
            min_activity_[act_undo_[u].row] = act_undo_[u].old_min;
            max_activity_[act_undo_[u].row] = act_undo_[u].old_max;
        }
        act_undo_.resize(act_mark_target);
    }
    // Restore domain priority queue by replaying undo entries in reverse.
    // Each entry stores the exact PQ key before and after the change, so
    // we don't depend on vs_ state during replay.
    if (pq_mark_target >= 0) {
        for (HighsInt u = static_cast<HighsInt>(pq_undo_.size()) - 1; u >= pq_mark_target; --u) {
            auto& undo = pq_undo_[u];
            if (undo.is_present) {
                domain_pq_.erase({undo.new_dom, undo.var});
            }
            if (undo.was_present) {
                domain_pq_.insert({undo.old_dom, undo.var});
            }
        }
        pq_undo_.resize(pq_mark_target);
    }
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

HighsInt PropEngine::act_mark() const {
    return static_cast<HighsInt>(act_undo_.size());
}

void PropEngine::init_activities() {
    min_activity_.assign(nrow_, 0.0);
    max_activity_.assign(nrow_, 0.0);
    act_undo_.clear();
    act_undo_.reserve(4 * ncol_);  // grows with nnz touched during DFS

    for (HighsInt i = 0; i < nrow_; ++i) {
        double lo = 0.0, hi = 0.0;
        for (HighsInt k = ar_start_[i]; k < ar_start_[i + 1]; ++k) {
            HighsInt j = ar_index_[k];
            double a = ar_value_[k];
            if (vs_[j].fixed) {
                lo += a * vs_[j].val;
                hi += a * vs_[j].val;
            } else if (a > 0) {
                lo += a * vs_[j].lb;
                hi += a * vs_[j].ub;
            } else {
                lo += a * vs_[j].ub;
                hi += a * vs_[j].lb;
            }
        }
        min_activity_[i] = lo;
        max_activity_[i] = hi;
    }
}

void PropEngine::update_activities(HighsInt j, const VarState& old_vs) {
    if (min_activity_.empty()) {
        return;
    }

    const VarState& new_vs = vs_[j];

    for (HighsInt p = col_start_[j]; p < col_start_[j + 1]; ++p) {
        HighsInt i = col_row_[p];
        double a = col_val_[p];

        // Compute old contribution to [min, max] activity
        double old_lo, old_hi;
        if (old_vs.fixed) {
            old_lo = a * old_vs.val;
            old_hi = old_lo;
        } else if (a > 0) {
            old_lo = a * old_vs.lb;
            old_hi = a * old_vs.ub;
        } else {
            old_lo = a * old_vs.ub;
            old_hi = a * old_vs.lb;
        }

        // Compute new contribution
        double new_lo, new_hi;
        if (new_vs.fixed) {
            new_lo = a * new_vs.val;
            new_hi = new_lo;
        } else if (a > 0) {
            new_lo = a * new_vs.lb;
            new_hi = a * new_vs.ub;
        } else {
            new_lo = a * new_vs.ub;
            new_hi = a * new_vs.lb;
        }

        if (old_lo != new_lo || old_hi != new_hi) {
            act_undo_.push_back({i, min_activity_[i], max_activity_[i]});
            min_activity_[i] += (new_lo - old_lo);
            max_activity_[i] += (new_hi - old_hi);
        }
    }
}

void PropEngine::init_domain_pq() {
    domain_pq_.clear();
    pq_undo_.clear();
    pq_undo_.reserve(4 * ncol_);
    for (HighsInt j = 0; j < ncol_; ++j) {
        if (is_int(j) && !vs_[j].fixed) {
            domain_pq_.insert({vs_[j].ub - vs_[j].lb, j});
        }
    }
    pq_active_ = true;
}

HighsInt PropEngine::pq_top() const {
    if (domain_pq_.empty()) {
        return -1;
    }
    return domain_pq_.begin()->second;
}

HighsInt PropEngine::pq_mark() const {
    return static_cast<HighsInt>(pq_undo_.size());
}

void PropEngine::pq_notify(HighsInt j, const VarState& old_vs) {
    if (!pq_active_) {
        return;
    }
    if (!is_int(j)) {
        return;
    }

    double old_dom = old_vs.ub - old_vs.lb;
    bool was_present = !old_vs.fixed;
    bool is_present = !vs_[j].fixed;
    double new_dom = vs_[j].ub - vs_[j].lb;

    // Remove old entry
    if (was_present) {
        domain_pq_.erase({old_dom, j});
    }

    // Log undo with both old and new state
    pq_undo_.push_back({j, old_dom, new_dom, was_present, is_present});

    // Insert new entry if not fixed
    if (is_present) {
        domain_pq_.insert({new_dom, j});
    }
}
