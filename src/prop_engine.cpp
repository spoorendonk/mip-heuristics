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
    // DFS + propagation can push many more than ncol undo entries (every
    // bound tightening pushes one). Over-reserve to reduce reallocation
    // churn in the hot path; the memory cost is bounded and transient.
    vs_undo_.reserve(static_cast<size_t>(4) * static_cast<size_t>(ncol));
    sol_undo_.reserve(static_cast<size_t>(4) * static_cast<size_t>(ncol));
    prop_worklist_.reserve(nrow);
    // Preallocate domain-PQ storage (vector-backed indexed heap replaces
    // the prior std::set; reserve here so subsequent reset() cycles reuse
    // this allocation).
    domain_pq_.reserve(ncol);
    reset();
}

// --- IndexedMinHeap implementation -----------------------------------------
void IndexedMinHeap::reserve(HighsInt ncol) {
    pos_.assign(ncol, kNotPresent);
    heap_.reserve(ncol);
}

void IndexedMinHeap::clear() {
    for (const auto& entry : heap_) {
        pos_[entry.var] = kNotPresent;
    }
    heap_.clear();
}

void IndexedMinHeap::insert(double key, HighsInt var) {
    const auto idx = static_cast<HighsInt>(heap_.size());
    heap_.push_back({key, var});
    pos_[var] = idx;
    sift_up(idx);
}

void IndexedMinHeap::erase(HighsInt var) {
    const HighsInt idx = pos_[var];
    const auto last_idx = static_cast<HighsInt>(heap_.size()) - 1;
    pos_[var] = kNotPresent;
    if (idx == last_idx) {
        heap_.pop_back();
        return;
    }
    // Move last entry into the hole, then re-heapify from there.
    heap_[idx] = heap_[last_idx];
    pos_[heap_[idx].var] = idx;
    heap_.pop_back();
    // The moved entry might be smaller than its new parent or larger than a
    // child — try both directions (at most one will do work).
    if (idx > 0 && entry_less(heap_[idx], heap_[(idx - 1) / 2])) {
        sift_up(idx);
    } else {
        sift_down(idx);
    }
}

void IndexedMinHeap::update(HighsInt var, double new_key) {
    const HighsInt idx = pos_[var];
    const double old_key = heap_[idx].key;
    heap_[idx].key = new_key;
    if (new_key < old_key) {
        sift_up(idx);
    } else if (new_key > old_key) {
        sift_down(idx);
    }
    // Exact tie on key: position still satisfies heap property (tiebreak by
    // var is unchanged), so no sift needed.
}

void IndexedMinHeap::sift_up(HighsInt idx) {
    const Entry entry = heap_[idx];
    while (idx > 0) {
        const HighsInt parent = (idx - 1) / 2;
        if (!entry_less(entry, heap_[parent])) {
            break;
        }
        heap_[idx] = heap_[parent];
        pos_[heap_[idx].var] = idx;
        idx = parent;
    }
    heap_[idx] = entry;
    pos_[entry.var] = idx;
}

void IndexedMinHeap::sift_down(HighsInt idx) {
    const Entry entry = heap_[idx];
    const auto size = static_cast<HighsInt>(heap_.size());
    while (true) {
        const HighsInt left = 2 * idx + 1;
        if (left >= size) {
            break;
        }
        const HighsInt right = left + 1;
        HighsInt smallest = left;
        if (right < size && entry_less(heap_[right], heap_[left])) {
            smallest = right;
        }
        if (!entry_less(heap_[smallest], entry)) {
            break;
        }
        heap_[idx] = heap_[smallest];
        pos_[heap_[idx].var] = idx;
        idx = smallest;
    }
    heap_[idx] = entry;
    pos_[entry.var] = idx;
}
// --- end IndexedMinHeap implementation -------------------------------------

void PropEngine::reset() {
    // Reset variable states to global bounds and solution values to 0. We
    // re-derive from col_lb_/col_ub_ (the problem-data pointers, unchanged
    // across reset) rather than replaying vs_undo_ because the previous
    // attempt may have finished on a non-backtracked path (e.g. found a
    // complete assignment) and the undo stacks do not fully unwind in that
    // case.
    for (HighsInt j = 0; j < ncol_; ++j) {
        vs_[j].lb = col_lb_[j];
        vs_[j].ub = col_ub_[j];
        vs_[j].val = 0.0;
        vs_[j].fixed = false;
        solution_[j] = 0.0;
    }
    vs_undo_.clear();
    sol_undo_.clear();
    // Clear worklist membership flags that might still be set from the
    // previous attempt's propagation. Iterating prop_worklist_ is cheapest
    // when the worklist is non-empty; if it's already empty this is a
    // no-op.
    for (HighsInt i : prop_worklist_) {
        prop_in_wl_[i] = 0;
    }
    prop_worklist_.clear();
    // Activity tracking is opt-in (caller must invoke init_activities() if
    // needed). Dropping the size signals "not initialized" to downstream
    // code via activities_initialized(). Capacity is preserved so the next
    // init_activities() call does not reallocate.
    min_activity_.clear();
    max_activity_.clear();
    act_undo_.clear();
    pq_active_ = false;
    domain_pq_.clear();
    pq_undo_.clear();
    // Zero the accumulated propagation effort so the Phase 1-2 DFS gate
    // (E.effort() < cfg.max_effort) starts each attempt fresh. Without
    // this, a second fpr_attempt call on the same PropEngine would see
    // effort already at or above the previous attempt's total and exit
    // the DFS loop immediately.
    prop_work_ = 0;
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
    const HighsInt kbeg = col_start_[j];
    const HighsInt kend = col_start_[j + 1];
    const HighsInt* __restrict col_row = col_row_;
    char* __restrict in_wl = prop_in_wl_.data();
    for (HighsInt p = kbeg; p < kend; ++p) {
        HighsInt i = col_row[p];
        if (!in_wl[i]) {
            in_wl[i] = 1;
            prop_worklist_.push_back(i);
        }
    }
}

bool PropEngine::propagate(HighsInt fixed_var) {
    // Hoist raw pointers once so the tight loop doesn't re-load member
    // fields on every iteration. All of these arrays are read-only (the
    // problem data) except vs_/solution_, which we mutate via vs_data and
    // also push undo entries for.
    const HighsInt* __restrict ar_start = ar_start_;
    const HighsInt* __restrict ar_index = ar_index_;
    const double* __restrict ar_value = ar_value_;
    const double* __restrict col_lb = col_lb_;
    const double* __restrict col_ub = col_ub_;
    const double feastol = feastol_;
    VarState* __restrict vs_data = vs_.data();
    char* __restrict in_wl = prop_in_wl_.data();

    if (fixed_var >= 0) {
        // Seed only the rows containing the just-fixed variable (AC-3).
        // Clear stale prop_in_wl_ flags before clearing the worklist.
        for (HighsInt wi : prop_worklist_) {
            in_wl[wi] = 0;
        }
        prop_worklist_.clear();
        const HighsInt cbeg = col_start_[fixed_var];
        const HighsInt cend = col_start_[fixed_var + 1];
        const HighsInt* __restrict col_row = col_row_;
        for (HighsInt p = cbeg; p < cend; ++p) {
            HighsInt i = col_row[p];
            if (!in_wl[i]) {
                in_wl[i] = 1;
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
        in_wl[i] = 0;
        const HighsInt kbeg = ar_start[i];
        const HighsInt kend = ar_start[i + 1];
        prop_work += static_cast<size_t>(kend - kbeg);
        if (prop_work > prop_budget) {
            prop_work_ += prop_work;
            // Clear stale worklist markers
            for (HighsInt wi : prop_worklist_) {
                in_wl[wi] = 0;
            }
            prop_worklist_.clear();
            return false;
        }

        // Pass 1: compute row aggregates (fixed_sum, min_act, max_act) and
        // count unfixed entries. Pull the coefficient, index, and VarState
        // into locals to keep the inner loop read-dominated.
        double fixed_sum = 0.0;
        double min_act = 0.0;
        double max_act = 0.0;
        HighsInt num_unfixed = 0;

        for (HighsInt k = kbeg; k < kend; ++k) {
            const HighsInt j = ar_index[k];
            const double a = ar_value[k];
            const VarState& vj = vs_data[j];
            if (vj.fixed) {
                fixed_sum += a * vj.val;
            } else {
                ++num_unfixed;
                if (a > 0) {
                    min_act += a * vj.lb;
                    max_act += a * vj.ub;
                } else {
                    min_act += a * vj.ub;
                    max_act += a * vj.lb;
                }
            }
        }

        if (num_unfixed == 0) {
            continue;
        }

        const bool has_upper = row_hi_[i] < kHighsInf;
        const bool has_lower = row_lo_[i] > -kHighsInf;
        const double row_hi_i = row_hi_[i];
        const double row_lo_i = row_lo_[i];

        // Pass 2: tighten bounds on each unfixed variable. We re-access
        // vs_data[j] here because the first pass only reads; we need a
        // mutable reference for bound tightening.
        for (HighsInt k = kbeg; k < kend; ++k) {
            const HighsInt j = ar_index[k];
            VarState& vj = vs_data[j];
            if (vj.fixed) {
                continue;
            }
            const double a = ar_value[k];
            if (std::abs(a) < 1e-15) {
                continue;
            }

            const double old_lb = vj.lb;
            const double old_ub = vj.ub;
            const bool is_integer = integrality_[j] != HighsVarType::kContinuous;

            double min_others;
            double max_others;
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
                const double bound = row_hi_i - fixed_sum - min_others;
                if (a > 0) {
                    new_ub = std::min(new_ub, bound / a);
                } else {
                    new_lb = std::max(new_lb, bound / a);
                }
            }

            if (has_lower) {
                const double bound = row_lo_i - fixed_sum - max_others;
                if (a > 0) {
                    new_lb = std::max(new_lb, bound / a);
                } else {
                    new_ub = std::min(new_ub, bound / a);
                }
            }

            if (is_integer) {
                new_lb = std::ceil(new_lb - feastol);
                new_ub = std::floor(new_ub + feastol);
            }

            new_lb = std::max(new_lb, col_lb[j]);
            new_ub = std::min(new_ub, col_ub[j]);

            if (new_lb > new_ub + feastol) {
                prop_work_ += prop_work;
                // Clear stale worklist markers on infeasibility too
                for (HighsInt wi : prop_worklist_) {
                    in_wl[wi] = 0;
                }
                prop_worklist_.clear();
                return false;
            }

            const bool tighter_lb = new_lb > old_lb + feastol;
            const bool tighter_ub = new_ub < old_ub - feastol;
            bool changed = false;
            VarState pre_change_vs = vj;

            if (tighter_lb || tighter_ub) {
                vs_undo_.push_back({j, pre_change_vs});
                sol_undo_.push_back({j, solution_[j]});
                if (tighter_lb) {
                    vj.lb = new_lb;
                }
                if (tighter_ub) {
                    vj.ub = new_ub;
                }
                changed = true;
            }

            if (!vj.fixed && vj.ub - vj.lb < feastol) {
                if (!changed) {
                    vs_undo_.push_back({j, pre_change_vs});
                    sol_undo_.push_back({j, solution_[j]});
                }
                double val = (vj.lb + vj.ub) * 0.5;
                if (is_integer) {
                    val = std::round(val);
                }
                vj.fixed = true;
                vj.val = val;
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
    // we don't depend on vs_ state during replay.  Collapse the
    // erase-then-insert pair into a single update() when both old and new
    // states had the var in the PQ (the common bound-tightening case).
    if (pq_mark_target >= 0) {
        for (HighsInt u = static_cast<HighsInt>(pq_undo_.size()) - 1; u >= pq_mark_target; --u) {
            auto& undo = pq_undo_[u];
            if (undo.is_present && undo.was_present) {
                if (undo.old_dom != undo.new_dom) {
                    domain_pq_.update(undo.var, undo.old_dom);
                }
            } else if (undo.is_present) {
                domain_pq_.erase(undo.var);
            } else if (undo.was_present) {
                domain_pq_.insert(undo.old_dom, undo.var);
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
    act_undo_.reserve(static_cast<size_t>(4) *
                      static_cast<size_t>(ncol_));  // grows with nnz touched during DFS

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
    const HighsInt kbeg = col_start_[j];
    const HighsInt kend = col_start_[j + 1];
    const HighsInt* __restrict col_row = col_row_;
    const double* __restrict col_val = col_val_;
    double* __restrict min_a = min_activity_.data();
    double* __restrict max_a = max_activity_.data();

    for (HighsInt p = kbeg; p < kend; ++p) {
        const HighsInt i = col_row[p];
        const double a = col_val[p];

        // Compute old contribution to [min, max] activity
        double old_lo;
        double old_hi;
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
        double new_lo;
        double new_hi;
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

        // Use component comparisons rather than (delta != 0) so that
        // infinite contributions (e.g. a * -inf when a bound is -kHighsInf)
        // compare equal to themselves and skip the update; the delta would
        // be NaN in that case and silently corrupt the activity.
        if (old_lo != new_lo || old_hi != new_hi) {
            act_undo_.push_back({i, min_a[i], max_a[i]});
            min_a[i] += (new_lo - old_lo);
            max_a[i] += (new_hi - old_hi);
        }
    }
}

void PropEngine::init_domain_pq() {
    domain_pq_.clear();
    pq_undo_.clear();
    pq_undo_.reserve(4 * ncol_);
    for (HighsInt j = 0; j < ncol_; ++j) {
        if (is_int(j) && !vs_[j].fixed) {
            domain_pq_.insert(vs_[j].ub - vs_[j].lb, j);
        }
    }
    pq_active_ = true;
}

HighsInt PropEngine::pq_top() const {
    if (domain_pq_.empty()) {
        return -1;
    }
    return domain_pq_.top_var();
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

    // Log undo with both old and new state
    pq_undo_.push_back({j, old_dom, new_dom, was_present, is_present});

    // Apply the transition.  `update(var, new_key)` handles the common
    // "still in PQ but key changed" case without the erase+insert pair
    // the std::set version required.
    if (was_present && is_present) {
        if (old_dom != new_dom) {
            domain_pq_.update(j, new_dom);
        }
    } else if (was_present) {
        domain_pq_.erase(j);
    } else if (is_present) {
        domain_pq_.insert(new_dom, j);
    }
}
