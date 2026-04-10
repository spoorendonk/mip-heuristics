#include "repair_search.h"

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "prop_engine.h"
#include "walksat.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

// SyncChanges(E, R): transfer domain deductions from R to E (paper line 13).
bool sync_changes(PropEngine& E, const PropEngine& R) {
    bool any_seeded = false;
    for (HighsInt j = 0; j < E.ncol(); ++j) {
        if (E.var(j).fixed) {
            continue;
        }
        const auto& rv = R.var(j);
        const auto& ev = E.var(j);
        bool r_tightened_lb = rv.lb > ev.lb + E.feastol();
        bool r_tightened_ub = rv.ub < ev.ub - E.feastol();
        if (!r_tightened_lb && !r_tightened_ub) {
            continue;
        }

        if (rv.fixed || rv.ub - rv.lb < E.feastol()) {
            double val = rv.fixed ? rv.val : (rv.lb + rv.ub) * 0.5;
            if (!E.fix(j, val)) {
                return false;
            }
            E.seed_worklist(j);
            any_seeded = true;
        } else {
            if (r_tightened_lb) {
                if (!E.tighten_lb(j, rv.lb)) {
                    return false;
                }
                any_seeded = true;
            }
            if (r_tightened_ub) {
                if (!E.tighten_ub(j, rv.ub)) {
                    return false;
                }
                any_seeded = true;
            }
        }
    }
    if (any_seeded) {
        if (!E.propagate(-1)) {
            return false;
        }
    }
    return true;
}

// DFS node for repair disjunctions (paper Fig. 5).
struct RepairNode {
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

struct UndoEntry {
    HighsInt idx;
    double old_val;
};

// MoveToDisjunction (paper p.128).
struct Branch {
    HighsInt var;
    double val;
    bool is_fix;
    bool is_lb;
};

std::pair<Branch, Branch> move_to_disjunction(const PropEngine& E, const PropEngine& R,
                                              HighsInt var, double move_val) {
    double e_lb = E.var(var).lb;
    double e_ub = E.var(var).ub;

    // Binary: fix to move_val vs fix to 1-move_val
    if (e_lb == 0.0 && e_ub == 1.0 && E.is_int(var)) {
        double alt = (move_val < 0.5) ? 1.0 : 0.0;
        return {{var, move_val, true, false}, {var, alt, true, false}};
    }

    // Non-binary: gap-based split using R's propagated domain.
    // E domain [a, b], R domain [c, d].
    // Gaps: l = a - c (positive if R extended left), r = d - b (right).
    double left_gap = e_lb - R.var(var).lb;
    double right_gap = R.var(var).ub - e_ub;

    if (left_gap <= right_gap) {
        // Paper: x_j ≤ b ∨ x_j ≥ a
        return {{var, e_ub, false, false},  // tighten_ub — preferred
                {var, e_lb, false, true}};  // tighten_lb — alternative
    }
    // Paper: x_j ≤ a ∨ x_j ≥ b
    return {{var, e_lb, false, false},  // tighten_ub(a) — preferred
            {var, e_ub, false, true}};  // tighten_lb(b) — alternative
}

// BacktrackBestOpen: swap the lowest-violation node to the back of Q.
void backtrack_best_open(std::vector<RepairNode>& Q) {
    if (Q.empty()) {
        return;
    }
    auto best = std::min_element(Q.begin(), Q.end(), [](const RepairNode& a, const RepairNode& b) {
        return a.violation < b.violation;
    });
    if (best != Q.end() - 1) {
        std::iter_swap(best, Q.end() - 1);
    }
}

// Apply a branch to R: fix or tighten, then propagate.
// Returns false if infeasible.
bool apply_branch_to_r(PropEngine& R, const RepairNode& node) {
    if (node.var < 0) {
        return true;  // root node — no branch
    }
    bool ok;
    if (node.is_fix) {
        ok = R.fix(node.var, node.val);
        if (ok) {
            ok = R.propagate(node.var);
        }
    } else if (node.is_lb) {
        ok = R.tighten_lb(node.var, node.val);
        if (ok) {
            ok = R.propagate(-1);
        }
    } else {
        ok = R.tighten_ub(node.var, node.val);
        if (ok) {
            ok = R.propagate(-1);
        }
    }
    return ok;
}

}  // namespace

bool repair_search(PropEngine& E, std::vector<double>& solution, std::vector<double>& lhs_cache,
                   const double* col_lb, const double* col_ub, const double* row_lo,
                   const double* row_hi, HighsInt repair_iterations, double repair_noise,
                   bool repair_track_best, size_t max_effort, std::mt19937& rng,
                   size_t& effort_out) {
    const HighsInt ncol = E.ncol();
    const HighsInt nrow = E.nrow();
    const double feastol = E.feastol();
    const HighsInt* csc_start = E.csc_start();
    const HighsInt* csc_row = E.csc_row();
    const double* csc_val = E.csc_val();

    auto viol = [&](HighsInt i, double lhs) -> double {
        return row_violation(lhs, row_lo[i], row_hi[i]);
    };
    auto is_violated = [&](HighsInt i, double lhs) -> bool {
        return lhs > row_hi[i] + feastol || lhs < row_lo[i] - feastol;
    };

    // --- Violated set ---
    std::vector<HighsInt> violated;
    std::vector<HighsInt> violated_pos(nrow, -1);
    violated.reserve(nrow);

    auto add_violated = [&](HighsInt i) {
        if (violated_pos[i] != -1) {
            return;
        }
        violated_pos[i] = static_cast<HighsInt>(violated.size());
        violated.push_back(i);
    };
    auto remove_violated = [&](HighsInt i) {
        HighsInt pos = violated_pos[i];
        if (pos == -1) {
            return;
        }
        HighsInt last = violated.back();
        violated[pos] = last;
        violated_pos[last] = pos;
        violated.pop_back();
        violated_pos[i] = -1;
    };
    auto rebuild_violated = [&]() {
        for (auto vi : violated) {
            violated_pos[vi] = -1;
        }
        violated.clear();
        for (HighsInt i = 0; i < nrow; ++i) {
            if (is_violated(i, lhs_cache[i])) {
                add_violated(i);
            }
        }
    };

    // --- Initialize total violation and violated set ---
    double total_viol = 0.0;
    for (HighsInt i = 0; i < nrow; ++i) {
        total_viol += viol(i, lhs_cache[i]);
    }
    rebuild_violated();
    if (violated.empty()) {
        effort_out = 0;
        return true;
    }

    // --- Solution/LHS undo stacks ---
    std::vector<UndoEntry> sol_undo;
    std::vector<UndoEntry> lhs_undo;

    // apply_move updates solution, lhs_cache, total_viol, and violated set
    // incrementally (O(column_degree) instead of O(nrow)).
    auto apply_move = [&](HighsInt var, double new_val, size_t& effort) {
        double old_val = solution[var];
        sol_undo.push_back({var, old_val});
        solution[var] = new_val;
        double delta = new_val - old_val;
        effort += csc_start[var + 1] - csc_start[var];
        for (HighsInt p = csc_start[var]; p < csc_start[var + 1]; ++p) {
            HighsInt row = csc_row[p];
            double old_v = row_violation(lhs_cache[row], row_lo[row], row_hi[row]);
            lhs_undo.push_back({row, lhs_cache[row]});
            lhs_cache[row] += csc_val[p] * delta;
            double new_v = row_violation(lhs_cache[row], row_lo[row], row_hi[row]);
            total_viol += new_v - old_v;

            bool was = violated_pos[row] != -1;
            bool now = is_row_violated(lhs_cache[row], row_lo[row], row_hi[row], feastol);
            if (was && !now) {
                remove_violated(row);
            } else if (!was && now) {
                add_violated(row);
            }
        }
    };

    auto backtrack_sol_lhs = [&](HighsInt s_mark, HighsInt l_mark) {
        for (HighsInt u = static_cast<HighsInt>(sol_undo.size()) - 1; u >= s_mark; --u) {
            solution[sol_undo[u].idx] = sol_undo[u].old_val;
        }
        sol_undo.resize(s_mark);
        for (HighsInt u = static_cast<HighsInt>(lhs_undo.size()) - 1; u >= l_mark; --u) {
            lhs_cache[lhs_undo[u].idx] = lhs_undo[u].old_val;
        }
        lhs_undo.resize(l_mark);
    };
    double best_viol = total_viol;
    std::vector<double> best_solution;
    std::vector<double> best_lhs;
    if (repair_track_best) {
        best_solution = solution;
        best_lhs = lhs_cache;
    }

    // --- Secondary engine R from global bounds ---
    PropEngine R(ncol, nrow, E.ar_start(), E.ar_index(), E.ar_value(), E.csc_start(), E.csc_row(),
                 E.csc_val(), col_lb, col_ub, row_lo, row_hi, E.integrality(), feastol);

    size_t total_effort = 0;
    size_t e_effort_baseline = E.effort();
    size_t r_effort_baseline = R.effort();
    WalkSatScratch scratch;
    HighsInt nodes_without_progress = 0;
    constexpr HighsInt kProgressThreshold = 10;

    // --- DFS stack (paper Fig. 5, lines 3-4) ---
    std::vector<RepairNode> Q;
    Q.push_back({-1, 0.0, true, false, E.vs_mark(), E.sol_mark(), R.vs_mark(), R.sol_mark(), 0, 0,
                 total_viol});

    HighsInt nodes_visited = 0;

    while (!Q.empty() && nodes_visited < repair_iterations && total_effort < max_effort) {
        RepairNode node = Q.back();
        Q.pop_back();
        ++nodes_visited;

        // Restore parent state (paper lines 7-8)
        backtrack_sol_lhs(node.sol_undo_mark, node.lhs_undo_mark);
        E.backtrack_to(node.e_vs_mark, node.e_sol_mark);
        R.backtrack_to(node.r_vs_mark, node.r_sol_mark);
        total_viol = node.violation;
        rebuild_violated();

        // Apply branch to R, propagate (paper lines 8-10)
        if (!apply_branch_to_r(R, node)) {
            continue;  // infeasible — prune (paper lines 11-12)
        }

        // SyncChanges R→E (paper line 13)
        if (!sync_changes(E, R)) {
            continue;  // E infeasible after sync
        }

        // Apply branch to solution/lhs (our extension for complete-assignment)
        // apply_move updates total_viol and violated set incrementally.
        if (node.var >= 0) {
            if (node.is_fix) {
                apply_move(node.var, node.val, total_effort);
            } else {
                double cur = solution[node.var];
                double new_lb = E.var(node.var).lb;
                double new_ub = E.var(node.var).ub;
                double clamped = std::max(new_lb, std::min(new_ub, cur));
                if (std::abs(clamped - cur) > feastol) {
                    apply_move(node.var, clamped, total_effort);
                }
            }
        }

        if (violated.empty()) {
            // Feasible! (paper lines 15-16)
            effort_out =
                total_effort + (E.effort() - e_effort_baseline) + (R.effort() - r_effort_baseline);
            return true;
        }

        // Update best state (paper line 17)
        if (total_viol < best_viol - feastol) {
            best_viol = total_viol;
            if (repair_track_best) {
                best_solution = solution;
                best_lhs = lhs_cache;
            }
            nodes_without_progress = 0;
        } else {
            ++nodes_without_progress;
        }

        // Check progress — jump to best open node if stuck (paper lines 18-19)
        if (nodes_without_progress >= kProgressThreshold && !Q.empty()) {
            backtrack_best_open(Q);
            nodes_without_progress = 0;
        }

        // FindRepairMove: WalkSAT on current solution (paper line 20)
        HighsInt pick = std::uniform_int_distribution<HighsInt>(
            0, static_cast<HighsInt>(violated.size()) - 1)(rng);
        HighsInt row = violated[pick];

        auto move = walksat_select_move(row, solution.data(), lhs_cache.data(), col_lb, col_ub, E,
                                        repair_noise, rng, total_effort, scratch);
        if (move.var < 0) {
            continue;  // no valid move (paper lines 21-22)
        }

        // MoveToDisjunction (paper lines 24-26)
        auto [preferred, alternative] = move_to_disjunction(E, R, move.var, move.val);

        // Save current state marks
        HighsInt cur_e_vs = E.vs_mark();
        HighsInt cur_e_sol = E.sol_mark();
        HighsInt cur_r_vs = R.vs_mark();
        HighsInt cur_r_sol = R.sol_mark();
        HighsInt cur_sol = static_cast<HighsInt>(sol_undo.size());
        HighsInt cur_lhs = static_cast<HighsInt>(lhs_undo.size());

        // Push alternative first (explored second), then preferred (explored first)
        Q.push_back({alternative.var, alternative.val, alternative.is_fix, alternative.is_lb,
                     cur_e_vs, cur_e_sol, cur_r_vs, cur_r_sol, cur_sol, cur_lhs, total_viol});
        Q.push_back({preferred.var, preferred.val, preferred.is_fix, preferred.is_lb, cur_e_vs,
                     cur_e_sol, cur_r_vs, cur_r_sol, cur_sol, cur_lhs, total_viol});

        // Best-first steering (paper line 27)
        backtrack_best_open(Q);
    }

    // Restore best state (paper line 28)
    if (repair_track_best && best_viol < total_viol) {
        solution = best_solution;
        lhs_cache = best_lhs;
        rebuild_violated();
    }

    effort_out = total_effort + (E.effort() - e_effort_baseline) + (R.effort() - r_effort_baseline);
    return violated.empty();
}
