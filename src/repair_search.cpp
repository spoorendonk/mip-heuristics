#include "repair_search.h"

#include "fpr_core.h"
#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "prop_engine.h"
#include "walksat.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

// SyncChanges(E, R): transfer domain deductions from R to E (paper Fig. 5
// line 13, Sect. 5.1).  Exposed (not in the anonymous namespace below,
// declared in repair_search.h) for direct unit testing (issue #125) --
// this is the paper's own named function and every other caller reaches it
// only through repair_search().
//
// Sect. 5.1's case analysis, unified across the binary/non-binary split the
// prose makes (verified equivalent -- see the two dedicated tests in
// tests/test_repair_search.cpp):
//
//   - E already fixed to `ev.val`: if R's domain still contains `ev.val`
//     the two domains agree and there is nothing to do ("leave it as
//     is"); otherwise R has ruled `ev.val` out (e.g. clique propagation
//     on a repair move), so E is *re-fixed* to R's domain -- a flip, in
//     the binary case, and its direct generalization (nearest reachable
//     point) otherwise.  This is the case the pre-#125 code skipped
//     entirely (`if (E.var(j).fixed) continue;`), making the paper's
//     motivating binary swap unreachable.
//   - E unfixed, R's domain a superset of E's (`Dr ⊇ D`): nothing to
//     sync.
//   - E unfixed, non-empty intersection: tighten D to D ∩ Dr (paper case
//     1).
//   - E unfixed, disjoint intervals: fix to the endpoint of Dr closer to
//     D (paper case 2; worked example D=[1,3], Dr=[4,5] -> 4).  This is
//     the case the pre-#125 code got wrong by calling `tighten_lb`/
//     `tighten_ub`, which validates against E's *current* bounds and
//     therefore always rejects a value outside them -- the very
//     definition of "disjoint" -- so `sync_changes` returned false and
//     the node was dropped instead of repaired.
//
// The two "re-fix / fix past the current domain" branches use
// `PropEngine::refix`, not `fix()`: both target a value outside E's
// *current* domain by construction (that is what makes them the
// override cases rather than the refine cases), and `fix()` validates
// against exactly that current domain.  See `refix`'s own doc comment.
//
// Cognitive complexity 34 (threshold 25).  Kept whole: SyncChanges(E, R) from Fig. 5 line 13, the
// domain-transfer case analysis unifying the binary and non-binary cases the paper's prose splits.
// Decomposing it would move work across a worker's inner loop, and the closeout takes no unmeasured
// performance risk; the standards also rank fidelity to the reference algorithm above mechanical
// extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool sync_changes(PropEngine& E, const PropEngine& R) {
    bool any_seeded = false;
    const double feastol = E.feastol();
    for (HighsInt j = 0; j < E.ncol(); ++j) {
        const auto& rv = R.var(j);
        const auto& ev = E.var(j);
        // R's effective domain for j: a single point if R has fixed it
        // (regardless of whether R's own vs_[j].lb/ub were narrowed --
        // R.fix() does not narrow them, so the raw bounds can understate
        // what R has actually decided).
        const double r_lb = rv.fixed ? rv.val : rv.lb;
        const double r_ub = rv.fixed ? rv.val : rv.ub;

        if (ev.fixed) {
            if (ev.val >= r_lb - feastol && ev.val <= r_ub + feastol) {
                continue;  // domains agree -- leave as is
            }
            const double target = (ev.val < r_lb) ? r_lb : r_ub;
            if (!E.refix(j, target)) {
                return false;
            }
            E.seed_worklist(j);
            any_seeded = true;
            continue;
        }

        // E unfixed.  Dr ⊇ D means R hasn't restricted this column at
        // all relative to what E already knows -- nothing to sync.
        if (r_lb <= ev.lb + feastol && r_ub >= ev.ub - feastol) {
            continue;
        }

        const double new_lb = std::max(ev.lb, r_lb);
        const double new_ub = std::min(ev.ub, r_ub);
        if (new_lb <= new_ub + feastol) {
            // Case 1: non-empty intersection -- tighten D to D ∩ Dr.
            // `new_lb`/`new_ub` are within E's current [ev.lb, ev.ub] by
            // construction, so `fix()`'s current-domain validation is the
            // right (and sufficient) check here.
            if (new_ub - new_lb < feastol) {
                if (!E.fix(j, (new_lb + new_ub) * 0.5)) {
                    return false;
                }
                E.seed_worklist(j);
            } else {
                if (new_lb > ev.lb + feastol && !E.tighten_lb(j, new_lb)) {
                    return false;
                }
                if (new_ub < ev.ub - feastol && !E.tighten_ub(j, new_ub)) {
                    return false;
                }
            }
            any_seeded = true;
        } else {
            // Case 2: disjoint intervals -- fix to the endpoint of Dr
            // closer to D.  The target is outside E's current domain by
            // definition, hence `refix`.
            const double target = (r_lb > ev.ub) ? r_lb : r_ub;
            if (!E.refix(j, target)) {
                return false;
            }
            E.seed_worklist(j);
            any_seeded = true;
        }
    }
    if (any_seeded) {
        // Budget exhaustion (issue #127) is sound-but-incomplete, not a
        // verdict; only a proven inconsistency fails the sync.
        if (E.propagate(-1) == PropResult::kInfeasible) {
            return false;
        }
    }
    return true;
}

namespace {

// MoveToDisjunction (paper p.128).
struct Branch {
    HighsInt var;
    double val;
    bool is_fix;
    bool is_lb;
};

// Known limitation (#131, discovered while implementing #125, not fixed
// here -- out of both issues' stated scope): the binary detection below
// is `[lb, ub] == [0, 1]`, which degenerates once a binary column's
// domain has been narrowed to a singleton -- by ordinary AC-3 auto-fix
// (`tighten_lb`/`tighten_ub` in prop_engine.cpp) or, since #125, by
// `PropEngine::refix`, which produces the identical narrow shape by
// design (see its header comment in prop_engine.h). Such a column falls
// through to the non-binary gap-split path below and produces a vacuous
// `x <= v \/ x >= v` disjunction instead of an actual flip choice --
// gating the production binary-swap pipeline one level further up than
// the `sync_changes` fix addresses.
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
void backtrack_best_open(std::vector<RepairSearchNode>& Q) {
    if (Q.empty()) {
        return;
    }
    auto best =
        std::ranges::min_element(Q, [](const RepairSearchNode& a, const RepairSearchNode& b) {
            return a.violation < b.violation;
        });
    if (best != Q.end() - 1) {
        std::iter_swap(best, Q.end() - 1);
    }
}

// Apply a branch to R: fix or tighten, then propagate.
// Returns false only on a proven inconsistency; budget exhaustion
// (issue #127) leaves R sound but incomplete and does not fail the branch.
bool apply_branch_to_r(PropEngine& R, const RepairSearchNode& node) {
    if (node.var < 0) {
        return true;  // root node — no branch
    }
    bool ok;
    PropResult pr = PropResult::kFixpoint;
    if (node.is_fix) {
        ok = R.fix(node.var, node.val);
        if (ok) {
            pr = R.propagate(node.var);
        }
    } else if (node.is_lb) {
        ok = R.tighten_lb(node.var, node.val);
        if (ok) {
            pr = R.propagate(-1);
        }
    } else {
        ok = R.tighten_ub(node.var, node.val);
        if (ok) {
            pr = R.propagate(-1);
        }
    }
    ok = ok && pr != PropResult::kInfeasible;
    return ok;
}

}  // namespace

// Cognitive complexity 68 (threshold 25).  Kept whole: RepairSearch from Fig. 5 in full, including
// the secondary backtracks and the e_pq_mark threading that keeps E's domain PQ consistent.
// Decomposing it would move work across a worker's inner loop, and the
// closeout takes no unmeasured performance risk; the standards also rank
// fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool repair_search(PropEngine& E, std::vector<double>& solution, std::vector<double>& lhs_cache,
                   const double* col_lb, const double* col_ub, const double* row_lo,
                   const double* row_hi, HighsInt repair_iterations, double repair_noise,
                   bool repair_track_best, size_t max_effort, Rng& rng, size_t& effort_out,
                   FprScratch& scratch, const Deadline& deadline) {
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

    // --- Violated set (reuses WalkSatScratch buffers; never runs concurrently
    // with walksat_repair — the two functions are alternative Phase 3 paths
    // at the same call site in fpr_attempt). ---
    auto& violated = scratch.walksat.violated;
    auto& violated_pos = scratch.walksat.violated_pos;
    violated.clear();
    if (std::cmp_less(violated_pos.size(), nrow)) {
        violated_pos.assign(nrow, -1);
    } else {
        std::fill(violated_pos.begin(), violated_pos.begin() + nrow, -1);
    }
    if (violated.capacity() < static_cast<size_t>(nrow)) {
        violated.reserve(nrow);
    }

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

    // --- Solution/LHS undo stacks (reuse WalkSatScratch buffers) ---
    auto& sol_undo = scratch.walksat.sol_undo;
    auto& lhs_undo = scratch.walksat.lhs_undo;
    sol_undo.clear();
    lhs_undo.clear();

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
    auto& best_solution = scratch.repair_best_solution;
    auto& best_lhs = scratch.repair_best_lhs;
    if (repair_track_best) {
        best_solution.assign(solution.begin(), solution.end());
        best_lhs.assign(lhs_cache.begin(), lhs_cache.end());
    } else {
        best_solution.clear();
        best_lhs.clear();
    }

    // --- Secondary engine R from global bounds ---
    // Reused across calls via FprScratch to eliminate the per-call allocation
    // (vs_/solution_/prop_in_wl_/undo reserves) that mirrors the primary
    // engine's reuse pattern.  Guard: R is always constructed from E's
    // problem-data pointers, so any cached R built against a prior E is
    // safe to reuse iff every pointer E exposes still matches.  Since E
    // is itself pooled with a pointer-identity guard in fpr_attempt, the
    // only way to invalidate R mid-solve is to swap E's underlying
    // problem data — which implies fpr_attempt already re-emplaced E;
    // in that case R's guard below also mismatches and R is re-emplaced.
    auto& engine_r_opt = scratch.repair_prop_engine_r;
    const bool r_valid =
        engine_r_opt.has_value() && engine_r_opt->ncol() == ncol && engine_r_opt->nrow() == nrow &&
        engine_r_opt->ar_start() == E.ar_start() && engine_r_opt->ar_index() == E.ar_index() &&
        engine_r_opt->ar_value() == E.ar_value() && engine_r_opt->csc_start() == E.csc_start() &&
        engine_r_opt->csc_row() == E.csc_row() && engine_r_opt->csc_val() == E.csc_val() &&
        engine_r_opt->col_lb() == col_lb && engine_r_opt->col_ub() == col_ub &&
        engine_r_opt->row_lo() == row_lo && engine_r_opt->row_hi() == row_hi &&
        engine_r_opt->integrality() == E.integrality() && engine_r_opt->feastol() == feastol;
    if (r_valid) {
        engine_r_opt->reset();
    } else {
        engine_r_opt.emplace(ncol, nrow, E.ar_start(), E.ar_index(), E.ar_value(), E.csc_start(),
                             E.csc_row(), E.csc_val(), col_lb, col_ub, row_lo, row_hi,
                             E.integrality(), feastol);
    }
    // NOLINT rationale: `R` is the paper's own symbol for this object
    // (Fig. 5), used under that name in the surrounding prose in
    // fpr_core.h, repair_search.h and fpr_strategies.h, and — for the
    // primary engine — as the parameter name of repair_search() itself.
    // Lower-casing only the locals would split one symbol across two
    // spellings; renaming the documentation too would cost the mapping
    // back to the paper, which the standards rank above naming.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine& R = *engine_r_opt;

    size_t total_effort = 0;
    size_t e_effort_baseline = E.effort();
    size_t r_effort_baseline = R.effort();
    HighsInt nodes_without_progress = 0;
    constexpr HighsInt kProgressThreshold = 10;

    // --- DFS stack (paper Fig. 5, lines 3-4).  Reused across calls via
    // scratch to avoid per-call allocations. ---
    // NOLINT rationale: `Q` is the paper's own symbol for this object
    // (Fig. 5), used under that name in the surrounding prose in
    // fpr_core.h, repair_search.h and fpr_strategies.h, and — for the
    // primary engine — as the parameter name of repair_search() itself.
    // Lower-casing only the locals would split one symbol across two
    // spellings; renaming the documentation too would cost the mapping
    // back to the paper, which the standards rank above naming.
    // NOLINTNEXTLINE(readability-identifier-naming)
    auto& Q = scratch.repair_dfs_stack;
    Q.clear();
    const HighsInt root_e_pq = E.pq_initialized() ? E.pq_mark() : -1;
    Q.push_back({-1, 0.0, true, false, E.vs_mark(), E.sol_mark(), root_e_pq, R.vs_mark(),
                 R.sol_mark(), 0, 0, total_viol});

    HighsInt nodes_visited = 0;

    // The budget gate must include PropEngine propagation effort; `total_effort`
    // only tracks WalkSAT move counters, so without the `E.effort()`/`R.effort()`
    // deltas the AC-3 fixpoints below (one per node on `R` in apply_branch_to_r,
    // one on `E` in sync_changes) would be unbounded.  The same compound sum is
    // charged to `effort_out` below, so guard and report stay consistent.
    auto effort_spent = [&]() -> size_t {
        return total_effort + (E.effort() - e_effort_baseline) + (R.effort() - r_effort_baseline);
    };
    // The wall clock joins the node and effort gates (issue #117).  Polled
    // on every node rather than on a cadence: `repair_iterations` is 50, so
    // fifty clock reads is the whole cost, while one node is two
    // propagation fixpoints — the reason this loop can outlive a time limit
    // at all.
    while (!Q.empty() && nodes_visited < repair_iterations && effort_spent() < max_effort &&
           !deadline.expired()) {
        RepairSearchNode node = Q.back();
        Q.pop_back();
        ++nodes_visited;

        // Restore parent state (paper lines 7-8).  Pass `node.e_pq_mark`
        // explicitly: when E was `init_domain_pq`'d in Phase 2 (any
        // dynamic-var strategy), omitting the PQ undo target here leaves
        // E's heap inconsistent with vs_ across backtracks — a subsequent
        // `pq_notify` triggered by the new branch's E.fix/tighten then
        // erases a var that's no longer in the heap.  R has no PQ active
        // (repair_search never calls init_domain_pq on R), so the default
        // -1 there is harmless.
        backtrack_sol_lhs(node.sol_undo_mark, node.lhs_undo_mark);
        E.backtrack_to(node.e_vs_mark, node.e_sol_mark, /*act_mark=*/-1, node.e_pq_mark);
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
            effort_out = effort_spent();
            return true;
        }

        // Update best state (paper line 17)
        if (total_viol < best_viol - feastol) {
            best_viol = total_viol;
            if (repair_track_best) {
                best_solution.assign(solution.begin(), solution.end());
                best_lhs.assign(lhs_cache.begin(), lhs_cache.end());
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
                                        repair_noise, rng, total_effort, scratch.walksat);
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
        auto cur_sol = static_cast<HighsInt>(sol_undo.size());
        auto cur_lhs = static_cast<HighsInt>(lhs_undo.size());

        // Push alternative first (explored second), then preferred (explored first)
        const HighsInt cur_e_pq = E.pq_initialized() ? E.pq_mark() : -1;
        Q.push_back({alternative.var, alternative.val, alternative.is_fix, alternative.is_lb,
                     cur_e_vs, cur_e_sol, cur_e_pq, cur_r_vs, cur_r_sol, cur_sol, cur_lhs,
                     total_viol});
        Q.push_back({preferred.var, preferred.val, preferred.is_fix, preferred.is_lb, cur_e_vs,
                     cur_e_sol, cur_e_pq, cur_r_vs, cur_r_sol, cur_sol, cur_lhs, total_viol});

        // Best-first steering (paper line 27)
        backtrack_best_open(Q);
    }

    // Restore best state (paper line 28)
    if (repair_track_best && best_viol < total_viol) {
        solution.assign(best_solution.begin(), best_solution.end());
        lhs_cache.assign(best_lhs.begin(), best_lhs.end());
        rebuild_violated();
    }

    effort_out = effort_spent();
    return violated.empty();
}
