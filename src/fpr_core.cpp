#include "fpr_core.h"

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "prop_engine.h"
#include "repair_search.h"
#include "walksat.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

HeuristicResult fpr_attempt(HighsMipSolver &mipsolver, const FprConfig &cfg, std::mt19937 &rng,
                            int attempt_idx, const double *initial_solution) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const auto &ARstart = mipdata->ARstart_;
    const auto &ARindex = mipdata->ARindex_;
    const auto &ARvalue = mipdata->ARvalue_;
    const auto &col_lb = model->col_lower_;
    const auto &col_ub = model->col_upper_;
    const auto &col_cost = model->col_cost_;
    const auto &row_lo = model->row_lower_;
    const auto &row_hi = model->row_upper_;
    const auto &integrality = model->integrality_;
    const double feastol = mipdata->feastol;
    const bool minimize = (model->sense_ == ObjSense::kMinimize);

    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return {};
    }

    // Use caller's CSC if provided, otherwise build our own
    CscMatrix owned_csc;
    if (!cfg.csc) {
        owned_csc = build_csc(ncol, nrow, ARstart, ARindex, ARvalue);
    }
    const auto &csc_ref = cfg.csc ? *cfg.csc : owned_csc;
    const auto &col_start = csc_ref.col_start;
    const auto &col_row = csc_ref.col_row;
    const auto &col_val = csc_ref.col_val;

    auto is_int = [&](HighsInt j) { return is_integer(integrality, j); };

    // Paper: artificial bounding box [-100000, +100000] for infinite bounds
    auto finite_clamp = [](double val, double lo, double hi) -> double {
        constexpr double kBox = 1e5;
        if (lo > -kHighsInf && hi < kHighsInf) {
            return std::max(lo, std::min(hi, val));
        }
        if (lo > -kHighsInf) {
            return std::max(lo, std::min(lo + kBox, val));
        }
        if (hi < kHighsInf) {
            return std::min(hi, std::max(hi - kBox, val));
        }
        return std::max(-kBox, std::min(kBox, val));
    };

    // Phase 1: Rank variables
    std::vector<HighsInt> var_order;
    if (cfg.precomputed_var_order != nullptr) {
        // Use pre-computed order (avoids data races on cliquePartition)
        var_order.assign(cfg.precomputed_var_order,
                         cfg.precomputed_var_order + cfg.precomputed_var_order_size);
    } else if (cfg.strategy) {
        var_order = compute_var_order(mipsolver, cfg.strategy->var_strategy, rng, cfg.lp_ref);
    } else {
        // Legacy: sort by caller-provided scores
        var_order.resize(ncol);
        for (HighsInt j = 0; j < ncol; ++j) {
            var_order[j] = j;
        }
        std::sort(var_order.begin(), var_order.end(),
                  [&](HighsInt a, HighsInt b) { return cfg.scores[a] > cfg.scores[b]; });
    }

    std::vector<double> lhs_cache(nrow);

    const HighsInt repair_budget = cfg.repair_iterations;

    auto is_violated = [&](HighsInt i, double lhs) -> bool {
        return is_row_violated(lhs, row_lo[i], row_hi[i], feastol);
    };

    // --- Create PropEngine for Phase 2 ---
    PropEngine E(ncol, nrow, ARstart.data(), ARindex.data(), ARvalue.data(), csc_ref, col_lb.data(),
                 col_ub.data(), row_lo.data(), row_hi.data(), integrality.data(), feastol);

    // --- Initialize solution in E ---
    if (initial_solution) {
        for (HighsInt j = 0; j < ncol; ++j) {
            double v = initial_solution[j];
            if (is_int(j)) {
                v = std::round(v);
            }
            E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], v));
        }
    } else if (attempt_idx == 0 && cfg.hint) {
        for (HighsInt j = 0; j < ncol; ++j) {
            double v = cfg.hint[j];
            if (is_int(j)) {
                v = std::round(v);
            }
            E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], v));
        }
    } else if (attempt_idx == 0) {
        for (HighsInt j = 0; j < ncol; ++j) {
            if (mipdata->domain.isBinary(j)) {
                E.sol(j) = 0.0;
            } else if (is_int(j)) {
                double lo = std::max(col_lb[j], -1e8);
                double hi = std::min(col_ub[j], lo + 100.0);
                E.sol(j) = std::round((lo + hi) * 0.5);
                E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], E.sol(j)));
            } else {
                E.sol(j) = finite_clamp(0.0, col_lb[j], col_ub[j]);
            }
        }
    } else {
        for (HighsInt j = 0; j < ncol; ++j) {
            if (mipdata->domain.isBinary(j)) {
                E.sol(j) = std::uniform_int_distribution<int>(0, 1)(rng);
            } else if (is_int(j)) {
                double lo = std::max(col_lb[j], -1e8);
                double hi = std::min(col_ub[j], lo + 100.0);
                E.sol(j) = std::round(std::uniform_real_distribution<double>(lo, hi)(rng));
                E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], E.sol(j)));
            } else {
                double lo = finite_clamp(0.0, col_lb[j], col_ub[j]);
                double hi = std::min(col_ub[j], lo + 1e6);
                if (hi < kHighsInf && lo > -kHighsInf && hi > lo) {
                    E.sol(j) = std::uniform_real_distribution<double>(lo, hi)(rng);
                } else {
                    E.sol(j) = lo;
                }
            }
        }
    }

    // Shuffle top 30% of ranking for diversity on later attempts
    if (attempt_idx > 0) {
        HighsInt shuffle_len = std::max(HighsInt{1}, ncol * 3 / 10);
        std::shuffle(var_order.begin(), var_order.begin() + shuffle_len, rng);
    }

    // --- Phase 2: Fix & Propagate ---
    // (E already initialized with global bounds via constructor)

    // Initialize incremental row activities if loosedyn is used
    if (cfg.strategy && cfg.strategy->val_strategy == ValStrategy::kLoosedyn) {
        E.init_activities();
    }

    // choose_fix_value: strategy-aware or legacy hint+objective-greedy fallback
    const bool use_hint = (attempt_idx == 0 && cfg.hint != nullptr);
    auto choose_fix_value = [&](HighsInt j) -> double {
        // Strategy-based value selection (paper Table 2)
        if (cfg.strategy) {
            return choose_value(
                j, E.var(j).lb, E.var(j).ub, is_int(j), minimize, col_cost[j],
                cfg.strategy->val_strategy, rng, cfg.lp_ref, row_lo.data(), row_hi.data(),
                E.activities_initialized() ? E.min_activity_data() : nullptr,
                E.activities_initialized() ? E.max_activity_data() : nullptr, &csc_ref);
        }

        // Legacy behavior
        double lo = E.var(j).lb;
        double hi = E.var(j).ub;

        if (use_hint) {
            double h = cfg.hint[j];
            if (is_int(j)) {
                h = std::round(h);
            }
            if (h >= lo - feastol && h <= hi + feastol) {
                return std::max(lo, std::min(hi, h));
            }
        }

        if (mipdata->domain.isBinary(j)) {
            if (minimize) {
                return (col_cost[j] >= 0) ? lo : hi;
            }
            return (col_cost[j] >= 0) ? hi : lo;
        }

        if (std::abs(col_cost[j]) < 1e-15) {
            double mid = std::round((lo + hi) * 0.5);
            return std::max(lo, std::min(hi, mid));
        }
        if (minimize) {
            return (col_cost[j] > 0) ? lo : hi;
        }
        return (col_cost[j] > 0) ? hi : lo;
    };

    // Paper Section 6: "fix all trivially-roundable variables (if any) to the
    // corresponding bound" before running strategies.
    if (!mipdata->uplocks.empty()) {
        const auto &uplocks = mipdata->uplocks;
        const auto &downlocks = mipdata->downlocks;
        for (HighsInt j = 0; j < ncol; ++j) {
            if (!is_int(j) || E.var(j).fixed) {
                continue;
            }
            if (uplocks[j] == 0 && downlocks[j] != 0) {
                E.fix(j, E.var(j).ub);
            } else if (downlocks[j] == 0 && uplocks[j] != 0) {
                E.fix(j, E.var(j).lb);
            }
        }
    }

    // Paper Section 6: "perform a first round of constraint propagation, until
    // a fixpoint is reached" before starting the DFS.
    for (HighsInt j = 0; j < ncol; ++j) {
        if (E.var(j).fixed) {
            E.seed_worklist(j);
        }
    }
    E.propagate(-1);

    // --- Phase 2: DFS Fix & Propagate (paper Fig. 1) ---

    const bool dynamic_var = cfg.strategy && is_dynamic_var_strategy(cfg.strategy->var_strategy);

    // Cursor into var_order: tracks how far we've scanned. Advances on the
    // forward path so each position is visited at most once. Reset on backtrack.
    const auto var_order_size = static_cast<HighsInt>(var_order.size());
    HighsInt var_order_cursor = 0;

    // Static: scan from cursor (O(1) amortized).
    auto find_next_unfixed_int_static = [&]() -> std::pair<HighsInt, HighsInt> {
        for (; var_order_cursor < var_order_size; ++var_order_cursor) {
            HighsInt j = var_order[var_order_cursor];
            if (is_int(j) && !E.var(j).fixed) {
                return {j, var_order_cursor};
            }
        }
        return {-1, -1};
    };

    // Dynamic: smallest-domain-first via priority queue maintained by PropEngine.
    if (dynamic_var) {
        E.init_domain_pq();
    }

    auto find_next_unfixed_int = [&]() -> std::pair<HighsInt, HighsInt> {
        if (dynamic_var) {
            return {E.pq_top(), 0};
        }
        return find_next_unfixed_int_static();
    };

    // Compute alternative value for branching
    auto compute_alt = [&](HighsInt j, double preferred) -> double {
        if (mipdata->domain.isBinary(j)) {
            return (preferred < 0.5) ? 1.0 : 0.0;
        }
        double alt = (std::abs(preferred - E.var(j).lb) < feastol) ? E.var(j).ub : E.var(j).lb;
        if (is_int(j)) {
            alt = std::round(alt);
        }
        return alt;
    };

    struct DfsNode {
        HighsInt var;
        double val;
        HighsInt vs_mark;
        HighsInt sol_mark;
        HighsInt act_mark;
        HighsInt pq_mark;
        HighsInt cursor_reset;  // backtrack point: reset var_order_cursor here
    };

    const bool do_propagate = mode_propagates(cfg.mode);
    const bool do_backtrack = mode_backtracks(cfg.mode);
    const HighsInt node_limit = ncol + 1;

    std::vector<DfsNode> dfs_stack;
    dfs_stack.reserve(do_backtrack ? 2 * static_cast<size_t>(ncol) : ncol);
    HighsInt nodes_visited = 0;
    bool found_complete = false;

    // Seed the DFS with the first unfixed integer variable
    auto [first_var, first_idx] = find_next_unfixed_int();
    if (first_var < 0) {
        // All integer variables already fixed (e.g., by propagation)
        found_complete = true;
    } else {
        double pref = choose_fix_value(first_var);
        double alt = compute_alt(first_var, pref);
        HighsInt vs_m = E.vs_mark();
        HighsInt sol_m = E.sol_mark();
        HighsInt act_m = E.act_mark();
        HighsInt pq_m = E.pq_initialized() ? E.pq_mark() : -1;
        HighsInt cursor_pt = first_idx + 1;

        if (do_backtrack) {
            dfs_stack.push_back({first_var, alt, vs_m, sol_m, act_m, pq_m, cursor_pt});
        }
        dfs_stack.push_back({first_var, pref, vs_m, sol_m, act_m, pq_m, cursor_pt});
    }

    // Gate Phase 1-2 DFS on cfg.max_effort.  Without this, the DFS runs up to
    // node_limit = ncol + 1 with a full propagation fixpoint per node, which on
    // tight models (e.g. 4k cols × 9k nnz) consumes ~150M effort regardless of
    // the budget the caller passed.  Phase 3 (repair_search / walksat) already
    // respected max_effort, but if Phase 1-2 exhausted the DFS node cap we
    // never reached Phase 3 — the arm simply returned "no complete assignment"
    // with a huge effort count.
    //
    // The pre-loop `E.propagate(-1)` during DFS seeding is intentionally not
    // gated; the initial fixpoint is required to place the root node.  If
    // cfg.max_effort is 0 (caller asked for zero budget), the comparison
    // `E.effort() < 0` is false as a size_t, so the loop never iterates —
    // the arm returns "no complete assignment" with effort ≈ initial
    // propagation cost.  That matches the rest of the solver's "best effort"
    // budget contract.
    while (!dfs_stack.empty() && nodes_visited < node_limit && !found_complete &&
           E.effort() < cfg.max_effort) {
        auto node = dfs_stack.back();
        dfs_stack.pop_back();
        ++nodes_visited;

        // Backtrack to parent state and reset cursor
        E.backtrack_to(node.vs_mark, node.sol_mark, node.act_mark, node.pq_mark);
        var_order_cursor = node.cursor_reset;

        // Apply the branching fixing
        if (!E.fix(node.var, node.val)) {
            continue;  // can't fix, try next node (sibling)
        }

        // Propagate
        if (do_propagate) {
            if (!E.propagate(node.var)) {
                continue;  // infeasible, try next node (sibling)
            }
        }

        // Find next unfixed integer variable (cursor advances from last position)
        auto [next_var, next_idx] = find_next_unfixed_int();

        if (next_var < 0) {
            // All integer variables fixed
            found_complete = true;
            break;
        }

        // Branch on next variable: push children to DFS stack
        double pref = choose_fix_value(next_var);
        double alt = compute_alt(next_var, pref);
        HighsInt vs_m = E.vs_mark();
        HighsInt sol_m = E.sol_mark();
        HighsInt act_m = E.act_mark();
        HighsInt pq_m = E.pq_initialized() ? E.pq_mark() : -1;
        HighsInt cursor_pt = next_idx + 1;

        if (do_backtrack) {
            dfs_stack.push_back({next_var, alt, vs_m, sol_m, act_m, pq_m, cursor_pt});
        }
        dfs_stack.push_back({next_var, pref, vs_m, sol_m, act_m, pq_m, cursor_pt});
    }

    if (!found_complete) {
        return HeuristicResult::failed(E.effort());
    }

    // Fix remaining unfixed variables (continuous and residual integers)
    for (HighsInt j = 0; j < ncol; ++j) {
        if (E.var(j).fixed) {
            continue;
        }
        double lo = E.var(j).lb;
        double hi = E.var(j).ub;

        if (!is_int(j)) {
            if (std::abs(col_cost[j]) > 1e-15) {
                bool want_low = (minimize == (col_cost[j] > 0));
                E.sol(j) = finite_clamp(want_low ? lo : hi, lo, hi);
            } else {
                double fallback = cfg.cont_fallback ? cfg.cont_fallback[j] : 0.0;
                E.sol(j) = finite_clamp(fallback, lo, hi);
            }
        } else {
            E.sol(j) = choose_fix_value(j);
            E.sol(j) = std::round(std::max(lo, std::min(hi, E.sol(j))));
        }
        E.sol(j) = std::max(col_lb[j], std::min(col_ub[j], E.sol(j)));
    }

    // --- Copy solution out of E for Phase 3 and result ---
    std::vector<double> solution(E.sol_data(), E.sol_data() + ncol);
    size_t total_prop_work = E.effort();

    // --- Compute LHS cache ---
    total_prop_work += ARindex.size();
    for (HighsInt i = 0; i < nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
            lhs += ARvalue[k] * solution[ARindex[k]];
        }
        lhs_cache[i] = lhs;
    }

    bool feasible = true;
    for (HighsInt i = 0; i < nrow; ++i) {
        if (is_violated(i, lhs_cache[i])) {
            feasible = false;
            break;
        }
    }

    // --- Phase 3: RepairSearch (Fig. 5) or WalkSAT Repair ---
    if (!feasible && cfg.mode == FrameworkMode::kRepairSearch) {
        size_t rs_effort = 0;
        feasible = repair_search(
            E, solution, lhs_cache, col_lb.data(), col_ub.data(), row_lo.data(), row_hi.data(),
            cfg.repair_iterations, cfg.repair_noise, cfg.repair_track_best,
            cfg.max_effort > total_prop_work ? cfg.max_effort - total_prop_work : 0, rng,
            rs_effort);
        total_prop_work += rs_effort;
    } else if (!feasible && mode_repairs(cfg.mode)) {
        size_t walk_effort = 0;
        feasible =
            walksat_repair(E, solution, lhs_cache, col_lb.data(), col_ub.data(), repair_budget,
                           cfg.repair_noise, cfg.repair_track_best,
                           cfg.max_effort > total_prop_work ? cfg.max_effort - total_prop_work : 0,
                           rng, walk_effort);
        total_prop_work += walk_effort;
    }

    if (!feasible) {
        return HeuristicResult::failed(total_prop_work);
    }

    // Verify feasibility using cached LHS values (O(nrow) vs O(nnz))
    for (HighsInt i = 0; i < nrow; ++i) {
        if (is_violated(i, lhs_cache[i])) {
            return HeuristicResult::failed(total_prop_work);
        }
    }

    // Greedy 1-opt (paper Section 6)
    greedy_1opt(E, solution, lhs_cache, col_cost.data(), minimize, total_prop_work);

    double obj = model->offset_;
    for (HighsInt j = 0; j < ncol; ++j) {
        obj += col_cost[j] * solution[j];
    }

    HeuristicResult result;
    result.found_feasible = true;
    result.solution = std::move(solution);
    result.objective = obj;
    result.effort = total_prop_work;
    return result;
}
