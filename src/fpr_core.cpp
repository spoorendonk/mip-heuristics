#include "fpr_core.h"

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "prop_engine.h"
#include "repair_search.h"
#include "walksat.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

// Bundle the per-call references that begin/step/finish all need to
// rehydrate the lambdas (`is_int`, `finite_clamp`, `choose_fix_value`,
// `is_violated`).  The lambdas themselves cannot live in
// `FprAttemptState` (closures over references are not portably
// stashable), so each function rebuilds them from this struct.  Cheap
// — the lambdas are stateless wrappers over const refs.
struct AttemptCtx {
    HighsMipSolver &mipsolver;
    const HighsLp *model;
    HighsMipSolverData *mipdata;
    const std::vector<HighsInt> &ARstart;
    const std::vector<HighsInt> &ARindex;
    const std::vector<double> &ARvalue;
    const std::vector<double> &col_lb;
    const std::vector<double> &col_ub;
    const std::vector<double> &col_cost;
    const std::vector<double> &row_lo;
    const std::vector<double> &row_hi;
    const std::vector<HighsVarType> &integrality;
    double feastol;
    bool minimize;
    HighsInt ncol;
    HighsInt nrow;
};

AttemptCtx make_ctx(HighsMipSolver &mipsolver) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    return AttemptCtx{
        mipsolver,         model,
        mipdata,           mipdata->ARstart_,
        mipdata->ARindex_, mipdata->ARvalue_,
        model->col_lower_, model->col_upper_,
        model->col_cost_,  model->row_lower_,
        model->row_upper_, model->integrality_,
        mipdata->feastol,  model->sense_ == ObjSense::kMinimize,
        model->num_col_,   model->num_row_,
    };
}

// Paper: artificial bounding box [-100000, +100000] for infinite bounds.
double finite_clamp_helper(double val, double lo, double hi) {
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
}

// Lazy-construct (or reset) the cached PropEngine inside the scratch.
// Called exactly once per attempt (only from `fpr_attempt_begin`) —
// `fpr_attempt_step` must NOT call this or the DFS undo stacks underneath
// the in-flight attempt are wiped.  The pointer-identity check below
// guards against problem-buffer reuse across attempts on a stale
// scratch; comparing dangling pointers to .data() of vectors that have
// since been freed is technically indeterminate per the C++ standard
// but benign on all mainstream toolchains.  Hot-path callers (the FPR
// worker, scylla, fpr_lp, portfolio) pair a stable `cfg.csc` and a
// stable `mipsolver` with the scratch's lifetime — see the lifetime
// comment on `FprConfig::scratch` in `fpr_core.h`.
PropEngine &acquire_engine(FprScratch &scratch, const AttemptCtx &c, const CscMatrix &csc) {
    std::optional<PropEngine> &engine_opt = scratch.prop_engine;
    const bool engine_valid =
        engine_opt.has_value() && engine_opt->ncol() == c.ncol && engine_opt->nrow() == c.nrow &&
        engine_opt->ar_start() == c.ARstart.data() && engine_opt->ar_index() == c.ARindex.data() &&
        engine_opt->ar_value() == c.ARvalue.data() &&
        engine_opt->csc_start() == csc.col_start.data() &&
        engine_opt->csc_row() == csc.col_row.data() &&
        engine_opt->csc_val() == csc.col_val.data() && engine_opt->col_lb() == c.col_lb.data() &&
        engine_opt->col_ub() == c.col_ub.data() && engine_opt->row_lo() == c.row_lo.data() &&
        engine_opt->row_hi() == c.row_hi.data() &&
        engine_opt->integrality() == c.integrality.data() && engine_opt->feastol() == c.feastol;
    if (!engine_valid) {
        engine_opt.emplace(c.ncol, c.nrow, c.ARstart.data(), c.ARindex.data(), c.ARvalue.data(),
                           csc, c.col_lb.data(), c.col_ub.data(), c.row_lo.data(), c.row_hi.data(),
                           c.integrality.data(), c.feastol);
    } else {
        engine_opt->reset();
    }
    return *engine_opt;
}

// Strategy-aware or legacy hint+objective-greedy value selection.
// Pure (no state outside its arguments); rebuild fresh in each begin/step/finish.
double choose_fix_value(HighsInt j, const FprConfig &cfg, const AttemptCtx &c, PropEngine &E,
                        const CscMatrix &csc, Rng &rng, bool use_hint) {
    if (cfg.strategy) {
        return choose_value(j, E.var(j).lb, E.var(j).ub, is_integer(c.integrality, j), c.minimize,
                            c.col_cost[j], cfg.strategy->val_strategy, rng, cfg.lp_ref,
                            c.row_lo.data(), c.row_hi.data(),
                            E.activities_initialized() ? E.min_activity_data() : nullptr,
                            E.activities_initialized() ? E.max_activity_data() : nullptr, &csc);
    }

    double lo = E.var(j).lb;
    double hi = E.var(j).ub;
    const bool is_int = is_integer(c.integrality, j);

    if (use_hint && cfg.hint != nullptr) {
        double h = cfg.hint[j];
        if (is_int) {
            h = std::round(h);
        }
        if (h >= lo - c.feastol && h <= hi + c.feastol) {
            return std::max(lo, std::min(hi, h));
        }
    }

    if (c.mipdata->domain.isBinary(j)) {
        if (c.minimize) {
            return (c.col_cost[j] >= 0) ? lo : hi;
        }
        return (c.col_cost[j] >= 0) ? hi : lo;
    }

    if (std::abs(c.col_cost[j]) < 1e-15) {
        double mid = std::round((lo + hi) * 0.5);
        return std::max(lo, std::min(hi, mid));
    }
    if (c.minimize) {
        return (c.col_cost[j] > 0) ? lo : hi;
    }
    return (c.col_cost[j] > 0) ? hi : lo;
}

double compute_alt(HighsInt j, double preferred, const AttemptCtx &c, PropEngine &E) {
    if (c.mipdata->domain.isBinary(j)) {
        return (preferred < 0.5) ? 1.0 : 0.0;
    }
    double alt = (std::abs(preferred - E.var(j).lb) < c.feastol) ? E.var(j).ub : E.var(j).lb;
    if (is_integer(c.integrality, j)) {
        alt = std::round(alt);
    }
    return alt;
}

bool is_row_violated_in_ctx(HighsInt i, double lhs, const AttemptCtx &c) {
    return is_row_violated(lhs, c.row_lo[i], c.row_hi[i], c.feastol);
}

}  // namespace

// ---------------------------------------------------------------------------
// fpr_attempt_begin
// ---------------------------------------------------------------------------

void fpr_attempt_begin(FprAttemptState &state, HighsMipSolver &mipsolver, const FprConfig &cfg,
                       Rng &rng, int attempt_idx, const double *initial_solution) {
    assert(cfg.scratch != nullptr && "fpr_attempt_begin requires cfg.scratch");
    FprScratch &scratch = *cfg.scratch;
    const AttemptCtx c = make_ctx(mipsolver);

    // Lifecycle reset.
    state = FprAttemptState{};
    state.ncol = c.ncol;
    state.nrow = c.nrow;
    state.attempt_idx = attempt_idx;

    if (c.ncol == 0 || c.nrow == 0) {
        // Degenerate model — no DFS to do.  Leave phase = kIdle so a
        // subsequent finish would short-circuit on found_complete=false.
        // Match the legacy fpr_attempt early-return shape.
        state.phase = FprAttemptState::Phase::kIdle;
        return;
    }

    // The lifecycle API requires cfg.csc — one-shot callers go via
    // `fpr_attempt` which builds a local CSC.  Persistent callers (the
    // FPR worker) all carry a stable cfg.csc.
    assert(cfg.csc != nullptr && "fpr_attempt_begin requires cfg.csc");
    const CscMatrix &csc = *cfg.csc;

    // --- Phase 1: variable ranking -------------------------------------------------
    auto &var_order = scratch.var_order;
    var_order.clear();
    if (cfg.precomputed_var_order != nullptr) {
        var_order.assign(cfg.precomputed_var_order,
                         cfg.precomputed_var_order + cfg.precomputed_var_order_size);
    } else if (cfg.strategy) {
        var_order = compute_var_order(mipsolver, cfg.strategy->var_strategy, rng, cfg.lp_ref);
    } else {
        var_order.resize(c.ncol);
        for (HighsInt j = 0; j < c.ncol; ++j) {
            var_order[j] = j;
        }
        std::sort(var_order.begin(), var_order.end(),
                  [&](HighsInt a, HighsInt b) { return cfg.scores[a] > cfg.scores[b]; });
    }
    state.var_order_size = static_cast<HighsInt>(var_order.size());

    // Ensure scratch.lhs_cache has capacity for finish().
    scratch.lhs_cache.resize(c.nrow);

    // --- Acquire PropEngine (resets if cached engine is from a previous attempt) ---
    PropEngine &E = acquire_engine(scratch, c, csc);

    // --- Initial solution -----------------------------------------------------------
    auto is_int = [&](HighsInt j) { return is_integer(c.integrality, j); };

    if (initial_solution) {
        for (HighsInt j = 0; j < c.ncol; ++j) {
            double v = initial_solution[j];
            if (is_int(j)) {
                v = std::round(v);
            }
            E.sol(j) = std::max(c.col_lb[j], std::min(c.col_ub[j], v));
        }
    } else if (attempt_idx == 0 && cfg.hint) {
        for (HighsInt j = 0; j < c.ncol; ++j) {
            double v = cfg.hint[j];
            if (is_int(j)) {
                v = std::round(v);
            }
            E.sol(j) = std::max(c.col_lb[j], std::min(c.col_ub[j], v));
        }
    } else if (attempt_idx == 0) {
        for (HighsInt j = 0; j < c.ncol; ++j) {
            if (c.mipdata->domain.isBinary(j)) {
                E.sol(j) = 0.0;
            } else if (is_int(j)) {
                double lo = std::max(c.col_lb[j], -1e8);
                double hi = std::min(c.col_ub[j], lo + 100.0);
                E.sol(j) = std::round((lo + hi) * 0.5);
                E.sol(j) = std::max(c.col_lb[j], std::min(c.col_ub[j], E.sol(j)));
            } else {
                E.sol(j) = finite_clamp_helper(0.0, c.col_lb[j], c.col_ub[j]);
            }
        }
    } else {
        for (HighsInt j = 0; j < c.ncol; ++j) {
            if (c.mipdata->domain.isBinary(j)) {
                E.sol(j) = std::uniform_int_distribution<int>(0, 1)(rng);
            } else if (is_int(j)) {
                double lo = std::max(c.col_lb[j], -1e8);
                double hi = std::min(c.col_ub[j], lo + 100.0);
                E.sol(j) = std::round(std::uniform_real_distribution<double>(lo, hi)(rng));
                E.sol(j) = std::max(c.col_lb[j], std::min(c.col_ub[j], E.sol(j)));
            } else {
                double lo = finite_clamp_helper(0.0, c.col_lb[j], c.col_ub[j]);
                double hi = std::min(c.col_ub[j], lo + 1e6);
                if (hi < kHighsInf && lo > -kHighsInf && hi > lo) {
                    E.sol(j) = std::uniform_real_distribution<double>(lo, hi)(rng);
                } else {
                    E.sol(j) = lo;
                }
            }
        }
    }

    // Shuffle top 30% of ranking for diversity on later attempts.
    if (attempt_idx > 0) {
        HighsInt shuffle_len = std::max(HighsInt{1}, c.ncol * 3 / 10);
        std::shuffle(var_order.begin(), var_order.begin() + shuffle_len, rng);
    }

    if (cfg.strategy && cfg.strategy->val_strategy == ValStrategy::kLoosedyn) {
        E.init_activities();
    }

    // Trivially-roundable fixings (paper Section 6).
    if (!c.mipdata->uplocks.empty()) {
        const auto &uplocks = c.mipdata->uplocks;
        const auto &downlocks = c.mipdata->downlocks;
        for (HighsInt j = 0; j < c.ncol; ++j) {
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

    // First round of constraint propagation to fixpoint.
    for (HighsInt j = 0; j < c.ncol; ++j) {
        if (E.var(j).fixed) {
            E.seed_worklist(j);
        }
    }
    E.propagate(-1);

    // --- Phase 2 setup -------------------------------------------------------------
    state.dynamic_var = cfg.strategy && is_dynamic_var_strategy(cfg.strategy->var_strategy);
    state.do_propagate = mode_propagates(cfg.mode);
    state.do_backtrack = mode_backtracks(cfg.mode);
    state.node_limit = c.ncol + 1;
    state.var_order_cursor = 0;
    state.nodes_visited = 0;
    state.found_complete = false;

    auto &dfs_stack = scratch.dfs_stack;
    dfs_stack.clear();
    const size_t dfs_reserve =
        state.do_backtrack ? 2 * static_cast<size_t>(c.ncol) : static_cast<size_t>(c.ncol);
    if (dfs_stack.capacity() < dfs_reserve) {
        dfs_stack.reserve(dfs_reserve);
    }

    if (state.dynamic_var) {
        E.init_domain_pq();
    }

    // Seed root DFS node.
    HighsInt first_var = -1;
    HighsInt first_idx = -1;
    if (state.dynamic_var) {
        first_var = E.pq_top();
        first_idx = 0;
    } else {
        for (; state.var_order_cursor < state.var_order_size; ++state.var_order_cursor) {
            HighsInt j = var_order[state.var_order_cursor];
            if (is_int(j) && !E.var(j).fixed) {
                first_var = j;
                first_idx = state.var_order_cursor;
                break;
            }
        }
    }

    if (first_var < 0) {
        // All integers fixed by propagation; DFS is trivial — go straight
        // to finish.
        state.found_complete = true;
        state.phase = FprAttemptState::Phase::kReadyToFinish;
    } else {
        const bool use_hint = (attempt_idx == 0 && cfg.hint != nullptr);
        double pref = choose_fix_value(first_var, cfg, c, E, csc, rng, use_hint);
        double alt = compute_alt(first_var, pref, c, E);
        HighsInt vs_m = E.vs_mark();
        HighsInt sol_m = E.sol_mark();
        HighsInt act_m = E.act_mark();
        HighsInt pq_m = E.pq_initialized() ? E.pq_mark() : -1;
        HighsInt cursor_pt = first_idx + 1;

        if (state.do_backtrack) {
            dfs_stack.push_back({first_var, alt, vs_m, sol_m, act_m, pq_m, cursor_pt});
        }
        dfs_stack.push_back({first_var, pref, vs_m, sol_m, act_m, pq_m, cursor_pt});
        state.phase = FprAttemptState::Phase::kDfs;
    }

    state.effort_consumed = E.effort();
}

// ---------------------------------------------------------------------------
// fpr_attempt_step
// ---------------------------------------------------------------------------

FprStepResult fpr_attempt_step(FprAttemptState &state, HighsMipSolver &mipsolver,
                               const FprConfig &cfg, Rng &rng, size_t effort_remaining) {
    assert(state.phase == FprAttemptState::Phase::kDfs &&
           "fpr_attempt_step called outside kDfs phase");
    assert(cfg.scratch != nullptr);
    assert(cfg.csc != nullptr);

    FprScratch &scratch = *cfg.scratch;
    const AttemptCtx c = make_ctx(mipsolver);
    const CscMatrix &csc = *cfg.csc;
    PropEngine &E = *scratch.prop_engine;
    auto &dfs_stack = scratch.dfs_stack;
    auto &var_order = scratch.var_order;

    auto is_int = [&](HighsInt j) { return is_integer(c.integrality, j); };

    auto find_next_unfixed_int = [&]() -> std::pair<HighsInt, HighsInt> {
        if (state.dynamic_var) {
            return {E.pq_top(), 0};
        }
        for (; state.var_order_cursor < state.var_order_size; ++state.var_order_cursor) {
            HighsInt j = var_order[state.var_order_cursor];
            if (is_int(j) && !E.var(j).fixed) {
                return {j, state.var_order_cursor};
            }
        }
        return {-1, -1};
    };

    // Per-call DFS budget is `effort_remaining` (a slice of the worker's
    // `epoch_budget`).  Crucially: gate on the *delta* `E.effort() -
    // effort_at_call_start`, not on absolute `E.effort()`.  After a
    // paused attempt resumes from kBudgetGate, `E.effort()` is already
    // at the previous call's slice high-water mark; comparing against
    // an absolute target derived from current effort would treat the
    // DFS as already-exhausted on entry and exit immediately, making
    // forward progress impossible (the bug that hangs `infeasible-mip0`
    // when run alongside FJ/LocalMIP/Scylla).  cfg.max_effort still
    // bounds Phase 3 (repair/walksat) inside `fpr_attempt_finish`; it
    // is no longer the DFS gate's cap.
    const size_t effort_at_call_start = E.effort();
    // Target as a delta from this call's start, not an absolute.
    const size_t effort_target_delta = effort_remaining;
    const bool use_hint = (state.attempt_idx == 0 && cfg.hint != nullptr);

    while (!dfs_stack.empty() && state.nodes_visited < state.node_limit && !state.found_complete &&
           (E.effort() - effort_at_call_start) < effort_target_delta) {
        auto node = dfs_stack.back();
        dfs_stack.pop_back();
        ++state.nodes_visited;

        E.backtrack_to(node.vs_mark, node.sol_mark, node.act_mark, node.pq_mark);
        state.var_order_cursor = node.cursor_reset;

        if (!E.fix(node.var, node.val)) {
            continue;
        }

        if (state.do_propagate) {
            if (!E.propagate(node.var)) {
                continue;
            }
        }

        auto [next_var, next_idx] = find_next_unfixed_int();

        if (next_var < 0) {
            state.found_complete = true;
            break;
        }

        double pref = choose_fix_value(next_var, cfg, c, E, csc, rng, use_hint);
        double alt = compute_alt(next_var, pref, c, E);
        HighsInt vs_m = E.vs_mark();
        HighsInt sol_m = E.sol_mark();
        HighsInt act_m = E.act_mark();
        HighsInt pq_m = E.pq_initialized() ? E.pq_mark() : -1;
        HighsInt cursor_pt = next_idx + 1;

        if (state.do_backtrack) {
            dfs_stack.push_back({next_var, alt, vs_m, sol_m, act_m, pq_m, cursor_pt});
        }
        dfs_stack.push_back({next_var, pref, vs_m, sol_m, act_m, pq_m, cursor_pt});
    }

    state.effort_consumed = E.effort();

    // Verdict determined?  Found a leaf or stack/node-limit exhausted.
    if (state.found_complete || dfs_stack.empty() || state.nodes_visited >= state.node_limit) {
        state.phase = FprAttemptState::Phase::kReadyToFinish;
        return FprStepResult::kVerdictReady;
    }

    // Budget gate hit; attempt is alive.
    return FprStepResult::kBudgetGate;
}

// ---------------------------------------------------------------------------
// fpr_attempt_finish
// ---------------------------------------------------------------------------

HeuristicResult fpr_attempt_finish(FprAttemptState &state, HighsMipSolver &mipsolver,
                                   const FprConfig &cfg, Rng &rng) {
    assert(cfg.scratch != nullptr);

    FprScratch &scratch = *cfg.scratch;
    const AttemptCtx c = make_ctx(mipsolver);

    // Degenerate model from begin() — short-circuit cleanly.
    if (c.ncol == 0 || c.nrow == 0) {
        state.phase = FprAttemptState::Phase::kIdle;
        return {};
    }

    assert(cfg.csc != nullptr);
    const CscMatrix &csc = *cfg.csc;
    PropEngine &E = *scratch.prop_engine;

    auto is_int = [&](HighsInt j) { return is_integer(c.integrality, j); };

    if (!state.found_complete) {
        state.phase = FprAttemptState::Phase::kIdle;
        return HeuristicResult::failed(E.effort());
    }

    // Phase 2.5: fix remaining unfixed variables (continuous + residual integers).
    const bool use_hint = (state.attempt_idx == 0 && cfg.hint != nullptr);
    for (HighsInt j = 0; j < c.ncol; ++j) {
        if (E.var(j).fixed) {
            continue;
        }
        double lo = E.var(j).lb;
        double hi = E.var(j).ub;

        if (!is_int(j)) {
            if (std::abs(c.col_cost[j]) > 1e-15) {
                bool want_low = (c.minimize == (c.col_cost[j] > 0));
                E.sol(j) = finite_clamp_helper(want_low ? lo : hi, lo, hi);
            } else {
                double fallback = cfg.cont_fallback ? cfg.cont_fallback[j] : 0.0;
                E.sol(j) = finite_clamp_helper(fallback, lo, hi);
            }
        } else {
            E.sol(j) = choose_fix_value(j, cfg, c, E, csc, rng, use_hint);
            E.sol(j) = std::round(std::max(lo, std::min(hi, E.sol(j))));
        }
        E.sol(j) = std::max(c.col_lb[j], std::min(c.col_ub[j], E.sol(j)));
    }

    auto &solution = scratch.solution;
    solution.assign(E.sol_data(), E.sol_data() + c.ncol);
    size_t total_prop_work = E.effort();

    auto &lhs_cache = scratch.lhs_cache;
    lhs_cache.resize(c.nrow);
    total_prop_work += c.ARindex.size();
    for (HighsInt i = 0; i < c.nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = c.ARstart[i]; k < c.ARstart[i + 1]; ++k) {
            lhs += c.ARvalue[k] * solution[c.ARindex[k]];
        }
        lhs_cache[i] = lhs;
    }

    bool feasible = true;
    for (HighsInt i = 0; i < c.nrow; ++i) {
        if (is_row_violated_in_ctx(i, lhs_cache[i], c)) {
            feasible = false;
            break;
        }
    }

    // Phase 3: RepairSearch (Fig. 5) or WalkSAT.
    if (!feasible && cfg.mode == FrameworkMode::kRepairSearch) {
        size_t rs_effort = 0;
        feasible = repair_search(
            E, solution, lhs_cache, c.col_lb.data(), c.col_ub.data(), c.row_lo.data(),
            c.row_hi.data(), cfg.repair_iterations, cfg.repair_noise, cfg.repair_track_best,
            cfg.max_effort > total_prop_work ? cfg.max_effort - total_prop_work : 0, rng, rs_effort,
            scratch);
        total_prop_work += rs_effort;
    } else if (!feasible && mode_repairs(cfg.mode)) {
        size_t walk_effort = 0;
        feasible =
            walksat_repair(E, solution, lhs_cache, c.col_lb.data(), c.col_ub.data(),
                           cfg.walksat_iterations, cfg.repair_noise, cfg.repair_track_best,
                           cfg.max_effort > total_prop_work ? cfg.max_effort - total_prop_work : 0,
                           rng, walk_effort, scratch.walksat);
        total_prop_work += walk_effort;
    }

    if (!feasible) {
        state.phase = FprAttemptState::Phase::kIdle;
        state.effort_consumed = total_prop_work;
        return HeuristicResult::failed(total_prop_work);
    }

    for (HighsInt i = 0; i < c.nrow; ++i) {
        if (is_row_violated_in_ctx(i, lhs_cache[i], c)) {
            state.phase = FprAttemptState::Phase::kIdle;
            state.effort_consumed = total_prop_work;
            return HeuristicResult::failed(total_prop_work);
        }
    }

    greedy_1opt(E, solution, lhs_cache, c.col_cost.data(), c.minimize, total_prop_work);

    double obj = c.model->offset_;
    for (HighsInt j = 0; j < c.ncol; ++j) {
        obj += c.col_cost[j] * solution[j];
    }

    HeuristicResult result;
    result.found_feasible = true;
    result.solution = std::move(solution);
    result.objective = obj;
    result.effort = total_prop_work;
    state.phase = FprAttemptState::Phase::kIdle;
    state.effort_consumed = total_prop_work;
    return result;
}

// ---------------------------------------------------------------------------
// fpr_attempt — backward-compatible one-shot wrapper
// ---------------------------------------------------------------------------
//
// One-shot callers (tests, portfolio, scylla, fpr_lp) keep this entry point.
// It runs begin → step (uncapped) → finish in sequence on a local state,
// and accepts a null cfg.scratch by routing through a function-local scratch
// (matches the pre-#77 contract for those callers).

HeuristicResult fpr_attempt(HighsMipSolver &mipsolver, const FprConfig &cfg, Rng &rng,
                            int attempt_idx, const double *initial_solution) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return {};
    }

    FprScratch local_scratch;
    CscMatrix owned_csc;
    if (!cfg.csc) {
        owned_csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);
    }

    FprConfig effective_cfg = cfg;
    if (effective_cfg.scratch == nullptr) {
        effective_cfg.scratch = &local_scratch;
    }
    if (effective_cfg.csc == nullptr) {
        effective_cfg.csc = &owned_csc;
    }

    FprAttemptState state;
    fpr_attempt_begin(state, mipsolver, effective_cfg, rng, attempt_idx, initial_solution);

    // Single-shot DFS gated by `cfg.max_effort` — matches the pre-#77
    // contract for one-shot callers (portfolio / scylla / fpr_lp / tests).
    // The `if` (not a `while`) reflects the actual control flow: step
    // either returns `kVerdictReady` (which finish handles, possibly via
    // its `!found_complete` shortcut to `failed`) or `kBudgetGate`, in
    // which case we force a `kReadyToFinish` and let finish emit
    // `failed(E.effort())`.  Either way, exactly one step call.
    if (state.phase == FprAttemptState::Phase::kDfs) {
        const size_t already_used =
            effective_cfg.scratch->prop_engine ? effective_cfg.scratch->prop_engine->effort() : 0;
        const size_t remaining =
            effective_cfg.max_effort > already_used ? effective_cfg.max_effort - already_used : 0;
        const FprStepResult outcome =
            fpr_attempt_step(state, mipsolver, effective_cfg, rng, remaining);
        if (outcome == FprStepResult::kBudgetGate) {
            state.phase = FprAttemptState::Phase::kReadyToFinish;
        }
    }

    return fpr_attempt_finish(state, mipsolver, effective_cfg, rng);
}
