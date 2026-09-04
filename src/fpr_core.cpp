#include "fpr_core.h"

#include "heuristic_common.h"
#include "heuristic_context.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "prop_engine.h"
#include "repair_search.h"
#include "repair_walk.h"
#include "walksat.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace {

// DFS nodes between two polls of the wall-clock deadline (issue #117).
//
// A node is a `fix` plus, in every propagating mode, a propagation
// fixpoint, so it already costs orders of magnitude more than the
// `clock_gettime` behind `Deadline::expired()`; the cadence exists to keep
// the poll off the hottest instruction path, not because the clock read is
// expensive relative to the work it guards.
//
// What this cadence still governs, after #151, is the *non*-propagating
// modes (`mode_propagates(cfg.mode) == false`), where a node is a `fix`
// plus a value choice and nothing bigger.  A propagating node no longer
// waits for it: the fixpoint polls the same clock itself every
// `kPropagateDeadlinePollWork` counted accesses and the loop below breaks
// out on `PropResult::kDeadlineExpired`, so the residual floor there is
// part of one fixpoint rather than sixteen whole ones — see
// `docs/PARAMETERS.md`, "What bounds the deadline's tightness".
constexpr HighsInt kDeadlinePollNodes = 16;

// Bundle the per-call references that begin/step/finish all need to
// rehydrate the lambdas (`is_int`, `finite_clamp`, `choose_fix_value`,
// `is_violated`).  The lambdas themselves cannot live in
// `FprAttemptState` (closures over references are not portably
// stashable), so each function rebuilds them from this struct.  Cheap
// — the lambdas are stateless wrappers over const refs.
struct AttemptCtx {
    HighsMipSolver& mipsolver;
    const HighsLp* model;
    HighsMipSolverData* mipdata;
    const std::vector<HighsInt>& ar_start;
    const std::vector<HighsInt>& ar_index;
    const std::vector<double>& ar_value;
    const std::vector<double>& col_lb;
    const std::vector<double>& col_ub;
    const std::vector<double>& col_cost;
    const std::vector<double>& row_lo;
    const std::vector<double>& row_hi;
    const std::vector<HighsVarType>& integrality;
    double feastol;
    bool minimize;
    HighsInt ncol;
    HighsInt nrow;
    // Dispatch-time `isBinary` snapshot; see `FprConfig::binary_mask`.
    const uint8_t* binary;

    [[nodiscard]] bool is_binary(HighsInt j) const { return binary[j] != 0; }
};

AttemptCtx make_ctx(HighsMipSolver& mipsolver, const uint8_t* binary) {
    assert(binary != nullptr && "FprConfig::binary_mask must be set");
    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();
    return AttemptCtx{
        mipsolver,
        model,
        mipdata,
        mipdata->ARstart_,
        mipdata->ARindex_,
        mipdata->ARvalue_,
        model->col_lower_,
        model->col_upper_,
        model->col_cost_,
        model->row_lower_,
        model->row_upper_,
        model->integrality_,
        mipdata->feastol,
        model->sense_ == ObjSense::kMinimize,
        model->num_col_,
        model->num_row_,
        binary,
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
// worker, scylla, fpr_lp) pair a stable `cfg.csc` and a
// stable `mipsolver` with the scratch's lifetime — see the lifetime
// comment on `FprConfig::scratch` in `fpr_core.h`.
PropEngine& acquire_engine(FprScratch& scratch, const AttemptCtx& c, const CscMatrix& csc) {
    std::optional<PropEngine>& engine_opt = scratch.prop_engine;
    const bool engine_valid =
        engine_opt.has_value() && engine_opt->ncol() == c.ncol && engine_opt->nrow() == c.nrow &&
        engine_opt->ar_start() == c.ar_start.data() &&
        engine_opt->ar_index() == c.ar_index.data() &&
        engine_opt->ar_value() == c.ar_value.data() &&
        engine_opt->csc_start() == csc.col_start.data() &&
        engine_opt->csc_row() == csc.col_row.data() &&
        engine_opt->csc_val() == csc.col_val.data() && engine_opt->col_lb() == c.col_lb.data() &&
        engine_opt->col_ub() == c.col_ub.data() && engine_opt->row_lo() == c.row_lo.data() &&
        engine_opt->row_hi() == c.row_hi.data() &&
        engine_opt->integrality() == c.integrality.data() && engine_opt->feastol() == c.feastol;
    if (!engine_valid) {
        engine_opt.emplace(c.ncol, c.nrow, c.ar_start.data(), c.ar_index.data(), c.ar_value.data(),
                           csc, c.col_lb.data(), c.col_ub.data(), c.row_lo.data(), c.row_hi.data(),
                           c.integrality.data(), c.feastol);
    } else {
        engine_opt->reset();
    }
    // Arm the engine's own wall-clock poll (issue #151) on both paths: the
    // cached engine survives across attempts, so a `set_deadline` only on
    // the emplace path would leave every attempt after the first
    // propagating against a null deadline.  `deadline_of` is one of the two
    // sanctioned constructors, so the fixpoint stops against exactly the
    // clock and limit the DFS loop below polls.
    engine_opt->set_deadline(deadline_of(c.mipsolver));
    return *engine_opt;
}

// Strategy-aware value selection.  `cfg.strategy` must be non-null — issue
// #120 deleted the legacy null-strategy branch (scores-based ranking plus a
// hint/goodobj value fallback): no production caller ever left `strategy`
// null, and no test reached the branch either, so keeping it as an
// unreachable fallback was strictly dead code with a maintenance cost.
// Pure (no state outside its arguments); rebuild fresh in each begin/step/finish.
double choose_fix_value(HighsInt j, const FprConfig& cfg, const AttemptCtx& c, PropEngine& E,
                        const CscMatrix& csc, Rng& rng) {
    assert(cfg.strategy != nullptr && "FprConfig::strategy must be set (issue #120)");
    return choose_value(j, E.var(j).lb, E.var(j).ub, is_integer(c.integrality, j), c.minimize,
                        c.col_cost[j], cfg.strategy->val_strategy, rng, cfg.lp_ref, c.row_lo.data(),
                        c.row_hi.data(),
                        E.activities_initialized() ? E.min_activity_data() : nullptr,
                        E.activities_initialized() ? E.max_activity_data() : nullptr, &csc);
}

double compute_alt(HighsInt j, double preferred, const AttemptCtx& c, PropEngine& E) {
    if (c.is_binary(j)) {
        return (preferred < 0.5) ? 1.0 : 0.0;
    }
    double alt = (std::abs(preferred - E.var(j).lb) < c.feastol) ? E.var(j).ub : E.var(j).lb;
    if (is_integer(c.integrality, j)) {
        alt = std::round(alt);
    }
    return alt;
}

bool is_row_violated_in_ctx(HighsInt i, double lhs, const AttemptCtx& c) {
    return is_row_violated(lhs, c.row_lo[i], c.row_hi[i], c.feastol);
}

}  // namespace

// ---------------------------------------------------------------------------
// fpr_attempt_begin
// ---------------------------------------------------------------------------

// Cognitive complexity 80 (threshold 25).  Kept whole: Phases 1-2 of the paper's
// Fix-Propagate-Repair (Fig. 4) — variable order, value selection, propagation and DFS seeding —
// expressed as one resumable state machine so an attempt can pause at the budget gate. Decomposing
// it would move work across a worker's inner loop, and the closeout takes no unmeasured performance
// risk; the standards also rank fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void fpr_attempt_begin(FprAttemptState& state, HighsMipSolver& mipsolver, const FprConfig& cfg,
                       Rng& rng, int attempt_idx) {
    assert(cfg.scratch != nullptr && "fpr_attempt_begin requires cfg.scratch");
    FprScratch& scratch = *cfg.scratch;
    const AttemptCtx c = make_ctx(mipsolver, cfg.binary_mask);

    // Lifecycle reset.
    state = FprAttemptState{};
    state.ncol = c.ncol;
    state.nrow = c.nrow;
    state.attempt_idx = attempt_idx;

    if (c.ncol == 0 || c.nrow == 0) {
        // Degenerate model — no DFS to do.  Leave phase = kIdle so a
        // subsequent finish takes its `ncol == 0 || nrow == 0` guard, which
        // since #155 is the only return that hands back an empty solution.
        // Match the legacy fpr_attempt early-return shape.
        state.phase = FprAttemptState::Phase::kIdle;
        return;
    }

    // The lifecycle API requires cfg.csc — one-shot callers go via
    // `fpr_attempt` which builds a local CSC.  Persistent callers (the
    // FPR worker) all carry a stable cfg.csc.
    assert(cfg.csc != nullptr && "fpr_attempt_begin requires cfg.csc");
    const CscMatrix& csc = *cfg.csc;
    assert(cfg.strategy != nullptr && "FprConfig::strategy must be set (issue #120)");

    // --- Phase 1: variable ranking -------------------------------------------------
    auto& var_order = scratch.var_order;
    var_order.clear();
    if (cfg.precomputed_var_order != nullptr) {
        var_order.assign(cfg.precomputed_var_order,
                         cfg.precomputed_var_order + cfg.precomputed_var_order_size);
    } else {
        var_order = compute_var_order(mipsolver, cfg.strategy->var_strategy, rng, cfg.lp_ref);
    }
    state.var_order_size = static_cast<HighsInt>(var_order.size());

    // Ensure scratch.lhs_cache has capacity for finish().
    scratch.lhs_cache.resize(c.nrow);

    // --- Acquire PropEngine (resets if cached engine is from a previous attempt) ---
    // NOLINT rationale: `E` is the paper's own symbol for this object
    // (Fig. 5), used under that name in the surrounding prose in
    // fpr_core.h, repair_search.h and fpr_strategies.h, and — for the
    // primary engine — as the parameter name of repair_search() itself.
    // Lower-casing only the locals would split one symbol across two
    // spellings; renaming the documentation too would cost the mapping
    // back to the paper, which the standards rank above naming.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine& E = acquire_engine(scratch, c, csc);

    // No E.sol(j) seeding here (issue #122): a prior block wrote a
    // deterministic-or-random starting solution into every column before
    // Phase 2, but nothing between `acquire_engine`'s reset() (which
    // already zeros the array) and the final extraction ever reads a
    // column's value before writing it — `E.fix()`/propagation auto-fix
    // overwrite every column the DFS visits, and the Phase 2.5 fill loop
    // in `fpr_attempt_finish` overwrites every column it does not. Confirmed
    // both by that code-path argument and empirically (two `fpr_attempt`
    // runs with wildly different seeds produced identical `fixed`/`sol`
    // state on every column PropEngine actually touched). Removing this
    // also drops the RNG draws the random-start branch (attempt_idx > 0,
    // no caller-supplied seed) used to make, so downstream draws in this
    // and later attempts shift — expected, not a regression.
    auto is_int = [&](HighsInt j) { return is_integer(c.integrality, j); };

    // Shuffle top 30% of ranking for diversity on later attempts.
    if (attempt_idx > 0) {
        HighsInt shuffle_len = std::max(HighsInt{1}, c.ncol * 3 / 10);
        std::shuffle(var_order.begin(), var_order.begin() + shuffle_len, rng);
    }

    // Row activities are needed by two independent consumers, and either
    // one alone arms them: `val_loosedyn`'s dynamic lock counts, and --
    // since #124 -- `repair_walk`, whose whole violation measure *is* the
    // activity range (paper Sect. 5: "the very same quantities are needed
    // for constraint propagation as well, a fact that is exploited by our
    // implementation").  Arming them changes nothing else: `choose_value`
    // reads `min_act`/`max_act` on the `kLoosedyn` branch and on no other,
    // so a repairing mode paired with any other value strategy selects
    // exactly the values it did before -- it only pays for the O(nnz)
    // initialization and the incremental `update_activities` on each
    // fix/tighten.
    if (cfg.strategy->val_strategy == ValStrategy::kLoosedyn || mode_repairs(cfg.mode)) {
        E.init_activities();
    }

    // Trivially-roundable fixings (paper Section 6).
    if (!c.mipdata->uplocks.empty()) {
        const auto& uplocks = c.mipdata->uplocks;
        const auto& downlocks = c.mipdata->downlocks;
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
    // Verdict deliberately discarded (pre-existing; propagate() becoming
    // [[nodiscard]] under #127 is what makes this explicit now). Nothing
    // below reads it: the DFS root-node seeding just past this point walks
    // var_order looking for the first unfixed integer and does not consult
    // whether this trivial-fixings round reached a full fixpoint, ran into
    // the budget or the wall clock (kBudgetExhausted / kDeadlineExpired --
    // sound either way, see PropResult), or proved the model inconsistent
    // (kInfeasible) before it could finish.
    // The clock case needs nothing extra here: the round is now bounded by
    // `kPropagateDeadlinePollWork` past an expiry (#151), and `step` then
    // breaks out on its first propagating node (or, in a non-propagating
    // mode, on its own `kDeadlinePollNodes` poll).
    // An undetected kInfeasible here is not silently wrong: any column left
    // with lb > ub fails every subsequent E.fix() the DFS tries on it, so
    // the attempt backtracks and eventually reports failed rather than
    // producing an unsound result -- just later and less directly than
    // catching it here would. Not fixed in #125/#127; out of scope for both.
    static_cast<void>(E.propagate(-1));

    // --- Phase 2 setup -------------------------------------------------------------
    state.dynamic_var = is_dynamic_var_strategy(cfg.strategy->var_strategy);
    state.do_propagate = mode_propagates(cfg.mode);
    state.do_backtrack = mode_backtracks(cfg.mode);
    // Fig. 1's `repair` parameter.  `mode_repairs` is exactly the paper's
    // three repair-enabled presets (dfsrep / dive / diveprop); it excludes
    // `kRepairSearch`, whose repair procedure (Fig. 5) still runs only at
    // the leaf.  Moving *that* one into the tree needs `repair_search` to
    // work on a partial assignment rather than a complete one, and that
    // is still open: #130 and #131 fixed two defects *inside* Fig. 5 (the
    // stall gate, and the disjunction ignoring the repair move) without
    // changing which assignment it runs on.
    state.do_repair = mode_repairs(cfg.mode);
    state.node_limit = c.ncol + 1;
    state.var_order_cursor = 0;
    state.nodes_visited = 0;
    state.found_complete = false;

    auto& dfs_stack = scratch.dfs_stack;
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
        double pref = choose_fix_value(first_var, cfg, c, E, csc, rng);
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

// Cognitive complexity 26 (threshold 25).  Kept whole: one resumable DFS step of Fig. 4, including
// the per-call budget gate that lets an in-flight attempt pause and resume. Decomposing it would
// move work across a worker's inner loop, and the closeout takes no unmeasured performance risk;
// the standards also rank fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
FprStepResult fpr_attempt_step(FprAttemptState& state, HighsMipSolver& mipsolver,
                               const FprConfig& cfg, Rng& rng, size_t effort_remaining) {
    assert(state.phase == FprAttemptState::Phase::kDfs &&
           "fpr_attempt_step called outside kDfs phase");
    assert(cfg.scratch != nullptr);
    assert(cfg.csc != nullptr);

    FprScratch& scratch = *cfg.scratch;
    const AttemptCtx c = make_ctx(mipsolver, cfg.binary_mask);
    const CscMatrix& csc = *cfg.csc;
    // NOLINT rationale: `E` is the paper's own symbol for the primary
    // propagation engine (Fig. 5), used under that name in the prose in
    // fpr_core.h, repair_search.h and fpr_strategies.h and as the
    // parameter name of repair_search() itself.  Lower-casing only the
    // locals would split one symbol across two spellings.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine& E = *scratch.prop_engine;
    auto& dfs_stack = scratch.dfs_stack;
    auto& var_order = scratch.var_order;

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
    // `attempt_budget`).  Crucially: gate on the *delta* `E.effort() -
    // effort_at_call_start`, not on absolute `E.effort()`.  After a
    // paused attempt resumes from kBudgetGate, `E.effort()` is already
    // at the previous call's slice high-water mark; comparing against
    // an absolute target derived from current effort would treat the
    // DFS as already-exhausted on entry and exit immediately, making
    // forward progress impossible (the bug that hangs `infeasible-mip0`
    // when run alongside FJ/LocalMIP/Scylla).  cfg.max_effort bounds
    // neither this loop nor Phase 3 under the lifecycle API: it is the
    // one-shot `fpr_attempt` wrapper's DFS gate and nothing else (issue
    // #156 removed the Phase 3 caps derived from it, which arrived as 0
    // on every attempt whose effort had outgrown it).
    const size_t effort_at_call_start = E.effort();
    // Target as a delta from this call's start, not an absolute.
    const size_t effort_target_delta = effort_remaining;

    // The wall clock, alongside the effort gate above (issue #117).
    // `effort_remaining` is the caller's slice, and every caller sizes it
    // from its effort option — `HeuristicBudget::attempt_cap` for
    // `FprWorker`, the remainder of a pump attempt for `ScyllaWorker`, the
    // whole attempt budget for `fpr_lp` — so a large option makes this one
    // loop the coarsest indivisible unit in the solve, and the deadline
    // only as tight as it is long.  Measured at 1.9x a 30 s limit on
    // `rail02` before this poll existed.
    const Deadline deadline = deadline_of(mipsolver);
    HighsInt nodes_since_poll = 0;

    while (!dfs_stack.empty() && state.nodes_visited < state.node_limit && !state.found_complete &&
           (E.effort() - effort_at_call_start) < effort_target_delta) {
        if (++nodes_since_poll >= kDeadlinePollNodes) {
            nodes_since_poll = 0;
            if (deadline.expired()) {
                // Leave the attempt alive: the stack is non-empty (the
                // loop condition just held), so the verdict test below
                // reports `kBudgetGate` and the DFS resumes on the next
                // call — which, for every caller, does not come, because
                // its own deadline poll fires first.  Nothing here decides
                // that; a paused attempt is simply what "stop now" means
                // mid-DFS, and it is what the one-shot wrapper turns into
                // a not-feasible verdict — carrying Phase 2.5's point
                // since #155, not an empty result.
                break;
            }
        }
        auto node = dfs_stack.back();
        dfs_stack.pop_back();
        ++state.nodes_visited;

        E.backtrack_to(node.vs_mark, node.sol_mark, node.act_mark, node.pq_mark);
        state.var_order_cursor = node.cursor_reset;

        // Node processing, Fig. 1 lines 4-10, in the paper's own order.
        // `infeas` is the figure's variable: it starts as the verdict of
        // `Apply(fixing, P)` and is then handed to propagation, to repair,
        // and finally to the backtrack decision.  Reading it as "prune
        // here" at any earlier point is the defect issue #124 names.
        bool infeas = !E.fix(node.var, node.val);

        // The rest of `Apply(fixing, P)`: the fixing is also infeasible
        // when it leaves one of the fixed column's rows unsatisfiable by
        // *any* completion of the current domain (issue #124).
        // `PropEngine::fix` alone answers a narrower question -- is the
        // value inside the column's own domain -- and never looks at a
        // row, which is why `dive` (propagation off, so nothing else can
        // report an infeasible node) could never reach the repair below,
        // although the paper calls `dive` "an incremental repair strategy
        // that constructs a complete solution in a single big dive".
        //
        // Gated on `do_repair` rather than run unconditionally: in a
        // non-repairing mode `infeas` feeds only the backtrack decision,
        // where propagation already supplies it, and the activity arrays
        // this reads are armed for exactly the repairing modes.  Arming
        // them in `dfs`/`repairsearch` too, to prune a fully-fixed
        // violated row marginally sooner, is a cost with no repair to pay
        // for it and is not this issue's business.
        if (!infeas && state.do_repair) {
            size_t apply_effort = 0;
            infeas = any_violated_row_in_column(E, node.var, apply_effort);
            E.add_effort(apply_effort);
        }

        if (!infeas && state.do_propagate) {
            // Only a proven inconsistency makes the node infeasible (issue
            // #127).  Budget exhaustion (`kBudgetExhausted`) is sound but
            // incomplete propagation -- the DFS continues with whatever
            // was deduced so far rather than discarding a subtree that
            // may well be feasible.  Neither truncation may reach the
            // repair call below either: repair is what Fig. 1 does with a
            // *refuted* node, and running it on a node that merely ran out
            // of budget would spend the walk's whole step limit repairing
            // nothing.
            const PropResult pr = E.propagate(node.var);
            if (pr == PropResult::kDeadlineExpired) {
                // The fixpoint's own poll fired (issue #151) -- this
                // loop's poll arriving from one level down.  Stop here
                // rather than carrying on for up to `kDeadlinePollNodes`
                // more nodes, and stop the way the pre-node
                // `deadline.expired()` break above stops: that one breaks
                // *before* popping, so push `node` back to leave a paused
                // stack of exactly the same shape.  This is not a prune --
                // nothing was refuted, the node is simply unfinished, and a
                // resume re-runs it from its own marks like any other
                // (`backtrack_to` then `fix`) -- which is the whole reason
                // #151 gave the clock a state distinct from `kInfeasible`.
                dfs_stack.push_back(node);
                break;
            }
            infeas = pr == PropResult::kInfeasible;
        }

        // Fig. 1 lines 7-8: `if infeas and repair: infeas = RepairWalk(P)`.
        // The repair runs on the *partial* assignment this node's domain
        // encodes -- see `repair_walk` -- which is the whole of issue #124.
        // Bounded three ways so it cannot become a new indivisible unit
        // between two deadline polls: the paper's own per-call step limit,
        // `repair_walk`'s own `kRepairWalkBudgetPerNnz` effort valve
        // (deliberately internal -- neither the call slice nor the attempt
        // budget bounds `E.effort()` in a way that could be handed in
        // here without silently switching the repair off mid-attempt), and
        // the same `Deadline` this loop polls, which `repair_walk` polls
        // once per step.
        if (infeas && state.do_repair) {
            size_t walk_effort = 0;
            const bool repaired = repair_walk(E, cfg.walksat_iterations, cfg.repair_noise, rng,
                                              walk_effort, scratch.repair_walk, deadline);
            // Charge into the engine's own counter, which is what both the
            // loop's budget gate above and `state.effort_consumed` below
            // read -- an in-tree repair that reported its effort anywhere
            // else would be work no gate in the attempt can see.
            E.add_effort(walk_effort);
            infeas = !repaired;
        }

        // Fig. 1 lines 9-10: `if infeas and backtrackOnInfeas: Backtrack`.
        // Popping the next node *is* the backtrack here -- in a
        // backtracking mode the sibling this node's parent pushed is
        // directly underneath it on the stack.
        if (infeas && state.do_backtrack) {
            continue;
        }

        // Fig. 1 line 11: `branches = Branch(P)`.  Note this is reached
        // with `infeas` still true in a non-backtracking mode: `dive` and
        // `diveprop` construct "a complete solution in a single big dive"
        // and are not entitled to stop early, which is why a diveprop
        // whose propagation failed used to return failure exactly where
        // the paper's would have carried on.
        auto [next_var, next_idx] = find_next_unfixed_int();

        if (next_var < 0) {
            // Fig. 1 lines 12-16: no branches left.  Line 14's backtrack
            // has already run for every mode that has one -- the
            // `if (infeas && state.do_backtrack)` above is the same test
            // and nothing between the two writes `infeas` -- so this
            // states the invariant rather than re-testing it.
            assert(!(infeas && state.do_backtrack) &&
                   "Fig. 1 line 14 already backtracked this node");

            // Non-backtracking mode.  `found_complete` is set even on a
            // still-infeasible leaf, and that is **not** a deviation from
            // Fig. 1: the figure ends at line 16.  Phase 2.5 and Phase 3
            // are a local post-process whose precondition has always been
            // `found_complete`, declared as "a leaf with every integer
            // fixed" (see `FprAttemptState` in fpr_core.h) and never as
            // "Fig. 1's running `infeas` was false".  Those two were
            // indistinguishable until #124 gave a fixing a way to set
            // `infeas` at all; reading it the other way is what would be
            // new.  So lines 9-16 are transcribed exactly and the only
            // difference from the paper is the leaf post-process that
            // always existed.
            //
            // The two non-backtracking modes are not affected alike.  For
            // `dive` this *restores* what every dive did before the
            // activity half of `Apply` existed -- which is what the
            // recorded benchmark numbers were measured with.  For
            // `diveprop` it is genuinely new: a refuted node used to end
            // the attempt outright, so no refuted diveprop ever reached
            // Phase 3.  Same rule and same precondition, different blast
            // radius -- four shipped `dive` configurations
            // (`kInitialFprConfigs[2]`, Scylla's `kFprConfigs[2]`, and the
            // `ZerocoreDive` / `LpDive` arms of `kLpArmTable`) against
            // five `diveprop` ones (`kInitialFprConfigs[5]` and the four
            // `*Diveprop` arms).
            //
            // What this leaves is **one verdict site**.  Fig. 1's
            // `infeas` feeds the repair call and the backtrack decision
            // and nothing else; whether anything is *returned* is decided
            // solely by `fpr_attempt_finish`'s point re-check over every
            // row.  That is also what makes
            // `any_violated_row_in_column`'s one-column scan sound: it is
            // a correct definition of `Apply`'s verdict and an incorrect
            // one of "the problem is infeasible", and this deletes the
            // only consumer that read the per-node fact as a global one.
            // `infeas` is non-global for two further reasons that predate
            // #124 and that the same re-check also covers: Phase 1's
            // batch fixings bypass `Apply` entirely, so a row they both
            // complete and violate is invisible to it and to propagation
            // (which skips a row with no unfixed columns); and a repair's
            // collateral damage on rows no later column touches is never
            // re-reported.
            state.found_complete = true;
            break;
        }

        double pref = choose_fix_value(next_var, cfg, c, E, csc, rng);
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

// Cognitive complexity 41 (threshold 25).  Kept whole: Phase 3 hand-off to WalkSAT / RepairSearch
// plus the teardown every early exit in Fig. 4 shares. Decomposing it would move work across a
// worker's inner loop, and the closeout takes no unmeasured performance risk; the standards also
// rank fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
HeuristicResult fpr_attempt_finish(FprAttemptState& state, HighsMipSolver& mipsolver,
                                   const FprConfig& cfg, Rng& rng) {
    assert(cfg.scratch != nullptr);

    FprScratch& scratch = *cfg.scratch;
    const AttemptCtx c = make_ctx(mipsolver, cfg.binary_mask);

    // Degenerate model from begin() — short-circuit cleanly.
    if (c.ncol == 0 || c.nrow == 0) {
        state.phase = FprAttemptState::Phase::kIdle;
        return {};
    }

    assert(cfg.csc != nullptr);
    const CscMatrix& csc = *cfg.csc;
    // NOLINT rationale: `E` is the paper's own symbol for the primary
    // propagation engine (Fig. 5), used under that name in the prose in
    // fpr_core.h, repair_search.h and fpr_strategies.h and as the
    // parameter name of repair_search() itself.  Lower-casing only the
    // locals would split one symbol across two spellings.
    // NOLINTNEXTLINE(readability-identifier-naming)
    PropEngine& E = *scratch.prop_engine;

    auto is_int = [&](HighsInt j) { return is_integer(c.integrality, j); };

    // Phase 2.5: fix remaining unfixed variables (continuous + residual integers).
    //
    // This runs whether or not the DFS reached a leaf (issue #155).  Sect.
    // 2.3 of Mexi et al. is explicit that fix-and-propagate produces an
    // integer point even when it ran into an infeasibility: "In the latter
    // case, fix-and-propagate continues in order to produce an integer
    // vector by ignoring any constraint that would lead to empty domains.
    // At the end of the propagation, all remaining unfixed variables are
    // fixed to their values in the fractional reference solution or its
    // projection on to their domain.  The procedure always produces an
    // integer-feasible, but not necessarily LP-feasible, solution."  That
    // sentence is what makes the `!found_complete` case determined rather
    // than a guess: the fill is the same fill, and the columns the DFS
    // never decided are exactly the "remaining unfixed variables".
    //
    // A `!found_complete` attempt is still a *failed* one and returns
    // below without ever reaching Phase 3 — whose precondition has always
    // been a leaf with every integer fixed — but it returns carrying the
    // point, which is what Algorithm 1.1 line 12 hands to lines 14-16.
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
                double fallback = (cfg.cont_fallback != nullptr) ? cfg.cont_fallback[j] : 0.0;
                E.sol(j) = finite_clamp_helper(fallback, lo, hi);
            }
        } else {
            E.sol(j) = choose_fix_value(j, cfg, c, E, csc, rng);
            E.sol(j) = std::round(std::max(lo, std::min(hi, E.sol(j))));
        }
        E.sol(j) = std::max(c.col_lb[j], std::min(c.col_ub[j], E.sol(j)));
    }

    auto& solution = scratch.solution;
    solution.assign(E.sol_data(), E.sol_data() + c.ncol);
    size_t total_prop_work = E.effort();

    auto& lhs_cache = scratch.lhs_cache;
    lhs_cache.resize(c.nrow);
    total_prop_work += c.ar_index.size();
    for (HighsInt i = 0; i < c.nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = c.ar_start[i]; k < c.ar_start[i + 1]; ++k) {
            lhs += c.ar_value[k] * solution[c.ar_index[k]];
        }
        lhs_cache[i] = lhs;
    }

    // The DFS never reached a leaf: no Phase 3, no verdict, but the point
    // the fill just completed goes back to the caller (issue #155).  It is
    // charged the row rebuild rather than the untouched `E.effort()` the
    // old shortcut returned, because that is what it now costs — the
    // whole-dispatch and per-attempt gates read this number.  Note the
    // charge is the rebuild and *only* the rebuild: `E.effort()` is
    // `prop_work_`, which nothing but `propagate` and `reset` writes, and
    // the fill runs through `choose_fix_value` and `E.sol(j) = ...`, which
    // touch no counter.  So the fill is real wall time that no effort gate
    // can see.
    //
    // Worth knowing when sizing that charge: of the two halves only the
    // fill buys anything here.  `lhs_cache` is rebuilt above and then
    // discarded on this path, an O(nnz) charge for nothing, and this is
    // now the *common* Scylla round — three of four `kFprConfigs` are
    // backtracking modes that routinely exhaust the `ncol + 1` node
    // budget.  It stays on the shared path deliberately: one code path is
    // harder to break than a second fill-only branch, and splitting it
    // would put the two effort charges out of each other's sight.
    if (!state.found_complete) {
        state.phase = FprAttemptState::Phase::kIdle;
        state.effort_consumed = total_prop_work;
        return HeuristicResult::infeasible_point(std::move(solution), total_prop_work);
    }

    bool feasible = true;
    for (HighsInt i = 0; i < c.nrow; ++i) {
        if (is_row_violated_in_ctx(i, lhs_cache[i], c)) {
            feasible = false;
            break;
        }
    }

    // Phase 3: RepairSearch (Fig. 5) or WalkSAT.
    //
    // The deadline gates entry to both (issue #117).  `step`'s own poll
    // does not cover this: a DFS that reached a leaf *at* the deadline
    // returns `kVerdictReady`, so finish runs, and the shortcut above
    // catches only the attempts that failed.  RepairSearch is the
    // expensive half — `repair_iterations` nodes, each two propagation
    // fixpoints — and takes the deadline itself so it stops between
    // nodes rather than only before the first.  WalkSAT does not, and
    // its residual is bounded by its own internal valve rather than by a
    // poll: a step is O(row degree x column degree), which a dense row
    // makes arbitrarily large, so `walksat_iterations` alone was never
    // the argument -- `kWalkSatBudgetPerNnz * nnz` is.  Note the one
    // direction #156 loosens this: past the crossing the old cap arrived
    // as 0 and the walk broke at step 0, so the unpolled residual here
    // grew from ~0 to that valve.  A poll inside `walksat_repair` is the
    // fix if it ever matters; it did not before, because the walk could
    // not run at all.
    //
    // Neither takes an effort cap from here any more (issue #156).  The
    // only two numbers this site could derive are the per-call DFS slice
    // — already spent by the time a leaf is reached — and
    // `cfg.max_effort`, which under the lifecycle API is not an upper
    // bound on `E.effort()` at all, so past the crossing the cap arrived
    // as 0 and Phase 3 silently became a no-op that still paid its entry
    // scan.  `repair_search` is governed by `cfg.repair_iterations` (each
    // node's two fixpoints already answer to `kPropagateBudgetPerNnz`)
    // and `walksat_repair` by `cfg.walksat_iterations` plus its own
    // internal per-nnz valve, the shape `repair_walk` has used since
    // #124.  What both spend is still charged to the attempt through
    // `total_prop_work`.
    const Deadline deadline = deadline_of(mipsolver);
    if (!feasible && !deadline.expired() && cfg.mode == FrameworkMode::kRepairSearch) {
        size_t rs_effort = 0;
        feasible = repair_search(E, solution, lhs_cache, c.col_lb.data(), c.col_ub.data(),
                                 c.row_lo.data(), c.row_hi.data(), cfg.repair_iterations,
                                 cfg.repair_noise, cfg.repair_track_best, rng, rs_effort, scratch,
                                 deadline, /*stats=*/nullptr);
        total_prop_work += rs_effort;
    } else if (!feasible && !deadline.expired() && mode_repairs(cfg.mode)) {
        size_t walk_effort = 0;
        feasible = walksat_repair(E, solution, lhs_cache, c.col_lb.data(), c.col_ub.data(),
                                  cfg.walksat_iterations, cfg.repair_noise, cfg.repair_track_best,
                                  rng, walk_effort, scratch.walksat);
        total_prop_work += walk_effort;
    }

    // The failure return below hands back the point too (issue #155).
    // What the point *is* on this path is whatever `scratch.solution`
    // holds at the return: `walksat_repair` and `repair_search`
    // mutate it in place and, under `repair_track_best`, restore their own
    // best state into it before returning.  So it is the leaf as Phase 3
    // left it, not the pre-repair leaf — a distinction with no consumer
    // (the pump wants a direction, and Phase 3's best-known point is at
    // least as good a one), but not one to misdescribe.
    // Phase 3's answer is not the verdict; this re-check is (see
    // `fpr_attempt_step`, which names this the one verdict site).  A
    // `true` from `walksat_repair` / `repair_search` says only that their
    // own `lhs_cache` had no violated row left, so the point is re-checked
    // against every row before anything is returned, and the two answers
    // are folded into one `feasible` rather than into two returns.
    //
    // **One return, deliberately.** A separate one for "Phase 3 said yes
    // and a row disagreed" is a return no fixture can drive: for
    // `walksat_repair` it is bug-or-nothing, since it returns
    // `violated.empty()` on this very cache; for `repair_search` it is
    // reachable only on a drift between the cache it maintains
    // incrementally through `apply_move` and this recomputation from
    // `solution` -- which is exactly what the re-check exists to catch,
    // and so is unreachable-as-far-as-we-know rather than unreachable.
    // A second return would therefore be one no test could pin and a
    // mutation could silently delete.  Folded in, every failure return in
    // this function is driven by a test.
    if (feasible) {
        for (HighsInt i = 0; i < c.nrow; ++i) {
            if (is_row_violated_in_ctx(i, lhs_cache[i], c)) {
                feasible = false;
                break;
            }
        }
    }

    if (!feasible) {
        state.phase = FprAttemptState::Phase::kIdle;
        state.effort_consumed = total_prop_work;
        return HeuristicResult::infeasible_point(std::move(solution), total_prop_work);
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
// One-shot callers (tests, scylla, fpr_lp) keep this entry point.
// It runs begin → step (uncapped) → finish in sequence on a local state,
// and accepts a null cfg.scratch by routing through a function-local scratch
// (matches the pre-#77 contract for those callers).

HeuristicResult fpr_attempt(HighsMipSolver& mipsolver, const FprConfig& cfg, Rng& rng,
                            int attempt_idx) {
    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return {};
    }

    FprScratch local_scratch;
    CscMatrix owned_csc;
    if (cfg.csc == nullptr) {
        owned_csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);
    }

    FprConfig effective_cfg = cfg;
    if (effective_cfg.scratch == nullptr) {
        effective_cfg.scratch = &local_scratch;
    }
    if (effective_cfg.csc == nullptr) {
        effective_cfg.csc = &owned_csc;
    }
    // Same fallback shape as `csc`/`scratch` above: a one-shot caller that
    // took no dispatch snapshot gets one here.  Callers inside a parallel
    // region (fpr_lp, scylla, FprWorker) always set it and skip this.
    std::vector<uint8_t> owned_binary;
    if (effective_cfg.binary_mask == nullptr) {
        owned_binary = build_binary_mask(mipsolver);
        effective_cfg.binary_mask = owned_binary.data();
    }

    FprAttemptState state;
    fpr_attempt_begin(state, mipsolver, effective_cfg, rng, attempt_idx);

    // Single-shot DFS gated by `cfg.max_effort` — matches the pre-#77
    // contract for one-shot callers (scylla / fpr_lp / tests).
    // The `if` (not a `while`) reflects the actual control flow: step
    // either returns `kVerdictReady` or `kBudgetGate`, in which case we
    // force a `kReadyToFinish`.  Either way, exactly one step call, and
    // either way finish runs Phase 2.5 and returns a point — since #155 a
    // `found_complete == false` attempt is an infeasible *point*, not an
    // empty `failed`, which is what lets Scylla's pump advance on it.
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
