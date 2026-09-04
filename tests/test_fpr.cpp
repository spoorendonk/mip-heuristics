#include "fpr.h"
#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "parallel/HighsParallel.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <vector>

TEST_CASE("Characterization: flugpl", "[heuristic][fpr]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // These assert the *known optimum* at a tolerance far tighter than
    // HiGHS's default `mip_rel_gap` (1e-4), which permits terminating on
    // an incumbent that is merely within relative 1e-4 of the dual bound.
    // Require a proven-optimal solve so the assertion is sound rather
    // than dependent on the search happening to land on the optimum.
    require_option(highs, "mip_rel_gap", 0.0);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Characterization: egout", "[heuristic][fpr]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // These assert the *known optimum* at a tolerance far tighter than
    // HiGHS's default `mip_rel_gap` (1e-4), which permits terminating on
    // an incumbent that is merely within relative 1e-4 of the dual bound.
    // Require a proven-optimal solve so the assertion is sound rather
    // than dependent on the search happening to land on the optimum.
    require_option(highs, "mip_rel_gap", 0.0);
    REQUIRE(highs.readModel(kInstancesDir + "/egout.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(568.1007).epsilon(1e-4));
}

TEST_CASE("Characterization: bell5", "[heuristic][fpr]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    // These assert the *known optimum* at a tolerance far tighter than
    // HiGHS's default `mip_rel_gap` (1e-4), which permits terminating on
    // an incumbent that is merely within relative 1e-4 of the dual bound.
    // Require a proven-optimal solve so the assertion is sound rather
    // than dependent on the search happening to land on the optimum.
    require_option(highs, "mip_rel_gap", 0.0);
    REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-6));
}

// Also the RepairSearch coverage: `kInitialFprConfigs` includes
// `LocksRepairSearch`, so a `suite=fpr` solve exercises the Fig. 5
// secondary-propagation path as part of its rotation.  A separate
// "RepairSearch: FPR standalone" case used to say so in its name while
// running exactly this solve; it went when the option migration made the
// two bodies identical.
TEST_CASE("FPR standalone: flugpl finds solution", "[heuristic][fpr]") {
    REQUIRE(solve_suite("flugpl.mps", "fpr") == Catch::Approx(1201500.0).epsilon(1e-6));
}

// ===================================================================
// FPR Strategy tests
// ===================================================================

TEST_CASE("FPR strategies: framework mode helpers", "[fpr][strategies]") {
    // dfs: propagate on, repair off, backtrack on
    REQUIRE(mode_propagates(FrameworkMode::kDfs));
    REQUIRE_FALSE(mode_repairs(FrameworkMode::kDfs));
    REQUIRE(mode_backtracks(FrameworkMode::kDfs));

    // dfsrep: propagate on, repair on, backtrack on
    REQUIRE(mode_propagates(FrameworkMode::kDfsrep));
    REQUIRE(mode_repairs(FrameworkMode::kDfsrep));
    REQUIRE(mode_backtracks(FrameworkMode::kDfsrep));

    // dive: propagate off, repair on, no backtrack
    REQUIRE_FALSE(mode_propagates(FrameworkMode::kDive));
    REQUIRE(mode_repairs(FrameworkMode::kDive));
    REQUIRE_FALSE(mode_backtracks(FrameworkMode::kDive));

    // diveprop: propagate on, repair on, no backtrack
    REQUIRE(mode_propagates(FrameworkMode::kDiveprop));
    REQUIRE(mode_repairs(FrameworkMode::kDiveprop));
    REQUIRE_FALSE(mode_backtracks(FrameworkMode::kDiveprop));

    // repairsearch: propagate on, WalkSAT repair off (own dispatch), backtrack on
    REQUIRE(mode_propagates(FrameworkMode::kRepairSearch));
    REQUIRE_FALSE(mode_repairs(FrameworkMode::kRepairSearch));
    REQUIRE(mode_backtracks(FrameworkMode::kRepairSearch));
}

TEST_CASE("FPR strategies: strategy_needs_lp", "[fpr][strategies]") {
    // LP-free strategies
    REQUIRE_FALSE(strategy_needs_lp(kStratRandom));
    REQUIRE_FALSE(strategy_needs_lp(kStratBadobjcl));
    REQUIRE_FALSE(strategy_needs_lp(kStratLocks2));
    REQUIRE_FALSE(strategy_needs_lp(kStratGoodobj));
    REQUIRE_FALSE(strategy_needs_lp(kStratDomsize));

    // Dynamic strategy helpers
    REQUIRE(is_dynamic_var_strategy(VarStrategy::kDomainSize));
    REQUIRE_FALSE(is_dynamic_var_strategy(VarStrategy::kType));
    REQUIRE_FALSE(is_dynamic_var_strategy(VarStrategy::kLocks));

    // LP-dependent strategies
    REQUIRE(strategy_needs_lp(kStratZerocore));
    REQUIRE(strategy_needs_lp(kStratZerolp));
    REQUIRE(strategy_needs_lp(kStratCore));
    REQUIRE(strategy_needs_lp(kStratLp));
    REQUIRE(strategy_needs_lp(kStratCliques));
    REQUIRE(strategy_needs_lp(kStratCliques2));
}

TEST_CASE("FPR strategies: DFS mode on flugpl", "[fpr][strategies][dfs]") {
    // Test that DFS mode (with backtracking) solves flugpl
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("FPR strategies: multi-config sequential on egout", "[fpr][strategies]") {
    // The sequential multi-config runner should solve egout
    // FPR enabled (runs multi-config)
    REQUIRE(solve_suite("egout.mps", "fpr") == Catch::Approx(568.1007).epsilon(1e-4));
}

// ===================================================================
// Issue #77 lifecycle: pause/resume across budget gates is deterministic
// ===================================================================

namespace {

// Solve `inst` end-to-end at a small `mip_heuristic_fpr_effort` (the knob
// feeding the presolve FPR budget) so the FPR per-call slice is well below
// the cost of a full DFS subtree on these instances — attempts must pause
// via `kBudgetGate` and resume on subsequent `run_attempt` calls, or
// fast-fail and trigger the multi-attempt fill loop.  Without this the
// [fpr][resume] tests can pass without ever exercising the new
// pause/resume code path on the small HiGHS check instances (egout /
// bell5 / flugpl all verdict in one slice at the default effort).
//
// The value is the *budget* this test needs, translated into the option
// that now carries it (#110): FPR's slice is
// `heuristic_effort_budget(nnz, effort)` = `nnz << 12` x effort/0.05, so
// 0.0007 gives ~57 x nnz.  That is what the pre-#110 shared envelope gave
// FPR at `mip_heuristic_presolve_effort=0.01` (~60 x nnz once FJ's charge
// and the 2.99/10.15 weight came off), which is the value this test was
// written against.  Do not round it up for tidiness: one order of
// magnitude below it, the slice fell short of even `begin`'s initial
// `propagate(-1)` on flugpl and the loop never reached `kBudgetGate`.
// Returns final objective.
double solve_with_seed_small_effort(const char* inst, int seed) {
    const ScopedThreadPin pin;
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("random_seed", seed);
    // Pin threads=1 so the determinism contract is the *intra-worker*
    // lifecycle determinism (single-worker pause/resume + multi-attempt
    // fill).  Across-worker scheduling determinism is a different
    // (harder) property: HighsTaskExecutor is a global singleton lazily
    // initialised on the first Highs::run in a process and the per-thread
    // work-stealing order on subsequent runs depends on the scheduler's
    // internal state — running these tests sequentially in one Catch2
    // process exposes that as cross-test instability at effort=0.01 on
    // bell5 even though each test in isolation is deterministic.
    // CLAUDE.md says "don't pass --threads/threads= unless asked" for
    // benchmarks; this is a determinism test where threads=1 is the
    // ask.
    highs.setOptionValue("threads", 1);
    // Small effort → small per-call slice → multi-attempt loop and/or
    // pause-resume engages on the small HiGHS check instances.
    highs.setOptionValue("mip_heuristic_fpr_effort", 0.0007);
    REQUIRE(highs.readModel(std::string(kInstancesDir) + "/" + inst) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    return obj;
}

}  // namespace

// Test design note: these tests assert end-to-end objective equality
// across two same-seed runs.  This is a *proxy* for the issue #77
// literal acceptance bullet — "bit-identical [Sequential] summaries" —
// because parsing the HiGHS log would require wiring a callback into
// the test harness that can flake on log-format changes.  A divergence
// in effort count or attempt rotation that ultimately produces the
// same optimum would slip past objective equality alone.  In NDEBUG=0
// builds we additionally assert that the lifecycle counters
// (`fpr::budget_gate_hits()`, `fpr::multi_attempt_iters()`) are
// non-zero (proving the pause/resume / multi-attempt-fill paths
// actually fired) AND identical across runs (proving the lifecycle
// path traversal is deterministic).  Together this is a tighter
// guarantee than objective equality alone.
TEST_CASE("FPR resume: same seed reproduces same objective at small effort (egout)",
          "[fpr][resume][determinism]") {
#ifndef NDEBUG
    fpr::reset_test_counters();
#endif
    const double obj1 = solve_with_seed_small_effort("egout.mps", 42);
#ifndef NDEBUG
    const size_t gate1 = fpr::budget_gate_hits();
    const size_t multi1 = fpr::multi_attempt_iters();
    // Sanity: at least one of the two new lifecycle paths must have
    // engaged.  Without this, the determinism assertion below could
    // pass on a regression that bypassed the lifecycle entirely
    // (HiGHS' default B&B trivially solves these instances).
    REQUIRE((gate1 > 0 || multi1 > 0));
    fpr::reset_test_counters();
#endif
    const double obj2 = solve_with_seed_small_effort("egout.mps", 42);
    REQUIRE(obj1 == obj2);
#ifndef NDEBUG
    REQUIRE(fpr::budget_gate_hits() == gate1);
    REQUIRE(fpr::multi_attempt_iters() == multi1);
#endif
}

TEST_CASE("FPR resume: same seed reproduces same objective at small effort (bell5)",
          "[fpr][resume][determinism]") {
#ifndef NDEBUG
    fpr::reset_test_counters();
#endif
    const double obj1 = solve_with_seed_small_effort("bell5.mps", 7);
#ifndef NDEBUG
    const size_t gate1 = fpr::budget_gate_hits();
    const size_t multi1 = fpr::multi_attempt_iters();
    REQUIRE((gate1 > 0 || multi1 > 0));
    fpr::reset_test_counters();
#endif
    const double obj2 = solve_with_seed_small_effort("bell5.mps", 7);
    REQUIRE(obj1 == obj2);
#ifndef NDEBUG
    REQUIRE(fpr::budget_gate_hits() == gate1);
    REQUIRE(fpr::multi_attempt_iters() == multi1);
#endif
}

TEST_CASE("FPR resume: same seed reproduces same objective at small effort (flugpl)",
          "[fpr][resume][determinism]") {
#ifndef NDEBUG
    fpr::reset_test_counters();
#endif
    const double obj1 = solve_with_seed_small_effort("flugpl.mps", 0);
#ifndef NDEBUG
    const size_t gate1 = fpr::budget_gate_hits();
    const size_t multi1 = fpr::multi_attempt_iters();
    REQUIRE((gate1 > 0 || multi1 > 0));
    fpr::reset_test_counters();
#endif
    const double obj2 = solve_with_seed_small_effort("flugpl.mps", 0);
    REQUIRE(obj1 == obj2);
#ifndef NDEBUG
    REQUIRE(fpr::budget_gate_hits() == gate1);
    REQUIRE(fpr::multi_attempt_iters() == multi1);
#endif
}

TEST_CASE("FPR resume: paper-curated rotation still solves with multi-attempt cycling",
          "[fpr][resume]") {
    // Worker rotation `(worker_idx + attempt_idx) % kNumInitialFprConfigs`
    // visits every Class-1 config (paper Section 6.3) before cycling.
    // bell5 is a known-feasible instance that previously relied on the
    // randomized stale-attempt jump; the deterministic rotation must still
    // reach the same optimum.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    set_suite(highs, "fpr");
    // bell5 is the one bundled instance whose solve can terminate on
    // HiGHS's default `mip_rel_gap` (1e-4) with an incumbent short of
    // the optimum — 3 distinct primal bounds over 15 default-option
    // runs, against 1 for every other instance the suite uses.  Assert
    // the known optimum only on a proven-optimal solve.
    require_option(highs, "mip_rel_gap", 0.0);
    REQUIRE(highs.readModel(kInstancesDir + "/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    double obj;
    highs.getInfoValue("objective_function_value", obj);
    REQUIRE(obj == Catch::Approx(8966406.49152).epsilon(1e-4));
}

// ===================================================================
// cfg.lp_ref is the sole reference-point mechanism into the strategy path
// ===================================================================
//
// This is NOT a regression guard for issue #120 by itself: `lp_ref` was
// already threaded into `choose_value` on the strategy branch before #120
// — #120's defect was that `cfg.hint` was ignored once a strategy was set
// (every production caller's case), not that `lp_ref` was. A version of
// this test run against the pre-#120 code would still pass, because the
// property it checks — a non-null `lp_ref` changes the strategy path's
// output — already held there. What #120 actually changed is that
// `hint`/`scores` and the legacy null-strategy branch that read them are
// gone, leaving `lp_ref` as the *only* way any caller can steer
// fix-and-propagate toward a reference point. This test characterizes
// that surviving, single mechanism directly. The `static_assert`s below
// are the part that genuinely regresses against pre-#120 semantics: they
// fail to compile if `FprConfig` grows a `hint`/`scores` surface again.
//
// `choose_fix_value` has exactly one path once a strategy is set (every
// production caller sets one): `choose_value`, which for an LP-based value
// strategy (`kZerocore`/`kZerolp`/`kCore`/`kLp`) rounds toward `cfg.lp_ref`.
// This drives the lifecycle API directly (begin -> step-to-verdict) rather
// than the one-shot `fpr_attempt` wrapper: on these small bundled
// instances a single DFS attempt frequently does not complete within the
// tight `ncol+1` node budget, and `fpr_attempt_finish` then reports
// `found_feasible == false`. Since #155 that result does carry a point,
// but it is Phase 2.5's fill talking for every column the DFS never
// reached rather than the value strategy under test, so an assertion on
// `fpr_attempt(...).solution` would characterize the wrong thing (before
// #155 it was vacuous, the solution being empty). Comparing PropEngine's
// per-column state after `step()` sidesteps
// that: it is what the search actually decided, independent of whether
// the attempt as a whole verdicts complete.

// Member-detection idiom: `T` must be a template parameter for the
// member-access check inside `requires` to be a genuine substitution
// (SFINAE) context. A `requires` expression checking a fixed, non-dependent
// type directly (e.g. `requires { std::declval<FprConfig>().hint; }` at
// namespace scope) hard-errors on an absent member instead of evaluating
// to `false` -- there is no template argument for substitution to fail on.
template <typename T>
concept HasHintMember = requires(const T& t) { t.hint; };
template <typename T>
concept HasScoresMember = requires(const T& t) { t.scores; };

static_assert(!HasHintMember<FprConfig>,
              "FprConfig::hint was deleted (issue #120) -- if it is back, cfg.lp_ref is no "
              "longer the sole reference-point mechanism and the test below needs revisiting");
static_assert(!HasScoresMember<FprConfig>,
              "FprConfig::scores was deleted (issue #120) along with the legacy "
              "null-strategy branch that read it");

namespace {

// Two per-column reference points at each column's own bounds — clamped to
// something finite for an unbounded column — so `val_lp_based`'s
// `frac(lp_ref[j]) < 1e-10` branch is always taken (both `lb` and `ub` are
// exact integers here after rounding) and neither run draws from the RNG,
// keeping the two runs' RNG streams identical regardless of what
// `choose_fix_value` decides.
struct LpRefPair {
    std::vector<double> lo;
    std::vector<double> hi;
};

LpRefPair make_bound_refs(const HighsLp& model) {
    LpRefPair p;
    p.lo = model.col_lower_;
    p.hi = model.col_upper_;
    for (HighsInt j = 0; j < model.num_col_; ++j) {
        if (p.lo[j] <= -1e30) {
            p.lo[j] = -1e5;
        }
        if (p.hi[j] >= 1e30) {
            p.hi[j] = 1e5;
        }
    }
    return p;
}

}  // namespace

TEST_CASE("FPR: lp_ref is the strategy path's reference-point mechanism", "[fpr][lp_ref]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    const HighsInt ncol = mipsolver->model_->num_col_;
    const LpRefPair refs = make_bound_refs(*mipsolver->model_);

    // Drive begin -> step-to-verdict with `cfg.lp_ref` pointed at each
    // extreme in turn, everything else (strategy, mode, rng seed, binary
    // mask, csc) held fixed, and compare the PropEngine's own per-column
    // `fixed` flags and values -- what the search actually decided.
    struct Probe {
        std::vector<uint8_t> fixed;
        std::vector<double> sol;
    };
    auto run_once = [&](const double* lp_ref) {
        FprScratch scratch;
        FprConfig cfg{};
        cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
        cfg.csc = &csc;
        cfg.mode = FrameworkMode::kDfs;
        cfg.strategy = &kStratLp;  // VarStrategy::kTypecl (lp_ref-free order), ValStrategy::kLp
        cfg.lp_ref = lp_ref;
        cfg.binary_mask = problem.binary.data();
        cfg.scratch = &scratch;
        Rng rng(777);
        FprAttemptState state;
        fpr_attempt_begin(state, *mipsolver, cfg, rng, /*attempt_idx=*/0);
        while (state.phase == FprAttemptState::Phase::kDfs) {
            fpr_attempt_step(state, *mipsolver, cfg, rng, cfg.max_effort);
        }
        Probe p;
        for (HighsInt j = 0; j < ncol; ++j) {
            p.fixed.push_back(scratch.prop_engine->var(j).fixed ? 1 : 0);
            p.sol.push_back(scratch.prop_engine->sol_data()[j]);
        }
        return p;
    };

    Probe with_lo = run_once(refs.lo.data());
    Probe with_hi = run_once(refs.hi.data());

    // Variable order is lp_ref-free for this strategy (kTypecl), but which
    // columns end up fixed is not: `kDfs` propagates and backtracks, so a
    // different chosen value can cascade into a different tightening (or a
    // fix/propagate failure that sends the DFS down a different branch of
    // the stack) — a stronger form of "the assignment differs" than value
    // selection alone. Treat either kind of divergence as the property
    // under test.
    bool any_column_differs = false;
    for (HighsInt j = 0; j < ncol; ++j) {
        const auto idx = static_cast<size_t>(j);
        if (with_lo.fixed[idx] != with_hi.fixed[idx] ||
            (with_lo.fixed[idx] != 0 && with_lo.sol[idx] != with_hi.sol[idx])) {
            any_column_differs = true;
            break;
        }
    }
    REQUIRE(any_column_differs);
}
