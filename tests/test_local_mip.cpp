#include "fj.h"
#include "fpr.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "incumbent_sink.h"
#include "local_mip.h"
#include "local_mip_construction.h"
#include "local_mip_worker.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"  // for kSolutionSource* constants
#include "parallel/HighsParallel.h"
#include "rng.h"
#include "scylla.h"
#include "solution_pool.h"
#include "test_common.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ── LocalMIP only: neighborhood search finds feasible solution ──
// There used to be a "standalone" and a "parallel" case per instance,
// differing only in whether FJ was left enabled alongside LocalMIP.
// `suite=local_mip` cannot express that split — and since #92 there is
// one runner, so it named nothing — leaving byte-identical duplicates.

TEST_CASE("LocalMIP only: flugpl", "[heuristic][local_mip]") {
    REQUIRE(solve_suite("flugpl.mps", "local_mip") == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("LocalMIP only: egout", "[heuristic][local_mip]") {
    REQUIRE(solve_suite("egout.mps", "local_mip") == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── LocalMIP cold-start construction phase (issue #75) ────────────────

// Build a tiny feasibility MIP by hand to drive the construction phase
// directly.  Two binary variables x0, x1 and one row:
//
//     x0 + x1 >= 1,   x0 ∈ {0,1},   x1 ∈ {0,1}
//
// The paper's zero-start (value closest to 0 in global bounds) sets
// both to 0, yielding an infeasible assignment.  The greedy sweep
// then picks at least one variable and flips it to 1 because the
// row is violated.  This tests both the zero-start and the
// feasibility-first greedy refinement.
TEST_CASE("LocalMIP construction: feasibility-first sweep repairs tiny MIP",
          "[heuristic][local_mip][construction]") {
    using local_mip_detail::construct_initial_solution;
    using local_mip_detail::ConstructionInputs;

    const HighsInt ncol = 2;
    const HighsInt nrow = 1;
    // Row-major: one row with (x0 + x1) >= 1.
    std::vector<HighsInt> ar_start = {0, 2};
    std::vector<HighsInt> ar_index = {0, 1};
    std::vector<double> ar_value = {1.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1.0, 1.0};
    std::vector<double> row_lo = {1.0};
    std::vector<double> row_hi = {kHighsInf};
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);

    ConstructionInputs inputs;
    inputs.ncol = ncol;
    inputs.nrow = nrow;
    inputs.ar_start = &ar_start;
    inputs.ar_index = &ar_index;
    inputs.ar_value = &ar_value;
    inputs.col_lb = &col_lb;
    inputs.col_ub = &col_ub;
    inputs.row_lo = &row_lo;
    inputs.row_hi = &row_hi;
    inputs.integrality = &integrality;
    inputs.csc = &csc;
    inputs.feastol = 1e-6;

    std::vector<double> solution;
    Rng rng(42);
    // Generous budget: this is a 2-variable model so any positive
    // budget is more than enough.
    construct_initial_solution(inputs, rng, /*max_effort=*/1000, solution);

    REQUIRE(solution.size() == 2);
    // Bounds respected.
    REQUIRE(solution[0] >= 0.0);
    REQUIRE(solution[0] <= 1.0);
    REQUIRE(solution[1] >= 0.0);
    REQUIRE(solution[1] <= 1.0);
    // Integer-valued.
    REQUIRE(solution[0] == Catch::Approx(std::round(solution[0])));
    REQUIRE(solution[1] == Catch::Approx(std::round(solution[1])));
    // Feasibility-first rule: the greedy sweep should have satisfied
    // the one violated row (x0 + x1 >= 1) by flipping at least one
    // variable to 1.
    REQUIRE((solution[0] + solution[1]) >= 1.0 - 1e-9);
}

// Construction starting point is inside bounds even when zero-start
// is infeasible (lb > 0 or ub < 0 forces a non-zero start).
TEST_CASE("LocalMIP construction: zero-start respects bounds with lb > 0 / ub < 0",
          "[heuristic][local_mip][construction]") {
    using local_mip_detail::construct_initial_solution;
    using local_mip_detail::ConstructionInputs;

    const HighsInt ncol = 3;
    const HighsInt nrow = 0;  // no constraints → only zero-start phase runs
    std::vector<HighsInt> ar_start = {0};
    std::vector<HighsInt> ar_index;
    std::vector<double> ar_value;
    std::vector<double> col_lb = {2.0, -5.0, -3.0};  // lb>0 / lb<0 / ub<0
    std::vector<double> col_ub = {5.0, -1.0, -1.0};
    std::vector<double> row_lo;
    std::vector<double> row_hi;
    std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kContinuous,
                                             HighsVarType::kInteger};
    CscMatrix csc = build_csc(ncol, nrow, ar_start, ar_index, ar_value);

    ConstructionInputs inputs;
    inputs.ncol = ncol;
    inputs.nrow = nrow;
    inputs.ar_start = &ar_start;
    inputs.ar_index = &ar_index;
    inputs.ar_value = &ar_value;
    inputs.col_lb = &col_lb;
    inputs.col_ub = &col_ub;
    inputs.row_lo = &row_lo;
    inputs.row_hi = &row_hi;
    inputs.integrality = &integrality;
    inputs.csc = &csc;
    inputs.feastol = 1e-6;

    std::vector<double> solution;
    Rng rng(7);
    construct_initial_solution(inputs, rng, /*max_effort=*/1000, solution);

    REQUIRE(solution.size() == 3);
    // x0: lb=2 (>0), value-closest-to-0 = lb = 2.
    REQUIRE(solution[0] == Catch::Approx(2.0));
    // x1: bounds straddle 0? actually lb=-5, ub=-1 → ub<0, so value closest to 0 = ub = -1.
    REQUIRE(solution[1] == Catch::Approx(-1.0));
    // x2: lb=-3, ub=-1, integer → closest-to-0 = ub = -1.
    REQUIRE(solution[2] == Catch::Approx(-1.0));
}

// End-to-end integration test: drive a full HiGHS solve with FJ / FPR /
// Scylla disabled and only LocalMIP enabled, on a small feasibility
// MIP.  Exercises the cold-start pathway: even if upstream HiGHS
// presolve doesn't find an incumbent, our LocalMIP construction +
// search phase should progress (the key acceptance criterion of issue
// #75).  Using flugpl (a small MIPLIB-like MIP) as the vehicle — the
// optimal is known and the solver chain still has to find it.
TEST_CASE("LocalMIP cold-start: emits non-zero [Sequential] when upstream heuristics off",
          "[heuristic][local_mip][construction][cold-start]") {
    // Reviewer R3-2 (round-3) flagged that the previous version of
    // this test ran full B&B and asserted `obj == 1201500.0` on
    // flugpl — which HiGHS finds via its own LP-driven branching even
    // if `construct_initial_solution` is a no-op.  Here we constrain
    // the solve to the presolve chain via `mip_root_presolve_only` and
    // assert directly on the `[Sequential] heur=local_mip effort=…`
    // line with non-zero effort.  That's the real signal that #75's
    // cold-start construction kicked in.
    const std::vector<std::string> lines = solve_capturing_log("flugpl.mps", [](Highs& h) {
        h.setOptionValue("log_dev_level", 3);
        h.setOptionValue("mip_root_presolve_only", true);
        set_suite(h, "local_mip");
    });

    REQUIRE(heuristic_reported_effort(lines, "local_mip"));
}

// Regression guard for `local_mip::run`'s warm-start path
// (issue #74).  The presolve chain in `mode_dispatch::run_sequential`
// flushes the shared `SolutionPool` into `mipdata->incumbent` only after
// all four heuristics have run.  Before #74 (and before the #75
// construction cold-start), `local_mip::run` bailed out on
// `mipdata->incumbent.empty()`, so an FJ solution sitting in the pool
// was invisible and local_mip's `[Sequential]` line read
// `effort=0 wall_ms=0`.  After the fix, `resolve_worker_start` prefers
// the pool's best entry over `mipdata->incumbent`, so local_mip sees
// FJ's fresh primal as its warm-start base.  The test runs the full
// chain, because `suite=all` is the only value that runs FJ and LocalMIP
// together — FPR runs between them and can add to the pool as well, so
// what is pinned is that both ran with non-zero effort, not that FJ
// specifically filled the pool.  It captures the developer-level log via
// HiGHS's logging callback and asserts that both the `heur=fj` and
// `heur=local_mip` `[Sequential]` lines report non-zero effort.
// `lseu.mps` is chosen because FJ reliably produces a feasible for
// it inside the presolve budget.
TEST_CASE("LocalMIP: warm-starts from pool when FJ finds feasible before it (#74)",
          "[heuristic][local_mip][pool-aware]") {
    const std::vector<std::string> lines = solve_capturing_log("lseu.mps", [](Highs& h) {
        h.setOptionValue("log_dev_level", 3);
        set_suite(h, "all");
        // Force HiGHS to run the full root-presolve chain (fj → local_mip)
        // before any branching, so the [Sequential] lines are guaranteed
        // to appear regardless of whether HiGHS would otherwise shortcut
        // into B&B.  Reviewers (R3) flagged that the test would silently
        // fail if the chain never ran.
        h.setOptionValue("mip_root_presolve_only", true);
    });

    REQUIRE(heuristic_reported_effort(lines, "fj"));
    REQUIRE(heuristic_reported_effort(lines, "local_mip"));
}

// Unit-level regression for #74's pool-aware helper (complements the
// log-based integration test above).  `resolve_worker_start` prefers
// the pool's best over `mipdata->incumbent` and over the cold-start
// construction; the reasoning delegates to `SolutionPool::copy_best`
// for that first branch.  Round-2 reviewers flagged that the
// integration test can't distinguish pool-warm-start from cold-start
// construction (both produce non-zero effort); testing `copy_best`
// directly proves the pool-first branch returns exactly the seeded
// vector, which is the cheap half of #74 to pin down.  The full
// integration-level distinction (did the worker start from the pool
// or construct fresh?) still relies on the `lseu.mps` test above.
TEST_CASE("SolutionPool::copy_best returns exactly the seeded best entry (#74 unit)",
          "[heuristic][local_mip][pool-aware][unit]") {
    SolutionPool pool(/*capacity=*/4, /*minimize=*/true);
    std::vector<double> probe;
    // Empty pool: no best, copy_best returns false and leaves `probe`
    // untouched.
    probe.assign(3, 9.9);  // sentinel to confirm no write
    REQUIRE_FALSE(pool.copy_best(probe));
    REQUIRE(probe == std::vector<double>{9.9, 9.9, 9.9});

    // Seed a worse and a better entry; copy_best must return the better.
    const std::vector<double> worse_sol{1.0, 2.0, 3.0};
    const std::vector<double> better_sol{4.0, 5.0, 6.0};
    REQUIRE(pool.try_add(/*obj=*/100.0, worse_sol, kSolutionSourceFJ).accepted);
    REQUIRE(pool.try_add(/*obj=*/10.0, better_sol, kSolutionSourceLocalMIP).accepted);
    probe.clear();
    REQUIRE(pool.copy_best(probe));
    REQUIRE(probe == better_sol);
}

// ── Distinguish #74 (pool warm-start) vs #75 (cold-start construction) ──
//
// Reviewers R1-8, R2-7, R3-3 (round-3) flagged that the existing
// integration tests assert "local_mip ran with non-zero effort", which
// is true regardless of whether the warm-start came from the shared
// pool (#74) or the paper's cold-start construction (#75).  These two
// tests use the warm-start branch counters in `local_mip.h` to pin
// down which path actually fired in each scenario.
//
// Counter contract (from `resolve_worker_start`):
//   - `pool`: SolutionPool::copy_best returned a vector (warm).
//   - `incumbent`: pool empty, mipdata->incumbent picked up.
//   - `construction`: both empty → paper's Phase A/B ran.

// Scenario A: nothing populates the pool/incumbent before LocalMIP, so
// the cold-start construction must fire on every worker (#75 active,
// #74 unreachable).
//
// State the test asserts on entry: with FJ, FPR, and Scylla disabled
// and `mip_root_presolve_only` set, no upstream heuristic populates
// either the shared pool or `mipdata->incumbent` before LocalMIP runs;
// the construction branch is therefore the only reachable warm-start
// path.  R2-6 / R3-4 round-4 review: assert *both* `pool == 0` AND
// `incumbent == 0` so a future HiGHS presolve change that pre-populates
// the incumbent surfaces as a clean test failure rather than silently
// reaching cold-start through a different (no-op) branch.  The
// `construction >= 1` assertion alone can't tell those apart; the
// purpose of this scenario is the cold-start path is *reachable*, not
// just "construction fired".
TEST_CASE("LocalMIP: cold-start construction fires when pool and incumbent are empty (#75)",
          "[heuristic][local_mip][cold-start][warm-start-counters]") {
    if constexpr (!local_mip::kInstrumented) {
        SKIP("Built with MIP_HEURISTICS_INSTRUMENT=OFF — counters compiled out");
    }
    local_mip::reset_warm_start_counters();
    const ScopedThreadPin pin;
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_root_presolve_only", true);
    set_suite(h, "local_mip");
    // threads=1 is load-bearing for the `pool == 0` assertion below.  The
    // continuous runner resolves each worker's start inside the parallel
    // region, so with several workers a late starter can legitimately
    // warm-start from a *peer LocalMIP worker's* freshly-added solution
    // and trip the pool branch — the old epoch-gated runner resolved all
    // starts before any worker searched, which is what made that
    // assertion unconditional.  One worker restores the precondition the
    // scenario is about: nothing has populated the pool when the start is
    // resolved.
    require_option(h, "threads", 1);
    REQUIRE(h.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    auto counters = local_mip::warm_start_counters();
    // Cold-start construction must have run at least once (one per
    // worker, modulo the cold-start cache de-duplication).
    REQUIRE(counters.construction >= 1);
    // Pool was empty before LocalMIP fired (no upstream heuristic
    // populated it), so the pool branch must not have triggered.
    REQUIRE(counters.pool == 0);
    // Likewise, `mipdata->incumbent` must have been empty: a future
    // HiGHS presolve change that pre-populates the incumbent would
    // otherwise let the warm-start fall into the (different) incumbent
    // branch and silently bypass the cold-start construction this
    // scenario is meant to exercise.
    REQUIRE(counters.incumbent == 0);
}

// Standalone LocalMIP at the default worker count (issue #94).  Distinct
// from Scenario A above, which pins `threads=1` so it can also assert
// `pool == 0`: what this one covers is the multi-worker standalone path
// the heuristic config table made reachable.  At `suite=local_mip` the
// table filters to a single entry, so LocalMIP is dispatched with the
// entire post-FJ envelope and no upstream heuristic has run — the
// cold-start construction is the only way it can obtain a starting
// point, and it must actually produce one on every worker team, not
// just on a single-worker one.
TEST_CASE("LocalMIP standalone: cold-start construction fires at the default worker count",
          "[heuristic][local_mip][cold-start][warm-start-counters]") {
    if constexpr (!local_mip::kInstrumented) {
        SKIP("Built with MIP_HEURISTICS_INSTRUMENT=OFF — counters compiled out");
    }
    local_mip::reset_warm_start_counters();
    const ScopedThreadPin pin;
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_root_presolve_only", true);
    set_suite(h, "local_mip");
    REQUIRE(h.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    auto counters = local_mip::warm_start_counters();
    REQUIRE(counters.construction >= 1);
    // No upstream heuristic ran, so nothing can have seeded the
    // incumbent.  The `pool` branch is deliberately *not* asserted zero:
    // with several workers a late starter may legitimately warm-start
    // from a peer LocalMIP worker's freshly added solution.
    REQUIRE(counters.incumbent == 0);
}

// Scenario B: FJ runs first and populates the pool with a feasible
// solution; LocalMIP must then warm-start from the pool (#74 active,
// #75 unreachable for the worker setup paths).
TEST_CASE("LocalMIP: pool warm-start fires when FJ pre-populates pool (#74)",
          "[heuristic][local_mip][pool-aware][warm-start-counters]") {
    if constexpr (!local_mip::kInstrumented) {
        SKIP("Built with MIP_HEURISTICS_INSTRUMENT=OFF — counters compiled out");
    }
    local_mip::reset_warm_start_counters();
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_root_presolve_only", true);
    set_suite(h, "all");
    // `lseu.mps` is the same instance the existing #74 regression test
    // uses — FJ reliably finds a feasible inside the presolve budget,
    // so the pool is non-empty by the time LocalMIP fires.  FPR runs
    // between the two at `suite=all` and can add to the pool as well;
    // the assertion is that LocalMIP started from the pool at all, not
    // that FJ specifically filled it.
    REQUIRE(h.readModel(kInstancesDir + "/lseu.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    auto counters = local_mip::warm_start_counters();
    // The decisive assertion: at least one worker's start came from the
    // pool, not construction.  Without #74's pool-aware lookup the only
    // path that produces non-zero warm-start counts is `construction`.
    REQUIRE(counters.pool >= 1);
}

namespace {

// Stand up a real `HighsMipSolver` (with `mipdata_`) on `instance`
// without going through `Highs::run`'s heuristics, so a test can call a
// heuristic's `run` — or `make_problem` — itself.  Mirrors the minimal
// init sequence from `HighsMipSolver::run` (init → runMipPresolve →
// runSetup); the heuristics and B&B that follow are skipped.
//
// Callers must have started the HiGHS task scheduler first (see the
// `initialize_scheduler()` note at each call site).
std::unique_ptr<HighsMipSolver> build_bare_mipsolver(Highs& highs, HighsCallback& cb,
                                                     const char* instance = "flugpl.mps") {
    // Disable HiGHS presolve so `runMipPresolve` is a near-no-op
    // that leaves `mipsolver.model_` pointing at the original LP.
    // The heuristics' `run` only needs the LP shape and
    // the `mipdata_` row-major buffers (`ARstart_/ARindex_/ARvalue_`)
    // that `runSetup` populates; the heavier LP-relaxation
    // machinery that comes later in `Highs::run` is not needed and
    // skipping presolve keeps this minimal.
    highs.setOptionValue("presolve", "off");
    REQUIRE(highs.readModel(kInstancesDir + "/" + instance) == HighsStatus::kOk);
    auto mipsolver = std::make_unique<HighsMipSolver>(cb, highs.getOptions(), highs.getLp(),
                                                      highs.getSolution());
    mipsolver->timer_.start();
    // `HighsMipSolver::run` initialises this before anything can find a
    // solution; constructing the solver directly leaves it holding
    // garbage, and `addIncumbent` -> `saveReportMipSolution` writes to
    // it unconditionally when non-null.
    mipsolver->improving_solution_file_ = nullptr;
    mipsolver->mipdata_ = std::make_unique<HighsMipSolverData>(*mipsolver);
    mipsolver->mipdata_->init();
    mipsolver->mipdata_->runMipPresolve(mipsolver->options_mip_->presolve_reduction_limit);
    mipsolver->mipdata_->runSetup();
    // `HighsMipSolver::run` creates the master worker right after
    // runSetup and before it dispatches the presolve heuristics.  Do
    // the same: `addIncumbent` — reached through the sink's accept
    // callback as soon as a heuristic finds something — reads
    // `mipdata_->workers[0]`, so a harness that skips this crashes on
    // the first solution rather than on any assertion.
    mipsolver->mipdata_->workers.emplace_back(
        *mipsolver, &mipsolver->mipdata_->getLp(), &mipsolver->mipdata_->getDomain(),
        &mipsolver->mipdata_->getCutPool(), &mipsolver->mipdata_->getConflictPool(),
        &mipsolver->mipdata_->getPseudoCost());
    return mipsolver;
}

}  // namespace

// ── Effort-accounting contract: run return value matches delta (#79) ──
//
// `local_mip::run` returns the effort it consumed; the dispatcher
// (`mode_dispatch::run_sequential`) books that exact value into
// `mipdata->heuristic_effort_used`.  The reviewer for #79 flagged that the
// existing tests (#30, #33) only assert "[Sequential] effort != 0", which
// would silently pass a regression that returns just `total_effort` and
// drops the cold-start `construction_effort` from the sum.
//
// This test pins the contract directly: it constructs a `HighsMipSolver`
// with a valid `mipdata_`, calls `local_mip::run` itself, and
// asserts `(after - before) == returned`.  One section per heuristic —
// the bug class the reviewer flagged is per-runner.  The instance is
// `flugpl.mps` — small, fast, and used by the existing
// LocalMIP tests; running with no upstream heuristic populating the pool
// or incumbent ensures the cold-start construction path fires (so
// `construction_effort > 0` is part of the returned sum in the
// local_mip section).
TEST_CASE("Heuristics: run return value matches heuristic_effort_used delta",
          "[heuristic][effort-accounting]") {
    // The HiGHS task scheduler is normally started by `Highs::run`, which
    // this harness bypasses — so initialise it once explicitly before any
    // `HighsMipSolverData::init` call (`init` reads
    // `parallel::num_threads()` for the cliquetable parallelism threshold
    // and segfaults on a null worker deque).  Subsequent
    // `initialize_scheduler()` calls are no-ops.
    highs::parallel::initialize_scheduler();

    using RunFn =
        size_t (*)(const ProblemView&, const HeuristicBudget&, ExecutionContext&, IncumbentSink&);
    auto check_invariant = [&](RunFn run_fn) {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        HighsCallback cb(&highs);
        auto mipsolver = build_bare_mipsolver(highs, cb);
        CscMatrix csc;
        const ProblemView problem = make_problem(*mipsolver, csc);
        ExecutionContext exec = make_exec(*mipsolver);
        IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);

        // Mirror exactly what `mode_dispatch::run_sequential` does:
        // read `mipdata->heuristic_effort_used`, call `run`,
        // then `+=` the returned value into the bookkeeping field.
        // The invariant the dispatcher relies on is
        // `(after - before) == returned`; that holds iff `run`
        // itself did NOT also touch the field.
        const size_t before = mipsolver->mipdata_->heuristic_effort_used;
        // A modest budget that is plenty for flugpl: large enough that
        // each runner will execute meaningful work (so `returned > 0` is
        // very likely), small enough that the test stays sub-second.
        const size_t budget = 200000;
        // `budget >> 2` is what `make_budget` used to derive internally.
        // This test is about the effort-booking contract, not about where
        // the patience gate sits, so it keeps the pre-#111 number rather
        // than picking one of the four per-heuristic constants — it runs
        // all four `run` functions through the same `RunFn`.
        const size_t returned =
            run_fn(problem, make_budget(budget, exec.num_workers, budget >> 2), exec, sink);
        mipsolver->mipdata_->heuristic_effort_used += returned;
        const size_t after = mipsolver->mipdata_->heuristic_effort_used;

        // The contract under test (issue #79 + its FJ/FPR/Scylla
        // extension): the dispatcher's `+=` booking is the *only* path
        // that updates `mipdata->heuristic_effort_used` for any of the
        // four sequential heuristics.  If a future refactor reintroduces
        // self-booking inside any `run` the delta becomes
        // `2 * returned` (or more) and this fires.
        REQUIRE(after - before == returned);
        // Sanity guard: a broken implementation that always returns 0
        // would make the invariant above vacuously true.  flugpl with a
        // 200k budget and no incumbent will do real work in every
        // heuristic (FJ runs jumps, FPR fixes integers and propagates,
        // LocalMIP constructs and searches, Scylla runs at least one
        // PDLP+rounding cycle), so `returned > 0` for all four.  The
        // exact value isn't pinned (depends on parallelism + seeds) —
        // only the lower bound is.
        REQUIRE(returned > 0);
    };

    SECTION("fj") {
        check_invariant(&fj::run);
    }
    SECTION("fpr") {
        check_invariant(&fpr::run);
    }
    SECTION("local_mip") {
        check_invariant(&local_mip::run);
    }
    SECTION("scylla") {
        check_invariant(&scylla::run);
    }
}

// ── Issue #98: the incumbent workers see is a dispatch snapshot ──
//
// Workers used to read `mipdata->incumbent` live, from inside a parallel
// region, while a peer's accepted solution ran `addIncumbent` — whose
// `incumbent = sol;` rewrites that buffer under the reader.
// `ProblemView::incumbent` is a copy taken on the dispatching thread, so a
// concurrent write cannot move or change what a worker holds.
//
// The two sections split the invariant in half.  Neither reproduces the
// race itself — that needs a write to land inside a concurrent read — but
// together they fail for every regression that would reintroduce it.
TEST_CASE("ProblemView::incumbent is a dispatch snapshot, not the live vector (#98)",
          "[heuristic][heuristic-context][warm-start-counters]") {
    highs::parallel::initialize_scheduler();

    // The cheap half: `make_problem` copies rather than aliases.  Fails if
    // the field goes back to being a pointer or a reference.
    SECTION("make_problem copies, and the copy is immune to later writes") {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        HighsCallback cb(&highs);
        auto mipsolver = build_bare_mipsolver(highs, cb);
        auto* mipdata = mipsolver->mipdata_.get();
        const HighsInt ncol = mipsolver->model_->num_col_;
        REQUIRE(ncol > 0);

        // Pre-dispatch state: the solver holds an incumbent, as it does
        // whenever an earlier heuristic (or HiGHS itself) found one.
        mipdata->incumbent.assign(static_cast<size_t>(ncol), 1.0);

        CscMatrix csc;
        const ProblemView problem = make_problem(*mipsolver, csc);
        REQUIRE(problem.incumbent.size() == static_cast<size_t>(ncol));
        const double* snapshot_data = problem.incumbent.data();
        REQUIRE(snapshot_data != mipdata->incumbent.data());

        // What `addIncumbent` does on a peer worker's accepted solution: a
        // whole-vector assignment, free to reallocate.  Change the size so
        // the allocation is guaranteed to move.
        mipdata->incumbent = std::vector<double>(static_cast<size_t>(ncol) * 2, 2.0);

        REQUIRE(problem.incumbent.data() == snapshot_data);
        REQUIRE(problem.incumbent.size() == static_cast<size_t>(ncol));
        for (HighsInt j = 0; j < ncol; ++j) {
            REQUIRE(problem.incumbent[j] == 1.0);
        }
    }

    // The half that matters: the workers actually read the snapshot.  The
    // two vectors are made to disagree — snapshot non-empty, live vector
    // emptied right after — which is impossible in production but is
    // exactly what distinguishes the two reads.  A worker (or the
    // dispatch-thread prime in `local_mip::run`) reading `mipdata` live
    // would find nothing and fall through to cold-start construction.
    SECTION("LocalMIP resolves its start from the snapshot, not from mipdata") {
        if constexpr (!local_mip::kInstrumented) {
            SKIP("Built with MIP_HEURISTICS_INSTRUMENT=OFF — counters compiled out");
        }
        Highs highs;
        highs.setOptionValue("output_flag", false);
        HighsCallback cb(&highs);
        auto mipsolver = build_bare_mipsolver(highs, cb);
        auto* mipdata = mipsolver->mipdata_.get();
        const HighsInt ncol = mipsolver->model_->num_col_;

        mipdata->incumbent.assign(static_cast<size_t>(ncol), 0.0);
        CscMatrix csc;
        const ProblemView problem = make_problem(*mipsolver, csc);
        REQUIRE(!problem.incumbent.empty());

        // Diverge the live vector from the snapshot, then build the sink:
        // `seed_pool` reads the live one, so the pool stays empty and the
        // incumbent branch of `resolve_worker_start` — dead in production,
        // where the pool always carries the incumbent — becomes reachable.
        mipdata->incumbent.clear();
        ExecutionContext exec = make_exec(*mipsolver);
        IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);

        local_mip::reset_warm_start_counters();
        local_mip::run(problem, make_budget(200000, exec.num_workers, 200000 >> 2), exec, sink);
        auto counters = local_mip::warm_start_counters();

        // Reading the snapshot: the incumbent branch fires (at minimum on
        // the dispatch-thread prime).  Reading `mipdata` live: it is empty,
        // so every start resolves through construction instead.
        REQUIRE(counters.incumbent >= 1);
        REQUIRE(counters.construction == 0);
    }
}

// ── Issue #99: column classification comes from a dispatch snapshot ──
//
// `addIncumbent` propagates the root domain, tightening the very bound
// vectors `HighsDomain::isBinary` reads, while workers classify columns
// from them.  `ProblemView::binary` (and `FprConfig::binary_mask`, and
// `LpFprSetup::binary`) is the snapshot that replaces those live reads.
TEST_CASE("ProblemView::binary is a dispatch snapshot of isBinary (#99)",
          "[heuristic][heuristic-context]") {
    highs::parallel::initialize_scheduler();

    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    // `lseu.mps`, not the `flugpl.mps` default: flugpl's integers are all
    // general, so the binary half of this test would be vacuous on it.
    auto mipsolver = build_bare_mipsolver(highs, cb, "lseu.mps");
    auto* mipdata = mipsolver->mipdata_.get();
    const HighsInt ncol = mipsolver->model_->num_col_;
    REQUIRE(ncol > 0);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    REQUIRE(problem.binary.size() == static_cast<size_t>(ncol));

    // Agrees with the domain it was taken from, column for column.
    for (HighsInt j = 0; j < ncol; ++j) {
        REQUIRE(static_cast<bool>(problem.binary[j]) == mipdata->getDomain().isBinary(j));
    }

    // A snapshot, not a live view: tightening a column's bounds the way
    // `getDomain().propagate()` does must not change what a worker sees
    // mid-dispatch.  `lseu.mps` is binary-heavy so the search below always
    // finds a probe; the `REQUIRE(probe >= 0)` guards a future instance swap
    // that would quietly make the rest of this vacuous.
    HighsInt probe = -1;
    for (HighsInt j = 0; j < ncol; ++j) {
        if (problem.binary[j] != 0U) {
            probe = j;
            break;
        }
    }
    REQUIRE(probe >= 0);
    mipdata->getDomain().changeBound(HighsBoundType::kUpper, probe, 0.0,
                                     HighsDomain::Reason::unspecified());
    REQUIRE(!mipdata->getDomain().isBinary(probe));
    REQUIRE(problem.binary[probe] == 1);
}

// The half that matters for #99: the consumers read the mask they are
// handed, not the solver.  `perturb_solution` is the cleanest probe —
// it is a free function whose only classification input is now the mask,
// so the same columns take the binary-flip path or the general-integer
// shift path purely on what the caller passed.
TEST_CASE("perturb_solution classifies columns from the mask it is given (#99)",
          "[heuristic][local_mip][heuristic-context]") {
    // Bounds [0, 5] make the two paths distinguishable: the binary path
    // flips to exactly 0 or 1, the general-integer path shifts by a
    // non-zero amount within the range and so can land above 1.
    constexpr HighsInt kNcol = 400;
    const std::vector<HighsVarType> integrality(kNcol, HighsVarType::kInteger);
    const std::vector<double> col_lb(kNcol, 0.0);
    const std::vector<double> col_ub(kNcol, 5.0);

    const std::vector<uint8_t> all_binary(kNcol, 1);
    const std::vector<uint8_t> none_binary(kNcol, 0);

    std::vector<double> as_binary(kNcol, 0.0);
    Rng rng_a(12345);
    local_mip_detail::perturb_solution(as_binary, all_binary.data(), integrality, col_lb, col_ub,
                                       kNcol, rng_a);

    std::vector<double> as_general(kNcol, 0.0);
    Rng rng_b(12345);
    local_mip_detail::perturb_solution(as_general, none_binary.data(), integrality, col_lb, col_ub,
                                       kNcol, rng_b);

    // Called binary: every perturbed column flipped 0 -> 1, and nothing
    // ever leaves {0, 1}.
    int flipped = 0;
    for (HighsInt j = 0; j < kNcol; ++j) {
        REQUIRE((as_binary[j] == 0.0 || as_binary[j] == 1.0));
        if (as_binary[j] == 1.0) {
            ++flipped;
        }
    }
    // ~20% of 400 columns; a wide band, since the point is that the
    // branch fired at all, not how often.
    REQUIRE(flipped > 20);

    // Called general-integer: the shift path consumes RNG draws the binary
    // path does not, so the two streams diverge after the first perturbed
    // column — this is not a column-for-column A/B.  What it pins is that
    // the shift path ran at all, since it reaches values the binary path
    // cannot produce.
    int above_one = 0;
    for (HighsInt j = 0; j < kNcol; ++j) {
        if (as_general[j] > 1.0) {
            ++above_one;
        }
    }
    REQUIRE(above_one > 0);
}
