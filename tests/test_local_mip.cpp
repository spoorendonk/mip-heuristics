#include "fj.h"
#include "fpr.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "incumbent_sink.h"
#include "local_mip.h"
#include "local_mip_construction.h"
#include "local_mip_core.h"
#include "local_mip_worker.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"  // for kSolutionSource* constants
#include "parallel/HighsParallel.h"
#include "rng.h"
#include "scylla.h"
#include "solution_pool.h"
#include "test_common.h"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
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

    using RunFn = DispatchOutcome (*)(const ProblemView&, const HeuristicBudget&, ExecutionContext&,
                                      IncumbentSink&);
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
            run_fn(problem, make_budget(budget, exec.num_workers, budget >> 2), exec, sink).effort;
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
        // The outcome is deliberately dropped: this case asserts on the
        // warm-start counters the dispatch leaves behind, not on what it
        // spent or whether it bailed.
        static_cast<void>(local_mip::run(
            problem, make_budget(200000, exec.num_workers, 200000 >> 2), exec, sink));
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

// ── Mixed tight move: the integer rounding rule (issue #123) ──────────
//
// `WorkerCtx::compute_tight_delta` used to round an integer tight-move
// delta by the coefficient's sign alone (`coeff > 0 -> floor`,
// `coeff < 0 -> ceil`).  That collapse is valid only for the paper's
// one-sided `A_i x <= b_i` form (Lin, Zou, Cai, CP 2024, Def 4 / Eq 5);
// HiGHS rows are two-sided, and on a lower-bound violation with a
// positive coefficient the sign rule undershoots and leaves the row
// violated.  Both call sites now round through the shared
// `round_tight_delta`: away from zero when the row is violated, toward
// zero when it is satisfied.
//
// Three levels are covered, deliberately: `compute_tight_delta` itself
// (the only function whose behaviour this issue changes, driven over a
// real `HighsMipSolver` from `build_bare_mipsolver`), the construction
// copy `tight_delta_for_row`, and the shared `round_tight_delta` the
// two now call.

namespace {

struct TightMoveCase {
    const char* name;
    double row_lo;
    double row_hi;
    double lhs;
    double coeff;
    double expected_delta;
};

// All four combinations of {upper violated, lower violated} x
// {coeff > 0, coeff < 0} on an integer variable.  "upper violated" is
// `x0 + x1 <= 2` (resp. `-x0 + x1 <= 2`) sitting at lhs 4.5; "lower
// violated" is `x0 + x1 >= 3` (resp. `-x0 + x1 >= 3`) sitting at lhs
// 0.5.  The unrounded delta is +-2.5 in every case.
//
// The upper-violated pair rounds identically under the retired
// sign-based rule and is kept for coverage, not because it ever
// failed; the lower-violated pair is what issue #123 reports — the old
// rule gave 2 and -2 there, leaving lhs at 2.5, still violated.
const std::array<TightMoveCase, 4> kTightMoveCases = {{
    {"upper violated, coeff > 0", -kHighsInf, 2.0, 4.5, 1.0, -3.0},
    {"lower violated, coeff > 0", 3.0, kHighsInf, 0.5, 1.0, 3.0},
    {"upper violated, coeff < 0", -kHighsInf, 2.0, 4.5, -1.0, 3.0},
    {"lower violated, coeff < 0", 3.0, kHighsInf, 0.5, -1.0, -3.0},
}};

}  // namespace

TEST_CASE("LocalMIP tight move: the construction copy satisfies the row it targets (#123)",
          "[heuristic][local_mip]") {
    const double feastol = 1e-6;

    for (const auto& c : kTightMoveCases) {
        INFO(c.name);
        std::vector<double> lhs = {c.lhs};
        std::vector<double> row_lo = {c.row_lo};
        std::vector<double> row_hi = {c.row_hi};
        // Bounds deliberately far wider than the move: a clamped delta
        // would satisfy the row for the wrong reason and hide the
        // rounding under test.
        std::vector<double> col_lb = {-10.0, -10.0};
        std::vector<double> col_ub = {10.0, 10.0};
        std::vector<double> solution = {0.0, 0.0};

        double delta = local_mip_detail::tight_delta_for_row(
            /*i=*/0, /*j=*/0, c.coeff, lhs, row_lo, row_hi, col_lb, col_ub, solution, feastol,
            /*integer=*/true);

        REQUIRE(delta == Catch::Approx(c.expected_delta));
        REQUIRE(delta == std::floor(delta));  // the move stays integral

        // The point of the operator (paper Def 4): the row it was
        // computed for is satisfied afterwards.
        double new_lhs = c.lhs + (c.coeff * delta);
        REQUIRE(new_lhs >= c.row_lo - feastol);
        REQUIRE(new_lhs <= c.row_hi + feastol);
    }
}

TEST_CASE("LocalMIP tight move: satisfied rows round toward zero (#123)",
          "[heuristic][local_mip]") {
    using local_mip_detail::round_tight_delta;

    // Violated: away from zero, so the shift crosses the bound it aims at.
    REQUIRE(round_tight_delta(2.5, /*row_violated=*/true) == Catch::Approx(3.0));
    REQUIRE(round_tight_delta(-2.5, /*row_violated=*/true) == Catch::Approx(-3.0));

    // Satisfied: toward zero, so the shift stops short of the bound it
    // aims at.  Both nearest-bound sides are covered, because the sign
    // of the shift is what distinguishes them.
    //
    // Nearest bound above (`gap_hi` wins, `coeff > 0` => `delta > 0`):
    // e.g. `x1 + x2 <= 6` at lhs 3.5 gives delta 2.5; 2 keeps lhs at
    // 5.5 <= 6, while away-from-zero 3 would land exactly on the bound.
    REQUIRE(round_tight_delta(2.5, /*row_violated=*/false) == Catch::Approx(2.0));

    // Nearest bound below (`gap_lo` wins, `coeff > 0` => `delta < 0`).
    // This is the case the issue's first acceptance criterion got
    // wrong: on `x1 + x2 >= 3` at lhs 5.5 with coeff 1 the satisfied
    // branch picks `gap = 2.5`, so `delta = -2.5`, and the sign rule's
    // `floor` gives -3 — lhs 2.5, now violated.  Toward zero gives -2,
    // leaving lhs 3.5, still satisfied.
    REQUIRE(round_tight_delta(-2.5, /*row_violated=*/false) == Catch::Approx(-2.0));

    const double satisfied_lhs = 5.5;
    const double row_lo = 3.0;
    double new_lhs = satisfied_lhs + (1.0 * round_tight_delta(-2.5, /*row_violated=*/false));
    REQUIRE(new_lhs >= row_lo);
}

// The production function this issue changes.  `build_bare_mipsolver`
// stands up a real `HighsMipSolver`, and `WorkerCtx`'s `lhs` /
// `solution` are public and mutable, so the rule can be driven
// directly: pick a row of the shape a case needs out of the instance,
// park `lhs[i]` on the side of the bound the case is about, and pass
// the coefficient as an argument — `compute_tight_delta` takes it
// rather than looking it up, so the instance only has to supply a row
// of the right *shape*, not a particular coefficient.
//
// Reinstating the retired sign rule (`coeff > 0 -> floor`,
// `coeff < 0 -> ceil`) fails the two lower-violated cases and the
// satisfied-nearest-bound-below case; the other three are unchanged by
// it and are here for coverage of the operator's promise.
TEST_CASE("LocalMIP tight move: compute_tight_delta satisfies the row it targets (#123)",
          "[heuristic][local_mip]") {
    struct CtxTightMoveCase {
        const char* name;
        bool lower_anchored;  // which bound the case is measured against
        bool violated;        // park `lhs` outside (true) or inside (false) it
        double coeff;
        double expected_delta;
    };
    // The unrounded delta is +-2.5 in every case, so the rounding is the
    // only thing that separates them.
    const std::array<CtxTightMoveCase, 6> cases = {{
        {"upper violated, coeff > 0", false, true, 1.0, -3.0},
        {"lower violated, coeff > 0", true, true, 1.0, 3.0},
        {"upper violated, coeff < 0", false, true, -1.0, 3.0},
        {"lower violated, coeff < 0", true, true, -1.0, -3.0},
        {"satisfied, nearest bound below", true, false, 1.0, -2.0},
        {"satisfied, nearest bound above", false, false, 1.0, 2.0},
    }};

    // `build_bare_mipsolver` skips `Highs::run`, which is what normally
    // starts the task scheduler; `HighsMipSolverData::init` needs it.
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb);
    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    local_mip_detail::WorkerCtx ctx(*mipsolver, csc, problem.binary.data());

    // Slack demanded of the opposite bound, and of the variable's own
    // bounds: wide enough that neither the nearest-bound choice nor the
    // clamp in `compute_tight_delta` can stand in for the rounding.
    const double room = 8.0;

    // A non-equality row with the anchor bound finite and the opposite
    // bound either infinite or `room` away.
    auto find_row = [&](bool lower_anchored) {
        for (HighsInt i = 0; i < ctx.nrow; ++i) {
            if (ctx.is_equality(i)) {
                continue;
            }
            const double lo = ctx.row_lo[i];
            const double hi = ctx.row_hi[i];
            const double anchor = lower_anchored ? lo : hi;
            const double opposite = lower_anchored ? hi : lo;
            if (std::abs(anchor) >= kHighsInf) {
                continue;
            }
            if (std::abs(opposite) < kHighsInf && hi - lo <= room) {
                continue;
            }
            return i;
        }
        return HighsInt{-1};
    };

    HighsInt col = -1;
    for (HighsInt j = 0; j < ctx.ncol; ++j) {
        if (ctx.is_int(j) && ctx.col_lb[j] > -kHighsInf && ctx.col_ub[j] < kHighsInf &&
            ctx.col_ub[j] - ctx.col_lb[j] >= room) {
            col = j;
            break;
        }
    }
    REQUIRE(col >= 0);
    // Midpoint, so a +-3 move stays clear of both column bounds.
    const double base_val = std::floor((ctx.col_lb[col] + ctx.col_ub[col]) / 2.0);
    REQUIRE(base_val - 3.0 >= ctx.col_lb[col]);
    REQUIRE(base_val + 3.0 <= ctx.col_ub[col]);

    for (const auto& c : cases) {
        INFO(c.name);
        const HighsInt row = find_row(c.lower_anchored);
        REQUIRE(row >= 0);

        const double anchor = c.lower_anchored ? ctx.row_lo[row] : ctx.row_hi[row];
        // Outward from the anchor when the case wants the row violated,
        // inward when it wants it satisfied.
        const double direction = (c.lower_anchored ? -1.0 : 1.0) * (c.violated ? 1.0 : -1.0);
        ctx.lhs[row] = anchor + (direction * 2.5);
        ctx.solution[col] = base_val;
        REQUIRE(ctx.is_violated(row, ctx.lhs[row]) == c.violated);

        const double delta = ctx.compute_tight_delta(row, col, c.coeff);
        // `CHECK`, not `REQUIRE`: the cases are independent, and a
        // regression in the rounding rule breaks three of the six — a
        // report naming all of them is worth more than the first.
        CHECK(delta == Catch::Approx(c.expected_delta));

        // The operator's promise (paper Def 4): the row it was computed
        // for is satisfied afterwards.
        const double new_lhs = ctx.lhs[row] + (c.coeff * delta);
        CHECK(new_lhs >= ctx.row_lo[row] - ctx.feastol);
        CHECK(new_lhs <= ctx.row_hi[row] + ctx.feastol);
    }
}

// ── LocalMIP: one tolerance governs the violation partition (#148) ────

// Before this fix, three different questions about "is this row
// violated?" disagreed. `kViolTol` (5e-7) gated set membership
// (`WorkerCtx::update_violated`, `full_recheck`) and the worker's
// submission gate; HiGHS's own `feastol` (1e-6 by default) gated
// `WorkerCtx::is_violated` and the branch `compute_tight_delta` uses to
// pick its rounding rule. A row violated by an amount in the window
// (5e-7, 1e-6] therefore landed in the `violated` set — but the
// tight-move operator, reading `feastol`, treated it as already
// satisfied and produced no repairing candidate; and the submission
// gate refused a solution HiGHS's own `trySolution`
// (`mip/HighsMipSolverData.cpp`, which bounds every row check to
// `feastol` alone) would have accepted.
//
// The fix threads `feastol` through every one of those sites. A row in
// this window is therefore not "violated" by any of them any more — it
// agrees with HiGHS's own tolerance instead of being pickier than it,
// so no repair is needed and the worker is willing to submit. This
// pins that outcome directly on the issue's repro shape:
// `x1 + x2 <= 2`, both integer, `lhs = 2 + 7e-7`.
TEST_CASE("LocalMIP: violation partition agrees with HiGHS's feastol, not a stricter one (#148)",
          "[heuristic][local_mip]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);

    // x1 + x2 <= 2, x1, x2 in [0, 10], both integer.
    highs.addVar(0.0, 10.0);
    highs.addVar(0.0, 10.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);
    const auto idx = std::to_array<HighsInt>({0, 1});
    const auto val = std::to_array<double>({1.0, 1.0});
    highs.addRow(-kHighsInf, 2.0, 2, idx.data(), val.data());

    // Mirrors `build_bare_mipsolver` (test_common.h), minus the
    // `readModel` call: the repro needs a row shaped exactly
    // `x1 + x2 <= 2` with unit coefficients, which no bundled instance
    // is guaranteed to contain, so the model is built directly above.
    highs.setOptionValue("presolve", "off");
    require_option(highs, "time_limit", kHighsInf);
    HighsCallback cb(&highs);
    auto mipsolver = std::make_unique<HighsMipSolver>(cb, highs.getOptions(), highs.getLp(),
                                                      highs.getSolution());
    mipsolver->timer_.start();
    mipsolver->improving_solution_file_ = nullptr;
    mipsolver->mipdata_ = std::make_unique<HighsMipSolverData>(*mipsolver);
    mipsolver->mipdata_->init();
    mipsolver->mipdata_->runMipPresolve(mipsolver->options_mip_->presolve_reduction_limit);
    mipsolver->mipdata_->runSetup();
    mipsolver->mipdata_->workers.emplace_back(
        *mipsolver, &mipsolver->mipdata_->getLp(), &mipsolver->mipdata_->getDomain(),
        &mipsolver->mipdata_->getCutPool(), &mipsolver->mipdata_->getConflictPool(),
        &mipsolver->mipdata_->getPseudoCost());

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    local_mip_detail::WorkerCtx ctx(*mipsolver, csc, problem.binary.data());

    REQUIRE(ctx.nrow == 1);
    const HighsInt row = 0;
    REQUIRE(ctx.row_hi[row] == Catch::Approx(2.0));
    REQUIRE_FALSE(ctx.is_equality(row));

    // The window this issue is about: strictly between the retired
    // kViolTol (5e-7) and HiGHS's feastol (1e-6). Confirm the fixture's
    // own feastol matches what the repro assumes before leaning on it.
    REQUIRE(ctx.feastol == Catch::Approx(1e-6));
    const double violation = 7e-7;
    REQUIRE(violation > 5e-7);
    REQUIRE(violation <= ctx.feastol);

    // Column values chosen only to make the row's real dot product
    // equal `2 + violation` via floating-point arithmetic (both
    // coefficients are 1.0) — `full_recheck` recomputes `lhs` from
    // `solution`, so the target activity has to be reached that way,
    // not by poking `ctx.lhs` directly. This is a row-activity
    // tolerance test, not an integer-feasibility one (that is checked
    // separately, e.g. by `local_mip::is_solution_feasible`).
    ctx.solution[0] = 2.0;
    ctx.solution[1] = violation;

    // `is_violated` already read `feastol` before this fix and still
    // does; check it first so a regression there is not conflated with
    // the set-membership fix below.
    REQUIRE_FALSE(ctx.is_violated(row, 2.0 + violation));

    // `rebuild_state` drives `full_recheck(update_sets=true, ...)`,
    // which is both the set-membership classifier and (in its
    // `update_sets=false` form below) the submission gate's own
    // feasibility check.
    ctx.rebuild_state();
    REQUIRE(ctx.lhs[row] == Catch::Approx(2.0 + violation));

    // The heart of #148: set membership must agree with `is_violated`.
    // Before the fix, `kViolTol` (5e-7) made this `true` even though
    // `is_violated` above says `false` for the same row.
    REQUIRE_FALSE(ctx.violated.contains(row));
    REQUIRE(ctx.satisfied.contains(row));

    // The submission gate (mirrors the production call site in
    // `local_mip_worker.cpp`'s post-improvement recheck): a worker
    // willing to submit this solution is exactly as strict as HiGHS's
    // own `trySolution`, which accepts any row activity within
    // `feastol`. Before the fix `full_recheck` used `kViolTol` and
    // returned `false` here, refusing a solution HiGHS itself accepts.
    REQUIRE(ctx.full_recheck(/*update_sets=*/false, /*early_exit=*/true));

    // And the tight-move operator agrees rather than disagreeing with
    // set membership: since the row is (now, correctly) not violated,
    // its satisfied-branch delta rounds toward zero to a no-op — the
    // consistent answer for a row already within tolerance, replacing
    // the old mismatch where the same row sat in `violated` but got no
    // repairing candidate. This assertion is *not* discriminating on
    // its own: `compute_tight_delta` already read `feastol` before
    // this fix, so it passes on both sides of it. It documents intent
    // (the operator and set membership now genuinely agree), not
    // coverage — the `violated`/`satisfied`/`full_recheck` assertions
    // above are what a regression trips.
    const double delta = ctx.compute_tight_delta(row, /*j=*/0, /*coeff=*/1.0);
    REQUIRE(std::abs(delta) < local_mip_detail::kEpsZero);

    // The other side of the same threshold (issue #148 follow-up): a
    // row violated by more than `feastol` must still land in
    // `violated` and still fail the submission gate — pinning only the
    // "not violated within the window" side above would pass a
    // degenerate regression such as `is_violated`'s `>` becoming
    // unconditionally false, or `full_recheck`'s predicate becoming
    // `> 1.0`. `2e-6` is comfortably past `feastol` (1e-6) so the
    // margin cannot be mistaken for more summation noise.
    const double clear_violation = 2e-6;
    REQUIRE(clear_violation > ctx.feastol);
    ctx.solution[1] = clear_violation;
    REQUIRE(ctx.is_violated(row, 2.0 + clear_violation));

    ctx.rebuild_state();
    REQUIRE(ctx.lhs[row] == Catch::Approx(2.0 + clear_violation));
    REQUIRE(ctx.violated.contains(row));
    REQUIRE_FALSE(ctx.satisfied.contains(row));
    REQUIRE_FALSE(ctx.full_recheck(/*update_sets=*/false, /*early_exit=*/true));
}
