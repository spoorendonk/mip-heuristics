#include "Highs.h"
#include "lp_data/HStruct.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>

// ===================================================================
// HiGHS's own standalone FeasibilityJump at `suite=off` (#147)
//
// `off` is the "our four presolve heuristics disabled" ablation, and the
// patch deliberately hands HiGHS's own standalone FeasibilityJump call
// site back at that value — without it, `off` would conflate two
// ablations ("our heuristics off" and "HiGHS's own FJ off"), and the
// second one already has its own switch in
// `mip_heuristic_run_feasibility_jump`.
//
// `bench/check_vanilla_equivalence.py` used to cover this incidentally,
// by running `off` against an unpatched binary and diffing the logs.  It
// no longer does: that check is now taken with FeasibilityJump disabled
// on both sides, because FJ is the one component the patch deliberately
// changes.  This case is what covers the call site instead — presence and
// accounting, not bit-identity, and cheap enough to be a ctest, which
// that script can never be (it needs a second binary).
//
// The accounting is not visible in the log: at `suite=off` the dispatcher
// returns before the effort ledger exists, so no `[Heur]` or
// `[Sequential]` line is emitted, and the `heuristic_effort_used +=
// fj_last_effort` store the patch adds inside stock `feasibilityJump()`
// is the only writer of that counter on this path.  Reading the counter
// therefore means owning the `HighsMipSolver`, which means driving
// `run()` directly rather than through `Highs::run`.
// ===================================================================

namespace {

// The whole solve, driven on a solver object the test keeps, so
// `mipdata_->heuristic_effort_used` is still readable afterwards
// (`cleanupSolve` leaves `mipdata_` alive).  `flugpl` is small enough to
// solve to optimality in well under a second.
struct OffRun {
    size_t heuristic_effort_used = 0;
    HighsModelStatus status = HighsModelStatus::kNotset;
};

OffRun solve_at_off(bool run_feasibility_jump) {
    // `HighsMipSolverData::init` reads `parallel::num_threads()`; see the
    // note at the `build_bare_mipsolver` call sites.  A no-op once started.
    highs::parallel::initialize_scheduler();

    // Declared before `highs`, so it is destroyed *after* it: `Highs` holds
    // the pointer this hands it (see `initializeProfiling` below), and a
    // `~Highs` that ever reads `profiling_` would otherwise read a destroyed
    // object.  Today's does not; the ordering is what keeps that from being
    // something a HiGHS bump can turn into a silent use-after-free.
    HighsProfiling profiling;

    Highs highs;
    highs.setOptionValue("output_flag", false);
    set_suite(highs, "off");
    require_option(highs, "mip_heuristic_run_feasibility_jump", run_feasibility_jump);
    // The counter is an effort total, and FJ's effort depends on its seed.
    // Pin it so a failure is about the call site and not about the draw.
    require_option(highs, "random_seed", 0);
    REQUIRE(highs.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);

    // `HighsMipSolver::run` dereferences `profiling_` unconditionally, and
    // the member defaults to null — `Highs::callSolveMip` hands its own in
    // right after constructing the solver, so a test that drives `run()`
    // itself has to do the same.
    highs.initializeProfiling(&profiling);

    HighsCallback cb(&highs);
    HighsMipSolver mipsolver(cb, highs.getOptions(), highs.getLp(), highs.getSolution());
    mipsolver.setProfiling(&profiling);
    mipsolver.run();

    REQUIRE(mipsolver.mipdata_ != nullptr);
    return {mipsolver.mipdata_->heuristic_effort_used, mipsolver.modelstatus_};
}

}  // namespace

TEST_CASE("native FJ: suite=off runs HiGHS's own FJ and charges its effort",
          "[mode-matrix][suite][native-fj]") {
    const OffRun ran = solve_at_off(/*run_feasibility_jump=*/true);
    // Accounting: something charged `heuristic_effort_used`, and at
    // `suite=off` the standalone FeasibilityJump call site is the only
    // thing that can — the presolve chain never dispatches, so the ledger
    // is never reached, and `fpr_lp` is disabled at `off` too.
    REQUIRE(ran.heuristic_effort_used > 0);
    REQUIRE(ran.status == HighsModelStatus::kOptimal);
}

TEST_CASE("native FJ: the pure patch-overhead configuration charges nothing",
          "[mode-matrix][suite][native-fj]") {
    // `suite=off` plus `mip_heuristic_run_feasibility_jump=false` is the
    // configuration `bench/check_vanilla_equivalence.py` compares against
    // an unpatched binary, so "no heuristic ran at all" is part of what
    // that check rests on.  It is also the control for the case above:
    // without it, a non-zero counter could come from anywhere.
    const OffRun disabled = solve_at_off(/*run_feasibility_jump=*/false);
    REQUIRE(disabled.heuristic_effort_used == 0);
    REQUIRE(disabled.status == HighsModelStatus::kOptimal);
}

TEST_CASE("native FJ: the call site is silent when FJ is switched off",
          "[mode-matrix][suite][native-fj]") {
    // The presence half, from the other side.  `Feasibility Jump: starting
    // solve` is logged once per FJ solver instance, so a zero count is the
    // gate holding: upstream's own option disables the native call site at
    // `off` exactly as it disables our parallel FJ elsewhere.  (The count
    // at `off` with FJ enabled — exactly one, upstream's single-threaded
    // instance — is pinned by `test_execution_modes.cpp`.)
    const auto lines = solve_capturing_log("lseu.mps", [](Highs& h) {
        require_option(h, "log_dev_level", 3);
        set_suite(h, "off");
        require_option(h, "mip_heuristic_run_feasibility_jump", false);
    });
    REQUIRE_FALSE(log_contains(lines, "Feasibility Jump: starting solve"));
}
