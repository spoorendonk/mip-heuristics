#include "Highs.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>
#include <utility>

// ===================================================================
// mip_heuristic_presolve_only (#106)
//
// The option exits the solve after the presolve heuristic chain and
// before the root LP, keeping whatever incumbent the chain produced.
// It exists because the presolve heuristics are unmeasurable inside a
// full solve: a heuristic runs for ~2 s of a 60 s limit and B&B owns the
// rest, so a primal-integral score over the whole solve dilutes the
// thing being tuned into seed noise.  A presolve-only run scores exactly
// the chain.
//
// Two alternatives that look like they should work do not, and both were
// tried: `mip_max_nodes = 0` is checked *inside* the B&B loop, so the
// root LP and the dive heuristics all run first, and
// `mip_root_presolve_only` controls where presolve is applied rather
// than when the solve stops.
//
// The reported status is `kSolutionLimit` — see the justification beside
// the inserted early exit in
// `third_party/highs_patch/apply_patch.cmake`.  In one line: it is what
// HiGHS itself assigns for its own search-size limits
// (`mip_max_nodes` / `mip_max_leaves` / `mip_max_improving_sols`), and
// `cleanupSolve` overwrites only `kNotset` and `kInfeasible`, so it is
// also the shape of status that survives to the caller with the
// incumbent intact.
// ===================================================================

namespace {

// One solve's outcome, in the fields these tests assert on.
//
// The suite's shared `solve_capturing_log` cannot be used here: it
// `REQUIRE`s `run() == kOk`, and a presolve-only solve returns
// `kWarning` by construction — every limit status maps to `kWarning` in
// `highsStatusFromHighsModelStatus`.  Asserting the status is half of
// what this file is for, so it is captured rather than required.
struct SolveOutcome {
    HighsStatus status = HighsStatus::kError;
    HighsModelStatus model_status = HighsModelStatus::kNotset;
    double objective = 0.0;
    double dual_bound = 0.0;
    HighsInt solution_status = kSolutionStatusNone;
    int64_t node_count = -1;
    HighsInt lp_iterations = -1;
};

// Solve `inst` under `configure` and report the outcome.
//
// `threads` and `random_seed` are pinned for reproducibility, not
// coverage: "did the chain find anything on this instance" has to be the
// same answer on a 2-vCPU runner and a 12-core laptop, and one worker per
// heuristic under a fixed seed is the project's documented reproducible
// configuration.  `ScopedThreadPin` is what makes the pin survive
// whatever initialised the process-global task executor first.
template <typename Configure>
SolveOutcome solve_outcome(const char* inst, Configure&& configure) {
    const ScopedThreadPin pin;
    Highs h;
    h.setOptionValue("output_flag", false);
    require_option(h, "threads", 1);
    require_option(h, "random_seed", 0);
    std::forward<Configure>(configure)(h);
    REQUIRE(h.readModel(kInstancesDir + "/" + inst) == HighsStatus::kOk);

    SolveOutcome out;
    out.status = h.run();
    out.model_status = h.getModelStatus();
    REQUIRE(h.getInfoValue("objective_function_value", out.objective) == HighsStatus::kOk);
    REQUIRE(h.getInfoValue("mip_dual_bound", out.dual_bound) == HighsStatus::kOk);
    REQUIRE(h.getInfoValue("primal_solution_status", out.solution_status) == HighsStatus::kOk);
    REQUIRE(h.getInfoValue("mip_node_count", out.node_count) == HighsStatus::kOk);
    REQUIRE(h.getInfoValue("simplex_iteration_count", out.lp_iterations) == HighsStatus::kOk);
    return out;
}

// The one instance these cases run on.  flugpl is small enough to solve
// to optimality in the default path in well under a second, and its
// presolve chain reliably produces an incumbent at `threads=1, seed=0`
// (the stall-gate suite measures FPR earning pool acceptances on it).
constexpr const char* kInstance = "flugpl.mps";
constexpr double kFlugplOptimum = 1201500.0;

}  // namespace

TEST_CASE("presolve-only: the solve stops before the root LP", "[presolve-only]") {
    const SolveOutcome out = solve_outcome(
        kInstance, [](Highs& h) { require_option(h, "mip_heuristic_presolve_only", true); });

    // A limit status, not an error and not a claim of optimality.
    REQUIRE(out.status == HighsStatus::kWarning);
    REQUIRE(out.model_status == HighsModelStatus::kSolutionLimit);

    // Three independent witnesses that the root node was never evaluated.
    // Any one of them alone is arguable; together they pin it.
    //
    // No B&B node was processed — necessary but not sufficient, since the
    // root LP itself is not a node.
    REQUIRE(out.node_count == 0);
    // No LP iteration was performed anywhere in the solve.  This is the
    // direct one: `evaluateRootNode` cannot run without solving the root
    // relaxation, and `mip_max_nodes = 0` fails exactly here.
    REQUIRE(out.lp_iterations == 0);
    // And therefore no dual bound was ever computed.  A completed root LP
    // always yields a finite one on this instance.
    REQUIRE(out.dual_bound == -kHighsInf);
}

TEST_CASE("presolve-only: a solution found during presolve survives", "[presolve-only]") {
    const SolveOutcome out = solve_outcome(
        kInstance, [](Highs& h) { require_option(h, "mip_heuristic_presolve_only", true); });

    // The point of the mode: the incumbent the chain produced is reported,
    // not discarded.  `kInfeasible` — the status the pre-existing early
    // exit uses — would have been rewritten to `kOptimal` by
    // `cleanupSolve` here, which is why it could not be reused.
    REQUIRE(out.solution_status == kSolutionStatusFeasible);
    REQUIRE(out.objective < kHighsInf);
    // A heuristic incumbent on a minimisation is at or above the optimum.
    // Asserting a bound rather than a value: which heuristic wins the race
    // is not part of this contract.
    REQUIRE(out.objective >= kFlugplOptimum);
}

TEST_CASE("presolve-only: with no heuristic enabled it reports no solution", "[presolve-only]") {
    // The honest empty case.  `suite=off` hands the standalone FJ call
    // site back to HiGHS, and switching that off too leaves a solve that
    // runs no primal heuristic at all before the root LP — so presolve-only
    // has nothing to report.  It must still be a clean limit exit rather
    // than an infeasibility claim: the model is feasible, we simply never
    // looked.
    const SolveOutcome out = solve_outcome(kInstance, [](Highs& h) {
        require_option(h, "mip_heuristic_presolve_only", true);
        set_suite(h, "off");
        require_option(h, "mip_heuristic_run_feasibility_jump", false);
    });

    REQUIRE(out.model_status == HighsModelStatus::kSolutionLimit);
    REQUIRE(out.solution_status != kSolutionStatusFeasible);
    REQUIRE(out.objective == kHighsInf);
    REQUIRE(out.node_count == 0);
    REQUIRE(out.lp_iterations == 0);
}

TEST_CASE("presolve-only: the default path is unchanged", "[presolve-only]") {
    // The other half of the contract, and the reason the default is
    // `false`: an option that silently truncated every solve would be
    // catastrophic and would look exactly like a heuristic regression.
    // Same instance, same pins, option left alone.
    const SolveOutcome out = solve_outcome(kInstance, [](Highs& h) {
        // Tightened so the assertion below is on the true optimum rather
        // than on whatever the default 1e-4 relative gap permits.
        require_option(h, "mip_rel_gap", 0.0);
    });

    REQUIRE(out.status == HighsStatus::kOk);
    REQUIRE(out.model_status == HighsModelStatus::kOptimal);
    REQUIRE(out.objective == kFlugplOptimum);
    // The solve really did run B&B: the presolve-only witnesses above all
    // come out the other way round.
    REQUIRE(out.node_count > 0);
    REQUIRE(out.lp_iterations > 0);
    REQUIRE(out.dual_bound > -kHighsInf);
}

TEST_CASE("presolve-only: setting it false explicitly is the default path", "[presolve-only]") {
    // Guards the direction of the flag.  A patch that registered the
    // option with an inverted sense, or a call site that tested it
    // backwards, passes every case above and fails here.
    const SolveOutcome out = solve_outcome(kInstance, [](Highs& h) {
        require_option(h, "mip_heuristic_presolve_only", false);
        require_option(h, "mip_rel_gap", 0.0);
    });

    REQUIRE(out.model_status == HighsModelStatus::kOptimal);
    REQUIRE(out.objective == kFlugplOptimum);
    REQUIRE(out.node_count > 0);
}
