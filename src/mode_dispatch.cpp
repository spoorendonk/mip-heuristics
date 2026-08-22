#include "mode_dispatch.h"

#include "effort_ledger.h"
#include "fj.h"
#include "fpr.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "io/HighsIO.h"
#include "local_mip.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "scylla.h"

#include <algorithm>
#include <array>
#include <string>

namespace heuristics {

namespace {

// Per-heuristic effort options (#110).  Each of the four presolve
// heuristics reads its own `mip_heuristic_<name>_effort` multiplier —
// registered by `third_party/highs_patch/apply_patch.cmake`, defaults
// documented in `docs/PARAMETERS.md` — and turns it into a budget with
// `heuristic_effort_budget(nnz, value)`: `nnz << 12` effort units at the
// anchor 0.05, linear in the value, so a budget still scales with model
// size.
//
// This replaced one shared envelope split by `kWeight*` constants
// proportional to each heuristic's `effort_per_ms`.  That model could not
// express what a calibration needs: the heuristics' effort counters are in
// genuinely different units (FJ step-units; FPR/LocalMIP coefficient
// accesses; Scylla PDLP iters x nnz), so the split had to be measured, and
// because the envelope was shared, raising one heuristic's budget lowered
// the other two — there was no way to ask what a good budget for LocalMIP
// is without confounding it with FPR and Scylla.  FJ sat outside the
// scheme entirely, on a fixed allowance no option reached.  The weights,
// their calibration procedure, and the measured limits of the
// equal-weight/equal-wall contract they never quite delivered are in git
// history (issue #71; #110 removed them).
//
// The defaults are the closest *scalar* approximation to what the shared
// envelope handed each heuristic, not a reproduction of it — no scalar can
// be, because the old share depended on the worker count and on which other
// heuristics the suite enabled, neither of which a constant can see.  They
// run 1.04x the old budget at N=1 and 4x from N=18 at `suite=all`, and
// 0.29x / 0.61x / 0.10x for fpr / local_mip / scylla when that heuristic
// runs alone.  Only FJ is exact, at every N and every suite.  The full
// accounting is in `third_party/highs_patch/apply_patch.cmake`, where the
// defaults themselves live; retuning them is a separate change with its own
// measurements (#106).

// One heuristic's entry in the fixed FJ -> FPR -> LocalMIP -> Scylla
// chain.  `run_sequential` is a filtered loop over the table below; the
// four near-identical `if (enabled && !deadline) { ... }` blocks it
// replaced were the last place a fifth heuristic would have had to be
// wired in by hand.
struct HeuristicConfig {
    const char* name;
    // kSolutionSource* tag the sink attributes this heuristic's solutions
    // with, so the HiGHS log credits the right finder.
    int source_tag;
    // Which `mip_heuristic_suite` bit enables this entry.
    bool HeuristicFlags::* flag;
    // This entry's effort-budget multiplier option.
    double HighsOptionsStruct::* effort;
    // Whether that option sizes one *worker's* allowance rather than the
    // whole dispatch.  Only FJ sets it: vanilla HiGHS gives its single FJ
    // thread `nnz << 10` steps, and each of our N workers matches that, so
    // FJ's dispatch total scales with the worker count where the other
    // three are divided across it by `make_budget`.  Spelled out rather
    // than `per_worker`, which in this translation unit already means
    // `HeuristicBudget::per_worker` — a size_t budget, not a flag.
    bool budget_is_per_worker;
    size_t (*run)(const ProblemView&, const HeuristicBudget&, ExecutionContext&, IncumbentSink&);
};

constexpr auto kChain = std::to_array<HeuristicConfig>({
    {"fj", kSolutionSourceFJ, &HeuristicFlags::fj, &HighsOptionsStruct::mip_heuristic_fj_effort,
     true, &fj::run},
    {"fpr", kSolutionSourceFPR, &HeuristicFlags::fpr, &HighsOptionsStruct::mip_heuristic_fpr_effort,
     false, &fpr::run},
    {"local_mip", kSolutionSourceLocalMIP, &HeuristicFlags::local_mip,
     &HighsOptionsStruct::mip_heuristic_local_mip_effort, false, &local_mip::run},
    {"scylla", kSolutionSourceScylla, &HeuristicFlags::scylla,
     &HighsOptionsStruct::mip_heuristic_scylla_effort, false, &scylla::run},
});

// Each enabled heuristic runs in turn, with its own effort budget and the
// full thread pool.
//
// A single `IncumbentSink` is constructed here and threaded through all
// heuristics so that solutions found by an earlier heuristic (e.g. FJ)
// become available as pool-restart seeds for later heuristics (FPR,
// LocalMIP).  Each entry carries its originating heuristic's source tag
// (see incumbent_sink.h / #73).
bool run_sequential(HighsMipSolver& mipsolver, const HeuristicFlags& flags) {
    const bool any_enabled =
        std::ranges::any_of(kChain, [&](const HeuristicConfig& h) { return flags.*h.flag; });
    if (!any_enabled) {
        return false;
    }

    const HighsOptions& options = *mipsolver.options_mip_;
    ExecutionContext exec = make_exec(mipsolver);

    // Check out before the transpose, not only before each heuristic.  Each
    // heuristic used to build its own CSC behind its own deadline check, so
    // an already-terminated dispatch built none; hoisting the build out of
    // all four would otherwise make it unconditional, and it is the single
    // most expensive piece of setup in this function.
    if (exec.terminated()) {
        return false;
    }

    EffortLedger ledger(mipsolver);

    // Built once for the whole chain: the CSC transpose and the derived
    // sizes are the same for all four heuristics, and the row-major buffers
    // they come from are frozen by `runSetup()` before dispatch.  Each
    // heuristic used to build its own identical copy.  `csc` owns the
    // storage `problem` views, so it has to outlive the loop below.
    CscMatrix csc;
    const ProblemView problem = make_problem(mipsolver, csc);

    // One sink for the whole sequential chain, so a solution found by an
    // earlier heuristic (say FJ) is available as a pool-restart seed for
    // the later ones.  Its constructor seeds the pool from the incumbent
    // with the generic kSolutionSourceHeuristic tag; `set_source` below
    // re-tags it per heuristic so each entry carries its finder's tag.
    IncumbentSink sink(mipsolver, kSolutionSourceHeuristic);

    // All four heuristics return the effort they consumed and hand it to
    // the ledger, which is the single point of effort accounting for the
    // whole patch (issue #79 and its follow-up that extended LocalMIP's
    // contract to FJ, FPR and Scylla; #94 brought the dive-time `fpr_lp`
    // onto the same path).  No heuristic self-books.  All
    // bookings happen on the main thread after each parallel region has
    // joined, so `EffortLedger` reads/writes the counter without
    // synchronisation — do not move any of them into a worker without
    // revisiting this, and the matching note in effort_ledger.h.
    // (Historical note: local_mip used to early-return when
    // `mipdata->incumbent.empty()` so its [Sequential] line was absent
    // on a first solve.  Since issue #75 it runs the paper's
    // construction phase on cold start and emits a non-zero effort even
    // when no upstream heuristic produced a feasible solution.)
    //
    // Wall-ms is measured in this outer frame so all four measurements
    // share a clock and include each heuristic's own setup
    // (`precompute_var_orders`, `ContestedPdlp` construction, worker
    // construction) — what users actually pay for.  The shared CSC build
    // sits outside all four, since it is no longer any one of them.
    auto run_and_charge = [&](const char* name, auto&& call) {
        // `found` is the sink's accepted-offer count moving across this
        // heuristic's dispatch.  Read either side of the call, on this
        // thread, with the parallel region joined at both points.
        const size_t accepted_before = sink.accepted();
        const double t0_s = ledger.now_s();
        const size_t effort = call();
        ledger.charge_presolve(name, effort, sink.accepted() > accepted_before, t0_s,
                               ledger.now_s());
    };

    // Each heuristic's inner loops also poll the deadline, but their own
    // setup (precompute_var_orders, ContestedPdlp construction) runs before
    // that first inner poll; re-checking here skips it once the budget is
    // exhausted.  `exec.terminated()` is safe to call from this sequential
    // outer loop — the previous heuristic's parallel region has already
    // joined, so there is no concurrent access.
    for (const HeuristicConfig& h : kChain) {
        if (!(flags.*h.flag) || exec.terminated()) {
            continue;
        }
        // The heuristic's own option, sized against this model: a
        // whole-dispatch total, except for FJ, whose option sizes one
        // worker's allowance and therefore scales with the pool.
        const size_t sized = heuristic_effort_budget(problem.nnz, options.*h.effort);
        const HeuristicBudget slice = make_budget(
            h.budget_is_per_worker ? sized * exec.num_workers : sized, exec.num_workers);
        sink.set_source(h.source_tag);
        run_and_charge(h.name, [&]() -> size_t { return h.run(problem, slice, exec, sink); });
    }

    return false;
}

}  // namespace

HeuristicFlags effective_flags(const HighsOptions& options, bool* recognized) {
    const std::string& suite = options.mip_heuristic_suite;

    // Fail open on an unrecognised value: running everything is the same
    // thing the default does, and silently disabling all four heuristics
    // because of a typo is the worse failure.  The caller warns.
    HeuristicFlags flags{true, true, true, true};
    bool known = true;
    if (suite == "off") {
        flags = {false, false, false, false};
    } else if (suite == "fj") {
        flags = {true, false, false, false};
    } else if (suite == "fpr") {
        flags = {false, true, false, false};
    } else if (suite == "local_mip") {
        flags = {false, false, true, false};
    } else if (suite == "scylla") {
        flags = {false, false, false, true};
    } else if (suite != "all") {
        known = false;
    }

    // Upstream's own FJ switch still means what it says.  At suite=off the
    // patch leaves it gating HiGHS's native FJ call site; everywhere else it
    // gates ours, so `mip_heuristic_run_feasibility_jump=false` turns
    // FeasibilityJump off in every configuration rather than only one.
    flags.fj = flags.fj && options.mip_heuristic_run_feasibility_jump;

    if (recognized != nullptr) {
        *recognized = known;
    }
    return flags;
}

bool run_presolve(HighsMipSolver& mipsolver) {
    const HighsOptions& options = *mipsolver.options_mip_;

    // The two warnings below are **API, not prose**.  Both describe a solve
    // that ran something other than what its configuration asked for while
    // still exiting cleanly with an ordinary-looking log, so they are the only
    // signal distinguishing such a run from a good one.
    // `bench/run_benchmark.py` greps for them (`CONFIG_IGNORED_WARNINGS`) and
    // discards the affected result rather than recording a mislabelled tree —
    // a benchmark directory named for one configuration holding runs of
    // another is exactly the silent-failure mode that harness exists to
    // prevent.  If you reword either string, update that list in the same
    // commit; `tests/test_smoke.cpp` pins both substrings against this
    // binary's real output and will fail until you do.
    bool recognized = false;
    const HeuristicFlags flags = effective_flags(options, &recognized);
    if (!recognized) {
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Unknown mip_heuristic_suite value \"%s\"; running all heuristics.\n",
                     options.mip_heuristic_suite.c_str());
    } else if (!flags.fj && !flags.fpr && !flags.local_mip && !flags.scylla &&
               options.mip_heuristic_suite != "off") {
        // Only reachable as `suite=fj` with mip_heuristic_run_feasibility_jump
        // false, which asks for FJ and then takes it away.  That run is
        // heuristic-free without being `off`, so it also loses the native FJ
        // call site — a benchmark row labelled "FJ isolated" would silently
        // measure vanilla-minus-FJ.  Say so rather than leave it silent.
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "mip_heuristic_suite=\"%s\" selects only FeasibilityJump, which "
                     "mip_heuristic_run_feasibility_jump=false disables; no heuristic will "
                     "run. Use mip_heuristic_suite=off for a vanilla-equivalent run.\n",
                     options.mip_heuristic_suite.c_str());
    }

    return run_sequential(mipsolver, flags);
}

}  // namespace heuristics
