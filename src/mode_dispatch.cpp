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

namespace heuristics {

namespace {

// Per-heuristic effort options (#110).  Each of the four presolve
// heuristics reads its own `mip_heuristic_<name>_effort` multiplier —
// registered by `third_party/highs_patch/apply_patch.cmake`, defaults
// documented in `docs/PARAMETERS.md` — and turns it into a budget with
// `heuristic_effort_budget(nnz, value)`: `nnz << 10` effort units per
// unit of the option, linear in it, so a budget still scales with model
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
// heuristics were enabled, neither of which a constant can see.  They
// run 1.04x the old budget at N=1 and 4x from N=18 with every heuristic on, and
// 0.29x / 0.61x / 0.10x for fpr / local_mip / scylla when that heuristic
// runs alone.  Only FJ is exact, at every N and every selection.  The full
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
    // This entry's effort-budget multiplier option, which is also what
    // selects it: at or below zero the heuristic does not run (#167).
    double HighsOptionsStruct::* effort;
    // An upstream `mip_heuristic_run_*` switch that additionally has to be
    // true, or null when the entry answers to its effort option alone.
    // Only FJ names one: `mip_heuristic_run_feasibility_jump` is HiGHS's
    // own switch for the heuristic ours replaces, so it keeps its meaning
    // rather than becoming a dead option on a patched binary.  The other
    // five upstream switches (rens, rins, root_reduced_cost, shifting,
    // zi_round) name heuristics we do not touch.
    const bool HighsOptionsStruct::* enable_switch;
    // Whether that option sizes one *worker's* allowance rather than the
    // whole dispatch.  Only FJ sets it: vanilla HiGHS gives its single FJ
    // thread `nnz << 10` steps, and each of our N workers matches that, so
    // FJ's dispatch total scales with the worker count where the other
    // three are divided across it by `make_budget`.  It governs the
    // patience option too — that value is expressed in the same scope as
    // the effort option it sits next to.  Spelled out rather than
    // `per_worker`, which in this translation unit already means
    // `HeuristicBudget::per_worker` — a size_t budget, not a flag.
    bool budget_is_per_worker;
    // This entry's patience option, as a multiple of `nnz << 10` -- the
    // same unit as `effort` above, so the two are directly comparable and
    // `patience < effort` is legible without a conversion (#116; an option
    // since #106, absolute since #111).
    // Absolute and instance-scaled: a heuristic that stops improving the
    // incumbent exits on this rather than on a fraction of an allowance
    // that someone tuned in isolation.  It was a `constexpr` in each
    // heuristic's own header until the calibration needed to search the
    // patience axis, which a rebuild-per-point cannot do; the
    // per-heuristic header still carries what that heuristic's effort
    // counter counts, which is why the four values are not comparable with
    // each other.
    // **0 means no gate at all** — see `patience_threshold`.
    double HighsOptionsStruct::* patience;
    DispatchOutcome (*run)(const ProblemView&, const HeuristicBudget&, ExecutionContext&,
                           IncumbentSink&);
};

constexpr auto kChain = std::to_array<HeuristicConfig>({
    {"fj", kSolutionSourceFJ, &HighsOptionsStruct::mip_heuristic_fj_effort,
     &HighsOptionsStruct::mip_heuristic_run_feasibility_jump, true,
     &HighsOptionsStruct::mip_heuristic_fj_patience, &fj::run},
    {"fpr", kSolutionSourceFPR, &HighsOptionsStruct::mip_heuristic_fpr_effort, nullptr, false,
     &HighsOptionsStruct::mip_heuristic_fpr_patience, &fpr::run},
    {"local_mip", kSolutionSourceLocalMIP, &HighsOptionsStruct::mip_heuristic_local_mip_effort,
     nullptr, false, &HighsOptionsStruct::mip_heuristic_local_mip_patience, &local_mip::run},
    {"scylla", kSolutionSourceScylla, &HighsOptionsStruct::mip_heuristic_scylla_effort, nullptr,
     false, &HighsOptionsStruct::mip_heuristic_scylla_patience, &scylla::run},
});

// Whether `h` runs at all under `options` (#167).
//
// The effort option is the selector: `mip_heuristic_<name>_effort` at or
// below zero means the heuristic does not run, which is the only way to
// exclude one.  It replaced `mip_heuristic_suite`, a string naming the
// heuristics to enable, and the replacement is a narrowing of two spellings
// to one rather than a new mechanism — a zero budget already declined every
// piece of work a heuristic does, down to `ProblemView`-sized setup, since
// `HeuristicBudget::disabled()` reached all four entry points (#106).  What
// the suite value added on top was the ability to say the same thing twice,
// and a string HiGHS does not validate, so a typo inside it ran a
// configuration nobody asked for.  What it is *not* is a loss of
// expressiveness: every subset of the five is still selectable, as the
// zero-pattern of five continuous options, which is the form #107's search
// already used.
//
// `enable_switch`, when the entry names one, is ANDed on top: upstream's
// own `mip_heuristic_run_feasibility_jump` still means what it says, so
// setting it false disables FeasibilityJump exactly as a zero effort does.
bool entry_enabled(const HighsOptions& options, const HeuristicConfig& h) {
    if (options.*h.effort <= 0.0) {
        return false;
    }
    return h.enable_switch == nullptr || options.*h.enable_switch;
}

// Each enabled heuristic runs in turn, with its own effort budget and the
// full thread pool.
//
// A single `IncumbentSink` is constructed here and threaded through all
// heuristics so that solutions found by an earlier heuristic (e.g. FJ)
// become available as pool-restart seeds for later heuristics (FPR,
// LocalMIP).  Each entry carries its originating heuristic's source tag
// (see incumbent_sink.h / #73).
bool run_sequential(HighsMipSolver& mipsolver) {
    const HighsOptions& options = *mipsolver.options_mip_;

    // Nothing to run means nothing to *build*: returning here is what keeps
    // a fully zeroed configuration free of the shared CSC transpose below,
    // which is the single most expensive piece of setup in this function
    // and is charged to no heuristic.  `fpr_lp` is deliberately not part of
    // this test — it runs during the B&B dive, reads its own option there,
    // and shares none of this setup.
    const bool any_chain_enabled = std::ranges::any_of(
        kChain, [&](const HeuristicConfig& h) { return entry_enabled(options, h); });
    if (!any_chain_enabled) {
        return false;
    }
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
        // `abandoned_setup` comes back from the heuristic rather than
        // being inferred here (issue #119).  Inferring it from
        // `outcome.effort == 0 && exec.past_deadline()` would be the same
        // mistake the log made, moved in-process: a dispatch entered with
        // a millisecond left constructs its workers, searches nothing and
        // returns zero without ever bailing in setup, and would be
        // mislabelled.  Only the bail site knows.
        const DispatchOutcome outcome = call();
        ledger.charge_presolve(name, outcome.effort, sink.accepted() > accepted_before, t0_s,
                               ledger.now_s(), outcome.abandoned_setup);
    };

    // Each heuristic's inner loops also poll the deadline, but their own
    // setup (precompute_var_orders, ContestedPdlp construction) runs before
    // that first inner poll; re-checking here skips it once the budget is
    // exhausted.  `exec.terminated()` is safe to call from this sequential
    // outer loop — the previous heuristic's parallel region has already
    // joined, so there is no concurrent access.
    for (const HeuristicConfig& h : kChain) {
        // A skipped heuristic emits no `[Sequential]` / `[Heur]` line at
        // all, which is what "skipped" has always meant here — the line is
        // written by `run_and_charge` and a heuristic excluded from the
        // configuration never reaches it.  So a zeroed heuristic is absent
        // from the trace rather than present with `effort=0`, and at the
        // shipped defaults, which disable Scylla and `fpr_lp`, a solve's
        // presolve trace is three lines.
        if (!entry_enabled(options, h) || exec.terminated()) {
            continue;
        }
        // The heuristic's own option, sized against this model: a
        // whole-dispatch total, except for FJ, whose option sizes one
        // worker's allowance and therefore scales with the pool.
        const size_t sized = heuristic_effort_budget(problem.nnz, options.*h.effort);
        const size_t total =
            h.budget_is_per_worker ? saturating_mul(sized, exec.num_workers) : sized;
        // The runner-level patience gate (issue #111).  Absolute, not
        // `total / 4`: the runner's counter aggregates every worker, so a
        // per-worker option is multiplied by the pool, and a
        // whole-dispatch one is used as it stands.  Clamped to a quarter
        // of `total` rather than to `total`, so a gate that exists fires
        // strictly before exhaustion instead of coinciding with it (#116)
        // — except at `patience_option == 0`, which is not a gate at all:
        // the threshold is unbounded and no clamp applies.  See
        // `patience_threshold`.
        //
        // FJ's option is per worker, like its effort option, so the
        // runner-level gate is it times the pool size; the other three are
        // dispatch-scoped and `make_budget` divides them back down.  The
        // product is taken in `double` and `heuristic_effort_budget`
        // saturates on the way to `size_t`, so the top of the option range
        // against a large `nnz` cannot wrap into a tiny threshold that
        // stops the heuristic almost immediately.  Zero survives it, which
        // is what makes "no gate" expressible from the option.
        const double patience_option = options.*h.patience;
        const double patience_per_base =
            h.budget_is_per_worker ? patience_option * static_cast<double>(exec.num_workers)
                                   : patience_option;
        const HeuristicBudget slice = make_budget(
            total, exec.num_workers, patience_threshold(problem.nnz, patience_per_base, total));
        sink.set_source(h.source_tag);
        run_and_charge(h.name,
                       [&]() -> DispatchOutcome { return h.run(problem, slice, exec, sink); });
    }

    return false;
}

}  // namespace

bool any_enabled(const HighsOptions& options) {
    // The four presolve entries, then the one heuristic that is not a chain
    // entry.  `fpr_lp` runs during the B&B dive and reads its own option in
    // `fpr_lp::run`; it is included here because this predicate answers
    // "can a solution of ours appear in this log", and an `FPR LP` display
    // row is one of those.  It honours no `enable_switch`: upstream's FJ
    // option has nothing to say about it.
    for (const HeuristicConfig& h : kChain) {
        if (entry_enabled(options, h)) {
            return true;
        }
    }
    return options.mip_heuristic_fpr_lp_effort > 0.0;
}

bool run_presolve(HighsMipSolver& mipsolver) {
    const HighsOptions& options = *mipsolver.options_mip_;

    // This warning is **API, not prose**.  It describes a solve that ran
    // something other than what its configuration asked for while still
    // exiting cleanly with an ordinary-looking log, so it is the only signal
    // distinguishing such a run from a good one.  `bench/run_benchmark.py`
    // greps for it (`CONFIG_IGNORED_WARNINGS`) and discards the affected
    // result rather than recording a mislabelled tree — a benchmark
    // directory named for one configuration holding runs of another is
    // exactly the silent-failure mode that harness exists to prevent.  If
    // you reword the string, update that list in the same commit;
    // `tests/test_smoke.cpp` pins the substring against this binary's real
    // output and will fail until you do.
    //
    // The condition is a configuration that asks for FeasibilityJump
    // through `mip_heuristic_fj_effort` and then takes it away through
    // upstream's `mip_heuristic_run_feasibility_jump`, with nothing else
    // enabled to take its place.  That run is heuristic-free while being
    // spelled like an "FJ isolated" row, which is the one contradiction the
    // five effort options can express — every other way of running nothing
    // is spelled as running nothing.
    if (options.mip_heuristic_fj_effort > 0.0 && !options.mip_heuristic_run_feasibility_jump &&
        !any_enabled(options)) {
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "mip_heuristic_fj_effort=%g selects only FeasibilityJump, which "
                     "mip_heuristic_run_feasibility_jump=false disables; no heuristic will "
                     "run.\n",
                     options.mip_heuristic_fj_effort);
    }

    return run_sequential(mipsolver);
}

}  // namespace heuristics
