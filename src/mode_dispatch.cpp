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
#include <string_view>
#include <vector>

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
    // three are divided across it by `make_budget`.  It governs
    // `stall_per_nnz` too — that constant is expressed in the same scope
    // as the effort option it sits next to.  Spelled out rather than
    // `per_worker`, which in this translation unit already means
    // `HeuristicBudget::per_worker` — a size_t budget, not a flag.
    bool budget_is_per_worker;
    // This entry's stall threshold, in effort units per constraint-matrix
    // nonzero (issue #111).  Absolute and instance-scaled: a heuristic
    // that stops finding things exits on this rather than on a fraction
    // of an allowance that someone tuned in isolation.  Defined in the
    // heuristic's own header, since only it knows what its effort counter
    // counts.
    size_t stall_per_nnz;
    size_t (*run)(const ProblemView&, const HeuristicBudget&, ExecutionContext&, IncumbentSink&);
};

constexpr auto kChain = std::to_array<HeuristicConfig>({
    {"fj", kSolutionSourceFJ, &HeuristicFlags::fj, &HighsOptionsStruct::mip_heuristic_fj_effort,
     true, fj::kStallPerNnzFj, &fj::run},
    {"fpr", kSolutionSourceFPR, &HeuristicFlags::fpr, &HighsOptionsStruct::mip_heuristic_fpr_effort,
     false, fpr::kStallPerNnzFpr, &fpr::run},
    {"local_mip", kSolutionSourceLocalMIP, &HeuristicFlags::local_mip,
     &HighsOptionsStruct::mip_heuristic_local_mip_effort, false, local_mip::kStallPerNnzLocalMip,
     &local_mip::run},
    {"scylla", kSolutionSourceScylla, &HeuristicFlags::scylla,
     &HighsOptionsStruct::mip_heuristic_scylla_effort, false, scylla::kStallPerNnzScylla,
     &scylla::run},
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
        const size_t total = h.budget_is_per_worker ? sized * exec.num_workers : sized;
        // The runner-level stall gate (issue #111).  Absolute, not
        // `total / 4`: the runner's counter aggregates every worker, so a
        // per-worker constant is multiplied by the pool, and a
        // whole-dispatch one is used as it stands.  Clamped to `total`,
        // which is the only thing the gate may not exceed.
        const size_t stall_per_nnz =
            h.budget_is_per_worker ? h.stall_per_nnz * exec.num_workers : h.stall_per_nnz;
        const HeuristicBudget slice = make_budget(
            total, exec.num_workers, stall_threshold(problem.nnz, stall_per_nnz, total));
        sink.set_source(h.source_tag);
        run_and_charge(h.name, [&]() -> size_t { return h.run(problem, slice, exec, sink); });
    }

    return false;
}

// ── mip_heuristic_suite ──
//
// The value is either one of two whole-value aliases — `off` (no heuristic)
// and `all` (every one) — or a comma-separated list of heuristic names,
// unioned: `fj,fpr` runs those two and nothing else.  Order is irrelevant,
// whitespace around a token is ignored, and repeating a name is harmless.
// Fifteen non-empty subsets exist and the six single values could express
// five of them (#112), which left the FJ+FPR+LocalMIP composition the
// recorded benchmark table was measured at inexpressible.
//
// The legal names are `kChain`'s own `name` field rather than a second
// table, so they cannot drift from the `[Heur] name=<n>` traces those same
// strings produce: the name a user reads in the log is the name they select
// with, and a fifth heuristic stays a single table edit.
//
// `off` is an alias only as the *whole* value, never as a token in a list.
// It is not merely "the empty set": the patched HiGHS tree tests
// `mip_heuristic_suite == "off"` verbatim to hand back upstream's own
// FeasibilityJump call site and its display key (see
// `third_party/highs_patch/apply_patch.cmake`), so a value that selected
// nothing without being that exact string would run no heuristic at all
// while quietly not being the vanilla-equivalent configuration.  `fj,off`
// is therefore an unrecognised token, and warns.  `setLocalOptionValue`
// strips *spaces* — only spaces — from both ends of a string option's value
// and lower-cases it before storing (the options-file loader strips tabs,
// newlines and quotes first), so ` OFF ` arrives as `off` and the exact
// comparison on both sides of the patch boundary is safe: whatever neither
// strips fails `== "off"` identically here and in the patched tree.

// `token` without surrounding ASCII whitespace.  HiGHS strips spaces around
// the whole value but not around a separator inside it, so `fj, fpr` needs
// this to mean the same thing as `fj,fpr`.
std::string_view trim(std::string_view token) {
    constexpr std::string_view kSpace = " \t\n\v\f\r";
    const size_t first = token.find_first_not_of(kSpace);
    if (first == std::string_view::npos) {
        return {};
    }
    return token.substr(first, token.find_last_not_of(kSpace) - first + 1);
}

// The HeuristicFlags bit `token` names, or nullptr if it names no heuristic.
//
// A `std::ranges::find` over `kChain` would read better, but the iterator it
// returns has nowhere portable to live: `std::array::const_iterator` is a raw
// pointer on libstdc++ and libc++ and a class type on MSVC, so `const auto`
// trips readability-qualified-auto while the `const auto *const` that check
// asks for is the assumption that breaks on MSVC.  Returning the member
// pointer sidesteps the choice.  Do not "simplify" this back.
bool HeuristicFlags::* suite_flag(std::string_view token) {
    for (const HeuristicConfig& h : kChain) {
        if (token == h.name) {
            return h.flag;
        }
    }
    return nullptr;
}

// Union the heuristics named by the comma-separated `suite`, appending every
// token that names none to `unknown` (deduplicated) for the caller's
// warning.  The empty string is one empty token rather than zero tokens, so
// a bare `mip_heuristic_suite=` is an unrecognised value and not a silent
// `off` — as is the empty token a stray trailing comma leaves behind.
HeuristicFlags parse_suite_list(std::string_view suite, std::vector<std::string_view>& unknown) {
    HeuristicFlags flags{false, false, false, false};
    for (size_t pos = 0;;) {
        const size_t comma = suite.find(',', pos);
        const size_t count = comma == std::string_view::npos ? comma : comma - pos;
        const std::string_view token = trim(suite.substr(pos, count));
        if (bool HeuristicFlags::* const flag = suite_flag(token); flag != nullptr) {
            flags.*flag = true;
        } else if (std::ranges::find(unknown, token) == unknown.end()) {
            unknown.push_back(token);
        }
        if (comma == std::string_view::npos) {
            return flags;
        }
        pos = comma + 1;
    }
}

// `tokens` quoted and comma-joined, for a warning that has to name what it
// rejected: {fpr2, walksat} -> `"fpr2", "walksat"`.
std::string quote_join(const std::vector<std::string_view>& tokens) {
    std::string joined;
    for (const std::string_view token : tokens) {
        if (!joined.empty()) {
            joined += ", ";
        }
        joined += '"';
        joined += token;
        joined += '"';
    }
    return joined;
}

}  // namespace

HeuristicFlags effective_flags(const HighsOptions& options, SuiteDiagnosis* diagnosis) {
    const std::string& suite = options.mip_heuristic_suite;

    std::vector<std::string_view> unknown;
    HeuristicFlags flags{true, true, true, true};
    if (suite == "off") {
        flags = {false, false, false, false};
    } else if (suite != "all") {
        flags = parse_suite_list(suite, unknown);
        // Fail open on an unrecognised token: running everything is the same
        // thing the default does, and silently disabling heuristics because
        // of a typo is the worse failure — inside a list it would quietly
        // demote a two-heuristic run to a one-heuristic one, so a results
        // tree named `fj+fpr` would hold runs of `fj`.  The caller warns and
        // names the token.
        if (!unknown.empty()) {
            flags = {true, true, true, true};
        }
    }

    // Upstream's own FJ switch still means what it says.  At suite=off the
    // patch leaves it gating HiGHS's native FJ call site; everywhere else it
    // gates ours, so `mip_heuristic_run_feasibility_jump=false` turns
    // FeasibilityJump off in every configuration rather than only one.
    flags.fj = flags.fj && options.mip_heuristic_run_feasibility_jump;

    if (diagnosis != nullptr) {
        diagnosis->unknown_tokens = quote_join(unknown);
        diagnosis->unknown_count = unknown.size();
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
    SuiteDiagnosis diagnosis;
    const HeuristicFlags flags = effective_flags(options, &diagnosis);
    if (diagnosis.unknown_count > 0) {
        // Naming the token is what makes this usable on a list value: the
        // value alone leaves the reader to spot which of `fj,fpr,locl_mip`
        // is wrong, and the run it describes silently executed all four.
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Unknown mip_heuristic_suite value \"%s\": unrecognised %s %s; running all "
                     "heuristics.\n",
                     options.mip_heuristic_suite.c_str(),
                     diagnosis.unknown_count == 1 ? "token" : "tokens",
                     diagnosis.unknown_tokens.c_str());
    } else if (!flags.fj && !flags.fpr && !flags.local_mip && !flags.scylla &&
               options.mip_heuristic_suite != "off") {
        // Only reachable from a value naming FJ and nothing else (`fj`, or a
        // list whose tokens are all `fj`) with mip_heuristic_run_feasibility_jump
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
