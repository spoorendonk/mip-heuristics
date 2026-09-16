#include "Highs.h"
#include "test_common.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <string>
#include <vector>

// ===================================================================
// `mip_heuristic_<name>_effort = 0` genuinely disables a heuristic
// (#106, #167)
//
// Issue #107 expresses "this heuristic is excluded from the configuration"
// as effort 0, which turns the subset choice into a zero-pattern of five
// continuous parameters instead of a separate discrete dimension — the
// reduction that keeps its search tractable.  #167 made that the only
// encoding by retiring `mip_heuristic_suite`, so the option is now both the
// budget and the selector, and this file pins what the zero buys.
//
// It was not always free.  Three things had to be fixed before a zero
// budget was worth exactly what omitting a heuristic is worth:
//
//   * `run_opportunistic_loop` did decline a zero total, but three of the
//     four heuristics do real work before they reach it.  Scylla built a
//     `ContestedPdlp` (a whole `Highs` LP copy), the per-config variable
//     orders and N workers; FPR precomputed its variable orders.  None of
//     that is charged, so it was invisible in the effort total while still
//     costing wall time — and Scylla's is the expensive one.
//   * `make_budget` floored `attempt_cap` at 1, so a zero budget licensed
//     one attempt.  For Scylla one attempt is a whole PDLP solve, which
//     `attempt_cap` does not govern once started.
//   * `patience_threshold` special-cases a zero budget by returning the
//     *unclamped* threshold, so the one ceiling that could still have
//     bounded such a run was the one that did not apply.
//
// `HeuristicBudget::disabled()` is checked at the top of all four entry
// points, alongside `ProblemView::degenerate()`, and `make_budget` returns
// an all-zero budget at a zero total.
//
// Since #167 there are **two** gates, not one, and they are not redundant.
// `heuristics::entry_enabled` skips the `kChain` entry outright at effort
// <= 0, so nothing downstream is reached at all and no `[Heur]` line is
// emitted.  `HeuristicBudget::disabled()` still guards the case that gate
// cannot see: an effort small enough that `heuristic_effort_budget` floors
// the derived budget to zero while the option itself is above zero.  The
// cases below cover both, and the tiny-effort one is what keeps the entry
// point guard from becoming dead code no test can reach.
//
// The `fpr_lp` half of the same property is asserted where it can be seen —
// on the dispatch, in `tests/test_fpr_lp.cpp` — since `flugpl` never
// reaches a B&B dive.
// ===================================================================

namespace {

// Value of `key=` in `line`, or an empty string when absent.
std::string field_of(const std::string& line, const std::string& key) {
    const std::string tag = key + "=";
    const auto pos = line.find(tag);
    if (pos == std::string::npos) {
        return {};
    }
    const auto start = pos + tag.size();
    const auto end = line.find_first_of(" \n", start);
    return line.substr(start, end == std::string::npos ? end : end - start);
}

// The `name=` of every `[Heur] phase=presolve` line, in emission order.
std::vector<std::string> presolve_heur_names(const std::vector<std::string>& lines) {
    std::vector<std::string> out;
    for (const auto& line : lines) {
        if (line.contains("[Heur] name=") && field_of(line, "phase") == "presolve") {
            out.push_back(field_of(line, "name"));
        }
    }
    return out;
}

// `[HeurSol]` lines attributed to `name`, i.e. the solutions that
// heuristic offered the shared pool.
size_t offers_by(const std::vector<std::string>& lines, const std::string& name) {
    const std::string tag = "[HeurSol] name=" + name + " ";
    return static_cast<size_t>(
        std::ranges::count_if(lines, [&](const std::string& l) { return l.contains(tag); }));
}

// One reproducible solve, with `configure` applied on top.
//
// `threads=1` and a pinned `random_seed` are what make the trace
// reproducible; `log_dev_level=3` is what makes it exist.
template <typename Configure>
std::vector<std::string> trace_solve(const char* inst, Configure&& configure) {
    ScopedThreadPin pin;
    return solve_capturing_log(inst, [&](Highs& h) {
        require_option(h, "log_dev_level", 3);
        require_option(h, "threads", 1);
        require_option(h, "random_seed", 1);
        configure(h);
    });
}

// The four presolve heuristics and the solution-source character each
// prints in HiGHS's incumbent display.
struct Case {
    const char* name;
    char source_code;
};

constexpr std::array<Case, 4> kCases = {{
    {"fj", 'J'},
    {"fpr", 'A'},
    {"local_mip", 'M'},
    {"scylla", 'G'},
}};

// Every heuristic enabled except `name`.  Scylla ships at effort 0, so it
// is raised first and then zeroed again by the selection on the arms that
// exclude it — `select_heuristics` only ever writes zeros, so the order
// matters.
std::vector<std::string> all_but(const char* inst, const char* name) {
    return trace_solve(inst, [&](Highs& h) {
        enable_scylla(h);
        require_option(h, std::string("mip_heuristic_") + name + "_effort", 0.0);
    });
}

}  // namespace

// A zeroed heuristic is not dispatched at all: `run_sequential` skips its
// `kChain` entry before the ledger is touched, so it emits no `[Heur]`
// line, offers nothing to the pool, and reaches no incumbent display row.
//
// The absent line is the load-bearing assertion, and it is stronger than
// the `effort=0` line it replaced.  A dispatched-but-idle heuristic could
// still have paid for setup — `precompute_var_orders`, `ContestedPdlp`
// construction — and charged none of it, which is exactly the hole #106
// was filed against.  No line means `run_and_charge` never ran, which means
// the entry point was never called, which is a structural statement about
// the setup rather than a measurement of it.  That is what retired the
// `[serial]`-tagged wall-clock case this file used to carry: it compared a
// disabled heuristic's `[Heur]` window against a threshold, and there is no
// longer a window to time.
TEST_CASE("effort-zero: a zeroed heuristic is absent from the trace", "[effort-zero]") {
    for (const Case& c : kCases) {
        INFO("heuristic " << c.name);
        const auto lines = all_but("flugpl.mps", c.name);
        const auto names = presolve_heur_names(lines);

        CHECK(std::ranges::find(names, std::string(c.name)) == names.end());
        CHECK(offers_by(lines, c.name) == 0);
        CHECK(!source_codes(lines).contains(c.source_code));
    }
}

// The complement, and the guard against every case above passing because
// the trace is empty: each heuristic *does* appear when it is the one left
// enabled.
TEST_CASE("effort-zero: an enabled heuristic is present in the trace", "[effort-zero]") {
    for (const Case& c : kCases) {
        INFO("heuristic " << c.name);
        const auto lines = trace_solve("flugpl.mps", [&](Highs& h) {
            enable_scylla(h);
            select_heuristics(h, c.name);
        });
        const auto names = presolve_heur_names(lines);

        CHECK(names == std::vector<std::string>{c.name});
    }
}

// Zeroing every heuristic leaves the chain with nothing to build, and
// `run_sequential` returns before `make_problem` — the shared CSC
// transpose, which is the most expensive piece of setup in that function
// and is charged to no heuristic.  Asserted as an empty presolve trace,
// which is what "returned before the ledger existed" looks like from
// outside.
//
// `p0548` is the bundled instance with the most columns, so the one whose
// setup would cost most if any of it ran.
TEST_CASE("effort-zero: a fully zeroed chain builds nothing", "[effort-zero]") {
    const auto lines = trace_solve("p0548.mps", [](Highs& h) { select_heuristics(h, "off"); });
    CHECK(presolve_heur_names(lines).empty());
}

// Scylla's setup is the expensive one — a `ContestedPdlp` wraps a whole
// `Highs` LP copy, and the per-config variable orders reach the clique
// table (`clique_cover::build_clique_cover`, once per config).
// `[ScyllaOverlap]` is emitted at the end of `scylla::run` from the workers
// it constructed, so its absence is a direct observable that none of that
// setup ran.
//
// The third arm is the one that reaches `HeuristicBudget::disabled()`
// rather than `entry_enabled`: an effort far above zero but small enough
// that `heuristic_effort_budget(nnz, effort)` floors to zero, so the chain
// dispatches Scylla — the `[Heur]` line is emitted — and the entry point
// declines before building anything.  Without this arm, deleting
// `budget.disabled()` from `scylla::run` would leave the whole file green.
TEST_CASE("effort-zero: scylla builds no PDLP wrapper at a zero budget", "[effort-zero]") {
    CHECK(log_contains(trace_solve("flugpl.mps",
                                   [](Highs& h) {
                                       // The control arm has to actually run
                                       // Scylla, which ships at effort 0.
                                       enable_scylla(h);
                                       select_heuristics(h, "scylla");
                                   }),
                       "[ScyllaOverlap]"));

    CHECK(!log_contains(all_but("flugpl.mps", "scylla"), "[ScyllaOverlap]"));

    const auto tiny = trace_solve("flugpl.mps", [](Highs& h) {
        select_heuristics(h, "scylla");
        require_option(h, "mip_heuristic_scylla_effort", 1e-9);
    });
    INFO("a sub-unit effort must still dispatch, so the entry point guard is what declines");
    CHECK(presolve_heur_names(tiny) == std::vector<std::string>{"scylla"});
    CHECK(!log_contains(tiny, "[ScyllaOverlap]"));
}
