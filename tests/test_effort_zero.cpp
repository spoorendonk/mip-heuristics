#include "Highs.h"
#include "test_common.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

// ===================================================================
// `mip_heuristic_<name>_effort = 0` genuinely disables a heuristic (#106)
//
// Issue #107 expresses "this heuristic is excluded from the configuration"
// as effort 0, which turns the subset choice into a zero-pattern of four
// continuous parameters instead of a separate discrete dimension — the
// reduction that keeps its search tractable.  That is only sound if a zero
// budget is indistinguishable from omitting the heuristic, and it was not:
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
//   * `stall_threshold` special-cases a zero budget by returning the
//     *unclamped* threshold, so the one ceiling that could still have
//     bounded such a run was the one that did not apply.
//
// `HeuristicBudget::disabled()` is now checked at the top of all four
// entry points, alongside `ProblemView::degenerate()`, and `make_budget`
// returns an all-zero budget at a zero total.
//
// One asymmetry is deliberate and is *not* fixed here: omitting `fpr` from
// `mip_heuristic_suite` also disables the dive-time `fpr_lp`, via
// `heuristics::effective_flags`, while `mip_heuristic_fpr_effort = 0` does
// not — `fpr_lp` draws from upstream's `mip_heuristic_effort` envelope and
// never reads the presolve option.  That is a real property of the option,
// not a limitation of these tests, so every comparison below is scoped to
// `phase=presolve`.  The tuning target runner derives the suite value from
// the zero-pattern, which is where the two are reconciled.
// ===================================================================

namespace {

// One `[Heur]` observation, reduced to the fields that are deterministic.
//
// `effort` and `found` are: at `threads=1` with a pinned `random_seed`,
// repeated solves of a bundled instance reproduce them exactly (only the
// wall-clock fields move).  That is what lets the equivalence check below
// compare traces rather than just objectives.
struct HeurLine {
    std::string name;
    std::string phase;
    unsigned long long effort = 0;
    bool found = false;

    // Deliberately outside `operator==`: the equivalence assertions below
    // compare traces across two solves, and wall time is the one field that
    // is not reproducible.  It is carried anyway for the setup-cost test,
    // which is the only reader.
    double wall_ms = 0.0;

    bool operator==(const HeurLine& other) const {
        return name == other.name && phase == other.phase && effort == other.effort &&
               found == other.found;
    }
};

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

std::vector<HeurLine> heur_lines(const std::vector<std::string>& lines) {
    std::vector<HeurLine> out;
    for (const auto& line : lines) {
        if (!line.contains("[Heur] name=")) {
            continue;
        }
        out.push_back(HeurLine{field_of(line, "name"), field_of(line, "phase"),
                               std::strtoull(field_of(line, "effort").c_str(), nullptr, 10),
                               field_of(line, "found") == "1",
                               std::strtod(field_of(line, "wall_ms").c_str(), nullptr)});
    }
    return out;
}

// The presolve-phase entries only — see the `fpr_lp` note in the header
// comment.
std::vector<HeurLine> presolve_lines(const std::vector<std::string>& lines) {
    std::vector<HeurLine> out;
    for (auto& entry : heur_lines(lines)) {
        if (entry.phase == "presolve") {
            out.push_back(entry);
        }
    }
    return out;
}

std::vector<HeurLine> without(const std::vector<HeurLine>& lines, const std::string& name) {
    std::vector<HeurLine> out;
    for (const auto& entry : lines) {
        if (entry.name != name) {
            out.push_back(entry);
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
// `threads=1` and a pinned `random_seed` are what make `[Heur] effort`
// reproducible; `log_dev_level=3` is what makes the traces exist.
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

// Solve with every heuristic enabled but `name` zeroed.
std::vector<std::string> zeroed(const char* inst, const char* name) {
    const std::string option = std::string("mip_heuristic_") + name + "_effort";
    return trace_solve(inst, [&](Highs& h) {
        set_suite(h, "all");
        require_option(h, option, 0.0);
    });
}

// Solve with `name` left out of the suite entirely.
std::vector<std::string> omitted(const char* inst, const char* suite) {
    return trace_solve(inst, [&](Highs& h) { set_suite(h, suite); });
}

// `all` minus one, in chain order — the spelling `mip_heuristic_suite`
// requires (`run_benchmark.py` names the same subsets with `+`).
struct Case {
    const char* name;
    const char* complement;
    char source_code;  // the solution-source character this heuristic prints
};

constexpr std::array<Case, 4> kCases = {{
    {"fj", "fpr,local_mip,scylla", 'J'},
    {"fpr", "fj,local_mip,scylla", 'A'},
    {"local_mip", "fj,fpr,scylla", 'M'},
    {"scylla", "fj,fpr,local_mip", 'G'},
}};

}  // namespace

// A zeroed heuristic is dispatched — `run_sequential` still walks its
// `kChain` entry — but does nothing: no charged effort, no accepted
// solution, and not one offer to the pool.  Asserting on the charged
// effort and on `found` rather than on wall time, which is not a
// property of the change.
TEST_CASE("effort-zero: a zeroed heuristic charges nothing and offers nothing", "[effort-zero]") {
    for (const Case& c : kCases) {
        INFO("heuristic " << c.name);
        const auto lines = zeroed("flugpl.mps", c.name);
        const auto presolve = presolve_lines(lines);

        const auto entry =
            std::ranges::find_if(presolve, [&](const HeurLine& l) { return l.name == c.name; });
        REQUIRE(entry != presolve.end());
        CHECK(entry->effort == 0);
        CHECK(entry->found == false);

        // Nothing was offered, so nothing could have been accepted — the
        // stronger statement, since `found` only reports acceptance.
        CHECK(offers_by(lines, c.name) == 0);
        // And nothing reached HiGHS's incumbent display under this
        // heuristic's source character either.
        CHECK(!source_codes(lines).contains(c.source_code));
    }
}

// Scylla's setup is the expensive one — a `ContestedPdlp` wraps a whole
// `Highs` LP copy, and the per-config variable orders reach
// `cliquePartition`.  `[ScyllaOverlap]` is emitted at the end of
// `scylla::run` from the workers it constructed, so its absence is a
// direct observable that none of that setup ran.  It is also exactly what
// omitting Scylla from the suite produces, which is the point.
TEST_CASE("effort-zero: scylla builds no PDLP wrapper at effort 0", "[effort-zero]") {
    CHECK(log_contains(trace_solve("flugpl.mps", [](Highs& h) { set_suite(h, "all"); }),
                       "[ScyllaOverlap]"));
    CHECK(!log_contains(zeroed("flugpl.mps", "scylla"), "[ScyllaOverlap]"));
    CHECK(!log_contains(omitted("flugpl.mps", "fj,fpr,local_mip"), "[ScyllaOverlap]"));
}

// The half the assertions above cannot reach: that the guard removes the
// *setup*, not merely the search.
//
// Charged effort and offer counts are both blind to it — `precompute_var_orders`
// and `ContestedPdlp` construction are uncharged, so deleting
// `budget.disabled()` from all four entry points leaves every other
// assertion in this file passing (measured: 3 of 4 cases green, only the
// `[ScyllaOverlap]` one below failing).  The only observable that moves is
// the wall-clock window the ledger already reports.
//
// This is therefore the one test here that reads a time, and it is written
// to be a *structural* comparison rather than a performance one.  With the
// guard, a disabled heuristic's `[Heur]` window is two `timer_.read()`
// calls around a function that returns immediately: `%.1f` prints `0.0`,
// i.e. under 0.05 ms, on every bundled instance, 3 runs each.  Without it,
// on `p0548` — the bundled instance with the most columns, so the one whose
// setup costs most — FPR reports 0.7 ms and Scylla 0.5 ms, reproducibly to
// the digit.  The threshold below sits an order of magnitude above the
// guarded value and well under half the unguarded one.
//
// The minimum over repeats, not one sample: a scheduler hiccup between two
// clock reads can inflate any single window on a loaded machine, while the
// cost the counterfactual pays is real work that every repeat pays again.
// So the min keeps the false-failure rate near zero without weakening what
// the test detects.
TEST_CASE("effort-zero: a zeroed heuristic runs no setup either", "[effort-zero]") {
    constexpr int kRepeats = 3;
    constexpr double kSetupFreeMs = 0.3;

    std::map<std::string, double> fastest;
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        // `p0548`, and every heuristic zeroed at once: the point is that no
        // entry point pays for setup, and one solve measures all four.
        const auto lines = trace_solve("p0548.mps", [](Highs& h) {
            set_suite(h, "all");
            for (const Case& c : kCases) {
                require_option(h, std::string("mip_heuristic_") + c.name + "_effort", 0.0);
            }
        });
        for (const HeurLine& entry : presolve_lines(lines)) {
            const auto it = fastest.find(entry.name);
            if (it == fastest.end() || entry.wall_ms < it->second) {
                fastest[entry.name] = entry.wall_ms;
            }
        }
    }

    REQUIRE(fastest.size() == kCases.size());
    for (const auto& [name, wall_ms] : fastest) {
        INFO("heuristic " << name << " best-of-" << kRepeats << " wall_ms " << wall_ms);
        CHECK(wall_ms < kSetupFreeMs);
    }
}

// The equivalence #107's parameter encoding rests on: zeroing a
// heuristic's effort leaves the rest of the presolve chain doing exactly
// what omitting it from the suite would.
//
// Compared as traces, not just objectives: two configurations can agree on
// the final objective while spending completely different budgets, and it
// is the budgets a calibration search reads.  The zeroed heuristic's own
// entry is excluded from the comparison because it is the one documented
// difference — `run_sequential` still books it, so `[Heur] name=<n>
// effort=0 found=0` is emitted where omission emits nothing at all.  That
// is a log difference and not a behavioural one; asserting on solver state
// rather than log identity is what keeps it out of the way.
TEST_CASE("effort-zero: equivalent to omitting the heuristic from the suite", "[effort-zero]") {
    for (const Case& c : kCases) {
        INFO("heuristic " << c.name);
        const auto zero_lines = zeroed("flugpl.mps", c.name);
        const auto omit_lines = omitted("flugpl.mps", c.complement);

        CHECK(without(presolve_lines(zero_lines), c.name) == presolve_lines(omit_lines));
        CHECK(source_codes(zero_lines) == source_codes(omit_lines));

        // The zeroed heuristic is the only extra presolve entry.
        CHECK(presolve_lines(zero_lines).size() == presolve_lines(omit_lines).size() + 1);
    }
}

// The objective itself, on an instance whose optimum every configuration
// reaches: disabling a heuristic by either spelling must not change what
// the solver returns.  `mip_rel_gap = 0` on both sides so the comparison is
// between two proven optima rather than between two incumbents each
// allowed to stop 1e-4 short.
TEST_CASE("effort-zero: the reported objective is unchanged", "[effort-zero]") {
    const auto objective = [](auto&& configure) {
        Highs h;
        h.setOptionValue("output_flag", false);
        require_option(h, "mip_rel_gap", 0.0);
        configure(h);
        REQUIRE(h.readModel(kInstancesDir + "/flugpl.mps") == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        double obj = 0.0;
        h.getInfoValue("objective_function_value", obj);
        return obj;
    };

    for (const Case& c : kCases) {
        INFO("heuristic " << c.name);
        const std::string option = std::string("mip_heuristic_") + c.name + "_effort";
        const double zero_obj = objective([&](Highs& h) {
            set_suite(h, "all");
            require_option(h, option, 0.0);
        });
        const double omit_obj = objective([&](Highs& h) { set_suite(h, c.complement); });
        CHECK(zero_obj == omit_obj);
    }
}
