#include "Highs.h"
#include "mode_dispatch.h"
#include "test_common.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

// ===================================================================
// mip_heuristic_suite as a set (#112, #164)
//
// The option takes a comma-separated list of heuristic names, so all
// thirty-one non-empty subsets of the five heuristics are expressible
// instead of the six the single-valued surface could name.  `off` and
// `all` stay the two whole-value aliases they always were.
//
// Two things are tested separately here.  `effective_flags` is the single
// place the string becomes five booleans, so the parsing cases (ordering,
// whitespace, duplicates, rejection) belong on it directly — they are
// cheap and can be exhaustive.  But the flags struct agreeing with the
// string proves nothing about the *chain* honouring it, so the subset
// cases are also asserted end to end, against the `[Heur] name=<n>`
// traces of a real solve.
//
// The fifth token, `fpr_lp`, is the dive-time LP-based FPR (#164).  It used
// to follow `fpr`'s bit, so "presolve FPR without fpr_lp" had no spelling.
// Adding it must not have moved what any existing value does to the
// *presolve chain*, which the four-heuristic recalibration of #113 rests
// on, so that is pinned exhaustively below rather than left to the cases
// that happen to mention a subset.
// ===================================================================

namespace {

// The flags `suite` selects, read through the same function the dispatcher
// and fpr_lp both call.
//
// `setOptionValue` rather than writing the member directly: that is the path
// a user takes, and it applies HiGHS's own trimming and lower-casing of a
// string option value, which the `off` alias comparison depends on.
heuristics::HeuristicFlags flags_for(const char* suite, heuristics::SuiteDiagnosis* diagnosis) {
    Highs h;
    h.setOptionValue("output_flag", false);
    set_suite(h, suite);
    return heuristics::effective_flags(h.getOptions(), diagnosis);
}

// The enabled heuristics as a `+`-joined name in chain order, so a failing
// assertion prints the set that was selected rather than four opaque bools.
std::string enabled_names(const heuristics::HeuristicFlags& flags) {
    std::string names;
    const auto add = [&names](bool enabled, const char* name) {
        if (enabled) {
            if (!names.empty()) {
                names += '+';
            }
            names += name;
        }
    };
    add(flags.fj, "fj");
    add(flags.fpr, "fpr");
    add(flags.local_mip, "local_mip");
    add(flags.scylla, "scylla");
    // Last, as in a config name: it is the one entry that does not run in
    // presolve.
    add(flags.fpr_lp, "fpr_lp");
    return names.empty() ? "none" : names;
}

std::string enabled_for(const char* suite) {
    return enabled_names(flags_for(suite, nullptr));
}

// Whether `suite` was understood in full, i.e. carried no unknown token.
bool recognized(const char* suite) {
    heuristics::SuiteDiagnosis diagnosis;
    flags_for(suite, &diagnosis);
    return diagnosis.unknown_count == 0;
}

// Every heuristic, in chain order with the dive-time entry last — what a
// fail-open selects.
const std::string kAll = "fj+fpr+local_mip+scylla+fpr_lp";

// The four presolve bits alone, as a `+`-joined name.  The regression case
// below compares *only* these across every suite value, because the four
// presolve heuristics' gating is the thing #164 was required not to move.
std::string presolve_names(const heuristics::HeuristicFlags& flags) {
    heuristics::HeuristicFlags chain_only = flags;
    chain_only.fpr_lp = false;
    return enabled_names(chain_only);
}

// Solve flugpl with `suite` selected and return the captured log.  The
// dev-level pin is what makes the `[Heur]` traces exist at all.
std::vector<std::string> log_for(const char* suite) {
    return solve_capturing_log("flugpl.mps", [&](Highs& h) {
        require_option(h, "log_dev_level", 3);
        set_suite(h, suite);
    });
}

// Whether the presolve chain dispatched `heur` — its `[Heur]` trace exists.
// The trailing space keeps `name=fpr` from matching fpr_lp's `name=fpr_lp`.
bool dispatched(const std::vector<std::string>& lines, const char* heur) {
    return log_contains(lines, std::string("[Heur] name=") + heur + " ");
}

}  // namespace

// --- the six values that predate lists ------------------------------------

TEST_CASE("suite: the singletons and aliases mean what they always did", "[options][suite]") {
    REQUIRE(enabled_for("off") == "none");
    REQUIRE(enabled_for("fj") == "fj");
    REQUIRE(enabled_for("fpr") == "fpr");
    REQUIRE(enabled_for("local_mip") == "local_mip");
    REQUIRE(enabled_for("scylla") == "scylla");
    REQUIRE(enabled_for("fpr_lp") == "fpr_lp");
    REQUIRE(enabled_for("all") == kAll);
    for (const char* value : {"off", "fj", "fpr", "local_mip", "scylla", "fpr_lp", "all"}) {
        INFO("value " << value);
        REQUIRE(recognized(value));
    }
}

// --- lists ----------------------------------------------------------------

TEST_CASE("suite: a list enables exactly the heuristics it names", "[options][suite]") {
    REQUIRE(enabled_for("fj,fpr") == "fj+fpr");
    REQUIRE(enabled_for("fpr,scylla") == "fpr+scylla");
    // The composition the recorded PLATO table was measured at, which the
    // single-valued option could not express.
    REQUIRE(enabled_for("fj,fpr,local_mip") == "fj+fpr+local_mip");
    REQUIRE(enabled_for("fj,fpr,local_mip,scylla,fpr_lp") == kAll);
}

TEST_CASE("suite: ordering, whitespace and repetition do not change a list", "[options][suite]") {
    REQUIRE(enabled_for("fpr,fj") == "fj+fpr");
    REQUIRE(enabled_for("fj, fpr") == "fj+fpr");
    REQUIRE(enabled_for(" fj ,\tfpr ") == "fj+fpr");
    REQUIRE(enabled_for("fj,fpr,fj") == "fj+fpr");
}

TEST_CASE("suite: fpr_lp is selected by its own token, not by fpr's", "[options][suite]") {
    // The one intended behaviour change of #164, and the whole point of it:
    // the dive-time variant no longer follows presolve FPR's bit, so all
    // four cells of the issue's matrix are reachable.  `suite=fpr` used to
    // imply `fpr_lp`; a configuration spelled that way now means presolve
    // FPR alone.
    REQUIRE(enabled_for("fpr") == "fpr");
    REQUIRE(enabled_for("fpr_lp") == "fpr_lp");
    REQUIRE(enabled_for("fpr,fpr_lp") == "fpr+fpr_lp");
    REQUIRE(enabled_for("fj,local_mip,scylla") == "fj+local_mip+scylla");
    // Order is irrelevant here too — `fpr_lp` is last in the canonical
    // spelling, not in the parse.
    REQUIRE(enabled_for("fpr_lp,fj") == "fj+fpr_lp");
    // `off` still dominates everything, the dive-time entry included.
    REQUIRE(enabled_for("off") == "none");
}

// The regression the issue's comment asks for: adding a fifth token moved
// nothing about which of the four presolve heuristics run, at any value.
//
// Exhaustive over the sixteen values that existed before #164 — `off`,
// `all`, and the fifteen non-empty subsets of the chain — plus each of them
// with `fpr_lp` appended, since a value naming the new token must still
// leave the chain alone.  The expectation is written out rather than
// derived from the parse, so a change in `parse_suite_list` cannot make
// this test agree with itself.
TEST_CASE("suite: the presolve chain's gating is unchanged at every value",
          "[options][suite][regression]") {
    struct Expectation {
        const char* suite;
        const char* presolve;  // the four presolve bits, `+`-joined
    };
    // `all` and `off` are the two aliases; the rest are the chain's
    // subsets, in the canonical chain-order spelling.
    constexpr std::array<Expectation, 17> kChainValues = {{
        {"off", "none"},
        {"all", "fj+fpr+local_mip+scylla"},
        {"fj", "fj"},
        {"fpr", "fpr"},
        {"local_mip", "local_mip"},
        {"scylla", "scylla"},
        {"fj,fpr", "fj+fpr"},
        {"fj,local_mip", "fj+local_mip"},
        {"fj,scylla", "fj+scylla"},
        {"fpr,local_mip", "fpr+local_mip"},
        {"fpr,scylla", "fpr+scylla"},
        {"local_mip,scylla", "local_mip+scylla"},
        {"fj,fpr,local_mip", "fj+fpr+local_mip"},
        {"fj,fpr,scylla", "fj+fpr+scylla"},
        {"fj,local_mip,scylla", "fj+local_mip+scylla"},
        {"fpr,local_mip,scylla", "fpr+local_mip+scylla"},
        {"fj,fpr,local_mip,scylla", "fj+fpr+local_mip+scylla"},
    }};

    for (const Expectation& e : kChainValues) {
        INFO("suite " << e.suite);
        CHECK(presolve_names(flags_for(e.suite, nullptr)) == e.presolve);
        CHECK(recognized(e.suite));

        // The same value with the new token added.  `off` and `all` are
        // whole-value aliases, so appending to them is not a legal value
        // and is skipped rather than asserted on.
        if (std::string(e.suite) == "off" || std::string(e.suite) == "all") {
            continue;
        }
        const std::string with_fpr_lp = std::string(e.suite) + ",fpr_lp";
        INFO("suite " << with_fpr_lp);
        CHECK(presolve_names(flags_for(with_fpr_lp.c_str(), nullptr)) == e.presolve);
        CHECK(flags_for(with_fpr_lp.c_str(), nullptr).fpr_lp);
    }
}

// --- rejection ------------------------------------------------------------

TEST_CASE("suite: an unknown token fails open and is named", "[options][suite]") {
    heuristics::SuiteDiagnosis diagnosis;
    // A typo in one token of an otherwise valid list.  Failing open promotes
    // a two-heuristic run to a four-heuristic one, so the warning naming the
    // token is the only thing separating this from a silently mislabelled
    // benchmark row.
    REQUIRE(enabled_names(flags_for("fj,fpr2", &diagnosis)) == kAll);
    REQUIRE(diagnosis.unknown_count == 1);
    REQUIRE(diagnosis.unknown_tokens == "\"fpr2\"");

    REQUIRE(enabled_names(flags_for("bogus,fj,walksat", &diagnosis)) == kAll);
    REQUIRE(diagnosis.unknown_count == 2);
    REQUIRE(diagnosis.unknown_tokens == "\"bogus\", \"walksat\"");
}

TEST_CASE("suite: off is an alias for the whole value, never a token", "[options][suite]") {
    // `off` is not merely the empty set: the patched HiGHS tree compares this
    // option to "off" verbatim to hand back the native FeasibilityJump call
    // site.  A list containing it would select nothing on our side while not
    // being that string — a run with no heuristic at all, HiGHS's own FJ
    // included.  Loud instead.
    REQUIRE_FALSE(recognized("fj,off"));
    REQUIRE(enabled_for("fj,off") == kAll);
}

TEST_CASE("suite: an empty value or a stray comma is not a silent off", "[options][suite]") {
    REQUIRE_FALSE(recognized(""));
    REQUIRE(enabled_for("") == kAll);
    REQUIRE_FALSE(recognized("fj,"));
    REQUIRE(enabled_for("fj,") == kAll);
}

// --- end to end: what the chain actually ran ------------------------------

// The default configuration, pinned as a whole (#164).
//
// Adding a fifth token and a fifth effort option touches the code path that
// decides what the presolve chain does, and that chain's budgets were
// measured in #113 against the binary as it shipped.  So this asserts what
// a user gets with nothing configured: all five heuristics enabled, and the
// four presolve ones dispatched.
//
// The fifth is pinned on its **gating** rather than on a dispatch, and that
// is a deliberate limit rather than a weaker version of the same check.
// Both of `fpr_lp`'s gates are asserted here — the suite bit and
// `mip_heuristic_fpr_lp_effort > 0`, which are the two returns at the top of
// `fpr_lp::run` — so nothing about #164 can disable it at defaults without
// failing this.  Whether the B&B dive then *reaches* it is a property of the
// instance and of what the presolve chain already found: on `bell5` at
// defaults the chain usually solves the model before the dive needs it (see
// the fixture note in `tests/test_heursol_trace.cpp`, measuring 0, 0, 520, 0
// offers over four runs), so a dispatch assertion here would be a flake.
// `tests/test_fpr_lp.cpp` pins the dispatch at the values that make it
// deterministic.
TEST_CASE("suite: the default configuration enables all five heuristics",
          "[options][suite][regression]") {
    Highs h;
    h.setOptionValue("output_flag", false);
    // Nothing configured: no `set_suite`, no effort option.  That is the
    // point — this is the shipped default.
    REQUIRE(enabled_names(heuristics::effective_flags(h.getOptions())) == kAll);
    double fpr_lp_effort = -1.0;
    REQUIRE(h.getOptionValue("mip_heuristic_fpr_lp_effort", fpr_lp_effort) == HighsStatus::kOk);
    // The second of `fpr_lp`'s two gates: a zero here would disable it just
    // as surely as an absent token.  The value itself is pinned in
    // `tests/test_smoke.cpp`; what matters here is only that it does not
    // disable.
    REQUIRE(fpr_lp_effort > 0.0);

    // And the presolve chain really dispatches all four, unconfigured.
    const std::vector<std::string> lines = solve_capturing_log(
        "flugpl.mps", [](Highs& solver) { require_option(solver, "log_dev_level", 3); });
    for (const char* heur : {"fj", "fpr", "local_mip", "scylla"}) {
        INFO("heuristic " << heur);
        REQUIRE(dispatched(lines, heur));
    }
}

TEST_CASE("suite: a two-element list dispatches exactly those two", "[options][suite]") {
    const std::vector<std::string> lines = log_for("fj,fpr");
    REQUIRE(dispatched(lines, "fj"));
    REQUIRE(dispatched(lines, "fpr"));
    REQUIRE_FALSE(dispatched(lines, "local_mip"));
    REQUIRE_FALSE(dispatched(lines, "scylla"));
}

TEST_CASE("suite: a three-element list dispatches exactly those three", "[options][suite]") {
    const std::vector<std::string> lines = log_for("fj,fpr,local_mip");
    REQUIRE(dispatched(lines, "fj"));
    REQUIRE(dispatched(lines, "fpr"));
    REQUIRE(dispatched(lines, "local_mip"));
    REQUIRE_FALSE(dispatched(lines, "scylla"));
}

TEST_CASE("suite: a mistyped token in a list warns, names it, and runs everything",
          "[options][suite][bench-contract]") {
    const std::vector<std::string> lines = log_for("fj,locl_mip");
    // The harness greps `Unknown mip_heuristic_suite value` and discards the
    // run; the token is what tells a human which name to fix.
    REQUIRE(log_contains(lines, "Unknown mip_heuristic_suite value \"fj,locl_mip\""));
    REQUIRE(log_contains(lines, "unrecognised token \"locl_mip\""));
    for (const char* heur : {"fj", "fpr", "local_mip", "scylla"}) {
        INFO("heuristic " << heur);
        REQUIRE(dispatched(lines, heur));
    }
}
