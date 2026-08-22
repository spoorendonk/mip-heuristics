#include "Highs.h"
#include "mode_dispatch.h"
#include "test_common.h"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

// ===================================================================
// mip_heuristic_suite as a set (#112)
//
// The option takes a comma-separated list of heuristic names, so all
// fifteen non-empty subsets of the chain are expressible instead of the
// five the single-valued surface could name.  `off` and `all` stay the
// two whole-value aliases they always were.
//
// Two things are tested separately here.  `effective_flags` is the single
// place the string becomes four booleans, so the parsing cases (ordering,
// whitespace, duplicates, rejection) belong on it directly — they are
// cheap and can be exhaustive.  But the flags struct agreeing with the
// string proves nothing about the *chain* honouring it, so the subset
// cases are also asserted end to end, against the `[Heur] name=<n>`
// traces of a real solve.
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

// Every heuristic, in chain order — what a fail-open selects.
const std::string kAll = "fj+fpr+local_mip+scylla";

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
    REQUIRE(enabled_for("all") == kAll);
    for (const char* value : {"off", "fj", "fpr", "local_mip", "scylla", "all"}) {
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
    REQUIRE(enabled_for("fj,fpr,local_mip,scylla") == kAll);
}

TEST_CASE("suite: ordering, whitespace and repetition do not change a list", "[options][suite]") {
    REQUIRE(enabled_for("fpr,fj") == "fj+fpr");
    REQUIRE(enabled_for("fj, fpr") == "fj+fpr");
    REQUIRE(enabled_for(" fj ,\tfpr ") == "fj+fpr");
    REQUIRE(enabled_for("fj,fpr,fj") == "fj+fpr");
}

TEST_CASE("suite: fpr_lp follows the fpr bit out of a list", "[options][suite]") {
    // The dive-time variant is gated on the same flag as presolve FPR, so a
    // subset that omits `fpr` disables it too — otherwise a mix-selection row
    // measures a second FPR variant it did not ask for.
    REQUIRE(flags_for("fj,fpr", nullptr).fpr);
    REQUIRE_FALSE(flags_for("fj,local_mip,scylla", nullptr).fpr);
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
    // being that string — a heuristic-free run that is not the
    // vanilla-equivalent one.  Loud instead.
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
