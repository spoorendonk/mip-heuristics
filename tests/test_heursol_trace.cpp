#include "Highs.h"
#include "test_common.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

// ===================================================================
// The `[HeurSol]` per-offered-solution trace (#106)
//
//   [HeurSol] name=<n> dispatch=<i> worker=<w> effort_at=<E> wall_ms=<X>
//             obj=<O> accepted=<0|1>
//
// `[Heur]` is one line per dispatch carrying aggregate effort, which
// cannot show what happens *inside* one — and the stall thresholds #107
// calibrates are exactly an intra-dispatch quantity ("how much
// improvement-free effort is enough before this is going nowhere?").  This
// line is emitted from `IncumbentSink::offer`, the project's single
// definition of production, once per offered solution, accepted or not.
//
// Three properties are load-bearing and are asserted here rather than left
// to the consumers in `bench/`:
//
//   * `(name, dispatch)` identifies one dispatch.  The id is drawn from a
//     process-global counter, so it is neither zero-based nor dense within
//     a solve; what must hold is that one id is never shared by two names,
//     and that a name's ids partition its dispatches.
//   * `effort_at` is monotone non-decreasing within `(name, dispatch,
//     worker)`.  FJ, LocalMIP, Scylla and fpr_lp all rebuild a retired
//     worker in place with a fresh, zeroed `WorkerBudgetState`;
//     `WorkerTrace::effort_base` carries the outgoing worker's charge into
//     the replacement so the emitted value keeps rising.  Without it every
//     rebuild would silently swallow one occupant's worth of effort from
//     the inter-acceptance gap distribution, biasing its quantiles
//     downward — towards a tighter gate, the direction that costs
//     solutions.
//   * `name` agrees with the `[Heur] name=` of the same solve.  The sink
//     is handed a `kSolutionSource*` tag, not a name, so it maps the tag
//     back to `kChain`'s spelling in a `switch` of its own; this is the
//     check that keeps that second spelling from drifting.
// ===================================================================

namespace {

struct Offer {
    std::string name;
    unsigned long long dispatch = 0;
    long long worker = 0;
    unsigned long long effort_at = 0;
    double wall_ms = 0.0;
    double objective = 0.0;
    bool accepted = false;
};

// The `key=value` tokens of one line, in emission order.  Parsed as a
// dictionary rather than positionally, mirroring `bench/parse_highs_log.py`
// — the line has already gained a field once and must tolerate gaining
// another.
std::map<std::string, std::string> fields_of(const std::string& line) {
    std::map<std::string, std::string> fields;
    size_t pos = line.find("[HeurSol]");
    if (pos == std::string::npos) {
        return fields;
    }
    pos += std::string("[HeurSol]").size();
    while (pos < line.size()) {
        const auto start = line.find_first_not_of(" \t\r\n", pos);
        if (start == std::string::npos) {
            break;
        }
        auto end = line.find_first_of(" \t\r\n", start);
        if (end == std::string::npos) {
            end = line.size();
        }
        const auto token = line.substr(start, end - start);
        const auto eq = token.find('=');
        if (eq != std::string::npos) {
            fields[token.substr(0, eq)] = token.substr(eq + 1);
        }
        pos = end;
    }
    return fields;
}

// Every well-formed `[HeurSol]` line of `lines`.  A line that carries the
// tag but not all seven keys is a failure, not something to skip, so the
// caller checks the counts against `count_tagged` below.
std::vector<Offer> offers(const std::vector<std::string>& lines) {
    std::vector<Offer> out;
    for (const auto& line : lines) {
        const auto fields = fields_of(line);
        if (fields.empty()) {
            continue;
        }
        const auto get = [&](const char* key) -> const std::string* {
            const auto it = fields.find(key);
            return it == fields.end() ? nullptr : &it->second;
        };
        const std::string* name = get("name");
        const std::string* dispatch = get("dispatch");
        const std::string* worker = get("worker");
        const std::string* effort = get("effort_at");
        const std::string* wall = get("wall_ms");
        const std::string* obj = get("obj");
        const std::string* accepted = get("accepted");
        if (name == nullptr || dispatch == nullptr || worker == nullptr || effort == nullptr ||
            wall == nullptr || obj == nullptr || accepted == nullptr) {
            continue;
        }
        out.push_back(Offer{*name, std::strtoull(dispatch->c_str(), nullptr, 10),
                            std::strtoll(worker->c_str(), nullptr, 10),
                            std::strtoull(effort->c_str(), nullptr, 10),
                            std::strtod(wall->c_str(), nullptr), std::strtod(obj->c_str(), nullptr),
                            *accepted == "1"});
    }
    return out;
}

size_t count_tagged(const std::vector<std::string>& lines) {
    return static_cast<size_t>(
        std::ranges::count_if(lines, [](const std::string& l) { return l.contains("[HeurSol]"); }));
}

// The `name=` values of the solve's `[Heur]` lines.
std::set<std::string> heur_names(const std::vector<std::string>& lines) {
    std::set<std::string> names;
    const std::string tag = "[Heur] name=";
    for (const auto& line : lines) {
        const auto pos = line.find(tag);
        if (pos == std::string::npos) {
            continue;
        }
        const auto start = pos + tag.size();
        const auto end = line.find_first_of(" \n", start);
        names.insert(line.substr(start, end == std::string::npos ? end : end - start));
    }
    return names;
}

// A solve that produces traces: `log_dev_level=3` is what makes them
// exist, and `egout` is small enough to finish fast while its FPR earns
// dozens of pool acceptances, so the accepted-offer assertions have
// something to bite on.
std::vector<std::string> traced_solve(int dev_level) {
    return solve_capturing_log("egout.mps", [&](Highs& h) {
        require_option(h, "log_dev_level", dev_level);
        set_suite(h, "all");
    });
}

// One fixture per rebuild path, because `egout` at default options covers
// only two of them.
//
// The carry that keeps `effort_at` monotone is written out separately at
// every site that replaces a retired worker, so a test that never sees a
// site cannot fail when its carry is deleted — measured: removing FJ's
// harvest alone left the monotonicity case green, because every one of its
// assertions came from `local_mip` slots.  `egout` emits no Scylla or
// `fpr_lp` offers at all at the shipped defaults.
//
// So: `egout` at defaults for FJ / FPR / LocalMIP, `gt2` with Scylla's
// effort raised (25 offers against 1 at the default), and `bell5` at
// `suite=fpr` for the dive-time `fpr_lp` — the recipe `test_fpr_lp.cpp`
// already uses, and the reason it is not `bell5` at defaults: with the
// whole chain enabled the presolve heuristics usually solve bell5 before
// the dive needs `fpr_lp`, and the offer count came out 0, 0, 520, 0 over
// four runs.  At `suite=fpr` it was 360-457 over six.
//
// `kTracedNames` below asserts the union actually covers all five, so the
// coverage cannot silently lapse again if a default moves.
struct Fixture {
    const char* instance;
    const char* suite;
    double scylla_effort;  // negative to leave the option alone
};

constexpr std::array<Fixture, 3> kFixtures = {{
    {"egout.mps", "all", -1.0},
    {"gt2.mps", "all", 1.0},
    {"bell5.mps", "fpr", -1.0},
}};

std::vector<std::string> traced_fixture(const Fixture& fixture) {
    return solve_capturing_log(fixture.instance, [&](Highs& h) {
        require_option(h, "log_dev_level", 3);
        set_suite(h, fixture.suite);
        if (fixture.scylla_effort >= 0.0) {
            require_option(h, "mip_heuristic_scylla_effort", fixture.scylla_effort);
        }
    });
}

// Every heuristic that can offer a solution, so the fixture set can be
// checked for coverage rather than assumed to have it.
const std::set<std::string> kTracedNames = {"fj", "fpr", "local_mip", "scylla", "fpr_lp"};

}  // namespace

TEST_CASE("heursol: a dev-level-3 solve emits well-formed lines", "[heursol]") {
    const auto lines = traced_solve(3);
    const auto parsed = offers(lines);

    REQUIRE(!parsed.empty());
    // Every tagged line parsed: a line carrying the tag but missing a key
    // would be dropped by `offers` and would show up here as a mismatch.
    CHECK(parsed.size() == count_tagged(lines));

    const std::set<std::string> legal = {"fj", "fpr", "local_mip", "scylla", "fpr_lp"};
    const auto announced = heur_names(lines);
    for (const auto& offer : parsed) {
        INFO("offer from " << offer.name);
        CHECK(legal.contains(offer.name));
        // The tag-to-name map in `incumbent_sink.cpp` agrees with the
        // ledger's `kChain` spelling — the drift guard described above.
        CHECK(announced.contains(offer.name));
        // `-1` is the documented sentinel for an offer made off any worker
        // slot (LocalMIP's cold-start publish on the dispatching thread).
        CHECK(offer.worker >= -1);
        CHECK(std::isfinite(offer.objective));
    }
}

TEST_CASE("heursol: (name, dispatch) identifies one dispatch", "[heursol]") {
    const auto parsed = offers(traced_solve(3));
    REQUIRE(!parsed.empty());

    // One id never belongs to two heuristics.  The counter is
    // process-global, so this holds by construction — the check is that
    // nothing re-uses or re-derives an id behind its back.
    std::map<unsigned long long, std::string> owner;
    for (const auto& offer : parsed) {
        const auto [it, inserted] = owner.try_emplace(offer.dispatch, offer.name);
        INFO("dispatch " << offer.dispatch);
        CHECK(it->second == offer.name);
    }

    // The presolve chain runs each heuristic once per solve, so each name
    // that offered anything did so under exactly one id.
    std::map<std::string, std::set<unsigned long long>> per_name;
    for (const auto& offer : parsed) {
        per_name[offer.name].insert(offer.dispatch);
    }
    for (const auto& [name, ids] : per_name) {
        if (name == "fpr_lp") {
            // The dive heuristic is dispatched once per dive, so it may
            // legitimately carry several ids.
            continue;
        }
        INFO("heuristic " << name);
        CHECK(ids.size() == 1);
    }
}

TEST_CASE("heursol: effort_at is monotone within a worker slot", "[heursol]") {
    std::set<std::string> covered;
    size_t checked = 0;

    for (const Fixture& fixture : kFixtures) {
        INFO("fixture " << fixture.instance);
        const auto parsed = offers(traced_fixture(fixture));
        REQUIRE(!parsed.empty());

        std::map<std::tuple<std::string, unsigned long long, long long>, unsigned long long> last;
        for (const auto& offer : parsed) {
            covered.insert(offer.name);
            const auto key = std::tuple{offer.name, offer.dispatch, offer.worker};
            const auto it = last.find(key);
            if (it != last.end()) {
                INFO("slot " << offer.name << "/" << offer.dispatch << "/" << offer.worker);
                CHECK(offer.effort_at >= it->second);
                ++checked;
            }
            last[key] = offer.effort_at;
        }
    }

    // The assertions above are vacuous for a slot that offered once, and
    // the whole case is vacuous for a heuristic that never appeared.  Both
    // are checked, so a shifted default cannot quietly empty this test.
    CHECK(checked > 0);
    CHECK(covered == kTracedNames);
}

TEST_CASE("heursol: an accepted offer is what [Heur] reports as found", "[heursol]") {
    const auto lines = traced_solve(3);
    const auto parsed = offers(lines);
    REQUIRE(!parsed.empty());

    // At least one offer was taken — otherwise `accepted` would be
    // trivially satisfiable and the field untested.
    CHECK(std::ranges::any_of(parsed, [](const Offer& o) { return o.accepted; }));

    // Every heuristic that had an offer accepted reported `found=1`.
    std::set<std::string> producers;
    for (const auto& offer : parsed) {
        if (offer.accepted) {
            producers.insert(offer.name);
        }
    }
    for (const auto& name : producers) {
        INFO("heuristic " << name);
        CHECK(log_contains(lines, "[Heur] name=" + name + " "));
        CHECK(std::ranges::any_of(lines, [&](const std::string& l) {
            return l.contains("[Heur] name=" + name + " ") && l.contains("found=1");
        }));
    }
}

// The line is `kVerbose`, like `[Heur]` and `[Sequential]`, so it must
// cost a run nothing below `log_dev_level=3`.  Level 2 as well as the
// default, because `kDetailed` is the level immediately below and is the
// one a "turn on some tracing" run would reach for.
TEST_CASE("heursol: absent below log_dev_level 3", "[heursol]") {
    CHECK(count_tagged(traced_solve(0)) == 0);
    CHECK(count_tagged(traced_solve(1)) == 0);
    CHECK(count_tagged(traced_solve(2)) == 0);
    CHECK(count_tagged(traced_solve(3)) > 0);
}
