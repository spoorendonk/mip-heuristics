// One FJ model reused across searches (#316): a reset `FjWorker` searches
// exactly as a fresh one with the same arguments, and the row-wise matrix
// it is built from is the one FJ used to transpose for itself.

#include "core_test_support.h"
#include "fj_worker.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "test_paths.h"
#include "util/HighsSparseMatrix.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

using namespace core_test;

namespace {

// Every offer the sink accepted: objective, point and reported effort.
class OfferLog : public SolutionSink {
public:
    using SolutionSink::SolutionSink;
    struct Offer {
        double objective;
        std::vector<double> x;
        size_t effort_at;
        bool operator==(const Offer&) const = default;
    };
    std::mutex mtx;
    std::vector<Offer> offers;

private:
    void on_accept(double obj, const std::vector<double>& x, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t effort_at) override {
        std::scoped_lock guard(mtx);
        offers.push_back({obj, x, effort_at});
    }
};

// What one search did: each attempt's result, then what it found.
struct Search {
    std::vector<size_t> effort;
    std::vector<bool> improved;
    std::vector<double> best_found;
    double best_found_objective = 0.0;
    size_t traced_effort = 0;
    bool finished = false;
    bool operator==(const Search&) const = default;
};

constexpr int kSearches = 6;
constexpr int kAttemptsPerSearch = 3;
constexpr size_t kTotal = size_t{1} << 20;
constexpr size_t kPerAttempt = size_t{1} << 17;

Search run_search(FjWorker& fj) {
    Search out;
    for (int a = 0; a < kAttemptsPerSearch && !fj.finished(); ++a) {
        const AttemptResult r = fj.run_attempt(kPerAttempt);
        out.effort.push_back(r.effort);
        out.improved.push_back(r.found_improvement);
    }
    out.best_found = fj.best_found();
    out.best_found_objective = fj.best_found_objective();
    out.traced_effort = fj.traced_effort();
    out.finished = fj.finished();
    return out;
}

// The arguments of search `k`: a start (none for the first, then the
// previous search's best or the one before's), a seed, budgets, and a
// cutoff that moves when `cut` is set.
struct Args {
    std::vector<double> start;
    uint32_t seed;
    size_t total;
    size_t stale;
    WorkerTrace trace;
    double cutoff;
};

// Runs `kSearches` searches on `s` twice, once through one worker reset
// between searches and once through a fresh worker per search, and checks
// they match search for search, offer for offer.
void check_reuse_matches_fresh(const Setup& s, bool cut, bool hold) {
    const double inf = std::numeric_limits<double>::infinity();
    OfferLog reused_sink(*s.problem.integrality, 0);
    OfferLog fresh_sink(*s.problem.integrality, 0);
    std::unique_ptr<FjWorker> reused;
    std::vector<double> prev;
    double prev_objective = inf;
    int searches_that_found = 0;
    for (int k = 0; k < kSearches; ++k) {
        Args args{.start = prev,
                  .seed = static_cast<uint32_t>(17 + (31 * k)),
                  .total = kTotal + (static_cast<size_t>(k) * 4096),
                  // Some searches end on FJ's own stall, some on the total.
                  .stale = k % 2 == 0 ? std::numeric_limits<size_t>::max() : 600000,
                  .trace = WorkerTrace{0, static_cast<size_t>(k) * 1000},
                  .cutoff = inf};
        if (cut) {
            // Finite from the first search on; looser than the best so far
            // until there is one, so every search has room.
            args.cutoff =
                std::isfinite(prev_objective) ? prev_objective - (1e-6 * (k + 1)) : 1e12 - k;
        }

        if (!reused) {
            reused =
                std::make_unique<FjWorker>(s.problem, s.exec, reused_sink, args.total, args.stale,
                                           args.seed, args.start, args.trace, args.cutoff);
            if (hold) {
                reused->hold_to_budget();
            }
        } else {
            reused->reset(args.start, args.seed, args.total, args.stale, args.trace, args.cutoff);
        }
        FjWorker fresh(s.problem, s.exec, fresh_sink, args.total, args.stale, args.seed, args.start,
                       args.trace, args.cutoff);
        if (hold) {
            fresh.hold_to_budget();
        }

        const Search a = run_search(*reused);
        const Search b = run_search(fresh);
        INFO("search " << k);
        REQUIRE_FALSE(a.effort.empty());
        CHECK(a.effort == b.effort);
        CHECK(a.improved == b.improved);
        CHECK(a.best_found == b.best_found);
        CHECK(a.best_found_objective == b.best_found_objective);
        CHECK(a.traced_effort == b.traced_effort);
        CHECK(a.finished == b.finished);
        REQUIRE(reused_sink.offers == fresh_sink.offers);
        // The next start: this search's best, else the one before's.
        if (!a.best_found.empty()) {
            ++searches_that_found;
            prev = a.best_found;
            prev_objective = a.best_found_objective;
        }
    }
    // Not vacuous: a reset worker found something after a search that had
    // found something too, so its starts and cutoffs moved.
    CHECK(searches_that_found >= 2);
    CHECK_FALSE(reused_sink.offers.empty());
}

}  // namespace

TEST_CASE("fj_reuse: a reset FjWorker searches as a fresh one", "[fj][fj_reuse]") {
    // Instances FJ alone solves within a few thousand effort units, with
    // several improvements, so each search finds something.  lseu and
    // qap04: under ruin's cutoff, which moves every search.  gt2 and p01: no
    // cutoff, as recombine's local search runs.
    struct Case {
        const char* instance;
        bool cut;
    };
    for (const Case c : {Case{"lseu.mps", true}, Case{"gt2.mps", false}, Case{"p01.mps", false},
                         Case{"qap04.mps", true}}) {
        for (const bool hold : {true, false}) {
            INFO(c.instance << " cut=" << c.cut << " hold=" << hold);
            Highs highs;
            highs.setOptionValue("output_flag", false);
            REQUIRE(highs.readModel(instances_dir() + "/" + c.instance) == HighsStatus::kOk);
            const Setup s(highs.getLp(), highs.getOptions().log_options);
            check_reuse_matches_fresh(s, c.cut, hold);
        }
    }
}

TEST_CASE("fj_reuse: the view's row-wise matrix is createRowwise's", "[fj][fj_reuse]") {
    // FJ used to transpose the column-wise matrix itself
    // (`HighsSparseMatrix::createRowwise`) and now reads the view's
    // (`highsSparseTranspose`, in `make_problem` and in HiGHS's own
    // `HighsMipSolverData::runSetup`): same rows, same order, same values,
    // so every sum FJ makes is the same.
    for (const char* instance : {"egout.mps", "bell5.mps", "flugpl.mps", "dcmulti.mps"}) {
        INFO(instance);
        Highs highs;
        highs.setOptionValue("output_flag", false);
        REQUIRE(highs.readModel(instances_dir() + "/" + instance) == HighsStatus::kOk);
        const HighsLp& lp = highs.getLp();
        REQUIRE(lp.a_matrix_.isColwise());
        const Setup s(lp, highs.getOptions().log_options);

        HighsSparseMatrix rowwise;
        rowwise.createRowwise(lp.a_matrix_);
        CHECK(rowwise.start_ == *s.problem.ar_start);
        CHECK(rowwise.index_ == *s.problem.ar_index);
        CHECK(rowwise.value_ == *s.problem.ar_value);
    }
}
