// FJ's structural personas (#206): row classification, cuOpt's covering and
// exact-k starts, and which slot gets which persona.  Core only, on
// hand-built models small enough to check by eye.

#include "core_test_support.h"
#include "fj.h"
#include "fj_personas.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "parallel/HighsParallel.h"
#include "structure.h"
#include "test_paths.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace core_test;

namespace {

struct Entry {
    HighsInt col;
    double value;
};

struct Row {
    double lower;
    double upper;
    std::vector<Entry> entries;
};

// A minimisation model with zero costs from its rows; `integrality` sizes
// the columns, binaries are `kInteger` on `[0, 1]`.
HighsLp model(const std::vector<HighsVarType>& integrality, const std::vector<double>& upper,
              const std::vector<Row>& rows) {
    HighsLp lp;
    lp.num_col_ = static_cast<HighsInt>(integrality.size());
    lp.num_row_ = static_cast<HighsInt>(rows.size());
    lp.col_cost_.assign(integrality.size(), 0.0);
    lp.col_lower_.assign(integrality.size(), 0.0);
    lp.col_upper_ = upper;
    lp.integrality_ = integrality;
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_.assign(1, 0);
    for (HighsInt col = 0; col < lp.num_col_; ++col) {
        for (size_t r = 0; r < rows.size(); ++r) {
            for (const Entry& e : rows[r].entries) {
                if (e.col == col) {
                    lp.a_matrix_.index_.push_back(static_cast<HighsInt>(r));
                    lp.a_matrix_.value_.push_back(e.value);
                }
            }
        }
        lp.a_matrix_.start_.push_back(static_cast<HighsInt>(lp.a_matrix_.index_.size()));
    }
    for (const Row& row : rows) {
        lp.row_lower_.push_back(row.lower);
        lp.row_upper_.push_back(row.upper);
    }
    return lp;
}

HighsLp binary_model(HighsInt ncol, const std::vector<Row>& rows) {
    return model(std::vector<HighsVarType>(static_cast<size_t>(ncol), HighsVarType::kInteger),
                 std::vector<double>(static_cast<size_t>(ncol), 1.0), rows);
}

// x0 + x1 >= 1, x1 + x2 >= 1, x2 + x3 >= 1 over binaries.
HighsLp chain_cover() {
    return binary_model(4, {{1, kHighsInf, {{0, 1}, {1, 1}}},
                            {1, kHighsInf, {{1, 1}, {2, 1}}},
                            {1, kHighsInf, {{2, 1}, {3, 1}}}});
}

}  // namespace

TEST_CASE("personas: every row kind is detected", "[core][fj_personas]") {
    using V = HighsVarType;
    // b0..b5 binary, i6 integer in [0, 10], c7 continuous in [0, 10].
    const HighsLp lp =
        model({V::kInteger, V::kInteger, V::kInteger, V::kInteger, V::kInteger, V::kInteger,
               V::kInteger, V::kContinuous},
              {1, 1, 1, 1, 1, 1, 10, 10},
              {
                  {1, 1, {{0, 1}, {1, 1}, {2, 1}}},           // partitioning, exact-k 1
                  {-kHighsInf, 1, {{0, 1}, {1, 1}}},          // packing
                  {1, kHighsInf, {{2, 1}, {3, 1}}},           // covering
                  {4, 4, {{0, 2}, {1, 2}, {2, 2}, {3, 2}}},   // cardinality, exact-k 2
                  {-kHighsInf, 2, {{3, 1}, {4, 1}, {5, 1}}},  // cardinality (<= 2)
                  {-kHighsInf, 0, {{6, 1}, {7, -1}}},         // precedence
                  {0, 0, {{7, 1}, {6, -1}, {5, -1}}},         // flow conservation
                  {2, kHighsInf, {{6, 3}, {7, 1}}},           // general
                  {0, kHighsInf, {{4, 1}, {5, 1}}},           // binds nothing: general
                  {-kHighsInf, 5, {{0, 1}, {1, 1}}},          // binds nothing: general
                  {1, 1, {{0, 1}, {1, 0.5}}},                 // mixed coefficients: general
              });
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    const RowStructure structure = detect_structure(s.problem);

    CHECK(structure.count_of(RowKind::kPartitioning) == 1);
    CHECK(structure.count_of(RowKind::kPacking) == 1);
    CHECK(structure.count_of(RowKind::kCovering) == 1);
    CHECK(structure.count_of(RowKind::kCardinality) == 2);
    CHECK(structure.count_of(RowKind::kPrecedence) == 1);
    CHECK(structure.count_of(RowKind::kFlowConservation) == 1);
    CHECK(structure.count_of(RowKind::kGeneral) == 4);
    // Partitioning, covering and the `= 4` row bind from below; packing and
    // the `<= 2` row do not.
    CHECK(structure.lower_bound_rows == 3);
    REQUIRE(structure.exact_k.size() == 2);
    CHECK(structure.exact_k[0].row == 0);
    CHECK(structure.exact_k[0].k == 1);
    CHECK(structure.exact_k[1].row == 3);
    CHECK(structure.exact_k[1].k == 2);
}

TEST_CASE("personas: the covering start repairs a chain cover from all zeros",
          "[core][fj_personas]") {
    const HighsLp lp = chain_cover();
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    std::vector<double> x(4, 0.0);
    StartBudget budget(s.exec);
    REQUIRE(count_violated_rows(s.problem, x, budget) == 3);
    REQUIRE(greedy_covering_start(s.problem, x, budget));
    CHECK(count_violated_rows(s.problem, x, budget) == 0);
    CHECK(is_feasible(lp, x));
}

TEST_CASE("personas: a start the covering step cannot improve comes back unchanged",
          "[core][fj_personas]") {
    const HighsLp lp = chain_cover();
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> feasible = {0, 1, 1, 0};
    std::vector<double> x = feasible;
    StartBudget budget(s.exec);
    CHECK_FALSE(greedy_covering_start(s.problem, x, budget));
    CHECK(x == feasible);
}

TEST_CASE("personas: the exact-k start fills narrow rows first, low degree first",
          "[core][fj_personas]") {
    // x0 + x1 + x2 = 1 and x2 + x3 = 1.  The width-2 row goes first and
    // takes x3 (degree 1) over x2 (degree 2), which leaves the wide row
    // x0; x1 and x2 are set to 0.
    const HighsLp lp =
        binary_model(4, {{1, 1, {{0, 1}, {1, 1}, {2, 1}}}, {1, 1, {{2, 1}, {3, 1}}}});
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    const RowStructure structure = detect_structure(s.problem);
    REQUIRE(structure.exact_k.size() == 2);
    std::vector<double> x(4, 0.0);
    StartBudget budget(s.exec);
    REQUIRE(exact_k_start(s.problem, structure.exact_k, x, budget));
    CHECK(x == std::vector<double>{1, 0, 0, 1});
}

TEST_CASE("personas: an exact-k start that leaves no fewer violated rows is dropped",
          "[core][fj_personas]") {
    // x0 + x1 = 1 from (0, 1), which satisfies it: the exact-k pick (1, 0)
    // leaves no fewer violated rows, so the start is kept as it was.
    const HighsLp lp = binary_model(2, {{1, 1, {{0, 1}, {1, 1}}}});
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    const RowStructure structure = detect_structure(s.problem);
    std::vector<double> x = {0, 1};
    StartBudget budget(s.exec);
    CHECK_FALSE(exact_k_start(s.problem, structure.exact_k, x, budget));
    CHECK(x == std::vector<double>{0, 1});
}

TEST_CASE("personas: an empty start becomes the bound start, then the persona's",
          "[core][fj_personas]") {
    const HighsLp lp = chain_cover();
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    const RowStructure structure = detect_structure(s.problem);
    std::vector<double> start;
    StartBudget budget(s.exec);
    apply_persona(FjPersona::kCovering, s.problem, structure, start, budget);
    REQUIRE(start.size() == 4);
    CHECK(is_feasible(lp, start));
}

// ── The start's charge and deadline (#326) ──

TEST_CASE("personas: a start on a bundled MIP is charged its passes over the rows",
          "[core][fj_personas]") {
    // issue-2095's `= 1` binary rows bind from below and are exact-k rows.
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/issue-2095.mps") == HighsStatus::kOk);
    const HighsLp& lp = highs.getLp();
    const Setup s(lp, highs.getOptions().log_options);
    const RowStructure structure = detect_structure(s.problem);
    REQUIRE(structure.lower_bound_rows > 0);
    for (const FjPersona persona : {FjPersona::kCovering, FjPersona::kExactK}) {
        std::vector<double> start;
        StartBudget budget(s.exec);
        apply_persona(persona, s.problem, structure, start, budget);
        CHECK_FALSE(budget.expired());
        // At least the covering step's two counts of every row.
        CHECK(budget.effort() >= 2 * s.problem.nnz);
    }
    // A seeded slot builds no start, so is charged nothing.
    std::vector<double> start;
    StartBudget seeded(s.exec);
    apply_persona(FjPersona::kSeeded, s.problem, structure, start, seeded);
    CHECK(seeded.effort() == 0);
}

TEST_CASE("personas: a start past the deadline returns at once, unchanged", "[core][fj_personas]") {
    const HighsLp lp = chain_cover();
    Highs highs;
    Setup s(lp, highs.getOptions().log_options);
    const std::atomic<bool> stopped{true};
    s.exec.stop = &stopped;
    const RowStructure structure = detect_structure(s.problem);
    const std::vector<double> zeros(4, 0.0);
    for (const FjPersona persona : {FjPersona::kCovering, FjPersona::kExactK}) {
        std::vector<double> start = zeros;
        StartBudget budget(s.exec);
        apply_persona(persona, s.problem, structure, start, budget);
        CHECK(budget.expired());
        CHECK(budget.effort() == 0);
        CHECK(start == zeros);
        std::vector<double> empty;
        StartBudget again(s.exec);
        apply_persona(persona, s.problem, structure, empty, again);
        CHECK(empty.empty());
    }
    // Each step keeps its input, never a result the deadline cut short.
    std::vector<double> x = zeros;
    StartBudget budget(s.exec);
    CHECK_FALSE(greedy_covering_start(s.problem, x, budget));
    CHECK_FALSE(exact_k_start(s.problem, structure.exact_k, x, budget));
    CHECK(x == zeros);
}

TEST_CASE("personas: fj::run charges the structure pass and the start to the dispatch",
          "[core][fj_personas]") {
    // x0 + x1 >= 1 and x0 + x1 <= 0 over binaries: a covering row, and no
    // feasible point, so nothing reaches the pool and every slot's first
    // attempt is the same whatever the threads do.  A search total of one
    // unit lets each slot that starts before the first attempt ends run that
    // attempt alone, so the dispatch's effort is the structure pass plus one
    // of three sums of first attempts, each slot's computed here by a worker
    // built as `fj::run` builds it.
    const HighsLp lp =
        binary_model(2, {{1, kHighsInf, {{0, 1}, {1, 1}}}, {-kHighsInf, 0, {{0, 1}, {1, 1}}}});
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs::parallel::initialize_scheduler();
    Setup s(lp, highs.getOptions().log_options);
    s.exec.num_workers = 2;
    const RowStructure structure = detect_structure(s.problem);
    REQUIRE(assign_personas(structure, s.problem.nnz, 2) ==
            std::vector<FjPersona>{FjPersona::kSeeded, FjPersona::kCovering});

    constexpr size_t kLarge = size_t{1} << 40;
    const HeuristicBudget budget{.total = s.problem.nnz + 1,
                                 .per_worker = kLarge,
                                 .attempt_cap = kLarge,
                                 .stale = kLarge,
                                 .worker_stale = kLarge,
                                 .fj_personas = true};

    std::vector<double> persona_start;
    StartBudget start_budget(s.exec);
    apply_persona(FjPersona::kCovering, s.problem, structure, persona_start, start_budget);
    const size_t start_effort = start_budget.effort();
    REQUIRE(start_effort > 0);

    const uint32_t seed = s.exec.base_seed - kBaseSeedOffset;
    SolutionSink scratch(*s.problem.integrality, 0);
    FjWorker slot0(s.problem, s.exec, scratch, kLarge, kLarge, seed, {}, WorkerTrace{0, 0});
    const size_t first0 = slot0.run_attempt(1).effort;
    FjWorker slot1(s.problem, s.exec, scratch, kLarge - start_effort, kLarge, seed + 1,
                   persona_start, WorkerTrace{1, start_effort});
    const size_t first1 = slot1.run_attempt(0).effort;
    REQUIRE(scratch.accepted() == 0);

    const size_t pass = s.problem.nnz;
    const std::set<size_t> charged = {pass + first0, pass + start_effort + first1,
                                      pass + first0 + start_effort + first1};
    // The same three without the two charges, so a match above is one only
    // a dispatch that books them can make.
    for (const size_t uncharged : {first0, first1, first0 + first1}) {
        REQUIRE_FALSE(charged.contains(uncharged));
    }
    SolutionSink sink(*s.problem.integrality, 0);
    const DispatchOutcome outcome = fj::run(s.problem, budget, s.exec, sink);
    CHECK(charged.contains(outcome.effort));
}

TEST_CASE("personas: slots take what the structure supports, slot 0 always seeded",
          "[core][fj_personas]") {
    using P = FjPersona;
    RowStructure none;
    CHECK(assign_personas(none, 10, 4) == std::vector<P>(4, P::kSeeded));

    // Packing and `<= k` rows hold at the bound start: no covering slot.
    RowStructure packing;
    packing.count[static_cast<size_t>(RowKind::kPacking)] = 3;
    packing.count[static_cast<size_t>(RowKind::kCardinality)] = 2;
    CHECK(assign_personas(packing, 10, 4) == std::vector<P>(4, P::kSeeded));

    RowStructure cover;
    cover.count[static_cast<size_t>(RowKind::kCovering)] = 3;
    cover.lower_bound_rows = 3;
    CHECK(assign_personas(cover, 10, 4) ==
          std::vector<P>{P::kSeeded, P::kCovering, P::kSeeded, P::kSeeded});

    // Only `= 0` exact-k rows: nothing for the exact-k start to set.
    RowStructure zero_k = cover;
    zero_k.exact_k.push_back({0, 0});
    CHECK(assign_personas(zero_k, 10, 4) ==
          std::vector<P>{P::kSeeded, P::kCovering, P::kSeeded, P::kSeeded});

    RowStructure both = cover;
    both.exact_k.push_back({0, 1});
    CHECK(assign_personas(both, 10, 4) ==
          std::vector<P>{P::kSeeded, P::kCovering, P::kExactK, P::kSeeded});
    CHECK(assign_personas(both, 10, 2) == std::vector<P>{P::kSeeded, P::kCovering});
    CHECK(assign_personas(both, 10, 1) == std::vector<P>{P::kSeeded});
    CHECK(assign_personas(both, kPersonaNnzLimit + 1, 4) == std::vector<P>(4, P::kSeeded));
}

// cuOpt's LP start (#217).  i0 integer in [0, 10], c1 continuous in
// [0, 10], b2 binary; one row so the model has one.
TEST_CASE("personas: the LP start clamps, keeps integral values, rounds by fraction",
          "[core][fj_personas]") {
    using V = HighsVarType;
    const HighsLp lp = model({V::kInteger, V::kContinuous, V::kInteger}, {10, 10, 1},
                             {{-kHighsInf, 20, {{0, 1}, {1, 1}, {2, 1}}}});
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    Rng rng(7);
    std::vector<double> x;
    // Clamped into the bounds; the integral value and the continuous one
    // pass through.
    REQUIRE(lp_rounded_start(s.problem, std::vector<double>{3.0, 12.5, -0.2}, rng, x));
    CHECK(x == std::vector<double>{3, 10, 0});
    // A fractional value rounds to one of its two neighbours, and over many
    // draws to both: up with probability its fractional part.
    int up = 0;
    constexpr int kDraws = 200;
    for (int i = 0; i < kDraws; ++i) {
        REQUIRE(lp_rounded_start(s.problem, std::vector<double>{2.25, 4.5, 0.5}, rng, x));
        REQUIRE((x[0] == 2.0 || x[0] == 3.0));
        CHECK(x[1] == 4.5);
        up += x[0] == 3.0 ? 1 : 0;
    }
    CHECK(up > 0);
    CHECK(up < kDraws / 2);
}

TEST_CASE("personas: the LP start gives up where cuOpt drops it", "[core][fj_personas]") {
    using V = HighsVarType;
    Highs highs;
    Rng rng(7);
    std::vector<double> x;
    // A continuous column with no finite bound and an infinite LP value.
    HighsLp free_col = model({V::kContinuous}, {kHighsInf}, {{-kHighsInf, 1, {{0, 1}}}});
    free_col.col_lower_ = {-kHighsInf};
    const Setup free_setup(free_col, highs.getOptions().log_options);
    CHECK_FALSE(lp_rounded_start(free_setup.problem, std::vector<double>{kHighsInf}, rng, x));
    // A NaN LP value on a bounded column: the clamp would land it on a bound.
    HighsLp boxed = model({V::kContinuous}, {kHighsInf}, {{-kHighsInf, 1, {{0, 1}}}});
    boxed.col_lower_ = {0.0};
    boxed.col_upper_ = {1.0};
    const Setup boxed_setup(boxed, highs.getOptions().log_options);
    CHECK_FALSE(lp_rounded_start(boxed_setup.problem,
                                 std::vector<double>{std::numeric_limits<double>::quiet_NaN()}, rng,
                                 x));
    // An integer column with no integral value in its bounds is cuOpt's
    // other give-up, but the core `make_problem` refuses such a model, so
    // no view reaches it here.
}
