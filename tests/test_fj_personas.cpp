// FJ's structural personas (#206): row classification, cuOpt's covering and
// exact-k starts, and which slot gets which persona.  Core only, on
// hand-built models small enough to check by eye.

#include "core_test_support.h"
#include "fj_personas.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "structure.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
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
    REQUIRE(count_violated_rows(s.problem, x) == 3);
    REQUIRE(greedy_covering_start(s.problem, x));
    CHECK(count_violated_rows(s.problem, x) == 0);
    CHECK(is_feasible(lp, x));
}

TEST_CASE("personas: a start the covering step cannot improve comes back unchanged",
          "[core][fj_personas]") {
    const HighsLp lp = chain_cover();
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    const std::vector<double> feasible = {0, 1, 1, 0};
    std::vector<double> x = feasible;
    CHECK_FALSE(greedy_covering_start(s.problem, x));
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
    REQUIRE(exact_k_start(s.problem, structure.exact_k, x));
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
    CHECK_FALSE(exact_k_start(s.problem, structure.exact_k, x));
    CHECK(x == std::vector<double>{0, 1});
}

TEST_CASE("personas: an empty start becomes the bound start, then the persona's",
          "[core][fj_personas]") {
    const HighsLp lp = chain_cover();
    Highs highs;
    const Setup s(lp, highs.getOptions().log_options);
    const RowStructure structure = detect_structure(s.problem);
    std::vector<double> start;
    apply_persona(FjPersona::kCovering, s.problem, structure, start);
    REQUIRE(start.size() == 4);
    CHECK(is_feasible(lp, start));
}

TEST_CASE("personas: slots take what the structure supports, slot 0 always seeded",
          "[core][fj_personas]") {
    using P = FjPersona;
    RowStructure none;
    CHECK(assign_personas(none, 10, 4) == std::vector<P>(4, P::kSeeded));

    RowStructure cover;
    cover.count[static_cast<size_t>(RowKind::kCovering)] = 3;
    CHECK(assign_personas(cover, 10, 4) ==
          std::vector<P>{P::kSeeded, P::kCovering, P::kSeeded, P::kSeeded});

    RowStructure both = cover;
    both.exact_k.push_back({0, 1});
    CHECK(assign_personas(both, 10, 4) ==
          std::vector<P>{P::kSeeded, P::kCovering, P::kExactK, P::kSeeded});
    CHECK(assign_personas(both, 10, 2) == std::vector<P>{P::kSeeded, P::kCovering});
    CHECK(assign_personas(both, 10, 1) == std::vector<P>{P::kSeeded});
    CHECK(assign_personas(both, kPersonaNnzLimit + 1, 4) == std::vector<P>(4, P::kSeeded));
}
