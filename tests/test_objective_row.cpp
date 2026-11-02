// The objective cut as a model row (#208, #231): the view `ObjectiveRowStorage`
// builds, and a `PropEngine` propagating it.  No clock, no random numbers.

#include "core_test_support.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "objective_row.h"
#include "prop_engine.h"
#include "test_paths.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <tuple>
#include <vector>

using namespace core_test;

namespace {

using Triple = std::tuple<HighsInt, HighsInt, double>;  // (row, column, value)

std::vector<Triple> row_wise_triples(const ProblemView& v) {
    std::vector<Triple> out;
    for (HighsInt i = 0; i < v.nrow; ++i) {
        for (HighsInt k = (*v.ar_start)[i]; k < (*v.ar_start)[i + 1]; ++k) {
            out.emplace_back(i, (*v.ar_index)[k], (*v.ar_value)[k]);
        }
    }
    std::ranges::sort(out);
    return out;
}

std::vector<Triple> column_wise_triples(const ProblemView& v) {
    std::vector<Triple> out;
    for (HighsInt j = 0; j < v.ncol; ++j) {
        for (HighsInt p = v.csc->col_start[j]; p < v.csc->col_start[j + 1]; ++p) {
            out.emplace_back(v.csc->col_row[p], j, v.csc->col_val[p]);
        }
    }
    std::ranges::sort(out);
    return out;
}

PropEngine make_engine(const ProblemView& v) {
    return {v.ncol,
            v.nrow,
            v.ar_start->data(),
            v.ar_index->data(),
            v.ar_value->data(),
            *v.csc,
            v.col_lower->data(),
            v.col_upper->data(),
            v.row_lower->data(),
            v.row_upper->data(),
            v.integrality->data(),
            v.feastol};
}

// Three binaries, costs 3, 2, 1, offset 10, and one cover row
// `x0 + x1 + x2 >= 1`.
HighsLp cover_model() {
    HighsLp lp;
    lp.num_col_ = 3;
    lp.num_row_ = 1;
    lp.col_cost_ = {3.0, 2.0, 1.0};
    lp.offset_ = 10.0;
    lp.col_lower_ = {0.0, 0.0, 0.0};
    lp.col_upper_ = {1.0, 1.0, 1.0};
    lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger, HighsVarType::kInteger};
    lp.row_lower_ = {1.0};
    lp.row_upper_ = {kHighsInf};
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_ = {0, 1, 2, 3};
    lp.a_matrix_.index_ = {0, 0, 0};
    lp.a_matrix_.value_ = {1.0, 1.0, 1.0};
    return lp;
}

}  // namespace

TEST_CASE("Objective row: the view appends the objective as row nrow", "[objective-cut]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/p0548.mps") == HighsStatus::kOk);
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    const ProblemView& base = s.problem;

    const ObjectiveRowStorage storage(base);
    const ProblemView& v = storage.view();

    REQUIRE(v.nrow == base.nrow + 1);
    REQUIRE(storage.objective_row() == base.nrow);
    REQUIRE(v.ncol == base.ncol);
    REQUIRE(v.ar_start->size() == static_cast<size_t>(v.nrow) + 1);
    REQUIRE(v.nnz == v.ar_index->size());
    REQUIRE(v.ar_value->size() == v.nnz);
    REQUIRE(v.csc->col_start.size() == static_cast<size_t>(v.ncol) + 1);
    REQUIRE(v.csc->col_row.size() == v.nnz);

    // Row-wise and column-wise hold the same entries.
    const std::vector<Triple> rows = row_wise_triples(v);
    REQUIRE(rows == column_wise_triples(v));

    // The first nrow rows are the base view's, and row nrow is the costs that
    // are exactly nonzero.
    std::vector<Triple> expected = row_wise_triples(base);
    size_t cost_nnz = 0;
    for (HighsInt j = 0; j < base.ncol; ++j) {
        if ((*base.col_cost)[j] != 0.0) {
            expected.emplace_back(base.nrow, j, (*base.col_cost)[j]);
            ++cost_nnz;
        }
    }
    std::ranges::sort(expected);
    REQUIRE(cost_nnz > 0);
    REQUIRE(rows == expected);
    REQUIRE(v.nnz == base.nnz + cost_nnz);

    // Row bounds: the base rows' unchanged, the objective row uncut.
    for (HighsInt i = 0; i < base.nrow; ++i) {
        REQUIRE((*v.row_lower)[i] == (*base.row_lower)[i]);
        REQUIRE((*v.row_upper)[i] == (*base.row_upper)[i]);
    }
    REQUIRE((*v.row_lower)[base.nrow] == -kHighsInf);
    REQUIRE((*v.row_upper)[base.nrow] == kHighsInf);

    // Locks and everything per-column are the base view's own.
    REQUIRE(v.uplocks == base.uplocks);
    REQUIRE(v.downlocks == base.downlocks);
    REQUIRE(v.col_lower == base.col_lower);
    REQUIRE(v.col_upper == base.col_upper);
    REQUIRE(v.col_cost == base.col_cost);
    REQUIRE(v.offset == base.offset);
    REQUIRE(v.integrality == base.integrality);
    REQUIRE(v.model == base.model);
}

TEST_CASE("Objective row: each worker's view owns its cutoff", "[objective-cut]") {
    const HighsLp lp = cover_model();
    const Setup s(lp, HighsLogOptions{});
    const ObjectiveRowStorage storage(s.problem);

    ObjectiveRowView a(storage);
    const ObjectiveRowView b(storage);
    REQUIRE(a.view().row_upper != storage.view().row_upper);
    REQUIRE(a.view().row_upper != b.view().row_upper);
    REQUIRE(a.view().row_lower == storage.view().row_lower);
    REQUIRE(a.view().ar_start == storage.view().ar_start);
    REQUIRE(a.view().csc == storage.view().csc);
    REQUIRE(a.row_bound() == kHighsInf);

    // The target is a pool objective, offset included; the row is without it.
    a.set_cutoff(12.5);
    REQUIRE(a.row_bound() == 2.5);
    REQUIRE((*a.view().row_upper)[0] == kHighsInf);  // the cover row untouched
    REQUIRE(b.row_bound() == kHighsInf);
    REQUIRE((*storage.view().row_upper)[1] == kHighsInf);

    a.set_cutoff(kHighsInf);
    REQUIRE(a.row_bound() == kHighsInf);
}

TEST_CASE("Objective row: propagation fixes the columns the cutoff forces", "[objective-cut]") {
    const HighsLp lp = cover_model();
    const Setup s(lp, HighsLogOptions{});
    const ObjectiveRowStorage storage(s.problem);
    ObjectiveRowView worker(storage);

    // Objective <= 12.5 is `3 x0 + 2 x1 + x2 <= 2.5`.  With x2 = 0 the
    // objective row rules x0 out (3 > 2.5), and the cover row then forces
    // x1 = 1.
    worker.set_cutoff(12.5);
    PropEngine cut = make_engine(worker.view());
    REQUIRE(cut.fix(2, 0.0));
    REQUIRE(cut.propagate(2) == PropResult::kFixpoint);
    REQUIRE(cut.var(0).ub == 0.0);
    REQUIRE(cut.var(1).lb == 1.0);

    // Uncut, the same fix leaves both free.
    worker.set_cutoff(kHighsInf);
    PropEngine uncut = make_engine(worker.view());
    REQUIRE(uncut.fix(2, 0.0));
    REQUIRE(uncut.propagate(2) == PropResult::kFixpoint);
    REQUIRE(uncut.var(0).lb == 0.0);
    REQUIRE(uncut.var(0).ub == 1.0);
    REQUIRE(uncut.var(1).lb == 0.0);
    REQUIRE(uncut.var(1).ub == 1.0);
}

TEST_CASE("Objective row: uncut, it propagates as the base view", "[objective-cut]") {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/p0548.mps") == HighsStatus::kOk);
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    const ObjectiveRowStorage storage(s.problem);
    const ProblemView& base = s.problem;

    PropEngine plain = make_engine(base);
    PropEngine with_row = make_engine(storage.view());

    // Fix every fourth integer column to its upper bound, propagating each,
    // until the base view's engine stops at an infeasibility or runs out.
    // Upper bounds, so p0548's knapsack rows have something to propagate.
    std::vector<char> decided(static_cast<size_t>(base.ncol), 0);
    for (HighsInt j = 0; j < base.ncol; j += 4) {
        if (!is_integer(*base.integrality, j) || plain.var(j).fixed) {
            continue;
        }
        const double value = plain.var(j).ub;
        REQUIRE(plain.fix(j, value) == with_row.fix(j, value));
        decided[j] = 1;
        const PropResult result = plain.propagate(j);
        REQUIRE(with_row.propagate(j) == result);
        if (result != PropResult::kFixpoint) {
            break;
        }
    }
    // Propagation itself moved some column no decision fixed, so the
    // comparison below exercises the rows rather than only the decisions.
    bool propagated = false;
    for (HighsInt j = 0; j < base.ncol; ++j) {
        propagated = propagated || (decided[j] == 0 && (plain.var(j).lb != (*base.col_lower)[j] ||
                                                        plain.var(j).ub != (*base.col_upper)[j]));
    }
    REQUIRE(propagated);
    for (HighsInt j = 0; j < base.ncol; ++j) {
        REQUIRE(plain.var(j).fixed == with_row.var(j).fixed);
        REQUIRE(plain.var(j).lb == with_row.var(j).lb);
        REQUIRE(plain.var(j).ub == with_row.var(j).ub);
    }
}
