#include "objective_row.h"

#include <cmath>
#include <cstddef>

ObjectiveRowStorage::ObjectiveRowStorage(const ProblemView& base) : view_(base) {
    const HighsInt ncol = base.ncol;
    const HighsInt nrow = base.nrow;
    const std::vector<double>& cost = *base.col_cost;

    size_t cost_nnz = 0;
    for (HighsInt j = 0; j < ncol; ++j) {
        cost_nnz += cost[j] != 0.0 ? 1 : 0;
    }

    ar_start_.reserve(static_cast<size_t>(nrow) + 2);
    ar_start_.assign(base.ar_start->begin(), base.ar_start->end());
    ar_index_.reserve(base.nnz + cost_nnz);
    ar_index_.assign(base.ar_index->begin(), base.ar_index->end());
    ar_value_.reserve(base.nnz + cost_nnz);
    ar_value_.assign(base.ar_value->begin(), base.ar_value->end());
    for (HighsInt j = 0; j < ncol; ++j) {
        if (cost[j] != 0.0) {
            ar_index_.push_back(j);
            ar_value_.push_back(cost[j]);
        }
    }
    ar_start_.push_back(static_cast<HighsInt>(ar_index_.size()));
    csc_ = build_csc(ncol, nrow + 1, ar_start_, ar_index_, ar_value_);

    row_lower_.reserve(static_cast<size_t>(nrow) + 1);
    row_lower_.assign(base.row_lower->begin(), base.row_lower->end());
    row_lower_.push_back(-kHighsInf);
    row_upper_.reserve(static_cast<size_t>(nrow) + 1);
    row_upper_.assign(base.row_upper->begin(), base.row_upper->end());
    row_upper_.push_back(kHighsInf);

    view_.ar_start = &ar_start_;
    view_.ar_index = &ar_index_;
    view_.ar_value = &ar_value_;
    view_.csc = &csc_;
    view_.row_lower = &row_lower_;
    view_.row_upper = &row_upper_;
    view_.nrow = nrow + 1;
    view_.nnz = ar_index_.size();
}

ObjectiveRowView::ObjectiveRowView(const ObjectiveRowStorage& storage)
    : row_upper_(*storage.view().row_upper), view_(storage.view()) {
    view_.row_upper = &row_upper_;
}

void ObjectiveRowView::set_cutoff(double target) {
    row_upper_.back() = std::isfinite(target) ? target - view_.offset : kHighsInf;
}
