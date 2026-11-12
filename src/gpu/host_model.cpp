#include "host_model.h"

#include "heuristic_context.h"
#include "lp_data/HConst.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace gpu_fj {

#ifdef HIGHSINT64
// `FjWorker` finishes at once in this build; so does the GPU model.
HostModel::HostModel(const ProblemView& /*problem*/) : legal_bounds_(false) {}
#else
static_assert(std::is_same_v<HighsInt, std::int32_t>,
              "the GPU model's indices are 32-bit, as FjWorker's are");

HostModel::HostModel(const ProblemView& problem) {
    const auto ncol = static_cast<size_t>(problem.ncol);
    const std::vector<HighsVarType>& integrality = *problem.integrality;
    col_lower_.resize(ncol);
    col_upper_.resize(ncol);
    col_is_int_.resize(ncol);
    // Bounds as `FjWorker::run_attempt` hands them to `addVar`.
    for (size_t col = 0; col < ncol; ++col) {
        double lower = (*problem.col_lower)[col];
        double upper = (*problem.col_upper)[col];
        const bool is_int = integrality[col] != HighsVarType::kContinuous;
        if (is_int) {
            lower = std::ceil(lower - problem.feastol);
            upper = std::floor(upper + problem.feastol);
        }
        legal_bounds_ = legal_bounds_ && lower <= upper && lower < kHighsInf &&
                        upper > -kHighsInf && !std::isnan(lower) && !std::isnan(upper);
        col_lower_[col] = lower;
        col_upper_[col] = upper;
        col_is_int_[col] = is_int ? 1 : 0;
    }

    // The constraints in `FjWorker`'s order: a row's `>=` half, then its
    // `<=` half; a free row neither, and an empty row neither, since
    // upstream's `addConstraint` drops a constraint with no coefficient.
    const std::vector<double>& row_lower = *problem.row_lower;
    const std::vector<double>& row_upper = *problem.row_upper;
    const std::vector<HighsInt>& row_start = *problem.ar_start;
    const auto nrow = static_cast<size_t>(problem.nrow);
    const auto halves = [&](size_t row) {
        if (row_start[row + 1] == row_start[row]) {
            return std::pair{false, false};
        }
        return std::pair{std::isfinite(row_lower[row]), std::isfinite(row_upper[row])};
    };
    size_t ncon = 0;
    for (size_t row = 0; row < nrow; ++row) {
        const auto [gte, lte] = halves(row);
        ncon += (gte ? 1 : 0) + (lte ? 1 : 0);
    }
    con_row_.reserve(ncon);
    con_rhs_.reserve(ncon);
    con_sense_.reserve(ncon);
    row_con_start_.reserve(nrow + 1);
    for (size_t row = 0; row < nrow; ++row) {
        row_con_start_.push_back(static_cast<std::int32_t>(con_row_.size()));
        const auto [gte, lte] = halves(row);
        if (gte) {
            con_row_.push_back(static_cast<std::int32_t>(row));
            con_rhs_.push_back(row_lower[row]);
            con_sense_.push_back(ConSense::kGte);
        }
        if (lte) {
            con_row_.push_back(static_cast<std::int32_t>(row));
            con_rhs_.push_back(row_upper[row]);
            con_sense_.push_back(ConSense::kLte);
        }
    }
    row_con_start_.push_back(static_cast<std::int32_t>(con_row_.size()));

    pod_.equality_tol = problem.epsilon;
    pod_.violation_tol = problem.feastol;
    pod_.ncol = problem.ncol;
    pod_.nrow = problem.nrow;
    pod_.nnz = static_cast<std::int64_t>(problem.nnz);
    pod_.row_start = problem.ar_start->data();
    pod_.row_index = problem.ar_index->data();
    pod_.row_value = problem.ar_value->data();
    pod_.col_start = problem.csc->col_start.data();
    pod_.col_row = problem.csc->col_row.data();
    pod_.col_value = problem.csc->col_val.data();
    pod_.col_lower = col_lower_.data();
    pod_.col_upper = col_upper_.data();
    pod_.col_cost = problem.col_cost->data();
    pod_.col_is_int = col_is_int_.data();
    pod_.ncon = static_cast<std::int32_t>(con_row_.size());
    pod_.con_row = con_row_.data();
    pod_.con_rhs = con_rhs_.data();
    pod_.con_sense = con_sense_.data();
    pod_.row_con_start = row_con_start_.data();
}
#endif

}  // namespace gpu_fj
