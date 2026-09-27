#include "heuristic_context.h"

#include "lp_data/HConst.h"
#include "lp_data/HighsLp.h"
#include "util/HighsUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using Kind = ProblemError::Kind;

// HiGHS's defaults for the three options its model assessment reads
// (`lp_data/HighsOptions.h`, v1.15.1): a bound or cost of this magnitude is
// infinite, and a coefficient of this magnitude is refused.
constexpr double kInfiniteBound = 1e20;
constexpr double kInfiniteCost = 1e20;
constexpr double kLargeMatrixValue = 1e15;

ProblemError error(Kind kind, std::string message) {
    return ProblemError{.kind = kind, .message = std::move(message)};
}

std::string where(const char* what, size_t i) {
    return std::string(what) + " " + std::to_string(i);
}

bool valid_type(HighsVarType type) {
    switch (type) {
        case HighsVarType::kContinuous:
        case HighsVarType::kInteger:
        case HighsVarType::kSemiContinuous:
        case HighsVarType::kSemiInteger:
        case HighsVarType::kImplicitInteger:
            return true;
    }
    return false;
}

// Every array the model's dimensions size, and the model's scalars.  Must
// pass before anything indexes the model.
std::optional<ProblemError> check_shape(const HighsLp& m) {
    const auto ncol = static_cast<size_t>(m.num_col_);
    const auto nrow = static_cast<size_t>(m.num_row_);
    if (m.num_col_ < 0 || m.num_row_ < 0 || m.col_cost_.size() != ncol ||
        m.col_lower_.size() != ncol || m.col_upper_.size() != ncol || m.row_lower_.size() != nrow ||
        m.row_upper_.size() != nrow || (!m.integrality_.empty() && m.integrality_.size() != ncol)) {
        return error(Kind::kMalformed, "an array's size disagrees with the model's dimensions");
    }
    if (m.sense_ != ObjSense::kMinimize && m.sense_ != ObjSense::kMaximize) {
        return error(Kind::kMalformed, "the objective sense is neither minimise nor maximise");
    }
    if (!std::isfinite(m.offset_)) {
        return error(Kind::kMalformed, "the objective offset is not finite");
    }
    return std::nullopt;
}

// The matrix, as HiGHS's `assessMatrix` sees it: column-wise with the
// model's own dimensions (FJ's `createRowwise` reads the matrix's
// `num_col_`/`num_row_`), starts that begin at 0, are monotone and cover
// exactly the index and value arrays, row indices in range and not repeated
// within a column, and coefficients finite and below `kLargeMatrixValue`.
std::optional<ProblemError> check_matrix(const HighsLp& m) {
    const auto ncol = static_cast<size_t>(m.num_col_);
    const HighsSparseMatrix& a = m.a_matrix_;
    if (!a.isColwise() || a.num_col_ != m.num_col_ || a.num_row_ != m.num_row_ ||
        a.start_.size() != ncol + 1) {
        return error(Kind::kMalformed,
                     "the constraint matrix is not column-wise with the model's dimensions");
    }
    if (a.start_[0] != 0) {
        return error(Kind::kMalformed, "the matrix's column starts do not begin at 0");
    }
    for (size_t j = 0; j < ncol; ++j) {
        if (a.start_[j + 1] < a.start_[j]) {
            return error(Kind::kMalformed, "the matrix's column starts are not monotone");
        }
    }
    const auto nnz = static_cast<size_t>(a.start_[ncol]);
    if (a.index_.size() != nnz || a.value_.size() != nnz) {
        return error(Kind::kMalformed,
                     "the matrix's starts do not cover its index and value arrays");
    }
    // The column that last used each row, to find a repeat in one pass.
    std::vector<HighsInt> last_col(static_cast<size_t>(m.num_row_), -1);
    for (HighsInt j = 0; j < m.num_col_; ++j) {
        for (HighsInt k = a.start_[j]; k < a.start_[j + 1]; ++k) {
            const HighsInt i = a.index_[k];
            if (i < 0 || i >= m.num_row_) {
                return error(Kind::kMalformed,
                             where("column", j) + " has a row index out of range");
            }
            if (last_col[i] == j) {
                return error(Kind::kMalformed, where("column", j) + " names a row twice");
            }
            last_col[i] = j;
            if (!(std::abs(a.value_[k]) < kLargeMatrixValue)) {
                return error(Kind::kMalformed,
                             where("column", j) + " has a non-finite or huge coefficient");
            }
        }
    }
    return std::nullopt;
}

// One bound pair into `lower`/`upper`, as HiGHS's `assessBounds` treats it:
// a magnitude at or above `kInfiniteBound` becomes infinite, and a lower
// bound at `+kInfiniteBound` or an upper one at `-kInfiniteBound` is an
// error.  Crossing is checked by the caller, after any rounding.
std::optional<ProblemError> map_bounds(double& lower, double& upper, const std::string& name) {
    if (std::isnan(lower) || std::isnan(upper)) {
        return error(Kind::kMalformed, name + " has a NaN bound");
    }
    if (lower <= -kInfiniteBound) {
        lower = -kHighsInf;
    }
    if (upper >= kInfiniteBound) {
        upper = kHighsInf;
    }
    if (lower >= kInfiniteBound || upper <= -kInfiniteBound) {
        return error(Kind::kMalformed,
                     name + " has an infinite lower bound or -infinite upper one");
    }
    return std::nullopt;
}

// Column by column: types, costs and bounds as the workers use them — an
// integer column's rounded inward into `storage`, then refused if they cross.
std::optional<ProblemError> normalise_columns(const HighsLp& m,
                                              const std::vector<HighsVarType>& integrality,
                                              double feastol, ProblemStorage& storage) {
    storage.col_lower = m.col_lower_;
    storage.col_upper = m.col_upper_;
    for (HighsInt j = 0; j < m.num_col_; ++j) {
        const std::string name = where("column", j);
        const HighsVarType type = integrality[j];
        if (!valid_type(type)) {
            return error(Kind::kMalformed, name + " has an integrality outside HighsVarType");
        }
        if (type == HighsVarType::kSemiContinuous || type == HighsVarType::kSemiInteger) {
            return error(Kind::kUnsupported,
                         name +
                             " is semi-continuous or semi-integer, which the heuristics do not "
                             "model");
        }
        if (!(std::abs(m.col_cost_[j]) < kInfiniteCost)) {
            return error(Kind::kMalformed, name + " has a non-finite cost");
        }
        double& lb = storage.col_lower[j];
        double& ub = storage.col_upper[j];
        if (auto refused = map_bounds(lb, ub, name)) {
            return refused;
        }
        const bool integer = type != HighsVarType::kContinuous;
        if (integer) {
            // `+ 0.0` turns the `-0.0` that `ceil` gives a bound just below
            // zero into `0.0`, so no worker clamps a value to a negative zero.
            lb = std::ceil(lb - feastol) + 0.0;
            ub = std::floor(ub + feastol) + 0.0;
        }
        if (lb > ub) {
            return error(
                Kind::kInfeasible,
                name + "'s bounds cross" + (integer ? " once rounded inward to integers" : ""));
        }
    }
    return std::nullopt;
}

// Row by row, with the row-wise starts already built: bounds mapped into
// `storage`, crossed bounds, and an empty row whose bounds exclude 0 —
// nothing any assignment can do satisfies it, so a worker would report a
// violating point.
std::optional<ProblemError> normalise_rows(const HighsLp& m, double feastol,
                                           ProblemStorage& storage) {
    storage.row_lower = m.row_lower_;
    storage.row_upper = m.row_upper_;
    for (HighsInt i = 0; i < m.num_row_; ++i) {
        const std::string name = where("row", i);
        double& lo = storage.row_lower[i];
        double& hi = storage.row_upper[i];
        if (auto refused = map_bounds(lo, hi, name)) {
            return refused;
        }
        if (lo > hi) {
            return error(Kind::kInfeasible, name + "'s bounds cross");
        }
        if (storage.ar_start[i] == storage.ar_start[i + 1] && (lo > feastol || hi < -feastol)) {
            return error(Kind::kInfeasible, name + " is empty and its bounds exclude 0");
        }
    }
    return std::nullopt;
}

// `HighsMipSolverData::runSetup()`'s rule: a finite row lower bound locks a
// column against moving in the direction that decreases the row, a finite
// upper bound in the direction that increases it.
void count_locks(const HighsLp& model, ProblemStorage& storage) {
    const HighsSparseMatrix& a = model.a_matrix_;
    storage.uplocks.assign(model.num_col_, 0);
    storage.downlocks.assign(model.num_col_, 0);
    for (HighsInt j = 0; j < model.num_col_; ++j) {
        for (HighsInt k = a.start_[j]; k < a.start_[j + 1]; ++k) {
            const HighsInt i = a.index_[k];
            const bool negative = a.value_[k] < 0;
            if (storage.row_lower[i] != -kHighsInf) {
                ++(negative ? storage.uplocks : storage.downlocks)[j];
            }
            if (storage.row_upper[i] != kHighsInf) {
                ++(negative ? storage.downlocks : storage.uplocks)[j];
            }
        }
    }
}

}  // namespace

std::expected<ProblemView, ProblemError> make_problem(const HighsLp& model, ProblemStorage& storage,
                                                      double feastol, double epsilon) {
    if (auto refused = check_shape(model)) {
        return std::unexpected(std::move(*refused));
    }
    if (auto refused = check_matrix(model)) {
        return std::unexpected(std::move(*refused));
    }
    const HighsInt ncol = model.num_col_;
    const HighsInt nrow = model.num_row_;

    const std::vector<HighsVarType>* integrality = &model.integrality_;
    if (model.integrality_.empty()) {
        storage.integrality.assign(ncol, HighsVarType::kContinuous);
        integrality = &storage.integrality;
    }
    if (auto refused = normalise_columns(model, *integrality, feastol, storage)) {
        return std::unexpected(std::move(*refused));
    }

    // Minimisation form, as HPresolve leaves a model before any heuristic
    // runs inside HiGHS.
    const std::vector<double>* col_cost = &model.col_cost_;
    double offset = model.offset_;
    if (model.sense_ == ObjSense::kMaximize) {
        storage.col_cost.resize(ncol);
        for (HighsInt j = 0; j < ncol; ++j) {
            storage.col_cost[j] = -model.col_cost_[j];
        }
        col_cost = &storage.col_cost;
        offset = -model.offset_;
    }

    const HighsSparseMatrix& a = model.a_matrix_;
    // The transpose `resize`s rather than assigns its starts, so a reused
    // `storage` would keep a stale `ar_start[0]`.
    storage.ar_start.clear();
    highsSparseTranspose(nrow, ncol, a.start_, a.index_, a.value_, storage.ar_start,
                         storage.ar_index, storage.ar_value);
    if (auto refused = normalise_rows(model, feastol, storage)) {
        return std::unexpected(std::move(*refused));
    }
    storage.csc = build_csc(ncol, nrow, storage.ar_start, storage.ar_index, storage.ar_value);
    count_locks(model, storage);

    std::vector<uint8_t> binary(static_cast<size_t>(ncol), 0);
    for (HighsInt j = 0; j < ncol; ++j) {
        binary[j] = is_integer(*integrality, j) && storage.col_lower[j] == 0.0 &&
                            storage.col_upper[j] == 1.0
                        ? 1
                        : 0;
    }

    return ProblemView{.model = &model,
                       .col_lower = &storage.col_lower,
                       .col_upper = &storage.col_upper,
                       .row_lower = &storage.row_lower,
                       .row_upper = &storage.row_upper,
                       .col_cost = col_cost,
                       .offset = offset,
                       .integrality = integrality,
                       .ar_start = &storage.ar_start,
                       .ar_index = &storage.ar_index,
                       .ar_value = &storage.ar_value,
                       .csc = &storage.csc,
                       .uplocks = &storage.uplocks,
                       .downlocks = &storage.downlocks,
                       .feastol = feastol,
                       .epsilon = epsilon,
                       .ncol = ncol,
                       .nrow = nrow,
                       .nnz = storage.ar_index.size(),
                       .incumbent = {},
                       .binary = std::move(binary)};
}

bool take_restart(const RestartSource& source, Rng& rng, std::vector<double>& out,
                  const ProblemView& problem) {
    if (source && source(rng, out) && std::cmp_equal(out.size(), problem.ncol)) {
        bool usable = true;
        for (HighsInt j = 0; j < problem.ncol && usable; ++j) {
            double v = out[j];
            usable = std::isfinite(v);
            if ((*problem.integrality)[j] != HighsVarType::kContinuous) {
                v = std::round(v);
            }
            out[j] = std::max((*problem.col_lower)[j], std::min((*problem.col_upper)[j], v));
        }
        if (usable) {
            return true;
        }
    }
    out.clear();
    return false;
}
