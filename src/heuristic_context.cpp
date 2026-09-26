#include "heuristic_context.h"

#include "lp_data/HConst.h"
#include "lp_data/HighsLp.h"
#include "util/HighsUtils.h"

#include <cstdint>
#include <utility>
#include <vector>

ProblemView make_problem(const HighsLp& model, ProblemStorage& storage, double feastol,
                         double epsilon) {
    const HighsInt ncol = model.num_col_;
    const HighsInt nrow = model.num_row_;
    const HighsSparseMatrix& a = model.a_matrix_;
    // The transpose `resize`s rather than assigns its starts, so a reused
    // `storage` would keep a stale `ar_start[0]`.
    storage.ar_start.clear();
    highsSparseTranspose(nrow, ncol, a.start_, a.index_, a.value_, storage.ar_start,
                         storage.ar_index, storage.ar_value);
    storage.csc = build_csc(ncol, nrow, storage.ar_start, storage.ar_index, storage.ar_value);

    // `HighsMipSolverData::runSetup()`'s rule: a finite row lower bound
    // locks a column against moving in the direction that decreases the
    // row, a finite upper bound in the direction that increases it.
    storage.uplocks.assign(ncol, 0);
    storage.downlocks.assign(ncol, 0);
    for (HighsInt j = 0; j < ncol; ++j) {
        for (HighsInt k = a.start_[j]; k < a.start_[j + 1]; ++k) {
            const HighsInt i = a.index_[k];
            const bool negative = a.value_[k] < 0;
            if (model.row_lower_[i] != -kHighsInf) {
                ++(negative ? storage.uplocks : storage.downlocks)[j];
            }
            if (model.row_upper_[i] != kHighsInf) {
                ++(negative ? storage.downlocks : storage.uplocks)[j];
            }
        }
    }

    std::vector<uint8_t> binary(static_cast<size_t>(ncol), 0);
    for (HighsInt j = 0; j < ncol; ++j) {
        binary[j] = is_integer(model.integrality_, j) && model.col_lower_[j] == 0.0 &&
                            model.col_upper_[j] == 1.0
                        ? 1
                        : 0;
    }

    return ProblemView{.model = &model,
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
