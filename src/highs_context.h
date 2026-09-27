#pragma once

// The HiGHS side of the execution scaffold (#170): everything that builds a
// `ProblemView` or an `ExecutionContext` from a live `HighsMipSolver`.  The
// types themselves, and everything the heuristics do with them, are in
// heuristic_context.h and include no `mip/` header.

#include "deadline.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

// Snapshot `HighsDomain::isBinary` for every column.  Must run on the
// dispatching thread, before any parallel region — see `ProblemView::binary`.
inline std::vector<uint8_t> build_binary_mask(const HighsMipSolver& mipsolver) {
    const HighsInt ncol = mipsolver.model_->num_col_;
    const HighsDomain& domain = mipsolver.mipdata_->getDomain();
    std::vector<uint8_t> mask(static_cast<size_t>(ncol), 0);
    for (HighsInt j = 0; j < ncol; ++j) {
        mask[j] = domain.isBinary(j) ? 1 : 0;
    }
    return mask;
}

// The dispatch's deadline, for a callee that was handed the solver but not
// the `ExecutionContext` built from it — `ContestedPdlp` and `fpr_lp`'s
// setup (issue #117).  `make_exec` builds
// `ExecutionContext::deadline` with it, so the two cannot drift: there is one
// option and one clock.
inline Deadline deadline_of(const HighsMipSolver& mipsolver) {
    return make_deadline(mipsolver.timer_, mipsolver.options_mip_->time_limit);
}

// Derive one dispatch's execution parameters.  Shared by `run_sequential`
// and by `fpr_lp`, which runs on the same continuous parallel runner from a
// setup of its own shape.
inline ExecutionContext make_exec(HighsMipSolver& mipsolver) {
    return ExecutionContext{
        .num_workers = static_cast<size_t>(std::max(1, highs::parallel::num_threads())),
        .base_seed = heuristic_base_seed(mipsolver.options_mip_->random_seed),
        .deadline = deadline_of(mipsolver),
        .log_options = mipsolver.options_mip_->log_options,
        .terminator = [&mipsolver] { return mipsolver.mipdata_->terminatorTerminated(); }};
}

// A view over the solver's model and row-wise buffers plus a CSC transpose
// the caller already holds, with the incumbent and `isBinary` snapshots.
// `csc` must outlive every use of the returned view.  Must be called on the
// dispatching thread, before any parallel region — see
// `ProblemView::incumbent`.  `fpr_lp` calls it directly, over the CSC its
// setup built; everything else goes through `make_problem` below.
inline ProblemView problem_view(HighsMipSolver& mipsolver, const CscMatrix& csc) {
    const HighsLp* model = mipsolver.model_;
    HighsMipSolverData* mipdata = mipsolver.mipdata_.get();
    // The presolved model is already in minimisation form: `HPresolve`
    // negates a maximisation model's costs and offset before anything else,
    // presolve on or off, so the view points at its costs unchanged.
    assert(model->sense_ == ObjSense::kMinimize);
    // Designated initialisers: two snapshots have been appended to this
    // aggregate in as many issues, and three of the members in the middle
    // are a positional `HighsInt, HighsInt, size_t` run that a mis-ordered
    // addition would silently convert between.
    return ProblemView{.model = model,
                       .col_lower = &model->col_lower_,
                       .col_upper = &model->col_upper_,
                       .row_lower = &model->row_lower_,
                       .row_upper = &model->row_upper_,
                       .col_cost = &model->col_cost_,
                       .offset = model->offset_,
                       .integrality = &model->integrality_,
                       .ar_start = &mipdata->ARstart_,
                       .ar_index = &mipdata->ARindex_,
                       .ar_value = &mipdata->ARvalue_,
                       .csc = &csc,
                       .uplocks = &mipdata->uplocks,
                       .downlocks = &mipdata->downlocks,
                       .feastol = mipdata->feastol,
                       .epsilon = mipdata->epsilon,
                       .ncol = model->num_col_,
                       .nrow = model->num_row_,
                       .nnz = mipdata->ARindex_.size(),
                       .incumbent = mipdata->incumbent,
                       .binary = build_binary_mask(mipsolver)};
}

// Build the CSC transpose into caller-owned `csc` and return
// `problem_view` over it.
//
// One call covers a whole FJ -> FPR -> LocalMIP -> Scylla chain: the
// row-major buffers the transpose is built from are written by
// `HighsMipSolverData::runSetup()` before any heuristic dispatch and are
// not touched again while the chain runs, so a single snapshot is valid for
// all four.  (Each heuristic used to build its own identical copy.)
inline ProblemView make_problem(HighsMipSolver& mipsolver, CscMatrix& csc) {
    const HighsMipSolverData* mipdata = mipsolver.mipdata_.get();
    csc = build_csc(mipsolver.model_->num_col_, mipsolver.model_->num_row_, mipdata->ARstart_,
                    mipdata->ARindex_, mipdata->ARvalue_);
    return problem_view(mipsolver, csc);
}
