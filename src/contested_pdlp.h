#pragma once

#include "Highs.h"
#include "lp_data/HighsStatus.h"
#include "util/HighsInt.h"

#include <cstddef>
#include <mutex>
#include <vector>

class HighsMipSolver;

// Thread-safe wrapper around a single PDLP `Highs` instance shared by N
// Scylla workers.  One mutex guards the entire `changeColsCost →
// setSolution → run → getSolution` critical section so only one PDLP
// solve is in flight at a time.  This eliminates concurrency questions
// around the underlying (possibly GPU-backed cuPDLP) solver and keeps
// memory to a single LP copy + single iterate regardless of N.
class ContestedPdlp {
public:
    struct SolveResult {
        std::vector<double> col_value;
        std::vector<double> row_dual;
        HighsInt pdlp_iters = 0;
        HighsStatus status = HighsStatus::kError;
        HighsModelStatus model_status = HighsModelStatus::kNotset;
        bool value_valid = false;
        bool dual_valid = false;
    };

    // Builds the shared PDLP Highs instance from the presolved MIP
    // relaxation.  `initialized()==false` when the instance has no
    // rows / no nonzeros; callers should short-circuit.
    ContestedPdlp(HighsMipSolver &mipsolver, HighsInt pdlp_iter_cap);

    ContestedPdlp(const ContestedPdlp &) = delete;
    ContestedPdlp &operator=(const ContestedPdlp &) = delete;

    bool initialized() const { return initialized_; }
    size_t nnz_lp() const { return nnz_lp_; }
    HighsInt num_col() const { return ncol_; }

    // Solve PDLP with the caller's objective and warm-start.  The mutex
    // is held for the full changeColsCost + setSolution + run +
    // getSolution path; callers block when another chain is active.
    //
    // `warm_start_col_value` / `warm_start_row_dual` may be empty (cold
    // start) but must otherwise have length == ncol/nrow respectively.
    // `epsilon` is passed as `pdlp_optimality_tolerance`.  `time_limit`
    // is a wall-clock cap for this single solve (seconds).
    SolveResult solve(const std::vector<double> &modified_cost,
                      const std::vector<double> &warm_start_col_value,
                      const std::vector<double> &warm_start_row_dual, bool warm_start_valid,
                      double epsilon, double time_limit);

private:
    std::mutex mu_;
    Highs highs_;
    bool initialized_ = false;
    size_t nnz_lp_ = 0;
    HighsInt ncol_ = 0;
    HighsInt nrow_ = 0;
};
