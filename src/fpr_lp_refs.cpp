#include "fpr_lp_refs.h"

#include "Highs.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <utility>
#include <vector>

// ===================================================================
// LP reference solutions
// ===================================================================

namespace {

// Solve an LP relaxation of the presolved MIP model.
// use_ipm: barrier solver (analytic center); otherwise simplex (vertex).
// run_crossover: false disables crossover (for analytic center).
// use_objective: true uses model cost; false uses zero objective.
std::vector<double> solve_lp_relaxation(const HighsMipSolver& mipsolver, bool use_ipm,
                                        bool run_crossover, bool use_objective) {
    const auto* model = mipsolver.model_;
    const auto& mipdata = *mipsolver.mipdata_;
    const HighsInt ncol = model->num_col_;

    HighsLp lp;
    lp.num_col_ = ncol;
    lp.num_row_ = model->num_row_;
    lp.col_lower_ = model->col_lower_;
    lp.col_upper_ = model->col_upper_;
    lp.row_lower_ = model->row_lower_;
    lp.row_upper_ = model->row_upper_;
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.num_col_ = ncol;
    lp.a_matrix_.num_row_ = model->num_row_;
    lp.a_matrix_.start_ = mipdata.ARstart_;
    lp.a_matrix_.index_ = mipdata.ARindex_;
    lp.a_matrix_.value_ = mipdata.ARvalue_;

    if (use_objective) {
        lp.col_cost_ = model->col_cost_;
        lp.sense_ = model->sense_;
        lp.offset_ = model->offset_;
    } else {
        lp.col_cost_.assign(ncol, 0.0);
    }

    // Respect the outer MIP time limit: never exceed what remains, cap at 30s.
    // HiGHS treats `time_limit == 0.0` as "no limit" (the guard in Highs.cpp is
    // `time_limit > 0 && time_limit < kHighsInf`), so when we have already
    // blown past the outer deadline we must short-circuit before constructing
    // `Highs`; otherwise we would accidentally disable the cap and let the
    // analytic-center LP run unbounded.
    const double outer_limit = mipsolver.options_mip_->time_limit;
    const double remaining = outer_limit - mipsolver.timer_.read();
    if (remaining <= 0.0) {
        return {};
    }
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("time_limit", std::min(30.0, remaining));
    if (use_ipm) {
        highs.setOptionValue("solver", "ipm");
    }
    if (!run_crossover) {
        highs.setOptionValue("run_crossover", "off");
    }

    highs.passModel(std::move(lp));
    highs.run();

    const auto& sol = highs.getSolution();
    if (static_cast<HighsInt>(sol.col_value.size()) == ncol) {
        return sol.col_value;
    }
    return {};
}

}  // namespace

std::vector<double> compute_analytic_center(const HighsMipSolver& mipsolver, bool use_objective) {
    return solve_lp_relaxation(mipsolver, /*use_ipm=*/true,
                               /*run_crossover=*/false, use_objective);
}

std::vector<double> compute_zero_obj_vertex(const HighsMipSolver& mipsolver) {
    return solve_lp_relaxation(mipsolver, /*use_ipm=*/false,
                               /*run_crossover=*/true, /*use_objective=*/false);
}
