#include "contested_pdlp.h"

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "pump_common.h"

#include <cassert>

namespace {

constexpr HighsInt kMinPdlpIterCap = 100;

}  // namespace

ContestedPdlp::ContestedPdlp(HighsMipSolver &mipsolver, HighsInt pdlp_iter_cap) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    ncol_ = model->num_col_;
    nrow_ = model->num_row_;
    nnz_lp_ = mipdata->ARindex_.size();
    if (ncol_ == 0 || nrow_ == 0 || nnz_lp_ == 0) {
        return;
    }

    auto lp = pump::build_lp_relaxation(*model, *mipdata);
    highs_.setOptionValue("solver", "pdlp");
    highs_.setOptionValue("output_flag", false);
    highs_.setOptionValue("pdlp_scaling", true);
    highs_.setOptionValue("pdlp_e_restart_method", 2);
    highs_.setOptionValue("pdlp_iteration_limit",
                          pdlp_iter_cap > kMinPdlpIterCap ? pdlp_iter_cap : kMinPdlpIterCap);
    highs_.passModel(std::move(lp));

    initialized_ = true;
}

ContestedPdlp::SolveResult ContestedPdlp::solve(const std::vector<double> &modified_cost,
                                                const std::vector<double> &warm_start_col_value,
                                                const std::vector<double> &warm_start_row_dual,
                                                bool warm_start_valid, double epsilon,
                                                double time_limit) {
    SolveResult result;
    if (!initialized_) {
        return result;
    }
    assert(static_cast<HighsInt>(modified_cost.size()) == ncol_);

    std::lock_guard<std::mutex> lock(mu_);

    highs_.changeColsCost(0, ncol_ - 1, modified_cost.data());
    highs_.setOptionValue("pdlp_optimality_tolerance", epsilon);
    highs_.setOptionValue("time_limit", time_limit);

    if (warm_start_valid && static_cast<HighsInt>(warm_start_col_value.size()) == ncol_ &&
        static_cast<HighsInt>(warm_start_row_dual.size()) == nrow_) {
        HighsSolution warm;
        warm.col_value = warm_start_col_value;
        warm.row_dual = warm_start_row_dual;
        warm.value_valid = true;
        warm.dual_valid = true;
        highs_.setSolution(warm);
    }

    result.status = highs_.run();
    result.model_status = highs_.getModelStatus();
    highs_.getInfoValue("pdlp_iteration_count", result.pdlp_iters);

    const auto &sol = highs_.getSolution();
    result.col_value = sol.col_value;
    result.row_dual = sol.row_dual;
    result.value_valid = sol.value_valid;
    result.dual_valid = sol.dual_valid;

    return result;
}
