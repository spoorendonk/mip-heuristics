#include "fpr_lp_refs.h"

#include "Highs.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

// ===================================================================
// LP reference solutions
// ===================================================================

namespace {

// `setOptionValue` whose failure is not silent.  Every `Highs` instance we
// build sets `output_flag=false`, so a rejected write — an option renamed
// by a HiGHS tag bump, a value out of domain — reports itself only through
// the return status nobody was reading here.  One of the four writes below
// is `time_limit`, i.e. the whole reason this solve is bounded at all, so
// losing it silently is the failure this helper exists to prevent.
//
// Deliberately a second copy of `contested_pdlp.cpp`'s helper rather than a
// shared one: it is nine lines, the message names its own subsystem, and a
// common header for it would be a dependency between two translation units
// that otherwise share nothing.
template <typename T>
void set_option_or_die(Highs& highs, const char* name, T value) {
    if (highs.setOptionValue(name, value) != HighsStatus::kOk) {
        std::fprintf(stderr,
                     "fpr_lp reference LP: HiGHS rejected option '%s' (unknown name or invalid "
                     "value). This is a build-time bug, not a solve failure.\n",
                     name);
        // `assert` is a no-op under `NDEBUG`, which is every Release build,
        // and a silently unset `time_limit` is exactly what this guards —
        // so abort unconditionally, as the ContestedPdlp helper does.  Every
        // call site passes a compile-time-constant name, and the one
        // runtime-valued write is provably in domain (`time_limit` is
        // `min(30, remaining)` with `remaining > 0` checked just above).
        assert(false && "fpr_lp reference LP: unknown or invalid HiGHS option");
        std::abort();
    }
}

// Solve an LP relaxation of the presolved MIP model.
// use_ipm: barrier solver (analytic center); otherwise simplex (vertex).
// run_crossover: false disables crossover (for analytic center).
// use_objective: true uses model cost; false uses zero objective.
// lp_iterations: incremented by the LP iterations this solve consumed
// (simplex + IPM + crossover counts summed — a deliberate simple proxy;
// an IPM iteration costs more than a simplex one, but the counts are
// small and only feed the shared B&B heuristic-budget accounting).
std::vector<double> solve_lp_relaxation(const HighsMipSolver& mipsolver, bool use_ipm,
                                        bool run_crossover, bool use_objective,
                                        const Deadline& deadline, int64_t& lp_iterations) {
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
    // The scalar comes from the shared `Deadline` the caller threads in
    // (issue #118); it used to be `time_limit - timer_.read()` inlined here,
    // which is the same clock read and the same subtraction, but left this
    // file with a second spelling of the deadline and no way for the caller
    // to tell an expiry here from a failed solve.  `remaining()` is
    // `kHighsInf` when the solve carries no limit, which the cap below
    // collapses to 30 s exactly as before.
    const double remaining = deadline.remaining();
    if (remaining <= 0.0) {
        return {};
    }
    Highs highs;
    set_option_or_die(highs, "output_flag", false);
    set_option_or_die(highs, "time_limit", std::min(30.0, remaining));
    if (use_ipm) {
        set_option_or_die(highs, "solver", "ipm");
    }
    if (!run_crossover) {
        set_option_or_die(highs, "run_crossover", "off");
    }

    highs.passModel(std::move(lp));
    highs.run();

    const auto& info = highs.getInfo();
    lp_iterations += static_cast<int64_t>(info.simplex_iteration_count) +
                     static_cast<int64_t>(info.ipm_iteration_count) +
                     static_cast<int64_t>(info.crossover_iteration_count);

    const auto& sol = highs.getSolution();
    if (std::cmp_equal(sol.col_value.size(), ncol)) {
        return sol.col_value;
    }
    return {};
}

}  // namespace

std::vector<double> compute_analytic_center(const HighsMipSolver& mipsolver, bool use_objective,
                                            const Deadline& deadline, int64_t& lp_iterations) {
    return solve_lp_relaxation(mipsolver, /*use_ipm=*/true,
                               /*run_crossover=*/false, use_objective, deadline, lp_iterations);
}

std::vector<double> compute_zero_obj_vertex(const HighsMipSolver& mipsolver,
                                            const Deadline& deadline, int64_t& lp_iterations) {
    return solve_lp_relaxation(mipsolver, /*use_ipm=*/false,
                               /*run_crossover=*/true, /*use_objective=*/false, deadline,
                               lp_iterations);
}
