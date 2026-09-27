#pragma once

// What the core tests share: the setup a caller owning its model does, a
// sink that keeps its best offer, and checks made against the `HighsLp`
// itself — its column-wise matrix, its own integrality, sense and offset —
// never against the view the workers searched, so a view that misread the
// model cannot vouch for its own answers.  Core headers only.

#include "deadline.h"
#include "fj_worker.h"
#include "heuristic_context.h"
#include "io/HighsIO.h"
#include "local_mip_worker.h"
#include "lp_data/HighsLp.h"
#include "solution_sink.h"
#include "worker_base.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace core_test {

// A budget that cannot bind, and a cap on the attempts, so a worker that
// never finds anything fails the test instead of spinning.
constexpr size_t kBudget = size_t{1} << 30;
constexpr size_t kAttemptBudget = size_t{1} << 16;
constexpr int kMaxAttempts = 2000;
constexpr double kFeasTol = 1e-6;
constexpr double kEpsilon = 1e-9;

// A sink that keeps the best offer it accepted, objective included: the
// objective is what the maximisation case checks.
class BestSink : public SolutionSink {
public:
    using SolutionSink::SolutionSink;

    std::mutex mtx;
    bool found = false;
    double objective = 0.0;
    std::vector<double> solution;

private:
    void on_accept(double obj, const std::vector<double>& x, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t /*effort_at*/) override {
        std::scoped_lock guard(mtx);
        if (!found || obj < objective) {
            found = true;
            objective = obj;
            solution = x;
        }
    }
};

// What a caller owning its model sets up: the storage its view points at,
// the view, and an execution context of its own — one worker, no deadline,
// no terminator.
struct Setup {
    ProblemStorage storage;
    ProblemView problem;
    ExecutionContext exec;

    Setup(const HighsLp& lp, const HighsLogOptions& log_options)
        : problem(make_view(lp, storage)),
          exec{.num_workers = 1,
               .base_seed = heuristic_base_seed(0),
               .deadline = Deadline{},
               .log_options = log_options,
               .terminator = {}} {}

    static ProblemView make_view(const HighsLp& lp, ProblemStorage& storage) {
        auto made = make_problem(lp, storage, kFeasTol, kEpsilon);
        REQUIRE(made.has_value());
        return std::move(*made);
    }
};

// Bounds, integrality and every row, from `lp`'s own column-wise matrix.
inline bool is_feasible(const HighsLp& lp, const std::vector<double>& x) {
    if (x.size() != static_cast<size_t>(lp.num_col_)) {
        return false;
    }
    std::vector<double> activity(static_cast<size_t>(lp.num_row_), 0.0);
    for (HighsInt j = 0; j < lp.num_col_; ++j) {
        if (x[j] < lp.col_lower_[j] - kFeasTol || x[j] > lp.col_upper_[j] + kFeasTol) {
            return false;
        }
        const bool integer =
            !lp.integrality_.empty() && lp.integrality_[j] != HighsVarType::kContinuous;
        if (integer && std::abs(x[j] - std::round(x[j])) > kFeasTol) {
            return false;
        }
        for (HighsInt k = lp.a_matrix_.start_[j]; k < lp.a_matrix_.start_[j + 1]; ++k) {
            activity[lp.a_matrix_.index_[k]] += lp.a_matrix_.value_[k] * x[j];
        }
    }
    for (HighsInt i = 0; i < lp.num_row_; ++i) {
        if (activity[i] < lp.row_lower_[i] - kFeasTol ||
            activity[i] > lp.row_upper_[i] + kFeasTol) {
            return false;
        }
    }
    return true;
}

// `lp`'s own objective, in its own sense.
inline double original_objective(const HighsLp& lp, const std::vector<double>& x) {
    double obj = lp.offset_;
    for (HighsInt j = 0; j < lp.num_col_; ++j) {
        obj += lp.col_cost_[j] * x[j];
    }
    return obj;
}

// Run `worker` until its sink holds a solution or the worker retires.
template <typename Worker>
void run_until_found(Worker& worker, const SolutionSink& sink) {
    for (int attempt = 0; attempt < kMaxAttempts && sink.accepted() == 0; ++attempt) {
        if (worker.finished()) {
            break;
        }
        static_cast<void>(worker.run_attempt(kAttemptBudget));
    }
}

// Both workers on `lp`, each into a fresh sink; returns their best offers.
struct Found {
    BestSink fj;
    BestSink local_mip;
    explicit Found(const std::vector<HighsVarType>& integrality)
        : fj(integrality, 0), local_mip(integrality, 0) {}
};

inline void run_both(const Setup& s, Found& found) {
    FjWorker fj(s.problem, s.exec, found.fj, kBudget, kBudget, /*seed=*/0, /*start=*/{},
                WorkerTrace{0, 0});
    run_until_found(fj, found.fj);
    local_mip_detail::LocalMipWorker local_mip(s.problem, s.exec, found.local_mip, kBudget, kBudget,
                                               /*seed=*/0,
                                               /*initial_solution=*/nullptr, WorkerTrace{0, 0});
    run_until_found(local_mip, found.local_mip);
}

// maximise 3 x0 + 2 x1 + 4 x2 + x3 + 7
//   s.t.   x0 + x1 + x2        <= 2
//          2 x0 + x2 - x3      <= 3
//          x1 + x3             >= 1
//          x0, x1, x2 in {0, 1}, x3 in [0, 2.5] continuous.
// Small enough to read by eye, sense and offset both non-trivial, one
// continuous column so the objective is not integral by construction.
inline HighsLp maximisation_model() {
    HighsLp lp;
    lp.num_col_ = 4;
    lp.num_row_ = 3;
    lp.sense_ = ObjSense::kMaximize;
    lp.offset_ = 7.0;
    lp.col_cost_ = {3.0, 2.0, 4.0, 1.0};
    lp.col_lower_ = {0.0, 0.0, 0.0, 0.0};
    lp.col_upper_ = {1.0, 1.0, 1.0, 2.5};
    lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger, HighsVarType::kInteger,
                       HighsVarType::kContinuous};
    lp.row_lower_ = {-kHighsInf, -kHighsInf, 1.0};
    lp.row_upper_ = {2.0, 3.0, kHighsInf};
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_ = {0, 2, 4, 6, 8};
    lp.a_matrix_.index_ = {0, 1, 0, 2, 0, 1, 1, 2};
    lp.a_matrix_.value_ = {1.0, 2.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0};
    return lp;
}

}  // namespace core_test
