#pragma once

// Shared by the objective-cut tests of the heuristics that propagate the cut
// as a row (#232 FPR, #233 Scylla).

#include "heuristic_context.h"
#include "solution_sink.h"

#include <cmath>
#include <mutex>
#include <vector>

inline constexpr ObjectiveCutParams kCutOn{.on = true, .rel = 1e-3, .abs = 1e-7};
// The objective row is checked to `mip_feasibility_tolerance` on its
// activity, as every row is.
inline constexpr double kRowTol = 1e-6;

// Every accepted offer, in order.
class RecordingSink : public SolutionSink {
public:
    using SolutionSink::SolutionSink;
    std::mutex mtx;
    std::vector<double> objectives;
    std::vector<std::vector<double>> solutions;

private:
    void on_accept(double obj, const std::vector<double>& x, int /*source*/,
                   const WorkerTrace& /*trace*/, size_t /*effort_at*/) override {
        std::scoped_lock guard(mtx);
        objectives.push_back(obj);
        solutions.push_back(x);
    }
};

// `x` with every column whose cost is positive raised to its upper bound,
// when the result stays feasible: a point far worse than `x`, so a cutoff
// below it leaves room for a find.  Empty when it is not feasible (on
// egout, a fixed-charge network, raising a charge keeps it so).
inline std::vector<double> inflate(const ProblemView& problem, std::vector<double> x) {
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        if ((*problem.col_cost)[j] > 0.0 && std::isfinite((*problem.col_upper)[j])) {
            x[j] = (*problem.col_upper)[j];
        }
    }
    for (HighsInt i = 0; i < problem.nrow; ++i) {
        double act = 0.0;
        for (HighsInt k = (*problem.ar_start)[i]; k < (*problem.ar_start)[i + 1]; ++k) {
            act += (*problem.ar_value)[k] * x[(*problem.ar_index)[k]];
        }
        if (act > (*problem.row_upper)[i] + kRowTol || act < (*problem.row_lower)[i] - kRowTol) {
            return {};
        }
    }
    return x;
}
