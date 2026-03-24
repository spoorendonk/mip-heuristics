#include "scylla.h"

#include <cmath>
#include <random>
#include <vector>

#include "fpr_core.h"
#include "heuristic_common.h"
#include "solution_pool.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"

namespace scylla {

namespace {

constexpr int kPoolCapacity = 10;
constexpr int kMaxEpochs = 4;

// Noise scaling: proximity-weighted perturbation for workers 1..N-1
constexpr double kNoiseBase = 0.3;
constexpr double kNoiseProximityScale = 0.6;

void seed_pool(SolutionPool& pool, const HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  if (mipdata->incumbent.empty()) return;
  const HighsInt ncol = model->num_col_;
  double obj = model->offset_;
  for (HighsInt j = 0; j < ncol; ++j)
    obj += model->col_cost_[j] * mipdata->incumbent[j];
  pool.try_add(obj, mipdata->incumbent);
}

}  // namespace

void run(HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) return;

  // Guard: need an optimal LP relaxation
  auto lp_status = mipdata->lp.getStatus();
  if (!HighsLpRelaxation::scaledOptimal(lp_status)) return;

  const bool minimize = (model->sense_ == ObjSense::kMinimize);
  const int N = highs::parallel::num_threads();

  SolutionPool pool(kPoolCapacity, minimize);
  seed_pool(pool, mipsolver);

  const double deadline =
      heuristic_deadline(mipsolver.options_mip_->time_limit,
                         mipsolver.timer_.read());

  // Snapshot LP solution once (read-only for all workers)
  const auto& lp_sol = mipdata->lp.getLpSolver().getSolution().col_value;
  const auto& integrality = model->integrality_;
  const auto& col_lb = model->col_lower_;
  const auto& col_ub = model->col_upper_;

  // Build CSC once for all workers
  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                        mipdata->ARvalue_);

  // Pre-allocate per-worker scores to avoid repeated allocation in epoch loop
  std::vector<std::vector<double>> worker_scores(N);
  for (int w = 0; w < N; ++w) worker_scores[w].resize(ncol);

  for (int epoch = 0; epoch < kMaxEpochs; ++epoch) {
    if (mipsolver.timer_.read() >= deadline) break;
    if (mipdata->terminatorTerminated()) break;

    std::vector<HeuristicResult> results(N);
    highs::parallel::for_each(0, static_cast<HighsInt>(N),
        [&](HighsInt lo, HighsInt hi) {
          for (HighsInt w = lo; w < hi; ++w) {
            std::mt19937 rng(42 + epoch * N + static_cast<int>(w));

            // Compute scores: LP fractionality + per-worker noise
            auto& scores = worker_scores[w];
            for (HighsInt j = 0; j < ncol; ++j) {
              if (!is_integer(integrality, j)) {
                scores[j] = -1.0;
              } else {
                double s = std::abs(lp_sol[j] - std::round(lp_sol[j]));
                if (w > 0) {  // perturb for workers 1..N-1
                  double range = col_ub[j] - col_lb[j];
                  double proximity = (range > 0 && range < 1e8)
                      ? std::min(lp_sol[j] - col_lb[j],
                                 col_ub[j] - lp_sol[j]) / range
                      : 0.0;
                  double noise_scale =
                      kNoiseBase + kNoiseProximityScale * proximity;
                  s *= 1.0 + std::uniform_real_distribution<>(
                      -noise_scale, noise_scale)(rng);
                }
                scores[j] = s;
              }
            }

            FprConfig cfg{};
            cfg.max_attempts = 1;
            cfg.rng_seed_offset = 42 + epoch * N + static_cast<int>(w);
            cfg.hint = lp_sol.data();
            cfg.scores = scores.data();
            cfg.cont_fallback = lp_sol.data();
            cfg.csc = &csc;
            cfg.deadline = deadline;

            results[w] = fpr_attempt(mipsolver, cfg, rng, 0, nullptr);
          }
        }, 1);

    for (int w = 0; w < N; ++w)
      if (results[w].found_feasible)
        pool.try_add(results[w].objective, results[w].solution);
  }

  // Submit best solutions to solver
  for (auto& entry : pool.sorted_entries())
    mipdata->trySolution(entry.solution, kSolutionSourceScyllaFPR);
}

}  // namespace scylla
