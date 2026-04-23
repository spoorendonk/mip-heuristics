#include "local_mip.h"

#include "epoch_runner.h"
#include "heuristic_common.h"
#include "local_mip_construction.h"
#include "local_mip_worker.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "opportunistic_runner.h"
#include "parallel/HighsParallel.h"
#include "parallel_setup.h"
#include "rng.h"
#include "solution_pool.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace local_mip {

using local_mip_detail::construct_initial_solution;
using local_mip_detail::construction_effort_cap;
using local_mip_detail::LocalMipWorker;
using local_mip_detail::perturb_solution;

namespace {

// Row + integer feasibility check for a candidate solution. Used once
// per cold-construct branch so a feasible construction lands in the
// shared pool with the LocalMIP source tag; if infeasible the caller
// still uses it as the search's starting point (paper's intended
// behaviour).
bool is_solution_feasible(const HighsMipSolver &mipsolver, const std::vector<double> &solution) {
    const auto *model = mipsolver.model_;
    const auto *mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    const double feastol = mipdata->feastol;
    const double inttol = mipdata->epsilon;
    if (static_cast<HighsInt>(solution.size()) != ncol) {
        return false;
    }
    // Integer feasibility.
    for (HighsInt j = 0; j < ncol; ++j) {
        if (model->integrality_[j] == HighsVarType::kInteger ||
            model->integrality_[j] == HighsVarType::kImplicitInteger) {
            if (std::abs(solution[j] - std::round(solution[j])) > inttol) {
                return false;
            }
        }
        if (solution[j] < model->col_lower_[j] - feastol ||
            solution[j] > model->col_upper_[j] + feastol) {
            return false;
        }
    }
    // Row feasibility — walk ARstart/ARindex/ARvalue once.
    for (HighsInt i = 0; i < nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = mipdata->ARstart_[i]; k < mipdata->ARstart_[i + 1]; ++k) {
            lhs += mipdata->ARvalue_[k] * solution[mipdata->ARindex_[k]];
        }
        if (lhs < model->row_lower_[i] - feastol || lhs > model->row_upper_[i] + feastol) {
            return false;
        }
    }
    return true;
}

double compute_solution_objective(const HighsMipSolver &mipsolver,
                                  const std::vector<double> &solution) {
    const auto *model = mipsolver.model_;
    double obj = model->offset_;
    for (HighsInt j = 0; j < model->num_col_; ++j) {
        obj += model->col_cost_[j] * solution[j];
    }
    return obj;
}

}  // namespace

HeuristicResult worker(HighsMipSolver &mipsolver, const CscMatrix &csc, uint32_t seed,
                       const double *initial_solution, size_t max_effort) {
    const HighsInt ncol = mipsolver.model_->num_col_;
    const HighsInt nrow = mipsolver.model_->num_row_;

    HeuristicResult result;
    if (ncol == 0 || nrow == 0) {
        return result;
    }

    const bool minimize = (mipsolver.model_->sense_ == ObjSense::kMinimize);
    SolutionPool pool(1, minimize);

    // Disable stale_budget: pass max_effort so it can never fire before
    // the total budget is exhausted (original worker() had no stale_budget).
    LocalMipWorker w(mipsolver, csc, pool, max_effort, seed, initial_solution,
                     /*stale_budget=*/max_effort);

    size_t total_effort = 0;
    while (!w.finished()) {
        auto epoch = w.run_epoch(max_effort);
        total_effort += epoch.effort;
        if (epoch.effort == 0) {
            break;
        }
    }

    result.effort = total_effort;
    auto entries = pool.sorted_entries();
    if (!entries.empty()) {
        result.found_feasible = true;
        result.objective = entries[0].objective;
        result.solution = std::move(entries[0].solution);
    }

    return result;
}

namespace {

// Resolve the starting point for a worker with the paper's cold-start
// fallback (issue #75):
//
//   1. Prefer the pool's best if one exists (an earlier heuristic in
//      the same presolve chain or another worker may have landed one).
//   2. Else prefer `mipdata->incumbent` if non-empty (warm start).
//   3. Else run the paper's construction phase
//      (`construct_initial_solution`), capped at
//      `construction_effort_cap(max_effort)`, with a per-worker
//      seeded RNG so cold-start diversity matches the existing
//      perturbation-based diversity of workers 1..N-1.
//
// The callers read the returned vector via `.data()` and feed it
// straight into `LocalMipWorker`'s `initial_solution` pointer — the
// worker's constructor clamps and rounds defensively, so the
// construction result being mildly infeasible is fine.
//
// NOTE: The pool-first branch is the cold/warm boundary that issue
// #74 is expected to further refine (pool-aware warm-start).  This
// file's contract with #74: return the pool's best when
// `snap.has_solution` is true, never fall through to incumbent in
// that case.  That matches #75's out-of-scope note.
std::vector<double> resolve_worker_start(HighsMipSolver &mipsolver, const CscMatrix &csc,
                                         SolutionPool &pool, size_t max_effort, uint32_t seed) {
    // Single pool lookup: `sorted_entries` returns best-first, so the
    // non-empty test replaces a separate `snapshot()` call (which
    // would otherwise take the pool's lock twice in a row for no
    // gain).  Reviewers flagged the double-lock across R1/R2/R3.
    auto entries = pool.sorted_entries();
    if (!entries.empty()) {
        return std::move(entries[0].solution);
    }
    auto *mipdata = mipsolver.mipdata_.get();
    if (!mipdata->incumbent.empty()) {
        return mipdata->incumbent;
    }
    // Cold start: neither the pool nor the incumbent has a solution.
    // Run the paper's construction phase.
    Rng rng(seed);
    std::vector<double> constructed;
    construct_initial_solution(mipsolver, csc, rng, construction_effort_cap(max_effort),
                               constructed);
    // If the construction happens to land on a feasible integer point,
    // publish it to the shared pool so downstream heuristics (and
    // HiGHS's own incumbent path) pick it up.  Tag as `LocalMIP`
    // (source char 'M') — we don't mint a new `Construction` source
    // tag because that would require an upstream HiGHS patch.
    // Infeasible constructions are the paper's intended input to the
    // search phase and are not inserted.
    if (!constructed.empty() && is_solution_feasible(mipsolver, constructed)) {
        double obj = compute_solution_objective(mipsolver, constructed);
        pool.try_add(obj, constructed, kSolutionSourceLocalMIP);
    }
    return constructed;
}

// Build a fresh per-worker starting solution for the deterministic
// epoch loop: worker 0 gets the unperturbed start; workers 1..N-1
// get the start + perturbation.  Cold-start workers all get
// independently-constructed starts (different seeds) so the
// perturbation step is a no-op in spirit but we still apply it to
// stay on the existing worker-diversity path.
std::vector<double> build_starting_solution_for_worker(HighsMipSolver &mipsolver,
                                                       const ParallelSetup &setup,
                                                       SolutionPool &pool, size_t w,
                                                       uint32_t seed) {
    std::vector<double> start =
        resolve_worker_start(mipsolver, setup.csc, pool, setup.worker_budget, seed);
    if (w == 0) {
        return start;
    }
    // Derive a distinct perturbation seed so it doesn't reproduce the
    // RNG trajectory the construction already consumed when cold
    // starting.  `0x9E3779B9u` is the golden-ratio 32-bit constant
    // (same trick `boost::hash_combine` uses).  Flagged by R1/R2.
    Rng perturb_rng(seed ^ 0x9E3779B9u);
    perturb_solution(start, *setup.mipdata, setup.model.integrality_, setup.model.col_lower_,
                     setup.model.col_upper_, setup.model.num_col_, perturb_rng);
    return start;
}

void run_parallel_deterministic(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);
    const HighsInt ncol = setup.model.num_col_;

    // Create per-worker LocalMipWorker instances.  Starting points are
    // resolved via the paper's cold-start fallback chain (pool → incumbent
    // → construction), then perturbed for workers 1..N-1 to mirror the
    // original warm-start diversity path.
    std::vector<std::unique_ptr<LocalMipWorker>> workers;
    workers.reserve(setup.N);

    for (size_t w = 0; w < setup.N; ++w) {
        uint32_t seed = setup.base_seed + static_cast<uint32_t>(w) * kSeedStride;
        std::vector<double> start =
            build_starting_solution_for_worker(mipsolver, setup, pool, w, seed);
        workers.push_back(std::make_unique<LocalMipWorker>(
            mipsolver, setup.csc, pool, setup.worker_budget, seed, start.data()));
    }

    // Track restart seed counter for deterministic restart seeding.
    uint32_t restart_seed_counter = static_cast<uint32_t>(setup.N);

    size_t total_effort = run_epoch_loop(
        mipsolver, workers, max_effort, setup.epoch_budget(kEpochsPerWorker),
        [&](int w) {
            // Restart stalled worker: prefer pool restart, fall back to
            // incumbent, fall back again to a fresh construction with a
            // new seed (cold-start branch of issue #75 also covers
            // restart-after-exhaustion on stubbornly-infeasible
            // instances).
            uint32_t new_seed =
                setup.base_seed + static_cast<uint32_t>(restart_seed_counter++) * kSeedStride;

            std::vector<double> restart_sol;
            Rng restart_rng(new_seed);
            if (!pool.get_restart(restart_rng, restart_sol)) {
                if (!setup.mipdata->incumbent.empty()) {
                    restart_sol = setup.mipdata->incumbent;
                } else {
                    // Cold restart: rebuild via construction.
                    Rng construct_rng(new_seed);
                    construct_initial_solution(mipsolver, setup.csc, construct_rng,
                                               construction_effort_cap(setup.worker_budget),
                                               restart_sol);
                }
            }
            perturb_solution(restart_sol, *setup.mipdata, setup.model.integrality_,
                             setup.model.col_lower_, setup.model.col_upper_, ncol, restart_rng);
            workers[w] = std::make_unique<LocalMipWorker>(
                mipsolver, setup.csc, pool, setup.worker_budget, new_seed, restart_sol.data());
        },
        setup.stale_budget);

    setup.mipdata->heuristic_effort_used += total_effort;
}

void run_parallel_opportunistic(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort) {
    ParallelSetup setup(mipsolver, max_effort);
    const HighsInt ncol = setup.model.num_col_;

    struct LmState {
        std::unique_ptr<LocalMipWorker> worker;
    };

    size_t total_effort = run_opportunistic_loop(
        mipsolver, static_cast<int>(setup.N), max_effort, setup.stale_budget, setup.default_run_cap,
        setup.base_seed,
        [&](int worker_idx, Rng &rng) -> LmState {
            uint32_t seed = static_cast<uint32_t>(rng());
            std::vector<double> start =
                resolve_worker_start(mipsolver, setup.csc, pool, setup.worker_budget, seed);
            if (worker_idx != 0) {
                perturb_solution(start, *setup.mipdata, setup.model.integrality_,
                                 setup.model.col_lower_, setup.model.col_upper_, ncol, rng);
            }
            return LmState{std::make_unique<LocalMipWorker>(
                mipsolver, setup.csc, pool, setup.worker_budget, seed, start.data())};
        },
        [&](LmState &state, Rng &rng, size_t run_cap) -> HeuristicResult {
            if (!state.worker || state.worker->finished()) {
                // Restart from pool, incumbent, or fresh construction
                // (cold-start), with fresh perturbation.
                std::vector<double> restart_sol;
                if (!pool.get_restart(rng, restart_sol)) {
                    if (!setup.mipdata->incumbent.empty()) {
                        restart_sol = setup.mipdata->incumbent;
                    } else {
                        uint32_t cseed = static_cast<uint32_t>(rng());
                        Rng construct_rng(cseed);
                        construct_initial_solution(mipsolver, setup.csc, construct_rng,
                                                   construction_effort_cap(setup.worker_budget),
                                                   restart_sol);
                    }
                }
                perturb_solution(restart_sol, *setup.mipdata, setup.model.integrality_,
                                 setup.model.col_lower_, setup.model.col_upper_, ncol, rng);
                uint32_t seed = static_cast<uint32_t>(rng());
                state.worker = std::make_unique<LocalMipWorker>(
                    mipsolver, setup.csc, pool, setup.worker_budget, seed, restart_sol.data());
            }
            auto epoch = state.worker->run_epoch(run_cap);
            HeuristicResult result;
            result.effort = epoch.effort;
            if (epoch.found_improvement) {
                result.found_feasible = true;
                result.objective = pool.snapshot().best_objective;
            }
            return result;
        });

    setup.mipdata->heuristic_effort_used += total_effort;
}

}  // namespace

void run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort,
                  bool opportunistic) {
    const auto *model = mipsolver.model_;
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return;
    }
    // Issue #75: the old `mipdata->incumbent.empty()` early-return is
    // gone.  Cold-start is now handled by `resolve_worker_start` which
    // runs the paper's construction phase when neither pool nor
    // incumbent has a solution.  The sibling issue #74 handles the
    // warm-start-with-pool path; this function stays neutral on that
    // (pool-first lookup in `resolve_worker_start` already covers it).

    if (opportunistic) {
        run_parallel_opportunistic(mipsolver, pool, max_effort);
    } else {
        run_parallel_deterministic(mipsolver, pool, max_effort);
    }
}

}  // namespace local_mip
