#pragma once

#include "heuristic_common.h"
#include "local_mip_caches.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace local_mip_detail {

// --- WorkerCtx: central context for the local search worker ---
struct WorkerCtx {
    // Model refs
    const HighsLp *model;
    const std::vector<HighsInt> &ARstart;
    const std::vector<HighsInt> &ARindex;
    const std::vector<double> &ARvalue;
    const std::vector<double> &col_lb;
    const std::vector<double> &col_ub;
    const std::vector<double> &col_cost;
    const std::vector<double> &row_lo;
    const std::vector<double> &row_hi;
    const std::vector<HighsVarType> &integrality;
    const CscMatrix &csc;
    const double feastol;
    const double epsilon;
    const bool minimize;
    const HighsInt ncol;
    const HighsInt nrow;
    HighsMipSolverData *mipdata;

    // Mutable state
    std::vector<double> solution;
    std::vector<double> lhs;
    std::vector<uint64_t> weight;
    uint64_t obj_weight = 1;
    double current_obj = 0.0;

    // Sub-structures
    IndexedSet violated;
    IndexedSet satisfied;
    ViolCache viol_cache;
    LiftCache lift;

    // Tabu
    std::vector<HighsInt> tabu_inc_until;
    std::vector<HighsInt> tabu_dec_until;

    // Reusable buffers
    std::vector<BatchCand> batch;
    std::vector<WeightedCon> sampled;

    // Feasibility tracking
    bool was_infeasible = true;
    HighsInt feasible_recheck_counter = 0;

    // Effort tracking (coefficient accesses)
    size_t effort = 0;

    WorkerCtx(HighsMipSolver &mipsolver, const CscMatrix &csc_);

    bool is_int(HighsInt j) const { return ::is_integer(integrality, j); }

    double clamp_and_round(HighsInt j, double val) const {
        return clamp_round(val, col_lb[j], col_ub[j], is_int(j));
    }

    double compute_violation(HighsInt i, double l) const {
        return row_violation(l, row_lo[i], row_hi[i]);
    }

    bool is_violated(HighsInt i, double l) const {
        return l > row_hi[i] + feastol || l < row_lo[i] - feastol;
    }

    bool is_equality(HighsInt i) const { return row_lo[i] == row_hi[i] && row_lo[i] > -kHighsInf; }

    bool is_tabu(HighsInt j, double delta, HighsInt step) const {
        if (delta > 0 && step < tabu_inc_until[j]) {
            return true;
        }
        if (delta < 0 && step < tabu_dec_until[j]) {
            return true;
        }
        return false;
    }

    void update_violated(HighsInt i);

    void apply_move(HighsInt j, double new_val);

    void apply_move_with_tabu(HighsInt j, double new_val, HighsInt step, Rng &rng);

    // Recompute all LHS from scratch and check feasibility.
    // update_sets: rebuild violated/satisfied partition from scratch.
    // early_exit:  return false on first violation without full scan.
    // Always recomputes lhs[] and charges effort.
    bool full_recheck(bool update_sets, bool early_exit);

    void rebuild_state();

    double compute_tight_delta(HighsInt i, HighsInt j, double coeff) const;

    // Paper Section 4.1: weighting scheme for MIP.
    // Called when at a local optimum (no positive operation found).
    void update_weights(Rng &rng, bool is_feasible, bool best_feasible, double best_obj);
};

// --- Candidate selection / scoring ---

// Paper Definitions 5-10: two-level scoring function.
// Progress score (level 1): discrete constraint-transition scores + objective.
// Bonus score (level 2): breakthrough bonus + robustness bonus.
std::pair<double, double> compute_candidate_scores(WorkerCtx &ctx, HighsInt j, double new_val,
                                                   bool best_feasible, double best_obj);

bool is_aspiration(const WorkerCtx &ctx, HighsInt j, double new_val, double best_obj,
                   bool best_feasible);

double compute_breakthrough_delta(const WorkerCtx &ctx, HighsInt j, double cur_obj,
                                  double best_obj);

Candidate select_best_from_batch(WorkerCtx &ctx, std::vector<BatchCand> &batch, HighsInt step,
                                 bool aspiration, double best_obj, bool best_feasible);

// --- infeasible_step: candidate generation following paper's Algorithm 2 ---
//
// Phase ordering (Algorithm 2):
// 1. MTM in violated (+ BM if post-feasible)
// 2. MTM in satisfied constraints (Alg 2 lines 7-8)
// 3. Boolean flips (Alg 2 lines 9-11)
// 4. Weight update + random constraint fallback (Alg 2 lines 12-14)
// Additional (our engineering additions):
// 5. Perturbation (generalizes Boolean flip to non-binary)
// 6. Easy moves
Candidate infeasible_step(WorkerCtx &ctx, Rng &rng, HighsInt step, bool best_feasible,
                          double best_objective, const std::vector<HighsInt> &costed_vars,
                          const std::vector<HighsInt> &binary_vars);

}  // namespace local_mip_detail
