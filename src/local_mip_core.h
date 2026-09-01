#pragma once

#include "heuristic_common.h"
#include "local_mip_caches.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace local_mip_detail {

// --- The mixed tight move's integer rounding rule (issue #123) ---
//
// Paper reference: Lin, Zou, Cai — "An Efficient Local Search Solver for
// Mixed Integer Programming", Proc. CP 2024, Article 19,
// doi:10.4230/LIPIcs.CP.2024.19.  The mixed tight move (Def 4) assigns a
// variable the threshold value that makes one row tight; Eq 5 rounds the
// resulting shift for an integer variable.
//
// The rule is stated in terms of the *row's* state, not the coefficient's
// sign:
//
//   * row currently violated  -> round the shift **away from zero**, so it
//     crosses the bound it is aiming at and leaves the row satisfied;
//   * row currently satisfied -> round the shift **toward zero**, so it
//     stops short of the bound it is aiming at and leaves the row satisfied.
//
// Both call sites used to collapse this to the coefficient's sign
// (`coeff > 0 -> floor`, `coeff < 0 -> ceil`).  That collapse is valid only
// for the paper's one-sided `A_i x <= b_i` form, where the sign of the
// coefficient determines the sign of the shift.  HiGHS rows are two-sided,
// and the collapse is wrong on both of the branches where it can disagree:
//
//   * violated, `lhs < row_lo` with `coeff > 0`: `gap < 0` gives
//     `delta > 0`, and `floor` undershoots.  Row `x1 + x2 >= 3` at
//     `lhs = 0.5` gives `delta = 2.5`; `floor` -> 2 leaves `lhs = 2.5`,
//     still violated, while `ceil` -> 3 leaves `lhs = 3.5`.
//   * satisfied, when the *nearest* bound is the lower one: there the
//     sign rule rounds away from zero and can push the row out of
//     feasibility.  Same row at `lhs = 5.5` with `coeff = 1` picks
//     `gap = 2.5`, so `delta = -2.5`; `floor` -> -3 leaves `lhs = 2.5`,
//     now violated, while `ceil` -> -2 leaves `lhs = 3.5`.
//
// Equality rows are the documented exception and do not use this helper —
// see `WorkerCtx::compute_tight_delta`.
[[nodiscard]] inline double round_tight_delta(double delta, bool row_violated) {
    if (row_violated) {
        return (delta > 0) ? std::ceil(delta) : std::floor(delta);
    }
    return (delta > 0) ? std::floor(delta) : std::ceil(delta);
}

// --- WorkerCtx: central context for the local search worker ---
struct WorkerCtx {
    // Model refs
    const HighsLp* model;
    const std::vector<HighsInt>& ar_start;
    const std::vector<HighsInt>& ar_index;
    const std::vector<double>& ar_value;
    const std::vector<double>& col_lb;
    const std::vector<double>& col_ub;
    const std::vector<double>& col_cost;
    const std::vector<double>& row_lo;
    const std::vector<double>& row_hi;
    const std::vector<HighsVarType>& integrality;
    const CscMatrix& csc;
    const double feastol;
    const double epsilon;
    const bool minimize;
    const HighsInt ncol;
    const HighsInt nrow;
    // Dispatch-time `isBinary` snapshot (`ProblemView::binary`), at least
    // `ncol` entries.  Never re-read the live root domain from here: a
    // peer's accepted solution propagates it while this worker runs
    // (issue #99).
    const uint8_t* binary;

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

    WorkerCtx(HighsMipSolver& mipsolver, const CscMatrix& csc_, const uint8_t* binary_);

    [[nodiscard]] bool is_int(HighsInt j) const { return ::is_integer(integrality, j); }

    [[nodiscard]] bool is_binary(HighsInt j) const { return binary[j] != 0; }

    [[nodiscard]] double clamp_and_round(HighsInt j, double val) const {
        return clamp_round(val, col_lb[j], col_ub[j], is_int(j));
    }

    [[nodiscard]] double compute_violation(HighsInt i, double l) const {
        return row_violation(l, row_lo[i], row_hi[i]);
    }

    [[nodiscard]] bool is_violated(HighsInt i, double l) const {
        return l > row_hi[i] + feastol || l < row_lo[i] - feastol;
    }

    [[nodiscard]] bool is_equality(HighsInt i) const {
        return row_lo[i] == row_hi[i] && row_lo[i] > -kHighsInf;
    }

    [[nodiscard]] bool is_tabu(HighsInt j, double delta, HighsInt step) const {
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

    void apply_move_with_tabu(HighsInt j, double new_val, HighsInt step, Rng& rng);

    // Recompute all LHS from scratch and check feasibility.
    // update_sets: rebuild violated/satisfied partition from scratch.
    // early_exit:  return false on first violation without full scan.
    // Always recomputes lhs[] and charges effort.
    bool full_recheck(bool update_sets, bool early_exit);

    void rebuild_state();

    [[nodiscard]] double compute_tight_delta(HighsInt i, HighsInt j, double coeff) const;

    // Paper Section 4.1: weighting scheme for MIP.
    // Called when at a local optimum (no positive operation found).
    void update_weights(Rng& rng, bool is_feasible, bool best_feasible, double best_obj);

    // Reset constraint and objective weights to their initial state
    // (`w(obj) = 1`, `w(coni) = 1` per Lin, Zou, Cai §4.1 init).
    // Used after a random-walk perturbation: a fresh restart point
    // logically warrants a clean weighting state, otherwise the worker
    // re-explores the same direction the existing weights bias it
    // toward.  Engineering choice — paper §4.1 prescribes only
    // initialization and the PAWS-style update at local optima; it is
    // silent on weight handling at perturbation/restart (the paper's
    // Algorithm 1 has no such step).  See the call site in
    // `local_mip_worker.cpp` for the full justification.  R1-9 round-3
    // review motivated the reset; R2-8 round-4 review flagged that the
    // paper-citation framing was not grounded in the paper.
    void reset_weights() {
        std::ranges::fill(weight, uint64_t{1});
        obj_weight = 1;
    }
};

// --- Candidate selection / scoring ---

// Paper Definitions 5-10: two-level scoring function.
// Progress score (level 1): discrete constraint-transition scores + objective.
// Bonus score (level 2): breakthrough bonus + robustness bonus.
std::pair<double, double> compute_candidate_scores(WorkerCtx& ctx, HighsInt j, double new_val,
                                                   bool best_feasible, double best_obj);

bool is_aspiration(const WorkerCtx& ctx, HighsInt j, double new_val, double best_obj,
                   bool best_feasible);

double compute_breakthrough_delta(const WorkerCtx& ctx, HighsInt j, double cur_obj,
                                  double best_obj);

Candidate select_best_from_batch(WorkerCtx& ctx, std::vector<BatchCand>& batch, HighsInt step,
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
Candidate infeasible_step(WorkerCtx& ctx, Rng& rng, HighsInt step, bool best_feasible,
                          double best_objective, const std::vector<HighsInt>& costed_vars,
                          const std::vector<HighsInt>& binary_vars);

}  // namespace local_mip_detail
