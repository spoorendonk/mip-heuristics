// Construction phase for Local-MIP cold start (issue #75).
//
// Paper reference: Lin, Zou, Cai — "An Efficient Local Search Solver for
// Mixed Integer Programming", Proc. CP 2024, Article 19.  Algorithm 1
// Line 1 specifies the starting assignment as "all variables are set to
// the value closest to 0 within their global bounds".  The public
// reference implementation
// (https://github.com/shaowei-cai-group/Local-MIP) confirms this in
// `src/local_search/start/start.cpp`: `zero_start` clamps each variable
// to the nearest point in [lb, ub] to zero (lb if lb>0, ub if ub<0,
// else 0).  The accompanying `random_start` overlays a uniform integer
// draw on top.
//
// On top of that minimal zero-start this implementation adds the
// per-variable greedy refinement requested by the issue: after the
// zero-start, iterate the variables in a randomised
// constraint-coverage-weighted order and, for each variable,
// consider a small candidate set of domain points (lb, ub, zero, tight
// deltas from currently violated rows, current value) and pick the one
// that minimises total weighted row violation with a feasibility-first
// rule.  The feasibility-first rule: if a candidate drops the number
// of violated rows containing the variable, prefer it over candidates
// that merely reduce the violation magnitude.  Weights are uniform
// here, matching the paper which only activates its dynamic weighting
// scheme once the search loop is in a local optimum (§4.1) — before
// the first step, all row weights are 1.
//
// The sweep is capped by `kConstructionEffortFraction` of the outer
// `max_effort` budget (10% — small enough to leave the bulk of the
// budget for the search phase that actually produces improvements;
// large enough that the greedy pass can complete on typical MIPLIB
// instance sizes, each variable costing O(col_nnz) coefficient
// accesses).

#include "local_mip_construction.h"

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "rng.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>  // std::swap
#include <vector>

namespace local_mip_detail {

namespace {

// Clamp a value to [lb, ub] and round to integer if needed.
double clamp_round_var(double v, double lb, double ub, bool integer) {
    return clamp_round(v, lb, ub, integer);
}

// Initial zero-start (Algorithm 1 Line 1 / reference impl zero_start).
// Returns the value closest to 0 within each variable's global bounds.
double zero_start_value(double lb, double ub, bool integer) {
    double v;
    if (lb > 0.0) {
        v = lb;
    } else if (ub < 0.0) {
        v = ub;
    } else {
        v = 0.0;
    }
    return clamp_round_var(v, lb, ub, integer);
}

// Compute all row LHS from `solution` and fill `lhs`.  Charges
// `effort` proportional to the nnz (matches LHS computation cost).
void compute_all_lhs(HighsInt nrow, const std::vector<HighsInt> &ARstart,
                     const std::vector<HighsInt> &ARindex, const std::vector<double> &ARvalue,
                     const std::vector<double> &solution, std::vector<double> &lhs,
                     size_t &effort) {
    for (HighsInt i = 0; i < nrow; ++i) {
        double l = 0.0;
        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
            l += ARvalue[k] * solution[ARindex[k]];
        }
        lhs[i] = l;
    }
    effort += ARindex.size();
}

// For a single variable `j`, evaluate the effect of flipping it to
// `new_val`: returns (delta_total_viol, num_newly_violated_rows,
// num_newly_satisfied_rows) for the subset of rows that contain `j`.
// Does not mutate `lhs`.
struct CandidateEffect {
    double delta_viol;              // signed: negative = improves total weighted violation
    HighsInt rows_newly_violated;   // rows containing j that transition sat→viol
    HighsInt rows_newly_satisfied;  // rows containing j that transition viol→sat
};

CandidateEffect evaluate_move(HighsInt j, double old_val, double new_val, const CscMatrix &csc,
                              const std::vector<double> &lhs, const std::vector<double> &row_lo,
                              const std::vector<double> &row_hi, double feastol, size_t &effort) {
    CandidateEffect eff{0.0, 0, 0};
    double delta = new_val - old_val;
    if (std::abs(delta) < 1e-15) {
        return eff;
    }
    effort += csc.col_start[j + 1] - csc.col_start[j];
    for (HighsInt p = csc.col_start[j]; p < csc.col_start[j + 1]; ++p) {
        HighsInt i = csc.col_row[p];
        double old_lhs = lhs[i];
        double new_lhs = old_lhs + csc.col_val[p] * delta;
        double old_v = row_violation(old_lhs, row_lo[i], row_hi[i]);
        double new_v = row_violation(new_lhs, row_lo[i], row_hi[i]);
        eff.delta_viol += (new_v - old_v);
        bool was_viol = is_row_violated(old_lhs, row_lo[i], row_hi[i], feastol);
        bool now_viol = is_row_violated(new_lhs, row_lo[i], row_hi[i], feastol);
        if (was_viol && !now_viol) {
            ++eff.rows_newly_satisfied;
        } else if (!was_viol && now_viol) {
            ++eff.rows_newly_violated;
        }
    }
    return eff;
}

// Apply the chosen move to both `solution` and `lhs`, charging effort.
void apply_move_inplace(HighsInt j, double new_val, std::vector<double> &solution,
                        std::vector<double> &lhs, const CscMatrix &csc, size_t &effort) {
    double old_val = solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < 1e-15) {
        return;
    }
    effort += csc.col_start[j + 1] - csc.col_start[j];
    solution[j] = new_val;
    for (HighsInt p = csc.col_start[j]; p < csc.col_start[j + 1]; ++p) {
        HighsInt i = csc.col_row[p];
        lhs[i] += csc.col_val[p] * delta;
    }
}

// Compute a tight-move delta for row `i`, variable `j`, coefficient
// `coeff` that would satisfy row `i` from the current `lhs[i]`.
// Mirrors `WorkerCtx::compute_tight_delta` but as a free function so we
// don't need to build a WorkerCtx just for construction.
double tight_delta_for_row(HighsInt i, HighsInt j, double coeff, const std::vector<double> &lhs,
                           const std::vector<double> &row_lo, const std::vector<double> &row_hi,
                           const std::vector<double> &col_lb, const std::vector<double> &col_ub,
                           const std::vector<double> &solution, double feastol, bool integer) {
    if (std::abs(coeff) < 1e-15) {
        return 0.0;
    }
    double l = lhs[i];
    double gap;
    if (l > row_hi[i] + feastol) {
        gap = l - row_hi[i];
    } else if (l < row_lo[i] - feastol) {
        gap = l - row_lo[i];
    } else {
        return 0.0;  // already satisfied
    }
    double delta = -gap / coeff;
    if (integer) {
        // Round in the direction that *reaches* the violated bound,
        // i.e. away from zero (ceil for positive delta, floor for
        // negative).  The previous coeff>0 → floor / coeff<0 → ceil
        // rule was wrong for `lhs < row_lo` violations with positive
        // coeff: there `delta > 0` and we need to push lhs upward,
        // which requires `ceil(delta)`, not `floor`.  R1-15 round-3.
        delta = (delta > 0) ? std::ceil(delta) : std::floor(delta);
    }
    double new_val = solution[j] + delta;
    if (new_val < col_lb[j]) {
        delta = col_lb[j] - solution[j];
    } else if (new_val > col_ub[j]) {
        delta = col_ub[j] - solution[j];
    }
    return delta;
}

// Build a variable order weighted by constraint coverage (number of
// rows each variable appears in), with ties broken by the supplied
// RNG.  Matches the issue's "constraint-coverage-weighted order for
// warmth" hint: variables that hit many constraints get first pick so
// their choice informs the per-variable decisions of narrower
// variables.
//
// IMPORTANT — divergence from the paper.  Lin, Zou, Cai (CP 2024) and
// the public reference implementation
// (https://github.com/shaowei-cai-group/Local-MIP,
// src/local_search/start/start.cpp) specify only the trivial
// `zero_start` for the construction — every variable set to the value
// closest to 0 within bounds, no greedy sweep, no variable ordering.
// The coverage-weighted ordering + per-variable greedy refinement
// below is our engineering extension, motivated by issue #75's
// "greedy variable-ordering + per-variable minimise-weighted-violation"
// prose (which is richer than the paper actually prescribes).
// Round-2 reviewers (R1, R2, R3) flagged the gap; leaving the sweep
// active because it empirically produces a less-infeasible starting
// point than bare zero-start on the instances we've tested, but the
// ordering key is not load-bearing and can be replaced by a plain
// Fisher-Yates shuffle if a bench ever shows it matters.  Tracked for
// validation against the 4 Local-MIP new-record instances
// (`genus-sym-g31-8`, `genus-sym-g62-2`, `genus-g61-25`,
// `neos-4232544-orira`) per issue #75's acceptance criterion.
// Hoisted scratch buffers for `weighted_order`.  Per-thread to keep
// concurrent worker construction calls from racing.  R1-12 round-3
// review: the previous version allocated `order` and `tiebreak` on
// every call; on instances with thousands of cold-restarts this
// dominated the construction profile.  Resized in-place each call so
// memory stays bounded by the largest model the thread has constructed
// against.
struct WeightedOrderBuffers {
    std::vector<HighsInt> order;
    std::vector<uint32_t> tiebreak;
};

// thread_local lifetime trade-off (R3-10 round-4 review): the buffers
// persist for the thread's lifetime — fine for the current single-MIP-
// per-process deployment (HiGHS solves one model per `Highs::run()`,
// and the thread pool is short-lived).  Capacity is bounded by the
// largest model the thread has constructed against.  If this code is
// ever embedded in a long-running service that solves many models with
// different sizes from the same persistent worker thread, revisit:
// the high-water-mark capacity may stick around for the life of the
// service.  The thread_local was deliberately chosen over per-call
// allocation to avoid re-allocating on every cold-start (R1-12 round-3
// review's profile showed allocation dominating on instances with
// many cold restarts).
WeightedOrderBuffers &weighted_order_buffers() {
    thread_local WeightedOrderBuffers buffers;
    return buffers;
}

// Returns a non-owning view into the thread_local `WeightedOrderBuffers`
// owned by `weighted_order_buffers()`.  R2-7 round-4 review: we used to
// return `const std::vector<HighsInt>&`, which silently relied on the
// caller not invoking any sibling helper that touches the same
// thread_local before the view goes out of scope.  Returning
// `std::span` makes the lifetime contract visible in the type — a
// future maintainer who wants to call this twice in a row, or
// recursively from inside `construct_initial_solution`, will see the
// view aliasing as part of the signature.  The span's payload is still
// the same `WeightedOrderBuffers::order` vector; the view stays valid
// until the next call to `weighted_order_buffers()` on the same thread.
std::span<const HighsInt> weighted_order(const CscMatrix &csc, HighsInt ncol, Rng &rng) {
    auto &buffers = weighted_order_buffers();
    buffers.order.resize(ncol);
    buffers.tiebreak.resize(ncol);
    for (HighsInt j = 0; j < ncol; ++j) {
        buffers.order[j] = j;
        buffers.tiebreak[j] = static_cast<uint32_t>(rng());
    }
    // Primary key: descending col-nnz (high-coverage variables first).
    // Secondary key: a per-variable RNG draw, so two workers with
    // different seeds see different orderings even when many
    // variables share the same col-nnz value.  Using stable_sort
    // with a post-shuffle did *not* actually diversify across
    // workers — the shuffle's randomness is overwritten by the
    // deterministic col-nnz comparison for any two variables with
    // different nnz.  Review R2 flagged this.
    const auto &tiebreak = buffers.tiebreak;
    std::sort(buffers.order.begin(), buffers.order.end(), [&](HighsInt a, HighsInt b) {
        auto nnz_a = csc.col_start[a + 1] - csc.col_start[a];
        auto nnz_b = csc.col_start[b + 1] - csc.col_start[b];
        if (nnz_a != nnz_b) {
            return nnz_a > nnz_b;
        }
        return tiebreak[a] < tiebreak[b];
    });
    return std::span<const HighsInt>(buffers.order);
}

}  // namespace

size_t construct_initial_solution(const ConstructionInputs &inputs, Rng &rng, size_t max_effort,
                                  std::vector<double> &out_solution) {
    const HighsInt ncol = inputs.ncol;
    const HighsInt nrow = inputs.nrow;

    out_solution.assign(ncol, 0.0);
    if (ncol == 0) {
        return 0;
    }

    const auto &col_lb = *inputs.col_lb;
    const auto &col_ub = *inputs.col_ub;
    const auto &row_lo = *inputs.row_lo;
    const auto &row_hi = *inputs.row_hi;
    const auto &integrality = *inputs.integrality;
    const auto &ARstart = *inputs.ARstart;
    const auto &ARindex = *inputs.ARindex;
    const auto &ARvalue = *inputs.ARvalue;
    const CscMatrix &csc = *inputs.csc;
    const double feastol = inputs.feastol;

    // --- Phase A: zero-start (paper Alg 1 Line 1) -----------------------
    for (HighsInt j = 0; j < ncol; ++j) {
        out_solution[j] = zero_start_value(col_lb[j], col_ub[j], is_integer(integrality, j));
    }

    // Unit decision (R2-2 round-4 review): the rest of this function
    // accounts effort in *coefficient-access* (nnz) units, matching
    // `WorkerCtx::effort` and the system-wide `mode_dispatch::kWeight*`
    // calibration.  Phase A is a single column-write loop (`ncol` writes,
    // not nnz), so charging `+= ncol` on the normal path mixes units.
    // We therefore drop the unconditional charge and only book a small
    // `ncol` charge on the early-exit branch (nrow == 0 or
    // max_effort == 0) so callers still see a non-zero effort
    // representing the real wall-time spent (however small) when Phase B
    // doesn't run.  The early-exit value is technically still in mixed
    // units, but it's bounded by `ncol`, fires only on degenerate
    // inputs, and the alternative (returning 0) would underreport work.
    if (nrow == 0 || max_effort == 0) {
        return static_cast<size_t>(ncol);
    }

    // Compute initial lhs[] vector; this is the anchor the greedy sweep
    // mutates incrementally.  Also acts as the effort "upfront" cost
    // (one full pass over nnz) against the construction cap.
    std::vector<double> lhs(nrow, 0.0);
    size_t effort = 0;
    compute_all_lhs(nrow, ARstart, ARindex, ARvalue, out_solution, lhs, effort);

    // --- Phase B: greedy variable sweep --------------------------------
    // Iterate variables in constraint-coverage-weighted order.  For each
    // variable, build a small candidate set and pick the value that
    // minimises weighted row violation, with a feasibility-first
    // tie-break favouring moves that reduce the count of violated rows
    // containing the variable (paper §3.2 mtm operator on violated
    // constraints).  Uniform weights (1) during construction — dynamic
    // weighting only kicks in later inside the search loop (paper §4.1).
    std::span<const HighsInt> order = weighted_order(csc, ncol, rng);

    // Small cap on candidates per variable — bounded so one variable
    // sweep stays O(col_nnz) coefficient accesses.
    constexpr HighsInt kMaxTightPerVar = 4;

    // Per-variable candidate buffer.
    std::vector<double> candidates;
    candidates.reserve(8);

    for (HighsInt j : order) {
        if (effort >= max_effort) {
            break;
        }
        bool integer = is_integer(integrality, j);
        double old_val = out_solution[j];

        candidates.clear();
        // Always consider the current value as a "no-op" baseline.
        candidates.push_back(old_val);

        // Bounds (finite only; clamp_round_var handles integer rounding).
        if (col_lb[j] > -kHighsInf) {
            candidates.push_back(clamp_round_var(col_lb[j], col_lb[j], col_ub[j], integer));
        }
        if (col_ub[j] < kHighsInf) {
            candidates.push_back(clamp_round_var(col_ub[j], col_lb[j], col_ub[j], integer));
        }
        // Zero-clamp baseline.
        candidates.push_back(zero_start_value(col_lb[j], col_ub[j], integer));

        // Tight deltas from up to kMaxTightPerVar rows containing j that
        // are currently violated: each row gives the mtm operator's
        // "make this row tight" candidate (paper Def 4 / §3.2).
        HighsInt tight_added = 0;
        for (HighsInt p = csc.col_start[j];
             p < csc.col_start[j + 1] && tight_added < kMaxTightPerVar; ++p) {
            HighsInt i = csc.col_row[p];
            if (!is_row_violated(lhs[i], row_lo[i], row_hi[i], feastol)) {
                continue;
            }
            double delta = tight_delta_for_row(i, j, csc.col_val[p], lhs, row_lo, row_hi, col_lb,
                                               col_ub, out_solution, feastol, integer);
            if (std::abs(delta) < 1e-15) {
                continue;
            }
            double cand = clamp_round_var(old_val + delta, col_lb[j], col_ub[j], integer);
            candidates.push_back(cand);
            ++tight_added;
        }

        // Evaluate each candidate.  Best = lowest (delta_viol), with a
        // feasibility-first rule: prefer candidates with strictly more
        // rows_newly_satisfied than rows_newly_violated over candidates
        // that merely reduce delta_viol.
        double best_new_val = old_val;
        double best_delta_viol = 0.0;
        HighsInt best_net_feas = 0;  // newly_satisfied - newly_violated

        for (double cand : candidates) {
            if (std::abs(cand - old_val) < 1e-15) {
                continue;
            }
            CandidateEffect eff =
                evaluate_move(j, old_val, cand, csc, lhs, row_lo, row_hi, feastol, effort);
            HighsInt net_feas = eff.rows_newly_satisfied - eff.rows_newly_violated;

            // Feasibility-first: a strictly larger net_feas always wins,
            // regardless of delta_viol sign.  Within the same net_feas,
            // pick the smaller delta_viol.
            if (net_feas > best_net_feas ||
                (net_feas == best_net_feas && eff.delta_viol < best_delta_viol - 1e-12)) {
                best_net_feas = net_feas;
                best_delta_viol = eff.delta_viol;
                best_new_val = cand;
            }
            if (effort >= max_effort) {
                break;
            }
        }

        if (std::abs(best_new_val - old_val) > 1e-15) {
            apply_move_inplace(j, best_new_val, out_solution, lhs, csc, effort);
        }
    }

    // Defensive final clamp — clamp_round_var + evaluate_move respect
    // bounds but rounding drift on pathological coefficients could push
    // a continuous variable ~feastol outside.  Cheap to enforce.
    for (HighsInt j = 0; j < ncol; ++j) {
        double v = out_solution[j];
        if (v < col_lb[j]) {
            v = col_lb[j];
        } else if (v > col_ub[j]) {
            v = col_ub[j];
        }
        if (is_integer(integrality, j)) {
            v = std::round(v);
            if (v < col_lb[j]) {
                v = std::ceil(col_lb[j]);
            }
            if (v > col_ub[j]) {
                v = std::floor(col_ub[j]);
            }
        }
        out_solution[j] = v;
    }

    // Total effort = Phase B greedy sweep only.  Phase A's column-write
    // loop (~ncol cheap writes) is dropped from the normal-path total
    // for unit consistency: `effort` here is the coefficient-access
    // signal LocalMIP's search loop uses (`WorkerCtx::effort`), and that
    // is the same signal `mode_dispatch::kWeight*` is calibrated against.
    // See the comment near the early-exit return above for the unit
    // rationale.
    return effort;
}

// HighsMipSolver& thin wrapper: assemble ConstructionInputs from the
// solver's model + mipdata and delegate.
size_t construct_initial_solution(HighsMipSolver &mipsolver, const CscMatrix &csc, Rng &rng,
                                  size_t max_effort, std::vector<double> &out_solution) {
    const auto *model = mipsolver.model_;
    auto *mipdata = mipsolver.mipdata_.get();
    ConstructionInputs inputs;
    inputs.ncol = model->num_col_;
    inputs.nrow = model->num_row_;
    inputs.ARstart = &mipdata->ARstart_;
    inputs.ARindex = &mipdata->ARindex_;
    inputs.ARvalue = &mipdata->ARvalue_;
    inputs.col_lb = &model->col_lower_;
    inputs.col_ub = &model->col_upper_;
    inputs.row_lo = &model->row_lower_;
    inputs.row_hi = &model->row_upper_;
    inputs.integrality = &model->integrality_;
    inputs.csc = &csc;
    inputs.feastol = mipdata->feastol;
    return construct_initial_solution(inputs, rng, max_effort, out_solution);
}

}  // namespace local_mip_detail
