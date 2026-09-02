// Candidate scoring and the Algorithm-2 `infeasible_step` driver.
//
// Declared in local_mip_core.h alongside WorkerCtx / LiftCache.  Split
// from local_mip_core.cpp to keep each translation unit under ~500 LoC
// (issue #66 acceptance criterion).

#include "local_mip_caches.h"
#include "local_mip_core.h"
#include "lp_data/HConst.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace local_mip_detail {

namespace {

void append_candidate(WorkerCtx& ctx, std::vector<BatchCand>& batch, HighsInt j, double delta) {
    double new_val = ctx.clamp_and_round(j, ctx.solution[j] + delta);
    if (std::abs(new_val - ctx.solution[j]) < kEpsZero) {
        return;
    }
    batch.push_back({j, new_val});
}

}  // namespace

// Paper Definitions 5-10: two-level scoring function.
// Progress score (level 1): discrete constraint-transition scores + objective.
// Bonus score (level 2): breakthrough bonus + robustness bonus.
// Cognitive complexity 41 (threshold 25).  Kept whole: Defs 6-7 and 9-10 evaluated in a single pass
// over the column, which is the point — separate passes would re-walk the CSC column. Decomposing
// it would move work across a worker's inner loop, and the closeout takes no unmeasured performance
// risk; the standards also rank fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::pair<double, double> compute_candidate_scores(WorkerCtx& ctx, HighsInt j, double new_val,
                                                   bool best_feasible, double best_obj) {
    double old_val = ctx.solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < kEpsZero) {
        return {-std::numeric_limits<double>::infinity(), 0.0};
    }

    ctx.effort += ctx.csc.col_start[j + 1] - ctx.csc.col_start[j];

    // Def 5: progress score for objective
    double obj_delta = ctx.col_cost[j] * delta;
    double new_obj = ctx.current_obj + obj_delta;
    double eps = ctx.epsilon;
    double progress = 0.0;
    if ((!ctx.minimize && new_obj > ctx.current_obj + eps) ||
        (ctx.minimize && new_obj < ctx.current_obj - eps)) {
        progress += static_cast<double>(ctx.obj_weight);  // objective improved
    } else if ((!ctx.minimize && new_obj < ctx.current_obj - eps) ||
               (ctx.minimize && new_obj > ctx.current_obj + eps)) {
        progress -= static_cast<double>(ctx.obj_weight);  // objective worsened
    }

    // Def 8: breakthrough bonus (beats best-found solution)
    double bonus = 0.0;
    if (best_feasible) {
        bool beats_best = ctx.minimize ? (new_obj < best_obj - eps) : (new_obj > best_obj + eps);
        if (beats_best) {
            bonus += static_cast<double>(ctx.obj_weight);
        }
    }

    // Defs 6-7, 9-10: constraint progress + robustness
    for (HighsInt p = ctx.csc.col_start[j]; p < ctx.csc.col_start[j + 1]; ++p) {
        HighsInt i = ctx.csc.col_row[p];
        double coeff = ctx.csc.col_val[p];
        double old_lhs = ctx.lhs[i];
        double new_lhs = old_lhs + (coeff * delta);
        double old_viol = ctx.viol_cache.get_or_compute(i, old_lhs, ctx.row_lo[i], ctx.row_hi[i]);
        double new_viol = ctx.compute_violation(i, new_lhs);
        auto w = static_cast<double>(ctx.weight[i]);

        // Def 6: constraint progress score.
        //
        // DELIBERATE DEVIATION FROM THE PAPER, MATCHING THE AUTHORS' CODE:
        // Def 6 awards +/- w(con_i)/2 for a constraint that stays violated
        // but improves/worsens, reserving full weight for a satisfy/violate
        // transition.  We award full weight in both cases, as the authors'
        // own implementation does (github.com/shaowei-cai-group/Local-MIP,
        // `explore_unsat.cpp`: `if (new_gap < pre_gap) score += con_weight;
        // else score -= con_weight;`).  Consequence either way: "satisfy a
        // constraint" and "reduce a violation" score identically, flattening
        // the preference Def 6 states.  Do NOT "fix" this toward the paper
        // without measuring — it would move us away from the reference
        // implementation.  (The reference also scales equality rows by 2x,
        // which we do not do; that part is a genuine gap.)
        // `feastol`, not `kScoreTol` or `kViolDeltaTol` (issue #148):
        // this is the same "is this row violated?" question
        // `WorkerCtx::is_violated` and the `violated`/`satisfied` sets
        // answer, just evaluated on a hypothetical (pre-move / post-move)
        // lhs rather than the worker's live state — it has to use the
        // one tolerance that governs violation everywhere else, or this
        // scoring pass would reintroduce the same kind of mismatch #148
        // fixed elsewhere.
        bool was_viol = (old_viol > ctx.feastol);
        bool now_viol = (new_viol > ctx.feastol);
        if (was_viol && !now_viol) {
            progress += w;  // violated → satisfied
        } else if (!was_viol && now_viol) {
            progress -= w;  // satisfied → violated
        } else if (was_viol && now_viol) {
            // `kViolDeltaTol`, not `kScoreTol` and not `feastol`: this
            // asks whether the violation *magnitude* moved by more than
            // floating-point noise, not whether the row is violated
            // (already decided above) and not whether a candidate
            // *score* changed meaningfully — a violation magnitude is a
            // row-activity quantity, in the same units as `feastol`,
            // and has no relationship to the score comparisons below.
            // See `kViolDeltaTol`'s definition in `local_mip_caches.h`.
            if (new_viol < old_viol - kViolDeltaTol) {
                progress += w;  // still violated, improved
            } else if (new_viol > old_viol + kViolDeltaTol) {
                progress -= w;  // still violated, worsened
            }
        }

        // Def 9: robustness bonus — only for transitions into strictly
        // satisfied (was violated or tight, now strictly interior).
        //
        // DELIBERATE DEVIATION FROM THE PAPER, MATCHING THE AUTHORS' CODE:
        // Def 9 awards w(con_i) for *every* constraint that ends strictly
        // satisfied.  Summed over all rows and cancelling the rows this
        // column does not touch, the paper's per-column form is
        // sum_i w_i * ([new strict] - [old strict]) — i.e. it also
        // *penalises* losing strict satisfaction.  We implement only the
        // positive term.  The authors' implementation is likewise
        // transition-shaped (`if (!pre_sat && now_sat) score +=
        // scaled_con_weight * 2`).  Same caution as Def 6 above.
        if (!now_viol) {
            bool old_strict =
                !was_viol &&
                (ctx.row_hi[i] >= kHighsInf || old_lhs < ctx.row_hi[i] - ctx.feastol) &&
                (ctx.row_lo[i] <= -kHighsInf || old_lhs > ctx.row_lo[i] + ctx.feastol);
            if (!old_strict) {
                bool new_strict =
                    (ctx.row_hi[i] >= kHighsInf || new_lhs < ctx.row_hi[i] - ctx.feastol) &&
                    (ctx.row_lo[i] <= -kHighsInf || new_lhs > ctx.row_lo[i] + ctx.feastol);
                if (new_strict) {
                    bonus += w;
                }
            }
        }
    }

    return {progress, bonus};
}

bool is_aspiration(const WorkerCtx& ctx, HighsInt j, double new_val, double best_obj,
                   bool best_feasible) {
    if (!best_feasible) {
        return false;
    }
    double delta = new_val - ctx.solution[j];
    double obj_delta = ctx.col_cost[j] * delta;
    double new_obj = ctx.current_obj + obj_delta;
    return ctx.minimize ? (new_obj < best_obj - ctx.epsilon) : (new_obj > best_obj + ctx.epsilon);
}

// Paper Definition 2: `Delta_j = (obj(s*) - obj(s) - eps) / c_j` -- the
// shift that moves x_j alone to exactly `eps` past the best found
// objective, `eps` being, in the paper's own words, "for making the
// objective value strictly better". The reference implementation
// (github.com/shaowei-cai-group/Local-MIP, `explore_unsat.cpp`) does the
// same thing by setting the objective pseudo-constraint's RHS to
// `m_best_obj - m_opt_tolerance` before solving for the delta.
//
// This used to omit `eps` entirely, computing only the delta that
// reaches `obj(s*)` exactly (issue #129 cold review). At a feasible
// local optimum `obj(s) == obj(s*)` -- the current solution IS the best
// found -- so the omitted-eps version computed `delta == 0`, literally
// at the state Algorithm 2 lines 5-6 exist to escape. The `is_int`
// rounding below (away from zero: floor a negative delta, ceil a
// positive one) papered over this for an INTEGER variable away from the
// tie, since rounding away from zero happens to overshoot into
// strictly-better territory -- but never for a delta that started at
// exactly zero, and never for a continuous variable, which this
// function never rounds at all. `ctx.epsilon` is the same margin
// `compute_candidate_scores`'s `beats_best`, `is_aspiration`, and
// `LocalMipWorker::run_attempt`'s own `improved` check already use for
// "strictly better than the best found", so it is reused here rather
// than introducing a second tolerance for the same question.
//
// What this changes for a CONTINUOUS costed variable sitting at the tie
// (`obj(s) == obj(s*)`): `delta = -eps/c_j` is never rounded, so it
// stays at `ctx.epsilon`'s own scale (`mipdata->epsilon`, HiGHS's
// `small_matrix_value`, 1e-9). `append_candidate` keeps it (`kEpsZero`
// is 1e-15), `compute_candidate_scores` scores it ~0 -- it lands exactly
// at the eps margin it was built to just clear -- and it reaches Phase
// 4's final `if (cand.var_idx != -1) return cand;` unbeaten, since
// nothing else scores strictly better. On a continuous-costed model
// that DISPLACES the Phase 5/6 diversification move that used to be
// returned there instead: a fall-through move at 1e-9 is closer to a
// rounding artifact than a search step. This is Definition 2 exactly as
// written, not a defect -- do not add a guard or a magnitude threshold
// to suppress it, which would be an unreviewed heuristic approximation
// this codebase does not otherwise make -- but it is a measured
// behaviour change, not a neutral one. Isolated by reverting only the
// `+ ctx.epsilon` below with a fall-through-move probe still in place,
// counting moves with `|delta| < 1e-6`: rgn 6/284 -> 16,711/16,986
// (98.4%), egout 28/554 -> 384/808 (47.5%), dcmulti 2/27 -> 7/52, bell5
// 0/476 -> 3/492. Full-solve A/B across the bundled set (suite=local_mip,
// presolve_only, threads=1, seed 0, 15s) between the commit before this
// fix and the one after: 6 better / 7 worse / 92 unchanged -- better on
// 3015, dcmulti, gt2, issue-2173, p01, rgn (129.3 -> 114.2, the instance
// most saturated with these nudges), worse on egout, gesa2, issue-2095,
// lseu, sp150x300d, and marginally on the two OBJSENSE MAX instances.
//
// Two corrections to how this fix was first reported (issue #129 cold
// review, round 2): the escape this restores is *returned* by Phase 4's
// own unconditional `if (cand.var_idx != -1) return cand;` fallback
// check, not by Phase 1's early return -- its score is 0, not
// `> kScoreTol` -- even though it originates in Phase 1b below and
// survives Phases 2-4 unbeaten; and it is not returned on *every*
// feasible-mode entry, only most of them -- one entry in an 8-entry
// trace (`cur=2 best=3`) still correctly returned from Phase 6, because
// Definition 2's precondition just below did not hold there.
//
// Definition 2 also states the operator only for a solution `s` with
// `obj(s) >= obj(s*)`, and the first guard below is that precondition
// (issue #150; the #129 cold review that added the `+ eps` filed it
// separately rather than fold a second behaviour change in, so this
// function was unguarded until then -- the paragraph here used to say
// so).  The formula is a move *toward* `obj(s*)`, so evaluated from a
// solution that already beats the best found it points backwards:
// `obj_gap < 0` flips `delta`'s sign, and the operator offers an
// objective-worsening candidate under the label of a breakthrough. That
// state is reachable -- during an infeasible episode the current
// solution may beat the incumbent while violating rows.
//
// The authors' implementation spells the same precondition at the call
// site rather than in the operator
// (github.com/shaowei-cai-group/Local-MIP): both of its breakthrough
// loops, in `explore_unsat.cpp` and `explore_unsat_random.cpp`, are
// gated on `m_is_found_feasible && !m_current_obj_breakthrough`, and
// `Local_Search.cpp` maintains `m_current_obj_breakthrough` as
// `obj(s) <= m_best_obj - m_opt_tolerance` -- current strictly better
// than best found, the same question `ctx.epsilon` asks here.  Paper
// and reference agree, so this is Definition 2 implemented, not a
// deviation recorded.  It sits in the operator, not at Phase 1b's loop,
// because Definition 2 attaches it to `bm(x_j, s)` itself and because
// Phase 4 has already once considered re-scoring breakthrough
// candidates from a second call site; a `return 0.0` is equivalent to
// skipping the loop, since `append_candidate` drops a zero delta.
//
// The comparison is the same strict form `is_aspiration` and
// `compute_candidate_scores`'s `beats_best` use for "strictly better
// than the best found", rather than a second spelling of it.  It
// differs from the reference's `<=` only at `cur_obj == best_obj - eps`
// exactly, where the unguarded formula yields `delta == 0` and
// `append_candidate` drops the candidate anyway -- so the two are
// behaviourally identical, not merely close.
double compute_breakthrough_delta(const WorkerCtx& ctx, HighsInt j, double cur_obj,
                                  double best_obj) {
    bool already_breaks_through =
        ctx.minimize ? (cur_obj < best_obj - ctx.epsilon) : (cur_obj > best_obj + ctx.epsilon);
    if (already_breaks_through) {
        return 0.0;
    }

    double obj_coeff = ctx.col_cost[j];
    if (std::abs(obj_coeff) < kEpsZero) {
        return 0.0;
    }

    double obj_gap = cur_obj - best_obj;
    if (!ctx.minimize) {
        // Computes the delta in the objective-*worsening* direction
        // (unverified against the paper) and is dead in this
        // integration -- HiGHS normalizes every model to minimization,
        // so `ctx.minimize` was `false` on none of the 105 bundled
        // instances, the two OBJSENSE MAX ones included. Before the
        // `+ ctx.epsilon` fix below, a wrong sign here was harmless
        // whenever it landed on `delta == 0`; with `eps` added it now
        // produces a small non-zero move in the wrong direction
        // instead of a no-op, should this branch ever go live.
        obj_gap = -obj_gap;
    }

    // `+ ctx.epsilon`: paper Definition 2's own eps, dropped before this
    // fix -- see the comment above, including its continuous-variable
    // and trajectory consequences.
    double delta = -(obj_gap + ctx.epsilon) / obj_coeff;

    if (ctx.is_int(j)) {
        delta = (obj_coeff > 0) ? std::floor(delta) : std::ceil(delta);
    }
    double new_val = ctx.solution[j] + delta;
    if (new_val < ctx.col_lb[j] || new_val > ctx.col_ub[j]) {
        delta =
            (obj_coeff > 0) ? (ctx.col_lb[j] - ctx.solution[j]) : (ctx.col_ub[j] - ctx.solution[j]);
    }
    return delta;
}

Candidate select_best_from_batch(WorkerCtx& ctx, std::vector<BatchCand>& batch, HighsInt step,
                                 bool aspiration, double best_obj, bool best_feasible) {
    Candidate best;
    for (const auto& c : batch) {
        double delta = c.new_val - ctx.solution[c.var_idx];
        if (std::abs(delta) < kEpsZero) {
            continue;
        }

        if (ctx.is_tabu(c.var_idx, delta, step)) {
            if (!(aspiration &&
                  is_aspiration(ctx, c.var_idx, c.new_val, best_obj, best_feasible))) {
                continue;
            }
        }

        auto [prog, bon] =
            compute_candidate_scores(ctx, c.var_idx, c.new_val, best_feasible, best_obj);

        // `kScoreTol`, every use from here down in this file: comparing
        // two candidate scores for a numerically meaningful difference,
        // not a violation question — see its definition in
        // `local_mip_caches.h` (issue #148).
        if (prog > best.score + kScoreTol) {
            best = {c.var_idx, c.new_val, prog, bon};
        } else if (prog > best.score - kScoreTol) {
            if (bon > best.bonus) {
                best = {c.var_idx, c.new_val, prog, bon};
            }
        }
    }
    ctx.viol_cache.reset();
    return best;
}

// --- Lift move selection (paper Algorithm 1 line 5) ---
//
// Returns the highest-scoring entry of `LiftCache`'s positive-score list
// -- the feasibility-preserving objective improvement the paper calls the
// lift move process -- compacting the list's lazily-removed entries on
// the way.  Extracted verbatim out of `LocalMipWorker::run_attempt`
// (issue #149) so that the tabu behaviour documented below is pinnable by
// a unit test rather than only reachable through a full attempt loop.
//
// DELIBERATE DEVIATION FROM THE PAPER, MATCHING THE AUTHORS' CODE:
// this consults no tabu list, even though `apply_move_with_tabu` -- the
// only applier of what it returns -- sets one.  The paper states the
// forbidding strategy in general terms (Sect. 5: "Once a variable is
// modified, it forbids the modification for the reverse direction in the
// following tt iterations") without scoping it to a phase; the authors'
// implementation (github.com/shaowei-cai-group/Local-MIP) resolves that
// silence unambiguously.  `Neighbor::tabu` / `Neighbor::tabu_latest` is
// consulted in every one of its five neighbourhood explorers
// (`explore_unsat.cpp`, `explore_unsat_random.cpp`, `explore_sat.cpp`,
// `explore_flip.cpp`, `explore_easy.cpp`) and in no part of the lift path
// (`lift_move.cpp`'s `lift_move()` / `lift_move_operation()`,
// `lift_scoring.cpp`'s `score_lift` / `lift_age` / `lift_random`) --
// while their shared `Local_Search::apply_move`, which both paths call,
// sets the reverse-direction tabu for every move it applies, lift moves
// included.  The lift phase writes the tabu lists and never reads them:
// an asymmetry inside one solver, not an omission in a code path that
// forgot the lists existed.  The reference's own anti-cycling device
// here is a different one -- `lift_age` breaks equal-score ties toward
// the variable whose last modification is oldest.
//
// The consequence is the one issue #149 recorded, and it is pinned by
// `tests/test_local_mip.cpp` ("LocalMIP: the lift phase applies a move
// the tabu list forbids (#149)"): a neighbourhood move on column j
// immediately followed by a lift move reversing it is not prevented.
// It cannot repeat indefinitely, because only one half of the
// alternation ignores the lists.  The reversing lift move itself goes
// through `apply_move_with_tabu`, which sets the opposite-direction
// tabu on j, and the exploration side does check
// (`select_best_from_batch`'s `ctx.is_tabu`), so it will not re-apply
// the move it just made for the next `kTabuBase + rand(kTabuVar)`
// steps -- unless the aspiration criterion fires, which requires the
// move to strictly beat the best-found objective and is therefore
// wanted.  The cycle costs one round trip and is then broken at the
// half that checks.
//
// Do not "fix" this by adding an `is_tabu` filter without re-reading the
// two reference files named above: a lift move is objective-improving
// and feasibility-preserving by construction
// (`LiftCache::recompute_one` lists only columns whose best in-domain
// target strictly improves the objective), so a tabu filter here
// suppresses guaranteed progress, which is not a change this project
// makes to a reference algorithm on its own judgement.
Candidate select_lift_move(WorkerCtx& ctx) {
    Candidate best;
    best.score = 0.0;

    HighsInt write = 0;
    // Compaction, not traversal: the body writes back into the same
    // vector at `write <= read` and never resizes it (the `resize`
    // below is what shortens the list), so the bound is
    // loop-invariant.  Hoisting it drops a size() load per iteration
    // of LocalMIP's per-restart loop and keeps modernize-loop-convert
    // from proposing a range-for that would hide the rewrite.
    const auto n_positive = static_cast<HighsInt>(ctx.lift.positive_list.size());
    for (HighsInt read = 0; read < n_positive; ++read) {
        HighsInt j = ctx.lift.positive_list[read];
        if (!ctx.lift.in_positive[j]) {
            continue;
        }
        ctx.lift.positive_list[write++] = j;
        if (ctx.lift.score[j] <= best.score) {
            continue;
        }
        double lo = ctx.lift.lo[j];
        double hi = ctx.lift.hi[j];
        if (lo > hi) {
            continue;
        }
        double target;
        if (ctx.minimize) {
            target = (ctx.col_cost[j] > 0) ? lo : hi;
        } else {
            target = (ctx.col_cost[j] > 0) ? hi : lo;
        }
        target = ctx.clamp_and_round(j, target);
        if (std::abs(target - ctx.solution[j]) < kEpsZero) {
            continue;
        }
        best = {j, target, ctx.lift.score[j], 0.0};
    }
    ctx.lift.positive_list.resize(write);

    return best;
}

// Cognitive complexity 93 (threshold 25).  Kept whole: Algorithm 2 in full: six numbered phases,
// each the fallback for the previous one, sharing the candidate batch and effort counter.
// Decomposing it would move work across a worker's inner loop, and the
// closeout takes no unmeasured performance risk; the standards also rank
// fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Candidate infeasible_step(WorkerCtx& ctx, Rng& rng, HighsInt step, bool best_feasible,
                          double best_objective, const std::vector<HighsInt>& costed_vars,
                          const std::vector<HighsInt>& binary_vars) {
    // Issue #129: this is now also entered from a *feasible* state, on a
    // failed lift (`LocalMipWorker::run_attempt`'s fall-through) -- not
    // only from a genuinely infeasible one.  `was_infeasible` gates the
    // lift cache's incremental dirty-tracking and the caller's next
    // `need_full_recheck` (both in `local_mip_worker.cpp`), so it must
    // record whether *this* entry was genuinely infeasible, not merely
    // that this function ran: forcing it to `true` unconditionally would
    // charge a full recheck on every plateau iteration reached through
    // the fall-through path, and would suppress dirty-marking for moves
    // that never left feasible territory. Identical to the old
    // unconditional `true` for the pre-existing (genuinely infeasible)
    // caller, since `ctx.violated` is non-empty there by construction.
    ctx.was_infeasible = !ctx.violated.empty();

    auto& batch = ctx.batch;
    auto& sampled = ctx.sampled;

    // --- Phase 1: BMS tight moves from violated constraints ---
    //
    // DELIBERATE DEVIATION FROM THE PAPER, MATCHING THE AUTHORS' CODE:
    // Algorithm 2 enumerates *all* violated constraints.  BMS (Best from
    // Multiple Selections) appears nowhere in the paper, but the authors'
    // implementation samples exactly this way
    // (github.com/shaowei-cai-group/Local-MIP, `explore_unsat.cpp`:
    // `sample_idxs(..., m_bms_con, ...)` and `sample_op(m_bms_op, ...)`).
    // The same applies to the analogous caps in the satisfied-constraint
    // and boolean-flip phases below.  Reference-faithful, not
    // paper-faithful; see docs/PARAMETERS.md `kBmsConstraints`.
    HighsInt num_to_sample = std::min(kBmsConstraints * 3, ctx.violated.size());
    HighsInt num_to_keep = std::min(kBmsConstraints, ctx.violated.size());

    sampled.clear();
    if (num_to_sample == ctx.violated.size()) {
        for (auto ci : ctx.violated) {
            sampled.push_back({ci, ctx.weight[ci]});
        }
    } else {
        for (HighsInt s = 0; s < num_to_sample; ++s) {
            auto idx = static_cast<HighsInt>(rng() % ctx.violated.size());
            sampled.push_back({ctx.violated[idx], ctx.weight[ctx.violated[idx]]});
        }
    }

    if (std::cmp_greater(sampled.size(), num_to_keep)) {
        std::partial_sort(sampled.begin(), sampled.begin() + num_to_keep, sampled.end(),
                          [](const WeightedCon& a, const WeightedCon& b) { return a.w > b.w; });
        sampled.resize(num_to_keep);
    }

    batch.clear();
    HighsInt budget_remaining = kBmsBudget;

    for (auto& [ci, w] : sampled) {
        (void)w;
        if (budget_remaining <= 0) {
            break;
        }
        for (HighsInt k = ctx.ar_start[ci]; k < ctx.ar_start[ci + 1] && budget_remaining > 0; ++k) {
            HighsInt j = ctx.ar_index[k];
            --budget_remaining;
            double delta = ctx.compute_tight_delta(ci, j, ctx.ar_value[k]);
            append_candidate(ctx, batch, j, delta);
        }
    }

    // --- Phase 1b: Breakthrough moves (only post-feasible, Alg 2 line 5-6) ---
    //
    // Definition 2's other precondition, `obj(s) >= obj(s*)`, is enforced
    // inside `compute_breakthrough_delta` rather than here (issue #150 --
    // the authors' code spells it at this loop instead; see that
    // function's comment for why the operator is the place for it).
    if (best_feasible) {
        for (HighsInt j : costed_vars) {
            double delta = compute_breakthrough_delta(ctx, j, ctx.current_obj, best_objective);
            append_candidate(ctx, batch, j, delta);
        }
    }

    Candidate cand = select_best_from_batch(ctx, batch, step, true, best_objective, best_feasible);

    // If positive candidate found, done (Alg 2 lines 1-6)
    if (cand.var_idx != -1 && cand.score > kScoreTol) {
        return cand;
    }

    // --- Phase 2: MTM in satisfied constraints (Alg 2 lines 7-8) ---
    if (!ctx.satisfied.empty()) {
        batch.clear();
        HighsInt num_sat_sample = std::min(kBmsSatCon, ctx.satisfied.size());
        HighsInt sat_budget = kBmsSatBudget;
        for (HighsInt s = 0; s < num_sat_sample && sat_budget > 0; ++s) {
            HighsInt ci = ctx.satisfied[static_cast<HighsInt>(
                rng() % static_cast<uint64_t>(ctx.satisfied.size()))];
            for (HighsInt k = ctx.ar_start[ci]; k < ctx.ar_start[ci + 1] && sat_budget > 0; ++k) {
                HighsInt j = ctx.ar_index[k];
                --sat_budget;
                double delta = ctx.compute_tight_delta(ci, j, ctx.ar_value[k]);
                append_candidate(ctx, batch, j, delta);
            }
        }
        auto sat_cand =
            select_best_from_batch(ctx, batch, step, false, best_objective, best_feasible);
        if (sat_cand.var_idx != -1 && sat_cand.score > cand.score + kScoreTol) {
            cand = sat_cand;
        }
    }

    if (cand.var_idx != -1 && cand.score > kScoreTol) {
        return cand;
    }

    // --- Phase 3: Boolean flip (Alg 2 lines 9-11) ---
    if (!binary_vars.empty()) {
        batch.clear();
        auto nbinary = static_cast<HighsInt>(binary_vars.size());
        auto offset = static_cast<HighsInt>(rng() % nbinary);
        for (HighsInt idx = 0; idx < nbinary && idx < kBoolFlipBudget; ++idx) {
            HighsInt j = binary_vars[(offset + idx) % nbinary];
            double new_val = (ctx.solution[j] < 0.5) ? 1.0 : 0.0;
            if (std::abs(new_val - ctx.solution[j]) < kEpsZero) {
                continue;
            }
            batch.push_back({j, new_val});
        }
        if (!batch.empty()) {
            auto flip_cand =
                select_best_from_batch(ctx, batch, step, true, best_objective, best_feasible);
            if (flip_cand.var_idx != -1 && flip_cand.score > cand.score + kScoreTol) {
                cand = flip_cand;
            }
        }
    }

    if (cand.var_idx != -1 && cand.score > kScoreTol) {
        return cand;
    }

    // --- Phase 4: Weight update + random constraint fallback (Alg 2 lines 12-14) ---
    //
    // `is_feasible` reads the CURRENT state (`ctx.violated`, unchanged by
    // anything above -- phases 1-3 only score candidates, never apply
    // one), not a hardcoded `false`: issue #129 also reaches this phase
    // from a feasible worker whose lift just failed, and the paper's
    // PAWS-style weight scheme strengthens `w(obj)` at a feasible local
    // optimum, `w(coni)` at an infeasible one (Algorithm 1 lines 4-7 vs.
    // Algorithm 2 lines 12-14 -- the same update, two triggers). Identical
    // to the old hardcoded `false` for the pre-existing (genuinely
    // infeasible) caller.
    ctx.update_weights(rng, /*is_feasible=*/ctx.violated.empty(), best_feasible, best_objective);

    if (!ctx.violated.empty()) {
        batch.clear();
        HighsInt ci =
            ctx.violated[static_cast<HighsInt>(rng() % static_cast<uint64_t>(ctx.violated.size()))];
        for (HighsInt k = ctx.ar_start[ci]; k < ctx.ar_start[ci + 1]; ++k) {
            HighsInt j = ctx.ar_index[k];
            double delta = ctx.compute_tight_delta(ci, j, ctx.ar_value[k]);
            append_candidate(ctx, batch, j, delta);
        }
        // Breakthrough candidates already scored in Phase 1; skip re-scoring.
        auto fallback =
            select_best_from_batch(ctx, batch, step, false, best_objective, best_feasible);
        if (fallback.var_idx != -1 &&
            (cand.var_idx == -1 || fallback.score > cand.score + kScoreTol ||
             (fallback.score > cand.score - kScoreTol && fallback.bonus > cand.bonus))) {
            cand = fallback;
        }
    }

    if (cand.var_idx != -1) {
        return cand;
    }

    // --- Phase 5: Perturbation (our addition, last resort) ---
    if (!ctx.violated.empty()) {
        HighsInt ci =
            ctx.violated[static_cast<HighsInt>(rng() % static_cast<uint64_t>(ctx.violated.size()))];
        HighsInt row_len = ctx.ar_start[ci + 1] - ctx.ar_start[ci];
        if (row_len > 0) {
            HighsInt k = ctx.ar_start[ci] + static_cast<HighsInt>(rng() % row_len);
            HighsInt j = ctx.ar_index[k];
            double new_val;
            if (ctx.is_binary(j)) {
                new_val = (ctx.solution[j] < 0.5) ? 1.0 : 0.0;
            } else if (ctx.is_int(j)) {
                HighsInt dir = (rng() % 2 == 0) ? 1 : -1;
                new_val = ctx.clamp_and_round(j, ctx.solution[j] + dir);
            } else {
                double range = std::min(ctx.col_ub[j], ctx.col_lb[j] + 1e6) - ctx.col_lb[j];
                double perturbation =
                    std::uniform_real_distribution<double>(-0.1 * range, 0.1 * range)(rng);
                new_val = ctx.clamp_and_round(j, ctx.solution[j] + perturbation);
            }
            if (std::abs(new_val - ctx.solution[j]) > kEpsZero) {
                auto [prog, bon] =
                    compute_candidate_scores(ctx, j, new_val, best_feasible, best_objective);
                ctx.viol_cache.reset();
                cand = {j, new_val, prog, bon};
            }
        }
    }

    if (cand.var_idx != -1) {
        return cand;
    }

    // --- Phase 6: Easy moves (our addition) ---
    {
        batch.clear();
        HighsInt num_easy = std::min(kEasyBudget, ctx.ncol);
        for (HighsInt s = 0; s < num_easy; ++s) {
            auto j = static_cast<HighsInt>(rng() % ctx.ncol);
            double target;
            if (ctx.col_lb[j] > 0) {
                target = ctx.col_lb[j];
            } else if (ctx.col_ub[j] < 0) {
                target = ctx.col_ub[j];
            } else {
                target = 0.0;
            }
            append_candidate(ctx, batch, j, target - ctx.solution[j]);
            // Try: toward lower bound
            if (ctx.col_lb[j] > -1e15 && ctx.col_lb[j] < 0) {
                append_candidate(ctx, batch, j, ctx.col_lb[j] - ctx.solution[j]);
            }
            // Try: toward upper bound
            if (ctx.col_ub[j] < 1e15 && ctx.col_ub[j] > 0) {
                append_candidate(ctx, batch, j, ctx.col_ub[j] - ctx.solution[j]);
            }
            // Try: midpoint for continuous
            if (!ctx.is_int(j) && ctx.col_lb[j] > -1e15 && ctx.col_ub[j] < 1e15) {
                append_candidate(ctx, batch, j,
                                 ((ctx.col_lb[j] + ctx.col_ub[j]) * 0.5) - ctx.solution[j]);
            }
        }
        auto easy_cand =
            select_best_from_batch(ctx, batch, step, false, best_objective, best_feasible);
        if (easy_cand.var_idx != -1) {
            cand = easy_cand;
        }
    }

    return cand;
}

}  // namespace local_mip_detail
