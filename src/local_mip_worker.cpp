#include "local_mip_worker.h"

#include "heuristic_common.h"
#include "incumbent_sink.h"
#include "local_mip.h"
#include "local_mip_caches.h"
#include "local_mip_core.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace local_mip_detail {

void perturb_solution(std::vector<double>& solution, const uint8_t* binary,
                      const std::vector<HighsVarType>& integrality,
                      const std::vector<double>& col_lb, const std::vector<double>& col_ub,
                      HighsInt ncol, Rng& rng) {
    // `kInfBoundShiftWindow` and `kSafeInt64DoubleRange` are shared
    // with `pump::perturb` via `heuristic_common.h` (R1-4 / R3-11
    // round-5 review): the two perturbation paths must use the same
    // window so cross-heuristic behaviour stays in lock-step.
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(integrality, j)) {
            continue;
        }
        if (coin(rng) > kPerturbBinaryFraction) {
            continue;
        }
        if (binary[j] != 0U) {
            solution[j] = (solution[j] < 0.5) ? 1.0 : 0.0;
        } else {
            // Skip variables whose current value is non-finite (NaN or
            // ±inf): casting NaN to int64_t is UB and `current ±
            // kInfBoundShiftWindow` would propagate NaN through the
            // shift arithmetic below.
            if (!std::isfinite(solution[j])) {
                continue;
            }
            double lo = std::ceil(col_lb[j]);
            double hi = std::floor(col_ub[j]);
            // Clamp `lo`/`hi` to a finite window around the current
            // value when either bound is non-finite OR finite-but-huge.
            // Without this guard the `static_cast<int64_t>(hi - lo)`
            // below overflows: `kHighsInf` (== std::infinity per HiGHS
            // HConst.h) is caught by `!std::isfinite`, but adversarial
            // user-supplied bounds at e.g. ±1e20 satisfy isfinite yet
            // still overflow int64_t (R1-3 round-5 review).  The
            // `kSafeInt64DoubleRange` check catches both cases at once.
            double current = std::round(solution[j]);
            if (!std::isfinite(lo) || !std::isfinite(hi) || hi - lo > kSafeInt64DoubleRange) {
                lo = current - kInfBoundShiftWindow;
                hi = current + kInfBoundShiftWindow;
            }
            if (hi <= lo) {
                continue;
            }
            // `lo`/`hi` are integer-valued (ceil/floor or finite ±64
            // window above), so `hi - lo` is an exact non-negative
            // integer; the `hi <= lo` guard already eliminated the
            // zero case.  Keep the post-cast guard purely as a safety
            // net for any future refactor that drops the integer
            // rounding above.
            auto irange = static_cast<int64_t>(hi - lo);
            if (irange < 1) {
                continue;
            }
            int64_t shift = std::uniform_int_distribution<int64_t>(1, irange)(rng);
            solution[j] = lo + std::fmod(current - lo + static_cast<double>(shift),
                                         static_cast<double>(irange) + 1.0);
            solution[j] = std::max(col_lb[j], std::min(col_ub[j], solution[j]));
        }
    }
}

LocalMipWorker::LocalMipWorker(HighsMipSolver& mipsolver, const ExecutionContext& exec,
                               const CscMatrix& csc, IncumbentSink& sink, size_t total_budget,
                               size_t stale_budget, uint32_t seed, const double* initial_solution,
                               const uint8_t* binary, WorkerTrace trace)
    : mipsolver_(mipsolver),
      exec_(exec),
      csc_(csc),
      sink_(sink),
      rng_(seed),
      trace_(trace),
      ctx_(mipsolver, csc, binary) {
    base_.total_budget = total_budget;
    base_.stale_budget = stale_budget;
    const HighsInt ncol = mipsolver.model_->num_col_;

    // Precompute variable subsets
    for (HighsInt j = 0; j < ncol; ++j) {
        if (std::abs(ctx_.col_cost[j]) >= kEpsZero) {
            costed_vars_.push_back(j);
        }
        if (ctx_.is_binary(j)) {
            binary_vars_.push_back(j);
        }
    }
    ctx_.lift.costed_vars = &costed_vars_;

    // Initialize solution
    const double* src = initial_solution;
    if (src != nullptr) {
        for (HighsInt j = 0; j < ncol; ++j) {
            double v = src[j];
            if (ctx_.is_int(j)) {
                v = std::round(v);
            }
            ctx_.solution[j] = std::max(ctx_.col_lb[j], std::min(ctx_.col_ub[j], v));
        }
    } else {
        for (HighsInt j = 0; j < ncol; ++j) {
            double v = std::clamp(0.0, ctx_.col_lb[j], ctx_.col_ub[j]);
            if (ctx_.is_int(j)) {
                v = std::round(v);
            }
            ctx_.solution[j] = v;
        }
    }

    ctx_.rebuild_state();
    best_objective_ = ctx_.minimize ? std::numeric_limits<double>::infinity()
                                    : -std::numeric_limits<double>::infinity();
    best_solution_.resize(ncol);
}

// Cognitive complexity 115 (threshold 25).  Kept whole: one LocalMIP attempt: the
// feasible/infeasible step loop with tabu and constraint-weight updates, restart, and the lift
// phase. Decomposing it would move work across a worker's inner loop, and the closeout takes no
// unmeasured performance risk; the standards also rank fidelity to the reference algorithm above
// mechanical extraction. NOLINTNEXTLINE(readability-function-cognitive-complexity)
AttemptResult LocalMipWorker::run_attempt(size_t attempt_budget) {
    if (base_.finished) {
        return {};
    }

    const HighsInt ncol = mipsolver_.model_->num_col_;

    AttemptResult attempt{};
    size_t effort_start = ctx_.effort;
    size_t effort_at_last_improvement = effort_start;

    // Effort is the deterministic work signal, and the outer loop
    // (opportunistic_runner.h / continuous_loop.h) still enforces the
    // wall-clock deadline between attempts.  That alone is not a bound:
    // one attempt is `attempt_cap` = `total / (10N)`, which scales with
    // `mip_heuristic_local_mip_effort`, so the overshoot grows with the
    // option (issue #114).  Poll the deadline here too, every
    // `kTermCheckWork` counted units of `WorkerCtx::effort`.
    //
    // The cadence is the point.  A clock_gettime per iteration was
    // measured at ~3% of total instruction refs on small instances, which
    // is why this loop had no deadline check at all; paced, it is not
    // measurable, and `past_deadline()` is the write-free half of
    // `ExecutionContext::terminated()`, so it needs no poller seat.
    //
    // It is paced on *work* and not on steps (#162).  A step is not a unit
    // of bounded size: in feasible mode every `kFeasibleRecheckPeriod`-th
    // step calls `WorkerCtx::full_recheck`, which charges one `nnz`, so a
    // fixed step count buys a wall-clock interval that grows with the
    // model.  On a 1.4M-nonzero model one 1000-step batch ran for tens of
    // seconds and the dispatch overran a 15 s limit by ~46 s while polling
    // 49 times, every poll landing before the deadline and the batch
    // between the last two spanning it.  This is #151's argument about
    // `PropEngine::propagate` one heuristic over, and it has the same
    // answer: the residual becomes one step plus a constant of charged
    // work, instead of one constant of steps whose cost the model sets.
    auto spent = [&]() { return ctx_.effort - effort_start; };
    size_t next_deadline_poll = 0;
    while (spent() < attempt_budget && !base_.exhausted(spent())) {
        if (base_.stale(spent())) {
            base_.finished = true;
            break;
        }
        if (spent() >= next_deadline_poll) {
            local_mip::note_deadline_poll();
            if (exec_.past_deadline()) {
                break;
            }
            next_deadline_poll = spent() + kTermCheckWork;
        }

        bool feasible_mode = ctx_.violated.empty();

        if (feasible_mode) {
            bool need_full_recheck = ctx_.was_infeasible ||
                                     (ctx_.feasible_recheck_counter % kFeasibleRecheckPeriod == 0);
            ctx_.was_infeasible = false;
            ++ctx_.feasible_recheck_counter;

            bool truly_feasible = true;
            if (need_full_recheck) {
                truly_feasible = ctx_.full_recheck(/*update_sets=*/true, /*early_exit=*/false);
            }
            if (!truly_feasible) {
                ++step_;
                continue;
            }

            double obj = ctx_.current_obj;
            bool improved = false;
            if (!best_feasible_) {
                improved = true;
            } else if (ctx_.minimize) {
                improved = (obj < best_objective_ - ctx_.epsilon);
            } else {
                improved = (obj > best_objective_ + ctx_.epsilon);
            }

            if (improved) {
                if (!need_full_recheck) {
                    if (!ctx_.full_recheck(/*update_sets=*/false,
                                           /*early_exit=*/true)) {
                        ctx_.rebuild_state();
                        ++step_;
                        continue;
                    }
                }
                best_feasible_ = true;
                best_objective_ = obj;
                best_solution_ = ctx_.solution;
                steps_since_improvement_ = 0;
                // Random walks that *led* to this improvement were
                // productive — reset the lifetime counter so the cap
                // doesn't cumulatively retire a worker that's still
                // finding improvements.  R3-7 round-3 review.
                feasible_random_walks_done_ = 0;

                // Three different notions of "improved" meet here, and
                // issues #111 and #116 are about keeping them apart.
                // Everything above is the *local search's* bookkeeping and
                // stays on the worker-local flag: beating this worker's
                // own best is what makes a step productive to the search,
                // and `best_objective_` restarts at infinity on every
                // rebuild by design.  Everything below is the *patience
                // gate's*, and it reads whether the offer moved the
                // incumbent.  Feeding the local flag to the gate let a
                // rebuilt worker clear the dispatch's staleness counter by
                // rediscovering a solution the pool already held; feeding
                // it the pool's admission verdict instead (#111) left a
                // top-K policy driving the gate, which is another way of
                // never running out of patience — LocalMIP earned ~3.3 M
                // acceptances against 24,598 incumbent improvements over
                // #113's 233 instances.
                // `effort_at`: `ctx_.effort` is this worker's cumulative
                // coefficient-access counter — the same quantity
                // `base_.total_effort` accumulates at the end of each
                // attempt, read here mid-attempt.
                if (sink_.offer(obj, ctx_.solution, trace_, trace_.at(ctx_.effort))
                        .improved_incumbent) {
                    attempt.found_improvement = true;
                    base_.reset_staleness();
                    // Part of the staleness accounting, not the search's:
                    // the tail of `run_attempt` charges
                    // `ctx_.effort - effort_at_last_improvement` to
                    // `effort_since_improvement`, so advancing this on an
                    // offer that moved nothing would silently forgive that
                    // effort.
                    effort_at_last_improvement = ctx_.effort;
                }
            }

            ctx_.lift.recompute_all(ctx_);
            Candidate lift_best = select_lift_move(ctx_);

            if (lift_best.var_idx != -1) {
                ctx_.apply_move_with_tabu(lift_best.var_idx, lift_best.new_val, step_, rng_);
                ++steps_since_improvement_;
            } else {
                // Issue #129: a failed lift falls through to Algorithm 2's
                // candidate generation in the SAME iteration, instead of a
                // standalone weight update followed by looping. The
                // reference implementation's `run_search` is unconditional
                // about calling the neighbourhood search every iteration
                // and only `continue`s past it on a *successful* lift
                // (github.com/shaowei-cai-group/Local-MIP); paper
                // Algorithm 1 line 8 reads the same way. `infeasible_step`
                // handles an empty `ctx_.violated` on its own (see its
                // comment in local_mip_core.h): Phase 1's tight-move half
                // is vacuous, Phase 1b's breakthrough moves are
                // unconditional on `best_feasible_`, and its own Phase 4
                // weight update reads `ctx_.violated` to bump `w(obj)`
                // rather than the standalone call this replaces.
                Candidate cand = infeasible_step(ctx_, rng_, step_, best_feasible_, best_objective_,
                                                 costed_vars_, binary_vars_);
                if (cand.var_idx != -1) {
                    ctx_.apply_move_with_tabu(cand.var_idx, cand.new_val, step_, rng_);
                    ++steps_since_improvement_;
                    if (!ctx_.violated.empty()) {
                        // Cold-review fix (#129): this move can leave the
                        // row set violated while `ctx_.lift.all_dirty` is
                        // still false (the lift cache was being maintained
                        // incrementally, since we were feasible up to this
                        // move). Before this fall-through existed, an
                        // infeasible episode could only be *entered* via
                        // `rebuild_state()` (restart / random walk /
                        // activity refresh), which already marks the
                        // cache fully dirty -- so "any infeasible episode
                        // starts with `all_dirty == true`" held for free.
                        // This move is a second way to enter one, and it
                        // does not go through `rebuild_state()`. Every
                        // move made while genuinely infeasible marks
                        // nothing dirty (`apply_move`'s `dirty_lift` is
                        // false whenever `ctx_.was_infeasible` is true,
                        // which `infeasible_step` now sets from this same
                        // `ctx_.violated` on its *next* call), so without
                        // restoring the invariant here, the eventual
                        // return to feasible mode would recompute only
                        // the column this move touched via
                        // `ctx_.lift.recompute_all`'s `dirty_list` path and
                        // serve stale scores -- including a stale
                        // `in_positive` membership, not just a stale
                        // value -- for every other column the infeasible
                        // episode moves. Marked here, at the one place
                        // that can newly *cause* such an episode, rather
                        // than on every `need_full_recheck` in the
                        // feasible branch above: that fires every
                        // `kFeasibleRecheckPeriod` (100) steps regardless
                        // of infeasibility, and forcing a full lift
                        // recompute there would defeat the incremental
                        // cache on a much hotter path than this one.
                        ctx_.lift.mark_all_dirty();
                    }
                }
                // else: every phase found nothing to do -- a true no-op
                // iteration.  The plateau counter must not accumulate it
                // (issue #129 point 3); `infeasible_step`'s own Phase 4
                // already charged whatever weight update it decided on.
                // Rare on real instances (measured: 0/6 on a spot check
                // of p0548/egout/bell5/3015/gt2/lseu) but not unreachable
                // -- every candidate can be tabu simultaneously.
            }

            if (steps_since_improvement_ >= kFeasiblePlateau) {
                // Random-walk diversification on plateau (engineering
                // extension).  Paper Algorithm 1 (Lin, Zou, Cai CP 2024)
                // runs a single search loop until time cutoff with no
                // notion of plateau, restart, or random walk; this
                // perturb-on-plateau scheme is our addition so workers
                // running under a coefficient-effort budget (rather
                // than wall-clock cutoff) don't burn out on stuck
                // assignments.  `kFeasibleMaxRandomWalks` caps the
                // walks for pathological instances.
                if (feasible_random_walks_done_ < kFeasibleMaxRandomWalks) {
                    perturb_solution(ctx_.solution, ctx_.binary, ctx_.integrality, ctx_.col_lb,
                                     ctx_.col_ub, ctx_.ncol, rng_);
                    // Reset weights on random walk (engineering choice;
                    // R2-8 round-4 review).  Paper §4.1 specifies only
                    // initialization (`w(obj)=1, w(coni)=1`) and the
                    // PAWS-style update rule fired at local optima — it
                    // is silent on weight handling at the random-walk /
                    // perturbation step we add above (the paper's main
                    // loop has no such step).  Two defensible reads:
                    //   - retain weights: keep the learned constraint
                    //     difficulty signal across the walk;
                    //   - reset weights: clear the bias toward the
                    //     direction that led to the just-failed plateau.
                    // We chose reset on the rationale that the existing
                    // weights bias the search back toward the just-failed
                    // plateau; this is intuition, not benchmarked.
                    // Documented as an engineering extension rather than
                    // paper-faithful behaviour (R3-1 round-5 review
                    // flagged the prior comment's fabricated empirical
                    // citation).
                    ctx_.reset_weights();
                    ctx_.rebuild_state();
                    ++feasible_random_walks_done_;
                    steps_since_improvement_ = 0;
                } else {
                    base_.finished = true;
                    break;
                }
            }
        } else {
            Candidate cand = infeasible_step(ctx_, rng_, step_, best_feasible_, best_objective_,
                                             costed_vars_, binary_vars_);

            if (cand.var_idx != -1) {
                ctx_.apply_move_with_tabu(cand.var_idx, cand.new_val, step_, rng_);
                ++steps_since_improvement_;
                if (ctx_.violated.empty()) {
                    steps_since_improvement_ = 0;
                }
            }
            // else: every phase found nothing to do -- a true no-op
            // iteration, symmetric with the feasible branch above (issue
            // #129 point 3): nothing changed, so the plateau counter must
            // not move either. Rare (see the feasible branch's comment)
            // but reachable, not dead code.
        }

        // Activity refresh
        if (step_ % kActivityPeriod == 0 && step_ > 0) {
            ctx_.rebuild_state();
        }

        // Restart logic
        if (steps_since_improvement_ >= kRestartInterval) {
            steps_since_improvement_ = 0;
            ++restart_count_;

            if (best_feasible_ && (restart_count_ % 2 == 1)) {
                ctx_.solution = best_solution_;
            } else {
                for (HighsInt j = 0; j < ncol; ++j) {
                    if (ctx_.is_binary(j)) {
                        ctx_.solution[j] = (rng_() % 2 == 0) ? 0.0 : 1.0;
                    } else if (ctx_.is_int(j)) {
                        double lo = std::max(ctx_.col_lb[j], -1e8);
                        double hi = std::min(ctx_.col_ub[j], lo + 100.0);
                        ctx_.solution[j] = std::max(
                            ctx_.col_lb[j],
                            std::min(
                                ctx_.col_ub[j],
                                std::round(std::uniform_real_distribution<double>(lo, hi)(rng_))));
                    } else {
                        double lo = ctx_.col_lb[j] > -kHighsInf ? ctx_.col_lb[j] : -1e6;
                        double hi = ctx_.col_ub[j] < kHighsInf ? ctx_.col_ub[j] : lo + 1e6;
                        if (hi > lo) {
                            ctx_.solution[j] = std::uniform_real_distribution<double>(lo, hi)(rng_);
                        } else {
                            ctx_.solution[j] = lo;
                        }
                    }
                }
            }

            ctx_.rebuild_state();
            std::ranges::fill(ctx_.tabu_inc_until, 0);
            std::ranges::fill(ctx_.tabu_dec_until, 0);
        }

        ++step_;
    }

    size_t attempt_effort = ctx_.effort - effort_start;
    local_mip::note_attempt_effort(static_cast<int64_t>(attempt_effort));
    base_.total_effort += attempt_effort;
    // Only add effort consumed since the last improvement within this
    // attempt (avoid double-counting when improvement resets the counter).
    base_.effort_since_improvement += ctx_.effort - effort_at_last_improvement;
    // Set finished if either budget is exhausted so the runner does not
    // re-enter this worker after its budget is spent.  (FjWorker gets this
    // via charge_improvement/charge_no_improvement; LocalMIP does its own
    // accounting because improvements can occur mid-attempt.)
    if (base_.exhausted() || base_.stale()) {
        base_.finished = true;
    }
    attempt.effort = attempt_effort;

    return attempt;
}

}  // namespace local_mip_detail
