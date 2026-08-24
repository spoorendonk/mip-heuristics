#include "fpr.h"

#include "fpr_core.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "mip/HighsMipSolver.h"
#include "opportunistic_runner.h"
#include "worker_base.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace fpr {

#ifndef NDEBUG
// Test-only lifecycle counters.  Plain `std::atomic<size_t>` with
// relaxed `fetch_add` — the increments fire at most once per
// `kBudgetGate` return and once per multi-attempt-loop iter past the
// first, so the atomic cost is negligible relative to a per-thread
// accumulator + flush.  See issue #77 review residual finding 4.
namespace {
std::atomic<size_t> g_budget_gate_hits{0};
std::atomic<size_t> g_multi_attempt_iters{0};
}  // namespace
size_t budget_gate_hits() {
    return g_budget_gate_hits.load(std::memory_order_relaxed);
}
size_t multi_attempt_iters() {
    return g_multi_attempt_iters.load(std::memory_order_relaxed);
}
void reset_test_counters() {
    g_budget_gate_hits.store(0, std::memory_order_relaxed);
    g_multi_attempt_iters.store(0, std::memory_order_relaxed);
}
#endif

// Precomputed variable orders indexed by kFprStrategies position.  Built
// once sequentially before any parallel region (some strategies call
// HighsCliqueTable::cliquePartition which is not thread-safe).
using VarOrderTable = std::vector<std::vector<HighsInt>>;

// Worker driving the lifecycle introduced in issue #77: an attempt
// is the unit of work, and an attempt's DFS may pause at the per-run
// budget gate and resume next call with state intact.  When an attempt
// verdicts (feasible / failed), the worker advances `attempt_idx_` and
// picks the next (strategy, mode) from a deterministic per-worker
// rotation `(worker_idx_ + attempt_idx_)`.  Attempts never share a
// queue across workers — that determinism rule is what lets two runs
// with the same seed produce bit-identical [Sequential] traces at
// `threads=1`.
//
// Termination: `finished()` used to return false unconditionally, on the
// grounds that the runner's stale gate was the backstop.  Issue #111
// removed that reasoning — under an independently tuned per-heuristic
// budget (#110) nothing else bounds one worker — so the worker now owns
// an absolute stall gate of its own (`base_.stale_budget`, sized from
// `mip_heuristic_fpr_stall`).  It has no rebuild path: a retired FPR worker stays
// retired and `run_attempt` reports zero effort, which is how the
// opportunistic runner retires the slot.
//
// **Pause/resume is stall-neutral.**  The gate counts effort since this
// worker's last accepted solution, never attempts and never calls, so a
// `kBudgetGate` pause neither advances it by itself nor resets it: an
// attempt spanning K calls is charged exactly the sum of what those K
// calls spent, which is what one uninterrupted call of the same total
// size would have charged.  A worker that was interrupted has therefore
// not stalled *for having been interrupted*.  Counting attempts, or
// counting "calls that returned without a solution", would break that —
// a paused attempt returns without a solution every time.
class FprWorker {
public:
    // `binary` is the dispatch's `isBinary` snapshot (`ProblemView::binary`,
    // issue #99); it must outlive the worker.
    FprWorker(const ExecutionContext& exec, const CscMatrix& csc, IncumbentSink& sink,
              const VarOrderTable& var_orders, const uint8_t* binary, int worker_idx, uint32_t seed,
              size_t attempt_budget, size_t stale_budget);

    AttemptResult run_attempt(size_t attempt_budget);

    [[nodiscard]] bool finished() const { return base_.finished; }

private:
    // Pick the (strategy, mode) for `attempt_idx_`.  Cycles the
    // paper-curated `kInitialFprConfigs` list (8 entries) keyed on
    // `(worker_idx + attempt_idx) % kNumInitialFprConfigs`.  See the body
    // comment for why this is the curated list rather than the full 8×5
    // grid (a second `repair_search` activity-undo gap is the residual
    // blocker).
    void select_config_for_current_attempt();

    // Book one call's effort against the staleness gate.  Called on both
    // exits from `run_attempt` — the `kBudgetGate` pause and the normal
    // return — so every unit of effort is counted exactly once, which is
    // what makes the gate independent of how an attempt was sliced.
    void charge(const AttemptResult& attempt);

    const ExecutionContext& exec_;
    HighsMipSolver& mipsolver_;
    const CscMatrix& csc_;
    IncumbentSink& sink_;
    const VarOrderTable& var_orders_;
    const uint8_t* binary_;

    int worker_idx_;
    size_t attempt_budget_;  // hint for cfg.max_effort per attempt

    int strat_idx_ = 0;
    FrameworkMode mode_ = FrameworkMode::kDfs;

    int attempt_idx_ = 0;
    FprAttemptState attempt_state_;

    // The "is an attempt currently mid-flight" predicate is computed
    // from `attempt_state_.phase` rather than mirrored in a separate
    // bool.  `kIdle` = no attempt or just finalized; `kDfs` = DFS in
    // progress (paused or running); `kReadyToFinish` = step verdicted,
    // finish pending.  The 3-state Phase enum stays — `kReadyToFinish`
    // earns its keep via `fpr_attempt_step`'s assert that catches
    // "step called after step already returned kVerdictReady" — but
    // having a redundant bool in the worker is pure drift risk.
    [[nodiscard]] bool attempt_alive() const {
        return attempt_state_.phase != FprAttemptState::Phase::kIdle;
    }

    // Effort / staleness / finished bookkeeping, shared with FJ, LocalMIP
    // and Scylla (worker_base.h).  Only the staleness half is armed:
    // `total_budget` is left at SIZE_MAX because the whole-dispatch
    // ceiling belongs to `run_opportunistic_loop`, and giving each FPR
    // worker a hard `total / N` share as well would cap a fast worker at
    // its share instead of letting it absorb a slow peer's — a different
    // change from the one issue #111 asks for.
    WorkerBudgetState base_;

    Rng rng_;
    FprScratch scratch_;
    // Reused across attempts to avoid `std::vector<double>` churn — the
    // multi-attempt loop in `run_attempt` calls `sink_.get_restart` once
    // per attempt, and an unhoisted local would re-allocate every
    // iteration on instances large enough to matter (review R2 CF-1).
    std::vector<double> initial_solution_buf_;
};

namespace {

// Master strategy pool for all FPR parallel paths.  var_orders are
// precomputed for each entry (see precompute_var_orders) so any strategy
// — including clique-based ones like kStratBadobjcl whose compute_var_order
// calls HighsCliqueTable::cliquePartition — can be used inside a parallel
// region without racing on cliquePartition's internal state.
constexpr auto kFprStrategies = std::to_array<FprStrategyConfig>({
    // Strategies used by the paper's curated initial configs.
    kStratBadobjcl,  // 0: type+cliques / badobj
    kStratLocks2,    // 1: locks / loosedyn
    kStratLocks,     // 2: LR / loosedyn
    kStratRandom,    // 3: type+cliques / random
    kStratDomsize,   // 4: domainSize / loosedyn
    // Extra strategies kept for randomization diversity at restart.
    kStratRandom2,  // 5: random / random
    kStratBadobj,   // 6: type / badobj
    kStratGoodobj,  // 7: type / goodobj
});
constexpr int kNumFprStrategies = static_cast<int>(std::size(kFprStrategies));

// Paper Section 6.3, Class 1 — LP-free initial configs.  Each entry gives
// a worker its starting (strategy, mode); strat_idx is an index into
// kFprStrategies.
struct InitialFprConfig {
    int strat_idx;
    FrameworkMode mode;
};
constexpr auto kInitialFprConfigs = std::to_array<InitialFprConfig>({
    {0, FrameworkMode::kDfs},           // kStratBadobjcl, dfs
    {1, FrameworkMode::kDfs},           // kStratLocks2, dfs
    {1, FrameworkMode::kDive},          // kStratLocks2, dive
    {2, FrameworkMode::kDfsrep},        // kStratLocks, dfsrep
    {0, FrameworkMode::kDfsrep},        // kStratBadobjcl, dfsrep
    {3, FrameworkMode::kDiveprop},      // kStratRandom, diveprop
    {2, FrameworkMode::kRepairSearch},  // kStratLocks, repairsearch
    {4, FrameworkMode::kDfs},           // kStratDomsize, dfs
});
constexpr int kNumInitialFprConfigs = static_cast<int>(std::size(kInitialFprConfigs));

// Compute variable orders for every strategy in kFprStrategies.  MUST be
// called from a sequential context: clique-based var_strategies invoke
// HighsCliqueTable::cliquePartition which mutates internal state and is
// not thread-safe.
VarOrderTable precompute_var_orders(HighsMipSolver& mipsolver) {
    VarOrderTable orders(kNumFprStrategies);
    const uint32_t base = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    for (int i = 0; i < kNumFprStrategies; ++i) {
        Rng rng(base + static_cast<uint32_t>(i));
        orders[i] = compute_var_order(mipsolver, kFprStrategies[i].var_strategy, rng, nullptr);
    }
    return orders;
}

}  // namespace

// ---------------------------------------------------------------------------
// FprWorker implementation
// ---------------------------------------------------------------------------

FprWorker::FprWorker(const ExecutionContext& exec, const CscMatrix& csc, IncumbentSink& sink,
                     const VarOrderTable& var_orders, const uint8_t* binary, int worker_idx,
                     uint32_t seed, size_t attempt_budget, size_t stale_budget)
    : exec_(exec),
      mipsolver_(exec.mipsolver),
      csc_(csc),
      sink_(sink),
      var_orders_(var_orders),
      binary_(binary),
      worker_idx_(worker_idx),
      attempt_budget_(attempt_budget),
      rng_(seed) {
    base_.total_budget = std::numeric_limits<size_t>::max();
    base_.stale_budget = stale_budget;
    select_config_for_current_attempt();
}

void FprWorker::select_config_for_current_attempt() {
    // Per-worker rotation through the paper-curated `kInitialFprConfigs`
    // list (8 entries), keyed deterministically on
    // `(worker_idx + attempt_idx) % kNumInitialFprConfigs`.  Each worker
    // visits every Class-1 config exactly once before wrapping.  Issue #77's
    // determinism rule is satisfied because the rotation is purely a
    // function of (worker identity, attempt count) — no shared queue, no
    // rng dependency, no per-attempt randomisation.
    //
    // Why not the full 8 × 5 = 40-pair (strategy, mode) grid?  An earlier
    // draft widened to it once `e_pq_mark` threading was in place, but
    // the `(kStratDomsize, kRepairSearch)` pairing exposed a second
    // latent state-restoration gap in `repair_search`'s secondary
    // backtrack: `act_mark` is not threaded through `RepairSearchNode`
    // analogously to `e_pq_mark`, so when `init_activities()` ran in
    // Phase 2 (any `kLoosedyn` value strategy) the activity vectors and
    // `vs_` diverge across the secondary backtrack.  `kStratDomsize` is
    // the only entry that simultaneously uses `init_domain_pq` AND a
    // `kLoosedyn` val strategy AND was widened to a `kRepairSearch`
    // mode the curated list never exercised — so it is the smallest
    // reproducer.  Fix is the same shape as `e_pq_mark` (extend
    // `RepairSearchNode` with `e_act_mark` and pass it to
    // `E.backtrack_to`); kept out of this change to bound scope.  Until
    // then the curated list keeps the rotation safe.  Multi-attempt
    // looping inside `run_attempt` still lets fast workers fill the slice
    // by cycling through the 8-config list, which the issue's #1
    // acceptance bullet (FPR CPU% on tbfp-network) cares about.
    const int idx =
        (((worker_idx_ + attempt_idx_) % kNumInitialFprConfigs) + kNumInitialFprConfigs) %
        kNumInitialFprConfigs;
    const auto& cfg = kInitialFprConfigs[idx];
    strat_idx_ = cfg.strat_idx;
    mode_ = cfg.mode;
}

void FprWorker::charge(const AttemptResult& attempt) {
    if (attempt.found_improvement) {
        base_.charge_improvement(attempt.effort);
    } else {
        base_.charge_no_improvement(attempt.effort);
    }
}

// Cognitive complexity 26 (threshold 25).  Kept whole: the worker's multi-attempt loop:
// pause/resume of an in-flight DFS and rotation through kInitialFprConfigs. Decomposing it would
// move work across a worker's inner loop, and the closeout takes no unmeasured performance risk;
// the standards also rank fidelity to the reference algorithm above mechanical extraction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
AttemptResult FprWorker::run_attempt(size_t attempt_budget) {
    AttemptResult attempt{};
    if (base_.finished) {
        return attempt;
    }

    // Issue #77 lifecycle.  Two mechanics in play:
    //
    // (1) Pause/resume *across* calls.  When the DFS exhausts the per-call
    //     slice, `fpr_attempt_step` returns `kBudgetGate` and we return so
    //     peers can run their next slice.  The DFS state (var_order_cursor,
    //     nodes_visited, found_complete, dfs_stack, prop_engine) lives in
    //     `attempt_state_` + `scratch_` and is preserved until the next call
    //     resumes the same attempt.  Without this, a long DFS subtree gets
    //     truncated and discarded each attempt — the parallelism bottleneck
    //     this issue exists to fix.
    //
    // (2) Multi-attempt fill *within* a call.  When an attempt verdicts
    //     before exhausting the slice (e.g. a fast-failing strategy on a
    //     hard instance, or a leaf found in 5% of the budget), we start
    //     the next attempt with the next rotation slot rather than idling
    //     at the runner's barrier waiting for slow peers.  This is what
    //     drives FPR's CPU% up on `tbfp-network` (acceptance bullet #1) —
    //     pause/resume alone fixes truncation but leaves fast workers
    //     idle while a slow worker holds the barrier.
    //
    // The safety cap (`kMaxAttemptsPerCall`) and no-progress guard guard
    // against degenerate models where every attempt verdicts with near-zero
    // recorded effort (`infeasible-mip0` initial-propagation short-circuit):
    // without bounds we'd burn the slice on `fpr_attempt_begin`'s
    // O(ncol+nrow) setup churn alone.

    // Snapshot the pool restart once per call so all attempts inside the
    // multi-attempt loop see the same `initial_solution`.  Per-attempt
    // `sink_.get_restart` would observe interleaved peer `offer` inserts
    // from other workers running concurrently in `parallel::for_each`,
    // breaking the issue-#77 determinism guarantee that two runs at
    // identical seed produce bit-identical [Sequential] summaries —
    // `initial_solution` is the only non-deterministic input into
    // `fpr_attempt_begin`.  `initial_solution_buf_` is a member to amortise
    // the `ncol`-sized allocation across calls (review R2 CF-1).
    initial_solution_buf_.clear();
    const bool have_restart = sink_.get_restart(rng_, initial_solution_buf_);

    // 32 attempts × 2 mutex ops × N workers is a theoretical upper bound on
    // pool-mutex acquisitions per outer attempt.  In practice the cap is
    // rarely approached: each attempt's begin charges O(nnz) coefficient
    // accesses (initial propagate), and the per-call slice is sized so an
    // instance with non-trivial DFS spends most of the slice inside step
    // rather than restarting attempts.  HighsSpinMutex critical sections
    // in `try_add` / `get_restart` are sub-microsecond (lower_bound over
    // <= kPoolCapacity entries plus an O(ncol) Hamming or single solution
    // copy).  Even worst-case, total mutex-time per attempt is bounded ms
    // (review R2 U-1 / Finding 3).
    constexpr int kMaxAttemptsPerCall = 32;
    int attempts_started = 0;
    size_t prev_loop_effort = 0;

    // Issue #111: this call may not outrun what is left of the worker's
    // stall allowance.  Without the clamp the *slice* bounds the call,
    // and the slice is `HeuristicBudget::attempt_cap` = `total / (10N)`,
    // which grows with the effort option — so a stalled worker would
    // still overshoot its absolute ceiling by a budget-proportional
    // amount and the gate would only half-bind.  Clamping here also makes
    // the DFS pause at the ceiling rather than past it, since
    // `fpr_attempt_step` is handed `budget_remaining` derived from
    // `call_cap`.
    const size_t stall_room = base_.stale_budget > base_.effort_since_improvement
                                  ? base_.stale_budget - base_.effort_since_improvement
                                  : 0;
    const size_t call_cap = std::min(attempt_budget, std::max<size_t>(stall_room, 1));

    while (attempt.effort < call_cap) {
        if (exec_.terminated()) {
            break;
        }
        if (attempts_started > 0 && attempt.effort == prev_loop_effort) {
            // Defensive belt-and-braces guard.  Today this branch is
            // unreachable: degenerate `ncol==0||nrow==0` models are filtered
            // out by `fpr::run` before workers are constructed,
            // every begin runs at least one `E.propagate(-1)` round (>0 ops
            // on any non-empty model), and finish always adds
            // `c.ar_index.size() > 0` for the LHS sum.  Keep the guard so a
            // future change that relaxes any of the above (e.g., a Phase 1
            // shortcut that skips initial propagate) cannot silently turn
            // this loop into an infinite attempt-cycler.
            break;
        }
        prev_loop_effort = attempt.effort;

        // Advance the per-worker rotation BEFORE building cfg so that
        // `cfg.strategy` / `cfg.mode` / `cfg.precomputed_var_order` reflect
        // the current attempt's choice.  Earlier draft built cfg from the
        // previous attempt's strat/mode and re-assigned 4 fields after the
        // rotation advance — a maintenance hazard if a future cfg field
        // is added (review R2 CF-2).
        if (!attempt_alive()) {
            if (attempts_started >= kMaxAttemptsPerCall) {
                break;
            }
#ifndef NDEBUG
            if (attempts_started > 0) {
                g_multi_attempt_iters.fetch_add(1, std::memory_order_relaxed);
            }
#endif
            ++attempts_started;
            select_config_for_current_attempt();
        }

        const auto& strat = kFprStrategies[strat_idx_];
        const auto& var_order = var_orders_[strat_idx_];
        FprConfig cfg{};
        // `cfg.max_effort` is the attempt-wide cap consumed by Phase 3 sub-
        // budgets (`cfg.max_effort - total_prop_work` for repair_search /
        // walksat).  Sized at the worker's `attempt_budget_`, which is
        // `HeuristicBudget::total >> 2` — a quarter of the dispatch
        // allowance.  It used to be spelled `HeuristicBudget::stale`
        // because the two were the same number; issue #111 made `stale`
        // an absolute instance-scaled ceiling, so this is now written out
        // at the construction site (`fpr::run`) and the two have parted
        // company.  Not the per-call
        // `attempt_budget`: when an attempt spans multiple `run_attempt` calls,
        // the cumulative `total_prop_work` arriving at Phase 3 already
        // exceeds any single slice, so a slice-sized cap clamps the repair
        // budget to 0 (review R1 CF-1).  The DFS gate inside
        // `fpr_attempt_step` uses `effort_remaining` (the per-call slice)
        // and is unaffected by this size — Phase 3's iteration counts
        // (`cfg.repair_iterations`, `cfg.walksat_iterations`) self-throttle
        // even when the effort budget is large.
        cfg.max_effort = std::max<size_t>(attempt_budget_, 1);
        cfg.hint = nullptr;
        cfg.scores = nullptr;
        cfg.cont_fallback = nullptr;
        cfg.csc = &csc_;
        cfg.mode = mode_;
        cfg.strategy = &strat;
        cfg.lp_ref = nullptr;
        cfg.precomputed_var_order = var_order.data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order.size());
        cfg.binary_mask = binary_;
        cfg.scratch = &scratch_;

        if (!attempt_alive()) {
            // Reuse the restart snapshot taken at the start of `run_attempt`
            // (review R1 / Finding 1) — `initial_solution_buf_` is the
            // member buffer the snapshot landed in.
            const double* init_ptr = have_restart ? initial_solution_buf_.data() : nullptr;
            fpr_attempt_begin(attempt_state_, mipsolver_, cfg, rng_, attempt_idx_, init_ptr);
            // `attempt_state_.phase` is now `kDfs` (or `kReadyToFinish`
            // if Phase 1 already produced a complete fixing); either way
            // `attempt_alive()` is true on the next iteration.
            attempt.effort += attempt_state_.effort_consumed;
        }

        if (attempt_state_.phase == FprAttemptState::Phase::kDfs) {
            const size_t before_step = attempt_state_.effort_consumed;
            const size_t budget_remaining =
                call_cap > attempt.effort ? call_cap - attempt.effort : 0;
            const FprStepResult outcome =
                fpr_attempt_step(attempt_state_, mipsolver_, cfg, rng_, budget_remaining);
            attempt.effort += attempt_state_.effort_consumed - before_step;
            if (outcome == FprStepResult::kBudgetGate) {
#ifndef NDEBUG
                g_budget_gate_hits.fetch_add(1, std::memory_order_relaxed);
#endif
                // Attempt paused at the per-call slice boundary — return so
                // peers do their next attempt's work and we resume here
                // next call.  Charge what the pause spent: the stall
                // counter is in effort units, so slicing an attempt
                // across calls costs it exactly what running it in one
                // call would (see the class comment).
                charge(attempt);
                return attempt;
            }
            // kVerdictReady — DFS ended (leaf found or stack/node-limit
            // exhausted), proceed to finish.
        }

        const size_t before_finish = attempt_state_.effort_consumed;
        HeuristicResult result = fpr_attempt_finish(attempt_state_, mipsolver_, cfg, rng_);
        attempt.effort += attempt_state_.effort_consumed - before_finish;

        // Pool acceptance, not "this attempt reached a feasible point"
        // (#111): FPR reaches the same feasible point over and over on
        // some models, and each rediscovery used to clear the staleness
        // counter the gate reads.  `offer` is still called for every
        // feasible result, so nothing that reached HiGHS before stops
        // reaching it — only the verdict is now read.
        if (result.found_feasible && sink_.offer(result.objective, result.solution)) {
            attempt.found_improvement = true;
        }

        ++attempt_idx_;
        // `fpr_attempt_finish` set `attempt_state_.phase = kIdle`, so
        // `attempt_alive()` is false on the next iteration.
    }

    charge(attempt);
    return attempt;
}

// ---------------------------------------------------------------------------
// Parallel FPR
// ---------------------------------------------------------------------------

size_t run(const ProblemView& problem, const HeuristicBudget& budget, ExecutionContext& exec,
           IncumbentSink& sink) {
    if (problem.degenerate()) {
        return 0;
    }

    HighsMipSolver& mipsolver = exec.mipsolver;

    // Precompute var_orders sequentially before any parallel region.
    VarOrderTable var_orders = precompute_var_orders(mipsolver);

    std::vector<std::unique_ptr<FprWorker>> workers;
    workers.reserve(exec.num_workers);
    for (size_t w = 0; w < exec.num_workers; ++w) {
        uint32_t seed = exec.worker_seed(static_cast<int>(w));
        // `budget.total >> 2` is the *attempt-wide* `cfg.max_effort` hint,
        // not a stall threshold: it sizes Phase 3's repair/WalkSAT
        // sub-budgets and used to be spelled `budget.stale` only because
        // the two happened to be the same number.  Issue #111 made
        // `budget.stale` absolute, so the hint is written out here rather
        // than silently following it.  `budget.worker_stale` is this
        // worker's share of the dispatch's absolute stall ceiling.
        workers.push_back(std::make_unique<FprWorker>(
            exec, *problem.csc, sink, var_orders, problem.binary.data(), static_cast<int>(w), seed,
            budget.total >> 2, budget.worker_stale));
    }

    struct FprOppState {
        int worker_idx;
    };

    return run_opportunistic_loop(
        exec, budget,
        [](int worker_idx, Rng& /*rng*/) -> FprOppState { return FprOppState{worker_idx}; },
        [&](FprOppState& state, Rng& /*rng*/, size_t run_cap) -> AttemptResult {
            auto& worker = workers[state.worker_idx];
            // A retired worker reports zero effort, which retires its
            // slot in `run_opportunistic_loop` (issue #111 gave FPR the
            // worker-level gate it had been doing without).  Deliberately
            // no `attempt_with_rebuild`: FPR's diversity already comes
            // from the per-attempt rotation through `kInitialFprConfigs`,
            // so a rebuilt worker would resume the same rotation with a
            // fresh stall allowance and the gate would bound nothing.
            return worker->run_attempt(run_cap);
        });
}

}  // namespace fpr
