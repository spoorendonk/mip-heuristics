#include "fpr_lp.h"

#include "effort_ledger.h"
#include "fpr_core.h"
#include "fpr_lp_arms.h"
#include "fpr_lp_refs.h"
#include "fpr_strategies.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "incumbent_sink.h"
#include "io/HighsIO.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "mode_dispatch.h"
#include "opportunistic_runner.h"
#include "worker_base.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace fpr_lp {

// ---------------------------------------------------------------------------
// LP-dependent arms (paper Section 6.3, Classes 2 and 3)
// ---------------------------------------------------------------------------
//
// Each arm is a (strategy, framework mode) pair bound to the LP reference
// solution its strategy is defined against (Sect. 4.1, Sect. 6.3):
//   Class 2  — zero-obj LP strategies     — analytic center  (ac_ptr)
//   Class 3a — zero-obj-vertex strategies — zero-obj vertex  (zv_ptr)
//   Class 3b — full-obj LP strategies     — full-obj LP      (lp_ptr)
//
// `kLpArmTable` is the single ordered source of truth for the whole
// portfolio: a name, a (strategy, mode) pair, and a reference class travel
// together in one record, so `kNumLpArms` and each arm's `LpRefClass` are
// *derived* from it rather than kept in sync by hand across parallel
// arrays that can silently disagree — which is exactly how issue #128
// happened: `cliques2` sat in the full-obj-LP array even though its
// ranking (`fpr_var_order.cpp`'s `rank_cliques2`) is defined by the paper
// against the zero-obj vertex. Sect. 4.1: "in strategy cliques2, we
// construct a clique cover dynamically using both the clique table and a
// reference LP solution, in this case, a zero-objective vertex." The
// Sect. 6.3 portfolio puts `diveprop-cliques2` in the zero-objective-vertex
// stage alongside `dfs-zerolp` / `diveprop-zerolp` — not in the
// full-objective stage `lp` uses.
//
// `lp_arm_table()` (declared in `fpr_lp_arms.h`, deliberately not
// `fpr_lp.h` — see that header's comment for why) exposes this table so
// `tests/test_fpr_lp.cpp` can assert every arm's `ref_class` against what
// its own strategy needs, independent of table position. That covers the
// arm-to-class assignment; `build_setup` below turns a `ref_class` into an
// actual pointer through `select_ref` (also in `fpr_lp_arms.h`), a second
// mapping a test exercises separately — see its comment for why that split
// matters.
constexpr auto kLpArmTable = std::to_array<LpArmInfo>({
    // Class 2 — zero-obj analytic center
    {"ZerocoreDfs", {kStratZerocore, FrameworkMode::kDfs}, LpRefClass::kAnalyticCenter},
    {"ZerocoreDive", {kStratZerocore, FrameworkMode::kDive}, LpRefClass::kAnalyticCenter},
    {"ZerocoreDiveprop", {kStratZerocore, FrameworkMode::kDiveprop}, LpRefClass::kAnalyticCenter},
    {"CliquesDfs",
     {kStratCliques, FrameworkMode::kDfs},
     LpRefClass::kAnalyticCenter},  // paper: "if predominant clique structure"; run
                                    // unconditionally, degrades gracefully on
                                    // non-clique models

    // Class 3a — zero-obj simplex vertex
    {"ZerolpDfs", {kStratZerolp, FrameworkMode::kDfs}, LpRefClass::kZeroObjVertex},
    {"ZerolpDiveprop", {kStratZerolp, FrameworkMode::kDiveprop}, LpRefClass::kZeroObjVertex},
    {"Cliques2Diveprop", {kStratCliques2, FrameworkMode::kDiveprop}, LpRefClass::kZeroObjVertex},

    // Class 3b — full-obj LP solution
    {"LpDfs", {kStratLp, FrameworkMode::kDfs}, LpRefClass::kFullObjLp},
    {"LpDive", {kStratLp, FrameworkMode::kDive}, LpRefClass::kFullObjLp},
    {"LpDiveprop", {kStratLp, FrameworkMode::kDiveprop}, LpRefClass::kFullObjLp},
});
constexpr int kNumLpArms = static_cast<int>(std::size(kLpArmTable));

// See the declaration in fpr_lp_arms.h for why this is its own function
// rather than a switch inlined into `build_setup` below: it exists so a
// test can exercise the class-to-pointer mapping in isolation from the
// arm-to-class table above, which is a separate fact `kLpArmTable` alone
// does not pin.
const double* select_ref(LpRefClass ref_class, const double* ac, const double* zv,
                         const double* lp) {
    switch (ref_class) {
        case LpRefClass::kAnalyticCenter:
            return ac;
        case LpRefClass::kZeroObjVertex:
            return zv;
        case LpRefClass::kFullObjLp:
            return lp;
    }
    // Unreachable for any of today's enumerators — `-Werror=switch`
    // (CMakeLists.txt) fails the build if a future one lands here without
    // a case above. No pointer here is a plausible stand-in for a real
    // reference class, deliberately: returning one of ac/zv/lp would repeat
    // #128's failure mode of an unhandled case silently picking a
    // real-looking answer. Existing `lp_ref` consumers already branch on
    // `!= nullptr` (e.g. `rank_cliques2`, `choose_value`), so nullptr
    // degrades through a path that already exists rather than inventing a
    // new one.
    return nullptr;
}

namespace {

// Test hook counters; see fpr_lp.h.  std::atomic so concurrent entry
// points don't race; relaxed is fine (monotonic, not used for
// synchronization).
std::atomic<size_t> g_dispatch_count{0};

// An arm binds a NamedConfig to the LP reference pointer it requires.
struct LpArm {
    const NamedConfig* config;
    const double* lp_ref;
};

// ---------------------------------------------------------------------------
// Shared setup: LP references, CSC matrix, precomputed var_orders
// ---------------------------------------------------------------------------

using VarOrderTable = std::vector<std::vector<HighsInt>>;

struct LpFprSetup {
    // Combined arm table (Class 2 + 3a + 3b, in that order).  Owning.
    std::vector<LpArm> arms;

    // Per-arm variable orderings, precomputed sequentially before any
    // parallel region to avoid races on the live clique table and domain.
    VarOrderTable var_orders;

    // CSC matrix of the model — built once, shared read-only.
    CscMatrix csc;

    // LP reference vectors.  Owned here so raw pointers stored in `arms`
    // remain valid for the lifetime of the setup.  Each may be empty if
    // the corresponding LP computation failed; in that case the pointer
    // fallback is the full-obj LP solution.
    std::vector<double> analytic_center;  // ac_ptr source (Class 2)
    std::vector<double> zero_vertex;      // zv_ptr source (Class 3a)

    // Per-column `HighsDomain::isBinary`, snapshotted here because the
    // workers below classify columns from it while an accepted solution
    // may be propagating the live root domain (issue #99).
    std::vector<uint8_t> binary;

    size_t budget = 0;
};

// What `build_setup` produced, and what it spent getting there (issue
// #118).
//
// The setup used to be a bare `std::optional<LpFprSetup>` carrying its own
// `setup_lp_iterations`, and the invariant that made that safe was written
// above it: *every* nullopt exit happened before the reference-LP solves,
// so a nullopt could never leave unaccounted LP work behind.  A wall-clock
// bail breaks that invariant deliberately — it can happen after a
// reference solve — and an empty optional cannot carry a counter out.  So
// the counter lives here, outside the optional, where the caller reads it
// on every path.
struct SetupResult {
    // Engaged only for a complete setup.  Empty means the dispatch does
    // not run, for one of the three reasons below.
    std::optional<LpFprSetup> setup;

    // LP iterations the reference solves (analytic center + zero-obj
    // vertex) consumed, complete setup or not.  `run()` charges these to
    // the shared RENS/RINS envelope exactly once, on whichever of its two
    // mutually exclusive booking paths it takes.  Nothing is cached across
    // dives, so a bail costs the envelope exactly what it spent and the
    // next dive re-pays only for the work it redoes.
    int64_t lp_iterations = 0;

    // True when the solve's wall-clock deadline is what stopped the setup,
    // as opposed to the model-shape and LP-status skips, which are not a
    // deadline event and consume nothing.  The caller books the two
    // differently, and keeping them apart is also what makes the bail
    // observable — through `probe_setup` in a test, and through the
    // `[Heur]` line for a benchmark run.
    bool deadline_bail = false;
};

// Build the shared LP-FPR setup.
//
// Three ways to come back without one, and the caller must tell them
// apart: the model is empty, the LP relaxation is not at an optimal scaled
// state (skip LP-FPR entirely, nothing consumed), or `deadline` passed
// part-way through (`deadline_bail`, with whatever the reference solves
// had already spent in `lp_iterations`).
//
// The deadline is polled between this setup's indivisible units — around
// each reference LP solve and before each arm's `compute_var_order` — for
// the reason `fpr::precompute_var_orders` is (issue #117): this whole
// function runs sequentially on the dispatching thread before a worker
// exists, so no gate the dispatch derives from its budget has any bearing
// on it, and one `compute_var_order` is a clique-cover greedy over the
// whole model.  Ten of them here, against eight for presolve FPR.
//
// It was not merely unpolled before: an expiry was actively *masked*.  The
// reference solves return an empty vector when the clock has passed, the
// `ac_ptr`/`zv_ptr` fallbacks read that as "LP failed, use the full-obj
// solution", and the setup went on to build all ten orders — so the one
// bounded part of this function hid the unbounded part.  Polling the clock
// rather than testing the vector is what separates the two cases.
SetupResult build_setup(HighsMipSolver& mipsolver, size_t max_effort, const Deadline& deadline) {
    SetupResult out;

    // Deliberately the first thing, ahead of the model-shape and LP-status
    // skips: a clock that has already passed is the answer whatever those
    // would say, and this ordering is the only seam at which a bail is
    // distinguishable from a skip on a model small enough to test
    // (`probe_setup`).  Do not sink it below them.
    if (deadline.expired()) {
        out.deadline_bail = true;
        return out;
    }

    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return out;
    }

    auto lp_status = mipdata->getLp().getStatus();
    if (!HighsLpRelaxation::scaledOptimal(lp_status)) {
        return out;
    }

    LpFprSetup s;
    s.budget = max_effort;

    s.csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    s.binary = build_binary_mask(mipsolver);

    // Full-obj LP solution — direct reference to the solver's col_value
    // vector (stable while we run because we do not trigger further LP
    // solves during LP-FPR).
    const auto& lp_sol = mipdata->getLp().getLpSolver().getSolution().col_value;
    const double* lp_ptr = lp_sol.data();

    // Zero-obj analytic center (for Class 2 zerocore strategies).  Each
    // solve is itself bounded — it gets what is left of the deadline,
    // capped at 30 s — so the poll after it is not what stops the solve;
    // it is what stops the *setup* from carrying on with a reference the
    // clock denied it.
    s.analytic_center =
        compute_analytic_center(mipsolver, /*use_objective=*/false, deadline, out.lp_iterations);
    const double* ac_ptr = s.analytic_center.empty() ? lp_ptr : s.analytic_center.data();

    if (deadline.expired()) {
        out.deadline_bail = true;
        return out;
    }

    // Zero-obj LP vertex (for Class 3a zerolp strategies, and — since
    // #128 — cliques2 too).  A failed or deadline-truncated solve here
    // falls back to the full-obj LP solution rather than an empty
    // pointer, same as the analytic-center fallback above: pre-#128 that
    // was two arms' fallback, and now, with cliques2 correctly reading
    // `zv_ptr`, it is a third.  Arguably the right degradation (a stale
    // reference beats none), but worth knowing: on that path cliques2
    // silently runs against the full-obj LP anyway, the exact reference
    // #128 exists to move it off of, just for a different reason (a
    // failed solve, not a wiring bug).
    s.zero_vertex = compute_zero_obj_vertex(mipsolver, deadline, out.lp_iterations);
    const double* zv_ptr = s.zero_vertex.empty() ? lp_ptr : s.zero_vertex.data();

    s.arms.reserve(kNumLpArms);
    for (const auto& spec : kLpArmTable) {
        s.arms.push_back({&spec.config, select_ref(spec.ref_class, ac_ptr, zv_ptr, lp_ptr)});
    }

    // Precompute var_orders sequentially — required before any parallel
    // region because clique-based var_strategies read the live clique
    // table and root domain, which `addIncumbent` mutates.
    s.var_orders.resize(kNumLpArms);
    const uint32_t base = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    for (int i = 0; i < kNumLpArms; ++i) {
        // Checked *between* orders, not inside one: `compute_var_order` is
        // indivisible from here and is this path's residual floor, exactly
        // as it is for the two presolve heuristics (#117).  A partial table
        // is of no use to anyone — the workers index it by arm — so the
        // whole dispatch is declined.
        if (deadline.expired()) {
            out.deadline_bail = true;
            return out;
        }
        // +200 offset spaces these seeds away from the presolve-FPR
        // var-order seeds (also derived from the same base) so the two
        // heuristics' RNG streams don't collide on small seed values.
        Rng rng(base + static_cast<uint32_t>(i) + 200);
        s.var_orders[i] = compute_var_order(mipsolver, s.arms[i].config->strat.var_strategy, rng,
                                            s.arms[i].lp_ref);
    }

    out.setup = std::move(s);
    return out;
}

// ---------------------------------------------------------------------------
// LpFprWorker: runs one LP-dependent FPR arm at a time
// ---------------------------------------------------------------------------

class LpFprWorker {
public:
    LpFprWorker(HighsMipSolver& mipsolver, const LpFprSetup& setup, IncumbentSink& sink,
                int arm_idx, uint32_t seed, WorkerTrace trace)
        : mipsolver_(mipsolver),
          setup_(setup),
          sink_(sink),
          arm_idx_(arm_idx),
          trace_(trace),
          rng_(seed) {}

    AttemptResult run_attempt(size_t attempt_budget) {
        AttemptResult attempt{};

        // After K stale attempts, randomize to another LP arm from the full
        // 10-element pool.  var_orders are precomputed for every arm so the
        // switch is race-free.  Track total randomisations separately so
        // the hard cap can fire even though the soft threshold resets
        // `attempts_without_improvement_` each trigger (R2-2 round-3 review).
        if (attempts_without_improvement_ >= kStaleAttemptThreshold) {
            randomize_arm();
            attempts_without_improvement_ = 0;
            ++randomizations_without_improvement_;
            if (randomizations_without_improvement_ >= kHardRandomizationLimit) {
                finished_ = true;
                return attempt;
            }
        }

        // No pool restart is pulled here (issue #122): `fpr_attempt` has no
        // seed parameter left to feed one to. The removed
        // `sink_.get_restart(rng_, initial_solution_buf_)` call bought a
        // pool-mutex acquisition, an `ncol`-sized copy, and RNG draws from
        // `rng_` on every call whenever the pool was non-empty (a roll, two
        // parent indices, up to `ncol` crossover coin flips — see
        // `SolutionPool::get_restart`), all for a seed value that was never
        // read. Removing it moves this worker's RNG stream on top of the
        // shift from deleting the seeding block itself.
        const LpArm& arm = setup_.arms[arm_idx_];
        const auto& var_order = setup_.var_orders[arm_idx_];

        FprConfig cfg{};
        cfg.max_effort = attempt_budget;
        cfg.cont_fallback = nullptr;
        cfg.csc = &setup_.csc;
        cfg.mode = arm.config->mode;
        cfg.strategy = &arm.config->strat;
        cfg.lp_ref = arm.lp_ref;
        cfg.precomputed_var_order = var_order.data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order.size());
        cfg.binary_mask = setup_.binary.data();
        cfg.scratch = &scratch_;

        auto result = fpr_attempt(mipsolver_, cfg, rng_, attempt_idx_);
        ++attempt_idx_;

        attempt.effort = result.effort;

        if (result.found_feasible) {
            // Deliberately discarded, and the only worker site that does.
            // Issues #111 and #116 both moved the improvement signal for
            // the four *presolve* heuristics and both left fpr_lp alone:
            // it draws from upstream's dive-time LP-iteration envelope
            // rather than a per-heuristic effort option, it has no
            // patience option, and it gates itself on its own
            // `kStaleAttemptThreshold` attempt counter as well as this
            // flag.  Neither the prototype that measured #111 nor #113's
            // presolve-only probe covered the dive, so this stays on
            // "reached a feasible point" until someone measures that
            // envelope.
            // `effort_at`: this worker's cumulative charge including the
            // attempt that just produced the solution.  `LpFprWorker` keeps
            // no `WorkerBudgetState`, so `total_effort_` below is the
            // running sum it charges; nothing else reads it, and no budget
            // or gate is derived from it (#106).
            static_cast<void>(sink_.offer(result.objective, result.solution, trace_,
                                          trace_.at(total_effort_ + result.effort)));
            attempt.found_improvement = true;
            attempts_without_improvement_ = 0;
            randomizations_without_improvement_ = 0;
        } else {
            ++attempts_without_improvement_;
        }

        total_effort_ += attempt.effort;
        return attempt;
    }

    [[nodiscard]] bool finished() const { return finished_; }

    // Monotone charged effort for the `[HeurSol]` trace (#106); see
    // `WorkerTrace` in worker_base.h.  `fpr_lp` rebuilds a retired worker
    // in place like Scylla does, so the slot's base has to absorb the
    // outgoing worker's charge.
    [[nodiscard]] size_t traced_effort() const { return trace_.at(total_effort_); }

private:
    void randomize_arm() { arm_idx_ = std::uniform_int_distribution<int>(0, kNumLpArms - 1)(rng_); }

    HighsMipSolver& mipsolver_;
    const LpFprSetup& setup_;
    IncumbentSink& sink_;

    int arm_idx_;
    // Trace-only slot identity; see `WorkerTrace` in worker_base.h.
    const WorkerTrace trace_;
    // Trace-only running charge.  `LpFprWorker` has no `WorkerBudgetState`
    // (it gates on its own attempt counters), so the `[HeurSol]` line needs
    // this sum; it feeds nothing else.
    size_t total_effort_ = 0;
    int attempt_idx_ = 0;
    int attempts_without_improvement_ = 0;
    int randomizations_without_improvement_ = 0;
    bool finished_ = false;

    Rng rng_;
    // Per-worker scratch reused across fpr_attempt calls to avoid malloc
    // churn on the DFS + WalkSAT repair hot path.
    FprScratch scratch_;

    // Hard cap on the number of soft-threshold arm randomisations
    // without an improvement before the worker declares itself
    // finished.  At 50 we expect to visit a substantial portion of the
    // 10 LP arms before giving up; Salvagnin et al. 2025 don't
    // prescribe an early finish — this is our engineering guard
    // against pathological loops.  Note: `fpr.cpp` used to carry the
    // same constant but issue #77 replaced its FprWorker with a
    // pause/resume lifecycle that has no per-worker stale counter;
    // LpFprWorker keeps the staleness shape because it has no DFS
    // state to resume across attempts.
    static constexpr int kHardRandomizationLimit = 50;

    // Number of stale attempts before a worker randomizes its arm.
    static constexpr int kStaleAttemptThreshold = 3;
};

// ---------------------------------------------------------------------------
// Worker dispatch
// ---------------------------------------------------------------------------

// Spawn `exec.num_workers` workers; worker w binds to arm `w % kNumLpArms`.
// The worker count comes from the shared `make_exec`, which floors at 1 —
// fpr_lp used to no-op on a hypothetical `num_threads() <= 0`.  Harmonising
// with the presolve heuristics, which have always floored, is the point.
// Matches the presolve FPR pattern (src/fpr.cpp) where excess workers
// wrap around the curated config list with distinct seeds for diversity.
size_t run_workers(const LpFprSetup& setup, const ExecutionContext& exec,
                   const HeuristicBudget& budget, IncumbentSink& sink) {
    g_dispatch_count.fetch_add(1, std::memory_order_relaxed);

    HighsMipSolver& mipsolver = exec.mipsolver;

    // Per-worker lightweight state: just the LpFprWorker instance.
    struct LpFprOppState {
        std::unique_ptr<LpFprWorker> worker;
        // Trace-only slot identity, carried across rebuilds (#106).
        WorkerTrace trace;
    };

    return run_opportunistic_loop(
        exec, budget,
        [&](int worker_idx, Rng& /*rng*/) -> LpFprOppState {
            // Initial arm is worker_idx modulo the arm pool.
            int arm = worker_idx % kNumLpArms;
            uint32_t seed = exec.worker_seed(worker_idx);
            return LpFprOppState{std::make_unique<LpFprWorker>(mipsolver, setup, sink, arm, seed,
                                                               WorkerTrace{worker_idx, 0}),
                                 WorkerTrace{worker_idx, 0}};
        },
        [&](LpFprOppState& state, Rng& rng, size_t run_cap) -> AttemptResult {
            // A retired worker here hit its hard randomisation cap; the
            // replacement draws a fresh arm so the slot keeps contributing.
            return attempt_with_rebuild(state.worker, run_cap, [&]() {
                int arm = std::uniform_int_distribution<int>(0, kNumLpArms - 1)(rng);
                auto seed = static_cast<uint32_t>(rng());
                state.trace.effort_base = state.worker->traced_effort();
                state.worker =
                    std::make_unique<LpFprWorker>(mipsolver, setup, sink, arm, seed, state.trace);
            });
        });
}

}  // namespace

void run(HighsMipSolver& mipsolver) {
    auto* mipdata = mipsolver.mipdata_.get();

    // Parallel-search guard: under `parallel=on` HiGHS spawns concurrent
    // processNode tasks (runTask with the parallel lock held), so this
    // function would race on the shared mipdata counters it reads and
    // charges below — RENS/RINS avoid that by accumulating worker-locally
    // and flushing at serial sync points, an infrastructure fpr_lp does
    // not have.  Skipping keeps the budget accounting exact and avoids
    // oversubscribing the thread pool with nested fpr_lp worker teams.
    // parallelLockActive() is false whenever there is a single search
    // worker (the HiGHS default), so this never fires on default runs.
    if (mipdata->parallelLockActive()) {
        return;
    }

    // Suite gating: fpr_lp runs only at a mip_heuristic_suite value naming fpr.
    // suite=off must disable it so an off run has nothing of ours running at
    // dive time either — this return sits above every read and write of
    // heuristic_lp_iterations / total_lp_iterations below, which feed
    // moreHeuristicsAllowed() and therefore decide whether RENS and RINS
    // run.  Do not move it down.
    if (!heuristics::effective_flags(*mipsolver.options_mip_).fpr) {
        return;
    }

    const size_t nnz = mipdata->ARindex_.size();
    if (nnz == 0) {
        return;
    }

    // Shared B&B heuristic budget, in vanilla's currency (LP iterations).
    // moreHeuristicsAllowed() — which gates the runHeuristics lambda we
    // are called from — admits heuristics early in the search while
    //   heuristic_lp_iterations < total_lp_iterations * heuristic_effort + 10000
    // (the initial-offset branch; the later branches are estimate-based,
    // so in submip/late-search regimes this formula over-estimates the
    // true remaining envelope by up to the 10000 offset — acceptable
    // because the per-call cap below bounds any single draw and the
    // charge-back still depletes the real counters the gate reads).
    // Size each call to the remaining headroom of that envelope, converted
    // at nnz effort-units per LP iteration (a simplex iteration touches
    // O(nnz) coefficients), and cap it at vanilla_effort_budget(nnz,
    // mip_heuristic_effort) — exactly nnz<<12 at the vanilla default 0.05
    // — so one call cannot drain a large late-search envelope in one go.
    // The charge-back below depletes the same envelope RENS/RINS draw
    // from; that is the point: fpr_lp competes for the vanilla heuristic
    // budget instead of consuming unaccounted work (and the budget scales
    // with the one vanilla knob, mip_heuristic_effort).
    const double allowed_iters =
        (static_cast<double>(mipdata->total_lp_iterations) * mipdata->heuristic_effort) + 10000.0;
    const double headroom_iters =
        allowed_iters - static_cast<double>(mipdata->heuristic_lp_iterations);
    if (headroom_iters <= 0.0) {
        return;
    }
    const double headroom_units = headroom_iters * static_cast<double>(nnz);
    const auto cap_units =
        static_cast<double>(vanilla_effort_budget(nnz, mipdata->heuristic_effort));
    const auto max_effort = static_cast<size_t>(std::min(headroom_units, cap_units));

    // Below ~256 LP-iteration equivalents the CSC build / var-order /
    // worker-spawn overhead dominates any useful DFS work; skip the call
    // (including the reference-LP solves) and let the envelope regrow.
    const size_t min_effort = nnz << 8;
    if (max_effort < min_effort) {
        return;
    }

    // Wall clock starts here: everything from this point on is fpr_lp's
    // spend, the reference-LP solves inside `build_setup` included.
    // The ledger is constructed first because it owns the clock — its
    // `now_s` reads the solver's timer, so this window shares an origin
    // with HiGHS's own display-line time column.
    EffortLedger ledger(mipsolver);
    const double t0_s = ledger.now_s();

    auto built = build_setup(mipsolver, max_effort, deadline_of(mipsolver));
    if (!built.setup) {
        // A deadline bail is a dispatch that happened and stopped: it may
        // have paid for a reference LP solve before the clock passed, and
        // that work owes the shared envelope whether or not a worker ever
        // ran — the whole reason this path is not #117's, which could
        // report zero and be done.  It books through the same
        // `charge_dive` the normal path uses, so the two are one
        // accounting rule, and it books even at zero iterations so the
        // declined dispatch leaves a `[Heur]` line rather than no trace at
        // all.
        //
        // The model-shape and LP-status skips book nothing, exactly as
        // before: no work was consumed and no dispatch was declined for a
        // reason a reader of the trace would want to see.
        //
        // `abandoned_setup=true` is what keeps this line out of the #113
        // probe's barren population (#119).  Without it a dive that never
        // searched would be byte-identical to one that searched and found
        // nothing — and this is the *dive* instance of exactly the
        // population that issue exists to separate, so the flag has to
        // reach the same field from here as it does from the presolve
        // chain.
        if (built.deadline_bail) {
            ledger.charge_dive("fpr_lp", 0, false, built.lp_iterations, nnz, t0_s, ledger.now_s(),
                               /*abandoned_setup=*/true);
        }
        return;
    }
    auto& setup = *built.setup;

    // The reference-LP solves are part of fpr_lp's spend: subtract them
    // from the worker budget so setup + workers together stay within
    // max_effort.  If setup ate (nearly) everything, skip the workers but
    // still charge the setup below.
    const auto setup_units = static_cast<size_t>(built.lp_iterations) * nnz;
    const size_t worker_budget = setup_units < setup.budget ? setup.budget - setup_units : 0;

    size_t worker_effort = 0;
    bool found = false;
    if (worker_budget >= min_effort) {
        // The sink owns the pool, seeds it from the incumbent, and wires
        // immediate submission so incumbent timestamps reflect find time
        // rather than the end-of-run flush time.
        IncumbentSink sink(mipsolver, kSolutionSourceFprLp);

        // fpr_lp is one heuristic family (LP-dependent FPR, Classes 2-3), so
        // it always runs arm-aligned parallel workers — num_threads workers
        // bound to the top-N arms from kClass2/3a/3b, sharing the sink.
        // The execution context and budget split are derived exactly as the
        // presolve heuristics' are, even though the setup around them is
        // fpr_lp's own.
        const ExecutionContext exec = make_exec(mipsolver);
        // `worker_budget >> 2` is the pre-#111 staleness rule, kept here
        // deliberately: issue #111 replaced the fraction-of-budget stale
        // thresholds in the *presolve* chain and put fpr_lp out of scope.
        // fpr_lp draws from upstream's dive-time LP-iteration envelope,
        // not from a per-heuristic effort option, and `LpFprWorker` keeps
        // its own private stale counter (`kStaleAttemptThreshold`) besides
        // this one — so the argument for an absolute, instance-scaled
        // ceiling has to be made against that envelope, not restated from
        // the presolve chain.
        worker_effort = run_workers(
            setup, exec, make_budget(worker_budget, exec.num_workers, worker_budget >> 2), sink);
        // Read after the worker loop has joined; the sink starts at zero
        // because it is constructed per dispatch.
        found = sink.accepted() > 0;
    }

    // The ledger books the observability counter, emits the per-heuristic
    // log line, and charges the consumed work back to the shared RENS/RINS
    // envelope.  fpr_lp is the only caller of `charge_dive`; that envelope
    // depletion is what makes it compete for the vanilla heuristic budget
    // rather than draw unaccounted work.
    // `abandoned_setup=false` stated rather than defaulted: reaching here
    // means the setup completed, and a dispatch that then found nothing is
    // barren in the sense the probe bins — which is the fact the bail path
    // above denies about itself.  Both answers are spelled out at both
    // sites (#119).
    ledger.charge_dive("fpr_lp", worker_effort, found, built.lp_iterations, nnz, t0_s,
                       ledger.now_s(), /*abandoned_setup=*/false);
}

SetupProbe probe_setup(HighsMipSolver& mipsolver, size_t max_effort) {
    const SetupResult result = build_setup(mipsolver, max_effort, deadline_of(mipsolver));
    return {result.setup.has_value(), result.deadline_bail, result.lp_iterations};
}

DispatchCounts dispatch_counts() {
    return {g_dispatch_count.load(std::memory_order_relaxed)};
}

void reset_dispatch_counts() {
    g_dispatch_count.store(0, std::memory_order_relaxed);
}

std::vector<LpArmInfo> lp_arm_table() {
    return {kLpArmTable.begin(), kLpArmTable.end()};
}

}  // namespace fpr_lp
