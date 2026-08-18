#include "fpr_lp.h"

#include "effort_ledger.h"
#include "fpr_core.h"
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
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace fpr_lp {

namespace {

// Test hook counters; see fpr_lp.h.  std::atomic so concurrent entry
// points don't race; relaxed is fine (monotonic, not used for
// synchronization).
std::atomic<size_t> g_dispatch_count{0};

// ---------------------------------------------------------------------------
// LP-dependent arms (paper Section 6.3, Classes 2 and 3)
// ---------------------------------------------------------------------------
//
// Each arm is a (strategy, framework mode) pair bound to a reference LP
// solution:
//   Class 2  — zero-obj LP strategies — analytic center  (ac_ptr)
//   Class 3a — zerolp configs         — zero-obj vertex  (zv_ptr)
//   Class 3b — lp/cliques2 configs    — full-obj LP      (lp_ptr)

constexpr NamedConfig kClass2Configs[] = {
    {kStratZerocore, FrameworkMode::kDfs},
    {kStratZerocore, FrameworkMode::kDive},
    {kStratZerocore, FrameworkMode::kDiveprop},
    {kStratCliques, FrameworkMode::kDfs},  // paper: "if predominant clique
                                           // structure"; run unconditionally,
                                           // degrades gracefully on non-clique
                                           // models
};
constexpr int kNumClass2 = static_cast<int>(std::size(kClass2Configs));

constexpr NamedConfig kClass3aConfigs[] = {
    {kStratZerolp, FrameworkMode::kDfs},
    {kStratZerolp, FrameworkMode::kDiveprop},
};
constexpr int kNumClass3a = static_cast<int>(std::size(kClass3aConfigs));

constexpr NamedConfig kClass3bConfigs[] = {
    {kStratCliques2, FrameworkMode::kDiveprop},
    {kStratLp, FrameworkMode::kDfs},
    {kStratLp, FrameworkMode::kDive},
    {kStratLp, FrameworkMode::kDiveprop},
};
constexpr int kNumClass3b = static_cast<int>(std::size(kClass3bConfigs));

constexpr int kNumLpArms = kNumClass2 + kNumClass3a + kNumClass3b;

constexpr const char* kLpArmNames[] = {
    "ZerocoreDfs",       // Class 2
    "ZerocoreDive",      // Class 2
    "ZerocoreDiveprop",  // Class 2
    "CliquesDfs",        // Class 2
    "ZerolpDfs",         // Class 3a
    "ZerolpDiveprop",    // Class 3a
    "Cliques2Diveprop",  // Class 3b
    "LpDfs",             // Class 3b
    "LpDive",            // Class 3b
    "LpDiveprop",        // Class 3b
};
static_assert(std::size(kLpArmNames) == kNumLpArms, "kLpArmNames must match total LP arm count");

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
    // parallel region to avoid races on HighsCliqueTable::cliquePartition.
    VarOrderTable var_orders;

    // CSC matrix of the model — built once, shared read-only.
    CscMatrix csc;

    // LP reference vectors.  Owned here so raw pointers stored in `arms`
    // remain valid for the lifetime of the setup.  Each may be empty if
    // the corresponding LP computation failed; in that case the pointer
    // fallback is the full-obj LP solution.
    std::vector<double> analytic_center;  // ac_ptr source (Class 2)
    std::vector<double> zero_vertex;      // zv_ptr source (Class 3a)

    // Incumbent hint (copy — snapshot taken before dispatch to keep the
    // pointer stable while mipdata->incumbent may be mutated by HiGHS).
    std::vector<double> incumbent_snapshot;

    // Per-column `HighsDomain::isBinary`, snapshotted here for the same
    // reason (issue #99): the workers below classify columns from it while
    // an accepted solution may be propagating the live root domain.
    std::vector<uint8_t> binary;

    // LP iterations spent solving the reference LPs (analytic center +
    // zero-obj vertex).  Charged against the shared B&B heuristic budget
    // by run() whether or not the workers subsequently run.
    int64_t setup_lp_iterations = 0;

    size_t budget = 0;
};

// Build the shared LP-FPR setup.  Returns nullopt when the model is
// empty or the LP relaxation is not at an optimal scaled state (the
// caller should skip LP-FPR entirely in that case).  All nullopt exits
// happen before the reference-LP solves, so a nullopt return never
// leaves unaccounted LP work behind.
std::optional<LpFprSetup> build_setup(HighsMipSolver& mipsolver, size_t max_effort) {
    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    const HighsInt nrow = model->num_row_;
    if (ncol == 0 || nrow == 0) {
        return std::nullopt;
    }

    auto lp_status = mipdata->getLp().getStatus();
    if (!HighsLpRelaxation::scaledOptimal(lp_status)) {
        return std::nullopt;
    }

    LpFprSetup s;
    s.budget = max_effort;

    s.csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_, mipdata->ARvalue_);

    s.incumbent_snapshot = mipdata->incumbent;
    s.binary = build_binary_mask(mipsolver);

    // Full-obj LP solution — direct reference to the solver's col_value
    // vector (stable while we run because we do not trigger further LP
    // solves during LP-FPR).
    const auto& lp_sol = mipdata->getLp().getLpSolver().getSolution().col_value;
    const double* lp_ptr = lp_sol.data();

    // Zero-obj analytic center (for Class 2 zerocore strategies).
    s.analytic_center =
        compute_analytic_center(mipsolver, /*use_objective=*/false, s.setup_lp_iterations);
    const double* ac_ptr = s.analytic_center.empty() ? lp_ptr : s.analytic_center.data();

    // Zero-obj LP vertex (for Class 3a zerolp strategies).
    s.zero_vertex = compute_zero_obj_vertex(mipsolver, s.setup_lp_iterations);
    const double* zv_ptr = s.zero_vertex.empty() ? lp_ptr : s.zero_vertex.data();

    s.arms.reserve(kNumLpArms);
    for (const auto& cfg : kClass2Configs) {
        s.arms.push_back({&cfg, ac_ptr});
    }
    for (const auto& cfg : kClass3aConfigs) {
        s.arms.push_back({&cfg, zv_ptr});
    }
    for (const auto& cfg : kClass3bConfigs) {
        s.arms.push_back({&cfg, lp_ptr});
    }

    // Precompute var_orders sequentially — required before any parallel
    // region because clique-based var_strategies call
    // HighsCliqueTable::cliquePartition which mutates internal state.
    s.var_orders.resize(kNumLpArms);
    const uint32_t base = heuristic_base_seed(mipsolver.options_mip_->random_seed);
    for (int i = 0; i < kNumLpArms; ++i) {
        // +200 offset spaces these seeds away from the presolve-FPR
        // var-order seeds (also derived from the same base) so the two
        // heuristics' RNG streams don't collide on small seed values.
        Rng rng(base + static_cast<uint32_t>(i) + 200);
        s.var_orders[i] = compute_var_order(mipsolver, s.arms[i].config->strat.var_strategy, rng,
                                            s.arms[i].lp_ref);
    }

    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// LpFprWorker: runs one LP-dependent FPR arm at a time
// ---------------------------------------------------------------------------

class LpFprWorker {
public:
    LpFprWorker(HighsMipSolver& mipsolver, const LpFprSetup& setup, IncumbentSink& sink,
                int arm_idx, uint32_t seed)
        : mipsolver_(mipsolver), setup_(setup), sink_(sink), arm_idx_(arm_idx), rng_(seed) {}

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

        initial_solution_buf_.clear();
        const double* init_ptr = nullptr;
        if (sink_.get_restart(rng_, initial_solution_buf_)) {
            init_ptr = initial_solution_buf_.data();
        }

        const LpArm& arm = setup_.arms[arm_idx_];
        const auto& var_order = setup_.var_orders[arm_idx_];

        FprConfig cfg{};
        cfg.max_effort = attempt_budget;
        cfg.hint = setup_.incumbent_snapshot.empty() ? nullptr : setup_.incumbent_snapshot.data();
        cfg.scores = nullptr;
        cfg.cont_fallback = nullptr;
        cfg.csc = &setup_.csc;
        cfg.mode = arm.config->mode;
        cfg.strategy = &arm.config->strat;
        cfg.lp_ref = arm.lp_ref;
        cfg.precomputed_var_order = var_order.data();
        cfg.precomputed_var_order_size = static_cast<HighsInt>(var_order.size());
        cfg.binary_mask = setup_.binary.data();
        cfg.scratch = &scratch_;

        auto result = fpr_attempt(mipsolver_, cfg, rng_, attempt_idx_, init_ptr);
        ++attempt_idx_;

        attempt.effort = result.effort;

        if (result.found_feasible) {
            sink_.offer(result.objective, result.solution);
            attempt.found_improvement = true;
            attempts_without_improvement_ = 0;
            randomizations_without_improvement_ = 0;
        } else {
            ++attempts_without_improvement_;
        }

        return attempt;
    }

    [[nodiscard]] bool finished() const { return finished_; }

private:
    void randomize_arm() { arm_idx_ = std::uniform_int_distribution<int>(0, kNumLpArms - 1)(rng_); }

    HighsMipSolver& mipsolver_;
    const LpFprSetup& setup_;
    IncumbentSink& sink_;

    int arm_idx_;
    int attempt_idx_ = 0;
    int attempts_without_improvement_ = 0;
    int randomizations_without_improvement_ = 0;
    bool finished_ = false;

    Rng rng_;
    // Per-worker scratch reused across fpr_attempt calls to avoid malloc
    // churn on the DFS + WalkSAT repair hot path.
    FprScratch scratch_;
    // Reused across attempts so the per-attempt pool restart does not
    // re-allocate an `ncol`-sized vector every call.  Mirrors
    // `FprWorker::initial_solution_buf_` in fpr.cpp.
    std::vector<double> initial_solution_buf_;

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

namespace {

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
    };

    return run_opportunistic_loop(
        exec, budget,
        [&](int worker_idx, Rng& /*rng*/) -> LpFprOppState {
            // Initial arm is worker_idx modulo the arm pool.
            int arm = worker_idx % kNumLpArms;
            uint32_t seed = exec.worker_seed(worker_idx);
            return LpFprOppState{std::make_unique<LpFprWorker>(mipsolver, setup, sink, arm, seed)};
        },
        [&](LpFprOppState& state, Rng& rng, size_t run_cap) -> AttemptResult {
            // A retired worker here hit its hard randomisation cap; the
            // replacement draws a fresh arm so the slot keeps contributing.
            return attempt_with_rebuild(state.worker, run_cap, [&]() {
                int arm = std::uniform_int_distribution<int>(0, kNumLpArms - 1)(rng);
                auto seed = static_cast<uint32_t>(rng());
                state.worker = std::make_unique<LpFprWorker>(mipsolver, setup, sink, arm, seed);
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

    // Suite gating: fpr_lp runs at mip_heuristic_suite=fpr and =all only.
    // suite=off must disable it so an off run is comparable to vanilla
    // HiGHS — this return sits above every read and write of
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
    // O(nnz) coefficients), and cap it at heuristic_effort_budget(nnz,
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
        static_cast<double>(heuristic_effort_budget(nnz, mipdata->heuristic_effort));
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
    // `now_s` reads the solver's timer so this window is comparable with
    // the `[Root]` timestamps (issue #95).
    EffortLedger ledger(mipsolver);
    const double t0_s = ledger.now_s();

    auto setup_opt = build_setup(mipsolver, max_effort);
    if (!setup_opt) {
        return;
    }
    auto& setup = *setup_opt;

    // The reference-LP solves are part of fpr_lp's spend: subtract them
    // from the worker budget so setup + workers together stay within
    // max_effort.  If setup ate (nearly) everything, skip the workers but
    // still charge the setup below.
    const auto setup_units = static_cast<size_t>(setup.setup_lp_iterations) * nnz;
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
        worker_effort =
            run_workers(setup, exec, make_budget(worker_budget, exec.num_workers), sink);
        // Read after the worker loop has joined; the sink starts at zero
        // because it is constructed per dispatch.
        found = sink.accepted() > 0;
    }

    // The ledger books the observability counter, emits the per-heuristic
    // log line, and charges the consumed work back to the shared RENS/RINS
    // envelope.  fpr_lp is the only caller of `charge_dive`; that envelope
    // depletion is what makes it compete for the vanilla heuristic budget
    // rather than draw unaccounted work.
    ledger.charge_dive("fpr_lp", worker_effort, found, setup.setup_lp_iterations, nnz, t0_s,
                       ledger.now_s());
}

DispatchCounts dispatch_counts() {
    return {g_dispatch_count.load(std::memory_order_relaxed)};
}

void reset_dispatch_counts() {
    g_dispatch_count.store(0, std::memory_order_relaxed);
}

}  // namespace fpr_lp
