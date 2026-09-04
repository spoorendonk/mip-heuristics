#include "contested_pdlp.h"
#include "fpr_core.h"
#include "fpr_var_order.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "incumbent_sink.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "pump_common.h"
#include "rng.h"
#include "scylla_worker.h"
#include "test_common.h"
#include "util/HighsInt.h"
#include "worker_base.h"

#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <utility>
#include <vector>

// ── Scylla as the only enabled heuristic ──
// `suite=scylla` clears FJ too, so a solution here can only have come from
// the pump chains.  There used to be three sections here — "standalone",
// "parallel" and "only" — distinguished by `mip_heuristic_opportunistic`
// (deleted in #92) and then by the per-heuristic bool flags (#93).  With
// neither left they are the same configuration, so the duplicates were
// removed rather than kept as six extra full HiGHS solves per suite run
// under names claiming distinct coverage.

TEST_CASE("Scylla standalone: flugpl general integers", "[heuristic][scylla]") {
    REQUIRE(solve_suite("flugpl.mps", "scylla") == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Scylla standalone: gt2 pure binary instance", "[heuristic][scylla]") {
    REQUIRE(solve_suite("gt2.mps", "scylla") == Catch::Approx(21166.0).epsilon(1e-3));
}

TEST_CASE("Scylla standalone: egout mixed integers", "[heuristic][scylla]") {
    REQUIRE(solve_suite("egout.mps", "scylla") == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── Sequential orchestrator: weighted effort allocation ──

TEST_CASE("Sequential orchestrator: flugpl weighted effort", "[heuristic][sequential]") {
    REQUIRE(solve_suite("flugpl.mps", "all") == Catch::Approx(1201500.0).epsilon(1e-6));
}

TEST_CASE("Sequential orchestrator: egout all arms", "[heuristic][sequential]") {
    REQUIRE(solve_suite("egout.mps", "all") == Catch::Approx(568.1007).epsilon(1e-4));
}

// ── Scylla stale-snapshot overlap (issue #76) ──
//
// Regression guard for the new `[ScyllaOverlap] fresh=<F> stale=<S>
// ratio=<R>` trace line emitted at the end of Scylla's parallel
// runners.  The line surfaces the #76 acceptance criterion — operators
// running with `log_dev_level=3` can read the overlap ratio from the
// log.  We assert the line is emitted at all and that `fresh >= 1`
// (Scylla ran at least one real solve).  Stale rounds are environment-
// dependent (contention between N workers fighting the PDLP mutex);
// on small instances the PDLP solve is fast enough that a single
// worker can finish before peers retry, so we don't require
// `stale > 0` as a hard assertion.  Coverage of the full stale
// branches is via the `ContestedPdlp` unit tests in
// `tests/test_contested_pdlp.cpp` plus MIPLIB bench runs.
TEST_CASE("Scylla overlap trace line: fresh count emitted (#76)", "[heuristic][scylla][overlap]") {
    const std::vector<std::string> lines = solve_capturing_log("flugpl.mps", [](Highs& h) {
        h.setOptionValue("log_dev_level", 3);
        set_suite(h, "scylla");
    });

    // Parse out the fresh / stale counts from the [ScyllaOverlap] line
    // so we assert the plumbing, not just the presence of a substring.
    const std::regex re(R"(\[ScyllaOverlap\] fresh=(\d+) stale=(\d+) ratio=([0-9.]+))");
    std::uint64_t fresh = 0;
    std::uint64_t stale = 0;
    bool seen = false;
    for (const auto& line : lines) {
        std::smatch match;
        if (std::regex_search(line, match, re)) {
            fresh = std::stoull(match[1].str());
            stale = std::stoull(match[2].str());
            seen = true;
            break;
        }
    }
    REQUIRE(seen);        // Line was emitted — closes #76's "new trace lines" ask.
    REQUIRE(fresh >= 1);  // Scylla actually ran at least one solve.
    (void)stale;          // Best-effort — see comment above.
}

// ── Cycle-history ring buffer (#126) ──
// `pump::record_cycle_entry` is the slot rule of Algorithm 1.1 line 13
// (Mexi, Besancon, Pokutta et al., *Scylla*).  It used to write
// `(K - 1) % kCycleWindow`, which overwrote the newest entry at the first
// wrap: the window at K4 was {K0, K1, K3} and at K5 {K1, K3, K4} instead
// of the last three iterates.  This drives six fresh rounds in the real
// order ScyllaWorker uses — record with the pre-increment counter, then
// increment — and pins the window contents after every one.  The old rule
// gets rounds 4 and 5 wrong; round 4 is where an execution dies, since
// `REQUIRE` aborts the case at the first failure.

TEST_CASE("Scylla cycle history keeps the last kCycleWindow iterates", "[heuristic][scylla]") {
    constexpr int kRounds = 6;
    std::vector<std::vector<double>> history;
    history.reserve(pump::kCycleWindow);
    int completed_rounds = 0;

    for (int round = 1; round <= kRounds; ++round) {
        // Distinguishable iterate: round 1 is {1.0}, round 2 is {2.0}, ...
        const std::vector<double> x{static_cast<double>(round)};
        pump::record_cycle_entry(history, completed_rounds, x);
        ++completed_rounds;

        const int expected_size = std::min(round, pump::kCycleWindow);
        REQUIRE(std::cmp_equal(history.size(), expected_size));

        // The window is the last `expected_size` iterates, as a set.
        std::vector<double> got;
        got.reserve(history.size());
        for (const auto& entry : history) {
            REQUIRE(entry.size() == 1);
            got.push_back(entry[0]);
        }
        std::ranges::sort(got);

        std::vector<double> want;
        want.reserve(static_cast<size_t>(expected_size));
        for (int r = round - expected_size + 1; r <= round; ++r) {
            want.push_back(static_cast<double>(r));
        }
        // `want` ends at `round`, so this pins the newest iterate's presence
        // too — the entry the old rule dropped at the wrap.
        REQUIRE(got == want);
    }
}

// ===================================================================
// Issue #121: the PDLP iterate reaches the rounding (Algorithm 1.1 line 12)
// ===================================================================
//
// `ScyllaWorker::run_attempt` builds its `FprConfig` with `cfg.lp_ref =
// x_bar.data()` and a `kFprConfigs` entry whose value strategy is
// `ValStrategy::kLp`. This drives that exact call shape (strategy +
// lp_ref + cont_fallback, all from `kFprConfigs`) at the `fpr_attempt`
// level directly, with two very different `x_bar` inputs, and — like the
// #120 test in test_fpr.cpp — via the begin/step lifecycle rather than
// the one-shot wrapper, since a single DFS attempt on these small bundled
// instances routinely does not complete within the `ncol+1` node budget.
namespace {

struct ScyllaProbe {
    std::vector<uint8_t> fixed;
    std::vector<double> sol;
};

ScyllaProbe round_once(HighsMipSolver& mipsolver, const CscMatrix& csc, const ProblemView& problem,
                       const NamedConfig& named, const double* x_bar, HighsInt ncol) {
    FprScratch scratch;
    FprConfig cfg{};
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.csc = &csc;
    cfg.mode = named.mode;
    cfg.strategy = &named.strat;
    cfg.lp_ref = x_bar;
    cfg.cont_fallback = x_bar;
    cfg.binary_mask = problem.binary.data();
    cfg.scratch = &scratch;
    Rng rng(2024);
    FprAttemptState state;
    fpr_attempt_begin(state, mipsolver, cfg, rng, /*attempt_idx=*/0);
    while (state.phase == FprAttemptState::Phase::kDfs) {
        fpr_attempt_step(state, mipsolver, cfg, rng, cfg.max_effort);
    }
    ScyllaProbe p;
    for (HighsInt j = 0; j < ncol; ++j) {
        p.fixed.push_back(scratch.prop_engine->var(j).fixed ? 1 : 0);
        p.sol.push_back(scratch.prop_engine->sol_data()[j]);
    }
    return p;
}

}  // namespace

TEST_CASE("Scylla: two different x_bar inputs round to different x_hat (#121)",
          "[heuristic][scylla][lp_ref]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    const HighsInt ncol = mipsolver->model_->num_col_;

    std::vector<double> x_bar_lo(mipsolver->model_->col_lower_);
    std::vector<double> x_bar_hi(mipsolver->model_->col_upper_);
    for (HighsInt j = 0; j < ncol; ++j) {
        if (x_bar_lo[j] <= -1e30) {
            x_bar_lo[j] = -1e5;
        }
        if (x_bar_hi[j] >= 1e30) {
            x_bar_hi[j] = 1e5;
        }
    }

    // Pool pre-seeded for narrative completeness, not as coverage: `fpr_attempt`
    // takes no `IncumbentSink&`, so nothing here can reach this call — the
    // restart fetch was removed from ScyllaWorker in #121, and the
    // `initial_solution` parameter it fed was removed in #122.
    IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);
    std::vector<double> adversarial(x_bar_lo.begin(), x_bar_lo.end());
    static_cast<void>(sink.offer(0.0, adversarial, WorkerTrace{0, 0}, 0));

    for (int i = 0; i < kNumFprConfigs; ++i) {
        INFO("kFprConfigs[" << i << "]");
        const ScyllaProbe with_lo =
            round_once(*mipsolver, csc, problem, kFprConfigs[i], x_bar_lo.data(), ncol);
        const ScyllaProbe with_hi =
            round_once(*mipsolver, csc, problem, kFprConfigs[i], x_bar_hi.data(), ncol);

        bool any_column_differs = false;
        for (HighsInt j = 0; j < ncol; ++j) {
            const auto idx = static_cast<size_t>(j);
            if (with_lo.fixed[idx] != with_hi.fixed[idx] ||
                (with_lo.fixed[idx] != 0 && with_lo.sol[idx] != with_hi.sol[idx])) {
                any_column_differs = true;
                break;
            }
        }
        REQUIRE(any_column_differs);
    }
}

// ===================================================================
// Issue #155: the pump advances on a failed FPR rounding
// ===================================================================
//
// Mexi et al., arXiv 2307.03466v2, Algorithm 1.1:
//
//     12  x_hat(K) = fix-and-propagate(x_bar(k))
//     13  if x_hat(K) MIP-feasible then return x_hat(K)
//     14  if cycling on x_hat(K) detected then x_hat(K) = perturb(x_hat(K))
//     15  c_hat = alpha^K sqrt(|I|)/||c|| c + (1 - alpha^K) Delta(x, x_hat(K))
//     16  K = K + 1
//
// No branch skips 14-16, and Sect. 2.3 is explicit that line 12 always
// produces an integer vector even when a domain went empty.  A feasibility
// pump is *defined* by being pulled toward the rounded point whether or
// not it is feasible.  `ScyllaWorker::run_attempt` used to `continue` past
// the whole block whenever the rounding returned no solution, so
// `modified_cost_` was never rewritten and the next round re-solved a
// byte-identical LP from a byte-identical warm start.
namespace {

// `2*x0 + 2*x1 = 1`, both binary: LP-feasible, integer-infeasible.  No
// rounding of it can succeed, so *every* round in this test is a failed
// one and the observable below is attributable to the failure path alone.
// A zero objective makes the assertion exact rather than approximate:
// `orig_cost` is the all-zero vector, and one pass of
// `pump::compute_pump_objective` writes `(1 - alpha_K) * (1 - 2*x_hat[j])`
// = +-0.1 into every binary column, so solve #2's cost differs from solve
// #1's in every entry the moment line 15 runs at all.
void build_parity_pump_mip(Highs& highs) {
    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);
    const auto idx = std::to_array<HighsInt>({0, 1});
    const auto val = std::to_array<double>({2.0, 2.0});
    highs.addRow(1.0, 1.0, 2, idx.data(), val.data());
}

// Mirrors `build_bare_mipsolver` (test_common.h) minus the `readModel`:
// no bundled instance is guaranteed to be integer-infeasible, and this
// case needs one.
std::unique_ptr<HighsMipSolver> bare_mipsolver_on_parity(Highs& highs, HighsCallback& cb) {
    build_parity_pump_mip(highs);
    // `Highs::addRow` leaves the matrix row-wise; round-trip through
    // `passModel`, whose master overload calls `ensureColwise()`.
    REQUIRE(highs.passModel(highs.getLp()) == HighsStatus::kOk);
    highs.setOptionValue("presolve", "off");
    require_option(highs, "time_limit", kHighsInf);
    auto mipsolver = std::make_unique<HighsMipSolver>(cb, highs.getOptions(), highs.getLp(),
                                                      highs.getSolution());
    mipsolver->timer_.start();
    mipsolver->improving_solution_file_ = nullptr;
    mipsolver->mipdata_ = std::make_unique<HighsMipSolverData>(*mipsolver);
    mipsolver->mipdata_->init();
    mipsolver->mipdata_->runMipPresolve(mipsolver->options_mip_->presolve_reduction_limit);
    mipsolver->mipdata_->runSetup();
    mipsolver->mipdata_->workers.emplace_back(
        *mipsolver, &mipsolver->mipdata_->getLp(), &mipsolver->mipdata_->getDomain(),
        &mipsolver->mipdata_->getCutPool(), &mipsolver->mipdata_->getConflictPool(),
        &mipsolver->mipdata_->getPseudoCost());
    return mipsolver;
}

// Records what each PDLP solve was asked to solve, then delegates to the
// real thing.  `solve_locked` is protected virtual, so this needs no new
// seam; and it goes over the *real* constructor rather than the
// `ForTesting` one, because that one leaves `nnz_lp_ == 0` and
// `ScyllaWorker`'s constructor retires on it.
class RecordingPdlp : public ContestedPdlp {
public:
    RecordingPdlp(HighsMipSolver& mipsolver, HighsInt pdlp_iter_cap)
        : ContestedPdlp(mipsolver, pdlp_iter_cap) {}

    struct Call {
        std::vector<double> modified_cost;
        std::vector<double> warm_start_col_value;
        bool warm_start_valid = false;
    };

    // Written under `mu_` (the caller holds it across `solve_locked`), and
    // read only after `run_attempt` has returned on the single test thread.
    //
    // Note before copying this pattern into a threaded test: the two
    // vector copies below happen *inside* the PDLP mutex, so every other
    // worker's chain waits on them.  Harmless here -- this fixture is
    // single-threaded -- but the wrapper's whole contract is that the
    // mutex covers the solve and nothing avoidable beside it.
    std::vector<Call> calls;

protected:
    SolveResult solve_locked(const std::vector<double>& modified_cost,
                             const std::vector<double>& warm_start_col_value,
                             const std::vector<double>& warm_start_row_dual, bool warm_start_valid,
                             double epsilon, double time_limit) override {
        calls.push_back({modified_cost, warm_start_col_value, warm_start_valid});
        return ContestedPdlp::solve_locked(modified_cost, warm_start_col_value, warm_start_row_dual,
                                           warm_start_valid, epsilon, time_limit);
    }
};

}  // namespace

TEST_CASE("Scylla: the pump advances on a failed FPR rounding (#155)",
          "[heuristic][scylla][pump]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on_parity(highs, cb);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    ExecutionContext exec = make_exec(*mipsolver);
    IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);

    RecordingPdlp pdlp(*mipsolver, /*pdlp_iter_cap=*/1000);
    REQUIRE(pdlp.initialized());

    // The dispatching-thread rule (#99): `compute_var_order` reads the live
    // root domain and the clique table, so it runs here and not inside the
    // worker.  This is what `scylla::precompute_config_var_orders` does for
    // production; that function is file-local to scylla.cpp.
    std::vector<std::vector<HighsInt>> var_orders;
    var_orders.reserve(kNumFprConfigs);
    Rng order_rng(11);
    for (int i = 0; i < kNumFprConfigs; ++i) {
        var_orders.push_back(
            compute_var_order(*mipsolver, kFprConfigs[i].strat.var_strategy, order_rng, nullptr));
    }

    // One worker, so every round takes the mutex uncontended and is
    // `fresh` — the guard the pump-state block sits behind.
    constexpr size_t kBudget = size_t{1} << 20;
    ScyllaWorker worker(*mipsolver, exec, pdlp, csc, sink, problem.binary.data(), var_orders,
                        /*total_budget=*/kBudget,
                        /*stale_budget=*/std::numeric_limits<size_t>::max(),
                        /*seed=*/7, /*worker_idx=*/0, /*num_workers=*/1, WorkerTrace{0, 0});
    static_cast<void>(worker.run_attempt(kBudget));

    // Two fresh solves is the premise; without them there is nothing to
    // compare.  `absorb_fresh_solve` retires the chain on `kError`,
    // `kInfeasible`, an invalid primal, or `pump::kMaxPdlpStalls`
    // consecutive zero-iteration solves — the last of which still allows
    // three solves, so the first two are always observed.
    REQUIRE(worker.fresh_solves() >= 2);
    REQUIRE(pdlp.calls.size() >= 2);

    // No offer was ever made: the model is integer-infeasible, so every
    // rounding failed, and `found_feasible` still gates every `offer` call
    // site.  This is what makes the difference below attributable to the
    // fix rather than to a lucky feasible round.
    REQUIRE(sink.accepted() == 0);

    // Printed by Catch2 only when an assertion below fails, which is
    // exactly the pre-#155 case: the two cost vectors came back
    // byte-identical, and so did the warm start.
    CAPTURE(pdlp.calls[0].modified_cost, pdlp.calls[1].modified_cost,
            pdlp.calls[0].warm_start_col_value, pdlp.calls[1].warm_start_col_value);

    // Solve #1 gets the original objective (`modified_cost_` is
    // initialised to `orig_cost` in the constructor)...
    REQUIRE(pdlp.calls[0].modified_cost == mipsolver->model_->col_cost_);
    // ...and solve #2 must not.  Before #155 the failed rounding skipped
    // lines 14-16 wholesale, so this vector came back byte-identical and
    // the chain re-solved the same LP from the same warm start forever.
    REQUIRE(pdlp.calls[1].modified_cost != pdlp.calls[0].modified_cost);

    // The objective blend is only half of what a failed rounding must
    // reach.  Algorithm 1.1 line 13 (`if x_hat MIP-feasible then return`)
    // is the *only* branch between the rounding and lines 14-16, and it
    // exits the loop -- so a rounding that is not MIP-feasible falls
    // straight into line 14's cycling check, and Sect. 2.3's "always
    // produces an integer-feasible, but not necessarily LP-feasible,
    // solution" is exactly the statement that such an `x_hat` is a
    // legitimate cycling operand.  The assertion above cannot see this:
    // `compute_pump_objective` alone moves the cost.  So pin the operand
    // the blend does not move -- one recorded iterate per fresh round,
    // every one of them from a failed rounding.  Without it, two
    // identical failed roundings stay invisible to `detect_cycling` and
    // the perturbation that exists to break them never fires.
    // `fresh_solves() == K_` is a property of *this fixture*, not of the
    // class: `fresh_solves_` is bumped before both the MIP-feasible fast
    // path and the exhausted-budget break, neither of which reaches
    // `++K_`.  Here neither can fire -- the model is integer-infeasible so
    // the fast path is unreachable, and the budget is orders above what
    // three rounds spend -- so the identity holds and is the sharper
    // assertion.  The `>= 2` below is the part that survives if that ever
    // stops being true.
    CHECK(worker.cycle_history_size_for_test() ==
          std::min<size_t>(static_cast<size_t>(worker.fresh_solves()), pump::kCycleWindow));
    CHECK(worker.cycle_history_size_for_test() >= 2);
}
