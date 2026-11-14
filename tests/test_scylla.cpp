#include "contested_pdlp.h"
#include "fpr_core.h"
#include "fpr_var_order.h"
#include "heuristic_context.h"
#include "Highs.h"
#include "highs_context.h"
#include "incumbent_sink.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsParallel.h"
#include "pump_common.h"
#include "rng.h"
#include "scylla.h"
#include "scylla_worker.h"
#include "test_common.h"
#include "util/HighsInt.h"
#include "worker_base.h"

#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <regex>
#include <span>
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
        select_heuristics(h, "scylla");
        enable_scylla(h);
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

ScyllaProbe round_once(HighsMipSolver& mipsolver, const ProblemView& problem,
                       const NamedConfig& named, const double* x_bar, HighsInt ncol) {
    FprScratch scratch;
    FprConfig cfg{};
    std::vector<HighsInt> var_order;
    cfg.max_effort = std::numeric_limits<size_t>::max() / 2;
    cfg.mode = named.mode;
    cfg.strategy = &named.strat;
    cfg.lp_ref = x_bar;
    cfg.cont_fallback = x_bar;
    cfg.scratch = &scratch;
    Rng rng(2024);
    FprAttemptState state;
    set_var_order(cfg, var_order, mipsolver, rng);
    fpr_attempt_begin(state, problem, deadline_of(mipsolver), cfg, rng, /*attempt_idx=*/0);
    while (state.phase == FprAttemptState::Phase::kDfs) {
        fpr_attempt_step(state, problem, deadline_of(mipsolver), cfg, rng, cfg.max_effort);
    }
    ScyllaProbe p;
    for (HighsInt j = 0; j < ncol; ++j) {
        p.fixed.push_back(require_engaged(scratch.prop_engine).var(j).fixed ? 1 : 0);
        p.sol.push_back(require_engaged(scratch.prop_engine).sol_data()[j]);
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
            round_once(*mipsolver, problem, kFprConfigs[i], x_bar_lo.data(), ncol);
        const ScyllaProbe with_hi =
            round_once(*mipsolver, problem, kFprConfigs[i], x_bar_hi.data(), ncol);

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
// #1's in every entry the moment line 15 runs at all.  `rhs = 2` makes it
// feasible (one of the two at 1), for the cycle escape's test (#297).
void build_parity_pump_mip(Highs& highs, double rhs) {
    highs.addVar(0.0, 1.0);
    highs.addVar(0.0, 1.0);
    highs.changeColIntegrality(0, HighsVarType::kInteger);
    highs.changeColIntegrality(1, HighsVarType::kInteger);
    const auto idx = std::to_array<HighsInt>({0, 1});
    const auto val = std::to_array<double>({2.0, 2.0});
    highs.addRow(rhs, rhs, 2, idx.data(), val.data());
}

// Mirrors `build_bare_mipsolver` (test_common.h) minus the `readModel`:
// no bundled instance is guaranteed to be integer-infeasible, and this
// case needs one.
std::unique_ptr<HighsMipSolver> bare_mipsolver_on_parity(Highs& highs, HighsCallback& cb,
                                                         double rhs = 1.0) {
    build_parity_pump_mip(highs, rhs);
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
                             double epsilon, double time_limit,
                             std::span<const double> aux_row_lower) override {
        calls.push_back({modified_cost, warm_start_col_value, warm_start_valid});
        return ContestedPdlp::solve_locked(modified_cost, warm_start_col_value, warm_start_row_dual,
                                           warm_start_valid, epsilon, time_limit, aux_row_lower);
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
    ScyllaWorker worker(*mipsolver, problem, exec, pdlp, sink, var_orders,
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

// ── cuOpt projection (#290, `mip_heuristic_scylla_cuopt_pump`) ──

TEST_CASE("cuOpt projection: distance per column kind and the blend", "[scylla][cuopt]") {
    // Columns: binary rounded to 0, binary rounded to 1, general integer
    // [0, 10] rounded to 4 (interior, aux column 0), continuous.
    const std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger,
                                                   HighsVarType::kInteger,
                                                   HighsVarType::kContinuous};
    const std::vector<double> lb = {0.0, 0.0, 0.0, 0.0};
    const std::vector<double> ub = {1.0, 1.0, 10.0, 5.0};
    const std::vector<HighsInt> aux_cols = {2};
    std::vector<double> cost;
    std::vector<double> rows;

    SECTION("pure distance: c = 0 drops the objective term") {
        const std::vector<double> c = {0.0, 0.0, 0.0, 0.0};
        pump::cuopt::build_projection(c, std::vector<double>{0.0, 1.0, 4.0, 2.5}, integrality, lb,
                                      ub, aux_cols, 0.5, 1e-6, cost, rows);
        const double w = 0.5 / std::numbers::sqrt3;
        REQUIRE(cost.size() == 5);
        CHECK(cost[0] == Catch::Approx(w));   // at lower bound: +1
        CHECK(cost[1] == Catch::Approx(-w));  // at upper bound: -1
        CHECK(cost[2] == 0.0);                // interior: through its aux column
        CHECK(cost[3] == 0.0);                // continuous
        CHECK(cost[4] == Catch::Approx(w));   // aux d_0, cost 1 before scaling
        REQUIRE(rows.size() == 2);
        CHECK(rows[0] == -4.0);  // d - x >= -x_hat
        CHECK(rows[1] == 4.0);   // d + x >= x_hat
    }
    SECTION("general integer at a bound: no aux cost, rows relaxed, objective blended") {
        const std::vector<double> c = {3.0, 0.0, 0.0, 4.0};
        pump::cuopt::build_projection(c, std::vector<double>{0.0, 1.0, 10.0, 2.5}, integrality, lb,
                                      ub, aux_cols, 0.5, 1e-6, cost, rows);
        const double w = 0.5 / std::numbers::sqrt3;
        CHECK(cost[0] == Catch::Approx(w + (0.5 * 3.0 / 5.0)));
        CHECK(cost[1] == Catch::Approx(-w));
        CHECK(cost[2] == Catch::Approx(-w));  // at its upper bound
        CHECK(cost[3] == Catch::Approx(0.5 * 4.0 / 5.0));
        CHECK(cost[4] == 0.0);
        CHECK(rows[0] == -kHighsInf);
        CHECK(rows[1] == -kHighsInf);
    }
}

TEST_CASE("cuOpt projection: tolerance schedule thresholds", "[scylla][cuopt]") {
    using namespace pump::cuopt;
    CHECK(lp_tolerance(0.0) == 0.1);
    CHECK(lp_tolerance(0.7999) == 0.1);
    CHECK(lp_tolerance(0.80) == 0.01);
    CHECK(lp_tolerance(0.9299) == 0.01);
    CHECK(lp_tolerance(0.93) == 1e-3);
    CHECK(lp_tolerance(0.9699) == 1e-3);
    CHECK(lp_tolerance(0.97) == pump::kEpsilonFloor);
    CHECK(lp_tolerance(1.0) == pump::kEpsilonFloor);
    const std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger,
                                                   HighsVarType::kContinuous};
    CHECK(integral_fraction(std::vector<double>{1.0, 0.5, 0.3}, integrality, 1e-6) == 0.5);
}

namespace {

// Keeps every solve's full LP-space primal: the aux values `ContestedPdlp`
// cuts off before a caller sees the point.
class AuxPdlp : public ContestedPdlp {
public:
    AuxPdlp(HighsMipSolver& mipsolver, HighsInt pdlp_iter_cap)
        : ContestedPdlp(mipsolver, pdlp_iter_cap, /*aux_distance=*/true) {}
    std::vector<double> last_full_primal;

protected:
    SolveResult solve_locked(const std::vector<double>& modified_cost,
                             const std::vector<double>& warm_start_col_value,
                             const std::vector<double>& warm_start_row_dual, bool warm_start_valid,
                             double epsilon, double time_limit,
                             std::span<const double> aux_row_lower) override {
        auto result =
            ContestedPdlp::solve_locked(modified_cost, warm_start_col_value, warm_start_row_dual,
                                        warm_start_valid, epsilon, time_limit, aux_row_lower);
        last_full_primal = result.col_value;
        return result;
    }
};

}  // namespace

TEST_CASE("cuOpt cycles: the distance detector needs a full window", "[scylla][cuopt]") {
    using namespace pump::cuopt;
    // One short of the window: never a cycle, however flat.
    std::vector<double> prev(kDistanceWindow - 1, 10.0);
    CHECK_FALSE(distance_cycle(prev, 10.0));
    prev.push_back(10.0);
    // Full: avg 10, a cycle once cur is above avg - 0.1 * avg = 9.
    CHECK(distance_cycle(prev, 10.0));
    CHECK(distance_cycle(prev, 9.5));
    CHECK_FALSE(distance_cycle(prev, 8.5));
    // All zero: avg - cur < 0 never holds.
    CHECK_FALSE(distance_cycle(std::vector<double>(kDistanceWindow, 0.0), 0.0));
    // The ring keeps the last `kDistanceWindow` and drops the oldest.
    std::vector<double> ring;
    ring.reserve(kDistanceWindow);
    size_t next = 0;
    for (int i = 0; i < kDistanceWindow + 5; ++i) {
        ring_push(ring, next, kDistanceWindow, static_cast<double>(i));
    }
    CHECK(ring.size() == static_cast<size_t>(kDistanceWindow));
    CHECK(std::ranges::find(ring, 4.0) == ring.end());
    CHECK(std::ranges::find(ring, 5.0) != ring.end());
}

TEST_CASE("cuOpt cycles: the assignment ring and its alpha gate", "[scylla][cuopt]") {
    using namespace pump::cuopt;
    const std::vector<HighsVarType> integrality = {
        HighsVarType::kInteger, HighsVarType::kContinuous, HighsVarType::kInteger};
    // Equal on the integer columns, whatever the continuous one holds.
    CHECK(assignment_hash(std::vector<double>{1.0, 0.25, -0.0}, integrality) ==
          assignment_hash(std::vector<double>{1.0, 7.0, 0.0}, integrality));
    CHECK(assignment_hash(std::vector<double>{1.0, 0.0, 2.0}, integrality) !=
          assignment_hash(std::vector<double>{2.0, 0.0, 1.0}, integrality));

    // A rounding `kAssignmentRing` rounds back is still found; one more back
    // is not.
    std::vector<uint64_t> ring;
    ring.reserve(kAssignmentRing);
    size_t next = 0;
    const auto h = [&](int i) {
        return assignment_hash(std::vector<double>{static_cast<double>(i), 0.0, 0.0}, integrality);
    };
    for (int i = 0; i < kAssignmentRing; ++i) {
        ring_push(ring, next, kAssignmentRing, h(i));
    }
    CHECK(in_ring(ring, h(0)));
    ring_push(ring, next, kAssignmentRing, h(kAssignmentRing));
    CHECK_FALSE(in_ring(ring, h(0)));
    CHECK(in_ring(ring, h(1)));

    // The gate: a/0.9 - a = a/9 < 0.005, i.e. a < 0.045.
    CHECK_FALSE(assignment_check_due(kAlphaInit));
    CHECK_FALSE(assignment_check_due(0.05));
    CHECK(assignment_check_due(0.04));
}

TEST_CASE("cuOpt cycles: the perturbation stays in bounds and on integers", "[scylla][cuopt]") {
    const std::vector<HighsVarType> integrality = {HighsVarType::kInteger, HighsVarType::kInteger,
                                                   HighsVarType::kContinuous,
                                                   HighsVarType::kInteger, HighsVarType::kInteger};
    const std::vector<double> lb = {0.0, -3.0, 0.0, -kHighsInf, 5.0};
    const std::vector<double> ub = {1.0, 4.0, 10.0, kHighsInf, 5.0};
    const std::vector<double> x0 = {1.0, 2.0, 3.7, 100.0, 5.0};
    Rng rng(3);
    int changed = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<double> x = x0;
        pump::cuopt::perturb(x, integrality, lb, ub, rng);
        CHECK(x[2] == x0[2]);
        CHECK(x[4] == 5.0);
        for (const int j : {0, 1, 3}) {
            CHECK(x[j] == std::round(x[j]));
            CHECK(x[j] >= lb[j]);
            CHECK(x[j] <= ub[j]);
        }
        // The free column draws from the window around its value.
        CHECK(std::abs(x[3] - x0[3]) <= kInfBoundShiftWindow);
        changed += static_cast<int>(x != x0);
    }
    // The mechanism, not a rate: some trial redraws a column to a new value.
    CHECK(changed > 0);
}

TEST_CASE("cuOpt projection: the aux LP's distance is the L1 distance at an optimum",
          "[scylla][cuopt][pump]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "flugpl.mps");
    AuxPdlp pdlp(*mipsolver, /*pdlp_iter_cap=*/100000);
    REQUIRE(pdlp.initialized());
    const auto& aux = pdlp.aux_cols();
    REQUIRE_FALSE(aux.empty());  // flugpl has general integers
    const auto* model = mipsolver->model_;
    const HighsInt ncol = model->num_col_;

    // Round the relaxation's point; aux columns rounded strictly inside
    // their bounds get live rows.
    std::vector<double> c0 = model->col_cost_;
    c0.resize(ncol + aux.size(), 0.0);
    auto relax = pdlp.solve(c0, {}, {}, false, 1e-8);
    REQUIRE(relax.value_valid);
    std::vector<double> x_hat = relax.col_value;
    for (HighsInt j = 0; j < ncol; ++j) {
        if (is_integer(model->integrality_, j)) {
            x_hat[j] = std::round(x_hat[j]);
        }
    }
    std::vector<double> cost;
    std::vector<double> rows;
    pump::cuopt::build_projection(model->col_cost_, x_hat, model->integrality_, model->col_lower_,
                                  model->col_upper_, aux, /*alpha=*/0.0, 1e-6, cost, rows);
    auto proj = pdlp.solve(cost, {}, {}, false, 1e-8, rows);
    REQUIRE(proj.value_valid);
    REQUIRE(proj.col_value.size() == static_cast<size_t>(ncol));  // aux cut off
    REQUIRE(pdlp.last_full_primal.size() == ncol + aux.size());
    int live = 0;
    for (size_t k = 0; k < aux.size(); ++k) {
        if (rows[2 * k] == -kHighsInf) {
            continue;
        }
        ++live;
        const HighsInt j = aux[k];
        CHECK(pdlp.last_full_primal[ncol + k] ==
              Catch::Approx(std::abs(proj.col_value[j] - x_hat[j])).margin(1e-5));
    }
    CHECK(live > 0);
}

namespace {

// Drives one cuOpt-variant chain attempt by attempt and checks alpha after
// each one.  An attempt of `n` rounds with no improvement leaves alpha `n`
// decays below where it started.  One with an improvement resets it to
// `kAlphaInit` before that round's decay, so it ends within `n` decays of
// `kAlphaInit`, exactly one when `n == 1`.  Returns how many `n == 1`
// improving attempts came after the first projection: those are the ones
// where a missing reset would show (alpha two or more decays down).
int drive_cuopt_alpha_chain(const char* inst, int worker_idx) {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_scylla_cuopt_pump", true);
    HighsCallback cb(&highs);
    auto mipsolver = inst == nullptr ? bare_mipsolver_on_parity(highs, cb)
                                     : build_bare_mipsolver(highs, cb, inst);
    REQUIRE(mipsolver->options_mip_->mip_heuristic_scylla_cuopt_pump);

    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    ExecutionContext exec = make_exec(*mipsolver);
    IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);
    ContestedPdlp pdlp(*mipsolver, /*pdlp_iter_cap=*/1000, /*aux_distance=*/true);
    REQUIRE(pdlp.initialized());
    std::vector<std::vector<HighsInt>> var_orders;
    var_orders.reserve(kNumFprConfigs);
    Rng order_rng(11);
    for (int i = 0; i < kNumFprConfigs; ++i) {
        var_orders.push_back(
            compute_var_order(*mipsolver, kFprConfigs[i].strat.var_strategy, order_rng, nullptr));
    }
    constexpr size_t kBudget = size_t{1} << 20;
    ScyllaWorker worker(*mipsolver, problem, exec, pdlp, sink, var_orders, kBudget,
                        std::numeric_limits<size_t>::max(), /*seed=*/7, worker_idx,
                        /*num_workers=*/1, WorkerTrace{0, 0});
    CHECK(worker.cuopt_alpha_for_test() == pump::cuopt::kAlphaInit);
    const double a1 = pump::cuopt::kAlphaInit * pump::cuopt::kAlphaDecay;
    int resets_after_first = 0;
    for (int i = 0; i < 40 && !worker.finished(); ++i) {
        const double alpha_before = worker.cuopt_alpha_for_test();
        const int rounds_before = worker.pump_rounds_for_test();
        const AttemptResult res = worker.run_attempt(1000);
        const int n = worker.pump_rounds_for_test() - rounds_before;
        if (!worker.finished()) {
            // From the first solve on (one worker: every attempt solves
            // fresh), the warm primal spans model and aux columns, also
            // before any projection, so `solve_locked` never drops it and
            // the row duals with it.
            CHECK(worker.cuopt_warm_primal_size_for_test() ==
                  static_cast<size_t>(mipsolver->model_->num_col_) + pdlp.aux_cols().size());
        }
        if (n == 0) {
            continue;
        }
        const double alpha = worker.cuopt_alpha_for_test();
        if (!res.found_improvement) {
            CHECK(alpha == Catch::Approx(alpha_before * std::pow(pump::cuopt::kAlphaDecay, n)));
        } else if (n == 1) {
            CHECK(alpha == Catch::Approx(a1));
            if (rounds_before >= 1) {
                ++resets_after_first;
            }
        } else {
            CHECK(alpha >= a1 * std::pow(pump::cuopt::kAlphaDecay, n - 1) * (1.0 - 1e-12));
        }
    }
    // The variant's cycle state (#295) replaces Mexi's window and stays
    // within its rings' capacities.
    CHECK(worker.cycle_history_size_for_test() == 0);
    CHECK(worker.cuopt_assignments_size_for_test() <=
          static_cast<size_t>(pump::cuopt::kAssignmentRing));
    CHECK(worker.cuopt_distances_size_for_test() <=
          static_cast<size_t>(pump::cuopt::kDistanceWindow));
    if (inst == nullptr) {
        CHECK(worker.pump_rounds_for_test() >= 2);
        // Integer-infeasible: every rounding is infeasible, so each fresh
        // round enters the ring.
        CHECK(worker.cuopt_assignments_size_for_test() >= 1);
    }
    return resets_after_first;
}

}  // namespace

TEST_CASE("cuOpt projection: alpha decays per projection and resets on improvement",
          "[scylla][cuopt][pump]") {
    highs::parallel::initialize_scheduler();
    // Integer-infeasible: nothing improves, so alpha only decays.
    CHECK(drive_cuopt_alpha_chain(nullptr, 0) == 0);
    // Feasible instances, every chain: each attempt's alpha must match the
    // relation above.  None of these chains improves after its first
    // projection, so the reset's discriminating case is not reached here
    // (#290 notes the gap); an improving attempt is still checked exactly.
    for (const char* inst : {"bell5.mps", "p0548.mps"}) {
        for (int w = 0; w < kNumFprConfigs; ++w) {
            static_cast<void>(drive_cuopt_alpha_chain(inst, w));
        }
    }
}

// Drives `handle_cuopt_cycles` with constructed points on the two-binary
// parity model: the order of checks and pushes, and what each cycle clears.
TEST_CASE("cuOpt cycles: the worker's detectors, perturbation and restarts",
          "[scylla][cuopt][pump]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_scylla_cuopt_pump", true);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on_parity(highs, cb);
    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    ExecutionContext exec = make_exec(*mipsolver);
    IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);
    ContestedPdlp pdlp(*mipsolver, /*pdlp_iter_cap=*/1000, /*aux_distance=*/true);
    REQUIRE(pdlp.initialized());
    std::vector<std::vector<HighsInt>> var_orders;
    var_orders.reserve(kNumFprConfigs);
    Rng order_rng(11);
    for (int i = 0; i < kNumFprConfigs; ++i) {
        var_orders.push_back(
            compute_var_order(*mipsolver, kFprConfigs[i].strat.var_strategy, order_rng, nullptr));
    }
    ScyllaWorker worker(*mipsolver, problem, exec, pdlp, sink, var_orders, size_t{1} << 20,
                        std::numeric_limits<size_t>::max(), /*seed=*/7, 0,
                        /*num_workers=*/1, WorkerTrace{0, 0});
    const std::vector<double> x_bar = {0.5, 0.5};  // L1 distance 1 to any 0/1 point
    const auto round = [&](std::vector<double> x_hat, bool improved = false) {
        worker.handle_cuopt_cycles(x_hat, x_bar, improved);
        return x_hat;
    };

    SECTION("an assignment cycle needs a settled alpha and a repeat") {
        // The first rounding has no rounding before it: no distance.
        round({0.0, 0.0});
        CHECK(worker.cuopt_assignments_size_for_test() == 1);
        CHECK(worker.cuopt_distances_size_for_test() == 0);
        // A repeat while alpha still moves: recorded, not a cycle.
        round({0.0, 0.0});
        CHECK(worker.cuopt_assignments_size_for_test() == 2);
        CHECK(worker.cuopt_distances_size_for_test() == 1);
        worker.set_cuopt_alpha_for_test(0.01);
        // Settled, but new: checked before it is pushed, so not a cycle.
        round({1.0, 0.0});
        CHECK(worker.cuopt_assignments_size_for_test() == 3);
        CHECK(worker.cuopt_distances_size_for_test() == 2);
        // Settled and a repeat: a cycle.  The FJ escape runs and, on this
        // integer-infeasible model, finds nothing, so the cycle persists:
        // the rounding and its perturbation both enter the ring, and the
        // distances are cleared.
        std::vector<double> out = {0.0, 0.0};
        const AttemptResult escape = worker.handle_cuopt_cycles(out, x_bar, false);
        CHECK(escape.effort > 0);
        CHECK_FALSE(escape.found_improvement);
        CHECK(worker.cuopt_assignments_size_for_test() == 5);
        CHECK(worker.cuopt_distances_size_for_test() == 0);
        for (const double v : out) {
            CHECK((v == 0.0 || v == 1.0));
        }
        // The distance window restarts from empty.
        round({1.0, 1.0});
        CHECK(worker.cuopt_distances_size_for_test() == 1);
    }

    SECTION("a distance cycle fires on a full window, not one short") {
        round({0.0, 0.0});
        for (int i = 0; i < pump::cuopt::kDistanceWindow; ++i) {
            round({0.0, 0.0});
        }
        // Every check so far saw fewer than `kDistanceWindow` distances.
        CHECK(worker.cuopt_distances_size_for_test() ==
              static_cast<size_t>(pump::cuopt::kDistanceWindow));
        const size_t ring_before = worker.cuopt_assignments_size_for_test();
        // Full and flat: a cycle.  The rounding is in the ring, so it is
        // perturbed and its result pushed once; the window is cleared.
        round({0.0, 0.0});
        CHECK(worker.cuopt_distances_size_for_test() == 0);
        CHECK(worker.cuopt_assignments_size_for_test() ==
              std::min(ring_before + 1, static_cast<size_t>(pump::cuopt::kAssignmentRing)));
    }

    SECTION("an improving round skips the detectors") {
        round({0.0, 0.0});
        round({0.0, 0.0});
        worker.set_cuopt_alpha_for_test(0.01);
        round({0.0, 0.0}, /*improved=*/true);
        CHECK(worker.cuopt_assignments_size_for_test() == 2);
        CHECK(worker.cuopt_distances_size_for_test() == 1);
    }
}

// The FJ escape (#297) on the feasible parity model `2*x0 + 2*x1 = 2`: an
// assignment cycle at the infeasible rounding (0, 0) reaches FJ before any
// perturbation, and FJ's feasible point replaces the rounding, improves the
// incumbent and restarts the descent.  No perturbation follows: FJ's point
// is not in the ring.
TEST_CASE("cuOpt cycles: a cycle reaches the FJ escape before perturbing",
          "[scylla][cuopt][pump]") {
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_scylla_cuopt_pump", true);
    HighsCallback cb(&highs);
    auto mipsolver = bare_mipsolver_on_parity(highs, cb, /*rhs=*/2.0);
    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    ExecutionContext exec = make_exec(*mipsolver);
    IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);
    ContestedPdlp pdlp(*mipsolver, /*pdlp_iter_cap=*/1000, /*aux_distance=*/true);
    REQUIRE(pdlp.initialized());
    std::vector<std::vector<HighsInt>> var_orders;
    var_orders.reserve(kNumFprConfigs);
    Rng order_rng(11);
    for (int i = 0; i < kNumFprConfigs; ++i) {
        var_orders.push_back(
            compute_var_order(*mipsolver, kFprConfigs[i].strat.var_strategy, order_rng, nullptr));
    }
    ScyllaWorker worker(*mipsolver, problem, exec, pdlp, sink, var_orders, size_t{1} << 20,
                        std::numeric_limits<size_t>::max(), /*seed=*/7, 0,
                        /*num_workers=*/1, WorkerTrace{0, 0});
    const std::vector<double> x_bar = {0.5, 0.5};
    std::vector<double> x_hat = {0.0, 0.0};
    CHECK(worker.handle_cuopt_cycles(x_hat, x_bar, false).effort == 0);
    x_hat = {0.0, 0.0};
    CHECK(worker.handle_cuopt_cycles(x_hat, x_bar, false).effort == 0);
    worker.set_cuopt_alpha_for_test(0.01);
    x_hat = {0.0, 0.0};
    const AttemptResult escape = worker.handle_cuopt_cycles(x_hat, x_bar, false);
    CHECK(escape.effort > 0);
    CHECK(escape.found_improvement);
    CHECK(x_hat[0] + x_hat[1] == 1.0);
    // The cycle's rounding and FJ's point, not a perturbation of either.
    CHECK(worker.cuopt_assignments_size_for_test() == 4);
    CHECK(worker.cuopt_distances_size_for_test() == 0);
    CHECK(worker.cuopt_alpha_for_test() == pump::cuopt::kAlphaInit);
}

// bell5 has general integers, so the dispatch's LP carries aux columns.
TEST_CASE("cuOpt projection: a dispatch finds a feasible point", "[scylla][cuopt][pump]") {
    // `HighsMipSolverData::init` reads `parallel::num_threads()`.
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("mip_heuristic_scylla_cuopt_pump", true);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "bell5.mps");
    CscMatrix csc;
    const ProblemView problem = make_problem(*mipsolver, csc);
    ExecutionContext exec = make_exec(*mipsolver);
    IncumbentSink sink(*mipsolver, kSolutionSourceHeuristic);
    constexpr size_t kBudget = size_t{1} << 22;
    const DispatchOutcome out = scylla::run(
        *mipsolver, problem, make_budget(kBudget, exec.num_workers, kBudget >> 2), exec, sink);
    CHECK(out.effort > 0);
    // Only Scylla offered into this sink, and the bare solver has no
    // incumbent of its own.
    CHECK(sink.best_objective() < kHighsInf);
}
