#pragma once

#include "deadline.h"
#include "heuristic_common.h"
#include "lp_data/HighsLp.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

struct HighsLogOptions;

// Common execution scaffold for the four presolve heuristics (issue #94).
//
// Before this header, each of FJ / FPR / LocalMIP / Scylla re-derived the
// same ~10-line block of constants for itself (`ParallelSetup`: csc,
// num_workers, base_seed, worker_budget, default_run_cap, stale_budget),
// and the model sizes on top of that.  The three structs below carve that
// single struct along its actual seams — what the heuristic *searches*,
// what it may *spend*, and how it *runs* — so a heuristic's entry point can
// take exactly the parts it needs, and `mode_dispatch` can build the
// expensive part (the CSC transpose) once for the whole chain.
//
// `fpr_lp.cpp` takes `make_exec` / `make_budget` and builds its view with
// `problem_view` over the CSC its setup built: it runs on the same
// continuous parallel runner but keeps its own `LpFprSetup` for the LP
// references, reduced costs and shared `ContestedPdlp` that the presolve
// heuristics have no equivalent of.
//
// Ownership: solution submission is not here — it lives in `SolutionSink`
// (solution_sink.h), which inside HiGHS is the `IncumbentSink` that
// `mode_dispatch::run_sequential` owns and threads through each heuristic's
// entry point.  Per-worker effort/staleness bookkeeping lives in
// `WorkerBudgetState` (worker_base.h); `HeuristicBudget` holds the
// *derived* values each worker's base struct receives on construction.
// All four heuristics' budgets come from their own
// `mip_heuristic_<name>_effort` option; FJ's sizes `per_worker` and lets
// `total` scale with the pool, the other three size `total` and are
// divided across it (see mode_dispatch.cpp).

// The uniform runner contract every presolve heuristic implements:
//
//     DispatchOutcome <ns>::run(const ProblemView &problem, const HeuristicBudget &budget,
//                               ExecutionContext &exec, SolutionSink &sink);
//
// `mode_dispatch::run_sequential` owns all four arguments — including the
// source tag the sink attributes this heuristic's solutions with — and books
// the returned outcome through `EffortLedger`, the single point of effort
// accounting.  No heuristic self-books.  The per-heuristic headers describe
// only what their own runner does differently.
//
// `ExecutionContext` is passed by non-const reference to match the entry
// signature, but it is immutable for the duration of a dispatch: it is
// shared by every worker of every heuristic in the chain, so caching
// mutable per-dispatch state on it would be an unsynchronised shared
// write.  Its methods are `const`.

// What one dispatch of a heuristic did, as its runner reports it (issue
// #119).  Used to be a bare `size_t` effort count.
//
// The second field exists because zero effort has two causes that a log
// cannot tell apart.  A heuristic that searched and produced nothing and
// one that never searched at all — because #117 made the *sequential*
// setup abandon its work on an already-passed deadline — both booked
// `effort=0 found=0`, and the #113 calibration bins the second as
// "barren", which is the population its patience estimate rests on.  A
// setup bail is not a barren dispatch: it is a cost of setup, and it
// happens precisely on the large, hard instances the calibration is most
// sensitive to.
//
// Why the *return type* rather than a narrower channel.  The flag has to
// travel from a bail site deep inside one heuristic to `run_sequential`,
// which is the only caller and the only booker.  The two alternatives were
// a mutable field on `ExecutionContext` — narrower in signatures, but that
// object is shared by every worker of every heuristic in the chain and is
// documented immutable for exactly that reason, and it would need a reset
// per heuristic that nothing forces anyone to write — and a flag on
// `IncumbentSink`, which is the per-heuristic mutable channel
// `run_and_charge` already reads across the call but has nothing to do
// with solution submission.  Both smuggle per-dispatch state into an
// object that outlives the dispatch; a return value cannot leak into the
// next heuristic because there is no state to forget to clear.
//
// `[[nodiscard]]`, for the reason `SolutionSink::offer` is: a dropped
// outcome loses both the effort — which `run_sequential` books into
// `heuristic_effort_used` and nothing else can recover — and the bail flag
// this issue exists to carry.  `-Werror=unused-result` on our library and
// test targets is what makes the attribute a build failure
// rather than a warning that scrolls past.  It costs a caller that wants
// one field nothing: `run(...).effort` is a *use* of the return value, so
// only a call whose whole result is thrown away trips it, and the single
// such caller (a test driving warm-start counters) spells the discard.
struct [[nodiscard]] DispatchOutcome {
    // Effort charged, in this heuristic's own unit.  `run_sequential`
    // books exactly this into `heuristic_effort_used`.
    size_t effort = 0;

    // This dispatch abandoned its sequential setup on the wall-clock
    // deadline and never searched (issue #117's bail, made visible).  It
    // implies `effort == 0`, but the converse is what this field exists to
    // deny.
    //
    // Scope: *only* the deadline bails in `fpr::precompute_var_orders` and
    // `scylla::precompute_config_var_orders` set it, plus the dive-time
    // equivalent.  A heuristic declining for another reason — a degenerate
    // model, a zero budget, a `ContestedPdlp` that failed to initialise —
    // reports a plain zero, because none of those is a dispatch the clock
    // cut short.
    bool abandoned_setup = false;

    // The bail, spelled once so the two sites cannot disagree.
    static DispatchOutcome abandoned() { return {.effort = 0, .abandoned_setup = true}; }
};

// Read-only view of the model a heuristic searches: everything a worker
// reads, and nothing of the solver it was read from (issue #170).  Every
// member but the two snapshots is a non-owning pointer, a derived size or
// a tolerance; the pointees are owned by whoever built the view — the
// solver and `run_sequential`'s CSC local for the presolve chain
// (`make_problem(HighsMipSolver&, CscMatrix&)`), or a `ProblemStorage` for
// a caller that owns its model — and outlive every worker reading them.
// Built once per dispatch and passed by const reference — the snapshots
// make it no longer trivially cheap to copy.
struct ProblemView {
    // The column-wise matrix, which FJ builds its own row-wise copy from.
    // Never read its bounds, cost, offset, sense or integrality: the fields
    // below replace them.
    const HighsLp* model = nullptr;
    // Column and row bounds as the workers enforce them.  The HiGHS adapter
    // points these at the presolved model's own; the core `make_problem`
    // maps a magnitude at or above HiGHS's `infinite_bound` to infinity, as
    // HiGHS's own model assessment does, and rounds an integer column's
    // inward (`ceil(lb - feastol)`, `floor(ub + feastol)`), as the presolved
    // model's already are.
    const std::vector<double>* col_lower = nullptr;
    const std::vector<double>* col_upper = nullptr;
    const std::vector<double>* row_lower = nullptr;
    const std::vector<double>* row_upper = nullptr;
    // The objective in *minimisation* form, and the integrality of every
    // column (#170).  Every worker reads these, so the heuristics only ever
    // see a minimisation problem — as they do inside HiGHS, whose presolve
    // turns a maximisation model into one before any heuristic runs.  Every
    // objective a worker computes, and so every objective a `SolutionSink`
    // or `SolutionPool` holds, is `offset + col_cost . x` in this form: the
    // negated original objective for a maximisation model.
    const std::vector<double>* col_cost = nullptr;
    double offset = 0.0;
    const std::vector<HighsVarType>* integrality = nullptr;
    // Row-wise copy of `model->a_matrix_`.
    const std::vector<HighsInt>* ar_start = nullptr;
    const std::vector<HighsInt>* ar_index = nullptr;
    const std::vector<double>* ar_value = nullptr;
    const CscMatrix* csc = nullptr;
    // Per-column up- and down-lock counts, for FPR's trivially-roundable
    // fixings.
    const std::vector<HighsInt>* uplocks = nullptr;
    const std::vector<HighsInt>* downlocks = nullptr;

    // `mip_feasibility_tolerance` and `small_matrix_value`, as
    // `HighsMipSolverData::feastol` / `epsilon` hold them.
    double feastol = 0.0;
    double epsilon = 0.0;

    // Derived sizes, previously recomputed at every call site that wanted
    // one of them.
    HighsInt ncol = 0;
    HighsInt nrow = 0;
    size_t nnz = 0;

    // Snapshot of `HighsMipSolverData::incumbent`, copied once per dispatch
    // on the dispatching thread (issue #98).  Workers read *this*, never
    // `mipdata->incumbent`: submission is immediate, so a peer worker's
    // accepted solution runs `addIncumbent`, whose whole-vector assignment
    // (`incumbent = sol;`) rewrites the live buffer under a concurrent
    // reader — element-wise while the sizes match, reallocating out from
    // under it on the empty-to-sized transition.  Empty when the solver had
    // no incumbent at dispatch time.  `fpr_lp` takes the same snapshot, through
    // `problem_view`.
    //
    // The snapshot is the *floor* a worker starts from, not necessarily what
    // it gets: both readers consult the shared pool first, which holds the
    // seeded incumbent plus everything accepted since (`IncumbentSink` seeds
    // it at construction and every accept goes through it, under its own
    // lock).  So dropping the live read costs neither of them a warm start —
    // LocalMIP's `resolve_worker_start` only reaches this copy with an empty
    // pool, i.e. when nothing has been submitted and the live incumbent
    // still equals it, and `fj.cpp` resolves pool-first for the same reason.
    // Nothing may read this *instead* of the pool: an `FjWorker` rebuilt
    // mid-dispatch would then silently lose a peer's find.
    std::vector<double> incumbent;

    // Per-column snapshot of `HighsDomain::isBinary`, taken with the
    // incumbent and for the same reason (issue #99).  `addIncumbent` runs
    // `getDomain().propagate()` and `redcostfixing.propagateRootRedcost`,
    // both of which tighten the root domain's bound vectors element-wise,
    // while workers classify columns from those same vectors.  Unlike the
    // incumbent this can never dangle — the bound vectors are sized once at
    // setup — so it is a torn read rather than a use-after-free, but it is
    // still a race, and a column's classification flipping mid-dispatch is
    // not something any worker is written to expect.
    //
    // Scope: taken once for the *whole* FJ -> FPR -> LocalMIP -> Scylla
    // chain, so a column that root propagation fixes after FJ's first
    // incumbent is still classified with its pre-FJ value by the other
    // three.  Deliberate, and cheap: workers enforce the view's
    // `col_lower`/`col_upper`, never `HighsDomain`'s, so the classification
    // was already decoupled from the bounds they respect.
    //
    // `uint8_t` rather than `std::vector<bool>`: workers index this from
    // hot loops, and the bit-packed specialisation costs a shift and mask
    // per read plus a proxy object.  Concurrent *reads* of a frozen
    // `vector<bool>` would be well-defined — the hazard is concurrent
    // read/write of neighbouring bits, which cannot arise for a mask built
    // before the parallel region — so this is a throughput choice, not a
    // correctness one.
    std::vector<uint8_t> binary;

    // A model with no columns or no rows: every heuristic declines it.
    [[nodiscard]] bool degenerate() const { return ncol == 0 || nrow == 0; }

    // `x`'s objective in the view's minimisation form, offset included.
    [[nodiscard]] double objective(const std::vector<double>& x) const {
        double obj = offset;
        for (HighsInt j = 0; j < ncol; ++j) {
            obj += (*col_cost)[j] * x[j];
        }
        return obj;
    }
};

// What a `ProblemView` points at when there is no `HighsMipSolverData` to
// borrow it from — a caller running the heuristics on a model it owns,
// such as the original model read with `Highs::readModel`.  Must outlive
// every view built over it.
struct ProblemStorage {
    std::vector<HighsInt> ar_start;
    std::vector<HighsInt> ar_index;
    std::vector<double> ar_value;
    std::vector<HighsInt> uplocks;
    std::vector<HighsInt> downlocks;
    CscMatrix csc;
    // The negated costs of a maximisation model; empty otherwise.
    std::vector<double> col_cost;
    // The bounds, huge ones mapped to infinity and integer columns' rounded
    // inward.
    std::vector<double> col_lower;
    std::vector<double> col_upper;
    std::vector<double> row_lower;
    std::vector<double> row_upper;
    // All-continuous for a model with no `integrality_` (a pure LP); empty
    // otherwise.
    std::vector<HighsVarType> integrality;
};

// Why `make_problem` refused a model.  The kinds are distinct because the
// caller acts on them differently: a malformed model is a bug in whoever
// built it, an infeasible one is an answer about the model, and an
// unsupported one needs another tool.  The line between the first two is
// HiGHS's own (`assessLp`, `assessBounds`, `assessMatrix`): what it refuses
// with an error is malformed, what it lets through with a warning or
// solves to infeasibility is infeasible.
struct ProblemError {
    enum class Kind {
        // Structurally broken: sizes that disagree with the dimensions, a
        // matrix that is not column-wise with them, starts that do not begin
        // at 0 or are not monotone, row indices out of range or repeated
        // within a column, an integrality or sense outside its enum, a NaN
        // anywhere, a non-finite (or `infinite_cost`-sized) cost or offset,
        // a coefficient at or above `large_matrix_value`, or a lower bound
        // at `+infinite_bound` or an upper one at `-infinite_bound`.
        kMalformed,
        // Provably infeasible on its face: crossed bounds (an integer column's
        // after rounding inward), or an empty row whose bounds exclude 0.
        kInfeasible,
        // A semi-continuous or semi-integer column, which the workers do
        // not model: they read integrality as "anything but continuous is an
        // integer", which would drop the column's zero branch.
        kUnsupported,
    };
    Kind kind;
    std::string message;
};

// A view over a caller's own `model`, with `storage` filled the way
// `HighsMipSolverData::runSetup()` fills the solver's copies: the row-wise
// matrix through `highsSparseTranspose`, the lock counts by the same rule,
// and the CSC from the row-wise matrix as `make_problem` builds it.  The
// binary mask is `HighsDomain::isBinary` at the normalised bounds (an
// integer column with bounds exactly `[0, 1]`), and there is no incumbent.
// `feastol` / `epsilon` are the caller's `mip_feasibility_tolerance` /
// `small_matrix_value`.  `model` must outlive the view.
//
// Normalised as HiGHS's presolve normalises a model before its heuristics
// see it, so that nothing a worker reports can violate what it was given:
// minimisation form (a maximisation model's costs and offset negated into
// `storage`), bounds of magnitude `infinite_bound` or more mapped to
// infinity, integer column bounds rounded inward, and a model without
// `integrality_` all continuous.  Everything is checked in one linear pass
// over the model; see `ProblemError` for what is refused.
std::expected<ProblemView, ProblemError> make_problem(const HighsLp& model, ProblemStorage& storage,
                                                      double feastol, double epsilon);

// One heuristic's slice of the presolve effort envelope.  `total` used to
// travel separately as a bare `max_effort` parameter while the other three
// were fields of `ParallelSetup`; they are one thing and now travel as one.
struct HeuristicBudget {
    size_t total = 0;        // whole-dispatch ceiling, summed over workers
    size_t per_worker = 0;   // total / N (floor division)
    size_t attempt_cap = 0;  // per-attempt cap: max(total / (N * 10), 1)

    // Runner-level staleness ceiling: the dispatch stops once
    // `ContinuousLoopState::effort_since_improvement` — summed over every
    // worker — crosses this.  Absolute and instance-scaled since issue
    // #111 (`patience_threshold(nnz, <this heuristic's patience option>,
    // total)`, sized by the caller from `mip_heuristic_<name>_patience`),
    // not the `total / 4` it used to be.  A quarter of the budget is not a
    // patience criterion: it says "I have spent a quarter of what I was
    // given", which is true at every budget and therefore bounds nothing.
    // A quarter of the budget is what the *clamp* is (#116), which is a
    // different job — see `kPatienceCeilingDivisor`.
    size_t stale = 0;

    // Per-worker share of the same ceiling.  The runner's counter
    // aggregates N workers, so one worker's share is `stale / N`; that
    // relation is not new, it is what `total / 4` per worker against
    // `total / 4` per dispatch already meant, and what FJ's `nnz << 8`
    // against a `N * nnz << 8` runner gate already was at the shipped
    // default.  Scylla is the documented exception — see scylla_worker.cpp.
    size_t worker_stale = 0;

    // This heuristic was handed nothing, and must therefore do nothing
    // (issue #106).
    //
    // `mip_heuristic_<name>_effort = 0` sizes `total` to zero, and #107
    // spells "this heuristic is excluded from the configuration" exactly
    // that way — a zero-pattern of four continuous parameters rather than a
    // separate discrete subset dimension, which is what keeps its search
    // tractable.  That reduction needs a zero budget to be worth exactly
    // what omitting the heuristic is worth, so every entry point checks
    // this alongside `ProblemView::degenerate()` and returns before any
    // setup.
    //
    // Since #167 the zero *is* the omission — `mip_heuristic_suite` is
    // gone and the effort option is the only selector — so what this
    // guarantees is no longer an equivalence between two spellings but the
    // meaning of the one that is left.  The history is worth keeping
    // because it is what the guarantee costs: `fpr_lp` draws from
    // upstream's `mip_heuristic_effort` envelope and never reads a presolve
    // option, so until #164 gave it an option of its own, zeroing `fpr`
    // left the dive-time heuristic running.
    // `run_opportunistic_loop` already declined a zero total,
    // but three of the four heuristics do real work before they reach it —
    // Scylla builds a `ContestedPdlp` (a whole `Highs` LP copy), the
    // per-config variable orders and N workers; FPR precomputes its
    // variable orders — and that work is neither free nor charged, so it
    // was invisible in the effort total while still costing wall time.
    [[nodiscard]] bool disabled() const { return total == 0; }
};

// How a heuristic runs: the worker count, the RNG base seed, the deadline,
// the log, and the one termination predicate.  Nothing in it names the
// solver (#170): the HiGHS adapter builds one with `make_exec`
// (highs_context.h), and a caller running the workers on its own threads
// builds its own.
//
// Historical note on `attempt_cap`: the deleted epoch-gated runner gave FJ a
// separate cadence (`kEpochsPerWorkerFj = 20` against 10 for the rest), on
// the grounds that "FJ's synchronization cadence matters for pool-crossover
// behaviour and a change could regress on FJ-dominant instances".  That only
// ever applied to the epoch runner — the continuous runner has always used
// this cap for FJ too — so #92 removed a constant, not a behaviour.  The
// concern was never benchmarked; recorded here so it is not lost with the
// constant, but no cadence changed for any surviving execution path.
struct ExecutionContext {
    size_t num_workers;  // highs::parallel::num_threads(), at least 1
    uint32_t base_seed;  // seeded from `random_seed` via heuristic_base_seed

    // The run's wall-clock deadline, and the only half of "should we
    // stop?" a worker thread may poll (issue #114).  Also handed as it is
    // to the layers below a heuristic's runner: `fpr_core`'s DFS and the
    // repair search under it have no `ExecutionContext` (issue #117), and
    // taking this one rather than building their own is what keeps them
    // from stopping against a different clock or limit than their runner.
    //
    // `HighsTimer::read()` is `const` and, for the solve clock, writes
    // nothing — it reads `clock_start`/`clock_time` and calls
    // `getWallTime()`.  Nothing starts or stops that clock while a
    // heuristic dispatch is in flight, so concurrent readers are safe.
    // That is what separates this from `terminated()` below, and it is why
    // every presolve heuristic can poll the deadline from inside its own
    // inner loop without a poller seat.
    //
    // In the HiGHS adapter it is the solver's own clock rather than a
    // `steady_clock` snapshot (`make_exec`): a second origin would no longer
    // agree with the `[Heur] start_s`/`end_s` the ledger emits, which the
    // tests and `bench/parse_highs_log.py` both read against this same
    // limit.  `HighsTimer` bottoms out in `high_resolution_clock` and is
    // therefore not monotonic (see `effort_ledger.h`); that risk is
    // pre-existing and is the price of one shared origin.
    Deadline deadline;

    // Where the workers' own solvers log (FJ's `FeasibilityJumpSolver`).
    // Held by reference, so it must outlive the context, and it must be a
    // working one: a default-constructed `HighsLogOptions` holds null
    // `output_flag` / `log_to_console` / `log_dev_level` pointers, which the
    // logger dereferences.  `Highs::getOptions().log_options` is one; a
    // caller without a `Highs` points those three at its own values.
    const HighsLogOptions& log_options;

    // The external half of "should we stop?", or empty when nothing
    // outside the clock can end the run.  In the HiGHS adapter it is
    // `HighsMipSolverData::terminatorTerminated()`, which *writes*
    // `mipsolver.termination_status_` when a terminator is attached — so
    // it is called only through `terminated()`, and only by one caller at
    // a time.
    std::function<bool()> terminator;

    // A caller's own stop flag, or null (#171).  The HiGHS adapter leaves
    // it null; a caller driving a worker with `run_until_stopped` points it
    // at the flag it sets to end the run.  It is read, never written, so it
    // joins the clock on the write-free side of the split above: any worker
    // thread may poll it, and it is polled wherever the deadline is — inside
    // an attempt as well as between attempts, which is what bounds a stop's
    // latency by a worker's poll cadence rather than by one attempt's budget.
    const std::atomic<bool>* stop = nullptr;

    // The write-free half of "should we stop?": the caller's stop flag or
    // the wall-clock deadline.  Any worker thread may call it.
    [[nodiscard]] bool past_deadline() const {
        return (stop != nullptr && stop->load(std::memory_order_relaxed)) || deadline.expired();
    }

    // The full "should we stop?" predicate.  Three hand-rolled copies of
    // it existed before this struct.
    //
    // Not thread-safe for concurrent callers: `terminator` may write (see
    // above).  That write — not the clock read — is the whole reason this
    // one needs a single caller.  `mode_dispatch` calls it between
    // heuristics, with every parallel region already joined, and inside a
    // parallel region the worker holding `ContinuousLoopState`'s claimable
    // poller seat calls it on everyone's behalf.
    //
    // There is no longer an exception: FPR's multi-attempt inner loop used
    // to poll this directly from its own worker thread, which was a race
    // whenever a terminator was attached.  It polls `past_deadline()`
    // instead (issue #114), which is the half it actually needed, so the
    // seat is now the only route to the terminator.
    [[nodiscard]] bool terminated() const {
        return (terminator && terminator()) || past_deadline();
    }

    // Deterministic seed for worker `w`.  The runner seeds its own per-worker
    // `Rng` with this, and heuristics that pre-construct their workers seed
    // them with it too — three hand-written copies of the expression before
    // it lived here, which is three chances for one of them to drift.
    [[nodiscard]] uint32_t worker_seed(int w) const {
        return base_seed + (static_cast<uint32_t>(w) * kSeedStride);
    }
};

// Split a heuristic's slice of the effort envelope into the per-worker,
// per-attempt and staleness ceilings its workers and the runner use.
//
// `stale` is passed in rather than derived here (issue #111): it is the
// one quantity in this struct that must *not* be a function of `total`,
// and every caller knows the model's `nnz` and its heuristic's
// patience value.  See `patience_threshold` in heuristic_common.h.
inline HeuristicBudget make_budget(size_t total, size_t num_workers, size_t stale) {
    // A zero total is "this heuristic is excluded" (issue #106), so every
    // derived ceiling is zero too and `disabled()` holds.  Spelling it out
    // rather than letting the expressions below run: `attempt_cap` floors
    // at 1, so a zero budget used to license one attempt — the whole of
    // Scylla's, which charges a full PDLP solve and does not stop for
    // `attempt_cap` once started — and `stale` arrives here *unclamped*,
    // because `patience_threshold` special-cases a zero budget by skipping
    // the clamp.  Both are ceilings that only make sense above a budget
    // that exists.
    if (total == 0) {
        return HeuristicBudget{};
    }
    // Designated initialisers for the same reason `make_problem` below
    // gives them: this aggregate is five `size_t` members in a row, so a
    // mis-ordered addition converts silently between them.  #111 appended
    // the fifth.
    return HeuristicBudget{.total = total,
                           .per_worker = total / num_workers,
                           .attempt_cap = std::max<size_t>(total / (num_workers * 10), 1),
                           .stale = stale,
                           .worker_stale = std::max<size_t>(stale / num_workers, 1)};
}

// The second runner contract (#171).  FJ and LocalMIP also expose
//
//     size_t <ns>::run_until_stopped(const ProblemView &problem, const HeuristicBudget &budget,
//                                    const ExecutionContext &exec, int worker,
//                                    const RestartSource &source, SolutionSink &sink);
//
// which runs worker slot `worker` on the calling thread — a thread the
// caller owns, not the HiGHS task scheduler — until `exec` says stop.  It
// is the same slot loop the presolve dispatch runs N of in parallel
// (`run_on_caller_thread` in opportunistic_runner.h), with two differences:
// a stalled worker is always rebuilt and never retires its slot
// (`attempt_with_rebuild`), and a rebuild asks `source` for its start point
// before the sink's own pool.  The in-HiGHS dispatch does not use it.
//
// What ends it: `exec.stop` or the deadline, both polled through
// `past_deadline()` inside an attempt on the worker's own cadence (FJ at
// every upstream callback, i.e. every 500000 FJ effort units or found
// solution; LocalMIP every `kTermCheckWork` counted units or steps) and
// again before every attempt; `exec.terminator`, before every other
// attempt; `budget.total`; or `budget.stale`, the run-level patience.
// `make_until_stopped_budget` lifts the last two.  So a stop is seen within
// one poll interval, except while a worker is being built: FJ's solver
// construction and LocalMIP's cold-start construction sweep (one O(nnz)
// pass) poll nothing, as in the presolve dispatch.  A worker is rebuilt
// when it stalls (`budget.worker_stale`) or spends `budget.per_worker`.
// `exec.num_workers` is not read; `worker` picks the slot's seed
// (`exec.worker_seed(worker)`) and its trace identity, and for LocalMIP a
// slot other than 0 perturbs its first start, as in the presolve dispatch.
// Returns the effort charged, cold-start construction included.
//
// Thread-safety.  Everything runs on the calling thread, and the calls into
// the caller's code are:
//   - `source`, when the slot's first worker is built and whenever a
//     stalled one is rebuilt;
//   - `sink`'s `on_accept`, for every offer the sink's pool accepts, on
//     this thread and outside the pool lock;
//   - `exec.terminator`, every other attempt: this thread holds the poller
//     seat for the whole run;
//   - `*exec.stop`, a relaxed load, from inside attempts and between them.
// Several threads may each run a slot against one `sink`, `source`, `exec`
// and stop flag — `SolutionSink` is thread-safe — but then `source` and
// `terminator` are called concurrently and must be thread-safe too.

// Where a `run_until_stopped` slot takes its start point, ahead of the
// sink's own pool (#171): fill `out` with a point in the view's
// column space and return true, or return false to fall back to the pool,
// the view's incumbent and finally a fresh construction, in that order.  A
// point that is not `ncol` long is treated as a false.  The workers clamp
// it into the view's bounds, and LocalMIP rounds its integer columns; FJ
// takes it as its starting assignment and LocalMIP perturbs it as it does
// any restart.  `rng` is the slot's own generator, there for a source that
// wants to draw — `SolutionSink::get_restart` has this signature, so a
// source can forward to another sink's pool crossover.  Empty means no
// source; the presolve dispatch passes an empty one.
using RestartSource = std::function<bool(Rng& rng, std::vector<double>& out)>;

// `source`'s point into `out`, when there is a source and its point has
// the view's width; otherwise false, with `out` empty.
[[nodiscard]] inline bool take_restart(const RestartSource& source, Rng& rng,
                                       std::vector<double>& out, HighsInt ncol) {
    if (source && source(rng, out) && std::cmp_equal(out.size(), ncol)) {
        return true;
    }
    out.clear();
    return false;
}

// The budget for a `run_until_stopped` slot that only the caller's stop
// should end: no ceiling on the run or on one worker, no run-level patience,
// `attempt_cap` effort between two of the runner's own checks, and a worker
// rebuilt once `patience` of its effort has gone by without moving the
// sink's best.  Both in the heuristic's own effort unit.
inline HeuristicBudget make_until_stopped_budget(size_t attempt_cap, size_t patience) {
    constexpr size_t kUnbounded = std::numeric_limits<size_t>::max();
    return HeuristicBudget{.total = kUnbounded,
                           .per_worker = kUnbounded,
                           .attempt_cap = attempt_cap,
                           .stale = kUnbounded,
                           .worker_stale = patience};
}
