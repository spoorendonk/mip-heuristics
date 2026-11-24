#include "fj.h"

#include "fj_personas.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "io/HighsIO.h"
#include "opportunistic_runner.h"
#include "solution_sink.h"
#include "structure.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace fj {

namespace {

// `set_fresh_rebuilds_for_test`.
bool g_fresh_rebuilds = false;

struct FjState {
    std::unique_ptr<FjWorker> worker;
    uint32_t initial_seed = 0;
    bool first_creation = true;
    // Trace-only slot identity, carried across rebuilds (#106).
    WorkerTrace trace;
    // The start the slot's first worker takes (#206); rebuilds ignore it.
    FjPersona persona = FjPersona::kSeeded;
    // Every worker of the slot searches with the exchange (#218).
    bool exchange = false;
};

// Slot `worker_idx`, with no worker built yet.  Its first worker is pinned
// to `random_seed + worker_idx`, so worker 0 matches vanilla FJ's seed.
FjState make_state(const ExecutionContext& exec, int worker_idx,
                   FjPersona persona = FjPersona::kSeeded, bool exchange = false) {
    // The `random_seed` option itself: `base_seed` is it plus
    // `kBaseSeedOffset` (`heuristic_base_seed`), so this is the exact
    // inverse, in the same modular `uint32_t` arithmetic.
    const uint32_t random_seed_opp = exec.base_seed - kBaseSeedOffset;
    return FjState{nullptr, random_seed_opp + static_cast<uint32_t>(worker_idx),
                   true,    WorkerTrace{worker_idx, 0},
                   persona, exchange};
}

// The persona of every slot, which slots exchange, and the structure both
// read: all seeded and none exchanging, with no detection pass, unless
// `budget.fj_personas` (#206) or `budget.fj_exchange` (#218) and some slot
// can use it.  Run on the dispatching thread, ahead of the parallel region.
struct Personas {
    RowStructure structure;
    std::vector<FjPersona> of_slot;
    std::vector<uint8_t> exchange_of_slot;
    // The detection pass's charge (#326): one read of every nonzero.
    size_t effort = 0;
};

Personas plan_personas(const ProblemView& problem, const HeuristicBudget& budget,
                       const ExecutionContext& exec) {
    Personas out;
    out.of_slot.assign(exec.num_workers, FjPersona::kSeeded);
    out.exchange_of_slot.assign(exec.num_workers, 0);
    // Where `assign_personas` would seed every slot anyway, and slot 1, the
    // first exchange lane, does not exist, skip the O(nnz) detection pass.
    // `kPersonaNnzLimit` bounds the pass for the exchange as well: it is one
    // pass, charged to FJ's effort, and not cut short by the deadline, which
    // is polled once ahead of it — the dispatch would stop at its first poll.
    if (!(budget.fj_personas || budget.fj_exchange) || exec.num_workers < 2 ||
        problem.nnz > kPersonaNnzLimit || exec.past_deadline()) {
        return out;
    }
    out.structure = detect_structure(problem);
    out.effort = problem.nnz;
    if (budget.fj_personas) {
        out.of_slot = assign_personas(out.structure, problem.nnz, exec.num_workers);
    }
    if (budget.fj_exchange && !out.structure.exact_k.empty()) {
        for (size_t slot = 0; slot < out.exchange_of_slot.size(); ++slot) {
            out.exchange_of_slot[slot] = is_exchange_slot(slot) ? 1 : 0;
        }
    }
    const auto slots_with = [&](FjPersona p) {
        return static_cast<int>(std::ranges::count(out.of_slot, p));
    };
    const auto kind = [&](RowKind k) { return static_cast<int>(out.structure.count_of(k)); };
    highsLogDev(exec.log_options, HighsLogType::kVerbose,
                "[FjPersonas] partitioning=%d packing=%d covering=%d cardinality=%d "
                "precedence=%d flow=%d exact_k=%d slots_covering=%d slots_exact_k=%d "
                "slots_exchange=%d\n",
                kind(RowKind::kPartitioning), kind(RowKind::kPacking), kind(RowKind::kCovering),
                kind(RowKind::kCardinality), kind(RowKind::kPrecedence),
                kind(RowKind::kFlowConservation), static_cast<int>(out.structure.exact_k.size()),
                slots_with(FjPersona::kCovering), slots_with(FjPersona::kExactK),
                static_cast<int>(std::ranges::count(out.exchange_of_slot, uint8_t{1})));
    return out;
}

// What building a slot's start cost (#326): `StartBudget`'s charge, and
// whether the deadline cut it short.  Zero for every start but a persona's.
struct StartCharge {
    size_t effort = 0;
    bool cut = false;
};

// Put a fresh worker in the slot, replacing a finished one.  A persona
// start's charge comes back for the caller to add to the attempt, and is
// already taken off the worker's own budget and added to its trace base.
StartCharge build_worker(FjState& state, Rng& rng, const ProblemView& problem,
                         const HeuristicBudget& budget, const ExecutionContext& exec,
                         SolutionSink& sink, const RestartSource& source,
                         const RowStructure* structure = nullptr) {
    // A slot's first worker is not a restart, so `budget.restart` is not
    // read for it (#200).
    const bool restart = !state.first_creation;
    uint32_t seed;
    if (state.first_creation) {
        seed = state.initial_seed;
        state.first_creation = false;
    } else {
        seed = static_cast<uint32_t>(rng());
    }
    // The caller's source first (only `run_until_stopped` passes one), then
    // the pool, then the dispatch snapshot.  Pool before snapshot is the
    // order LocalMIP's `resolve_worker_start` uses, and the reason dropping
    // the live `mipdata->incumbent` read (issue #98) costs FJ nothing.  The runner
    // rebuilds a stalled worker inside the parallel region, and that rebuild
    // used to warm-start from whatever a peer had just found; the pool holds
    // every such solution (`IncumbentSink` seeds it from the incumbent and
    // every accept goes through it) and `copy_best` takes its own lock, so
    // this reads the same material without racing a concurrent
    // `addIncumbent`.
    //
    // `kBest` is this order already, so only `kHybrid` changes the point;
    // with no point at all it falls back to it, as `kDefault` would.
    std::vector<double> start;
    if (!restart || !rebuild_restart_point(budget.restart, source, sink, rng, start, problem)) {
        // A source that just declined is not asked twice.
        const bool ask_source = !restart || budget.restart == RestartPoint::kDefault;
        if (!(ask_source && take_restart(source, rng, start, problem)) && !sink.copy_best(start)) {
            start = problem.incumbent;
        }
    }
    // A persona rewrites the first worker's start only (#206).
    StartCharge charge;
    if (!restart && state.persona != FjPersona::kSeeded && structure != nullptr) {
        StartBudget start_budget(exec);
        apply_persona(state.persona, problem, *structure, start, start_budget);
        charge = {.effort = start_budget.effort(), .cut = start_budget.expired()};
    }
    // Carry the outgoing worker's charge into the replacement's trace base,
    // so the `[HeurSol] effort_at` of this slot keeps rising instead of
    // restarting with the fresh `WorkerBudgetState` (#106).  Nothing about
    // what the budget counts changes: `base_` still starts at zero.  A
    // persona start's charge goes in the base too, and comes off the
    // worker's budget, since FJ's own counter cannot start above zero.
    if (state.worker) {
        state.trace.effort_base = state.worker->traced_effort();
    }
    state.trace.effort_base += charge.effort;
    const size_t per_worker = budget.per_worker - std::min(budget.per_worker, charge.effort);
    // `budget.worker_stale` is this worker's share of the dispatch's
    // absolute patience ceiling (issue #111) — the `nnz << 8` FJ used to
    // compute from its own copy of the matrix, now sized once alongside every
    // other heuristic's.
    // An exchange slot's rebuilt workers exchange too (#218): it is a move,
    // not a start.
    const std::span<const ExactKRow> exchange_rows =
        state.exchange && structure != nullptr ? std::span<const ExactKRow>(structure->exact_k)
                                               : std::span<const ExactKRow>();
    // The model, the exchange rows and the absent cutoff are the slot's, so
    // a rebuild resets the worker it has (#323): as a fresh one would search.
    if (state.worker && !g_fresh_rebuilds) {
        state.worker->reset(start, seed, per_worker, budget.worker_stale, state.trace);
        return charge;
    }
    state.worker = std::make_unique<FjWorker>(
        problem, exec, sink, per_worker, budget.worker_stale, seed, std::move(start), state.trace,
        std::numeric_limits<double>::infinity(), exchange_rows);
    return charge;
}

}  // namespace

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink) {
    return run(problem, budget, exec, sink, {});
}

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink,
                    const std::function<bool()>& found_elsewhere) {
    if (problem.degenerate() || budget.disabled()) {
        return {};
    }

    const RestartSource no_source;
    const Personas personas = plan_personas(problem, budget, exec);

    // No setup to abandon, so this dispatch can only ever report a plain
    // effort count.  MakeState below builds no worker at all — it returns
    // a null slot plus three scalars — and the `FjWorker` is constructed
    // lazily in the RunAttempt callback, which `run_opportunistic_loop`
    // reaches only *after* `should_stop`, whose first act is the deadline
    // poll.  So an already-expired dispatch never constructs one: FJ has
    // nothing ahead of that gate to give up on, where FPR and Scylla
    // precompute variable orders ahead of it and therefore can (#117).
    // The persona plan above (#206) is the one exception, and not worth a
    // bail: one O(nnz) classification pass, only with
    // `mip_heuristic_fj_personas` or `mip_heuristic_fj_exchange` on, skipped
    // past the deadline, and charged to this dispatch, out of its total
    // (#326).  The persona starts themselves run in `build_worker`, behind
    // the gate, charged to the slot's first attempt, and stop at the
    // deadline.  This does not touch the narrower standing caveat that once
    // construction has begun nothing bounds it — the deadline is polled
    // between attempts, not inside `FjWorker`'s constructor.
    HeuristicBudget search = budget;
    search.total -= std::min(search.total, personas.effort);
    DispatchOutcome outcome = run_opportunistic_loop(
        exec, search,
        [&exec, &personas](int worker_idx, Rng& /*rng*/) -> FjState {
            return make_state(exec, worker_idx, personas.of_slot[worker_idx],
                              personas.exchange_of_slot[worker_idx] != 0);
        },
        [&](FjState& state, Rng& rng, size_t run_cap) -> AttemptResult {
            StartCharge start;
            if (!state.worker || state.worker->finished()) {
                start = build_worker(state, rng, problem, budget, exec, sink, no_source,
                                     &personas.structure);
            }
            // A start the deadline cut short ends the attempt with its
            // charge alone, and the runner then stops on the clock (#326).
            // So does one that fills the cap: FJ checks its limit only at a
            // callback, so a zero-cap attempt would still cost a callback
            // period and break the runner's attempt_cap bound.
            if (start.cut || start.effort >= run_cap) {
                return {.effort = start.effort};
            }
            AttemptResult attempt = state.worker->run_attempt(run_cap - start.effort);
            attempt.effort += start.effort;
            // Asked even after an improving attempt, so a find from
            // elsewhere is not carried over to reset a later one; not after
            // a zero-effort one, which retires the slot before the runner
            // notes it, so the find would be lost.
            if (attempt.effort > 0 && found_elsewhere && found_elsewhere()) {
                attempt.found_improvement = true;
            }
            return attempt;
        });
    outcome.effort += personas.effort;
    return outcome;
}

size_t run_until_stopped(const ProblemView& problem, const HeuristicBudget& budget,
                         const ExecutionContext& exec, int worker, const RestartSource& source,
                         SolutionSink& sink) {
    if (problem.degenerate()) {
        return 0;
    }
    return run_on_caller_thread(
        exec, budget, worker,
        [&](int worker_idx, Rng& rng) -> FjState {
            // Built here rather than lazily: `attempt_with_rebuild` needs a
            // worker to ask `finished()`, and constructing one costs nothing
            // — the solver is built by its first attempt.
            FjState state = make_state(exec, worker_idx);
            build_worker(state, rng, problem, budget, exec, sink, source);
            return state;
        },
        [&](FjState& state, Rng& rng, size_t run_cap) -> AttemptResult {
            // A retired worker ends the run: zero effort retires the slot.
            if (budget.fj_stall_ends_run && state.worker->finished()) {
                return {};
            }
            auto rebuild = [&] { build_worker(state, rng, problem, budget, exec, sink, source); };
            if (exec.restart_now && exec.restart_now()) {
                rebuild();
            }
            return attempt_with_rebuild(state.worker, run_cap, rebuild);
        });
}

void set_fresh_rebuilds_for_test(bool fresh) {
    g_fresh_rebuilds = fresh;
}

}  // namespace fj
