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
};

Personas plan_personas(const ProblemView& problem, const HeuristicBudget& budget,
                       const ExecutionContext& exec) {
    Personas out;
    out.of_slot.assign(exec.num_workers, FjPersona::kSeeded);
    out.exchange_of_slot.assign(exec.num_workers, 0);
    // Where `assign_personas` would seed every slot anyway, and slot 1, the
    // first exchange lane, does not exist, skip the O(nnz) detection pass.
    const bool personas = budget.fj_personas && problem.nnz <= kPersonaNnzLimit;
    if (!(personas || budget.fj_exchange) || exec.num_workers < 2) {
        return out;
    }
    out.structure = detect_structure(problem);
    if (personas) {
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

// Put a fresh worker in the slot, replacing a finished one.
void build_worker(FjState& state, Rng& rng, const ProblemView& problem,
                  const HeuristicBudget& budget, const ExecutionContext& exec, SolutionSink& sink,
                  const RestartSource& source, const RowStructure* structure = nullptr) {
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
    if (!restart && state.persona != FjPersona::kSeeded && structure != nullptr) {
        apply_persona(state.persona, problem, *structure, start);
    }
    // Carry the outgoing worker's charge into the replacement's trace base,
    // so the `[HeurSol] effort_at` of this slot keeps rising instead of
    // restarting with the fresh `WorkerBudgetState` (#106).  Nothing about
    // what the budget counts changes: `base_` still starts at zero.
    if (state.worker) {
        state.trace.effort_base = state.worker->traced_effort();
    }
    // `budget.worker_stale` is this worker's share of the dispatch's
    // absolute patience ceiling (issue #111) — the `nnz << 8` FJ used to
    // compute from its own copy of the matrix, now sized once alongside every
    // other heuristic's.
    // The objective cut (#222) is polled from the sink, so a rebuilt worker
    // starts under the cutoff in force without being told it.  An exchange
    // slot's rebuilt workers exchange too (#218): it is a move, not a start.
    const std::span<const ExactKRow> exchange_rows =
        state.exchange && structure != nullptr ? std::span<const ExactKRow>(structure->exact_k)
                                               : std::span<const ExactKRow>();
    state.worker = std::make_unique<FjWorker>(
        problem, exec, sink, budget.per_worker, budget.worker_stale, seed, std::move(start),
        state.trace, std::numeric_limits<double>::infinity(), budget.objective_cut, exchange_rows);
}

}  // namespace

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink) {
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
    // bail: one O(nnz) classification pass, and only with
    // `mip_heuristic_fj_personas` or `mip_heuristic_fj_exchange` on; the persona starts themselves
    // run in `build_worker`, behind the gate.  This does not touch the narrower standing caveat
    // that once construction has begun nothing bounds it — the deadline is polled between attempts,
    // not inside `FjWorker`'s constructor.
    return {.effort = run_opportunistic_loop(
                exec, budget,
                [&exec, &personas](int worker_idx, Rng& /*rng*/) -> FjState {
                    return make_state(exec, worker_idx, personas.of_slot[worker_idx],
                                      personas.exchange_of_slot[worker_idx] != 0);
                },
                [&](FjState& state, Rng& rng, size_t run_cap) -> AttemptResult {
                    if (!state.worker || state.worker->finished()) {
                        build_worker(state, rng, problem, budget, exec, sink, no_source,
                                     &personas.structure);
                    }
                    return state.worker->run_attempt(run_cap);
                })};
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

}  // namespace fj
