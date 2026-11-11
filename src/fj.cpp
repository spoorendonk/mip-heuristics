#include "fj.h"

#include "fj_personas.h"
#include "fj_worker.h"
#include "heuristic_common.h"
#include "heuristic_context.h"
#include "io/HighsIO.h"
#include "opportunistic_runner.h"
#include "sat_search.h"
#include "solution_sink.h"
#include "structure.h"

#include <algorithm>
#include <cstdint>
#include <functional>
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
    // The slot's first attempt is the CDCL lane (#237); cleared once taken.
    bool cdcl = false;
};

// Slot `worker_idx`, with no worker built yet.  Its first worker is pinned
// to `random_seed + worker_idx`, so worker 0 matches vanilla FJ's seed.
FjState make_state(const ExecutionContext& exec, int worker_idx,
                   FjPersona persona = FjPersona::kSeeded, bool exchange = false,
                   bool cdcl = false) {
    // The `random_seed` option itself: `base_seed` is it plus
    // `kBaseSeedOffset` (`heuristic_base_seed`), so this is the exact
    // inverse, in the same modular `uint32_t` arithmetic.
    const uint32_t random_seed_opp = exec.base_seed - kBaseSeedOffset;
    return FjState{nullptr, random_seed_opp + static_cast<uint32_t>(worker_idx),
                   true,    WorkerTrace{worker_idx, 0},
                   persona, exchange,
                   cdcl};
}

const char* status_name(sat::SatSearchStatus status) {
    switch (status) {
        case sat::SatSearchStatus::kFeasible:
            return "feasible";
        case sat::SatSearchStatus::kInfeasible:
            return "infeasible";
        case sat::SatSearchStatus::kStopped:
            return "stopped";
        case sat::SatSearchStatus::kRefused:
            return "refused";
    }
    return "?";
}

const char* refusal_name(sat::SatRefusal refusal) {
    switch (refusal) {
        case sat::SatRefusal::kNone:
            return "none";
        case sat::SatRefusal::kNotPureBinary:
            return "not_pure_binary";
        case sat::SatRefusal::kNotIntegral:
            return "not_integral";
        case sat::SatRefusal::kTooLarge:
            return "too_large";
        case sat::SatRefusal::kCheckFailed:
            return "check_failed";
    }
    return "?";
}

// Whether a feasible point is known: the dispatch's snapshot, or anything
// the pool holds, which `IncumbentSink` seeds from the incumbent and every
// accept goes through.
bool point_known(const ProblemView& problem, SolutionSink& sink) {
    return !problem.incumbent.empty() || sink.best_objective() < kHighsInf;
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
    // `kPersonaNnzLimit` bounds the pass for the exchange as well: it is
    // uncharged and runs before the deadline is first polled.
    if (!(budget.fj_personas || budget.fj_exchange) || exec.num_workers < 2 ||
        problem.nnz > kPersonaNnzLimit) {
        return out;
    }
    out.structure = detect_structure(problem);
    if (budget.fj_personas) {
        out.of_slot = assign_personas(out.structure, problem.nnz, exec.num_workers);
    }
    if (budget.fj_exchange && !out.structure.exact_k.empty()) {
        for (size_t slot = 0; slot < out.exchange_of_slot.size(); ++slot) {
            out.exchange_of_slot[slot] = is_exchange_slot(slot) ? 1 : 0;
        }
    }
    // The CDCL lane's slot (#237), the last, is plain FJ after the lane: a
    // persona would rewrite the start it takes from the pool, the lane's
    // point, and with either it or the exchange the options could not be
    // measured apart.  Decided before the lane runs, so a model the lane
    // refuses loses that slot's persona or exchange too.
    if (budget.fj_cdcl && is_cdcl_slot(exec.num_workers - 1, exec.num_workers)) {
        out.of_slot.back() = FjPersona::kSeeded;
        out.exchange_of_slot.back() = 0;
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
    // An exchange slot's rebuilt workers exchange too (#218): it is a move,
    // not a start.
    const std::span<const ExactKRow> exchange_rows =
        state.exchange && structure != nullptr ? std::span<const ExactKRow>(structure->exact_k)
                                               : std::span<const ExactKRow>();
    state.worker = std::make_unique<FjWorker>(
        problem, exec, sink, budget.per_worker, budget.worker_stale, seed, std::move(start),
        state.trace, std::numeric_limits<double>::infinity(), exchange_rows);
}

}  // namespace

AttemptResult run_cdcl_lane(const ProblemView& problem, size_t max_effort,
                            const ExecutionContext& exec, SolutionSink& sink,
                            const WorkerTrace& trace, uint64_t seed,
                            const std::atomic<bool>* loop_stop) {
    // The search ignores the objective, so a point it finds can only be worth
    // more than one already known if it lands in a better basin by chance;
    // FJ, which starts from the pool's best, is the better use of the slot
    // then (#237).
    if (point_known(problem, sink)) {
        highsLogDev(exec.log_options, HighsLogType::kVerbose,
                    "[FjCdcl] skipped: a feasible point is known\n");
        return {};
    }
    sat::SatSearchConfig config;
    config.seed = seed;
    config.limits.max_effort = max_effort;
    // The sink is polled at the solver's cadence (every `kStopPollEffort`
    // units), under the pool's lock: a peer's find ends the search, for the
    // reason above.  So does the runner's stop flag, which an interrupt, the
    // budget or the patience gate sets while the other slots step: the
    // search is one attempt of up to a whole allowance, so without it the
    // join would wait that long.
    const std::function<bool()> stop = [&] {
        return (loop_stop != nullptr && loop_stop->load(std::memory_order_relaxed)) ||
               exec.past_deadline() || sink.best_objective() < kHighsInf;
    };
    std::vector<double> point;
    const sat::SatSearchResult found = sat::sat_search(problem, config, stop, point);
    AttemptResult out{.effort = found.effort, .exempt_from_staleness = true};
    if (found.status == sat::SatSearchStatus::kFeasible) {
        out.found_improvement =
            sink.offer(problem.objective(point), point, trace, found.effort).improved_incumbent;
    }
    highsLogDev(exec.log_options, HighsLogType::kVerbose,
                "[FjCdcl] status=%s refusal=%s clauses=%zu aux_vars=%zu conflicts=%zu "
                "effort=%zu max_effort=%zu\n",
                status_name(found.status), refusal_name(found.refusal), found.clauses,
                found.aux_vars, found.conflicts, found.effort, max_effort);
    return out;
}

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink) {
    if (problem.degenerate() || budget.disabled()) {
        return {};
    }

    const RestartSource no_source;
    const Personas personas = plan_personas(problem, budget, exec);
    if (budget.fj_cdcl) {
        // On the dispatching thread, so it is logged whether or not the
        // slot reaches its first attempt.
        const size_t last = exec.num_workers - 1;
        const int slot = exec.num_workers > 0 && is_cdcl_slot(last, exec.num_workers)
                             ? static_cast<int>(last)
                             : -1;
        highsLogDev(exec.log_options, HighsLogType::kVerbose, "[FjCdcl] slot=%d workers=%d\n", slot,
                    static_cast<int>(exec.num_workers));
    }

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
                [&exec, &personas, &budget](int worker_idx, Rng& /*rng*/) -> FjState {
                    return make_state(
                        exec, worker_idx, personas.of_slot[worker_idx],
                        personas.exchange_of_slot[worker_idx] != 0,
                        budget.fj_cdcl &&
                            is_cdcl_slot(static_cast<size_t>(worker_idx), exec.num_workers));
                },
                [&](FjState& state, Rng& rng, size_t run_cap,
                    const ContinuousLoopState& loop) -> AttemptResult {
                    // The CDCL lane is one attempt, capped at the slot's whole
                    // allowance (or what is left of the dispatch's) rather than
                    // `run_cap`: the search does not resume (#237).  Its effort
                    // joins the slot's trace base, as a retired worker's does,
                    // and a lane that did not run falls through to FJ, since a
                    // zero-effort attempt would retire the slot.
                    if (state.cdcl) {
                        state.cdcl = false;
                        const size_t spent = loop.total_effort.load(std::memory_order_relaxed);
                        const size_t cap = std::min(budget.per_worker,
                                                    budget.total - std::min(budget.total, spent));
                        const AttemptResult lane = run_cdcl_lane(
                            problem, cap, exec, sink, state.trace, state.initial_seed, &loop.stop);
                        state.trace.effort_base += lane.effort;
                        if (lane.effort > 0) {
                            return lane;
                        }
                        // Stopped before its first row: the dispatch is
                        // over, so retire the slot rather than build FJ.
                        if (loop.stopped() || exec.past_deadline()) {
                            return {};
                        }
                    }
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
