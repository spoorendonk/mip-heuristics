#include "sat_solver.h"

#include "rng.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace sat {

namespace {

// VSIDS decay per conflict, and the activity at which all are rescaled.
constexpr double kVarDecay = 0.95;
constexpr double kActivityRescale = 1e100;
// Initial activities are drawn below this, so they only break ties.
constexpr double kInitialActivityMax = 1e-5;
// Conflicts in one Luby unit.
constexpr size_t kRestartUnit = 100;
// Learnt clauses kept before the first reduction, and the growth per reduction.
constexpr size_t kReduceBase = 2000;
constexpr size_t kReduceIncrement = 300;
// Learnt clauses at or below this LBD are never deleted (Audemard and Simon's
// "glue" clauses).
constexpr std::int32_t kGlueLbd = 2;
// Effort between polls of the caller's stop check.
constexpr size_t kStopPollEffort = 1U << 16;

// The Luby sequence 1, 1, 2, 1, 1, 2, 4, ... at 0-based `i`.
size_t luby(size_t i) {
    size_t size = 1;
    size_t seq = 0;
    while (size < i + 1) {
        ++seq;
        size = (2 * size) + 1;
    }
    while (size - 1 != i) {
        size = (size - 1) / 2;
        --seq;
        i %= size;
    }
    return size_t{1} << seq;
}

}  // namespace

SatSolver::SatSolver(std::int32_t num_vars, std::uint64_t seed) : seed_state_(seed) {
    const auto n = static_cast<size_t>(num_vars);
    activity_.reserve(n);
    assign_.reserve(n);
    phase_.reserve(n);
    level_.reserve(n);
    reason_.reserve(n);
    seen_.reserve(n);
    heap_.reserve(n);
    heap_pos_.reserve(n);
    watches_.reserve(2 * n);
    trail_.reserve(n);
    for (std::int32_t v = 0; v < num_vars; ++v) {
        new_var();
    }
}

std::int32_t SatSolver::new_var() {
    const auto var = static_cast<std::int32_t>(activity_.size());
    Rng rng(seed_state_ + static_cast<std::uint64_t>(var));
    activity_.push_back(std::uniform_real_distribution<double>(0.0, kInitialActivityMax)(rng));
    assign_.push_back(0);
    phase_.push_back(0);
    level_.push_back(0);
    reason_.push_back(kNoClause);
    seen_.push_back(0);
    heap_pos_.push_back(-1);
    watches_.emplace_back();
    watches_.emplace_back();
    level_stamp_.push_back(0);
    level_stamp_.push_back(0);
    heap_insert(var);
    return var;
}

bool SatSolver::add_clause(std::span<const Lit> lits) {
    if (!ok_) {
        return false;
    }
    learnt_.assign(lits.begin(), lits.end());
    std::ranges::sort(learnt_);
    size_t out = 0;
    for (size_t k = 0; k < learnt_.size(); ++k) {
        const Lit l = learnt_[k];
        if (lit_value(l) > 0 || (k + 1 < learnt_.size() && learnt_[k + 1] == (l ^ 1))) {
            return true;  // Satisfied at the root, or a tautology.
        }
        if (lit_value(l) == 0 && (out == 0 || learnt_[out - 1] != l)) {
            learnt_[out++] = l;
        }
    }
    learnt_.resize(out);
    if (learnt_.empty()) {
        ok_ = false;
    } else if (learnt_.size() == 1) {
        enqueue(learnt_[0], kNoClause);
    } else {
        const ClauseRef c = alloc_clause(learnt_, 0);
        clauses_.push_back(c);
        attach(c);
    }
    return ok_;
}

SatSolver::ClauseRef SatSolver::alloc_clause(std::span<const Lit> lits, std::int32_t lbd) {
    const auto c = static_cast<ClauseRef>(arena_.size());
    arena_.push_back(static_cast<std::int32_t>(lits.size()));
    arena_.push_back(lbd);
    arena_.insert(arena_.end(), lits.begin(), lits.end());
    return c;
}

void SatSolver::attach(ClauseRef c) {
    const Lit* lits = clause_lits(c);
    watches_[static_cast<size_t>(lits[0] ^ 1)].push_back({c, lits[1]});
    watches_[static_cast<size_t>(lits[1] ^ 1)].push_back({c, lits[0]});
}

void SatSolver::enqueue(Lit l, ClauseRef reason) {
    const auto v = static_cast<size_t>(l >> 1);
    assign_[v] = (l & 1) != 0 ? std::int8_t{-1} : std::int8_t{1};
    level_[v] = decision_level();
    reason_[v] = reason;
    trail_.push_back(l);
}

SatSolver::ClauseRef SatSolver::visit_watch(Lit false_lit, Watch watch, std::vector<Watch>& kept) {
    if (lit_value(watch.blocker) > 0) {
        kept.push_back(watch);
        return kNoClause;
    }
    const ClauseRef c = watch.clause;
    Lit* lits = clause_lits(c);
    const std::int32_t size = clause_size(c);
    effort_ += static_cast<size_t>(size);
    if (lits[0] == false_lit) {
        std::swap(lits[0], lits[1]);
    }
    const Lit first = lits[0];
    if (lit_value(first) > 0) {
        kept.push_back({c, first});
        return kNoClause;
    }
    for (std::int32_t k = 2; k < size; ++k) {
        if (lit_value(lits[k]) >= 0) {
            std::swap(lits[1], lits[k]);
            watches_[static_cast<size_t>(lits[1] ^ 1)].push_back({c, first});
            return kNoClause;
        }
    }
    kept.push_back({c, first});
    if (lit_value(first) < 0) {
        return c;
    }
    enqueue(first, c);
    return kNoClause;
}

SatSolver::ClauseRef SatSolver::propagate() {
    while (trail_head_ < trail_.size()) {
        const Lit assigned = trail_[trail_head_++];
        const Lit false_lit = assigned ^ 1;
        // `visit_watch` may push onto another literal's list, never this one:
        // the replacement it watches is not false, and `false_lit` is.
        auto& list = watches_[static_cast<size_t>(assigned)];
        effort_ += list.size() + 1;
        kept_.clear();
        for (size_t k = 0; k < list.size(); ++k) {
            const ClauseRef conflict = visit_watch(false_lit, list[k], kept_);
            if (conflict != kNoClause) {
                kept_.insert(kept_.end(), list.begin() + static_cast<std::ptrdiff_t>(k) + 1,
                             list.end());
                list.assign(kept_.begin(), kept_.end());
                trail_head_ = trail_.size();
                return conflict;
            }
        }
        list.assign(kept_.begin(), kept_.end());
    }
    return kNoClause;
}

// A literal of the learnt clause is redundant when its reason's other literals
// are all in the clause or fixed at the root (local minimisation).
bool SatSolver::literal_redundant(Lit l) {
    const ClauseRef r = reason_[static_cast<size_t>(l >> 1)];
    if (r == kNoClause) {
        return false;
    }
    const Lit* lits = clause_lits(r);
    const std::int32_t size = clause_size(r);
    effort_ += static_cast<size_t>(size);
    for (std::int32_t k = 0; k < size; ++k) {
        const auto v = static_cast<size_t>(lits[k] >> 1);
        if (seen_[v] == 0 && level_[v] > 0) {
            return false;
        }
    }
    return true;
}

// First-UIP analysis; leaves the clause in `learnt_` with the asserting literal
// first, and the level to return to in `backjump_level_`.
void SatSolver::analyze(ClauseRef conflict) {
    learnt_.clear();
    learnt_.push_back(0);  // The asserting literal goes here.
    to_clear_.clear();
    std::int32_t open = 0;
    size_t index = trail_.size();
    Lit uip = 0;
    ClauseRef c = conflict;
    do {
        const Lit* lits = clause_lits(c);
        const std::int32_t size = clause_size(c);
        effort_ += static_cast<size_t>(size);
        for (std::int32_t k = 0; k < size; ++k) {
            const Lit q = lits[k];
            const auto v = static_cast<size_t>(q >> 1);
            if (seen_[v] != 0 || level_[v] == 0) {
                continue;
            }
            seen_[v] = 1;
            to_clear_.push_back(q);
            bump_var(q >> 1);
            if (level_[v] == decision_level()) {
                ++open;
            } else {
                learnt_.push_back(q);
            }
        }
        while (seen_[static_cast<size_t>(trail_[index - 1] >> 1)] == 0) {
            --index;
        }
        --index;
        uip = trail_[index];
        c = reason_[static_cast<size_t>(uip >> 1)];
        --open;
    } while (open > 0);
    learnt_[0] = uip ^ 1;

    size_t out = 1;
    for (size_t k = 1; k < learnt_.size(); ++k) {
        if (!literal_redundant(learnt_[k])) {
            learnt_[out++] = learnt_[k];
        }
    }
    learnt_.resize(out);

    backjump_level_ = 0;
    if (learnt_.size() > 1) {
        size_t best = 1;
        for (size_t k = 2; k < learnt_.size(); ++k) {
            if (level_of(learnt_[k]) > level_of(learnt_[best])) {
                best = k;
            }
        }
        std::swap(learnt_[1], learnt_[best]);
        backjump_level_ = level_of(learnt_[1]);
    }
    for (const Lit q : to_clear_) {
        seen_[static_cast<size_t>(q >> 1)] = 0;
    }
}

std::int32_t SatSolver::compute_lbd() {
    ++stamp_;
    std::int32_t lbd = 0;
    for (const Lit l : learnt_) {
        const auto level = static_cast<size_t>(level_of(l));
        if (level_stamp_[level] != stamp_) {
            level_stamp_[level] = stamp_;
            ++lbd;
        }
    }
    return lbd;
}

void SatSolver::cancel_until(std::int32_t level) {
    if (decision_level() <= level) {
        return;
    }
    const auto keep = static_cast<size_t>(trail_lim_[static_cast<size_t>(level)]);
    for (size_t k = trail_.size(); k > keep; --k) {
        const Lit l = trail_[k - 1];
        const auto v = static_cast<size_t>(l >> 1);
        phase_[v] = assign_[v];
        assign_[v] = 0;
        if (heap_pos_[v] < 0) {
            heap_insert(l >> 1);
        }
    }
    trail_.resize(keep);
    trail_lim_.resize(static_cast<size_t>(level));
    trail_head_ = keep;
}

void SatSolver::learn_and_backjump(ClauseRef conflict) {
    analyze(conflict);
    const std::int32_t lbd = compute_lbd();
    cancel_until(backjump_level_);
    if (learnt_.size() == 1) {
        enqueue(learnt_[0], kNoClause);
    } else {
        const ClauseRef c = alloc_clause(learnt_, std::max(lbd, 1));
        learnts_.push_back(c);
        attach(c);
        enqueue(learnt_[0], c);
    }
    var_inc_ /= kVarDecay;
}

// Returns false when every variable is assigned.
bool SatSolver::decide() {
    while (!heap_.empty()) {
        const std::int32_t var = heap_pop();
        const auto v = static_cast<size_t>(var);
        if (assign_[v] == 0) {
            trail_lim_.push_back(static_cast<std::int32_t>(trail_.size()));
            enqueue(make_lit(var, phase_[v] <= 0), kNoClause);
            ++effort_;
            return true;
        }
    }
    return false;
}

// Deletes the worst half of the deletable learnt clauses, and rebuilds the arena
// and the watches without them.  Called at the root at a propagation fixpoint,
// so no reason refers to a clause anything will read again, and a watched
// literal that is false has a true partner, which the literal order preserves.
void SatSolver::reduce_learnts() {
    std::ranges::sort(learnts_, [this](ClauseRef a, ClauseRef b) {
        const std::int32_t la = clause_lbd(a);
        const std::int32_t lb = clause_lbd(b);
        return la != lb ? la < lb : clause_size(a) < clause_size(b);
    });
    size_t glue = 0;
    while (glue < learnts_.size() && clause_lbd(learnts_[glue]) <= kGlueLbd) {
        ++glue;
    }
    learnts_.resize(glue + ((learnts_.size() - glue) / 2));

    std::vector<std::int32_t> arena;
    arena.reserve(arena_.size());
    auto relocate = [&](std::vector<ClauseRef>& refs) {
        for (ClauseRef& c : refs) {
            const auto start = arena_.begin() + c;
            const auto length = static_cast<std::ptrdiff_t>(kHeaderSize) + clause_size(c);
            c = static_cast<ClauseRef>(arena.size());
            arena.insert(arena.end(), start, start + length);
        }
    };
    relocate(clauses_);
    relocate(learnts_);
    arena_ = std::move(arena);
    for (auto& list : watches_) {
        list.clear();
    }
    for (const ClauseRef c : clauses_) {
        attach(c);
    }
    for (const ClauseRef c : learnts_) {
        attach(c);
    }
    for (const Lit l : trail_) {
        reason_[static_cast<size_t>(l >> 1)] = kNoClause;
    }
}

void SatSolver::save_model() {
    model_.resize(assign_.size());
    for (size_t v = 0; v < assign_.size(); ++v) {
        model_[v] = assign_[v] > 0 ? std::int8_t{1} : std::int8_t{0};
    }
}

bool SatSolver::should_stop(const SatLimits& limits, size_t conflicts_at_entry,
                            size_t effort_at_entry, size_t& next_poll,
                            const std::function<bool()>& stop) const {
    if (conflicts_ - conflicts_at_entry >= limits.max_conflicts ||
        effort_ - effort_at_entry >= limits.max_effort) {
        return true;
    }
    if (effort_ < next_poll) {
        return false;
    }
    next_poll = effort_ + kStopPollEffort;
    return stop && stop();
}

SatResult SatSolver::solve(const SatLimits& limits, const std::function<bool()>& stop) {
    if (!ok_) {
        return SatResult::kUnsatisfiable;
    }
    const size_t conflicts_at_entry = conflicts_;
    const size_t effort_at_entry = effort_;
    size_t next_poll = effort_ + kStopPollEffort;
    size_t restarts = 0;
    size_t conflicts_since_restart = 0;
    size_t restart_limit = kRestartUnit * luby(restarts);
    size_t reduce_limit = kReduceBase;

    while (true) {
        // Before propagating, so a run of conflicts cannot overshoot the budget.
        // Root literals not yet propagated stay on the trail for a later `solve`.
        if (should_stop(limits, conflicts_at_entry, effort_at_entry, next_poll, stop)) {
            cancel_until(0);
            return SatResult::kStopped;
        }
        const ClauseRef conflict = propagate();
        if (conflict != kNoClause) {
            ++conflicts_;
            ++conflicts_since_restart;
            if (decision_level() == 0) {
                ok_ = false;
                return SatResult::kUnsatisfiable;
            }
            learn_and_backjump(conflict);
            continue;
        }
        if (conflicts_since_restart >= restart_limit) {
            conflicts_since_restart = 0;
            restart_limit = kRestartUnit * luby(++restarts);
            cancel_until(0);
            // The root's part of the trail was propagated before any decision.
            if (learnts_.size() >= reduce_limit) {
                reduce_learnts();
                reduce_limit += kReduceIncrement;
            }
            continue;
        }
        if (!decide()) {
            save_model();
            cancel_until(0);
            return SatResult::kSatisfiable;
        }
    }
}

void SatSolver::bump_var(std::int32_t var) {
    const auto v = static_cast<size_t>(var);
    activity_[v] += var_inc_;
    if (activity_[v] > kActivityRescale) {
        for (double& a : activity_) {
            a /= kActivityRescale;
        }
        var_inc_ /= kActivityRescale;
    }
    if (heap_pos_[v] >= 0) {
        heap_sift_up(static_cast<size_t>(heap_pos_[v]));
    }
}

void SatSolver::heap_insert(std::int32_t var) {
    heap_pos_[static_cast<size_t>(var)] = static_cast<std::int32_t>(heap_.size());
    heap_.push_back(var);
    heap_sift_up(heap_.size() - 1);
}

void SatSolver::heap_sift_up(size_t pos) {
    const std::int32_t var = heap_[pos];
    const double score = activity_[static_cast<size_t>(var)];
    while (pos > 0) {
        const size_t parent = (pos - 1) / 2;
        const std::int32_t up = heap_[parent];
        if (activity_[static_cast<size_t>(up)] >= score) {
            break;
        }
        heap_[pos] = up;
        heap_pos_[static_cast<size_t>(up)] = static_cast<std::int32_t>(pos);
        pos = parent;
    }
    heap_[pos] = var;
    heap_pos_[static_cast<size_t>(var)] = static_cast<std::int32_t>(pos);
}

void SatSolver::heap_sift_down(size_t pos) {
    const std::int32_t var = heap_[pos];
    const double score = activity_[static_cast<size_t>(var)];
    while (true) {
        size_t child = (2 * pos) + 1;
        if (child >= heap_.size()) {
            break;
        }
        if (child + 1 < heap_.size() && activity_[static_cast<size_t>(heap_[child + 1])] >
                                            activity_[static_cast<size_t>(heap_[child])]) {
            ++child;
        }
        const std::int32_t down = heap_[child];
        if (activity_[static_cast<size_t>(down)] <= score) {
            break;
        }
        heap_[pos] = down;
        heap_pos_[static_cast<size_t>(down)] = static_cast<std::int32_t>(pos);
        pos = child;
    }
    heap_[pos] = var;
    heap_pos_[static_cast<size_t>(var)] = static_cast<std::int32_t>(pos);
}

std::int32_t SatSolver::heap_pop() {
    const std::int32_t top = heap_.front();
    heap_pos_[static_cast<size_t>(top)] = -1;
    const std::int32_t last = heap_.back();
    heap_.pop_back();
    if (!heap_.empty()) {
        heap_[0] = last;
        heap_pos_[static_cast<size_t>(last)] = 0;
        heap_sift_down(0);
    }
    return top;
}

}  // namespace sat
