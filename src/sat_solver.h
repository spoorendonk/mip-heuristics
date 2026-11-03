#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

// A plain CDCL SAT solver (#236): two watched literals, first-UIP learning with
// local minimisation, VSIDS, phase saving, Luby restarts and LBD-ranked learnt
// clause reduction.  Textbook technique (Marques-Silva and Sakallah's GRASP,
// Moskewicz et al.'s Chaff, Een and Sorensson's MiniSat, Audemard and Simon's
// Glucose for LBD).  Single-threaded; a solver is owned by one worker.
namespace sat {

// A literal is `2 * var + negated`, so `lit ^ 1` is its negation.
using Lit = std::int32_t;

constexpr Lit make_lit(std::int32_t var, bool negated = false) {
    return (2 * var) + (negated ? 1 : 0);
}

enum class SatResult { kSatisfiable, kUnsatisfiable, kStopped };

struct SatLimits {
    // Both count from the start of each `solve` call.  Effort is watch visits
    // and clause literals read, the solver's analogue of coefficient accesses.
    size_t max_conflicts = SIZE_MAX;
    size_t max_effort = SIZE_MAX;
};

class SatSolver {
public:
    // `seed` draws the tiny initial VSIDS activities that break ties.
    SatSolver(std::int32_t num_vars, std::uint64_t seed);

    std::int32_t new_var();
    [[nodiscard]] std::int32_t num_vars() const {
        return static_cast<std::int32_t>(activity_.size());
    }

    // Adds a clause at the root.  Returns false once the formula is known
    // unsatisfiable, which an empty clause or two opposed units make it.
    bool add_clause(std::span<const Lit> lits);

    // `stop` is polled from this thread, every `kStopPollEffort` units of
    // effort, so it must be cheap and safe to call here; it may be empty.
    SatResult solve(const SatLimits& limits, const std::function<bool()>& stop);

    // Valid after `solve` returned `kSatisfiable`.
    [[nodiscard]] bool model_value(std::int32_t var) const {
        return model_[static_cast<size_t>(var)] != 0;
    }

    [[nodiscard]] size_t conflicts() const { return conflicts_; }
    [[nodiscard]] size_t effort() const { return effort_; }
    [[nodiscard]] size_t reductions() const { return reductions_; }
    [[nodiscard]] size_t num_clauses() const { return clauses_.size(); }

private:
    using ClauseRef = std::uint32_t;
    static constexpr ClauseRef kNoClause = UINT32_MAX;

    // `blocker` is any other literal of the clause: when it is true the clause
    // is skipped without touching the arena, which is most visits.
    struct Watch {
        ClauseRef clause;
        Lit blocker;
    };

    // The arena holds `[size, lbd, lit...]`; `lbd` is 0 for an original clause.
    static constexpr size_t kHeaderSize = 2;
    std::int32_t& clause_size(ClauseRef c) { return arena_[c]; }
    std::int32_t& clause_lbd(ClauseRef c) { return arena_[c + 1]; }
    Lit* clause_lits(ClauseRef c) { return arena_.data() + c + kHeaderSize; }

    // +1 true, -1 false, 0 unassigned.
    [[nodiscard]] std::int8_t lit_value(Lit l) const {
        const std::int8_t v = assign_[static_cast<size_t>(l >> 1)];
        return (l & 1) != 0 ? static_cast<std::int8_t>(-v) : v;
    }
    [[nodiscard]] std::int32_t level_of(Lit l) const { return level_[static_cast<size_t>(l >> 1)]; }
    [[nodiscard]] std::int32_t decision_level() const {
        return static_cast<std::int32_t>(trail_lim_.size());
    }

    ClauseRef alloc_clause(std::span<const Lit> lits, std::int32_t lbd);
    void attach(ClauseRef c);
    void enqueue(Lit l, ClauseRef reason);
    ClauseRef propagate();
    // Returns the clause on a conflict, else `kNoClause` with `watch` kept.
    ClauseRef visit_watch(Lit false_lit, Watch watch, std::vector<Watch>& kept);
    void analyze(ClauseRef conflict);
    bool literal_redundant(Lit l);
    std::int32_t compute_lbd();
    void cancel_until(std::int32_t level);
    void learn_and_backjump(ClauseRef conflict);
    bool decide();
    void save_model();
    // Spent budgets, then the caller's stop check once `next_poll` is reached.
    bool should_stop(const SatLimits& limits, size_t conflicts_at_entry, size_t effort_at_entry,
                     size_t& next_poll, const std::function<bool()>& stop) const;
    void reduce_learnts();

    void bump_var(std::int32_t var);
    void heap_insert(std::int32_t var);
    void heap_sift_up(size_t pos);
    void heap_sift_down(size_t pos);
    std::int32_t heap_pop();

    bool ok_ = true;
    std::vector<std::int32_t> arena_;
    std::vector<ClauseRef> clauses_;
    std::vector<ClauseRef> learnts_;
    // Indexed by literal, holding the clauses watching it.  Each list keeps its
    // capacity, so a propagation step allocates only as a list grows.
    std::vector<std::vector<Watch>> watches_;
    std::vector<Watch> kept_;

    std::vector<std::int8_t> assign_;
    std::vector<std::int8_t> phase_;
    std::vector<std::int8_t> model_;
    std::vector<std::int32_t> level_;
    std::vector<ClauseRef> reason_;
    std::vector<Lit> trail_;
    std::vector<std::int32_t> trail_lim_;
    size_t trail_head_ = 0;

    std::vector<double> activity_;
    double var_inc_ = 1.0;
    std::vector<std::int32_t> heap_;
    std::vector<std::int32_t> heap_pos_;

    std::vector<std::uint8_t> seen_;
    std::vector<Lit> learnt_;
    std::vector<Lit> to_clear_;
    std::vector<std::int64_t> level_stamp_;
    std::int64_t stamp_ = 0;
    std::int32_t backjump_level_ = 0;

    size_t conflicts_ = 0;
    size_t effort_ = 0;
    size_t reductions_ = 0;
    std::uint64_t seed_state_;
};

}  // namespace sat
