#pragma once

#include "parallel/HighsSpinMutex.h"
#include "rng.h"

#include <functional>
#include <vector>

class HighsMipSolver;

inline constexpr int kPoolCapacity = 10;

// Diversity-aware insertion constants.
// Accept a solution within this fraction of best objective if it adds diversity.
inline constexpr double kDiversityObjTolerance = 0.10;
// Minimum Hamming distance (as fraction of integer vars) to qualify as diverse.
inline constexpr double kDiversityMinHammingFrac = 0.05;

// Relative margin an offer must clear to count as having moved the pool's
// best objective, with an absolute floor of the same size (issue #116).
//
// Deliberately not a re-use of `kDiversityObjTolerance`: that one is an
// *admission* band — how far below best a structurally different solution
// may sit and still be worth keeping — so reading it as an improvement
// test would call a 9%-worse solution an improvement.  This is the margin
// `bench/analyze_presolve_probe.py`'s `improving_offers` reconstructs an
// improvement trajectory with, copied on purpose: the shipped patience
// defaults are p95 inter-improvement gaps measured under that definition,
// and a gate resetting on a different one is not the gate they calibrate.
inline constexpr double kImprovementObjMargin = 1e-9;

// Thread-safe solution pool. Keeps top-K solutions sorted by objective.
// Supports restart strategies: guided crossover, neighborhood crossover,
// and biased copy.  When an integer mask is set, insertion is
// diversity-aware: solutions that don't improve the worst objective can
// still enter the pool if they are structurally different from existing
// entries (measured by Hamming distance on integer variables).
class SolutionPool {
public:
    struct Snapshot {
        bool has_solution;
        double best_objective;
    };

    struct Entry {
        double objective;
        std::vector<double> solution;
        // Per-entry provenance tag (one of the kSolutionSource* constants
        // from HiGHS's HighsMipSolverData.h).  Carried so the shared pool
        // can attribute each solution to the heuristic that produced it,
        // rather than falling back on the generic kSolutionSourceHeuristic
        // tag.
        int source;
    };

    SolutionPool(int capacity, bool minimize);

    // Set the integer variable mask.  Must be called before diversity-aware
    // insertion can take effect.  is_integer[j] == true iff variable j is
    // integer.  Thread-safe (acquires lock).
    void set_integer_mask(std::vector<bool> mask);

    // Register a callback invoked (outside the pool lock) whenever a solution
    // is accepted. Call once before dispatching workers. The callback receives
    // the solution vector and source tag and must be thread-safe (multiple
    // workers may trigger it concurrently).
    void set_on_accept(std::function<void(const std::vector<double>&, int)> callback);

    // What one offer did to the pool.  Two facts, because they are two
    // questions with different answers and #116 is about not confusing
    // them: `accepted` is the admission policy's verdict — is this worth
    // keeping? — and `improved_best` is whether the offer moved the best
    // objective the solve knows, which the pool tracks as a monotone
    // watermark rather than as its own front entry (see `try_add`).
    //
    // `accepted && !improved_best` is the ordinary case, not an edge one:
    // the pool admits its first `capacity_` offers unconditionally while
    // it fills, and structurally diverse near-best solutions afterwards.
    // Measured over 233 instances at 16 workers, FPR earns ~3.3 M
    // acceptances against 590 incumbent improvements.  So a staleness gate
    // reading `accepted` is reading the admission policy rather than the
    // heuristic's productivity, and cannot be calibrated (issue #116);
    // gates read `improved_best`, while `[Heur] found` and
    // `[HeurSol] accepted` keep reporting `accepted`.
    //
    // `improved_best` implies `accepted`: an offer that beats the best
    // beats the worst too, so it takes the standard replacement path.
    struct AddResult {
        bool accepted = false;
        bool improved_best = false;
    };

    // Try to add a solution.  Invokes the on_accept callback (if set)
    // after releasing the pool lock.
    // `source` is one of the kSolutionSource* constants and is stored on
    // the inserted entry for later provenance-aware flushing.
    // Insertion policy (when pool is full):
    //   1. If obj improves on worst: replace worst (standard).
    //   2. Else if obj is within kDiversityObjTolerance of best and Hamming
    //      diversity exceeds kDiversityMinHammingFrac: replace the most
    //      similar entry *other than the best*, which is never evicted —
    //      this path admits solutions that do not even beat the worst
    //      entry, so it must not be able to spend the best one on one of
    //      them.  Diversity itself is still measured against every entry,
    //      the best included.
    //
    // `improved_best` is decided against `best_seen_`, the best objective
    // this pool has ever accepted, read under the same hold of the pool
    // lock — the only place that comparison is race-free: a worker cannot
    // ask the solver for the pre-offer incumbent without racing
    // `addIncumbent` (see `ProblemView::incumbent`, #98), and two workers
    // offering concurrently would otherwise both read the same "before".
    // Deliberately not `entries_.front()`: rule 2 above can evict the
    // front entry, so the pool's best objective goes backwards where the
    // solve's never does.
    [[nodiscard]] AddResult try_add(double obj, const std::vector<double>& sol, int source);

    // Atomically snapshot feasibility and current best objective.
    Snapshot snapshot();

    // Get a restart solution via one of three strategies (roll order):
    //   [0.0, 0.4)  — guided crossover: keep agreed integer values, coin-flip
    //                  disagreements.
    //   [0.4, 0.7)  — neighborhood crossover: better parent provides base,
    //                  coin-flip only on disagreeing integers.
    //   [0.7, 1.0)  — biased copy toward better entries.
    //
    // Post-crossover repair is handled naturally by the calling heuristic,
    // which treats the restart as an initial solution and runs its own
    // feasibility restoration. LocalMIP (`src/local_mip.cpp`) is the only
    // caller left: FPR, fpr_lp and Scylla all called this too until issue
    // #122 found that the value they fed as an FPR fix-and-propagate seed
    // was unread on every path and removed those three call sites.
    bool get_restart(Rng& rng, std::vector<double>& out);

    // Return sorted entries (best first). Caller should hold no lock.
    std::vector<Entry> sorted_entries();

    // Copy only the best entry's solution vector into `out`.  Acquires
    // the pool lock once and copies one vector (vs `sorted_entries`,
    // which copies the whole pool including up to kPoolCapacity - 1
    // unused vectors).  Returns false when the pool is empty, with
    // `out` untouched.  Preferred entry point when the caller only
    // needs the top-ranked warm-start.
    bool copy_best(std::vector<double>& out);

    int size();

private:
    // Hamming distance on integer variables between two solutions.
    // Caller must hold mtx_.
    int hamming_distance(const std::vector<double>& a, const std::vector<double>& b) const;

    // Number of integer variables (cached from integer_mask_).
    int num_integers() const;

    // Whether column `j` is integer, tolerating a mask shorter than the
    // solution vector (or absent entirely, which is how a pool built
    // without integrality information behaves).
    [[nodiscard]] bool is_integer_col(int j) const;

    // The three restart strategies `get_restart` dispatches between; see
    // its documentation above for the roll ranges.  All three are called
    // with mtx_ already held and write their result into `out`.
    void guided_crossover(const std::vector<double>& sol_a, const std::vector<double>& sol_b,
                          Rng& rng, std::vector<double>& out) const;
    void neighborhood_crossover(const std::vector<double>& sol_better,
                                const std::vector<double>& sol_other, Rng& rng,
                                std::vector<double>& out) const;
    void biased_copy(Rng& rng, std::vector<double>& out) const;

    mutable HighsSpinMutex mtx_;
    std::vector<Entry> entries_;
    int capacity_;
    bool minimize_;
    std::vector<bool> integer_mask_;  // true for integer variables
    int num_integers_ = 0;            // cached count of integer vars
    // Best objective ever accepted, and whether anything has been.  The
    // predicate `improved_best` is decided against, under mtx_; monotone
    // by construction, unlike `entries_.front()` — see `try_add`.
    double best_seen_ = 0.0;
    bool has_best_seen_ = false;
    // Invoked outside pool lock after a successful insertion. Set once before
    // workers start; reads from worker threads are unsynchronized but safe
    // because the happens-before from thread creation covers the write.
    std::function<void(const std::vector<double>&, int)> on_accept_;
};

// Seed a pool with the current incumbent (if any). Defined inline to
// avoid pulling HighsMipSolver includes into the header — callers
// already include both solution_pool.h and HighsMipSolver.h.
void seed_pool(SolutionPool& pool, const HighsMipSolver& mipsolver);
