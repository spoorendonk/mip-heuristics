#pragma once

#include "heuristic_common.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace local_mip_detail {

// --- File-scope constants (paper + engineering) ---
//
// Three different questions used to share one constant here (issue
// #148), and conflating any two of them is the bug class this file is
// now careful about: "is this row violated?" is a feasibility
// question, and LocalMIP already has an authoritative answer for it —
// `WorkerCtx::feastol`, HiGHS's own runtime feasibility tolerance, the
// same one `HighsMipSolverData::trySolution` checks a row against
// before accepting a solution. The retired `kViolTol` (5e-7) was
// *tighter* than that (HiGHS's default `feastol` is 1e-6), so
// `WorkerCtx::update_violated` / `full_recheck` could classify a row
// "violated" — and refuse to submit the solution — in a window,
// (5e-7, 1e-6], where `WorkerCtx::is_violated` and
// `compute_tight_delta`, which already read `feastol`, disagreed and
// called the same row satisfied. A row stuck in that window got no
// repairing candidate: the tight-move operator's satisfied branch is a
// no-op there by construction. Every violation question in the search
// worker now reads `feastol` — set membership
// (`WorkerCtx::update_violated`, `full_recheck`), the submission gate
// (also `full_recheck`), `is_violated`, `compute_tight_delta`'s branch,
// and the analogous was-this-row-violated check in
// `compute_candidate_scores` (`local_mip_search.cpp`) — so the worker
// is exactly as strict as HiGHS's own acceptance check, never stricter.
//
// `kViolDeltaTol` answers a related but distinct question: given two
// violation *magnitudes* for the same row (before/after a candidate
// move, both already known violated), did the magnitude change by more
// than floating-point noise — i.e. did the move meaningfully reduce or
// worsen the violation. Still about row activity, still in the same
// units as `feastol`, but not "is this row violated" (that question is
// `feastol`'s alone), so it is not folded into `feastol` either.
// `compute_candidate_scores` (`local_mip_search.cpp`) is the only
// reader.
//
// `kScoreTol` is the third question: whether two accumulated candidate
// *scores* (Defs 5-10 — sums of per-row integer constraint weights, an
// objective-improvement flag, and a bonus) differ by more than
// floating-point summation noise. A score has no relationship to a
// row's activity or bounds, so it is a different quantity from
// `kViolDeltaTol` even though both exist to filter numerical noise —
// merging them would leave a reader unable to tune one without
// silently retuning the other, which is exactly the kind of coupling
// issue #148 removed for the violation questions. `kScoreTol` and
// `kViolDeltaTol` happen to share `kViolTol`'s old value because
// nothing has ever needed them to differ, not because the roles are
// related.
inline constexpr double kScoreTol = 5e-7;

// See `kScoreTol`'s comment above: this is the violation-magnitude
// noise floor, not the score-comparison one. Same value today; tune
// them independently.
inline constexpr double kViolDeltaTol = 5e-7;
inline constexpr HighsInt kRestartInterval = 200000;
// Counted units of `WorkerCtx::effort` between wall-clock polls in
// `LocalMipWorker::run_attempt`'s search loop (#162).  This was a *step*
// count until then, and a step is not a unit of bounded size: a feasible-
// mode step runs `WorkerCtx::full_recheck` every `kFeasibleRecheckPeriod`
// steps, which charges one `nnz` per call, so the wall time between two
// polls scales with the model where the constant does not.  Measured on a
// 1.4M-nonzero model, one 1000-step batch took tens of seconds and the
// dispatch overran a 15 s limit by ~46 s.
//
// The same argument #151 applied to `PropEngine::propagate`, and the same
// answer: pace the poll on the work actually charged, so a cheap step on a
// small model still polls rarely (the ~3% instruction-ref cost that made a
// per-iteration read unaffordable is what the cadence exists to avoid)
// while an expensive one polls after essentially every step.  The residual
// is one step plus this constant, whatever the model looks like.
inline constexpr size_t kTermCheckWork = 65536;
inline constexpr HighsInt kActivityPeriod = 100000;
inline constexpr double kSmoothProb = 3e-4;
inline constexpr HighsInt kBmsConstraints = 12;
inline constexpr HighsInt kBmsBudget = 2250;
inline constexpr HighsInt kBmsSatCon = 1;
inline constexpr HighsInt kBmsSatBudget = 80;
inline constexpr HighsInt kBoolFlipBudget = 5000;
inline constexpr HighsInt kEasyBudget = 5;
inline constexpr HighsInt kTabuBase = 3;
inline constexpr HighsInt kTabuVar = 10;
inline constexpr HighsInt kFeasibleRecheckPeriod = 100;
// After `kFeasiblePlateau` steps without an improving feasible move,
// trigger a random-walk diversification (perturb the solution and keep
// searching) instead of immediately declaring the worker finished.
// `kFeasibleMaxRandomWalks` bounds how many such perturbations we
// attempt before giving up, so pathological cases still terminate.
//
// This is an engineering addition, NOT from the paper.  Lin, Zou, Cai
// describe no random walk, plateau escape or perturbation anywhere;
// §4.1 is the PAWS weighting scheme.  An earlier comment here cited
// "§4.1's random walk to escape plateau recipe", which does not exist
// — the sibling copy of this rationale in `local_mip_worker.cpp` was
// corrected and this one was missed.  Keep both honest.
inline constexpr HighsInt kFeasiblePlateau = 5000;
inline constexpr HighsInt kFeasibleMaxRandomWalks = 20;
inline constexpr double kEpsZero = 1e-15;

// Forward declaration: LiftCache::recompute_* take a WorkerCtx& (defined
// in local_mip_core.h).  Defined here to keep LiftCache colocated with
// the other caches and the shared constants, matching the issue's
// suggested split.
struct WorkerCtx;

// --- IndexedSet: O(1) add/remove with iteration ---
struct IndexedSet {
    std::vector<HighsInt> elements;
    std::vector<HighsInt> pos;  // -1 = absent

    explicit IndexedSet(HighsInt n) : pos(n, -1) { elements.reserve(n); }

    void add(HighsInt i) {
        if (pos[i] != -1) {
            return;
        }
        pos[i] = static_cast<HighsInt>(elements.size());
        elements.push_back(i);
    }

    void remove(HighsInt i) {
        HighsInt p = pos[i];
        if (p == -1) {
            return;
        }
        HighsInt last = elements.back();
        elements[p] = last;
        pos[last] = p;
        elements.pop_back();
        pos[i] = -1;
    }

    [[nodiscard]] bool contains(HighsInt i) const { return pos[i] != -1; }
    [[nodiscard]] bool empty() const { return elements.empty(); }
    [[nodiscard]] HighsInt size() const { return static_cast<HighsInt>(elements.size()); }
    HighsInt operator[](HighsInt idx) const { return elements[idx]; }

    void clear() {
        for (HighsInt e : elements) {
            pos[e] = -1;
        }
        elements.clear();
    }

    [[nodiscard]] auto begin() const { return elements.begin(); }
    [[nodiscard]] auto end() const { return elements.end(); }
};

// --- ViolCache: memoize row violations within a candidate batch ---
struct ViolCache {
    std::vector<double> cache;
    std::vector<HighsInt> used;
    static constexpr double kSentinel = -1.0;

    explicit ViolCache(HighsInt n) : cache(n, kSentinel) { used.reserve(n); }

    double get_or_compute(HighsInt i, double lhs_i, double row_lo_i, double row_hi_i) {
        if (cache[i] >= 0.0) {
            return cache[i];
        }
        double v = row_violation(lhs_i, row_lo_i, row_hi_i);
        cache[i] = v;
        used.push_back(i);
        return v;
    }

    void reset() {
        for (HighsInt i : used) {
            cache[i] = kSentinel;
        }
        used.clear();
    }
};

// --- Candidate structs ---
struct Candidate {
    HighsInt var_idx = -1;
    double new_val = 0.0;
    double score = -std::numeric_limits<double>::infinity();
    double bonus = 0.0;
};

struct BatchCand {
    HighsInt var_idx;
    double new_val;
};

struct WeightedCon {
    HighsInt ci;
    uint64_t w;
};

// --- LiftCache: cached lift bounds / scores per variable ---
//
// Methods are defined in local_mip_core.cpp because they depend on the
// full WorkerCtx definition.
struct LiftCache {
    std::vector<double> lo, hi, score;
    std::vector<bool> dirty;
    std::vector<HighsInt> dirty_list;
    bool all_dirty = true;
    std::vector<HighsInt> positive_list;
    std::vector<bool> in_positive;
    const std::vector<HighsInt>* costed_vars = nullptr;

    explicit LiftCache(HighsInt ncol)
        : lo(ncol), hi(ncol), score(ncol), dirty(ncol, true), in_positive(ncol, false) {
        dirty_list.reserve(ncol);
        positive_list.reserve(ncol);
    }

    void mark_dirty(HighsInt j) {
        if (!dirty[j]) {
            dirty[j] = true;
            dirty_list.push_back(j);
        }
    }

    void mark_all_dirty() {
        all_dirty = true;
        dirty_list.clear();
        std::ranges::fill(dirty, true);
        positive_list.clear();
        std::ranges::fill(in_positive, false);
    }

    void recompute_one(HighsInt j, WorkerCtx& ctx);
    void recompute_all(WorkerCtx& ctx);
};

}  // namespace local_mip_detail
