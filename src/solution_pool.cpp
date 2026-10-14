#include "solution_pool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <utility>

SolutionPool::SolutionPool(int capacity)
    : capacity_(capacity), integer_mask_(std::make_shared<const std::vector<bool>>()) {
    entries_.reserve(capacity_ > 0 ? capacity_ : 0);
}

void SolutionPool::set_integer_mask(std::vector<bool> mask) {
    int count = 0;
    for (bool b : mask) {
        if (b) {
            ++count;
        }
    }
    auto next = std::make_shared<const std::vector<bool>>(std::move(mask));
    std::scoped_lock lock(mtx_);
    integer_mask_.swap(next);  // the old mask is released after the lock
    num_integers_ = count;
    ++generation_;
}

int SolutionPool::hamming_distance(const std::vector<bool>& mask, const std::vector<double>& a,
                                   const std::vector<double>& b) {
    int dist = 0;
    int n = static_cast<int>(std::min(a.size(), b.size()));
    if (mask.empty()) {
        return 0;
    }
    for (int j = 0; j < n; ++j) {
        if (mask[j] && std::round(a[j]) != std::round(b[j])) {
            ++dist;
        }
    }
    return dist;
}

namespace {

// The diversity path's verdict over one scan: `dists[i]` is the offer's
// Hamming distance to entry `i`, best first.  Two different questions, so
// two different reductions.
//
// *Diversity* is measured against every entry, the best
// included: an offer that duplicates the best solution adds
// nothing to a crossover pool however different it is from
// the rest.
//
// *Eviction* skips index 0.  This path only runs on an offer
// that does not even beat the worst entry, so letting it
// replace the best — which it could, whenever the best was
// also its nearest neighbour — discards the pool's most
// valuable solution to make room for a dominated one.  HiGHS
// keeps its own incumbent, so no solution left the solve, but
// `copy_best` and the crossover's better parent then work
// from something worse until the pool refills.  A diversity
// rule has no reason to discard the best solution it holds.
//
// Returns the index to evict, or -1 to refuse the offer.
int diversity_victim(const std::vector<int>& dists, int num_integers) {
    int min_dist = std::numeric_limits<int>::max();
    int evict_dist = std::numeric_limits<int>::max();
    int most_similar_idx = -1;
    for (int idx = 0; std::cmp_less(idx, dists.size()); ++idx) {
        const int dist = dists[idx];
        min_dist = std::min(min_dist, dist);
        if (idx > 0 && dist < evict_dist) {
            evict_dist = dist;
            most_similar_idx = idx;
        }
    }
    double min_frac = static_cast<double>(min_dist) / static_cast<double>(num_integers);
    // `most_similar_idx < 0` means the best entry is the only one there
    // is, so nothing may be evicted.
    if (min_frac >= kDiversityMinHammingFrac && most_similar_idx > 0) {
        return most_similar_idx;
    }
    return -1;
}

}  // namespace

SolutionPool::AddResult SolutionPool::try_add(double obj, const std::vector<double>& sol,
                                              int source) {
    return try_add(obj, std::make_shared<const std::vector<double>>(sol), source);
}

// Cognitive complexity over the threshold.  Kept whole because capacity, dominance and
// diversity replacement are one decision, taken at one hold of the pool spin-lock: the
// diversity scan alone leaves the lock, and the loop around it exists to re-take that one
// decision when the pool moved during the scan.  Splitting the decision would thread the
// lock and the scan's state through helpers.  Not a hot path for the lock's sake: the
// argument is lock scope, not throughput.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
SolutionPool::AddResult SolutionPool::try_add(double obj, SolutionPtr sol, int source) {
    // Pointers whose last reference this call drops — an evicted entry, the
    // scan's snapshot — are destroyed at return, after the lock: freeing a
    // large vector is not work to do while peers spin.
    SolutionPtr evicted;
    // The scan's state across attempts.  `seen` keeps every entry a scan
    // has measured alive alongside its distance, so its address cannot be
    // reused by a new entry, and a rescan after the pool moved measures only
    // entries it has not seen.
    std::vector<EntryRef> scan;
    std::vector<std::pair<SolutionPtr, int>> seen;
    std::vector<int> dists;
    MaskPtr mask;
    std::uint64_t scanned_generation = 0;
    bool scanned = false;
    // Sized here so the snapshot taken under the lock never allocates.
    scan.reserve(capacity_ > 0 ? capacity_ : 0);

    for (;;) {
        AddResult result;
        bool need_scan = false;
        {
            std::scoped_lock lock(mtx_);

            // The best objective offered so far, decided here rather than by
            // the caller because this is the one point where "the best before
            // this offer" is race-free: two workers offering concurrently would
            // otherwise both read the same "before" outside the lock and both
            // call themselves an improvement (#116).  No watermark yet means
            // the solve has no feasible solution at all — the sink seeds the
            // pool from the incumbent at construction — so any offer improves
            // on it.
            //
            // A *watermark* and not `entries_.front()`, which is not the same
            // thing: the diversity path below replaces the entry most similar
            // to the offer, and that can be the front one, so the pool's best
            // objective goes backwards while the solve's does not (HiGHS keeps
            // whatever `addIncumbent` was given).  Reading the front entry
            // would then call the next offer to clear the degraded value an
            // improvement, handing a patience gate a free reset for a solution
            // the incumbent already dominates — which is the very thing #116
            // exists to stop.  A monotone watermark is also exactly what
            // `improving_offers` in `bench/analyze_presolve_probe.py` tracks,
            // and that is where the shipped patience defaults were measured.
            //
            // The margin mirrors that same function; see `kImprovementObjMargin`.
            if (!has_best_seen_) {
                result.improved_best = true;
            } else {
                const double margin = kImprovementObjMargin * std::max(1.0, std::abs(best_seen_));
                result.improved_best = obj < best_seen_ - margin;
            }

            // Find insertion point (entries_ kept sorted, best first)
            // Heterogeneous (const EntryRef&, double) on purpose.  That is
            // fine for std::lower_bound but does not model
            // indirect_strict_weak_order, so std::ranges::lower_bound does
            // not accept it — hence the NOLINTs on the call sites below.
            auto cmp = [](const EntryRef& entry, double val) { return entry.objective < val; };
            auto insert_sorted = [&] {
                // NOLINTNEXTLINE(modernize-use-ranges)
                auto pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);
                entries_.insert(pos, EntryRef{obj, source, std::move(sol)});
                result.accepted = true;
            };

            if (std::cmp_less(entries_.size(), capacity_)) {
                insert_sorted();
            } else if (!(obj >= entries_.back().objective)) {
                // Spelled as the negation of "dominated", not `obj < worst`,
                // so a NaN objective takes this path as it always has.
                // Standard path: improves on worst — replace worst.
                evicted = std::move(entries_.back().solution);
                entries_.pop_back();
                insert_sorted();
            } else if (!integer_mask_->empty() && num_integers_ > 0 && !entries_.empty()) {
                // Diversity-aware path: pool is full and obj doesn't beat worst.
                // Accept if (a) integer mask is set, (b) obj is within tolerance of
                // best, and (c) solution is sufficiently diverse from all entries.
                double best_obj = entries_.front().objective;
                double gap = std::abs(obj - best_obj);
                // Continuous fallback: fraction of |best_obj|, floored to avoid
                // a discontinuous jump near zero.
                double threshold = std::max(kDiversityObjTolerance * std::abs(best_obj),
                                            kDiversityObjTolerance * 1e-6);

                if (gap <= threshold) {
                    if (scanned && scanned_generation == generation_) {
                        // The scan describes these very entries: decide.
                        const int victim = diversity_victim(dists, num_integers_);
                        if (victim > 0) {
                            // Replace the most similar entry other than the best.
                            evicted = std::move(entries_[victim].solution);
                            entries_.erase(entries_.begin() + victim);
                            insert_sorted();
                        }
                    } else {
                        // Measure outside the lock, then decide again.
                        scan.assign(entries_.begin(), entries_.end());
                        mask = integer_mask_;
                        scanned_generation = generation_;
                        need_scan = true;
                    }
                }
            }

            if (result.accepted) {
                ++generation_;
                // Advance the watermark only on an offer the pool kept, so a
                // refused solution cannot raise the bar the next one is judged
                // against.  It never moves backwards, which is what makes it
                // track the solve's incumbent rather than the pool's contents.
                if (result.improved_best) {
                    best_seen_ = obj;
                    has_best_seen_ = true;
                }
            }
        }

        if (!need_scan) {
            // `improved_best` implies `accepted` in every branch above — an
            // offer that beats the best beats the worst too, so it takes the
            // standard replacement path — but tie them together rather than
            // leaving that to a reader of the admission policy: a staleness
            // gate must never reset on a solution the pool did not keep.
            result.improved_best = result.improved_best && result.accepted;
            return result;
        }

        // The scan, outside the lock.  Entries are immutable, so a pointer
        // seen before is the same vector and its distance stands.
        dists.resize(scan.size());
        for (size_t idx = 0; idx < scan.size(); ++idx) {
            const auto& entry = scan[idx].solution;
            auto hit = std::ranges::find(seen, entry, &std::pair<SolutionPtr, int>::first);
            if (hit != seen.end()) {
                dists[idx] = hit->second;
            } else {
                dists[idx] = hamming_distance(*mask, *sol, *entry);
                seen.emplace_back(entry, dists[idx]);
            }
        }
        scanned = true;
    }
}

SolutionPool::Snapshot SolutionPool::snapshot() {
    std::scoped_lock lock(mtx_);
    if (entries_.empty()) {
        return {false, std::numeric_limits<double>::infinity()};
    }
    return {true, entries_[0].objective};
}

bool SolutionPool::is_integer_col(const std::vector<bool>& mask, int j) {
    return std::cmp_less(j, mask.size()) && mask[j];
}

// Guided crossover: keep integer values the parents agree on, coin-flip
// everything else.
void SolutionPool::guided_crossover(const std::vector<bool>& mask, const std::vector<double>& sol_a,
                                    const std::vector<double>& sol_b, Rng& rng,
                                    std::vector<double>& out) {
    int ncol = static_cast<int>(sol_a.size());
    out.resize(ncol);
    for (int j = 0; j < ncol; ++j) {
        if (is_integer_col(mask, j) && std::round(sol_a[j]) == std::round(sol_b[j])) {
            out[j] = sol_a[j];
        } else {
            // Disagree or continuous — coin flip.
            out[j] = std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? sol_a[j] : sol_b[j];
        }
    }
}

// Neighborhood crossover: the better parent provides the base and only
// disagreeing integer variables are coin-flipped.
void SolutionPool::neighborhood_crossover(const std::vector<bool>& mask,
                                          const std::vector<double>& sol_better,
                                          const std::vector<double>& sol_other, Rng& rng,
                                          std::vector<double>& out) {
    int ncol = static_cast<int>(sol_better.size());
    out.resize(ncol);
    for (int j = 0; j < ncol; ++j) {
        if (is_integer_col(mask, j) && std::round(sol_better[j]) != std::round(sol_other[j])) {
            out[j] =
                std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? sol_better[j] : sol_other[j];
        } else {
            // Agree or continuous — keep the better parent's value.
            out[j] = sol_better[j];
        }
    }
}

// Biased copy: half the time draw from the better half of the pool, half the
// time from anywhere.
int SolutionPool::biased_index(Rng& rng, int pool_size) {
    if (pool_size > 1 && std::uniform_int_distribution<int>(0, 1)(rng) == 0) {
        return std::uniform_int_distribution<int>(0, ((pool_size + 1) / 2) - 1)(rng);
    }
    return std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
}

bool SolutionPool::get_restart(Rng& rng, std::vector<double>& out) {
    enum class Strategy { kGuided, kNeighborhood, kCopy };
    Strategy strategy = Strategy::kCopy;
    SolutionPtr first;
    SolutionPtr second;
    MaskPtr mask;
    {
        // Every RNG draw that picks entries is made here, in the order the
        // strategies have always made them; the crossover's own coin flips
        // follow outside the lock, on the same stream.
        std::scoped_lock lock(mtx_);
        if (entries_.empty()) {
            return false;
        }

        int pool_size = static_cast<int>(entries_.size());
        double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

        // Two distinct parents, drawn uniformly.
        auto pick_two_parents = [&](int& a, int& b) {
            a = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
            b = std::uniform_int_distribution<int>(0, pool_size - 2)(rng);
            if (b >= a) {
                ++b;
            }
        };

        if (roll < 0.4 && pool_size >= 2) {
            int a = 0;
            int b = 0;
            pick_two_parents(a, b);
            strategy = Strategy::kGuided;
            first = entries_[a].solution;
            second = entries_[b].solution;
        } else if (roll < 0.7 && pool_size >= 2) {
            int a = 0;
            int b = 0;
            pick_two_parents(a, b);
            // Better parent = lower index (entries_ sorted best-first).
            strategy = Strategy::kNeighborhood;
            first = entries_[std::min(a, b)].solution;
            second = entries_[std::max(a, b)].solution;
        } else {
            first = entries_[biased_index(rng, pool_size)].solution;
        }
        mask = integer_mask_;
    }

    switch (strategy) {
        case Strategy::kGuided:
            guided_crossover(*mask, *first, *second, rng, out);
            break;
        case Strategy::kNeighborhood:
            neighborhood_crossover(*mask, *first, *second, rng, out);
            break;
        case Strategy::kCopy:
            out = *first;
            break;
    }
    return true;
}

void SolutionPool::entry_refs(std::vector<EntryRef>& out) {
    // Released and sized before the lock: dropping a stale reference can
    // free a vector, and growing `out` allocates.
    out.clear();
    out.reserve(capacity_ > 0 ? capacity_ : 0);
    std::scoped_lock lock(mtx_);
    out.assign(entries_.begin(), entries_.end());
}

std::vector<SolutionPool::Entry> SolutionPool::sorted_entries() {
    std::vector<EntryRef> refs;
    entry_refs(refs);
    std::vector<Entry> entries;
    entries.reserve(refs.size());
    for (const auto& ref : refs) {
        entries.push_back({ref.objective, *ref.solution, ref.source});
    }
    return entries;
}

bool SolutionPool::copy_best(std::vector<double>& out) {
    SolutionPtr best;
    {
        std::scoped_lock lock(mtx_);
        if (entries_.empty()) {
            return false;
        }
        best = entries_[0].solution;  // best — entries_ kept sorted by try_add.
    }
    out = *best;
    return true;
}

int SolutionPool::size() {
    std::scoped_lock lock(mtx_);
    return static_cast<int>(entries_.size());
}
