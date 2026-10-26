// SolutionPool's pointer entries (#180): the overload equivalence, the
// lifetime of a handed-out pointer, and concurrent use.  Needs nothing of
// HiGHS but the spin-lock header, so `mip_heuristics_pool_tsan_tests` builds
// this file alone with `src/solution_pool.cpp` under ThreadSanitizer.

#include "rng.h"
#include "solution_pool.h"

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <ios>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

// Source tags.  Any ints do: the pool stores them and compares nothing.
constexpr int kSourceA = 1;
constexpr int kSourceB = 2;
constexpr int kSourceC = 3;

namespace {

// A random offer stream built to exercise every admission path: a small
// pool, objectives in a narrow band so the diversity tolerance admits
// dominated offers, and binary columns so their Hamming distances vary.
struct Offer {
    double obj;
    std::vector<double> sol;
};

std::vector<Offer> make_offers(int count, int ncol, std::uint64_t seed) {
    Rng rng(seed);
    std::vector<Offer> offers;
    offers.reserve(count);
    for (int i = 0; i < count; ++i) {
        Offer offer{std::uniform_real_distribution<double>(100.0, 108.0)(rng),
                    std::vector<double>(ncol)};
        for (auto& v : offer.sol) {
            v = static_cast<double>(std::uniform_int_distribution<int>(0, 1)(rng));
        }
        offers.push_back(std::move(offer));
    }
    return offers;
}

}  // namespace

TEST_CASE("SolutionPool: the pointer try_add decides every offer as the vector one does",
          "[pool]") {
    constexpr int kNcol = 40;
    const auto offers = make_offers(/*count=*/400, kNcol, /*seed=*/7);

    SolutionPool by_ref(/*capacity=*/4);
    SolutionPool by_ptr(/*capacity=*/4);
    by_ref.set_integer_mask(std::vector<bool>(kNcol, true));
    by_ptr.set_integer_mask(std::vector<bool>(kNcol, true));

    int diversity_admissions = 0;
    for (const auto& offer : offers) {
        const double worst_before = by_ref.size() == 4 ? by_ref.sorted_entries().back().objective
                                                       : std::numeric_limits<double>::infinity();
        const auto a = by_ref.try_add(offer.obj, offer.sol, kSourceA);
        const auto b = by_ptr.try_add(
            offer.obj, std::make_shared<const std::vector<double>>(offer.sol), kSourceA);
        REQUIRE(a.accepted == b.accepted);
        REQUIRE(a.improved_best == b.improved_best);
        if (a.accepted && offer.obj >= worst_before) {
            ++diversity_admissions;
        }
    }
    // The stream must actually reach the path the scan outside the lock
    // changed, or the comparison above proves nothing about it.
    REQUIRE(diversity_admissions > 0);

    const auto ea = by_ref.sorted_entries();
    const auto eb = by_ptr.sorted_entries();
    REQUIRE(ea.size() == eb.size());
    for (size_t i = 0; i < ea.size(); ++i) {
        REQUIRE(ea[i].objective == eb[i].objective);
        REQUIRE(ea[i].source == eb[i].source);
        REQUIRE(ea[i].solution == eb[i].solution);
    }

    // Same entries and the same RNG state give the same restart draws.
    Rng ra(11);
    Rng rb(11);
    std::vector<double> out_a;
    std::vector<double> out_b;
    for (int i = 0; i < 200; ++i) {
        REQUIRE(by_ref.get_restart(ra, out_a));
        REQUIRE(by_ptr.get_restart(rb, out_b));
        REQUIRE(out_a == out_b);
    }
}

TEST_CASE("SolutionPool: an entry_refs pointer outlives the entry's eviction", "[pool]") {
    SolutionPool pool(/*capacity=*/2);
    REQUIRE(pool.try_add(10.0, {1.0, 2.0}, kSourceA).accepted);
    REQUIRE(pool.try_add(20.0, {3.0, 4.0}, kSourceB).accepted);

    std::vector<SolutionPool::EntryRef> refs;
    pool.entry_refs(refs);
    REQUIRE(refs.size() == 2);
    REQUIRE(refs[0].objective == 10.0);
    REQUIRE(refs[1].objective == 20.0);
    REQUIRE(refs[1].source == kSourceB);
    const SolutionPool::SolutionPtr held = refs[1].solution;

    // Both better than 20: the obj-20 entry is evicted on the first.
    REQUIRE(pool.try_add(5.0, {5.0, 6.0}, kSourceC).accepted);
    REQUIRE(pool.try_add(1.0, {7.0, 8.0}, kSourceC).accepted);
    for (const auto& entry : pool.sorted_entries()) {
        REQUIRE(entry.objective != 20.0);
    }
    REQUIRE(*held == std::vector<double>{3.0, 4.0});

    // A refill reuses `refs`' capacity and replaces its contents.
    const auto* storage = refs.data();
    pool.entry_refs(refs);
    REQUIRE(refs.data() == storage);
    REQUIRE(refs.size() == 2);
    REQUIRE(refs[0].objective == 1.0);
    REQUIRE(*refs[0].solution == std::vector<double>{7.0, 8.0});
}

TEST_CASE("SolutionPool: a non-finite objective or point is refused", "[pool]") {
    // #174: a NaN objective used to be admitted and ranked like any other.
    constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();
    SolutionPool pool(/*capacity=*/4);
    for (const double obj : {kNan, kInf, -kInf}) {
        const auto r = pool.try_add(obj, {1.0, 2.0}, kSourceA);
        REQUIRE_FALSE(r.accepted);
        REQUIRE_FALSE(r.improved_best);
    }
    for (const double v : {kNan, kInf, -kInf}) {
        const auto r = pool.try_add(1.0, {v, 2.0}, kSourceA);
        REQUIRE_FALSE(r.accepted);
        REQUIRE_FALSE(r.improved_best);
    }
    REQUIRE(pool.sorted_entries().empty());
    REQUIRE_FALSE(pool.snapshot().has_solution);

    // A finite offer after them is the pool's first.
    const auto r = pool.try_add(3.0, {1.0, 2.0}, kSourceA);
    REQUIRE(r.accepted);
    REQUIRE(r.improved_best);
}

// Also run under ThreadSanitizer by `mip_heuristics_pool_tsan_tests`.
TEST_CASE("SolutionPool: concurrent offers, reads and restarts on pointer entries",
          "[pool][thread-safety]") {
    constexpr int kNcol = 32;
    SolutionPool pool(/*capacity=*/6);
    pool.set_integer_mask(std::vector<bool>(kNcol, true));

    constexpr int kNumThreads = 4;
    constexpr int kOpsPerThread = 300;
    // Catch2 assertions are not thread-safe: workers count what they see,
    // and the test asserts on the count after the join.
    std::atomic<int> violations{0};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&pool, &violations, t]() {
            auto check = [&](bool ok) {
                if (!ok) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
            };
            const auto offers = make_offers(kOpsPerThread, kNcol, 100 + t);
            Rng rng(42 + t);
            std::vector<SolutionPool::EntryRef> refs;
            std::vector<double> buf;
            for (const auto& offer : offers) {
                if (t % 2 == 0) {
                    static_cast<void>(pool.try_add(offer.obj, offer.sol, kSourceA));
                } else {
                    static_cast<void>(pool.try_add(
                        offer.obj, std::make_shared<const std::vector<double>>(offer.sol),
                        kSourceB));
                }
                pool.entry_refs(refs);
                for (size_t i = 1; i < refs.size(); ++i) {
                    check(refs[i - 1].objective <= refs[i].objective);
                }
                for (const auto& ref : refs) {
                    check(ref.solution->size() == static_cast<size_t>(kNcol));
                }
                if (pool.copy_best(buf)) {
                    check(buf.size() == static_cast<size_t>(kNcol));
                }
                if (pool.get_restart(rng, buf)) {
                    check(buf.size() == static_cast<size_t>(kNcol));
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    REQUIRE(violations.load() == 0);

    const auto entries = pool.sorted_entries();
    REQUIRE(entries.size() == 6);
    for (size_t i = 1; i < entries.size(); ++i) {
        REQUIRE(entries[i - 1].objective <= entries[i].objective);
    }
}

// ── The scan outside the lock, re-decided when the pool moved under it ──

namespace {

constexpr int kRedecideNcol = 20;

// Three entries and an offer built so the offer is dominated (it ties the
// worst), sits within the diversity tolerance of the best, and is two
// columns from the best and far from the rest: under an all-integer mask it
// is diverse enough to be admitted.
void seed_three(SolutionPool& pool) {
    std::vector<double> best(kRedecideNcol, 0.0);
    std::vector<double> mid(kRedecideNcol, 1.0);
    std::vector<double> worst(kRedecideNcol, 1.0);
    worst[0] = 0.0;
    REQUIRE(pool.try_add(100.0, best, kSourceA).accepted);
    REQUIRE(pool.try_add(101.0, mid, kSourceA).accepted);
    REQUIRE(pool.try_add(102.0, worst, kSourceA).accepted);
}

std::vector<double> near_best_offer() {
    std::vector<double> offer(kRedecideNcol, 0.0);
    offer[0] = 1.0;
    offer[1] = 1.0;
    return offer;
}

void require_same_pool(SolutionPool& a, SolutionPool& b) {
    const auto ea = a.sorted_entries();
    const auto eb = b.sorted_entries();
    REQUIRE(ea.size() == eb.size());
    for (size_t i = 0; i < ea.size(); ++i) {
        REQUIRE(ea[i].objective == eb[i].objective);
        REQUIRE(ea[i].source == eb[i].source);
        REQUIRE(ea[i].solution == eb[i].solution);
    }
}

}  // namespace

TEST_CASE("SolutionPool: a mask changed mid-scan is decided under the new mask", "[pool]") {
    // The mask the offer ends up judged under marks only the columns where
    // it agrees with the best entry, so it is no longer diverse.  Distances
    // measured under the old mask must not be reused against the new one.
    std::vector<bool> narrow(kRedecideNcol, true);
    narrow[0] = false;
    narrow[1] = false;

    SolutionPool pool(/*capacity=*/3);
    pool.set_integer_mask(std::vector<bool>(kRedecideNcol, true));
    seed_three(pool);
    int hook_calls = 0;
    pool.set_scan_hook_for_test([&] {
        if (hook_calls++ == 0) {
            pool.set_integer_mask(narrow);
        }
    });
    const auto moved = pool.try_add(102.0, near_best_offer(), kSourceB);
    REQUIRE(hook_calls == 2);  // scanned, then rescanned under the new mask

    // One hold of the lock in the final state decides it this way.
    SolutionPool reference(/*capacity=*/3);
    reference.set_integer_mask(narrow);
    seed_three(reference);
    const auto direct = reference.try_add(102.0, near_best_offer(), kSourceB);

    REQUIRE_FALSE(direct.accepted);
    REQUIRE(moved.accepted == direct.accepted);
    REQUIRE(moved.improved_best == direct.improved_best);
    require_same_pool(pool, reference);
}

TEST_CASE("SolutionPool: entries changed mid-scan are rescanned before the decision", "[pool]") {
    // A peer's solution lands during the scan: the offer's own vector, at
    // an objective inside the diversity band.  The decision must then go
    // through a rescan against the entries as they now stand, where the
    // offer duplicates the new best and is refused.  Applying the first
    // scan's distances to the new entries instead admits it (they say it is
    // two columns from its nearest entry) and evicts by a stale index.
    SolutionPool pool(/*capacity=*/3);
    pool.set_integer_mask(std::vector<bool>(kRedecideNcol, true));
    seed_three(pool);
    int hook_calls = 0;
    pool.set_scan_hook_for_test([&] {
        if (hook_calls++ == 0) {
            REQUIRE(pool.try_add(99.5, near_best_offer(), kSourceC).accepted);
        }
    });
    const auto moved = pool.try_add(102.0, near_best_offer(), kSourceB);
    REQUIRE(hook_calls == 2);  // scanned, then rescanned against the new entries

    SolutionPool reference(/*capacity=*/3);
    reference.set_integer_mask(std::vector<bool>(kRedecideNcol, true));
    seed_three(reference);
    REQUIRE(reference.try_add(99.5, near_best_offer(), kSourceC).accepted);
    const auto direct = reference.try_add(102.0, near_best_offer(), kSourceB);

    REQUIRE_FALSE(direct.accepted);
    REQUIRE(moved.accepted == direct.accepted);
    REQUIRE(moved.improved_best == direct.improved_best);
    require_same_pool(pool, reference);

    // Without the move, the same offer is admitted: the first scan's verdict.
    SolutionPool unmoved(/*capacity=*/3);
    unmoved.set_integer_mask(std::vector<bool>(kRedecideNcol, true));
    seed_three(unmoved);
    REQUIRE(unmoved.try_add(102.0, near_best_offer(), kSourceB).accepted);
}

TEST_CASE("SolutionPool: a best-only restart is the best entry, drawing nothing", "[pool]") {
    SolutionPool pool(/*capacity=*/4);
    std::vector<double> out;
    Rng rng(7);
    REQUIRE_FALSE(pool.get_restart(rng, out, SolutionPool::RestartPolicy::kBestOnly));
    REQUIRE(pool.try_add(20.0, {0.0, 1.0, 0.0}, kSourceA).accepted);
    REQUIRE(pool.try_add(10.0, {1.0, 1.0, 0.0}, kSourceB).accepted);
    REQUIRE(pool.try_add(30.0, {0.0, 0.0, 1.0}, kSourceC).accepted);

    Rng untouched = rng;
    for (int i = 0; i < 50; ++i) {
        REQUIRE(pool.get_restart(rng, out, SolutionPool::RestartPolicy::kBestOnly));
        REQUIRE(out == std::vector<double>{1.0, 1.0, 0.0});
    }
    REQUIRE(rng() == untouched());  // the same stream: nothing was drawn
    // A new best is what the next restart draws.
    REQUIRE(pool.try_add(5.0, {1.0, 0.0, 1.0}, kSourceA).accepted);
    REQUIRE(pool.get_restart(rng, out, SolutionPool::RestartPolicy::kBestOnly));
    REQUIRE(out == std::vector<double>{1.0, 0.0, 1.0});
}

namespace {

// The diversity rule as it stood before #195 made the scan fast, kept here
// as the reference the pool is held to: every entry measured, `std::round`
// on every column the flag array marks.
class ReferencePool {
public:
    ReferencePool(int capacity, std::vector<bool> mask)
        : capacity_(capacity), mask_(std::move(mask)) {
        for (const bool b : mask_) {
            num_integers_ += b ? 1 : 0;
        }
    }

    bool try_add(double obj, const std::vector<double>& sol) {
        if (std::cmp_less(entries_.size(), capacity_)) {
            insert(obj, sol);
            return true;
        }
        if (obj < entries_.back().first) {
            entries_.pop_back();
            insert(obj, sol);
            return true;
        }
        const double best = entries_.front().first;
        if (num_integers_ == 0 ||
            std::abs(obj - best) >
                std::max(kDiversityObjTolerance * std::abs(best), kDiversityObjTolerance * 1e-6)) {
            return false;
        }
        ++diversity_offers_;
        int min_dist = std::numeric_limits<int>::max();
        int evict_dist = std::numeric_limits<int>::max();
        int victim = -1;
        for (int idx = 0; std::cmp_less(idx, entries_.size()); ++idx) {
            int dist = 0;
            for (int j = 0; std::cmp_less(j, sol.size()); ++j) {
                if (mask_[j] && std::round(sol[j]) != std::round(entries_[idx].second[j])) {
                    ++dist;
                }
            }
            min_dist = std::min(min_dist, dist);
            if (idx > 0 && dist < evict_dist) {
                evict_dist = dist;
                victim = idx;
            }
        }
        if (static_cast<double>(min_dist) / num_integers_ < kDiversityMinHammingFrac ||
            victim <= 0) {
            return false;
        }
        ++diversity_accepts_;
        entries_.erase(entries_.begin() + victim);
        insert(obj, sol);
        return true;
    }

    [[nodiscard]] int diversity_offers() const { return diversity_offers_; }
    [[nodiscard]] int diversity_accepts() const { return diversity_accepts_; }

    [[nodiscard]] const std::vector<std::pair<double, std::vector<double>>>& entries() const {
        return entries_;
    }

private:
    void insert(double obj, const std::vector<double>& sol) {
        // NOLINTNEXTLINE(modernize-use-ranges)
        auto pos = std::lower_bound(entries_.begin(), entries_.end(), obj,
                                    [](const auto& e, double v) { return e.first < v; });
        entries_.insert(pos, {obj, sol});
    }

    int capacity_;
    std::vector<bool> mask_;
    int num_integers_ = 0;
    int diversity_offers_ = 0;
    int diversity_accepts_ = 0;
    std::vector<std::pair<double, std::vector<double>>> entries_;
};

}  // namespace

TEST_CASE("SolutionPool: the diversity scan decides every offer as the full std::round scan does",
          "[pool]") {
    // Mixed columns, a third continuous, and values on and off the rounding
    // ties, negative ones included, so the distances depend on how a value
    // rounds and on which columns count; objectives in a band narrow enough
    // that most offers take the diversity path, where the scan stops at the
    // first entry too close to the offer.
    constexpr int kNcol = 30;
    constexpr int kOffers = 3000;
    std::vector<bool> mask(kNcol);
    for (int j = 0; j < kNcol; ++j) {
        mask[j] = j % 3 != 1;
    }
    const std::vector<double> values = {
        -2.5, -1.5, -1.0, -0.5, -0.49999999999999994, -0.0, 0.0, 0.3, 0.5,
        0.7,  1.0,  1.5,  2.5,  0.49999999999999994};
    Rng rng(11);
    auto pick_value = [&] {
        return values[std::uniform_int_distribution<size_t>(0, values.size() - 1)(rng)];
    };
    // Offers are a few bank vectors with a column or two redrawn, so they
    // sit near the entries and the scan refuses as well as admits.
    std::vector<std::vector<double>> bank(4, std::vector<double>(kNcol));
    for (auto& base : bank) {
        std::ranges::generate(base, pick_value);
    }
    SolutionPool pool(/*capacity=*/5);
    pool.set_integer_mask(mask);
    ReferencePool reference(/*capacity=*/5, mask);
    for (int i = 0; i < kOffers; ++i) {
        const double obj = std::uniform_real_distribution<double>(100.0, 104.0)(rng);
        std::vector<double> sol =
            bank[std::uniform_int_distribution<size_t>(0, bank.size() - 1)(rng)];
        for (auto& v : sol) {
            if (std::uniform_int_distribution<int>(0, kNcol - 1)(rng) == 0) {
                v = pick_value();
            }
        }
        const bool expected = reference.try_add(obj, sol);
        REQUIRE(pool.try_add(obj, sol, kSourceA).accepted == expected);
    }
    // Both diversity verdicts were reached, many times over.
    REQUIRE(reference.diversity_accepts() > 50);
    REQUIRE(reference.diversity_offers() - reference.diversity_accepts() > 50);
    const auto entries = pool.sorted_entries();
    REQUIRE(entries.size() == reference.entries().size());
    for (size_t i = 0; i < entries.size(); ++i) {
        REQUIRE(entries[i].objective == reference.entries()[i].first);
        REQUIRE(entries[i].solution == reference.entries()[i].second);
    }
}

TEST_CASE("SolutionPool: a column's Hamming term is std::round's verdict at every rounding edge",
          "[pool]") {
    // Twenty integer columns, so one differing column is exactly the 5%
    // diversity threshold.  The best entry and the offer agree everywhere
    // but column 0, where they hold `y` and `x`; the worst entry also
    // differs in column 19, so it is never the offer's nearest.  The offer,
    // inside the diversity band and not beating the worst, is admitted
    // exactly when `x` and `y` round apart.
    constexpr int kNcol = 21;  // the last column continuous
    std::vector<bool> mask(kNcol, true);
    mask.back() = false;
    const double big = 4503599627370496.0;  // 2^52
    const std::vector<double> values = {0.0,
                                        -0.0,
                                        0.49999999999999994,
                                        0.5,
                                        0.5000000000000001,
                                        -0.49999999999999994,
                                        -0.5,
                                        1.5,
                                        2.5,
                                        -2.5,
                                        1e-300,
                                        big - 0.5,
                                        big - 1.0,
                                        big,
                                        big + 2.0,
                                        -(big - 0.5),
                                        1e300,
                                        -1e300};
    for (const double x : values) {
        for (const double y : values) {
            SolutionPool pool(/*capacity=*/2);
            pool.set_integer_mask(mask);
            std::vector<double> best(kNcol, 0.0);
            best[0] = y;
            std::vector<double> worst = best;
            worst[19] = 7.0;
            REQUIRE(pool.try_add(1.0, best, kSourceA).accepted);
            REQUIRE(pool.try_add(1.05, worst, kSourceB).accepted);
            std::vector<double> offer = best;
            offer[0] = x;
            offer.back() = 3.25;  // continuous: never counted
            INFO("x = " << std::hexfloat << x << ", y = " << y);
            REQUIRE(pool.try_add(1.05, offer, kSourceC).accepted ==
                    (std::round(x) != std::round(y)));
        }
    }
}
