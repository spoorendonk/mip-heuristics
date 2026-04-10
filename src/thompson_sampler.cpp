#include "thompson_sampler.h"

#include <cassert>
#include <mutex>

ThompsonSampler::ThompsonSampler(int num_arms, const double* prior_alpha, bool use_mutex)
    : use_mutex_(use_mutex) {
    arms_.reserve(num_arms);
    for (int i = 0; i < num_arms; ++i) {
        arms_.push_back({prior_alpha[i], 1.0, 0, 0.0, false});
    }
}

int ThompsonSampler::select_unlocked(std::mt19937& rng) {
    int best_arm = 0;
    double best_sample = -1.0;

    for (int i = 0; i < static_cast<int>(arms_.size()); ++i) {
        std::gamma_distribution<double> gamma_a(arms_[i].alpha, 1.0);
        std::gamma_distribution<double> gamma_b(arms_[i].beta, 1.0);
        double a = gamma_a(rng);
        double b = gamma_b(rng);
        double sample = (a + b > 0.0) ? a / (a + b) : 0.5;
        if (sample > best_sample) {
            best_sample = sample;
            best_arm = i;
        }
    }
    return best_arm;
}

int ThompsonSampler::select(std::mt19937& rng) {
    if (use_mutex_) {
        std::lock_guard<HighsSpinMutex> lock(mtx_);
        return select_unlocked(rng);
    }
    return select_unlocked(rng);
}

int ThompsonSampler::select_effort_aware(std::mt19937& rng) {
    auto impl = [&]() -> int {
        // Check if all arms have effort observations
        bool all_have_effort = true;
        for (const auto& a : arms_) {
            if (!a.has_effort) {
                all_have_effort = false;
                break;
            }
        }
        if (!all_have_effort) {
            return select_unlocked(rng);
        }

        // Effort-aware: weight Thompson sample by 1/avg_effort (reward rate)
        int best_arm = 0;
        double best_score = -1.0;
        for (int i = 0; i < static_cast<int>(arms_.size()); ++i) {
            std::gamma_distribution<double> gamma_a(arms_[i].alpha, 1.0);
            std::gamma_distribution<double> gamma_b(arms_[i].beta, 1.0);
            double a = gamma_a(rng);
            double b = gamma_b(rng);
            double sample = (a + b > 0.0) ? a / (a + b) : 0.5;
            // Divide by effort to get reward-per-unit-effort rate.
            // Guard against zero-effort arms (near-instant returns).
            double effort = std::max(arms_[i].avg_effort, 1.0);
            double score = sample / effort;
            if (score > best_score) {
                best_score = score;
                best_arm = i;
            }
        }
        return best_arm;
    };

    if (use_mutex_) {
        std::lock_guard<HighsSpinMutex> lock(mtx_);
        return impl();
    }
    return impl();
}

void ThompsonSampler::record_effort(int arm, size_t effort_units) {
    assert(arm >= 0 && arm < static_cast<int>(arms_.size()));

    auto impl = [&]() {
        constexpr double kEmaAlpha = 0.3;
        double effort = static_cast<double>(effort_units);
        if (!arms_[arm].has_effort) {
            arms_[arm].avg_effort = effort;
            arms_[arm].has_effort = true;
        } else {
            arms_[arm].avg_effort = kEmaAlpha * effort + (1.0 - kEmaAlpha) * arms_[arm].avg_effort;
        }
    };

    if (use_mutex_) {
        std::lock_guard<HighsSpinMutex> lock(mtx_);
        impl();
    } else {
        impl();
    }
}

void ThompsonSampler::update(int arm, int reward) {
    assert(arm >= 0 && arm < static_cast<int>(arms_.size()));
    assert(reward >= 0 && reward <= 3);

    auto impl = [&]() {
        arms_[arm].pulls++;
        switch (reward) {
            case 0:
                arms_[arm].beta += 1.0;
                break;
            case 1:
                arms_[arm].beta += 0.25;
                break;
            case 2:
                arms_[arm].alpha += 1.0;
                break;
            case 3:
                arms_[arm].alpha += 1.5;
                break;
            default:
                break;
        }
    };

    if (use_mutex_) {
        std::lock_guard<HighsSpinMutex> lock(mtx_);
        impl();
    } else {
        impl();
    }
}

ThompsonSampler::ArmStats ThompsonSampler::stats(int arm) const {
    assert(arm >= 0 && arm < static_cast<int>(arms_.size()));
    if (use_mutex_) {
        std::lock_guard<HighsSpinMutex> lock(mtx_);
        return {arms_[arm].alpha, arms_[arm].beta, arms_[arm].pulls, arms_[arm].avg_effort};
    }
    return {arms_[arm].alpha, arms_[arm].beta, arms_[arm].pulls, arms_[arm].avg_effort};
}
