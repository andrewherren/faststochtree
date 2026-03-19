#pragma once
#include <random>

namespace bart {

struct RNG {
    std::mt19937 gen;
    explicit RNG(unsigned seed = 42) : gen(seed) {}

    float uniform() {
        return std::uniform_real_distribution<float>(0.f, 1.f)(gen);
    }
    float normal() {
        return std::normal_distribution<float>(0.f, 1.f)(gen);
    }
    // Draw x ~ Gamma(shape, 1)
    float gamma(float shape) {
        return std::gamma_distribution<float>(shape, 1.f)(gen);
    }
    // Uniform integer in [lo, hi)
    int randint(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi - 1)(gen);
    }
};

} // namespace bart
