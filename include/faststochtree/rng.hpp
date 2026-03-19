#pragma once
#include <random>

namespace bart {

struct RNG {
    std::mt19937 gen;
    explicit RNG(unsigned seed = 42) : gen(seed) {}

    double uniform() {
        return std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    }
    double normal() {
        return std::normal_distribution<double>(0.0, 1.0)(gen);
    }
    // Draw x ~ Gamma(shape, 1)
    double gamma(double shape) {
        return std::gamma_distribution<double>(shape, 1.0)(gen);
    }
    // Uniform integer in [lo, hi)
    int randint(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi - 1)(gen);
    }
};

} // namespace bart
