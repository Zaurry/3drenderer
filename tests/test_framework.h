#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

inline void test_check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        std::cerr << file << ":" << line << " check failed: " << expression << "\n";
        std::exit(1);
    }
}

inline bool nearly_equal(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

#define RENDER_CHECK(expr) test_check((expr), #expr, __FILE__, __LINE__)
