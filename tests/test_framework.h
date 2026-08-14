#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Tiny registry-based test harness shared by every test binary.
//
// RENDER_TEST(name) registers a test function; tests/test_main.cpp provides
// main(), which runs every registered test, supports --filter <substring[,substring]>,
// catches exceptions per test, and exits:
//   0  - every executed test passed (skips allowed)
//   1  - at least one failure
//   77 - nothing was verified (every executed test skipped); CTest treats 77
//        as "skipped" via SKIP_RETURN_CODE.

struct TestFailure {
    std::string message;
};

inline void test_check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw TestFailure{
            std::string(file) + ":" + std::to_string(line) +
            " check failed: " + expression};
    }
}

#define RENDER_CHECK(expression) test_check((expression), #expression, __FILE__, __LINE__)

inline bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

struct TestCase {
    const char* name;
    void (*function)();
};

inline std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

struct TestRegistrar {
    TestRegistrar(const char* name, void (*function)()) {
        test_registry().push_back(TestCase{name, function});
    }
};

#define RENDER_TEST(name)                                \
    static void name();                                  \
    static TestRegistrar name##_registrar(#name, &name); \
    static void name()

struct TestSkipSignal {
    std::string reason;
};

// Marks the current test as skipped: it was not verified on this machine
// (e.g. CUDA unavailable). The runner counts it separately from passes.
#define RENDER_SKIP(reason) throw TestSkipSignal{reason}
