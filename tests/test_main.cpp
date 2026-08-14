#include "test_framework.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool matches_filter(const char* name, const std::vector<std::string>& filters) {
    if (filters.empty()) {
        return true;
    }
    const std::string test_name(name);
    for (const std::string& filter : filters) {
        if (!filter.empty() && test_name.find(filter) != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    // Flush every line so crashes inside tests do not swallow diagnostics.
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    std::vector<std::string> filters;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--filter" && index + 1 < argc) {
            ++index;
            const std::string value = argv[index];
            std::size_t begin = 0;
            while (begin <= value.size()) {
                const std::size_t end = value.find(',', begin);
                const std::size_t stop =
                    end == std::string::npos ? value.size() : end;
                filters.push_back(value.substr(begin, stop - begin));
                if (end == std::string::npos) {
                    break;
                }
                begin = end + 1;
            }
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return 1;
        }
    }

    int passed = 0;
    int failed = 0;
    int skipped = 0;
    for (const TestCase& test : test_registry()) {
        if (!matches_filter(test.name, filters)) {
            continue;
        }
        std::cout << "[RUN] " << test.name << '\n';
        try {
            test.function();
            ++passed;
        } catch (const TestFailure& failure) {
            ++failed;
            std::cerr << "[FAIL] " << test.name << ": "
                      << failure.message << '\n';
        } catch (const TestSkipSignal& skip) {
            ++skipped;
            std::cout << "[SKIP] " << test.name << ": "
                      << skip.reason << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << test.name
                      << ": unexpected exception: " << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << test.name
                      << ": unknown exception\n";
        }
    }

    std::cout << "tests: " << passed << " passed, " << failed << " failed, "
              << skipped << " skipped\n";
    if (failed > 0) {
        return 1;
    }
    if (passed == 0 && skipped > 0) {
        return 77;
    }
    return 0;
}
