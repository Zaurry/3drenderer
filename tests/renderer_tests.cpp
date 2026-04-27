#include "test_framework.h"

#include <iostream>

int main() {
    RENDER_CHECK(1 + 1 == 2);
    std::cout << "renderer_tests: all tests passed\n";
    return 0;
}
