#pragma once

namespace renderer {

struct Vec2 {
    double x;
    double y;

    constexpr Vec2() : x(0.0), y(0.0) {}
    constexpr Vec2(double x_value, double y_value) : x(x_value), y(y_value) {}
};

}  // namespace renderer
