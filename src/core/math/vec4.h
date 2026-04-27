#pragma once

namespace renderer {

struct Vec4 {
    double x;
    double y;
    double z;
    double w;

    constexpr Vec4() : x(0.0), y(0.0), z(0.0), w(0.0) {}
    constexpr Vec4(double x_value, double y_value, double z_value, double w_value)
        : x(x_value), y(y_value), z(z_value), w(w_value) {}
};

}  // namespace renderer
