#pragma once

#include <algorithm>
#include <cmath>

namespace renderer {

struct Vec3 {
    double x;
    double y;
    double z;

    constexpr Vec3() : x(0.0), y(0.0), z(0.0) {}
    constexpr Vec3(double x_value, double y_value, double z_value) : x(x_value), y(y_value), z(z_value) {}

    constexpr Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    constexpr Vec3& operator-=(const Vec3& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    constexpr Vec3& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    constexpr Vec3& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

constexpr Vec3 operator+(Vec3 lhs, const Vec3& rhs) {
    lhs += rhs;
    return lhs;
}

constexpr Vec3 operator-(Vec3 lhs, const Vec3& rhs) {
    lhs -= rhs;
    return lhs;
}

constexpr Vec3 operator-(const Vec3& v) {
    return Vec3(-v.x, -v.y, -v.z);
}

constexpr Vec3 operator*(Vec3 v, double scalar) {
    v *= scalar;
    return v;
}

constexpr Vec3 operator*(double scalar, Vec3 v) {
    v *= scalar;
    return v;
}

constexpr Vec3 operator/(Vec3 v, double scalar) {
    v /= scalar;
    return v;
}

constexpr double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

constexpr double length_squared(const Vec3& v) {
    return dot(v, v);
}

inline double length(const Vec3& v) {
    return std::sqrt(length_squared(v));
}

inline Vec3 normalize(const Vec3& v) {
    const double len = length(v);
    if (len == 0.0) {
        return Vec3();
    }
    return v / len;
}

inline Vec3 min_components(const Vec3& a, const Vec3& b) {
    return Vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

inline Vec3 max_components(const Vec3& a, const Vec3& b) {
    return Vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

}  // namespace renderer
