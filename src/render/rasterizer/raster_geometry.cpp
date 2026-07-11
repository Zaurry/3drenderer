#include "render/rasterizer/raster_geometry.h"

#include <cmath>

namespace renderer {

namespace {

RasterVertex interpolate_vertex(
    const RasterVertex& a,
    const RasterVertex& b,
    double t) {
    const float interpolation_t = static_cast<float>(t);
    RasterVertex result;
    result.view = a.view * (1.0f - interpolation_t) + b.view * interpolation_t;
    result.world = a.world * (1.0f - interpolation_t) + b.world * interpolation_t;
    result.uv = Vec2(
        a.uv.x() * (1.0f - interpolation_t) + b.uv.x() * interpolation_t,
        a.uv.y() * (1.0f - interpolation_t) + b.uv.y() * interpolation_t);
    result.normal = a.normal * (1.0f - interpolation_t) + b.normal * interpolation_t;
    return result;
}

}  // namespace

std::vector<RasterVertex> clip_triangle_to_near_plane(
    const std::array<RasterVertex, 3>& triangle,
    double near_z) {
    std::vector<RasterVertex> output;
    output.reserve(4);

    RasterVertex previous = triangle.back();
    bool previous_inside = previous.view.z() >= near_z;
    for (const RasterVertex& current : triangle) {
        const bool current_inside = current.view.z() >= near_z;
        if (current_inside != previous_inside) {
            const double denominator = current.view.z() - previous.view.z();
            if (std::abs(denominator) > 1e-15) {
                const double t = (near_z - previous.view.z()) / denominator;
                RasterVertex crossing = interpolate_vertex(previous, current, t);
                crossing.view.z() = static_cast<float>(near_z);
                output.push_back(crossing);
            }
        }
        if (current_inside) {
            output.push_back(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    return output;
}

Vec3 perspective_correct_weights(
    const Vec3& screen_weights,
    const Vec3& view_depths) {
    if (view_depths.x() <= 0.0 || view_depths.y() <= 0.0 || view_depths.z() <= 0.0) {
        return Vec3::Zero();
    }
    const Vec3 weighted(
        screen_weights.x() / view_depths.x(),
        screen_weights.y() / view_depths.y(),
        screen_weights.z() / view_depths.z());
    const double sum = weighted.x() + weighted.y() + weighted.z();
    if (!std::isfinite(sum) || sum <= 0.0) {
        return Vec3::Zero();
    }
    return weighted / static_cast<float>(sum);
}

}  // namespace renderer
