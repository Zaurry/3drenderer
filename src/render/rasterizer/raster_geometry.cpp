#include "render/rasterizer/raster_geometry.h"

#include <cmath>

namespace renderer {

namespace {

RasterVertex interpolate_vertex(
    const RasterVertex& a,
    const RasterVertex& b,
    float t) {
    RasterVertex result;
    result.view = a.view * (1.0f - t) + b.view * t;
    result.world = a.world * (1.0f - t) + b.world * t;
    result.uv = a.uv * (1.0f - t) + b.uv * t;
    result.normal = a.normal * (1.0f - t) + b.normal * t;
    return result;
}

}  // namespace

std::vector<RasterVertex> clip_triangle_to_near_plane(
    const std::array<RasterVertex, 3>& triangle,
    float near_z) {
    std::vector<RasterVertex> output;
    output.reserve(4);

    RasterVertex previous = triangle.back();
    bool previous_inside = previous.view.z() >= near_z;
    for (const RasterVertex& current : triangle) {
        const bool current_inside = current.view.z() >= near_z;
        if (current_inside != previous_inside) {
            const float denominator = current.view.z() - previous.view.z();
            const float t = (near_z - previous.view.z()) / denominator;
            RasterVertex crossing = interpolate_vertex(previous, current, t);
            crossing.view.z() = near_z;
            output.push_back(crossing);
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
    if (view_depths.x() <= 0.0f || view_depths.y() <= 0.0f || view_depths.z() <= 0.0f) {
        return Vec3::Zero();
    }
    const Vec3 weighted(
        screen_weights.x() / view_depths.x(),
        screen_weights.y() / view_depths.y(),
        screen_weights.z() / view_depths.z());
    const float sum = weighted.sum();
    if (!std::isfinite(sum) || sum <= 0.0f) {
        return Vec3::Zero();
    }
    return weighted / sum;
}

}  // namespace renderer
