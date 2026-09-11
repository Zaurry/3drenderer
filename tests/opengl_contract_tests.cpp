#include "test_framework.h"

#include "render/opengl/opengl_raster_renderer.h"
#include "render/opengl/opengl_ao_math.h"
#include "render/opengl/opengl_shadow_math.h"
#include "render/opengl/opengl_ssr_math.h"
#include "render/opengl/tonal_art_map.h"

#include <array>

RENDER_TEST(test_tonal_art_map_nesting_and_mips) {
    const auto map = renderer::make_tonal_art_map();
    RENDER_CHECK(map.mips == renderer::make_tonal_art_map().mips);
    int width = renderer::TonalArtMap::size;
    for (const auto& mip : map.mips) {
        const int area = width * width;
        RENDER_CHECK(mip.size() == static_cast<std::size_t>(area * renderer::TonalArtMap::tones));
        int previous_coverage = -1;
        for (int tone = 0; tone < renderer::TonalArtMap::tones; ++tone) {
            int coverage = 0;
            for (int pixel = 0; pixel < area; ++pixel) {
                const auto ink = mip[tone * area + pixel];
                if (tone == 0) RENDER_CHECK(ink == 0);
                else RENDER_CHECK(ink >= mip[(tone - 1) * area + pixel]);
                coverage += ink;
            }
            RENDER_CHECK(coverage > previous_coverage);
            previous_coverage = coverage;
        }
        width /= 2;
    }
    RENDER_CHECK(width == 0);
}

RENDER_TEST(test_npr_style_and_diagnostic_routing) {
    renderer::OpenGlRenderSettings settings;
    RENDER_CHECK(!renderer::open_gl_npr_active(settings));
    for (const auto style : {renderer::OpenGlRenderStyle::Toon, renderer::OpenGlRenderStyle::Sketch}) {
        settings.npr.style = style;
        RENDER_CHECK(renderer::open_gl_npr_active(settings));
        RENDER_CHECK(!renderer::open_gl_ssr_requested(settings));
        settings.ssr.debug_view = renderer::OpenGlSsrDebugView::RawIndirect;
        RENDER_CHECK(!renderer::open_gl_npr_active(settings));
        RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
        settings.ssr.debug_view = renderer::OpenGlSsrDebugView::Final;
    }
    settings.npr.style = renderer::OpenGlRenderStyle::Realistic;
    RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
}

RENDER_TEST(test_open_gl_geometry_upload_predicate) {
    RENDER_CHECK(renderer::open_gl_requires_geometry_upload(
        renderer::SceneChange::Geometry));
    RENDER_CHECK(renderer::open_gl_requires_geometry_upload(
        renderer::SceneChange::MaterialBindings));
    RENDER_CHECK(renderer::open_gl_requires_geometry_upload(
        renderer::SceneChange::InstanceTransforms));
    RENDER_CHECK(!renderer::open_gl_requires_geometry_upload(
        renderer::SceneChange::Materials));
    RENDER_CHECK(!renderer::open_gl_requires_geometry_upload(
        renderer::SceneChange::Lighting));
}

RENDER_TEST(test_open_gl_shadow_depth_roundtrip_and_pcss_math) {
    constexpr float near_plane = 0.25f;
    constexpr float far_plane = 20.0f;
    for (const float distance : {near_plane, 1.0f, 7.5f, far_plane}) {
        const float depth = renderer::open_gl_normalized_linear_shadow_depth(
            distance,
            near_plane,
            far_plane);
        RENDER_CHECK(nearly_equal(
            renderer::open_gl_reconstruct_linear_shadow_distance(
                depth,
                near_plane,
                far_plane),
            distance,
            1.0e-6f));
    }
    const float close_receiver = renderer::open_gl_pcss_penumbra_texels(
        4.0f, 3.0f, 8.0f, 64.0f);
    const float distant_receiver = renderer::open_gl_pcss_penumbra_texels(
        8.0f, 3.0f, 8.0f, 64.0f);
    const float closer_blocker = renderer::open_gl_pcss_penumbra_texels(
        8.0f, 2.0f, 8.0f, 64.0f);
    RENDER_CHECK(close_receiver > 0.0f);
    RENDER_CHECK(distant_receiver > close_receiver);
    RENDER_CHECK(closer_blocker > distant_receiver);
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_pcss_penumbra_texels(
            100.0f, 1.0f, 8.0f, 64.0f),
        64.0f));
    RENDER_CHECK(renderer::open_gl_shadow_budget_precedes(
        2, 1.0f, 1, 1000.0f));
    RENDER_CHECK(renderer::open_gl_shadow_budget_precedes(
        2, 10.0f, 2, 1.0f));
    RENDER_CHECK(!renderer::open_gl_shadow_budget_precedes(
        1, 1000.0f, 2, 1.0f));
}

RENDER_TEST(test_open_gl_directional_shadow_fit) {
    const renderer::Bounds3 fitted_bounds(
        renderer::Vec3(-3.0f, -1.0f, -5.0f),
        renderer::Vec3(4.0f, 6.0f, 2.0f));
    const renderer::OpenGlDirectionalShadowFit fitted =
        renderer::open_gl_fit_directional_shadow(
            fitted_bounds,
            renderer::Vec3(-0.3f, -1.0f, 0.2f),
            0.0f,
            1024);
    for (const renderer::Vec3& corner :
         renderer::open_gl_shadow_bounds_corners(fitted_bounds)) {
        const renderer::Vec3 relative = corner - fitted.position;
        RENDER_CHECK(std::abs(fitted.right.dot(relative)) <=
            fitted.half_width + 1.0e-4f);
        RENDER_CHECK(std::abs(fitted.up.dot(relative)) <=
            fitted.half_height + 1.0e-4f);
        const float depth = fitted.direction.dot(relative);
        RENDER_CHECK(depth >= fitted.near_plane - 1.0e-4f);
        RENDER_CHECK(depth <= fitted.far_plane + 1.0e-4f);
    }
}

RENDER_TEST(test_open_gl_ao_view_position_and_projected_radius) {
    const renderer::Vec2 camera_viewport(2.0f, 1.0f);
    RENDER_CHECK(renderer::open_gl_ao_reconstruct_view_position(
        renderer::Vec2(0.5f, 0.5f), 4.0f, camera_viewport).isApprox(
        renderer::Vec3(0.0f, 0.0f, -4.0f)));
    RENDER_CHECK(renderer::open_gl_ao_reconstruct_view_position(
        renderer::Vec2(1.0f, 1.0f), 4.0f, camera_viewport).isApprox(
        renderer::Vec3(4.0f, 2.0f, -4.0f)));
    const float near_projected_radius =
        renderer::open_gl_ao_projected_radius_pixels(
            0.5f, 2.0f, 1.0f, 1080);
    const float far_projected_radius =
        renderer::open_gl_ao_projected_radius_pixels(
            0.5f, 8.0f, 1.0f, 1080);
    RENDER_CHECK(near_projected_radius > far_projected_radius);
    RENDER_CHECK(renderer::open_gl_ssao_range_weight(1.0f, 0.25f) >
        renderer::open_gl_ssao_range_weight(1.0f, 4.0f));
}

RENDER_TEST(test_open_gl_gtao_slice_math) {
    constexpr float half_pi = 1.57079632679489661923f;
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_gtao_slice_visibility(0.0f, -half_pi, half_pi),
        1.0f,
        1.0e-5f));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_gtao_slice_visibility(0.0f, 0.0f, 0.0f),
        0.0f,
        1.0e-5f));
    const renderer::Vec3 symmetric_bent =
        renderer::open_gl_gtao_slice_bent_local(
            0.0f,
            -half_pi,
            half_pi,
            renderer::Vec2::UnitX());
    const renderer::Vec3 asymmetric_bent =
        renderer::open_gl_gtao_slice_bent_local(
            0.0f,
            -half_pi,
            0.25f,
            renderer::Vec2::UnitX());
    RENDER_CHECK(std::abs(symmetric_bent.x()) < 1.0e-5f);
    RENDER_CHECK(symmetric_bent.z() > 0.0f);
    RENDER_CHECK(std::abs(asymmetric_bent.x()) > 1.0e-3f);
    const renderer::Mat3 center_view_rotation =
        renderer::open_gl_ao_rotation_positive_z_to(
            renderer::Vec3::UnitZ());
    RENDER_CHECK(center_view_rotation.isApprox(renderer::Mat3::Identity()));
    for (const renderer::Vec3& view_direction : {
             renderer::Vec3(0.0f, 0.0f, 1.0f),
             renderer::Vec3(-0.2f, 0.0f, 1.0f).normalized(),
             renderer::Vec3(0.3f, -0.25f, 1.0f).normalized()}) {
        const renderer::Mat3 rotation =
            renderer::open_gl_ao_rotation_positive_z_to(view_direction);
        RENDER_CHECK((rotation * renderer::Vec3::UnitZ()).isApprox(
            view_direction,
            1.0e-5f));
        RENDER_CHECK(nearly_equal(rotation.determinant(), 1.0f, 1.0e-5f));
    }
}

RENDER_TEST(test_open_gl_gtso_visibility_range) {
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_gtso_visibility(0.0f, 0.5f, 1.0f, 0.5f),
        1.0f));
    for (const float alignment : {-1.0f, 0.0f, 1.0f}) {
        const float visibility = renderer::open_gl_gtso_visibility(
            alignment, 0.25f, 0.5f, 0.75f);
        RENDER_CHECK(visibility >= 0.0f && visibility <= 1.0f);
    }
}

RENDER_TEST(test_open_gl_ssr_resolution_hiz_and_sampling_math) {
    RENDER_CHECK(renderer::open_gl_ssr_hiz_level_count(1, 1) == 1);
    RENDER_CHECK(renderer::open_gl_ssr_hiz_level_count(8, 3) == 4);
    RENDER_CHECK(renderer::open_gl_ssr_hiz_level_count(513, 257) == 10);
    const auto first_odd_coverage =
        renderer::open_gl_ssr_reduction_coverage(0, 513, 256);
    const auto last_odd_coverage =
        renderer::open_gl_ssr_reduction_coverage(255, 513, 256);
    RENDER_CHECK(first_odd_coverage.first == 0);
    RENDER_CHECK(first_odd_coverage.last == 2);
    RENDER_CHECK(last_odd_coverage.first == 510);
    RENDER_CHECK(last_odd_coverage.last == 512);

    const std::array<renderer::OpenGlSsrDepthRange, 4> ranges{{
        {3.0f, 4.0f},
        {},
        {1.5f, 2.0f},
        {7.0f, 8.0f},
    }};
    const renderer::OpenGlSsrDepthRange reduced =
        renderer::open_gl_ssr_reduce_depth_ranges(ranges);
    RENDER_CHECK(nearly_equal(reduced.minimum, 1.5f));
    RENDER_CHECK(nearly_equal(reduced.maximum, 8.0f));
    RENDER_CHECK(renderer::open_gl_ssr_hiz_intersects(
        1.0f, 1.5f, reduced, 0.05f));
    RENDER_CHECK(!renderer::open_gl_ssr_hiz_intersects(
        0.5f, 1.0f, reduced, 0.05f));
    RENDER_CHECK(renderer::open_gl_ssr_hiz_intersects(
        8.1f, 7.9f, reduced, 0.1f));
    RENDER_CHECK(
        renderer::open_gl_ssr_depth_interval_relation(
            1.0f, 2.0f, {}, 0.1f) ==
        renderer::OpenGlSsrDepthIntervalRelation::Empty);
    RENDER_CHECK(
        renderer::open_gl_ssr_depth_interval_relation(
            0.5f, 1.0f, reduced, 0.05f) ==
        renderer::OpenGlSsrDepthIntervalRelation::InFront);
    RENDER_CHECK(
        renderer::open_gl_ssr_depth_interval_relation(
            2.0f, 1.0f, reduced, 0.05f) ==
        renderer::OpenGlSsrDepthIntervalRelation::Overlap);
    RENDER_CHECK(
        renderer::open_gl_ssr_depth_interval_relation(
            9.0f, 10.0f, reduced, 0.05f) ==
        renderer::OpenGlSsrDepthIntervalRelation::Behind);

    const renderer::Vec3 center =
        renderer::open_gl_ssr_cosine_hemisphere_sample(
            renderer::Vec2(0.0f, 0.25f));
    const renderer::Vec3 rim =
        renderer::open_gl_ssr_cosine_hemisphere_sample(
            renderer::Vec2(1.0f, 0.0f));
    RENDER_CHECK(center.isApprox(renderer::Vec3::UnitZ(), 1.0e-6f));
    RENDER_CHECK(nearly_equal(rim.norm(), 1.0f, 1.0e-6f));
    RENDER_CHECK(rim.z() >= 0.0f);
}

RENDER_TEST(test_open_gl_ssr_temporal_validation_and_projection_math) {
    RENDER_CHECK(renderer::open_gl_ssr_history_valid(
        10.0f, 10.09f, 0.01f, 0.9f));
    RENDER_CHECK(!renderer::open_gl_ssr_history_valid(
        10.0f, 10.11f, 0.01f, 0.9f));
    RENDER_CHECK(!renderer::open_gl_ssr_history_valid(
        10.0f, 10.0f, 0.01f, 0.84f));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_history_weight(0.0f, 32), 0.0f));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_history_weight(1.0f, 32), 0.5f));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_history_weight(100.0f, 32), 31.0f / 32.0f));

    const float boundary_distance =
        renderer::open_gl_ssr_projected_boundary_distance(
            renderer::Vec3(0.0f, 0.0f, -2.0f),
            renderer::Vec3(1.0f, 0.0f, 0.0f),
            0.75f,
            2.0f,
            true);
    RENDER_CHECK(nearly_equal(boundary_distance, 1.0f));
    RENDER_CHECK(std::isinf(
        renderer::open_gl_ssr_projected_boundary_distance(
            renderer::Vec3(0.0f, 0.0f, -2.0f),
            renderer::Vec3(-1.0f, 0.0f, 0.0f),
            0.75f,
            2.0f,
            true)));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_projected_boundary_distance(
            renderer::Vec3(0.0f, 0.0f, -2.0f),
            renderer::Vec3(-1.0f, 0.0f, 0.0f),
            0.25f,
            2.0f,
            true),
        1.0f));

    renderer::OpenGlRenderSettings settings;
    RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
    settings.ssr.enabled = false;
    RENDER_CHECK(!renderer::open_gl_ssr_requested(settings));
    settings.ssr.debug_view = renderer::OpenGlSsrDebugView::RawIndirect;
    RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
}

RENDER_TEST(test_open_gl_ssr_edge_fade) {
    // Screen-center UVs stay untouched; the fade band hugs the viewport edge.
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_edge_fade(0.5f, 0.5f, 0.15f),
        1.0f));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_edge_fade(1.0f, 0.5f, 0.15f),
        0.0f));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_edge_fade(0.5f, 0.0f, 0.15f),
        0.0f));
    // Fade decreases monotonically toward the edge inside the fade band.
    RENDER_CHECK(renderer::open_gl_ssr_edge_fade(0.94f, 0.5f, 0.15f) >
        renderer::open_gl_ssr_edge_fade(0.97f, 0.5f, 0.15f));
    RENDER_CHECK(renderer::open_gl_ssr_edge_fade(0.97f, 0.5f, 0.15f) >
        renderer::open_gl_ssr_edge_fade(0.99f, 0.5f, 0.15f));
    // A wider fade band attenuates further into the frame.
    RENDER_CHECK(renderer::open_gl_ssr_edge_fade(0.90f, 0.5f, 0.30f) <
        renderer::open_gl_ssr_edge_fade(0.90f, 0.5f, 0.10f));
    // Zero fade width disables the falloff for in-screen UVs.
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_edge_fade(0.0f, 1.0f, 0.0f),
        1.0f));
    for (const float uv_x : {0.0f, 0.3f, 0.6f, 1.0f}) {
        for (const float uv_y : {0.0f, 0.25f, 0.75f, 1.0f}) {
            const float fade = renderer::open_gl_ssr_edge_fade(
                uv_x, uv_y, 0.15f);
            RENDER_CHECK(fade >= 0.0f && fade <= 1.0f);
        }
    }
}

RENDER_TEST(test_open_gl_ssr_depth_bounds_and_activation) {
    // A hit is a front-to-back crossing of the sampled screen-space surface.
    // The rule is independent of whether the reflected ray itself moves
    // toward or away from the camera.
    RENDER_CHECK(renderer::open_gl_ssr_depth_crossing(-0.1f, 0.0f));
    RENDER_CHECK(renderer::open_gl_ssr_depth_crossing(-0.1f, 0.1f));
    RENDER_CHECK(!renderer::open_gl_ssr_depth_crossing(0.1f, -0.1f));
    RENDER_CHECK(!renderer::open_gl_ssr_depth_crossing(0.0f, 0.1f));

    RENDER_CHECK(renderer::open_gl_ssr_depth_within_thickness(
        5.0f, 5.0f, 0.1f));
    RENDER_CHECK(renderer::open_gl_ssr_depth_within_thickness(
        5.1f, 5.0f, 0.1f));
    RENDER_CHECK(!renderer::open_gl_ssr_depth_within_thickness(
        5.1001f, 5.0f, 0.1f));
    RENDER_CHECK(!renderer::open_gl_ssr_depth_within_thickness(
        4.99f, 5.0f, 0.1f));

    renderer::OpenGlRenderSettings settings;
    RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
    settings.ssr.enabled = false;
    RENDER_CHECK(!renderer::open_gl_ssr_requested(settings));
    settings.ssr.debug_view = renderer::OpenGlSsrDebugView::HitConfidence;
    RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
    settings.shadow_map.debug_view =
        renderer::OpenGlShadowDebugView::Visibility;
    RENDER_CHECK(!renderer::open_gl_ssr_requested(settings));
}
