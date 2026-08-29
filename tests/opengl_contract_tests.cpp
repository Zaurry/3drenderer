#include "test_framework.h"

#include "render/opengl/opengl_raster_renderer.h"
#include "render/opengl/opengl_ao_math.h"
#include "render/opengl/opengl_shadow_math.h"
#include "render/opengl/opengl_ssr_math.h"

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

RENDER_TEST(test_open_gl_ssr_depth_hit_and_ggx_cone_lod) {
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

    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_ggx_cone_tangent(0.0f),
        0.0f));
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_ssr_ggx_cone_tangent(0.5f),
        0.3303486f,
        1.0e-5f));
    RENDER_CHECK(std::isfinite(
        renderer::open_gl_ssr_ggx_cone_tangent(1.0f)));

    const float mirror_lod = renderer::open_gl_ssr_reflection_lod(
        0.0f, 2.0f, 4.0f, 2.0f, 1.0f, 800, 400, 10);
    const float glossy_lod = renderer::open_gl_ssr_reflection_lod(
        0.5f, 2.0f, 4.0f, 2.0f, 1.0f, 800, 400, 10);
    const float close_hit_lod = renderer::open_gl_ssr_reflection_lod(
        0.5f, 0.25f, 4.0f, 2.0f, 1.0f, 800, 400, 10);
    const float deep_hit_lod = renderer::open_gl_ssr_reflection_lod(
        0.5f, 2.0f, 8.0f, 2.0f, 1.0f, 800, 400, 10);
    RENDER_CHECK(nearly_equal(mirror_lod, 0.0f));
    RENDER_CHECK(glossy_lod > 5.5f && glossy_lod < 6.5f);
    RENDER_CHECK(close_hit_lod < glossy_lod);
    RENDER_CHECK(deep_hit_lod < glossy_lod);
    const float fully_rough_lod = renderer::open_gl_ssr_reflection_lod(
        1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 800, 400, 10);
    RENDER_CHECK(fully_rough_lod > glossy_lod);
    RENDER_CHECK(fully_rough_lod > 8.5f && fully_rough_lod <= 9.0f);

    RENDER_CHECK(nearly_equal(renderer::open_gl_ssr_cone_edge_fade(
        0.5f, 0.5f, 0.1f, 0.1f), 1.0f));
    RENDER_CHECK(nearly_equal(renderer::open_gl_ssr_cone_edge_fade(
        0.0f, 0.5f, 0.1f, 0.1f), 0.0f));
    RENDER_CHECK(renderer::open_gl_ssr_cone_edge_fade(
        0.25f, 0.5f, 0.5f, 0.5f) <
        renderer::open_gl_ssr_cone_edge_fade(
            0.25f, 0.5f, 0.1f, 0.1f));

    renderer::OpenGlRenderSettings settings;
    RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
    settings.ssr.enabled = false;
    RENDER_CHECK(!renderer::open_gl_ssr_requested(settings));
    settings.ssr.debug_view = renderer::OpenGlSsrDebugView::Confidence;
    RENDER_CHECK(renderer::open_gl_ssr_requested(settings));
    settings.shadow_map.debug_view =
        renderer::OpenGlShadowDebugView::Visibility;
    RENDER_CHECK(!renderer::open_gl_ssr_requested(settings));
}
