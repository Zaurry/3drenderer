#include "test_framework.h"

#include "render/opengl/opengl_raster_renderer.h"
#include "render/opengl/opengl_ao_math.h"
#include "render/opengl/opengl_shader_contract.h"
#include "render/opengl/opengl_shadow_math.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    RENDER_CHECK(input.good());
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
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
    RENDER_CHECK(nearly_equal(
        renderer::open_gl_gtso_visibility(0.0f, 0.5f, 1.0f, 0.5f),
        1.0f));
    for (const float alignment : {-1.0f, 0.0f, 1.0f}) {
        const float visibility = renderer::open_gl_gtso_visibility(
            alignment, 0.25f, 0.5f, 0.75f);
        RENDER_CHECK(visibility >= 0.0f && visibility <= 1.0f);
    }

    const std::filesystem::path shader_root =
        std::filesystem::path(RENDERER_SOURCE_DIR) / "shaders" / "opengl";
    const std::string vertex = read_text(shader_root / "raster.vert");
    const std::string fragment = read_text(shader_root / "raster.frag");
    const std::string ao_gbuffer = read_text(shader_root / "ao_gbuffer.frag");
    const std::string ao_fragment = read_text(
        shader_root / "ambient_occlusion.frag");
    const std::string ao_denoise = read_text(shader_root / "ao_denoise.frag");
    const std::string renderer_source = read_text(
        std::filesystem::path(RENDERER_SOURCE_DIR) /
        "src" / "render" / "opengl" / "opengl_raster_renderer.cpp");
    for (const renderer::OpenGlShaderBinding& binding :
         renderer::kOpenGlVertexAttributeContract) {
        const std::string declaration =
            "layout(location = " + std::to_string(binding.location) + ") in";
        RENDER_CHECK(vertex.find(declaration) != std::string::npos);
        RENDER_CHECK(vertex.find(binding.name) != std::string::npos);
    }
    for (const renderer::OpenGlShaderBinding& binding :
         renderer::kOpenGlTextureBindingContract) {
        const std::string declaration =
            "layout(binding = " + std::to_string(binding.location) + ") uniform";
        RENDER_CHECK(fragment.find(declaration) != std::string::npos);
        RENDER_CHECK(fragment.find(binding.name) != std::string::npos);
    }
    RENDER_CHECK(fragment.find("vogel_disk") != std::string::npos);
    RENDER_CHECK(fragment.find("ltc_evaluate") != std::string::npos);
    RENDER_CHECK(fragment.find(
        "dot(position - points[0], light_normal) < 0.0") !=
        std::string::npos);
    RENDER_CHECK(fragment.find(
        "vec3 bitangent = -cross(normal, tangent)") != std::string::npos);
    RENDER_CHECK(fragment.find(
        "if (!crosses_receiver_horizon)") != std::string::npos);
    RENDER_CHECK(fragment.find(
        "rect_area_light_emits_toward_receiver") != std::string::npos);
    RENDER_CHECK(fragment.find(
        "bool casts_shadows = u_rect_area_lights[index].shadow.w > 0.5") !=
        std::string::npos);
    RENDER_CHECK(fragment.find("sampler2DArray") != std::string::npos);
    RENDER_CHECK(fragment.find("samplerCubeArray") != std::string::npos);
    RENDER_CHECK(fragment.find("gtso_visibility") != std::string::npos);
    RENDER_CHECK(fragment.find(
        "diffuse_ibl * (occlusion * screen_ao)") != std::string::npos);
    RENDER_CHECK(ao_gbuffer.find("u_alpha_cutoff") != std::string::npos);
    RENDER_CHECK(ao_gbuffer.find("u_has_normal_texture") != std::string::npos);
    RENDER_CHECK(ao_gbuffer.find("u_has_bump_texture") != std::string::npos);
    RENDER_CHECK(ao_fragment.find("evaluate_ssao") != std::string::npos);
    RENDER_CHECK(ao_fragment.find("evaluate_gtao") != std::string::npos);
    RENDER_CHECK(ao_fragment.find("bent_normal_sum") != std::string::npos);
    RENDER_CHECK(ao_fragment.find("horizon_cosine") != std::string::npos);
    RENDER_CHECK(ao_fragment.find("rotate_positive_z_to") !=
        std::string::npos);
    RENDER_CHECK(ao_fragment.find("rotate_minus_z_to") ==
        std::string::npos);
    RENDER_CHECK(ao_denoise.find("u_depth_sigma_fraction") !=
        std::string::npos);
    RENDER_CHECK(ao_denoise.find("u_normal_power") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_R32F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_DEPTH_COMPONENT32F") !=
        std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_RGB16F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_RGBA16F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("u_alpha_cutoff") != std::string::npos);
    RENDER_CHECK(renderer_source.find("AlphaMode::Blend") != std::string::npos);
    RENDER_CHECK(renderer_source.find(
        "light.casts_shadows ? 1.0f : 0.0f") != std::string::npos);
    for (const char* filename : {"ltc_1.dds", "ltc_2.dds", "LTC_LICENSE.txt"}) {
        const std::filesystem::path asset = shader_root / filename;
        RENDER_CHECK(std::filesystem::exists(asset));
        RENDER_CHECK(std::filesystem::file_size(asset) > 0);
    }
    for (const char* filename : {
             "fullscreen.vert",
             "ao_gbuffer.frag",
             "ambient_occlusion.frag",
             "ao_denoise.frag"}) {
        const std::filesystem::path shader = shader_root / filename;
        RENDER_CHECK(std::filesystem::exists(shader));
        RENDER_CHECK(std::filesystem::file_size(shader) > 0);
    }
    for (std::string_view uniform :
         renderer::kOpenGlMaterialUniformContract) {
        RENDER_CHECK(fragment.find(uniform) != std::string::npos);
    }
    for (const renderer::OpenGlShaderBinding& binding :
         renderer::kOpenGlLightBufferContract) {
        const std::string declaration =
            "binding = " + std::to_string(binding.location);
        RENDER_CHECK(fragment.find(declaration) != std::string::npos);
        RENDER_CHECK(fragment.find(binding.name) != std::string::npos);
    }
    std::cout << "opengl_contract_tests: all tests passed\n";
}
