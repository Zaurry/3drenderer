#include "test_framework.h"

#include "render/opengl/opengl_shader_contract.h"
#include "render/opengl/opengl_ssr_math.h"
#include "render/shading_constants.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

// Source-text lint for the C++/GLSL contract, NOT a runtime verification:
// it greps shader sources and the OpenGL renderer source for the binding
// declarations and formats declared by opengl_shader_contract.h. It cannot
// prove shaders compile, link, or bind at runtime (that needs a live GL
// context); it exists to catch accidental renames/format drift early.
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    RENDER_CHECK(input.good());
    std::string text{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
}

}  // namespace

RENDER_TEST(test_opengl_shader_source_contract_lint) {
    const std::filesystem::path shader_root =
        std::filesystem::path(RENDERER_SOURCE_DIR) / "shaders" / "opengl";
    const std::string vertex = read_text(shader_root / "raster.vert");
    const std::string fragment = read_text(shader_root / "raster.frag");
    const std::string screen_space_gbuffer = read_text(
        shader_root / "screen_space_gbuffer.frag");
    const std::string ao_fragment = read_text(
        shader_root / "ambient_occlusion.frag");
    const std::string ao_denoise = read_text(shader_root / "ao_denoise.frag");
    const std::string ssr_fragment = read_text(shader_root / "ssr.frag");
    const std::string ssgi_hiz = read_text(shader_root / "ssgi_hiz.frag");
    const std::string ssgi_trace = read_text(shader_root / "ssgi_trace.frag");
    const std::string ssgi_temporal = read_text(
        shader_root / "ssgi_temporal.frag");
    const std::string ssgi_denoise = read_text(
        shader_root / "ssgi_denoise.frag");
    const std::string ssgi_composite = read_text(
        shader_root / "ssgi_composite.frag");
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
    RENDER_CHECK(fragment.find("out_diffuse_ibl") != std::string::npos);
    RENDER_CHECK(fragment.find(
        "applied_diffuse_ibl = diffuse_ibl * (occlusion * screen_ao)") !=
        std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("u_alpha_cutoff") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("u_has_normal_texture") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("u_has_bump_texture") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("out_ssr_pbr") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("u_metallic") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find(
        "u_has_metallic_roughness_texture") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("evaluate_ssr_material") !=
        std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("out_ssgi_material") !=
        std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("diffuse_response") !=
        std::string::npos);
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
    RENDER_CHECK(ssr_fragment.find("u_max_steps") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("u_refinement_steps") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("reconstruct_view_position") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("project_view_position") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("gtso_visibility") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("screen_edge_fade") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("ggx_reflection_cone_tangent") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("reflection_lobe_basis") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("project_view_offset") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("reflection_footprint") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("reflection_filter_tap_count") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("sample_elliptical_reflection") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("minor_scale = max(n_dot_v") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("footprint.minor_radius_pixels") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("specular_response") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find(
        "previous_delta < 0.0 && depth_delta >= 0.0") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("refined_delta <= thickness") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("reflection_view.z < 0.0") ==
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("textureQueryLevels") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("textureLod(\n            u_opaque") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("u_ssr_pbr") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find(
        "float roughness = clamp(pbr.a, 0.02, 1.0)") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("u_debug_view") != std::string::npos);
    const std::string ssr_filter_tap_cap =
        "const int SSR_MAX_FILTER_TAPS = " +
        std::to_string(renderer::kOpenGlSsrMaxFilterTaps) + ";";
    RENDER_CHECK(ssr_fragment.find(ssr_filter_tap_cap) != std::string::npos);
    const std::string ssr_anisotropy_cap =
        "const float SSR_MAX_ANISOTROPY = " +
        std::to_string(static_cast<int>(renderer::kOpenGlSsrMaxAnisotropy)) +
        ".0;";
    RENDER_CHECK(ssr_fragment.find(ssr_anisotropy_cap) != std::string::npos);
    const std::string ssr_step_cap =
        "index < " + std::to_string(renderer::kOpenGlMaxSsrSteps);
    RENDER_CHECK(ssr_fragment.find(ssr_step_cap) != std::string::npos);
    const std::string ssr_refinement_cap =
        "refine < " + std::to_string(renderer::kOpenGlMaxSsrRefinementSteps);
    RENDER_CHECK(ssr_fragment.find(ssr_refinement_cap) != std::string::npos);
    RENDER_CHECK(ssgi_hiz.find("out_depth_range") != std::string::npos);
    RENDER_CHECK(ssgi_hiz.find("3.402823466e+38, 0.0") !=
        std::string::npos);
    RENDER_CHECK(ssgi_hiz.find("y < 3") != std::string::npos);
    RENDER_CHECK(ssgi_hiz.find("x < 3") != std::string::npos);
    RENDER_CHECK(ssgi_trace.find("march_hiz") != std::string::npos);
    RENDER_CHECK(ssgi_trace.find("cell_exit_distance") != std::string::npos);
    RENDER_CHECK(ssgi_trace.find("textureLod(u_opaque") !=
        std::string::npos);
    RENDER_CHECK(ssgi_composite.find("u_original_diffuse_ibl") !=
        std::string::npos);
    RENDER_CHECK(ssgi_trace.find(
        "radiance_residual = confidence *") !=
        std::string::npos);
    RENDER_CHECK(ssgi_trace.find("environment_radiance(direction)") !=
        std::string::npos);
    RENDER_CHECK(ssgi_trace.find("R2_SEQUENCE") != std::string::npos);
    RENDER_CHECK(ssgi_trace.find("stable_hit_radiance") !=
        std::string::npos);
    RENDER_CHECK(ssgi_trace.find("dot(normalize(hit_normal), -direction)") !=
        std::string::npos);
    RENDER_CHECK(ssgi_trace.find(
        "far_distance <= 2.0 * u_thickness") != std::string::npos);
    const std::string ssgi_ray_cap =
        "ray_index < " + std::to_string(renderer::kOpenGlMaxSsgiRays);
    const std::string ssgi_step_cap =
        "visit < " + std::to_string(renderer::kOpenGlMaxSsgiSteps);
    const std::string ssgi_refinement_cap =
        "refine < " +
        std::to_string(renderer::kOpenGlMaxSsgiRefinementSteps);
    RENDER_CHECK(ssgi_trace.find(ssgi_ray_cap) != std::string::npos);
    RENDER_CHECK(ssgi_trace.find(ssgi_step_cap) != std::string::npos);
    RENDER_CHECK(ssgi_trace.find(ssgi_refinement_cap) != std::string::npos);
    RENDER_CHECK(ssgi_temporal.find("valid_history_sample") !=
        std::string::npos);
    RENDER_CHECK(ssgi_temporal.find("0.01 * predicted_depth") !=
        std::string::npos);
    RENDER_CHECK(ssgi_temporal.find(">= 0.85") != std::string::npos);
    RENDER_CHECK(ssgi_temporal.find("1.5 * sigma") != std::string::npos);
    RENDER_CHECK(ssgi_temporal.find("rgb_to_ycocg") != std::string::npos);
    RENDER_CHECK(ssgi_temporal.find("current_neighborhood_statistics") !=
        std::string::npos);
    RENDER_CHECK(ssgi_temporal.find(
        "history = ycocg_to_rgb(clipped_history_ycocg)") !=
        std::string::npos);
    RENDER_CHECK(ssgi_denoise.find("offset = -2; offset <= 2") !=
        std::string::npos);
    RENDER_CHECK(ssgi_denoise.find("u_stride") != std::string::npos);
    RENDER_CHECK(ssgi_composite.find("bilateral_upsample") !=
        std::string::npos);
    RENDER_CHECK(ssgi_composite.find("filtered_residual") !=
        std::string::npos);
    RENDER_CHECK(ssgi_composite.find(
        "filtered_residual = vec3(0.0)") !=
        std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_R32F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_RG32F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_DEPTH_COMPONENT32F") !=
        std::string::npos);
    RENDER_CHECK(renderer_source.find("open_gl_ssr_requested(settings.opengl)") !=
        std::string::npos);
    RENDER_CHECK(renderer_source.find(
        "glGenerateMipmap(GL_TEXTURE_2D);\n            "
        "render_screen_space_reflections") !=
        std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_RGB16F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_RGBA16F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_RG16F") != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_COLOR_ATTACHMENT4") !=
        std::string::npos);
    RENDER_CHECK(renderer_source.find("glTextureBarrier();") !=
        std::string::npos);
    const std::size_t opaque_pass = renderer_source.find("draw_batches(false);");
    const std::size_t ssgi_pass = renderer_source.find(
        "render_ssgi(scene, camera, settings, ao_active);");
    const std::size_t ssr_pass = renderer_source.find(
        "render_screen_space_reflections(\n                scene");
    const std::size_t oit_composite = renderer_source.find(
        "glBindFramebuffer(GL_FRAMEBUFFER, composite_framebuffer_);",
        ssr_pass);
    RENDER_CHECK(opaque_pass < ssgi_pass);
    RENDER_CHECK(ssgi_pass < ssr_pass);
    RENDER_CHECK(ssr_pass < oit_composite);
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
             "sky.vert",
             "sky.frag",
             "shadow.vert",
             "shadow.frag",
             "composite.frag",
             "screen_space_gbuffer.frag",
             "ambient_occlusion.frag",
             "ao_denoise.frag",
             "ssr.frag",
             "ssgi_hiz.frag",
             "ssgi_trace.frag",
             "ssgi_temporal.frag",
             "ssgi_denoise.frag",
             "ssgi_composite.frag"}) {
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
    // Shared C++/GLSL constants must stay in sync.
    const std::string shadow_slot_declaration =
        "u_shadow_matrices[" +
        std::to_string(renderer::kOpenGlMaxShadowLightSlots) + "]";
    RENDER_CHECK(
        fragment.find(shadow_slot_declaration) != std::string::npos);
    const std::string pcss_sample_cap =
        "sample_index < " +
        std::to_string(renderer::kOpenGlMaxPcssSamples);
    RENDER_CHECK(fragment.find(pcss_sample_cap) != std::string::npos);
    const std::string luminance_literal = "0.2126, 0.7152, 0.0722";
    RENDER_CHECK(fragment.find(luminance_literal) != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find(luminance_literal) !=
        std::string::npos);
    const std::string shadow_fragment_source =
        read_text(shader_root / "shadow.frag");
    RENDER_CHECK(
        shadow_fragment_source.find(luminance_literal) !=
        std::string::npos);
    // Punctual falloff (raster.frag) matches shading_constants.h.
    RENDER_CHECK(
        fragment.find("ratio * ratio * ratio * ratio") !=
        std::string::npos);
    RENDER_CHECK(renderer::kOpenGlShadowResolutionMin > 0);
    RENDER_CHECK(
        renderer::kOpenGlShadowResolutionMax >=
        renderer::kOpenGlShadowResolutionMin);
}
