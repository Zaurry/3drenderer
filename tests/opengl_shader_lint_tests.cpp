#include "test_framework.h"

#include "render/opengl/opengl_shader_contract.h"
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
    RENDER_CHECK(screen_space_gbuffer.find("u_alpha_cutoff") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("u_has_normal_texture") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("u_has_bump_texture") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("out_ssr_pbr") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("u_metallic") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find(
        "u_has_metallic_roughness_texture") != std::string::npos);
    RENDER_CHECK(screen_space_gbuffer.find("evaluate_ssr_material") !=
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
    RENDER_CHECK(ssr_fragment.find("fresnel_schlick") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("gtso_visibility") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("screen_edge_fade") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find(
        "previous_delta < 0.0 && depth_delta >= 0.0") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("refined_delta <= thickness") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("reflection_view.z < 0.0") ==
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("textureQueryLevels") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("textureLod(\n                u_opaque") !=
        std::string::npos);
    RENDER_CHECK(ssr_fragment.find("u_ssr_pbr") != std::string::npos);
    RENDER_CHECK(ssr_fragment.find("u_debug_view") != std::string::npos);
    const std::string ssr_step_cap =
        "index < " + std::to_string(renderer::kOpenGlMaxSsrSteps);
    RENDER_CHECK(ssr_fragment.find(ssr_step_cap) != std::string::npos);
    const std::string ssr_refinement_cap =
        "refine < " + std::to_string(renderer::kOpenGlMaxSsrRefinementSteps);
    RENDER_CHECK(ssr_fragment.find(ssr_refinement_cap) != std::string::npos);
    RENDER_CHECK(renderer_source.find("GL_R32F") != std::string::npos);
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
             "ssr.frag"}) {
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
