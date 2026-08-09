#include "test_framework.h"

#include "render/opengl/opengl_shader_contract.h"

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
    const std::filesystem::path shader_root =
        std::filesystem::path(RENDERER_SOURCE_DIR) / "shaders" / "opengl";
    const std::string vertex = read_text(shader_root / "raster.vert");
    const std::string fragment = read_text(shader_root / "raster.frag");
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
