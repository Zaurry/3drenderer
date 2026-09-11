#include "render/opengl/opengl_raster_renderer.h"
#include "scene/scene.h"

#include <SDL3/SDL.h>
#include <glad/gl.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void run() {
    using namespace renderer;
    const auto shaders = std::filesystem::path(RENDERER_SOURCE_DIR) / "shaders/opengl";
    OpenGlRasterRenderer renderer(shaders / "raster.vert", shaders / "raster.frag");
    Scene scene = make_gradient_sphere_scene();
    scene.environment = Color(0.08f, 0.10f, 0.12f);
    scene.directional_lights.push_back({Vec3(-1, -1, -1).normalized(), Color(2, 2, 2)});
    tessellate_spheres(scene);
    renderer.reset(scene);
    RenderSettings settings;
    settings.width = 256;
    settings.height = 256;
    // NPR must render its own G-buffer even when all optional effects are off.
    settings.opengl.ambient_occlusion.mode = OpenGlAmbientOcclusionMode::Off;
    settings.opengl.ssr.enabled = false;
    const Camera camera(Vec3(0, 0, 1), Vec3(0, 0, -1), Vec3(0, 1, 0), 45, 1);
    const auto render = [&]() {
        renderer.render(scene, camera, settings, {});
        check(renderer.has_valid_shader() && renderer.shader_error().empty(), "Shader compilation failed");
        std::vector<float> pixels(static_cast<std::size_t>(settings.width) * settings.height * 4);
        glBindTexture(GL_TEXTURE_2D, renderer.output_texture());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels.data());
        check(glGetError() == GL_NO_ERROR, "OpenGL error while switching NPR styles");
        for (float value : pixels) check(std::isfinite(value), "Non-finite output");
        return pixels;
    };
    const auto physical = render();
    settings.opengl.npr.style = OpenGlRenderStyle::Toon;
    const auto toon = render();
    check(toon != physical, "Toon did not change shading");
    settings.opengl.npr.style = OpenGlRenderStyle::Sketch;
    const auto sketch = render();
    check(sketch != toon, "Sketch did not change shading");
    check(std::abs(sketch[0] - 0.92f) < 0.005f, "Sketch paper background missing with AO off");
    settings.opengl.npr.outline_strength = 0;
    const auto no_outline = render();
    check(no_outline != sketch, "Outline control did not change the image");
    settings.opengl.npr.sketch_use_uv = true;
    check(render() != no_outline, "UV hatching did not change the image");
    settings.opengl.npr.sketch_use_uv = false;
    settings.opengl.npr.outline_strength = 0.85f;
    settings.width = 320;
    settings.height = 192;
    render();
    check(renderer.output_width() == 320 && renderer.output_height() == 192, "NPR resize failed");
    settings.width = settings.height = 256;
    check(render() == sketch, "NPR resize did not restore deterministic output");
    settings.opengl.npr.style = OpenGlRenderStyle::Realistic;
    check(render() == physical, "Switching back changed the original realistic image");
    settings.opengl.ssr.enabled = true;
    render();
    settings.opengl.npr.style = OpenGlRenderStyle::Sketch;
    check(render() == sketch, "SSR contaminated sketch shading");
    settings.opengl.ambient_occlusion.mode = OpenGlAmbientOcclusionMode::Gtao;
    settings.opengl.ambient_occlusion.debug_view = OpenGlAmbientOcclusionDebugView::ViewNormal;
    check(render() != sketch, "NPR hid the diagnostic view");
    settings.opengl.ambient_occlusion.debug_view = OpenGlAmbientOcclusionDebugView::Final;
    settings.opengl.npr.style = OpenGlRenderStyle::Toon;
    settings.opengl.ibl_enabled = false;
    settings.opengl.shadow_map.enabled = false;
    scene.directional_lights.clear();
    RectAreaLight light;
    light.position = Vec3(0, 0, 1);
    light.axis_u = Vec3(0.05f, 0, 0);
    light.axis_v = Vec3(0, 0.05f, 0);
    light.radiance = Color(10, 10, 10);
    light.casts_shadows = false;
    scene.rect_area_lights.push_back(light);
    renderer.reset(scene);
    const auto area_lit = render();
    check(area_lit[(128 * 256 + 128) * 4] > 0.0001f,
        "Toon quantization extinguished a small area light");
    std::cout << "NPR GPU smoke passed: style switching, outline, UV, resize, diagnostics, SSR/SSGI isolation\n";
}
}  // namespace

int main() {
    SDL_Window* window = nullptr;
    SDL_GLContext context = nullptr;
    int result = 0;
    try {
        check(SDL_Init(SDL_INIT_VIDEO), "SDL initialization failed");
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        window = SDL_CreateWindow("NPR smoke", 256, 256, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        check(window != nullptr, "Hidden test window failed");
        context = SDL_GL_CreateContext(window);
        check(context != nullptr, "OpenGL context failed");
        check(gladLoadGL(reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress)) != 0,
            "OpenGL loader failed");
        run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    if (context) SDL_GL_DestroyContext(context);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
    return result;
}
