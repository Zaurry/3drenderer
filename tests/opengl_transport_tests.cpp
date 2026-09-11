#include "test_framework.h"
#include "core/image.h"
#include "platform/opengl/gl_shader_program.h"
#include "render/display_settings.h"
#include "render/ggx_energy_compensation.h"
#include "render/opengl/opengl_raster_renderer.h"
#include "scene/scene.h"

#include <SDL3/SDL.h>
#include <glad/gl.h>
#include <array>
#include <filesystem>
#include <limits>
#include <vector>

namespace {
const auto shader_root = std::filesystem::path(RENDERER_SOURCE_DIR) / "shaders/opengl";

struct GlContext {
    SDL_Window* window = nullptr;
    SDL_GLContext context = nullptr;
    GlContext() {
        if (!SDL_Init(SDL_INIT_VIDEO)) RENDER_SKIP(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        window = SDL_CreateWindow("OpenGL transport tests", 64, 64,
            SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        if (window) context = SDL_GL_CreateContext(window);
        if (!context) {
            const std::string reason = SDL_GetError();
            if (window) SDL_DestroyWindow(window);
            SDL_Quit();
            RENDER_SKIP(reason);
        }
        RENDER_CHECK(SDL_GL_MakeCurrent(window, context));
        RENDER_CHECK(gladLoadGL(reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress)) != 0);
        RENDER_CHECK(GLAD_GL_VERSION_4_5);
    }
    ~GlContext() {
        SDL_GL_DestroyContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

using Pixel = std::array<float, 4>;
constexpr int extent = 65; // Exercise odd/NPOT hierarchy coverage.
constexpr int center = extent / 2;
constexpr int center_index = center * extent + center;

struct TraceFixture {
    renderer::GlShaderProgram trace;
    renderer::GlShaderProgram hiz;
    renderer::GlShaderProgram composite;
    std::array<GLuint, 11> textures{};
    GLuint framebuffer = 0;
    GLuint output = 0;
    GLuint vao = 0;
    int levels = 7;

    TraceFixture() {
        std::string error;
        for (auto [program, filename] : {
                 std::pair{&trace, "ssr_trace.frag"},
                 std::pair{&hiz, "ssr_hiz.frag"},
                 std::pair{&composite, "ssr_composite.frag"}}) {
            if (!program->load(shader_root / "fullscreen.vert", shader_root / filename, error)) {
                std::cerr << error << '\n';
                RENDER_CHECK(false);
            }
        }
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenFramebuffers(1, &framebuffer);
        glGenTextures(static_cast<GLsizei>(textures.size()), textures.data());
        for (int unit = 0; unit < 11; ++unit) {
            glActiveTexture(GL_TEXTURE0 + unit);
            if (unit == 6) {
                glBindTexture(GL_TEXTURE_CUBE_MAP, textures[unit]);
                const Pixel white{1, 1, 1, 1};
                for (int face = 0; face < 6; ++face) {
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA32F,
                        1, 1, 0, GL_RGBA, GL_FLOAT, white.data());
                }
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                continue;
            }
            glBindTexture(GL_TEXTURE_2D, textures[unit]);
            glTexStorage2D(GL_TEXTURE_2D, unit == 4 ? levels : 1,
                GL_RGBA32F, extent, extent);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                unit == 4 ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glGenTextures(1, &output);
        glBindTexture(GL_TEXTURE_2D, output);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, extent, extent);
        std::vector<Pixel> lut(extent * extent);
        for (int y = 0; y < extent; ++y) {
            for (int x = 0; x < extent; ++x) {
                const float r = (y + 0.5f) / extent;
                const float e = renderer::ggx_directional_albedo((x + 0.5f) / extent, r);
                lut[y * extent + x] = {e, 0, e, renderer::ggx_average_albedo(r)};
            }
        }
        upload(10, lut);
    }
    ~TraceFixture() {
        glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
        glDeleteTextures(1, &output);
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteVertexArrays(1, &vao);
    }
    void upload(int unit, const std::vector<Pixel>& pixels) {
        glTextureSubImage2D(textures[unit], 0, 0, 0, extent, extent,
            GL_RGBA, GL_FLOAT, pixels.data());
    }
    void fill(int unit, Pixel value) { upload(unit, std::vector<Pixel>(extent * extent, value)); }
    void integer(GLuint program, const char* name, int value) {
        glProgramUniform1i(program, glGetUniformLocation(program, name), value);
    }
    void scalar(const char* name, float value) {
        glProgramUniform1f(trace.id(), glGetUniformLocation(trace.id(), name), value);
    }
    void vector(const char* name, float x, float y, float z) {
        glProgramUniform3f(trace.id(), glGetUniformLocation(trace.id(), name), x, y, z);
    }
    void setup(float blocker_depth, Pixel normal, Pixel hit_normal = {0, 0, 1, 0}) {
        fill(0, {0, 0, 0, 0});
        std::vector<Pixel> depths(extent * extent, Pixel{blocker_depth, 0, 0, 0});
        depths[center_index] = {4, 0, 0, 0};
        upload(1, depths);
        std::vector<Pixel> normals(extent * extent, hit_normal);
        normals[center_index] = normal;
        upload(2, normals);
        fill(3, {0, 0, 0, 0});
        fill(5, {0, 0, 1, 1});
        fill(7, {1, 1, 1, 0.02f});
        fill(8, {1, 1, 1, 1});
        fill(9, {0, 0, 0, 0});
        integer(trace.id(), "u_rays_per_pixel", 8);
        integer(trace.id(), "u_max_steps", 256);
        integer(trace.id(), "u_hiz_levels", levels);
        integer(trace.id(), "u_ao_enabled", 1);
        integer(trace.id(), "u_ibl_enabled", 1);
        integer(trace.id(), "u_has_environment_map", 0);
        scalar("u_max_distance", 20);
        scalar("u_thickness", 0.02f);
        scalar("u_edge_fade", 0);
        scalar("u_environment_intensity", 1);
        vector("u_environment_color", 1, 1, 1);
        vector("u_camera_forward", 0, 0, -1);
        vector("u_camera_right", 1, 0, 0);
        vector("u_camera_up", 0, 1, 0);
        glProgramUniform2f(trace.id(), glGetUniformLocation(trace.id(), "u_camera_viewport"), 2, 2);
        for (const char* name : {"u_full_resolution", "u_trace_resolution"}) {
            glProgramUniform2i(trace.id(), glGetUniformLocation(trace.id(), name), extent, extent);
        }
        build_hierarchy();
    }
    void build_hierarchy() {
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glUseProgram(hiz.id());
        glBindTextureUnit(0, textures[1]);
        int source_size = extent;
        for (int level = 0; level < levels; ++level) {
            int size = std::max(1, extent >> level);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[4], level);
            glViewport(0, 0, size, size);
            integer(hiz.id(), "u_initialize", level == 0);
            integer(hiz.id(), "u_source_level", std::max(0, level - 1));
            glProgramUniform2i(hiz.id(), glGetUniformLocation(hiz.id(), "u_source_size"), source_size, source_size);
            glProgramUniform2i(hiz.id(), glGetUniformLocation(hiz.id(), "u_destination_size"), size, size);
            glBindTextureUnit(1, level == 0 ? 0 : textures[4]);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glTextureBarrier();
            source_size = size;
        }
    }
    Pixel render(int frame = 0) {
        glUseProgram(trace.id());
        integer(trace.id(), "u_frame_index", frame);
        for (int unit = 0; unit < 11; ++unit) glBindTextureUnit(unit, textures[unit]);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output, 0);
        glViewport(0, 0, extent, extent);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        Pixel result;
        glReadPixels(center, center, 1, 1, GL_RGBA, GL_FLOAT, result.data());
        RENDER_CHECK(glGetError() == GL_NO_ERROR);
        for (float value : result) RENDER_CHECK(std::isfinite(value));
        return result;
    }
};
} // namespace

RENDER_TEST(test_gpu_ssr_visibility_replaces_environment_and_does_not_double_occlude_hits) {
    GlContext context;
    TraceFixture f;
    f.setup(6, {0.9238795f, 0, 0.3826834f, 0});
    auto blocked = f.render();
    RENDER_CHECK(blocked[3] > 0.99f);
    RENDER_CHECK(blocked[0] < 1.0e-5f);
    f.fill(0, {2, 0.5f, 0.25f, 1});
    f.fill(5, {0, 0, 1, 0}); // Screen AO must not suppress the known connection.
    f.fill(8, {1, 1, 1, 0}); // Material AO is also a fallback approximation.
    auto hit = f.render();
    RENDER_CHECK(nearly_equal(hit[0], 2, 0.01f));
    RENDER_CHECK(nearly_equal(hit[1], 0.5f, 0.01f));
    f.integer(f.trace.id(), "u_ibl_enabled", 0);
    RENDER_CHECK(nearly_equal(f.render()[0], hit[0], 1.0e-5f));
    f.setup(6, {0.9238795f, 0, 0.3826834f, 0}, {0, 0, -1, 0});
    f.fill(0, {100, 100, 100, 1});
    f.scalar("u_edge_fade", 0.5f);
    auto backface = f.render();
    RENDER_CHECK(backface[3] > 0.99f);
    RENDER_CHECK(backface[0] < 1.0e-5f);
    f.fill(3, {0, 0, 0, 2}); // Two-sided emitters remain visible to the ray.
    RENDER_CHECK(f.render()[0] > 1.0f);
    // A ray travelling towards the camera still intersects a depth slab.
    f.setup(3, {0.3826834f, 0, 0.9238795f, 0});
    auto towards_camera = f.render();
    RENDER_CHECK(towards_camera[3] > 0.99f);
    RENDER_CHECK(towards_camera[0] < 1.0e-5f);
}

RENDER_TEST(test_gpu_ssr_miss_fallback_and_diffuse_ggx_white_furnace) {
    GlContext context;
    TraceFixture f;
    f.setup(0, {0, 0, 1, 0});
    RENDER_CHECK(nearly_equal(f.render()[0], 1.0f, 0.01f));
    f.fill(5, {0, 0, 1, 0.25f});
    RENDER_CHECK(nearly_equal(f.render()[0], 0.25f, 0.01f));
    f.integer(f.trace.id(), "u_ibl_enabled", 0);
    RENDER_CHECK(f.render()[0] < 1.0e-5f);
    f.setup(0, {0, 0, 1, 0});
    f.fill(3, {0.8f, 0.4f, 0.2f, 0});
    f.fill(7, {0, 0, 0, 1});
    f.fill(8, {0, 0, 0, 1});
    auto diffuse = f.render();
    RENDER_CHECK(nearly_equal(diffuse[0], 0.8f, 1.0e-5f));
    RENDER_CHECK(nearly_equal(diffuse[1], 0.4f, 1.0e-5f));
    // Unit-Fresnel GGX plus energy compensation integrates to one across
    // roughness and view angle, including rejected below-horizon samples.
    for (float roughness : {0.02f, 0.3f, 0.65f, 1.0f}) {
        f.setup(0, {0.6f, 0, 0.8f, 0});
        f.fill(7, {1, 1, 1, roughness});
        float mean = 0;
        for (int frame = 0; frame < 128; ++frame) mean += f.render(frame)[0] / 128.0f;
        std::cout << "GGX furnace roughness=" << roughness << " mean=" << mean << '\n';
        RENDER_CHECK(nearly_equal(mean, 1.0f, 0.035f));
    }
    // A sloped plane in a white environment cannot occlude its own outgoing
    // reflection, including at a grazing angle to the camera.
    const Pixel plane_normal{0.9238795f, 0, 0.3826834f, 0};
    f.setup(0, plane_normal);
    f.fill(2, plane_normal);
    std::vector<Pixel> plane_depths(extent * extent);
    for (int y = 0; y < extent; ++y) {
        for (int x = 0; x < extent; ++x) {
            const float ndc = 2.0f * (x + 0.5f) / extent - 1.0f;
            const float denominator = plane_normal[2] - plane_normal[0] * ndc;
            const float depth = denominator > 0.0f ? 4.0f * plane_normal[2] / denominator : 0.0f;
            plane_depths[y * extent + x] = {depth, 0, 0, 0};
        }
    }
    f.upload(1, plane_depths);
    f.build_hierarchy();
    RENDER_CHECK(nearly_equal(f.render()[0], 1.0f, 0.01f));
}

RENDER_TEST(test_gpu_ssr_compositor_preserves_direct_light_and_sky) {
    GlContext context;
    TraceFixture f;
    f.setup(6, {0, 0, 1, 0});
    // Deliberately huge old IBL: it must not survive the replacement.
    f.fill(0, {100, 100, 100, 1});
    f.fill(3, {2, 3, 4, 1});
    f.fill(5, {0, 0, 0, 0});
    glUseProgram(f.composite.id());
    glProgramUniform2i(f.composite.id(), glGetUniformLocation(f.composite.id(), "u_full_resolution"), extent, extent);
    glBindTextureUnit(0, f.textures[0]);
    glBindTextureUnit(1, f.textures[3]);
    glBindTextureUnit(4, f.textures[5]);
    glBindTextureUnit(5, f.textures[1]);
    glBindTextureUnit(6, f.textures[2]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, f.output, 0);
    glViewport(0, 0, extent, extent);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    Pixel result;
    glReadPixels(center, center, 1, 1, GL_RGBA, GL_FLOAT, result.data());
    RENDER_CHECK(nearly_equal(result[0], 2));
    RENDER_CHECK(nearly_equal(result[1], 3));
    f.fill(1, {0, 0, 0, 0});
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadPixels(center, center, 1, 1, GL_RGBA, GL_FLOAT, result.data());
    RENDER_CHECK(nearly_equal(result[0], 100));
    RENDER_CHECK(glGetError() == GL_NO_ERROR);
}

RENDER_TEST(test_gpu_ssr_does_not_count_ltc_emission_twice) {
    GlContext context;
    renderer::OpenGlRasterRenderer raster(shader_root / "raster.vert", shader_root / "raster.frag");
    raster.set_auto_reload(false);
    renderer::Scene scene;
    scene.environment = renderer::Color::Zero();
    renderer::Material receiver;
    receiver.type = renderer::MaterialType::Pbr;
    receiver.base_color = renderer::Color::Ones();
    receiver.specular_factor = 0;
    renderer::Material emitter;
    emitter.type = renderer::MaterialType::Emissive;
    emitter.emission = renderer::Color::Constant(10);
    scene.materials = {receiver, emitter};
    const auto quad = [&](renderer::Vec3 a, renderer::Vec3 b,
                          renderer::Vec3 c, renderer::Vec3 d, int material) {
        scene.triangles.emplace_back(a, b, c, material);
        scene.triangles.emplace_back(a, c, d, material);
    };
    quad({-4, -4, -4}, {4, -4, -4}, {4, 4, -4}, {-4, 4, -4}, 0);
    renderer::RectAreaLight light;
    light.position = {1, 0, -2};
    light.axis_u = {0, 0.7f, 0};
    light.axis_v = renderer::Vec3(0.3f, 0, 0.95f) * 0.5f;
    light.radiance = emitter.emission;
    quad(light.position - light.axis_u - light.axis_v,
        light.position - light.axis_u + light.axis_v,
        light.position + light.axis_u + light.axis_v,
        light.position + light.axis_u - light.axis_v, 1);
    scene.rect_area_lights.push_back(light);
    raster.reset(scene);
    renderer::RenderSettings settings;
    settings.width = settings.height = extent;
    settings.opengl.shadow_map.enabled = false;
    settings.opengl.ssr.rays_per_pixel = 8;
    settings.opengl.ssr.max_steps = 256;
    settings.opengl.ssr.denoise_passes = 0;
    settings.opengl.ssr.debug_view = renderer::OpenGlSsrDebugView::RawIndirect;
    renderer::Camera camera({0, 0, 0}, {0, 0, -4}, {0, 1, 0}, 90, 1);
    const auto average_indirect = [&] {
        float average = 0;
        for (int frame = 0; frame < 128; ++frame) {
            raster.render(scene, camera, settings, {});
            RENDER_CHECK(raster.shader_error().empty());
            Pixel p;
            glGetTextureSubImage(raster.output_texture(), 0, center, center, 0,
                1, 1, 1, GL_RGBA, GL_FLOAT, sizeof(Pixel), p.data());
            RENDER_CHECK(glGetError() == GL_NO_ERROR);
            average += p[0] / 128.0f;
        }
        return average;
    };
    RENDER_CHECK(average_indirect() < 1.0e-5f);
    // Unregistered emissive geometry must still illuminate through SSR.
    scene.rect_area_lights.clear();
    raster.reset(scene);
    const float mesh_emission = average_indirect();
    std::cout << "Unregistered emitter indirect=" << mesh_emission << '\n';
    RENDER_CHECK(mesh_emission > 0.02f);
}

RENDER_TEST(test_gpu_ssr_complete_pipeline_resize_settings_and_history) {
    GlContext context;
    renderer::OpenGlRasterRenderer raster(shader_root / "raster.vert", shader_root / "raster.frag");
    raster.set_auto_reload(false);
    auto scene = renderer::make_cornell_box_scene();
    raster.reset(scene);
    renderer::RenderSettings settings;
    settings.width = 65;
    settings.height = 47;
    renderer::Camera camera({0, 1, 4}, {0, 1, 0}, {0, 1, 0}, 50, 65.0f / 47);
    renderer::InteractiveFrameState frame;
    const auto render = [&] {
        raster.render(scene, camera, settings, frame);
        RENDER_CHECK(raster.shader_error().empty());
        RENDER_CHECK(glGetError() == GL_NO_ERROR);
        std::vector<Pixel> pixels(settings.width * settings.height);
        glGetTextureImage(raster.output_texture(), 0, GL_RGBA, GL_FLOAT,
            static_cast<GLsizei>(pixels.size() * sizeof(Pixel)), pixels.data());
        for (const auto& p : pixels) for (float value : p) RENDER_CHECK(std::isfinite(value));
        return pixels;
    };
    render();
    render();
    settings.opengl.ssr.debug_view = renderer::OpenGlSsrDebugView::HistoryLength;
    render(); // Settings changes invalidate history.
    const auto history = render();
    bool accumulated = false;
    for (const auto& p : history) accumulated |= p[0] >= 2.0f / settings.opengl.ssr.max_history_frames - 1.0e-5f;
    RENDER_CHECK(accumulated);
    settings.width = 1;
    settings.height = 1;
    render();
    settings.width = 67;
    settings.height = 49;
    settings.opengl.ssr.debug_view = renderer::OpenGlSsrDebugView::Final;
    settings.opengl.ibl_enabled = false;
    render();
    settings.opengl.ssr.enabled = false;
    render();
    settings.opengl.ssr.enabled = true;
    render();

    // Optional artifacts for visual QA; normal CTest runs do not write files.
    if (const char* capture_dir = SDL_getenv("RENDERER_OPENGL_CAPTURE_DIR")) {
        settings.width = 384;
        settings.height = 384;
        settings.opengl.ibl_enabled = true;
        scene.environment = renderer::Color(0.5f, 0.6f, 0.8f);
        raster.reset(scene);
        camera = renderer::Camera({0, 0, 1.2f}, {0, 0, -2}, {0, 1, 0}, 45, 1);
        const auto capture = [&](const char* filename, int frame_count) {
            for (int index = 1; index < frame_count; ++index) render();
            GLuint query = 0;
            glGenQueries(1, &query);
            glBeginQuery(GL_TIME_ELAPSED, query);
            const auto pixels = render();
            glEndQuery(GL_TIME_ELAPSED);
            GLuint64 elapsed = 0;
            glGetQueryObjectui64v(query, GL_QUERY_RESULT, &elapsed);
            glDeleteQueries(1, &query);
            std::cout << filename << " GPU ms=" << elapsed / 1.0e6 << '\n';
            renderer::Image image(settings.width, settings.height);
            renderer::DisplaySettings display;
            display.tone_mapper = renderer::ToneMapper::Aces;
            for (int y = 0; y < settings.height; ++y) {
                for (int x = 0; x < settings.width; ++x) {
                    const auto& p = pixels[y * settings.width + x];
                    image.set_pixel(x, settings.height - 1 - y,
                        renderer::apply_display_transform({p[0], p[1], p[2]}, display));
                }
            }
            RENDER_CHECK(image.write_png((std::filesystem::path(capture_dir) / filename).string()));
        };
        settings.opengl.ssr.enabled = false;
        capture("cornell-ibl-fallback.png", 1);
        settings.opengl.ssr.enabled = true;
        capture("cornell-unified-ssr.png", 64);
        settings.opengl.ssr.debug_view = renderer::OpenGlSsrDebugView::FilteredIndirect;
        capture("cornell-indirect.png", 64);
    }
}
