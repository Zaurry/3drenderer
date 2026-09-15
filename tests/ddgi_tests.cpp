#include "core/image.h"
#include "render/ddgi/cuda_ddgi_volume.h"
#include "render/ddgi/ddgi_settings_json.h"
#include "render/display_settings.h"
#include "render/interactive/viewer_render_backend.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "scene/scene_document.h"
#include "test_framework.h"
#include <SDL3/SDL.h>
#include <cstdlib>
#include <fstream>
#include <glad/gl.h>
#include <limits>

namespace {
using namespace renderer;
using Pixel = std::array<float, 4>;
const auto shaders = std::filesystem::path(RENDERER_SOURCE_DIR) / "shaders/opengl";
CudaDeviceContext device() {
    std::string reason;
    auto result = CudaDeviceContext::try_create(0, &reason);
    if (!result) {
        if (std::getenv("DDGI_REQUIRE_GPU"))
            throw TestFailure{reason};
        RENDER_SKIP(reason);
    }
    return *result;
}
struct GlContext {
    SDL_Window *window = nullptr;
    SDL_GLContext context = nullptr;
    GlContext() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            const std::string reason = SDL_GetError();
            if (std::getenv("DDGI_REQUIRE_GPU"))
                throw TestFailure{reason};
            RENDER_SKIP(reason);
        }
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        window = SDL_CreateWindow("DDGI tests", 96, 96, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        if (window)
            context = SDL_GL_CreateContext(window);
        if (!context) {
            const std::string reason = SDL_GetError();
            if (window)
                SDL_DestroyWindow(window);
            SDL_Quit();
            if (std::getenv("DDGI_REQUIRE_GPU"))
                throw TestFailure{reason};
            RENDER_SKIP(reason);
        }
        RENDER_CHECK(SDL_GL_MakeCurrent(window, context));
        RENDER_CHECK(gladLoadGL(reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress)) != 0);
    }
    ~GlContext() {
        SDL_GL_DestroyContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};
OpenGlRenderSettings small_settings() {
    OpenGlRenderSettings s;
    s.ddgi.auto_fit = false;
    s.ddgi.origin = {-2, -2, -2};
    s.ddgi.extent = {4, 4, 4};
    s.ddgi.probe_counts = {3, 3, 3};
    s.ddgi.probes_per_frame = 27;
    s.ddgi.rays_per_probe = 256;
    s.ddgi.relocation = false;
    s.ddgi.classification = false;
    s.ddgi.hysteresis = .95f;
    s.dominant_light.enabled = false;
    s.ambient_occlusion.mode = OpenGlAmbientOcclusionMode::Off;
    s.ssr.enabled = false;
    return s;
}
InteractiveFrameState tick() {
    InteractiveFrameState s;
    s.delta_seconds = 1.f / 60;
    return s;
}
void quad(Scene &s, Vec3 a, Vec3 b, Vec3 c, Vec3 d, int material = 0) {
    s.triangles.emplace_back(a, b, c, material);
    s.triangles.emplace_back(a, c, d, material);
}
Scene cube_scene() {
    Scene s;
    s.environment = Color(.2f, .3f, .4f);
    Material m;
    m.type = MaterialType::Pbr;
    m.specular_factor = 0;
    m.two_sided = false;
    s.materials = {m};
    quad(s, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1});
    quad(s, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}, {1, 1, -1});
    quad(s, {1, -1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1});
    quad(s, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1});
    quad(s, {-1, 1, 1}, {1, 1, 1}, {1, 1, -1}, {-1, 1, -1});
    quad(s, {-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {-1, -1, 1});
    return s;
}
std::vector<Pixel> read_output(ViewerRenderBackend &backend, bool expect_ddgi = true) {
    const auto &output = std::get<OpenGlTextureHandle>(backend.output());
    std::vector<Pixel> pixels(std::size_t(output.width * output.height));
    glGetTextureImage(output.texture, 0, GL_RGBA, GL_FLOAT,
                      static_cast<GLsizei>(pixels.size() * sizeof(Pixel)), pixels.data());
    RENDER_CHECK(glGetError() == GL_NO_ERROR);
    for (const auto &p : pixels)
        for (float c : p)
            RENDER_CHECK(std::isfinite(c));
    const auto stats = std::get<OpenGlViewerStatistics>(backend.statistics());
    if (!stats.shader_error.empty())
        throw TestFailure{stats.shader_error};
    if (expect_ddgi && !stats.ddgi.active)
        throw TestFailure{stats.ddgi.status + ": " + stats.ddgi.detail};
    return pixels;
}
Color center_mean(const std::vector<Pixel> &pixels, int width, int height) {
    Color sum = Color::Zero();
    int count = 0;
    for (int y = height / 3; y < height * 2 / 3; ++y)
        for (int x = width / 3; x < width * 2 / 3; ++x) {
            const auto p = pixels[std::size_t(y * width + x)];
            sum += Color(p[0], p[1], p[2]);
            ++count;
        }
    return sum / float(count);
}
void capture(const char *name, const std::vector<Pixel> &pixels, int w, int h) {
    const char *directory = std::getenv("DDGI_CAPTURE_DIR");
    if (!directory)
        return;
    std::filesystem::create_directories(directory);
    Image image(w, h);
    DisplaySettings display;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const auto p = pixels[std::size_t((h - 1 - y) * w + x)];
            image.set_pixel(x, y, apply_display_transform(Color(p[0], p[1], p[2]), display));
        }
    RENDER_CHECK(image.write_png((std::filesystem::path(directory) / name).string()));
}
void metrics(const char *name, const nlohmann::json &data) {
    if (const char *directory = std::getenv("DDGI_CAPTURE_DIR")) {
        std::filesystem::create_directories(directory);
        std::ofstream output(std::filesystem::path(directory) / name);
        output << data.dump(2) << '\n';
    }
}
std::unique_ptr<ViewerRenderBackend> gl_backend() {
    return make_viewer_render_backend(InteractiveRenderMode::OpenGl, shaders / "raster.vert",
                                      shaders / "raster.frag");
}
void settle(ViewerRenderBackend &backend, const RenderSceneSnapshot &scene, const Camera &camera,
            const RenderSettings &settings, int frames) {
    for (int i = 0; i < frames; ++i)
        backend.render(scene, camera, settings, tick());
}
double region_mse(const std::vector<Pixel> &pixels, const Image &reference, int x0, int y0, int x1,
                  int y1) {
    double error = 0;
    int samples = 0;
    const int width = reference.width(), height = reference.height();
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            const auto p = pixels[std::size_t((height - 1 - y) * width + x)];
            error += (Color(p[0], p[1], p[2]) - reference.pixel(x, y)).squaredNorm();
            ++samples;
        }
    return error / (3 * samples);
}
} // namespace

RENDER_TEST(test_ddgi_settings_layout_and_session_fields) {
    using namespace renderer;
    DdgiSettings s;
    s.auto_fit = false;
    s.origin = {-3, 2, 1};
    s.extent = {6, 2, 4};
    s.probe_counts = {3, 2, 5};
    s.probes_per_frame = 4;
    s.hysteresis = .8f;
    s.paused = true;
    s.show_probes = true;
    s.reset_generation = 4;
    s.fit_generation = 2;
    const auto json = ddgi_settings_json(s);
    auto expected = s;
    expected.reset_generation = expected.fit_generation = 0;
    RENDER_CHECK(parse_ddgi_settings(json) == expected);
    RENDER_CHECK(!json.contains("reset_generation"));
    auto malformed = json;
    malformed["rays_per_probe"] = "wrong";
    malformed["probe_counts"] = {100000, 100000, 100000};
    malformed["intensity"] = nullptr;
    malformed["extent"] = {0, -1, 3};
    auto clamped = parse_ddgi_settings(malformed);
    RENDER_CHECK(clamped.rays_per_probe == 128);
    RENDER_CHECK(clamped.probe_counts[0] * clamped.probe_counts[1] * clamped.probe_counts[2] <=
                 8192);
    RENDER_CHECK(clamped.extent[0] > 0 && clamped.extent[1] > 0);
    const auto layout = make_ddgi_layout({}, s);
    RENDER_CHECK(layout.count() == 30);
    RENDER_CHECK(layout.position(0).isApprox(Vec3(-3, 2, 1)));
    RENDER_CHECK(layout.position(29).isApprox(Vec3(3, 4, 5)));
}

RENDER_TEST(test_ddgi_constant_environment_hdr_and_octahedral_borders) {
    const auto ctx = device();
    Scene scene;
    scene.environment = Color(.25f, .5f, 2);
    auto snapshot = make_render_scene_snapshot(scene);
    auto settings = small_settings();
    CudaDdgiVolume volume(ctx);
    for (int i = 0; i < 4; ++i)
        volume.update(snapshot, settings, tick());
    const auto data = volume.download_atlases();
    const auto layout = volume.statistics().layout;
    RENDER_CHECK(volume.statistics().active_probes == 27);
    for (int probe = 0; probe < layout.count(); ++probe)
        for (int n : {8, 16}) {
            const auto &atlas = n == 8 ? data.irradiance : data.distance;
            const int pitch = layout.width(n), base = (probe / layout.columns * (n + 2)) * pitch +
                                                      probe % layout.columns * (n + 2);
            for (int y = 0; y < n + 2; ++y)
                for (int x = 0; x < n + 2; ++x) {
                    const auto [sx, sy] = ddgi_border_source(x, y, n);
                    for (int c = 0; c < 3; ++c)
                        RENDER_CHECK(nearly_equal(atlas[std::size_t(base + y * pitch + x)][c],
                                                  atlas[std::size_t(base + sy * pitch + sx)][c],
                                                  1e-5f));
                    if (n == 8)
                        for (int c = 0; c < 3; ++c)
                            RENDER_CHECK(nearly_equal(atlas[std::size_t(base + y * pitch + x)][c],
                                                      scene.environment[c] * 3.141592654f, 3e-5f));
                }
        }
    snapshot.environment = Color::Zero();
    ++snapshot.revisions.environment;
    for (int i = 0; i < 8; ++i)
        volume.update(snapshot, settings, tick());
    const auto dark = volume.download_atlases();
    float peak = 0;
    for (const auto &p : dark.irradiance)
        peak = std::max(peak, p[2]);
    std::cout << "DDGI light removal residual=" << peak << '\n';
    RENDER_CHECK(peak < .001f);
    const auto resets = volume.statistics().reset_count;
    auto state = tick();
    state.camera_changed = true;
    state.camera_cut = true;
    state.framebuffer_resized = true;
    volume.update(snapshot, settings, state);
    RENDER_CHECK(volume.statistics().reset_count == resets);
    settings.ddgi.paused = true;
    const auto frozen = volume.download_atlases();
    volume.update(snapshot, settings, tick());
    RENDER_CHECK(volume.download_atlases().irradiance == frozen.irradiance);
}

RENDER_TEST(test_ddgi_inside_geometry_classification_relocation_and_reactivation) {
    const auto ctx = device();
    auto snapshot = make_render_scene_snapshot(cube_scene());
    auto settings = small_settings();
    settings.ddgi.origin = {-.1f, -.1f, -.1f};
    settings.ddgi.extent = {.2f, .2f, .2f};
    settings.ddgi.classification = true;
    CudaDdgiVolume volume(ctx);
    volume.update(snapshot, settings, tick());
    RENDER_CHECK(volume.statistics().active_probes == 0);
    snapshot.instances.clear();
    ++snapshot.revisions.topology;
    volume.update(snapshot, settings, tick());
    RENDER_CHECK(volume.statistics().active_probes == 27);
    snapshot = make_render_scene_snapshot(cube_scene());
    settings.ddgi.origin = {.95f, 0, 0};
    settings.ddgi.extent = {1, 1, 1};
    settings.ddgi.relocation = true;
    for (int i = 0; i < 4; ++i)
        volume.update(snapshot, settings, tick());
    const auto data = volume.download_atlases();
    bool relocated = false;
    for (int i = 0; i < 27; ++i)
        relocated |= std::abs(data.metadata[std::size_t(i)][0]) > 0.001f;
    RENDER_CHECK(relocated);
    RENDER_CHECK(volume.statistics().active_probes > 0);
}

RENDER_TEST(test_ddgi_instanced_motion_refits_without_moving_volume) {
    const auto ctx = device();
    auto snapshot = make_render_scene_snapshot(cube_scene());
    auto settings = small_settings();
    settings.ddgi.auto_fit = true;
    CudaDdgiVolume volume(ctx);
    volume.update(snapshot, settings, tick());
    const auto before = volume.statistics();
    auto &instance = snapshot.instances[0];
    instance.object_to_world(0, 3) = 2;
    instance.world_to_object = instance.object_to_world.inverse();
    instance.world_bounds.min.x() += 2;
    instance.world_bounds.max.x() += 2;
    ++snapshot.revisions.transforms;
    volume.update(snapshot, settings, tick());
    RENDER_CHECK(volume.statistics().layout == before.layout);
    RENDER_CHECK(volume.statistics().reset_count == before.reset_count);
    RENDER_CHECK(volume.statistics().tlas_refits > before.tlas_refits);
    RENDER_CHECK(volume.statistics().blas_builds == before.blas_builds);
    ++settings.ddgi.fit_generation;
    volume.update(snapshot, settings, tick());
    RENDER_CHECK(volume.statistics().layout != before.layout);
}

RENDER_TEST(test_ddgi_opengl_energy_interop_fallback_and_switches) {
    const auto ctx = device();
    (void)ctx;
    GlContext context;
    Scene scene;
    scene.environment = Color(.25f, .5f, 2);
    Material m;
    m.type = MaterialType::Pbr;
    m.base_color = Color::Constant(.5f);
    m.specular_factor = 0;
    scene.materials = {m};
    quad(scene, {-8, -8, 0}, {8, -8, 0}, {8, 8, 0}, {-8, 8, 0});
    auto snapshot = make_render_scene_snapshot(scene);
    RenderSettings settings;
    settings.width = 65;
    settings.height = 49;
    settings.opengl = small_settings();
    settings.opengl.ddgi.origin = {-2, -2, -1};
    settings.opengl.ddgi.extent = {4, 4, 4};
    Camera camera({0, 0, 3}, {0, 0, 0}, {0, 1, 0}, 45, 65.f / 49);
    std::vector<Pixel> interop_pixels;
    for (bool fallback : {false, true}) {
        auto backend =
            make_viewer_render_backend(InteractiveRenderMode::OpenGl, shaders / "raster.vert",
                                       shaders / "raster.frag", fallback);
        backend->reset(snapshot, settings);
        for (int i = 0; i < 24; ++i)
            backend->render(snapshot, camera, settings, tick());
        auto pixels = read_output(*backend);
        const auto stats = std::get<OpenGlViewerStatistics>(backend->statistics());
        std::cout << "DDGI GL " << stats.ddgi.status
                  << " mean=" << center_mean(pixels, 65, 49).transpose() << '\n';
        RENDER_CHECK(stats.ddgi.status == (fallback ? "fallback" : "active"));
        RENDER_CHECK(stats.ddgi.atlas_downloads == (fallback ? 24u : 0u));
        RENDER_CHECK(center_mean(pixels, 65, 49).isApprox(scene.environment * .5f, .05f));
        if (!fallback)
            interop_pixels = pixels;
        else
            for (std::size_t i = 0; i < pixels.size(); ++i)
                for (int c = 0; c < 4; ++c)
                    RENDER_CHECK(nearly_equal(pixels[i][c], interop_pixels[i][c], 1e-5f));
        const auto resets = stats.ddgi.reset_count;
        auto resized = settings;
        resized.width = 1;
        resized.height = 1;
        auto state = tick();
        state.framebuffer_resized = true;
        backend->render(snapshot, camera, resized, state);
        RENDER_CHECK(read_output(*backend).size() == 1);
        RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.reset_count ==
                     resets);
        settings.opengl.ssr.enabled = true;
        for (int i = 0; i < 8; ++i)
            backend->render(snapshot, camera, settings, tick());
        pixels = read_output(*backend);
        RENDER_CHECK(center_mean(pixels, 65, 49).isApprox(scene.environment * .5f, .06f));
        settings.opengl.ssr.enabled = false;
        settings.opengl.ibl_enabled = false;
        for (bool ssr : {false, true}) {
            settings.opengl.ssr.enabled = ssr;
            settle(*backend, snapshot, camera, settings, 8);
            RENDER_CHECK(center_mean(read_output(*backend), 65, 49).maxCoeff() < 1e-5f);
        }
        settings.opengl.ibl_enabled = true;
        settings.opengl.ssr.enabled = false;
        auto metal = snapshot;
        metal.instances[0].materials[0].metallic = 1;
        metal.instances[0].materials[0].specular_factor = 1;
        ++metal.revisions.materials;
        auto debug = settings;
        debug.opengl.ddgi.debug_view = DdgiDebugView::Indirect;
        backend->render(metal, camera, debug, tick());
        RENDER_CHECK(center_mean(read_output(*backend), 65, 49).maxCoeff() < 1e-5f);
        settings.opengl.ddgi.show_probes = true;
        backend->render(snapshot, camera, settings, tick());
        read_output(*backend);
        settings.opengl.ddgi.show_probes = false;
    }
    // Zero user bias can place a receiver exactly at a probe. Its normal weight
    // must be defined there, otherwise trilinear interpolation returns black.
    auto coincident = settings;
    coincident.opengl.ddgi.origin = {-2, -2, 0};
    coincident.opengl.ddgi.normal_bias = coincident.opengl.ddgi.view_bias = 0;
    auto backend = gl_backend();
    backend->reset(snapshot, coincident);
    settle(*backend, snapshot, camera, coincident, 2);
    const auto exact = read_output(*backend)[std::size_t(24 * 65 + 32)];
    for (int c = 0; c < 3; ++c)
        RENDER_CHECK(nearly_equal(exact[c], scene.environment[c] * .5f, 2e-5f));
}

RENDER_TEST(test_ddgi_analytic_emitter_partition_and_alpha_mask) {
    const auto ctx = device();
    Scene scene;
    scene.environment = Color::Zero();
    Material emitter;
    emitter.type = MaterialType::Emissive;
    emitter.emission = Color(2, 3, 4);
    emitter.two_sided = false;
    scene.materials = {emitter};
    RectAreaLight rect;
    rect.position = {0, 0, 2};
    rect.axis_u = {2, 0, 0};
    rect.axis_v = {0, 2, 0};
    rect.radiance = emitter.emission;
    scene.rect_area_lights.push_back(rect);
    auto snapshot = make_render_scene_snapshot(scene);
    auto settings = small_settings();
    settings.ddgi.origin = {-.5f, -.5f, -.5f};
    settings.ddgi.extent = {1, 1, 1};
    CudaDdgiVolume volume(ctx);
    volume.update(snapshot, settings, tick());
    auto peak = [](const DdgiAtlasReadback &data) {
        float result = 0;
        for (const auto &p : data.irradiance)
            result = std::max(result, p[2]);
        return result;
    };
    RENDER_CHECK(peak(volume.download_atlases()) <
                 1e-6f); // LTC already represents this direct emitter.
    settings.ltc_area_lights_enabled = false;
    ++settings.ddgi.reset_generation;
    volume.update(snapshot, settings, tick());
    const float unrepresented = peak(volume.download_atlases());
    RENDER_CHECK(unrepresented > 1); // Disabling LTC returns emission to the probe field.
    snapshot.instances[0].materials[0].type = MaterialType::Pbr;
    snapshot.instances[0].materials[0].alpha_mode = AlphaMode::Mask;
    snapshot.instances[0].materials[0].opacity = 0;
    ++snapshot.revisions.materials;
    ++settings.ddgi.reset_generation;
    volume.update(snapshot, settings, tick());
    RENDER_CHECK(peak(volume.download_atlases()) < 1e-6f);
    snapshot.instances[0].materials[0].opacity = 1;
    ++snapshot.revisions.materials;
    ++settings.ddgi.reset_generation;
    volume.update(snapshot, settings, tick());
    RENDER_CHECK(peak(volume.download_atlases()) > 1);
}

RENDER_TEST(test_ddgi_dominant_environment_residual_and_rotation) {
    const auto ctx = device();
    Scene scene = make_cornell_box_scene();
    scene.triangles.erase(scene.triangles.end() - 2, scene.triangles.end());
    std::vector<Color> hdr(32 * 16, Color::Constant(.1f));
    for (int y = 6; y < 10; ++y)
        for (int x = 23; x < 25; ++x)
            hdr[std::size_t(y * 32 + x)] = {20, 8, 4};
    scene.environment_map = std::make_shared<const EnvironmentMap>(32, 16, hdr);
    scene.environment = {.7f, .8f, 1};
    scene.environment_intensity = 1.4f;
    scene.environment_rotation_degrees = 37;
    auto snapshot = make_render_scene_snapshot(scene);
    auto settings = small_settings();
    settings.ddgi.auto_fit = true;
    settings.dominant_light.enabled = true;
    settings.dominant_light.intensity_scale = .8f;
    const auto dominant = extract_dominant_environment_light(
        scene.environment_map, settings.dominant_light.peak_threshold_ev,
        settings.dominant_light.minimum_energy_fraction);
    RENDER_CHECK(dominant.valid);
    auto explicit_scene = snapshot;
    explicit_scene.environment_map = dominant.residual_map;
    const Mat3 rotation =
        Eigen::AngleAxisf(37 * 3.141592654f / 180, Vec3::UnitY()).toRotationMatrix();
    DirectionalLight sun;
    sun.direction = -rotation * dominant.direction;
    sun.radiance = dominant.integrated_radiance.cwiseProduct(scene.environment) *
                   (scene.environment_intensity * .8f);
    sun.angular_radius_radians = dominant.angular_radius_radians;
    explicit_scene.directional_lights.push_back(sun);
    auto explicit_settings = settings;
    explicit_settings.dominant_light.enabled = false;
    CudaDdgiVolume extracted(ctx), explicit_volume(ctx);
    for (int i = 0; i < 12; ++i) {
        extracted.update(snapshot, settings, tick());
        explicit_volume.update(explicit_scene, explicit_settings, tick());
    }
    const auto a = extracted.download_atlases(), b = explicit_volume.download_atlases();
    for (std::size_t i = 0; i < a.irradiance.size(); ++i)
        for (int c = 0; c < 3; ++c)
            RENDER_CHECK(nearly_equal(a.irradiance[i][c], b.irradiance[i][c], 2e-5f));
    // Replacing the HDR object must update transport without resetting the layout.
    const auto resets = extracted.statistics().reset_count;
    snapshot.environment_map =
        std::make_shared<const EnvironmentMap>(4, 2, std::vector<Color>(8, Color::Constant(.2f)));
    ++snapshot.revisions.environment;
    extracted.update(snapshot, settings, tick());
    RENDER_CHECK(extracted.statistics().reset_count == resets);
}

RENDER_TEST(test_ddgi_cornell_multibounce_against_offline_reference) {
    const auto ctx = device();
    (void)ctx;
    GlContext context;
    auto snapshot = make_render_scene_snapshot(make_cornell_box_scene());
    RenderSettings settings;
    settings.width = settings.height = 96;
    settings.path.max_bounces = 8;
    settings.path.samples_per_pixel = 1024;
    settings.opengl.dominant_light.enabled = false;
    settings.opengl.ambient_occlusion.mode = OpenGlAmbientOcclusionMode::Off;
    settings.opengl.ssr.enabled = false;
    settings.opengl.ddgi.rays_per_probe = 256;
    const Camera camera({0, .15f, 1.5f}, {0, .15f, -2}, Vec3::UnitY(), 45, 1);
    const auto reference = render_cuda_path(snapshot, camera, settings).image;
    std::vector<Pixel> ref(96 * 96);
    for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 96; ++x) {
            const Color c = reference.pixel(x, 95 - y);
            ref[std::size_t(y * 96 + x)] = {c.x(), c.y(), c.z(), 1};
        }
    capture("cornell-reference-1024spp.png", ref, 96, 96);
    auto backend = gl_backend();
    backend->reset(snapshot, settings);
    settle(*backend, snapshot, camera, settings, 96);
    const auto on = read_output(*backend);
    capture("cornell-ddgi.png", on, 96, 96);
    settings.opengl.ddgi.debug_view = DdgiDebugView::Indirect;
    backend->render(snapshot, camera, settings, tick());
    capture("cornell-indirect.png", read_output(*backend), 96, 96);
    settings.opengl.ddgi.enabled = false;
    settings.opengl.ddgi.debug_view = DdgiDebugView::Final;
    backend->render(snapshot, camera, settings, tick());
    const auto off = read_output(*backend, false);
    capture("cornell-ddgi-off.png", off, 96, 96);
    // Fixed back-wall region excludes the visible ceiling emitter and silhouettes.
    const double enabled_error = region_mse(on, reference, 28, 34, 68, 62),
                 disabled_error = region_mse(off, reference, 28, 34, 68, 62);
    const Color mean = center_mean(on, 96, 96);
    std::cout << "Cornell DDGI ROI MSE=" << enabled_error << " disabled=" << disabled_error
              << " mean=" << mean.transpose() << '\n';
    metrics("cornell.json", {{"reference_spp", 1024},
                             {"frames", 96},
                             {"region_top_left_xyxy", {28, 34, 68, 62}},
                             {"ddgi_mse", enabled_error},
                             {"disabled_mse", disabled_error}});
    RENDER_CHECK(enabled_error < disabled_error * .7);
    RENDER_CHECK(mean.minCoeff() > .005f);
}

RENDER_TEST(test_ddgi_viewer_document_default_ao_ssr_combinations) {
    const auto ctx = device(); (void)ctx; GlContext context;
    auto document = SceneDocument::from_scene(make_cornell_box_scene(), "Cornell", "cornell_box");
    const auto& snapshot = document.render_scene_snapshot();
    const Camera camera({0,.15f,1.5f},{0,.15f,-2},Vec3::UnitY(),45,1);
    RenderSettings settings; settings.width=settings.height=96;
    auto backend=gl_backend();backend->reset(snapshot,settings);
    settle(*backend,snapshot,camera,settings,128);
    const auto initial=read_output(*backend);
    capture("viewer-default-ao-ssr.png",initial,96,96);
    std::vector<float> means{center_mean(initial,96,96).mean()};
    std::cout<<"DDGI viewer initial default mean="<<means.front()<<'\n';
    for(int mode=0;mode<4;++mode) {
        settings.opengl.ssr.enabled=(mode&1)!=0;
        settings.opengl.ambient_occlusion.mode=(mode&2)!=0?OpenGlAmbientOcclusionMode::Gtao:OpenGlAmbientOcclusionMode::Off;
        settle(*backend,snapshot,camera,settings,8);
        auto pixels=read_output(*backend);means.push_back(center_mean(pixels,96,96).mean());
        std::cout<<"DDGI viewer document SSR/AO="<<mode<<" mean="<<means.back()<<'\n';
        capture(("viewer-combination-"+std::to_string(mode)+".png").c_str(),pixels,96,96);
    }
    for(float mean:means)RENDER_CHECK(mean>.05f);
}

RENDER_TEST(test_ddgi_two_rooms_visibility_and_offscreen_emitter) {
    const auto ctx = device();
    (void)ctx;
    GlContext context;
    Scene scene;
    scene.environment = Color::Zero();
    Material wall;
    wall.base_color = Color::Constant(.6f);
    Material emitter;
    emitter.type = MaterialType::Emissive;
    emitter.emission = {8, 5, 2};
    scene.materials = {wall, emitter};
    quad(scene, {-2, -1, -2}, {2, -1, -2}, {2, -1, 2}, {-2, -1, 2});
    quad(scene, {-2, 1, 2}, {2, 1, 2}, {2, 1, -2}, {-2, 1, -2});
    quad(scene, {-2, -1, -2}, {-2, -1, 2}, {-2, 1, 2}, {-2, 1, -2});
    quad(scene, {2, -1, 2}, {2, -1, -2}, {2, 1, -2}, {2, 1, 2});
    quad(scene, {-2, -1, -2}, {-2, 1, -2}, {2, 1, -2}, {2, -1, -2});
    quad(scene, {2, -1, 2}, {2, 1, 2}, {-2, 1, 2}, {-2, -1, 2});
    quad(scene, {0, -1, -2}, {0, 1, -2}, {0, 1, 2},
         {0, -1, 2}); // Zero-thickness, two-sided divider.
    quad(scene, {-1.8f, .99f, -.6f}, {-1.8f, .99f, .6f}, {-.2f, .99f, .6f}, {-.2f, .99f, -.6f}, 1);
    auto snapshot = make_render_scene_snapshot(scene);
    RenderSettings settings;
    settings.width = settings.height = 96;
    settings.opengl.dominant_light.enabled = false;
    settings.opengl.ambient_occlusion.mode = OpenGlAmbientOcclusionMode::Off;
    settings.opengl.ssr.enabled = false;
    settings.opengl.ddgi.rays_per_probe = 256;
    const Camera left({-1, 0, 1.7f}, {-1, 0, -2}, Vec3::UnitY(), 40, 1),
        right({1, 0, 1.7f}, {1, 0, -2}, Vec3::UnitY(), 40, 1);
    auto backend = gl_backend();
    backend->reset(snapshot, settings);
    settle(*backend, snapshot, left, settings, 128);
    const auto lit = read_output(*backend);
    capture("rooms-offscreen-emitter.png", lit, 96, 96);
    const auto resets = std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.reset_count;
    auto move = tick();
    move.camera_changed = true;
    move.camera_cut = true;
    backend->render(snapshot, right, settings, move);
    const auto dark = read_output(*backend);
    capture("rooms-dark.png", dark, 96, 96);
    RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.reset_count ==
                 resets);
    const float lit_mean = center_mean(lit, 96, 96).mean(),
                dark_mean = center_mean(dark, 96, 96).mean();
    std::cout << "DDGI separated rooms lit=" << lit_mean << " dark=" << dark_mean
              << " leakage=" << dark_mean / lit_mean << '\n';
    metrics("rooms.json", {{"lit_mean", lit_mean},
                           {"dark_mean", dark_mean},
                           {"leakage_ratio", dark_mean / lit_mean}});
    RENDER_CHECK(lit_mean > .01f);
    RENDER_CHECK(dark_mean < lit_mean * .02f);
    // An enclosing room must also remain dark with bright environmental IBL outside.
    snapshot.environment = Color::Ones();
    ++snapshot.revisions.environment;
    settings.opengl.ddgi.debug_view = DdgiDebugView::Indirect;
    settle(*backend, snapshot, right, settings, 64);
    const auto enclosed = read_output(*backend);
    capture("rooms-enclosed-environment.png", enclosed, 96, 96);
    std::cout << "DDGI enclosed diffuse=" << center_mean(enclosed, 96, 96).transpose() << '\n';
    RENDER_CHECK(center_mean(enclosed, 96, 96).mean() < .02f);
}

RENDER_TEST(test_ddgi_dynamic_changes_converge_within_eight_sweeps) {
    const auto ctx = device();
    (void)ctx;
    GlContext context;
    Scene scene = make_cornell_box_scene();
    scene.triangles.erase(scene.triangles.end() - 2,
                          scene.triangles.end()); // Use an editable punctual source.
    scene.point_lights.push_back(PointLight{{-.4f, .75f, -1.8f}, {5, 4, 3}});
    scene.environment = Color::Constant(.04f);
    scene.spheres.emplace_back(Vec3(-.5f, -.55f, -2), .4f, 2);
    auto original = make_render_scene_snapshot(scene);
    RenderSettings settings;
    settings.width = settings.height = 64;
    settings.opengl = small_settings();
    settings.opengl.ddgi.origin = {-1.1f, -1.1f, -3.1f};
    settings.opengl.ddgi.extent = {2.2f, 2.2f, 2.2f};
    settings.opengl.ddgi.probe_counts = {8, 6, 8};
    settings.opengl.ddgi.probes_per_frame = 96;
    settings.opengl.ddgi.rays_per_probe = 512;
    settings.opengl.ddgi.relocation = true;
    settings.opengl.ddgi.classification = true;
    const Camera camera({0, .15f, 1.5f}, {0, .15f, -2}, Vec3::UnitY(), 45, 1);
    nlohmann::json results = nlohmann::json::array();
    for (int scenario = 0; scenario < 5; ++scenario) {
        auto changed = original;
        const char *name = "";
        if (scenario == 0) {
            name = "move-light";
            changed.point_lights[0].position = {.5f, .6f, -2.5f};
            ++changed.revisions.lighting;
        }
        if (scenario == 1) {
            name = "disable-light";
            changed.point_lights.clear();
            ++changed.revisions.lighting;
        }
        if (scenario == 2) {
            name = "material";
            changed.instances[0].materials[0].base_color = {.05f, .08f, .6f};
            ++changed.revisions.materials;
        }
        if (scenario == 3) {
            name = "environment";
            changed.environment = {.7f, .3f, .1f};
            ++changed.revisions.environment;
        }
        if (scenario == 4) {
            name = "move-occluder";
            auto &instance = changed.instances[1];
            instance.object_to_world(0, 3) += 1;
            instance.world_to_object = instance.object_to_world.inverse();
            instance.world_bounds.min.x() += 1;
            instance.world_bounds.max.x() += 1;
            ++changed.revisions.transforms;
        }
        auto backend = gl_backend();
        backend->reset(original, settings);
        settle(*backend, original, camera, settings, 96);
        const auto resets =
            std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.reset_count;
        settle(*backend, changed, camera, settings, 32);
        const auto updated = read_output(*backend);
        const auto stats = std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi;
        RENDER_CHECK(stats.reset_count == resets);
        RENDER_CHECK(stats.maximum_age <
                     8); // Priority updates cannot starve the round-robin sweep.
        auto fresh = gl_backend();
        fresh->reset(changed, settings);
        settle(*fresh, changed, camera, settings, 160);
        const auto converged = read_output(*fresh);
        const float actual = center_mean(updated, 64, 64).mean(),
                    expected = center_mean(converged, 64, 64).mean();
        const float error = std::abs(actual - expected) / std::max(expected, 1e-5f);
        results.push_back({{"change", name},
                           {"mean_after_8_sweeps", actual},
                           {"converged_mean", expected},
                           {"relative_error", error}});
        std::cout << "DDGI dynamic " << name << " actual=" << actual << " expected=" << expected
                  << " relative_error=" << error << '\n';
        capture((std::string("dynamic-") + name + ".png").c_str(), updated, 64, 64);
    }
    metrics("dynamic.json", {{"sweeps", 8}, {"frames_per_sweep", 4}, {"results", results}});
    for (const auto &result : results)
        RENDER_CHECK(result.at("relative_error").get<float>() < .10f);
}

RENDER_TEST(test_ddgi_lifecycle_and_unavailable_device) {
    GlContext context;
    RenderSettings settings;
    settings.width = settings.height = 32;
    settings.opengl = small_settings();
    auto snapshot = make_render_scene_snapshot(cube_scene());
    const Camera camera({0, 0, 3}, {0, 0, 0}, Vec3::UnitY(), 45, 1);
    auto backend = gl_backend();
    backend->reset(snapshot, settings);
    std::string reason;
    const bool available = CudaDeviceContext::try_create(0, &reason).has_value();
    if (!available) {
        settle(*backend, snapshot, camera, settings, 3);
        read_output(*backend, false);
        const auto stats = std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi;
        RENDER_CHECK(!stats.active);
        RENDER_CHECK(stats.status == "unavailable");
        RENDER_CHECK(!stats.detail.empty());
        return;
    }
    settle(*backend, snapshot, camera, settings, 4);
    const auto initial = std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi;
    settings.opengl.ddgi.paused = true;
    settle(*backend, snapshot, camera, settings, 3);
    RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.updated_probes == 0);
    settings.opengl.ddgi.paused = false;
    backend->render(snapshot, camera, settings, tick());
    RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.updated_probes == 27);
    settings.opengl.ddgi.enabled = false;
    backend->render(snapshot, camera, settings, tick());
    read_output(*backend, false);
    RENDER_CHECK(!std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.active);
    settings.opengl.ddgi.enabled = true;
    backend->render(snapshot, camera, settings, tick());
    read_output(*backend);
    RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.reset_count ==
                 initial.reset_count);
    ++settings.opengl.ddgi.reset_generation;
    backend->render(snapshot, camera, settings, tick());
    RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.reset_count ==
                 initial.reset_count + 1);
    settings.opengl.ddgi.probe_counts = {4, 3, 3};
    backend->render(snapshot, camera, settings, tick());
    RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.probe_count == 36);
    snapshot.source_id = allocate_render_scene_source_id();
    backend->render(snapshot, camera, settings, tick());
    RENDER_CHECK(std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.reset_count ==
                 initial.reset_count + 3);
    settings.opengl.npr.style = OpenGlRenderStyle::Toon;
    backend->render(snapshot, camera, settings, tick());
    read_output(*backend, false);
    RENDER_CHECK(!std::get<OpenGlViewerStatistics>(backend->statistics()).ddgi.active);
}
