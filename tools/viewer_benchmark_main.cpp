#define SDL_MAIN_HANDLED

#include "benchmark/benchmark_framework.h"
#include "interactive/viewer_session.h"
#include "render/interactive/viewer_render_backend.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "scene/scene_document.h"

#if !defined(RENDERER_BENCHMARK_DIAGNOSTICS)
#include <SDL3/SDL.h>
#include <glad/gl.h>
#endif

#if RENDERER_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using renderer::benchmark::CaseConfig;
using renderer::benchmark::Metric;
using renderer::benchmark::PhaseResult;
using renderer::benchmark::Report;

struct Options {
    std::filesystem::path case_path =
        std::filesystem::path("benchmarks") / "cases" /
        "san_miguel_first_scene.json";
    std::string backend = "all";
    std::string profile = "full";
    std::string metrics = "timing";
    std::filesystem::path output_directory;
    std::filesystem::path baseline;
    std::string git_commit = "unknown";
    bool git_dirty = false;
    std::filesystem::path import_session;
    std::filesystem::path write_case;
};

std::string require_value(
    int argc,
    char** argv,
    int& index,
    const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(flag + " requires a value");
    }
    return argv[++index];
}

void print_help() {
    std::cout
        << "Viewer benchmark\n\n"
        << "  viewer_benchmark [--case file] [--backend all|opengl|cuda]\n"
        << "                   [--profile full] [--metrics timing]\n"
        << "                   [--output-dir path] [--baseline raw.json]\n"
        << "  viewer_benchmark --import-session last-session.json --write-case case.json\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    options.backend = "cuda";
    options.metrics = "rays,bounces";
#endif
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_help();
            std::exit(0);
        }
        if (argument == "--case") {
            options.case_path = require_value(argc, argv, index, argument);
        } else if (argument == "--backend") {
            options.backend = require_value(argc, argv, index, argument);
        } else if (argument == "--profile") {
            options.profile = require_value(argc, argv, index, argument);
        } else if (argument == "--metrics") {
            options.metrics = require_value(argc, argv, index, argument);
        } else if (argument == "--output-dir") {
            options.output_directory = require_value(argc, argv, index, argument);
        } else if (argument == "--baseline") {
            options.baseline = require_value(argc, argv, index, argument);
        } else if (argument == "--git-commit") {
            options.git_commit = require_value(argc, argv, index, argument);
        } else if (argument == "--git-dirty") {
            options.git_dirty = true;
        } else if (argument == "--import-session") {
            options.import_session = require_value(argc, argv, index, argument);
        } else if (argument == "--write-case") {
            options.write_case = require_value(argc, argv, index, argument);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.profile != "full") {
        throw std::invalid_argument("--profile must be full");
    }
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    if (options.backend != "cuda") {
        throw std::invalid_argument("diagnostics benchmark only supports --backend cuda");
    }
#else
    if (options.backend != "all" && options.backend != "opengl" &&
        options.backend != "cuda") {
        throw std::invalid_argument("--backend must be all, opengl, or cuda");
    }
    if (options.metrics != "timing") {
        throw std::invalid_argument(
            "timing benchmark only supports --metrics timing; use the diagnostics executable for counters");
    }
#endif
    if (options.import_session.empty() != options.write_case.empty()) {
        throw std::invalid_argument(
            "--import-session and --write-case must be provided together");
    }
    return options;
}

const char* tone_mapper_name(renderer::ToneMapper tone_mapper) {
    switch (tone_mapper) {
    case renderer::ToneMapper::None:
        return "none";
    case renderer::ToneMapper::Reinhard:
        return "reinhard";
    case renderer::ToneMapper::Aces:
        return "aces";
    }
    return "none";
}

nlohmann::json vec3_json(const renderer::Vec3& value) {
    return nlohmann::json::array({value.x(), value.y(), value.z()});
}

void import_session_as_case(
    const std::filesystem::path& session_path,
    const std::filesystem::path& descriptor_path) {
    renderer::ViewerSessionState state =
        renderer::ViewerSessionStore::load(session_path);
    const std::filesystem::path absolute_descriptor =
        std::filesystem::absolute(descriptor_path).lexically_normal();
    std::filesystem::create_directories(absolute_descriptor.parent_path());
    const std::filesystem::path scene_path =
        absolute_descriptor.parent_path() /
        (absolute_descriptor.stem().string() + ".rscene");
    state.document.save(scene_path);
    nlohmann::json descriptor{
        {"schema_version", 1},
        {"name", absolute_descriptor.stem().string()},
        {"description", "Explicitly imported from " + session_path.generic_string()},
        {"scene", scene_path.filename().generic_string()},
        {"output", {
            {"width", state.window_width},
            {"height", state.window_height},
            {"render_scale", state.ui.render_scale},
        }},
        {"camera", {
            {"eye", vec3_json(state.camera.eye)},
            {"forward", vec3_json(state.camera.forward)},
            {"up", vec3_json(state.camera.up)},
            {"vertical_fov_degrees", state.camera.vertical_fov_degrees},
        }},
        {"display", {
            {"exposure_ev", state.ui.display.exposure_ev},
            {"tone_mapper", tone_mapper_name(state.ui.display.tone_mapper)},
        }},
        {"render_settings", {
            {"path", {
            {"samples_per_pixel", 1},
            {"cuda_device", state.render_settings.path.cuda_device},
            {"rr_start_bounce", state.render_settings.path.russian_roulette_start_bounce},
            {"rr_min_probability", state.render_settings.path.russian_roulette_min_probability},
            {"rr_max_probability", state.render_settings.path.russian_roulette_max_probability},
            {"sample_seed_offset", state.render_settings.path.sample_seed_offset},
            {"backend", "cuda"},
            }},
            {"opengl", {{"ssr", {
                {"enabled", state.render_settings.opengl.ssr.enabled},
                {"rays_per_pixel", state.render_settings.opengl.ssr.rays_per_pixel},
                {"max_steps", state.render_settings.opengl.ssr.max_steps},
                {"max_distance_scale", state.render_settings.opengl.ssr.max_distance_scale},
                {"thickness_scale", state.render_settings.opengl.ssr.thickness_scale},
                {"edge_fade", state.render_settings.opengl.ssr.edge_fade},
                {"max_history_frames", state.render_settings.opengl.ssr.max_history_frames},
                {"denoise_passes", state.render_settings.opengl.ssr.denoise_passes},
                {"denoise_depth_sigma_fraction", state.render_settings.opengl.ssr.denoise_depth_sigma_fraction},
                {"denoise_normal_power", state.render_settings.opengl.ssr.denoise_normal_power},
            }}}},
        }},
        {"phases", {
            {"opengl_warmup_frames", 60},
            {"opengl_measure_frames", 300},
            {"cuda_full_warmup_frames", 1},
            {"cuda_full_measure_frames", 5},
            {"cuda_interaction_frames", 30},
            {"cuda_native_sweeps", 10},
            {"diagnostic_samples", 1},
        }},
    };
    std::ofstream output(absolute_descriptor);
    if (!output) {
        throw std::runtime_error("failed to write imported benchmark case");
    }
    output << std::setw(2) << descriptor << '\n';
    std::cout << "benchmark case written to "
              << absolute_descriptor.string() << '\n';
}

double elapsed_milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

Metric scalar_metric(
    std::string name,
    std::string unit,
    double value,
    bool instrumented = false) {
    return Metric{
        std::move(name),
        std::move(unit),
        instrumented,
        {value}};
}

Metric series_metric(
    std::string name,
    std::string unit,
    std::vector<double> values,
    bool instrumented = false) {
    return Metric{
        std::move(name),
        std::move(unit),
        instrumented,
        std::move(values)};
}

std::string compiler_name() {
#ifdef _MSC_VER
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#else
    return "unknown";
#endif
}

nlohmann::json system_metadata() {
    nlohmann::json result{
        {"compiler", compiler_name()},
        {"build_type", RENDERER_BUILD_TYPE},
        {"logical_cpu_threads", std::thread::hardware_concurrency()},
    };
#ifdef _WIN32
    result["os"] = "Windows";
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) {
        result["physical_memory_bytes"] = memory.ullTotalPhys;
    }
    HKEY key = nullptr;
    if (RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0,
            KEY_READ,
            &key) == ERROR_SUCCESS) {
        wchar_t name[256]{};
        DWORD bytes = sizeof(name);
        if (RegQueryValueExW(
                key,
                L"ProcessorNameString",
                nullptr,
                nullptr,
                reinterpret_cast<LPBYTE>(name),
                &bytes) == ERROR_SUCCESS) {
            std::wstring wide(name);
            result["cpu"] = std::string(wide.begin(), wide.end());
        }
        RegCloseKey(key);
    }
#else
    result["os"] = "non-Windows";
#endif
#if RENDERER_HAS_CUDA
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        result["cuda_device"] = properties.name;
        result["cuda_compute_capability"] =
            std::to_string(properties.major) + "." +
            std::to_string(properties.minor);
        result["cuda_global_memory_bytes"] = properties.totalGlobalMem;
    }
    int runtime = 0;
    int driver = 0;
    if (cudaRuntimeGetVersion(&runtime) == cudaSuccess) {
        result["cuda_runtime_version"] = runtime;
    }
    if (cudaDriverGetVersion(&driver) == cudaSuccess) {
        result["cuda_driver_version"] = driver;
    }
#endif
    return result;
}

nlohmann::json case_metadata(
    const CaseConfig& config,
    const renderer::SceneDocument& document) {
    const renderer::RenderSceneSnapshot& instanced = document.render_scene_snapshot();
    std::size_t triangles = 0;
    std::size_t materials = 0;
    std::size_t textures = 0;
    for (const renderer::RenderSceneAssetSnapshot& asset : instanced.assets) {
        if (!asset.local_scene) {
            continue;
        }
        triangles += asset.local_scene->triangles.size();
        materials += asset.local_scene->materials.size();
        textures += asset.local_scene->textures.size();
    }
    nlohmann::json result = config.source;
    result["resolved_scene"] = config.scene_path.generic_string();
    result["output"]["effective_width"] = config.render_settings.width;
    result["output"]["effective_height"] = config.render_settings.height;
    result["scene_counts"] = {
        {"unique_assets", instanced.assets.size()},
        {"instances", instanced.instances.size()},
        {"triangles", triangles},
        {"materials", materials},
        {"textures", textures},
        {"point_lights", instanced.point_lights.size()},
        {"directional_lights", instanced.directional_lights.size()},
        {"spot_lights", instanced.spot_lights.size()},
    };
    return result;
}

std::string make_compatibility_key(const Report& report) {
    nlohmann::json comparable_case = report.case_metadata;
    comparable_case.erase("resolved_scene");
    comparable_case.erase("description");
    nlohmann::json identity{
        {"schema_version", report.schema_version},
        {"kind", report.kind},
        {"case", std::move(comparable_case)},
        {"inputs", nlohmann::json::array()},
        {"cpu", report.metadata.value("cpu", std::string())},
        {"cuda_device", report.metadata.value("cuda_device", std::string())},
        {"opengl_renderer", report.metadata.value("opengl_renderer", std::string())},
    };
    for (const auto& input : report.inputs) {
        identity["inputs"].push_back({input.path, input.sha256});
    }
    return renderer::benchmark::sha256_text(identity.dump());
}

std::filesystem::path default_output_directory(
    const std::filesystem::path& source_root,
    const std::string& kind) {
    std::string stamp = renderer::benchmark::utc_timestamp();
    std::replace(stamp.begin(), stamp.end(), ':', '-');
    return source_root / "benchmarks" / "results" /
        (stamp + "-" + kind);
}

#if !defined(RENDERER_BENCHMARK_DIAGNOSTICS)

class HiddenGlContext {
public:
    HiddenGlContext() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(
                std::string("SDL_Init failed: ") + SDL_GetError());
        }
        initialized_ = true;
        if (!SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4) ||
            !SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5) ||
            !SDL_GL_SetAttribute(
                SDL_GL_CONTEXT_PROFILE_MASK,
                SDL_GL_CONTEXT_PROFILE_CORE) ||
            !SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0) ||
            !SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24)) {
            throw std::runtime_error(
                std::string("SDL_GL_SetAttribute failed: ") + SDL_GetError());
        }
        window_ = SDL_CreateWindow(
            "3D Renderer Benchmark",
            64,
            64,
            SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        if (!window_) {
            throw std::runtime_error(
                std::string("SDL_CreateWindow failed: ") + SDL_GetError());
        }
        context_ = SDL_GL_CreateContext(window_);
        if (!context_ || !SDL_GL_MakeCurrent(window_, context_)) {
            throw std::runtime_error(
                std::string("SDL OpenGL context creation failed: ") + SDL_GetError());
        }
        const int loaded = gladLoadGL(
            reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress));
        if (loaded == 0 || GLAD_VERSION_MAJOR(loaded) < 4 ||
            (GLAD_VERSION_MAJOR(loaded) == 4 && GLAD_VERSION_MINOR(loaded) < 5)) {
            throw std::runtime_error("OpenGL 4.5 Core is required for benchmarking");
        }
        SDL_GL_SetSwapInterval(0);
    }

    ~HiddenGlContext() {
        if (context_) {
            SDL_GL_DestroyContext(context_);
        }
        if (window_) {
            SDL_DestroyWindow(window_);
        }
        if (initialized_) {
            SDL_Quit();
        }
    }

private:
    bool initialized_ = false;
    SDL_Window* window_ = nullptr;
    SDL_GLContext context_ = nullptr;
};

void add_opengl_metadata(Report& report) {
    const auto text = [](GLenum name) {
        const auto* value = glGetString(name);
        return value ? reinterpret_cast<const char*>(value) : "unknown";
    };
    report.metadata["opengl_vendor"] = text(GL_VENDOR);
    report.metadata["opengl_renderer"] = text(GL_RENDERER);
    report.metadata["opengl_version"] = text(GL_VERSION);
}

void run_opengl(
    Report& report,
    const CaseConfig& config,
    const renderer::RenderSceneSnapshot& snapshot,
    const std::filesystem::path& source_root) {
    auto backend = renderer::make_viewer_render_backend(
        renderer::InteractiveRenderMode::OpenGl,
        source_root / "shaders/opengl/raster.vert",
        source_root / "shaders/opengl/raster.frag");
    const auto prepare_start = std::chrono::steady_clock::now();
    backend->reset(snapshot, config.render_settings);
    PhaseResult prepare{"backend_prepare", "opengl"};
    prepare.metrics.push_back(scalar_metric(
        "opengl.backend_prepare.wall_ms",
        "ms",
        elapsed_milliseconds(prepare_start)));
    report.phases.push_back(std::move(prepare));

    renderer::InteractiveFrameState frame;
    frame.camera_changed = true;
    GLuint first_query = 0;
    glGenQueries(1, &first_query);
    const auto first_start = std::chrono::steady_clock::now();
    glBeginQuery(GL_TIME_ELAPSED, first_query);
    backend->render(snapshot, config.camera, config.render_settings, frame);
    glEndQuery(GL_TIME_ELAPSED);
    GLuint64 first_nanoseconds = 0;
    glGetQueryObjectui64v(first_query, GL_QUERY_RESULT, &first_nanoseconds);
    glDeleteQueries(1, &first_query);
    PhaseResult first{"first_frame", "opengl"};
    first.metrics.push_back(scalar_metric(
        "opengl.first_frame.wall_ms",
        "ms",
        elapsed_milliseconds(first_start)));
    first.metrics.push_back(scalar_metric(
        "opengl.first_frame.gpu_ms",
        "ms",
        static_cast<double>(first_nanoseconds) / 1.0e6));
    report.phases.push_back(std::move(first));

    frame.camera_changed = false;
    for (int index = 0; index < config.opengl_warmup_frames; ++index) {
        backend->render(snapshot, config.camera, config.render_settings, frame);
    }
    glFinish();
    std::vector<GLuint> queries(
        static_cast<std::size_t>(config.opengl_measure_frames));
    glGenQueries(config.opengl_measure_frames, queries.data());
    std::vector<double> wall_samples;
    wall_samples.reserve(queries.size());
    for (GLuint query : queries) {
        const auto start = std::chrono::steady_clock::now();
        glBeginQuery(GL_TIME_ELAPSED, query);
        backend->render(snapshot, config.camera, config.render_settings, frame);
        glEndQuery(GL_TIME_ELAPSED);
        wall_samples.push_back(elapsed_milliseconds(start));
    }
    std::vector<double> gpu_samples;
    gpu_samples.reserve(queries.size());
    for (GLuint query : queries) {
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
        gpu_samples.push_back(static_cast<double>(nanoseconds) / 1.0e6);
    }
    glDeleteQueries(config.opengl_measure_frames, queries.data());
    PhaseResult steady{"steady_opengl", "opengl"};
    steady.metrics.push_back(series_metric(
        "opengl.frame.wall_ms",
        "ms",
        std::move(wall_samples)));
    steady.metrics.push_back(series_metric(
        "opengl.frame.gpu_ms",
        "ms",
        std::move(gpu_samples)));
    report.phases.push_back(std::move(steady));
}

void append_cuda_statistics(
    PhaseResult& phase,
    const renderer::CudaPathStatistics& statistics) {
    phase.metrics.push_back(scalar_metric(
        "cuda.upload.last_ms", "ms", statistics.upload_milliseconds));
    phase.metrics.push_back(scalar_metric(
        "cuda.upload.cumulative_geometry_bytes", "bytes",
        static_cast<double>(statistics.geometry_upload_bytes)));
    phase.metrics.push_back(scalar_metric(
        "cuda.upload.cumulative_texture_bytes", "bytes",
        static_cast<double>(statistics.texture_upload_bytes)));
    phase.metrics.push_back(scalar_metric(
        "cuda.upload.cumulative_lighting_bytes", "bytes",
        static_cast<double>(statistics.lighting_upload_bytes)));
    phase.metrics.push_back(scalar_metric(
        "cuda.blas.build_count", "count",
        static_cast<double>(statistics.blas_build_count)));
    phase.metrics.push_back(scalar_metric(
        "cuda.tlas.build_count", "count",
        static_cast<double>(statistics.tlas_build_count)));
    phase.metrics.push_back(scalar_metric(
        "cuda.framebuffer.downloads", "count",
        static_cast<double>(statistics.framebuffer_downloads)));
}

void synchronize_cuda_benchmark_sample() {
#if RENDERER_HAS_CUDA
    const cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA benchmark synchronization failed: ") +
            cudaGetErrorString(result));
    }
#endif
}

void run_cuda_timing(
    Report& report,
    const CaseConfig& config,
    const renderer::RenderSceneSnapshot& instanced,
    const std::filesystem::path& source_root) {
    std::string unavailable_reason;
    if (!renderer::cuda_path_backend_available(&unavailable_reason)) {
        report.phases.push_back(PhaseResult{
            "cuda", "cuda", "skipped", unavailable_reason});
        return;
    }
    renderer::RenderSettings settings = config.render_settings;
    auto backend = renderer::make_viewer_render_backend(
        renderer::InteractiveRenderMode::Path,
        source_root / "shaders/opengl/raster.vert",
        source_root / "shaders/opengl/raster.frag");
    const auto prepare_start = std::chrono::steady_clock::now();
    backend->reset(instanced, settings);
    PhaseResult prepare{"backend_prepare", "cuda"};
    prepare.metrics.push_back(scalar_metric(
        "cuda.backend_prepare.wall_ms",
        "ms",
        elapsed_milliseconds(prepare_start)));
    append_cuda_statistics(
        prepare,
        std::get<renderer::CudaPathViewerStatistics>(backend->statistics()).cuda);
    report.phases.push_back(std::move(prepare));

    renderer::InteractiveFrameState frame;
    frame.camera_changed = true;
    frame.automatic_interaction_quality = false;
    const auto first_start = std::chrono::steady_clock::now();
    backend->render(instanced, config.camera, settings, frame);
    synchronize_cuda_benchmark_sample();
    const double first_wall = elapsed_milliseconds(first_start);
    auto statistics =
        std::get<renderer::CudaPathViewerStatistics>(backend->statistics()).cuda;
    PhaseResult first{"first_frame", "cuda"};
    first.metrics.push_back(scalar_metric(
        "cuda.first_frame.wall_ms", "ms", first_wall));
    first.metrics.push_back(scalar_metric(
        "cuda.first_frame.trace_ms", "ms", statistics.trace_milliseconds));
    report.phases.push_back(std::move(first));

    frame.camera_changed = false;
    for (int index = 0; index < config.cuda_full_warmup_frames; ++index) {
        backend->render(instanced, config.camera, settings, frame);
        synchronize_cuda_benchmark_sample();
    }
    std::vector<double> full_wall;
    std::vector<double> full_trace;
    std::vector<double> full_present;
    for (int index = 0; index < config.cuda_full_measure_frames; ++index) {
        const auto start = std::chrono::steady_clock::now();
        backend->render(instanced, config.camera, settings, frame);
        synchronize_cuda_benchmark_sample();
        full_wall.push_back(elapsed_milliseconds(start));
        statistics = std::get<renderer::CudaPathViewerStatistics>(
            backend->statistics()).cuda;
        full_trace.push_back(statistics.trace_milliseconds);
        full_present.push_back(statistics.presentation_milliseconds);
    }
    PhaseResult full{"cuda_full", "cuda"};
    full.metrics.push_back(series_metric(
        "cuda.frame.wall_ms", "ms", std::move(full_wall)));
    full.metrics.push_back(series_metric(
        "cuda.frame.trace_ms", "ms", std::move(full_trace)));
    full.metrics.push_back(series_metric(
        "cuda.frame.presentation_ms", "ms", std::move(full_present)));
    report.phases.push_back(std::move(full));

    backend->reset(instanced, settings);
    frame.automatic_interaction_quality = true;
    std::vector<double> interaction_wall;
    std::vector<double> interaction_trace;
    std::vector<double> interaction_pixels;
    for (int index = 0; index < config.cuda_interaction_frames; ++index) {
        frame.camera_changed = true;
        const auto start = std::chrono::steady_clock::now();
        backend->render(instanced, config.camera, settings, frame);
        synchronize_cuda_benchmark_sample();
        const double wall = elapsed_milliseconds(start);
        statistics = std::get<renderer::CudaPathViewerStatistics>(
            backend->statistics()).cuda;
        if (index > 0) {
            interaction_wall.push_back(wall);
            interaction_trace.push_back(statistics.trace_milliseconds);
            interaction_pixels.push_back(
                static_cast<double>(statistics.internal_width) *
                static_cast<double>(statistics.internal_height));
        }
    }
    PhaseResult interaction{"cuda_interaction", "cuda"};
    interaction.metrics.push_back(series_metric(
        "cuda.interaction.wall_ms", "ms", std::move(interaction_wall)));
    interaction.metrics.push_back(series_metric(
        "cuda.interaction.trace_ms", "ms", std::move(interaction_trace)));
    interaction.metrics.push_back(series_metric(
        "cuda.interaction.internal_pixels", "pixels", std::move(interaction_pixels)));
    report.phases.push_back(std::move(interaction));

    backend->reset(instanced, settings);
    frame.camera_changed = true;
    backend->render(instanced, config.camera, settings, frame);
    synchronize_cuda_benchmark_sample();
    frame.camera_changed = false;
    std::vector<double> native_wall;
    std::vector<double> native_trace;
    std::vector<double> sweep_rates;
    int completed_sweeps = 0;
    int previous_samples = std::get<renderer::CudaPathViewerStatistics>(
        backend->statistics()).accumulated_samples;
    constexpr int max_frames = 100000;
    for (int frame_index = 0;
         frame_index < max_frames && completed_sweeps < config.cuda_native_sweeps;
         ++frame_index) {
        const auto start = std::chrono::steady_clock::now();
        backend->render(instanced, config.camera, settings, frame);
        synchronize_cuda_benchmark_sample();
        const double wall = elapsed_milliseconds(start);
        const auto backend_statistics =
            std::get<renderer::CudaPathViewerStatistics>(
                backend->statistics());
        statistics = backend_statistics.cuda;
        if (statistics.work_mode == renderer::CudaPathWorkMode::NativeTile) {
            native_wall.push_back(wall);
            native_trace.push_back(statistics.trace_milliseconds);
            if (statistics.complete_sweeps_per_second > 0.0f) {
                sweep_rates.push_back(statistics.complete_sweeps_per_second);
            }
        }
        if (backend_statistics.accumulated_samples > previous_samples) {
            completed_sweeps +=
                backend_statistics.accumulated_samples - previous_samples;
            previous_samples = backend_statistics.accumulated_samples;
        }
    }
    if (completed_sweeps < config.cuda_native_sweeps) {
        throw std::runtime_error("CUDA native sweep phase exceeded its frame guard");
    }
    PhaseResult native{"cuda_native_sweep", "cuda"};
    native.metrics.push_back(series_metric(
        "cuda.native_quantum.wall_ms", "ms", std::move(native_wall)));
    native.metrics.push_back(series_metric(
        "cuda.native_quantum.trace_ms", "ms", std::move(native_trace)));
    if (!sweep_rates.empty()) {
        native.metrics.push_back(series_metric(
            "cuda.native.complete_sweeps_per_second",
            "sweeps/s",
            std::move(sweep_rates)));
    }
    native.metrics.push_back(scalar_metric(
        "cuda.native.completed_sweeps", "count", completed_sweeps));
    append_cuda_statistics(native, statistics);
    report.phases.push_back(std::move(native));
}

#endif

Report create_report(
    const Options& options,
    const CaseConfig& config,
    const renderer::SceneDocument& document,
    const std::filesystem::path& source_root) {
    Report report;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    report.kind = "diagnostics";
#else
    report.kind = "timing";
#endif
    report.generated_at_utc = renderer::benchmark::utc_timestamp();
    report.metadata = system_metadata();
    report.metadata["profile"] = options.profile;
    report.metadata["requested_backend"] = options.backend;
    report.metadata["requested_metrics"] = options.metrics;
    report.metadata["git_commit"] = options.git_commit;
    report.metadata["git_dirty"] = options.git_dirty;
    report.case_metadata = case_metadata(config, document);
    report.inputs = renderer::benchmark::fingerprint_case_inputs(config, source_root);
    return report;
}

#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)

void run_diagnostics(
    Report& report,
    const CaseConfig& config,
    const renderer::RenderSceneSnapshot& instanced) {
    std::string unavailable_reason;
    if (!renderer::cuda_path_backend_available(&unavailable_reason)) {
        report.phases.push_back(PhaseResult{
            "cuda_diagnostics", "cuda", "skipped", unavailable_reason});
        return;
    }
    renderer::RenderSettings settings = config.render_settings;
    renderer::CudaPathInteractiveRenderer renderer_instance;
    renderer_instance.reset(instanced, settings);
    renderer::Framebuffer framebuffer(settings.width, settings.height);
    renderer::InteractiveFrameState frame;
    frame.automatic_interaction_quality = false;
    for (int index = 0; index < config.diagnostic_samples; ++index) {
        frame.camera_changed = index == 0;
        renderer_instance.render_next_frame(
            instanced,
            config.camera,
            settings,
            frame,
            framebuffer);
    }
    const renderer::CudaPathDiagnosticProfile profile =
        renderer_instance.download_diagnostic_profile();
    const auto count_metric = [](const char* name, std::uint64_t value) {
        return scalar_metric(
            name,
            "rays",
            static_cast<double>(value),
            true);
    };
    PhaseResult phase{"cuda_diagnostics", "cuda"};
    phase.metrics.push_back(count_metric("cuda.rays.primary", profile.primary_rays));
    phase.metrics.push_back(count_metric("cuda.rays.continuation", profile.continuation_rays));
    phase.metrics.push_back(count_metric("cuda.rays.shadow.directional", profile.directional_shadow_rays));
    phase.metrics.push_back(count_metric("cuda.rays.shadow.point", profile.point_shadow_rays));
    phase.metrics.push_back(count_metric("cuda.rays.shadow.spot", profile.spot_shadow_rays));
    phase.metrics.push_back(count_metric("cuda.rays.shadow.emissive", profile.emissive_shadow_rays));
    phase.metrics.push_back(count_metric("cuda.rays.shadow.environment", profile.environment_shadow_rays));
    const std::uint64_t total =
        profile.primary_rays + profile.continuation_rays +
        profile.directional_shadow_rays + profile.point_shadow_rays +
        profile.spot_shadow_rays + profile.emissive_shadow_rays +
        profile.environment_shadow_rays;
    phase.metrics.push_back(count_metric("cuda.rays.total", total));
    phase.metrics.push_back(scalar_metric(
        "cuda.rays.per_camera_sample",
        "rays/sample",
        profile.primary_rays == 0
            ? 0.0
            : static_cast<double>(total) /
                static_cast<double>(profile.primary_rays),
        true));
    phase.metrics.push_back(scalar_metric(
        "cuda.primary.hits", "paths",
        static_cast<double>(profile.primary_hits), true));
    phase.metrics.push_back(scalar_metric(
        "cuda.primary.misses", "paths",
        static_cast<double>(profile.primary_misses), true));
    nlohmann::json rays_by_bounce = nlohmann::json::array();
    nlohmann::json termination = nlohmann::json::array();
    for (std::size_t bounce = 0; bounce < profile.rays_by_bounce.size(); ++bounce) {
        rays_by_bounce.push_back({
            {"bounce", bounce},
            {"rays", profile.rays_by_bounce[bounce]},
        });
        termination.push_back({
            {"bounce", bounce},
            {"paths", profile.termination_by_bounce[bounce]},
        });
    }
    Metric ray_histogram;
    ray_histogram.name = "cuda.rays.by_bounce";
    ray_histogram.unit = "rays";
    ray_histogram.instrumented = true;
    ray_histogram.histogram = std::move(rays_by_bounce);
    phase.metrics.push_back(std::move(ray_histogram));
    Metric termination_histogram;
    termination_histogram.name = "cuda.paths.bounce_histogram";
    termination_histogram.unit = "paths";
    termination_histogram.instrumented = true;
    termination_histogram.histogram = std::move(termination);
    phase.metrics.push_back(std::move(termination_histogram));
    report.phases.push_back(std::move(phase));
}

#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!options.import_session.empty()) {
            import_session_as_case(options.import_session, options.write_case);
            return 0;
        }
        const std::filesystem::path source_root(RENDERER_SOURCE_DIR);
        const CaseConfig config = renderer::benchmark::load_case(
            options.case_path,
            source_root);
        const auto load_start = std::chrono::steady_clock::now();
        renderer::SceneDocument document = renderer::SceneDocument::load(
            config.scene_path,
            config.width,
            config.height);
        const double load_milliseconds = elapsed_milliseconds(load_start);
        Report report = create_report(options, config, document, source_root);
        PhaseResult load{"asset_load", "shared"};
        load.metrics.push_back(scalar_metric(
            "scene.asset_load.wall_ms", "ms", load_milliseconds));
        report.phases.push_back(std::move(load));

#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        run_diagnostics(report, config, document.render_scene_snapshot());
#else
        HiddenGlContext context;
        add_opengl_metadata(report);
        if (options.backend == "all" || options.backend == "opengl") {
            run_opengl(
                report,
                config,
                document.render_scene_snapshot(),
                source_root);
        }
        if (options.backend == "all" || options.backend == "cuda") {
            run_cuda_timing(
                report,
                config,
                document.render_scene_snapshot(),
                source_root);
        }
#endif
        report.compatibility_key = make_compatibility_key(report);
        if (!options.baseline.empty()) {
            renderer::benchmark::compare_with_baseline(report, options.baseline);
        }
        const std::filesystem::path output_directory =
            options.output_directory.empty()
            ? default_output_directory(source_root, report.kind)
            : options.output_directory;
        renderer::benchmark::write_report_files(report, output_directory);
        std::cout << "benchmark kind=" << report.kind
                  << " case=" << config.name
                  << " output=" << output_directory.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        std::cerr << "viewer_benchmark_diagnostics: ";
#else
        std::cerr << "viewer_benchmark: ";
#endif
        std::cerr << error.what() << '\n';
        return 1;
    }
}
