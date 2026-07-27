#define SDL_MAIN_HANDLED
#include "platform/sdl/sdl_display_backend.h"

#include "platform/opengl/gl_shader_program.h"

#include <SDL3/SDL.h>
#include <glad/gl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace renderer {

struct SdlDisplayBackend::DialogInbox {
    std::mutex mutex;
    std::vector<FileDialogResult> results;
    std::atomic_bool open = false;
};

struct SdlDisplayBackend::DialogCallbackData {
    std::weak_ptr<DialogInbox> inbox;
    FileDialogKind kind = FileDialogKind::ImportFiles;
};

namespace {

constexpr const char* compositor_vertex_shader = R"GLSL(#version 450 core
out vec2 v_uv;

void main() {
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0));
    const vec2 position = positions[gl_VertexID];
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
)GLSL";

constexpr const char* compositor_fragment_shader = R"GLSL(#version 450 core
layout(binding = 0) uniform sampler2D u_linear_image;
uniform float u_exposure_ev;
uniform int u_tone_mapper;
uniform int u_flip_y;

in vec2 v_uv;
layout(location = 0) out vec4 out_color;

float sanitize_channel(float value) {
    if (isnan(value)) {
        return 0.0;
    }
    if (isinf(value)) {
        return value > 0.0 ? 1.0e20 : 0.0;
    }
    return value;
}

vec3 sanitize_color(vec3 color) {
    return vec3(
        sanitize_channel(color.r),
        sanitize_channel(color.g),
        sanitize_channel(color.b));
}

vec3 reinhard(vec3 color) {
    return color / (vec3(1.0) + color);
}

vec3 aces(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 color) {
    const bvec3 low = lessThanEqual(color, vec3(0.0031308));
    const vec3 lower = color * 12.92;
    const vec3 upper = 1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(upper, lower, low);
}

void main() {
    vec2 uv = v_uv;
    if (u_flip_y != 0) {
        uv.y = 1.0 - uv.y;
    }
    vec3 color = max(sanitize_color(texture(u_linear_image, uv).rgb), vec3(0.0));
    color *= exp2(u_exposure_ev);
    if (u_tone_mapper == 1) {
        color = reinhard(color);
    } else if (u_tone_mapper == 2) {
        color = aces(color);
    }
    out_color = vec4(clamp(linear_to_srgb(color), 0.0, 1.0), 1.0);
}
)GLSL";

int tone_mapper_value(ToneMapper tone_mapper) {
    if (tone_mapper == ToneMapper::Reinhard) {
        return 1;
    }
    if (tone_mapper == ToneMapper::Aces) {
        return 2;
    }
    return 0;
}

}  // namespace

SdlDisplayBackend::SdlDisplayBackend()
    : compositor_program_(std::make_unique<GlShaderProgram>()),
      dialog_inbox_(std::make_shared<DialogInbox>()) {}

SdlDisplayBackend::~SdlDisplayBackend() {
    dialog_inbox_.reset();
    if (gl_context_ && window_) {
        SDL_GL_MakeCurrent(window_, static_cast<SDL_GLContext>(gl_context_));
    }
    if (imgui_initialized_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        imgui_initialized_ = false;
    }
    release_gl_resources();
    if (gl_context_) {
        SDL_GL_DestroyContext(static_cast<SDL_GLContext>(gl_context_));
        gl_context_ = nullptr;
    }
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    if (sdl_initialized_) {
        SDL_Quit();
        sdl_initialized_ = false;
    }
}

bool SdlDisplayBackend::initialize(int width, int height, const char* title) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        set_error_from_sdl("SDL_Init failed");
        return false;
    }
    sdl_initialized_ = true;

    if (!SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4) ||
        !SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5) ||
        !SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE) ||
        !SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1) ||
        !SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24)) {
        set_error_from_sdl("SDL_GL_SetAttribute failed");
        return false;
    }
#ifndef NDEBUG
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif

    window_ = SDL_CreateWindow(
        title,
        width,
        height,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window_) {
        set_error_from_sdl("SDL_CreateWindow failed");
        return false;
    }
    SDL_SetWindowMinimumSize(window_, 1, 1);

    gl_context_ = SDL_GL_CreateContext(window_);
    if (!gl_context_) {
        set_error_from_sdl("SDL_GL_CreateContext failed");
        return false;
    }
    if (!SDL_GL_MakeCurrent(window_, static_cast<SDL_GLContext>(gl_context_))) {
        set_error_from_sdl("SDL_GL_MakeCurrent failed");
        return false;
    }

    const int loaded_version = gladLoadGL(
        reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress));
    if (loaded_version == 0 ||
        GLAD_VERSION_MAJOR(loaded_version) < 4 ||
        (GLAD_VERSION_MAJOR(loaded_version) == 4 && GLAD_VERSION_MINOR(loaded_version) < 5)) {
        last_error_ = "OpenGL 4.5 Core is required";
        return false;
    }
    SDL_GL_SetSwapInterval(0);

    std::string shader_error;
    if (!compositor_program_->load_sources(
            compositor_vertex_shader,
            compositor_fragment_shader,
            shader_error)) {
        last_error_ = "Display compositor initialization failed: " + shader_error;
        return false;
    }
    glGenVertexArrays(1, &fullscreen_vao_);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.Fonts->AddFontDefaultVector();
    ImGui::StyleColorsDark();
    if (!ImGui_ImplSDL3_InitForOpenGL(
            window_,
            static_cast<SDL_GLContext>(gl_context_))) {
        last_error_ = "ImGui SDL3/OpenGL initialization failed";
        ImGui::DestroyContext();
        return false;
    }
    if (!ImGui_ImplOpenGL3_Init("#version 450 core")) {
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        last_error_ = "ImGui OpenGL initialization failed";
        return false;
    }
    imgui_initialized_ = true;

    if (!SDL_GetWindowSizeInPixels(window_, &window_width_, &window_height_)) {
        window_width_ = std::max(1, width);
        window_height_ = std::max(1, height);
    }
    return true;
}

InputState SdlDisplayBackend::poll_input() {
    InputState input;
    input.window_width = window_width_;
    input.window_height = window_height_;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (imgui_initialized_) {
            ImGui_ImplSDL3_ProcessEvent(&event);
        }
        switch (event.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                input.quit_requested = true;
                break;
            case SDL_EVENT_WINDOW_RESIZED:
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                if (!SDL_GetWindowSizeInPixels(window_, &window_width_, &window_height_)) {
                    window_width_ = std::max(1, event.window.data1);
                    window_height_ = std::max(1, event.window.data2);
                }
                window_width_ = std::max(1, window_width_);
                window_height_ = std::max(1, window_height_);
                input.window_resized = true;
                input.window_width = window_width_;
                input.window_height = window_height_;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    left_mouse_down_ = true;
                    left_mouse_dragged_ = false;
                    input.left_mouse_pressed = true;
                } else if (event.button.button == SDL_BUTTON_RIGHT) {
                    right_mouse_down_ = true;
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    left_mouse_down_ = false;
                    input.left_mouse_released = true;
                    input.left_mouse_clicked = !left_mouse_dragged_;
                    left_mouse_dragged_ = false;
                } else if (event.button.button == SDL_BUTTON_RIGHT) {
                    right_mouse_down_ = false;
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                input.mouse_x = event.motion.x;
                input.mouse_y = event.motion.y;
                if (left_mouse_down_ &&
                    (std::abs(event.motion.xrel) + std::abs(event.motion.yrel) > 0.5f)) {
                    left_mouse_dragged_ = true;
                }
                if (left_mouse_down_ || right_mouse_down_ || relative_mouse_mode_) {
                    input.mouse_delta_x += event.motion.xrel;
                    input.mouse_delta_y += event.motion.yrel;
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                input.wheel_delta += event.wheel.y;
                break;
            case SDL_EVENT_DROP_FILE:
                if (event.drop.data && event.drop.data[0] != '\0') {
                    input.dropped_paths.emplace_back(event.drop.data);
                }
                break;
            case SDL_EVENT_KEY_DOWN:
                if (!event.key.repeat) {
                    if (event.key.scancode == SDL_SCANCODE_1) {
                        input.select_raster = true;
                    } else if (event.key.scancode == SDL_SCANCODE_2) {
                        input.select_ray = true;
                    } else if (event.key.scancode == SDL_SCANCODE_3) {
                        input.select_path = true;
                    } else if (event.key.scancode == SDL_SCANCODE_4) {
                        input.select_opengl = true;
                    } else if (event.key.scancode == SDL_SCANCODE_F5) {
                        input.reload_shaders = true;
                    } else if (
                        event.key.scancode == SDL_SCANCODE_R &&
                        (event.key.mod & SDL_KMOD_CTRL) != 0) {
                        input.reset_render = true;
                    } else if (event.key.scancode == SDL_SCANCODE_C) {
                        input.toggle_camera_mode = true;
                    } else if (event.key.scancode == SDL_SCANCODE_TAB) {
                        input.toggle_ui = true;
                    } else if (event.key.scancode == SDL_SCANCODE_ESCAPE) {
                        input.quit_requested = true;
                    }
                }
                break;
            default:
                break;
        }
    }

    input.left_mouse_down = left_mouse_down_;
    input.right_mouse_down = right_mouse_down_;
    float current_mouse_x = 0.0f;
    float current_mouse_y = 0.0f;
    SDL_GetMouseState(&current_mouse_x, &current_mouse_y);
    input.mouse_x = current_mouse_x;
    input.mouse_y = current_mouse_y;
    if (SDL_GetKeyboardFocus() == window_) {
        const bool* keyboard = SDL_GetKeyboardState(nullptr);
        input.move_forward = keyboard[SDL_SCANCODE_W];
        input.move_backward = keyboard[SDL_SCANCODE_S];
        input.move_left = keyboard[SDL_SCANCODE_A];
        input.move_right = keyboard[SDL_SCANCODE_D];
        input.move_up = keyboard[SDL_SCANCODE_SPACE];
        input.move_down =
            keyboard[SDL_SCANCODE_LSHIFT] || keyboard[SDL_SCANCODE_RSHIFT];
    }
    if (dialog_inbox_) {
        std::scoped_lock lock(dialog_inbox_->mutex);
        input.dialog_results.swap(dialog_inbox_->results);
    }
    return input;
}

void SdlDisplayBackend::begin_ui_frame() {
    if (!imgui_initialized_) {
        throw std::runtime_error("ImGui frame requires an initialized display");
    }
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    ui_frame_started_ = true;
}

bool SdlDisplayBackend::wants_mouse_capture() const {
    return imgui_initialized_ && ImGui::GetIO().WantCaptureMouse;
}

bool SdlDisplayBackend::wants_keyboard_capture() const {
    return imgui_initialized_ && ImGui::GetIO().WantCaptureKeyboard;
}

bool SdlDisplayBackend::set_relative_mouse_mode(bool enabled) {
    if (!window_) {
        last_error_ = "SDL relative mouse mode requires an initialized window";
        return false;
    }
    if (enabled == relative_mouse_mode_) {
        return true;
    }
    if (!SDL_SetWindowRelativeMouseMode(window_, enabled)) {
        set_error_from_sdl("SDL_SetWindowRelativeMouseMode failed");
        return false;
    }
    relative_mouse_mode_ = enabled;
    return true;
}

void SdlDisplayBackend::dialog_callback(
    void* userdata,
    const char* const* filelist,
    int) {
    const std::unique_ptr<DialogCallbackData> callback_data(
        static_cast<DialogCallbackData*>(userdata));
    const std::shared_ptr<DialogInbox> inbox = callback_data->inbox.lock();
    if (!inbox) {
        return;
    }

    FileDialogResult result;
    result.kind = callback_data->kind;
    if (!filelist) {
        const char* error = SDL_GetError();
        result.error = error ? error : "file dialog failed";
    } else {
        for (const char* const* path = filelist; *path; ++path) {
            result.paths.emplace_back(*path);
        }
    }
    {
        std::scoped_lock lock(inbox->mutex);
        inbox->results.push_back(std::move(result));
    }
    inbox->open = false;
}

bool SdlDisplayBackend::show_dialog(
    FileDialogKind kind,
    const std::string& default_location) {
    if (!window_ || !dialog_inbox_) {
        last_error_ = "file dialog requires an initialized window";
        return false;
    }
    bool expected = false;
    if (!dialog_inbox_->open.compare_exchange_strong(expected, true)) {
        last_error_ = "a file dialog is already open";
        return false;
    }

    auto* callback_data = new DialogCallbackData{dialog_inbox_, kind};
    const char* location = default_location.empty() ? nullptr : default_location.c_str();
    static constexpr SDL_DialogFileFilter model_filters[]{
        {"Wavefront OBJ", "obj"},
    };
    static constexpr SDL_DialogFileFilter scene_filters[]{
        {"Renderer scene", "rscene"},
    };
    switch (kind) {
        case FileDialogKind::ImportFiles:
            SDL_ShowOpenFileDialog(
                &SdlDisplayBackend::dialog_callback,
                callback_data,
                window_,
                model_filters,
                1,
                location,
                true);
            break;
        case FileDialogKind::ImportFolder:
            SDL_ShowOpenFolderDialog(
                &SdlDisplayBackend::dialog_callback,
                callback_data,
                window_,
                location,
                false);
            break;
        case FileDialogKind::OpenScene:
            SDL_ShowOpenFileDialog(
                &SdlDisplayBackend::dialog_callback,
                callback_data,
                window_,
                scene_filters,
                1,
                location,
                false);
            break;
        case FileDialogKind::SaveScene:
            SDL_ShowSaveFileDialog(
                &SdlDisplayBackend::dialog_callback,
                callback_data,
                window_,
                scene_filters,
                1,
                location);
            break;
    }
    return true;
}

bool SdlDisplayBackend::show_import_files_dialog(const std::string& default_location) {
    return show_dialog(FileDialogKind::ImportFiles, default_location);
}

bool SdlDisplayBackend::show_import_folder_dialog(const std::string& default_location) {
    return show_dialog(FileDialogKind::ImportFolder, default_location);
}

bool SdlDisplayBackend::show_open_scene_dialog(const std::string& default_location) {
    return show_dialog(FileDialogKind::OpenScene, default_location);
}

bool SdlDisplayBackend::show_save_scene_dialog(const std::string& default_location) {
    return show_dialog(FileDialogKind::SaveScene, default_location);
}

void SdlDisplayBackend::present(
    const Framebuffer& framebuffer,
    const DisplaySettings& display_settings) {
    ensure_framebuffer_texture(framebuffer.width(), framebuffer.height());
    const std::vector<float> rgba = framebuffer.to_rgba32f();
    glBindTexture(GL_TEXTURE_2D, framebuffer_texture_);
    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        framebuffer.width(),
        framebuffer.height(),
        GL_RGBA,
        GL_FLOAT,
        rgba.data());
    present_texture(
        framebuffer_texture_,
        framebuffer.width(),
        framebuffer.height(),
        true,
        display_settings);
}

void SdlDisplayBackend::present_texture(
    unsigned int linear_texture,
    int texture_width,
    int texture_height,
    bool flip_y,
    const DisplaySettings& display_settings) {
    if (!gl_context_ || !ui_frame_started_ || linear_texture == 0 ||
        texture_width <= 0 || texture_height <= 0) {
        throw std::runtime_error(
            "OpenGL present requires an initialized display, active UI frame, and valid texture");
    }

    SDL_GetWindowSizeInPixels(window_, &window_width_, &window_height_);
    window_width_ = std::max(1, window_width_);
    window_height_ = std::max(1, window_height_);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, window_width_, window_height_);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(compositor_program_->id());
    const GLint exposure_location = glGetUniformLocation(
        compositor_program_->id(), "u_exposure_ev");
    const GLint tone_mapper_location = glGetUniformLocation(
        compositor_program_->id(), "u_tone_mapper");
    const GLint flip_location = glGetUniformLocation(compositor_program_->id(), "u_flip_y");
    glUniform1f(
        exposure_location,
        std::isfinite(display_settings.exposure_ev) ? display_settings.exposure_ev : 0.0f);
    glUniform1i(tone_mapper_location, tone_mapper_value(display_settings.tone_mapper));
    glUniform1i(flip_location, flip_y ? 1 : 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, linear_texture);
    glBindVertexArray(fullscreen_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window_);
    ui_frame_started_ = false;
}

void SdlDisplayBackend::set_title(const std::string& title) {
    if (window_ && !SDL_SetWindowTitle(window_, title.c_str())) {
        set_error_from_sdl("SDL_SetWindowTitle failed");
    }
}

const std::string& SdlDisplayBackend::last_error() const {
    return last_error_;
}

void SdlDisplayBackend::set_error_from_sdl(const char* prefix) {
    last_error_ = prefix;
    const char* error = SDL_GetError();
    if (error && std::strlen(error) > 0) {
        last_error_ += ": ";
        last_error_ += error;
    }
}

void SdlDisplayBackend::ensure_framebuffer_texture(int width, int height) {
    if (framebuffer_texture_ != 0 && width == texture_width_ && height == texture_height_) {
        return;
    }
    if (framebuffer_texture_ == 0) {
        glGenTextures(1, &framebuffer_texture_);
    }
    glBindTexture(GL_TEXTURE_2D, framebuffer_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA32F,
        width,
        height,
        0,
        GL_RGBA,
        GL_FLOAT,
        nullptr);
    texture_width_ = width;
    texture_height_ = height;
}

void SdlDisplayBackend::release_gl_resources() {
    if (compositor_program_) {
        compositor_program_->reset();
    }
    if (framebuffer_texture_ != 0) {
        glDeleteTextures(1, &framebuffer_texture_);
        framebuffer_texture_ = 0;
    }
    if (fullscreen_vao_ != 0) {
        glDeleteVertexArrays(1, &fullscreen_vao_);
        fullscreen_vao_ = 0;
    }
}

}  // namespace renderer
