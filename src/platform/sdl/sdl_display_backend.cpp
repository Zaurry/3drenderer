#define SDL_MAIN_HANDLED
#include "platform/sdl/sdl_display_backend.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace renderer {

SdlDisplayBackend::SdlDisplayBackend() = default;

SdlDisplayBackend::~SdlDisplayBackend() {
    if (imgui_initialized_) {
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        imgui_initialized_ = false;
    }
    if (framebuffer_texture_) {
        SDL_DestroyTexture(framebuffer_texture_);
        framebuffer_texture_ = nullptr;
    }
    if (renderer_) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
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

    window_ = SDL_CreateWindow(title, width, height, SDL_WINDOW_RESIZABLE);
    if (!window_) {
        set_error_from_sdl("SDL_CreateWindow failed");
        return false;
    }
    SDL_SetWindowMinimumSize(window_, 1, 1);

    renderer_ = SDL_CreateRenderer(window_, nullptr);
    if (!renderer_) {
        set_error_from_sdl("SDL_CreateRenderer failed");
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.Fonts->AddFontDefaultVector();
    ImGui::StyleColorsDark();
    if (!ImGui_ImplSDL3_InitForSDLRenderer(window_, renderer_)) {
        last_error_ = "ImGui SDL3 initialization failed";
        ImGui::DestroyContext();
        return false;
    }
    if (!ImGui_ImplSDLRenderer3_Init(renderer_)) {
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        last_error_ = "ImGui SDL_Renderer initialization failed";
        return false;
    }
    imgui_initialized_ = true;

    window_width_ = std::max(1, width);
    window_height_ = std::max(1, height);
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
                window_width_ = std::max(1, event.window.data1);
                window_height_ = std::max(1, event.window.data2);
                input.window_resized = true;
                input.window_width = window_width_;
                input.window_height = window_height_;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    left_mouse_down_ = true;
                } else if (event.button.button == SDL_BUTTON_RIGHT) {
                    right_mouse_down_ = true;
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    left_mouse_down_ = false;
                } else if (event.button.button == SDL_BUTTON_RIGHT) {
                    right_mouse_down_ = false;
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (left_mouse_down_ || right_mouse_down_ || relative_mouse_mode_) {
                    input.mouse_delta_x += event.motion.xrel;
                    input.mouse_delta_y += event.motion.yrel;
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                input.wheel_delta += event.wheel.y;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (!event.key.repeat) {
                    if (event.key.scancode == SDL_SCANCODE_1) {
                        input.select_raster = true;
                    } else if (event.key.scancode == SDL_SCANCODE_2) {
                        input.select_ray = true;
                    } else if (event.key.scancode == SDL_SCANCODE_3) {
                        input.select_path = true;
                    } else if (event.key.scancode == SDL_SCANCODE_R) {
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
    return input;
}

void SdlDisplayBackend::begin_ui_frame() {
    if (!imgui_initialized_) {
        throw std::runtime_error("ImGui frame requires an initialized display");
    }
    ImGui_ImplSDLRenderer3_NewFrame();
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

void SdlDisplayBackend::present(
    const Framebuffer& framebuffer,
    const DisplaySettings& display_settings) {
    if (!renderer_ || !ui_frame_started_) {
        throw std::runtime_error("SDL present requires an initialized renderer and active UI frame");
    }

    ensure_framebuffer_texture(framebuffer.width(), framebuffer.height());
    const std::vector<std::uint8_t> rgba = framebuffer.to_rgba8(display_settings);
    if (!SDL_UpdateTexture(
            framebuffer_texture_,
            nullptr,
            rgba.data(),
            framebuffer.width() * 4)) {
        set_error_from_sdl("SDL_UpdateTexture failed");
        throw std::runtime_error(last_error_);
    }

    if (!SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255) || !SDL_RenderClear(renderer_)) {
        set_error_from_sdl("SDL_RenderClear failed");
        throw std::runtime_error(last_error_);
    }
    int output_width = 0;
    int output_height = 0;
    if (!SDL_GetCurrentRenderOutputSize(renderer_, &output_width, &output_height)) {
        set_error_from_sdl("SDL_GetCurrentRenderOutputSize failed");
        throw std::runtime_error(last_error_);
    }
    const SDL_FRect destination{
        0.0f,
        0.0f,
        static_cast<float>(output_width),
        static_cast<float>(output_height)};
    if (!SDL_RenderTexture(renderer_, framebuffer_texture_, nullptr, &destination)) {
        set_error_from_sdl("SDL_RenderTexture failed");
        throw std::runtime_error(last_error_);
    }

    ImGui::Render();
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer_);
    if (!SDL_RenderPresent(renderer_)) {
        set_error_from_sdl("SDL_RenderPresent failed");
        throw std::runtime_error(last_error_);
    }
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
    if (framebuffer_texture_ && width == texture_width_ && height == texture_height_) {
        return;
    }
    if (framebuffer_texture_) {
        SDL_DestroyTexture(framebuffer_texture_);
        framebuffer_texture_ = nullptr;
    }
    framebuffer_texture_ = SDL_CreateTexture(
        renderer_,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height);
    if (!framebuffer_texture_) {
        set_error_from_sdl("SDL_CreateTexture failed");
        throw std::runtime_error(last_error_);
    }
    if (!SDL_SetTextureScaleMode(framebuffer_texture_, SDL_SCALEMODE_LINEAR)) {
        set_error_from_sdl("SDL_SetTextureScaleMode failed");
        throw std::runtime_error(last_error_);
    }
    texture_width_ = width;
    texture_height_ = height;
}

}  // namespace renderer
