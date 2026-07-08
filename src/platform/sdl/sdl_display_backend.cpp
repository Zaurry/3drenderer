#define SDL_MAIN_HANDLED
#include "platform/sdl/sdl_display_backend.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace renderer {

namespace {

bool convert_to_surface(const Framebuffer& framebuffer, SDL_Surface* surface) {
    if (!surface || !surface->pixels) {
        return false;
    }

    const std::vector<std::uint8_t> rgba = framebuffer.to_rgba8();
    const int width = std::min(framebuffer.width(), surface->w);
    const int height = std::min(framebuffer.height(), surface->h);
    return SDL_ConvertPixels(
        width,
        height,
        SDL_PIXELFORMAT_RGBA32,
        rgba.data(),
        framebuffer.width() * 4,
        surface->format,
        surface->pixels,
        surface->pitch);
}

}  // namespace

SdlDisplayBackend::SdlDisplayBackend() = default;

SdlDisplayBackend::~SdlDisplayBackend() {
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
        switch (event.type) {
            case SDL_EVENT_QUIT:
                input.quit_requested = true;
                break;
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
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    left_mouse_down_ = false;
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (left_mouse_down_) {
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
    return input;
}

void SdlDisplayBackend::present(const Framebuffer& framebuffer) {
    SDL_Surface* surface = SDL_GetWindowSurface(window_);
    if (!surface) {
        set_error_from_sdl("SDL_GetWindowSurface failed");
        throw std::runtime_error(last_error_);
    }

    if (!SDL_LockSurface(surface)) {
        set_error_from_sdl("SDL_LockSurface failed");
        throw std::runtime_error(last_error_);
    }
    const bool converted = convert_to_surface(framebuffer, surface);
    SDL_UnlockSurface(surface);
    if (!converted) {
        set_error_from_sdl("SDL_ConvertPixels failed");
        throw std::runtime_error(last_error_);
    }

    if (!SDL_UpdateWindowSurface(window_)) {
        set_error_from_sdl("SDL_UpdateWindowSurface failed");
        throw std::runtime_error(last_error_);
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

}  // namespace renderer
