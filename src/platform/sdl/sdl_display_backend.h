#pragma once

#include "render/framebuffer.h"

#include <string>

struct SDL_Window;

namespace renderer {

struct InputState {
    bool quit_requested = false;
    bool window_resized = false;
    int window_width = 0;
    int window_height = 0;

    bool left_mouse_down = false;
    double mouse_delta_x = 0.0;
    double mouse_delta_y = 0.0;
    double wheel_delta = 0.0;

    bool select_raster = false;
    bool select_ray = false;
    bool select_path = false;
    bool reset_render = false;
};

class SdlDisplayBackend {
public:
    SdlDisplayBackend();
    ~SdlDisplayBackend();

    SdlDisplayBackend(const SdlDisplayBackend&) = delete;
    SdlDisplayBackend& operator=(const SdlDisplayBackend&) = delete;

    bool initialize(int width, int height, const char* title);
    InputState poll_input();
    void present(const Framebuffer& framebuffer);
    void set_title(const std::string& title);
    const std::string& last_error() const;

private:
    SDL_Window* window_ = nullptr;
    bool sdl_initialized_ = false;
    bool left_mouse_down_ = false;
    int window_width_ = 0;
    int window_height_ = 0;
    std::string last_error_;

    void set_error_from_sdl(const char* prefix);
};

}  // namespace renderer
