#pragma once

#include "render/framebuffer.h"

#include <memory>
#include <string>

struct SDL_Window;

namespace renderer {

class GlShaderProgram;

struct InputState {
    bool quit_requested = false;
    bool window_resized = false;
    int window_width = 0;
    int window_height = 0;

    bool left_mouse_down = false;
    bool right_mouse_down = false;
    float mouse_delta_x = 0.0f;
    float mouse_delta_y = 0.0f;
    float wheel_delta = 0.0f;

    bool toggle_camera_mode = false;
    bool move_forward = false;
    bool move_backward = false;
    bool move_left = false;
    bool move_right = false;
    bool move_up = false;
    bool move_down = false;

    bool select_raster = false;
    bool select_ray = false;
    bool select_path = false;
    bool select_opengl = false;
    bool reload_shaders = false;
    bool reset_render = false;
    bool toggle_ui = false;
};

class SdlDisplayBackend {
public:
    SdlDisplayBackend();
    ~SdlDisplayBackend();

    SdlDisplayBackend(const SdlDisplayBackend&) = delete;
    SdlDisplayBackend& operator=(const SdlDisplayBackend&) = delete;

    bool initialize(int width, int height, const char* title);
    InputState poll_input();
    void begin_ui_frame();
    bool wants_mouse_capture() const;
    bool wants_keyboard_capture() const;
    bool set_relative_mouse_mode(bool enabled);
    void present(const Framebuffer& framebuffer, const DisplaySettings& display_settings);
    void present_texture(
        unsigned int linear_texture,
        int texture_width,
        int texture_height,
        bool flip_y,
        const DisplaySettings& display_settings);
    void set_title(const std::string& title);
    const std::string& last_error() const;

private:
    SDL_Window* window_ = nullptr;
    void* gl_context_ = nullptr;
    unsigned int framebuffer_texture_ = 0;
    unsigned int fullscreen_vao_ = 0;
    std::unique_ptr<GlShaderProgram> compositor_program_;
    bool sdl_initialized_ = false;
    bool imgui_initialized_ = false;
    bool ui_frame_started_ = false;
    bool left_mouse_down_ = false;
    bool right_mouse_down_ = false;
    bool relative_mouse_mode_ = false;
    int window_width_ = 0;
    int window_height_ = 0;
    int texture_width_ = 0;
    int texture_height_ = 0;
    std::string last_error_;

    void set_error_from_sdl(const char* prefix);
    void ensure_framebuffer_texture(int width, int height);
    void release_gl_resources();
};

}  // namespace renderer
