#pragma once

#include "render/framebuffer.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct SDL_Window;

namespace renderer {

class GlShaderProgram;

enum class FileDialogKind {
    ImportFiles,
    ImportFolder,
    OpenScene,
    SaveScene,
};

struct FileDialogResult {
    FileDialogKind kind = FileDialogKind::ImportFiles;
    std::vector<std::string> paths;
    std::string error;
};

struct InputState {
    bool quit_requested = false;
    bool window_resized = false;
    int window_width = 0;
    int window_height = 0;

    bool left_mouse_down = false;
    bool left_mouse_pressed = false;
    bool left_mouse_released = false;
    bool left_mouse_clicked = false;
    bool right_mouse_down = false;
    float mouse_x = 0.0f;
    float mouse_y = 0.0f;
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
    std::vector<std::string> dropped_paths;
    std::vector<FileDialogResult> dialog_results;
};

class SdlDisplayBackend {
public:
    SdlDisplayBackend();
    ~SdlDisplayBackend();

    SdlDisplayBackend(const SdlDisplayBackend&) = delete;
    SdlDisplayBackend& operator=(const SdlDisplayBackend&) = delete;

    static std::filesystem::path preferred_session_path();

    bool initialize(int width, int height, const char* title);
    bool constrain_window_to_display();
    std::pair<int, int> logical_window_size() const;
    std::pair<int, int> drawable_size() const;
    InputState poll_input();
    void begin_ui_frame();
    bool wants_mouse_capture() const;
    bool wants_keyboard_capture() const;
    bool main_window_has_keyboard_focus() const;
    bool set_relative_mouse_mode(bool enabled);
    bool show_import_files_dialog(const std::string& default_location = {});
    bool show_import_folder_dialog(const std::string& default_location = {});
    bool show_open_scene_dialog(const std::string& default_location = {});
    bool show_save_scene_dialog(const std::string& default_location = {});
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
    struct DialogInbox;
    struct DialogCallbackData;

    SDL_Window* window_ = nullptr;
    void* gl_context_ = nullptr;
    unsigned int framebuffer_texture_ = 0;
    unsigned int fullscreen_vao_ = 0;
    std::unique_ptr<GlShaderProgram> compositor_program_;
    bool sdl_initialized_ = false;
    bool imgui_initialized_ = false;
    bool ui_frame_started_ = false;
    bool left_mouse_down_ = false;
    bool left_mouse_dragged_ = false;
    bool right_mouse_down_ = false;
    bool relative_mouse_mode_ = false;
    int window_width_ = 0;
    int window_height_ = 0;
    int texture_width_ = 0;
    int texture_height_ = 0;
    std::string imgui_ini_path_;
    std::string last_error_;
    std::shared_ptr<DialogInbox> dialog_inbox_;

    bool show_dialog(FileDialogKind kind, const std::string& default_location);
    static void dialog_callback(
        void* userdata,
        const char* const* filelist,
        int filter);
    void set_error_from_sdl(const char* prefix);
    void ensure_framebuffer_texture(int width, int height);
    void release_gl_resources();
};

}  // namespace renderer
