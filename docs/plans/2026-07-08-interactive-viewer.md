# Interactive Viewer Implementation Plan

> **历史文档（已过期）：** 本文记录当时的实现计划。CPU 软件光栅化与 Whitted 光追已于 2026-07-29 删除；当前架构仅支持 OpenGL 与 Path。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an SDL3 + Eigen interactive viewer that displays the project's own CPU framebuffer, supports raster/ray/path modes, and loads Computer Graphics Archive CornellBox OBJ/MTL scenes correctly.

**Architecture:** Keep the existing offline `IRenderer` path intact. Add small renderer-owned framebuffer/depth/interactive session abstractions, a tiny SDL3 platform layer, an Eigen orbit camera controller, and a scene asset loader that maps OBJ/MTL assets into `renderer::Scene`. The viewer displays CPU-generated pixels only; SDL3 handles window/input/presentation.

**Tech Stack:** C++20, CMake, SDL3, Eigen, tinyobjloader, existing single-file C++ test harness.

## Global Constraints

- SDL3 is platform/presentation only: no OpenGL, Vulkan, DirectX, SDL GPU API, SDL render pipeline, OptiX, RT Core, or CUDA rendering path.
- Keep `renderer.exe` and existing offline rendering behavior working.
- `viewer.exe` supports raster, ray, and path modes; `1/2/3` switch modes at runtime.
- Ray/path modes may be slow, but they must continue displaying progress in the window.
- Path mode uses progressive accumulation and resets on camera/window/scene/mode changes.
- External CornellBox loading must preserve per-face materials, emissive lights, mirror-like materials, and dielectric materials from OBJ/MTL.
- Eigen is introduced at the interaction boundary first; do not replace all existing math primitives in this version.
- `Computer Graphics Archive/` is a local untracked asset directory and must not be committed.

---

## File Structure

- Create `src/render/framebuffer.h/.cpp`: CPU color framebuffer and RGBA8 conversion.
- Create `src/render/depth_buffer.h/.cpp`: software depth buffer for raster sessions.
- Create `src/render/interactive/interactive_render_session.h`: shared session interface and frame state.
- Create `src/render/interactive/raster_interactive_session.h/.cpp`: software raster output to `Framebuffer`.
- Create `src/render/interactive/ray_interactive_session.h/.cpp`: Whitted ray output to `Framebuffer`.
- Create `src/render/interactive/path_interactive_session.h/.cpp`: progressive path tracing session.
- Create `src/scene/scene_asset_loader.h/.cpp`: OBJ/MTL scene loader using tinyobjloader material data.
- Create `src/interactive/orbit_camera_controller.h/.cpp`: Eigen-backed orbit camera controller.
- Create `src/platform/sdl/sdl_display_backend.h/.cpp`: SDL3 window/input/presentation backend.
- Create `src/viewer_main.cpp`: viewer CLI, scene selection, loop, hotkeys.
- Modify `CMakeLists.txt`: optional SDL3/Eigen dependencies, new sources, `viewer` target.
- Modify `tests/renderer_tests.cpp`: tests for buffers, asset loading, camera controller, and interactive sessions.

---

### Task 1: Core CPU Buffers

**Files:**
- Create: `src/render/framebuffer.h`
- Create: `src/render/framebuffer.cpp`
- Create: `src/render/depth_buffer.h`
- Create: `src/render/depth_buffer.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `renderer::Framebuffer`, `renderer::DepthBuffer`
- Consumes: `renderer::Color`, `renderer::Rgb8`, `renderer::to_rgb8`

- [ ] **Step 1: Write failing framebuffer/depth tests**

Add tests:

```cpp
void test_framebuffer_clear_set_and_rgba8_conversion() {
    renderer::Framebuffer framebuffer(2, 1);
    framebuffer.clear(renderer::Color(0.25, 0.0, 1.0));
    framebuffer.set_pixel(1, 0, renderer::Color(1.0, 0.25, 0.0));
    RENDER_CHECK(framebuffer.width() == 2);
    RENDER_CHECK(framebuffer.height() == 1);
    RENDER_CHECK(nearly_equal(framebuffer.pixel(0, 0).z, 1.0));
    const std::vector<std::uint8_t> rgba = framebuffer.to_rgba8();
    RENDER_CHECK(rgba.size() == 8);
    RENDER_CHECK(rgba[3] == 255);
    RENDER_CHECK(rgba[4] == 255);
    RENDER_CHECK(rgba[7] == 255);
}

void test_depth_buffer_clear_resize_and_access() {
    renderer::DepthBuffer depth(2, 2);
    depth.clear(42.0);
    depth.set(1, 0, 0.5);
    RENDER_CHECK(nearly_equal(depth.get(0, 0), 42.0));
    RENDER_CHECK(nearly_equal(depth.get(1, 0), 0.5));
    depth.resize(1, 1);
    depth.clear(7.0);
    RENDER_CHECK(depth.width() == 1);
    RENDER_CHECK(depth.height() == 1);
    RENDER_CHECK(nearly_equal(depth.get(0, 0), 7.0));
}
```

- [ ] **Step 2: Run RED**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: compile fails because `render/framebuffer.h` and `render/depth_buffer.h` do not exist.

- [ ] **Step 3: Implement buffers**

Implement:

```cpp
class Framebuffer {
public:
    Framebuffer(int width, int height);
    int width() const;
    int height() const;
    void resize(int width, int height);
    void clear(const Color& color);
    void set_pixel(int x, int y, const Color& color);
    const Color& pixel(int x, int y) const;
    std::vector<std::uint8_t> to_rgba8() const;
};

class DepthBuffer {
public:
    DepthBuffer(int width, int height);
    int width() const;
    int height() const;
    void resize(int width, int height);
    void clear(double value);
    double get(int x, int y) const;
    void set(int x, int y, double value);
};
```

Validate dimensions and coordinate bounds with `std::invalid_argument` / `std::out_of_range`.

- [ ] **Step 4: Run GREEN**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt src/render/framebuffer.* src/render/depth_buffer.* tests/renderer_tests.cpp
git commit -m "feat: add software render buffers"
```

---

### Task 2: Scene Asset Loader For OBJ/MTL CornellBox

**Files:**
- Create: `src/scene/scene_asset_loader.h`
- Create: `src/scene/scene_asset_loader.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Consumes: `renderer::Scene`, `renderer::Material`, `renderer::Triangle`, `renderer::Bounds3`, tinyobjloader
- Produces: `renderer::LoadedScene renderer::load_scene_asset(const std::string& path, int width, int height)`

- [ ] **Step 1: Write failing asset loader tests**

Add a temporary OBJ/MTL test:

```cpp
void test_scene_asset_loader_preserves_obj_mtl_materials() {
    const std::string obj_path = "test_asset_loader.obj";
    const std::string mtl_path = "test_asset_loader.mtl";
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl red\nKd 0.8 0.1 0.1\nKs 0 0 0\nillum 2\n";
        mtl << "newmtl light\nKd 1 1 1\nKe 4 3 2\nillum 2\n";
        mtl << "newmtl mirror\nKd 0.02 0.02 0.02\nKs 0.95 0.95 0.95\nNs 1000\nillum 5\n";
        mtl << "newmtl glass\nKd 0.01 0.01 0.01\nKs 0.3 0.3 0.3\nNi 1.33\nillum 7\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 0 1 1\n";
        obj << "usemtl red\nf 1 2 3\n";
        obj << "usemtl light\nf 4 5 6\n";
        obj << "usemtl mirror\nf 1 4 2\n";
        obj << "usemtl glass\nf 2 5 3\n";
    }
    renderer::LoadedScene loaded = renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.triangles.size() == 4);
    RENDER_CHECK(has_red_like_material(loaded.scene));
    RENDER_CHECK(has_nonzero_emissive_material(loaded.scene));
    RENDER_CHECK(has_material_type(loaded.scene, renderer::MaterialType::Metal));
    RENDER_CHECK(has_material_type(loaded.scene, renderer::MaterialType::Dielectric));
    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
}
```

- [ ] **Step 2: Run RED**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: compile fails because `scene/scene_asset_loader.h` does not exist.

- [ ] **Step 3: Implement loader**

Use tinyobjloader with `config.triangulate = true`. Convert materials:

```cpp
struct LoadedScene {
    Scene scene;
    Camera camera;
    Bounds3 bounds;
};

LoadedScene load_scene_asset(const std::string& path, int width, int height);
```

Mapping:

```cpp
Kd -> base_color
Ke length > 0 -> Emissive
illum == 5 or high Ks -> Metal
illum == 7 or Ni > 1.0 with transmission -> Dielectric
else -> Diffuse
```

Use each shape face's `material_ids[face]` to assign triangle material ids. Compute bounds and default camera from bounds.

- [ ] **Step 4: Run GREEN**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt src/scene/scene_asset_loader.* tests/renderer_tests.cpp
git commit -m "feat: load obj mtl scene assets"
```

---

### Task 3: Orbit Camera Controller

**Files:**
- Create: `src/interactive/orbit_camera_controller.h`
- Create: `src/interactive/orbit_camera_controller.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Consumes: Eigen, `renderer::Camera`, `renderer::Vec3`, `renderer::Bounds3`
- Produces: `renderer::OrbitCameraController`, `to_eigen`, `to_vec3`

- [ ] **Step 1: Write failing camera controller tests**

Add:

```cpp
void test_orbit_camera_controller_zoom_and_orbit_change_camera() {
    renderer::Bounds3 bounds(renderer::Vec3(-1, 0, -1), renderer::Vec3(1, 2, 1));
    renderer::OrbitCameraController controller(bounds, 1.0);
    renderer::Camera before = controller.camera();
    controller.orbit(0.5, 0.25);
    controller.zoom(-1.0);
    renderer::Camera after = controller.camera();
    RENDER_CHECK(renderer::length(after.eye() - before.eye()) > 0.001);
    RENDER_CHECK(after.viewport_width() > 0.0);
}
```

- [ ] **Step 2: Run RED**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: compile fails because `interactive/orbit_camera_controller.h` does not exist.

- [ ] **Step 3: Implement controller**

Implement yaw/pitch/distance around a target. Clamp pitch to `[-1.5, 1.5]` radians and distance to a small positive minimum.

- [ ] **Step 4: Run GREEN**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt src/interactive/orbit_camera_controller.* tests/renderer_tests.cpp
git commit -m "feat: add orbit camera controller"
```

---

### Task 4: Interactive Render Sessions

**Files:**
- Create: `src/render/interactive/interactive_render_session.h`
- Create: `src/render/interactive/raster_interactive_session.h/.cpp`
- Create: `src/render/interactive/ray_interactive_session.h/.cpp`
- Create: `src/render/interactive/path_interactive_session.h/.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Consumes: `Framebuffer`, `Scene`, `Camera`, `RenderSettings`
- Produces: `RasterInteractiveSession`, `RayInteractiveSession`, `PathInteractiveSession`

- [ ] **Step 1: Write failing session tests**

Add:

```cpp
int count_lit_pixels(const renderer::Framebuffer& framebuffer) {
    int count = 0;
    for (int y = 0; y < framebuffer.height(); ++y) {
        for (int x = 0; x < framebuffer.width(); ++x) {
            renderer::Color c = framebuffer.pixel(x, y);
            if (c.x + c.y + c.z > 0.05) {
                ++count;
            }
        }
    }
    return count;
}

void test_interactive_sessions_render_visible_pixels() {
    renderer::Scene scene = renderer::make_raster_triangle_scene();
    renderer::Camera camera(renderer::Vec3(0, 0, 2), renderer::Vec3(0, 0, 0), renderer::Vec3(0, 1, 0), 45.0, 1.0);
    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    renderer::Framebuffer framebuffer(32, 32);
    renderer::InteractiveFrameState frame_state;

    renderer::RasterInteractiveSession raster;
    raster.reset(scene, settings);
    raster.render_next_frame(scene, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(count_lit_pixels(framebuffer) > 0);

    renderer::RayInteractiveSession ray;
    ray.reset(scene, settings);
    ray.render_next_frame(scene, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(count_lit_pixels(framebuffer) > 0);
}

void test_path_interactive_session_accumulates_and_resets() {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Camera camera(renderer::Vec3(0, 0.15, 1.5), renderer::Vec3(0, 0.15, -2), renderer::Vec3(0, 1, 0), 45.0, 1.0);
    renderer::RenderSettings settings;
    settings.width = 8;
    settings.height = 8;
    settings.samples_per_pixel = 1;
    settings.max_depth = 3;
    settings.thread_count = 1;
    renderer::Framebuffer framebuffer(8, 8);
    renderer::InteractiveFrameState state;
    renderer::PathInteractiveSession path;
    path.reset(scene, settings);
    path.render_next_frame(scene, camera, settings, state, framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 1);
    path.render_next_frame(scene, camera, settings, state, framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 2);
    state.camera_changed = true;
    path.render_next_frame(scene, camera, settings, state, framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 1);
}
```

- [ ] **Step 2: Run RED**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: compile fails because interactive session headers do not exist.

- [ ] **Step 3: Implement sessions**

For raster and ray, call existing offline renderers and copy `Image` pixels into `Framebuffer`. This is acceptable for the first working version; later optimize by sharing pixel kernels.

For path, render one spp per call into a temporary `Image`, add to an accumulation buffer, and write average color to `Framebuffer`. Reset accumulation on changed frame state, settings size changes, or explicit `reset`.

- [ ] **Step 4: Run GREEN**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt src/render/interactive tests/renderer_tests.cpp
git commit -m "feat: add interactive render sessions"
```

---

### Task 5: SDL3 Viewer Target

**Files:**
- Create: `src/platform/sdl/sdl_display_backend.h`
- Create: `src/platform/sdl/sdl_display_backend.cpp`
- Create: `src/viewer_main.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: SDL3, Eigen, `DisplayBackend`, `InputState`, `InteractiveRenderSession`, `SceneAssetLoader`
- Produces: `viewer.exe`

- [ ] **Step 1: Add CMake dependency path**

Use `find_package(SDL3 CONFIG QUIET)` and `find_package(Eigen3 CONFIG QUIET)`. If missing, use `FetchContent` for SDL3 and Eigen. Build `viewer` only when dependencies are available or fetched.

- [ ] **Step 2: Implement SDL backend**

Use SDL3 window surface/software presentation. Convert SDL events to:

```cpp
select_raster, select_ray, select_path, reset_render, left_mouse_down, mouse_delta_x, mouse_delta_y, wheel_delta, window_resized, quit_requested
```

- [ ] **Step 3: Implement viewer main loop**

Parse:

```text
viewer --scene builtin|asset --asset path --mode raster|ray|path --width N --height N
```

Default to `Computer Graphics Archive/CornellBox/CornellBox-Original.obj` when present, otherwise built-in Cornell box.

- [ ] **Step 4: Build viewer**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release`

Expected: `build\bin\viewer.exe` exists.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt src/platform/sdl src/viewer_main.cpp
git commit -m "feat: add sdl interactive viewer"
```

---

### Task 6: End-To-End Verification

**Files:**
- Modify only if verification finds issues.

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: verified working version.

- [ ] **Step 1: Run automated tests**

Run: `cmake --build build --config Release && .\build\bin\renderer_tests.exe`

Expected: all tests pass.

- [ ] **Step 2: Verify offline renderer still works**

Run:

```powershell
.\build\bin\renderer.exe --mode raster --scene raster_triangle --width 128 --height 128 --output output\verify_raster.png
.\build\bin\renderer.exe --mode ray --scene mirror_spheres --width 128 --height 128 --output output\verify_ray.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 64 --height 64 --spp 2 --max-depth 3 --output output\verify_path.png
```

Expected: all commands exit 0 and produce PNG files.

- [ ] **Step 3: Verify CornellBox asset loader through viewer CLI**

Run:

```powershell
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\CornellBox\CornellBox-Original.obj" --mode raster --width 320 --height 240 --frames 2
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\CornellBox\CornellBox-Original.obj" --mode ray --width 160 --height 120 --frames 1
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\CornellBox\CornellBox-Original.obj" --mode path --width 80 --height 60 --frames 2
```

Expected: headless/smoke frame limit exits 0 for all modes.

- [ ] **Step 4: Manual launch command**

Run for user-visible window:

```powershell
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\CornellBox\CornellBox-Original.obj" --mode raster --width 960 --height 540
```

Expected: a window opens, mouse drag orbits, wheel zooms, `1/2/3` switch modes, and close exits.

- [ ] **Step 5: Final commit if fixes were needed**

```powershell
git status --short
git add CMakeLists.txt src tests docs
git commit -m "fix: verify interactive viewer"
```
