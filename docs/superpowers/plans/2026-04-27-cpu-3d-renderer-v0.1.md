# CPU 3D Renderer v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++20 CPU renderer that outputs PNG images through software rasterization, Whitted ray tracing, and path tracing.

**Architecture:** The renderer is split into core math/image utilities, renderer-neutral scene data, a shared BVH acceleration layer, and three concrete implementations behind `IRenderer`. The public interface stays small so v0.1 remains teachable while preserving a path toward a future unified render pipeline.

**Tech Stack:** C++20, CMake, standard library threads, `stb_image_write` for PNG output, `tinyobjloader` for OBJ loading, no OpenGL/Vulkan/DirectX/Embree/OptiX.

---

## Guardrails

- Do not use OpenGL, Vulkan, DirectX, Embree, OptiX, CUDA, or platform graphics APIs.
- Keep core algorithms handwritten: vector math, intersections, rasterization, BVH traversal, ray tracing, path tracing, sampling.
- Use Chinese comments for teaching, with standard English graphics terms where useful.
- Every production behavior added in a task gets a failing test first unless the file is build metadata or documentation.
- Prefer simple, readable implementations over clever micro-optimizations in v0.1.
- Commit after each task.

## File Structure

Create or modify these files over the plan:

```text
CMakeLists.txt
README.md
external/stb/stb_image_write.h
external/tinyobjloader/tiny_obj_loader.h
src/main.cpp
src/core/color.h
src/core/image.h
src/core/image.cpp
src/core/random.h
src/core/timer.h
src/core/math/vec2.h
src/core/math/vec3.h
src/core/math/vec4.h
src/core/math/mat4.h
src/core/math/ray.h
src/core/math/bounds.h
src/scene/camera.h
src/scene/camera.cpp
src/scene/material.h
src/scene/texture.h
src/scene/light.h
src/scene/primitive.h
src/scene/mesh.h
src/scene/mesh.cpp
src/scene/scene.h
src/scene/scene.cpp
src/scene/obj_loader.h
src/scene/obj_loader.cpp
src/acceleration/bvh.h
src/acceleration/bvh.cpp
src/sampling/sampler.h
src/sampling/sampler.cpp
src/render/render_settings.h
src/render/renderer.h
src/render/rasterizer/rasterizer_renderer.h
src/render/rasterizer/rasterizer_renderer.cpp
src/render/raytracer/raytracer_renderer.h
src/render/raytracer/raytracer_renderer.cpp
src/render/pathtracer/pathtracer_renderer.h
src/render/pathtracer/pathtracer_renderer.cpp
tests/test_framework.h
tests/renderer_tests.cpp
output/.gitkeep
docs/output/v0.1-results.md
```

## Common Build Commands

Use these commands throughout the plan:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Renderer smoke commands after the relevant renderer tasks:

```powershell
.\build\bin\renderer.exe --mode raster --scene raster_triangle --width 512 --height 512 --output output\raster_triangle.png
.\build\bin\renderer.exe --mode ray --scene mirror_spheres --width 512 --height 512 --output output\mirror_spheres.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 32 --max-depth 5 --output output\cornell_box.png
```

---

### Task 1: Project Skeleton and Test Harness

**Files:**
- Create: `CMakeLists.txt`
- Create: `src/main.cpp`
- Create: `tests/test_framework.h`
- Create: `tests/renderer_tests.cpp`
- Create: `README.md`
- Create: `output/.gitkeep`

- [ ] **Step 1: Write the first failing smoke test**

Create `tests/test_framework.h` with a tiny assertion helper:

```cpp
#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

inline void test_check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        std::cerr << file << ":" << line << " check failed: " << expression << "\n";
        std::exit(1);
    }
}

inline bool nearly_equal(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

#define RENDER_CHECK(expr) test_check((expr), #expr, __FILE__, __LINE__)
```

Create `tests/renderer_tests.cpp`:

```cpp
#include "test_framework.h"

#include <iostream>

int main() {
    RENDER_CHECK(1 + 1 == 2);
    std::cout << "renderer_tests: all tests passed\n";
    return 0;
}
```

- [ ] **Step 2: Run the test before build support exists**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because no CMake project exists yet.

- [ ] **Step 3: Add the minimal CMake project and CLI stub**

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(cpu_3d_renderer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(OUTPUTCONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER "${OUTPUTCONFIG}" OUTPUTCONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin")
endforeach()

add_library(renderer_core INTERFACE)

target_include_directories(renderer_core INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
)

add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE renderer_core)

add_executable(renderer_tests tests/renderer_tests.cpp)
target_link_libraries(renderer_tests PRIVATE renderer_core)

if (MSVC)
    target_compile_options(renderer PRIVATE /W4 /permissive-)
    target_compile_options(renderer_tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(renderer PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(renderer_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

Create `src/main.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "CPU 3D Renderer v0.1\n";
    return 0;
}
```

Create `README.md` with build and run commands:

````markdown
# CPU 3D Renderer

这是一个 C++20 CPU 3D 图形渲染器，目标是用清晰代码学习软件光栅化、光线追踪和路径追踪。

## 构建

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\renderer_tests.exe
.\build\bin\renderer.exe
```
````

Create an empty `output/.gitkeep` so render output has a stable directory.

- [ ] **Step 4: Verify the smoke test passes**

Run:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS and prints `renderer_tests: all tests passed`.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt README.md src/main.cpp tests/test_framework.h tests/renderer_tests.cpp output/.gitkeep
git commit -m "chore: add renderer project skeleton"
```

---

### Task 2: Core Math Types

**Files:**
- Create: `src/core/math/vec2.h`
- Create: `src/core/math/vec3.h`
- Create: `src/core/math/vec4.h`
- Create: `src/core/math/mat4.h`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing math tests**

Add these tests to `tests/renderer_tests.cpp`:

```cpp
#include "core/math/mat4.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"
#include "core/math/vec4.h"

void test_vec3_arithmetic() {
    renderer::Vec3 a(1.0, 2.0, 3.0);
    renderer::Vec3 b(4.0, -2.0, 0.5);

    renderer::Vec3 sum = a + b;
    RENDER_CHECK(nearly_equal(sum.x, 5.0));
    RENDER_CHECK(nearly_equal(sum.y, 0.0));
    RENDER_CHECK(nearly_equal(sum.z, 3.5));

    RENDER_CHECK(nearly_equal(renderer::dot(a, b), 1.0 * 4.0 + 2.0 * -2.0 + 3.0 * 0.5));

    renderer::Vec3 c = renderer::cross(renderer::Vec3(1, 0, 0), renderer::Vec3(0, 1, 0));
    RENDER_CHECK(nearly_equal(c.x, 0.0));
    RENDER_CHECK(nearly_equal(c.y, 0.0));
    RENDER_CHECK(nearly_equal(c.z, 1.0));

    renderer::Vec3 n = renderer::normalize(renderer::Vec3(0, 3, 4));
    RENDER_CHECK(nearly_equal(renderer::length(n), 1.0));
    RENDER_CHECK(nearly_equal(n.y, 0.6));
    RENDER_CHECK(nearly_equal(n.z, 0.8));
}

void test_mat4_translation_and_perspective_divide() {
    renderer::Mat4 t = renderer::Mat4::translation(renderer::Vec3(2, 3, 4));
    renderer::Vec4 p = t * renderer::Vec4(1, 1, 1, 1);
    RENDER_CHECK(nearly_equal(p.x, 3.0));
    RENDER_CHECK(nearly_equal(p.y, 4.0));
    RENDER_CHECK(nearly_equal(p.z, 5.0));
    RENDER_CHECK(nearly_equal(p.w, 1.0));
}
```

Call them from `main()` before printing success.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because `core/math/*.h` does not exist.

- [ ] **Step 3: Implement math headers**

Create the four math headers as header-only types in namespace `renderer`.

Required API:

```cpp
namespace renderer {
struct Vec2 { double x, y; };
struct Vec3 { double x, y, z; };
struct Vec4 { double x, y, z, w; };

Vec3 operator+(const Vec3& a, const Vec3& b);
Vec3 operator-(const Vec3& a, const Vec3& b);
Vec3 operator*(const Vec3& v, double s);
Vec3 operator/(const Vec3& v, double s);
double dot(const Vec3& a, const Vec3& b);
Vec3 cross(const Vec3& a, const Vec3& b);
double length_squared(const Vec3& v);
double length(const Vec3& v);
Vec3 normalize(const Vec3& v);
Vec3 min_components(const Vec3& a, const Vec3& b);
Vec3 max_components(const Vec3& a, const Vec3& b);

struct Mat4 {
    double m[4][4];
    static Mat4 identity();
    static Mat4 translation(const Vec3& offset);
    static Mat4 scale(const Vec3& scale);
    static Mat4 perspective(double vertical_fov_degrees, double aspect, double near_z, double far_z);
    static Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up);
};

Vec4 operator*(const Mat4& matrix, const Vec4& v);
Mat4 operator*(const Mat4& a, const Mat4& b);
}
```

Document the convention in `mat4.h`: matrices are row-major in memory, vectors are multiplied as column vectors, and clip-space depth follows the common `[-1, 1]` teaching convention.

- [ ] **Step 4: Verify math tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt src/core/math tests/renderer_tests.cpp
git commit -m "feat: add core math types"
```

---

### Task 3: Rays, Bounds, Color, Image Output, Random, Timer

**Files:**
- Create: `src/core/math/ray.h`
- Create: `src/core/math/bounds.h`
- Create: `src/core/color.h`
- Create: `src/core/image.h`
- Create: `src/core/image.cpp`
- Create: `src/core/random.h`
- Create: `src/core/timer.h`
- Add: `external/stb/stb_image_write.h`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing tests for ray, bounds, and image**

Add tests:

```cpp
#include "core/color.h"
#include "core/image.h"
#include "core/math/bounds.h"
#include "core/math/ray.h"

void test_ray_and_bounds_intersection() {
    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::Bounds3 box(renderer::Vec3(-1, -1, -1), renderer::Vec3(1, 1, 1));
    RENDER_CHECK(box.intersect(ray, 0.001, 1000.0));

    renderer::Ray miss(renderer::Vec3(5, 5, -5), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!box.intersect(miss, 0.001, 1000.0));
}

void test_image_stores_gamma_corrected_pixels() {
    renderer::Image image(2, 1);
    image.set_pixel(0, 0, renderer::Color(1.0, 0.25, 0.0));
    renderer::Rgb8 pixel = image.pixel_rgb8(0, 0);
    RENDER_CHECK(pixel.r == 255);
    RENDER_CHECK(pixel.g >= 126 && pixel.g <= 128);
    RENDER_CHECK(pixel.b == 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because ray, bounds, image, and color files do not exist.

- [ ] **Step 3: Add external STB header**

Place the official `stb_image_write.h` single header at `external/stb/stb_image_write.h`.

Use this exact header source:

```text
https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

Do not modify the vendor header except for line endings.

- [ ] **Step 4: Implement core utility files**

Required API:

```cpp
namespace renderer {
using Color = Vec3;

struct Rgb8 {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

Color clamp_color(const Color& color);
Rgb8 to_rgb8(const Color& linear_color);

struct Ray {
    Vec3 origin;
    Vec3 direction;
    Ray(const Vec3& origin, const Vec3& direction);
    Vec3 at(double t) const;
};

struct Bounds3 {
    Vec3 min;
    Vec3 max;
    Bounds3();
    Bounds3(const Vec3& min_point, const Vec3& max_point);
    void expand(const Vec3& p);
    void expand(const Bounds3& bounds);
    Vec3 extent() const;
    int longest_axis() const;
    bool intersect(const Ray& ray, double t_min, double t_max) const;
};

class Image {
public:
    Image(int width, int height);
    int width() const;
    int height() const;
    void set_pixel(int x, int y, const Color& color);
    const Color& pixel(int x, int y) const;
    Rgb8 pixel_rgb8(int x, int y) const;
    bool write_png(const std::string& path) const;
};
}
```

`to_rgb8` must clamp linear color to `[0, 1]`, apply gamma `1/2.2`, then convert to `[0, 255]`.

`Bounds3::intersect` must use the slab method and handle rays parallel to an axis without division crashes.

- [ ] **Step 5: Update CMake source list**

Replace the Task 1 interface library with a static library that now has a source file:

```cmake
add_library(renderer_core STATIC
    src/core/image.cpp
)

target_include_directories(renderer_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
)
```

Also add compile options for `renderer_core` in the existing MSVC/non-MSVC branches.

- [ ] **Step 6: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add external/stb src/core CMakeLists.txt tests/renderer_tests.cpp
git commit -m "feat: add image and ray utilities"
```

---

### Task 4: Scene Primitives, Materials, Textures, Lights

**Files:**
- Create: `src/scene/material.h`
- Create: `src/scene/texture.h`
- Create: `src/scene/light.h`
- Create: `src/scene/primitive.h`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing primitive tests**

Add tests:

```cpp
#include "scene/material.h"
#include "scene/primitive.h"

void test_sphere_intersection() {
    renderer::Material material;
    material.base_color = renderer::Color(1, 0, 0);
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 0);

    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(sphere.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(nearly_equal(hit.t, 4.0));
    RENDER_CHECK(nearly_equal(hit.position.z, -1.0));
    RENDER_CHECK(nearly_equal(renderer::length(hit.normal), 1.0));
    RENDER_CHECK(hit.material_id == 0);
}

void test_triangle_intersection() {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::Ray ray(renderer::Vec3(0, 0.25, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(tri.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(nearly_equal(hit.position.x, 0.0));
    RENDER_CHECK(nearly_equal(hit.position.y, 0.25));
    RENDER_CHECK(hit.material_id == 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because scene primitive files do not exist.

- [ ] **Step 3: Implement materials and primitives**

Required types:

```cpp
namespace renderer {
enum class MaterialType {
    Diffuse,
    Metal,
    Dielectric,
    Emissive
};

struct Material {
    MaterialType type = MaterialType::Diffuse;
    Color base_color = Color(0.8, 0.8, 0.8);
    Color emission = Color(0, 0, 0);
    double roughness = 0.0;
    double metallic = 0.0;
    double ior = 1.5;
};

struct HitRecord {
    double t = 0.0;
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    int material_id = -1;
    bool front_face = true;
    void set_face_normal(const Ray& ray, const Vec3& outward_normal);
};

class Sphere {
public:
    Sphere(const Vec3& center, double radius, int material_id);
    bool intersect(const Ray& ray, double t_min, double t_max, HitRecord& hit) const;
    Bounds3 bounds() const;
};

class Triangle {
public:
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c, int material_id);
    bool intersect(const Ray& ray, double t_min, double t_max, HitRecord& hit) const;
    Bounds3 bounds() const;
    Vec3 centroid() const;
};
}
```

Use the quadratic equation for spheres and Moller-Trumbore for triangles. Add Chinese comments explaining what `t_min` prevents: it avoids self-intersection acne when a shadow or bounce ray starts at a surface.

- [ ] **Step 4: Add constant and checker texture helpers**

Implement `src/scene/texture.h` with simple value types:

```cpp
namespace renderer {
struct ConstantTexture {
    Color color;
    Color sample(const Vec2& uv, const Vec3& p) const;
};

struct CheckerTexture {
    Color even;
    Color odd;
    double scale = 8.0;
    Color sample(const Vec2& uv, const Vec3& p) const;
};
}
```

- [ ] **Step 5: Add light definitions**

Implement `src/scene/light.h`:

```cpp
namespace renderer {
struct PointLight {
    Vec3 position;
    Color intensity;
};

struct DirectionalLight {
    Vec3 direction;
    Color radiance;
};
}
```

- [ ] **Step 6: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/scene CMakeLists.txt tests/renderer_tests.cpp
git commit -m "feat: add scene primitives and materials"
```

---

### Task 5: Camera, Mesh, Scene, Built-In Scenes

**Files:**
- Create: `src/scene/camera.h`
- Create: `src/scene/camera.cpp`
- Create: `src/scene/mesh.h`
- Create: `src/scene/mesh.cpp`
- Create: `src/scene/scene.h`
- Create: `src/scene/scene.cpp`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing camera and scene tests**

Add tests:

```cpp
#include "scene/camera.h"
#include "scene/scene.h"

void test_camera_center_ray_points_forward() {
    renderer::Camera camera(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        60.0,
        1.0);

    renderer::Ray ray = camera.generate_ray(0.5, 0.5);
    RENDER_CHECK(nearly_equal(ray.direction.x, 0.0, 1e-6));
    RENDER_CHECK(nearly_equal(ray.direction.y, 0.0, 1e-6));
    RENDER_CHECK(ray.direction.z < -0.999);
}

void test_builtin_scene_contains_renderable_geometry() {
    renderer::Scene scene = renderer::make_gradient_sphere_scene();
    RENDER_CHECK(!scene.materials.empty());
    RENDER_CHECK(!scene.spheres.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because camera and scene files do not exist.

- [ ] **Step 3: Implement camera**

Required API:

```cpp
namespace renderer {
class Camera {
public:
    Camera(const Vec3& eye, const Vec3& target, const Vec3& up, double vertical_fov_degrees, double aspect_ratio);
    Ray generate_ray(double u, double v) const;
    const Vec3& eye() const;
};
}
```

Use a pinhole camera. Document the basis vectors: `forward`, `right`, and `true_up`.

- [ ] **Step 4: Implement mesh and scene containers**

Required API:

```cpp
namespace renderer {
struct Mesh {
    std::vector<Triangle> triangles;
};

struct Scene {
    std::vector<Material> materials;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<PointLight> point_lights;
    std::vector<DirectionalLight> directional_lights;
    Color environment = Color(0.02, 0.03, 0.05);
};

Scene make_gradient_sphere_scene();
Scene make_raster_triangle_scene();
Scene make_mirror_spheres_scene();
Scene make_cornell_box_scene();
}
```

Cornell box geometry must use triangles for the walls and an emissive rectangle on the ceiling. `mirror_spheres` must include at least one diffuse floor sphere/plane approximation, one metal sphere, and one point light.

- [ ] **Step 5: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/scene CMakeLists.txt tests/renderer_tests.cpp
git commit -m "feat: add camera and built-in scenes"
```

---

### Task 6: BVH Acceleration

**Files:**
- Create: `src/acceleration/bvh.h`
- Create: `src/acceleration/bvh.cpp`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing BVH tests**

Add tests:

```cpp
#include "acceleration/bvh.h"

void test_bvh_matches_bruteforce_triangle_hit() {
    std::vector<renderer::Triangle> tris;
    tris.emplace_back(renderer::Vec3(-1, 0, 0), renderer::Vec3(1, 0, 0), renderer::Vec3(0, 1, 0), 0);
    tris.emplace_back(renderer::Vec3(-1, 0, 5), renderer::Vec3(1, 0, 5), renderer::Vec3(0, 1, 5), 0);

    renderer::Bvh bvh;
    bvh.build(tris);

    renderer::Ray ray(renderer::Vec3(0, 0.25, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord bvh_hit;
    RENDER_CHECK(bvh.intersect(ray, 0.001, 1000.0, bvh_hit));
    RENDER_CHECK(nearly_equal(bvh_hit.t, 2.0));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because BVH does not exist.

- [ ] **Step 3: Implement BVH**

Required API:

```cpp
namespace renderer {
struct BvhNode {
    Bounds3 bounds;
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;
    bool is_leaf() const;
};

class Bvh {
public:
    void build(const std::vector<Triangle>& triangles);
    bool intersect(const Ray& ray, double t_min, double t_max, HitRecord& hit) const;
    const std::vector<int>& primitive_indices() const;
    const std::vector<BvhNode>& nodes() const;
private:
    int build_recursive(int first, int count);
};
}
```

Implementation requirements:

- Store a copy of input triangles or a non-owning pointer with a documented lifetime; prefer a copy in v0.1 for safety.
- Split by centroid bounds along the longest axis.
- Leaf size: 4 triangles.
- Traversal returns closest hit.
- Add comments explaining why BVH reduces ray tracing from checking every triangle to rejecting large groups with AABBs.

- [ ] **Step 4: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/acceleration CMakeLists.txt tests/renderer_tests.cpp
git commit -m "feat: add bvh acceleration"
```

---

### Task 7: Sampling and Renderer Interfaces

**Files:**
- Create: `src/sampling/sampler.h`
- Create: `src/sampling/sampler.cpp`
- Create: `src/render/render_settings.h`
- Create: `src/render/renderer.h`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing sampler and interface tests**

Add tests:

```cpp
#include "render/render_settings.h"
#include "sampling/sampler.h"

void test_cosine_sample_is_in_upper_hemisphere() {
    renderer::PcgRandom rng(42);
    for (int i = 0; i < 100; ++i) {
        renderer::Vec3 d = renderer::cosine_weighted_hemisphere(rng);
        RENDER_CHECK(d.z >= -1e-9);
        RENDER_CHECK(nearly_equal(renderer::length(d), 1.0, 1e-6));
    }
}

void test_render_settings_defaults_are_useful() {
    renderer::RenderSettings settings;
    RENDER_CHECK(settings.width == 512);
    RENDER_CHECK(settings.height == 512);
    RENDER_CHECK(settings.samples_per_pixel == 1);
    RENDER_CHECK(settings.max_depth == 5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because sampling and renderer interface files do not exist.

- [ ] **Step 3: Implement random and sampling**

Required API:

```cpp
namespace renderer {
class PcgRandom {
public:
    explicit PcgRandom(std::uint64_t seed = 1);
    std::uint32_t next_u32();
    double next_double();
};

Vec3 cosine_weighted_hemisphere(PcgRandom& rng);
Vec3 random_in_unit_sphere(PcgRandom& rng);
Vec3 reflect(const Vec3& v, const Vec3& normal);
bool refract(const Vec3& unit_direction, const Vec3& normal, double eta_ratio, Vec3& refracted);
}
```

Use PCG-style integer hashing or a compact LCG for reproducibility. Document why deterministic RNG helps tests and image comparisons.

- [ ] **Step 4: Implement render settings and renderer interface**

Required API:

```cpp
namespace renderer {
enum class RenderMode {
    Raster,
    Ray,
    Path
};

struct RenderSettings {
    int width = 512;
    int height = 512;
    int samples_per_pixel = 1;
    int max_depth = 5;
    int tile_size = 16;
    int thread_count = 0;
    Color background = Color(0.02, 0.03, 0.05);
};

struct RenderResult {
    Image image;
    double seconds = 0.0;
};

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) = 0;
};
}
```

- [ ] **Step 5: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/sampling src/render CMakeLists.txt tests/renderer_tests.cpp
git commit -m "feat: add sampling and renderer interfaces"
```

---

### Task 8: Whitted Ray Tracer

**Files:**
- Create: `src/render/raytracer/raytracer_renderer.h`
- Create: `src/render/raytracer/raytracer_renderer.cpp`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing ray tracer tests**

Add tests:

```cpp
#include "render/raytracer/raytracer_renderer.h"

void test_raytracer_renders_visible_sphere() {
    renderer::Scene scene = renderer::make_gradient_sphere_scene();
    renderer::Camera camera(renderer::Vec3(0, 0, 2), renderer::Vec3(0, 0, -1), renderer::Vec3(0, 1, 0), 45.0, 1.0);

    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    settings.max_depth = 3;

    renderer::RayTracerRenderer renderer_instance;
    renderer::RenderResult result = renderer_instance.render(scene, camera, settings);
    renderer::Color center = result.image.pixel(16, 16);
    RENDER_CHECK(center.x > 0.05 || center.y > 0.05 || center.z > 0.05);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because ray tracer files do not exist.

- [ ] **Step 3: Implement ray tracer**

Required API:

```cpp
namespace renderer {
class RayTracerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;
private:
    Color trace_ray(const Ray& ray, const Scene& scene, const Bvh& bvh, int depth, const RenderSettings& settings) const;
    bool hit_scene(const Ray& ray, const Scene& scene, const Bvh& bvh, double t_min, double t_max, HitRecord& hit) const;
};
}
```

Implementation requirements:

- Build a BVH from scene triangles at the start of render.
- Test spheres directly and triangles through BVH.
- Shade diffuse surfaces with point and directional lights.
- Cast shadow rays using `t_min = 0.001`.
- Support metal reflection with recursive tracing.
- Support dielectric refraction using Schlick reflectance approximation.
- Stop recursion when `depth <= 0`.
- Add comments explaining primary rays, shadow rays, reflection rays, and why recursion depth is required.

- [ ] **Step 4: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 5: Render mirror spheres smoke image**

Run:

```powershell
New-Item -ItemType Directory -Force -Path output | Out-Null
.\build\bin\renderer.exe --mode ray --scene mirror_spheres --width 512 --height 512 --output output\mirror_spheres.png
```

Expected: file `output/mirror_spheres.png` exists and is non-empty.

- [ ] **Step 6: Commit**

```powershell
git add src/render/raytracer CMakeLists.txt tests/renderer_tests.cpp output/.gitkeep
git commit -m "feat: add whitted ray tracer"
```

---

### Task 9: Path Tracer

**Files:**
- Create: `src/render/pathtracer/pathtracer_renderer.h`
- Create: `src/render/pathtracer/pathtracer_renderer.cpp`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing path tracer tests**

Add tests:

```cpp
#include "render/pathtracer/pathtracer_renderer.h"

void test_pathtracer_renders_emissive_scene() {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Camera camera(renderer::Vec3(0, 1, 4), renderer::Vec3(0, 1, 0), renderer::Vec3(0, 1, 0), 40.0, 1.0);

    renderer::RenderSettings settings;
    settings.width = 16;
    settings.height = 16;
    settings.samples_per_pixel = 2;
    settings.max_depth = 3;
    settings.thread_count = 1;

    renderer::PathTracerRenderer renderer_instance;
    renderer::RenderResult result = renderer_instance.render(scene, camera, settings);

    double luminance_sum = 0.0;
    for (int y = 0; y < result.image.height(); ++y) {
        for (int x = 0; x < result.image.width(); ++x) {
            renderer::Color c = result.image.pixel(x, y);
            luminance_sum += c.x + c.y + c.z;
        }
    }
    RENDER_CHECK(luminance_sum > 0.1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because path tracer files do not exist.

- [ ] **Step 3: Implement path tracer**

Required API:

```cpp
namespace renderer {
class PathTracerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;
private:
    Color trace_path(const Ray& ray, const Scene& scene, const Bvh& bvh, PcgRandom& rng, int depth) const;
    bool scatter(const Ray& ray, const HitRecord& hit, const Material& material, PcgRandom& rng, Color& attenuation, Ray& scattered) const;
    bool hit_scene(const Ray& ray, const Scene& scene, const Bvh& bvh, double t_min, double t_max, HitRecord& hit) const;
};
}
```

Implementation requirements:

- Use tiled rendering. Tile size comes from `RenderSettings::tile_size`.
- Use `std::thread` or `std::async` with an atomic tile counter.
- If `thread_count <= 0`, use `std::thread::hardware_concurrency()` with a minimum of 1.
- For each pixel, jitter samples within the pixel.
- Average samples and write linear color to `Image`.
- Diffuse material uses cosine-weighted hemisphere sampling in local normal space.
- Metal material reflects with roughness perturbation.
- Dielectric material reflects or refracts using Schlick reflectance.
- Emissive material returns emission.
- Use explicit `max_depth`, no Russian roulette in v0.1.
- Add comments explaining the rendering equation approximation: throughput multiplied by emitted or scattered radiance.

- [ ] **Step 4: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 5: Render Cornell box smoke image**

Run:

```powershell
New-Item -ItemType Directory -Force -Path output | Out-Null
.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 32 --max-depth 5 --output output\cornell_box.png
```

Expected: file `output/cornell_box.png` exists and is non-empty. Image should show a Cornell-box-like room with visible diffuse light contribution.

- [ ] **Step 6: Commit**

```powershell
git add src/render/pathtracer CMakeLists.txt tests/renderer_tests.cpp output/.gitkeep
git commit -m "feat: add path tracer"
```

---

### Task 10: Software Rasterizer

**Files:**
- Create: `src/render/rasterizer/rasterizer_renderer.h`
- Create: `src/render/rasterizer/rasterizer_renderer.cpp`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing rasterizer tests**

Add tests:

```cpp
#include "render/rasterizer/rasterizer_renderer.h"

void test_rasterizer_draws_triangle() {
    renderer::Scene scene = renderer::make_raster_triangle_scene();
    renderer::Camera camera(renderer::Vec3(0, 0, 2), renderer::Vec3(0, 0, 0), renderer::Vec3(0, 1, 0), 45.0, 1.0);

    renderer::RenderSettings settings;
    settings.width = 64;
    settings.height = 64;

    renderer::RasterizerRenderer renderer_instance;
    renderer::RenderResult result = renderer_instance.render(scene, camera, settings);

    int lit_pixels = 0;
    for (int y = 0; y < result.image.height(); ++y) {
        for (int x = 0; x < result.image.width(); ++x) {
            renderer::Color c = result.image.pixel(x, y);
            if (c.x + c.y + c.z > 0.05) {
                ++lit_pixels;
            }
        }
    }
    RENDER_CHECK(lit_pixels > 20);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because rasterizer files do not exist.

- [ ] **Step 3: Implement rasterizer**

Required API:

```cpp
namespace renderer {
class RasterizerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;
private:
    double edge_function(const Vec3& a, const Vec3& b, const Vec3& c) const;
};
}
```

Implementation requirements:

- Build view/projection matrices from the camera or expose camera view/projection helpers.
- Transform triangles from world to clip space.
- Perform perspective divide to NDC.
- Convert NDC to screen coordinates.
- Use a depth buffer initialized to positive infinity.
- Compute barycentric coordinates with edge functions.
- Interpolate depth and normal.
- Shade with Lambert and Blinn-Phong using point and directional lights.
- Support constant material base color and checker texture through the material/texture helpers.
- Document the v0.1 clipping limitation: example geometry should remain inside the near plane.

- [ ] **Step 4: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 5: Render raster triangle smoke image**

Run:

```powershell
New-Item -ItemType Directory -Force -Path output | Out-Null
.\build\bin\renderer.exe --mode raster --scene raster_triangle --width 512 --height 512 --output output\raster_triangle.png
```

Expected: file `output/raster_triangle.png` exists and is non-empty. Image should show a depth-tested shaded triangle.

- [ ] **Step 6: Commit**

```powershell
git add src/render/rasterizer CMakeLists.txt tests/renderer_tests.cpp output/.gitkeep
git commit -m "feat: add software rasterizer"
```

---

### Task 11: OBJ Loading

**Files:**
- Add: `external/tinyobjloader/tiny_obj_loader.h`
- Create: `src/scene/obj_loader.h`
- Create: `src/scene/obj_loader.cpp`
- Modify: `tests/renderer_tests.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing OBJ loader test**

Add a temporary OBJ test string writer inside `tests/renderer_tests.cpp`:

```cpp
#include "scene/obj_loader.h"
#include <fstream>

void test_obj_loader_reads_single_triangle() {
    const std::string path = "test_single_triangle.obj";
    {
        std::ofstream out(path);
        out << "v 0 0 0\n";
        out << "v 1 0 0\n";
        out << "v 0 1 0\n";
        out << "f 1 2 3\n";
    }

    renderer::Mesh mesh = renderer::load_obj_mesh(path, 0);
    RENDER_CHECK(mesh.triangles.size() == 1);
    std::remove(path.c_str());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build build --config Release
```

Expected: FAIL because OBJ loader files and tinyobjloader header do not exist.

- [ ] **Step 3: Add external tinyobjloader header**

Place the official `tiny_obj_loader.h` single header at `external/tinyobjloader/tiny_obj_loader.h`.

Use this exact header source:

```text
https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/release/tiny_obj_loader.h
```

Do not modify the vendor header except for line endings.

- [ ] **Step 4: Implement OBJ loader**

Required API:

```cpp
namespace renderer {
Mesh load_obj_mesh(const std::string& path, int material_id);
}
```

Implementation requirements:

- Define `TINYOBJLOADER_IMPLEMENTATION` in exactly one `.cpp` file.
- Triangulate faces through tinyobjloader configuration or reject non-triangles with a clear exception.
- Convert OBJ vertices into `Triangle` objects with the provided material id.
- Throw `std::runtime_error` with the tinyobj warning/error text when loading fails.

- [ ] **Step 5: Verify tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add external/tinyobjloader src/scene/obj_loader.* CMakeLists.txt tests/renderer_tests.cpp
git commit -m "feat: add obj mesh loading"
```

---

### Task 12: CLI, Example Rendering, README, Result Report

**Files:**
- Modify: `src/main.cpp`
- Modify: `README.md`
- Create: `docs/output/v0.1-results.md`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a failing CLI expectation**

Run:

```powershell
.\build\bin\renderer.exe --mode ray --scene gradient_sphere --width 64 --height 64 --output output\cli_test.png
```

Expected: FAIL because the CLI stub does not parse arguments or write images.

- [ ] **Step 2: Implement CLI parsing**

`src/main.cpp` must support:

```text
--mode raster|ray|path
--scene gradient_sphere|raster_triangle|mirror_spheres|cornell_box|obj_viewer
--obj path/to/model.obj
--width integer
--height integer
--spp integer
--max-depth integer
--threads integer
--output path/to/file.png
--help
```

Implementation requirements:

- Use a small hand-written parser over `argv`.
- Print helpful errors for unknown modes, unknown scenes, missing output, invalid integer values, and missing `--obj` for `obj_viewer`.
- Construct only one concrete renderer based on `--mode`.
- Use `IRenderer&` for the render call.
- Create parent output directory if it does not exist.
- Write PNG with `RenderResult::image.write_png`.
- Print render time from `RenderResult::seconds`.

- [ ] **Step 3: Verify CLI smoke render passes**

Run:

```powershell
New-Item -ItemType Directory -Force -Path output | Out-Null
.\build\bin\renderer.exe --mode ray --scene gradient_sphere --width 64 --height 64 --output output\cli_test.png
```

Expected: PASS and `output/cli_test.png` exists.

- [ ] **Step 4: Render v0.1 example images**

Run:

```powershell
.\build\bin\renderer.exe --mode raster --scene raster_triangle --width 512 --height 512 --output output\raster_triangle.png
.\build\bin\renderer.exe --mode ray --scene mirror_spheres --width 512 --height 512 --output output\mirror_spheres.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 32 --max-depth 5 --output output\cornell_box.png
```

Expected:

```text
output/raster_triangle.png
output/mirror_spheres.png
output/cornell_box.png
```

All three files exist and are non-empty.

- [ ] **Step 5: Update README**

README must include:

- Project purpose
- Build commands
- Test command
- Three renderer modes
- Example render commands
- Learning order
- Dependency explanation
- Clear statement that OpenGL/Vulkan/DirectX are not used

- [ ] **Step 6: Write result report**

Create `docs/output/v0.1-results.md` after running the examples.

Required sections:

```markdown
# v0.1 输出结果说明

## 版本信息

## 构建环境

## 本版实现内容

## 示例图片

## 复现命令

## 性能记录

## 已知限制

## 下一步路线图
```

Record actual render times printed by the CLI. If exact hardware info is unavailable, record the OS, compiler, CMake generator, resolution, samples per pixel, max depth, and thread count that were actually used.

- [ ] **Step 7: Final verification**

Run:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\renderer_tests.exe
.\build\bin\renderer.exe --mode raster --scene raster_triangle --width 512 --height 512 --output output\raster_triangle.png
.\build\bin\renderer.exe --mode ray --scene mirror_spheres --width 512 --height 512 --output output\mirror_spheres.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 32 --max-depth 5 --output output\cornell_box.png
```

Expected: all commands pass and produce the three PNG files.

- [ ] **Step 8: Commit**

```powershell
git add src/main.cpp README.md docs/output/v0.1-results.md output CMakeLists.txt
git commit -m "docs: document v0.1 renderer results"
```

---

## Plan Self-Review

- Spec coverage: This plan covers CMake, PNG output, OBJ loading, handwritten math, shared scene data, BVH, rasterization, Whitted ray tracing, path tracing, tests, examples, README, and `docs/output/v0.1-results.md`.
- Scope control: v0.1 avoids GPU APIs, render graph, full PBR, MIS, denoising, animation, and real-time windows.
- Interface path: `IRenderer`, `RenderSettings`, and renderer-neutral `Scene` preserve the path toward a future unified rendering interface.
- Testing: Every renderer-facing behavior has a failing test step before implementation. Build metadata and documentation are verified by build or CLI smoke commands.
