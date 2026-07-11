# Eigen Float Full-Project Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the renderer's custom double-precision math with Eigen float types and native Eigen operations across all first-party code while preserving rendering behavior and measuring performance.

**Architecture:** Introduce one Eigen type boundary in `core/math/types.h`, cut vector storage over first with a temporary compatibility operation layer, then migrate modules to native Eigen APIs and float scalars in dependency order. Remove all compatibility headers/functions at the end, verify both portable and native builds, and compare fixed visual/performance baselines.

**Tech Stack:** C++20, Eigen 3.4, CMake 3.20+, MSVC/Clang/GCC, SDL3, tinyobjloader, stb_image, existing renderer tests.

## Global Constraints

- All first-party floating-point code in `src/` and `tests/` must use `float`; no `double` or Eigen `*d` types remain.
- `external/` is read-only and must not be reformatted or modified.
- Final vector/matrix types are aliases of Eigen float types, not wrapper classes.
- Final code uses Eigen native operations; temporary compatibility functions are allowed only during migration and must be deleted in Task 9.
- Do not add CUDA, GPU APIs, packet rays, SoA, BVH4/BVH8, new materials, or rendering features.
- Preserve OBJ/MTL, raster, ray, path, viewer, and offline CLI behavior.
- Use TDD for every behavioral or API change and keep each task independently reviewable.
- Preserve the unrelated untracked `learning/` directory.

---

## File Map

- Create `src/core/math/types.h`: Eigen float aliases and temporary migration helpers.
- Create `src/core/math/constants.h`: float tolerances grouped by algorithmic purpose.
- Create `src/core/math/transforms.h`: float perspective/look-at matrix factories.
- Delete at completion: `src/core/math/vec2.h`, `vec3.h`, `vec4.h`, `mat4.h`.
- Modify `src/core/color.h`, `image.*`, `random.h`, `timer.h`: float core utilities.
- Modify `src/core/math/ray.h`, `bounds.h`: Eigen domain math.
- Modify `src/scene/*`: float/Eigen camera, geometry, materials, textures, loaders and scene data.
- Modify `src/acceleration/*`, `src/sampling/*`: float BVH and sampling.
- Modify `src/render/*`: float depth, settings, intersection and all three renderers.
- Modify `src/interactive/*`, `src/platform/sdl/*`, `src/main.cpp`, `src/viewer_main.cpp`: float timing/input/application boundaries.
- Modify `tests/renderer_tests.cpp`, `tests/test_framework.h`: Eigen assertions and float tolerances.
- Modify `CMakeLists.txt`: native architecture option and new math headers.
- Create `docs/output/eigen-float-migration-results.md`: baseline, post-migration timings and image checks.
- Modify `README.md`: Eigen float architecture, build option and measured results.

---

### Task 1: Capture Double-Precision Baseline

**Files:**
- Create: `docs/output/eigen-float-migration-results.md`

**Interfaces:**
- Produces: committed baseline commands, timings, image paths and current source token counts.
- Consumes: current `master` renderer before any math code changes.

- [ ] **Step 1: Verify the current baseline**

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
ctest --test-dir build -C Release --output-on-failure
```

Expected: build exits 0, `renderer_tests: all tests passed`, CTest `1/1` passed.

- [ ] **Step 2: Record static baseline counts**

```powershell
rg -o "\bdouble\b" src tests | Measure-Object
rg -o "\bVec3\b" src tests | Measure-Object
rg -n "Eigen::.*d\b" src tests
```

Expected before migration: approximately 396 `double`, 587 `Vec3`, and only orbit-camera Eigen double usage.

- [ ] **Step 3: Render baseline reference images**

```powershell
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --output output\eigen_baseline_mary_raster.png
.\build\bin\renderer.exe --mode path --scene obj_viewer --obj "Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --spp 1 --max-depth 4 --output output\eigen_baseline_mary_path.png
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "Computer Graphics Archive\sponza\sponza.obj" --width 1280 --height 720 --output output\eigen_baseline_sponza_raster.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 8 --max-depth 5 --output output\eigen_baseline_cornell_path.png
```

Expected: all commands exit 0 and produce nonblank PNGs.

- [ ] **Step 4: Measure five-run medians**

Run each Step 3 render command five times after one warmup. Copy the printed `seconds=` values into the results document and calculate the median; do not time process startup with `Measure-Command`.

- [ ] **Step 5: Write and commit the baseline report**

The report must contain a table with scene, command, five samples, median, output path, CPU (`Ryzen 7 9800X3D`), GPU (`RTX 5080`, unused), branch SHA, compiler and build mode.

```powershell
git add -- docs/output/eigen-float-migration-results.md
git commit -m "docs: record double renderer baseline"
```

---

### Task 2: Cut Vector Storage Over to Eigen Float

**Files:**
- Create: `src/core/math/types.h`
- Create: `src/core/math/constants.h`
- Modify temporarily: `src/core/math/vec2.h`, `vec3.h`, `vec4.h`
- Modify: every first-party file containing `.x`, `.y`, `.z`, or `.w` vector field access
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `Scalar`, `Vec2`, `Vec3`, `Vec4`, `Mat3`, `Color`; float constants; temporary `dot/cross/length/normalize/min_components/max_components` helpers. The `Mat4` alias is deliberately deferred to Task 3 so it cannot collide with the still-present custom `Mat4`.
- Consumes: Eigen3 target already linked publicly by `renderer_core`.

- [ ] **Step 1: Add failing Eigen type assertions**

```cpp
#include "core/math/types.h"
#include <type_traits>

static_assert(std::is_same_v<renderer::Scalar, float>);
static_assert(std::is_same_v<renderer::Vec2, Eigen::Vector2f>);
static_assert(std::is_same_v<renderer::Vec3, Eigen::Vector3f>);
static_assert(std::is_same_v<renderer::Vec4, Eigen::Vector4f>);
static_assert(std::is_same_v<renderer::Mat3, Eigen::Matrix3f>);
static_assert(std::is_same_v<renderer::Color, Eigen::Vector3f>);
```

- [ ] **Step 2: Run RED**

Run: `cmake --build build --config Release`

Expected: compile fails because `core/math/types.h` does not exist.

- [ ] **Step 3: Add Eigen aliases and float constants**

Implement the approved aliases except `Mat4`, which remains custom until Task 3. Add temporary helpers implemented only with Eigen methods:

```cpp
inline float dot(const Vec3& a, const Vec3& b) { return a.dot(b); }
inline Vec3 cross(const Vec3& a, const Vec3& b) { return a.cross(b); }
inline float length(const Vec3& v) { return v.norm(); }
inline float length_squared(const Vec3& v) { return v.squaredNorm(); }
inline Vec3 normalize(const Vec3& v) { return v.isZero() ? Vec3::Zero() : v.normalized(); }
inline Vec3 min_components(const Vec3& a, const Vec3& b) { return a.cwiseMin(b); }
inline Vec3 max_components(const Vec3& a, const Vec3& b) { return a.cwiseMax(b); }
```

`vec2.h`, `vec3.h`, `vec4.h` become forwarding includes to `types.h` only.

- [ ] **Step 4: Convert component access mechanically**

Within `src/` and `tests/`, convert vector component fields to Eigen accessors: `.x` to `.x()`, `.y` to `.y()`, `.z` to `.z()`, `.w` to `.w()`. Do not alter unrelated struct fields whose names merely end in x/y/z/w. Compile after each directory group: `core`, `scene`, `acceleration`, `sampling`, `render`, `interactive`, application/tests.

- [ ] **Step 5: Verify GREEN**

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: all tests pass while scalar APIs and matrix implementation may still be double.

- [ ] **Step 6: Commit vector cutover**

```powershell
git add -- src tests
git commit -m "refactor: store renderer vectors with Eigen float"
```

---

### Task 3: Replace the Custom Matrix Implementation

**Files:**
- Create: `src/core/math/transforms.h`
- Modify temporarily: `src/core/math/mat4.h`
- Modify: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `Mat4 = Eigen::Matrix4f`, `make_perspective_matrix`, `make_look_at_matrix`.
- Consumes: Task 2 Eigen vectors and constants.

- [ ] **Step 1: Change matrix tests to the desired API**

```cpp
static_assert(std::is_same_v<renderer::Mat4, Eigen::Matrix4f>);

const renderer::Mat4 transform =
    (Eigen::Translation3f(renderer::Vec3(1.0f, 2.0f, 3.0f)) *
     Eigen::Scaling(2.0f, 3.0f, 4.0f)).matrix();
const renderer::Vec4 p = transform * renderer::Vec4(1.0f, 1.0f, 1.0f, 1.0f);

const renderer::Mat4 projection =
    renderer::make_perspective_matrix(90.0f, 1.0f, 1.0f, 10.0f);
```

- [ ] **Step 2: Run RED**

Expected: compile fails because `Mat4` is still custom and matrix factory functions do not exist.

- [ ] **Step 3: Implement float Eigen matrix factories**

`types.h` adds `using Mat4 = Eigen::Matrix4f`. `transforms.h` validates finite float inputs and constructs the same right-handed `[-1, 1]` depth projection and look-at matrices as the old tests. `mat4.h` becomes a temporary forwarding include to `transforms.h`.

- [ ] **Step 4: Verify matrix and complete tests**

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/core/math tests/renderer_tests.cpp
git commit -m "refactor: use Eigen float matrices"
```

---

### Task 4: Migrate Core Domains, Randomness, Camera and Timing

**Files:**
- Modify: `src/core/color.h`, `image.*`, `random.h`, `timer.h`
- Modify: `src/core/math/ray.h`, `bounds.h`
- Modify: `src/scene/camera.*`
- Modify: `src/interactive/orbit_camera_controller.*`, `frame_rate_counter.*`
- Modify: `src/platform/sdl/sdl_display_backend.h`
- Modify: `src/render/depth_buffer.*`, `renderer.h`, `render_settings.h`
- Test: `tests/renderer_tests.cpp`, `tests/test_framework.h`

**Interfaces:**
- Produces: `Ray::at(float)`, float `Bounds3::intersect`, `PcgRandom::next_float`, float Timer/FPS/depth/camera/input APIs.

- [ ] **Step 1: Add failing scalar type checks**

```cpp
static_assert(std::is_same_v<decltype(renderer::PcgRandom().next_float()), float>);
static_assert(std::is_same_v<decltype(renderer::Timer().elapsed_seconds()), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::RenderResult>().seconds), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::FrameRateSnapshot>().frames_per_second), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::Camera>().viewport_width()), float>);
```

Include `<type_traits>` and `<utility>` for these compile-time checks.

- [ ] **Step 2: Run RED**

Expected: compile fails on missing `next_float` and double return/member types.

- [ ] **Step 3: Implement float APIs and native Eigen operations**

Replace all doubles in listed files with float, use `std::uniform_real_distribution<float>`, `std::chrono::duration<float>`, `std::numeric_limits<float>`, and float literals. Delete orbit `to_eigen/to_vec3`; hold `Vec3 target_` directly. Replace temporary vector helpers in these files with `.dot()`, `.cross()`, `.norm()`, `.squaredNorm()`, `.normalized()`, `.allFinite()`, `cwiseMin/cwiseMax`.

- [ ] **Step 4: Adjust test tolerance intentionally**

Change `nearly_equal` to:

```cpp
inline bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}
```

Keep tighter explicit tolerances where exact behavior requires them; do not globally weaken assertions above `1e-4f` without a failing numerical case.

- [ ] **Step 5: Verify and commit**

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
git add -- src/core src/scene/camera.* src/interactive src/platform/sdl/sdl_display_backend.h src/render/depth_buffer.* src/render/renderer.h src/render/render_settings.h tests
git commit -m "refactor: migrate core renderer scalars to float"
```

---

### Task 5: Migrate Scene, Materials, Textures and Asset Loading

**Files:**
- Modify: `src/scene/light.h`, `material.h`, `primitive.h`, `scene.*`, `mesh.*`
- Modify: `src/scene/texture.*`, `material_evaluator.*`
- Modify: `src/scene/obj_loader.*`, `scene_asset_loader.*`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: float `HitRecord::t`, material fields, texture sampling, TBN and loader conversion boundary.

- [ ] **Step 1: Add failing scene type checks**

```cpp
static_assert(std::is_same_v<decltype(std::declval<renderer::HitRecord>().t), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::Material>().roughness), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::Material>().opacity), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::SurfaceMaterialSample>().opacity), float>);
```

- [ ] **Step 2: Run RED**

Expected: assertions fail because scene scalar members remain double.

- [ ] **Step 3: Migrate scene code**

Convert all scalar fields and methods to float. Replace free helper math with Eigen native operations. Use `cwiseProduct` for colors. At tinyobj/stb boundaries use `static_cast<float>` and never modify vendor code. Replace all `std::pow`, `sqrt`, `floor`, `isfinite` inputs with float values and `f` literals.

- [ ] **Step 4: Preserve numerical fallbacks**

Run focused tests for invalid centers/radii, degenerate normals/UV, alpha, bump and OBJ normal interpolation. Confirm no NaN enters `HitRecord` or texture output.

- [ ] **Step 5: Verify and commit**

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
git add -- src/scene tests/renderer_tests.cpp
git commit -m "refactor: migrate scene math to Eigen float"
```

---

### Task 6: Migrate BVH, Sampling and Scene Queries

**Files:**
- Modify: `src/acceleration/bvh.*`
- Modify: `src/sampling/sampler.*`
- Modify: `src/render/scene_intersector.*`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: float BVH/intersection distances, Eigen-native sampling and ray offsets.

- [ ] **Step 1: Add failing interface checks and a large-coordinate regression**

```cpp
static_assert(std::is_same_v<
    decltype(renderer::offset_ray_origin(
        renderer::Vec3::Zero(), renderer::Vec3::UnitX(), renderer::Vec3::UnitX())),
    renderer::Vec3>);

void test_float_bvh_matches_bruteforce_at_large_coordinates() {
    // Build triangles centered around 100000.0f and compare hit/miss, material id,
    // t within 1e-2f, and shading-normal dot above 0.999f.
}
```

- [ ] **Step 2: Run RED for remaining double APIs or missing fixture**

- [ ] **Step 3: Implement float Eigen algorithms**

Use Eigen indexing (`vector[axis]`) in BVH instead of component helper functions. Use approved float constants for determinant, direction and offset thresholds. Rename all RNG calls to `next_float()` and use Eigen normalized vectors in sampling.

- [ ] **Step 4: Verify BVH/bruteforce and sampling tests**

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/acceleration src/sampling src/render/scene_intersector.* tests/renderer_tests.cpp
git commit -m "refactor: migrate intersections and sampling to float"
```

---

### Task 7: Migrate Raster, Ray, Path and Interactive Sessions

**Files:**
- Modify: `src/render/rasterizer/*`
- Modify: `src/render/raytracer/*`
- Modify: `src/render/pathtracer/*`
- Modify: `src/render/interactive/*`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: float-only rendering pipelines using Eigen native expressions.
- Consumes: Tasks 4-6 float scene, query, RNG and settings APIs.

- [ ] **Step 1: Add failing renderer API checks**

```cpp
static_assert(std::is_same_v<
    decltype(renderer::perspective_correct_weights(
        renderer::Vec3::Ones(), renderer::Vec3::Ones())),
    renderer::Vec3>);
static_assert(std::is_same_v<decltype(std::declval<renderer::RasterVertex>().view), renderer::Vec3>);
```

Add assertions that rendered Image and Framebuffer colors are finite after raster/ray/path fixture renders.

- [ ] **Step 2: Run RED on remaining double signatures**

- [ ] **Step 3: Migrate renderer implementations**

Convert edge functions, barycentrics, depth buffers, path throughput, Fresnel, light attenuation, worker timing and accumulation weights to float. Replace color multiply helpers with `cwiseProduct`; replace vector free functions with Eigen methods. Use `.eval()` when storing a nontrivial Eigen expression whose operands may alias or outlive the expression.

- [ ] **Step 4: Verify focused renderer tests**

Run the complete test executable and specifically inspect failures in near clipping, alpha, direct light, inverse-square falloff, reflection/refraction and path accumulation before adjusting any tolerance.

- [ ] **Step 5: Commit**

```powershell
git add -- src/render tests/renderer_tests.cpp
git commit -m "refactor: render with Eigen float math"
```

---

### Task 8: Migrate Application Boundaries and Remove First-Party Double

**Files:**
- Modify: `src/main.cpp`, `src/viewer_main.cpp`
- Modify: `src/platform/sdl/sdl_display_backend.*`
- Modify: remaining `src/` and `tests/` files reported by static search
- Test: `tests/renderer_tests.cpp`, `tests/test_framework.h`

**Interfaces:**
- Produces: float CLI/viewer/input/timing boundaries and zero first-party `double` tokens.

- [ ] **Step 1: Add static scan as an execution gate**

```powershell
$doubleHits = rg -n "\bdouble\b|Eigen::(Vector[234]d|Matrix[^ ]*d)" src tests
if ($LASTEXITCODE -eq 0) { $doubleHits; exit 1 }
exit 0
```

Expected before implementation: command exits 1 and lists remaining application/test locations.

- [ ] **Step 2: Convert every reported first-party location**

Use `std::stof`, `static_cast<float>`, `std::chrono::duration<float>`, float SDL/input fields and `f` literals. Preserve integer parsing and dimensions. Replace any remaining temporary helper calls with Eigen native operations.

- [ ] **Step 3: Verify zero-token and full tests**

```powershell
rg -n "\bdouble\b|Eigen::(Vector[234]d|Matrix[^ ]*d)" src tests
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: `rg` returns exit code 1 with no output; build/tests pass.

- [ ] **Step 4: Commit**

```powershell
git add -- src tests
git commit -m "refactor: remove remaining first-party double math"
```

---

### Task 9: Remove Compatibility Math and Add Native Architecture Option

**Files:**
- Modify: `CMakeLists.txt`
- Modify: all includes of old math headers
- Modify: `src/core/math/types.h`
- Delete: `src/core/math/vec2.h`, `vec3.h`, `vec4.h`, `mat4.h`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: final Eigen-native API and `RENDERER_NATIVE_ARCH` CMake option.

- [ ] **Step 1: Add a source-policy test/check**

```powershell
rg -n "core/math/(vec2|vec3|vec4|mat4)\.h|(^|[^.[:alnum:]_])(dot|cross|normalize|length|length_squared|min_components|max_components)\s*\(" src tests
```

Expected before cleanup: forwarding includes and temporary helper calls are listed.

- [ ] **Step 2: Replace includes and operations**

All users include `core/math/types.h`, `constants.h`, `transforms.h`, `ray.h`, or `bounds.h` according to responsibility. Remove temporary helpers from `types.h`; use Eigen methods at every call site. Delete four forwarding headers.

- [ ] **Step 3: Add CMake option**

```cmake
option(RENDERER_NATIVE_ARCH "Enable native CPU instruction tuning" OFF)
function(target_enable_native_arch target_name)
    if(RENDERER_NATIVE_ARCH)
        if(MSVC)
            target_compile_options(${target_name} PRIVATE /arch:AVX2)
        else()
            target_compile_options(${target_name} PRIVATE -march=native)
        endif()
    endif()
endfunction()

foreach(target_name IN ITEMS renderer_core renderer viewer renderer_tests)
    target_enable_native_arch(${target_name})
endforeach()
```

Apply the architecture option to all first-party compiled targets; do not attach it to imported third-party targets.

- [ ] **Step 4: Verify final source policy and portable build**

```powershell
rg -n "\bdouble\b|Eigen::(Vector[234]d|Matrix[^ ]*d)|core/math/(vec2|vec3|vec4|mat4)\.h" src tests
cmake -S . -B build-portable -DRENDERER_NATIVE_ARCH=OFF
cmake --build build-portable --config Release
ctest --test-dir build-portable -C Release --output-on-failure
```

Expected: source scan has no output; portable tests 100% pass.

- [ ] **Step 5: Verify native build**

```powershell
cmake -S . -B build-native -DRENDERER_NATIVE_ARCH=ON
cmake --build build-native --config Release
ctest --test-dir build-native -C Release --output-on-failure
```

Expected: native tests 100% pass and MSVC compile command uses `/arch:AVX2`.

- [ ] **Step 6: Commit**

```powershell
git add -- CMakeLists.txt src tests
git commit -m "build: finalize Eigen native float math"
```

---

### Task 10: Visual Regression, Benchmark and Documentation

**Files:**
- Modify: `docs/output/eigen-float-migration-results.md`
- Modify: `README.md`

**Interfaces:**
- Produces: measured portable/native results and user-facing build instructions.

- [ ] **Step 1: Render post-migration images with `build-native`**

Repeat Task 1 Step 3 using `build-native\bin\renderer.exe` and output filenames prefixed `eigen_float_`. Also run:

```powershell
.\build-native\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\mary\Marry.obj" --mode path --width 1920 --height 1080 --frames 1
```

- [ ] **Step 2: Inspect image correctness**

Mary must remain smooth and textured in raster/path; Sponza must retain diffuse/bump detail without speckle; CornellBox must remain nonblank with emissive panel and colored walls. Reject NaN, magenta fallback, missing geometry, changed winding or widespread edge cracks.

- [ ] **Step 3: Measure five-run float/native medians**

Repeat the exact baseline procedure. For each scene calculate `(float_native_median / double_baseline_median - 1) * 100%`. Any slowdown above 5% is a blocker requiring profiling before completion.

- [ ] **Step 4: Update README and results**

Document Eigen float aliases, native option, portable/native commands, measured table, image validation, remaining optimization opportunities and the fact that RTX 5080 remains unused by this CPU backend.

- [ ] **Step 5: Run complete final verification**

```powershell
ctest --test-dir build-portable -C Release --output-on-failure
ctest --test-dir build-native -C Release --output-on-failure
rg -n "\bdouble\b|Eigen::(Vector[234]d|Matrix[^ ]*d)|core/math/(vec2|vec3|vec4|mat4)\.h" src tests
git diff --check
git status --short --branch
```

Expected: both test matrices pass, source scan has no hits, no whitespace errors, and only intentionally ignored/untracked user files remain.

- [ ] **Step 6: Request review and commit documentation**

Use `superpowers:requesting-code-review` against the approved spec and full branch diff. Fix Critical/Important findings with failing tests, rerun Step 5, then:

```powershell
git add -- README.md docs/output/eigen-float-migration-results.md
git commit -m "docs: report Eigen float migration results"
```

- [ ] **Step 7: Complete the branch**

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Do not merge or push until the user chooses the integration option.
