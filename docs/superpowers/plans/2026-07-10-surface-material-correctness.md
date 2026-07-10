# Surface Material Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OBJ/MTL surfaces shade consistently across raster, ray, and path modes by preserving smooth normals, sharing material evaluation, supporting alpha cutout and bump maps, and adding direct lights to the path tracer.

**Architecture:** `Triangle` and `HitRecord` become the single source of geometric and interpolated surface data. A shared material evaluator converts textures and MTL constants into a `SurfaceMaterialSample`, while a shared `SceneIntersector` owns alpha-aware ray queries and origin offsets. Raster keeps its edge-function pipeline but consumes the same surface sample after near-plane clipping and perspective-correct interpolation.

**Tech Stack:** C++20, CMake 3.20+, tinyobjloader, stb_image, SDL3, Eigen3, existing `renderer_tests` executable.

## Global Constraints

- Keep SDL3 as a window/input/presentation layer; do not add OpenGL, Vulkan, DirectX, SDL GPU, CUDA, OptiX, or RT Core.
- Limit asset work to OBJ/MTL: `map_Kd`, `d`, `Tr`, `map_d`, `bump`, and `map_bump`.
- Alpha support is cutout only; do not implement alpha blending or transmission in this plan.
- Bump maps are scalar height maps; do not interpret them as tangent-space normal maps.
- Imported OBJ materials default to two-sided shading.
- Decode color textures from sRGB to linear; keep opacity and height textures linear.
- Preserve all pre-existing user changes and the ignored `Computer Graphics Archive` assets.
- Use test-first development for every new behavior and run `renderer_tests.exe` after each green step.

---

## File Map

- `src/scene/primitive.h`: triangle vertex attributes, geometric/shading normals, UV/TBN interpolation, hit data.
- `src/scene/material.h`: opacity, alpha cutoff, bump slots, bump strength, and sidedness.
- `src/scene/texture.h`, `src/scene/texture.cpp`: image decoding semantics, scalar sampling, texel size.
- `src/scene/material_evaluator.h`, `src/scene/material_evaluator.cpp`: shared base-color, opacity, and bump evaluation.
- `src/scene/scene_asset_loader.h`, `src/scene/scene_asset_loader.cpp`: OBJ normals, MTL texture slots, warnings and fallback behavior.
- `src/render/scene_intersector.h`, `src/render/scene_intersector.cpp`: nearest opaque hit, alpha-aware occlusion, and ray-origin offset.
- `src/render/raytracer/raytracer_renderer.*`: consume shared intersections and material samples.
- `src/render/pathtracer/pathtracer_renderer.*`: consume shared intersections and add delta-light direct illumination.
- `src/render/rasterizer/raster_geometry.h`, `src/render/rasterizer/raster_geometry.cpp`: near-plane clipping and perspective weights.
- `src/render/rasterizer/rasterizer_renderer.cpp`: smooth/bump normals, alpha cutout, clipping and perspective interpolation.
- `src/main.cpp`, `src/viewer_main.cpp`: display deduplicated asset warnings.
- `tests/renderer_tests.cpp`: focused unit/integration regression tests.
- `README.md`: supported MTL fields, current limits, and verification commands.
- `CMakeLists.txt`: register new implementation files.

---

### Task 1: Preserve the Existing map_Kd Baseline

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Modify: `src/main.cpp`
- Modify: `src/scene/material.h`
- Modify: `src/scene/primitive.h`
- Modify: `src/scene/scene.h`
- Modify: `src/scene/scene_asset_loader.cpp`
- Modify: `src/scene/texture.h`
- Create: `src/scene/texture.cpp`
- Create: `external/stb/stb_image.h`
- Modify: `src/render/rasterizer/rasterizer_renderer.cpp`
- Modify: `src/render/raytracer/raytracer_renderer.cpp`
- Modify: `src/render/pathtracer/pathtracer_renderer.cpp`
- Modify: `src/render/pathtracer/pathtracer_renderer.h`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `ImageTexture::load(const std::string&)`, `ImageTexture::sample(const Vec2&)`, and `sample_material_base_color(...)` as the baseline that Task 3 replaces with explicit encoding and shared material evaluation.

- [ ] **Step 1: Verify the current baseline compiles and its existing tests pass**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: build exits 0 and prints `renderer_tests: all tests passed`.

- [ ] **Step 2: Run map_Kd smoke tests before changing its API**

Run:

```powershell
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "Computer Graphics Archive\sponza\sponza.obj" --width 160 --height 100 --output output\sponza_map_kd_baseline.png
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "Computer Graphics Archive\mary\Marry.obj" --width 160 --height 100 --output output\mary_map_kd_baseline.png
```

Expected: both commands exit 0 and write non-empty PNG files.

- [ ] **Step 3: Commit only the pre-existing map_Kd implementation**

```powershell
git add -- CMakeLists.txt README.md external/stb/stb_image.h src/main.cpp src/render/pathtracer/pathtracer_renderer.cpp src/render/pathtracer/pathtracer_renderer.h src/render/rasterizer/rasterizer_renderer.cpp src/render/raytracer/raytracer_renderer.cpp src/scene/material.h src/scene/primitive.h src/scene/scene.h src/scene/scene_asset_loader.cpp src/scene/texture.h src/scene/texture.cpp tests/renderer_tests.cpp
git commit -m "feat: support OBJ diffuse texture maps"
```

Expected: commit succeeds without staging `learning/` or `Computer Graphics Archive/`.

---

### Task 2: Preserve OBJ Smooth Normals in Surface Hits

**Files:**
- Modify: `src/scene/primitive.h`
- Modify: `src/scene/scene_asset_loader.cpp`
- Modify: `src/scene/obj_loader.cpp`
- Modify: `src/render/rasterizer/rasterizer_renderer.cpp`
- Modify: `src/render/raytracer/raytracer_renderer.cpp`
- Modify: `src/render/pathtracer/pathtracer_renderer.cpp`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `TriangleVertex`, `HitRecord::geometric_normal`, `HitRecord::shading_normal`, `HitRecord::tangent`, `HitRecord::bitangent`, `Triangle::vertex(int)`, and `Triangle::interpolate_shading_normal(...)`.
- Consumes: existing `Vec2`, `Vec3`, `Ray`, and tinyobjloader indices.

- [ ] **Step 1: Add failing triangle and OBJ-loader normal tests**

Add tests equivalent to:

```cpp
void test_triangle_interpolates_shading_normal_separately_from_geometry() {
    const renderer::Triangle triangle(
        renderer::TriangleVertex{renderer::Vec3(-1, -1, -1), renderer::Vec2(0, 0), renderer::normalize(renderer::Vec3(0, 1, 1)), true},
        renderer::TriangleVertex{renderer::Vec3(1, -1, -1), renderer::Vec2(1, 0), renderer::normalize(renderer::Vec3(1, 0, 1)), true},
        renderer::TriangleVertex{renderer::Vec3(0, 1, -1), renderer::Vec2(0.5, 1), renderer::normalize(renderer::Vec3(0, 0, 1)), true},
        0);
    renderer::HitRecord hit;
    RENDER_CHECK(triangle.intersect(renderer::Ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, -1)), 1e-6, 10.0, hit));
    RENDER_CHECK(renderer::dot(hit.geometric_normal, renderer::Vec3(0, 0, 1)) > 0.999);
    RENDER_CHECK(renderer::dot(hit.shading_normal, hit.geometric_normal) > 0.0);
    RENDER_CHECK(renderer::length(hit.shading_normal - hit.geometric_normal) > 0.01);
}

void test_scene_asset_loader_preserves_obj_vertex_normals() {
    // Write a one-triangle OBJ with three distinct vn records and f v/vt/vn indices.
    const renderer::LoadedScene loaded = renderer::load_scene_asset("test_smooth_normals.obj", 64, 64);
    renderer::HitRecord hit;
    RENDER_CHECK(loaded.scene.triangles[0].intersect(test_ray, 1e-6, 10.0, hit));
    RENDER_CHECK(renderer::length(hit.shading_normal - hit.geometric_normal) > 0.01);
}
```

- [ ] **Step 2: Run the suite and verify RED**

Run: `.\build\bin\renderer_tests.exe`

Expected: compilation fails because `TriangleVertex`, `geometric_normal`, and `shading_normal` do not exist.

- [ ] **Step 3: Implement triangle vertex data and robust normal orientation**

Implement these public shapes in `primitive.h`:

```cpp
struct TriangleVertex {
    Vec3 position;
    Vec2 uv;
    Vec3 normal;
    bool has_normal = false;
};

struct HitRecord {
    double t = 0.0;
    Vec3 position;
    Vec2 uv;
    Vec3 geometric_normal;
    Vec3 shading_normal;
    Vec3 tangent;
    Vec3 bitangent;
    int material_id = -1;
    bool front_face = true;

    void set_normals(const Ray& ray, const Vec3& outward_geometric, const Vec3& outward_shading) {
        front_face = dot(ray.direction, outward_geometric) < 0.0;
        geometric_normal = front_face ? outward_geometric : -outward_geometric;
        Vec3 corrected = dot(outward_shading, outward_geometric) < 0.0 ? -outward_shading : outward_shading;
        shading_normal = front_face ? corrected : -corrected;
    }
};
```

Keep legacy position-only and position-plus-UV constructors by delegating to the `TriangleVertex` constructor. Compute a finite normalized face normal, interpolate valid vertex normals by barycentric weight, then fall back to the face normal. Derive tangent/bitangent from position and UV edges; use a stable orthonormal basis when the UV determinant is near zero.

- [ ] **Step 4: Load `vn` indices and migrate renderer normal reads**

Add:

```cpp
Vec3 normal_from_index(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index, bool& valid);
```

Construct each loaded triangle from three `TriangleVertex` values. Replace renderer and test reads of `hit.normal` with `hit.shading_normal`; use `hit.geometric_normal` only for orientation and ray offsets. Raster should call `triangle.interpolate_shading_normal(...)` instead of using one face normal.

- [ ] **Step 5: Build and verify GREEN**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: build exits 0 and all tests pass.

- [ ] **Step 6: Commit smooth-surface data**

```powershell
git add -- src/scene/primitive.h src/scene/scene_asset_loader.cpp src/scene/obj_loader.cpp src/render/rasterizer/rasterizer_renderer.cpp src/render/raytracer/raytracer_renderer.cpp src/render/pathtracer/pathtracer_renderer.cpp tests/renderer_tests.cpp
git commit -m "feat: preserve smooth OBJ vertex normals"
```

---

### Task 3: Add Encoded Textures and Shared Material Evaluation

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `src/scene/material.h`
- Modify: `src/scene/texture.h`
- Modify: `src/scene/texture.cpp`
- Create: `src/scene/material_evaluator.h`
- Create: `src/scene/material_evaluator.cpp`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `TextureEncoding`, `ImageTexture::load(path, encoding)`, `ImageTexture::sample_scalar`, `ImageTexture::texel_size`, `SurfaceMaterialSample`, `evaluate_surface_material`, and `sample_material_opacity`.
- Consumes: Task 2 `HitRecord` TBN and shading normal.

- [ ] **Step 1: Add failing encoding, opacity, and bump tests**

```cpp
void test_texture_encoding_distinguishes_srgb_from_linear() {
    write_single_pixel_ppm("test_encoding.ppm", 128, 128, 128);
    const auto srgb = renderer::ImageTexture::load("test_encoding.ppm", renderer::TextureEncoding::Srgb);
    const auto linear = renderer::ImageTexture::load("test_encoding.ppm", renderer::TextureEncoding::Linear);
    RENDER_CHECK(srgb.sample(renderer::Vec2()).x < 0.25);
    RENDER_CHECK(linear.sample(renderer::Vec2()).x > 0.49);
}

void test_material_evaluator_combines_opacity_and_perturbs_bump_normal() {
    renderer::Scene scene;
    scene.textures.emplace_back(2, 2, height_pixels);
    renderer::Material material;
    material.opacity = 0.8;
    material.opacity_texture_id = 0;
    material.bump_texture_id = 0;
    material.bump_scale = 1.0;
    renderer::HitRecord hit = make_test_surface_interaction();
    const renderer::SurfaceMaterialSample sample = renderer::evaluate_surface_material(scene, material, hit);
    RENDER_CHECK(sample.opacity < 0.8);
    RENDER_CHECK(renderer::length(sample.shading_normal - hit.shading_normal) > 0.01);
    RENDER_CHECK(renderer::dot(sample.shading_normal, hit.geometric_normal) > 0.0);
}
```

- [ ] **Step 2: Run the suite and verify RED**

Run: `.\build\bin\renderer_tests.exe`

Expected: compilation fails because `TextureEncoding`, opacity fields, and `evaluate_surface_material` do not exist.

- [ ] **Step 3: Implement texture encoding and scalar access**

```cpp
enum class TextureEncoding { Srgb, Linear };

class ImageTexture {
public:
    static ImageTexture load(const std::string& path, TextureEncoding encoding = TextureEncoding::Srgb);
    Color sample(const Vec2& uv) const;
    double sample_scalar(const Vec2& uv) const;
    Vec2 texel_size() const;
};
```

Decode sRGB channels with the standard piecewise transfer function (`c / 12.92` below 0.04045, otherwise `pow((c + 0.055) / 1.055, 2.4)`). Linear textures use `byte / 255.0`. Define scalar sampling as luminance `0.2126 R + 0.7152 G + 0.0722 B`.

- [ ] **Step 4: Implement material fields and shared evaluator**

```cpp
struct Material {
    // existing fields
    double opacity = 1.0;
    double alpha_cutoff = 0.5;
    double bump_scale = 1.0;
    bool two_sided = true;
    int diffuse_texture_id = -1;
    int opacity_texture_id = -1;
    int bump_texture_id = -1;
};

struct SurfaceMaterialSample {
    Color base_color;
    double opacity = 1.0;
    Vec3 shading_normal;
};

SurfaceMaterialSample evaluate_surface_material(
    const Scene& scene, const Material& material, const HitRecord& hit);
double sample_material_opacity(const Scene& scene, const Material& material, const Vec2& uv);
```

The evaluator multiplies `base_color` by the diffuse texture, multiplies constant opacity by a linear opacity texture, computes bump finite differences using `texel_size()`, transforms them through hit TBN, normalizes the result, and flips it when necessary to keep it in the geometric-normal hemisphere.

- [ ] **Step 5: Register the evaluator and verify GREEN**

Add `src/scene/material_evaluator.cpp` to `renderer_core`, then run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: all tests pass.

- [ ] **Step 6: Commit material evaluation**

```powershell
git add -- CMakeLists.txt src/scene/material.h src/scene/texture.h src/scene/texture.cpp src/scene/material_evaluator.h src/scene/material_evaluator.cpp tests/renderer_tests.cpp
git commit -m "feat: unify textured surface evaluation"
```

---

### Task 4: Import OBJ Alpha, Bump, and Diagnostics

**Files:**
- Modify: `src/scene/scene_asset_loader.h`
- Modify: `src/scene/scene_asset_loader.cpp`
- Modify: `src/main.cpp`
- Modify: `src/viewer_main.cpp`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `LoadedScene::warnings`, encoded texture-cache keys, and populated opacity/bump material fields.
- Consumes: tinyobjloader `material_t::dissolve`, `alpha_texname`, `bump_texname`, and `bump_texopt.bump_multiplier`.

- [ ] **Step 1: Add a failing MTL integration test**

```cpp
void test_scene_asset_loader_imports_alpha_bump_and_warnings() {
    // MTL: d 0.8, map_d opacity.ppm, bump -bm 0.25 height.ppm, map_Kd color.ppm.
    const renderer::LoadedScene loaded = renderer::load_scene_asset("test_surface_maps.obj", 64, 64);
    const renderer::Material& material = loaded.scene.materials[0];
    RENDER_CHECK(nearly_equal(material.opacity, 0.8));
    RENDER_CHECK(material.opacity_texture_id >= 0);
    RENDER_CHECK(material.bump_texture_id >= 0);
    RENDER_CHECK(nearly_equal(material.bump_scale, 0.25));
    RENDER_CHECK(loaded.scene.textures[material.opacity_texture_id].sample(renderer::Vec2()).x > 0.49);

    const renderer::LoadedScene missing = renderer::load_scene_asset("test_missing_optional_map.obj", 64, 64);
    RENDER_CHECK(!missing.warnings.empty());
}
```

- [ ] **Step 2: Run and verify RED**

Run: `.\build\bin\renderer_tests.exe`

Expected: compilation fails because `LoadedScene::warnings` and new material fields are unavailable to the loader.

- [ ] **Step 3: Implement semantic texture loading and warnings**

Replace the texture cache key with:

```cpp
struct TextureCacheKey {
    std::string normalized_path;
    TextureEncoding encoding;
    bool operator==(const TextureCacheKey&) const = default;
};

struct TextureCacheKeyHash {
    std::size_t operator()(const TextureCacheKey& key) const {
        const std::size_t path_hash = std::hash<std::string>{}(key.normalized_path);
        const std::size_t encoding_hash = std::hash<int>{}(static_cast<int>(key.encoding));
        return path_hash ^ (encoding_hash + 0x9e3779b9U + (path_hash << 6U) + (path_hash >> 2U));
    }
};
```

Load `diffuse_texname` as `Srgb`, and `alpha_texname`/`bump_texname` as `Linear`. Set `opacity = clamp(source.dissolve, 0.0, 1.0)`, set `bump_scale` from `source.bump_texopt.bump_multiplier`, and default imported materials to `two_sided = true`.

Add deduplicated warnings through a local `std::unordered_set<std::string>` and append the first occurrence to `LoadedScene::warnings`. Preserve tinyobjloader's parser warning as a loaded-scene warning when parsing succeeds. Record one category-level warning when indexed normals are missing/invalid and geometry-normal fallback is used. Record one material-level warning when a bump map is present but a triangle has degenerate UVs; deduplicate both categories so large assets do not flood stderr.

- [ ] **Step 4: Display warnings in both entry points**

After `load_scene_asset`, print each warning once:

```cpp
for (const std::string& warning : loaded.warnings) {
    std::cerr << "warning: " << warning << '\n';
}
```

- [ ] **Step 5: Build and verify GREEN**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: all tests pass and the missing-map fixture records one deduplicated warning.

- [ ] **Step 6: Commit OBJ/MTL material import**

```powershell
git add -- src/scene/scene_asset_loader.h src/scene/scene_asset_loader.cpp src/main.cpp src/viewer_main.cpp tests/renderer_tests.cpp
git commit -m "feat: import OBJ alpha and bump maps"
```

---

### Task 5: Share Alpha-Aware Ray Queries and Origin Offsets

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/render/scene_intersector.h`
- Create: `src/render/scene_intersector.cpp`
- Modify: `src/render/raytracer/raytracer_renderer.h`
- Modify: `src/render/raytracer/raytracer_renderer.cpp`
- Modify: `src/render/pathtracer/pathtracer_renderer.h`
- Modify: `src/render/pathtracer/pathtracer_renderer.cpp`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `SceneIntersector::intersect`, `SceneIntersector::occluded`, and `offset_ray_origin`.
- Consumes: Task 3 `sample_material_opacity` and Task 2 geometric normals.

- [ ] **Step 1: Add failing transparent-hit and offset tests**

```cpp
void test_scene_intersector_skips_alpha_cutout_hits() {
    renderer::Scene scene = make_transparent_front_opaque_back_scene();
    renderer::SceneIntersector intersector(scene);
    renderer::HitRecord hit;
    RENDER_CHECK(intersector.intersect(test_ray, 0.0, 100.0, hit));
    RENDER_CHECK(hit.material_id == opaque_material_id);
}

void test_two_sided_surface_orients_both_normals_toward_the_ray() {
    renderer::Scene scene = make_back_facing_triangle_scene(true);
    renderer::SceneIntersector intersector(scene);
    renderer::HitRecord hit;
    RENDER_CHECK(intersector.intersect(back_side_ray, 0.0, 100.0, hit));
    RENDER_CHECK(renderer::dot(hit.geometric_normal, back_side_ray.direction) < 0.0);
    RENDER_CHECK(renderer::dot(hit.shading_normal, hit.geometric_normal) > 0.0);
}

void test_offset_ray_origin_moves_to_outgoing_side() {
    const renderer::Vec3 p(1000.0, 0.0, 0.0);
    const renderer::Vec3 n(1.0, 0.0, 0.0);
    RENDER_CHECK(renderer::offset_ray_origin(p, n, n).x > p.x);
    RENDER_CHECK(renderer::offset_ray_origin(p, n, -n).x < p.x);
}
```

- [ ] **Step 2: Run and verify RED**

Run: `.\build\bin\renderer_tests.exe`

Expected: compilation fails because `SceneIntersector` and `offset_ray_origin` do not exist.

- [ ] **Step 3: Implement the shared query**

```cpp
class SceneIntersector {
public:
    explicit SceneIntersector(const Scene& scene);
    bool intersect(const Ray& ray, double t_min, double t_max, HitRecord& hit) const;
    bool occluded(const Ray& ray, double t_min, double t_max) const;
private:
    bool intersect_nearest(const Ray& ray, double t_min, double t_max, HitRecord& hit) const;
    const Scene& scene_;
    Bvh bvh_;
};

Vec3 offset_ray_origin(const Vec3& position, const Vec3& geometric_normal, const Vec3& direction);
```

`intersect` repeatedly calls `intersect_nearest` on the original ray, tests opacity against `alpha_cutoff`, and advances `t_min` with `std::nextafter(hit.t, t_max)`. Stop after 64 transparent layers and return false. `occluded` delegates to `intersect`. Offset magnitude is `1e-7 * max(1.0, max(abs(position components)))` and its sign follows `dot(direction, geometric_normal)`.

- [ ] **Step 4: Migrate ray and path renderers**

Construct one `SceneIntersector` per render call. Remove duplicated `hit_scene` functions. Evaluate `SurfaceMaterialSample` once per hit and use its base color and bump-adjusted shading normal for ray/path lighting, reflection, refraction, and scattering. Every reflection, refraction, scatter, and shadow ray origin must use `offset_ray_origin(hit.position, hit.geometric_normal, direction)`.

- [ ] **Step 5: Build and verify GREEN**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: all tests pass, including existing BVH and shadow tests.

- [ ] **Step 6: Commit shared scene queries**

```powershell
git add -- CMakeLists.txt src/render/scene_intersector.h src/render/scene_intersector.cpp src/render/raytracer/raytracer_renderer.h src/render/raytracer/raytracer_renderer.cpp src/render/pathtracer/pathtracer_renderer.h src/render/pathtracer/pathtracer_renderer.cpp tests/renderer_tests.cpp
git commit -m "feat: add alpha-aware scene intersections"
```

---

### Task 6: Add Direct Point and Directional Lights to Path Tracing

**Files:**
- Modify: `src/render/pathtracer/pathtracer_renderer.h`
- Modify: `src/render/pathtracer/pathtracer_renderer.cpp`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: private `estimate_direct_lighting(...)` using `SceneIntersector` visibility.
- Consumes: `SurfaceMaterialSample`, point/directional lights, and `offset_ray_origin`.

- [ ] **Step 1: Add failing direct-light tests**

```cpp
void test_pathtracer_receives_unoccluded_directional_light() {
    renderer::Scene lit = make_diffuse_test_triangle_scene();
    lit.environment = renderer::Color();
    lit.directional_lights.push_back({renderer::Vec3(0, 0, -1), renderer::Color(2, 2, 2)});
    const renderer::Color visible = render_center_path_pixel(lit, 1, 1);
    lit.triangles.push_back(make_shadow_blocker());
    const renderer::Color shadowed = render_center_path_pixel(lit, 1, 1);
    RENDER_CHECK(renderer::length(visible) > renderer::length(shadowed) + 0.1);
}

void test_pathtracer_point_light_uses_inverse_square_falloff() {
    const renderer::Color near_value = render_with_point_light_distance(2.0);
    const renderer::Color far_value = render_with_point_light_distance(4.0);
    RENDER_CHECK(near_value.x > far_value.x * 3.5);
}
```

- [ ] **Step 2: Run and verify RED**

Run: `.\build\bin\renderer_tests.exe`

Expected: assertions fail because path tracing currently ignores both light arrays.

- [ ] **Step 3: Implement delta-light direct illumination**

```cpp
Color PathTracerRenderer::estimate_direct_lighting(
    const Scene& scene,
    const SceneIntersector& intersector,
    const HitRecord& hit,
    const SurfaceMaterialSample& surface) const;
```

For each direction light, use `light_dir = normalize(-light.direction)`. For each point light, use `to_light`, `distance_squared`, normalized direction, and `light.intensity / distance_squared`. Skip non-positive `dot(surface.shading_normal, light_dir)`. Send an alpha-aware shadow ray from the offset origin. Return `base_color * incoming_radiance * n_dot_l / pi` summed over visible lights.

In `trace_path`, add this result only for `MaterialType::Diffuse`; keep cosine-weighted indirect scattering unchanged.

- [ ] **Step 4: Build and verify GREEN**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: direct-light tests and all existing path tests pass.

- [ ] **Step 5: Commit path direct lighting**

```powershell
git add -- src/render/pathtracer/pathtracer_renderer.h src/render/pathtracer/pathtracer_renderer.cpp tests/renderer_tests.cpp
git commit -m "feat: sample direct lights in path tracing"
```

---

### Task 7: Make Raster Attributes and Near Clipping Correct

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/render/rasterizer/raster_geometry.h`
- Create: `src/render/rasterizer/raster_geometry.cpp`
- Modify: `src/render/rasterizer/rasterizer_renderer.cpp`
- Test: `tests/renderer_tests.cpp`

**Interfaces:**
- Produces: `RasterVertex`, `clip_triangle_to_near_plane`, and `perspective_correct_weights`.
- Consumes: `TriangleVertex`, `SurfaceMaterialSample`, and the existing camera basis.

- [ ] **Step 1: Add failing perspective and clipping tests**

```cpp
void test_perspective_correct_weights_favor_near_vertex() {
    const renderer::Vec3 corrected = renderer::perspective_correct_weights(renderer::Vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), renderer::Vec3(1.0, 2.0, 4.0));
    RENDER_CHECK(corrected.x > corrected.y);
    RENDER_CHECK(corrected.y > corrected.z);
    RENDER_CHECK(nearly_equal(corrected.x + corrected.y + corrected.z, 1.0));
}

void test_near_plane_clipping_keeps_visible_triangle_portion() {
    const std::vector<renderer::RasterVertex> clipped = renderer::clip_triangle_to_near_plane(vertices_with_one_behind, 1e-4);
    RENDER_CHECK(clipped.size() == 4);
    for (const auto& vertex : clipped) RENDER_CHECK(vertex.view.z >= 1e-4);
}
```

Add a render-level test asserting a triangle crossing the near plane produces at least one non-background pixel.

Add one render-level backface test: a two-sided imported-style material remains visible from the back, while `two_sided = false` leaves the same back-facing triangle at the background color.

- [ ] **Step 2: Run and verify RED**

Run: `.\build\bin\renderer_tests.exe`

Expected: compilation fails because the raster geometry helpers do not exist.

- [ ] **Step 3: Implement clipping and perspective helpers**

```cpp
struct RasterVertex {
    Vec3 view;
    Vec3 world;
    Vec2 uv;
    Vec3 normal;
};

std::vector<RasterVertex> clip_triangle_to_near_plane(
    const std::array<RasterVertex, 3>& triangle, double near_z);
Vec3 perspective_correct_weights(const Vec3& screen_weights, const Vec3& view_depths);
```

Use Sutherland-Hodgman clipping against `view.z >= near_z`. Interpolate world, view, UV, and normal at each crossing. Return a polygon of zero, three, or four vertices; triangulate four vertices as `(0,1,2)` and `(0,2,3)` in the renderer. Compute corrected weight `wi / zi` divided by the sum.

- [ ] **Step 4: Consume shared material data in raster**

For each clipped triangle, interpolate UV, world position, and shading normal with corrected weights. Fill a `HitRecord` with the original triangle's geometric normal and TBN, orient it according to camera view direction, and call `evaluate_surface_material`. Skip single-sided backfaces. Evaluate opacity before writing depth; cutout pixels leave both image and depth unchanged.

- [ ] **Step 5: Build and verify GREEN**

Run:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Expected: all clipping, interpolation, alpha, and existing raster tests pass.

- [ ] **Step 6: Commit raster correctness**

```powershell
git add -- CMakeLists.txt src/render/rasterizer/raster_geometry.h src/render/rasterizer/raster_geometry.cpp src/render/rasterizer/rasterizer_renderer.cpp tests/renderer_tests.cpp
git commit -m "fix: correct raster surface interpolation"
```

---

### Task 8: Documentation and Full Scene Regression

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-10-surface-material-correctness.md`

**Interfaces:**
- Consumes: all completed rendering behavior.
- Produces: accurate user-facing support matrix and recorded verification evidence.

- [ ] **Step 1: Update README support and limitations**

Document:

```text
Supported OBJ/MTL surface data:
- vertex normals (`vn`) with smooth interpolation
- diffuse color textures (`map_Kd`, sRGB)
- alpha cutout (`d`, `Tr`, `map_d`, linear)
- height bump maps (`bump`, `map_bump`, linear)
- two-sided imported materials
- point/directional direct lighting in raster, ray, and path modes
```

Keep alpha blend, normal maps, glTF, PBR, area-light MIS, mipmaps and denoising in the explicit limitations section.

- [ ] **Step 2: Run full automated verification**

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: all commands exit 0. CRLF conversion notices are acceptable; whitespace errors are not.

- [ ] **Step 3: Render Mary, Sponza, and CornellBox smoke images**

```powershell
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "Computer Graphics Archive\mary\Marry.obj" --width 640 --height 360 --output output\mary_raster_surface.png
.\build\bin\renderer.exe --mode path --scene obj_viewer --obj "Computer Graphics Archive\mary\Marry.obj" --width 640 --height 360 --spp 16 --max-depth 4 --output output\mary_path_surface.png
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "Computer Graphics Archive\sponza\sponza.obj" --width 640 --height 360 --output output\sponza_surface.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 320 --height 320 --spp 8 --max-depth 5 --output output\cornell_regression.png
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\mary\Marry.obj" --mode path --width 640 --height 360 --frames 2
```

Expected: all commands exit 0; Mary is smooth and directionally lit in path mode; Sponza textures remain visible; CornellBox remains nonblank; viewer reports `mode=path frames=2`.

- [ ] **Step 4: Inspect generated images**

Open all four PNGs. Reject the change if Mary still shows per-triangle lighting on smooth skin/clothing, if path Mary is environment-only dark, if Sponza becomes untextured, or if CornellBox becomes blank/magenta.

- [ ] **Step 5: Request code review and address only verified findings**

Use `superpowers:requesting-code-review`, review the full branch diff against `2026-07-10-surface-material-correctness-design.md`, and rerun the relevant failing test before each correction.

- [ ] **Step 6: Commit docs and final fixes**

```powershell
git add -- README.md docs/superpowers/plans/2026-07-10-surface-material-correctness.md
git commit -m "docs: describe OBJ surface material support"
```

- [ ] **Step 7: Final verification before completion**

Use `superpowers:verification-before-completion`, rerun Step 2 and the five Step 3 commands, then report exact results and remaining PBR limitations.
