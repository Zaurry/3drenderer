# Task 6 Report: BVH, Sampling and Scene Queries

## Status

Complete. BVH ranges, sampling scalars, and scene-query distances are float end-to-end in the Task 6 ownership scope. Owned vector math uses Eigen member operations and BVH axis indexing uses `vector[axis]`.

## RED / GREEN

### RED

Added exact interface assertions for:

- `Bvh::intersect(const Ray&, float, float, HitRecord&) const`
- `SceneIntersector::intersect(const Ray&, float, float, HitRecord&) const`
- `SceneIntersector::occluded(const Ray&, float, float) const`
- `refract(const Vec3&, const Vec3&, float, Vec3&)`
- `offset_ray_origin(...) -> Vec3`

Command:

```powershell
cmake --build build --config Release --target renderer_tests
```

Observed RED: build exited 1 with four `C2607: static assertion failed` diagnostics at `tests/renderer_tests.cpp:70`, `:77`, `:84`, and `:87`. After correcting test-only Eigen initializer-list construction, those four unmigrated API signatures were the only failures.

### GREEN

After migrating the owning APIs and implementation, the same Release target built successfully with no compiler warnings. The full test executable then reported:

```text
renderer_tests: all tests passed
```

## Regression Evidence

- Large-coordinate BVH fixture uses six triangles around `100000.0f`, forcing interior BVH traversal.
- A hit ray and a miss ray are compared against direct brute-force triangle traversal.
- Hit parity checks material ID, `t` within `1e-2f`, and shading-normal dot greater than `0.999f`.
- Ray-origin offset checks coordinates `1.0f` and `100000.0f`, normals in both axis directions, outgoing and incoming directions, finite outputs, and monotonic movement to the selected side.
- Thin alpha traversal places an opaque triangle `1e-5f` behind a transparent cutout and verifies continuation reaches the opaque material.
- Existing alpha-cutout and inclusive BVH/primitive range tests remain enabled.

## Implementation

- Converted BVH query ranges and nearest-hit storage to `float`; removed HitRecord range casts.
- Removed the BVH component adapter and compare centroids with `centroid()[axis]`.
- Converted sampler constants, intermediates, and refraction ratio to `float`; retained `PcgRandom::next_float()` and the existing distributions.
- Replaced owned free vector helpers with Eigen `.dot()`, `.squaredNorm()`, `.normalized()`, `.cwiseAbs()`, and `.maxCoeff()` operations.
- Converted scene intersection, occlusion, nearest-hit, and alpha continuation ranges to `float`; removed alpha traversal casts.
- Kept traversal ordering, nearest-hit replacement, inclusive ranges, transparent-layer limit, and `nextafter` continuation behavior unchanged.
- Added purpose-specific ray-origin constants for minimum scale, offset scale, and normal validity threshold.
- Updated only required ray/path call sites with float query literals and a scoped sampler argument conversion; ray/path algorithms remain deferred to Task 7.

## Source Audits

The following audits returned no matches in `src/acceleration`, `src/sampling`, and `src/render/scene_intersector.*`:

```powershell
rg -n "\bdouble\b|Eigen::[A-Za-z0-9_]*d\b|Vector[234]d\b|Matrix[234]d\b|static_cast<(float|double)>|next_double" ...
rg --pcre2 -n "(?<!\.)\b(?:renderer::)?(?:dot|cross|normalize|length|length_squared|component|min_components|max_components)\s*\(" ...
```

Verified separately that sampling uses `next_float()`, BVH uses `centroid()[axis]`, and alpha traversal uses `std::nextafter(candidate.t, t_max)` with float operands.

## Verification and Warnings

Clean full Release command:

```powershell
cmake --build build --config Release --clean-first
```

Result: exit 0; all renderer, viewer, core, and test targets rebuilt; no compiler warnings.

Full test command:

```powershell
.\build\bin\renderer_tests.exe
```

Result: exit 0; `renderer_tests: all tests passed`.

`git diff --check` exits 0. Git reports only the repository's existing LF-to-CRLF checkout notices, not source whitespace errors or compiler warnings.

## Files

- `src/acceleration/bvh.cpp`
- `src/acceleration/bvh.h`
- `src/sampling/sampler.cpp`
- `src/sampling/sampler.h`
- `src/render/scene_intersector.cpp`
- `src/render/scene_intersector.h`
- `src/render/pathtracer/pathtracer_renderer.cpp`
- `src/render/raytracer/raytracer_renderer.cpp`
- `tests/renderer_tests.cpp`
- `.superpowers/sdd/task-6-report.md`

## Commit

This report is included in the commit with subject:

```text
refactor: migrate intersections and sampling to float
```

## Self-Review

- Scope stays within Task 6 owners plus minimal ray/path call-site adapters.
- No raster, ray-tracing, or path-tracing algorithm was migrated.
- No external, learning, output, GPU/CUDA, feature, or unrelated files were changed.
- Tests cover the requested type surface and numerical regressions with real scene/BVH code.
- No traversal or sampling ordering change was introduced.

## Concerns

No Task 6 correctness concerns remain. Ray/path reflectance calculations intentionally remain double until Task 7; their sampler boundary uses an explicit float conversion so this task does not migrate those algorithms early.
