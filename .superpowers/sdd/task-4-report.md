# Task 4 Report: Core Domains, Randomness, Camera and Timing

## Status

Complete. Task 4 scalar APIs and owned storage use `float`, owned vector math uses native Eigen operations, required compatibility call sites compile, and the clean Release build and full test suite pass without warnings.

## RED / GREEN Evidence

### RED

Command:

```powershell
cmake --build build --config Release
```

Result: exit code 1. `tests/renderer_tests.cpp` failed to compile because `PcgRandom::next_float()` did not exist, and static assertions failed for `Timer::elapsed_seconds`, `RenderResult::seconds`, `FrameRateSnapshot::frames_per_second`, and `Camera::viewport_width` because they were still `double`.

### GREEN

Commands:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

Result: build exit code 0; test exit code 0 with `renderer_tests: all tests passed`.

Final clean verification:

```powershell
cmake --build build --config Release --clean-first
.\build\bin\renderer_tests.exe
```

Result: both commands exited 0. The clean Release build emitted no compiler warnings, and the full test executable reported all tests passed.

## API and Type Audit

- `Ray::at`, `Bounds3::intersect`, color conversion, camera, orbit control, FPS, SDL input deltas, depth buffer, timer, and `RenderResult::seconds` now use `float`.
- `PcgRandom` and `Random` expose `next_float`; `Random` uses `std::uniform_real_distribution<float>`.
- `PcgRandom::next_float` uses the upper 24 random bits and a `1 / 2^24` scale, preserving the promised `[0, 1)` range without float rounding producing `1.0f`.
- `Timer` and viewer frame timing use `std::chrono::duration<float>`.
- Orbit control stores `Vec3 target_` directly, explicitly initializes it with `Vec3::Zero()`, and no longer declares conversion helpers or Eigen double vectors.
- Bounds expansion uses `cwiseMin`/`cwiseMax`; camera and orbit math use Eigen `allFinite`, `squaredNorm`, `normalized`, `cross`, and `norm` methods.
- Repository audit found no remaining `next_double` references.
- Owned-file audit found no `double`, Eigen double types, `duration<double>`, or `numeric_limits<double>`.
- The requested `<type_traits>`/`<utility>` compile-time assertions are present.
- `nearly_equal` now accepts floats with a `1e-5f` default. Existing explicit `1e-6` tolerances were retained as `1e-6f`; no explicit tolerance was weakened.
- `Image` already stored float `Color` values with `Color::Zero()` initialization, and `RenderSettings` already used integer controls plus float `Color`; both were audited and required no source change.

## Files

Owned files modified:

- `src/core/color.h`
- `src/core/math/bounds.h`
- `src/core/math/ray.h`
- `src/core/random.h`
- `src/core/timer.h`
- `src/scene/camera.h`
- `src/scene/camera.cpp`
- `src/interactive/orbit_camera_controller.h`
- `src/interactive/orbit_camera_controller.cpp`
- `src/interactive/frame_rate_counter.h`
- `src/interactive/frame_rate_counter.cpp`
- `src/platform/sdl/sdl_display_backend.h`
- `src/render/depth_buffer.h`
- `src/render/depth_buffer.cpp`
- `src/render/renderer.h`
- `tests/renderer_tests.cpp`
- `tests/test_framework.h`

Required API compatibility call sites modified:

- `src/acceleration/bvh.cpp`
- `src/main.cpp`
- `src/render/pathtracer/pathtracer_renderer.cpp`
- `src/render/raytracer/raytracer_renderer.cpp`
- `src/sampling/sampler.cpp`
- `src/scene/primitive.h`
- `src/scene/scene_asset_loader.cpp`
- `src/viewer_main.cpp`

Audited without changes: `src/core/image.h`, `src/core/image.cpp`, and `src/render/render_settings.h`.

## Clean Warning Status

The final `--clean-first` Release build rebuilt SDL, renderer core, CLI renderer, tests, and viewer. It completed with exit code 0 and no warning diagnostics under the configured MSVC `/W4` policy.

## Self-Review

- `git diff --check` passed.
- Reviewed the full diff for ownership, initialization, float literals, finite checks, and unintended algorithm migration.
- Changes outside Task 4 ownership are limited to removed RNG API call sites and explicit adapters into new float `Ray`, `Bounds3`, camera, orbit, and FPS APIs.
- No files under `external/`, `learning/`, `docs/output`, CUDA, or GPU code were touched.
- No unrelated existing changes were reverted.

## Commit

Commit message: `refactor: migrate core renderer scalars to float` (the commit containing this report; final hash is reported after creation).

## Concerns

- PCG floating samples are intentionally derived at float precision from 24 high bits, so deterministic path-traced sample sequences and rendered pixels can differ from the previous double API even with the same seed.
- Primitive, material, BVH, and broader renderer algorithm scalars remain double until their assigned later tasks. This task uses explicit casts only at the newly migrated float API boundaries.
