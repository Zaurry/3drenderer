# Eigen Float Migration: Visual Regression and Benchmark Results

## Status and provenance

Task 10 is **BLOCKED**. The native Mary path median is 23.284% slower than
the committed double baseline, which exceeds the 5% blocker threshold and
requires profiling before this migration can be called complete.

- Branch: `codex/eigen-float-migration`
- Post-migration source commit: `b7cd8ff` (`build: finalize Eigen native float math`)
- Baseline source commit: `a54aa94d50b79a7f4a4c0be9ecb494b510f7156d`
- CPU: AMD Ryzen 7 9800X3D
- GPU: RTX 5080 (unused; this is a CPU renderer with no CUDA or RT Core backend)
- Compiler: MSVC `19.51.36248.0` (Visual Studio 18 2026 generator)
- Build mode: Release
- Asset root: `D:\Github\3drenderer\Computer Graphics Archive`
- Working directory: `D:\Github\3drenderer\.worktrees\eigen-float-migration`

The native tree was configured with `RENDERER_NATIVE_ARCH=ON`; the portable
tree used `RENDERER_NATIVE_ARCH=OFF`. On MSVC, the option adds `/arch:AVX2` to
the four first-party targets (`renderer_core`, `renderer`, `viewer`, and
`renderer_tests`). The portable compile lines contain no `/arch:AVX2`. SDL3
used the existing cache through the command-line-only
`-DFETCHCONTENT_SOURCE_DIR_SDL3=D:/Github/3drenderer/build/_deps/sdl3-src`
override. No machine-specific path was added to a project file.

## Native render commands

Each command was run from the worktree root with `build-native\bin` Release
binaries. These are the four required post-migration outputs:

```powershell
.\build-native\bin\renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --output output\eigen_float_mary_raster.png
.\build-native\bin\renderer.exe --mode path --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --spp 1 --max-depth 4 --output output\eigen_float_mary_path.png
.\build-native\bin\renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\sponza\sponza.obj" --width 1280 --height 720 --output output\eigen_float_sponza_raster.png
.\build-native\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 8 --max-depth 5 --output output\eigen_float_cornell_path.png
```

All four commands exited 0 and reported the requested dimensions, spp, and
depth. The Sponza command emitted the known loader warnings for conflicting
`d`/`Tr` fields, missing vertex normals, and degenerate bump-map UVs; it still
produced a valid image.

The required viewer smoke test was:

```powershell
.\build-native\bin\viewer.exe --scene asset --asset "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --mode path --width 1920 --height 1080 --frames 1
```

Result: exit 0, `viewer mode=path frames=1 size=1920x1080`.

## Benchmark results

The committed double baseline used one warmup followed by five measured runs.
The same procedure was repeated with `build-native\bin\renderer.exe`; all
four warmups and twenty measured processes exited 0. Values are the renderer's
reported `seconds=` values, not external process timings. Delta is
`(native median / double median - 1) * 100%`.

| Scene / mode | Exact native command | Five measured samples (s) | Double baseline median (s) | Native median (s) | Delta | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Mary / raster | `.\build-native\bin\renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --output output\eigen_float_bench_mary_raster.png` | 0.0601103, 0.0559124, 0.0557041, 0.0566670, 0.0586471 | 0.0675227 | **0.0566670** | **-16.077%** | pass |
| Mary / path | `.\build-native\bin\renderer.exe --mode path --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --spp 1 --max-depth 4 --output output\eigen_float_bench_mary_path.png` | 0.131503, 0.135715, 0.130779, 0.133672, 0.131084 | 0.1066670 | **0.1315030** | **+23.284%** | **BLOCKED** |
| Sponza / raster | `.\build-native\bin\renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\sponza\sponza.obj" --width 1280 --height 720 --output output\eigen_float_bench_sponza_raster.png` | 0.0788790, 0.0836402, 0.0782177, 0.0788246, 0.0786419 | 0.0967525 | **0.0788246** | **-18.530%** | pass |
| Cornell box / path | `.\build-native\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 8 --max-depth 5 --output output\eigen_float_bench_cornell_path.png` | 0.266756, 0.256036, 0.270077, 0.261226, 0.254947 | 0.2505130 | **0.2612260** | **+4.276%** | pass, under 5% |

The Mary path slowdown is a real blocker, not noise to be rationalized. The
next step is profiling native versus baseline path tracing before completion.

## Image and pixel checks

The four post images were decoded successfully with `System.Drawing.Bitmap`.
The decoder reported the exact dimensions below; PNG channels are finite
8-bit values by construction, and no magenta-fallback pixels were found. Lit
coverage is the fraction of pixels with Rec. 709 luminance greater than 0.01.
Mean RGB values are normalized to 0..1.

| Post image | Bytes | Dimensions | Lit coverage | Mean luma | Mean RGB (R, G, B) | Magenta pixels | Changed channels vs baseline | Mean abs channel delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `output/eigen_float_mary_raster.png` | 362,188 | 1920x1080 | 11.0031% | 0.025738 | 0.024130, 0.026010, 0.027771 | 0 | 117 | 0.000000 |
| `output/eigen_float_mary_path.png` | 593,651 | 1920x1080 | 97.2171% | 0.163849 | 0.147320, 0.166828, 0.183010 | 0 | 179,441 | 0.003862 |
| `output/eigen_float_sponza_raster.png` | 484,002 | 1280x720 | 26.3441% | 0.039046 | 0.042702, 0.038308, 0.035600 | 0 | 281 | 0.000000 |
| `output/eigen_float_cornell_path.png` | 196,200 | 512x512 | 33.1665% | 0.215692 | 0.243598, 0.213894, 0.151331 | 0 | 64,291 | 0.041128 |

Visual inspection against the committed `eigen_baseline_*.png` images found:

- Mary raster remains smooth, fully framed, and textured, including the dress,
  apron lettering, hair, and boots.
- Mary path retains the same smooth silhouette, textures, and lighting layout.
  Its low-spp Monte Carlo noise is visible but matches the baseline character;
  there are no missing geometry regions, magenta fallback areas, or winding
  cracks.
- Sponza retains its diffuse brick detail and the baseline bump/detail pattern.
  No new widespread speckle, missing geometry, or magenta fallback is visible.
- Cornell retains the bright emissive ceiling panel and the red/green wall
  balance. Its low-spp noise is already present in the baseline and is not a
  float-migration-specific artifact.

## Double baseline reference

The committed double baseline was captured from the clean pre-migration
renderer. It used the same worktree-root execution and one warmup followed by
five measured runs. The baseline commands and medians are retained here so the
percentage calculations above can be replayed directly.

| Scene / mode | Exact baseline command | Five measured samples (s) | Median (s) | Output |
| --- | --- | ---: | ---: | --- |
| Mary / raster | `.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --output output\eigen_baseline_mary_raster.png` | 0.0675227, 0.0662127, 0.0682396, 0.0693546, 0.0664713 | **0.0675227** | `output/eigen_baseline_mary_raster.png` |
| Mary / path | `.\build\bin\renderer.exe --mode path --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --spp 1 --max-depth 4 --output output\eigen_baseline_mary_path.png` | 0.107037, 0.106840, 0.106667, 0.106635, 0.106277 | **0.106667** | `output/eigen_baseline_mary_path.png` |
| Sponza / raster | `.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\sponza\sponza.obj" --width 1280 --height 720 --output output\eigen_baseline_sponza_raster.png` | 0.0955045, 0.0986501, 0.0984964, 0.0967525, 0.0962721 | **0.0967525** | `output/eigen_baseline_sponza_raster.png` |
| Cornell box / path | `.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 8 --max-depth 5 --output output\eigen_baseline_cornell_path.png` | 0.254427, 0.250895, 0.250513, 0.247663, 0.248048 | **0.250513** | `output/eigen_baseline_cornell_path.png` |

Baseline PNG sizes are 362,118 bytes (Mary raster), 526,214 bytes (Mary
path), 483,883 bytes (Sponza raster), and 204,340 bytes (Cornell path). The
baseline images are ignored artifacts in `output/eigen_baseline_*.png`; native
post images use the required `output/eigen_float_*.png` prefix and are also
ignored.

## Final verification

```powershell
ctest --test-dir build-portable -C Release --output-on-failure
ctest --test-dir build-native -C Release --output-on-failure
rg -n "\bdouble\b|Eigen::(Vector[234]d|Matrix[^ ]*d)|core/math/(vec2|vec3|vec4|mat4)\.h" src tests
git diff --check
git status --short --branch
```

Both CTest matrices passed `1/1`. The source-policy scan returned no matches
(expected `rg` exit 1), and `git diff --check` returned 0. The first-party
Release builds completed with 0 compiler warnings and 0 errors.

## Concerns

- **BLOCKER:** Mary path native median is +23.284%; profile before claiming
  completion.
- Sponza's known source-asset warnings remain as described above.
- The CPU backend does not use the RTX 5080, CUDA, or RT Cores.
- Generated PNGs and build trees are local evidence artifacts, not source
  deliverables.
