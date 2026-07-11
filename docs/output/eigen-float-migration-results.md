# Eigen Float Migration: Double-Precision Baseline

## Scope and provenance

This is the pre-migration evidence baseline captured from the clean renderer at
base commit `a54aa94d50b79a7f4a4c0be9ecb494b510f7156d` on branch
`codex/eigen-float-migration`. No renderer source code was modified.

- CPU: AMD Ryzen 7 9800X3D
- GPU: RTX 5080 (unused; all rendering was CPU-side)
- Compiler: MSVC `19.51.36248.0` (Visual Studio 18 2026 generator)
- Build mode: Release
- Asset root used for execution: `D:\Github\3drenderer\Computer Graphics Archive`

## Baseline verification

Commands:

```powershell
cmake --build build --config Release
.\build\bin\renderer_tests.exe
ctest --test-dir build -C Release --output-on-failure
```

Exact verification summary:

```text
Build: exit 0; renderer, renderer_tests, and viewer targets built.
renderer_tests: all tests passed
1/1 Test #1: renderer_tests ...................   Passed    0.01 sec
100% tests passed out of 1
Total Test time (real) =   0.01 sec
```

Static baseline commands and results:

```powershell
rg -o "\bdouble\b" src tests | Measure-Object
rg -o "\bVec3\b" src tests | Measure-Object
rg -n "Eigen::.*d\b" src tests
```

- `double`: `396`
- `Vec3`: `587`
- Eigen double matches: only `Eigen::Vector3d` declarations/conversions and
  orbit-camera state in `src/interactive/orbit_camera_controller.h` and
  `src/interactive/orbit_camera_controller.cpp` (9 matching lines).

## Render commands and timings

Each benchmark was run once as a warmup, followed by five measured executions.
Timings below are the renderer-reported `seconds=` values; process startup was
not timed externally. The median is the middle value after sorting the five
samples.

| Scene / mode | Command | Five measured samples (s) | Median (s) | Output |
| --- | --- | ---: | ---: | --- |
| Mary / raster | `renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --output output\eigen_baseline_mary_raster.png` | 0.0675227, 0.0662127, 0.0682396, 0.0693546, 0.0664713 | **0.0675227** | `output/eigen_baseline_mary_raster.png` |
| Mary / path | `renderer.exe --mode path --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\mary\Marry.obj" --width 1920 --height 1080 --spp 1 --max-depth 4 --output output\eigen_baseline_mary_path.png` | 0.107037, 0.106840, 0.106667, 0.106635, 0.106277 | **0.106667** | `output/eigen_baseline_mary_path.png` |
| Sponza / raster | `renderer.exe --mode raster --scene obj_viewer --obj "D:\Github\3drenderer\Computer Graphics Archive\sponza\sponza.obj" --width 1280 --height 720 --output output\eigen_baseline_sponza_raster.png` | 0.0955045, 0.0986501, 0.0984964, 0.0967525, 0.0962721 | **0.0967525** | `output/eigen_baseline_sponza_raster.png` |
| Cornell box / path | `renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 8 --max-depth 5 --output output\eigen_baseline_cornell_path.png` | 0.254427, 0.250895, 0.250513, 0.247663, 0.248048 | **0.250513** | `output/eigen_baseline_cornell_path.png` |

The commands were executed from the worktree root with the executable at
`build\bin\renderer.exe`. All 24 render processes (four warmups and twenty
measured runs) exited 0 and printed the requested mode, scene, dimensions, and
`seconds=` value.

## Output verification

All four PNGs were produced and were nonblank in sampled pixel checks:

| File | Dimensions | Bytes |
| --- | ---: | ---: |
| `output/eigen_baseline_mary_raster.png` | 1920x1080 | 362,118 |
| `output/eigen_baseline_mary_path.png` | 1920x1080 | 526,214 |
| `output/eigen_baseline_sponza_raster.png` | 1280x720 | 483,883 |
| `output/eigen_baseline_cornell_path.png` | 512x512 | 204,340 |

## Concerns

- Sponza emits existing asset-loader warnings about `d`/`Tr` material
  parameters, missing vertex normals, and degenerate UV bump-map triangles.
  Rendering still exits 0 and produces a nonblank PNG.
- The PNGs are generated baseline artifacts in the ignored `output/` area;
  the only tracked change for this task is this report.

## Files changed and commit

- Tracked: `docs/output/eigen-float-migration-results.md`
- Generated baseline outputs: `output/eigen_baseline_*.png` (ignored)
- Commit: this report is committed with subject `docs: record double renderer baseline`;
  obtain the authoritative SHA with `git rev-parse HEAD`.
