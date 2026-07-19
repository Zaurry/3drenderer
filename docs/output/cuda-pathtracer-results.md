# CUDA Path Tracer Upgrade Results

## Scope

This upgrade adds an optional CUDA backend only to the Monte Carlo path tracer.
The existing CPU path backend remains available, while the rasterizer and
Whitted ray tracer are unchanged. CUDA uses ordinary CUDA cores and the CUDA
Runtime API; it does not use OptiX or RT cores.

All work and measurements were performed on branch `codex/cuda-pathtracer`,
created from `master@81f0b739aec8d86369e5d11cdbe90dfe34382f8b`.

## Test system

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 9800X3D 8-Core Processor |
| GPU | NVIDIA GeForce RTX 5080, compute capability 12.0 |
| NVIDIA driver | 610.62 |
| CUDA toolkit | 13.3, nvcc 13.3.73 |
| CUDA architecture | `sm_120` |
| Host compiler | MSVC 19.51.36248 x64 |
| CMake | 4.4.0-rc3 |
| CPU tuning in CUDA matrix | `RENDERER_NATIVE_ARCH=ON` (`/arch:AVX2`) |

## Build and test matrices

The exact required matrices were used:

```powershell
cmake -S . -B build-cpu -DRENDERER_CUDA=OFF
cmake --build build-cpu --config Release --parallel 4
ctest --test-dir build-cpu -C Release --output-on-failure

cmake -S . -B build-cuda -DRENDERER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DRENDERER_NATIVE_ARCH=ON
cmake --build build-cuda --config Release --parallel 4
ctest --test-dir build-cuda -C Release --output-on-failure
```

Both Release builds completed and both CTest matrices passed `1/1`. The CUDA
tests cover backend selection, explicit unavailable-backend errors, statistical
CPU/CUDA parity, emissive and environment radiance, directional and point
lights, inverse-square falloff, hard shadows, Russian roulette, spheres,
triangles and BVH traversal, diffuse/metal/dielectric materials, two-sided and
alpha-cutout behavior, bilinear diffuse/opacity/bump textures, repeated CUDA
resource construction/destruction, progressive accumulation, and camera,
scene, size, manual, and backend-change resets.

The CPU-only CLI was also checked directly: `auto` warned once and selected
CPU, while explicit `cuda` failed. Offline raster mode rejected an explicitly
provided `--path-backend`. The viewer accepted a backend preset while starting
in raster mode, and rejected explicit CUDA when switching directly to path in
the CPU-only build.

CUDA memory checking used:

```powershell
compute-sanitizer --tool memcheck --error-exitcode 99 .\build-cuda\bin\renderer.exe --mode path --scene cornell_box --width 16 --height 16 --spp 2 --path-backend cuda --output output\cuda_memcheck_smoke.png
```

Result: `ERROR SUMMARY: 0 errors`.

## Performance

Each row used the same `build-cuda` Release binary for both backends. The CPU
therefore used the requested native architecture tuning. Each backend was run
once as a warmup and then five times. Values are the renderer's `seconds=`
field, which includes CPU BVH construction or CUDA BVH construction/upload,
CUDA initialization, kernel work, synchronization, and image download, but not
PNG encoding. OBJ parsing happens before the renderer timer for both backends.

| Scene | Backend | Warmup (s) | Five measured samples (s) | Median (s) | CUDA / CPU | Speedup |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Cornell Box, 512x512, 64 spp | CPU | 1.255910 | 1.245730, 1.246790, 1.218540, 1.233940, 1.300680 | **1.245730** | - | - |
| Cornell Box, 512x512, 64 spp | CUDA | 0.082534 | 0.095356, 0.080481, 0.082191, 0.083964, 0.080349 | **0.082191** | **6.598%** | **15.16x** |
| Mary OBJ, 1920x1080, 8 spp | CPU | 0.541093 | 0.592982, 0.529977, 0.532651, 0.510423, 0.518182 | **0.529977** | - | - |
| Mary OBJ, 1920x1080, 8 spp | CUDA | 0.147251 | 0.147266, 0.148713, 0.140242, 0.145591, 0.145412 | **0.145591** | **27.471%** | **3.64x** |

Both scenes pass the requirement that CUDA median time be no more than 50% of
CPU median time.

For a non-gating viewer observation, 120 progressive 1 spp frames of the
built-in Cornell Box at 512x512 took 15.820841 s on CPU and 1.003802 s on CUDA,
corresponding to approximately **7.585 FPS CPU** and **119.545 FPS CUDA**. This
measurement includes viewer startup, CPU-surface download, and presentation.

## Image comparison

The final measured runs generated:

- `output/cuda_bench_cornell_cpu.png`
- `output/cuda_bench_cornell_cuda.png`
- `output/cuda_bench_mary_cpu.png`
- `output/cuda_bench_mary_cuda.png`

The following metrics are calculated from decoded normalized 8-bit sRGB PNG
channels. Automated tests separately check finite linear floating-point values
before PNG conversion. Lit coverage is the percentage of pixels with Rec. 709
luminance greater than 0.01.

| Image | Bytes | Mean RGB | Mean luma | Lit coverage | Magenta fallback pixels |
| --- | ---: | --- | ---: | ---: | ---: |
| Cornell CPU | 704,577 | 0.440456, 0.369973, 0.258762 | 0.376928 | 80.418396% | 0 |
| Cornell CUDA | 704,577 | 0.440456, 0.369973, 0.258762 | 0.376928 | 80.418396% | 0 |
| Mary CPU | 608,065 | 0.151109, 0.171042, 0.187557 | 0.167997 | 99.918644% | 0 |
| Mary CUDA | 607,987 | 0.151108, 0.171042, 0.187557 | 0.167997 | 99.918644% | 0 |

| Pair | Mean RGB relative error | Mean luma relative error | Lit coverage delta | Mean absolute channel delta |
| --- | --- | ---: | ---: | ---: |
| Cornell CPU/CUDA | 0.000000%, 0.000000%, 0.000000% | 0.000000% | 0.000000 pp | 0.000000 |
| Mary CPU/CUDA | 0.000006%, 0.000001%, 0.000018% | 0.000003% | 0.000000 pp | 0.000000 |

Both comparisons are far inside the 5% mean-color/luminance and 3 percentage
point coverage limits. Visual inspection found matching framing, material and
texture detail, Cornell wall/light balance, and Monte Carlo noise, with no
missing geometry, winding cracks, or magenta fallback regions.

## Notes

- CUDA accumulation and PCG state stay in device memory between viewer frames;
  SDL presentation still uses a downloaded CPU framebuffer.
- Each CUDA launch renders one sample per pixel to keep kernel duration bounded
  for Windows TDR behavior.
- `auto` only falls back during initial selection when CUDA is not compiled or
  no device is available. Once a CUDA interactive session is selected, runtime
  memory, launch, synchronization, or traversal errors are reported rather
  than hidden by a CPU fallback.
- The fixed CUDA BVH traversal stack reports overflow through a device error
  flag and never writes beyond its capacity.
