# Native DXR renderer

DXR is a separate Windows backend (`--mode dxr`, key `3`). It uses a native
D3D12/DXGI presentation path and does not require CUDA or OptiX. The default
startup mode is unchanged. Viewer sessions use version 8 and load versions
1–7; scene documents remain version 5.

## Build and run

```powershell
cmake --preset dxr-no-cuda
cmake --build --preset dxr-no-cuda-release
build/dxr-no-cuda/bin/viewer.exe --mode dxr --strict-dxr --no-restore-last
```

`RENDERER_DXR=AUTO|ON|OFF` controls the backend. `AUTO` enables it on Windows.
The `dxr-off` preset exercises the dependency-free implementation; Linux uses
that implementation automatically. DXR 1.1 and SM 6.6 are the hardware baseline.
Standard SER and opacity micromaps are independently detected and use SM 6.9.
SER Auto measures both ray variants on completed GPU frames during warmup
(8 settling + 24 measured frames per variant). It excludes moving-camera,
scene-update and pending OMM-bake frames, and selects SER only with at least
2% measured ray-stage improvement. Scene/integrator/resolution changes restart
the comparison. Diagnostics report the two means and the measured speedup;
support alone does not establish a performance benefit.
`--strict-dxr` fails if DXR cannot start; interactive use reports the reason and
falls back to OpenGL.

Dependencies and archive SHA-256 values are pinned in
`cmake/RendererDxr.cmake` and `tools/fetch_dxr_sdks.ps1`: Agility 1.619.6,
DXC 1.9.2609, RTXDI 3.1.0 with RTXDI-Library revision
`f12037fa8e97ebc08e9e3edfd2de528ed1772a4b`, NRD 4.17.3 and Streamline 2.14.1.
The SDK archives and build artifacts live under ignored `build/` directories.
NRD shader outputs are isolated under each preset's `generated/nrd` directory,
so CUDA and no-CUDA builds do not race while writing ShaderMake output files.

Use `--dxr-config file.json` to load settings independently of RTRT settings:

```json
{
  "samples_per_pixel": 1,
  "max_bounces": 8,
  "restir_di": true,
  "restir_pt": true,
  "shader_execution_reordering": true,
  "opacity_micromaps": true,
  "full_resolution_materials": true,
  "reconstruction": "auto",
  "spatial_samples": 4,
  "pt_spatial_samples": 2,
  "pt_disocclusion_samples": 4,
  "internal_scale": 0.6666666667
}
```

The reference configuration uses `"reconstruction": "reference"`,
`"restir_di": false`, `"restir_pt": false`. It renders at output resolution
and accumulates stationary frames without denoising. Set
`"specular_antialiasing": false` to additionally disable PSR.
DI uses four spatial neighbors. PT uses two in established history and up to
four for reconnectible paths lacking sufficient history, with the SDK's MIS
correction unchanged. Both budgets are independently configurable. Older v8
configuration files containing only `spatial_samples` preserve their shared
DI/PT budget.

DI estimates the light-sampled MIS share of direct illumination. The first BRDF
continuation supplies the complementary emitter/environment share through the
ordinary path tracer or the PT reservoir. Camera-visible emission is added once;
subsequent NEE/BRDF connections retain their normal MIS weights. Analytical
lights without intersectable geometry use the full light-sampled contribution.
At the last bounce there is no BRDF continuation, so DI also uses full weight.
This avoids losing narrow glossy reflections when light-only candidates rarely
land inside the specular lobe, without adding another traced path per pixel.
The RTXDI bias-correction compile switch is set before the temporal header:
runtime `BASIC` selection alone cannot enable code that was compiled out.

## Shared editor costs

The DXR backend consumes a cached `RenderSceneSnapshot` by const reference;
geometry and texture pixels retain shared immutable storage. It does not call
the OpenGL or OptiX renderer. D3D12 frame outputs stay on the GPU through DXGI
presentation. The DXR-only preset verifies that CUDA is not required.

On document edits, unchanged local mesh bounds and primitive bindings are
reused instead of rescanning all triangles. Once OMM jobs settle, camera-only
frames also bypass repeated material/signature scans in the DXR scene cache.
Full-frame reports separate CPU UI/input, snapshot acquisition, command
recording, presentation, session work and the explicit benchmark completion
wait. GPU timestamps measure the rendering stages separately; the completion
wait is not additional CPU rendering work.

## Reconstruction and diagnostics

Auto selects DLSS Ray Reconstruction, then NRD RELAX + DLSS SR, then
NRD RELAX + TAAU. DLSS uses Quality mode. Streamline is loaded dynamically;
OTA and downloaded plugin loading are disabled. The actual loaded module
paths, file versions and hashes are recorded, including NVIDIA driver
overrides. A runtime hash mismatch disables DLSS and explains the fallback.
The project does not alter global driver settings. On driver 617.14 on the
development RTX 5080, the driver currently substitutes downloaded plugins,
so reproducible runs use RELAX + TAAU.

The DXR panel exposes the requested and active features, stage timings,
allocation counts, memory and reconstruction reason. Debug views include
raw/direct/indirect radiance, normals, albedo, depth, motion, reflection
motion, reservoir age/weight, history rejection and the NRD validation grid.
BRDF hit-distance and TAA history views help diagnose reconstruction inputs.
NRD distances come from an original first-bounce BRDF sample, independent of
the reused reservoir; missing lobes use NRD's 3×3 distance reconstruction.
With both ReSTIR stages enabled, RELAX diffuse/specular prepass radii are
8/16 pixels instead of the SDK's 30/50-pixel defaults, retaining more lighting
detail in the already-resampled signal. Other sampling combinations retain
the default prepasses. Both prepasses remain enabled for missing lobe distances.
TAAU integrates foreground/background coverage at silhouettes, reprojects with
the nearest foreground motion guide, and validates ordinary surfaces using
geometric normals. Normal-map variation therefore accumulates instead of
discarding history every jittered frame. Infinite backgrounds track camera
rotation. Short history limits apply to glass and glossy reflection; a pure
emitter's zero roughness does not shorten its coverage history.
When camera, geometry and lighting all remain unchanged, TAA accumulates the
same output footprint without clamping it to a noisy single-frame neighborhood.
It ramps toward at least 32 frames of history; camera motion, any scene edit,
cuts and resizes end this stationary accumulation immediately.

`full_resolution_materials` enables material-color reconstruction for TAAU.
It intersects the output sample with already-known primary triangles and
re-evaluates albedo, reflectance colors and emission, then combines them with
the denoised lighting. All four lighting contributors must belong to the same
continuous opaque surface. Silhouettes, alpha, glass and PSR use the coverage
resolve. The sampled lighting normals and lobe widths are retained: scalar
irradiance cannot reconstruct a different normal's reflection direction.
This pass adds no BVH traversal and does not increase path SPP. It does not
recover small geometry absent from the input guides or replace DLSS processing.
Capture only when needed: normal presentation does not download the framebuffer.

## Validation commands

```powershell
$env:DXR_STRICT='1'
$env:DXR_VALIDATE='1'
$env:DXR_GPU_VALIDATION='1'
$env:DXR_DISABLE_DLSS='1'
build/dxr-no-cuda/bin/dxr_tests.exe
build/dxr-no-cuda/bin/dxr_lifecycle_tests.exe
```

`DXR_LEGACY_BARRIERS=1` forces the traditional barrier path. GPU tests report
explicit skips on hosts without DXR; `DXR_STRICT=1` converts those to failures.
The lifecycle test uses a temporary in-memory UI layout and never persists it
to the user's ImGui settings. A CUDA-enabled build also exercises RTRT mode
switches.
Requested debug/validation interfaces must initialize successfully. Reports
record their actual activation instead of assuming the environment request
was honored.

The quality tool supports static, camera-stop, object-translation and moving
light sequences. It writes ACES PNGs, linear RGB PFM files, reference errors,
energy ratios and temporal residuals. Its readbacks and reference rendering
are intentionally excluded from performance acceptance:

```powershell
build/default/bin/dxr_quality.exe --output build/quality-cornell `
  --scene cornell --motion object --width 640 --height 360 `
  --warmup 120 --frames 120 --reference-spp 4096 --reference-stride 8
```

Use a CUDA/OptiX build for the RTRT comparison, or explicitly pass `--no-rtrt`.
Unavailable requested RTRT comparison produces a nonzero exit and a recorded
skip reason. `--scene-file` and `--camera-preset` load external inputs read-only.
Use independent scene copies from the performance runner's `inputs` folder
when comparing the user's edited cave. Glass and small-light validation use
4096 SPP; other reference cases use 1024 SPP.

The suite wrapper records executable, shader, runtime, configuration and scene
fingerprints alongside camera and RTRT settings. It includes mirror motion,
glass, occlusion, moving shadows, many lights, San Miguel and cave trajectories:

```powershell
python tools/run_dxr_quality.py --output build/dxr-quality-sequences
```

Its `completed` field records that measurements finished, not that image
quality passed. Inspect the reference errors, temporal residuals and exported
images before making a quality claim.
Quality report schema v2 records `stationary_start_frame` and
`stationary_pairs`. Static captures use all consecutive pairs; animated
trajectories use pairs after the stop. A sequence with no stationary pairs
reports `null` for stationary log-RMSE, not zero. One pair is insufficient
evidence for a temporal-stability claim.

```powershell
python tools/run_dxr_benchmarks.py --viewer build/dxr-no-cuda/bin/viewer.exe `
  --output build/dxr-acceptance
```

The default acceptance schedule is four scenes, 2560×1440, 120 warmup frames,
3000 measured frames and three repetitions. The runner checks full-frame
P95 ≤ 16.67 ms and P99 ≤ 20 ms, actual dimensions, active reconstruction,
runtime pinning and fixed quality settings. Short profiling runs cannot be
reported as acceptance. Original `.rscene` bytes are copied and fingerprinted;
asset paths are rebased only in a separate benchmark copy. Original hashes
are checked again after the run.
The runner uses `--dxr-settle-before-warmup`: OMM baking and SER comparison
finish before the full 120-frame warmup starts. Startup frames/time are reported
separately, with a 120-second failure limit. Scripted motion starts after this
initialization. No frames are removed from the subsequent measurement interval.
The manifest summarizes CPU and GPU stage means/P95/P99, active features, resource
counter deltas and GPU-only presentation checks. `allocated_bytes` counts managed
D3D12 resource allocations (including pools), not total driver-resident VRAM.
DXGI process-local video memory usage and budget are sampled every 60 frames and
reported separately, with explicit availability and peak/delta values.

## Acceptance status

Implementation and hardware validation are in progress. Passing shader builds
or correctness tests does not establish the complete performance/quality goal.
The full 1440p baseline in `build/dxr-validation/acceptance-1440-final`
completed startup settling, 120 warmup + 3000 measured frames, three repetitions
per scene (36000 measured frames). This baseline includes the DI energy fixes
but predates the newer footprint/alias and reconstruction changes below:

| Scene | Full-frame P95 range | Full-frame P99 range | All three runs within timing limits |
| --- | ---: | ---: | --- |
| Cornell | 12.43–12.76 ms | 12.67–12.99 ms | Yes |
| San Miguel | 23.83–24.12 ms | 24.04–24.47 ms | No |
| Cave | 32.59–32.67 ms | 32.85–32.97 ms | No |
| Many lights | 13.23–13.25 ms | 13.36–13.38 ms | Yes |

All runs used RELAX + TAAU, 1 SPP, 8 bounces, DI/PT enabled, a 1707×960 internal
image and no frame generation or vsync. SER Auto selected ordinary tracing.
All twelve runs had zero measured-frame image readbacks, unchanged allocation
and AS counters, and zero net process-local VRAM growth. San Miguel averaged
0.48–0.49 ms of active CPU work and 23.08–23.33 ms on the GPU. The shared editor
and scene snapshot are not the dominant static-scene cost in this measurement.
The complete performance target did not pass.

The subsequent 120 + 240-frame profile in
`build/dxr-validation/profile-footprint-alias-1440` measured P95 27.02 ms for
San Miguel and 28.70 ms for cave. HDR sampling uses a constant-time alias table;
PT uses the RTXDI footprint reconnection criterion. This combination helps cave
but slows San Miguel, and the short run is not acceptance. Work now prioritizes
reconstruction fidelity and stability rather than reducing quality to reach FPS.

With the quality changes below enabled, the final 120 + 240-frame profile in
`build/dxr-validation/profile-quality-final-1440` measured San Miguel
P95/P99 **29.48/30.17 ms** and cave **28.58/28.94 ms**. This is a short profile,
not a repeat of the complete acceptance schedule. Both retain GPU-only
presentation, fixed quality settings, settled startup and zero allocation/AS
counter growth or net VRAM growth over the measured interval. The performance
target remains unmet at these quality settings.

The quality sequences use 480×270 output, 320×180 internal resolution, 1 SPP,
8 bounces, 120 warmup and 64 captured frames. References use 1024 SPP (4096 for
glass, moving shadows and many lights). Reports, linear PFM frames and ACES PNGs
are under `build/dxr-validation/motion-quality-*`. Identical static reference
states reuse one high-SPP image, recorded in `reference_frames`, so reference
noise is not mistaken for temporal instability. These measurements do not
establish 1440p quality acceptance. The mirror-motion case demonstrates moving
emissive geometry in DXR reflections that the current RTRT baseline misses;
other scenes still need comparison against RTRT and the reference.

`build/dxr-validation/quality-priority-summary.json` compares the `motion-quality-footprint-alias`
baseline with `motion-quality-nrd-detail` at the above identical sample budget.
Relative L1 is measured at the last captured frame; stationary log-RMSE measures
frame-to-frame residuals after the motion stops (lower is better for both):

| Case | Relative L1 before → after | Stationary log-RMSE before → after |
| --- | ---: | ---: |
| Cornell occlusion | 0.08898 → 0.06738 | 0.00183 → 0.00272 |
| Moving shadow | 0.10972 → 0.08041 | 0.00373 → 0.00301 |
| Mirror motion | 0.00085 → 0.00091 | 0.000429 → 0.000420 |
| Glass | 0.13484 → 0.10497 | 0.00599 → 0.00302 |
| Many lights | 0.06863 → 0.05558 | 0.00321 → 0.00281 |
| San Miguel | 0.40935 → 0.29492 | 0.02511 → 0.00614 |
| Cave | 0.38364 → 0.36117 | 0.01506 → 0.00423 |

The changes improve error in six cases, but do not improve every stability
metric: Cornell occlusion's short post-motion interval is noisier. RTRT still
has lower error in San Miguel (0.26940 versus 0.29492) and lower stationary
residual in cave (0.00290 versus 0.00423). Cave's error is now slightly below
the matched RTRT result (0.36831). Fine specular detail and quick convergence
remain open work; these numbers are not a blanket image-quality pass.

A separate static cave capture at **2560×1440** (1707×960 internal, 1 SPP,
8 bounces, DI/PT, RELAX + TAAU, 240 warmup frames) is under
`build/dxr-validation/quality-cave-1440`. Against its 1024 SPP unreused,
undenoised reference, the last frame has relative L1 **0.27970**, display PSNR
**26.03 dB** and energy ratio **0.97650**. The exported `dxr/0001.png` and
`reference/0001.png` show remaining softness in rock and wet-specular detail.
Only two stationary frames were captured; this validates a still image, not
1440p motion stability. A 320×180 undenoised ReSTIR accumulation cross-check
(`cave-path-quality-final`) has energy ratio **0.99997** against 1024 SPP,
with no evidence of a global energy deficit in that scene.

The original two-frame v1 reports in these two directories incorrectly recorded
stationary log-RMSE as zero because they collected no pairs. Their
`report-v2.json` files retain the measurements and identify the correction,
recomputed from the saved linear images, with exactly one pair. Original
reports and images are preserved. `quality-priority-summary.json` links the
corrected reports, the latest performance profile and validation logs.

The final Windows CUDA and no-CUDA builds pass. The no-CUDA DXR binary passes
22 tests with both the D3D12 debug layer and GPU validation requested and
confirmed, with zero skipped tests (`quality-final-all-gbv.log`). DXR-off CTest
has 11 passed tests and two explicit hardware/backend skips. The material-color
regression uses an animated light and a deterministic emissive texture so it
checks responsive reconstruction independently of stationary accumulation.
Three reconstruction/resize tests also pass with traditional barriers forced
(`quality-final-legacy-gbv.log`).
The CUDA build passes both lifecycle tests with debug/GPU validation, including
two cycles through OpenGL, RTRT and DXR, resize/minimize, document/UI preservation
and the injected device-removal failure (`quality-final-lifecycle-gbv.log`).
The quality tool's two-frame static and moving-camera captures also pass with
debug/GPU validation: one measured stationary pair versus zero pairs/`null`.

Remaining work includes PT tail history validation local to edited geometry
and lighting (currently conservatively restarted), broader glass and moving
mirror coverage, and measured improvements to reconstruction stability and
the heavy indirect-lighting stages. DLSS with the pinned active runtime and
Linux compilation still require validation in an appropriate environment.
No 1440p/60 FPS acceptance claim is made yet.
