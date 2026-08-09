# Benchmark: san_miguel_first_scene

- Generated: `2026-08-09T08:08:20Z`
- Kind: `timing`
- Compatibility key: `cbb898c89a635aa8ba948375f7c43c2b2b9c211fbbb75a9fcccdaee58a49742f`

| Phase | Backend | Metric | Median | P95 | Max | Unit |
|---|---|---|---:|---:|---:|---|
| asset_load | shared | `scene.asset_load.wall_ms` | 7796.51 | 7796.51 | 7796.51 | ms |
| backend_prepare | cuda | `cuda.backend_prepare.wall_ms` | 4332.05 | 4332.05 | 4332.05 | ms |
| backend_prepare | cuda | `cuda.upload.last_ms` | 4276.88 | 4276.88 | 4276.88 | ms |
| backend_prepare | cuda | `cuda.upload.cumulative_geometry_bytes` | 1.30325e+09 | 1.30325e+09 | 1.30325e+09 | bytes |
| backend_prepare | cuda | `cuda.upload.cumulative_texture_bytes` | 1.43881e+09 | 1.43881e+09 | 1.43881e+09 | bytes |
| backend_prepare | cuda | `cuda.upload.cumulative_lighting_bytes` | 1.67772e+08 | 1.67772e+08 | 1.67772e+08 | bytes |
| backend_prepare | cuda | `cuda.blas.build_count` | 1 | 1 | 1 | count |
| backend_prepare | cuda | `cuda.tlas.build_count` | 1 | 1 | 1 | count |
| backend_prepare | cuda | `cuda.framebuffer.downloads` | 0 | 0 | 0 | count |
| first_frame | cuda | `cuda.first_frame.wall_ms` | 211.871 | 211.871 | 211.871 | ms |
| first_frame | cuda | `cuda.first_frame.trace_ms` | 209.999 | 209.999 | 209.999 | ms |
| cuda_full | cuda | `cuda.frame.wall_ms` | 210.593 | 212.609 | 212.609 | ms |
| cuda_full | cuda | `cuda.frame.trace_ms` | 210.359 | 212.448 | 212.448 | ms |
| cuda_full | cuda | `cuda.frame.presentation_ms` | 0 | 0 | 0 | ms |
| cuda_interaction | cuda | `cuda.interaction.wall_ms` | 10.7114 | 10.9746 | 11.0126 | ms |
| cuda_interaction | cuda | `cuda.interaction.trace_ms` | 10.41 | 10.6517 | 10.792 | ms |
| cuda_interaction | cuda | `cuda.interaction.internal_pixels` | 203280 | 203280 | 203280 | pixels |
| cuda_native_sweep | cuda | `cuda.native_quantum.wall_ms` | 12.0626 | 14.2267 | 17.1618 | ms |
| cuda_native_sweep | cuda | `cuda.native_quantum.trace_ms` | 11.7738 | 13.9575 | 16.8985 | ms |
| cuda_native_sweep | cuda | `cuda.native.complete_sweeps_per_second` | 1.85734 | 1.888 | 1.888 | sweeps/s |
| cuda_native_sweep | cuda | `cuda.native.completed_sweeps` | 10 | 10 | 10 | count |
| cuda_native_sweep | cuda | `cuda.upload.last_ms` | 4276.59 | 4276.59 | 4276.59 | ms |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_geometry_bytes` | 3.90975e+09 | 3.90975e+09 | 3.90975e+09 | bytes |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_texture_bytes` | 4.31643e+09 | 4.31643e+09 | 4.31643e+09 | bytes |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_lighting_bytes` | 5.03317e+08 | 5.03317e+08 | 5.03317e+08 | bytes |
| cuda_native_sweep | cuda | `cuda.blas.build_count` | 3 | 3 | 3 | count |
| cuda_native_sweep | cuda | `cuda.tlas.build_count` | 3 | 3 | 3 | count |
| cuda_native_sweep | cuda | `cuda.framebuffer.downloads` | 0 | 0 | 0 | count |
