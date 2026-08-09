# Benchmark: san_miguel_first_scene

- Generated: `2026-08-09T08:22:08Z`
- Kind: `timing`
- Compatibility key: `803cc7c26c07dc862d9ce14b8bbdc827943fd678a0c4d5ef7a4cbb08c9dbd56c`

| Phase | Backend | Metric | Median | P95 | Max | Unit |
|---|---|---|---:|---:|---:|---|
| asset_load | shared | `scene.asset_load.wall_ms` | 7699.07 | 7699.07 | 7699.07 | ms |
| backend_prepare | cuda | `cuda.backend_prepare.wall_ms` | 4587.7 | 4587.7 | 4587.7 | ms |
| backend_prepare | cuda | `cuda.upload.last_ms` | 4571.88 | 4571.88 | 4571.88 | ms |
| backend_prepare | cuda | `cuda.upload.cumulative_geometry_bytes` | 1.30325e+09 | 1.30325e+09 | 1.30325e+09 | bytes |
| backend_prepare | cuda | `cuda.upload.cumulative_texture_bytes` | 1.43881e+09 | 1.43881e+09 | 1.43881e+09 | bytes |
| backend_prepare | cuda | `cuda.upload.cumulative_lighting_bytes` | 1.67772e+08 | 1.67772e+08 | 1.67772e+08 | bytes |
| backend_prepare | cuda | `cuda.blas.build_count` | 1 | 1 | 1 | count |
| backend_prepare | cuda | `cuda.tlas.build_count` | 1 | 1 | 1 | count |
| backend_prepare | cuda | `cuda.framebuffer.downloads` | 0 | 0 | 0 | count |
| first_frame | cuda | `cuda.first_frame.wall_ms` | 226.262 | 226.262 | 226.262 | ms |
| first_frame | cuda | `cuda.first_frame.trace_ms` | 224.343 | 224.343 | 224.343 | ms |
| cuda_full | cuda | `cuda.frame.wall_ms` | 216.558 | 217.19 | 217.19 | ms |
| cuda_full | cuda | `cuda.frame.trace_ms` | 216.207 | 217.033 | 217.033 | ms |
| cuda_full | cuda | `cuda.frame.presentation_ms` | 0 | 0 | 0 | ms |
| cuda_interaction | cuda | `cuda.interaction.wall_ms` | 10.0695 | 10.5518 | 10.8796 | ms |
| cuda_interaction | cuda | `cuda.interaction.trace_ms` | 9.80589 | 10.3271 | 10.5901 | ms |
| cuda_interaction | cuda | `cuda.interaction.internal_pixels` | 203280 | 203280 | 203280 | pixels |
| cuda_native_sweep | cuda | `cuda.native_quantum.wall_ms` | 12.1211 | 14.193 | 18.2986 | ms |
| cuda_native_sweep | cuda | `cuda.native_quantum.trace_ms` | 11.8243 | 13.8644 | 17.8623 | ms |
| cuda_native_sweep | cuda | `cuda.native.complete_sweeps_per_second` | 1.88648 | 1.92947 | 1.92947 | sweeps/s |
| cuda_native_sweep | cuda | `cuda.native.completed_sweeps` | 10 | 10 | 10 | count |
| cuda_native_sweep | cuda | `cuda.upload.last_ms` | 4563.35 | 4563.35 | 4563.35 | ms |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_geometry_bytes` | 3.90975e+09 | 3.90975e+09 | 3.90975e+09 | bytes |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_texture_bytes` | 4.31643e+09 | 4.31643e+09 | 4.31643e+09 | bytes |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_lighting_bytes` | 5.03317e+08 | 5.03317e+08 | 5.03317e+08 | bytes |
| cuda_native_sweep | cuda | `cuda.blas.build_count` | 3 | 3 | 3 | count |
| cuda_native_sweep | cuda | `cuda.tlas.build_count` | 3 | 3 | 3 | count |
| cuda_native_sweep | cuda | `cuda.framebuffer.downloads` | 0 | 0 | 0 | count |
