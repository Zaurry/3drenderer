# Benchmark: san_miguel_first_scene

- Generated: `2026-08-09T02:43:54Z`
- Kind: `timing`
- Compatibility key: `52f9cde32e0fa2be2cbf294ac4e80f00a74e1da00dfe8e36ff932a994f45293c`

| Phase | Backend | Metric | Median | P95 | Max | Unit |
|---|---|---|---:|---:|---:|---|
| asset_load | shared | `scene.asset_load.wall_ms` | 7625.61 | 7625.61 | 7625.61 | ms |
| backend_prepare | opengl | `opengl.backend_prepare.wall_ms` | 3013.36 | 3013.36 | 3013.36 | ms |
| first_frame | opengl | `opengl.first_frame.wall_ms` | 11.1812 | 11.1812 | 11.1812 | ms |
| first_frame | opengl | `opengl.first_frame.gpu_ms` | 3.99146 | 3.99146 | 3.99146 | ms |
| steady_opengl | opengl | `opengl.frame.wall_ms` | 0.31525 | 0.4065 | 1.8169 | ms |
| steady_opengl | opengl | `opengl.frame.gpu_ms` | 2.47986 | 2.6952 | 2.8991 | ms |
| backend_prepare | cuda | `cuda.backend_prepare.wall_ms` | 4424.5 | 4424.5 | 4424.5 | ms |
| backend_prepare | cuda | `cuda.upload.last_ms` | 4303.35 | 4303.35 | 4303.35 | ms |
| backend_prepare | cuda | `cuda.upload.cumulative_geometry_bytes` | 1.30325e+09 | 1.30325e+09 | 1.30325e+09 | bytes |
| backend_prepare | cuda | `cuda.upload.cumulative_texture_bytes` | 1.43881e+09 | 1.43881e+09 | 1.43881e+09 | bytes |
| backend_prepare | cuda | `cuda.upload.cumulative_lighting_bytes` | 1.67772e+08 | 1.67772e+08 | 1.67772e+08 | bytes |
| backend_prepare | cuda | `cuda.blas.build_count` | 1 | 1 | 1 | count |
| backend_prepare | cuda | `cuda.tlas.build_count` | 1 | 1 | 1 | count |
| backend_prepare | cuda | `cuda.framebuffer.downloads` | 0 | 0 | 0 | count |
| first_frame | cuda | `cuda.first_frame.wall_ms` | 202.781 | 202.781 | 202.781 | ms |
| first_frame | cuda | `cuda.first_frame.trace_ms` | 201.164 | 201.164 | 201.164 | ms |
| cuda_full | cuda | `cuda.frame.wall_ms` | 201.866 | 204.191 | 204.191 | ms |
| cuda_full | cuda | `cuda.frame.trace_ms` | 201.688 | 204.009 | 204.009 | ms |
| cuda_full | cuda | `cuda.frame.presentation_ms` | 0 | 0 | 0 | ms |
| cuda_interaction | cuda | `cuda.interaction.wall_ms` | 9.7342 | 9.8188 | 9.8248 | ms |
| cuda_interaction | cuda | `cuda.interaction.trace_ms` | 9.48144 | 9.55616 | 9.55795 | ms |
| cuda_interaction | cuda | `cuda.interaction.internal_pixels` | 203280 | 203280 | 203280 | pixels |
| cuda_native_sweep | cuda | `cuda.native_quantum.wall_ms` | 12.0061 | 14.24 | 17.3038 | ms |
| cuda_native_sweep | cuda | `cuda.native_quantum.trace_ms` | 11.7828 | 13.9629 | 17.0902 | ms |
| cuda_native_sweep | cuda | `cuda.native.complete_sweeps_per_second` | 2.33616 | 2.38867 | 2.38867 | sweeps/s |
| cuda_native_sweep | cuda | `cuda.native.completed_sweeps` | 10 | 10 | 10 | count |
| cuda_native_sweep | cuda | `cuda.upload.last_ms` | 4462.77 | 4462.77 | 4462.77 | ms |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_geometry_bytes` | 3.90975e+09 | 3.90975e+09 | 3.90975e+09 | bytes |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_texture_bytes` | 4.31643e+09 | 4.31643e+09 | 4.31643e+09 | bytes |
| cuda_native_sweep | cuda | `cuda.upload.cumulative_lighting_bytes` | 5.03317e+08 | 5.03317e+08 | 5.03317e+08 | bytes |
| cuda_native_sweep | cuda | `cuda.blas.build_count` | 3 | 3 | 3 | count |
| cuda_native_sweep | cuda | `cuda.tlas.build_count` | 3 | 3 | 3 | count |
| cuda_native_sweep | cuda | `cuda.framebuffer.downloads` | 0 | 0 | 0 | count |
