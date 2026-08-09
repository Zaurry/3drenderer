# Benchmark: san_miguel_first_scene

- Generated: `2026-08-09T02:44:35Z`
- Kind: `diagnostics`
- Compatibility key: `f2cef61abee62aa9c49120dbe0fe9710d1778e43061b830b3670985c2f478750`

| Phase | Backend | Metric | Median | P95 | Max | Unit |
|---|---|---|---:|---:|---:|---|
| asset_load | shared | `scene.asset_load.wall_ms` | 7594.21 | 7594.21 | 7594.21 | ms |
| cuda_diagnostics | cuda | `cuda.rays.primary` | 3.24737e+06 | 3.24737e+06 | 3.24737e+06 | rays |
| cuda_diagnostics | cuda | `cuda.rays.continuation` | 6.40769e+06 | 6.40769e+06 | 6.40769e+06 | rays |
| cuda_diagnostics | cuda | `cuda.rays.shadow.directional` | 4.9402e+06 | 4.9402e+06 | 4.9402e+06 | rays |
| cuda_diagnostics | cuda | `cuda.rays.shadow.point` | 0 | 0 | 0 | rays |
| cuda_diagnostics | cuda | `cuda.rays.shadow.spot` | 0 | 0 | 0 | rays |
| cuda_diagnostics | cuda | `cuda.rays.shadow.emissive` | 0 | 0 | 0 | rays |
| cuda_diagnostics | cuda | `cuda.rays.shadow.environment` | 5.06798e+06 | 5.06798e+06 | 5.06798e+06 | rays |
| cuda_diagnostics | cuda | `cuda.rays.total` | 1.96632e+07 | 1.96632e+07 | 1.96632e+07 | rays |
| cuda_diagnostics | cuda | `cuda.rays.per_camera_sample` | 6.05512 | 6.05512 | 6.05512 | rays/sample |
| cuda_diagnostics | cuda | `cuda.primary.hits` | 3.24506e+06 | 3.24506e+06 | 3.24506e+06 | paths |
| cuda_diagnostics | cuda | `cuda.primary.misses` | 2316 | 2316 | 2316 | paths |

## Histograms

### `cuda.rays.by_bounce`

```json
[
  {
    "bounce": 0,
    "rays": 3247374
  },
  {
    "bounce": 1,
    "rays": 3177325
  },
  {
    "bounce": 2,
    "rays": 3038625
  },
  {
    "bounce": 3,
    "rays": 158729
  },
  {
    "bounce": 4,
    "rays": 23380
  },
  {
    "bounce": 5,
    "rays": 6574
  },
  {
    "bounce": 6,
    "rays": 1955
  },
  {
    "bounce": 7,
    "rays": 660
  },
  {
    "bounce": 8,
    "rays": 233
  },
  {
    "bounce": 9,
    "rays": 83
  },
  {
    "bounce": 10,
    "rays": 35
  },
  {
    "bounce": 11,
    "rays": 21
  },
  {
    "bounce": 12,
    "rays": 13
  },
  {
    "bounce": 13,
    "rays": 12
  },
  {
    "bounce": 14,
    "rays": 9
  },
  {
    "bounce": 15,
    "rays": 5
  },
  {
    "bounce": 16,
    "rays": 4
  },
  {
    "bounce": 17,
    "rays": 4
  },
  {
    "bounce": 18,
    "rays": 4
  },
  {
    "bounce": 19,
    "rays": 4
  },
  {
    "bounce": 20,
    "rays": 3
  },
  {
    "bounce": 21,
    "rays": 3
  },
  {
    "bounce": 22,
    "rays": 2
  },
  {
    "bounce": 23,
    "rays": 2
  },
  {
    "bounce": 24,
    "rays": 2
  },
  {
    "bounce": 25,
    "rays": 2
  },
  {
    "bounce": 26,
    "rays": 2
  },
  {
    "bounce": 27,
    "rays": 0
  },
  {
    "bounce": 28,
    "rays": 0
  },
  {
    "bounce": 29,
    "rays": 0
  },
  {
    "bounce": 30,
    "rays": 0
  },
  {
    "bounce": 31,
    "rays": 0
  },
  {
    "bounce": 32,
    "rays": 0
  },
  {
    "bounce": 33,
    "rays": 0
  },
  {
    "bounce": 34,
    "rays": 0
  },
  {
    "bounce": 35,
    "rays": 0
  },
  {
    "bounce": 36,
    "rays": 0
  },
  {
    "bounce": 37,
    "rays": 0
  },
  {
    "bounce": 38,
    "rays": 0
  },
  {
    "bounce": 39,
    "rays": 0
  },
  {
    "bounce": 40,
    "rays": 0
  },
  {
    "bounce": 41,
    "rays": 0
  },
  {
    "bounce": 42,
    "rays": 0
  },
  {
    "bounce": 43,
    "rays": 0
  },
  {
    "bounce": 44,
    "rays": 0
  },
  {
    "bounce": 45,
    "rays": 0
  },
  {
    "bounce": 46,
    "rays": 0
  },
  {
    "bounce": 47,
    "rays": 0
  },
  {
    "bounce": 48,
    "rays": 0
  },
  {
    "bounce": 49,
    "rays": 0
  },
  {
    "bounce": 50,
    "rays": 0
  },
  {
    "bounce": 51,
    "rays": 0
  },
  {
    "bounce": 52,
    "rays": 0
  },
  {
    "bounce": 53,
    "rays": 0
  },
  {
    "bounce": 54,
    "rays": 0
  },
  {
    "bounce": 55,
    "rays": 0
  },
  {
    "bounce": 56,
    "rays": 0
  },
  {
    "bounce": 57,
    "rays": 0
  },
  {
    "bounce": 58,
    "rays": 0
  },
  {
    "bounce": 59,
    "rays": 0
  },
  {
    "bounce": 60,
    "rays": 0
  },
  {
    "bounce": 61,
    "rays": 0
  },
  {
    "bounce": 62,
    "rays": 0
  },
  {
    "bounce": 63,
    "rays": 0
  },
  {
    "bounce": 64,
    "rays": 0
  }
]
```

### `cuda.paths.bounce_histogram`

```json
[
  {
    "bounce": 0,
    "paths": 2316
  },
  {
    "bounce": 1,
    "paths": 138115
  },
  {
    "bounce": 2,
    "paths": 130379
  },
  {
    "bounce": 3,
    "paths": 2820817
  },
  {
    "bounce": 4,
    "paths": 132765
  },
  {
    "bounce": 5,
    "paths": 16529
  },
  {
    "bounce": 6,
    "paths": 4530
  },
  {
    "bounce": 7,
    "paths": 1274
  },
  {
    "bounce": 8,
    "paths": 425
  },
  {
    "bounce": 9,
    "paths": 141
  },
  {
    "bounce": 10,
    "paths": 49
  },
  {
    "bounce": 11,
    "paths": 14
  },
  {
    "bounce": 12,
    "paths": 7
  },
  {
    "bounce": 13,
    "paths": 1
  },
  {
    "bounce": 14,
    "paths": 3
  },
  {
    "bounce": 15,
    "paths": 4
  },
  {
    "bounce": 16,
    "paths": 1
  },
  {
    "bounce": 17,
    "paths": 0
  },
  {
    "bounce": 18,
    "paths": 0
  },
  {
    "bounce": 19,
    "paths": 0
  },
  {
    "bounce": 20,
    "paths": 1
  },
  {
    "bounce": 21,
    "paths": 0
  },
  {
    "bounce": 22,
    "paths": 1
  },
  {
    "bounce": 23,
    "paths": 0
  },
  {
    "bounce": 24,
    "paths": 0
  },
  {
    "bounce": 25,
    "paths": 0
  },
  {
    "bounce": 26,
    "paths": 0
  },
  {
    "bounce": 27,
    "paths": 2
  },
  {
    "bounce": 28,
    "paths": 0
  },
  {
    "bounce": 29,
    "paths": 0
  },
  {
    "bounce": 30,
    "paths": 0
  },
  {
    "bounce": 31,
    "paths": 0
  },
  {
    "bounce": 32,
    "paths": 0
  },
  {
    "bounce": 33,
    "paths": 0
  },
  {
    "bounce": 34,
    "paths": 0
  },
  {
    "bounce": 35,
    "paths": 0
  },
  {
    "bounce": 36,
    "paths": 0
  },
  {
    "bounce": 37,
    "paths": 0
  },
  {
    "bounce": 38,
    "paths": 0
  },
  {
    "bounce": 39,
    "paths": 0
  },
  {
    "bounce": 40,
    "paths": 0
  },
  {
    "bounce": 41,
    "paths": 0
  },
  {
    "bounce": 42,
    "paths": 0
  },
  {
    "bounce": 43,
    "paths": 0
  },
  {
    "bounce": 44,
    "paths": 0
  },
  {
    "bounce": 45,
    "paths": 0
  },
  {
    "bounce": 46,
    "paths": 0
  },
  {
    "bounce": 47,
    "paths": 0
  },
  {
    "bounce": 48,
    "paths": 0
  },
  {
    "bounce": 49,
    "paths": 0
  },
  {
    "bounce": 50,
    "paths": 0
  },
  {
    "bounce": 51,
    "paths": 0
  },
  {
    "bounce": 52,
    "paths": 0
  },
  {
    "bounce": 53,
    "paths": 0
  },
  {
    "bounce": 54,
    "paths": 0
  },
  {
    "bounce": 55,
    "paths": 0
  },
  {
    "bounce": 56,
    "paths": 0
  },
  {
    "bounce": 57,
    "paths": 0
  },
  {
    "bounce": 58,
    "paths": 0
  },
  {
    "bounce": 59,
    "paths": 0
  },
  {
    "bounce": 60,
    "paths": 0
  },
  {
    "bounce": 61,
    "paths": 0
  },
  {
    "bounce": 62,
    "paths": 0
  },
  {
    "bounce": 63,
    "paths": 0
  },
  {
    "bounce": 64,
    "paths": 0
  }
]
```

