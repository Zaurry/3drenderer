# OpenGL Temporal Hi-Z SSGI 验收结果

记录日期：2026-08-29。对比基线为提交 `93096ca`，变更后为同一提交上的 SSGI 工作树；两次均使用 Release、no-CUDA、同一台 NVIDIA GeForce RTX 5080 和标准 `san_miguel_first_scene` case（2418×1343，60 帧预热、300 帧测量）。本记录用于描述新增默认工作量，不设置绝对性能门槛。

运行命令：

```powershell
viewer_benchmark.exe `
  --case benchmarks/cases/san_miguel_first_scene.json `
  --backend opengl `
  --output-dir <result-directory>
```

## steady_opengl

| 指标 | SSGI 前 median | SSGI 默认开启 median | 变化 | SSGI 前 P95 | SSGI 默认开启 P95 |
|---|---:|---:|---:|---:|---:|
| CPU submit wall | 0.96 ms | 1.17 ms | +21.6% | 1.28 ms | 1.62 ms |
| GPU timer | 6.56 ms | 7.83 ms | +19.4% | 6.87 ms | 8.32 ms |

变更后默认值为半分辨率、2 rays/pixel、64 次最大 Hi-Z cell visit、4 次 refinement、32 帧最大历史和三轮 stride 1/2/4 双边 à-trous。性能数字仅可与相同 case、输入指纹、驱动和硬件上的结果比较。

## 验证

- no-CUDA Release 全量 CTest：7/7 通过。
- 321×181、3 帧真实 OpenGL smoke：`shader=active`，覆盖奇数/NPOT Hi-Z 与历史 ping-pong。
- SSGI 数学、GLSL source contract、Viewer 会话 v5 往返/迁移/夹取测试通过。
