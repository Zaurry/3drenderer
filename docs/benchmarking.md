# 规范化 Benchmark 流程

Benchmark 是开发工具，与普通 `viewer.exe` 隔离。默认 `RENDERER_BUILD_BENCHMARKS=OFF`；正式 timing target 不包含 CUDA ray/bounce 诊断计数，instrumented diagnostics 使用独立 target。

## 标准 case

当前标准 case 为 `benchmarks/cases/san_miguel_first_scene.json`，冻结 San Miguel、Sun、HDRI、相机、输出分辨率、Path 设置和 phase 次数。资产位于 Git 忽略的 `Computer Graphics Archive`，至少需要：

- `San_Miguel/san-miguel-low-poly.obj` 及其 MTL/纹理；
- `environment map/quarry_04_puresky_4k.exr`。

runner 不读取 `%APPDATA%`，也不恢复 Viewer 会话。

## 运行

```powershell
# OpenGL + CUDA timing，并单独运行 CUDA diagnostics
.\tools\run_benchmarks.ps1

# 只测一个 backend
.\tools\run_benchmarks.ps1 -Backend opengl
.\tools\run_benchmarks.ps1 -Backend cuda

# 与已有 timing/raw.json 记录比较
.\tools\run_benchmarks.ps1 `
  -Backend cuda `
  -Baseline benchmarks/results/<run>/timing/raw.json
```

脚本会：

1. 用 `cuda-native` + `RENDERER_BUILD_BENCHMARKS=ON` 配置；
2. 构建 `viewer_benchmark`、`viewer_benchmark_diagnostics` 和两类 benchmark tests；
3. 先运行 benchmark contract tests；
4. 分别执行无插桩 timing 和有插桩 diagnostics；
5. 保存原始样本、统计摘要、Git dirty 状态、工具链和硬件信息。

## Phase 和指标

| Phase | 内容 |
|---|---|
| `asset_load` | `.rscene`、OBJ/MTL/纹理、EXR、环境重要性分布加载 |
| `backend_prepare` | OpenGL shader/IBL 或 CUDA snapshot/BLAS/TLAS/environment 上传 |
| `first_frame` | backend 准备后的第一帧 |
| `steady_opengl` | 预热后测 CPU submit wall 和 GPU timer query |
| `cuda_full` | 固定完整原生帧，显式等待 event 后取样 |
| `cuda_interaction` | 连续相机交互帧，排除首个 warm-up |
| `cuda_native_sweep` | 静止后的 native work quantum 和完整 sweep |
| `cuda_diagnostics` | 独立插桩构建中的 ray 分类、bounce 和 termination histogram |

与场景增量同步相关的验收不只看 wall time，还检查 `CudaPathStatistics`：

- 纯 instance transform：BLAS build、geometry/material/texture upload 不增加；
- TLAS `refit_count` 增加，`build_count` 不增加；
- 单一 asset geometry revision：只增加该 asset 的一次 BLAS build；
- material/texture/light/environment 修改只增加相应上传计数。

## 结果格式

```text
benchmarks/results/<UTC>-<commit>/
  raw.json
  summary.md
  timing/raw.json
  timing/summary.md
  diagnostics/raw.json
  diagnostics/summary.md
```

series metric 保存全部 samples 和 count/min/mean/median/P95/max/stddev，histogram 保存完整 bins。报告还记录：

- case schema 与 render settings；
- 输入文件相对路径、字节数和 SHA-256；
- effective render size；
- Git commit/dirty；
- compiler、build type、CPU、RAM、GPU、OpenGL、CUDA runtime/driver。

只有 compatibility key 相同的结果才适合由 runner 自动判定。若架构迁移同时改变 case schema 或输入指纹，必须在人工报告中明确列出相同 workload、命令和硬件，再进行手工 delta 计算，不能把不同 key 伪装成自动可比。

## 当前验收门槛

Renderer hardening 使用以下门槛：

- `cuda_full` median 相对基线回退不超过 5%；
- `cuda_interaction` median 不高于 16.7 ms；
- `cuda_interaction` P95 不高于 20 ms；
- 纯实例拖动不触发 BLAS、geometry/material/texture upload；
- TLAS 只 refit。

本次基线与结果见：

- `benchmarks/results/baseline-75d1992-hot-cuda/`
- `benchmarks/results/renderer-hardening-acceptance2-cuda/`
- [Renderer hardening 报告](output/renderer-hardening-results.md)
- [OpenGL Temporal Hi-Z SSGI 前后记录](output/opengl-ssgi-results.md)

## Diagnostics 与 Sanitizer

diagnostics target 使用 `instrumented: true`，耗时不能作为性能 baseline。它记录 primary/continuation、各类 shadow ray、hit/miss 和 bounce termination。

Compute Sanitizer 必须单独运行真实 CUDA executable，例如：

```powershell
compute-sanitizer --tool memcheck --error-exitcode 99 `
  .\build\cuda-native\bin\cuda_contract_tests.exe
```

普通 GitHub-hosted CI 不具备受支持的显示连接 GPU，因此 GPU tests 不得以“检测不到设备后 return 0”的方式计为覆盖；应在开发机或 self-hosted runner 明确执行并保存结果。

## 扩展规则

- 新场景优先增加 case JSON/`.rscene`，不要复制 runner。
- 新 phase 使用新的稳定命名空间，不能复用语义不同的旧 metric。
- 新 metric 必须声明单位、scalar/series/histogram 和是否插桩。
- 会改变 GPU 工作量的计数只进入 diagnostics target。
- 新增 `RenderSettings` 字段时，同步更新 case、session import、metadata 和 compatibility key。
- 历史 `docs/output` 中的旧 runner/CLI 只代表当时环境，不是当前入口。
