# 规范化 Benchmark 流程

Benchmark 是开发工具，与普通用户使用的 `viewer.exe` 隔离。默认配置
`RENDERER_BUILD_BENCHMARKS=OFF`，因此常规构建和发布不会编译、链接或打包任何
benchmark runner、报告代码或 CUDA 诊断计数。

## 标准运行

当前标准 case 是 `benchmarks/cases/san_miguel_first_scene.json`。它冻结了
2026-08-08 捕获的 San Miguel、Sun、HDRI、相机、分辨率和完整 Path 设置，运行时
不会读取 `%APPDATA%` 或恢复 Viewer 的当前会话。

资产仍位于被 Git 忽略的 `Computer Graphics Archive`。运行前需要存在：

- `San_Miguel/san-miguel-low-poly.obj` 及其 MTL/纹理；
- `environment map/quarry_04_puresky_4k.exr`。

标准命令：

```powershell
.\tools\run_benchmarks.ps1
```

脚本使用 CUDA Release preset，构建并运行：

- `viewer_benchmark.exe`：无插桩 OpenGL/CUDA timing；
- `viewer_benchmark_diagnostics.exe`：带 CUDA ray/bounce 计数；
- `benchmark_tests.exe` 与 `benchmark_diagnostics_tests.exe`。

只测单个后端：

```powershell
.\tools\run_benchmarks.ps1 -Backend opengl
.\tools\run_benchmarks.ps1 -Backend cuda
```

与既有 timing `raw.json` 比较：

```powershell
.\tools\run_benchmarks.ps1 `
  -Baseline benchmarks/results/20260809-120000Z-abcdef123456/timing/raw.json
```

首版只记录差异，不因回退自动返回失败。只有 compatibility key 相同的结果会比较；
key 包含 case/schema、完整渲染参数、输入 SHA-256 和硬件身份。

## 测量口径

标准完整档按以下 phase 输出独立数据：

| Phase | 内容 |
|---|---|
| `asset_load` | `.rscene`、OBJ/MTL/纹理、EXR 和环境重要性分布加载 |
| `backend_prepare` | OpenGL shader/资源/IBL，或 CUDA scene/BVH/environment 上传 |
| `first_frame` | 后端准备后的第一帧 |
| `steady_opengl` | 预热 60 帧后测量 300 帧 CPU submit wall 与 GPU timer query |
| `cuda_full` | 预热后 5 个原生完整帧，每帧同步后读取 CUDA event |
| `cuda_interaction` | 30 个相机变化帧，排除第一个 warm-up |
| `cuda_native_sweep` | 静止后至少 10 个完整 sweep 及其 native quantum |
| `cuda_diagnostics` | 独立插桩构建中的 ray 分类与 bounce 分布 |

CUDA timing runner 显式等待每个样本完成，再读取同一帧的 event；不会把异步提交耗时
误记成 GPU trace。OpenGL 同时记录 CPU submit wall 和 `GL_TIME_ELAPSED`，二者口径
不同，不互相替代。

diagnostics 统计实际进入求交或 visibility 的 ray：

- primary/camera ray；
- 每个 bounce 的 continuation ray；
- directional、point、spot、emissive、environment shadow ray；
- primary hit/miss；
- 0–64 bounce termination histogram。

诊断构建标记为 `instrumented: true`，其耗时不能作为性能 baseline。生产
`renderer_core` 不包含诊断 buffer、kernel 参数、原子计数或下载同步。

## 结果格式

每次标准运行生成：

```text
benchmarks/results/<UTC>-<commit>/
  raw.json                 # timing + diagnostics 合并数据
  summary.md               # 合并的人类可读报告
  timing/raw.json
  timing/summary.md
  diagnostics/raw.json
  diagnostics/summary.md
```

每个 metric 使用稳定命名空间、单位、原始 samples 和 count/min/mean/median/P95/max/
stddev。histogram 保留全部 bins，不只保存摘要。

`output.width/height` 是 Viewer 窗口尺寸，`render_scale` 遵循 Viewer 的 25%–100%
范围；报告中的 `effective_width/effective_height` 是后端实际渲染尺寸。

输入指纹包含 case、`.rscene`、OBJ、MTL、MTL 实际引用的纹理和环境图的相对路径、
字节数及 SHA-256。报告同时保存 Git commit/dirty、编译器、Release 配置、CPU、内存、
GPU、OpenGL 和 CUDA runtime/driver。

## 从 Viewer 会话创建新 case

当前 `%APPDATA%` 会话会随普通使用持续变化，不能直接作为 benchmark。需要显式冻结：

```powershell
$session = Join-Path $env:APPDATA 'Zaurry\3D Renderer\last-session.json'
.\build\cuda\bin\viewer_benchmark.exe `
  --import-session $session `
  --write-case benchmarks\cases\my_scene.json
```

该命令同时生成相邻的 `.rscene`。提交前应把其中资产路径调整为相对仓库根目录可解析的
稳定路径，并检查完整 RenderSettings、相机、输出尺寸和 phase 次数。

## 扩展规则

- 新场景只增加 case JSON/`.rscene`；runner 和报告 schema 不需要修改。
- 新 phase 通过 benchmark backend adapter 实现，不能复用含义不同的旧指标名。
- 新 metric 使用稳定命名空间并声明单位、是否插桩以及 scalar/series/histogram 形态。
- 会改变 GPU 工作量的计数必须只进入 diagnostics target。
- 新增 `RenderSettings` 字段时，case 读取、session import、结果 metadata 和
  compatibility key 必须同步覆盖。
- glTF/GLB case 已被 schema 和依赖指纹逻辑支持，但首版标准档暂不包含 PBR case。

旧的 `cuda_path_benchmark` 和临时 bounce 分析脚本已经删除；`docs/output` 中引用它们
的命令只代表历史测量环境，不再是当前入口。
