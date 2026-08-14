# 3D Renderer

一个 C++20 教学型 3D 渲染器，提供两条产品渲染链路：

- `OpenGL`：OpenGL 4.5 Core 实时预览，支持可热重载 GLSL、PBR、IBL、SSAO、GTAO + Bent Normal、Shadow Map/PCSS、环境主光提取、LTC 矩形面光和 weighted blended OIT。
- `Path`：CUDA Wavefront 路径追踪，支持渐进累积、环境/发光几何 NEE + MIS、矩形面光，以及 CUDA/OpenGL interop。

CPU Path 已从产品、CLI 和交互会话中删除。没有 CUDA 时仍可构建场景/文档系统、测试和 OpenGL Viewer；Path 模式会明确显示不可用原因，并回退到 OpenGL。

## 快速开始

日常构建会在存在 CUDA 编译器时启用 CUDA，否则构建 no-CUDA 版本：

```powershell
cmake --preset default
cmake --build --preset default-release
ctest --preset default-release --output-on-failure
```

启动 Viewer：

```powershell
.\build\default\bin\viewer.exe --scene builtin --mode opengl --no-restore-last
```

加载静态 OBJ、glTF/GLB 和 HDRI：

```powershell
.\build\default\bin\viewer.exe `
  --scene asset `
  --asset D:\assets\scene.glb `
  --environment D:\assets\studio.hdr `
  --environment-intensity 1.0 `
  --mode opengl `
  --no-restore-last
```

离线 CUDA Path：

```powershell
.\build\default\bin\renderer.exe `
  --scene cornell_box `
  --width 960 `
  --height 540 `
  --spp 64 `
  --cuda-device 0 `
  --output output\cornell.png
```

`renderer.exe` 直接使用 CUDA；`--mode path` 仅作为旧命令兼容 no-op。`--threads`、`--path-backend`、CPU/Auto backend 均已删除。

## 构建预设

项目要求 CMake 3.24 或更高版本。

```powershell
# CUDA 可用时自动启用，默认使用本机 GPU 架构
cmake --preset default
cmake --build --preset default-release

# 明确关闭 CUDA；仍构建 Viewer 和全部非 GPU 功能
cmake --preset no-cuda
cmake --build --preset no-cuda-release
ctest --preset no-cuda-release --output-on-failure

# 要求 CUDA，并为当前机器的 GPU 编译
cmake --preset cuda-native
cmake --build --preset cuda-native-release
ctest --preset cuda-native-release --output-on-failure
```

发行包应明确给出支持的架构，而不是使用 `native`：

```powershell
cmake --preset cuda-native -DCMAKE_CUDA_ARCHITECTURES="89;120"
cmake --build --preset cuda-native-release
```

无窗口核心构建：

```powershell
cmake -S . -B build\headless `
  -DRENDERER_BUILD_VIEWER=OFF `
  -DRENDERER_CUDA=OFF `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build\headless --config Release
ctest --test-dir build\headless -C Release --output-on-failure
```

## Viewer

Viewer 只有两种模式：

| 快捷键 | 模式 | 行为 |
|---|---|---|
| `1` | OpenGL | 实时编辑与 GLSL 热重载 |
| `2` | Path | CUDA 渐进路径追踪；无可用 CUDA 时禁用 |

常用操作：

- `W/A/S/D`、`Space`、`Shift`：自由相机移动。
- 鼠标左键拖动、滚轮：轨道相机旋转与缩放。
- 鼠标右键：自由相机观察。
- `C`：切换轨道/自由相机。
- `Ctrl+R`：重置当前渲染。
- `F5`：仅在 OpenGL 模式重载 shader。
- `Tab`：显示或隐藏编辑器界面。

Viewer 会把会话 v4 保存到 `%APPDATA%\Zaurry\3D Renderer\last-session.json`。不带参数启动时可恢复上次会话；`--no-restore-last` 禁用恢复。旧会话 v1–v3 可只读迁移：新增 OpenGL 技术参数使用默认值，CPU/Auto Path 字段会被忽略；CUDA 不可用时自动转到 OpenGL，并显示迁移警告。

默认可见的 `Techniques` 面板把 IBL、Shadow Map、PCSS、Dominant Light Extraction 和 LTC Area Lights 集中在 `Direct Lighting`，并提供独立的 `Ambient Occlusion` 区域切换 SSAO/GTAO、Bent Normal、空间滤波和调试视图。阴影默认使用 1024 分辨率、最多 8 盏灯；AO 默认使用全分辨率 GTAO。OpenGL 技术控件在 Path 模式中禁用，Path 始终使用完整环境和真实光线可见性。

## 场景与后端边界

`SceneDocument` 是编辑真值，所有渲染、拾取和离线输出都从同一份 `RenderSceneSnapshot` 派生：

```text
SceneDocument transaction
  -> typed edit + revision update + undo/dirty state
  -> immutable RenderSceneSnapshot
       -> OpenGL
       -> CUDA Path
       -> CPU picking BVH
       -> offline renderer
```

快照包含共享几何资产、实例矩阵、强类型材质槽、纹理、灯光、环境，以及八个单调 revision：`topology`、`geometry`、`transforms`、`material_bindings`、`materials`、`textures`、`lighting`、`environment`。后端自行比较 revision；UI 不负责维护上传失效状态。

对象的有限、可逆本地 4×4 仿射矩阵是真值。TRS 只是可分解矩阵的编辑视图，因此剪切矩阵可无损导入、重父级、撤销和保存。`.rscene` 当前写入 v5；v1–v4 只读迁移。

程序球在进入渲染快照前统一映射到一份共享的 64×32 平滑单位球网格，center/radius 进入实例矩阵，因此 OpenGL、CUDA、拾取和离线输出共享三角形、法线、UV 与材质槽语义。

矩形面光在快照中同时保留解析灯光数据和一份共享单位四边形实例。OpenGL 用解析矩形与内置 64×64 LTC LUT 求值；CUDA Path 把可见四边形作为两个发光三角形加入现有 emissive NEE/MIS 分布。LTC 数据来自 Eric Heitz 等人的 `selfshadow/ltc_code`，许可和论文引用见 `shaders/opengl/LTC_LICENSE.txt`。

## CUDA 设备与帧生命周期

Viewer 在创建任何 CUDA buffer、stream 或 interop 资源之前选择与当前 OpenGL context 兼容的设备；离线程序默认设备 0，可用 `--cuda-device N` 覆盖。资源记录所属 device ID，跨设备使用会被拒绝。可用性检查包含设备选择和最小探测 kernel，不以设备数量代替可执行性。

交互后端返回拥有生命周期租约的 `RenderFrameOutput`：

- `HostFrameHandle`：interop fallback 的只读 framebuffer。
- `OpenGlTextureHandle`：OpenGL 或 CUDA interop texture，并持有资源 owner。

统计信息使用 OpenGL/CUDA 类型化 variant。shader reload 控制只由 OpenGL 后端暴露。

## CLI

离线渲染：

```text
renderer [--mode path]
         --scene gradient_sphere|triangle|mirror_spheres|cornell_box|asset_viewer
         --output file.png
         [--asset file.obj|file.gltf|file.glb]
         [--obj file.obj]
         [--environment file.hdr|file.exr|file.png|file.jpg]
         [--environment-intensity value] [--environment-yaw degrees]
         [--hide-environment-background]
         [--width N] [--height N] [--spp N]
         [--cuda-device N]
```

Viewer：

```text
viewer --scene builtin|asset
       [--asset file-or-directory ...]
       [--scene-file scene.rscene]
       [--environment file.hdr|file.exr|file.png|file.jpg]
       [--environment-intensity value] [--environment-yaw degrees]
       [--hide-environment-background]
       [--mode opengl|path]
       [--cuda-device N]
       [--width N] [--height N] [--frames N]
       [--no-restore-last]
```

## 测试与基准

测试按领域拆分为 scene、sampling、document、OpenGL contract、CUDA contract、综合 renderer 和 benchmark targets。所有测试二进制共享一个注册表式测试运行器（支持 `--filter` 与显式 skip）；在无 CUDA 的构建上，需要 GPU 的用例会打印 `[SKIP]` 并通过 CTest 的 `SKIP_RETURN_CODE=77` 显示为 skipped，不会以绿色静默通过。普通 GitHub Actions 运行 Windows/Linux no-CUDA Release、Linux Mesa/Xvfb smoke 和 Linux ASan/UBSan core；需要真实 GPU 的 CUDA、interop 和 Compute Sanitizer 测试只在本机或自托管 runner 上运行。

San Miguel 规范化基准：

```powershell
.\tools\run_benchmarks.ps1
.\tools\run_benchmarks.ps1 -Backend cuda
```

详见 [Benchmark 流程](docs/benchmarking.md) 和 [Renderer hardening 结果](docs/output/renderer-hardening-results.md)。

## 文档

- [场景对象系统](docs/scene-object-system.md)
- [像素生命周期](docs/render-pixel-lifecycle.md)
- [GLSL Shader 合约](docs/glsl-shader-contract.md)
- [实时环境光、glTF 与 PBR](docs/realtime-environment-gltf-pbr.md)
- [Benchmark 流程](docs/benchmarking.md)
- [本次架构硬化报告](docs/output/renderer-hardening-results.md)

`docs/plans/`、日期化 `docs/specs/` 和旧 `docs/output/` 是历史记录，不代表当前入口或兼容承诺。
