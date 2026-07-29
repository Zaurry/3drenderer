# 3D Renderer

一个 C++20 教学型 3D 渲染器，当前由两条明确的渲染链路组成：

- `OpenGL`：OpenGL 4.5 Core + 可热重载 GLSL，面向实时预览与编辑。
- `Path`：Monte Carlo 路径追踪，支持 CPU、CUDA 和自动后端选择。

CPU 软件光栅化与 Whitted 风格光线追踪已被删除。通用的 `Ray`、BVH、`SceneIntersector`、材质与纹理系统仍由 CPU Path、CUDA Path 和编辑器拾取复用。

## 快速开始

Windows / PowerShell：

```powershell
cmake --preset default
cmake --build --preset default-release
ctest --preset default-release --output-on-failure
```

启动 Viewer：

```powershell
.\build\default\bin\viewer.exe --scene builtin
```

离线 CUDA Path：

```powershell
.\build\default\bin\renderer.exe `
  --mode path `
  --scene cornell_box `
  --width 960 `
  --height 540 `
  --spp 64 `
  --path-backend cuda `
  --output output\cornell.png
```

离线 CPU Path：

```powershell
.\build\default\bin\renderer.exe `
  --mode path `
  --scene obj_viewer `
  --obj "Computer Graphics Archive\sponza\sponza.obj" `
  --width 960 `
  --height 540 `
  --spp 16 `
  --path-backend cpu `
  --output output\sponza.png
```

`renderer` 默认就是 `--mode path`。旧的 `--mode raster` 与 `--mode ray` 会明确报错。

## 构建预设

```powershell
# 自动检测 CUDA，构建 Viewer
cmake --preset default
cmake --build --preset default-release

# 强制关闭 CUDA，验证 CPU 回退与 CUDA stub
cmake --preset cpu
cmake --build --preset cpu-release

# 强制启用 CUDA
cmake --preset cuda
cmake --build --preset cuda-release
```

无窗口构建：

```powershell
cmake -S . -B build\headless `
  -DRENDERER_BUILD_VIEWER=OFF `
  -DRENDERER_CUDA=OFF
cmake --build build\headless --config Release
ctest --test-dir build\headless -C Release --output-on-failure
```

## Viewer

Viewer 默认模式为 OpenGL，只保留两个模式：

| 快捷键 | 模式 | 用途 |
|---|---|---|
| `1` | OpenGL | 实时编辑、GLSL 热重载 |
| `2` | Path | CPU/CUDA 渐进累积 |

其他常用操作：

- `W/A/S/D`、`Space`、`Shift`：自由相机移动。
- 鼠标左键拖动、滚轮：轨道相机旋转与缩放。
- 鼠标右键：自由相机观察。
- `C`：切换轨道/自由相机。
- `Ctrl+R`：重置当前渲染。
- `F5`：重新加载 OpenGL shader。
- `Tab`：显示或隐藏编辑器界面。

不带参数启动时，Viewer 会从 `%APPDATA%\Zaurry\3D Renderer\last-session.json` 恢复上次会话。显式 CLI 调用保持确定性；`--no-restore-last` 可禁用恢复。

会话格式仍为版本 1：

- `path` 与 `opengl` 会话继续加载。
- 旧 `max_depth` 字段会被忽略。
- 保存为 `raster` 或 `ray` 的旧会话会被拒绝，并进入默认场景恢复流程。

## 渲染后端结构

模式名称、CLI 标识、快捷键与能力由 `RenderModeDescriptor` 集中定义。Viewer 通过统一后端工厂创建模式实例；所有后端返回 `RenderFrameOutput`：

- CPU Path 回退返回 Host `Framebuffer`。
- OpenGL 返回 GL texture view。
- CUDA interop 活跃时直接返回 `GL_RGBA32F` texture view。
- CUDA interop 失败时才创建 resolve buffer，并通过可复用 pinned staging 下载。

SDL 显示层只消费 `RenderFrameOutput`，不需要识别具体渲染器。

## SceneChangeSet

编辑器使用精确的场景变更分类驱动后端同步：

- `Geometry`：顶点、变换、可见拓扑；重建并上传 BVH。
- `MaterialBindings`：primitive 到 material 的绑定。
- `Materials`：材质参数。
- `Textures`：纹理描述与 texel。
- `Lighting`：环境色、点光源与方向光。
- `All`：导入、undo/redo 等无法可靠细分的操作。

相机、命名、锁定、选择等操作不会上传场景。灯光编辑只更新灯光缓冲；材质编辑不会重建 BVH。

## CUDA Path 链路

CUDA 后端使用：

- 持久 non-blocking stream。
- 按容量增长并复用的 device buffer。
- 几何、材质绑定、材质、纹理、灯光与 BVH 的独立缓冲。
- 三角形/球体 emissive primitive 的面积功率 CDF 与 primitive-to-light 映射。
- 同尺寸 reset 的原地初始化 kernel。
- Host 侧打包数组容量复用与异步上传。
- Host 构建阶段的 BVH 最大深度验证。
- CUDA 专用 16-bin SAH BVH4；叶节点最多 8 个三角形，inner/leaf 统一编码并按
  AABB 入口距离由近到远遍历。
- 求交数据使用 `v0 / edge1 / edge2` 紧凑布局，UV、normal 与 normal mask
  独立存放；每条射线只计算一次逆方向。
- continuation ray 使用 alpha-aware nearest-visible-hit；shadow ray 使用
  alpha-aware any-hit 并在首个有效遮挡处退出。
- CUDA Graph 驱动的 primary、intersection、shade/scatter、emissive sampling、
  direct visibility、accumulate/resolve Wavefront 阶段。
- 双缓冲 active/next path queue、warp 聚合入队和紧凑的
  `t / primitive id / barycentric` hit record。
- 最终可见命中确定后再重建完整法线与着色数据。
- 只有最终材质实际使用 bump texture 时才计算 tangent/bitangent。
- Diffuse bounce 对 emissive 三角形/球体执行一次 NEE，并用 β=2 power
  heuristic 与 cosine-weighted BSDF 样本做 MIS。
- Viewer 默认启用 `Auto interaction quality`：相机/场景交互使用自适应
  100%/75%/50%/25% 内部分辨率与最多 2 个 shading bounce；连续静止 8 帧后
  恢复原生 64-bounce 累积。
- 原生累积按 1–64 行水平 tile 调度，每次 Viewer 循环只提交一个 GPU work
  quantum；完整 sweep 后才增加 1 spp。低分辨率预览会放大写入完整 interop
  surface，原生 tile 随后逐块覆盖。

Performance 面板分别显示 UI FPS、完整原生 spp、GPU 工作模式、内部分辨率、
tile/sweep 进度、GPU trace/reset/upload 时间、各类累计上传字节、分配代次、
framebuffer 下载次数及 interop/fallback 状态。

重场景基准、真实相机与 RTX 5080 结果见
[CUDA 重场景性能结果](docs/output/cuda-heavy-scene-performance-results.md)。

CUDA Path 不依赖 OptiX 或 RT Core。这里的“降噪”来自 NEE/MIS 降低 Monte
Carlo 方差；没有引入 OptiX/OIDN 等后处理降噪器。CPU Path 的采样与输出算法保持不变。

## CLI

```text
renderer --mode path --scene gradient_sphere|triangle|mirror_spheres|cornell_box|obj_viewer
         --output file.png
         [--obj file.obj]
         [--width N] [--height N] [--spp N]
         [--threads N]
         [--path-backend auto|cpu|cuda]
```

`auto` 在 CUDA 可用时选择 CUDA，否则回退 CPU。显式 `cuda` 在不可用时会报错，不会静默换后端。

Viewer：

```text
viewer --scene builtin|asset
       [--asset file-or-directory ...]
       [--scene-file scene.rscene]
       [--mode opengl|path]
       [--path-backend auto|cpu|cuda]
       [--width N] [--height N] [--frames N]
```

## 目录

```text
src/
  acceleration/             BVH
  core/                     数学、颜色、图像、随机数、计时
  interactive/              UI、相机、会话
  platform/opengl/          Shader 与 CUDA/OpenGL interop
  platform/sdl/             窗口、输入、统一显示
  render/interactive/       模式目录、SceneChangeSet、Path session、统一帧输出与后端
  render/opengl/            OpenGL renderer
  render/pathtracer/        CPU/CUDA Path
  scene/                    场景、材质、纹理、OBJ/MTL、编辑器文档
tests/
docs/
shaders/opengl/
```

当前架构说明：

- [像素生命周期](docs/render-pixel-lifecycle.md)
- [场景对象系统](docs/scene-object-system.md)
- [GLSL Shader 合约](docs/glsl-shader-contract.md)
- [CUDA pipeline 优化结果](docs/output/cuda-pipeline-optimization-results.md)

`docs/plans/`、日期化 `docs/specs/` 与旧 `docs/output/` 是历史记录；其中涉及 CPU Raster/Whitted 的入口已标记为过期。
