# 3D Renderer

一个从零实现的 C++20 3D 渲染器，用于学习软件/OpenGL 光栅化、GLSL、光线求交、BVH、Whitted 光线追踪和 Monte Carlo 路径追踪。

Raster 与 Whitted Ray 运行在 CPU 上；OpenGL 模式使用 OpenGL 4.5 Core 和可热重载 GLSL；Path 模式支持 CPU 与 CUDA Core 后端。SDL3 负责窗口与输入，OpenGL 统一显示四种模式的线性 framebuffer 并叠加 Dear ImGui。CUDA Path 在兼容设备上通过 CUDA–OpenGL interop 直接写入显示 texture，失败时自动回退 CPU staging；Path 后端不依赖 OptiX 或 RT Core。

Viewer 内置完整的场景对象系统：可同时导入多个 OBJ 或整个目录，以层级对象组织共享网格资产，为每个对象独立设置变换和材质覆盖，并将编辑结果保存为 `.rscene`。

## 快速开始

需要 CMake 3.21+、支持 C++20 的编译器和 OpenGL 4.5 驱动。CUDA 是可选依赖。

```powershell
cmake --preset default
cmake --build --preset default-release
ctest --preset default-release

.\build\default\bin\viewer.exe --scene builtin --mode raster
.\build\default\bin\viewer.exe --scene builtin --mode opengl
```

首次配置时，CMake 会按需获取缺失的 Eigen3、SDL3 和 Dear ImGui 依赖；GLAD 4.5 Core loader 已固定在仓库中。

## 构建预设

所有生成文件都位于 `build/` 下，避免在仓库根目录散落多个构建树。

| Preset | 构建目录 | CUDA | 用途 |
| --- | --- | --- | --- |
| `default` | `build/default` | `AUTO` | 日常开发；有 CUDA 编译器时启用 CUDA，否则使用 CPU |
| `cpu` | `build/cpu` | `OFF` | CPU-only 构建与回归测试 |
| `cuda` | `build/cuda` | `ON` | 强制验证 CUDA 工具链和 CUDA 后端，默认架构为 `120` |

每个 configure preset 都有对应的 Release build/test preset：

```powershell
# 日常开发
cmake --preset default
cmake --build --preset default-release
ctest --preset default-release

# 强制 CPU-only
cmake --preset cpu
cmake --build --preset cpu-release
ctest --preset cpu-release

# 强制 CUDA；工具链不可用时配置会明确失败
cmake --preset cuda
cmake --build --preset cuda-release
ctest --preset cuda-release
```

项目使用 Visual Studio 等多配置生成器时，Release/Debug 由 build/test preset 的 `configuration` 决定，不需要设置 `CMAKE_BUILD_TYPE`。

## 交互 Viewer

启动内置场景：

```powershell
.\build\default\bin\viewer.exe --scene builtin --mode raster --width 960 --height 540
```

启动 OBJ 资产场景：

```powershell
.\build\default\bin\viewer.exe `
  --scene asset `
  --asset "Computer Graphics Archive\CornellBox\CornellBox-Original.obj" `
  --mode path `
  --path-backend auto `
  --width 960 `
  --height 540
```

`--asset` 可重复使用，也可直接传入目录；目录中的 OBJ 会被递归导入：

```powershell
.\build\default\bin\viewer.exe `
  --asset "Computer Graphics Archive\hw1\model-a.obj" `
  --asset "Computer Graphics Archive\hw1\model-b.obj" `
  --asset "Computer Graphics Archive\shared-assets" `
  --mode raster
```

打开已保存的场景文档：

```powershell
.\build\default\bin\viewer.exe `
  --scene-file path\to\scene.rscene `
  --mode path
```

如果 `--scene asset` 没有传入 `--asset`，viewer 默认查找：

```text
Computer Graphics Archive\CornellBox\CornellBox-Original.obj
```

Viewer 参数：

```text
--scene builtin|asset
--asset path\to\model-or-directory  (可重复)
--scene-file path\to\scene.rscene
--mode raster|ray|path|opengl
--width integer
--height integer
--frames integer
--path-backend auto|cpu|cuda
--gl-vertex-shader path\to\shader.vert
--gl-fragment-shader path\to\shader.frag
--help
```

### 参数面板

Dear ImGui 面板提供：

- Raster/Ray/Path/OpenGL 模式，以及 Path CPU/CUDA/Auto 后端。
- CUDA Path 的 CUDA–OpenGL interop active/fallback 状态和回退原因。
- OpenGL shader 路径、250 ms 自动热重载、F5 强制重载和编译/链接错误日志。
- Ray 最大深度、CPU 线程数、tile size 和 25%～100% 内部渲染比例。
- Path 累积暂停、继续、清零，以及 spp、FPS 和帧耗时显示。
- Orbit/Free 相机、FOV、轨道距离、移动速度和相机复位。
- 环境光、方向光和点光的 HDR 参数，以及灯光添加和删除。
- 场景 Outliner、层级重设、可见性、锁定、复制、删除和对象独立 TRS 变换。
- 按 MTL 名称选择材质槽，并为当前对象覆盖类型、颜色、粗糙度、IOR、自发光、透明、bump 和双面参数。
- OBJ/目录导入、`.rscene` 打开与保存，以及场景编辑 Undo/Redo。
- `-8～+8 EV` 曝光和 None/Reinhard/ACES tone mapping。
- 75%～200% UI 字体大小和一键恢复默认大小。

曝光、tone mapping 和 UI 字体大小只改变显示，不清空 Path 累积；相机、灯光和渲染尺寸变化会清空累积。UI 不会写入线性 framebuffer，也不会混入 Path 样本。

### 操作

- `Tab`：显示或隐藏参数面板。
- `C`：切换 Orbit/Free 相机。
- Orbit：鼠标左键拖动旋转，滚轮缩放。
- Free：在画面区域按住鼠标右键观察；`W/A/S/D` 移动，`Space` 上升，左右 `Shift` 下降。
- `1` / `2` / `3` / `4`：切换 raster / ray / path / opengl。
- `F5`：强制重新编译当前 OpenGL vertex/fragment shader。
- `R`：清空当前累积。
- `Esc`：退出。

ImGui 捕获鼠标或键盘时，相机和渲染热键不会抢占输入。面板状态仅在当前会话有效，不生成 `imgui.ini`。

### 场景对象与材质覆盖

重复导入同一 OBJ 时，多个对象共享原始网格、材质和纹理资产，但各自保存变换与材质覆盖。复制对象会继承当前外观，之后继续编辑不会影响原对象。

材质覆盖不会写回 OBJ/MTL，也不会修改引用同一资产的其他对象。`Base color / tint` 始终与原 `map_Kd` 相乘；Diffuse、Opacity 和 Bump texture 开关默认保留原贴图。当前版本只覆盖参数，不替换贴图文件。

`.rscene` 当前格式版本为 2，保存对象层级、变换、灯光和非空材质覆盖，并继续兼容版本 1。材质槽越界时保留覆盖数据、渲染时忽略并显示警告。完整操作与格式边界见 [场景对象系统](docs/scene-object-system.md)。

## 离线渲染

```powershell
.\build\default\bin\renderer.exe --mode raster --scene raster_triangle --width 512 --height 512 --output output\raster_triangle.png
.\build\default\bin\renderer.exe --mode ray --scene mirror_spheres --width 512 --height 512 --output output\mirror_spheres.png
.\build\default\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 32 --path-backend auto --output output\cornell_box.png
```

OBJ 示例：

```powershell
.\build\default\bin\renderer.exe --mode raster --scene obj_viewer --obj path\to\model.obj --width 512 --height 512 --output output\obj_viewer.png
```

Renderer 参数：

```text
--mode raster|ray|path
--scene gradient_sphere|raster_triangle|mirror_spheres|cornell_box|obj_viewer
--obj path\to\model.obj
--width integer
--height integer
--spp integer
--max-depth integer
--threads integer
--path-backend auto|cpu|cuda
--output path\to\file.png
--help
```

`auto` 会在已编译 CUDA 且检测到设备时选择 CUDA，否则提示并回退 CPU；`cuda` 不可用时明确报错；`cpu` 强制使用多线程 CPU 后端。离线 raster/ray 不接受显式 `--path-backend`。

## 渲染管线

| 模式 | 执行后端 | 当前能力 |
| --- | --- | --- |
| `raster` | CPU | 近面裁剪、透视正确插值、深度缓冲、平滑/bump 法线、alpha cutout、Lambert、Blinn-Phong |
| `ray` | CPU | BVH、球/三角形求交、alpha-aware 硬阴影、平滑/bump 法线、镜面反射、介质折射 |
| `path` | CPU/CUDA | 渐进累积、点光/方向光直接照明、漫反射半球采样、金属粗糙反射、介质反射/折射、自发光材质、Russian roulette |
| `opengl` | OpenGL 4.5/GLSL | GPU 光栅化、深度缓冲、平滑/bump 法线、alpha cutout、双面/自发光材质、点光/方向光、shader 热重载 |

CUDA 后端处理球、三角形、CPU 构建并上传的扁平 BVH、OBJ/纹理、alpha cutout、bump、现有材质与灯光；每次 kernel launch 为每个像素推进一个 sample。灯光变化只更新灯光数据，不重新上传几何、纹理或 BVH。Viewer 会在 OpenGL context 对应的 CUDA device 上注册 `GL_RGBA32F` texture，直接输出线性 HDR；无兼容设备或注册失败时自动使用原有 framebuffer 下载与上传路径。

## OBJ/MTL 支持

- `vn`：插值顶点法线；缺失或退化时回退到几何法线。
- `Kd`、`map_Kd`：漫反射基色；颜色纹理由 sRGB 解码到线性空间。
- `d`、`Tr`、`map_d`：alpha cutout；常量 opacity 与线性 opacity 纹理相乘。
- `illum 4/6/7/9`：映射为折射/玻璃材质；普通材质的默认 `Tf 1 1 1` 不会单独触发透明。
- `bump`、`map_bump`：线性高度图，通过 UV 导数构建 TBN 并扰动着色法线。
- OBJ 导入材质默认双面着色；几何法线和着色法线分离，次级光线使用几何法线偏移。
- 可选纹理缺失或解码失败时警告并回退到常量材质；损坏的 OBJ 会明确报错。

大型场景资源不提交到仓库。可将 Computer Graphics Archive 等资源包放在项目根目录的 `Computer Graphics Archive/` 中，该目录已被 Git 忽略。

## 构建细节

### CMake 选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `RENDERER_CUDA` | `AUTO` | `AUTO`、`ON` 或 `OFF`；控制 CUDA Path 后端是否参与构建 |
| `CMAKE_CUDA_ARCHITECTURES` | `120` | 未显式指定时使用的 CUDA 目标架构 |
| `RENDERER_NATIVE_ARCH` | `OFF` | MSVC 使用 `/arch:AVX2`，其他编译器使用 `-march=native`；仅作用于项目自有目标 |
| `RENDERER_BUILD_VIEWER` | `ON` | 构建 SDL3/OpenGL viewer；无图形环境可设为 `OFF`，仍构建离线 renderer 和测试 |

本机 CPU 性能测试可按需创建第四个临时构建树；它仍位于统一的 `build/` 目录中：

```powershell
cmake -S . -B build/native -DRENDERER_CUDA=OFF -DRENDERER_NATIVE_ARCH=ON
cmake --build build/native --config Release --parallel 4
ctest --test-dir build/native -C Release --output-on-failure
```

个人机器专用配置可写在不会提交的 `CMakeUserPresets.json` 中。

### 依赖

- Eigen 3.4.0：优先使用本机 CMake package，找不到时通过 FetchContent 获取。
- SDL 3.2.30：优先使用本机 CMake package，找不到时通过 FetchContent 获取。
- Dear ImGui 1.92.8：通过 FetchContent 固定版本，仅链接到 `viewer`。
- GLAD 2.0.8：仓库内固定的 OpenGL 4.5 Core loader，仅链接到 `viewer`。
- stb：读取纹理并写入 PNG。
- tinyobjloader：解析 OBJ/MTL 并三角化。

若要复用手动准备的 SDL3 源码，可在配置时传入绝对路径，不要把机器路径写进 `CMakeLists.txt`：

```powershell
cmake --preset default "-DFETCHCONTENT_SOURCE_DIR_SDL3=D:/path/to/SDL"
```

## 项目结构

```text
src/core/                 图像、数学与基础数据
src/scene/                相机、几何、材质、纹理与场景加载
src/acceleration/         BVH 构建与遍历
src/render/rasterizer/    软件光栅化
src/render/raytracer/     Whitted 光线追踪
src/render/pathtracer/    CPU/CUDA 路径追踪
src/render/opengl/        OpenGL GPU 光栅化
src/render/interactive/   交互式渲染 session
src/interactive/          相机控制、统计和 ImGui 面板
src/platform/opengl/      GLSL 生命周期与 CUDA–OpenGL texture interop
src/platform/sdl/         SDL3 窗口、输入、OpenGL context 与显示合成
shaders/opengl/           可热重载 GLSL shader
tests/                    自动化测试
docs/                     设计、计划、实现说明和验证结果
```

推荐按 `core → scene → acceleration → rasterizer/raytracer/pathtracer → interactive → platform` 的顺序阅读。

## 当前限制与路线图

- Raster 尚无通用六平面齐次裁剪、MSAA 和 mipmap。
- OpenGL 首版只绘制 `Scene::triangles`，不绘制程序化球体；没有阴影、MSAA、mipmap 或离线 PNG 输出。
- CUDA–OpenGL interop 只用于交互式 CUDA Path；离线输出和回退路径仍会下载 framebuffer。
- Path 尚无面积光/环境光重要性采样、MIS、降噪和 PBR 微表面模型；低 spp 噪声仍然明显。
- OBJ/MTL 尚无 tangent-space normal map、alpha blend、metallic/roughness、occlusion 和 emissive 贴图解析。
- Bump 使用逐像素高度有限差分且没有 mipmap，远距离或高频高度图可能走样。
- 材质编辑器尚不支持替换贴图文件、材质预设持久化、docking 或多窗口。

更完整的后续材质计划见 [现代 PBR 材质路线图](docs/pbr-roadmap.md)。

## 设计与验证文档

- [像素生命周期](docs/render-pixel-lifecycle.md)
- [交互 Viewer 设计](docs/specs/2026-07-07-interactive-viewer-design.md)
- [表面与材质正确性设计](docs/specs/2026-07-10-surface-material-correctness-design.md)
- [Eigen float 迁移结果](docs/output/eigen-float-migration-results.md)
- [CUDA Path Tracer 测试与基准](docs/output/cuda-pathtracer-results.md)
- [OpenGL GLSL Shader 合约](docs/glsl-shader-contract.md)
- [场景对象系统与材质覆盖](docs/scene-object-system.md)
