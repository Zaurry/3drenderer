# CPU 3D Renderer

一个从零实现的 C++20 软件 3D 渲染器，用于学习软件光栅化、光线求交、BVH、Whitted 光线追踪和 Monte Carlo 路径追踪。

光栅化器与 Whitted 光追运行在 CPU 上；Path 模式支持 CPU 与 CUDA Core 后端。核心渲染不依赖 OpenGL、Vulkan、DirectX、Embree、OptiX 或 RT Core。SDL3 的 `SDL_Renderer` 只负责显示线性 framebuffer 并叠加 Dear ImGui，Windows 上可能在内部使用 Direct3D 作为显示后端。

## 快速开始

需要 CMake 3.21+ 和支持 C++20 的编译器。CUDA 是可选依赖。

```powershell
cmake --preset default
cmake --build --preset default-release
ctest --preset default-release

.\build\default\bin\viewer.exe --scene builtin --mode raster
```

首次配置时，CMake 会按需获取缺失的 Eigen3、SDL3 和 Dear ImGui 依赖。

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

如果 `--scene asset` 没有传入 `--asset`，viewer 默认查找：

```text
Computer Graphics Archive\CornellBox\CornellBox-Original.obj
```

Viewer 参数：

```text
--scene builtin|asset
--asset path\to\scene.obj
--mode raster|ray|path
--width integer
--height integer
--frames integer
--path-backend auto|cpu|cuda
--help
```

### 参数面板

Dear ImGui 面板提供：

- Raster/Ray/Path 模式，以及 Path CPU/CUDA/Auto 后端。
- Ray 最大深度、CPU 线程数、tile size 和 25%～100% 内部渲染比例。
- Path 累积暂停、继续、清零，以及 spp、FPS 和帧耗时显示。
- Orbit/Free 相机、FOV、轨道距离、移动速度和相机复位。
- 环境光、方向光和点光的 HDR 参数，以及灯光添加和删除。
- `-8～+8 EV` 曝光和 None/Reinhard/ACES tone mapping。
- 75%～200% UI 字体大小和一键恢复默认大小。

曝光、tone mapping 和 UI 字体大小只改变显示，不清空 Path 累积；相机、灯光和渲染尺寸变化会清空累积。UI 不会写入线性 framebuffer，也不会混入 Path 样本。

### 操作

- `Tab`：显示或隐藏参数面板。
- `C`：切换 Orbit/Free 相机。
- Orbit：鼠标左键拖动旋转，滚轮缩放。
- Free：在画面区域按住鼠标右键观察；`W/A/S/D` 移动，`Space` 上升，左右 `Shift` 下降。
- `1` / `2` / `3`：切换 raster / ray / path。
- `R`：清空当前累积。
- `Esc`：退出。

ImGui 捕获鼠标或键盘时，相机和渲染热键不会抢占输入。面板状态仅在当前会话有效，不生成 `imgui.ini`。

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

CUDA 后端处理球、三角形、CPU 构建并上传的扁平 BVH、OBJ/纹理、alpha cutout、bump、现有材质与灯光；每次 kernel launch 为每个像素推进一个 sample。灯光变化只更新灯光数据，不重新上传几何、纹理或 BVH。

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
src/render/interactive/   交互式渲染 session
src/interactive/          相机控制、统计和 ImGui 面板
src/platform/sdl/         SDL3 窗口、输入与 framebuffer 显示
tests/                    自动化测试
docs/                     设计、计划、实现说明和验证结果
```

推荐按 `core → scene → acceleration → rasterizer/raytracer/pathtracer → interactive → platform` 的顺序阅读。

## 当前限制与路线图

- Raster 尚无通用六平面齐次裁剪、MSAA 和 mipmap。
- Path 尚无面积光/环境光重要性采样、MIS、降噪和 PBR 微表面模型；低 spp 噪声仍然明显。
- OBJ/MTL 尚无 tangent-space normal map、alpha blend、metallic/roughness、occlusion 和 emissive 贴图解析。
- Bump 使用逐像素高度有限差分且没有 mipmap，远距离或高频高度图可能走样。
- 参数面板尚无逐材质编辑、预设持久化、docking 或多窗口。

更完整的后续材质计划见 [现代 PBR 材质路线图](docs/pbr-roadmap.md)。

## 设计与验证文档

- [像素生命周期](docs/render-pixel-lifecycle.md)
- [交互 Viewer 设计](docs/specs/2026-07-07-interactive-viewer-design.md)
- [表面与材质正确性设计](docs/specs/2026-07-10-surface-material-correctness-design.md)
- [Eigen float 迁移结果](docs/output/eigen-float-migration-results.md)
- [CUDA Path Tracer 测试与基准](docs/output/cuda-pathtracer-results.md)
