# CPU 3D Renderer

这是一个从零实现的 C++20 CPU 软件 3D 渲染器，用来学习图形管线、光线求交、BVH、Whitted 光线追踪和基础路径追踪。当前版本不使用 OpenGL、Vulkan、DirectX、Embree、OptiX、CUDA 或 RT Core；SDL3 只用于创建窗口、读取输入和把 CPU framebuffer 贴到窗口 surface。

## 当前能力

- 离线 PNG 渲染：`raster`、`ray`、`path` 三种模式。
- 实时交互窗口：SDL3 窗口 + 自己的 CPU framebuffer，支持鼠标拖动、滚轮缩放和模式切换。
- OBJ/MTL 场景加载：可加载 Computer Graphics Archive 的 CornellBox OBJ 场景。
- Path 模式渐进式累积：交互窗口中每帧推进随机种子并累积样本。
- 标题栏性能显示：viewer 标题栏显示 FPS、单帧毫秒数，path 模式额外显示当前累计 spp。

## 构建

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

当前工程使用 CMake + C++20。Windows 下会生成 Visual Studio/MSBuild 项目。

依赖获取方式：

- Eigen3: 优先使用本机 CMake package，找不到时通过 FetchContent 下载 Eigen 3.4.0。
- SDL3: 优先使用本机 CMake package，找不到时通过 FetchContent 下载 SDL 3.2.30。
- `stb_image_write.h`: 写 PNG。
- `stb_image.h`: 读取 `map_Kd` 使用的 PNG/JPG/PPM 等图片贴图。
- `tiny_obj_loader.h`: 读取 OBJ/MTL 文本并三角化面。

如果 SDL3 已经手动解压到 `build/_deps/sdl3-src`，CMake 会优先使用这个本地目录。

## 离线渲染

```powershell
.\build\bin\renderer.exe --mode raster --scene raster_triangle --width 512 --height 512 --output output\raster_triangle.png
.\build\bin\renderer.exe --mode ray --scene mirror_spheres --width 512 --height 512 --output output\mirror_spheres.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 32 --max-depth 5 --output output\cornell_box.png
```

OBJ 离线查看器示例：

```powershell
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj path\to\model.obj --width 512 --height 512 --output output\obj_viewer.png
```

CLI 参数：

```text
--mode raster|ray|path
--scene gradient_sphere|raster_triangle|mirror_spheres|cornell_box|obj_viewer
--obj path\to\model.obj
--width integer
--height integer
--spp integer
--max-depth integer
--threads integer
--output path\to\file.png
--help
```

## 交互窗口

启动内置 Cornell Box：

```powershell
.\build\bin\viewer.exe --scene builtin --mode raster --width 960 --height 540
```

启动 Computer Graphics Archive 的 CornellBox：

```powershell
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\CornellBox\CornellBox-Original.obj" --mode path --width 960 --height 540
```

如果 `--scene asset` 没有指定 `--asset`，viewer 会默认查找：

```text
Computer Graphics Archive\CornellBox\CornellBox-Original.obj
```

交互控制：

- 鼠标左键拖动：轨道相机旋转。
- 鼠标滚轮：缩放。
- `1`: 切换到 raster。
- `2`: 切换到 ray。
- `3`: 切换到 path。
- `R`: 重置当前渲染累积。
- `Esc`: 退出。

标题栏示例：

```text
CPU 3D Renderer Viewer - path - 12.3 FPS - 81.2 ms - 24 spp
```

## 场景资源

Computer Graphics Archive 这类资源包不提交到仓库。手动下载后放在项目根目录，例如：

```text
D:\Github\3drenderer\Computer Graphics Archive\CornellBox\CornellBox-Original.obj
```

仓库会忽略 `Computer Graphics Archive/`，避免误提交大型素材目录。

## 渲染模式

- `raster`: 手写 CPU 光栅化器，包含世界到屏幕投影、edge function 重心坐标、深度缓冲、Lambert 和 Blinn-Phong 光照。
- `ray`: Whitted 风格光线追踪器，支持球和三角形求交、BVH、硬阴影、镜面反射、介质折射。
- `path`: 基础 Monte Carlo 路径追踪器，支持 tile 多线程、像素内抖动采样、漫反射半球采样、金属粗糙反射、介质反射/折射和自发光材质。

## 现代 PBR 材质路线图

当前材质系统还是教学用的简化模型：`Diffuse`、`Metal`、`Dielectric`、`Emissive` 加少量 OBJ/MTL 参数。要支持现代 PBR 材质，需要逐步补齐下面这些能力：

- 资产格式：优先支持 glTF 2.0 的 metallic-roughness 工作流，再兼容 OBJ/MTL 的 PBR 扩展字段，例如 `Pr`、`Pm`、`map_Pr`、`map_Pm`、`norm`、`map_Ke`。glTF 更适合作为现代 PBR 的主格式，因为它明确规定了贴图通道、颜色空间、alpha 模式和材质参数含义。
- 材质数据结构：把当前 `Material` 扩展成 PBR 参数集，包括 base color、metallic、roughness、normal、occlusion、emissive、alpha mode、alpha cutoff、ior、transmission、clearcoat 等字段。第一阶段可以只做 base color、metallic、roughness、normal、emissive 和 alpha cutout。
- 贴图系统：支持多贴图槽、UV set、贴图 wrap/filter、UV transform、mipmap，以及颜色空间区分。base color/emissive 通常按 sRGB 读取，normal/roughness/metallic/occlusion 必须按线性数据读取，不能做 sRGB gamma 转换。
- 几何属性：保存 OBJ/glTF 顶点 normal、tangent、bitangent 和多套 UV。normal map 需要切线空间 TBN；如果资产没有 tangent，需要生成 tangent，后续可对齐 MikkTSpace 规则。
- 着色模型：实现基于微表面的 BRDF，例如 GGX/Trowbridge-Reitz 法线分布、Smith 几何遮蔽、Fresnel-Schlick、能量守恒的 diffuse/specular 混合。raster/ray/path 三条管线都应通过同一个材质评估接口取样和求值。
- 路径追踪采样：PBR 不能只“算颜色”，还需要 BSDF sample/pdf/evaluate 三件套。后续应加入 next event estimation、MIS、Russian roulette、HDR 环境光和面积光采样，否则粗糙金属、室内间接光和小光源会很难收敛。
- Alpha 与透明：先支持 alpha cutout，用于树叶、栏杆、镂空贴图；再考虑 alpha blend、折射 transmission、薄表面和体积吸收。path tracer 中 alpha cutout 命中透明区域时需要继续追射线。
- 色彩与输出：补齐线性工作流、HDR framebuffer、tone mapping、曝光、白平衡和 sRGB 输出转换。PBR 结果是否可信，很大一部分取决于颜色空间是否正确。
- 测试与参考：加入小型 glTF/OBJ PBR fixture，分别覆盖 base color、metallic/roughness、normal、emissive、alpha cutout 和纹理颜色空间；再用 Khronos glTF sample models 或自制参考图做视觉回归。

## 学习顺序

1. `src/core/math`: 向量、矩阵、射线、AABB。
2. `src/scene`: 材质、光源、球、三角形、相机、内置场景和 OBJ/MTL 场景加载。
3. `src/acceleration/bvh.*`: BVH 构建与遍历。
4. `src/render/rasterizer`: 软件光栅化管线。
5. `src/render/raytracer`: Whitted 光线追踪。
6. `src/render/pathtracer`: Monte Carlo 路径追踪。
7. `src/render/interactive`: raster/ray/path 的交互式渲染 session。
8. `src/interactive`: 轨道相机和 FPS 统计等交互辅助逻辑。
9. `src/platform/sdl`: SDL3 窗口、输入和 CPU framebuffer 显示。
10. `src/main.cpp` 与 `src/viewer_main.cpp`: 离线 CLI 和实时 viewer 入口。

## 当前限制

- 光栅化 v0.1 还没有完整三角形裁剪。
- 路径追踪没有 next event estimation、MIS、Russian roulette、降噪或 PBR 微表面模型。
- 实时 path 模式已经能渐进累积，但低 spp 下噪声很重，帧率也取决于分辨率、场景复杂度和 CPU。
- OBJ/MTL 已支持 `map_Kd` diffuse 贴图；还没有 normal/bump/alpha、metallic/roughness、occlusion 或 emissive 贴图解析。
- 当前 UI 只有窗口标题栏 FPS 和键盘热键，还没有 ImGui 风格的画面内参数面板。
