# CPU 3D Renderer

这是一个从零实现的 C++20 CPU 软件 3D 渲染器，用来学习图形管线、光线求交、BVH、Whitted 光线追踪和基础路径追踪。当前版本不使用 OpenGL、Vulkan、DirectX、Embree、OptiX、CUDA 或 RT Core；SDL3 只用于创建窗口、读取输入和把 CPU framebuffer 贴到窗口 surface。

## 当前能力

- 离线 PNG 渲染：`raster`、`ray`、`path` 三种模式。
- 实时交互窗口：SDL3 窗口 + 自己的 CPU framebuffer，支持鼠标拖动、滚轮缩放和模式切换。
- OBJ/MTL 场景加载：保留顶点法线和 UV，支持漫反射、alpha cutout、bump 与双面材质。
- Path 模式渐进式累积：交互窗口中每帧推进随机种子并累积样本。
- Path 显式直接光：点光与方向光参与 Lambert 直接照明、硬阴影和距离平方衰减。
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
- `stb_image.h`: 读取 `map_Kd`、`map_d`、`bump/map_bump` 使用的 PNG/JPG/PPM 等图片贴图。
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

Mary 和 Sponza 示例：

```powershell
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\mary\Marry.obj" --mode path --width 1920 --height 1080
.\build\bin\viewer.exe --scene asset --asset "Computer Graphics Archive\sponza\sponza.obj" --mode raster --width 1280 --height 720
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

- `raster`: 手写 CPU 光栅化器，包含近面裁剪、透视正确属性插值、深度缓冲、平滑/bump 法线、alpha cutout、Lambert 和 Blinn-Phong 光照。
- `ray`: Whitted 风格光线追踪器，支持球和三角形求交、BVH、alpha-aware 硬阴影、平滑/bump 法线、镜面反射和介质折射。
- `path`: 基础 Monte Carlo 路径追踪器，支持 tile 多线程、像素内抖动采样、渐进累积、点光/方向光直接照明、漫反射半球采样、金属粗糙反射、介质反射/折射和自发光材质。

## OBJ/MTL 表面支持

- `vn`: 按重心坐标插值顶点法线；缺失或退化时回退到几何法线。
- `Kd`、`map_Kd`: 漫反射基色；颜色纹理从 sRGB 解码到线性空间。
- `d`、`Tr`、`map_d`: alpha cutout；常量与线性 opacity 纹理相乘。
- `bump`、`map_bump`: 线性高度图，通过 UV 导数构建的 TBN 扰动着色法线。
- OBJ 导入材质默认双面着色；几何法线和着色法线分离，次级光线使用几何法线偏移。
- 可选纹理缺失或解码失败时输出 warning 并回退到常量材质；结构损坏的 OBJ 仍明确报错退出。

## 现代 PBR 材质路线图

当前材质系统还是教学用的简化模型：`Diffuse`、`Metal`、`Dielectric`、`Emissive` 加少量 OBJ/MTL 参数。要支持现代 PBR 材质，需要逐步补齐下面这些能力：

- 资产格式：优先支持 glTF 2.0 的 metallic-roughness 工作流，再兼容 OBJ/MTL 的 PBR 扩展字段，例如 `Pr`、`Pm`、`map_Pr`、`map_Pm`、`norm`、`map_Ke`。glTF 更适合作为现代 PBR 的主格式，因为它明确规定了贴图通道、颜色空间、alpha 模式和材质参数含义。
- 材质数据结构：把当前 `Material` 扩展成 PBR 参数集，包括 base color、metallic、roughness、normal、occlusion、emissive、alpha mode、alpha cutoff、ior、transmission、clearcoat 等字段。第一阶段可以只做 base color、metallic、roughness、normal、emissive 和 alpha cutout。
- 贴图系统：当前已经区分 sRGB 颜色纹理与线性数据纹理；后续还需支持多 UV set、独立 wrap/filter、UV transform、mipmap 和各向异性过滤。emissive 通常按 sRGB 读取，normal/roughness/metallic/occlusion 必须保持线性数据。
- 几何属性：当前保存 OBJ 顶点 normal 和 UV，并能从三角形 UV 导数建立 bump 所需 TBN。normal map 仍需要更可靠的 tangent 生成，后续可对齐 MikkTSpace 规则并支持 glTF tangent 和多套 UV。
- 着色模型：实现基于微表面的 BRDF，例如 GGX/Trowbridge-Reitz 法线分布、Smith 几何遮蔽、Fresnel-Schlick、能量守恒的 diffuse/specular 混合。raster/ray/path 三条管线都应通过同一个材质评估接口取样和求值。
- 路径追踪采样：当前点光与方向光已经做显式直接采样，但 PBR 还需要 BSDF sample/pdf/evaluate 三件套、面积光与环境光采样、MIS 和 Russian roulette，否则粗糙金属、室内间接光和小面积光源会很难收敛。
- Alpha 与透明：当前三个渲染器均支持 alpha cutout，ray/path 会跳过透明命中并继续追踪。后续再考虑 alpha blend、折射 transmission、薄表面和体积吸收。
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

- 光栅化已支持近面三角形裁剪，但还没有通用六平面齐次裁剪、MSAA 或 mipmap。
- 路径追踪只对点光和方向光做显式直接采样；还没有面积光/环境光重要性采样、MIS、Russian roulette、降噪或 PBR 微表面模型。
- 实时 path 模式已经能渐进累积，但低 spp 下噪声很重，帧率也取决于分辨率、场景复杂度和 CPU。
- OBJ/MTL 已支持 `vn`、`map_Kd`、`d/Tr/map_d` 和 `bump/map_bump`；还没有 tangent-space normal map、alpha blend、metallic/roughness、occlusion 或 emissive 贴图解析。
- Bump 采用逐像素高度有限差分，没有 mipmap，远距离或高频高度图可能出现走样。
- 当前 UI 只有窗口标题栏 FPS 和键盘热键，还没有 ImGui 风格的画面内参数面板。

## Eigen float 架构与构建模式

项目自有渲染器的数学边界位于 `src/core/math/types.h`：

- `Vec2`、`Vec3` 和 `Vec4` 分别是 Eigen `Vector2f`、`Vector3f` 和
  `Vector4f` 的别名；`Mat4` 是 Eigen `Matrix4f` 的别名。
- 渲染器的几何、变换、颜色、采样、光栅化、光线追踪、路径追踪和交互状态均使用
  Eigen 原生运算与 `float` 标量。
- `RENDERER_NATIVE_ARCH=OFF` 是默认的可移植配置。设为 `ON` 时，MSVC 会为四个
  项目自有目标启用 `/arch:AVX2`，其他编译器则启用 `-march=native`。该选项不会
  重新配置 Eigen，也不会影响第三方目标。

可移植 Release 构建与测试：

```powershell
cmake -S . -B build-portable -DRENDERER_NATIVE_ARCH=OFF
cmake --build build-portable --config Release --parallel 2
ctest --test-dir build-portable -C Release --output-on-failure
```

本机优化 Release 构建与测试：

```powershell
cmake -S . -B build-native -DRENDERER_NATIVE_ARCH=ON
cmake --build build-native --config Release --parallel 2
ctest --test-dir build-native -C Release --output-on-failure
```

如果配置时需要使用本地 SDL3 源码缓存，请在命令行中通过
`-DFETCHCONTENT_SOURCE_DIR_SDL3=...` 传入其绝对路径。不要在 `CMakeLists.txt`
中写入特定机器的依赖路径。

```powershell
$sdl3 = (Resolve-Path ".\build\_deps\sdl3-src").Path

cmake -S . -B build-native `
  -DRENDERER_NATIVE_ARCH=ON `
  "-DFETCHCONTENT_SOURCE_DIR_SDL3=$sdl3"

cmake --build build-native --config Release --parallel 2
```

## Eigen 迁移验证结果

本机优化 Release 基准测试采用渲染器报告的秒数；每条命令先预热一次，再测量五次。
测试设备为 AMD Ryzen 7 9800X3D，比较对象是已经提交的 double 基线：

在源码提交 `b7cd8ff` 上进行的最初历史测量中，Mary path 本机优化版本的中位数为
`0.1315030s`，比 double 基线慢 `23.284%`。这个非最终结果超过 5% 的限制，因此
触发了性能分析。下表是提交 `476a0bc` 完成针对性修复后的最终测量结果。

| 场景 / 模式 | Double 中位数（秒） | 本机优化样本（秒） | 本机优化中位数（秒） | 变化 |
| --- | ---: | ---: | ---: | ---: |
| Mary / raster | 0.0675227 | 0.0580001, 0.0570466, 0.0550683, 0.0552859, 0.0570032 | 0.0570032 | -15.579% |
| Mary / path | 0.1066670 | 0.0963786, 0.0959811, 0.0967750, 0.0957497, 0.0951254 | 0.0959811 | -10.018% |
| Sponza / raster | 0.0967525 | 0.0793640, 0.0793388, 0.0781508, 0.0796694, 0.0783349 | 0.0793388 | -17.998% |
| Cornell box / path | 0.2505130 | 0.182832, 0.181961, 0.188832, 0.187693, 0.186658 | 0.1866580 | -25.490% |

本机优化版本按基线尺寸生成了 `output/eigen_float_*.png` 以进行视觉验证：Mary 的
raster/path 图像保持平滑并带有纹理，Sponza 保留了漫反射砖墙纹理和凹凸细节，Cornell
保留了发光面板和彩色墙壁。四张 PNG 均能解码为有效的 8 位通道，受光覆盖率非零，且
未发现洋红色回退像素或新的三角形绕序裂缝。由于输出转换会处理非有限值，仅成功解码
PNG 不能证明渲染器内部的浮点数全部有限。在量化之前，自动化渲染测试会对 raster、ray
和 path 图像执行 `image_colors_are_finite`，并对相应的交互式帧缓冲执行
`framebuffer_colors_are_finite`。低采样数 path 图像中的噪点在基线版本中同样存在。
完整验证证据和命令见 `docs/output/eigen-float-migration-results.md`。

ETW 分析将最初 Mary path 的性能下降定位到 BVH 遍历栈：每条光线都会为这个动态增长
的栈分配内存。提交 `476a0bc` 在保持子节点压栈顺序和遍历行为不变的前提下，将它替换
为局部固定容量数组。这只是一次针对性的内存分配消除，并不代表进行了更广泛的渲染器
优化。

这个 CPU 后端不会使用 RTX 5080，也没有 CUDA 路径或 RT Core 集成。后续切实可行的
优化方向包括降低路径采样方差、改进 BVH 与纹理/凹凸预处理、优化工作线程调度，以及在
性能分析之后开展经过测量的 SIMD 或运行时指令分派工作。这些内容不属于本次迁移范围。
