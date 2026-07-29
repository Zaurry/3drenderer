# SDL3 + Eigen 交互窗口设计

> **历史文档（已过期）：** 本文描述旧架构。CPU 软件光栅化与 Whitted 光追已于 2026-07-29 删除；当前架构仅支持 OpenGL 与 Path。

## 背景

当前项目是教学用 CPU 3D renderer，现有入口 `renderer.exe` 通过命令行创建场景、调用 `IRenderer::render(...)` 得到完整 `Image`，最后写出 PNG。这个流程适合离线验证，但不适合拖动相机或物体，因为它没有窗口、输入事件、持续帧循环、可复用 framebuffer 或逐帧显示层。

本设计的目标是在不引入 OpenGL、Vulkan、DirectX、OptiX、RT Core 等渲染 API 的前提下，增加一个基本交互窗口。SDL3 只作为平台层：创建窗口、接收鼠标键盘输入、把 CPU 生成的像素显示到屏幕。渲染算法和图形接口仍然由项目自己实现。

## 目标

- 新增 `viewer.exe`，保留现有 `renderer.exe` 离线 PNG 工作流。
- 使用 SDL3 创建窗口、处理输入、显示 CPU framebuffer。
- 使用 Eigen 支撑交互相机、矩阵和变换计算。
- 增加项目自己的软件 framebuffer / swapchain 风格接口，为未来统一图形接口铺路。
- 第一版支持 CPU rasterizer、ray tracer 和 path tracer 三种 viewer 模式。raster 追求低延迟拖动；ray/path 可以帧数低、噪声高，但必须能在窗口里持续出图。
- path tracer 使用渐进式累积，相机、窗口、场景或模式变化时重置累积，让用户能看到从很粗糙到逐渐可用的过程。
- 能正确加载 Computer Graphics Archive 的 CornellBox OBJ/MTL 场景，至少覆盖 `CornellBox-Original.obj`、`CornellBox-Mirror.obj`、`CornellBox-Sphere.obj`、`CornellBox-Water.obj` 等同目录资产。

## 非目标

- 不使用 OpenGL、Vulkan、DirectX 或 SDL GPU API 实现渲染管线。
- 不在第一版引入 RT Core、CUDA、OptiX 或硬件光追。
- 不立刻把现有 `Vec3`、`Mat4`、`Ray`、BVH 和相交代码全部迁移到 Eigen。
- 不实现完整现代图形 API 级别的 `Device`、`CommandBuffer`、`PipelineState`、资源同步系统。
- 不在第一版实现材质编辑器、场景层级面板、实时路径追踪降噪或复杂 UI。
- 不承诺 ray/path 模式达到流畅帧率。第一版的标准是“能交互触发、能持续显示、能逐步改善”，后续再优化。

## 架构概览

第一版采用“薄平台层 + 自有软件渲染目标 + 交互应用层”的结构：

```text
SDL3 window/input
    |
    v
SdlDisplayBackend
    |
    v
ViewerApp ---- OrbitCameraController
    |
    v
InteractiveRenderSession
    |
    v
Framebuffer / DepthBuffer
```

SDL3 只位于最外层。`ViewerApp` 不直接依赖 SDL 细节，而是通过一个小的显示后端接口收输入、提交 framebuffer。这样未来可以替换为 Win32、终端预览、测试 fake backend，或其他平台层。

渲染器不向 PNG 写文件，也不持有窗口。它只接收场景、相机、设置和目标 framebuffer，然后填充颜色与深度。raster/ray 可以每帧独立生成图像，path 需要保留累积状态，所以交互接口以“session”为单位。

## 组件设计

### Framebuffer

新增 `Framebuffer` 表示 CPU 端颜色缓冲：

```cpp
class Framebuffer {
public:
    Framebuffer(int width, int height);

    int width() const;
    int height() const;
    void resize(int width, int height);
    void clear(const Color& color);
    void set_pixel(int x, int y, const Color& color);
    const Color& pixel(int x, int y) const;

    std::vector<std::uint8_t> to_rgba8() const;
};
```

第一版可以每帧生成 RGBA8 presentation buffer，并通过 SDL3 的窗口 surface / software blit 风格路径显示。后续如果拷贝成本明显，可以缓存 RGBA8 buffer，只更新 dirty frame。

### DepthBuffer

光栅化交互后端需要独立深度缓冲：

```cpp
class DepthBuffer {
public:
    DepthBuffer(int width, int height);

    void resize(int width, int height);
    void clear(double depth);
    double get(int x, int y) const;
    void set(int x, int y, double depth);
};
```

深度缓冲不暴露给平台层，只属于软件渲染管线。

### DisplayBackend

平台显示接口隔离 SDL3：

```cpp
struct InputState {
    bool quit_requested = false;
    bool window_resized = false;
    int window_width = 0;
    int window_height = 0;

    bool left_mouse_down = false;
    double mouse_delta_x = 0.0;
    double mouse_delta_y = 0.0;
    double wheel_delta = 0.0;

    bool select_raster = false;
    bool select_ray = false;
    bool select_path = false;
    bool reset_render = false;
};

class DisplayBackend {
public:
    virtual ~DisplayBackend() = default;
    virtual bool initialize(int width, int height, const char* title) = 0;
    virtual InputState poll_input() = 0;
    virtual void present(const Framebuffer& framebuffer) = 0;
};
```

`SdlDisplayBackend` 是第一版唯一实现。它负责 SDL 初始化、窗口销毁、事件轮询，以及把 RGBA8 像素作为软件生成的最终图像显示到窗口。SDL3 不参与顶点处理、光栅化、shading、深度测试或任何渲染算法。

### InteractiveRenderSession

新增面向交互渲染的 session 接口，不替换现有离线 `IRenderer`。session 可以保存跨帧状态，例如 path tracing 的累积颜色、当前 sample index、低分辨率预览尺寸或 tile 游标。

```cpp
enum class InteractiveRenderMode {
    Raster,
    Ray,
    Path
};

struct InteractiveFrameState {
    bool camera_changed = false;
    bool scene_changed = false;
    bool framebuffer_resized = false;
    double delta_seconds = 0.0;
};

class InteractiveRenderSession {
public:
    virtual ~InteractiveRenderSession() = default;
    virtual void reset(const Scene& scene, const RenderSettings& settings) = 0;
    virtual void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) = 0;
};
```

第一版实现三个 session：

- `RasterInteractiveSession`：复用现有 `RasterizerRenderer` 的投影、edge function、深度测试和 Blinn-Phong 着色逻辑，输出目标从临时 `Image` 改为传入的 `Framebuffer`。
- `RayInteractiveSession`：复用 Whitted ray tracer 的 primary ray、BVH、阴影、反射和折射逻辑。第一版可以每次相机变化后整帧重绘；如果太慢，可以增加低分辨率 scale 再放大显示。
- `PathInteractiveSession`：复用 path tracer 的采样和递归逻辑，但改为 progressive accumulation。每次相机、窗口、场景或渲染设置变化时清空累积；不变时每帧增加一个或少量 samples per pixel，并把当前平均结果显示到 `Framebuffer`。

现有 `RasterizerRenderer::render(...)`、`RayTracerRenderer::render(...)` 和 `PathTracerRenderer::render(...)` 可以继续保留。为了避免重复，后续可以把三种离线 renderer 的核心像素函数抽到共享 helper 中，让离线和交互后端共用同一套像素路径。

### ViewerApp

`ViewerApp` 负责应用级循环：

```text
initialize SDL display
build or load scene
initialize orbit camera

while not quit:
    input = display.poll_input()
    update render mode from hotkeys
    update orbit camera from input
    resize framebuffer if window changed
    session.render_next_frame(scene, camera, settings, frame_state, framebuffer)
    display.present(framebuffer)
```

第一版先每帧调用当前 session。raster 和 ray 模式通常直接重绘；path 模式在画面不变时继续累积，画面变化时重置。Viewer 默认快捷键：

- `1`：raster 模式。
- `2`：ray 模式。
- `3`：path 模式。
- `R`：手动重置当前 session 的累积或缓存。

即使 ray/path 模式帧率较低，也要在窗口中显示当前正在计算出的结果。后续可以加入 dirty flag、tile budget、低分辨率预览和后台 worker，但第一版优先把模式跑通。

### OrbitCameraController

交互相机使用 Eigen 实现内部数学：

- 左键拖动：绕目标点 orbit。
- 滚轮：改变相机到目标点距离。
- 视角限制：pitch 限制在接近正负 90 度以内，避免翻转。
- 输出：转换成现有 `renderer::Camera`，供当前渲染器使用。

项目已有 `Vec3` 和 `Camera` 不在第一版删除。新增转换函数：

```cpp
Eigen::Vector3d to_eigen(const Vec3& value);
Vec3 to_vec3(const Eigen::Vector3d& value);
```

Eigen 的第一阶段使用范围只限交互控制和变换计算。核心相交、BVH 和 shading 仍保持现状，降低迁移风险。

### SceneAssetLoader

现有 `load_obj_mesh(path, material_id)` 只返回三角形网格，并把所有面绑定到同一个材质。为了正确加载 Computer Graphics Archive 的 CornellBox，需要新增更完整的 OBJ/MTL 场景加载入口：

```cpp
struct LoadedScene {
    Scene scene;
    Camera camera;
    Bounds3 bounds;
};

LoadedScene load_scene_asset(const std::string& path, int width, int height);
```

第一版重点支持本地目录 `Computer Graphics Archive/CornellBox` 中的 OBJ/MTL 文件。tinyobjloader 已经能解析 `mtllib`、`usemtl`、四边形三角化和材质表，因此实现应复用 tinyobjloader，而不是手写 OBJ/MTL parser。

材质映射规则：

- `Kd` -> `Material::base_color`。
- `Ke` 非零 -> `MaterialType::Emissive`，`emission = Ke`。
- `illum 5` 或较高 `Ks` 且非透明 -> `MaterialType::Metal`，`base_color` 优先使用 `Ks` 或 `Kd`，`roughness` 从 `Ns` 粗略换算并 clamp。
- `illum 7`、`Ni` 明确存在且 `Tf`/透明参数表示透射 -> `MaterialType::Dielectric`，`ior = Ni`。
- 其他材质 -> `MaterialType::Diffuse`。

面数据必须使用 tinyobj 的 per-face material id 生成三角形，保证 CornellBox 的红墙、绿墙、白墙、灯、镜面球、水体等材质不会被压成单一默认材质。

加载器还要为外部场景生成合理默认相机：

- 用 `Bounds3` 计算场景中心和半径。
- CornellBox 默认相机放在盒子正面外侧，朝向 bounds center。
- aspect ratio 使用窗口尺寸。
- 如果 OBJ 没有显式光源，但 MTL 中存在 `Ke` 材质，则 emissive 三角形作为 path/ray 可见光源；raster 模式同时可以增加一个临时弱环境或方向光，保证非发光区域在预览中可见。

## 数据流

每一帧的数据流如下：

1. SDL3 收集窗口、鼠标、滚轮事件。
2. `SdlDisplayBackend` 将 SDL 事件转换为项目自己的 `InputState`。
3. `ViewerApp` 根据快捷键更新 `InteractiveRenderMode`，必要时重置 session。
4. `ViewerApp` 把 `InputState` 交给 `OrbitCameraController`。
5. `OrbitCameraController` 更新 eye、target、up、fov 等相机参数。
6. `ViewerApp` 根据窗口尺寸更新 `RenderSettings` 和 `Framebuffer`。
7. 当前 `InteractiveRenderSession` 将下一帧或下一批 samples 写入 `Framebuffer`。
8. `SdlDisplayBackend` 把 `Framebuffer` 转成 RGBA8 presentation buffer 并显示到窗口。

## 依赖管理

CMake 增加可选依赖：

- Eigen：头文件库，可以通过系统安装、vcpkg 或 FetchContent 接入。
- SDL3：窗口和输入库，可以通过 vcpkg 或用户本机安装接入。

建议第一版优先支持 `find_package(SDL3 CONFIG REQUIRED)` 和 `find_package(Eigen3 CONFIG REQUIRED)`。这与 Windows 上 vcpkg 工作流匹配，也避免把第三方源码直接塞进项目。文档中补充构建示例。

如果用户环境没有 vcpkg，后续可以增加 FetchContent fallback，但不作为第一版必要条件。

## CMake 输出

保留：

- `renderer_core`
- `renderer`
- `renderer_tests`

新增：

- `viewer`

`viewer` 链接 `renderer_core`、`SDL3::SDL3`、`Eigen3::Eigen`。SDL3 相关代码只进入 `viewer` 或独立 `platform_sdl` 目标，不污染离线 renderer。

## Viewer 命令行

第一版 `viewer.exe` 提供最小命令行：

```text
viewer --scene builtin|asset --asset path\to\scene.obj --mode raster|ray|path --width 1280 --height 720
```

- `--scene builtin` 使用项目内置教学场景，默认可选 Cornell box 或 mirror spheres。
- `--scene asset --asset ...` 使用 `SceneAssetLoader` 加载外部 OBJ/MTL。
- `--mode` 选择启动时的交互模式；运行时仍可用 `1/2/3` 切换。
- 如果不给 `--asset`，viewer 默认优先尝试 `Computer Graphics Archive/CornellBox/CornellBox-Original.obj`；不存在时退回内置场景。

## 错误处理

- SDL 初始化失败时，`viewer` 输出错误信息并退出非零状态。
- framebuffer 尺寸必须为正数；窗口最小尺寸可以钳制到 1x1。
- OBJ/MTL 加载失败沿用现有错误路径，向用户显示命令行错误，并明确缺失的是 OBJ、MTL 还是材质引用。
- OBJ 中某个 face 的 material id 无效时，使用醒目的 fallback magenta 材质，同时输出 warning，方便调试资产问题。
- 如果 SDL presentation buffer 更新失败，当前帧跳过 present 并输出错误；连续失败时退出。

## 测试策略

自动化测试覆盖不依赖真实窗口的核心逻辑：

- `Framebuffer` resize、clear、set/get、RGBA8 转换。
- `DepthBuffer` resize、clear、set/get。
- Eigen 与 `Vec3` 转换。
- `OrbitCameraController` 的 orbit、zoom、pitch clamp。
- `RasterInteractiveSession` 输出尺寸正确，并能绘制简单三角形。
- `RayInteractiveSession` 能在小测试场景中生成非背景像素。
- `PathInteractiveSession` 在相机不变时 sample count 增加，相机变化时累积重置。
- `SceneAssetLoader` 能读取带 `mtllib`/`usemtl` 的 OBJ，并按 face material id 绑定材质。
- `SceneAssetLoader` 能把 CornellBox MTL 中的 `Kd`、`Ke`、`Ks`、`Ns`、`Ni` 和 `illum` 映射到项目材质。

SDL3 窗口行为作为手动验证：

- 启动 `viewer.exe` 能看到默认场景。
- 左键拖动能旋转相机。
- 滚轮能缩放。
- `1/2/3` 能切换 raster、ray、path 模式。
- path 模式静止时画面逐渐累积，拖动相机后重新从低 sample 图像开始。
- `viewer.exe --scene asset --asset "Computer Graphics Archive\CornellBox\CornellBox-Original.obj"` 能显示 CornellBox，且红墙、绿墙、白墙和灯材质可辨认。
- 调整窗口大小后图像尺寸同步更新。
- 关闭窗口后程序正常退出。

## 性能预期

第一版性能目标是“教学场景和 CornellBox 级别 OBJ/MTL 资产可在窗口中交互触发并持续出图”，不是现代游戏引擎级实时渲染。raster 应尽量接近流畅拖动；ray/path 可以明显低帧率，但用户要能看到当前模式正在工作。

优先优化顺序：

1. 保持 framebuffer 连续内存，减少每帧分配。
2. 三种交互 session 都避免不必要的 PNG 或 `Image` 中间转换。
3. path 模式优先做 progressive accumulation，每帧至少推进一个可见步骤。
4. ray/path 如果整帧太慢，先支持低分辨率渲染再放大显示。
5. 后续再加 tile 并行、背面剔除、视锥裁剪、SIMD 和后台 worker。

## 未来扩展

这个设计给后续统一图形接口留出空间，但第一版不提前实现完整复杂抽象。可自然演进为：

- `RenderTarget`：同时管理颜色、深度和尺寸。
- `SoftwareSwapchain`：管理 front/back framebuffer。
- `RenderDevice`：抽象 CPU backend、未来 CUDA backend 或其他后端。
- `RenderPass`：拆分清屏、几何 pass、显示 pass。
- `SceneView`：承载 camera、viewport、render mode 和调试选项。

扩展时应保持 SDL3 在平台层，渲染核心不依赖 SDL。

## 成功标准

- `renderer.exe` 和现有测试仍然正常。
- 新增 `viewer.exe` 可以打开 SDL3 窗口。
- 窗口显示的软件渲染结果来自项目自己的 framebuffer。
- 左键拖动可以 orbit 场景，滚轮可以缩放。
- viewer 中 `1/2/3` 可以切换 raster、ray、path 三种交互模式。
- raster 模式能低延迟重绘；ray 模式能持续显示 Whitted ray tracing 结果；path 模式能渐进累积并在相机变化时重置。
- `SceneAssetLoader` 能正确读取 Computer Graphics Archive CornellBox OBJ/MTL，保留 per-face 材质、emissive 灯、镜面材质和介质材质的基础映射。
- viewer 可以直接加载并显示 `Computer Graphics Archive\CornellBox\CornellBox-Original.obj`，并能用于其它同目录 CornellBox 变体。
- 窗口 resize 后 framebuffer 和相机 aspect ratio 正确更新。
- 第一版实现不把 OpenGL、Vulkan、DirectX、SDL GPU API、SDL 渲染管线或 RT Core 放进渲染管线。
