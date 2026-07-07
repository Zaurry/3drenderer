# SDL3 + Eigen 交互窗口设计

## 背景

当前项目是教学用 CPU 3D renderer，现有入口 `renderer.exe` 通过命令行创建场景、调用 `IRenderer::render(...)` 得到完整 `Image`，最后写出 PNG。这个流程适合离线验证，但不适合拖动相机或物体，因为它没有窗口、输入事件、持续帧循环、可复用 framebuffer 或逐帧显示层。

本设计的目标是在不引入 OpenGL、Vulkan、DirectX、OptiX、RT Core 等渲染 API 的前提下，增加一个基本交互窗口。SDL3 只作为平台层：创建窗口、接收鼠标键盘输入、把 CPU 生成的像素显示到屏幕。渲染算法和图形接口仍然由项目自己实现。

## 目标

- 新增 `viewer.exe`，保留现有 `renderer.exe` 离线 PNG 工作流。
- 使用 SDL3 创建窗口、处理输入、显示 CPU framebuffer。
- 使用 Eigen 支撑交互相机、矩阵和变换计算。
- 增加项目自己的软件 framebuffer / swapchain 风格接口，为未来统一图形接口铺路。
- 第一版以 CPU rasterizer 为交互预览后端，实现丝滑拖动基础场景和 OBJ 模型的目标。
- ray tracer 和 path tracer 暂不作为默认实时模式，只保留后续做低分辨率、渐进式预览的扩展空间。

## 非目标

- 不使用 OpenGL、Vulkan、DirectX 或 SDL GPU API 实现渲染管线。
- 不在第一版引入 RT Core、CUDA、OptiX 或硬件光追。
- 不立刻把现有 `Vec3`、`Mat4`、`Ray`、BVH 和相交代码全部迁移到 Eigen。
- 不实现完整现代图形 API 级别的 `Device`、`CommandBuffer`、`PipelineState`、资源同步系统。
- 不在第一版实现材质编辑器、场景层级面板、实时路径追踪降噪或复杂 UI。

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
InteractiveRenderer
    |
    v
Framebuffer / DepthBuffer
```

SDL3 只位于最外层。`ViewerApp` 不直接依赖 SDL 细节，而是通过一个小的显示后端接口收输入、提交 framebuffer。这样未来可以替换为 Win32、终端预览、测试 fake backend，或其他平台层。

渲染器不向 PNG 写文件，也不持有窗口。它只接收场景、相机、设置和目标 framebuffer，然后填充颜色与深度。

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

### InteractiveRenderer

新增面向逐帧渲染的接口，不替换现有离线 `IRenderer`：

```cpp
class InteractiveRenderer {
public:
    virtual ~InteractiveRenderer() = default;
    virtual void render_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        Framebuffer& target) = 0;
};
```

第一版实现 `SoftwareRasterInteractiveRenderer`。它可以复用现有 `RasterizerRenderer` 的投影、edge function、深度测试和 Blinn-Phong 着色逻辑，但输出目标从临时 `Image` 改为传入的 `Framebuffer`。

现有 `RasterizerRenderer::render(...)` 可以继续保留。为了避免重复，后续可以把核心光栅函数抽到共享 helper 中，让离线和交互后端共用同一套像素路径。

### ViewerApp

`ViewerApp` 负责应用级循环：

```text
initialize SDL display
build or load scene
initialize orbit camera

while not quit:
    input = display.poll_input()
    update orbit camera from input
    resize framebuffer if window changed
    renderer.render_frame(scene, camera, settings, framebuffer)
    display.present(framebuffer)
```

第一版先每帧重绘。场景较小时这最简单，也方便观察实时性能。后续可以加入 dirty flag：只有相机、窗口、场景或设置变化时才重绘。

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

## 数据流

每一帧的数据流如下：

1. SDL3 收集窗口、鼠标、滚轮事件。
2. `SdlDisplayBackend` 将 SDL 事件转换为项目自己的 `InputState`。
3. `ViewerApp` 把 `InputState` 交给 `OrbitCameraController`。
4. `OrbitCameraController` 更新 eye、target、up、fov 等相机参数。
5. `ViewerApp` 根据窗口尺寸更新 `RenderSettings` 和 `Framebuffer`。
6. `SoftwareRasterInteractiveRenderer` 将场景光栅化到 `Framebuffer`。
7. `SdlDisplayBackend` 把 `Framebuffer` 转成 RGBA8 presentation buffer 并显示到窗口。

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

## 错误处理

- SDL 初始化失败时，`viewer` 输出错误信息并退出非零状态。
- framebuffer 尺寸必须为正数；窗口最小尺寸可以钳制到 1x1。
- OBJ 加载失败沿用现有错误路径，向用户显示命令行错误。
- 如果 SDL presentation buffer 更新失败，当前帧跳过 present 并输出错误；连续失败时退出。

## 测试策略

自动化测试覆盖不依赖真实窗口的核心逻辑：

- `Framebuffer` resize、clear、set/get、RGBA8 转换。
- `DepthBuffer` resize、clear、set/get。
- Eigen 与 `Vec3` 转换。
- `OrbitCameraController` 的 orbit、zoom、pitch clamp。
- `SoftwareRasterInteractiveRenderer` 输出尺寸正确，并能绘制简单三角形。

SDL3 窗口行为作为手动验证：

- 启动 `viewer.exe` 能看到默认场景。
- 左键拖动能旋转相机。
- 滚轮能缩放。
- 调整窗口大小后图像尺寸同步更新。
- 关闭窗口后程序正常退出。

## 性能预期

第一版性能目标是“教学场景和中小 OBJ 模型可交互”，不是现代游戏引擎级实时渲染。

优先优化顺序：

1. 保持 framebuffer 连续内存，减少每帧分配。
2. 光栅化路径避免不必要的 PNG 或 `Image` 中间转换。
3. 后续再加 tile 并行、背面剔除、视锥裁剪和 SIMD。
4. ray/path 预览采用低分辨率或渐进累积，不阻塞第一版交互窗口。

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
- 窗口 resize 后 framebuffer 和相机 aspect ratio 正确更新。
- 第一版实现不把 OpenGL、Vulkan、DirectX、SDL GPU API、SDL 渲染管线或 RT Core 放进渲染管线。
