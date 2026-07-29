# CPU 3D 图形渲染器设计文档

> **历史文档（已过期）：** 本文描述旧架构。CPU 软件光栅化与 Whitted 光追已于 2026-07-29 删除；当前架构仅支持 OpenGL 与 Path。

## 目标

从零实现一个高性能、学习友好的 CPU 3D 图形渲染器。第一版要形成一个完整的教学型渲染器，可以通过三种渲染模式输出图片：

- CPU 软件光栅化
- Whitted 风格光线追踪
- Monte Carlo 路径追踪

项目不得使用 OpenGL、Vulkan、DirectX、Embree、OptiX 或其他集成式渲染框架。核心渲染算法必须手写。允许使用轻量工具库处理非核心工作，例如图片写出和 OBJ 模型读取。

## 已确认方向

第一版采用分阶段模块化架构。每种渲染模式都有清晰、独立的实现，但共享同一套场景表达、数学层、相机、材质、灯光、图片输出和加速结构。

架构需要从一开始预留未来演进到统一渲染接口的空间。第一版不引入复杂的 render graph 或 pass 系统，而是先定义一个小而稳定的 renderer 接口，后续可以自然扩展成 `RenderPipeline` / `RenderPass` / `RenderDevice`。

## 技术选择

- 语言：C++20
- 构建系统：CMake
- 图片输出：通过 `stb_image_write` 输出 PNG
- OBJ 读取：通过 `tinyobjloader`
- 渲染后端：仅 CPU
- 外部渲染 API：不使用
- 第一版测试方式：使用简单 C++ 断言测试可执行程序

## 架构

### Core 基础层

基础层只包含底层数据结构和工具，不感知场景对象，也不感知具体 renderer。

职责：

- 向量、矩阵、颜色、射线、包围盒、变换
- 图像缓冲区和 PNG 写出
- 随机数和采样辅助函数
- 计时和简单进度工具
- tiled rendering 所需的线程/任务辅助能力

示例文件：

```text
src/core/math/vec2.h
src/core/math/vec3.h
src/core/math/vec4.h
src/core/math/mat4.h
src/core/math/ray.h
src/core/math/bounds.h
src/core/image.h
src/core/image.cpp
src/core/color.h
src/core/random.h
src/core/timer.h
```

### Scene 场景层

场景层只描述一次世界数据，三种渲染器都消费同一套场景。

职责：

- 相机和相机射线生成
- 三角网格和基础几何体
- 材质和纹理
- 灯光
- 场景组装
- OBJ 加载

示例文件：

```text
src/scene/camera.h
src/scene/camera.cpp
src/scene/mesh.h
src/scene/mesh.cpp
src/scene/primitive.h
src/scene/material.h
src/scene/texture.h
src/scene/light.h
src/scene/scene.h
src/scene/scene.cpp
src/scene/obj_loader.h
src/scene/obj_loader.cpp
```

### Acceleration 加速结构层

加速结构单独成层，因为 BVH 会被光追、路径追踪和未来的拾取/可见性查询共同使用。

第一版使用 BVH，构建策略采用最长轴中点划分。这个方案比 SAH 更适合教学和调试。后续可以在不改变 renderer 查询接口的前提下扩展 SAH 构建器。

示例文件：

```text
src/acceleration/bvh.h
src/acceleration/bvh.cpp
```

### Renderer 渲染层

渲染层提供一个公共入口接口，并实现三个具体 renderer。

初始接口形状：

```cpp
class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual RenderResult render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings) = 0;
};
```

第一版具体 renderer：

- `RasterizerRenderer`
- `RayTracerRenderer`
- `PathTracerRenderer`

示例文件：

```text
src/render/renderer.h
src/render/render_settings.h
src/render/rasterizer/rasterizer_renderer.h
src/render/rasterizer/rasterizer_renderer.cpp
src/render/raytracer/raytracer_renderer.h
src/render/raytracer/raytracer_renderer.cpp
src/render/pathtracer/pathtracer_renderer.h
src/render/pathtracer/pathtracer_renderer.cpp
```

## 计划目录结构

```text
3drenderer/
  CMakeLists.txt
  README.md

  external/
    stb/
      stb_image_write.h
    tinyobjloader/
      tiny_obj_loader.h

  src/
    main.cpp

    core/
      math/
        vec2.h
        vec3.h
        vec4.h
        mat4.h
        ray.h
        bounds.h
      image.h
      image.cpp
      color.h
      random.h
      timer.h

    scene/
      camera.h
      camera.cpp
      mesh.h
      mesh.cpp
      primitive.h
      material.h
      texture.h
      light.h
      scene.h
      scene.cpp
      obj_loader.h
      obj_loader.cpp

    acceleration/
      bvh.h
      bvh.cpp

    render/
      renderer.h
      render_settings.h
      rasterizer/
        rasterizer_renderer.h
        rasterizer_renderer.cpp
      raytracer/
        raytracer_renderer.h
        raytracer_renderer.cpp
      pathtracer/
        pathtracer_renderer.h
        pathtracer_renderer.cpp

    sampling/
      sampler.h
      sampler.cpp

  tests/
    renderer_tests.cpp

  examples/
    cornell_box.cpp
    simple_obj.cpp

  assets/
    models/
    textures/

  output/

  docs/
    output/
      v0.1-results.md
```

## 第一版功能边界

### 公共能力

- C++20 + CMake 项目
- 使用 `stb_image_write` 输出 PNG
- 使用 `tinyobjloader` 加载 OBJ 网格
- 手写数学库
- 场景系统：相机、网格、基础几何体、材质、纹理、灯光、环境光
- 球体和三角形相交
- BVH 构建和遍历
- 命令行选择渲染模式

命令形态示例：

```bash
renderer --mode path --scene cornell_box --width 800 --height 600 --spp 64 --output output/cornell_box.png
```

### 软件光栅化

第一版光栅化包含：

- CPU model/view/projection 变换
- 屏幕空间三角形光栅化
- 深度缓冲
- 重心坐标插值
- 基础 Lambert 和 Blinn-Phong 着色
- 常量颜色和棋盘纹理支持
- 屏幕边界保守处理

完整近裁剪第一版可以简化，只要示例场景避免穿过近裁剪面的几何体即可。代码需要清楚注释这个限制。

### Whitted 光线追踪

第一版光追包含：

- primary ray 生成
- 最近交点查询
- 直接光照
- 硬阴影
- 镜面反射
- 透明折射
- 可配置最大递归深度
- BVH 加速

### 路径追踪

第一版路径追踪包含：

- 多采样抗锯齿
- cosine-weighted hemisphere sampling
- 漫反射、金属、玻璃、发光材质
- 环境颜色
- 可配置每像素采样数
- 可配置最大反弹次数
- BVH 加速
- 多线程 tiled rendering

Russian roulette 留作后续增强。第一版使用显式最大反弹深度，便于学习和复现调试。

## 暂不实现但预留

第一版明确不做：

- GPU 后端
- OpenGL、Vulkan、DirectX、Embree、OptiX 或类似框架
- 完整 PBR microfacet 材质
- 多重重要性采样
- 降噪器
- 动画
- 骨骼网格
- 实例化
- 实时交互窗口
- render graph
- 完整统一渲染管线

## 未来统一渲染接口

第一版避免过度抽象，但必须留下清晰迁移路径：

- 保留 `IRenderer` 作为公共入口。
- 尽量让 `RenderSettings` 与具体模式解耦。
- 场景数据保持 renderer-neutral。
- `main.cpp` 除了构造 renderer 和选择模式，不依赖具体 renderer 细节。
- 尽量把图片输出放在 renderer 内部算法之外。

后续版本可以引入：

- `RenderPipeline`
- `RenderPass`
- `RenderTarget`
- `RenderDevice`
- 共享资源管理
- 按模式组合 pass

## 测试策略

第一版测试系统是一个简单 C++ 可执行程序：`renderer_tests`。

需要测试：

- `Vec3` 加减乘除、点积、叉积、归一化
- 射线与球体相交
- 射线与三角形相交
- AABB 与射线相交
- BVH 最近命中结果与暴力遍历一致
- 重心坐标插值
- 相机中心射线方向
- 采样输出方向是否合法

测试要小而直接。后续可以引入 `doctest` 或 `Catch2`，但第一版不增加不必要依赖。

## 示例场景

第一版内置这些场景：

- `gradient_sphere`：验证相机、射线、颜色和图片输出
- `raster_triangle`：验证软件光栅化、深度缓冲和插值
- `mirror_spheres`：验证 Whitted 反射和硬阴影
- `cornell_box`：验证路径追踪、发光材质、间接光和 BVH 遍历
- `obj_viewer`：验证 OBJ 加载和共享场景渲染

示例命令：

```bash
renderer --mode raster --scene raster_triangle --output output/raster_triangle.png
renderer --mode ray --scene mirror_spheres --output output/mirror_spheres.png
renderer --mode path --scene cornell_box --width 512 --height 512 --spp 64 --output output/cornell_box.png
renderer_tests
```

## 文档和注释策略

文档和注释以中文为主，兼顾常见图形学英文术语，便于对照资料学习。

代码注释：

- 每个核心类说明它解决什么问题。
- 数学函数说明公式、坐标约定和常见坑。
- 相交算法、BVH 构建、路径追踪积分和光栅化插值写详细注释。
- 简单 getter/setter 不写无意义注释。

项目文档：

- `README.md` 说明构建命令、渲染模式、示例场景和推荐阅读顺序。
- 推荐学习顺序：
  1. `Vec3`
  2. `Ray`
  3. `Intersection`
  4. `Camera`
  5. `BVH`
  6. `RayTracerRenderer`
  7. `PathTracerRenderer`
  8. `RasterizerRenderer`

## 版本输出说明文档

第一版必须包含一份面向人的输出说明文档：

```text
docs/output/v0.1-results.md
```

这份文档解释当前版本做了什么，以及实际达到了什么效果。

文档需要包含：

- 版本名称和日期
- 构建环境和构建命令
- 已实现模块
- 已实现渲染功能
- 示例渲染命令
- 预期输出图片列表
- 每张示例图片说明
- 性能记录：分辨率、每像素采样数、线程数、渲染耗时
- 已知限制
- 下一步路线图

这份文档必须在实现和验证之后编写，因为它描述的是实际结果，不是计划结果。

## 成功标准

第一版满足以下条件即认为成功：

- 项目可以通过 CMake 构建。
- `renderer_tests` 通过。
- renderer 可以分别用光栅化、Whitted 光追和路径追踪模式输出 PNG 图片。
- `cornell_box` 在路径追踪模式下能看到间接光。
- `mirror_spheres` 能看到硬阴影和反射。
- `raster_triangle` 能看到通过深度测试绘制的三角形。
- 核心算法注释足够详细，学习者可以跟读。
- `README.md` 解释如何构建、运行和学习代码。
- `docs/output/v0.1-results.md` 总结本版完成内容和实际效果。

## 实现原则

第一版优先保证清晰，而不是炫技。性能很重要，但不能以牺牲学习可读性为代价。路径追踪第一版使用 tiled 多线程和 BVH 加速，这能提供实际性能收益，同时不会把核心算法藏到平台相关 API 或黑盒库后面。
