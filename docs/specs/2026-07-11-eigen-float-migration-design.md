# Eigen Float 全项目数学迁移设计

> **历史文档（已过期）：** 本文描述旧架构。CPU 软件光栅化与 Whitted 光追已于 2026-07-29 删除；当前架构仅支持 OpenGL 与 Path。

## 背景

项目目前同时存在两套数学实现：`src/core/math` 下自研的 `Vec2`、`Vec3`、`Vec4`、`Mat4` 使用 `double`，交互轨道相机局部使用 `Eigen::Vector3d`。渲染核心、场景、BVH、纹理、采样、三个渲染器、交互层和测试中约有 396 个 `double`、587 个 `Vec3` 使用点。

这种状态会增加维护成本，也无法充分利用统一的 Eigen 表达式和 float 数据宽度。本次迁移的目标是删除自研向量/矩阵实现，让整个自有代码统一使用 Eigen `float` 类型和原生运算，同时保留现有渲染行为并建立可重复性能基准。

## 目标

- `src/` 与 `tests/` 中所有自有浮点标量从 `double` 迁移为 `float`。
- 删除自研 `Vec2`、`Vec3`、`Vec4` 和 `Mat4` 实现。
- 使用 Eigen `Vector2f`、`Vector3f`、`Vector4f`、`Matrix3f` 和 `Matrix4f` 作为统一底层数学类型。
- 全项目使用 Eigen 原生向量与矩阵运算，不保留旧 `dot`、`cross`、`normalize`、`length` 等兼容 API。
- `Ray`、`Bounds3`、`Camera`、primitive、材质等领域结构保留，但字段与参数全部迁移为 Eigen/float。
- 调整 float 数值阈值，保持求交、BVH、裁剪、反射/折射和纹理计算稳定。
- 增加可选 native architecture 编译选项，让本机 Release 构建启用更积极的 SIMD 指令。
- 用自动化测试、视觉回归和固定场景基准验证迁移。

## 非目标

- 不修改 `external/` 下的 Eigen、stb 或 tinyobjloader 等第三方源码。
- 不加入 CUDA、OptiX、RT Core 或 GPU 后端。
- 不实现 packet ray、SoA、BVH4/BVH8 或新的并行算法。
- 不修改材质模型、光照算法、场景格式或 UI 功能。
- 不长期维护 double/float 双精度模板系统或旧数学兼容层。
- 不承诺仅靠 Eigen/float 获得数量级性能提升。

## 精度范围

用户选择全项目自有代码统一 float，而不是混合精度。迁移范围包括：

- 几何、矩阵、颜色、材质、纹理、射线、BVH 和深度缓冲。
- 相机参数、采样、随机浮点输出和路径吞吐量。
- SDL 输入增量、轨道相机、帧间隔、FPS 和计时结果。
- CLI 参数转换、中间计算和测试容差。

整数尺寸、索引、计数、随机状态和时间点类型保持原有整数/标准库类型。第三方库若返回 `double` 或其自有 `real_t`，只在项目边界显式转换为 `float`。

## 统一数学类型

新增 `src/core/math/types.h`：

```cpp
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace renderer {

using Scalar = float;
using Vec2 = Eigen::Vector2f;
using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;
using Color = Eigen::Vector3f;

}  // namespace renderer
```

最终删除：

- `src/core/math/vec2.h`
- `src/core/math/vec3.h`
- `src/core/math/vec4.h`
- 自研 `src/core/math/mat4.h`

迁移期间允许这些旧头文件短暂变成转发头，以保持每个阶段可编译；最终提交前必须更新全部 include 并删除转发层。

## Eigen 原生运算

项目代码直接使用 Eigen API：

```cpp
a.dot(b);
a.cross(b);
v.norm();
v.squaredNorm();
v.normalized();
a.cwiseMin(b);
a.cwiseMax(b);
v.allFinite();
```

分量访问改为 `x()`、`y()`、`z()`、`w()` 或下标。逐分量颜色乘法使用 `a.cwiseProduct(b)`，标量乘除继续使用 Eigen 运算符。

Eigen 没有项目当前静态投影接口，因此保留少量带领域约束的矩阵工厂函数：

```cpp
Mat4 make_perspective_matrix(
    float vertical_fov_degrees,
    float aspect,
    float near_z,
    float far_z);

Mat4 make_look_at_matrix(
    const Vec3& eye,
    const Vec3& target,
    const Vec3& up);
```

平移与缩放优先使用 `Eigen::Affine3f`、`Eigen::Translation3f` 和 `Eigen::Scaling`；需要 `Mat4` 时调用 `.matrix()`。测试不再依赖 `Mat4::translation()`、`Mat4::scale()` 等旧静态方法。

## 领域结构

`Ray` 继续保存 origin 与 direction，但类型为 `Vec3`，`at(float t)` 返回 Eigen 表达式求值后的 `Vec3`。

`Bounds3` 使用 `cwiseMin/cwiseMax` 扩展边界，slab 求交全部为 float。空 bounds 使用 `std::numeric_limits<float>::infinity()` 初始化。

`Color` 直接别名为 `Eigen::Vector3f`。颜色 clamp、sRGB 转换和非有限值处理使用 Eigen 分量访问；Framebuffer 与 Image 仍存储 `std::vector<Color>`。

`Camera` 的 eye、forward、right、up 和 viewport 全部迁移为 float/Eigen。轨道相机删除 `to_eigen`、`to_vec3` 转换，直接持有 `Vec3`。

`TriangleVertex`、`HitRecord`、灯光、材质、纹理像素、表面样本和 BVH 节点中的向量全部使用 Eigen 别名。C++17 以上的标准容器对过对齐类型提供支持，项目为 C++20，因此不引入旧式 `Eigen::aligned_allocator` 或手工对齐宏。

## Float 数值策略

现有 double 阈值不能机械替换。新增 `src/core/math/constants.h`，按用途定义：

```cpp
constexpr float kIntersectionEpsilon = 1e-6f;
constexpr float kDirectionEpsilonSquared = 1e-12f;
constexpr float kNearPlane = 1e-4f;
constexpr float kRayOriginScale = 1e-6f;
constexpr float kTestTolerance = 1e-5f;
```

实际常量名称和数值可在失败测试证明需要时细分，但不允许所有算法共用一个万能 epsilon。

重点审计：

- AABB slab 对零方向分量和边界接触的处理。
- Möller–Trumbore determinant、重心边界和极小三角形。
- 几何/着色法线归一化与半球校正。
- 折射判别式和 Fresnel 计算。
- BVH centroid、bounds 与 closest distance。
- Raster edge function、透视权重、近面裁剪和深度比较。
- Ray origin offset 在小场景和大坐标场景中的方向与尺度。
- 纹理 UV wrap、有限差分和 sRGB 转换。

所有浮点字面量使用 `f` 后缀，所有 `numeric_limits`、`chrono::duration` 和显式转换使用 float。

## 随机数与计时

`PcgRandom::next_double()` 改为 `next_float()`，从随机整数生成 `[0, 1)` float。路径追踪、半球采样和像素抖动只使用该接口。

计时器返回 float 秒数。FPS、毫秒、输入 delta、帧间隔和标题格式函数参数全部使用 float。`std::chrono::duration<float>` 用于时长转换；时间点本身继续使用标准库 clock 类型。

## CMake 与 SIMD

新增 CMake 选项：

```cmake
option(RENDERER_NATIVE_ARCH "Enable native CPU instruction tuning" OFF)
```

行为：

- 默认 `OFF`，保持可移植构建。
- MSVC x64 开启时为自有 targets 增加 `/arch:AVX2`。
- GCC/Clang 开启时增加 `-march=native`。
- 不定义 `EIGEN_DONT_VECTORIZE`、`EIGEN_DONT_ALIGN` 或固定 `EIGEN_MAX_ALIGN_BYTES`。
- Release 继续使用工具链正常优化选项。

本次不强制 AVX-512。AVX2 在用户的 9800X3D 上可用，也避免把本轮范围扩展到多套运行时指令分派。

## 迁移顺序

1. 记录迁移前可移植/现有 Release 基准和参考图片。
2. 新增 Eigen 类型入口、常量与矩阵工厂函数，先迁移数学测试。
3. 迁移 Color、Ray、Bounds3、Camera 和交互相机。
4. 迁移 primitive、材质、灯光、纹理、loader 和 Scene。
5. 迁移 BVH 与采样器。
6. 迁移 raster、ray、path 和 interactive sessions。
7. 迁移 SDL/CLI/FPS/计时与全部测试。
8. 删除旧数学头、兼容函数和所有自有 double。
9. 同时验证默认构建与 native architecture 构建。
10. 执行视觉回归与性能对比，更新 README。

每个阶段都使用 TDD，并在阶段结束时保持 Release 构建与完整测试通过。迁移问题必须在最接近来源的阶段解决，不把大量编译错误堆到最后。

## 测试策略

类型与静态检查：

- `static_assert(std::is_same_v<Vec3, Eigen::Vector3f>)`。
- `static_assert(std::is_same_v<Mat4, Eigen::Matrix4f>)`。
- 检查 `Scalar`、材质参数、Ray t、HitRecord t、深度缓冲值和 Timer 结果均为 float。
- `rg` 确认 `src/`、`tests/` 没有 `double`、`Vector2d/3d/4d`、`Matrix*d`。

数学与数值测试：

- Eigen 向量算术、点积、叉积、归一化和逐分量运算。
- 变换组合、投影、look-at 和裁剪空间深度。
- 零向量、平行 up、NaN 和 Infinity 输入。
- AABB 边界接触、平行射线和角点命中。
- 三角形顶点/边界命中、背面、退化面和大坐标。
- BVH 与暴力求交结果在 float 容差内一致。
- 反射、折射、Fresnel、半球采样和随机范围。

渲染回归：

- Raster 近面裁剪、透视插值、深度、alpha 和双面规则。
- Ray 阴影、镜面、介质和透明继续求交。
- Path 直接光、距离衰减、遮挡、发光和渐进累积。
- Mary、Sponza、CornellBox 输出保持正确材质、轮廓、阴影和有效像素覆盖。

构建矩阵：

```powershell
cmake -S . -B build-portable -DRENDERER_NATIVE_ARCH=OFF
cmake --build build-portable --config Release
ctest --test-dir build-portable -C Release --output-on-failure

cmake -S . -B build-native -DRENDERER_NATIVE_ARCH=ON
cmake --build build-native --config Release
ctest --test-dir build-native -C Release --output-on-failure
```

## 性能基准

每项使用 Release native build，预热后运行 5 次并取渲染器报告时间的中位数：

- Mary path：1920x1080、1 spp、4 bounce。
- Mary raster：1920x1080。
- Sponza raster：1280x720。
- CornellBox path：512x512、8 spp、5 bounce。

迁移前先记录当前 double 基线，迁移后使用同一机器、同一资产、同一命令对比。native Eigen/float 结果不能比当前 double 基线慢超过 5%。README 记录实际数据，不以理论推断代替测量。

Float 会改变随机路径和舍入，因此 path 图片不做逐像素完全相等；比较有效像素覆盖、平均亮度、有限值和视觉结果。Raster/ray 使用更严格的图像误差与轮廓检查。

## 风险与处理

- **编译面广：** 通过分阶段迁移和临时转发头控制错误范围。
- **Eigen 分量 API 不兼容：** 全面迁移 `.x` 到 `.x()`，不使用宏模拟旧字段。
- **表达式生命周期：** 领域结构返回具体 `Vec`/`Mat`，不在公共接口返回悬空 Eigen expression。
- **Float 自相交或漏交：** 用边界测试和尺度相关 ray offset 调整，而不是恢复 double。
- **性能未提升：** 保留正确迁移结果和基准证据；后续从数据布局、BVH 和批处理继续优化，不在本轮偷加算法改动。
- **第三方精度不同：** 在 loader/SDL 边界做显式 float 转换，不修改 vendor 代码。

## 成功标准

- `src/` 与 `tests/` 不再包含自有 `double` 或 Eigen double 类型。
- 自研向量/矩阵实现已删除，项目数学统一为 Eigen float。
- 所有项目代码使用 Eigen 原生运算，没有旧数学兼容函数。
- 默认与 native 两种 Release 构建均成功，全部测试通过。
- Mary、Sponza、CornellBox 和 1920x1080 viewer 启动回归通过。
- 性能基准完成并记录，native Eigen/float 中位数不比 double 基线慢超过 5%。
- README 准确说明 Eigen float、native 构建选项、实测性能和剩余优化方向。
