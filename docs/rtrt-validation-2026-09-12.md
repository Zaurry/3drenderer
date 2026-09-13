# RTRT 第三轮优化验证（2026-09-12）

后续同日的 OptiX / SER 与 1080p San Miguel 验收见 [复杂场景 60 FPS 验证](rtrt-60fps-validation-2026-09-12.md)。本页保留该后端接入前的测量与当时的后续计划。

本轮实现了可切换的光栅首交点、Owen Sobol 采样、较小的粗尺度滤波核与更稳定的镜面历史。使用用户指定的 **default / Release 构建**。采样和滤波优化默认启用；光栅首交点保留为实验选项，当前三组场景尚未测得包含互操作后的整帧优势。

## 实现

OpenGL 4.5 将首个覆盖三角形写入 RGBA32UI 可见性纹理：实例 ID、资产内三角形 ID、透视正确的重心坐标。深度缓冲负责可见性，CUDA 通过互操作读取纹理，重建原来的交点、纹理、材质和运动向量。相机 jitter、内部尺寸和坐标原点与 CUDA 路径一致。阴影、GI、离屏反射和透射继续调用共享 CUDA 场景与 BVH。

- Alpha Mask 或单面材质拒绝最近候选时，该像素执行完整 CUDA 遍历，寻找后面的有效表面。Alpha Blend 和玻璃仍进入原来的路径采样。
- 原始解析球资产、缺少 GL provider 或互操作不可用时回退 CUDA 首交点。常规导入中的球体使用共享三角网格，可以光栅化。
- 几何缓存随场景来源、资产、几何版本变化刷新，实例变换逐帧更新。绘制保存并恢复调用方 GL 状态。
- 可见性纹理不下载到 CPU。在全部 CUDA 工作入队后归还 GL，避免在首交点与光照之间引入 CPU/驱动等待空隙。

960×540 的可见性与深度占用约 9.89 MiB，另加每三角形 36 字节的位置缓冲；bedroom 总增量约 23.05 MiB。CUDA 帧缓冲保持约 433.08 MiB。共享边用例验证了光栅覆盖完整性：64×64 发光平面不再出现此前 CUDA 三角求交沿对角线偶发漏点。倾斜、镜像、非均匀变换的材质对照仍可能因 CUDA 原有边界覆盖差异出现少量不同像素，不能声称所有边缘逐位一致。

其余默认优化：

- 前 16 维 Sobol 采样逐像素固定扰乱，按帧和每帧样本序号推进，后续随机维度回退 PCG。离线路径仍使用原 PCG；此方案不是时空蓝噪声纹理。
- 默认镜面历史从 8 调整为 16，保留表面验证、亮度反应和玻璃的 4 帧上限。已有会话保存的历史值不被覆盖。
- à-trous 前两轮保留 5×5 核及第一轮历史反馈，后续大步长轮次改用 3×3 核。默认漫反射的空间采样点数从每像素 125 降到 77，镜面从 75 降到 59。没有降低 SPP、反弹上限或内部尺寸。
- 法线权重使用 CUDA 快速幂函数；背景在共享内存同步后跳过空信号统计与空间滤波；最终路径顶点不再计算不会使用的下一条射线。

采样参照 [PBRT Sobol 说明](https://pbr-book.org/4ed/Sampling_and_Reconstruction/Sobol_Samplers)、[方向矩阵](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/sobolmatrices.cpp) 和 [FastOwenScrambler](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/lowdiscrepancy.h)。提取的矩阵保留版权声明，附 [PBRT 许可](../src/render/realtime/PBRT_LICENSE.txt)，没有新增运行时 SDK 或 DLL 依赖。

## 性能

RTX 5080 16 GiB，驱动 616.64，CUDA 13.3，VS 18 / MSVC，`build/default`，sm_120。输出 960×540、内部 100%、1 SPP、8 层路径。各场景预热 48 帧，记录 300 帧，包含 CPU 提交及 CUDA stream 等待，不做整帧下载。导入资产使用加载器自动取景；bedroom 是自动包围盒视角，不能代表房间内部的最坏情况。

“本轮前”为已有的第二轮滤波代码，在本轮相同 default 构建和计时入口重新测量，不直接拿上一天 cuda-native 的数字计算本轮加速比。以下均使用 CUDA 首交点：

| 场景 | 三角形 | 本轮前整帧 P50 | 本轮后整帧 P50 | 本轮前 P95 | 本轮后 P95 |
|---|---:|---:|---:|---:|---:|
| builtin Cornell | 12 | 5.540 ms | **5.101 ms** | 5.886 ms | **5.336 ms** |
| CornellBox-Glossy | 1,110 | 6.104 ms | **5.782 ms** | 6.547 ms | **6.172 ms** |
| bedroom / iscv2 | 383,513 | 5.931 ms | **5.571 ms** | 6.284 ms | **5.892 ms** |

整帧中位耗时分别减少约 **7.9% / 5.3% / 6.1%**。独立四配置计时中，最终原生静止 CUDA P50 为 **5.010 ms**，横移为 **4.983 ms**，50% 内部比例静止为 **1.395 ms**。单独替换粗尺度核的实验中，à-trous P50 从 2.185 降到 1.440 ms，最终默认值下为 1.437 ms。

计时不含 Viewer UI、窗口呈现与垂直同步，不能把倒数当成实际 Viewer FPS。P99 保留在 JSON；300 帧也不足以代表长时间运行的尾部延迟。

最终代码的光栅与 CUDA 首交点对照：

| 场景 | CUDA 首交点 P50 | 光栅后 CUDA G-buffer P50 | GL 绘制 P50 | CUDA 路径整帧 P50 | 光栅路径整帧 P50 |
|---|---:|---:|---:|---:|---:|
| builtin Cornell | 0.117 ms | 0.135 ms | 0.009 ms | **5.101 ms** | 5.401 ms |
| CornellBox-Glossy | 0.162 ms | 0.140 ms | 0.009 ms | **5.782 ms** | 6.091 ms |
| bedroom / iscv2 | 0.367 ms | 0.152 ms | 0.097 ms | **5.571 ms** | 5.728 ms |

bedroom 的 CUDA 首交点部分减少约 59%，但新增的 GL 绘制与互操作抵消收益，因此 `raster_primary` 默认 **false**。不能只看 CUDA 总时间从 5.506 降到 5.270 ms 就认定光栅更快，它漏掉了之前的 GL 与交接时间。UI 的 CUDA 时间、GL 时间分开显示，整帧以 CPU 提交加 GPU 等待时间为准。

数据：[本轮前](../build/rtrt-hybrid-stage1/hybrid-performance.json)、[最终对照](../build/rtrt-final/hybrid/hybrid-performance.json)、[最终四配置分布](../build/rtrt-final/performance.json)。

## 画质与稳定性

Cornell 使用 96×96、1024 SPP 离线参考、48 帧重建，在线性 HDR 中计算。“上一轮”取自 2026-09-11 验证，同一离线参考未更换。

| 指标 | 上一轮 | 本轮 |
|---|---:|---:|
| SVGF + TAA MSE | 0.0142182 | **0.0139413** |
| 50% TAAU MSE | 0.051200 | **0.0510778** |
| Cornell 静态帧差 MSE | 0.00115917 | **0.00115541** |
| 粗糙反射静态 8×8 块均值帧差 MSE | 1.033×10⁻⁶ | **4.266×10⁻⁷** |
| 粗糙反射相机停止后块均值帧差 MSE | 1.128×10⁻⁶ | **4.076×10⁻⁷** |
| 粗糙反射平均误反应 | 0 | **0** |

粗糙反射恒定照明用例的静态低频帧差降低约 **58.7%**，停镜后降低约 **63.9%**。Cornell 最终 MSE 约降低 1.9%，整体帧差只小幅变化，不能宣称所有画面的闪烁都已解决。只打开 Sobol、保持 8 帧镜面历史的中间实验并未改善该反射用例，最终改善来自采样与历史长度的组合。

采样本身另用 64×64、2048 SPP 参考验证，四个种子各累积 64 帧原始辐亮度，不经过降噪、裁剪或 TAA。PCG 平均 MSE **0.00105100**，Sobol **0.000877158**，降低 **16.5%**；所有 Sobol 种子的总能量与参考偏差小于 0.23%。保留像素 jitter 以覆盖像素面积。

新增离屏发光物移动测试：主反射面的几何、材质和相机不变，只移动被反射物。中心区域平均亮度从 **0.974062** 在 8 帧后降至 **0**，全局历史重置次数仍为 1。移动阴影两方向在 4 帧时相对参考区域误差为 **6.691×10⁻⁵ / 5.125×10⁻⁵**。

数据：[画质](../build/rtrt-final/metrics.json)、[低频反射帧差](../build/rtrt-final/glossy-stability.json)、[四种子采样](../build/rtrt-final/sampling-convergence.json)、[反射物运动](../build/rtrt-final/reflected-motion.json)、[最终图像](../build/rtrt-final/cornell-svgf-taa.png)、[参考](../build/rtrt-final/cornell-reference-1024spp.png)。PNG 附带线性 PFM。

## 构建、回归与出图

- `cmake --build --preset default-release` 成功；default CTest **10/10** 通过。显式启用性能项的 `realtime_tests` **13 项通过**，混合管线 4 项行为测试及独立性能项均通过。
- `no-cuda-release` 构建成功，CTest **10 通过、1 跳过**（混合 GPU 测试）。其余二进制内部的 CUDA 用例也按设计跳过，不计为 GPU 通过。
- Compute Sanitizer：混合管线 4 项执行 `memcheck --leak-check full`，**0 errors、0 bytes leaked**；常量/非整块尺寸、粗糙反射与反射物移动 3 项执行 racecheck，**0 errors、0 warnings**。没有重新执行上一轮已记录失败的混合离线 sanitizer 检查。
- Viewer 混合与默认模式各完成 **300/300 帧**，均为 `interop=active`、`downloads=0`、`history_resets=1`、`allocations=30`。捕获分别为 [光栅首交点](../build/rtrt-final/viewer-hybrid.png) 和 [默认模式](../build/rtrt-final/viewer-default.png)。末帧 CUDA 样本不作为 FPS 分布。

日志：`build/rtrt-release-build.log`、`build/rtrt-default-ctest.log`、`build/rtrt-no-cuda-ctest.log`、`build/rtrt-hybrid-memcheck.log`、`build/rtrt-final-racecheck.log`、`build/rtrt-final.log`、`build/rtrt-final-hybrid16.log`、`build/rtrt-viewer-hybrid.log`、`build/rtrt-viewer-default.log`。

## 复现与后续方向

```powershell
cmake --build --preset default-release
ctest --preset default-release
$env:RTRT_VALIDATION_DIR = "$PWD/build/rtrt-check"
$env:RTRT_PERFORMANCE = "1"
.\build\default\bin\realtime_tests.exe
$env:RTRT_HYBRID_PERFORMANCE = "1"
.\build\default\bin\rtrt_hybrid_tests.exe
```

试用光栅首交点，在 `Techniques → RTRT Lighting & Sampling` 勾选 `Raster primary visibility`，或保存以下 JSON 并传给 `--rtrt-config`：

```json
{
  "raster_primary": true,
  "low_discrepancy": true,
  "specular_history": 16,
  "internal_scale": 1.0
}
```

下一步优先在实际问题场景及目标显卡上采集至少 3000 帧、相机轨迹和 HDR 连续帧。然后按耗时决定：合并可见性/呈现资源交接或评估跨 API 外部内存；将剩余软件 BVH 遍历迁移到 RT Core 后端；对高频 HDRI、多光源评估时空蓝噪声和光源样本复用；为复杂反射建立次级表面运动与独立历史。这些方向尚未实现。当前未接入 OptiX、ReSTIR 或神经降噪器，也未复测 RTX 3050 Laptop，不能保证目标场景 60 FPS。
