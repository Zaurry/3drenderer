# RTRT 首版验证记录

日期：2026-09-11。实现和参数见 [RTRT 使用说明](rtrt-renderer.md)。

后续光栅首交点、采样与滤波优化及 default 构建结果见 [2026-09-12 验证记录](rtrt-validation-2026-09-12.md)。本文件保留当日测量作为基线。

RTX 5080 / 1080p / San Miguel 的 OptiX、SER、原生材质重建及完整帧验收见 [复杂场景 60 FPS 验证](rtrt-60fps-validation-2026-09-12.md)。

镜面、玻璃 firefly 的后续修复、画质对照和停止点见 [2026-09-13 firefly 记录](rtrt-firefly-validation-2026-09-13.md)。该轮长测 P95 17.818 ms，未通过稳定 60 FPS 门槛；最新改动仍有构建和运行验收未完成。

本页前半部分保留 RTX 3050 Laptop 首版基线；针对帧率、闪烁和 boiling water 的本轮修复与 RTX 5080 同机前后对照见文末“第二轮优化”。两台设备的耗时不能直接比较。

## 构建与回归

- CUDA：MSVC / Ninja / Release，CUDA 13.3，`RENDERER_CUDA=ON`，`CMAKE_CUDA_ARCHITECTURES=86`；产物为 `build/cuda-ninja/bin/viewer.exe` 和离线 `renderer.exe`。
- 无 CUDA：`build/default`，`RENDERER_CUDA=OFF`，Release Viewer 可运行。
- 两套 CTest 均为 **8/8 通过**；无 CUDA 构建中的 GPU 用例按设计跳过。
- 新增 RTRT 测试 **8 项通过**。涵盖常量、缓冲复用、缩放、运动、jitter、斜平面、切镜、遮挡显露、灯光变化、阴影开关、离屏镜面、玻璃、Alpha 及图像对照。
- Compute Sanitizer 对其中 6 项 GPU 行为用例执行 `memcheck --leak-check full`：**0 errors，0 bytes leaked**。

日志：`build/rtrt-ctest.log`、`build/rtrt-no-cuda-ctest.log`、`build/rtrt-tests.log`、`build/rtrt-memcheck.log`。

## 实际运行

设备：NVIDIA GeForce RTX 3050 Laptop GPU，4 GiB。场景为内置 Cornell，1 SPP、8 层路径、SVGF 默认参数。

| 输出 / 内部尺寸 | 完整出图 | GPU 帧耗时 | 帧缓冲占用 | 互操作 / 整帧下载 |
|---|---:|---:|---:|---|
| 960×540 / 960×540，TAA | 300 / 300 | 69.57 ms | 419.24 MiB | active / 0 |
| 960×540 / 480×270，TAAU | 300 / 300 | 17.94 ms | 135.96 MiB | active / 0 |
| 640×360 / 640×360，TAA | 30 / 30 | 31.76 ms | 186.33 MiB | active / 0 |
| 640×360，强制主机呈现 | 30 / 30 | 32.21 ms | 186.33 MiB | fallback / 30 |

以上为日志最后一次完成的 GPU 计时样本，不含全部 CPU、窗口与驱动开销，也不是多场景平均帧率。原生 960×540 的阶段耗时为：首交点 0.89 ms、光照 10.92 ms、时域与亮点处理 28.60 ms、à-trous 20.27 ms、最终重建 8.89 ms。优化了共享邻域的几何验证和重复读取；同场景优化前约为 132.88 ms。

两次 300 帧测试均只有初始的一次历史重置，分配代数保持 29。主机呈现和互操作在相同尺寸、参数、30 帧后导出的 PNG **SHA-256 完全一致**：

```text
0B41EBEB13DF6D3707DE93EBAA77A9C09015DE5A8369486FC633FF6AB6700442
```

无 CUDA Viewer 以 `--mode rtrt` 启动时明确提示回退，运行 20 帧后报告 `mode=opengl shader=active`。`--mode path` 别名也已在主机呈现测试中确认映射到 RTRT。

日志：`build/rtrt-smoke.log`、`build/rtrt-taau-smoke.log`、`build/rtrt-host-smoke.log`、`build/rtrt-640-smoke.log`、`build/rtrt-no-cuda-smoke.log`。

## 画质对照

固定种子、96×96 Cornell、1024 SPP 离线参考、相同 8 层路径，重建运行 48 帧。指标在未曝光的线性 HDR 上计算，包含明亮发光边缘。

| 指标 | 结果 |
|---|---:|
| 原始单帧 1 SPP MSE | 0.104496 |
| SVGF + TAA MSE | 0.027969 |
| 50% 内部尺寸 SVGF + TAAU MSE | 0.074234 |
| 原始图连续帧差 MSE | 0.340988 |
| 重建后连续帧差 MSE | 0.007178 |

这一个场景中，原生重建误差下降约 **73.2%**，静态连续帧差下降约 **97.9%**。不能将这些比例直接外推到运动镜面、透明覆盖或其他资产。

对照图和线性 PFM 位于 `build/rtrt-validation/`：

- [1024 SPP 参考](../build/rtrt-validation/cornell-reference-1024spp.png)
- [原始 1 SPP](../build/rtrt-validation/cornell-raw-1spp.png)
- [SVGF + TAA](../build/rtrt-validation/cornell-svgf-taa.png)
- [50% TAAU](../build/rtrt-validation/cornell-svgf-taau-50.png)
- [960×540 Viewer 输出](../build/rtrt-validation/viewer-rtrt-960.png)
- [JSON 指标](../build/rtrt-validation/metrics.json)

## UI 与配置

真实 Viewer 中发送 `1 / 2`，确认 **RTRT → OpenGL → RTRT**，对应 `Techniques` 内容随之切换；退出码为 0，返回 RTRT 后仍为 `interop=active`。窄面板中的 RTRT 参数名已调整为完整显示。截图保存在检查用临时目录：

- [RTRT 面板](C:/Users/26041/AppData/Local/Temp/codex-shot-2026-09-11_07-11-33.png)
- [OpenGL 面板](C:/Users/26041/AppData/Local/Temp/codex-shot-2026-09-11_07-11-35.png)

会话测试验证 v6 中 OpenGL 与 RTRT 参数往返保存、旧 Path 别名、旧暂停 / 自动预览状态清除，以及无 CUDA 的模式回退。主机呈现路径也支持 `--capture`。

## 当前边界

首版通过 CUDA 软件 BVH 追踪，原生 960×540 尚未达到 60 FPS；降低内部尺寸可明显减少开销。反射依靠第一表面运动、次级命中距离和反应掩码，没有独立追踪次级反射对象的运动；快速反射、复杂透明和变形资产仍需更广的动态画质评估。这里交付的是自研 SVGF 风格重建、TAA / TAAU，没有接入神经降噪、ReSTIR 或 RT Core 后端。

## 第二轮优化：稳定历史并减少重建开销

### 诊断与修改

首版 69.57 ms 中约 57.76 ms 位于时域、空间滤波和最终重建。优先优化这些阶段，同时用回归用例区分采样噪声与真实场景变化。

新增的粗糙金属用例让不同距离的次级表面与环境具有相同辐亮度。在相机、物体、光照完全静止时，首版仍产生 **0.339167** 的平均反应掩码：逐帧随机的 BRDF 命中距离被当作场景变化，削弱镜面历史，再通过各分量反应掩码的最大值影响整个 TAA 输出。

本轮修改：

- 命中距离的时域反应仅用于不透明、粗糙度低于 0.06 的窄反射波瓣；粗糙反射的空间权重也移除随机命中距离项。粗糙反射与玻璃继续通过表面重投影、亮度统计和各自的历史上限响应变化。
- 历史裁剪和亮度反应同时参考历史矩方差，避免偶然全暗的低 SPP 邻域反复清空有效历史。显式灯光、材质变化仍使用当前邻域并提高反应权重。
- TAA 反应按分量辐亮度加权，包含变化前后的能量，使弱分量的随机变化不再控制整张图，亮度消失时也能及时响应。
- 16×8 CUDA 线程块共享 7×7 邻域的几何与解调后亮度，将亮点抑制、均值和方差计算合并；亮点阈值排除中心样本，避免异常值抬高自己的阈值。
- à-trous 为各分量共用邻居几何、深度与法线权重，并对方差做 3×3 预滤波，减轻空间权重随噪声跳变。
- 每个内部像素只合成一次线性 HDR，TAA / TAAU 从合成缓冲采样；历史深度容差每像素只计算一次。

### 同机性能对照

设备为 **RTX 5080 16 GiB**，驱动 616.64，CUDA 13.3，MSVC / Visual Studio 18 2026，Release / native architecture。构建目录 `build/cuda-native`。优化前后均为 Cornell、输出 960×540、1 SPP、8 层路径、默认 5 / 3 轮空间滤波。

每个配置预热 24 帧，记录随后 96 帧，逐帧同步以读取对应 CUDA event。运动配置为连续相机横移。计时路径生成完整图像但不创建窗口、不做整帧下载，因此以下是 **GPU 管线时间，不是 Viewer FPS**。JSON 同时保留 CPU 提交加 GPU 等待的时间及 P99；96 个样本的尾部分位数只能作本轮对照，不能代替长期帧率统计。

| 内部比例 / 相机 | 优化前 GPU P50 | 优化后 GPU P50 | 优化前 GPU P95 | 优化后 GPU P95 |
|---|---:|---:|---:|---:|
| 100% / 静止 | 12.66 ms | **6.28 ms** | 13.16 ms | **6.84 ms** |
| 100% / 横移 | 12.96 ms | **6.24 ms** | 13.44 ms | **6.79 ms** |
| 50% / 静止 | 3.50 ms | **1.54 ms** | 3.90 ms | **2.52 ms** |
| 50% / 横移 | 3.47 ms | **1.53 ms** | 3.83 ms | **2.52 ms** |

原生静止配置的中位耗时约减半；时域阶段 4.99 → 1.50 ms，à-trous 4.56 → 2.62 ms，重建 0.663 → 0.268 ms。各阶段分位数独立统计，不应直接相加。复用的统计与合成缓冲使原生帧缓冲占用从 419.24 增至 **433.08 MiB**，50% 内部比例从 135.96 增至 **139.42 MiB**。

日志及逐项数据：[优化前日志](../build/rtrt-before.log)、[优化后日志](../build/rtrt-after.log)、[优化前分布](../build/rtrt-before/performance.json)、[优化后分布](../build/rtrt-after/performance.json)。优化前新增的粗糙反射回归按预期失败，原有 8 项行为测试通过。

### 稳定性与动态响应

Cornell 继续使用首版相同的 96×96、1024 SPP 离线参考、48 帧重建及线性 HDR 指标；原始采样 MSE 仍为 0.104496。

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| SVGF + TAA MSE | 0.027969 | **0.014218** |
| 50% TAAU MSE | 0.074234 | **0.051200** |
| 静态连续帧差 MSE | 0.007178 | **0.001159** |
| 粗糙反射静态平均反应掩码 | 0.339167 | **0** |
| 粗糙反射静态 8×8 块均值帧差 MSE | 0.000013123 | **0.000001033** |
| 粗糙反射相机停止后块均值帧差 MSE | 0.000015598 | **0.000001128** |

原生图像误差降低约 **49.2%**，Cornell 静态帧差降低约 **83.9%**。块均值帧差用于观察低频斑块跳动，粗糙反射静态结果降低约 **92.1%**；它是专门的恒定照明用例，不能代表所有镜面场景。

新增移动阴影测试：遮挡物移开与移回后，分别在 4 帧时与 256 SPP 离线参考比较中心 6×6 区域，平方误差除以 `max(参考能量, 1)` 分别为 **0.0000718 / 0.0000734**。连续相机运动、阴影变化均未触发全局历史重置，原有切镜、显露、灯光变化、玻璃、Alpha 等回归继续通过。

对照产物：[优化前图像](../build/rtrt-before/cornell-svgf-taa.png)、[优化后图像](../build/rtrt-after/cornell-svgf-taa.png)、[1024 SPP 参考](../build/rtrt-after/cornell-reference-1024spp.png)、[画质指标](../build/rtrt-after/metrics.json)、[反射稳定性](../build/rtrt-after/glossy-stability.json)、[移动阴影](../build/rtrt-after/moving-shadow.json)。各 PNG 附带线性 PFM。

### 回归与实际出图

- CUDA 全套 CTest **11/11**、无 CUDA 全套 CTest **10/10** 通过。无 CUDA 可执行文件中的 GPU 行为按设计跳过；不能视为验证了 GPU。本轮显式启用性能用例时，`realtime_tests` 共 **11 项通过**。
- Compute Sanitizer 对 6 项纯 RTRT 用例执行 `memcheck --leak-check full`：**0 errors、0 bytes leaked**；对常量/非整块尺寸、相机与物体运动、粗糙反射 3 项执行 `racecheck`：**0 errors、0 warnings**。
- 检查限制：首次混合检查在移动阴影用例的离线 `resolve_frame_kernel` 启动时出现 CUDA 999，随后上下文报错；该次运行失败，不计为 sanitizer 通过。普通离线对照和完整 CTest 通过。日志保留在 `build/rtrt-memcheck-optimized.log`，六项纯 RTRT 的独立检查为 `build/rtrt-memcheck-pipeline.log`。
- Viewer 原生 960×540 连续 **300/300 帧**，`interop=active`、`downloads=0`、`history_resets=1`，分配代数稳定为 30。最后一帧 GPU 样本为 5.14 ms；Viewer 相机构图与上面的计时用例不同，此值不作为分布或 FPS 结论。[实际输出](../build/rtrt-after/viewer-rtrt-960.png)。
- 640×360、30 帧、相同参数下，互操作和主机呈现 PNG SHA-256 一致：`CCAD0B29E0A574C8174D42B089FDB52C6BF7714376AA3EFD268A06B810BF00F6`。

回归日志：`build/rtrt-ctest-optimized.log`、`build/rtrt-ctest-no-cuda-optimized.log`、`build/rtrt-racecheck-optimized.log`、`build/rtrt-viewer-optimized.log`、`build/rtrt-interop-optimized.log`、`build/rtrt-host-optimized.log`。

复现本轮画质与分布测试：

```powershell
cmake --build build/cuda-native --config Release --target realtime_tests viewer --parallel 4
$env:RTRT_VALIDATION_DIR = "$PWD/build/rtrt-after"
$env:RTRT_PERFORMANCE = "1"
.\build\cuda-native\bin\realtime_tests.exe
ctest --test-dir build/cuda-native -C Release --output-on-failure
```

### 下一阶段的验收重点

本轮针对已复现的历史误反应和重建开销完成修复。RTX 3050 Laptop 尚未复测，不能把 5080 的加速比当作它的 60 FPS 保证。快速镜面、复杂透明、高频 HDRI、密集资产与变形仍需目标场景验证。

后续按顺序推进：先保存实际问题场景及相机轨迹，在目标输出分辨率下统计至少 3000 帧的整帧 P50/P95/P99，同时观察停镜收敛、显露恢复和拖影；再针对残留的粗糙反射与透明噪声评估采样序列、独立镜面历史及次级运动；最后根据新分项计时决定 RT Core 后端与进一步缓冲布局优化。若目标是 60 FPS，整帧 P95 ≤ 16.67 ms 应作为明确验收条件，图像误差和动态拖影也需同时过关。
