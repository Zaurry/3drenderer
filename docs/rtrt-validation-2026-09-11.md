# RTRT 首版验证记录

日期：2026-09-11。实现和参数见 [RTRT 使用说明](rtrt-renderer.md)。

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
