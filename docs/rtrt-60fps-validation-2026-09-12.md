# RTRT 复杂场景 60 FPS 验证（2026-09-12）

验收对象为 **RTX 5080 / 1920×1080 / San Miguel**，使用用户指定的 `default-release` 构建。本轮完成 OptiX RT Core 后端、SER、原生分辨率材质重建和响应修复。排除外部 GPU 负载后，最终 **3540 帧长测平均 69.5 FPS，完整帧 P95 15.399 ms、P99 15.756 ms，超预算 0 帧**，达到约定的 60 FPS 门槛。结论限定于本页的配置、场景和相机轨迹；光照内部尺寸为 960×540。

此前光栅首交点、Sobol 和滤波优化见 [第三轮验证](rtrt-validation-2026-09-12.md)，更早基线见 [2026-09-11 验证](rtrt-validation-2026-09-11.md)。本页数据使用同机、同场景测量，不与 RTX 3050 Laptop 的时间计算加速比。

**后续状态：** 镜面和玻璃修复见 [2026-09-13 firefly 记录](rtrt-firefly-validation-2026-09-13.md)。新增原生光学补追后的长测平均 61.22 FPS、P95 17.818 ms，未通过相同门槛；最后独立着色程序调整尚未运行验收。本文的 69.5 FPS 成绩保留为前一阶段结果，不代表当前工作区的最终状态。

## 配置与计时口径

- NVIDIA GeForce RTX 5080，16 GiB，驱动 616.64；CUDA 13.3，MSVC Release，CUDA 核目标 `sm_120`，OptiX 9.1。
- 场景：[san_miguel_first_scene.rscene](../benchmarks/cases/san_miguel_first_scene.rscene)，**5,617,451 个三角形**；保持该文件中的材质、HDRI 和光源。
- 相机：[san_miguel_first_scene.json](../benchmarks/cases/san_miguel_first_scene.json)，固定初始视角；`--camera-motion` 每 900 帧重复 600 帧横移、300 帧静止，横移幅度 0.5 个世界单位。
- 最终配置：[rtrt_san_miguel_1080p.json](../benchmarks/cases/rtrt_san_miguel_1080p.json)。**光照采样 960×540，输出与材质 1920×1080**；1 SPP、8 层路径，直接光、软阴影、GI、反射、透射均开启。SER 开启，TAA 当前帧权重 0.03、裁剪 3。
- 普通默认参数仍是 100% 内部光照；面板的 `1080p RT quality preset` 按钮应用上述配置，不修改窗口尺寸。此配置不是原生 1080p 全路径追踪，也没有使用帧生成、DLSS、ReSTIR 或神经降噪。
- 报告从每帧 CPU 开始计时，直到呈现后的 `glFinish` 返回，包含 UI、提交、互操作与 GPU 完成；交换间隔为 0。加载、预热和末尾截图不计入分布。正常交互运行没有新增这项同步。
- 验收要求：至少 3000 个预热后样本，完整帧 **P95 ≤ 16.667 ms**，同时记录 P99、超预算帧数与历史/分配计数。GPU 分阶段计时异步收集，可能滞后一帧，仅用于定位瓶颈。

## 实现与性能探测

原来的 CUDA 软件 BVH 在该场景中耗费大量时间。新后端让首交点、阴影和路径续接使用 RT Core；材质、纹理、BRDF、NEE/MIS 与软件路径共享实现。静态网格缓存 GAS，刚体变换更新 IAS；Alpha 和双面规则变化会刷新几何分类。原始解析球和不支持 OptiX 的驱动回退原 CUDA 遍历，并显示原因。

不透明单面和双面三角形跳过无用的 any-hit 着色；需要 Alpha 接受规则或共享网格绑定不一致时保留接受程序。次级射线按材质执行 SER，首交点和阴影保持直接追踪。启动参数使用有完成事件保护的固定页循环缓冲，避免普通主机内存复制带来的提交停顿。

| 阶段 | 测量帧数 | 完整帧 P50 | P95 | P99 | 超过 16.667 ms |
|---|---:|---:|---:|---:|---:|
| 本轮前，1080p CUDA 软件遍历 | 180 | 181.071 ms | 188.443 ms | 190.268 ms | 180 |
| OptiX，1080p 原生光照 | 180 | 54.185 ms | 55.069 ms | 55.557 ms | 180 |
| 再分类不透明几何，1080p 原生光照 | 180 | 51.758 ms | 52.468 ms | 53.958 ms | 180 |
| 50% 光照，旧 RGB 重建 | 180 | 16.586 ms | 17.390 ms | 18.755 ms | 81 |
| 50% 光照、原生材质、固定页提交，SER 关 | 900 | 16.434 ms | 16.929 ms | 17.191 ms | 168 |
| 同配置，SER 开 | 900 | **14.508 ms** | **14.942 ms** | **15.121 ms** | **0** |

以上是开发阶段探测，帧数和阶段实现不同，不能替代最后代码的长测。SER 两组使用相同尺寸、8 层路径和相机轨迹；平均完整帧时间从 16.443 降为 14.506 ms。原生 1080p 光照仍没有达到 60 FPS。

原始报告：[软件基线](../build/rtrt60/baseline-probe.json)、[OptiX 原生](../build/rtrt60/optix-native-probe.json)、[不透明分类](../build/rtrt60/optix-opaque-probe.json)、[旧半分辨率](../build/rtrt60/optix-half-probe.json)、[SER 关](../build/rtrt60/pinned-half-probe.json)、[SER 开](../build/rtrt60/ser-half-probe.json)。这些 `build/` 文件是本地验证产物，不随源码提交。

### 最终长测与受干扰记录

最终运行 3600 帧、排除前 60 帧。测试前外部游戏进程已经退出，GPU 约 4% 占用；测试结束后恢复为约 1% / 2106 MiB。没有同时运行其他 GPU 测试。

| 阶段 | 样本数 | 平均完整帧 | 平均 FPS | P50 | P95 | P99 | 最大帧时间 | 超预算帧数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 3540 | 14.387 ms | **69.51** | 14.311 ms | **15.399 ms** | **15.756 ms** | 16.400 ms | **0** |
| 相机横移 | 2340 | 14.398 ms | 69.45 | 14.322 ms | 15.439 ms | 15.801 ms | 16.400 ms | 0 |
| 相机静止 | 1200 | 14.366 ms | 69.61 | 14.285 ms | 15.342 ms | 15.653 ms | 16.377 ms | 0 |

平均 FPS 使用 `1000 / 平均完整帧毫秒数`。GPU 阶段中位数：首交点（含原生材质）0.944 ms，光照 7.585 ms，时域 1.036 ms，空间滤波 2.267 ms，重建/输出 1.253 ms；CUDA 总计中位数 13.284 ms。异步阶段统计用于解释瓶颈，验收使用完整帧时间。

全部样本为 `hardware_rt=true`、960×540 内部尺寸、分配代数 40、历史重置 1、整帧下载 0；帧缓冲稳定在 794.971 MiB，显式 OptiX 缓冲约 331.323 MiB，后者不含驱动内部内存。

数据：[完整逐帧报告](../build/rtrt60/final-clean-3600.json)、[分阶段汇总](../build/rtrt60/final-clean-summary.json)、[最终运行日志](../build/rtrt60/final-clean-3600.log)、[1080p 截图](../build/rtrt60/final-1080p.png)、[测前环境](../build/rtrt60/final-clean-environment-before.log)、[测后环境](../build/rtrt60/final-clean-environment-after.log)。

第一次运行 3600 帧、排除前 60 帧，得到 3540 个样本：P50 **67.625 ms**，P95 **82.017 ms**，P99 **86.565 ms**，3390 帧超预算。后续 180 帧诊断同样变慢。两个失败结果均保留：[首次长测](../build/rtrt60/final-3600.json)、[诊断探测](../build/rtrt60/frame-copy-probe.json)。

渲染器退出后，`nvidia-smi` 仍报告 GPU 93% 占用、9148 MiB 显存；Windows GPU Engine 计数器将约 94–96% 的 3D 占用定位到另一程序 PID 30948。因此不能据该分布判断渲染器独占 GPU 的速度，也不能用它证明内存泄漏。CUDA 资源报告显示重建与降噪核 `STACK:0 / LOCAL:0`，没有支持“新增分支造成局部内存溢出”的证据。

受干扰的测量中，分配、历史和尺寸计数同样保持稳定。保留失败分布用于说明测试条件，不把它混入 GPU 空闲后的最终验收，也不删除这些异常样本后计算达标数字。

## 画质与稳定性

低分辨率光照不再直接放大 RGB。额外的输出分辨率首交点获取材质、发光和几何；按表面身份、几何法线与深度匹配光照，再调制原生材质。透明表面保留辐亮度重建。输出历史使用几何法线验证覆盖，避免法线贴图随 jitter 变化造成错误拒绝；反应掩码取贡献邻域最大值。

参考为同一 San Miguel 相机、1920×1080、1024 SPP 离线 CUDA 图像。各配置静态渲染 96 帧；最终 HDR 和 ACES 显示空间 MSE 对比参考，最后 16 帧测逐像素帧差与 8×8 块均值帧差。该参考仍含采样噪声，指标不代表所有视角或运动条件。

| 指标，越低越好 | 原生 1080p 光照 | 旧 50% RGB 重建 | 最终 50% 光照＋原生材质 |
|---|---:|---:|---:|
| HDR MSE | 0.0509783 | 0.0637421 | **0.0566115** |
| 显示空间 MSE | 0.00410880 | 0.00688067 | **0.00606033** |
| 静态像素帧差 MSE | 0.00121909 | 0.00234861 | **0.00103169** |
| 静态 8×8 块帧差 MSE | 5.83303×10⁻⁵ | 2.83211×10⁻⁴ | **6.25040×10⁻⁵** |
| 总能量 / 参考（接近 1 为佳） | 0.964541 | 0.963411 | **0.966924** |

相较旧的 50% RGB 重建，最终配置 HDR 误差降低 **11.2%**，显示误差降低 **11.9%**，像素帧差降低 **56.1%**，低频块帧差降低 **77.9%**。相较原生光照，像素帧差降低约 15.4%，但 HDR 误差增加约 11.1%、显示误差增加约 47.5%、块帧差增加约 7.2%。因此可以确认半分辨率方案画质与稳定性明显改善，不能宣称达到原生光照细节或消除了全部 boiling。

数据：[原生与中间配置](../build/rtrt60/quality1080/san-miguel-quality.json)、[最终配置](../build/rtrt60/final-quality/san-miguel-quality.json)。图像：[1024 SPP 参考](../build/rtrt60/quality1080/san-miguel-reference.png)、[原生光照](../build/rtrt60/quality1080/san-miguel-native.png)、[旧重建](../build/rtrt60/quality1080/san-miguel-bilinear.png)、[最终重建](../build/rtrt60/final-quality/san-miguel-final.png)；每张 PNG 附带同名 `.png.pfm` 线性图像。

额外回归：

- 纹理棋盘用例的 50% 重建 MSE 从 **0.00609541** 降为 **1.24485×10⁻⁸**，验证材质细节保留。
- Cornell 50% 重建的 HDR MSE 从 **0.0510778** 降为 **0.0124495**；原生光照仍为 0.0139413。该简单用例不代替复杂场景结果。
- 关闭软阴影，且没有环境和发光几何采样时，首表面确定性直接光不再经过宽空间核；保留历史用于检测变化，随机反射/间接光仍按原管线降噪。移动阴影在原生和最终配置中，4 帧后两个方向区域误差均低于 **2×10⁻⁷**，避免以长时间模糊掩盖拖影。
- 移走离屏反射物后，原生和最终配置的中心亮度在 8 帧后降为 0，历史重置计数仍为 1。SER 开关的原始路径对照通过；材质、Alpha、镜像实例与材质绑定变化通过软硬件路径对照。

## 构建和诊断

- `cmake --build --preset default-release` 成功；default CTest **10/10** 通过。`realtime_tests` **15 通过、2 项可选性能/场景测试跳过**；复杂场景画质测试另行显式执行。
- `no-cuda-release` 构建成功，CTest **10 通过、1 跳过**；GPU 子用例按能力跳过，不计为 GPU 验证成功。
- 最后一次 CUDA 辅助函数参数整理后，常量与尺寸、移动阴影、离屏反射物、原生纹理重建、SER 共 **5 项通过**。
- `RTRT_OPTIX_VALIDATE=1` 启用 OptiX 完整 API 验证、栈/追踪深度检查及完整调试信息，**7 项通过、0 失败、0 跳过**，硬件材质与 SER 用例确认后端实际启用。SDK 要求完整调试信息搭配优化等级 0；该模式用于诊断，不用于性能验收。正常模式保持优化等级 3。
- Compute Sanitizer 基础常量/尺寸与硬件材质更新两项通过，**0 errors、0 bytes leaked**。扩展检查在不同测试处出现 `Internal Sanitizer Error`、硬件异常处理失败及 CUDA 999，随后产生级联错误；GPU 空闲复测及完整 OptiX 调试模式仍能触发。普通执行和 OptiX API 验证未复现。**完整新后端 memcheck 尚未通过，根因未定位**，不能把它断言为纯工具问题，也不能把基础两项或前轮软件管线的结果推广到全部 OptiX/SER 路径。上下文异常后报告的泄漏包含未能正常清理的资源，不是稳定运行时泄漏的证据。

日志：`build/rtrt60/final-default-build.log`、`build/rtrt60/final-default-ctest.log`、`build/rtrt60-no-cuda-ctest.log`、`build/rtrt60/all-realtime-tests.log`、`build/rtrt60/final-helper-tests.log`、`build/rtrt60/optix-final-validation.log`、`build/rtrt60/memcheck-basic.log`、`build/rtrt60/memcheck-limited.log`、`build/rtrt60/memcheck-blocking.log`、`build/rtrt60/memcheck-clean.log`、`build/rtrt60/memcheck-optix-debug.log`。

## 复现

先关闭占用 GPU 的其他渲染/游戏任务。以下命令从仓库根目录执行，固定配置和相机：

```powershell
cmake --build --preset default-release
ctest --preset default-release
.\build\default\bin\viewer.exe `
  --scene-file benchmarks/cases/san_miguel_first_scene.rscene `
  --camera-preset benchmarks/cases/san_miguel_first_scene.json `
  --camera-motion --mode rtrt `
  --rtrt-config benchmarks/cases/rtrt_san_miguel_1080p.json `
  --width 1920 --height 1080 --frames 3600 --warmup-frames 60 `
  --frame-report build/rtrt60/recheck-3600.json `
  --capture build/rtrt60/recheck-1080p.png --no-restore-last
```

完整的 1080p 参考会消耗数分钟，勿与性能测试同时运行。复现最终画质：

```powershell
$env:RTRT_SAN_MIGUEL_QUALITY = "1"
$env:RTRT_QUALITY_WIDTH = "1920"
$env:RTRT_QUALITY_VARIANT = "7"
$env:RTRT_VALIDATION_DIR = "$PWD/build/rtrt60/recheck-quality"
.\build\default\bin\realtime_tests.exe --filter san_miguel_quality
```

可通过 `RTRT_REFERENCE_PFM` 显式复用已生成的同场景同相机参考；测试验证 PFM 格式和尺寸，调用者负责确保相机、灯光与路径设置一致。不要使用不同配置的缓存参考。

实现依据：[NVIDIA OptiX API](https://raytracing-docs.nvidia.com/optix9/api/group__optix__device__api.html)、[固定版本 OptiX 头文件](https://github.com/NVIDIA/optix-dev/tree/f1f6dd803f3159992d248178f6e09421c6eb8b6d)。许可随安装产物位于 `share/renderer/licenses/optix`。

内存诊断的调试信息设置参照 [NVIDIA Compute Sanitizer 的 OptiX 支持说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#optix-support)。
