# OptiX RTRT 性能与画质优化验证

基线为 `0b4945b`。验证环境：Windows、MSVC 14.51、CUDA 13.3、RTX 5080 16 GiB、驱动 616.64；使用 `default` / `default-release`。分辨率、SPP、反弹深度、SER 和降噪设置在前后对比中一致。结构化结果见 [summary.json](output/rtrt-optix-optimization-2026-09-25/summary.json)。

## 修改

- 前两轮 5×5 à-trous 使用 16×8 线程块和共享邻域。只缓存空间滤波所需的三类信号，颜色/方差记录填充为 13 个 float，避免原 64 字节记录在共享内存中产生 bank 冲突。保留边界检查、累加顺序、核权重和首轮历史反馈，粗尺度轮次继续直接读取稀疏邻域。未增加持久帧缓冲。
- 环境光和发光几何采样复用同一次 BRDF 求值的漫反射比例，减少每条有效直接光样本的重复着色计算，并使用采样方向分离分量，避免再次按偏移后的阴影射线方向求值。
- 最后一个允许的路径顶点不再对直接光应用双策略 MIS：此时没有 BSDF 续接样本，直接使用光源采样估计。共享采样函数的默认参数保留离线路径的行为。

NVIDIA 的 [OptiX 9.1 指南](https://raytracing-docs.nvidia.com/optix9/guide/optix_guide.251118.LTR.pdf) 说明了命中对象与 SER 的接口；[PBRT 的 GPU 路径追踪实现](https://pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Path_Tracer_Implementation) 说明了路径终止与 MIS 光源命中项之间的关系。本轮保留原有 OptiX 求交管线。直接读取命中对象、省去 closest-hit 的实验在玻璃轮廓测试中退化，已撤回，未纳入最终性能结果。

## 性能方法

GPU 阶段计时使用 Cornell 960×540，1 SPP、8 层路径、SVGF，各配置预热 24 帧、记录 96 帧。交替运行旧、新二进制共三组，报告各次 P50/P95 的中位数，避免将单次时钟波动当作收益。该测试同步 GPU，不含窗口呈现。

Viewer 使用 San Miguel 固定场景和相机，1920×1080 输出、50% 内部光照及原生材质，开启相机移动；每次运行 240 帧，前 60 帧预热，计时包含 UI 与呈现后的 `glFinish`。前后使用相同命令，并交替运行三组。

| 场景 / 配置 | 基线 P50 ms | 优化后 P50 ms | 耗时下降 |
|---|---:|---:|---:|
| Cornell 原生，静止，GPU | 3.9415 | 3.6651 | 7.0% |
| Cornell 原生，移动，GPU | 3.9399 | 3.6659 | 7.0% |
| Cornell 50%，静止，GPU | 1.5735 | 1.4883 | 5.4% |
| Cornell 50%，移动，GPU | 1.5757 | 1.4819 | 6.0% |
| San Miguel 1080p，移动，完整帧 | 15.7568 | 15.3190 | 2.8% |

原生 Cornell 的 à-trous 阶段 P50 从 1.4435 ms 降至 1.2264 ms，下降 15.0%；光照阶段从 1.2681 ms 降至 1.2161 ms。San Miguel 完整帧 P95 从 16.2043 ms 降至 15.7326 ms；三次新版本运行合计仍有 1 帧超过 16.667 ms，不视为长时间无掉帧保证。正常 Viewer 路径均报告 `primary=optix`、SER 生效、互操作生效、整帧下载 0、软件 BVH 构建 0。

## 画质方法

增加末端直接光能量回归：纯漫反射平面在均匀环境中使用解析反射率作参考；大面积发光面使用独立离线 4096 SPP 参考。组合 1/4 层路径和 1/4 光源样本，关闭降噪及 TAA，统计 16 帧 × 32 SPP 的均值。

此外保留 Cornell 1024 SPP 对照、OptiX AI 降噪四种配置、运动阴影/遮挡显露、镜面高光和原生纹理测试，并重跑 San Miguel 320×180、1024 SPP 参考对照。新增 1×1、17×13、33×9 的空间滤波常量保持测试，覆盖不完整线程块，以及整片玻璃轮廓的均匀环境亮度检查。

| 画质指标 | 基线 | 优化后 |
|---|---:|---:|
| 1 层路径，环境光能量 / 解析参考 | 0.182111 | 1.000016 |
| 1 层路径，面积光能量 / 离线参考 | 0.491221 | 0.999436 |
| Cornell SVGF + TAA 对参考 MSE | 0.013941643 | 0.013941639 |
| Cornell 50% TAAU 对参考 MSE | 0.012451830 | 0.012451818 |
| Cornell 静态帧差 MSE | 0.0011553693 | 0.0011553696 |
| San Miguel 对参考 HDR MSE | 0.049645564 | 0.049645547 |
| San Miguel 静态帧差 MSE | 0.006692815 | 0.006692802 |

前两行是关闭降噪/TAA 的能量回归，均使用 1 个光源样本。1/4 层路径 × 1/4 光源样本的全部八个组合，优化后能量误差小于 0.08%。默认 8 层路径下，Cornell 和 San Miguel 的参考误差与闪烁指标基本保持不变；画质修复主要针对较低反弹深度下的能量缺失，不宣称默认场景普遍更清晰。两份 Cornell 离线参考图（512/1024 SPP），以及均匀环境玻璃图，在前后比较中逐像素相同。

## 回归

- CUDA + OptiX 默认 CTest：11/11 通过。单独启用性能扫描的实时测试：28 通过、2 项可选长测跳过；San Miguel 高采样对照另行通过。
- no-CUDA Release：11 通过、OptiX 互操作 1 项跳过。本地已有 no-CUDA 配置启用了额外的 benchmark 测试，因此测试总数与默认配置不同。
- 玻璃轮廓每像素最大颜色误差为 `1.26e-6`，前后图像完全一致；新增直接光能量测试在基线失败、修正后通过。
- Compute Sanitizer 独立检查空间滤波不完整线程块和玻璃/Alpha：2 项通过，0 错误、0 字节泄漏。日志为 `memcheck-filter.log` 和 `memcheck-filter-tests.log`。

Compute Sanitizer 的扩大检查未完整通过：首次在离线参考的 `resolve_frame_kernel` 提交时返回 CUDA 999；剔除离线参考后的检查在 SER 对照用例中返回 OptiX 7900 / CUDA 999，原生光学工作列表用例也触发该错误。后续用例因同一 CUDA 上下文已失效而连带失败。独立运行未修改的 `0b4945b` 二进制，SER 用例在相同检查条件下重现该错误，加入 `--force-synchronization-limit 64` 后仍重现。因此不能宣称完整内存验证通过，也未将这些 API 错误归因于本轮修改。失败日志保存在 `build/optix-optimization/memcheck*.log`；正常 Release 的 SER 对照与完整回归均通过。

检查遵循 [NVIDIA Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)，使用 `RTRT_OPTIX_VALIDATE=1`、`--tool memcheck --check-optix-leaks --leak-check full --error-exitcode 99`。没有修改系统驱动、注册表或 GPU 超时设置。

## 复现

```powershell
cmake --preset default
cmake --build --preset default-release
$env:RTRT_REQUIRE_OPTIX = '1'
$env:RTRT_REQUIRE_OPTIX_DENOISER = '1'
ctest --preset default-release --output-on-failure

$env:RTRT_VALIDATION_DIR = "$PWD/build/optix-optimization/after"
$env:RTRT_PERFORMANCE = '1'
.\build\default\bin\realtime_tests.exe
$env:RTRT_PERFORMANCE = $null

.\build\default\bin\viewer.exe --scene-file benchmarks/cases/san_miguel_first_scene.rscene `
  --camera-preset benchmarks/cases/san_miguel_first_scene.json `
  --rtrt-config benchmarks/cases/rtrt_san_miguel_1080p.json `
  --mode rtrt --width 1920 --height 1080 --no-restore-last `
  --frames 240 --warmup-frames 60 --camera-motion `
  --frame-report build/optix-optimization/after/san-miguel.json
```

完整日志、PNG/PFM、重复计时报告及旧版二进制保存在本地 `build/optix-optimization/`。性能结论限于本次设备和场景；其他 GPU、HDR firefly 长测和长时间交互未在本轮重新验证。
