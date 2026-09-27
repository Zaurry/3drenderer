# 稳定画质默认参数与分区 Reset（2026-09-27）

本轮按“稳定画质优先、允许适当增加帧耗时”选择参数，验证设备为 RTX 5080，CUDA 13.3、OptiX 9.1、驱动 616.64。使用上一轮冻结的湿润溶洞面积光场景、镜面高光和 Cornell 静止/移动/停止序列；没有修改场景材质来降低噪声。它是一组有实测依据的默认值，不代表所有场景的全局最优。

## 默认值

| 板块 | 参数 |
|---|---|
| Rendering / Display | RTRT（不支持时 OpenGL）；输出比例 100%；曝光 0 EV；ACES |
| RTRT Lighting & Sampling | 1 SPP；8 层路径；第 3 层开始轮盘赌；1 个光源样本；SER、Sobol、稳定玻璃、高光抗锯齿、间接高光稳定开启 |
| RTRT Denoising | OptiX AI；时域复用与亮点抑制开启；firefly sigma 6；漫反射/静止历史 64 帧；几何拒绝阈值 0.02 / 0.85 |
| SVGF 可切换/回退参数 | 镜面/透射历史 16 / 4；历史裁剪 sigma 2；reactive strength 1；漫反射/镜面 à-trous 5 / 3 轮；depth sigma 1、normal power 64、luminance sigma 4 |
| TAA / Temporal Upscaling | 原生分辨率；TAA、TAAU、原生材质开启；当前帧权重 0.15；裁剪 sigma 1.5；轻度锐化 0.05 |
| OpenGL Direct Lighting | 保留已有基准值：IBL、LTC、PCSS 开启；阴影 1024、8 灯预算；PCSS 16 / 32 样本 |
| OpenGL AO | GTAO 3 个切片、每侧 3 样本；半径/场景半径 0.1；Bent Normal 与双边滤波开启 |
| OpenGL DDGI | 自动适配；12×8×12 探针；每探针 128 射线；每帧更新 256 探针；历史权重 0.95；重定位/分类开启 |
| OpenGL SSR | 2 rays/pixel；64 次 Hi-Z 访问；32 帧历史；3 轮滤波；thickness scale 0.002 |
| Interface / Diagnostics | 字号 100%；所有诊断视图恢复 Final |

完整参数由程序自身的会话序列化器导出：[RTRT JSON](output/rtrt-defaults-2026-09-27/rtrt-quality.json)、[全部渲染及显示参数](output/rtrt-defaults-2026-09-27/viewer-defaults.json)。OpenGL 使用原有基准值，本轮重点标定 RTRT，未对 OpenGL 的所有组合重新寻优。离线渲染器的显示默认值未改变。

`Diffuse / static history`、深度和法线拒绝阈值现在在 OptiX 模式也可调；它们影响两种降噪路径，原先界面错误地将它们一起置灰。64 帧增加静止收敛时间；运动继续使用原有当前帧权重和拒绝逻辑。

## 为什么没有直接提高 SPP

1080p 原生、OptiX AI，溶洞内正弦横移，每组 240 帧、预热 60 帧。完整帧计时包含 UI、呈现以及 `glFinish`，关闭垂直同步。候选只在指定参数上变化，不采用降低分辨率来掩盖开销。

| 候选 | P50 ms | P95 ms |
|---|---:|---:|
| 1 SPP / 1 光源样本，32 帧历史、锐化 0 | 29.50 | 30.27 |
| **最终：1 SPP / 1 光源样本，64 帧历史、锐化 0.05** | **29.43** | **30.10** |
| 1 SPP / 2 光源样本 | 48.70 | 49.29 |
| 2 SPP / 1 光源样本 | 78.82 | 79.89 |
| 2 SPP / 2 光源样本 | 148.40 | 150.34 |
| 4 SPP / 1 光源样本，TAA 权重 0.08、裁剪 2 | 145.47 | 147.60 |

前两行使用最终同一二进制交替配置；其余为本轮候选测量。小于 1% 的差异不作为性能提升结论。最终默认值约 34 FPS，记录帧均实际启用神经降噪及硬件求交，整帧下载计数为 0。增加样本改善部分空间误差，但本机代价明显超出这次希望的适当增幅。

## 静止与连续移动

沿用[上一轮溶洞视角和指标定义](rtrt-motion-stability-2026-09-27.md#同场景连续帧)：96 帧静止后连续横移 96 帧，每帧 0.004 单位；静止统计后 32 帧。比较同一种降噪器的旧标准参数（32 帧历史、无锐化）与最终参数（64 帧、0.05 锐化），其他参数相同。运动帧先按首表面重投影；数值为线性 ACES RGB RMSE，**包括真实反射和覆盖变化，不能当成纯噪声**。

| 分辨率 / 降噪器 | 静止旧值 | 静止新值 | 运动旧值 | 运动新值 |
|---|---:|---:|---:|---:|
| 320×180 / SVGF | 0.0023410 | 0.0014959 | 0.0331727 | 0.0342913 |
| 320×180 / OptiX | 0.0015519 | 0.0007852 | 0.0109962 | 0.0114609 |
| 1920×1080 / SVGF | 0.0005227 | 0.0003247 | 0.0095017 | 0.0096574 |
| 1920×1080 / OptiX | 0.0010974 | 0.0005541 | 0.0071185 | 0.0073994 |

1080p 静止帧差分别下降约 38% / 50%；运动指标分别增加约 1.6% / 3.9%。只延长历史、不锐化时 OptiX 运动指标基本不变。轻度锐化用于弥补神经降噪对细高光的软化，也会略放大运动中的细节变化。

320×180 的最后 16 个运动帧另与同视角 256 SPP 离线参考比较。OptiX 显示 RMSE 从 0.070364 变为 0.070236，时序残差从 0.049859 变为 0.049964，基本相当；SVGF 为 0.075309 → 0.076767、0.058182 → 0.058788。参考本身也含采样噪声，不能据此宣称运动已彻底稳定。

细镜面高光按两倍历史长度预热后，原生默认配置的 HDR MSE 为 2.83858（原门槛 < 3），能量比 0.86717（原门槛 0.8～1.2）。不锐化的 OptiX 在相同预热下 MSE 为 3.09157，0.1 锐化为 2.60078，但对溶洞运动波动的放大更多，因此选择较轻的 0.05。误差门槛未放宽，能量比仍说明它不是无损重建。

最终 1080p 输出：[OptiX](output/rtrt-defaults-2026-09-27/cave-optix-1080.png)、[SVGF](output/rtrt-defaults-2026-09-27/cave-svgf-1080.png)。[测量摘要](output/rtrt-defaults-2026-09-27/metrics.json)保留全部候选指标。

## Reset 行为

- Rendering、Display、Interface、Camera、Direct Lighting、AO、DDGI、SSR 和四个 RTRT 板块均有独立 Reset。环境光、Shadow Map、PCSS、Dominant Light、LTC、SSAO/GTAO、AO 滤波也有子区 Reset。GLSL 区 Reset 恢复自动重载并重载当前文件。
- 按钮使用独立 ID，放在禁用控件之外；折叠标题仍可点击，不会误触展开，也不会重置其他板块。Performance 的 Reset 专门清空渲染历史。
- Camera Reset 按当前场景边界恢复 45° 视场角、适配距离和移动速度；不会误回到启动时恢复的旧相机参数。
- DDGI Reset 每次都递增清空/适配命令，重复点击仍能重新计算探针。
- 环境光 Reset 保留 HDRI 来源，恢复颜色、强度、旋转和背景开关；场景对象、灯光位置和材质编辑继续通过原有编辑器处理。
- `Quality defaults` 恢复全部 RTRT 参数；`1080p performance preset` 保留原有半分辨率、1 SPP、SVGF、32 帧历史、TAA 权重 0.03/裁剪 3、锐化 0 的性能方案。

已有会话的显式参数继续正常保存/恢复，没有加入每次启动强制覆盖。按本次要求，本机已备份 `last-session.json` 并更新渲染/显示参数；恢复了此前关闭的亮点抑制、IBL，以及 8 帧的镜面历史等设置。场景文档、相机、窗口布局保持原内容，用户编辑过的 `cave.rscene` 未改写或提交。

## 验证

- `cmake --build --preset default-release`，`ctest --preset default-release --output-on-failure`：12/12 测试组通过，设置 `RTRT_REQUIRE_OPTIX=1` 和 `RTRT_REQUIRE_OPTIX_DENOISER=1`，避免静默跳过神经路径。
- `cmake --build --preset no-cuda-release`，`ctest --preset no-cuda-release --output-on-failure`：12 组通过，1 组 CUDA 互操作按预期跳过。
- 新增 ImGui 实际鼠标按下/释放测试：折叠状态 Reset、禁用效果 Reset、相邻板块互不影响、两种降噪器面板绘制。
- 默认设置会话往返、分区恢复、DDGI 重复清空/适配，以及使用实际默认参数的神经降噪、镜面、湿润高光、相机移动后停止回归通过。
- 溶洞静止/运动在 320×180 与 1080p 各测 SVGF/OptiX，验证有限非负输出和真实神经降噪。没有重跑可选 San Miguel 长测和全部高采样光学压力测试；这次没有改动渲染算法内核。

复用配置示例（场景需自行选择）：

```powershell
.\build\default\bin\viewer.exe --scene builtin --mode rtrt --no-restore-last --rtrt-config docs/output/rtrt-defaults-2026-09-27/rtrt-quality.json
```
