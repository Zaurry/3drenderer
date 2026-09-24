# RTRT 实时光线渲染器

交互 Viewer 提供 `OpenGL / RTRT` 两种模式，快捷键为 `1 / 2`。新会话默认 RTRT；OptiX 或 RT Core 不可用时回退 OpenGL，并在状态栏说明原因。`--mode path` 是 RTRT 的兼容别名。离线 `renderer` 仍使用高采样路径追踪，便于输出参考图像。

全光线 OptiX 迁移后的构建、画质对照和实际 GPU 运行结果见 [2026-09-23 迁移验证](rtrt-optix-migration-2026-09-23.md)。OptiX AI 降噪此前的验证见 [降噪记录](rtrt-optix-denoiser-2026-09-13.md)。镜面/玻璃此前的复杂场景长测见 [2026-09-13 修复记录](rtrt-firefly-validation-2026-09-13.md)，本轮没有重跑 San Miguel 长测。

```powershell
cmake --build --preset default-release
.\build\default\bin\viewer.exe --scene builtin --mode rtrt --no-restore-last
.\build\default\bin\viewer.exe --scene builtin --mode rtrt --frames 300 --no-restore-last --capture build/rtrt.png
```

## 管线

每帧发布完整图像：场景增量上传 → OptiX 首交点 G-buffer→ 低 SPP 路径采样 → 可选 SVGF / OptiX AI 降噪与材质合成 → TAA / TAAU → 曝光、色调映射与 sRGB 显示。

RTRT 的全部求交（主射线、续接路径、阴影、原生玻璃/镜面补射）统一进入 OptiX 9.1。三角形通过 RT Core 求交；解析球使用 OptiX 内置球体程序并参与其加速结构遍历。主射线、低采样光照和精细光学保持独立阶段，raygen 内迭代路径，closest-hit 只返回紧凑命中记录；阴影使用独立单值 payload、首命中终止和材质接受程序。

`src/render/optix/` 分离光线程序、路径积分器、共享设备参数、实时调度与 CUDA 后处理。材质、GGX/Kulla–Conty、纹理、NEE/MIS 继续共享场景着色实现。实时场景存储使用 `ShadingOnly`，不再构建软件 BVH；离线参考和 OpenGL DDGI 保持原有存储方式。

三角形与解析球分别建立 GAS，经 IAS 关联同一个逻辑实例及材质表。GAS 按资产缓存并压缩，刚体变换仅更新 IAS；颜色、粗糙度和灯光更新不重建 GAS，透明接受分类变化只重建相关几何。几何构建按批次等待，临时空间复用，实例及启动参数使用有完成事件保护的固定页内存循环缓冲。

SER 设置默认开启，但只有设备报告支持时，次级光线才使用 `optixTraverse → optixReorder → optixInvoke`；不支持时直接 `optixTrace`。面板与报告分别显示 `ser_supported`、`ser_active` 和 `rt_core_version`，不会把开启选项当作硬件加速生效的证据。

OptiX 头文件固定版本及 SHA256，PTX 以数值字节数组嵌入，兼容 MSVC 的字符串长度限制。`RENDERER_OPTIX=OFF` 保留 OpenGL 和离线 CUDA Path，但禁用 RTRT；无 CUDA 构建不下载 OptiX。离线参考保留原有 CUDA wavefront 队列。第三方许可随安装包保存在 `share/renderer/licenses`。

支持点光、聚光、太阳、矩形面积光、发光几何和环境光，以及多次反弹、离屏反射、玻璃和 Alpha Mask / Blend。点光和聚光的半径用于圆盘可见性采样，保留原有中心强度与范围衰减语义；太阳角半径用于方向采样。灯光的 `casts_shadows` 控制对应的直接阴影。

SVGF 将直接漫反射、间接漫反射、镜面和透射信号分别维护；直接可见发光和背景单独合成。漫反射先解调材质，再重投影，累积颜色、一二阶矩与历史长度，以短历史邻域估计方差，执行逐轮扩大步长的 à-trous。滤波使用独立读写缓冲，第一轮结果反馈颜色历史。粗糙镜面继续使用较短的颜色历史。粗糙度低于 0.06 的反射和透明表面的反射/透射保留历史统计用于响应与异常样本判断，颜色只在最终 TAA 中积累一次，同时跳过宽空间核。这样避免用首表面运动反复插值次级光学图像、扩散细高光。

`Stable glass sampling`（`split_dielectric`，默认开启）将首个玻璃界面的反射与折射按 Fresnel 权重分别追踪：一个反射分支，加两个半权重折射分支；后两者在下一个玻璃界面分别采样反射和折射。第一界面的权重为 `F`、`(1-F)/2`、`(1-F)/2`，第二界面再分别乘 `2F`、`2(1-F)`。连续玻璃链不做俄罗斯轮盘赌，后续随机路径仍按原规则处理。它减少低概率反射偶尔带回完整高亮辐射造成的闪点，不改变离线参考的采样器。Alpha Blend 覆盖保留原路径规则，玻璃表面自身发光只计一次。

清晰光学信号的异常样本阈值使用剔除最高一到两个邻居后的均值/方差；可信的同表面历史和匹配的次级命中距离保护已经存在的高光，新显露高光不直接按暗邻居剪掉。颜色按亮度等比缩放；清晰光学历史不设置邻域亮度下限，避免凭空填亮暗反射。它是有限采样下有偏的异常值抑制，并非无偏采样保证。

关闭软阴影且没有环境/发光几何采样时，不透明首表面的直接光是确定性的。它保留用于变化检测的历史，但直接使用当前值合成，避免宽空间核将移动硬阴影拖成模糊边缘；随机间接光与反射继续降噪。

邻域亮度和几何在 CUDA 线程块内共享，亮点抑制与时域邻域统计合并计算。历史裁剪同时参考历史矩方差；à-trous 预滤波方差，并为分量共享几何权重。最终 TAA 的反应按分量变化前后的辐亮度加权，输入来自只合成一次的线性 HDR 缓冲。本轮前后对照与限制见 [验证记录](rtrt-validation-2026-09-11.md#第二轮优化稳定历史并减少重建开销)。

后续优化保留前两轮 5×5 à-trous 与第一轮历史反馈，粗尺度轮次使用 3×3 核，减少访问量和滤波范围；背景跳过无用统计与空间滤波，最后一个路径顶点不再采样无人使用的续接射线。默认启用逐像素固定扰乱的 Owen Sobol 采样，前 16 个随机维度跨帧均匀覆盖，后续维度回退 PCG；它不是时空蓝噪声纹理。离线参考继续使用原 PCG。详细对照见 [2026-09-12 验证记录](rtrt-validation-2026-09-12.md)。

运动向量为“当前到历史”的 UV 位移。相机和刚体的连续运动按稳定对象 ID、上一帧变换重投影；逐个验证历史插值样本的深度、法线和表面身份。输入 jitter 单独补偿，最终 TAA 历史位于不抖动的输出网格。镜头切换、输出或内部尺寸变化、场景替换和不兼容几何变化清空历史；灯光、材质和遮挡变化使用历史裁剪与反应掩码更新。

正常互操作路径将结果直接写入 CUDA–OpenGL 纹理，整帧下载计数应为零。GPU 计时异步读取，帧缓冲持续复用。`--no-cuda-interop` 可强制使用主机呈现路径，以诊断互操作兼容性；它每帧下载完整图像，速度会降低。

低内部比例时，`Native materials / sharp optics`（沿用 JSON 字段 `full_resolution_materials`）默认额外获取输出分辨率的表面、材质和发光。重建按表面身份、几何法线和深度匹配光照样本，再调制原生分辨率的漫反射材质。透明表面与粗糙度低于 0.06、漫反射颜色最大分量低于 0.01 的抛光金属，复用原生首交点补追光学路径，避免把细反射和折射从低分辨率 RGB 放大。补追使用独立着色任务，以免提高整个首交点阶段的寄存器压力；额外首交点缓冲计入显存统计。输出历史按几何覆盖判断有效性，低分辨率反应掩码取贡献邻域的最大值。其他光照仍受内部采样率限制，这不是整幅原生分辨率全路径追踪或神经超分。

OptiX 启动参数使用有完成事件保护的固定页内存循环缓冲，避免普通主机内存的异步复制暗中串行化提交。软件遍历的错误回传已移除；OptiX API 和 CUDA 执行错误直接检查。验证/截图仍可显式下载，正常呈现不读取整帧。

## ImGui 与参数保存

`RTRT Denoising → Denoiser` 可在 `SVGF` 与 `OptiX AI` 间切换，`Enable denoising` 控制总开关，旧会话和默认配置继续使用 SVGF。OptiX 使用 9.1 的 `TEMPORAL_AOV` 神经模型；关闭 `Temporal reuse` 时使用单帧 `AOV` 模型。依赖已固定到 [NVIDIA OptiX 9.1 头文件](https://github.com/NVIDIA/optix-dev/blob/f1f6dd803f3159992d248178f6e09421c6eb8b6d/include/optix_types.h)，不需要单独下载模型，实际模型由兼容 NVIDIA 驱动提供。

OptiX 路径将当前帧线性 HDR 合成图直接送入网络，提供反照率、世界空间法线、前帧到当前帧的像素运动向量，以及依据深度、法线和对象身份计算的运动可信度。运动向量转换同时处理方向、分辨率和两帧采样抖动。玻璃和清晰镜面的首表面运动不能代表次级像，因此保守关闭这些像素的神经历史复用，仍执行空间降噪和可选输出 TAA。背景辐亮度原样保留。

启用 `Native materials / sharp optics` 且内部比例小于 100% 时，先在内部采样分辨率用 beauty + 四个光照 AOV 降噪，分别保留直接漫反射、间接漫反射、反射和透射；随后重建原生材质，并对原生玻璃/镜面补采样进行第二次神经降噪。第二次结果只用于这些光学像素，其余像素保留已重建的材质，避免放大原始噪声后再滤波，也避免重复模糊纹理。关闭原生材质选项时只在内部光照分辨率降噪，再由现有 TAAU 重建；100% 内部比例同样只调用一次网络。低分辨率下的细轮廓和发光体边缘仍受采样率限制。

OptiX 路径不串联 SVGF。神经状态、前帧图像、内部引导层、AOV 和 scratch 均留在 GPU 并复用，显式申请的显存计入统计；原生材质模式的两次神经调用需要更多时间和显存。切换、重置、切镜、尺寸或显式光照/材质变化时作废历史；连续刚体运动保留重投影并降低历史可信度，以响应阴影变化。关闭 OptiX 降噪时释放其图像和工作缓冲。

全部实时求交均使用 OptiX，降噪器仍可独立选择。神经降噪初始化或调用失败时自动回退 SVGF，保留用户选择并在状态面板说明原因。切回 SVGF 再选 OptiX 可以重新尝试。状态栏显示实际降噪器，显存计数包含神经缓冲。OptiX 模式的 `Filtered` 显示 TAA 前最终神经结果，`Temporal` 显示原始光照合成；分量视图显示内部光照信号（使用 AOV 时是降噪后的信号）。SVGF 的方差和历史诊断在该模式不适用。宽滤波可能损失细高光或纹理，性能和画质需按场景比较。

CLI 配置示例：`{"denoise": true, "denoiser": "optix", "temporal": true}`。`denoiser` 接受 `svgf` 或 `optix`；缺失、未知值或类型不匹配时按 SVGF 读取。选择会随现有会话保存。

`Rendering` 选择模式并显示相应性能统计；`Techniques` 保持同一停靠窗口，按模式切换技术面板。

| 面板 | 控制项 |
|---|---|
| 公共 | 场景、相机、灯光、HDRI、输出比例、曝光、色调映射 |
| OpenGL | NPR、IBL / LTC、阴影 / PCSS、AO、SSR、SSGI |
| RTRT Lighting & Sampling | SER、稳定玻璃采样、Owen Sobol 采样、每帧 SPP、路径深度、轮盘赌起始层、光源样本、直接光、阴影、软阴影、GI、反射、透射 |
| RTRT Denoising | SVGF / OptiX AI 切换、降噪总开关、时域复用；SVGF 的各分量历史、阈值、亮点抑制和空间滤波参数 |
| TAA / Temporal Upscaling | TAA、内部光照比例、原生材质和清晰光学采样、时域上采样、当前帧权重、裁剪、锐化 |
| RTRT Diagnostics | 原始及分量光照、Albedo / Normal / Depth、UV 运动、方差、历史长度、历史拒绝、反应掩码、时域和空间重建结果 |

默认 1 SPP、8 层路径，漫反射 / 镜面 / 透射历史分别为 32 / 16 / 4，空间滤波 5 / 3 / 0 轮。已有会话保存的镜面历史值仍会保留，可在面板调整为 16。`Output scale` 控制输出尺寸，RTRT 的 `Internal resolution` 仅控制内部采样尺寸。降低内部比例时默认使用 TAAU。关闭 SVGF 后仍可单独使用 TAA；检查原始采样可选择 `Raw lighting`。调试视图、曝光和显示变换不会污染光照历史。

会话版本 7 分别保存 `render.opengl` 和 `render.realtime`；旧会话中的交互 Path 自动映射至 RTRT，旧暂停和自动预览状态不再作用于新管线。模式切换保留两套参数。

CLI 可以读取与 `render.realtime` 相同的 JSON 对象：

```json
{
  "low_discrepancy": true,
  "specular_history": 16,
  "samples_per_pixel": 1,
  "max_bounces": 8,
  "internal_scale": 0.5,
  "denoise": true,
  "taa": true,
  "temporal_upscale": true
}
```

使用 `--rtrt-config path.json` 加载。未给出的字段使用默认值，数值会限制到支持范围。旧 `hardware_ray_tracing` / `raster_primary` 字段读取时忽略，保存时不再写出；它们不能恢复已删除的实时软件遍历或光栅首交点。

`1080p RT quality preset` 按钮提供 RTX 5080 / San Miguel 的配置：内部光照 50%、全分辨率材质、8 层路径、全部光照效果、SER、TAA 当前帧权重 0.03 / 裁剪 3。等价 CLI 配置为 `benchmarks/cases/rtrt_san_miguel_1080p.json`。按钮不改变窗口尺寸；测试时应设置输出 1920×1080。普通默认设置继续保留 100% 内部光照。

完整帧验收使用 `--frame-report report.json --warmup-frames 60 --camera-preset benchmarks/cases/san_miguel_first_scene.json --camera-motion --frames 3600`。报告计时直到呈现后的 `glFinish`，包含 GPU 完成和 UI；正常运行不为报告同步。`--camera-motion` 重复 600 帧横移、300 帧静止。详细数据与限制见 [复杂场景 60 FPS 验证](rtrt-60fps-validation-2026-09-12.md)。

上一轮配置在 RTX 5080 的 3540 帧有效样本中平均 69.5 FPS，完整帧 P95 15.399 ms、P99 15.756 ms，超出 16.667 ms 的帧数为 0。镜面和玻璃修复后的重新验收见 [firefly 验证](rtrt-firefly-validation-2026-09-13.md)；两轮结论均限于记录中的场景、配置和相机轨迹。

## 验证

`realtime_tests` 包含常量保持、尺寸与缓冲复用、jitter、斜平面运动、刚体运动、切镜、遮挡显露、灯光响应、移动阴影响应、四类光源阴影开关、离屏镜面、玻璃、Alpha、粗糙反射误反应与低频帧差及 Cornell 高采样对照。设置 `RTRT_VALIDATION_DIR` 后会输出 PNG、线性 HDR PFM 和 JSON 指标：

```powershell
$env:RTRT_VALIDATION_DIR = "$PWD/build/rtrt-validation"
.\build\default\bin\realtime_tests.exe
ctest --preset default-release
```

`RTRT_OPTIX_VALIDATE=1` 可启用 OptiX API、栈/追踪深度检查和完整调试信息；该模式按 SDK 要求关闭 OptiX 编译优化，不应用于性能测量。正常模式保留等级 3 优化。2026-09-24 完整实时套件在 Compute Sanitizer memcheck 下 25 项通过、0 错误、0 字节泄漏，3 项可选长测跳过；命令、范围和空场景修复见 [迁移验证](rtrt-optix-migration-2026-09-23.md#内存与资源检查)。

另外设置 `RTRT_PERFORMANCE=1` 可启用 960×540 原生/50% 内部比例、静止/横移四组计时，各预热 24 帧并记录 96 帧，输出 `performance.json` 的 P50/P95/P99。该路径同步 GPU 读取计时，不包含窗口呈现，不应直接换算为 Viewer FPS；默认回归跳过这项性能测试。

`optix_interop_tests` 验证 OptiX 首交点共享边覆盖、CUDA/OpenGL 与主机呈现一致、驻留缓冲复用、零整帧下载，以及实时路径不构建软件 BVH。`realtime_tests` 额外验证三角形/解析球混合资产、材质分类更新、镜像非均匀变换和 GAS/IAS 复用。设置 `RTRT_REQUIRE_OPTIX=1` 使 OptiX 不可用成为失败；设置 `RTRT_REQUIRE_OPTIX_DENOISER=1` 严格要求神经降噪可用。

首版没有为反射表面跟踪次级对象运动，也没有针对蒙皮变形建立上一帧顶点流；这些变化主要依靠命中距离、反应掩码和较短历史处理。快速变化的镜面与透明覆盖仍可能闪烁，低内部分辨率也会损失细节。性能和质量结果应结合分辨率、场景及 GPU 一起评估。
