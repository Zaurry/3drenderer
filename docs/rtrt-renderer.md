# RTRT 实时光线渲染器

交互 Viewer 提供 `OpenGL / RTRT` 两种模式，快捷键为 `1 / 2`。新会话默认 RTRT；CUDA 不可用时回退 OpenGL，并在状态栏说明原因。`--mode path` 是 RTRT 的兼容别名。离线 `renderer` 仍使用高采样路径追踪，便于输出参考图像。

OptiX AI 降噪集成后的构建、RTRT 回归、Viewer 呈现与内存检查见 [OptiX 降噪验证](rtrt-optix-denoiser-2026-09-13.md)。镜面/玻璃此前的复杂场景长测与 60 FPS 门槛仍以 [2026-09-13 修复记录](rtrt-firefly-validation-2026-09-13.md) 为准，本轮没有重跑 San Miguel 长测。

```powershell
cmake --build --preset default-release
.\build\default\bin\viewer.exe --scene builtin --mode rtrt --no-restore-last
.\build\default\bin\viewer.exe --scene builtin --mode rtrt --frames 300 --no-restore-last --capture build/rtrt.png
```

## 管线

每帧发布完整图像：场景增量上传 → 首交点 G-buffer（OptiX、CUDA 或可选光栅可见性）→ 低 SPP 路径采样 → 可选 SVGF / OptiX AI 降噪与材质合成 → TAA / TAAU → 曝光、色调映射与 sRGB 显示。

`Raster primary visibility` 在 OpenGL 中保存实例、三角形及透视正确的重心坐标，通过 CUDA–GL 互操作直接重建同一材质表面；阴影、GI、反射和透射继续追踪射线。首候选被 Alpha Mask / 背面规则拒绝时，该像素回退完整遍历；原始解析球资产或不可用互操作也回退 CUDA。普通场景中的球体使用共享三角网格。此选项默认关闭：当前测试中节省的首交点时间仍被 GL/CUDA 交接开销抵消，应比较整帧时间后选择。

材质、GGX / Kulla–Conty、纹理、NEE / MIS 继续使用共享的 `cuda_scene.cuh`。默认构建增加 OptiX 9.1 后端，将首交点、续接射线和阴影求交交给 RT Core；其余路径估计和 CUDA 降噪保持共享。三角形按不透明单面、不透明双面、需要材质接受测试分组，透明裁剪仍执行原有规则；镜像实例保留一致的正反面语义。静态网格缓存 GAS，实例变换更新 IAS，材质接受规则变化重新构建分类。

默认启用着色任务重排（SER）：次级射线使用 `optixTraverse → optixReorder → optixInvoke`，按材质提示组织后续着色。首交点与阴影不进行该重排。不支持重排的 OptiX 设备按 SDK 规则执行空操作；可通过面板单独关闭，比较实际场景整帧时间。实现依据 [NVIDIA OptiX API](https://raytracing-docs.nvidia.com/optix9/api/group__optix__device__api.html)。

OptiX 头文件通过固定版本和 SHA256 获取，PTX 在构建时嵌入程序；无需另行安装完整 SDK。驱动初始化不支持 OptiX，或资产包含原始解析球时，保留 CUDA 软件遍历，并显示回退原因。`-DRENDERER_OPTIX=OFF` 可禁用此后端；无 CUDA 构建不下载 OptiX。离线参考继续使用原有 CUDA wavefront 队列。第三方许可随安装包保存在 `share/renderer/licenses`。

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

OptiX 启动参数使用有完成事件保护的固定页内存循环缓冲；错误回传也使用固定页内存，避免普通主机内存的异步复制暗中串行化提交。验证/截图仍可显式下载，正常呈现不读取整帧。

## ImGui 与参数保存

`RTRT Denoising → Denoiser` 可在 `SVGF` 与 `OptiX AI` 间切换，`Enable denoising` 控制总开关，旧会话和默认配置继续使用 SVGF。OptiX 使用 9.1 的 `TEMPORAL_AOV` 神经模型；关闭 `Temporal reuse` 时使用单帧 `AOV` 模型。依赖已固定到 [NVIDIA OptiX 9.1 头文件](https://github.com/NVIDIA/optix-dev/blob/f1f6dd803f3159992d248178f6e09421c6eb8b6d/include/optix_types.h)，不需要单独下载模型，实际模型由兼容 NVIDIA 驱动提供。

OptiX 路径将当前帧线性 HDR 合成图直接送入网络，提供反照率、世界空间法线、前帧到当前帧的像素运动向量，以及依据深度、法线和对象身份计算的运动可信度。运动向量转换同时处理方向、分辨率和两帧采样抖动。玻璃和清晰镜面的首表面运动不能代表次级像，因此保守关闭这些像素的神经历史复用，仍执行空间降噪和可选输出 TAA。背景辐亮度原样保留。

启用 `Native materials / sharp optics` 且内部比例小于 100% 时，先在内部采样分辨率用 beauty + 四个光照 AOV 降噪，分别保留直接漫反射、间接漫反射、反射和透射；随后重建原生材质，并对原生玻璃/镜面补采样进行第二次神经降噪。第二次结果只用于这些光学像素，其余像素保留已重建的材质，避免放大原始噪声后再滤波，也避免重复模糊纹理。关闭原生材质选项时只在内部光照分辨率降噪，再由现有 TAAU 重建；100% 内部比例同样只调用一次网络。低分辨率下的细轮廓和发光体边缘仍受采样率限制。

OptiX 路径不串联 SVGF。神经状态、前帧图像、内部引导层、AOV 和 scratch 均留在 GPU 并复用，显式申请的显存计入统计；原生材质模式的两次神经调用需要更多时间和显存。切换、重置、切镜、尺寸或显式光照/材质变化时作废历史；连续刚体运动保留重投影并降低历史可信度，以响应阴影变化。关闭 OptiX 降噪时释放其图像和工作缓冲。

降噪与 `Hardware ray tracing (OptiX)` 独立；使用 CUDA BVH 或解析球也能神经降噪。构建关闭 `RENDERER_OPTIX`、驱动不兼容或初始化/调用失败时自动回退 SVGF，保留用户选择并在状态面板说明原因。切回 SVGF 再选 OptiX 可以重新尝试。状态栏显示实际降噪器，显存计数包含神经缓冲。OptiX 模式的 `Filtered` 显示 TAA 前最终神经结果，`Temporal` 显示原始光照合成；分量视图显示内部光照信号（使用 AOV 时是降噪后的信号）。SVGF 的方差和历史诊断在该模式不适用。宽滤波可能损失细高光或纹理，性能和画质需按场景比较。

CLI 配置示例：`{"denoise": true, "denoiser": "optix", "temporal": true}`。`denoiser` 接受 `svgf` 或 `optix`；缺失、未知值或类型不匹配时按 SVGF 读取。选择会随现有会话保存。

`Rendering` 选择模式并显示相应性能统计；`Techniques` 保持同一停靠窗口，按模式切换技术面板。

| 面板 | 控制项 |
|---|---|
| 公共 | 场景、相机、灯光、HDRI、输出比例、曝光、色调映射 |
| OpenGL | NPR、IBL / LTC、阴影 / PCSS、AO、SSR、SSGI |
| RTRT Lighting & Sampling | OptiX、SER、光栅首交点、稳定玻璃采样、Owen Sobol 采样、每帧 SPP、路径深度、轮盘赌起始层、光源样本、直接光、阴影、软阴影、GI、反射、透射 |
| RTRT Denoising | SVGF / OptiX AI 切换、降噪总开关、时域复用；SVGF 的各分量历史、阈值、亮点抑制和空间滤波参数 |
| TAA / Temporal Upscaling | TAA、内部光照比例、原生材质和清晰光学采样、时域上采样、当前帧权重、裁剪、锐化 |
| RTRT Diagnostics | 原始及分量光照、Albedo / Normal / Depth、UV 运动、方差、历史长度、历史拒绝、反应掩码、时域和空间重建结果 |

默认 1 SPP、8 层路径，漫反射 / 镜面 / 透射历史分别为 32 / 16 / 4，空间滤波 5 / 3 / 0 轮。已有会话保存的镜面历史值仍会保留，可在面板调整为 16。`Output scale` 控制输出尺寸，RTRT 的 `Internal resolution` 仅控制内部采样尺寸。降低内部比例时默认使用 TAAU。关闭 SVGF 后仍可单独使用 TAA；检查原始采样可选择 `Raw lighting`。调试视图、曝光和显示变换不会污染光照历史。

会话版本 6 分别保存 `render.opengl` 和 `render.realtime`；旧会话中的交互 Path 自动映射至 RTRT，旧暂停和自动预览状态不再作用于新管线。模式切换保留两套参数。

CLI 可以读取与 `render.realtime` 相同的 JSON 对象：

```json
{
  "raster_primary": true,
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

使用 `--rtrt-config path.json` 加载。未给出的字段使用默认值，数值会限制到支持范围。

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

`RTRT_OPTIX_VALIDATE=1` 可启用 OptiX API、栈/追踪深度检查和完整调试信息；该模式按 SDK 要求关闭 OptiX 编译优化，不应用于性能测量。正常模式保留等级 3 优化。当前完整 Compute Sanitizer 检查仍存在未定位异常，具体通过范围和失败日志见复杂场景验证记录。

另外设置 `RTRT_PERFORMANCE=1` 可启用 960×540 原生/50% 内部比例、静止/横移四组计时，各预热 24 帧并记录 96 帧，输出 `performance.json` 的 P50/P95/P99。该路径同步 GPU 读取计时，不包含窗口呈现，不应直接换算为 Viewer FPS；默认回归跳过这项性能测试。

`rtrt_hybrid_tests` 验证材质、Alpha、非均匀镜像变换、运动、切镜、尺寸、几何刷新、共享边覆盖和 GL/CUDA 完整呈现。设置 `RTRT_HYBRID_PERFORMANCE=1` 可启用三场景的光栅/射线对照，每种配置预热 48 帧、测量 300 帧；`RTRT_BENCH_ASSET` 可指定仓库相对路径。它单独记录 GL GPU 时间、CUDA 时间和 CPU 提交加 GPU 等待时间，后者包含输入可见性互操作，但不含 Viewer UI 和窗口呈现。

首版没有为反射表面跟踪次级对象运动，也没有针对蒙皮变形建立上一帧顶点流；这些变化主要依靠命中距离、反应掩码和较短历史处理。快速变化的镜面与透明覆盖仍可能闪烁，低内部分辨率也会损失细节。性能和质量结果应结合分辨率、场景及 GPU 一起评估。
