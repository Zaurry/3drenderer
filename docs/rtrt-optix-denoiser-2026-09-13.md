# RTRT OptiX AI 降噪集成验证

## 使用

打开 `Techniques → RTRT Denoising → Denoiser`，选择 `SVGF` 或 `OptiX AI`。默认和旧会话保持 SVGF。`Enable denoising` 是总开关，`Temporal reuse` 在神经时域和单帧模式之间切换。选择随现有会话保存。

CLI 的 `--rtrt-config` 接受 `{"denoise":true,"denoiser":"optix","temporal":true}`。无 OptiX 支持时自动回退 SVGF，状态栏显示实际后端和原因。降噪与硬件光追开关独立。

集成使用已固定的 [NVIDIA OptiX 9.1.0 头文件](https://github.com/NVIDIA/optix-dev/blob/f1f6dd803f3159992d248178f6e09421c6eb8b6d/include/optix_types.h)中的 AOV / TEMPORAL_AOV 模型，神经模型由驱动提供。输入是线性 HDR、反照率、世界空间法线、像素运动向量及运动可信度。实现细节见 [RTRT 文档](rtrt-renderer.md)。

100% 内部比例执行一次 beauty 降噪。低内部比例且启用原生材质时，先对内部 beauty 和四个光照分量降噪，再重建原生材质，最后为原生玻璃/镜面执行独立降噪；第二次输出只应用于光学像素。这样可保留纹理，也避免把已放大的相关噪声直接作为网络输入。

## 环境与功能回归

- Windows，RTX 5080，NVIDIA 驱动 616.64，CUDA Toolkit 13.3，Release。
- 默认 CUDA / OptiX Viewer 构建成功；`RENDERER_OPTIX=OFF` 的 CUDA 构建及无 CUDA Viewer 构建成功。
- 完整 `realtime_tests`：**22 通过，0 失败，4 跳过**。其中三个是按环境变量启用的性能/复杂场景/压力测试；另一个是需要无 OptiX 构建的回退测试。
- 无 OptiX 构建：设置序列化及回退一致性 **2 通过**。选择 OptiX 后的回退输出与直接选择 SVGF 逐像素一致。
- CUDA 与无 CUDA 的会话保存/恢复检查均通过，保存的 `optix` 选择能够恢复；无 CUDA 的设置清洗/序列化检查通过。
- 新增 GPU 回归覆盖单帧/时域、关闭/切换、异步驻留帧、固定尺寸缓冲复用、37×23 非整齐尺寸、内部比例、原生材质、相机/刚体运动、显露区域、材质变更、重置，以及原生/半分辨率的玻璃和镜面。

日志：[完整回归](../build/optix-denoiser-final-tests.log)、[无 OptiX 回退](../build/optix-off-complete-tests.log)。构建日志分别为 [默认 Viewer](../build/optix-denoiser-final-viewer-build.log)、[无 OptiX](../build/optix-off-complete-build.log)、[无 CUDA](../build/optix-denoiser-final-no-cuda-build.log)。

## 画质检查

96×96 Cornell，512 SPP 离线参考，实时 1 SPP、16 帧。关闭输出 TAA/TAAU，单独观察神经降噪；比较最后 8 帧的平均 HDR RGB MSE。原始输入与降噪输出使用相同内部比例和材质重建设置。表面区域为图像 `x=10..85, y=24..85`，与包含高亮发光体边缘的整图指标分别记录。

| 模式 | 整图原始 MSE | 整图降噪 MSE | 表面原始 MSE | 表面降噪 MSE |
|---|---:|---:|---:|---:|
| 原生，单帧 | 0.085524 | 0.023316 | 0.062950 | 0.000382 |
| 原生，时域 | 0.085524 | 0.024979 | 0.062950 | 0.000387 |
| 50% 光照＋原生材质/光学 | 0.086343 | 0.061356 | 0.022555 | 0.000767 |
| 50% 光照，普通上采样 | 0.144035 | 0.144500 | 0.021359 | 0.000831 |

最后一种配置的整图误差略高于原始输入，表面噪声则明显降低；高亮发光体和细轮廓的低分辨率误差仍然存在。该小场景中，时域模式也没有在所有指标上优于单帧模式。用户可按具体场景切换降噪器与内部比例。

数据：[完整画质 JSON](../build/optix-denoiser-validation/optix-quality.json)。图像：[原生时域](../build/optix-denoiser-validation/optix-cornell-1.png)、[50%＋原生材质](../build/optix-denoiser-validation/optix-cornell-2.png)、[512 SPP 参考](../build/optix-denoiser-validation/optix-reference-512spp.png)。每张 PNG 旁有线性 PFM。

## Viewer 呈现与内存

960×540 builtin 场景，每配置运行 90 帧，预热 15 帧后记录 75 帧；输出 TAA 开启。所有记录均为 `denoiser=optix`、硬件 RT 开启、GPU 互操作开启、整帧下载数为 0，历史重置次数保持 1。

| 内部光照 | 完整帧 P95 | 神经缓冲 | 总帧缓冲 |
|---|---:|---:|---:|
| 960×540 | 6.136 ms | 188.731 MiB | 621.812 MiB |
| 480×270＋原生材质/光学 | 7.339 ms | 264.217 MiB | 476.802 MiB |

以上描述 builtin 场景的短时呈现检查。半分辨率原生材质模式增加一次神经推理，因此在此简单场景中更慢；复杂场景的路径采样成本不同。显存统计为应用显式申请的缓冲，不包含驱动内部开销。

报告：[原生](../build/optix-denoiser-validation/viewer-optix-report.json)、[半分辨率](../build/optix-denoiser-validation/viewer-optix-half-report.json)。已人工检查 [原生截图](../build/optix-denoiser-validation/viewer-optix.png) 和 [半分辨率截图](../build/optix-denoiser-validation/viewer-optix-half.png)。

启用 `RTRT_OPTIX_VALIDATE=1`，通过 Compute Sanitizer memcheck 和完整泄漏检查运行切换/缩放、玻璃/镜面、场景变化三个用例：**3 通过，0 内存错误，0 字节泄漏**。结果见 [memcheck 日志](../build/optix-denoiser-memcheck.log)。

`build/` 中的日志、JSON 和图像均为本地验证产物，不随源码提交。
