# RTRT 实时光线渲染器

交互 Viewer 提供 `OpenGL / RTRT` 两种模式，快捷键为 `1 / 2`。新会话默认 RTRT；CUDA 不可用时回退 OpenGL，并在状态栏说明原因。`--mode path` 是 RTRT 的兼容别名。离线 `renderer` 仍使用高采样路径追踪，便于输出参考图像。

```powershell
.\build\cuda-ninja\bin\viewer.exe --scene builtin --mode rtrt --no-restore-last
.\build\cuda-ninja\bin\viewer.exe --scene builtin --mode rtrt --frames 300 --no-restore-last --capture build/rtrt.png
```

## 管线

每帧发布完整图像：场景增量上传 → CUDA 首交点 G-buffer → 低 SPP 路径采样 → SVGF → 合成 → TAA / TAAU → 曝光、色调映射与 sRGB 显示。

CUDA 场景、BLAS/TLAS、材质、GGX / Kulla–Conty、纹理、NEE / MIS 位于共享的 `cuda_scene.cuh`。交互 RTRT 使用逐像素路径内核；离线参考继续使用原来的 wavefront 队列。首版使用 CUDA 软件 BVH 遍历，没有调用 RTX 硬件光追、OptiX 或神经降噪 SDK。

支持点光、聚光、太阳、矩形面积光、发光几何和环境光，以及多次反弹、离屏反射、玻璃和 Alpha Mask / Blend。点光和聚光的半径用于圆盘可见性采样，保留原有中心强度与范围衰减语义；太阳角半径用于方向采样。灯光的 `casts_shadows` 控制对应的直接阴影。

SVGF 将直接漫反射、间接漫反射、镜面和透射信号分别维护；直接可见发光和背景单独合成。漫反射先解调材质，再重投影，累积颜色、一二阶矩与历史长度，以短历史邻域估计方差，执行逐轮扩大步长的 à-trous。滤波使用独立读写缓冲，第一轮结果反馈颜色历史。镜面使用粗糙度、次级命中距离和较短历史；理想镜面、玻璃透射跳过宽空间核。

运动向量为“当前到历史”的 UV 位移。相机和刚体的连续运动按稳定对象 ID、上一帧变换重投影；逐个验证历史插值样本的深度、法线和表面身份。输入 jitter 单独补偿，最终 TAA 历史位于不抖动的输出网格。镜头切换、输出或内部尺寸变化、场景替换和不兼容几何变化清空历史；灯光、材质和遮挡变化使用历史裁剪与反应掩码更新。

正常互操作路径将结果直接写入 CUDA–OpenGL 纹理，整帧下载计数应为零。GPU 计时异步读取，帧缓冲持续复用。`--no-cuda-interop` 可强制使用主机呈现路径，以诊断互操作兼容性；它每帧下载完整图像，速度会降低。

## ImGui 与参数保存

`Rendering` 选择模式并显示相应性能统计；`Techniques` 保持同一停靠窗口，按模式切换技术面板。

| 面板 | 控制项 |
|---|---|
| 公共 | 场景、相机、灯光、HDRI、输出比例、曝光、色调映射 |
| OpenGL | NPR、IBL / LTC、阴影 / PCSS、AO、SSR、SSGI |
| RTRT Lighting & Sampling | 每帧 SPP、路径深度、轮盘赌起始层、光源样本、直接光、阴影、软阴影、GI、反射、透射 |
| SVGF Denoising | 降噪、时域复用、各分量历史、深度和法线阈值、反应强度、历史钳制、亮点抑制、滤波轮数和权重 |
| TAA / Temporal Upscaling | TAA、内部采样比例、时域上采样、当前帧权重、裁剪、锐化 |
| RTRT Diagnostics | 原始及分量光照、Albedo / Normal / Depth、UV 运动、方差、历史长度、历史拒绝、反应掩码、时域和空间重建结果 |

默认 1 SPP、8 层路径，漫反射 / 镜面 / 透射历史分别为 32 / 8 / 4，空间滤波 5 / 3 / 0 轮。`Output scale` 控制输出尺寸，RTRT 的 `Internal resolution` 仅控制内部采样尺寸。降低内部比例时默认使用 TAAU。关闭 SVGF 后仍可单独使用 TAA；检查原始采样可选择 `Raw lighting`。调试视图、曝光和显示变换不会污染光照历史。

会话版本 6 分别保存 `render.opengl` 和 `render.realtime`；旧会话中的交互 Path 自动映射至 RTRT，旧暂停和自动预览状态不再作用于新管线。模式切换保留两套参数。

CLI 可以读取与 `render.realtime` 相同的 JSON 对象：

```json
{
  "samples_per_pixel": 1,
  "max_bounces": 8,
  "internal_scale": 0.5,
  "denoise": true,
  "taa": true,
  "temporal_upscale": true
}
```

使用 `--rtrt-config path.json` 加载。未给出的字段使用默认值，数值会限制到支持范围。

## 验证

`realtime_tests` 包含常量保持、尺寸与缓冲复用、jitter、斜平面运动、刚体运动、切镜、遮挡显露、灯光响应、四类光源阴影开关、离屏镜面、玻璃、Alpha 及 Cornell 高采样对照。设置 `RTRT_VALIDATION_DIR` 后会输出 PNG、线性 HDR PFM 和 JSON 指标：

```powershell
$env:RTRT_VALIDATION_DIR = "$PWD/build/rtrt-validation"
.\build\cuda-ninja\bin\realtime_tests.exe
ctest --test-dir build/cuda-ninja --output-on-failure
```

首版没有为反射表面跟踪次级对象运动，也没有针对蒙皮变形建立上一帧顶点流；这些变化主要依靠命中距离、反应掩码和较短历史处理。快速变化的镜面与透明覆盖仍可能闪烁，低内部分辨率也会损失细节。性能和质量结果应结合分辨率、场景及 GPU 一起评估。
