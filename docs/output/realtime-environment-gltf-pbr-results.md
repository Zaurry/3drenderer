# 实时环境光、glTF 与统一 PBR：验证及性能结果

## 结论

本次改动把环境光从“未命中射线返回常量颜色”升级为可编辑、可采样的 2:1 HDRI，
并把静态 glTF 2.0/GLB 与 metallic-roughness PBR 接入 OpenGL、CPU Path 和 CUDA
Path。基线不能加载 HDRI 或 glTF PBR，因此性能对比使用双方都支持的内置
Cornell Box；真实 HDR/EXR/GLB 用于当前版本的功能和跨后端验证。

固定工作量下，通用 PBR/GGX 与环境采样能力带来的代价为：CPU Path 渲染时间
增加 21.1%，CUDA Path 增加 7.6%。在基线/当前交替运行的 OpenGL 测试中，
20,000 帧总进程时间增加 5.6%，扣除启动后估算稳态增加 2.5%。加载一张
1024x512 HDRI 的 OpenGL IBL 预处理增加约 0.612 秒一次性启动时间；预处理
完成后，HDRI 相对当前无 HDRI 的稳态差异约 +0.5%，接近本次 wall-time 噪声。

## 测试环境

- 日期：2026-08-02
- 基线：`18d7503`
- 当前分支工作树：`codex/realtime-environment-gltf-pbr`
- CPU：AMD Ryzen 7 9800X3D 8-Core Processor
- GPU：NVIDIA GeForce RTX 5080
- 驱动：610.62
- CUDA Toolkit：13.3（nvcc 13.3.73）
- 构建：MSVC Release，`RENDERER_NATIVE_ARCH=OFF`

## 基线对比

离线 Path 使用 1280x720、64 spp 的内置 Cornell Box。每个后端在基线与当前
工作树之间交替运行五次，表中为 renderer 自报 trace time 的中位数。

| 后端 | 基线 | 当前 | 时间变化 | 吞吐变化 |
|---|---:|---:|---:|---:|
| CPU Path | 2.25890 s | 2.73480 s | +21.1% | -17.4% |
| CUDA Path | 0.180921 s | 0.194658 s | +7.6% | -7.1% |

原始样本：

- CPU 基线：2.24654、2.24771、2.25890、2.31687、2.26886 秒；
- CPU 当前：2.73480、2.74822、2.75055、2.73441、2.70971 秒；
- CUDA 基线：0.184999、0.180038、0.178911、0.182371、0.180921 秒；
- CUDA 当前：0.195766、0.203025、0.192244、0.194658、0.191775 秒。

复现命令：

```powershell
.\build\default\bin\renderer.exe `
  --mode path --scene cornell_box `
  --width 1280 --height 720 --spp 64 `
  --path-backend cpu --output output\bench-cpu.png

.\build\default\bin\renderer.exe `
  --mode path --scene cornell_box `
  --width 1280 --height 720 --spp 64 `
  --path-backend cuda --output output\bench-cuda.png
```

CPU 的主要新增成本是所有 Diffuse/Metal/PBR bounce 统一经过 visible-GGX
sample/pdf/evaluate，且启用环境时会增加一次环境 NEE 可见性查询。CUDA 保留
Wavefront、持久缓冲和 CUDA Graph，新增材质属性、环境混合 NEE/MIS 与纹理访问，
所以相对增幅较小。

## OpenGL 与 HDRI 预处理

OpenGL 使用内置场景、320x240、20,000 帧，各运行三次，以 PowerShell
`Stopwatch` 记录整个进程的 wall time。基线是在独立 worktree 中从 `18d7503`
重新生成的 CPU-only Release Viewer；两组测试均按“参照、对照”交替运行。

| 交替测试组 | 参照中位数 | 对照中位数 | 总进程变化 |
|---|---:|---:|---:|
| 基线无 HDRI → 当前无 HDRI | 9.0541 s | 9.5651 s | +5.6% |
| 当前无 HDRI → 当前 1024x512 HDRI | 9.7678 s | 10.4260 s | +6.7% |

对应的单帧启动中位数分别为：第一组 0.3337 / 0.6224 秒，第二组
0.6204 / 1.2322 秒。由此估算：

- 当前 BRDF LUT/IBL 基础初始化比基线多约 0.289 秒；
- 1024x512 HDRI 的 cubemap 与 GGX mip chain 预处理再增加约 0.612 秒；
- 扣除各自启动时间后，当前无 HDRI 稳态比基线慢约 2.5%；
- 扣除各自启动时间后，当前 HDRI 比当前无 HDRI 慢约 0.5%，接近本次
  wall-time 噪声，未观察到显著的额外逐帧成本。

OpenGL 的实时结果是确定性的 Split-Sum IBL，不承担 Path 后端的随机环境阴影光线。
更换 HDRI 会重新执行 load-time、memory-only 预处理；当前没有磁盘缓存。

## CUDA Viewer 快照

640x360、200 帧的一次 Viewer 统计用于辅助观察，不作为主基准：

| 配置 | `trace_ms` | 备注 |
|---|---:|---|
| 基线、无 HDRI | 2.35546 ms | interop active |
| 当前、无 HDRI | 2.56013 ms | 比基线 +8.7% |
| 当前、1024x512 HDRI | 2.85850 ms | 比当前无 HDRI +11.7% |

三次都保持 CUDA/OpenGL interop；当前真实 glTF + HDRI 冒烟报告
`allocations=25`、`downloads=0`。短 Viewer 统计会受预热、相机和采样状态影响，
应优先采用上面的五次离线中位数。

## 质量变化

基线没有同一 HDRI/glTF 输入路径，不能诚实地给出“噪声降低多少 dB”的
同图数值对比。可以确认的算法与功能改进是：

- HDR/EXR/PNG/JPG 环境图同时驱动背景和照明，并可独立隐藏背景；
- CPU/CUDA 使用 `luminance × texel solid angle` 重要性分布执行环境 NEE，
  与 visible-GGX BSDF 样本做 beta=2 MIS；小而亮的 HDR texel 不再只靠
  BSDF 偶然命中，固定时间下预期显著降低高亮环境造成的方差；
- OpenGL 使用 SH9 漫反射 irradiance、GGX 预过滤 specular cubemap 与 BRDF LUT；
- 三个后端统一 metallic-roughness、Smith masking-shadowing、Fresnel-Schlick
  与 roughness 语义，粗糙金属不再走旧的简化模型；
- glTF 的层级、共享 mesh 实例、PBR 纹理、顶点色/alpha、MASK/BLEND、
  double-sided、sampler、`KHR_texture_transform`、相机和 punctual lights
  可进入编辑器与保存/恢复流程。

## 内存估算

以 1024x512 环境图为例，不计容器与驱动对齐：

- 线性 RGB float texel 约 6 MiB；
- PMF 与 CDF 两个 float 数组合计约 4 MiB；
- CPU 环境表示约 10 MiB；CUDA 上传的 texel/PMF/CDF 另约 10 MiB；
- OpenGL 256 边长 RGBA16F cubemap 完整 mip chain 约 4 MiB，256x256 RG16F
  BRDF LUT 约 0.25 MiB。

环境对象按场景共享，不会按 mesh 或 instance 复制。CPU/CUDA 运行时可能同时保留
host 与 device 副本；OpenGL 还保留 host 环境对象用于重新预处理。

## 验证记录

- `default-release`：构建通过，CTest 1/1；
- `cpu-release`：构建通过，CTest 1/1；
- `cuda-release`：构建通过，CTest 1/1；
- OpenGL：真实 `DamagedHelmet.glb + studio_small_08_1k.hdr`，5 帧，
  `shader=active`；
- CPU/CUDA Path：同一 GLB + HDR，128x128、8 spp，均成功输出且人工检查外观一致；
- EXR：`studio_small_08_1k.exr`，CPU 32x32、1 spp 成功；
- CUDA Viewer：真实 GLB + HDR，`interop=active`、`downloads=0`；
- Compute Sanitizer：使用 `RENDERER_CUDA_SANITIZER_FALLBACK=1 -lineinfo`
  的固定拓扑构建；专用 instancing smoke 与真实 GLB + HDR memcheck 均为
  `ERROR SUMMARY: 0 errors`。

Sanitizer 构建与命令：

```powershell
cmake -S . -B build\cuda-memcheck `
  -DRENDERER_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=120 `
  '-DCMAKE_CUDA_FLAGS=-DRENDERER_CUDA_SANITIZER_FALLBACK=1 -lineinfo'
cmake --build build\cuda-memcheck --config Release -- /nodeReuse:false

compute-sanitizer --tool memcheck --error-exitcode 99 `
  .\build\cuda-memcheck\bin\renderer_tests.exe `
  --cuda-instancing-sanitizer
```

生产构建仍使用 CUDA Graph 条件节点；固定拓扑宏只用于 Sanitizer 插桩，
不是发布路径的性能配置。
