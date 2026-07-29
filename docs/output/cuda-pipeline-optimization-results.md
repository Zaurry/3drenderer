# CUDA Path 渲染链路优化结果

日期：2026-07-29

## 范围

本轮删除 CPU 软件光栅化与 Whitted 光追，只保留 OpenGL/GLSL 光栅化和
CPU/CUDA Path。优化仅涉及渲染链路、资源生命周期、数据布局与工程边界，
没有修改采样、材质、光照、Russian roulette 或 BVH 构建算法。

对照程序由改动前的 `HEAD` 源码归档到独立构建目录后以 Release 配置编译；
当前程序使用同一编译器、CUDA 工具链、驱动与资产。测试设备为 RTX 5080，
CUDA 13.3，Nsight Compute 2026.2.1，Nsight Systems 2026.1.3。

## GPU kernel

Cornell Box，512×512，1 spp，Nsight Compute `basic` 指标：

| 指标 | 改动前 | 改动后 | 变化 |
| --- | ---: | ---: | ---: |
| Registers/thread | 116 | 104 | -10.3% |
| Achieved occupancy | 23.55% | 26.44% | +2.89 pp |
| Kernel duration | 476.93 µs | 357.28 µs | -25.1% |
| Block size | 256 | 64 | - |

最终配置使用 64 threads/block。128 和 256 threads/block 都做过实测，64
在 Cornell 与三角形密集场景上同时更快。

Sponza，960×540，16 spp，Nsight Systems 中 16 次
`render_sample_kernel` 的中位数：

| 改动前 | 改动后 | 变化 |
| ---: | ---: | ---: |
| 716.6975 µs | 600.6290 µs | -16.2% |

该结果超过计划要求的三角形密集场景至少 10% trace 改善门槛。

## 端到端与回退检查

Sponza CUDA 离线渲染五次：

- 改动前：0.129977、0.124840、0.124201、0.126037、0.123869 秒，
  中位数 0.124840 秒。
- 改动后：0.111071、0.109126、0.108792、0.108553、0.107245 秒，
  中位数 0.108792 秒。
- 中位数改善 12.9%。

非目标链路的五次中位数回退均低于 5%：

| 链路 | 改动前 | 改动后 | 变化 |
| --- | ---: | ---: | ---: |
| CPU Path，Cornell 320×180×8，单线程 | 0.187027 s | 0.190678 s | +2.0% |
| OpenGL，320×240，20,000 帧 wall time | 3.613245 s | 3.691219 s | +2.2% |

## 资源与显示链路

内置场景、320×240、CUDA Path、20,000 帧：

| 指标 | 改动前 | 改动后 |
| --- | ---: | ---: |
| `nvidia-smi` 进程期间显存增量 | 408 MiB | 277 MiB |
| CUDA–OpenGL interop | active | active |
| 当前实现分配代次 | - | 7 |
| Framebuffer downloads | - | 0 |
| 初始 upload event | - | 0.302944 ms |
| 稳态 trace event | - | 0.092256 ms |

显存增量下降约 32.1%。Interop 活跃时 kernel 直接写入 `GL_RGBA32F`
surface，没有常驻 display buffer，也没有 framebuffer 下载。离线输出或
interop fallback 才延迟创建 resolve buffer，并复用 host staging。

资源计数测试还验证：

- 同尺寸相机复位不产生新分配或场景上传。
- 灯光编辑只增加灯光上传字节。
- 材质编辑只更新材质及 primitive-material binding，不上传 BVH。
- 几何/拓扑变化才重建并上传 BVH。

## 验证

通过的验证包括：

- `default`、`cpu`、`cuda` 三套 Release configure/build/CTest。
- `RENDERER_BUILD_VIEWER=OFF`、`RENDERER_CUDA=OFF` configure/build/CTest。
- CLI 默认 Path、显式 CPU/CUDA/Auto，以及 `raster`/`ray` 拒绝。
- Viewer 默认 OpenGL、CPU Path 回退、CUDA–OpenGL 120 帧与 20,000 帧。
- Shader 自动热重载错误注入 smoke test；编译失败时最后有效程序保持 active。
- Compute Sanitizer memcheck：`ERROR SUMMARY: 0 errors`。
- CPU/CUDA 图像一致性、alpha、bump、球/三角形、灯光、材质与累积复位测试。

关键命令：

```powershell
cmake --preset default
cmake --build --preset default-release
ctest --preset default-release --output-on-failure

cmake --preset cpu
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure

cmake --preset cuda
cmake --build --preset cuda-release
ctest --preset cuda-release --output-on-failure

.\build\cuda\bin\viewer.exe --scene builtin --mode path `
  --path-backend cuda --width 320 --height 240 --frames 120 --no-restore-last

compute-sanitizer --tool memcheck --error-exitcode 99 `
  .\build\cuda\bin\renderer.exe --mode path --scene cornell_box `
  --width 64 --height 64 --spp 2 --path-backend cuda `
  --output build\validation\compute-sanitizer.png
```
