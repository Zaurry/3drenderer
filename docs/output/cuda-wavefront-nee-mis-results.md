# CUDA Wavefront Path Tracer + NEE/MIS 结果

日期：2026-07-29

分支：`codex/gpu-wavefront-nee-mis`

基线提交：`881d30e` (`perf(cuda): optimize path rendering pipeline`)

## 范围与结论

本轮只修改 CUDA Path。CPU Path、`RenderSettings`、CLI、离线渲染、渐进累积、
CUDA/OpenGL interop 和 framebuffer fallback 的调用方式保持不变。

CUDA Path 已从单个 megakernel 改为 CUDA Graph 驱动的 Wavefront 管线，并为
emissive 三角形与球体加入 NEE/MIS。这里的降噪是 Monte Carlo 方差降低，没有
加入 OptiX、OIDN 或其他后处理降噪器。

验收结论：

- Cornell 固定时间 luminance RMSE 为基线的 **37.95%**，通过不高于 50% 的目标。
- Sponza 960×540×16 spp 五次中位数回退 **10.71%**，通过不超过 15% 的上限。
- `default`、`cpu`、`cuda` 三套 Release 构建与 CTest 通过。
- Compute Sanitizer memcheck 为 `ERROR SUMMARY: 0 errors`。
- CUDA/OpenGL interop 120 帧 smoke test 为 `interop=active`、`downloads=0`。

## 实现

每个样本的设备阶段为：

```text
primary ray
  -> compact intersection
  -> shade / scatter
  -> emissive-light sampling (conditional)
  -> direct-light visibility
  -> active / next queue advance
  -> accumulate / resolve
```

- active/next path queue 双缓冲；存活路径使用 warp 聚合原子操作紧凑入队。
- path state 保存 ray、throughput、pixel/RNG 索引、上次 BSDF PDF 与 delta 标记。
- hit queue 只保存 `t / primitive kind / primitive id / barycentric u/v`。
- shading、shadow task、计数器和 graph 参数位于按 framebuffer 容量增长的一块
  device arena；容量不增长时不重分配或重捕获 graph。
- 外层 conditional graph 批处理离线 spp；内层 conditional graph 最多执行 64
  bounce。所有 stage 读取设备端 queue count，不做逐 bounce CPU 同步。
- 无 emissive geometry 时设备端 conditional node 跳过面积光采样子图。
- 队列溢出和非法计数写入显式错误码，在同步或下一次已完成 stream 查询时报错。

### Emissive light distribution

场景上传收集有效、非退化且 emission 非零的三角形与球体。权重为：

```text
area * luminance(emission) * emitting-side-count
```

三角形均匀采样面积，球体均匀采样表面积。面积 PDF 转为立体角 PDF；diffuse
BSDF 为 `base_color / pi`，采样 PDF 为 `cos(theta) / pi`。NEE 与 BSDF 命中
使用 beta=2 power heuristic：

```text
w(a, b) = a^2 / (a^2 + b^2)
```

主射线或 delta 路径直接命中 emissive 时权重为 1。metal/dielectric 仍按 delta
事件处理，本轮没有引入 GGX PDF。点光与方向光继续逐灯确定性求和并使用
alpha-aware shadow ray，不参与 MIS。

CDF 在 `Geometry`、`MaterialBindings` 或 `Materials` 变更时重建；相机和显式
灯光变化不会重建 BVH。

## 合约测试

CUDA 测试覆盖：

- 无 emissive geometry 与常量环境 miss。
- emissive triangle / sphere。
- 单面、双面、遮挡、alpha cutout 和退化发光 primitive。
- 主射线直接命中、diffuse 路径命中和 metal/dielectric delta 行为。
- 显式点光/方向光的确定性求和与阴影。
- Russian roulette 能量均值、NaN/Inf 检查。
- 同 seed 重复渲染稳定。
- 场景增量上传、相机/累积 reset、resize、离线下载、interop 零下载和 CPU-only
  CUDA stub。

## Cornell 固定时间质量

设备：RTX 5080，CUDA 13.3。分辨率 512×512。reference 为新实现 4096 spp。
PNG 读回后先从 sRGB 还原线性值；mask 排除 reference 黑背景和直接可见的饱和
灯面，再计算 Rec.709 luminance RMSE。

| 项目 | spp | 五次时间中位数 | Luminance RMSE |
| --- | ---: | ---: | ---: |
| 基线 `881d30e` | 64 | 0.063345 s | 0.134202 |
| Wavefront + NEE/MIS | 16 | 0.063912 s | 0.050935 |

新实现时间比基线高 0.9%，RMSE 比值为 **0.379541**。

代表输出：

- `build/validation/cornell-reference-4096.png`
- `build/validation/cornell-baseline-64-1.png`
- `build/validation/cornell-wavefront-16-1.png`

## Sponza 吞吐

场景：`Computer Graphics Archive/sponza/sponza.obj`，960×540，16 spp。基线与
当前二进制交替执行，记录端到端 `RenderResult.seconds`。

| 实现 | 五次时间（秒） | 中位数 |
| --- | --- | ---: |
| 基线 `881d30e` | 0.112185, 0.107277, 0.122888, 0.111693, 0.110517 | 0.111693 |
| Wavefront + NEE/MIS | 0.126543, 0.132696, 0.123651, 0.123335, 0.121372 | 0.123651 |

中位数回退 **10.71%**，低于 15% 上限。

## Kernel / graph 时序

Nsight Compute 2026.2.1，Cornell 512×512×1 spp，首 bounce：

| Stage | 时间 | Registers/thread |
| --- | ---: | ---: |
| primary | 45.664 us | 30 |
| intersection | 46.176 us | 62 |
| shade/scatter | 101.088 us | 80 |
| emissive sampling | 51.488 us | 72 |
| direct visibility | 63.936 us | 92 |
| queue advance | 2.432 us | 24 |

shade/scatter 与 emissive sampling 分开，避免无面积光场景承担面积采样的寄存器
压力。最终 block size 为 128 threads。

## 显存

WDDM 驱动下 `nvidia-smi --query-compute-apps=used_gpu_memory` 对 viewer 进程返回
`N/A`，无法给出可靠的 per-process 驱动总量。核心 framebuffer 相关 device
容量可由布局精确复算：

```text
wavefront arena: 244 bytes/pixel
accumulation:     12 bytes/pixel
RNG state:        16 bytes/pixel
```

960×540 interop 模式合计约 **134.5 MiB**，不含场景、CUDA Graph/driver 和
OpenGL texture；interop 活跃时不分配常驻 resolve buffer。

## 验证命令

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
  --path-backend cuda --width 320 --height 240 --frames 120 `
  --no-restore-last
```

当前 Windows Compute Sanitizer 与 conditional graph 组合在 binary patching
阶段返回 `cudaErrorUnknown`，因此 memcheck 使用只由编译宏启用的等价 64-bounce
固定拓扑提交路径；产品 Release 仍使用 conditional graph：

```powershell
cmake --preset cuda -B build/cuda-memcheck `
  -DCMAKE_CUDA_FLAGS="-DRENDERER_CUDA_SANITIZER_FALLBACK=1 -lineinfo"
cmake --build build/cuda-memcheck --config Release --target renderer

compute-sanitizer --tool memcheck --error-exitcode 99 `
  .\build\cuda-memcheck\bin\renderer.exe --mode path `
  --scene cornell_box --width 64 --height 64 --spp 2 `
  --path-backend cuda `
  --output build\validation\compute-sanitizer-wavefront-64x64.png
```

结果：`ERROR SUMMARY: 0 errors`。
