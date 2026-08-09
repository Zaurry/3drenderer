# CUDA Path 资产 BLAS + 实例 TLAS 拖动结果

> 历史记录：本文数值保持原样，但当时使用的 `cuda_path_benchmark` 已删除。
> 当前 benchmark 架构和扩展 phase 的规则见
> [规范化 Benchmark 流程](../benchmarking.md)。

日期：2026-07-30

分支：`codex/gpu-wavefront-nee-mis`

设备：RTX 5080，CUDA 13.3，Release，输出 2418×1343。

## 结论

CUDA Viewer 不再在 Gizmo/Inspector/父组拖动的每一帧展开对象、变换并上传全部
三角形。每个唯一 asset 构建一份局部 BVH4 BLAS；可见对象使用实例矩阵、resolved
material table 与小型 TLAS。拖动帧只更新 instance buffer 并执行 GPU bottom-up
TLAS refit，同时继续显示 CUDA Path 的动态分辨率 2-bounce 预览。

移动时的大颗粒仍来自“场景变化后重置为约 1 spp + 25%/50% 动态分辨率放大”；
停止后恢复原生分辨率、64 bounce 与完整 sweep 渐进累积。它不是 OptiX/OIDN
后处理降噪。

## RTX 5080 结果

固化相机、Sun、动态预览规则与上一轮重场景基准相同。五次原生 full-frame trace
中位数：

| 场景 | 改造前 | 实例化后 | 变化 |
|---|---:|---:|---:|
| Sponza | 52.976 ms | 55.186 ms | +4.17% |
| San Miguel Low Poly | 93.000 ms | 92.056 ms | -1.02% |

连续拖动 30 帧，排除首次 warm-up 帧：

| 场景 | wall 中位数 | wall P95 | instance upload 中位数 / P95 | TLAS refit 中位数 / P95 |
|---|---:|---:|---:|---:|
| Sponza | 14.980 ms | 15.592 ms | 0.110 / 0.202 ms | 0.0046 / 0.0064 ms |
| San Miguel Low Poly | 11.130 ms | 11.698 ms | 0.108 / 0.126 ms | 0.0046 / 0.0061 ms |

两者均满足 UI 中位不高于 16.7 ms、P95 不高于 20 ms，以及 instance update +
TLAS refit 中位低于 1 ms、P95 低于 2 ms。30 个拖动帧中 BLAS build count 保持
为 2（floor 与主资产），没有新增 geometry/material/texture upload，也没有
framebuffer 扫描发布。Sponza 原生 trace 的 +4.17% 仍在不超过 5% 的回退上限内；
San Miguel 没有回退。

这些命令属于已退休的历史工具。当前标准档包含相机交互 phase；若要重新建立
对象拖动基线，应按新框架增加 `instance_drag` phase 和版本化 case，而不能复用
相机交互指标名。

原始记录位于：

- `build/validation/sponza-instanced-full.txt`
- `build/validation/sponza-instanced-drag.txt`
- `build/validation/san-miguel-instanced-full.txt`
- `build/validation/san-miguel-instanced-drag.txt`

## 合约验证

新增测试覆盖：

- 多实例共享一个 asset/BLAS、每实例材质 override。
- 平移、旋转、负缩放与非均匀缩放下，实例 CUDA 与扁平 CUDA 逐像素对照。
- 连续 100 帧拖动时 BLAS、geometry/material/texture upload 不增加，TLAS 每帧
  refit。
- 复制/删除已有 asset 实例只重建 TLAS，不重建 BLAS。
- emissive 三角形非均匀变换的 NEE/MIS 固定 SPP 能量偏差不超过 1%。
- 非均匀变换球体的表面 Jacobian、alpha-cutout emissive、直接命中与 CDF 更新。
- preview 样本与静止 accumulation 隔离、完整 sweep 原子发布以及 interop
  `downloads=0` 继续沿用原有合约。

`default`、`cpu`、`cuda` 三套 Release 构建和 CTest 均通过。

Windows Compute Sanitizer 对 conditional graph 的 binary patching 仍使用项目既有的
`RENDERER_CUDA_SANITIZER_FALLBACK=1` 等价固定拓扑路径。专用 instancing smoke
覆盖上述 BLAS/TLAS refit、负/非均匀变换、球体、alpha 与 emissive CDF，结果为：

```text
renderer_tests: CUDA instancing sanitizer smoke passed
========= ERROR SUMMARY: 0 errors
```

命令：

```powershell
cmake --preset cuda -B build/cuda-memcheck `
  -DCMAKE_CUDA_FLAGS="-DRENDERER_CUDA_SANITIZER_FALLBACK=1 -lineinfo"
cmake --build build/cuda-memcheck --config Release --target renderer_tests
compute-sanitizer --tool memcheck --error-exitcode 99 `
  .\build\cuda-memcheck\bin\renderer_tests.exe --cuda-instancing-sanitizer
```
