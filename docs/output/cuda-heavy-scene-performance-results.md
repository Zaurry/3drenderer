# CUDA Path 重场景性能结果

> 历史记录：本文数值保持原样，但当时使用的 `cuda_path_benchmark` 已删除。
> 当前入口、固定 San Miguel case 和结构化结果格式见
> [规范化 Benchmark 流程](../benchmarking.md)。

日期：2026-07-30

分支：`codex/gpu-wavefront-nee-mis`

设备：RTX 5080，CUDA 13.3，Release，输出 2418×1343。

## 结论

本轮只改变 CUDA Path 和 Viewer 的 CUDA 交互调度。CPU Path、离线
`RenderSettings`、CLI、NEE/MIS 权重及点光/方向光的确定性求和保持不变。

真实内部视角的原生 64-bounce trace 中位数：

| 场景 | 旧基线 | 当前 | 相对耗时 | 加速 |
| --- | ---: | ---: | ---: | ---: |
| Sponza | 217.69 ms | 52.976 ms | 24.34% | 4.11× |
| San Miguel Low Poly | 492.35 ms | 93.000 ms | 18.89% | 5.29× |

两项均通过 55 ms / 125 ms 目标。基准同时导入保存场景中的 floor，因此
实际测试三角形数分别为 66,452 和 5,617,453，不低于只统计主 OBJ 的数量。
相对上一版单 tile 调度的 51.411 / 89.731 ms，全帧基准分别回退 3.04% /
3.64%，仍在 5% 上限内。

## 固定相机

当时的历史工具固化以下相机，FOV 均为 45°：

| 场景 | eye | forward |
| --- | --- | --- |
| Sponza | `(12.0652, 1.5227, 0.6981)` | `(-0.9934, 0.0699, -0.0904)` |
| San Miguel Low Poly | `(25.2773, 1.3083, 2.3746)` | `(-0.9717, 0.2085, -0.1112)` |

目标不随默认 asset camera 或 session 是否被 Viewer 覆盖而改变：

当前 San Miguel 标准运行使用 `.\tools\run_benchmarks.ps1 -Backend cuda`。
Sponza 数值仅保留为历史记录；需要重新测量时应新增独立版本化 case。

五次 trace：

- Sponza：53.796、52.690、52.976、52.853、53.543 ms。
- San Miguel：94.610、93.000、94.543、92.333、91.941 ms。

## GPU 遍历

- 三角形 bounds/centroid 只预计算一次。
- 16-bin SAH 构建二叉树；无有效 SAH split 时使用 centroid 中位数。
- 二叉树折叠为 BVH4，叶节点最多 8 个三角形。
- BVH4 child bounds 为宽节点数组，inner 与 leaf 使用统一栈引用。
- child AABB 按入口距离排序后以正确的 LIFO 逆序压栈，实际访问为近到远。
- slab test 完全展开 x/y/z 三轴；每条射线只计算一次逆方向。
- 三角形求交只读取 `v0/edge1/edge2`，UV/normal 数据到最终 shading 时才读取。
- 无 opacity texture 的不透明材质不会在 traversal 中读取 shading triangle。
- continuation ray 在叶内过滤单面背面与 alpha cutout，不从根节点重启。
- shadow ray 使用 alpha-aware any-hit，首个有效遮挡立即返回。
- BVH 栈溢出、队列溢出和非法计数继续通过 device error code 显式报告。

San Miguel 改造后的前三次 deterministic Sun shadow visibility 合计约 15.9 ms；
此前为 220.5 ms。主要收益来自 BVH4、正确的近到远顺序和 any-hit。

曾验证 bounce 队列压缩、按 ray key 的 CUB radix sort 以及进一步拆分 direct
queue 的原型，但在 Sponza 的实际 active-count 分布下，排序/gather 和额外
kernel 的成本高于遍历一致性收益，并造成端到端回退。因此本次没有保留这些
原型；生产路径的 `sort_ms` 为 0，Performance 面板不会显示虚假的排序耗时。

## 自动交互质量

Viewer 的 `Auto interaction quality` 默认开启并写入 session；旧 session 缺少字段时
默认开启。

- 场景或相机变化时最多执行 2 个 shading bounce，保留首表面直接光、阴影和
  emissive NEE。
- 预览使用独立 accumulation/RNG，不混入原生 spp。
- 内部分辨率为 100%、75%、50%、25% 四档；EMA 高于 12 ms 连续两帧降档，
  低于 8 ms 连续 30 帧升档。
- 大场景首次交互按三角形数量保守 warm-start，避免第一张新视图超过输入延迟
  预算；随后仍由同一 EMA 规则调整。
- Sponza 稳定在 50%（1209×672），20 帧 GPU trace 中位 6.343 ms。
- San Miguel 稳定在 25%（605×336），20 帧 GPU trace 中位 4.706 ms。

上述 headless fallback 的稳定 wall time 分别约 13.8 ms 与 10.8 ms，包含一次
framebuffer 下载；interop 路径没有这次下载。

在更高的 2418×1343 输出尺寸下，20 帧交互 wall time 统计为：

| 场景 | 中位数 | P95 | 最大值 |
| --- | ---: | ---: | ---: |
| Sponza | 13.826 ms | 14.043 ms | 19.039 ms |
| San Miguel Low Poly | 10.788 ms | 11.234 ms | 14.829 ms |

## 静止原生 sweep

连续静止 8 帧后：

- 恢复 64 bounce 和现有 Russian roulette。
- accumulation 与 RNG 保持完整原生尺寸。
- Wavefront arena 只按最多 128 行分配和捕获 graph；总 quantum 为 1–512 行，
  超过 arena 容量时在同一帧、同一 stream 内拆成连续 graph launch。
- 百万级三角形初始为 32 行，其余为 64 行。每行成本使用
  `0.75 × 旧 EMA + 0.25 × 本次毫秒/行`，尾部不足标准批量时不更新 EMA。
- 下一批限制为上一批的 0.5–2 倍。实测 10/12 ms 控制器无法让 San Miguel
  达到完整 sweep/s 至少 2 倍，因此生产参数采用约 12 ms 目标、15 ms 硬限制；
  百万级三角形额外限制到 80 行，避免偶发长批次破坏 UI 延迟。
- 部分 quantum 仅更新 accumulation/RNG，不写 interop surface。完整 sweep
  结束时才增加 1 spp，并用一次全屏 resolve 发布统一样本数的结果；下一轮
  sweep 期间继续显示上一张完整结果。
- fallback 在部分 sweep 中既不下载也不覆盖 framebuffer；只有预览和完整
  sweep 发布才下载。如果 interop 中途失败且 host 尚无原生帧，则恢复最后
  一张完整预览。

2418×1343 静止阶段测量：

| 场景 | 稳态 quantum | UI 帧/sweep | 60 Hz 完整 spp/s | 对旧 32 行调度 | quantum wall 中位数 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sponza | 128 行 | 11 | 5.455 | 3.82× | 9.131 ms | 9.732 ms |
| San Miguel Low Poly | 约 70–80 行 | 19–20 | 3.077 | 2.15× | 11.116 ms | 15.766 ms |

San Miguel 的 P95 高于计划中的 12 ms，但 UI wall P95 仍低于 20 ms，且满足
完整 sweep/s 至少 2 倍；严格收紧到 12 ms 会使批量稳定在约 50–60 行并失去
吞吐验收。单次观测最大值为 17.859 ms，超过硬限制后下一批会立即减半。

当前 interop smoke：

```text
viewer mode=path frames=300 size=960x540
interop=active trace_ms=0.2072 present_ms=0.01824 published=1 downloads=0
```

该结果确认完整 sweep 使用全屏 resolve 发布，且 interop 路径保持零下载。

## 验证

新增合约覆盖：

- 自动交互预览不增加完整 spp。
- 自动质量首次启用时即使相机未变化，也会先生成一张完整预览。
- 8 个静止帧后进入 native quantum。
- 部分 sweep 不改变 host framebuffer，也不增加 framebuffer 下载次数。
- 相机在 sweep 中途变化时丢弃旧 sweep，并立即恢复完整预览。
- sweep 完成后才增加 1 spp，并且只发生一次整帧下载/发布。
- 分块 sweep 与 full-frame 在同 seed 下逐像素一致。
- session 中自动交互质量开关 round-trip。

验证命令：

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
```

Compute Sanitizer 使用 `RENDERER_CUDA_SANITIZER_FALLBACK=1` 的等价固定拓扑
构建运行完整 `renderer_tests`，结果为 `ERROR SUMMARY: 0 errors`。生产 CUDA
Graph 条件节点拓扑保持不变。

运行数据保存在 `build/validation/*-heavy-final.txt`、
`build/validation/*-interaction-final.txt` 与 `viewer-auto-out.txt`。
