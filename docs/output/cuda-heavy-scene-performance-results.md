# CUDA Path 重场景性能结果

日期：2026-07-30

分支：`codex/gpu-wavefront-nee-mis`

设备：RTX 5080，CUDA 13.3，Release，输出 2418×1343。

## 结论

本轮只改变 CUDA Path 和 Viewer 的 CUDA 交互调度。CPU Path、离线
`RenderSettings`、CLI、NEE/MIS 权重及点光/方向光的确定性求和保持不变。

真实内部视角的原生 64-bounce trace 中位数：

| 场景 | 旧基线 | 当前 | 相对耗时 | 加速 |
| --- | ---: | ---: | ---: | ---: |
| Sponza | 217.69 ms | 51.411 ms | 23.62% | 4.23× |
| San Miguel Low Poly | 492.35 ms | 89.731 ms | 18.23% | 5.49× |

两项均通过 55 ms / 125 ms 目标。基准同时导入保存场景中的 floor，因此
实际测试三角形数分别为 66,452 和 5,617,453，不低于只统计主 OBJ 的数量。

## 固定相机

`cuda_path_benchmark` 固化以下相机，FOV 均为 45°：

| 场景 | eye | forward |
| --- | --- | --- |
| Sponza | `(12.0652, 1.5227, 0.6981)` | `(-0.9934, 0.0699, -0.0904)` |
| San Miguel Low Poly | `(25.2773, 1.3083, 2.3746)` | `(-0.9717, 0.2085, -0.1112)` |

目标不随默认 asset camera 或 session 是否被 Viewer 覆盖而改变：

```powershell
cmake --build --preset cuda-release --target cuda_path_benchmark
.\build\cuda\bin\cuda_path_benchmark.exe sponza 5
.\build\cuda\bin\cuda_path_benchmark.exe san-miguel 5
```

五次 trace：

- Sponza：51.126、51.454、51.411、51.517、51.124 ms。
- San Miguel：90.694、88.749、91.070、89.731、88.693 ms。

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
- Wavefront arena 按最多 64 行容量分配和捕获 graph。
- 初始 tile 为 32 行；超过 10 ms 减半，低于 4 ms 连续 8 tile 后加倍，
  范围 1–64 行。
- 每次 Viewer 循环只渲染一个 tile，完整 sweep 后才增加 1 spp。
- interop texture 保持用户输出尺寸；预览先铺满 surface，原生 tile 逐块覆盖。

保存的 San Miguel session 隐藏运行 18 秒后正常关闭：

```text
viewer mode=path frames=1284 size=2418x1343
interop=active trace_ms=5.44093 downloads=0
```

该运行包含约 3.2 秒场景上传，整体循环仍超过 60 FPS；稳定渲染阶段余量更高。

## 验证

新增合约覆盖：

- 自动交互预览不增加完整 spp。
- 8 个静止帧后进入 native tile。
- fallback 只下载刚完成的 tile，未覆盖区域继续保留预览背景。
- tile sweep 完成后才增加 1 spp。
- tile 与 full-frame 在同 seed 下输出一致。
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
