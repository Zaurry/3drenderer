# RTRT 全光线 OptiX 迁移验证

本轮将 RTRT 的主射线、路径续接、阴影，以及原生分辨率镜面/玻璃补射全部迁入独立的 OptiX 后端。Viewer 仍显示 `RTRT`，`--mode path` 仍是兼容别名。OpenGL、DDGI 与离线 CUDA Wavefront 参考渲染保持原来的模式边界。

迁移开始于 2026-09-23，最终回归与内存检查更新于 2026-09-24。可移植的测试、画质与帧耗时数据见 [验证数据](output/rtrt-optix-migration-2026-09-23/summary.json)。

## 实现

- `src/render/optix/` 独立维护 raygen / miss / hit 程序、积分器、共享参数、加速结构、降噪和帧调度。路径在 raygen 内迭代，closest-hit 只返回紧凑命中；阴影使用独立 payload、首个有效命中终止并跳过 closest-hit。
- 三角形和解析球分别建 GAS，通过 IAS 共享逻辑实例和材质绑定。解析球使用 OptiX 内置球体程序，支持混合几何、非均匀缩放和镜像变换。
- GAS 按资产缓存、压缩；刚体运动仅更新 IAS。颜色/粗糙度修改不重建 GAS，影响透明接受方式的材质修改只重建相关几何。构建临时空间复用，参数和实例上传使用有事件保护的固定页内存循环缓冲。
- 实时场景使用 `ShadingOnly` 存储，不创建软件 BLAS/TLAS；删除旧实时 CUDA 遍历和光栅首交点提供器，以及无效的遍历错误回读。
- 场景无几何时由 CUDA 直接生成背景和运动引导，再进入相同的时域/呈现流程；此时没有求交任务，不提交空 OptiX launch。覆盖首次空场景、删除全部实例、原生/半分辨率和隐藏背景。
- 查询真实 RT Core 和 SER 能力。只有设备支持时才执行 SER；设置中的请求与实际启用状态分别报告。
- PTX 以数值字节嵌入，解决 MSVC `C2026` 超长字符串问题。实时构建目标与离线 CUDA 目标分离。
- SVGF 默认、OptiX AI、分量历史、TAA/TAAU 和原生材质/光学路径保留；旧 JSON 的 `hardware_ray_tracing`、`raster_primary` 字段忽略且不再保存。没有 OptiX / RT Core 时，实时 API 明确不可用，Viewer 回到 OpenGL。

## 环境与构建

Windows、MSVC、Ninja Release、CUDA 13.3、`sm_86`；RTX 3050 Laptop GPU 4 GiB，驱动 616.56。基于 `2f52909` 工作区，保留已有的 DDGI 修改。最终可执行文件为 `build/cuda-ninja/bin/viewer.exe`。

```powershell
# 已配置 CUDA / MSVC 开发环境时
cmake -S . -B build/cuda-ninja -DRENDERER_CUDA=ON -DRENDERER_OPTIX=ON
cmake --build build/cuda-ninja --parallel 4

$env:RTRT_REQUIRE_OPTIX = '1'
$env:RTRT_REQUIRE_OPTIX_DENOISER = '1'
$env:RTRT_VALIDATION_DIR = "$PWD/build/optix-migration/quality"
ctest --test-dir build/cuda-ninja -C Release --output-on-failure
```

CUDA + OptiX 的完整 CTest **11/11 通过**。其中 `realtime_tests` **25 通过、0 失败**，默认跳过 3 项可选长测：性能扫描、San Miguel 高采样画质、HDR firefly 压力。`optix_interop_tests` 的两个用例均通过。no-CUDA 构建 **10 通过、1 个 GPU 测试程序跳过**；GPU 子用例的跳过不计为硬件验证。另行启用 benchmark 构建，两个 benchmark 测试程序均通过。

`RENDERER_CUDA=ON / RENDERER_OPTIX=OFF` 独立配置同样 **10 通过、1 个互操作测试程序跳过**。离线 Cornell 64×64、4 SPP 实际运行报告 `backend=cuda`；请求 RTRT 的 Viewer 明确说明缺少 OptiX，并以 OpenGL + DDGI 成功运行 3 帧。最终工作区已恢复 `RENDERER_OPTIX=ON`。

新用例覆盖解析球/三角形混合、精确球体深度、材质与透明分类更新、镜像/非均匀实例、GAS/IAS 复用、空场景、三角形共边覆盖，以及带非对称图案的互操作/主机浮点图像对照。原有玻璃、阴影、反射、运动/遮挡历史、原生材质、神经降噪切换/尺寸/场景变更测试全部保留。

## 内存与资源检查

完整 `realtime_tests` 在 `RTRT_OPTIX_VALIDATE=1` 下使用 Compute Sanitizer memcheck，开启 OptiX 资源泄漏和完整 CUDA 泄漏检查：**25 通过、0 失败、3 项可选长测跳过；0 错误、0 字节泄漏**。日志保存在 `build/optix-migration/final-memcheck-tests.log` 和 `final-memcheck-tool.log`。此次检查包含实时 CUDA 核函数、OptiX 求交和神经降噪，不代表另行跳过的长测或其他 GPU 已验证。

首次检查暴露出“新建 renderer 后直接渲染空场景”在 memcheck 下触发硬件异常；已有 GAS 后清空实例则正常。空场景直接生成背景后，针对性 3 项测试和完整套件均通过。正常几何仍全部使用 OptiX 求交，画质参数未改变。

```powershell
$env:RTRT_REQUIRE_OPTIX = '1'
$env:RTRT_REQUIRE_OPTIX_DENOISER = '1'
$env:RTRT_OPTIX_VALIDATE = '1'
& "$env:CUDA_PATH/compute-sanitizer/compute-sanitizer.exe" --tool memcheck `
  --check-optix-leaks --leak-check full --error-exitcode 99 `
  --log-file build/optix-migration/final-memcheck-tool.log `
  build/cuda-ninja/bin/realtime_tests.exe
$env:RTRT_OPTIX_VALIDATE = $null  # 性能测量关闭验证模式
```

## 运行与性能

默认画质保持 100% 内部光照、1 SPP、8 层路径、SVGF、TAA。960×540；完整帧计时包含呈现后的 `glFinish` 和 UI，预热 30 帧。性能测量关闭 OptiX 验证模式，各 GPU 测试串行运行。

最终新后端数据于 2026-09-24 重测；Sponza 使用随验证数据保存的 [固定大厅相机](output/rtrt-optix-migration-2026-09-23/sponza-camera.json)。资产默认相机位于外墙前，工作量不同，不能用它与大厅视角比较。

| 场景/版本 | 帧数 | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| builtin，迁移前 CUDA 软件求交 | 120 | 31.825 | 33.185 | 33.502 |
| builtin，迁移前 OptiX（只修 PTX 嵌入） | 120 | 36.712 | 37.711 | 38.621 |
| builtin，新 OptiX | 120 | 36.442 | 37.045 | 37.565 |
| builtin，新 OptiX，相机移动 | 300 | 36.709 | 37.296 | 37.809 |
| Sponza，262,267 三角形，新 OptiX，固定大厅相机 | 300 | 143.345 | 145.389 | 147.610 |

新旧 OptiX 在这个 12 三角形 builtin 场景的帧耗时接近；相对软件求交没有加速，不能把启用 RT Core 解释成所有场景 FPS 都提高。当前优化主要消除重复的软件加速结构、复用 GAS/IAS、压缩存储并减少同步/回读。Sponza 本轮只有新后端数据，不宣称复杂场景的前后加速倍数。以上配置在这块 GPU 上没有达到 60 FPS。

300 帧运行均报告 `primary=optix`、`hardware_rt=1`、`rt_core_version=20`、`interop=active`、`downloads=0`、软件 `blas_builds=0`。RTX 3050 报告 `ser_supported=0`、`ser_active=0`；本机无法验证支持 SER 设备上的重排执行。

builtin 全程只建 1 个 GAS / 1 个 IAS；Sponza 全程只建 103 个 GAS / 1 个 IAS，加速结构显式缓冲约 9.06 MiB。builtin 场景/帧缓冲统计的分配次数从 31 降至 25，稳态不增长；帧缓冲仍为 433.081 MiB。上述显存计数不含 OptiX 驱动内部内存。`acceleration_ms` 是最近一次构建/更新的耗时，不是静态场景每帧成本。

互操作与 `--no-cuda-interop` 各运行相同的 30 帧，导出 PNG 逐像素相同。前者下载数为 0，后者为 30；浮点纹理与主机输出另由互操作测试逐像素核对。已检查 builtin 和 Sponza 截图。

```powershell
.\build\cuda-ninja\bin\viewer.exe --scene builtin --mode rtrt --no-restore-last `
  --frames 300 --warmup-frames 30 --camera-motion `
  --frame-report build/optix-migration/viewer-300.json

.\build\cuda-ninja\bin\viewer.exe --scene asset `
  --asset "Computer Graphics Archive/sponza-palace/source/scene.glb" `
  --mode rtrt --no-restore-last --width 960 --height 540 `
  --frames 300 --warmup-frames 30 `
  --camera-preset docs/output/rtrt-optix-migration-2026-09-23/sponza-camera.json `
  --frame-report build/optix-migration/sponza.json
```

## 画质对照

迁移前 OptiX 与新后端的 **26 张线性 HDR PFM 数值逐像素相同**，最大绝对差为 0，所有像素有限。不是通过降低采样、路径深度或滤波质量取得性能结果。

| 指标 | 迁移前 / 后 |
|---|---:|
| Cornell SVGF + TAA 对 1024 SPP 参考 MSE | 0.01394164 |
| Cornell 50% TAAU 对参考 MSE | 0.01245183 |
| Cornell 静态帧差 MSE | 0.00115537 |
| 稳定玻璃采样 Fresnel MSE | 7.8898e-14 |

对照包含 SVGF、OptiX AI 的原生/低内部比例、玻璃、镜面、移动阴影与遮挡显露。San Miguel 1024 SPP 和长时间 firefly 压力测试本轮未重跑，不沿用旧硬件的帧率结论。

完整帧报告、日志、PNG/PFM 与 `comparison.json` 位于本地 `build/optix-migration/`，不随源码提交。
