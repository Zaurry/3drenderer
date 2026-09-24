# DDGI 更新预算导致的亮度闪烁

基线为 `2f52909`。静止 Cornell 场景可复现随 `Probes per frame` 改变的亮度跳变。旧实现按所有未更新帧衰减历史：默认权重 0.95、1152 个探针、每帧更新 64 个、60 fps 时，一个探针更新前的历史权重已降到 `0.95^18 ≈ 0.397`，一批 128 条随机射线占据约 60% 的新结果。更新预算越小，单次替换越剧烈。距离矩同样受影响，放大为可见性权重变化。

修复包括：

- 为每批测量保留配置的历史权重下限；只有高于 60 Hz 的探针更新继续按时间校正，等待时间不再等价于增加样本量。
- 旋转射线和光源采样使用探针实际更新次数作为序列索引，改变预算不会跳过随机序列。
- 距离矩与辐照度的快速收敛分离。纯光照变更保留可见性历史；几何和可能影响 Alpha Mask 的变更用版本号逐探针刷新距离，之后恢复累积。

测试在 RTX 3050 Laptop、CUDA 13.3、Ninja Release 上运行。静态测试使用 12×8×12 探针、128 条照明射线，64×64 输出，关闭 AO/SSR，测量 DDGI 间接光。先更新 32 轮，再连续采集 180 帧。中央区域为 `[21,42) × [21,42)`；表中峰值是区域逐像素、逐通道的相邻帧 RMS 差异，除以该区域的平均 RGB 强度。它衡量局部闪烁，不等同于整幅画面的平均亮度差。

| 每帧更新探针数 | 修复前峰值 | 修复后峰值 | 区域均值标准差（前 → 后） |
| --- | ---: | ---: | ---: |
| 64 | 18.307% | 1.103% | 2.956% → 0.203% |
| 256 | 5.513% | 1.373% | 0.962% → 0.513% |
| 1152 | 1.587% | 1.590% | 0.980% → 0.881% |

低预算造成的大幅跳变得到抑制。全量更新仍有有限射线的采样噪声；没有停止探针更新或锁定静态照明。新增回归要求所有预算的区域均值标准差小于 1%、相邻帧 RMS 峰值小于 2%，旧实现不能通过。

[修复前逐帧数据](output/ddgi-flicker-2026-09-22/before.json)、[修复后逐帧数据](output/ddgi-flicker-2026-09-22/after.json)、[旧版本回归失败日志](output/ddgi-flicker-2026-09-22/before.log)。

动态变化回归保持八轮内收敛：移动灯光、关闭灯光、修改材质、修改环境和移动遮挡物的区域均值相对误差分别为 0.087%、0.808%、0.336%、0.306%、0.391%。另有回归比较同样数量的测量在 10 Hz 与 60 Hz 下的历史，并检查环境变更不会扰动距离矩。

CUDA 的 14 项 DDGI 测试全部通过，OpenGL 光照/白炉回归也通过；本次 CUDA CTest 选择这两个受影响的测试目标，没有重复运行此前已记录失败的 RTRT 玻璃测试。[CUDA CTest 日志](output/ddgi-flicker-2026-09-22/cuda-ctest.log)、[动态原始指标](output/ddgi-flicker-2026-09-22/dynamic.json)。

CUDA 和无 CUDA 构建均成功。无 CUDA 全量 CTest 为 10 项通过、1 项跳过、0 项失败。[无 CUDA CTest 日志](output/ddgi-flicker-2026-09-22/no-cuda-ctest.log)。新版查看器实测 300 帧，`ddgi=active`、`updated=256`、`probe_resets=1`、`atlas_downloads=0`。[查看器日志](output/ddgi-flicker-2026-09-22/viewer.log)。

新增的两项回归也通过 Compute Sanitizer `memcheck --leak-check full --error-exitcode 99`：进程退出码 0，报告 0 错误、0 字节泄漏。[内存检查日志](output/ddgi-flicker-2026-09-22/sanitizer.log)。

复现命令：

```powershell
$env:DDGI_REQUIRE_GPU = '1'
$env:DDGI_CAPTURE_DIR = "$PWD/build/ddgi-flicker/after"
.\build\cuda-ninja\bin\ddgi_tests.exe --filter static_lighting,sparse_updates
.\build\cuda-ninja\bin\ddgi_tests.exe
```
