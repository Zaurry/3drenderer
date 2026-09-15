# OpenGL DDGI 动态光照探针

OpenGL 写实模式使用 CUDA 更新世界空间探针，再以全屏阶段查询漫反射间接光。新会话默认打开 DDGI；没有 CUDA 的构建或设备保持原来的 OpenGL 照明，并在技术面板说明原因。旧版 v1–v6 会话迁移后默认关闭 DDGI，不改变原先的 IBL、SSR 参数。会话 v7 保存设置，不保存 GPU 历史。

```powershell
.\build\cuda-ninja\bin\viewer.exe --scene builtin --mode opengl --no-restore-last --ddgi on
.\build\cuda-ninja\bin\viewer.exe --scene builtin --mode opengl --no-restore-last --ddgi off
```

## 参数与使用

OpenGL 技术面板中的 **DDGI - Dynamic Diffuse GI** 提供开关、暂停、手动范围、网格、射线数、更新预算、历史权重、表面偏移、强度、重定位和分类。`Refit volume` 按当前场景重新适配；`Reset probes` 清除历史。移动物体不会自动扩大探针体。

默认参数是 12×8×12 个探针，每个更新探针 128 条旋转照明射线和 32 条固定检测射线，每帧更新 256 个探针。辐照度内部为 8×8，距离矩为 16×16，每张探针贴图各有一圈镜像八面体过滤边界。网格每轴限制 2–32，总数限制 8192；照明射线数为 32–512。

历史权重默认 0.95，按实际更新间隔换算为 `pow(weight, elapsed_seconds * 60)`。光照、材质、环境或几何变化后的八轮更新采用当前测量值，让旧间接光尽快消退；递归反射仍读取本帧开始时的探针历史。随后恢复常规历史滤波。暂停期间保留图集，恢复时按实际间隔更新。

`Show probes` 绘制带深度测试的探针：绿色有效、红色停用、黄色尚未初始化。调试视图提供间接光、有效状态、更新年龄以及两张图集。面板记录更新数量、最大年龄、追踪/混合/导出/查询耗时、探针缓冲显存与图集读回计数。显存数值包含探针双缓冲、射线缓冲和导出图集，不包括另行上传的 CUDA 场景和 OpenGL 全屏目标。

## 更新与查询

`CudaDdgiVolume` 接收原始 `RenderSceneSnapshot`，复用 `cuda_scene.cuh` 的实例化场景、BVH、材质和纹理采样。变换只更新实例并 refit TLAS。调度器将移动物体旧、新包围盒附近的探针加入优先队列，优先队列最多使用一半帧预算，其余预算继续轮转，停用探针也会复查。相机移动、相机切换和窗口缩放不清除历史；场景来源、网格布局、重新适配和显式重置会初始化图集。

每帧先复制历史，再执行固定射线、重定位/分类、照明射线、辐照度与距离矩混合、边界填充和元数据更新，最后交换历史缓冲。固定射线不参加光照积分；单面材质的背面只在探针检测路径中被接受。重定位不超过每轴 45% 探针间距。探针射程覆盖探针体和场景几何，手动放在厚墙内部的小探针体也能检测背面。

照明射线命中处累加漫反射直射光与上一帧探针间接光，逐轮传播多次漫反射。流程参考 [NVIDIA DDGI 集成说明](https://raw.githubusercontent.com/NVIDIAGameWorks/RTXGI-DDGI/main/docs/Integration.md)。点光、聚光、方向光使用直接估计；矩形光使用面积采样，不套用路径追踪的 BSDF/MIS 竞争权重。已由 LTC 表示的发光连接从探针场中剔除，其他发光几何继续贡献能量。主环境光提取与 OpenGL 使用同一提取器、旋转和强度，未提取部分留在残余环境中。

查询阶段始终准备所需 G-buffer，包括世界空间几何法线。八邻居插值使用距离一阶/二阶矩的可见性、几何法线方向权重和表面偏移；几何法线排除接收面背后的探针。着色法线用于辐照度方向查询。探针体外在一个网格间距内平滑过渡到环境漫反射，体内无有效可见邻居时保持黑色。

DDGI 替代不透明及 Alpha Mask 接收表面的漫反射 IBL 和 SSR 漫反射项，保留 SSR 的 GGX、Kulla–Conty 及其采样 PDF。关闭 SSR 时继续使用镜面 IBL。DDGI 同时加入最终颜色和 SSR 命中辐射源；材质 AO 与屏幕 AO 只衰减间接光。透明接收表面和 NPR 继续使用原有着色。DDGI 不计算镜面多次反射、折射或焦散。

## 互操作与验证

`OpenGlDdgiPass` 使用三张 RGBA32F CUDA/OpenGL 互操作纹理，分别导出完整辐照度、距离和两行元数据。正常路径不读回图集；更新统计和错误码会有少量 CPU 传输。注册或映射失败时改为主机暂存上传。`--no-cuda-interop` 可强制走回退路径。

追踪和混合耗时由 CUDA event 测量，查询由 OpenGL timer query 测量。导出耗时是包含映射、CUDA 导出和归还纹理的 CPU 墙钟时间；回退时包含下载与 GL 上传提交。`--frame-report` 通过 `glFinish` 测量包括呈现完成的整帧耗时。

```powershell
$env:DDGI_REQUIRE_GPU = '1'
$env:DDGI_CAPTURE_DIR = "$PWD/build/ddgi-validation"
.\build\cuda-ninja\bin\ddgi_tests.exe
ctest --test-dir build/cuda-ninja -C Release --output-on-failure -j 1
Remove-Item Env:DDGI_REQUIRE_GPU
ctest --test-dir build/no-cuda -C Release --output-on-failure -j 1

.\build\cuda-ninja\bin\viewer.exe --scene builtin --mode opengl --no-restore-last --ddgi on --frames 300 --warmup-frames 60 --frame-report build/ddgi-validation/interop-frames.json --capture build/ddgi-validation/interop.png
.\build\cuda-ninja\bin\viewer.exe --scene builtin --mode opengl --no-restore-last --ddgi on --no-cuda-interop --frames 300 --warmup-frames 60 --frame-report build/ddgi-validation/fallback-frames.json --capture build/ddgi-validation/fallback.png
$env:DDGI_REQUIRE_GPU = '1'
& 'D:\NVIDIA GPU Computing Toolkit\CUDA\v13.3\compute-sanitizer\compute-sanitizer.exe' --tool memcheck --leak-check full --error-exitcode 99 .\build\cuda-ninja\bin\ddgi_tests.exe
Remove-Item Env:DDGI_REQUIRE_GPU
```

`ddgi_tests` 包含恒定 HDR 辐照度、八面体边界、背面分类/重定位/重新激活、实例 refit、金属零漫反射、Alpha Mask、LTC 发光连接去重、主环境光与残余环境等价性、互操作/回退逐像素一致性、Cornell 离线参考、隔墙暗室、五类动态变化及生命周期。现有 SSR GGX 白炉测试同时覆盖 DDGI 开启与关闭。

数值比较、截图、构建和实机结果见 [2026-09-15 验证记录](opengl-ddgi-validation-2026-09-15.md)。
