# Renderer 全量修复与架构硬化结果

## 结论

本分支完成了计划内 10 个正确性/数据安全问题的修复，并把场景边界、CUDA 设备所有权、交互输出、构建目标、测试、CI 和当前文档收敛到同一架构。

- 分支：`codex/renderer-hardening`
- 基线：`master@75d19920dae24a36946a980fd1bbe59c78c5b5e3`
- 实现提交：`ceb7149`
- 报告提交：本文件所在的本地提交
- 发布范围：仅本地分支和本地提交；未 push，未创建 PR

验收结果：

- Windows no-CUDA Release：6/6 tests 通过。
- Windows CUDA-native Release：8/8 tests 通过。
- OpenGL Viewer、no-CUDA Path 回退、CUDA/OpenGL interop 和离线 renderer 冒烟通过。
- `CMAKE_CUDA_ARCHITECTURES=120` 自定义架构配置通过。
- San Miguel `cuda_full` median 相对基线 +2.83%，满足“不超过 5%”门槛。
- `cuda_interaction` median 10.0695 ms、P95 10.5518 ms，满足 16.7/20 ms 门槛。
- Compute Sanitizer 未取得有效 memcheck 结论：工具注入后首个 kernel 返回 CUDA 999；详见“验证限制”，本报告不把它记作通过。

## 架构结果

### 单一快照

公开渲染边界统一为 `RenderSceneSnapshot`。快照包含：

- `shared_ptr<const Scene>` 持有的共享几何资产；
- 稳定 `AssetId`、asset geometry revision；
- 实例 object/world/inverse/normal matrix 和 world bounds；
- 强类型 `MaterialSlot` 与逐实例材质表；
- 全局纹理、灯光、环境；
- `topology`、`geometry`、`transforms`、`material_bindings`、`materials`、`textures`、`lighting`、`environment` 八个单调 revision。

OpenGL、CUDA Path、Path interactive session 和离线 `renderer.exe` 的公开入口只接收这份快照。OpenGL 内部可按 revision 生成临时扁平批次，但扁平 `Scene` 不再是公共编辑或增量上传边界。CUDA 的旧 `sync(const Scene&, SceneChangeSet)` 已删除。

### 文档编辑与空间缓存

`SceneObject` 以有限、可逆 4×4 本地仿射矩阵为真值，TRS 只在无损可分解时提供。`SceneDocument` 的公开集合、查找、资产和环境读取均为只读；写入经过类型化 setter 或 `SceneEditTransaction`。

事务支持 merge key，把连续拖拽合并为一个 undo group；每次 preview setter 仍立即增加 revision、使缓存失效并设置 dirty。文档维护 ObjectId 哈希索引、父子邻接表、world matrix/world bounds 缓存，以及每个 mesh asset 的持久化拾取 BVH。

### CUDA 设备和后端边界

新增 `CudaDeviceContext`：

- Viewer 在创建 stream/buffer/interop 前选择当前 OpenGL context 的兼容 CUDA device；
- 离线 renderer 默认 device 0，支持 `--cuda-device N`；
- stream、renderer、interop registration 和 surface 记录并验证 device ID；
- interop 不调用隐式 `cudaSetDevice` 偷换设备；
- availability 使用真实 device 创建和最小 probe kernel，不只检查 device count。

CPU Path、`PathTracerRenderer`、`PathInteractiveSession` 的 CPU 分支、`PathBackend::Cpu/Auto`、`--threads` 和 `--path-backend` 已删除。no-CUDA 构建仍提供场景、文档、测试和 OpenGL；Path UI 明确不可用并显示原因。

### 帧和统计生命周期

`RenderFrameOutput` 改为拥有生命周期的 variant：

- `HostFrameHandle` 持有 `shared_ptr<const Framebuffer>`；
- `OpenGlTextureHandle` 持有 texture metadata 和 backend owner lease。

统计信息改为 OpenGL/CUDA 类型化 variant。shader reload capability 只由 OpenGL backend 暴露，Path 不再实现无效果的空控制接口。

### 文件版本和构建

- `.rscene` 写 v4，矩阵为四个明确 4 元行数组；v1–v3 可读迁移。
- Viewer session 写 v2；v1 的 CPU/Auto backend 字段只生成迁移提示。
- 保存使用同目录唯一临时文件 + flush/close + 原子 replace/rename；替换失败不删除原文件。
- CMake 最低版本 3.24，目标拆成 scene、render-common、interactive、CUDA、OpenGL 和 app。
- presets 为 `default`、`no-cuda`、`cuda-native`；默认 CUDA architecture 为 `native`，发行方可显式覆盖列表。

## 逐项问题报告

### 1. 环境重要性采样 PDF 不匹配

**现象与影响**

环境 texel 按 `luminance × solid angle` 选中后，旧实现在线性 theta 中均匀抖动，却报告 `texel_pmf / texel_solid_angle`。实际条件分布和 PDF 不同，尤其在极区产生系统性估计偏差；更多样本只能减小方差，不能消除偏差。

**根因**

等经纬 texel 在球面上的面积元素是 `sin(theta) dtheta dphi`。若要在 texel 覆盖的球面面积内均匀，必须在上下纬度边界的 `cos(theta)` 之间均匀，而不是在 theta 中均匀。

**修改内容**

CPU `EnvironmentMap::sample` 和 CUDA environment sampler 都执行：

1. 按 PMF/CDF 选择 texel；
2. 在行边界 `cos(theta0)`、`cos(theta1)` 之间均匀采样 cos(theta)；
3. 在列的 phi 边界内均匀采样；
4. 保持 PDF 为 `pmf / texel_solid_angle`。

**解决后的行为**

采样方向的条件分布与报告 PDF 一致，球面积分归一，极区不再过采样/欠采样。

**测试证据**

- `sampling_tests`：所有 texel 的 PDF×solid-angle 求和误差小于 `2e-5`。
- 固定 32768 样本直方图逐 texel 与目标 PMF 误差小于 `1e-4`。
- 综合 `renderer_tests` 包含相同的积分和直方图回归。

**兼容/限制**

PMF 定义和环境方向映射未改变；只修正 texel 内条件采样。全黑图仍使用均匀球面 fallback。

### 2. Geometry 更新留下旧材质绑定

**现象与影响**

几何 primitive 数量变化与 material binding 分开增量上传时，device view 可能短暂发布“新 triangle count + 旧 binding table”，导致越界读取或给新 primitive 使用错误材质。

**根因**

旧扁平 Scene 上传入口把 geometry 和 material binding 当作可独立发布的缓存，没有 schema 级原子性。

**修改内容**

- 删除 CUDA 扁平 `sync(const Scene&, SceneChangeSet)` 入口。
- `RenderSceneAssetSnapshot` 同时携带 immutable geometry 和等长强类型 material-slot table。
- snapshot build 验证 sphere/triangle slot 数量和范围。
- CUDA 在 host 端完整打包候选 assets、bindings、materials；相关 upload 全部成功后才更新 `DScene view_`。
- geometry revision 变化必然同步重新打包该 asset 的 primitive/binding 数据。

**解决后的行为**

CUDA 永远只看到同一版本的 primitive count 与 slot table；上传失败保持上一份完整 device scene 有效。

**测试证据**

- `cuda_contract_tests` 同时改变 triangle geometry revision 和 material-binding revision，渲染与统计正常。
- `scene_tests` 验证 snapshot slot table 与几何一致，正数越界在 snapshot build 被拒绝。
- CUDA 全套 8/8 通过。

**兼容/限制**

C++ 旧 flat upload API 不保留源码兼容，这是本次明确边界。

### 3. 实例资产变化不重建 BLAS

**现象与影响**

旧实现只看实例拓扑/矩阵；若实例继续引用同一位置的 asset record，但其几何已经改变，BLAS 可能仍对应旧顶点，产生漏交、幽灵命中或越界。

**根因**

asset 没有稳定身份与独立 geometry revision，BLAS cache 无法区分“同一实例布局、资产内容已更新”。

**修改内容**

- 每个 asset 增加稳定 `AssetId` 和 `geometry_revision`。
- CUDA cache 记录 asset ID、revision、geometry fingerprint 和 BLAS layout。
- 单 asset revision/fingerprint 变化只重建该 BLAS。
- instance topology 变化重建 TLAS；纯 matrix/bounds 变化只上传 instance 并 refit TLAS。

**解决后的行为**

资产几何更新立即进入求交结构；无关资产 BLAS 保持复用。纯拖拽不再触发 BLAS 或 geometry/material/texture upload。

**测试证据**

- `cuda_contract_tests`：两个 asset 初始 BLAS=2；只改第一个 asset 后 BLAS build count 恰好 +1。
- 随后纯 transform：BLAS 不变，TLAS refit 恰好 +1；geometry、binding、material、texture upload bytes 均不增加。
- 综合 instancing 回归覆盖拖拽、缩放、duplicate、hide/show 和 TLAS build/refit 计数。

**兼容/限制**

geometry fingerprint 是 revision 误用的防御层，不替代文档正确增加 revision。

### 4. `.rscene` 保存可能删除原文件

**现象与影响**

旧保存流程先删除目标再重命名临时文件；磁盘、权限、杀软或进程中断发生在两步之间时，原文件和新文件可能同时丢失。

**根因**

把“替换”实现成 delete + rename，而不是使用平台原子 replace primitive；场景和会话各自维护不一致的保存代码。

**修改内容**

抽取 `write_file_atomically`：

- 在目标同目录创建 PID + 单调序号唯一临时文件；
- binary write、flush、close 并检查 stream 状态；
- Windows 使用 `MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)`；
- POSIX 使用同文件系统 rename；
- replace 前或 replace 失败只删除临时文件；
- 场景和 session save 共用该工具；
- 增加 one-shot `BeforeReplace` fault injection。

**解决后的行为**

发布新文件只有一个平台原子操作。失败时目标路径仍包含原始完整字节，不存在先删原文件的窗口。

**测试证据**

- `document_tests` 和综合测试写入原始字节，注入 replace 前失败，再逐字节比较完全相同。
- no-CUDA 和 CUDA 两套 test presets 均通过。

**兼容/限制**

原子性依赖临时文件与目标位于同一目录/文件系统；实现已经强制这一条件。它不承诺远程文件系统超出其 rename/replace 语义的持久性。

### 5. 球体在后端和拾取中消失

**现象与影响**

程序球在某些扁平/实例转换中没有被带入 OpenGL、CUDA 或拾取，导致 Viewer 看不到、Path 没有交点、点击也选不中；不同后端还可能使用不同 UV/normal 语义。

**根因**

运行时同时存在 analytic sphere 和 triangle mesh 两种 primitive 分支，而各消费链路只实现了其中一部分。

**修改内容**

- 快照只保存一份 64×32 平滑单位球 asset，所有程序球通过 `world × translate(center) × scale(radius)` 生成实例。
- 生成一致 winding、UV0/UV1、平滑法线和 tangent，并保留 material slot。
- OpenGL、CUDA、拾取和离线 renderer 都只消费 snapshot 中的 canonical triangles。
- 负/非均匀缩放通过实例 world/inverse/inverse-transpose normal matrix 处理。
- 拾取为 asset 建立持久化 BVH，而不是每次点击线性遍历三角形。

**解决后的行为**

球体在四条消费链路使用相同共享单位几何和属性，不再因缺少 analytic sphere 分支而消失；多个球只构建一次共享 BLAS，负缩放下 front-face 和法线保持一致。

**测试证据**

- `test_document_sphere_mesh_negative_nonuniform_scale_and_pick`：snapshot 中无 sphere primitive，64×32 mesh 存在；bounds、负/非均匀缩放和 pick object ID 正确。
- CUDA emissive sphere/非均匀实例 NEE 和材质渲染回归通过。
- no-CUDA OpenGL Viewer 与 CUDA Path Viewer 冒烟通过。

**兼容/限制**

原始 `Scene` helper 仍可在构造阶段包含 sphere；快照边界后只有共享单位 mesh + instances，不提供运行时 analytic sphere API 兼容。

### 6. CUDA emissive NEE 的 alpha/UV 语义错误

**现象与影响**

旧 emissive NEE 只用 UV0/材质 alpha 的局部子集：会忽略 UV1、vertex alpha、texture transform，甚至让 Opaque 发光面被 opacity 提前剔除。直接采样和路径真正命中同一表面时可能给出不同发光结果，造成偏差。

**根因**

遍历、shading、emissive candidate build 和 NEE sample 各自实现表面求值；缺少统一 `SurfaceInputs`/alpha contract。

**修改内容**

- CUDA compact hit/triangle data 保留 UV0、UV1、vertex RGB/alpha 及 presence masks。
- 统一 `effective_alpha_mode` 和 `evaluate_alpha`。
- Opaque coverage 固定为 1，candidate weight 不再查看 opacity。
- Mask 对组合 coverage 使用 cutoff。
- Blend 对发光 contribution 乘 coverage。
- emissive texture 使用 material 指定的 UV set、offset/scale/rotation。
- traversal、shadow visibility、shading 和 emissive NEE 共用相同数据打包/求值函数。

**解决后的行为**

直接命中和 NEE 对 Opaque/Mask/Blend、vertex alpha、UV1 与 texture transform 的解释一致。Opaque 不会被透明数据误删，Mask 不发出被 cutoff 删除的能量，Blend 按 coverage 发光。

**测试证据**

- 新增 CUDA GPU 回归：Opaque + opacity=0 + vertex alpha=0 + 黑 opacity texture 仍发光；Mask + vertex alpha=0 为零；Blend 0.25 coverage 的结果位于 0 与 Opaque 的 50% 之间。
- UV1 + emissive offset 命中白 texel 时发光；移除 offset 后命中黑 texel，结果小于 `1e-6`。
- 既有 cutout、two-sided、blocked visibility、degenerate emissive 和 instanced NEE/MIS 回归通过。

**兼容/限制**

本次只修当前 LOD0 材质语义漂移。CUDA mip chain 和显式 texture LOD 未实现，不能把这项描述为已解决。

### 7. Progressive accumulation 依赖调用方重置

**现象与影响**

调用方漏传 reset 或 scene-change flag 时，新相机/材质/integrator 设置会继续与旧样本平均，生成不可恢复的混合图像；反过来，纯 UI 状态也可能无谓清空累积。

**根因**

renderer 没有能够代表当前辐射分布的内部 key，把缓存正确性委托给 UI 事件分类。

**修改内容**

新增 `ProgressiveRenderKey`，覆盖：

- snapshot 八域 revision；
- 相机 eye/forward/right/up 和 viewport；
- width/height；
- CUDA device ID；
- max bounces；
- RR start/min/max；
- seed；
- automatic interaction-quality policy。

每帧由 renderer 自行构造和比较 key。key 变化、尺寸变化或显式 reset 在现有 buffer 上执行初始化；非辐射 UI 字段不进入 key。

**解决后的行为**

所有影响样本分布的当前设置都会自动清空累积。UI 漏传 `SceneChangeSet` 不会留下旧样本，伪造 change flag 也不会触发不存在的场景上传。

**测试证据**

- `cuda_contract_tests` 逐项修改 max bounce、RR start、RR min、RR max、seed 和 environment revision，每次 accumulated samples 重回 1。
- delta time 等非辐射状态后样本从 1 增到 2。
- 相机和尺寸回归验证自动 reset；手工 `SceneChange::All` 但 revision 不变时不会 reset。

**兼容/限制**

显式 `reset_requested` 仍保留用于用户“清空重采样”；它不再承担缓存正确性的唯一责任。

### 8. TRS 分解吞掉剪切且失败污染状态

**现象与影响**

含 shear 的导入矩阵被近似分解为 TRS 后无法还原。重父级或 setter 分解失败时，如果先修改对象再失败，会留下 parent/matrix/history 不一致。

**根因**

把 TRS 当作存储真值，并以逐字段方式修改状态；分解既不是所有仿射矩阵的双射，也没有事务原子性。

**修改内容**

- `SceneTransform::local_matrix` 成为唯一真值。
- `trs()` 返回 optional，只在重建残差可接受时提供。
- `valid()` 整体检查 finite、affine 和 invertible。
- reparent 先计算/验证 `inverse(parent_world) * old_world`，成功后一次提交。
- invalid/non-affine/singular 候选整体拒绝。
- `.rscene` v4 保存完整矩阵；v1–v3 TRS 迁移到矩阵。

**解决后的行为**

剪切矩阵经过导入、reparent、undo/redo、保存和回读不丢失；失败操作保持原状态。

**测试证据**

- `document_tests` 和综合测试使用 XY/XZ shear，reparent、undo、redo、v4 roundtrip 最大残差 `<1e-5`。
- JSON 检查 `local_matrix` 为 4×4 行数组。
- 非仿射和奇异矩阵 setter 返回 false，原 local matrix 不变。

**兼容/限制**

旧 C++ 直接 TRS 字段写入不兼容；Inspector 对不可分解对象不显示可编辑 TRS，避免有损修改。

### 9. 可变指针绕过缓存、dirty 和历史

**现象与影响**

调用方通过 `objects()`、`find()`、`assets()` 或环境引用直接写内存时，不会统一更新 hash index、world cache、snapshot、revision、undo 和 saved identity，产生“画面已变但文档未 dirty”或“undo 无效”。

**根因**

聚合对象的可变引用跨越 `SceneDocument` 边界，文档无法观察写入。

**修改内容**

- 所有公共集合/查找/环境 getter 只读。
- mutable lookup 仅保留为 `SceneDocument` private。
- 名称、visible、locked、camera、light、matrix、TRS、material override、environment 均有类型化 setter。
- `mark_changed` 集中推进 revision、dirty 和缓存失效。
- `SceneEditTransaction` 负责 rollback、checkpoint 与 mergeable history。
- dirty 使用 saved history identity；从已保存节点分叉时不会误报 clean。

**解决后的行为**

不存在逃逸写入口；所有预览和最终修改都经过同一失效/历史路径。

**测试证据**

- compile-time API 只暴露 const view。
- `scene_tests` 连续两个相同 merge-key 拖拽只产生一个 undo step，undo 回原矩阵、redo 到最终矩阵。
- 综合 transaction 测试覆盖 preview revision、commit、cancel、dirty 和环境 revision 独立性。

**兼容/限制**

C++ 源码兼容明确不保留。事务当前不支持嵌套，嵌套请求会抛出 logic error。

### 10. 无效材质槽串到其他实例

**现象与影响**

旧 GPU accessor 先执行 `instance_material_offset + local_material_id`。当 local ID 为 `-1` 时，第二个实例可能得到前一实例材质表的最后一个元素；缺失材质随实例顺序随机串色。

**根因**

用裸 int 同时表达 missing 和 index，且在检查 sentinel 前做 offset arithmetic；正数越界也没有在 snapshot 边界拒绝。

**修改内容**

- host 使用 `MaterialSlot::missing/bound(uint32_t)`。
- snapshot build 只接受 `-1` 或 `[0, material_count)`。
- GPU 打包先检查 missing，再计算 instance offset。
- 全局 diagnostic material 固定为 index 0，missing 直接返回 0。
- flatten helper 为每实例 missing slot 生成明确洋红诊断材质。

**解决后的行为**

任意实例顺序下，missing 都稳定显示洋红诊断材质；不会引用相邻实例。正数越界在上传前报错。

**测试证据**

- `scene_tests` 创建两个实例共享 missing slot，全部 flattened triangles 的材质为洋红诊断材质。
- 正数越界 snapshot build 抛出 runtime error。
- CUDA contract/综合 instancing tests 通过。

**兼容/限制**

`-1` 只允许在 legacy primitive 输入边界表达 missing；进入 snapshot 后使用强类型，其他负数全部非法。

## 结构性整改与验证

### CPU Path 删除

删除：

- `src/render/pathtracer/pathtracer_renderer.*`
- `src/render/pathtracer/path_backend.*`
- CPU Path interactive branch 和 backend selector
- CLI `--threads`、`--path-backend`

因此原 CPU Path 每帧重建 BVH、创建线程和复制 Image 的产品性能问题随链路一起消失。CPU BVH 仅作为场景拾取的持久化加速结构保留。

### 测试拆分

当前 targets：

- `scene_tests`
- `sampling_tests`
- `document_tests`
- `opengl_contract_tests`
- `cuda_contract_tests`
- `renderer_tests`
- benchmark 配置下的 `benchmark_tests`、`benchmark_diagnostics_tests`

OpenGL contract 的属性、uniform、texture unit、SSBO binding 和 material field 由机器清单验证。

### CI

`.github/workflows/ci.yml` 增加：

- Windows no-CUDA Release build/test；
- Linux no-CUDA Release build/test；
- Linux Mesa + Xvfb OpenGL smoke；
- no-CUDA Path 请求必须显示 fallback 原因；
- Linux ASan/UBSan core tests。

CUDA GPU/interop/Sanitizer 不在普通 hosted runner 静默 skip；注释明确要求开发机或 self-hosted runner。

## 验证环境与命令

### 工具链/硬件

- OS：Windows，PowerShell
- CMake：4.4.0-rc3（项目最低要求 3.24）
- CUDA compiler：nvcc 13.3.73，build 13.3
- GPU：NVIDIA GeForce RTX 5080，16303 MiB
- NVIDIA driver：610.62
- Compute Sanitizer：2026.2.1

### 构建与测试

```powershell
cmake --preset no-cuda
cmake --build --preset no-cuda-release --parallel
ctest --preset no-cuda-release --output-on-failure

cmake --preset cuda-native
cmake --build --preset cuda-native-release --parallel
ctest --preset cuda-native-release --output-on-failure

cmake --preset cuda-native -DCMAKE_CUDA_ARCHITECTURES=120
```

结果：no-CUDA 6/6、CUDA 8/8。

额外冒烟：

- no-CUDA Viewer `--mode opengl --frames 1 --no-restore-last`：通过。
- no-CUDA Viewer `--mode path`：进程成功启动 OpenGL，控制台显示 `switched to OpenGL` 和 CUDA 不可用原因。
- CUDA Viewer Path：CUDA/OpenGL interop active，renderer/stream/interop device ID 一致。
- `renderer.exe --scene gradient_sphere --width 16 --height 16 --spp 1 --cuda-device 0`：PNG 输出通过。
- Linux Mesa/Xvfb：已进入 CI 配置，本 Windows 工作站未本地执行，不能把未运行的 CI job 记作通过。

### Compute Sanitizer 限制

尝试 memcheck 和 initcheck 时，普通裸跑的同一 CUDA test 可通过，但 sanitizer 注入后首个 kernel launch 返回 CUDA error 999；initcheck 报 application error，工具汇总为 0 sanitizer errors，但应用未完成。因此：

- 这不是“0 errors 通过”；
- 更像当前 RTX 5080 + driver 610.62 + CUDA 13.3 + Compute Sanitizer 2026.2.1 的注入/driver 兼容失败；
- 仍需在受支持的驱动/工具组合或 self-hosted GPU runner 复测；
- 普通 CUDA tests、device probe、interop 和离线 smoke 已通过，但不能替代 memcheck。

## San Miguel 性能

同一 RTX 5080、同一 San Miguel workload，基线 worktree 为 `75d1992`：

| 指标 | baseline | hardening | 变化 | 门槛 | 结果 |
|---|---:|---:|---:|---:|---|
| CUDA full median | 210.593 ms | 216.558 ms | +2.83% | 回退 ≤5% | 通过 |
| interaction median | 10.7114 ms | 10.0695 ms | -5.99% | ≤16.7 ms | 通过 |
| interaction P95 | 10.9746 ms | 10.5518 ms | -3.85% | ≤20 ms | 通过 |
| native quantum median | 12.0626 ms | 12.1211 ms | +0.48% | 观察项 | 通过 |

证据：

- `benchmarks/results/baseline-75d1992-hot-cuda/summary.md`
- `benchmarks/results/renderer-hardening-acceptance2-cuda/summary.md`

两份历史 raw report 的 compatibility key 因架构/case metadata 迁移不同，runner 没有自动宣称可比；上表是在同一机器、同一资产、同一分辨率和 workload 的配对命令上手工计算 delta。实例拖拽的“不上传/只 refit”由 `CudaPathStatistics` contract tests 单独证明。

## 已知后续工作与明确不在范围

以下没有在本分支实现，也没有在上文宣称解决：

- CUDA texture mip chain 和显式 LOD；
- 正交相机 Path 预览；
- animation、skin、morph target；
- 对现有 weighted blended OIT 的替换或精确排序透明；
- transmission、clearcoat、sheen、volume 等新 PBR 工作流；
- 在兼容 Compute Sanitizer 环境取得有效 memcheck 结果。

旧 `.rscene` v1–v3 和 session v1 提供读取迁移；旧 C++ API、CPU Path executable behavior 和 backend selector 不保留兼容。

## 2026-08-09 用户资源回归修复

本节记录硬化完成后由真实 glTF 资产和 Viewer 交互暴露的四项回归。它们在同一分支修复；没有改写上面的历史验收结论。

- 分支：`codex/renderer-hardening`
- 回归修复提交：本节所在的本地提交
- 发布范围：未 push，未创建 PR

### 11. 合法的毫米级 glTF 节点变换被拒绝

**现象与影响**

加载 `Computer Graphics Archive/lamborghini_revuelto/scene.gltf` 时，导入在节点 `xshell_6175_4` 处失败，错误为“node transform is not a finite invertible affine matrix”。该节点使用常见的均匀 `0.001` 单位缩放，矩阵本身有限、仿射且可逆。

**根因**

`SceneTransform::valid()` 用线性部分行列式的绝对值 `1e-6` 作为可逆阈值。均匀 `0.001` 缩放的行列式是 `1e-9`，因此仅因为模型以毫米为单位就被误判；这个阈值还把“单位大小”与“矩阵是否奇异”错误地绑定在一起。

**修改内容**

- 保留有限值与仿射末行校验；
- 对 3×3 线性部分改用 Eigen full-pivot LU 的相对秩判断；
- 增加 `0.001` 均匀缩放可接受、真正奇异矩阵仍拒绝的场景测试；
- 在 glTF 静态场景 fixture 中增加带毫米级父节点的 mesh，并验证文档导入和扁平化结果。

**解决后的行为**

尺度很小但条件良好的模型可正常导入；非有限、非仿射和秩不足的变换仍会被整体拒绝，不会污染文档状态。

**验证证据**

- 原始 Lamborghini glTF 在 no-CUDA OpenGL Viewer 中成功加载并渲染 1 帧；
- 同一资产在 CUDA Path + OpenGL interop 中成功加载并发布 1 帧，853 个 BLAS 完成构建；
- `scene_tests` 和 `renderer_tests` 覆盖小尺度与奇异矩阵边界。

**兼容/剩余限制**

对极端病态、数值上不可稳定求逆的矩阵仍可拒绝；这与“有限、可逆且可安全参与重父级/拾取”的场景契约一致。

### 12. OpenGL Gizmo 拖拽没有实时反映到画面

**现象与影响**

场景文档在拖拽期间已经更新对象矩阵、revision 和 dirty 状态，但 OpenGL 画面仍显示旧位置、旋转或大小，直到另一类场景变更或后端重置发生。

**根因**

OpenGL 当前消费的是由快照扁平化得到的 world-space 顶点。实例矩阵变化会改变这批顶点，但 `OpenGlRasterRenderer::sync_scene()` 只在 `Geometry` 或 `MaterialBindings` 变化时重传几何，错误地忽略了 `InstanceTransforms`。

**修改内容**

- 建立可测试的 `open_gl_requires_geometry_upload()` 同步策略；
- 把 `InstanceTransforms` 纳入 OpenGL 几何上传条件；
- 实例变换变化时同时重算 scene radius，保证相机裁剪和辅助尺度使用新边界；
- 后端直接比较快照 revision，拖拽每次事务预览产生的 transform revision 都会在下一帧被消费。

**解决后的行为**

移动、旋转和缩放的每次预览更新都会在下一帧重新生成并上传对应的扁平 OpenGL 几何，不再需要切换显示模式来刷新。

**验证证据**

- `opengl_contract_tests` 明确验证 transform 变化需要几何上传，而材质参数/灯光变化不会误触发该上传；
- no-CUDA OpenGL Release 编译和真实资产 Viewer smoke 通过；
- `scene_tests` 验证每次矩阵 setter 更新可被 snapshot revision 检出；
- Windows Viewer 实际拖动 X 平移手柄后，OpenGL 视口同帧更新，Inspector 的 Translation X 从 `0.000` 变为 `0.394`；实际拖动 Y 缩放手柄后视口同帧更新，Scale Y 从 `1.000` 变为 `2.345`。

**兼容/剩余限制**

这是当前扁平 OpenGL 后端的正确性修复，变换仍会重传 world-space 顶点。后续若 OpenGL 改为真正的实例 buffer，可把该路径优化为只更新矩阵，不改变本次快照/revision 契约。

### 13. Rotate/Scale Gizmo 操作状态串扰

**现象与影响**

从 Move 切到 Rotate 后可能仍显示 Scale 风格操作框；切到 Scale 时手柄状态又可能沿用前一个操作，表现为无响应。

**根因**

所有对象和三种操作共用 ImGuizmo 默认 ID。ImGuizmo 会按 ID 保存 active/hovered handle 状态，因此切换 operation 或对象时，上一操作的内部状态可能被新调用继承。嵌套三元表达式也让 operation 映射难以检查。

**修改内容**

- 用显式 `switch` 固定 `0=Translate`、`1=Rotate`、`2=Scale`；
- 用 `ObjectId + operation` 建立成对的 ImGuizmo `PushID/PopID` 作用域；
- hovered 查询限定到当前 operation，未知状态恢复为 Move。

**解决后的行为**

切换对象或操作会得到独立的 gizmo 状态；Rotate 和 Scale 的显示与命中逻辑不再串用前一次操作。

**验证证据**

- no-CUDA 与 CUDA Viewer 均通过包含 ImGuizmo 修改的 Release 编译；
- Windows Viewer 交互检查中，Move、Rotate、Scale 单选状态可连续切换；Rotate 显示独立旋转圆环，Scale 显示轴缩放手柄并可完成上述 Y 缩放拖拽。

**兼容/剩余限制**

快捷键和 Scene 面板的 0/1/2 状态值保持不变；场景文件和 session 格式不受影响。

### 14. 切换 Scene 后仍显示上一个 Scene

**现象与影响**

从一个场景文档切换到另一个文档时，画面可能继续显示旧场景；切换 OpenGL/Path 模式触发后端 reset 后才显示正确内容。Path 的 progressive accumulation 也存在把两个不同场景当成同一辐射状态的风险。

**根因**

revision 只在单个 `SceneDocument` 内单调。两个刚加载的文档可以拥有完全相同的八域 revision；后端仅比较 revision 数字，因而得到 `SceneChange::None`。显示模式切换会显式 reset，所以看起来像是“切模式才修好”。

**修改内容**

- 为每个 `SceneDocument`/独立快照谱系分配非零、进程内唯一的 `source_id`；
- 增加统一 `scene_changes_for_snapshot()`：source 改变必定返回 `All`，同一 source 才比较八域 revision；
- versioned snapshot 以 revision 为权威，忽略可能过期的 UI hint；只有旧的无 source、无 revision 快照保留 hint fallback；
- OpenGL Viewer backend、Path Viewer backend、CUDA scene upload 和 `ProgressiveRenderKey` 全部纳入 source identity。

**解决后的行为**

即使两个场景的 revision 数字逐项相同，切换文档也会完整同步并显示新场景；同一文档的普通下一帧不会误做全量更新；Path 会清空旧场景累积。

**验证证据**

- `scene_tests` 验证不同文档 source ID 不同、相同 revisions 仍得到 `All`、同一 source 得到 `None`；
- `cuda_contract_tests` 验证只改变 source ID 就会把 progressive samples 重置为 1；
- CUDA 8/8、no-CUDA 6/6 全部通过。

**兼容/剩余限制**

`source_id` 只用于进程内快照谱系，不写入 `.rscene`，因此不改变文档格式。旧的手工构造 C++ 快照仍可用 `source_id=0` 和 `SceneChangeSet` hint 工作，但正式 Viewer 路径不依赖 UI 手工维护变更集合。

### 本轮验证命令与结果

```powershell
cmake --build --preset no-cuda-release --parallel
ctest --preset no-cuda-release --output-on-failure

cmake --build --preset cuda-native-release --parallel
ctest --preset cuda-native-release --output-on-failure

build\no-cuda\bin\viewer.exe --scene asset `
  --asset '.\Computer Graphics Archive\lamborghini_revuelto\scene.gltf' `
  --mode opengl --frames 1 --width 320 --height 180 --no-restore-last

build\cuda-native\bin\viewer.exe --scene asset `
  --asset '.\Computer Graphics Archive\lamborghini_revuelto\scene.gltf' `
  --mode path --frames 1 --width 320 --height 180 --no-restore-last
```

结果：no-CUDA 6/6、CUDA 8/8；真实资产 OpenGL 与 CUDA Path smoke 均通过，CUDA/OpenGL interop 为 active。
