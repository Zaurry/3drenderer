# 3D Renderer 设计问题审查报告

> 审查日期：2026-08-11（以仓库当前工作区为准）
> 范围：`src/`、`shaders/`、`tests/`、`tools/`、`benchmarks/`、`CMakeLists.txt`、`CMakePresets.json`、`.github/`、`docs/`
> 方式：只读代码审查（未修改任何代码），逐文件核对实现并交叉验证引用关系
> 严重级别：🔴 高（正确性/可维护性/性能的主要风险）、🟡 中（明显设计缺陷）、🟢 低（局部问题）

---

## 0. 总览：最重要的 15 个问题

| # | 问题 | 级别 | 关键证据 |
|---|---|---|---|
| 1 | 编辑事务 `begin_edit` 深拷贝**整条撤销历史**，undo 为全量场景快照 | 🔴 | `src/scene/scene_document.cpp:2139-2175, 2267-2301` |
| 2 | `SceneMaterialOverride` 是 `Material` 的损副本，5 处手工同步、静默丢失 PBR/纹理变换字段 | 🔴 | `src/scene/scene_document.h:78-101` vs `src/scene/material.h:21-58` |
| 3 | BRDF/采样逻辑三份独立实现（pbr.cpp / CUDA / GLSL），`power_heuristic` 已经漂移 | 🔴 | `src/render/pbr.cpp:206-210` vs `src/render/pathtracer/cuda_pathtracer.cu:859-869` vs `shaders/opengl/raster.frag:188-250` |
| 4 | 主机→设备结构体用 ~50 字段的**位置聚合初始化**打包，无共享头文件、无布局断言 | 🔴 | `src/render/pathtracer/cuda_pathtracer.cu:117-296, 4753-4843` |
| 5 | CUDA 路径追踪器 7,749 行单文件；OpenGL 渲染器 3,222 行单类，无模块/测试接缝 | 🔴 | `cuda_pathtracer.cu:28-7749`；`opengl_raster_renderer.cpp:856-3149` |
| 6 | 渐进累积重置策略散布在 UI、主循环、渲染器三层（渲染器内部同一条件写了两遍） | 🔴 | `viewer_ui.h:118-129`；`viewer_main.cpp:1065-1074,1190-1203`；`cuda_pathtracer.cu:7110-7174` |
| 7 | 渲染全部在主线程；interop 不可用时回退路径阻塞 `cudaStreamSynchronize` | 🔴 | `viewer_main.cpp:1204-1224`；`viewer_render_backend.cpp:207-221`；`cuda_pathtracer.cu:6319-6362` |
| 8 | `viewer_main.cpp` 的 `main()` ≈825 行 god function；`ViewerUi::draw()` ≈1,807 行 13 参数 | 🔴 | `src/viewer_main.cpp:490-1314`；`src/interactive/viewer_ui.cpp:442-2248` |
| 9 | `mutable` 缓存由 `const` 方法修改：线程不安全，"不可变快照"契约泄漏引用 | 🔴 | `src/scene/scene_document.h:294-301`；`scene_document.cpp:743,1594,1756,2047-2050` |
| 10 | CUDA 相关测试在无 GPU 时**静默跳过仍报绿**，与 README/CI 的声明矛盾 | 🔴 | `tests/cuda_contract_tests.cpp:14-236`；`tests/renderer_tests.cpp:1606-1609`；`README.md:180` |
| 11 | 4 个 FetchContent 依赖无 `URL_HASH`；`renderer_core_diagnostics` 把整个核心（含 278KB .cu）重复编译一遍 | 🔴 | `CMakeLists.txt:81-148, 274-279, 403-428` |
| 12 | 测试框架只有 `exit(1)` 断言；`renderer_tests.cpp` 5,874 行混合 10+ 领域；`opengl_contract_tests` 实际是 grep 文本 | 🔴 | `tests/test_framework.h:8-19`；`tests/opengl_contract_tests.cpp:157-248` |
| 13 | `learning/`（GAMES202 课程作业）5,085 个文件、约 390MB 全部入库（占仓库跟踪文件 97%） | 🔴 | `git ls-files`；`.gitignore` 未包含 `learning/` |
| 14 | `ViewerUiActions` 22 个布尔标志的"穷人事件系统"，其中 2 个标志无人读取 | 🟡 | `src/interactive/viewer_ui.h:94-130`；`viewer_ui.cpp:814,820,831,836` |
| 15 | 两层会话抽象三重委托；GLSL↔C++ 契约字符串化且大部分未被使用 | 🟡 | `interactive_render_session.h:116-128`；`opengl_shader_contract.h` vs `opengl_raster_renderer.cpp:104-464` |

---

## 1. 数据模型与表示

### 1.1 🔴 `SceneMaterialOverride` 是 `Material` 的损副本，需要 5 处手工同步
**位置**：`src/scene/scene_document.h:78-101` vs `src/scene/material.h:21-58`；`src/scene/scene_document.cpp:208-233, 235-262, 264-308, 2471-2506, 2817-2883`

`SceneMaterialOverride` 复制了 `Material` 的 11 个标量字段，但**丢失**了 `pbr_workflow`、`specular_color/factor/glossiness`、3 个 specular 纹理 id 和全部 8 个 `TextureTransform`。字段映射在 5 个位置手工维护（校验、从 Material 转换、应用、序列化、反序列化）。给 `Material` 增加一个字段会静默破坏材质覆盖；并且覆盖用倒置的 `use_*_texture` 布尔表达纹理，`apply_material_override` 只会把 `texture_id` 置 -1（`scene_document.cpp:284-307`），覆盖永远无法"重新启用或改指"纹理——这是个单向门。

**影响**：编辑器里修改材质覆盖会丢 specular-glossiness 工作流的全部参数；纹理开关语义与名字相反；每次扩展材质都要碰 5 处代码。

**建议**：让覆盖直接持有完整 `Material` + 槽位，或用稀疏字段表；参照 `kMaterialTextureIds`（`scene_document.cpp:36-48` 的成员指针表）用一张表生成所有拷贝/校验/序列化代码。

### 1.2 🟡 `SceneObject` 是"大而全"结构体而非 tagged union
**位置**：`src/scene/scene_document.h:103-131`；类型校验散落在 `scene_document.cpp:843-851`

每个对象（无论 Group/Mesh/Camera/灯光）都携带约 15 个灯光浮点字段 + 6 个相机字段 + `std::vector<SceneMaterialOverride>` 头 + 完整 `Mat4`。类型-字段合法性只能靠运行时分支检查，`Group` 携带 `area_width` 这类非法组合可以被构造并被序列化。

**影响**：每个对象的内存浪费；语义合法性无法由类型系统保证；序列化/UI/快照三处都要按类型分派。

**建议**：改为 `std::variant<GroupData, MeshData, PointLightData, ...>` 判别联合。

### 1.3 🟡 `SceneLightProperties` / `SceneCameraProperties` 重复 `SceneObject` 的字段子集
**位置**：`src/scene/scene_document.h:138-160` vs `103-131`；`scene_document.cpp:813-908` 手工逐字段拷贝

属性结构体几乎逐字复制对象字段，setter 再逐字段搬运。新增灯光属性要同步改 `SceneObject`、属性结构体、setter、快照转换（`1875-1955`）、序列化/反序列化（`2452-2462, 2793-2807`）五处以上。

**建议**：从同一份共享的灯光/相机组件类型派生两个结构体，消除手工拷贝。

### 1.4 🟡 每个实例复制完整材质表（快照内存随实例数膨胀）
**位置**：`src/scene/instanced_scene.h:69`；`src/scene/scene_document.cpp:1977-1992`；`src/scene/instanced_scene.cpp:283-290`

`RenderSceneInstanceSnapshot.materials` 是完整的 `std::vector<Material>`，`ensure_render_scene_snapshot` 为每个实例拷贝整张材质表并重映射纹理 id；`flatten_render_scene_snapshot` 再拷贝一次并为每实例新建 `diagnostic_material()`。N 个同资产实例 = N 份相同的材质表，只为支持"每对象覆盖"。

**影响**：大规模实例化（如植被、粒子墙）时快照内存与构建时间线性膨胀，与"共享资产"的设计意图矛盾。

**建议**：资产级共享材质数组 + 每实例小的覆盖增量列表，仅在 flatten 时物化。

### 1.5 🟢 环境/灯光状态在 `Scene`、`RenderSceneSnapshot`、`SceneDocument::State` 三处逐字重复
**位置**：`src/scene/scene.h:22-30`；`src/scene/instanced_scene.h:80-88`；`src/scene/scene_document.h:284-289`

环境 5 个字段和 4 个灯光 vector 在三种表示中重复，新增一个环境参数要改三处 + 快照装配（`scene_document.cpp:1764-1768`）+ flatten（`instanced_scene.cpp:240-253`）。

**建议**：提取共享 `EnvironmentState` / `LightSet` 结构体。

### 1.6 🟢 程序球几何存了三份
**位置**：`src/scene/scene_document.cpp:559-609`；`src/scene/scene_document.h:63-76`

纯球场景同时持有：细分后的三角形（`local_scene`）、未细分的 `render_geometry`（`spheres.clear()` 后为空）、`procedural_spheres` 列表——同一几何三个副本，快照逻辑要靠"render_geometry 为空则回退 procedural_spheres"的分支（`scene_document.cpp:1803-1805, 1996-2040`）才能工作。

**建议**：只保留规范化实例 primitive 一种表示。

### 1.7 🟡 正交相机在数据模型中存在，但没有任何后端能渲染它
**位置**：`src/scene/scene_document.h:43,123`；运行时 `Camera` 只有透视构造（`src/scene/camera.h:10-15`）；`src/interactive/viewer_ui.cpp:1654-1655` 显示 "Orthographic (perspective preview)"

文档可创建/导入/保存正交相机（glTF 正交相机在 `gltf_loader.cpp:801` 被解析），但运行时 `Camera` 类没有正交投影，Viewer 明确只做透视预览。数据模型承诺了功能而渲染层没有实现。

**影响**：用户可编辑一个永远无法被任何后端真实渲染的属性，属于"假功能"。

**建议**：要么在 Camera/渲染器实现正交，要么在导入时显式降级并警告。

### 1.8 🟢 死代码与误导性残留
**位置**：`src/core/random.h:37-52`（`Random` 类无任何引用）；`src/scene/obj_loader.cpp:91`（`load_obj_mesh` 无调用者）；`src/scene/texture.cpp:360`（`sample_material_base_color` 无调用者）；`src/scene/scene_document.cpp:1661-1663`（`world_matrix_recursive` 忽略 depth 参数且无调用者）；`src/render/renderer.h:10-12`（`ExecutionBackend` 只有一个枚举值 `Cuda`）；`SceneMeshAsset::geometry_revision` 恒为 1 却被 CUDA BLAS 缓存作为键（`cuda_pathtracer.cu:5552`）

**影响**：误导后续开发者依赖不存在的语义（geometry_revision 尤其危险）。

**建议**：删除死代码；要么实现要么移除 `geometry_revision` 管线。

### 1.9 🟡 `Material` 双轨字段：legacy `Diffuse` 与 PBR 并存、语义按来源漂移
**位置**：`src/scene/material.h:39-48`；`material_evaluator.cpp:120-128` 优先 `base_color_texture_id` 回退 `diffuse_texture_id`；OBJ loader 只写 `diffuse_*`（`scene_asset_loader.cpp:252`），glTF loader 只写 `base_color_*`（`gltf_loader.cpp:389`）；默认 `two_sided = true`（`material.h:35`）与灯光侧的 `two_sided = false`（`light.h:44`）不一致且命名分叉为 `two_sided`/`light_two_sided`

**建议**：加载时统一到一套 base-color 字段；统一单面默认值（glTF 规范默认单面）与命名。

---

## 2. SceneDocument：状态、撤销与缓存

### 2.1 🔴 全量拷贝撤销 + 事务备份整条历史
**位置**：`src/scene/scene_document.h:282-290,307`；`src/scene/scene_document.cpp:28, 2139-2175, 2267-2301`

`history_` 是 `std::vector<State>`，`State` 内含完整 `std::vector<SceneObject>`（含名字、矩阵、材质覆盖 vector）。`checkpoint()` 每次离散操作深拷贝整棵树，上限 `kMaximumHistory = 256`。更糟的是 `SceneEditTransaction` 的 `Backup` 在 `begin_edit` 时拷贝 `state_`、`assets_` **和整个 `history_` vector**（`scene_document.cpp:2162-2166`）——启动一次事务的成本是 O(场景大小 × 历史深度)。gizmo 拖动等高频操作每次都走这条路。

**影响**：5 万对象的场景单次 checkpoint 数十 MB、驻留最多约 256 倍；每次 gizmo 拖动产生巨额拷贝；大场景编辑明显卡顿。

**建议**：命令对象 + 结构性共享（persistent vector / copy-on-write）；事务只备份增量。

### 2.2 🟡 undo/redo 不恢复 `assets_`，撤销导入泄漏孤儿资产
**位置**：`src/scene/scene_document.cpp:2267-2301`（checkpoint 只存 state_）vs `2303-2327`（undo/redo 只恢复 state_）；`Backup::assets` 只在事务路径存在（`2141,2239`）

撤销一次导入会删掉 mesh 对象，但 `SceneMeshAsset`（含几何、BVH、拾取加速结构）永久留在 `assets_`；`load_asset` 的路径去重（`1036-1042`）甚至会把它复活。正确性取决于"所有导入都包在事务里"这一未强制的不变量。

**建议**：撤销状态中包含资产引用集合/引用计数，或让资产生命周期跟随引用它的对象。

### 2.3 🔴 `mutable` 缓存由 `const` 方法修改，快照引用可被下一次编辑作废
**位置**：`src/scene/scene_document.h:294-301`；`scene_document.cpp:743,1594,1756,2047-2050`

`object_indices_`、`children_by_parent_`、`world_matrices_`、`world_bounds_`、`render_scene_snapshot_` 和三个 dirty 标志全是 `mutable`，由 `const` 方法重建。`render_scene_snapshot()` 返回指向共享可变缓存的 `const&`，OpenGL/CUDA/拾取后端都消费它：并发读 + 编辑即数据竞争；返回的引用也可能在下一次编辑后被改写——"不可变快照"契约不成立。没有任何锁或双缓冲。

**影响**：一旦未来引入后台线程（如资产异步加载回调）即触发数据竞争；后端持有跨帧引用存在被静默改写风险。

**建议**：显式构建快照并返回共享/不可变句柄，或 `std::shared_mutex` 保护并强制单线程消费。

### 2.4 🟡 `find_mutable` 的 const_cast 逃逸口绕过 revision/dirty 簿记
**位置**：`src/scene/scene_document.cpp:738-741`；直接改字段的调用点 `625-693, 1134-1135`

`find_mutable` 把 const 索引查到的指针 `const_cast` 成可变指针返回，且不置任何 dirty 标志。调用方直接改字段后依赖"事后批量 `mark_changed`"补账——忘一次就留下 stale 的 revisions/快照。返回指针在下次 `push_back` 重分配后还会悬空。

**建议**：所有修改走感知 `mark_changed` 的访问器，或返回插入即失效的代理/迭代器。

### 2.5 🟡 `find_asset` 在快照热路径上线性扫描 O(N·M)
**位置**：`src/scene/scene_document.cpp:1022-1028`；调用点 `1786, 1960`（每个 mesh 对象两次）

`ensure_render_scene_snapshot` 对每个 mesh 对象调用 `find_asset` 两次，而快照在每次编辑后重建。N 对象 × M 资产的场景是 O(N·M) 每次构建。

**建议**：与对象索引一样维护 `unordered_map<AssetId, index>` 惰性重建。

### 2.6 🟡 层级深度上限只在两个检查点生效，超限静默返回错误答案
**位置**：`src/scene/scene_document.cpp:27, 1399-1421, 1622-1627, 1661-1663`

`kMaximumHierarchyDepth = 1024` 只在 `is_descendant`/`is_effectively_visible` 检查：超深对象被当作"隐藏"（返回 false），制造 >1024 层环的重父级能通过守卫、随后在 `rebuild_spatial_cache` 以 throw 爆出。实际的 world 矩阵计算走迭代路径，`world_matrix_recursive` 的 depth 参数被忽略。

**建议**：在迭代遍历里强制深度上限，超深返回显式错误而不是静默误分类。

### 2.7 🟢 `merge_key` 合并靠手写游标簿记；`import_path` 为回滚先全量拷贝状态
**位置**：`src/scene/scene_document.cpp:2203-2231, 2267-2301, 2337-2345`；`1242-1248, 1330-1343`

`commit` 在 merge key 匹配时覆写历史顶端条目，`saved_cursor_` 在 4 个方法里各自做 `optional<size_t>` 算术修正；`import_path` 为异常回滚先 `const State previous_state = state_` 全量深拷贝，成功后再 checkpoint 拷贝一次。

**建议**：历史修改收敛到单一函数维护游标/saved 不变量（加断言）；导入用局部 staging + move 合并，避免快照拷贝。

### 2.8 🟡 错误处理契约四混：bool / throw / 静默 clamp / warnings
**位置**：`src/scene/scene_document.cpp:813-908`（bool + clamp）、`2060-2062`（非有限颜色静默忽略）、`2072-2073`（throw）；`scene_asset_loader.cpp:217-227`（纹理解码失败吞进 warnings）

同一类里 setter 返回 bool、环境颜色静默忽略非法输入、环境贴图加载 throw、灯光属性越界被 clamp——调用方无法预知哪种契约适用。

**建议**：统一为结果/状态返回，warnings 只留给非致命诊断；校验集中到结构体旁边。

### 2.9 🟢 `warnings_` 只增不清，去重仅限单词加载器调用
**位置**：`src/scene/scene_document.h:311`；`scene_asset_loader.cpp:148-156`；`scene_document.cpp:2149`（还被整段拷进事务 Backup）

重复导入/加载-恢复会重复追加警告；向量终生单调增长且参与每次事务备份拷贝。

**建议**：以规范化路径为键做文档级去重并设上限。

---

## 3. 渲染后端（CUDA / OpenGL）

### 3.1 🔴 CUDA 路径追踪器是 7,749 行单文件，无模块边界
**位置**：`src/render/pathtracer/cuda_pathtracer.cu:28-7749`（匿名命名空间内：设备数学 62-91、`GpuBvh4Builder` 494-723、内核 3640-4382、buffer 封装 4384-4524、`CudaSceneStorage` 4597-5922、`CudaFrameStorage` 5924-6998、`Impl` 7038-7670）；公开面仅 `cuda_pathtracer.h:85-122` 的 pimpl

所有 `D*` 结构体文件局部，其他代码无法复用或测试；任何改动都重编整个约 280KB 的 CUDA 翻译单元；设备内核只能通过带 GPU 的完整渲染器测试。

**建议**：拆成 `cuda_device_types.cuh`、`cuda_brdf.cuh`、`cuda_bvh.cuh/.cu`、`cuda_scene_storage.cu`、`cuda_wavefront.cu`，把 `D*` 与 `__device__` 数学放进 `.cuh` 供 host/device/测试共享。

### 3.2 🔴 OpenGL 渲染器 3,222 行单类、约 40 个裸 `GLuint` 成员
**位置**：`src/render/opengl/opengl_raster_renderer.cpp:856-3149`（Impl），成员 `1501-1568`，`render()` 995-1338（约 340 行顺序流水），`render_ambient_occlusion` 2668-2860，`prepare_shadow_maps` 2048-2248，`ensure_output` 2862-3042

Shadow/AO/OIT/合成全部是 Impl 的平铺方法，没有 `AoPass`/`ShadowPass`/`OitPass` 子对象；没有 GL 上下文就无法测试任何技巧接线。

**建议**：拆成各自拥有 FBO/纹理并带 `resize()/release()` 的 Pass 类，`Impl::render` 只做编排。

### 3.3 🔴 BRDF/采样三份独立实现，且 C++ 副本不作为有效交叉验证
**位置**：`src/render/pbr.cpp:100-210`；`cuda_pathtracer.cu:2933-3014, 3035-3060`；`shaders/opengl/raster.frag:188-250, 653-729`

GGX/`fresnel_schlick`/`smith_g1`/`sample_visible_ggx`/`evaluate_pbr` 在 C++、CUDA、GLSL 各写一份。C++ 版产品代码中无任何调用者，仅被 `tests/renderer_tests.cpp:5012-5077` 自测（只验证自身 pdf 一致性，**不与 CUDA/GLSL 对比**），所以它不是有效的防漂移 oracle。

### 3.4 🟡 `power_heuristic` 已实际漂移
**位置**：`src/render/pbr.cpp:206-210` vs `cuda_pathtracer.cu:859-869`

C++ 版：`a²/max(a²+b², 1e-20)`，无 NaN/0 处理；CUDA 版：非正/非有限 pdf 分别返回 0.0/1.0，否则直接除。pdf 恰为 0 或 NaN 时两版结果不同，且没有任何测试能发现。

**建议**：MIS 权重收敛为单一定义（边界行为一致）并在测试中断言两版等价。

### 3.5 🔴 主机→设备结构体打包：位置聚合初始化、无共享头、无布局断言
**位置**：`cuda_pathtracer.cu:117-296`（`DMaterial`/`DTexture`/`DScene`/`DInstance` 文件局部）；`4753-4843`（`pack_material`/`pack_instance` 约 50 字段 / 9 字段位置聚合初始化）

新增/重排 `DMaterial` 字段而不同步改 `pack_material` 会**静默交换值**（如 texcoord int 落到 rotation float）。没有任何 `static_assert(sizeof/offsetof)`，也没有与 `Material`/GLSL uniform block 共享的头。每个材质特性要手加到 `Material`、`DMaterial`、`pack_material`、`UniformLocations`+`find_uniforms`、`raster.frag` 五处。

**建议**：从一个材质 schema 生成设备结构体与打包器（以及 GLSL uniform/SSBO block），或至少把 `D*` 移进共享 `.cuh` 并对 `Material`/`DMaterial`/SSBO 做布局静态断言。

### 3.6 🟡 GLSL↔C++ 契约字符串化、不被消费、无校验
**位置**：`src/render/opengl/opengl_shader_contract.h:13-71` vs `opengl_raster_renderer.cpp:104-464`（约 70 个独立 `glGetUniformLocation(program, "u_…")`，契约数组几乎未被使用），另有 sky/shadow/composite/AO 程序的内联查询（`1056, 1930, 1999, 2758`）

重命名 `raster.frag` 里的 uniform 只会得到 `-1` 且 `set_uniform` 静默 no-op，启动时没有任何"契约中的 uniform 必须存在"的检查。

**建议**：让 `find_uniforms` 与纹理绑定由契约数组驱动；shader 编译成功后校验必需 uniform 全部解析（或显式 `layout(location=…)`）。

### 3.7 🟡 魔法数组大小与 clamp 范围在 C++ 与 GLSL 各写一遍
**位置**：shadow slot 数 32（`opengl_raster_renderer.cpp:1155,1164,2170` vs `raster.frag:84-88,427,514`）；PCSS 采样上限 64（C++ clamp `1221-1223` vs GLSL 硬编码 `463,489,553,582`）；阴影分辨率 clamp `128..4096`（`2182`）

两语言常量可静默不一致，纯字符串化配置。

**建议**：共享常量进契约头并从它生成/预处理 shader 源，或启动时断言一致。

### 3.8 🟡 四套几何遍历表示并存
**位置**：CPU `Bvh`（`src/acceleration/bvh.h:21-29`）+ `SceneIntersector`（仅拾取用，`scene_document.h:72`）；CUDA 自己的 `intersect_triangle_geometry`（`cuda_pathtracer.cu:1078-1111`）+ **另一个**主机侧 `GpuBvh4Builder`（`494-723`），而 `Bvh::build_layout`（`bvh.h:25` 注明 "for GPU upload"）**未被 CUDA 使用**；GLSL 走硬件光栅化

两套 CPU BVH 构建器 + 一套设备遍历 + 硬件光栅化。

**建议**：CUDA 路径复用 `Bvh::build_layout` 输出，或把 BVH4 构建器移入 `acceleration/` 作为唯一 GPU 构建器，节点布局经公共头共享。

### 3.9 🟡 射线原点偏移、alpha 模式判定、亮度权重、范围衰减等小原语多处复制
**位置**：`offset_ray_origin`（`scene_intersector.cpp:156-179`）vs 设备 `offset_origin`（`cuda_pathtracer.cu:2878-2896`）；alpha 模式规则在 `scene_intersector.cpp:28-35`、`cuda_pathtracer.cu:1177-1181, 2409-2416`、`opengl_raster_renderer.cpp:1925-1929, 2582-2586` 四处；亮度权重 `0.2126/0.7152/0.0722` 在 `pbr.cpp:42`、`cuda_pathtracer.cu:2956,4883-4884`、`raster.frag:134`、`ao_gbuffer.frag:40`、`opengl_raster_renderer.cpp:276` 五处；`punctual_range_attenuation` 在 `cuda_pathtracer.cu:2923-2931` 与 `raster.frag:207-214`

**建议**：提取进共享头（C++/host）与手工同步或生成的 GLSL/CUDA include，并用单元测试钉死数值行为。

### 3.10 🟡 SSAO/GTAO/PCSS 数学在 C++ 与 GLSL 镜像维护
**位置**：`src/render/opengl/opengl_ao_math.h:11-143` vs `shaders/opengl/ambient_occlusion.frag:32-46,244-268`、`raster.frag:621-651`；`opengl_shadow_math.h:128-148` vs `raster.frag:477-487`

C++ 副本用于 CPU 阴影拟合/诊断，但没有任何机制保证两语言一致；数值微调必须改两个语言。

**建议**：选一边为真值生成另一边，或加 shader/C++ 数值一致性测试。

### 3.11 🔴 渐进累积重置策略三层重复，渲染器内部同条件写两遍
**位置**：`viewer_ui.h:118-129`（`resets_path_accumulation()` 把渲染策略编码进 DTO）；`viewer_main.cpp:1065-1074, 1190-1203`；`cuda_pathtracer.cu:7110-7174`（`reset_accumulation` 7120-7125 与 `interaction_changed` 7169-7174 **同一函数内两遍**），另加浮点 `ProgressiveRenderKey`（7306-7374）

"什么会重置累积"的判定分散在 UI、主循环、渲染器，三者必须手工同步；渲染器内部还重复计算。

**建议**：让渲染器独占 `ProgressiveRenderKey` 决策，UI 只上报"为什么变了"；合并两个相同条件块。

### 3.12 🟢 `ProgressiveRenderKey` 用浮点 `==` 比较相机向量
**位置**：`cuda_pathtracer.cu:7306-7373`

任何亚像素漂移（空转 orbit 的 epsilon）都重置累积；NaN/denormal 相机分量会永久不等导致每帧重置。与 UI 的 `camera_changed` 布尔冗余。

**建议**：键中量化视角参数，或相机从键中移除、依赖 `camera_changed`，并加 NaN 安全。

### 3.13 🟡 交互状态机约 20 个标志/整数、ad hoc 分支与约 10 个魔法阈值
**位置**：`cuda_pathtracer.cu:7376-7381`（`kIdleFramesBeforeNative=8` 等）、`7632-7669`（约 40 个成员）、`7151-7204, 7441-7630`（preview/native-quantum/full 三态 if/else 演进）

没有 `enum class Phase` + 转移表，preview→native→full 的演进靠散落的标志改写表达。

**建议**：显式相位状态机 + 小型转移函数；调参常量收敛到命名配置结构。

### 3.14 🟡 wavefront arena 是手排布字节缓冲 + 手工偏移算术 + 魔法错误码
**位置**：`cuda_pathtracer.cu:6492-6596`（`reserve_region` 手工对齐/sizeof，`reinterpret_cast` 出 `DVec3*`/`DPathState*`/`DShadowTask*` 等）、`3971-3994`（手工 warp compaction）、`6929-6960`（错误码 1,3,5,6,7 → 字符串）

增删一个 wavefront buffer 要手工改布局。

**建议**：struct-of-arrays arena 类型（一处 `offsetof`/`alignas` 计算），错误码改枚举。

### 3.15 🟡 同一管线三套编译期变体
**位置**：`cuda_pathtracer.cu:36-53, 417-419, 3742-3756`（`RENDERER_BENCHMARK_DIAGNOSTICS` 门控的计数器/内核）；`6256-6262, 6404-6490`（`RENDERER_CUDA_SANITIZER_FALLBACK` 手写固定拓扑启动）vs `6599-6849`（生产 CUDA Graph 路径）

CUDA Graph、sanitizer 手启、diagnostics 计数三套拓扑需要同步维护。

**建议**：sanitizer 回退并入 graph 路径（或反向），诊断计数器走单一机制。

### 3.16 🔴 渲染全部在主线程；fallback 回读阻塞 UI
**位置**：`viewer_main.cpp:1204-1224`；`viewer_render_backend.cpp:207-221`；`cuda_pathtracer.cu:6319-6362`（`cudaStreamSynchronize`）；GL 上传无 PBO/持久映射（`opengl_raster_renderer.cpp:2325-2406`）

整个仓库只有 `cuda_device_context.cu:78` 一个互斥锁，没有任何渲染线程。interop 不可用时每次渲染后阻塞同步流再拷回 CPU framebuffer，UI 卡满整个追踪时间。

**建议**：提交与呈现分离（工作线程或 ping-pong pinned staging 双缓冲回读），`cudaStreamSynchronize` 移出 UI 线程。

### 3.17 🟢 `statistics() const` 内部调用非 const 实现并触发 CUDA API
**位置**：`cuda_pathtracer.cu:7270-7277, 7730-7736`（const getter 内部 `cudaEventQuery`/`cudaGetDevice` 并刷新计时器状态）

**建议**：拆分"读缓存统计"与显式 `update_timings()`。

### 3.18 🟡 统计仪表（约 40 字段）与核心逻辑深度交织
**位置**：`src/render/pathtracer/cuda_pathtracer.h:24-74`；`cuda_pathtracer.cu:4415,4438,4612-4670,5631-5717,6089-6115`（约 20 处 `*_upload_bytes` 累加点 + `CudaEventTimer` begin/end 散落各处）

**建议**：`StatisticsRecorder`/作用域守卫 + 单一字节记账助手，核心代码不直接摸约 20 个计数器。

### 3.19 🟡 两层会话抽象三重委托；场景 diff 每帧最多算三次
**位置**：`interactive_render_session.h:116-128` + `path_interactive_session.h/.cpp`（旧层）；`viewer_render_backend.h:43-59`（新层）；`viewer_render_backend.cpp:241`（Path 后端包 `PathInteractiveSession` 再包 `CudaPathInteractiveRenderer`）；OpenGL 后端 `62-70`、Path 后端 `168-173` 各自再算 `scene_changes_for_snapshot`，而 `viewer_main.cpp:942-945,1067` 也算了并传入

三层路径栈 + 同一 diff 每帧最多算三次。

**建议**：选一层抽象，`PathInteractiveSession` 并入 `CudaPathInteractiveRenderer`（或反向）；diff 由后端算一次向下传。

### 3.20 🟢 内嵌 shader 字符串 vs 磁盘热重载双轨
**位置**：`opengl_raster_renderer.cpp:686-852`（sky/shadow/composite 内嵌 GLSL）vs `1349-1426`（仅 raster + 3 个 AO shader 可热重载）

内嵌 pass 与可重载 shader 共享 uniform 概念却无法热重载，对 shader 作者不可见。

**建议**：内嵌 pass 移到磁盘走同一重载路径，或至少让可重载文件列表数据驱动。

### 3.21 🟢 可用性探测缓存有副作用；stub 行为不一致
**位置**：`cuda_device_context.cu:77-91`（进程级缓存命中仍 `cudaSetDevice`——查询会切换当前设备）；`cuda_pathtracer_stub.cpp:68-91`（`reset`/`render_next_frame` throw，但 `statistics()` 返回静态默认、`stream_handle()`/`device_id()` 静默返回 0/-1、`set_presentation_state` 静默 no-op）

stub 的"响亮/静默"取决于调用哪个方法。

**建议**：stub 统一显式（全 throw 或文档化的 `available()` 门），去掉缓存探测的 `cudaSetDevice` 副作用。

### 3.22 🟢 `RenderFrameOutput` 用 `shared_ptr<const void>` 类型擦除所有权；fallback 呈现两次全量拷贝
**位置**：`src/render/interactive/render_frame_output.h:10-26`；`cuda_pathtracer.cu:6333-6361`；`src/render/framebuffer.cpp:57-83`

设备→pinned→`vector<Color>`→`Framebuffer`→`to_rgba8` 再拷一遍。

**建议**：类型化 `TextureLease` 接口；fallback 直接呈现 pinned buffer，不经 Framebuffer 中转。

### 3.23 🟢 tone mapping 双实现
**位置**：`src/render/display_settings.cpp:10-45` vs 合成器 GLSL（`sdl_display_backend.cpp:50-110`）

Reinhard/ACES/exposure 在 C++ 与合成器 shader 各一份，人工同步；屏幕结果与离线 `to_display_rgb8` 路径可能静默分歧。

**建议**：单一真值来源（从同一 C++ 常量生成 shader，或加锁步一致性测试）。

---

## 4. Viewer 主循环与 UI

### 4.1 🔴 `main()` 是约 825 行 god function
**位置**：`src/viewer_main.cpp:490-1314`；循环 `769-1265`（约 500 行）；7 个内联 lambda（`reset_cameras_for_scene` 664、`set_viewer_title` 700、`save_session_now` 713、`import_asset_path` 782、`open_scene_path` 796、`save_scene_path` 813、`load_environment_path` 825）；约 30 个可变局部量被闭包捕获

启动、会话恢复、场景加载、CUDA 设备选择、后端初始化、整个帧循环、自动保存去抖、收尾统计全在一个函数里；四个路径处理 lambda 彼此复制 try/catch + `ui_state.scene_status` 模式。

**建议**：提取 `ViewerApp` 类；路径处理收敛为单一 `handle_dialog_result(kind, path)`。

### 4.2 🔴 `ViewerUi::draw()` 单函数约 1,807 行、13 个参数
**位置**：`src/interactive/viewer_ui.cpp:442-2248`；签名 `viewer_ui.h:134-147`

菜单、dock 布局、快捷键、五个面板（Rendering/Camera/Techniques/场景大纲/Inspector）全在一个函数里，直接改 `RenderSettings`、`SceneDocument`、两个相机控制器。面板无法复用/测试/重排。

**建议**：每个面板一个 `draw_*_panel()`（返回窄 `PanelActions`），`draw()` 只做编排；文档修改收敛到单一提交层。

### 4.3 🔴 `ViewerUiActions` 22 个布尔的"穷人事件系统"，2 个标志无人读取
**位置**：`src/interactive/viewer_ui.h:94-130`；`viewer_ui.cpp:814,820,831,836`（写 `display_changed`/`ui_style_changed`）——grep 全 `src/` 无任何读取点；`focus_object`/`look_through_camera` 作为带外 `ObjectId` 走私进标志袋；`resets_path_accumulation()` 把渲染策略塞进 DTO

布尔无法携带载荷；哪些变更需要标志、哪些直接读状态完全靠约定，新增 widget 时无法知道该不该置标志。

**建议**：类型化、带载荷的事件/命令列表（`std::variant` 或 `ViewerCommand` 队列），删除死标志。

### 4.4 🟡 `document_dirty` 三处冗余存储
**位置**：权威 `SceneDocument::dirty()`（`scene_document.cpp:2337-2340`）；镜像 `ViewerSessionState::document_dirty`（`viewer_session.h:24`）；签名 `ViewerSessionSignature::document_dirty`（`viewer_main.cpp:439,484,720`）；加载时经 `restore_file_state` 重建（`scene_document.cpp:2526-2538`，把历史坍缩成单个 checkpoint）

一个比特三个副本可漂移，且恢复时丢失原有撤销粒度。

**建议**：只持久化 `file_path`；dirty 保存时从 `SceneDocument::dirty()` 派生。

### 4.5 🟡 选择/激活/材质编辑对象状态三处 reconcile
**位置**：会话加载 `viewer_session.cpp:195-212`；gizmo 内懒校验 `viewer_ui.cpp:2255-2270`；场景打开 `viewer_main.cpp:803-804`；瞬态 `gizmo_was_using`/`gizmo_hovered` 混进持久化的 `ViewerUiState`（`viewer_ui.h:47-48`）

没有单一"选择不变量"执行点；过期 `ObjectId` 存活到下一个碰巧运行它的路径。

**建议**：`Selection` 值类型 + 单一 `reconcile(document)`；瞬态 gizmo 命中结果不放进持久化状态。

### 4.6 🟡 键盘处理三处三种机制；Escape 无条件退出
**位置**：SDL 扫描码 `sdl_display_backend.cpp:421-443,458-467`；ImGui 快捷键 `viewer_ui.cpp:507-550`；主循环解释 `viewer_main.cpp:950-982,1121-1131`；Escape → `quit_requested` 无捕获守卫（`sdl_display_backend.cpp:439-441`），而 `poll_input` 在 ImGui 帧之前运行（`viewer_main.cpp:775` vs `868`），上一帧的 `WantCaptureKeyboard` 也救不了

**影响**：关 ImGui 弹窗/文本编辑按 Escape 会直接退出程序；同一"快捷键"概念三种实现、两种捕获信号。

**建议**：单一 keymap 表产生统一输入流；Escape 等退出键走键盘捕获门控（或推迟到 UI 帧之后）。

### 4.7 🟡 相机状态三份 + reset 逻辑六处
**位置**：双控制器 + 持久化 `ViewerCameraSessionState`；重建在 `viewer_main.cpp:626-681,1016-1063`；`transition_camera_mode` `391-404`；快照式 reset（`initial_orbit_camera`/`initial_free_camera` `662-663`）；`look_through_camera` 在模式转换检查之后改模式（`1036-1063`，顺序敏感）

**建议**：`CameraState` 聚合类型拥有两种模式 + home 帧，只暴露 `reset()/focus()/look_through()/set_mode()`。

### 4.8 🟡 模式切换销毁重建整个后端并全量重传场景
**位置**：`viewer_main.cpp:989-1000`（`render_backend = make_viewer_render_backend(...)` + `reset(snapshot)`）；`viewer_render_backend.cpp:46-55, 140-160`

两个后端不共享 flatten 结果，每次切换付完整 teardown + 完整上传；大场景切换是秒级卡顿 + GPU 分配抖动。

**建议**：按文档 revision 缓存 flatten 快照并在后端实例间共享；模式切换只重建后端专属 GPU 资源。

### 4.9 🟡 后端接口泄漏：`dynamic_cast` 取 shader 控制 + `std::variant` 统计强制调用方分支
**位置**：`viewer_render_backend.cpp:254-256`；`viewer_render_backend.h:32-34`；`viewer_main.cpp:877-896,1213-1215`

**建议**：能力查询进接口；统计换成带 mode 判别符的公共结构体。

### 4.10 🟡 无 vsync/帧限速；delta 不 clamp，卡顿可瞬移自由相机
**位置**：`sdl_display_backend.cpp:228`（`SetSwapInterval(0)`）；`viewer_main.cpp:770-773`（delta 无上限）、`1131-1135`

原生文件对话框/shader 编译/窗口拖动产生的大 delta 直接喂给相机移动。

**建议**：clamp delta；提供 vsync/帧限速选项。

### 4.11 🟢 relative mouse 切换失败抛致命异常；HiDPI 逻辑/物理尺寸三源混用
**位置**：`viewer_main.cpp:1086-1092`；`984-987,1076-1084,721-727` vs `sdl_display_backend.cpp:331-363,693-698`

**建议**：输入切换失败降级为状态提示；确立唯一尺寸来源（drawable 像素）并显式派生渲染尺寸。

### 4.12 🟢 `output()` 是"上一帧结果"状态化成员，暂停时隐式依赖陈旧帧
**位置**：`viewer_render_backend.cpp:86-88, 225-227`；`viewer_main.cpp:1200-1224`

暂停时跳过 `render()` 仍呈现 `output_`——期望行为恰好成立但靠事故而非契约；`render()` 早退未写 `output_` 时会静默呈现更旧一帧。

**建议**：显式 `present_last_frame()` 语义。

---

## 5. 平台层与会话持久化

### 5.1 🟡 `SdlDisplayBackend` 是第二个上帝对象
**位置**：`src/platform/sdl/sdl_display_backend.h:65-135`；`sdl_display_backend.cpp:136-278`（初始化把 SDL+GL+ImGui 织在一起）、析构 `144-178`（硬编码多视口 teardown 顺序 hack：手工存 ini 再置空 `IniFilename`）

窗口、GL 上下文、ImGui 上下文、合成器、文件对话框、输入轮询六种职责耦合在一起。

**建议**：拆 `Window/GLContext`、`InputPump`、`DialogService`、`ImGuiPresenter/DisplayCompositor`。

### 5.2 🟡 会话序列化手工逐字段 ~60 项 + 内联版本判断 + 两个独立版本号
**位置**：`viewer_session.cpp:130-381`（load）与 `383-503`（save）双向重复字段表；`if (version >= 3)` 内联守卫 `241,352`；写入 `version = 4`（`388`）；会话内嵌完整文档快照（`392`），文档自身版本号 v5 独立演进（`scene_document.cpp:2547`）

一个文件两个版本号；没有迁移管线（v1→v2→… 顺序函数）；约 40 个嵌套字段的 `render.opengl` 子树手写。

**建议**：显式迁移函数链；要么停止内嵌完整文档、要么统一版本号。

### 5.3 🟢 基准工具 `viewer_benchmark_main` 复制 viewer 的 GL 引导、会话解析与帧语义
**位置**：`tools/viewer_benchmark_main.cpp:380-437`（`HiddenGlContext` 复制 SDL/GL 引导）、`141-155`（`tone_mapper_name`/`vec3_json` 复制 `viewer_session.cpp:18-20,62-72`）、`157-215`（`import_session_as_case` 硬编码会话布局）、`449-524,562-717`（手工驱动 `make_viewer_render_backend`）

会话/相机布局或"什么算一帧"的改动要在两处同步做，基准数字可能悄然偏离真实 viewer。

**建议**：共享引导/序列化/`render_one_frame` 助手库，基准尽可能驱动真实 viewer 帧路径。

### 5.4 🟢 对话框 inbox 的 `open` 标志在回调线程复位；回调线程读线程全局 `SDL_GetError`
**位置**：`sdl_display_backend.cpp:23-27, 513-539, 549`

结果在互斥锁下入队（正确），但 `open=false` 在锁外由回调线程置位、错误经线程全局 `SDL_GetError()` 读取——第一个结果未消费时第二个对话框即可打开；错误归属不保证对应本次对话框。

**建议**：`open` 只由主线程 drain 时复位（或同锁下）；错误在调用时捕获。

---

## 6. 构建系统与依赖

### 6.1 🔴 4 个 FetchContent 依赖缺 `URL_HASH`
**位置**：有哈希 `CMakeLists.txt:14-15, 26-27, 36-37`；无哈希 `:81-84`（eigen3）、`:118-122`（SDL3）、`:137-141`（imgui）、`:144-148`（imguizmo）

供应链固定策略不一致（7 个依赖只固定 3 个）；tag 被强推或下载损坏会注入代码或产生难排查的后期失败。

**建议**：四个全补 `URL_HASH`，把"必须带哈希"设为 lint 规则。

### 6.2 🔴 `renderer_core_diagnostics` 把整个核心（含 278KB `cuda_pathtracer.cu`）重复编译第二遍
**位置**：`CMakeLists.txt:274-279`（源码并集）、`:403-428`（第二个 STATIC 库，`RENDERER_BENCHMARK_DIAGNOSTICS=1`）

CUDA 基准构建把最贵的翻译单元编译两遍；且诊断库是生产代码的 define 分叉副本，存在"测的与交付的漂移"风险。

**建议**：诊断走运行时标志穿进现有 `renderer_cuda`，或只隔离被插桩的翻译单元（OBJECT 库），核心只编一次。

### 6.3 🔴 静态库分层不内聚：`renderer_scene` 是大杂烩，`renderer_interactive` PUBLIC 链接 `renderer_cuda`
**位置**：`CMakeLists.txt:173-196`（`renderer_scene` 含 pbr/scene_intersector/bvh/sampler/image/atomic_file）；`:259-272`（`renderer_interactive` PUBLIC `renderer_cuda`，`renderer_core` INTERFACE 再导出全部）

`scene_tests`/`document_tests` 链接 `renderer_scene` 被迫编译链接渲染代码；UI/相机库通过链接可见性耦合 GPU 路径追踪器——stub 层之所以存在正是因为这种耦合。

**建议**：按职责拆 `renderer_scene`（数据/文档）、`renderer_acceleration`、`renderer_sampling` 等；CUDA 依赖改 PRIVATE，或把 `path_interactive_session` 移出 `renderer_interactive`。

### 6.4 🟡 每目标手工枚举 warnings/native-arch 列表；网络强依赖 configure；SDL3 自引用 cache hack
**位置**：`CMakeLists.txt:460-497, 507-544`（两个 foreach 列表 + 诊断目标第三处重复）；`:12-40,137-149`（5 个依赖无条件 FetchContent，仅 Eigen/SDL 有 `find_package` 回退）；`:114-117`（`FETCHCONTENT_SOURCE_DIR_SDL3` 指向下载器自己的输出目录，绕开重复解压；存在后 URL 被静默忽略）

新目标会静默拿不到警告与原生指令调优；无网环境无法 configure；SDL3 hack 使依赖永不更新且难读。

**建议**：单一 `RENDERER_TARGETS` 变量驱动；每个依赖统一"find_package 回退或 vendor"策略并文档化；删 hack、用正常缓存 + `URL_HASH`。

### 6.5 🟡 `VERSION 0.2.0` 但无 `install()/export()/CPack`
**位置**：`CMakeLists.txt:3`；grep `install(|export(|CPack` 仅命中 `SDL_DISABLE_INSTALL`（`:113`）

6 个库和 2 个可执行文件无法安装、无法被下游 CMake 项目导入、无法打包——版本号形同虚设。

**建议**：补 `install()` + `install(EXPORT)` + 包配置模板，或明确文档化"仅源码构建"。

---

## 7. 测试与 CI

### 7.1 🔴 测试框架只有 `RENDER_CHECK`（首败即 `exit(1)`）
**位置**：`tests/test_framework.h:8-19`

无注册表、无筛选、无隔离、无 golden image、无期望异常助手；一次失败中止整个二进制，没有"N 通过 / M 失败"汇总；ctest 粒度只有 6 个二进制。

**建议**：接入 Catch2/doctest/GoogleTest（或扩展注册表 + 逐测试捕获 + 过滤器），并增加渲染输出 golden fixture。

### 7.2 🔴 `renderer_tests.cpp` 5,874 行 God-file，混合 10+ 领域且与分领域测试文件重复
**位置**：`tests/renderer_tests.cpp`（108 个 `test_*`、55 个 include；场景文档测试同时存在于 `document_tests.cpp:12-73` 与 `renderer_tests.cpp:5859-5871`）

**建议**：按领域拆文件并删重复，`renderer_tests` 只留集成测试。

### 7.3 🔴 CUDA 测试在无 GPU 时静默跳过仍报绿——与项目自己的声明矛盾
**位置**：`tests/cuda_contract_tests.cpp:14-31`（stub 下只测 5 个头文件纯函数断言）、`:32-235`（真实 CUDA 全部在 `if (available)` 内）、`:236` 打印 "all tests passed"；`tests/renderer_tests.cpp` 中 19 处可用性探测有 17 处是 `if (!available) return;` 静默跳过（如 `1607, 1699, 1715, 1733, 1800, 2085, 2507`）；`tests/benchmark_diagnostics_tests.cpp:30-34` 打印 skipped 后 `return 0`；声明见 `README.md:180` 与 `.github/workflows/ci.yml:77-79`

`renderer_tests` 二进制在普通 CI 上运行，其中全部 CUDA 依赖测试静默通过；CUDA 路径的回归在 CI 上永远不会被发现——README 声称"不会在普通 CI 中静默跳过后显示绿色"与实际不符。

**建议**：真实 CUDA 测试用 `#if RENDERER_HAS_CUDA` 编译门控，no-CUDA 二进制要么响亮失败要么经 CTest `SKIP_RETURN_CODE` 注册为 skipped，绿绝不能等于"没跑"。

### 7.4 🟡 `opengl_contract_tests` 从未打开 GL 上下文——它是 grep 文本
**位置**：`tests/opengl_contract_tests.cpp:157-248`（把 shader 源和 `opengl_raster_renderer.cpp` 读进内存做子串断言，如 `layout(location = N) in`、`GL_R32F`、`AlphaMode::Blend`）；`CMakeLists.txt:345-350`（只链接 `renderer_scene`）

格式改动/注释/重构就会弄坏它，且它证明不了 shader 能编译链接、绑定在运行期匹配——只证明某些子串共存。

**建议**：文本检查降级为 lint；加真实上下文（SDL/EGL headless GL）的契约测试：编译链接 shader、查询 program 输入/uniform。

### 7.5 🟡 CI 重复矩阵、无缓存/工件/性能门槛；Windows viewer 从不执行；CUDA 从未编译
**位置**：`.github/workflows/ci.yml:8-50`（Windows/Linux 跑同一套 no-CUDA 构建+ctest，Linux 多一个 Xvfb smoke）；`40-50`（断言 grep 字面量 `"switched to OpenGL"`，措辞一改就挂）；无 ccache/`_deps` 缓存、无工件、无基准回归 job（阈值 `docs/benchmarking.md:82-88` 只靠手工执行）

**建议**：跨平台保留编译覆盖但完整 ctest 只跑一次或按领域拆；Windows 加软 GL smoke；加编译器/依赖缓存与工件上传；自托管 GPU runner 上编译运行 `.cu`；基准门槛做成可选 job；smoke 断言改用稳定退出码/机器可读输出。

---

## 8. 基准与仓库卫生

### 8.1 🔴 `learning/` GAMES202 课程作业 5,085 文件（约 390MB）入库，占跟踪文件 97%
**位置**：`git ls-files` 共 5,255 个文件，`learning/` 占 5,085 个；内含 Eigen/OpenEXR/zlib/TBB/pugixml/nanogui/GLFW 完整 vendor；`.gitignore` 无 `learning/`

与渲染器构建无关；膨胀每次 clone 与历史；污染全仓库 grep/glob。

**建议**：移出仓库与历史（或独立 repo/submodule），至少先加 `.gitignore`。

### 8.2 🟡 `run_benchmarks.ps1` Windows-only 且硬编码 `cuda-native` 预设与 `.exe`
**位置**：`tools/run_benchmarks.ps1:28,36,38,54,62`

CI 跑 Linux，但"规范基准"在 CI 平台不可运行；没有 CUDA 工具链连 OpenGL-only 基准都跑不了。

**建议**：跨平台（经 CTest/CMake 驱动、`$<TARGET_FILE:...>`），允许 `-Backend opengl` 免 CUDA 配置。

### 8.3 🟡 基准结果（含 489KB raw.json）入库；基准资产未版本化
**位置**：`git ls-files benchmarks/*` 12 个文件含 `benchmarks/results/*/raw.json`；标准用例引用被 git 忽略的 `Computer Graphics Archive/`（`docs/benchmarking.md:7-10`），新 clone 无法复现规范基准

**建议**：忽略 raw.json 与诊断负载，只留 summary/baseline；提供小自包含基准场景（入库或带哈希 URL）。

### 8.4 🟢 `output/` 只忽略顶层图片后缀，10 个分析产物仍被跟踪；`tmp/` 未忽略
**位置**：`.gitignore:5-9` vs `git ls-files output/*`（`output/cuda-bounce-analysis/*.{png,csv,json}` 等）

**建议**：忽略 `output/**`（显式 allowlist），忽略 `tmp/`。

---

## 9. 文档一致性

### 9.1 🟡 文档与实际漂移；README 自相矛盾
**位置**：`docs/scene-object-system.md:115` 写"Viewer 会话当前写入 v3"，代码实际写 `version = 4`（`viewer_session.cpp:388`）；README 第 200 行声明 `docs/output/` 是历史记录，第 198 行又把 `renderer-hardening-results.md` 作为当前"架构硬化报告"链接；README 第 180 行与 CI 注释声称 CUDA 测试不会静默跳过，实际 §7.3 恰恰相反

**影响**：文档不能作为设计依据，新贡献者会照着错文档实现。

**建议**：把历史 plans/specs/output 移入 `archive/`；补一份与 6 库实际构建图一致的 `docs/architecture.md`；修正版本与测试声明。

---

## 10. 修复优先级建议

按"风险/收益"排序的路线图（不修改代码，仅建议方向）：

**第一阶段：正确性风险（数据损坏、静默错误）**
1. 设备结构体打包：`D*` 移入共享 `.cuh` + `Material`/`DMaterial`/SSBO 布局 `static_assert`（§3.5）。
2. `SceneMaterialOverride` 改为完整 `Material` 或字段表驱动（§1.1）。
3. `begin_edit` 停止拷贝整条历史；undo 引入结构性共享或命令对象（§2.1）。
4. CUDA 测试改为编译门控 + 显式 skip，兑现 README 承诺（§7.3）。
5. MIS/BRDF 收敛单一定义并加跨实现数值测试（§3.3/3.4）。

**第二阶段：结构与可维护性**
6. 拆 `cuda_pathtracer.cu` 与 GL Impl 为模块/Pass（§3.1/3.2）；提取 `ViewerApp` 与面板函数（§4.1/4.2）。
7. 渐进重置策略单一归属（§3.11）；会话抽象砍掉一层（§3.19）；`ViewerUiActions` 改事件列表（§4.3）。
8. CMake：补哈希、去掉 diagnostics 二次编译、修正库分层与 PUBLIC 链接（§6.1-6.3）。

**第三阶段：性能与体验**
9. 渲染提交/呈现与 UI 线程解耦；fallback 回读双缓冲（§3.16）。
10. 每实例材质表改共享 + 覆盖增量；`find_asset` 改哈希索引（§1.4/2.5）。
11. 模式切换共享 flatten 快照（§4.8）；delta clamp 与帧限速（§4.10）。

**第四阶段：工程卫生**
12. 移除 `learning/`；清理 output/benchmark 产物；补 install/export（§8.1-8.4、§6.5）。
13. 测试框架升级 + 领域拆分 + 真实 GL 契约测试（§7.1/7.2/7.4）；CI 加缓存/工件/基准门槛（§7.5）。
14. 文档归档与架构文档化（§9.1）。
