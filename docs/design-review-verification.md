# `design-review.md` 事实核查

> 核查日期：2026-08-15  
> 核查对象：`docs/design-review.md`  
> 代码基线：`0235f84d8aafe13a53d0a7629914af6bf0a26a73`

## 结论

`docs/design-review.md` 不能视为全部属实。

按当前代码核对其 72 个分项，并且只判断代码事实、不评价严重度和整改建议：

- 46 条基本成立；
- 25 条部分成立，但包含夸大、错误数量或未经证明的因果；
- 1 条实质不成立：§1.6“三份程序球几何”。

报告适合作为问题线索清单，但不宜直接作为事实基线或完全按照其中的严重度安排整改。

## 分项归类

### 基本成立

- §1.2、§1.3、§1.4、§1.7
- §2.1、§2.4、§2.5、§2.6、§2.7、§2.8、§2.9
- §3.1、§3.2、§3.6、§3.7、§3.9、§3.11、§3.13、§3.14、§3.16、§3.17、§3.18、§3.20、§3.21、§3.22、§3.23
- §4.1、§4.2、§4.6、§4.7、§4.9、§4.10、§4.11
- §5.1、§5.2、§5.3
- §6.1、§6.2、§6.3、§6.5
- §7.1、§7.2、§7.3、§7.5
- §8.2、§8.3

### 部分成立，需要改写

- §1.1、§1.5、§1.8、§1.9
- §2.2、§2.3
- §3.3、§3.4、§3.5、§3.8、§3.10、§3.12、§3.15、§3.19
- §4.3、§4.4、§4.5、§4.8、§4.12
- §5.4
- §6.4
- §7.4
- §8.1、§8.4
- §9.1

### 实质不成立

- §1.6

## 主要错误与修正

### §1.1 材质覆盖

“字段映射需要多处手工同步”属实，但以下结论不属实：

- `use_*_texture` 的语义没有与名称相反：`false` 表示禁用，`true` 表示保留源材质纹理。
- 覆盖不是在上一份已修改材质上继续应用。每次构建快照都会先复制资产的源材质，再调用 `apply_material_override`。因此把开关从 `false` 攥回 `true` 可以重新启用原纹理。
- 覆盖结构没有提供纹理 ID，确实不能把纹理改指到另一张纹理，但这不等于“永远无法重新启用”。
- `pbr_workflow`、specular 参数和纹理变换没有被覆盖结构保存，但应用覆盖时不会把源材质中的这些字段全部清空。Specular-glossiness 材质的 `glossiness` 还会由覆盖的 roughness 更新为 `1 - roughness`。

证据：`src/scene/scene_document.cpp:235-308, 1977-1992`。

### §1.5 环境与灯光状态

环境字段确实同时出现在 `Scene`、`RenderSceneSnapshot` 和 `SceneDocument::State`。

但 `SceneDocument::State` 没有四组灯光 vector。文档层的灯光被存储为 `SceneObject`，只有 `Scene` 和 `RenderSceneSnapshot` 直接持有 `point_lights`、`directional_lights`、`spot_lights`、`rect_area_lights`。因此“三处逐字重复灯光 vector”不成立。

证据：`src/scene/scene.h:22-30`、`src/scene/instanced_scene.h:80-88`、`src/scene/scene_document.h:282-290`。

### §1.6 程序球几何

“同一程序球几何存了三份”不成立。

对于包含程序球的场景：

1. `local_scene` 中保存球细分后的三角形，供文档拾取使用；
2. `procedural_spheres` 保存原始球参数，供快照实例化使用；
3. `render_geometry` 是在细分前复制的场景，但其 `spheres` 随即被清空。对于纯球场景，它不包含第三份球几何。

因此准确说法是存在两种球表示，而不是三份球几何。

证据：`src/scene/scene_document.cpp:559-609`。

### §1.8 死代码

该条只有部分成立：

- `Random` 和 `world_matrix_recursive` 没有实际调用者；
- `ExecutionBackend` 当前只有 `Cuda` 一个枚举值；
- `SceneMeshAsset::geometry_revision` 在文档资产中没有递增路径；
- 但 `load_obj_mesh` 和 `sample_material_base_color` 都被 `tests/renderer_tests.cpp` 调用，不能称为“无任何调用者”。它们最多只能被描述为没有生产代码调用者。

证据：`tests/renderer_tests.cpp:2159, 2310, 4096`。

### §1.9 双轨材质字段

OBJ 使用 legacy diffuse texture、glTF 使用 base-color texture，以及材质求值器优先 base-color、回退 diffuse，这些事实成立。

但报告中的两点不严谨：

- OBJ loader 同时会把 diffuse color 写入 `Material::base_color`，并非所有颜色数据都只进入 legacy 字段；
- 表面材质的 `two_sided` 与面积灯的 `light_two_sided` 表达不同对象的语义，默认值不同不能单独证明设计不一致。

### §2.2 撤销导入与资产生命周期

undo/redo 只恢复 `state_`、不会删除 `assets_`，所以撤销导入后资产会保留到文档销毁或再次复用。这个内存保留问题属实。

但更准确的描述是“文档生命周期内保留孤儿资产或缓存”，而不是不可达内存泄漏。`assets_` 仍持有资产，并且 `load_asset` 会按路径复用它。

另外，`import_path` 自身会备份 state、资产数量、warnings 和 ID，并在异常时回滚，因此其正确性并不依赖“所有导入都必须包在事务里”。

证据：`src/scene/scene_document.cpp:1233-1343, 2267-2327`。

### §2.3 mutable 缓存与线程安全

缓存由 const 方法重建、返回的 `const&` 会在下一次快照重建时更新，这些事实成立；类型本身也不支持并发编辑和读取。

但当前仓库没有独立渲染线程，现有 OpenGL/CUDA 后端也没有跨帧持有这个引用。因此报告描述的是未来引入并发后的风险，而不是当前已经发生的数据竞争或悬空引用。

### §3.3 BRDF 与采样副本

C++、CUDA、GLSL 各自实现了 GGX、Schlick Fresnel、Smith 几何项和 PBR 求值，这一核心问题成立。C++ 版本主要由测试调用，也没有与 CUDA/GLSL 做自动数值对比。

但 `sample_visible_ggx` 只存在于 C++ 和 CUDA 路径，光栅 GLSL 不进行这类路径采样，所以“采样逻辑三份”不准确。

### §3.4 `power_heuristic`

C++ 与 CUDA 实现对负数、NaN 和非有限 PDF 的防御行为不同，因此边界契约确实漂移。

但报告声称“PDF 恰为 0 时结果不同”是错误的：

- `a = 0, b > 0` 时两者都返回 0；
- `a > 0, b = 0` 时两者都返回 1；
- `a = 0, b = 0` 时两者都返回 0。

证据：`src/render/pbr.cpp:206-210`、`src/render/pathtracer/cuda_pathtracer.cu:859-869`。

### §3.5 主机到设备打包

`DMaterial` 使用很长的位置聚合初始化，重排同类型字段可能造成静默错位，这个维护风险成立。

但 `Material` 并不是按二进制布局直接复制成 `DMaterial`；代码逐字段构造 `DMaterial`，并由同一个 CUDA 翻译单元在主机和设备侧使用。因此缺少 `Material`/`DMaterial` 间布局断言不会直接造成 ABI 错配。真正的问题是字段映射和聚合初始化缺少编译期约束。

### §3.8 几何遍历表示

CPU BVH、CUDA BVH4/设备遍历和硬件光栅化确实并存，`Bvh::build_layout` 也没有被 CUDA 路径使用。

但 `SceneIntersector` 是 CPU BVH 的使用层，不是独立的几何表示；硬件光栅化也不等同于一份显式软件遍历结构。因此“四套表示”的计数混合了数据布局、构建器、遍历器和硬件管线。

### §3.10 AO、PCSS 数学镜像

C++ 与 GLSL 存在对应数学实现，且缺少跨语言自动数值一致性验证，这一点成立。

不过 C++ AO/PCSS helper 主要由 `opengl_contract_tests` 使用；生产 OpenGL renderer 直接使用的是方向光阴影拟合 helper。报告把测试 oracle、生产 CPU 算法和 shader 镜像概括成同一种“CPU 诊断路径”，不够准确。

### §3.12 浮点精确比较

`ProgressiveRenderKey` 的确使用浮点精确相等比较。

但相机发生任何变化时重置路径累积通常是正确行为。量化相机参数可能让已经变化的相机继续复用旧样本，产生鬼影。除非能够证明控制器在静止时持续产生数值抖动，否则不能仅凭 `==` 将其认定为缺陷。

### §3.15 编译期变体

生产 CUDA Graph 与 sanitizer 手工启动路径确实是两套需要同步的拓扑。Diagnostics 宏会加入计数器和参数，但仍主要是在现有执行路径上插桩；把它直接称为第三套完整拓扑有所夸大。

### §3.19 场景 diff 次数

Path 模式存在重复 revision diff，但最多是两次而不是三次：

1. `PathViewerRenderBackend::render` 调用一次 `scene_changes_for_snapshot`；
2. `CudaPathInteractiveRenderer::Impl` 再调用一次。

主循环只是合并 `external_scene_changes`、UI actions 和 gizmo changes，并将它们作为 hint 传入，不是第三次 revision diff。

证据：`src/viewer_main.cpp:942-945, 1067`、`src/render/interactive/viewer_render_backend.cpp:168-173`、`src/render/pathtracer/cuda_pathtracer.cu:7093-7098`。

### §4.3 `ViewerUiActions`

`ViewerUiActions` 实际有 19 个布尔数据字段，不是 22 个。`resets_path_accumulation()` 是成员函数，不应计入字段。

`display_changed` 和 `ui_style_changed` 确实只有写入点、没有读取点，这部分结论成立。

证据：`src/interactive/viewer_ui.h:94-130`。

### §4.4 `document_dirty`

三个位置都出现了 `document_dirty` 值，但不都是独立权威状态：

- `SceneDocument::dirty()` 是运行时权威状态；
- `ViewerSessionState::document_dirty` 是加载会话时使用的传输字段；
- `ViewerSessionSignature::document_dirty` 是自动保存变更检测使用的瞬时比较值。

因此“三份可漂移的权威存储”表述过度。会话没有保存 undo 历史、恢复后撤销粒度丢失则是另一个成立的问题。

### §4.5 选择状态

选择状态在多个路径执行 reconcile 属实。

但 `gizmo_was_using` 和 `gizmo_hovered` 虽然位于 `ViewerUiState`，并没有被 `viewer_session.cpp` 序列化，所以不能说它们被混入了持久化文件。

### §4.8 模式切换

切换模式会销毁旧后端、创建新后端并调用 `reset(snapshot)`，缺少跨后端共享的 flatten/GPU 资源缓存，这一点成立。

但“必然秒级卡顿”没有当前基准或测量证据，只能作为大场景下的风险推测。

### §4.12 上一帧输出

`output()` 返回保存的最后一帧是事实，暂停时主循环会跳过 `render()` 并继续呈现该输出。

不过这一控制流是明确写在主循环中的，不能由代码证明它只是“事故”。如果认为契约不清晰，更准确的问题应是接口和注释没有明确规定 last-frame presentation 的生命周期。

### §5.4 SDL 对话框回调

回调线程会在结果入队后把原子变量 `open` 设为 false，因此确实允许主线程在前一结果尚未 drain 时打开下一对话框。结果带有 `kind` 且保存在互斥队列中，这并不直接造成数据竞争。

报告关于 `SDL_GetError()` 的结论错误。SDL3 明确规定错误字符串是线程局部的；文件对话框回调文档也明确要求在 `filelist == nullptr` 时调用 `SDL_GetError()`。当前代码在回调中立即复制错误字符串，符合该契约。

证据：`src/platform/sdl/sdl_display_backend.cpp:513-539`，以及构建依赖中的 `SDL_error.h:128-145`、`SDL_dialog.h:76-91`。

### §6.4 FetchContent 与目标配置

依赖策略不统一、默认配置依赖网络、warnings/native-arch 目标列表手工维护等问题基本成立。

但 imgui 和 ImGuizmo 的 FetchContent 位于 `RENDERER_BUILD_VIEWER` 条件内，不是所有构建模式下都无条件下载。默认 viewer 构建仍会触发它们。

### §7.4 OpenGL contract tests

测试确实没有创建 GL 上下文，不能证明 shader 能实际编译、链接或完成运行时绑定。

但该测试不只是 grep：前半部分还执行 PCSS、方向光阴影拟合、SSAO 和 GTAO helper 的数值单元测试。准确说法应是“shader 契约部分主要依赖源码子串检查”。

证据：`tests/opengl_contract_tests.cpp:45-154, 157-248`。

### §8.1 `learning/` 大小

文件计数成立：

- Git 跟踪文件总数：5255；
- `learning/` 跟踪文件：5085；
- 占比：96.76%。

但“约 390MB 全部入库”不成立：

- Git 跟踪的 `learning/` 文件合计 283,069,473 字节，约 269.96 MiB；
- `learning/` 整个工作区约 389.37 MiB；
- 其中约 119.42 MiB、1105 个文件是被 Git 忽略的本地构建产物。

因此 390 MiB 是工作区占用，不是入库体积。

### §8.4 output 文件数量

`output/` 下当前有 10 个 Git 跟踪文件，但其中一个是 `output/.gitkeep`。真正的分析产物是 9 个，不是 10 个。

嵌套分析产物未被 `.gitignore` 中的顶层 `output/*.png` 等模式覆盖，以及 `tmp/` 未忽略，这两点成立。

### §9.1 文档一致性

以下两项成立：

- `docs/scene-object-system.md` 声称 Viewer session 写 v3，而代码写 v4；
- README/CI 关于 CUDA 测试不会以静默跳过的绿色检查出现的声明，与测试实现不完全一致。

但 README 第 198 行把 `renderer-hardening-results.md` 列为“本次架构硬化报告”，第 200 行说明“旧 `docs/output/`”是历史记录，两句话可以同时成立，并不自相矛盾。

## 测试验证

使用当前已有的 no-CUDA Release 构建执行：

```powershell
ctest --preset no-cuda-release --output-on-failure `
  -R 'cuda_contract_tests|renderer_tests|opengl_contract_tests'
```

结果为：

```text
renderer_tests          Passed
opengl_contract_tests   Passed
cuda_contract_tests     Passed
100% tests passed, 0 tests failed out of 3
```

这验证了 §7.3 的核心问题：在 no-CUDA 环境中，多个真实 CUDA 路径测试没有执行，但相关测试二进制和整个 CTest 运行仍显示绿色通过。

## 使用建议

在把原报告作为整改计划前，建议：

1. 修正本文件指出的事实错误和数量错误；
2. 把“当前已发生的问题”和“未来并发、大场景或维护风险”分开；
3. 为性能结论补充可复现基准，避免使用“秒级卡顿”“数十 MB”等未经测量的表述；
4. 把架构取舍、严重度和整改方案标记为评审意见，而不是代码事实。
