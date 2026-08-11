# 场景对象系统

## 单一渲染边界

Viewer 编辑 `SceneDocument`，OpenGL、CUDA Path、CPU 拾取和离线渲染都消费由它生成的同一类不可变数据：`RenderSceneSnapshot`。

```text
SceneDocument
  ├─ objects / hierarchy
  ├─ shared mesh assets
  ├─ canonical local matrices
  ├─ material overrides
  ├─ environment and lights
  ├─ undo / redo / saved identity
  └─ domain revisions
          │
          ▼
RenderSceneSnapshot
  ├─ shared immutable geometry
  ├─ instances and material tables
  ├─ textures, lights and environment
  └─ eight monotonic revisions
          ├─ OpenGL
          ├─ CUDA Path
          ├─ picking BVH
          └─ offline CUDA renderer
```

项目不再维护“文档对象 + 扁平 Scene + CUDA instance view”三套可独立漂移的数据。OpenGL 如需批处理，可在后端内部从当前快照生成临时扁平数据；这个派生结果不是公共编辑接口，也不能反向修改文档。

## 身份、索引和层级缓存

每个对象和资产分别拥有稳定 `ObjectId`、`AssetId`。`SceneDocument` 维护：

- `ObjectId -> vector index` 哈希索引；
- 父对象到子对象的邻接表；
- 按拓扑顺序生成的 world matrix 和 world bounds 缓存；
- 每个 mesh asset 的持久化 CPU `SceneIntersector`，供拾取复用。

对象查找不再扫描整个 vector，world matrix/bounds 也不再由每个查询重复递归。层级或空间数据变化会集中使对应缓存失效。

## 变换真值

`SceneObject::transform.local_matrix` 是对象变换的权威数据，必须满足：

- 所有元素有限；
- 最后一行为仿射形式 `[0, 0, 0, 1]`；
- 左上 3×3 线性部分可逆。

TRS 只是矩阵可无损分解时提供给 Inspector 的编辑视图。含剪切的矩阵不会被静默近似为 TRS。重父级直接计算：

```text
new_local = inverse(new_parent_world) * old_world
```

候选矩阵先整体验证，验证失败时不修改 parent、matrix、history 或 dirty 状态。成功后才能作为一次事务提交。因此剪切、负缩放和非均匀缩放可以经过导入、重父级、undo/redo 和保存回读而不丢失。

## 编辑事务和只读视图

公共读取接口只返回只读数据：

- `objects()` 返回 `const vector<SceneObject>&`；
- `find()`、`asset_for_object()` 返回 `const` 指针；
- `assets()` 返回 `shared_ptr<const SceneMeshAsset>`；
- 环境和快照均为只读视图。

修改通过类型化 setter 或 `SceneEditTransaction` 完成。事务启动时保存完整回滚状态；`commit()` 统一创建历史节点，`cancel()` 或析构未提交时整体恢复。连续 gizmo 拖动可使用相同 merge key 合并为一个 undo group，但每次预览 setter 仍立即：

- 增加相应 revision；
- 使快照/空间缓存失效；
- 更新 dirty 状态；
- 让渲染后端看到最新值。

这样不存在“先取得可变指针，再绕过 revision、dirty 或 history”的写入路径。

## Revision 契约

`SceneRevisions` 包含八个单调计数器：

| Domain | 内容 | 典型 CUDA 行为 |
|---|---|---|
| `topology` | 资产/实例集合与父子结构 | 重建 TLAS 拓扑 |
| `geometry` | 顶点、索引、primitive 数量 | 仅重建变化 asset 的 BLAS |
| `transforms` | 实例 world/inverse/normal matrix 与 bounds | 上传 instance，refit TLAS |
| `material_bindings` | primitive 到材质槽的映射 | 与几何一致地上传绑定 |
| `materials` | 材质参数和发光状态 | 更新材质与 emissive 表 |
| `textures` | 纹理描述、texel、alpha | 更新纹理资源 |
| `lighting` | point/directional/spot lights | 更新灯光 buffer |
| `environment` | HDRI、强度、旋转、背景可见性 | 更新环境分布和资源 |

后端保存上次已消费的 revisions 并自行比较。`InteractiveFrameState::scene_changes` 仅保留为内部兼容状态，不再是场景同步真值；UI 传错或漏传 change flag 都不能制造旧缓存。

## 资产、实例和材质槽

`RenderSceneAssetSnapshot` 持有 `shared_ptr<const Scene>`，因此快照中的共享几何在帧租约结束前不会悬空。资产有稳定 ID 和独立 geometry revision；CUDA BLAS cache 以两者识别变化。

材质槽使用 `MaterialSlot`：

- `missing()` 明确表示未绑定；
- `bound(index)` 表示有效的本地材质索引；
- `-1` 只在 GPU 打包边界表示 missing，不能先加 instance material offset；
- 正数越界在快照构建时抛错；
- missing 始终映射到全局诊断材质。

几何与材质槽表必须具有相同 primitive 数量。CUDA 先在 host 构建并验证完整候选 device scene，只有上传成功后才发布新 view，因此 geometry 数量变化不会与上一版本绑定表混用。

## 程序球

程序球进入快照时统一引用一份 64×32 平滑单位球网格，包含稳定 winding、平滑法线、UV0/UV1 和 tangent；center/radius 被编码为实例矩阵。多个程序球只上传/构建一次共享几何和 BLAS。文档拾取缓存使用由同一单位网格变换得到的三角形，因此运行时没有 OpenGL/CUDA/拾取各自解析 sphere 的分支。负缩放和非均匀缩放由实例矩阵、逆矩阵和 inverse-transpose normal matrix 统一处理。

## 文件和会话

`.rscene` 当前写入 v5，local matrix 保存为明确的四个 4 元行数组。v1–v4 仍可读取并迁移，保存时升级到 v5。v5 为各类灯加入投影开关、优先级和光源尺寸，并加入可编辑 `RectAreaLight`。

Viewer 会话当前写入 v3，并保存 Techniques 面板可见性及 IBL、Shadow Map、PCSS、环境主光提取和 LTC 参数。v1–v2 使用这些技术的默认值迁移；v1 中的 `path_backend` 只用于生成迁移提示，不恢复 CPU/Auto backend。若会话请求 Path 但 CUDA 设备或探测 kernel 不可用，Viewer 切到 OpenGL 并显示原因。

场景和会话保存都经过共享原子写入工具：同目录唯一临时文件完成 flush/close 后，Windows 使用 replace + write-through，POSIX 使用 rename。替换前失败会删除临时文件，目标原字节保持不变。
