# 场景对象系统

Viewer 编辑的是 `SceneDocument`。CPU Path 与 OpenGL 继续消费延迟生成的扁平
`Scene`；CUDA Path Viewer 直接消费缓存的 `InstancedSceneView`。二者分离，使对象
层级、身份、历史记录和文件语义不侵入 CPU/OpenGL，同时避免 CUDA 拖动时展开
数百万三角形。

## 数据流

```text
SceneDocument
  +-- assets
  +-- objects / hierarchy
  +-- transforms
  +-- material overrides
  +-- undo / redo
  +-- file/session state
          |
          v
      /                         \
     v                           v
rebuild_render_scene()    instanced_render_scene()
  (lazy)                    (cached metadata)
     |                           |
     v                           v
   Scene                  asset BLAS + instance TLAS
  /     \                         |
OpenGL  CPU Path               CUDA Path
```

## 对象身份与层级

每个对象都有稳定的 `ObjectId`。父子关系、独立 transform、visible、locked、名称与材质 override 都保留在文档层；导入多个 OBJ 不会把对象身份压平。

对象世界矩阵由父级到子级组合。mesh 顶点与法线在生成 `Scene` 时变换；点光源位置和方向光方向同样由对象世界矩阵得到。

## Asset

`SceneMeshAsset` 保存导入源、局部几何、局部材质、纹理与局部 bounds。多个对象可引用同一个 asset，同时保留独立 transform 与材质 override。

目录导入会递归发现支持的资产，并保持每个导入对象的独立身份。默认资产场景会添加一个方向光，便于首次预览。

## 材质 override

override 以对象和 material slot 为粒度。生成扁平 `Scene` 时：

1. 解析 asset 的基础材质与纹理索引。
2. 对对象覆盖的 slot 应用 override。
3. 生成全局 material 数组。
4. 写入 primitive-material binding。

因此材质编辑需要更新材质与 binding，但不需要重建 BVH。

## SceneChangeSet

编辑器操作必须返回足够精确的变更分类：

| 操作 | 变更 |
|---|---|
| 相机、选择、命名、locked | `None` |
| 环境色、点光源、方向光 | `Lighting` |
| 材质 override | `Materials | MaterialBindings` |
| mesh/group transform | `InstanceTransforms`，若包含 light 还包括 `Lighting` |
| visible 改变 | 对受影响子树按拓扑分类 |
| 导入、删除、复制、reparent、undo/redo | `All` |

`viewer_main` 只负责将 UI 与外部导入产生的变更合并。实际后端同步由统一 Viewer backend 完成。

## 保存与会话

`.rscene` 保存文档结构、asset 引用、对象层级、transform、材质 override 与灯光。Viewer 会话在此基础上额外保存：

- 主窗口尺寸，以及由 `imgui.ini` 保存的 docking、面板尺寸和外置多视口绝对位置。
- 当前模式（仅 `opengl` / `path`）。
- Path backend 与 CPU 参数。
- 相机状态。
- 选择、gizmo 与显示设置。

运行时资产表可以暂时保留无对象引用的资产以支持 undo/redo；`.rscene` 和 Viewer 会话只序列化当前 mesh 对象引用的资产。恢复旧快照时也会跳过孤立资产，避免重新加载已经删除的模型。

会话格式版本保持为 1。旧 `max_depth` 会被忽略；旧 `raster` / `ray` 模式会话被拒绝。

## 渲染边界

后端不得直接依赖编辑器对象：

- OpenGL backend 根据 `SceneChangeSet` 更新 GL 资源。
- CPU Path 使用扁平 `Scene` 与 `SceneIntersector`。
- CUDA Path Viewer 使用 asset-local BLAS 与 instance TLAS；只有切换到 CPU/OpenGL
  或真正需要扁平数据时才延迟生成 `Scene`。

这个边界允许后续增加新的对象组件或渲染后端，而不要求修改显示层或破坏现有对象身份。
