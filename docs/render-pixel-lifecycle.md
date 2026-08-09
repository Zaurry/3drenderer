# 当前像素生命周期

项目只有 OpenGL 和 CUDA Path 两种渲染模式。两条链路都输出线性 HDR，最后由同一个 OpenGL compositor 执行 exposure、tone mapping 和 linear-to-sRGB。

## 帧所有权

交互后端返回 `RenderFrameOutput` variant：

```text
HostFrameHandle
  └─ shared_ptr<const Framebuffer>

OpenGlTextureHandle
  ├─ texture / width / height / flip_y
  └─ shared lifetime lease
```

显示层按 variant 分发，不识别具体 renderer。handle 持有资源 owner，保证 compositor 完成采样前 framebuffer 或 GL texture 不会被后端 reset/析构。统计信息同样使用 `OpenGlViewerStatistics | CudaPathViewerStatistics` 类型化 variant。

## 从文档到帧

每帧渲染输入是 `const RenderSceneSnapshot&`：

```text
SceneDocument typed edit
  -> domain revision increment
  -> immutable RenderSceneSnapshot
  -> backend compares last consumed revisions
  -> minimum required resource update
  -> RenderFrameOutput lease
  -> compositor
  -> swap
```

相机、尺寸和 integrator settings 不放入场景 revision；CUDA 的 `ProgressiveRenderKey` 单独覆盖这些影响辐射分布的输入。

## OpenGL 像素

```text
RenderSceneSnapshot
  -> flatten only when relevant revisions changed
  -> VAO/VBO/material/texture/light/environment sync
  -> vertex shader
  -> triangle rasterization and depth
  -> opaque + weighted blended transparency passes
  -> GL_RGBA32F linear output texture
  -> OpenGlTextureHandle
```

OpenGL 只在快照 revision 变化时刷新派生批次。程序球已在快照边界转换为标准三角网格，不存在 OpenGL 特有 sphere 分支。

shader 热重载属于 OpenGL backend capability。编译或链接失败时保留最后一个有效 program，并在 GLSL 面板显示完整 driver log。Path backend 不提供空的 reload 实现。

## CUDA Path 像素

```text
RenderSceneSnapshot revisions
  -> validated host packing
  -> changed asset BLAS rebuild / instance TLAS build or refit
  -> persistent device-owned stream and buffers
  -> initialize frame when ProgressiveRenderKey changes
  -> reusable CUDA Graph wavefront stages
       primary
       intersection
       alpha-aware surface reconstruction
       shade / scatter
       direct-light and emissive/environment NEE
       visibility
       accumulate / resolve
  -> accumulation buffer
  -> CUDA/OpenGL surface or delayed host download
```

### 场景发布

几何和 primitive-material slot table 作为一个已验证 schema 打包。正数越界 material slot 在 host 阶段失败；missing slot 映射诊断材质。只有所有需要的 upload 成功后，新 device scene view 才会发布。

资产由稳定 `AssetId + geometry_revision` 标识：

- 某个资产几何变化：只重建该资产 BLAS。
- 实例集合变化：重建 TLAS 拓扑。
- 纯 world matrix/bounds 变化：只上传 instance 并 bottom-up refit TLAS。
- 材质、纹理、灯光或环境变化：只更新对应资源，不重建无关 BVH。

### Surface 和 alpha

遍历、shading 与 emissive NEE 共用 `SurfaceInputs`/alpha evaluation 语义。输入包含 UV0、UV1、顶点颜色/alpha、纹理变换和有效 `AlphaMode`：

- `Opaque` 忽略材质、顶点和纹理 alpha。
- `Mask` 用组合 alpha 与 cutoff 决定命中/发光候选。
- `Blend` 用 coverage 加权发光，并在路径可见性中使用同一 alpha。

这样 emissive 直接采样不会再用错误 UV、忽略 vertex alpha，或把 Opaque 当成 cutout。

### 环境采样

环境 PMF 的 texel 权重为 `luminance × texel solid angle`。选中一行后，CPU 公共契约和 CUDA 都在该行的两个纬度边界之间均匀采样 `cos(theta)`，并在 phi 区间均匀采样。方向 PDF 为：

```text
texel_pmf / texel_solid_angle
```

因此采样分布与报告的立体角 PDF 完全匹配；极区不会因为在 theta 中均匀采样而产生偏差。

### 渐进累积

`ProgressiveRenderKey` 至少包含：

- 八个 snapshot revisions；
- 完整相机 basis 与 viewport；
- 输出尺寸；
- CUDA device ID；
- max bounce、Russian roulette 参数；
- seed；
- 自动交互质量策略。

任何改变辐射分布的 key 变化都会由后端自动清零累积。选中、面板开关、delta time 等非辐射 UI 状态不会清零。调用方的 `SceneChangeSet` 不能替代 revision，也不能强制上传不存在的数据变化。

### Interop 与 fallback

Viewer 创建 CUDA 资源前，从当前 OpenGL context 查询兼容设备并创建 `CudaDeviceContext`。stream、buffer、interop registration 和 surface 必须属于相同 device ID；interop 层不会偷偷切换设备。

interop 活跃时，CUDA 直接写 `GL_RGBA32F` surface，并返回带资源租约的 texture handle。interop 不可用时才复用 pinned host staging 下载 framebuffer。离线 renderer 始终下载最终 image 并写 PNG。

## 显示变换

compositor 顺序：

```text
sanitize NaN/Inf
  -> clamp negative
  -> exposure EV
  -> None / Reinhard / ACES
  -> linear-to-sRGB
  -> display
```

Viewer display controls 不改变 Path 累积。离线 `Image::write_png()` 使用自己的输出转换，不经过 Viewer compositor。
