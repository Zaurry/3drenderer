# 当前像素生命周期

当前项目只有 OpenGL 与 Path 两种渲染模式。两条链路最终都产出线性 HDR 颜色，并由同一个 OpenGL compositor 完成曝光、tone mapping 与 linear-to-sRGB 转换。

## 统一帧输出

交互后端返回 `RenderFrameOutput`，显示层只处理两种存储：

```text
Host Framebuffer
  -> 上传到显示 texture
  -> compositor
  -> SDL/OpenGL swap

GL Texture view
  -> compositor
  -> SDL/OpenGL swap
```

OpenGL renderer 与 CUDA interop 都返回 texture view。CPU Path 或 CUDA interop fallback 返回 Host `Framebuffer`。

## OpenGL 像素

OpenGL 链路适合实时预览：

```text
SceneDocument
  -> SceneChangeSet
  -> OpenGlRasterRenderer resource sync
  -> vertex shader
  -> fixed-function triangle rasterization/depth test
  -> fragment shader
  -> GL_RGBA32F output texture
  -> RenderFrameOutput
```

顶点 shader 完成坐标变换并传递法线、UV 与 tangent。硬件光栅化产生 fragment；fragment shader 读取材质与纹理，计算当前点光源/方向光贡献，将线性 HDR 颜色写入 `GL_RGBA32F`。

shader 热重载失败时，`GlShaderProgram` 保留最后一个有效 program，因此预览不会因为一次编译错误而中断。

## CPU Path 像素

CPU Path 是无 CUDA 环境下的功能回退：

```text
pixel + sample seed
  -> Camera::generate_ray()
  -> SceneIntersector / BVH
  -> material evaluation
  -> direct point/directional light visibility
  -> stochastic scatter
  -> Russian roulette
  -> sample radiance
  -> progressive accumulation
  -> Host Framebuffer
```

交互 session 每帧为每个像素增加一个样本。相机、尺寸或影响渲染的场景变更会清零累积；暂停累积后仍会在必要变更发生时生成一帧预览。

## CUDA Path 像素

CUDA 与 CPU Path 保持同一采样和着色语义，但数据与生命周期针对 GPU 重构：

```text
SceneChangeSet
  -> incremental async upload
  -> persistent CUDA stream
  -> initialize_frame_kernel (仅 reset)
  -> render_sample_kernel
  -> accumulation buffer
  -> GL surface 或 delayed resolve
```

### 命中数据流

BVH 遍历阶段只维护紧凑候选：

```text
t + primitive kind/id + barycentric u/v
```

找到最近候选后，才重建 alpha cutout 与 two-sided 判断所需的材质 id、UV、几何朝向。最终最近可见命中确定后，才插值 shading normal。只有最终材质引用 bump texture 时才计算 tangent/bitangent。

阴影射线只要求可见性，因此不会重建不需要的完整着色数据。

### 直接 texture 输出

CUDA/OpenGL interop 活跃时：

```text
render_sample_kernel
  -> accumulation
  -> cudaSurfaceObject
  -> GL_RGBA32F texture
```

此时不分配永久 display buffer，也不下载 framebuffer。interop 不可用或离线输出时，后端才延迟创建 resolve buffer；交互 fallback 使用可复用 pinned host staging。

### Reset

同尺寸相机移动、手动 reset 或累积清零只在原有 buffer 上运行 `initialize_frame_kernel`。只有尺寸超过现有容量时才发生新分配。

## SceneChangeSet 对链路的影响

| 变更 | CPU 场景重建 | CUDA 上传 | BVH |
|---|---|---|---|
| 相机、选择、命名、锁定 | 否 | 无 | 不变 |
| 灯光/环境色 | 更新 render scene | 灯光 | 不变 |
| 材质参数 | 更新 render scene | 材质与 binding | 不变 |
| 纹理数据 | 更新 render scene | 纹理 | 不变 |
| 物体变换/几何 | 更新 render scene | 几何与 BVH | 重建 |
| 导入、undo/redo | 完整重建 | All | 重建 |

## 显示变换

曝光和 tone mapper 不改变 Path 累积。compositor 顺序为：

```text
sanitize NaN/Inf
  -> clamp negative values
  -> exposure EV
  -> None / Reinhard / ACES
  -> linear-to-sRGB
  -> display
```

离线 PNG 仍由 `Image::write_png()` 完成输出转换；它不经过 Viewer compositor。
