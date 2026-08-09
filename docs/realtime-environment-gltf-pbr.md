# 实时环境光、glTF 与统一 PBR

## 能力范围

项目现在可以从 Viewer、离线 `renderer` 和 `.rscene` 文档加载 2:1 经纬度环境图：

- Radiance HDR (`.hdr`)；
- OpenEXR (`.exr`)；
- PNG、JPG/JPEG（按 sRGB 解码为线性颜色）。

环境图同时用于光照和背景。强度、绕 Y 轴旋转、背景是否可见可以独立调整；隐藏背景
不会关闭光照。加载时会验证像素有限、尺寸为正且宽度严格等于高度的两倍。全黑资源
使用均匀球面分布，避免重要性分布退化。

资产入口支持 OBJ/MTL、glTF 2.0 (`.gltf`) 和 GLB (`.glb`)。glTF 默认场景会保留：

- 节点层级、局部变换、共享 mesh 与实例；
- POSITION、NORMAL、TANGENT、TEXCOORD_0/1、COLOR_0；
- metallic-roughness PBR、base color、normal、occlusion、emissive 纹理；
- alpha OPAQUE/MASK/BLEND、double-sided、sampler wrap/filter；
- `KHR_texture_transform`；
- 静态 perspective/orthographic camera；
- `KHR_lights_punctual` 的 directional、point、spot light，包括 range 平滑衰减与
  spot inner/outer cone。

直接以 glTF 启动 Viewer 时，第一台有效场景相机会成为初始视图。运行时选择 Camera
对象后可在 Inspector 编辑参数，并用 `Look through camera` 切换到该视图。

## 两条渲染链路

### OpenGL

环境图在加载时转换为 cubemap，并生成：

- SH9 漫反射辐照度；
- 按 GGX roughness 预过滤的 specular mip chain；
- Split-Sum BRDF LUT。

材质使用 GGX/Trowbridge-Reitz NDF、Smith masking-shadowing 和 Fresnel-Schlick。
alpha BLEND 使用 weighted blended OIT 的 accumulation/revealage 缓冲，避免按对象排序。

### CUDA Path

CUDA Wavefront 保留持久化资源与 CUDA/OpenGL interop；环境 texel、PMF、CDF 只在
`environment` revision 变化时上传。环境 NEE 与 emissive triangle NEE 组成
混合策略，并与 visible-GGX BSDF 样本做 MIS。interop 活跃时仍直接写 GL texture，
不会引入逐帧 framebuffer 下载。

## 命令行

实时预览：

```powershell
.\build\default\bin\viewer.exe `
  --scene asset `
  --asset D:\assets\model.glb `
  --environment D:\assets\studio.exr `
  --environment-intensity 1.25 `
  --environment-yaw 30 `
  --mode opengl
```

CUDA Path 离线输出：

```powershell
.\build\default\bin\renderer.exe `
  --mode path `
  --scene asset_viewer `
  --asset D:\assets\model.glb `
  --environment D:\assets\studio.hdr `
  --hide-environment-background `
  --width 1280 `
  --height 720 `
  --spp 256 `
  --cuda-device 0 `
  --output output\model-hdri.png
```

旧的 `--scene obj_viewer --obj file.obj` 仍兼容；新代码应优先使用
`--scene asset_viewer --asset file`。

## 与 GAMES202 第 5、6 讲的关系

第 5 讲是本实现的直接理论基础：经纬度环境映射、漫反射卷积、prefiltered environment、
Split-Sum approximation 和 BRDF LUT 都用于 OpenGL IBL；环境方向映射和重要性采样也
用于 Path 后端。

第 6 讲只使用了低频表示的边界思想：环境加载时计算 SH9，OpenGL 用它重建 diffuse
irradiance。项目没有实现完整 PRT 的预计算 transport、visibility transfer 或多次反弹
transfer，因为当前目标是环境、材质和对象都可动态编辑；完整 PRT 会引入静态几何假设
和额外烘焙流程。

构建、真实 HDR/EXR/GLB 冒烟、Compute Sanitizer 与基线性能数据见
[实时环境光、glTF 与 PBR 验证及性能结果](output/realtime-environment-gltf-pbr-results.md)。

## 明确限制

- 不支持 animation、skin、morph target；发现时明确拒绝，不静默丢数据。
- required Draco、meshopt、KTX2/BasisU 扩展会明确拒绝；核心未压缩 glTF/GLB 可加载。
- 正交相机参数会保留和序列化，但当前 Path 相机射线为透视模型，Viewer
  用 45° 透视视图预览正交相机。
- 环境预处理为 load-time、memory-only，不写磁盘缓存；更换 HDRI 会重新预处理。
- OpenGL weighted blended OIT 是顺序无关的实时近似，不等同于逐像素精确透明排序。
- 当前 PBR 核心不含 transmission、clearcoat、sheen、volume 等扩展工作流。
