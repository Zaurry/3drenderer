# OpenGL GLSL Shader 合约

Viewer 的 `opengl` 模式从磁盘加载 OpenGL 4.5 Core (`#version 450 core`) vertex/fragment shader。默认文件：

- `shaders/opengl/raster.vert`
- `shaders/opengl/raster.frag`

Viewer 每 250 ms 检查修改时间；成功 compile/link 后原子替换 program，失败时继续使用最后一个有效 program，并在 GLSL Shader 面板显示 driver log。`F5` 只对 OpenGL backend 生效。

合约的机器可读真值位于 `src/render/opengl/opengl_shader_contract.h`，`opengl_contract_tests` 会解析默认 shader 并验证属性 location、texture binding、SSBO binding 和 material uniform 名称。修改 shader 或 renderer binding 时必须同步更新该清单。

## Vertex attributes

| Location | GLSL type | Name | 含义 |
|---:|---|---|---|
| 0 | `vec3` | `a_position` | world-space position |
| 1 | `vec3` | `a_normal` | world-space shading normal |
| 2 | `vec2` | `a_uv` | UV0 |
| 3 | `vec4` | `a_tangent` | world tangent；`w` 为 handedness，0 表示无有效 basis |
| 4 | `vec2` | `a_uv1` | UV1 |
| 5 | `vec4` | `a_color` | vertex RGB + alpha |

几何在 OpenGL 派生批次中已经位于 world space，因此没有 model matrix。必需相机 uniform：

```glsl
uniform mat4 u_view_projection;
uniform vec3 u_camera_position;
```

## Light buffers

固定 `std430` binding：

| Binding | Block | Element layout |
|---:|---|---|
| 0 | `DirectionalLightBuffer` | `vec4 direction_angular; vec4 radiance; vec4 shadow;` |
| 1 | `PointLightBuffer` | `vec4 position_range; vec4 intensity; vec4 shadow;` |
| 2 | `SpotLightBuffer` | `vec4 position_range; vec4 direction_inner; vec4 intensity_outer; vec4 shadow;` |
| 3 | `RectAreaLightBuffer` | `vec4 position_two_sided; vec4 axis_u; vec4 axis_v; vec4 radiance; vec4 shadow;` |

计数 uniforms：

```glsl
uniform int u_directional_light_count;
uniform int u_point_light_count;
uniform int u_spot_light_count;
uniform int u_rect_area_light_count;
```

## Texture units

| Unit | Uniform |
|---:|---|
| 0 | `u_base_color_texture` |
| 1 | `u_opacity_texture` |
| 2 | `u_normal_or_bump_texture` |
| 3 | `u_metallic_roughness_texture` |
| 4 | `u_occlusion_texture` |
| 5 | `u_emissive_texture` |
| 6 | `u_specular_texture` |
| 7 | `u_specular_color_texture` |
| 8 | `u_specular_glossiness_texture` |
| 9 | `u_environment_prefilter` |
| 10 | `u_environment_brdf_lut` |
| 11 | `u_ltc_matrix_lut` |
| 12 | `u_ltc_amplitude_lut` |
| 13 | `u_shadow_maps_2d` (`R32F` 2D array) |
| 14 | `u_shadow_maps_cube` (`R32F` cube array) |
| 15 | `u_ambient_occlusion_texture` (`RGBA16F`: view-space bent normal + visibility) |

材质纹理 slot 0–8 还必须支持：

```glsl
uniform vec4  u_texture_offset_scale[9];
uniform float u_texture_rotation[9];
uniform int   u_texture_texcoord[9];
uniform int   u_texture_top_left[9];
```

其中 `u_texture_texcoord` 为 0/1，选择 UV0/UV1；transform 顺序与 `KHR_texture_transform` 契约一致。是否存在纹理由对应的 `u_has_*_texture` uniform 指示。

## Material uniforms

机器校验的核心 material field 清单：

```glsl
u_material_type
u_pbr_workflow
u_base_color
u_emission
u_ior
u_specular_color
u_specular_factor
u_glossiness
u_metallic
u_roughness
u_opacity
u_alpha_cutoff
u_bump_scale
u_normal_scale
u_occlusion_strength
u_alpha_mode
u_two_sided
u_texture_texcoord
```

默认 shader 还读取各 `u_has_*_texture`、texture transform、环境 SH/prefilter、环境强度/旋转，以及 `u_transparent_pass`。自定义 shader 可以不使用某些输入；OpenGL 允许把未使用 uniform 优化掉。

Alpha 语义：

- Opaque 忽略 alpha，写入不透明 color target。
- Mask 使用组合后的材质、纹理和 vertex alpha 与 cutoff。
- Blend 写入 weighted blended OIT accumulation/reveal targets。

## 直接阴影与 LTC

方向光、聚光和环境提取主光读取线性深度 2D array；点光和矩形灯读取 cube array。`shadow` 字段保存光源角/世界尺寸、纹理层与全局调试槽。PCSS 使用稳定旋转的 Vogel disk，先搜索 blocker、还原线性 blocker 距离，再执行可变半径 PCF。关闭 PCSS 时只比较中心样本。

Shadow pass 只绘制 Opaque/Mask 批次。Mask 必须复用 UV0/UV1、base-color/opacity alpha、`KHR_texture_transform` 和 alpha cutoff；Blend 不投影。该 pass 的 shader 内置于 OpenGL renderer，契约测试会校验 `R32F` 资源和 alpha 输入。

矩形面光的漫反射和 GGX 单次散射高光使用 64×64 LTC matrix/amplitude LUT；
Kulla–Conty 多次散射波瓣另以 2×2 面积中点积分计算，避免错误复用单次散射 LTC
幅值。环境 BRDF LUT 为 `RGBA16F`：RG 是 split-sum Fresnel scale/bias，BA 是
`E(N·V, roughness)` 与 `Eavg(roughness)`。LUT 及 BSD 许可位于
`shaders/opengl/ltc_1.dds`、`ltc_2.dds` 和 `LTC_LICENSE.txt`。

## Screen-space techniques

SSAO/GTAO 与 SSR 使用以下默认辅助 shader，并与主 raster shader 一同参与自动热重载：

- `fullscreen.vert`
- `screen_space_gbuffer.frag`
- `ambient_occlusion.frag`
- `ao_denoise.frag`
- `ssr.frag`

SSGI 使用独立的原子热重载组；组内任一 compile/link 失败时保留上一套全部有效 program，首次加载失败则旁路 SSGI：

- `ssgi_hiz.frag`
- `ssgi_trace.frag`
- `ssgi_temporal.frag`
- `ssgi_denoise.frag`
- `ssgi_composite.frag`

Screen-space G-buffer 只绘制 Opaque/Mask，复用材质 UV、normal/bump map、opacity、vertex alpha 和 alpha cutoff；Blend 不参与。资源格式为 `DEPTH_COMPONENT32F` 共享深度、`RGB16F` view normal、`R32F` linear depth、两张 `RGBA16F` SSR 材质纹理、一张 `RGBA16F` SSGI receiver 材质纹理（RGB 为与主 PBR 路径一致的视角相关漫反射响应，A 为材质 occlusion），以及 `RGBA16F` bent-normal/visibility ping-pong。主 Opaque pass 另写 `RGB16F` MRT，记录实际加入颜色的 diffuse IBL；天空、Unlit 和 Blend 写零。SSGI trace 命中时输出“屏幕命中辐射－同一余弦半球方向环境辐射”的有符号 control-variate 残差，未命中输出零。只有残差进入半分辨率时域、à-trous 与双边上采样，原始全分辨率 diffuse IBL 从不被滤波或重采样；最终把滤波残差直接加到 Opaque HDR。这样既不重复叠加 diffuse IBL，也不会模糊原始材质/法线细节；GTAO/Bent Normal 继续负责环境 IBL 的近场遮蔽。

默认 raster shader 仅将 AO 应用于环境 IBL。直接光、LTC、Shadow Map、Emission 和天空背景不乘 AO；GTAO 的 Bent Normal 用于漫反射环境方向，并用 GTSO 近似处理镜面环境遮蔽。自定义 raster fragment shader 如需接收该效果，必须声明 binding 15 及对应 AO uniforms。

SSR 在 view space 同时追踪朝向和远离相机的反射射线；候选交点必须由可见表面的前方跨越到后方，并在二分细化后落入配置的 thickness 区间。逐像素 normal（含 normal/bump map）决定射线方向，逐像素 roughness 决定 sharp/glossy 波瓣。Glossy 命中使用与 Path 一致的 GGX `alpha = roughness²`，以 GGX 中心波瓣的 FWHM 构造反射 footprint；半径随命中距离增长，从而保留接触处锐利、远处模糊的 contact hardening。波瓣在入射平面保持长轴，而垂直方向按 `N·V` 收缩，再经过命中点的透视 Jacobian 投影为定向椭圆；minor axis 选择 opaque HDR mip LOD，最多 9 个 FWHM 加权采样沿 major axis 完成 specular elongation，并将最大各向异性限制为 8。超出椭圆屏幕支持范围时会降低置信度并回退环境项。命中项与被替换的环境项共用 split-sum BRDF 响应和相同的 AO/GTSO 调制。Shadow debug view 会旁路 SSR，Blend 在 SSR 之后合成且不参与反射。

SSGI 顺序为 G-buffer → Hi-Z/AO → Opaque 光照 → SSGI trace/temporal/denoise/composite → SSR → transparent OIT。全分辨率 `RG32F` Hi-Z 的 R/G 分别保存区域内最近/最远的有效线性深度，空区域为 `(FLT_MAX, 0)`；奇数和 NPOT mip 按目标 texel 的实际覆盖归约最多 3×3 个源 texel。追踪在 `(width+1)/2 × (height+1)/2` 执行，每个 2×2 footprint 选择最近有效表面，最多使用 8 rays、256 次 Hi-Z cell visit 和 16 次二分细化。level 0 只接受从可见表面前方到后方的 crossing，并执行 thickness、自交、距离和命中背面验证。

SSGI 半分辨率历史为两套 ping-pong：`RGBA16F` signed residual/history length、`RG16F` signed-luminance moments、`R32F` linear depth 和 `RGB16F` view normal。重投影的四个 bilinear tap 分别用 `max(2×thickness, 1%×predictedDepth)` 与世界法线 dot ≥ 0.85 验证，之后重新归一化；历史残差使用当前 raw residual 的 3×3 YCoCg 邻域和 1.5σ 方差裁剪。默认执行 stride 1/2/4、横纵向 radius-2 的双边 à-trous，再以全分辨率深度/法线/材质响应进行 3×3 双边上采样。无可靠邻居时残差回退零，最终为 `max(opaque + strength × filteredResidual, 0)`。

命中读取完整的 pre-SSR Opaque HDR；未命中或边缘支持不足保留 raster pass 实际写入的 diffuse IBL，IBL 关闭时该基线及方向环境样本均为黑色。源纹理不含 SSGI/SSR，因此只有一次反弹。相机移动使用历史重投影；首次启用、resize、显式 reset、场景/光照设置变化和 shader 成功重载会使历史失效。SSGI 调试视图旁路 SSR 与透明合成。

## 输出

fragment shader 输出必须为线性 HDR：

```glsl
layout(location = 0) out vec4 out_linear_color;
layout(location = 1) out vec4 out_transparency_accum;
layout(location = 2) out float out_transparency_reveal;
layout(location = 3) out vec3 out_diffuse_ibl;
```

场景 shader 不应执行 exposure、tone mapping、gamma 或 sRGB encoding。共享 compositor 在 OpenGL/CUDA Path 都完成后统一做显示变换。

SSGI/SSR 都是屏幕空间近似：屏外、被前景遮挡的几何和透明表面不可见；SSGI 不提供透明 GI、镜面 GI、物体运动向量或多次反弹。现有 SSR 仍使用自己的线性 ray march，不读取 SSGI Hi-Z。

## 启动示例

默认 shader：

```powershell
.\build\default\bin\viewer.exe --scene builtin --mode opengl --no-restore-last
```

自定义 shader：

```powershell
.\build\default\bin\viewer.exe `
  --scene asset `
  --asset path\to\scene.obj `
  --mode opengl `
  --gl-vertex-shader path\to\custom.vert `
  --gl-fragment-shader path\to\custom.frag `
  --no-restore-last
```
