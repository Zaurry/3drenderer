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

SSR 在本项目中指 **Screen Space Ray Tracing**。旧 reflection pass 与 SSGI 已合并，只有一个完整反射 BRDF 积分器、一个 Hi-Z、一套历史与去噪流程，以及一次间接光合成。

顺序为 G-buffer → Hi-Z/AO → Opaque/Mask 光照与 OIT accumulation → SSR trace/temporal/denoise/composite → OIT composite。透明对象不写追踪深度，也不参与屏幕辐射源。Shadow/AO debug view 旁路 SSR；SSR debug view 旁路透明合成。

辅助 shader：

- `fullscreen.vert`、`screen_space_gbuffer.frag`、`ambient_occlusion.frag`、`ao_denoise.frag`。
- SSR 原子热重载组：`ssr_hiz.frag`、`ssr_trace.frag`、`ssr_temporal.frag`、`ssr_denoise.frag`、`ssr_composite.frag`。任一编译/链接失败时保留整组旧 program，首次失败则旁路。

### 几何与材质

G-buffer 只绘制 Opaque/Mask，复用主材质的 UV、normal/bump、vertex alpha、opacity 和 alpha cutoff。共享深度为 `DEPTH_COMPONENT32F`。六个 color attachment：

| Location | 格式 | 内容 |
|---:|---|---|
| 0 | RGB16F | view-space shading normal |
| 1 | R32F | 正线性深度 |
| 2 | RGBA16F | 镜面 F0、roughness |
| 3 | RGBA16F | 镜面 F90、材质 AO |
| 4 | RGBA16F | diffuse color；A 的 bit 0 为 diffuse Fresnel uses-max，bit 1 为 two-sided |
| 5 | RGBA16F | diffuse Fresnel F0；A 为无色 F90 |

完整材质输入支持 Metallic-Roughness 和 Specular-Glossiness；漫反射 Fresnel 使用每条射线的 V·H。Unlit/Emissive 材质自身不接收 SSR，但保留发光、遮挡及作为辐射源的作用。

### 追踪与积分

`RG32F` Hi-Z 存储区域最近/最远有效深度，空区域为 `(FLT_MAX, 0)`；奇数/NPOT mip 保守归约最多 3×3 个源 texel。追踪使用全分辨率，默认每像素 2 rays、64 cell visits，上限 8 rays/256 visits。level 0 对 `[depth, depth + thickness]` 深度薄层求解析交点，支持朝向和远离相机的射线；不再需要二分步数、粗糙度截止或反射 footprint 的 HDR mip 模糊。

按波瓣能量在余弦半球与 GGX VNDF 间选择采样，使用混合 PDF 和完整 BRDF 的 `f × N·L / pdf` 权重。GGX 使用 `alpha = roughness²`、height-correlated Smith G2 和与 raster/Path 相同的 Kulla–Conty 能量补偿 LUT。VNDF 落到接收面下方的样本贡献零，不重新采样。参考 [Heitz 2018](https://jcgt.org/published/0007/04/01/) 和 [McGuire/Mara 2014](https://jcgt.org/published/0003/04/04/)。

可见性与辐射可靠度分别处理：

- 已命中：采样本帧 `out_ray_radiance`，不乘接收面 AO。单面背面贡献零、双面材质可贡献辐射；两者都遮挡环境。屏幕边缘淡出只降低命中辐射，不重新加入环境。
- 未命中/屏外/达到预算：按方向采样剩余环境，乘材质 AO 和 SSAO/GTAO 可见度作为保守回退。IBL 关闭时回退为零。提取为方向主光的环境能量不再出现在剩余环境中。
- 原始 IBL 不再是每个表面必有的叠加项。SSR 最终颜色为 `directLighting + filteredIndirect`；天空保留 opaque background。

### 光照来源与去重

主 raster 另写两张 RGB32F MRT。`out_direct_lighting` 是本帧直接光 + emission；`out_ray_radiance` 以此为基础，排除与启用的 LTC 矩形灯位置/范围重合、已经由直接光积分器表示的发光辐射。LTC 与屏幕射线不会重复计算同一直接发光连接；没有对应解析灯的发光网格仍可由 SSR 采样。关闭 LTC 时该排除也关闭。

辐射源不包含 IBL、SSR 历史或合成结果，避免把未验证可见性的环境照明带入反弹，也避免两套效果顺序执行造成不一致的反弹次数。当前只传播一次直接光/发光表面连接；没有环境照亮的命中表面再反弹、屏外传播或多次表面反弹。

### 时域与空间滤波

全分辨率 ping-pong：RGBA32F indirect/history length、RG16F luminance moments、R32F linear depth、RGB16F view normal。RGBA32F 保存真实 HDR 辐射，luminance moments 限幅以保持半精度有限。

重投影的四个 bilinear tap 分别验证深度 `max(2×thickness, 1%×predictedDepth)` 和世界法线 dot ≥ 0.85。3×3 YCoCg 历史裁剪排除几何/材质不相容的邻居；相机移动、旋转或 FOV 变化时压低光滑镜面的历史权重。à-trous 默认 stride 1/2/4，使用深度、法线、亮度、diffuse color、F0 和 roughness 权重，光滑镜面旁路空间模糊。没有半分辨率上采样。

首次启用、resize、reset、场景/材质/光照变动、设置变化和 shader 重载会使历史失效。调试枚举依次为 Final、RawIndirect、HitConfidence（实际显示遮挡射线比例）、TemporalIndirect、FilteredIndirect、HistoryLength。

### 设置兼容与边界

仅保留 `OpenGlRenderSettings::ssr`。Viewer 会话 schema v6 写一个 `render.opengl.ssr` 对象；v5 优先继承旧 SSGI 的公共追踪/去噪参数，启用状态为旧 SSR/SSGI 的 OR。v3/v4 继承旧 SSR 的公共参数。旧 reflection intensity、max roughness、jitter、SSGI strength 和 refinement steps 已移除。Benchmark 读取 `ssr`，兼容旧 `ssgi` 参数对象。

默认 raster 在 SSR 关闭/不可用时仍使用原有 SH/prefilter IBL 与 AO/GTSO。屏外、被前景遮挡的几何、透明表面和真实体积厚度无法由单层深度恢复；AO 回退、边缘淡出和去噪仍有偏差。命中辐射来自相机方向的直接光 buffer，因此光泽命中表面的方向性也只是近似。完整物理可见性需要场景空间追踪或 Path backend。

## 输出

自定义 shader 必须输出线性 HDR，不执行 exposure、tone mapping 或 sRGB encoding：

```glsl
layout(location = 0) out vec4 out_linear_color;
layout(location = 1) out vec4 out_transparency_accum;
layout(location = 2) out float out_transparency_reveal;
layout(location = 3) out vec3 out_direct_lighting;
layout(location = 4) out vec3 out_ray_radiance;
```

Opaque pass 的 location 3/4 分别遵循上述直接光/追踪辐射语义，Blend pass 写零。旧自定义 shader 没有名为 `out_direct_lighting` 的 location 3 或 `out_ray_radiance` 的 location 4 时，renderer 自动旁路 SSR，保留该 shader 的 opaque 输出。


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
