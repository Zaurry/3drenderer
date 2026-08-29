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

矩形面光的漫反射和 GGX 高光使用 64×64 LTC matrix/amplitude LUT。LUT 及 BSD 许可位于 `shaders/opengl/ltc_1.dds`、`ltc_2.dds` 和 `LTC_LICENSE.txt`。

## Screen-space techniques

SSAO/GTAO 与 SSR 使用以下默认辅助 shader，并与主 raster shader 一同参与自动热重载：

- `fullscreen.vert`
- `screen_space_gbuffer.frag`
- `ambient_occlusion.frag`
- `ao_denoise.frag`
- `ssr.frag`

Screen-space G-buffer 只绘制 Opaque/Mask，复用材质 UV、normal/bump map、opacity、vertex alpha 和 alpha cutoff；Blend 不参与。资源格式为 `DEPTH_COMPONENT32F` 共享深度、`RGB16F` view normal、`R32F` linear depth、两张 `RGBA16F` SSR 材质纹理，以及 `RGBA16F` bent-normal/visibility ping-pong。

默认 raster shader 仅将 AO 应用于环境 IBL。直接光、LTC、Shadow Map、Emission 和天空背景不乘 AO；GTAO 的 Bent Normal 用于漫反射环境方向，并用 GTSO 近似处理镜面环境遮蔽。自定义 raster fragment shader 如需接收该效果，必须声明 binding 15 及对应 AO uniforms。

SSR 在 view space 同时追踪朝向和远离相机的反射射线；候选交点必须由可见表面的前方跨越到后方，并在二分细化后落入配置的 thickness 区间。Glossy 命中使用与 Path 一致的 GGX `alpha = roughness²`，以 GGX 中心波瓣的 FWHM 构造适合普通 box-filter mip 的有效反射锥，并由命中距离、命中深度、相机投影和输出分辨率计算 opaque HDR mip 金字塔的屏幕 footprint LOD；LOD 使用锥的像素半宽，以补偿 trilinear mip 采样本身覆盖相邻 texel 的约双倍支持范围。超出屏幕支持范围的锥会降低置信度并回退环境项。命中项与被替换的环境项共用 split-sum BRDF 响应和相同的 AO/GTSO 调制。Shadow debug view 会旁路 SSR，Blend 在 SSR 之后合成且不参与反射。

## 输出

fragment shader 输出必须为线性 HDR：

```glsl
layout(location = 0) out vec4 out_linear_color;
layout(location = 1) out vec4 out_transparency_accum;
layout(location = 2) out float out_transparency_reveal;
```

场景 shader 不应执行 exposure、tone mapping、gamma 或 sRGB encoding。共享 compositor 在 OpenGL/CUDA Path 都完成后统一做显示变换。

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
