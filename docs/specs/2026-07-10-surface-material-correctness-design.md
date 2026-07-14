# 表面数据与 OBJ/MTL 材质正确性设计

## 背景

当前软件渲染器已经具备 SDL3 交互窗口、raster、ray、path 三种模式、OBJ/MTL 场景加载、BVH、渐进式路径追踪和 `map_Kd` 漫反射纹理。Mary 场景暴露出两个基础问题：OBJ 中已有的顶点法线没有进入渲染管线，导致模型使用逐三角形面法线；asset loader 添加了方向光，但 path tracer 不计算显式点光和方向光，因此场景主要依靠弱环境光可见。

本轮采用“统一表面数据层”方案，先建立 raster、ray 和 path 共享的几何与材质语义，再完善 OBJ/MTL 范围内的透明裁剪、bump 和双面着色。目标是提高教学渲染器的正确性和一致性，同时为后续 PBR 留下自然扩展点。

## 目标

- 读取并插值 OBJ 顶点法线，消除 Mary 等平滑模型不应出现的三角形面感。
- 分离几何法线与着色法线，保证平滑着色不破坏求交、正反面判断和光线偏移。
- 让 path tracer 对点光和方向光执行显式直接光照与阴影测试。
- 统一 raster、ray 和 path 的 `map_Kd`、alpha、bump、双面材质和颜色空间解释。
- 支持 OBJ/MTL 的 `d`、`Tr`、`map_d`、`bump` 和 `map_bump`。
- 修复 raster 的属性透视插值和近裁剪面三角形裁剪。
- 对缺失或退化的可选资产提供可诊断、可回退的行为。
- 通过单元测试、集成 fixture 和 Mary/Sponza 本机视觉测试验证结果。

## 非目标

- 不加入 OpenGL、Vulkan、DirectX、SDL GPU API、CUDA 或 RT Core。
- 不加入可见或运动的光球。
- 不实现 alpha blend、薄表面透射或体积吸收；本轮透明只做 alpha cutout。
- 不实现 tangent-space normal map；本轮只实现高度图 bump。
- 不实现 glTF、完整 PBR、GGX、面光源重要性采样、环境贴图重要性采样或 MIS。
- 不实现 mipmap、各向异性过滤或完整抗锯齿管线。

## 方案选择

采用统一表面数据层，而不是在三个渲染器中分别打补丁，也不提前重构成完整 BSDF/PBR 系统。

三角形求交负责产生可靠的几何信息和插值属性；共享材质求值器负责把材质常量与纹理转换成表面样本；三个渲染器只负责各自的可见性、采样和光照算法。这个边界能立即修复当前问题，也能让后续 `normal map`、PBR texture slots 和 BSDF 接口复用同一份输入。

## 表面数据

`Triangle` 保存以下逐顶点属性：

- position
- UV
- normal
- normal 是否有效

求交产生扩展后的 `HitRecord`（语义上是 `SurfaceInteraction`）：

```cpp
struct HitRecord {
    double t;
    Vec3 position;
    Vec2 uv;
    Vec3 geometric_normal;
    Vec3 shading_normal;
    Vec3 tangent;
    Vec3 bitangent;
    int material_id;
    bool front_face;
};
```

几何法线来自三角形边叉积，用于正反面判断、双面规则、光线偏移和防止自相交。着色法线由三个 OBJ 顶点法线按重心坐标插值并归一化；缺失、非有限或接近零长度时回退到几何法线。

插值后的着色法线必须校正到几何法线所在半球。射线从背面命中双面材质时，几何法线和着色法线同时朝向入射射线的反方向，但二者仍保持同半球关系。

切线与副切线由三角形位置边和 UV 导数计算。UV 退化时，使用着色法线构造稳定正交基，保证 bump 求值不会产生 NaN 或让整个模型加载失败。

## 共享材质求值

新增共享材质求值边界，输入场景、材质和 `HitRecord`，输出：

```cpp
struct SurfaceMaterialSample {
    Color base_color;
    double opacity;
    Vec3 shading_normal;
};
```

`Material` 在现有字段上增加：

- `opacity`
- `opacity_texture_id`
- `bump_texture_id`
- `bump_scale`
- `alpha_cutoff`
- `two_sided`

纹理资源明确区分颜色数据与线性数据。`map_Kd` 以 sRGB 解码到线性空间；alpha 与高度纹理保持线性值。纹理缓存键包含规范化路径和颜色空间，避免同一文件以不同语义加载时错误复用。

OBJ/MTL 导入的材质默认双面着色，因为 MTL 没有可靠、统一的双面字段。内置场景可以显式使用单面材质。

## OBJ/MTL 映射

- `Kd` 与 `map_Kd` 组成 base color。
- `d` 直接映射到常量 opacity。
- `Tr` 映射为 `1 - Tr`；当文件同时提供 `d` 和 `Tr` 时采用 tinyobjloader 已解析的最终 dissolve 语义，避免重复反转。
- `map_d` 作为线性 opacity texture，与常量 opacity 相乘。
- `bump` 与 `map_bump` 作为线性 height texture。
- bump 选项中可可靠读取的强度参数映射到 `bump_scale`；无法识别时使用默认值。

缺失的可选纹理不会使 OBJ 加载失败。loader 记录警告并回退到材质常量。损坏的 OBJ、越界索引和无法构造任何三角形等结构错误继续作为 fatal error。

## Alpha Cutout

最终 opacity 为材质常量与 `map_d` 样本的乘积。低于 `alpha_cutoff` 的命中视为不存在：

- raster 不写颜色和深度。
- ray/path 在同一原始光线上提高 `t_min`，继续寻找后续交点。
- shadow ray 同样跳过透明命中，使镂空区域不产生实心阴影。

继续求交设置有限的最大透明层数，并在每次跳过后使用与命中尺度相关的小步进，避免恶意或退化资产造成无限循环。半透明数值不会做混合，只按 cutoff 分类。

## Bump Mapping

高度图在命中 UV 附近按一个纹素的偏移采样，计算有限差分 `dh/du` 和 `dh/dv`。差分通过 TBN 转换为扰动后的世界空间着色法线。结果必须归一化，并再次校正到几何法线半球。

Bump 只改变着色法线，不改变轮廓、深度、几何法线或 BVH。纹理缺失、尺寸无效、UV 退化或结果非有限时，回退到未扰动的着色法线。

## Raster 流程

Raster 顶点阶段携带 world position、UV、normal 和 reciprocal view depth。像素阶段使用 reciprocal depth 对这些属性执行透视正确插值；当前只对 world position 做正确插值、对 UV 做仿射插值的差异将被消除。

三角形穿过近裁剪面时，在 view space 对带属性顶点进行 Sutherland-Hodgman 裁剪，再将生成的三角形送入现有 edge-function 光栅化。完全位于近裁剪面后的三角形才被丢弃。

材质求值发生在深度写入之前，alpha cutout 像素不能占据 depth buffer。双面材质根据 view direction 调整着色法线；单面材质进行背面剔除。

## Ray 与 Path 流程

Ray tracer 和 path tracer 共享 alpha-aware scene intersection。所有反射、折射、散射和阴影射线通过统一的 `offset_ray_origin(position, geometric_normal, direction)` 发射。偏移方向由出射方向与几何法线的符号决定，偏移大小按位置尺度设置；固定 `t_min = 0.001` 不再承担掩盖自相交的职责。

Path tracer 在每次 diffuse 命中时显式计算场景中的点光和方向光：

- 点光使用距离平方衰减、Lambert BRDF、余弦项和 shadow ray。
- 方向光使用固定入射方向、Lambert BRDF、余弦项和 shadow ray。
- Metal 和 Dielectric 继续通过其递归方向采样处理，不额外添加漫反射直接光。
- 环境光仍通过路径未命中时返回 environment 处理。
- 不创建可见光球或隐式动画灯。

点光和方向光是 delta lights，因此本轮直接求和即可，不需要为它们引入 MIS。后续加入面积光和环境贴图采样时，再将直接光接口扩展为带 PDF 的 light sampler。

## 错误与诊断

`LoadedScene` 增加 warnings 集合。CLI renderer 和 viewer 在加载后输出每条 warning，但仍继续渲染。warning 至少覆盖：

- MTL 引用的纹理不存在或解码失败。
- 顶点法线索引缺失、越界或法线退化而发生回退。
- bump 使用退化 UV 而被忽略。
- 无法识别的可选 bump 参数。

重复问题按资源路径或类别去重，避免大型模型打印数万条相同警告。

## 测试策略

自动化测试使用仓库内生成或保存的小型 OBJ/MTL fixture，不依赖被 `.gitignore` 排除的 Computer Graphics Archive。

单元与集成测试覆盖：

- OBJ `vn` 加载、重心插值、缺失法线回退和半球校正。
- `d`、`Tr`、`map_d` 解析及 alpha-aware primary/shadow intersection。
- `bump/map_bump` 的线性采样、TBN 扰动与退化 UV 回退。
- sRGB color texture 与 linear data texture 的数值差异。
- raster UV、world position 和 normal 的透视正确插值。
- 穿越近裁剪面的三角形仍产生预期像素。
- path 点光/方向光直接照明、阴影和点光距离衰减。
- 光线偏移不会立即重新命中刚离开的表面。
- 双面材质在 raster、ray、path 中保持一致。

本机视觉验证覆盖：

- Mary raster/path：面状感明显消失，纹理仍正确，path 能被 asset 方向光照亮。
- Sponza raster/path：`map_Kd`、alpha cutout 和 bump 资产路径正确，缺失可选资源只产生去重 warning。
- CornellBox：现有 emissive、mirror、water 等场景不发生明显回归。

## 已知后续缺口

本轮完成后，渲染器仍不是现代完整 PBR renderer。后续按优先级处理：

1. tangent-space normal map 与可靠 tangent 生成（可对齐 MikkTSpace）。
2. GGX/Smith/Fresnel-Schlick 与 BSDF `evaluate/sample/pdf` 接口。
3. 面光源采样、环境贴图重要性采样、Russian roulette 和 MIS。
4. alpha blend、transmission、薄表面和体积。
5. mipmap、各向异性过滤和更完整的采样状态。
6. glTF 2.0、场景层级、实例与变换。
7. raster MSAA、ray/path 自适应采样和降噪。
8. BVH 构建与遍历优化、线程池、SIMD 和未来 CUDA backend。

## 成功标准

- 全部现有测试和新增测试通过。
- Mary 的 OBJ 顶点法线进入三个渲染模式，平滑区域不再使用逐面法线。
- Path 模式正确显示 asset loader 配置的方向光，并产生可解释的阴影。
- 三个渲染器对 base color、alpha、bump、双面和颜色空间使用共享语义。
- Raster 对属性执行透视正确插值，并正确处理穿越近裁剪面的三角形。
- 可选纹理错误可见且可回退，结构性场景错误仍明确失败。
- README 准确记录已实现能力、命令示例和剩余限制。
