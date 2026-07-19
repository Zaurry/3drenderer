# 现代 PBR 材质路线图

当前材质系统是教学用的简化模型：`Diffuse`、`Metal`、`Dielectric`、`Emissive`，再加上少量 OBJ/MTL 参数。要支持现代 PBR 材质，需要逐步补齐以下能力。

## 资产与材质数据

- 优先支持 glTF 2.0 metallic-roughness 工作流，再兼容 OBJ/MTL 的 `Pr`、`Pm`、`map_Pr`、`map_Pm`、`norm` 和 `map_Ke` 等扩展字段。
- 将 `Material` 扩展为包含 base color、metallic、roughness、normal、occlusion、emissive、alpha mode、alpha cutoff、IOR、transmission 和 clearcoat 的参数集。
- 第一阶段优先完成 base color、metallic、roughness、normal、emissive 和 alpha cutout。

## 纹理与几何属性

- 当前已经区分 sRGB 颜色纹理和线性数据纹理；后续还需支持多 UV set、独立 wrap/filter、UV transform、mipmap 和各向异性过滤。
- Emissive 通常按 sRGB 读取；normal、roughness、metallic 和 occlusion 必须保持线性数据。
- 当前能从 OBJ 三角形的 UV 导数建立 bump 所需 TBN；normal map 仍需可靠的 tangent 生成，后续可对齐 MikkTSpace 并支持 glTF tangent。

## 着色与路径采样

- 实现 GGX/Trowbridge-Reitz 法线分布、Smith 几何遮蔽、Fresnel-Schlick，以及能量守恒的 diffuse/specular 混合。
- 让 raster、ray 和 path 通过统一的材质评估接口取样和求值。
- Path 后端需要完整的 BSDF sample/pdf/evaluate、面积光与环境光采样和 MIS，改善粗糙金属、室内间接光和小面积光源的收敛。

## Alpha、输出与验证

- 当前三种渲染器均支持 alpha cutout；ray/path 会跳过透明命中。后续再加入 alpha blend、transmission、薄表面和体积吸收。
- 在线性 HDR framebuffer、曝光和 tone mapping 基础上继续补充白平衡与一致的 sRGB 输出转换。
- 加入小型 glTF/OBJ fixture，覆盖材质参数、贴图颜色空间和 alpha cutout，并使用 Khronos glTF sample models 或自制参考图进行视觉回归。
