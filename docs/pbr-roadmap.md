# 现代 PBR 材质状态与后续路线

基础 metallic-roughness 主线已经落地，具体能力与命令见
[实时环境光、glTF 与统一 PBR](realtime-environment-gltf-pbr.md)。

## 已完成

- glTF 2.0/GLB 静态默认场景、层级、共享 mesh 实例、camera 与 punctual lights；
- base color、metallic-roughness、normal、occlusion、emissive、alpha、double-sided；
- 双 UV、glTF tangent、vertex color、sampler wrap/filter、`KHR_texture_transform`；
- OpenGL / CPU Path / CUDA Path 统一 GGX、Smith 与 Fresnel-Schlick；
- HDRI 重要性采样、Path NEE/MIS、OpenGL Split-Sum IBL；
- OpenGL weighted blended OIT 与 Path 随机 alpha BLEND。

## 后续优先级

1. MikkTSpace tangent 生成与 mip/各向异性过滤的完整一致性。
2. glTF transmission、IOR、clearcoat、sheen 与 volume 扩展。
3. 正交相机的原生 raster/ray 支持，而非透视预览。
4. 可选的环境预处理磁盘缓存与后台异步预处理。
5. animation、skin、morph target 和压缩纹理/几何扩展。
6. Khronos sample models 与固定 HDRI 的图像回归基线。
