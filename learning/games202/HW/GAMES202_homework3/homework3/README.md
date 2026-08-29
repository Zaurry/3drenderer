# GAMES202 Homework 3 - Screen Space Ray Tracing

本实现完成了作业要求的三个核心部分：

1. 直接光照：`EvalDiffuse` 使用 Lambertian BSDF `kd / pi`，`EvalDirectionalLight` 读取 GBuffer 中由 Shadow Map 生成的可见性，并在主函数中计算 `Li * f * max(N dot L, 0)`。
2. 屏幕空间求交：`RayMarch` 在世界空间沿射线步进，将采样点投影到屏幕并与 GBuffer 的线性深度比较。候选交点必须从可见表面前方跨越到后方，之后执行 6 次二分细化，并通过深度厚度和命中法线检查抑制轮廓误命中。
3. 间接光照：在当前着色点的法线半球上进行余弦加权采样，使用 `BSDF * cos / pdf` 的蒙特卡洛权重；SSR 命中后计算命中点的带阴影直接光照，得到一次间接漫反射（两次表面反射）的估计。

另外补充了起点偏移、屏幕边界/相机后方检查、可调采样数和步进次数，以及用于检查求交正确性的镜面反射显示模式。深度 Mipmap/Hi-Z Bonus 未实现。

## 运行

必须通过 HTTP 服务器运行，不能直接双击 `index.html`：

```text
cd homework3
python -m http.server 8000
```

打开 `http://127.0.0.1:8000/`。右上角面板可以选择：

- `scene`: `cube1`、`cube2`、`cave`
- `mode`: `combined`、`direct`、`indirect`、`reflection`
- `samples`: 每像素 1-4 条间接光照射线，默认 2
- `raySteps`: 每条射线最多 16-128 步，默认 96

也可以通过 URL 直接设置，例如：

```text
http://127.0.0.1:8000/?scene=cave&mode=combined&samples=4&steps=128
```

`samples=4&steps=128` 适合生成提交截图；交互查看时建议使用默认值。

## 结果截图

| 场景 | 直接光照 | 直接 + 间接光照 |
| --- | --- | --- |
| Cube 1 | [cube1-direct.png](images/cube1-direct.png) | [cube1-combined.png](images/cube1-combined.png) |
| Cube 2 | [cube2-direct.png](images/cube2-direct.png) | [cube2-combined.png](images/cube2-combined.png) |
| Cave | [cave-direct.png](images/cave-direct.png) | [cave-combined.png](images/cave-combined.png) |

额外调试截图：纯间接光照 [cube1-indirect.png](images/cube1-indirect.png)；镜面求交验证 [cube1-reflection.png](images/cube1-reflection.png)。

## 与主项目 SSR 的区别

| 对比项 | Homework 3 | 主项目 OpenGL SSR |
| --- | --- | --- |
| 目标 | 用屏幕空间求交估计 diffuse 两次反射，即短程屏幕空间全局光照 | 用屏幕空间命中替换/修正 PBR 的镜面环境反射 |
| 射线方向 | 围绕表面法线余弦采样 1-4 条随机半球射线 | 沿观察射线关于逐像素法线的镜面反射方向追踪一条射线 |
| 命中后的着色 | 在命中点重新计算方向光、阴影、Lambertian BSDF，再乘当前点 BSDF 和蒙特卡洛权重 | 直接查询已经完成光照的 HDR opaque color，并乘 split-sum specular response |
| GBuffer/坐标 | 世界坐标位置、世界坐标法线、漫反射率、阴影可见性和线性深度；在世界空间步进 | view normal、线性深度、F0/F90、roughness、occlusion；重建 view-space 位置并在 view space 步进 |
| 材质范围 | 方向光 + 纯漫反射材质 | metallic-roughness、specular-glossiness、normal/bump、IBL、AO/GTSO 等 PBR 数据 |
| 求交 | 自适应世界空间步长、前后深度跨越、6 次二分细化、厚度和命中法线检查 | 场景尺度控制的固定步长、可选 jitter、可调二分细化与厚度，并计算屏幕边缘置信度 |
| 粗糙反射 | 不涉及 | 用 GGX 反射锥计算命中 footprint；opaque color mip LOD 加最多 9 个各向异性采样实现距离相关模糊 |
| Mipmap 含义 | 未实现作业 Bonus 的 depth Mipmap/Hi-Z 加速 | 使用的是颜色 Mipmap 来过滤 glossy reflection，不是 depth Hi-Z 求交加速 |
| Miss/边缘 | 未命中时该条间接光为 0 | 按命中置信度、屏幕边缘和最大 roughness 混合，并回退到环境 IBL |
| 管线位置 | Shadow -> 五目标 GBuffer -> 对场景网格再次着色 | opaque/mask GBuffer 与 HDR opaque pass -> 全屏 SSR correction -> 透明合成 |

两者共同的限制是只能看到当前屏幕已有的表面：屏幕外、被遮挡或背面的几何都无法提供命中信息。
