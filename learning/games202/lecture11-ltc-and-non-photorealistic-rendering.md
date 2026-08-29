# GAMES202 Lecture 11：LTC and Non-Photorealistic Rendering

![Lecture 11 标题](images/lecture11-01-title.png)

Lecture 11 前半完成 Linearly Transformed Cosines（LTC）：用线性变换把复杂微表面 BRDF 波瓣变成可解析积分的余弦波瓣，从而实时计算多边形面光源。后半进入 Non-Photorealistic Rendering（NPR），讨论轮廓线、色块量化和笔触纹理。

> 本讲看似包含两个不同主题，实际共享同一种思路：先找出视觉结果中最重要的结构，再用适合实时计算的表示重建它。

## 1. 本讲路线

![Lecture 11 内容提纲](images/lecture11-04-outline.png)

1. 完成 LTC 的变换与积分思路。
2. 简要回顾 Disney Principled BRDF 的设计理念。
3. 理解 NPR 不等于“随便加卡通滤镜”。
4. 学习三类实时风格化方法：轮廓、色块、笔触。

# 第一部分：Linearly Transformed Cosines

## 2. LTC 要解决的问题

![LTC 解决多边形光源下的微表面着色](images/lecture11-06-ltc-problem.png)

点光源只有一个入射方向，直接评估一次 BRDF 即可。多边形面光源覆盖一片方向区域 `P`，需要积分：

$$
L_o(\omega_o)
=
\int_P
L_i(\omega_i)
f_r(\omega_i,\omega_o)
(\mathbf{n}\cdot\omega_i)
\,\mathrm{d}\omega_i
$$

对 GGX 等微表面 BRDF，波瓣会随观察方向、粗糙度发生偏斜和拉伸。逐像素数值采样面光源既慢又有噪声。

LTC 的适用条件与边界：

- 主要用于无阴影的多边形面光源着色。
- 常拟合 GGX，也可以拟合其他二维 BRDF 波瓣。
- 它计算的是局部材质积分，不负责光源到表面的遮挡。

## 3. 把复杂波瓣变成余弦波瓣

![LTC 的核心变换](images/lecture11-07-ltc-key-idea.png)

余弦分布很简单：

$$
D_0(\omega')
=
\frac{\max(0,\omega'_z)}{\pi}
$$

LTC 假设复杂 BRDF 波瓣可由一个归一化线性变换 `M` 作用在余弦波瓣上近似得到。`M` 可以旋转、缩放、错切，把圆对称余弦波瓣变成倾斜、拉长的高光形状。

关键不是直接对复杂波瓣积分，而是反过来用 `M^{-1}` 把问题送回余弦空间。

## 4. 波瓣、方向和积分域一起变换

![BRDF、方向和多边形积分域同时变换](images/lecture11-08-ltc-transform.png)

设原空间中的入射方向为 `omega_i`，余弦空间方向为 `omega_i'`：

$$
\omega_i
=
\frac{
M\omega_i'
}{
\lVert M\omega_i'\rVert
}
$$

反向变换为：

$$
\omega_i'
=
\frac{
M^{-1}\omega_i
}{
\lVert M^{-1}\omega_i\rVert
}
$$

同一个逆变换还要作用到面光源多边形 `P` 的每条顶点方向，得到新积分域 `P'`。

这是 LTC 最容易漏掉的一点：只变换 BRDF 不够，积分域必须随变量替换一起变化。

## 5. 变量替换与 Jacobian

![LTC 的变量替换](images/lecture11-09-ltc-change-variable.png)

若面光源辐亮度近似常量 `L_i`，把 BRDF 与余弦合并为方向波瓣 `F`：

$$
L(\omega_o)
=
L_i
\int_P F(\omega_i)\,\mathrm{d}\omega_i
$$

通过变换后：

$$
L(\omega_o)
\approx
L_i
\int_{P'}
D_0(\omega_i')
J(\omega_i')
\,\mathrm{d}\omega_i'
$$

对归一化线性映射，立体角 Jacobian 与矩阵行列式和方向长度有关：

$$
J(\omega')
=
\frac{
|\det M|
}{
\lVert M\omega'\rVert^3
}
$$

实际 LTC 分布的定义会把对应 Jacobian 因子纳入归一化，使变换前后总能量一致。课程希望强调的是：困难 BRDF 被拟合为变换余弦后，对变换多边形的积分可以解析计算。

## 6. 为什么余弦在多边形上可以解析积分

多边形顶点先转换为从着色点出发的单位方向，再被 `M^{-1}` 变换和重新归一化。这些方向在单位球面上形成球面多边形。

余弦积分可以改写成球面多边形边界的边积分。运行时只需遍历多边形边，而不必在面积内部大量采样：

1. 将光源顶点转换到着色点局部坐标。
2. 乘 `M^{-1}` 并归一化。
3. 将多边形裁剪到上半球。
4. 对相邻顶点组成的球面边累加解析项。
5. 乘光源辐亮度和拟合幅度。

因此成本主要与多边形边数有关，而且无蒙特卡洛噪声。

## 7. 变换矩阵从哪里来

`M` 依赖两个主要参数：

- 观察方向与法线夹角，通常用 `NdotV` 表示。
- 材质粗糙度。

离线阶段对每组参数拟合一个线性变换，使 LTC 波瓣尽量接近目标 GGX 波瓣。运行时从二维 LUT 读取：

- 逆矩阵的压缩参数。
- 波瓣幅度或归一化系数。

无需每像素重新拟合。

```glsl
vec3 evaluateLTC(
    vec3 position,
    vec3 normal,
    vec3 viewDir,
    float roughness,
    vec3 lightVertices[4],
    vec3 lightRadiance)
{
    float NdotV = max(dot(normal, viewDir), 1e-4);
    vec2 lutUV = vec2(roughness, sqrt(1.0 - NdotV));

    mat3 Minv = decodeInverseMatrix(texture(ltcMatrixLUT, lutUV));
    float amplitude = texture(ltcAmplitudeLUT, lutUV).r;
    mat3 shadingFrame = buildFrame(normal, viewDir);

    vec3 polygon[4];
    for (int i = 0; i < 4; ++i) {
        vec3 local = transpose(shadingFrame) * (lightVertices[i] - position);
        polygon[i] = normalize(Minv * local);
    }

    float integral = integrateSphericalPolygonCosine(polygon);
    return lightRadiance * amplitude * max(integral, 0.0);
}
```

## 8. LTC 效果与限制

![LTC 的多边形面光源结果](images/lecture11-10-ltc-result.png)

**优点：**

- 多边形面光源下的镜面高光稳定、无采样噪声。
- 支持粗糙度和观察角变化。
- 光源形状会真实影响高光形状。
- 运行时只需 LUT 查询、矩阵变换和边积分。

**限制：**

- 波瓣是拟合结果，极端粗糙度或掠射角可能有误差。
- 基础 LTC 不计算阴影。
- 复杂非凸光源需拆分或谨慎裁剪。
- 多层、各向异性或多峰 BRDF 可能需要多个 LTC 波瓣。
- 坐标系、矩阵方向和球面裁剪实现较容易出错。

# 第二部分：Disney Principled BRDF 回顾

## 9. 从物理参数到美术参数

![Disney Principled BRDF 的动机](images/lecture11-13-disney-motivation.png)

课程再次强调 Disney 模型的需求：真实世界材质种类太多，而折射率等物理参数并不总适合制作流程。实时渲染更需要一套可统一编辑、可纹理化、插值稳定的参数系统。

![Principled 模型的设计规则](images/lecture11-14-principled-rules.png)

“Principled” 的重点是参数设计原则：直观、少量、范围统一、组合稳健，并允许美术指导。它是 physics-inspired，不等同于严格的 measured BRDF。

## 10. 参数不是独立滤镜

![Disney 参数对外观的影响](images/lecture11-15-parameter-effects.png)

每个参数控制一个或多个波瓣，但最终材质是这些波瓣按能量规则组合后的结果。例如：

- `metallic` 改变 base color 进入漫反射还是镜面 Fresnel。
- `roughness` 同时影响波瓣宽度、峰值和环境光预过滤层级。
- `clearcoat` 是额外介电质层，不应只理解为第二个白色高光。
- `sheen` 主要增强布料在掠射角的柔和反射。

![Disney 模型的优缺点](images/lecture11-16-disney-pros-cons.png)

统一模型提升资产制作效率，也会带来庞大参数空间。引擎应提供合理默认值、材质模板和范围约束，而不是把所有滑块直接交给使用者。

# 第三部分：Non-Photorealistic Rendering

## 11. NPR 就是风格化

![NPR 等于风格化](images/lecture11-18-npr-stylization.png)

Non-Photorealistic Rendering 的目标不是尽可能接近照片，而是主动产生艺术表达。课程进一步强调实时 NPR：

![实时 NPR 追求快速可靠的风格化](images/lecture11-19-realtime-stylization.png)

> 实时 NPR 是快速、可靠的风格化。

“可靠”意味着视角变化、动画和光照变化时，线条、色块和笔触不能随机跳动。

## 12. 照片真实与非照片真实

![照片真实渲染目标](images/lecture11-20-photorealistic.png)

照片真实渲染关注光照、阴影和材质是否与真实成像一致。

![NPR 的目标是艺术外观](images/lecture11-21-npr-goal.png)

NPR 关注哪些信息应该被保留、舍弃或强化。它通常不是从零开始，而是建立在可靠的几何、材质和光照基础上。

## 13. NPR 的三个特征

![NPR 的主要特征](images/lecture11-22-npr-characteristics.png)

1. **从照片真实模型出发**：仍需法线、深度、光照和材质结构。
2. **抽象**：减少连续色调、细节和纹理，只保留视觉语义。
3. **强化重点**：用轮廓、对比和笔触突出角色、结构或动作。

NPR 不是“计算得更少”，而是“把计算预算放到风格需要的信息上”。

## 14. NPR 的应用

![NPR 在动画和游戏中的应用](images/lecture11-24-npr-applications.png)

应用包括艺术创作、科学可视化、说明图、教育、动画与游戏。不同应用关心的重点不同：技术插图要求结构清楚，游戏角色则更关心造型识别和时间稳定性。

## 15. 如何拆解一种风格

![观察风格化角色](images/lecture11-25-style-example.png)

面对参考图，不要只说“做成动漫风”。应把风格拆成可渲染的结构：

![风格由轮廓、色块和表面笔触组成](images/lecture11-26-style-components.png)

- 粗细、颜色和位置不同的轮廓线
- 离散而非连续的明暗色块
- 表面上的排线、素描或材质笔触
- 不同角色部件采用不同规则

拆解之后，艺术需求才能转化为 shader、几何 pass 和后处理。

# 第四部分：Outline Rendering

## 16. 轮廓不只有 silhouette

![轮廓线的四种类型](images/lecture11-27-outline-types.png)

课程把线条分为四类：

- **Boundary / Border Edge**：开放网格边界，一条边只有一个相邻面。
- **Crease**：相邻面的夹角或法线差很大。
- **Material Edge**：边两侧材质、颜色或语义不同。
- **Silhouette**：一个相邻面朝向相机，另一个背向相机。

只做 silhouette 无法复现内部结构线和材质分界。

## 17. 方法一：着色法线轮廓

![通过法线与视线垂直检测轮廓](images/lecture11-28-shading-outline.png)

当法线接近垂直于视线时：

$$
|\mathbf{n}\cdot\mathbf{v}|
<
\varepsilon
$$

可将该表面区域压暗。这种方法简单，能产生柔和轮廓带，但线宽受曲率、距离和网格法线影响，不是稳定的屏幕像素宽度。

```glsl
float contour = 1.0 - smoothstep(
    contourStart,
    contourEnd,
    abs(dot(normalize(normalWS), normalize(viewDirWS))));

color = mix(color, outlineColor, contour);
```

## 18. 方法二：几何背面膨胀

![背面膨胀生成稳定外轮廓](images/lecture11-29-geometry-outline.png)

常见流程：

1. 正常渲染正面。
2. 将网格顶点沿法线向外膨胀。
3. 只渲染膨胀网格的背面，使用纯色。
4. 背面在原物体外露出的区域形成轮廓。

```glsl
// Outline vertex pass
vec4 positionCS = projection * view * model * vec4(positionOS, 1.0);
vec3 normalVS = normalize(normalMatrix * normalOS);

// A production version should convert the desired pixel width more carefully.
positionCS.xy += normalVS.xy * outlineWidth * positionCS.w;
gl_Position = positionCS;
```

优点是外轮廓连贯，缺点是尖角、硬边、凹模型和不同距离下线宽需要特殊处理。

## 19. 方法三：图像空间边缘检测

![Sobel 图像边缘检测](images/lecture11-30-sobel-outline.png)

Sobel 核估计水平与垂直梯度：

$$
G_x
=
\begin{bmatrix}
1 & 0 & -1 \\
2 & 0 & -2 \\
1 & 0 & -1
\end{bmatrix}
$$

$$
G_y
=
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

边缘强度近似为：

$$
g
=
\sqrt{g_x^2+g_y^2}
$$

## 20. 不要只在最终颜色上找边

![在深度、法线和材质等缓冲上检测边缘](images/lecture11-31-gbuffer-edges.png)

颜色边缘会把纹理细节也识别成轮廓。更稳定的方法是组合多种 G-buffer 差异：

$$
e
=
w_z e_{\mathrm{depth}}
+
w_n e_{\mathrm{normal}}
+
w_m e_{\mathrm{material}}
+
w_c e_{\mathrm{color}}
$$

```glsl
float edge = 0.0;

for (int i = 0; i < 8; ++i) {
    vec2 qUV = uv + neighborOffset[i] * texelSize;
    float qDepth = texture(depthBuffer, qUV).r;
    vec3 qNormal = normalize(texture(normalBuffer, qUV).xyz);
    float qMaterial = texture(materialIdBuffer, qUV).r;

    edge += depthWeight * abs(centerDepth - qDepth);
    edge += normalWeight * (1.0 - dot(centerNormal, qNormal));
    edge += materialWeight * step(materialThreshold,
                                  abs(centerMaterial - qMaterial));
}

float outline = smoothstep(edgeLow, edgeHigh, edge);
```

图像空间方法容易控制屏幕像素线宽，也能捕获材质边界，但会受到分辨率、抗锯齿、透明物体和遮挡顺序影响。

## 21. 三种轮廓方法对比

| 方法 | 优点 | 缺点 |
| --- | --- | --- |
| 法线着色 | 单 pass、柔和、实现简单 | 线宽不稳定，只能看到接近轮廓的区域 |
| 背面膨胀 | 外轮廓连贯，可独立着色 | 几何问题多，内部线条难处理 |
| 图像边缘 | 固定像素宽度，可组合多种信息 | 后处理伪影、遮挡和透明度复杂 |

实际 NPR 常混合三者：几何方法画主轮廓，G-buffer 方法补内部结构线，美术 mask 控制局部开关。

# 第五部分：Color Blocks

## 22. Hard Shading 与 Posterization

![通过着色阈值或最终颜色阈值生成色块](images/lecture11-32-color-blocks.png)

两种常见方式：

- **Hard shading**：在计算光照时量化 `NdotL`，改变光照模型本身。
- **Posterization**：对最终图像颜色量化，是后处理效果。

前者能保持材质和光照语义，后者容易实现但可能破坏高光、色调映射和不同物体的层次。

## 23. 多级量化

![将连续明暗量化为多个等级](images/lecture11-33-quantization.png)

将 0 到 1 的连续值量化为 `N` 个等级：

$$
Q(x)
=
\frac{
\operatorname{round}(x(N-1))
}{
N-1
}
$$

```glsl
float quantize(float x, float levels) {
    return round(clamp(x, 0.0, 1.0) * (levels - 1.0))
         / max(levels - 1.0, 1.0);
}

float diffuseBand = quantize(max(dot(n, l), 0.0), diffuseLevels);
vec3 toonDiffuse = baseColor * diffuseBand;
```

为了让明暗边界稳定，可用阈值表、平滑过渡、蓝噪声抖动或时域稳定策略。

## 24. 不同光照分量采用不同风格

![漫反射与镜面分量可使用不同风格](images/lecture11-34-component-styles.png)

漫反射、镜面、阴影和边缘不必共享同一种量化：

- 漫反射可使用 2 至 4 个宽色带。
- 镜面可使用单个硬高光或手绘形状。
- 阴影颜色可偏冷或偏暖，而不是简单乘黑。
- 轮廓颜色可随材质变化。

这比对最终 RGB 统一 posterize 更能表达设计意图。

# 第六部分：Strokes Surface Stylization

## 25. 用笔触替代逐点明暗

![用预生成笔触纹理表现表面明暗](images/lecture11-35-stroke-stylization.png)

素描风格不希望出现平滑色块，而希望暗部由更密集的排线组成。核心问题有两个：

- **Density**：亮度如何映射到笔触密度？
- **Continuity**：相机、物体和 MIP 层变化时，笔触如何保持连续？

直接每像素随机画线会严重闪烁，因此通常预生成结构一致的纹理集合。

## 26. Tonal Art Maps

![不同密度与不同 MIP 层级的 Tonal Art Maps](images/lecture11-36-tonal-art-maps.png)

Tonal Art Map（TAM）是二维纹理表：

- 一个维度是色调，越暗的层级笔触越密。
- 另一个维度是 MIP 层级，适应屏幕投影尺寸。

为了保持连续性，TAM 通常满足嵌套关系：

- 更暗色调保留较亮色调已有笔触，再增加新笔触。
- 更粗 MIP 层保留可识别的主要笔触，而不是独立随机生成。

这样色调或距离变化时，线条逐渐增加、合并，而不会整套替换。

## 27. TAM 渲染流程

![TAM 从预处理到实时渲染的流程](images/lecture11-37-tam-pipeline.png)

1. 离线生成多个密度等级及其 MIPMAP。
2. 为模型建立连续、方向合适的纹理参数化。
3. 运行时计算表面色调。
4. 在相邻两个 tone 层间插值。
5. 让硬件根据屏幕导数选择 MIP 层。

```glsl
float tone = clamp(1.0 - luminance(lighting), 0.0, 1.0);
float layer = tone * float(TONE_COUNT - 1);
int lo = int(floor(layer));
int hi = min(lo + 1, TONE_COUNT - 1);
float blend = fract(layer);

float strokeLo = texture(tamLayers[lo], strokeUV).r;
float strokeHi = texture(tamLayers[hi], strokeUV).r;
float strokes = mix(strokeLo, strokeHi, blend);

vec3 result = mix(paperColor, inkColor, strokes);
```

模型 UV 若在关节或接缝处不连续，笔触也会断裂。角色动画还需考虑纹理随表面运动，避免 screen-space 笔触像贴在镜头上。

# 第七部分：NPR 的工程判断

## 28. NPR 是艺术驱动的系统

![NPR 需要把艺术需求翻译成渲染规则](images/lecture11-38-art-driven.png)

“线条更有力量”不是可以直接写进 shader 的参数。渲染工程师需要与美术一起把需求拆成：

- 哪类边缘要出现
- 线宽是否随距离变化
- 哪些部件使用不同色阶
- 光照变化时高光如何移动
- 笔触属于世界、模型还是屏幕空间

同一角色甚至不同部位都可能需要独立规则。统一算法只是起点，美术 mask 和可控参数同样重要。

## 29. 照片真实模型对 NPR 仍然重要

![真实材质模型仍是 NPR 的基础](images/lecture11-39-physical-models.png)

风格化布料仍需要正确褶皱法线、各向异性和遮挡，之后才能选择性简化和强化。若基础法线、光照与材质响应错误，NPR 只是掩盖问题，难以得到稳定、可指导的结果。

这也是本讲把 LTC、Disney 和 NPR 放在一起的深层原因：艺术风格并不排斥物理模型，而是建立在可预测的物理信号之上。

## 30. 常见误解

### 误解一：LTC 是把光源变成点光源

不是。LTC 保留多边形光源的方向域，只把 BRDF 波瓣和光源顶点共同变换到余弦空间。

### 误解二：LTC 自带面光源阴影

基础 LTC 只解局部 BRDF 积分，不知道光源与着色点之间是否有遮挡。阴影要由其他方法提供。

### 误解三：NPR 是照片真实结果加描边

描边只是一个组件。完整风格还涉及色块、材质、高光、阴影颜色、笔触以及时间一致性。

### 误解四：Sobel 能找到所有语义轮廓

Sobel 只能找到输入信号的梯度。颜色相同但材质不同的边界，需要 material ID；平滑阴影产生的颜色梯度也可能被误识别。

### 误解五：减少色阶就是 cel shading

机械量化最终颜色通常不够。成熟 cel shading 会分别设计漫反射、镜面、阴影和轮廓，并提供角色级控制。

## 31. 调试清单

### LTC

1. 固定观察方向，逐步改变 roughness，检查矩阵 LUT 连续性。
2. 显示变换前后的多边形顶点方向。
3. 检查 `M` 与 `M^{-1}` 是否用反。
4. 在跨越地平线的光源上测试球面裁剪。
5. 用数值蒙特卡洛积分作为离线参考。
6. 分离 diffuse LTC 与 specular LTC。

### NPR

1. 分别显示深度边、法线边、材质边和颜色边。
2. 在不同分辨率检查轮廓像素宽度。
3. 用旋转相机检查线条和色块是否闪烁。
4. 检查背面膨胀在尖角、硬法线和凹模型上的破面。
5. 在线性空间量化光照，色调映射放在合理阶段。
6. 检查 TAM 的相邻 tone 与 MIP 是否保持嵌套。
7. 用动画角色检查 UV 接缝和笔触游动。
8. 为不同角色部件提供 mask，避免所有表面共享一个阈值。

## 32. 本讲小结

1. LTC 用线性变换将复杂 BRDF 波瓣拟合为余弦波瓣，并同步变换多边形光源积分域。
2. 变换矩阵由粗糙度和观察角查表获得，变换后的球面多边形可做解析边积分。
3. Principled BRDF 的价值在于直观、连续和稳健的统一材质工作流。
4. NPR 从可靠的物理信号出发，通过抽象和强化形成风格。
5. 轮廓可来自着色法线、膨胀几何或 G-buffer 边缘，各有不同适用范围。
6. 色块应优先量化有语义的光照分量，而不是盲目量化最终颜色。
7. TAM 通过色调和 MIP 两个维度的嵌套笔触保持密度与时间连续性。

![下一讲进入散射材质模型](images/lecture11-40-next-lecture.png)

下一讲将继续实时物理材质，但从表面 BRDF 转向云、皮肤、头发等散射模型。
