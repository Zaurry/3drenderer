# GAMES202 Lecture 7：Real-time Global Illumination in 3D

![Lecture 7 标题](images/lecture07-01-title.png)

这一讲先为上一讲的 PRT 收尾：把只适合漫反射的标量传输扩展到有光泽材质，并介绍另一类基函数 Wavelet。随后课程正式进入实时全局光照，用 Reflective Shadow Map（RSM）近似一次间接光照。

> 本讲的核心问题：如何把数量巨大的间接光路径，压缩成实时算法能够处理的数据和查询？

## 1. 本讲路线

![Lecture 7 内容提纲](images/lecture07-04-outline.png)

本讲可以分成三部分：

1. **Glossy PRT**：漫反射只有一个输出值，有光泽表面却会随观察方向变化，因此传输量从向量升级为矩阵。
2. **Wavelet PRT**：球谐函数擅长低频，Wavelet 能表达局部、尖锐的高频信号。
3. **RSM**：把从光源看到的每个直接受光表面小块，当作一个虚拟点光源，累加它们对接收点的贡献。

## 2. 回顾：PRT 到底预计算了什么

![PRT 回顾](images/lecture07-05-prt-recap.png)

渲染方程中的直接光部分可以写为：

$$
L_o(\mathbf{x},\omega_o)
=
\int_{\Omega^+}
L_i(\mathbf{x},\omega_i)
f_r(\mathbf{x},\omega_i,\omega_o)
V(\mathbf{x},\omega_i)
(\mathbf{n}\cdot\omega_i)
\,\mathrm{d}\omega_i
$$

PRT 将不随运行时光照变化的部分合并为传输函数：

$$
T(\mathbf{x},\omega_i,\omega_o)
=
f_r(\mathbf{x},\omega_i,\omega_o)
V(\mathbf{x},\omega_i)
(\mathbf{n}\cdot\omega_i)
$$

离线阶段计算并压缩 `T`，运行时只需把新的环境光照与它组合。遮挡、软阴影，甚至预计算范围内的多次反弹，都可以隐藏在传输数据中。

## 3. 漫反射 PRT 为什么只是点积

![漫反射 PRT 展开](images/lecture07-07-diffuse-prt.png)

对漫反射材质，输出不依赖观察方向。将光照和传输都投影到一组正交基函数 `B_i` 上：

$$
L(\omega)
\approx
\sum_{p=1}^{K} l_p B_p(\omega)
$$

$$
T(\omega)
\approx
\sum_{q=1}^{K} t_q B_q(\omega)
$$

代入积分：

$$
L_o
\approx
\sum_{p=1}^{K}\sum_{q=1}^{K}
l_p t_q
\int_{\Omega}B_p(\omega)B_q(\omega)\,\mathrm{d}\omega
$$

![球谐正交性把双重求和化为点积](images/lecture07-08-sh-orthogonality.png)

球谐基是正交归一的：

$$
\int_{\Omega}B_p(\omega)B_q(\omega)\,\mathrm{d}\omega
=
\delta_{pq}
$$

当 `p` 与 `q` 不同时，积分为零；只有编号相同的系数会留下：

$$
L_o
\approx
\sum_{p=1}^{K}l_p t_p
=
\mathbf{l}^{\mathsf T}\mathbf{t}
$$

因此表面每个顶点只需保存一个 `K` 维传输向量，着色成本为 `O(K)`。

## 4. Glossy PRT：传输向量为何变成矩阵

漫反射从任意方向看都一样，所以一个点只输出一个辐亮度。有光泽材质不同：观察方向移动时，高光也会移动。传输函数必须同时描述入射方向 `omega_i` 和出射方向 `omega_o`。

![Glossy PRT 的传输矩阵](images/lecture07-09-glossy-matrix.png)

先对入射方向展开，再让每个系数成为出射方向的函数：

$$
T(\omega_i,\omega_o)
\approx
\sum_{i=1}^{K}\sum_{j=1}^{K}
t_{ij}B_i(\omega_i)B_j(\omega_o)
$$

把光照展开后积分，可得：

$$
L_o(\omega_o)
\approx
\sum_{j=1}^{K}
\left(
\sum_{i=1}^{K}l_i t_{ij}
\right)
B_j(\omega_o)
$$

矩阵形式为：

$$
\mathbf{o}^{\mathsf T}
=
\mathbf{l}^{\mathsf T}\mathbf{M}
$$

`M` 的行对应入射基函数，列对应出射基函数。先用环境光系数乘传输矩阵，得到出射光的球谐系数 `o`；再在当前观察方向重建最终亮度。

![漫反射与 Glossy PRT 的复杂度](images/lecture07-10-complexity.png)

- 漫反射：每个点保存 `K` 个系数，运行时做一次长度为 `K` 的点积。
- 有光泽：每个点保存 `K x K` 个系数，运行时需要矩阵乘法，成本为 `O(K^2)`。

![Glossy PRT 效果](images/lecture07-11-glossy-result.png)

即使运行时只做线性代数，预计算的传输仍然可以包含复杂遮挡和反射路径。因此画面可能看起来像做了昂贵的光线追踪，实际运行成本却基本不随路径复杂度变化。

![复杂传输路径已被烘焙进系数](images/lecture07-12-transport-complexity.png)

## 5. 如何预计算传输矩阵

![逐个点亮基函数来预计算](images/lecture07-13-basis-precompute.png)

一种直观方法是逐个使用基函数作为环境光：

1. 用第 `i` 个基函数 `B_i` 照亮场景。
2. 离线求出每个表面点随观察方向变化的响应。
3. 再把这个方向响应投影到出射方向的基函数上。
4. 所得系数就是矩阵第 `i` 行的元素。

对所有入射基函数重复后，就得到完整的传输矩阵。任意运行时光照都是这些基光照的线性组合，因此可直接组合预计算结果。

![空间变化 BRDF 也可纳入预计算](images/lecture07-14-arbitrary-brdf.png)

如果材质固定，空间变化的 BRDF 也可一起烘焙。代价是交互自由度下降：几何、材质和复杂传输通常不能任意改变。

## 6. PRT 的能力边界

![PRT 局限性](images/lecture07-17-limitations.png)

经典 PRT 的主要限制是：

- 场景几何通常必须静态，否则预计算的可见性和反弹路径会失效。
- 传输对光照是线性的，难以表示非线性效果。
- 低阶球谐只能保留低频信号，尖锐阴影和高频高光会被抹平。
- Glossy PRT 的数据量从 `K` 增长到 `K^2`，顶点和纹理存储压力明显增加。

这解释了 PRT 的定位：它很适合静态场景配合动态环境光，但不是通用的动态全局光照方案。

## 7. Wavelet：另一种光照基函数

![二维 Haar Wavelet](images/lecture07-20-haar-wavelet.png)

球谐函数具有全局支撑：修改一个系数会影响整个球面。Wavelet 则具有局部支撑，可以把信号拆成不同位置、不同尺度上的变化。

以 Haar Wavelet 为例：

- 最低频项记录整体平均值。
- 其余项记录相邻区域之间的差异。
- 越高层的项描述越小尺度、越局部的变化。

![Wavelet 变换](images/lecture07-21-wavelet-transform.png)

现实光照往往只有少数区域存在强烈变化。变换后，许多 Wavelet 系数很小，可以直接丢弃，只保留绝对值最大的少量项。

![仅保留少量非零 Wavelet 项](images/lecture07-22-sparse-wavelet.png)

这种非线性近似可能只保留原系数的 `0.1%` 到 `1%`，仍能保存尖锐光源和高频阴影。但由于每次光照变化都要重新选择重要系数，它不像固定阶数 SH 那样简单稳定。

![SH 与 Wavelet 的频率特征](images/lecture07-23-sh-vs-wavelet.png)

| 特性 | 球谐函数 SH | Wavelet |
| --- | --- | --- |
| 擅长信号 | 平滑、低频 | 局部、高频也可表达 |
| 支撑范围 | 全局 | 局部 |
| 截断方式 | 保留固定低阶 | 按系数大小稀疏保留 |
| 旋转 | 有成熟快速方法 | 通常更麻烦 |
| 典型伪影 | 高频被模糊 | 稀疏截断可能出现块状或振铃 |

## 8. 从 PRT 转向实时全局光照

全局光照的困难不在于写出方程，而在于积分中每个入射方向又对应另一个表面点，那个点的出射光还可能来自更多反弹。路径数量会迅速膨胀。

![本讲目标：一次间接光](images/lecture07-28-one-bounce-target.png)

实时算法通常先做一个重要取舍：只计算**一次间接反弹**。也就是光源先照到表面 `q`，再由 `q` 反射到当前着色点 `p`。

## 9. 关键观察：受光表面就是次级光源

![直接受光表面可看作次级光源](images/lecture07-29-secondary-lights.png)

从点 `p` 看，所有被光源直接照亮的表面小块都在向外反射能量。将它们离散化后，每个小块都可以近似成一个 Virtual Point Light（VPL，虚拟点光源）。

于是一次间接光照变成：

1. 找到所有直接受光的表面小块。
2. 计算每个小块携带的能量和方向。
3. 累加这些 VPL 对当前点 `p` 的贡献。

![一次间接光照结果](images/lecture07-31-one-bounce-result.png)

![一次间接光的关键观察](images/lecture07-32-key-observations.png)

问题是如何低成本地获得这些直接受光的小块。答案就在阴影贴图的生成过程里。

## 10. Reflective Shadow Map

![RSM 将阴影贴图像素视为 VPL](images/lecture07-33-rsm-vpl.png)

普通 Shadow Map 从光源视角渲染，只保存最近深度。RSM 在同一次渲染中额外保存该表面片元的信息：

- 世界空间位置 `x_q`
- 世界空间法线 `n_q`
- 反射通量 `Phi_q`
- 深度，用于正常的阴影判断

每个 RSM 像素对应光源能直接看到并照亮的一小块表面，因此天然就是 VPL 候选。

## 11. 一个 RSM 像素贡献多少间接光

设 VPL 位于 `q`，接收点为 `p`，位移向量为：

$$
\mathbf{d}=\mathbf{p}-\mathbf{q}
$$

![立体角与面积之间的转换](images/lecture07-35-solid-angle-area.png)

面积与立体角之间满足投影和距离平方衰减。把发射端朝向、接收端朝向以及距离衰减合在一起，可得课程中的近似形式：

![RSM 的间接光贡献公式](images/lecture07-36-rsm-form-factor.png)

$$
E_p(q)
\approx
\Phi_q
\frac{
\max(0,\mathbf{n}_q\cdot\mathbf{d})
\max(0,\mathbf{n}_p\cdot(-\mathbf{d}))
}{
\lVert\mathbf{d}\rVert^4
}
$$

为什么分母是四次方？因为上式的两个点积使用了**未归一化**位移 `d`，每个点积都额外带有一个距离因子。若改用归一化方向：

$$
\widehat{\mathbf{d}}
=
\frac{\mathbf{d}}{\lVert\mathbf{d}\rVert}
$$

则等价地写成更熟悉的形式：

$$
E_p(q)
\approx
\Phi_q
\frac{
\max(0,\mathbf{n}_q\cdot\widehat{\mathbf{d}})
\max(0,\mathbf{n}_p\cdot(-\widehat{\mathbf{d}}))
}{
\lVert\mathbf{d}\rVert^2
}
$$

两个余弦项分别表示：VPL 是否朝向接收点，以及接收面是否朝向 VPL。

## 12. 哪些 VPL 才有贡献

![RSM 贡献需要满足的条件](images/lecture07-37-contribution-tests.png)

一个候选 VPL 通常要经过以下检查：

- `q` 必须被原始光源直接照亮，这由 RSM 本身保证。
- VPL 必须朝向接收点。
- 接收点必须朝向 VPL。
- 距离不能太远，否则贡献很小。
- 理论上还应检查 `q` 到 `p` 是否被遮挡。

最后一项最昂贵。基础 RSM 通常省略 VPL 到接收点的可见性测试，因此间接光可能穿墙。这是它用速度换准确性的关键近似。

## 13. 不能遍历整张 RSM

![从 RSM 中抽样而不是逐像素累加](images/lecture07-38-rsm-sampling.png)

若 RSM 有上百万像素，而屏幕也有上百万像素，逐点遍历会形成无法接受的二次复杂度。实际做法是在当前点投影到 RSM 的位置附近采样少量 VPL：

- 近处密集、远处稀疏地取样。
- 用随机旋转打散固定图案。
- 根据采样概率补偿权重。
- 限制搜索半径，只估计局部一次反弹。

这使算法足够快，但也带来噪声和漏光。

## 14. RSM 的缓冲数据与伪代码

![RSM 需要保存的 G-buffer](images/lecture07-39-rsm-gbuffer.png)

RSM 本质上是从光源视角生成的一组 G-buffer。一个简化实现如下：

```glsl
vec3 evaluateRSMIndirect(
    vec3 receiverPos,
    vec3 receiverNormal,
    vec2 receiverLightUV)
{
    vec3 indirect = vec3(0.0);

    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        vec2 uv = receiverLightUV + sampleOffset(i);
        vec3 vplPos = texture(rsmPosition, uv).xyz;
        vec3 vplNormal = normalize(texture(rsmNormal, uv).xyz);
        vec3 vplFlux = texture(rsmFlux, uv).rgb;

        vec3 d = receiverPos - vplPos;
        float r2 = max(dot(d, d), 1e-4);
        vec3 wi = d * inversesqrt(r2);

        float cosFromVPL = max(dot(vplNormal, wi), 0.0);
        float cosAtReceiver = max(dot(receiverNormal, -wi), 0.0);
        float geometry = cosFromVPL * cosAtReceiver / r2;

        indirect += vplFlux * geometry * sampleWeight(i);
    }

    return indirect;
}
```

真正实现还需处理 BRDF、坐标空间、RSM texel 面积、概率密度、颜色空间和数值稳定性。

## 15. RSM 的效果与局限

![RSM 在游戏中的效果](images/lecture07-40-rsm-game.png)

![RSM 优缺点](images/lecture07-41-rsm-pros-cons.png)

**优点：**

- 完全动态，可随光源和物体移动更新。
- 复用 Shadow Map 渲染流程，概念和工程实现都较直接。
- 能产生明显的颜色渗透，例如红墙把红色间接光反射到附近地面。
- 适合估计局部、一次反弹的漫反射间接光。

**缺点：**

- 假设 VPL 对外近似漫反射，不擅长有光泽的间接反射。
- 忽略 VPL 到接收点的可见性时会穿墙漏光。
- 采样数量有限，会产生噪声、闪烁和不稳定。
- 距离很近时几何项可能异常大，需要偏移或钳制。
- RSM 只包含光源能直接看到的表面，无法自然表达多次反弹。

## 16. 三种方法之间的关系

| 方法 | 压缩或近似对象 | 适合场景 | 主要代价 |
| --- | --- | --- | --- |
| SH PRT | 预计算光传输 | 静态几何、动态低频环境光 | 预计算与系数存储 |
| Wavelet PRT | 稀疏的高频光照与传输 | 静态场景、高频信号 | 稀疏匹配与运行时更新复杂 |
| RSM | 直接受光表面形成的 VPL 集合 | 动态场景、一次漫反射间接光 | 大量 VPL 抽样与漏光 |

它们的共同思想不是“精确解积分”，而是寻找适合当前信号结构的低维表示。

## 17. 常见误解

### 误解一：Glossy PRT 只是给漫反射结果加高光

不是。观察方向成为传输函数的一个独立维度，因此输出不再是一个数，而是一组随观察方向重建的系数。

### 误解二：SH 系数越多就一定越适合

高阶 SH 可以表达更高频率，但系数和传输矩阵会快速增大。实时系统要在带宽、存储、精度之间取平衡。

### 误解三：RSM 就是保存更多内容的 Shadow Map

数据结构上确实相似，但用途不同。Shadow Map 回答“这个点能否看见光源”，RSM 还要回答“光源照到的表面能向哪里反射多少能量”。

### 误解四：一次间接光只需做距离衰减

还必须考虑发射端和接收端的朝向。缺少任意一个余弦项，都可能让背面错误地发光或受光。

## 18. 调试清单

实现 RSM 时建议逐层检查：

1. 单独显示 RSM 深度、世界位置、法线和通量。
2. 确认所有向量处于同一坐标空间。
3. 关闭间接光，只验证直接光与 Shadow Map。
4. 固定一个 VPL，检查两个余弦项和距离衰减。
5. 增加样本数，观察噪声是否按预期下降。
6. 检查墙角和薄墙背面，定位可见性缺失导致的漏光。
7. 对很小的距离设置下限，避免亮点和数值爆炸。

## 19. 本讲小结

Lecture 7 完成了两次重要过渡：

- 从漫反射 PRT 的向量点积，过渡到 Glossy PRT 的传输矩阵。
- 从静态预计算光传输，过渡到动态场景中的一次间接光估计。

最值得记住的结论是：

1. 球谐正交性让漫反射 PRT 从双重求和降为点积。
2. 有光泽反射依赖观察方向，因此传输量升级为矩阵。
3. Wavelet 通过局部、多尺度基函数保留高频细节。
4. RSM 把直接受光的表面小块视为 VPL，并累加它们的一次反弹贡献。
5. RSM 的速度来自抽样和省略可见性，噪声与漏光也由此产生。

![下一讲将介绍 LPV 与 VXGI](images/lecture07-43-next-lecture.png)

下一讲会继续处理三维动态全局光照：不再为每个接收点遍历 VPL，而是把辐射信息注入三维网格或体素层次结构，再做传播和查询。

