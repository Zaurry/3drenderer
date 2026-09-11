# OpenGL 统一 SSR 验证记录

日期：2026-09-05。SSR 现在表示 Screen Space Ray Tracing，完整合成约定见 [GLSL shader 合约](../glsl-shader-contract.md)。

## 修正内容

- 合并原 SSR/SSGI 的追踪、Hi-Z、重投影、历史和去噪，移除顺序叠加及不同的反弹来源。
- 全分辨率混合余弦半球/GGX VNDF 采样，以完整 BRDF/PDF 权重求解漫反射、镜面和微表面多次散射补偿。
- SSR 启用时 opaque raster 跳过 IBL，最终只合成直接光/发光 + SSR 间接光。屏幕射线命中后不再保留该方向的环境项，AO 只作用于环境回退。
- 遮挡与屏幕边缘辐射可靠度分开。单面背面不发光但遮挡环境；双面材质可被两侧射线采样。
- 解析深度薄层交点支持朝向/远离相机的射线，保留奇数尺寸与 NPOT Hi-Z 归约。
- LTC 已表示的矩形发光表面从射线辐射源排除，防止与直接光重复计光。普通发光网格仍可照亮其他表面。
- Viewer 设置与会话 v6 只有一组 SSR，兼容旧会话公共参数；自定义 shader 缺少直接光/辐射源 MRT 时自动旁路。

## 验证

Windows、Release：

```powershell
cmake --build --preset no-cuda-release
ctest --preset no-cuda-release --output-on-failure
cmake --build --preset default-release --target viewer
```

no-CUDA 构建的全部 9 个 CTest 目标通过。该构建中的 CUDA Path 用例按既有规则跳过，不能据此声称验证了 CUDA 追踪行为；默认配置的 Viewer（含 CUDA）另外编译成功。

新增 `opengl_transport_tests` 使用隐藏 SDL OpenGL 4.5 context，实际编译 shader、绑定 framebuffer 并回读像素。5 项测试覆盖：

1. 可见遮挡物移除环境；材质/屏幕 AO 为零时命中光照仍存在；IBL 关闭不删除命中贡献；朝向相机的射线、单面背面和双面材质。
2. 未命中的环境回退、AO 可见度、IBL 关闭、diffuse/GGX 白炉积分，以及斜面掠射角自相交检查。
3. 合成中剔除原环境照明、精确保留直接光与天空背景。
4. LTC 发光不重复计入 SSR；移除解析灯后，对应网格发光仍得到非零间接贡献。
5. 完整 renderer 的奇数尺寸、1×1 resize、SSR 开关、IBL 开关、历史累积和输出有限性。

GGX 白炉：单位 Fresnel、N·V=0.8、128 帧 × 8 samples，期望反射能量 1。GPU 回读均值：

| Roughness | 均值 |
|---:|---:|
| 0.02 | 1.000000 |
| 0.30 | 1.002200 |
| 0.65 | 1.004570 |
| 1.00 | 0.989885 |

这些数值验证采样 PDF、下半球零贡献和能量补偿的组合，没有把着色器源码字符串匹配当作物理正确性的证明。

## 视觉检查

可选导出命令：

```powershell
$env:RENDERER_OPENGL_CAPTURE_DIR = "$PWD/output/opengl-ssr-validation"
./build/no-cuda/bin/opengl_transport_tests.exe --filter complete_pipeline
```

已检查 384×384、默认 2 rays/pixel、64 帧的 Cornell Box 图像：

- `output/opengl-ssr-validation/cornell-ibl-fallback.png`
- `output/opengl-ssr-validation/cornell-unified-ssr.png`
- `output/opengl-ssr-validation/cornell-indirect.png`

原始检查图保留在本地 output 目录，不作为可移植测试基线。全分辨率路径增加了 diffuse trace/history/filter 的像素数；本轮没有完整重跑大场景性能基准，不用小场景单帧计时推断高分辨率性能。

## 仍有的近似

屏外、隐藏几何、透明物体和真实厚度不能由单层屏幕深度恢复。未命中环境回退可能继续产生漏光，边缘淡出和去噪也有偏差。当前辐射源只含本帧直接光/发光，没有环境照亮命中表面后的再次反弹或多次表面反弹；命中表面的方向性使用相机观测值近似。因此这是对屏幕空间能量与可见性合成的修正，不能等同于完整物理路径追踪。
