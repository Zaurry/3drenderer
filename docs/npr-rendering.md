# OpenGL 非真实感渲染

在 ImGui 的 **Rendering → Style** 中选择 `Realistic`（写实）、`Toon`（卡通）或 `Sketch`（素描）。默认仍为写实，风格及参数随 Viewer 会话保存；CUDA Path 模式下此选项禁用。

实现依据 `learning/games202/lecture11-ltc-and-non-photorealistic-rendering.md` 的轮廓、光照量化和 Tonal Art Maps 方法。

- **卡通**：在线性光照阶段量化点光、方向光和聚光灯的 `NdotL`，镜面采用独立硬高光；矩形面光源按中心方向量化余弦，保留 LTC 积分的光源面积和距离尺度，避免较小的积分直接被量化为零。IBL 量化漫反射照度。`Diffuse bands` 可设置 2–6 档，色带边界使用屏幕导数进行窄范围抗锯齿。
- **素描**：初始化时生成六层确定性的 TAM。暗色调保留亮色调的已有笔触，再加入平行线和交叉排线；同一纹理集合生成共享面积过滤的完整 MIP 链。根据光照亮度选择相邻色调层并插值，硬件在 MIP 层间过滤。此实现是简化的程序化 TAM，远处笔触会按覆盖率合并成色调，不包含美术手工设计的粗层笔触。
- **轮廓**：复用 G-buffer 的线性深度和法线，以邻域差异识别外轮廓和内部硬边。`Outline width (px)` 按渲染分辨率计量，`Outline strength` 调整墨线强度；没有使用最终 RGB 的纹理边缘。

`Hatching scale` 控制排线频率，`Pencil darkness` 控制色调密度。默认使用按场景半径归一化的世界空间三平面投影，可用于无 UV 的模型，相机移动不会使笔触贴在屏幕上。移动或旋转物体时需要启用 `Use mesh UV for strokes`，让排线随表面运动；这要求模型有合适的 UV，接缝和不均匀拉伸仍会影响排线。

风格化期间保留直接光照、阴影、IBL 和 AO，暂停写实的 SSGI/SSR 合成，避免它们在排线和色块上叠加连续反射。切回写实后恢复原先开关。技术调试视图临时优先于风格化；重新选择风格会返回最终画面。

透明材质参与表面风格化和原有透明度合成，但不写入轮廓 G-buffer，因此不生成透明物体自身的深度/法线轮廓。素描背景使用统一纸色。曝光和色调映射仍由现有 Display 设置控制。

可复现运行和无 UI 截图：

```powershell
.\build\default\bin\viewer.exe --scene builtin --mode opengl --no-restore-last --style toon --frames 120 --capture output/npr/toon.png
.\build\default\bin\viewer.exe --scene builtin --mode opengl --no-restore-last --style sketch --frames 120 --capture output/npr/sketch.png
```

自动检查覆盖 TAM 各色调/MIP 的嵌套性、覆盖率、确定性，风格与调试视图的路由，以及会话参数的保存/恢复。

带 OpenGL 4.5 上下文的额外集成检查（独立于无需 GPU 的常规 CTest）：

```powershell
cmake --build --preset default-release --target opengl_npr_smoke
.\build\default\bin\opengl_npr_smoke.exe
```

此检查在隐藏窗口中读取实际 GPU 输出，验证同一实例中的写实/卡通/素描切换、AO 关闭时的轮廓、UV 模式、调整分辨率、调试视图，以及 SSR/SSGI 不污染风格化结果。
