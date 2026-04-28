# CPU 3D Renderer

这是一个从零实现的 C++20 CPU 软件 3D 渲染器，用来学习图形管线、光线求交、BVH、Whitted 光线追踪和基础路径追踪。当前 v0.1 不使用 OpenGL、Vulkan、DirectX、Embree、OptiX 或 CUDA。

## 构建

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\renderer_tests.exe
```

如果本机没有 CMake，可以安装后再运行上面的命令。当前工程使用 CMake + C++20，Windows 下会生成 Visual Studio/MSBuild 项目。

## 渲染模式

- `raster`: 手写 CPU 光栅化器，包含世界到屏幕投影、edge function 重心坐标、深度缓冲、Lambert 和 Blinn-Phong 光照。
- `ray`: Whitted 风格光线追踪器，支持球和三角形求交、BVH、硬阴影、镜面反射、介质折射。
- `path`: 基础 Monte Carlo 路径追踪器，支持 tile 多线程、像素内抖动采样、漫反射半球采样、金属粗糙反射、介质反射/折射和自发光材质。

## 示例命令

```powershell
.\build\bin\renderer.exe --mode raster --scene raster_triangle --width 512 --height 512 --output output\raster_triangle.png
.\build\bin\renderer.exe --mode ray --scene mirror_spheres --width 512 --height 512 --output output\mirror_spheres.png
.\build\bin\renderer.exe --mode path --scene cornell_box --width 512 --height 512 --spp 32 --max-depth 5 --output output\cornell_box.png
```

OBJ 查看器示例：

```powershell
.\build\bin\renderer.exe --mode raster --scene obj_viewer --obj path\to\model.obj --width 512 --height 512 --output output\obj_viewer.png
```

## CLI 参数

```text
--mode raster|ray|path
--scene gradient_sphere|raster_triangle|mirror_spheres|cornell_box|obj_viewer
--obj path\to\model.obj
--width integer
--height integer
--spp integer
--max-depth integer
--threads integer
--output path\to\file.png
--help
```

## 学习顺序

1. `src/core/math`: 向量、矩阵、射线、AABB。
2. `src/scene`: 材质、光源、球、三角形、相机和内置场景。
3. `src/acceleration/bvh.*`: BVH 构建与遍历。
4. `src/render/rasterizer`: 软件光栅化管线。
5. `src/render/raytracer`: Whitted 光线追踪。
6. `src/render/pathtracer`: Monte Carlo 路径追踪。
7. `src/main.cpp`: 统一 CLI 入口，展示未来统一渲染接口的方向。

## 依赖说明

核心算法都是项目内手写：数学、求交、BVH、光栅化、光追、路径追踪和采样。第三方单头文件只用于边界功能：

- `stb_image_write.h`: 写 PNG。
- `tiny_obj_loader.h`: 读取 OBJ 文本并三角化面。

## 当前限制

- 光栅化 v0.1 只处理示例几何在近裁剪面前方的情况，还没有完整三角形裁剪。
- 路径追踪没有 next event estimation、MIS、Russian roulette、降噪或 PBR 微表面模型。
- OBJ 查看器使用一个默认材质和简单自动取景，还没有材质贴图解析。
- 目前输出离线 PNG，没有实时窗口。
