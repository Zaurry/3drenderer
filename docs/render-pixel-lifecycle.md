# 三种渲染方式中的一个像素如何诞生

本文用“一个像素”的视角，串起当前项目里三种渲染方式的执行路径：软件光栅化、Whitted 光线追踪、路径追踪。这里说的“最终渲染到屏幕”，在 v0.1 里具体指：渲染器先把线性颜色写入 `Image`，再通过 `Image::write_png()` 做 gamma 转换并保存成 PNG，最后由图片查看器显示在屏幕上。

## 共同入口

无论选择哪种模式，命令行都会先走同一段入口代码。

```text
src/main.cpp
  parse_args()
  make_scene_bundle()
  make_renderer()
  renderer_instance->render(scene, camera, settings)
  result.image.write_png(output_path)
```

关键模块：

- `src/main.cpp`: 解析 `--mode`、`--scene`、`--width`、`--height`、`--spp`、`--max-depth` 等参数。
- `src/scene/scene.cpp`: 生成内置场景，例如 `raster_triangle`、`mirror_spheres`、`cornell_box`。
- `src/scene/camera.cpp`: 创建相机，并负责从像素坐标生成相机射线。
- `src/render/renderer.h`: 统一的 `IRenderer` 接口。
- `src/core/image.cpp`: 保存每个像素的线性颜色，并写出 PNG。

最后一步对三种模式完全一样：

```text
linear Color
  -> Image::set_pixel(x, y, color)
  -> Image::write_png()
  -> to_rgb8()
  -> channel_to_rgb8(): clamp + gamma 1/2.2 + 0..255
  -> PNG 文件
  -> 图片查看器显示到屏幕
```

## 1. 软件光栅化：像素被三角形“覆盖”出来

入口：

```text
src/render/rasterizer/rasterizer_renderer.cpp
  RasterizerRenderer::render()
```

光栅化不是从一个像素主动发射光线，而是从三角形出发：先把三角形投到屏幕上，再判断这个像素是否落在三角形内部。

### 像素生命周期

1. 创建目标图像和深度缓冲。

   ```cpp
   Image image(settings.width, settings.height);
   std::vector<double> depth_buffer(..., infinity);
   ```

   此时像素还没有颜色，深度缓冲表示“当前像素还没有任何表面挡在前面”。

2. 遍历场景里的每个三角形。

   ```cpp
   for (const Triangle& triangle : scene.triangles)
   ```

   几何数据来自：

   - `src/scene/primitive.h`: `Triangle`
   - `src/scene/scene.cpp`: 内置三角形场景
   - `src/scene/obj_loader.cpp`: OBJ 模型会被转换成多个 `Triangle`

3. 三角形顶点从世界空间投影到屏幕空间。

   ```text
   project_to_screen()
     world vertex
     -> camera.eye/right/up/forward
     -> view_x/view_y/view_z
     -> NDC
     -> screen x/y
   ```

   相关模块：

   - `src/scene/camera.h`: 暴露 `eye()`、`right()`、`up()`、`forward()`、`viewport_width()`、`viewport_height()`
   - `src/core/math/vec3.h`: `dot()`、`normalize()` 等数学工具

4. 计算三角形屏幕包围盒。

   ```cpp
   min_x/max_x/min_y/max_y
   start_x/end_x/start_y/end_y
   ```

   这一步确定“哪些像素可能被这个三角形影响”。一个像素只有落在包围盒内，才会进入下一步判断。

5. 用 edge function 判断像素中心是否在三角形内。

   ```cpp
   const Vec3 p(x + 0.5, y + 0.5, 0.0);
   w0 = edge_function(v1.screen, v2.screen, p);
   w1 = edge_function(v2.screen, v0.screen, p);
   w2 = edge_function(v0.screen, v1.screen, p);
   ```

   如果 `w0/w1/w2` 的符号说明像素中心在三角形外，这个像素保持原样。

6. 通过重心坐标计算深度，并做 depth test。

   ```text
   w0/w1/w2
     -> barycentric b0/b1/b2
     -> inverse_depth
     -> depth
     -> compare with depth_buffer[y * width + x]
   ```

   如果当前三角形比旧内容更远，它不会改变这个像素。只有更近的表面才会覆盖旧颜色。

7. 透视正确插值世界坐标。

   ```cpp
   world_position =
       (v0.world * (b0 / v0.view_depth) +
        v1.world * (b1 / v1.view_depth) +
        v2.world * (b2 / v2.view_depth)) /
       inverse_depth;
   ```

   到这里，这个屏幕像素已经知道自己对应三角形上的哪个世界位置。

8. 着色。

   ```text
   shade_surface()
     material.base_color
     scene.environment
     DirectionalLight
     PointLight
     Lambert diffuse
     Blinn-Phong specular
   ```

   相关模块：

   - `src/scene/material.h`: `Material`
   - `src/scene/light.h`: `PointLight`、`DirectionalLight`
   - `src/core/color.h`: `Color`

9. 写入像素。

   ```cpp
   image.set_pixel(x, y, color);
   ```

### 光栅化中的像素变化总结

```text
空像素
  -> 进入三角形包围盒
  -> edge function 判断是否在三角形内
  -> 深度测试决定是否可见
  -> 根据材质和灯光计算颜色
  -> 写入 Image
  -> PNG gamma 转换
  -> 屏幕显示
```

光栅化的核心特点是：像素颜色由“哪个三角形覆盖了它”决定。

## 2. Whitted 光线追踪：像素主动发出一条相机射线

入口：

```text
src/render/raytracer/raytracer_renderer.cpp
  RayTracerRenderer::render()
  RayTracerRenderer::trace_ray()
```

光线追踪从像素出发。每个像素中心生成一条 primary ray，问场景：“沿这个方向最先看到什么？”

### 像素生命周期

1. 创建图像并构建 BVH。

   ```cpp
   Image image(settings.width, settings.height);
   Bvh bvh;
   bvh.build(scene.triangles);
   ```

   相关模块：

   - `src/acceleration/bvh.cpp`: BVH 构建和三角形遍历
   - `src/core/math/bounds.h`: AABB 相交测试

   BVH 只加速三角形；球体在当前版本里仍直接遍历。

2. 遍历每个像素。

   ```cpp
   for (int y = 0; y < settings.height; ++y)
   for (int x = 0; x < settings.width; ++x)
   ```

3. 像素中心转换成归一化屏幕坐标。

   ```cpp
   u = (x + 0.5) / width;
   v = 1.0 - (y + 0.5) / height;
   ```

   这里的 `0.5` 表示采样像素中心，而不是像素左上角。

4. 相机生成 primary ray。

   ```cpp
   const Ray ray = camera.generate_ray(u, v);
   ```

   相关模块：

   - `src/scene/camera.cpp`: `Camera::generate_ray()`
   - `src/core/math/ray.h`: `Ray`

5. `trace_ray()` 查找最近命中。

   ```text
   hit_scene()
     -> sphere.intersect()
     -> bvh.intersect()
     -> closest HitRecord
   ```

   相关模块：

   - `src/scene/primitive.h`: `Sphere::intersect()`、`Triangle::intersect()`
   - `src/acceleration/bvh.cpp`: `Bvh::intersect()`
   - `HitRecord`: 保存 `t`、`position`、`normal`、`material_id`、`front_face`

6. 如果没有命中，像素得到背景颜色。

   ```cpp
   return background_color(scene, settings);
   ```

7. 如果命中，读取材质并计算直接光照。

   ```text
   material = scene.materials[hit.material_id]
   result = material.emission
   result += ambient/background contribution
   ```

8. 对每盏点光源发 shadow ray。

   ```cpp
   Ray shadow_ray(hit.position, light_dir);
   hit_scene(shadow_ray, scene, bvh, 0.001, distance - 0.001, shadow_hit)
   ```

   如果 shadow ray 中途碰到物体，说明光被挡住，这盏灯不贡献颜色。`t_min = 0.001` 用来跳过刚刚命中的表面，减少自相交造成的阴影斑点。

9. 对每盏方向光也发 shadow ray。

   方向光没有距离衰减，只需要确认沿光源方向是否被遮挡。

10. 根据材质类型递归。

   金属材质：

   ```cpp
   reflected = reflect(normalize(ray.direction), hit.normal);
   reflected_color = trace_ray(Ray(hit.position, reflected), ..., depth - 1);
   ```

   介质材质：

   ```text
   eta_ratio
   refract()
   reflectance() using Schlick approximation
   reflected_color / refracted_color
   ```

   每次递归都会减少 `depth`，防止镜面之间无限反射。

11. 写入像素。

   ```cpp
   image.set_pixel(x, y, trace_ray(...));
   ```

### Whitted 光追中的像素变化总结

```text
像素坐标
  -> 像素中心 u/v
  -> primary ray
  -> 最近命中 HitRecord
  -> 材质 + 直接光
  -> shadow ray 判断遮挡
  -> reflection/refraction ray 递归补充颜色
  -> 写入 Image
  -> PNG gamma 转换
  -> 屏幕显示
```

Whitted 光追的核心特点是：像素颜色由“这条相机射线看到的表面，以及它递归反射/折射看到的内容”决定。

## 3. 路径追踪：一个像素由多条随机路径平均得到

入口：

```text
src/render/pathtracer/pathtracer_renderer.cpp
  PathTracerRenderer::render()
  PathTracerRenderer::trace_path()
  PathTracerRenderer::scatter()
```

路径追踪也从像素发射相机射线，但它不是只算一条确定路径，而是在像素内部随机采样多次。每个 sample 会随机反弹，最后把多次结果平均。

### 像素生命周期

1. 创建图像并构建 BVH。

   ```cpp
   Image image(settings.width, settings.height);
   Bvh bvh;
   bvh.build(scene.triangles);
   ```

2. 按 tile 切分图像，并启动 worker。

   ```text
   tile_size
   tiles_x / tiles_y
   atomic next_tile
   std::thread workers
   ```

   相关模块：

   - `src/render/render_settings.h`: `tile_size`、`thread_count`、`samples_per_pixel`
   - `std::atomic<int> next_tile`: 多线程领取下一个 tile

   每个 worker 负责一批像素，但不同 worker 不会写同一个像素。

3. 为当前像素创建确定性随机数。

   ```cpp
   PcgRandom rng(pixel_seed(x, y, settings.width));
   ```

   相关模块：

   - `src/core/random.h`: `PcgRandom`

   同一个像素总是从同一个 seed 开始，所以测试和图片对比更稳定。

4. 像素内部做多次 jitter sampling。

   ```cpp
   u = (x + rng.next_double()) / width;
   v = 1.0 - (y + rng.next_double()) / height;
   ```

   和 Whitted 光追的像素中心采样不同，路径追踪会在像素面积内部随机取点。这样可以降低锯齿，并为 Monte Carlo 积分提供样本。

5. 每个 sample 生成一条 primary ray。

   ```cpp
   camera.generate_ray(u, v)
   ```

6. `trace_path()` 查找最近命中。

   ```text
   hit_scene()
     -> spheres
     -> bvh triangles
   ```

   和 Whitted 光追一样，路径追踪也用 `Sphere::intersect()`、`Bvh::intersect()` 和 `HitRecord`。

7. 如果没有命中，sample 返回环境光。

   ```cpp
   return scene.environment;
   ```

8. 如果命中自发光材质，sample 返回 emission。

   ```cpp
   if (material.type == MaterialType::Emissive) {
       return emitted;
   }
   ```

   这就是路径最终“找到光源”的情况。

9. 如果不是光源，`scatter()` 根据材质随机决定下一跳。

   漫反射：

   ```text
   cosine_weighted_hemisphere(rng)
   tangent_to_world(local_direction, hit.normal)
   attenuation = material.base_color
   ```

   相关模块：

   - `src/sampling/sampler.cpp`: `cosine_weighted_hemisphere()`
   - `tangent_to_world()`: 把局部半球采样方向转成世界方向

   金属：

   ```text
   reflect(ray.direction, hit.normal)
   + roughness * random_in_unit_sphere(rng)
   ```

   介质：

   ```text
   refract()
   reflectance() using Schlick approximation
   rng.next_double() chooses reflection or refraction
   ```

10. 迭代估计渲染方程，并维护累计辐射与路径吞吐量。

   ```cpp
   radiance += throughput.cwiseProduct(emitted + direct);
   throughput = throughput.cwiseProduct(attenuation);
   ```

   这两行是当前路径追踪器的核心。可以读成：

   ```text
   当前点自己发出的光
   + 材质吞吐量 * 下一跳路径带回来的光
   ```

   前 3 次散射总是继续。之后根据累计 `throughput` 的最大 RGB 分量计算存活概率，并限制在 `[0.05, 0.95]`。路径未存活时结束；存活时将吞吐量除以存活概率，使 Monte Carlo 估计在期望上保持能量不变。内部 64 跳上限只用于防止极端路径运行过久。

   `--max-depth` 不再控制路径追踪，只保留给 Whitted 光线追踪使用。与原来的固定深度截断相比，Russian roulette 会让低贡献路径更早结束，也允许高贡献路径继续传播到原先深度之外；代价是单条路径的执行时间和贡献具有更高方差。

11. 累加 sample 颜色并平均。

   ```cpp
   accumulated += trace_path(...);
   image.set_pixel(x, y, accumulated / samples_per_pixel);
   ```

### 路径追踪中的像素变化总结

```text
像素坐标
  -> 多个随机 u/v sample
  -> 每个 sample 一条 primary ray
  -> 命中表面
  -> 按材质随机 scatter
  -> 多次递归反弹
  -> 命中光源或环境，带回 radiance
  -> sample 累加
  -> 平均成像素颜色
  -> 写入 Image
  -> PNG gamma 转换
  -> 屏幕显示
```

路径追踪的核心特点是：像素颜色由“许多随机光路的平均结果”决定。采样越多，噪声越低，但耗时越高。

## 三种方式对比

| 维度 | 软件光栅化 | Whitted 光追 | 路径追踪 |
| --- | --- | --- | --- |
| 像素如何开始 | 三角形投影后覆盖像素 | 像素中心发 primary ray | 像素内随机发多条 primary ray |
| 可见性判断 | edge function + depth buffer | 最近射线命中 | 最近射线命中 |
| 光照方式 | 显式计算灯光 | 直接光 + 阴影 + 递归反射/折射 | 随机路径迭代估计间接光 |
| 随机性 | 无 | 无 | 有，来自 `PcgRandom` |
| 主要加速结构 | 深度缓冲 | BVH 加速三角形 | BVH + tile 多线程 |
| 当前输出 | 线性颜色写入 `Image` | 线性颜色写入 `Image` | 多 sample 平均后写入 `Image` |
| 最终显示 | `write_png()` gamma 转换 | `write_png()` gamma 转换 | `write_png()` gamma 转换 |

## 从代码角度看同一个像素

以像素 `(x, y)` 为例，三条路径可以简化成下面这样：

```text
raster:
  main.cpp
  -> RasterizerRenderer::render()
  -> project_to_screen(triangle vertices)
  -> edge_function(pixel center)
  -> depth_buffer test
  -> shade_surface()
  -> Image::set_pixel(x, y)
  -> Image::write_png()

ray:
  main.cpp
  -> RayTracerRenderer::render()
  -> Camera::generate_ray(pixel center)
  -> trace_ray()
  -> hit_scene()
  -> direct light + shadow ray
  -> reflection/refraction recursion
  -> Image::set_pixel(x, y)
  -> Image::write_png()

path:
  main.cpp
  -> PathTracerRenderer::render()
  -> tile worker
  -> PcgRandom(pixel_seed)
  -> Camera::generate_ray(jittered sample)
  -> trace_path()
  -> scatter()
  -> recursive path contribution
  -> average samples
  -> Image::set_pixel(x, y)
  -> Image::write_png()
```

## 共同的最终颜色转换

所有渲染器都输出线性空间的 `Color`，也就是 `Vec3(r, g, b)`。写 PNG 时走：

```text
src/core/image.cpp
  Image::write_png()
    pixel_rgb8()
      to_rgb8()
        channel_to_rgb8()
```

`channel_to_rgb8()` 会做三件事：

1. 处理非有限数值，例如 NaN 或 infinity。
2. 把颜色 clamp 到 `[0, 1]`。
3. 应用 `pow(channel, 1.0 / 2.2)` 做 gamma 转换，再变成 `0..255` 的字节。

所以屏幕上看到的不是渲染器内部的原始线性值，而是适合普通显示器观看的 PNG 像素值。

