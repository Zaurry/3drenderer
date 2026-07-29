# OpenGL GLSL Shader Contract

The interactive viewer's `opengl` mode loads a vertex/fragment pair from disk and targets
OpenGL 4.5 Core (`#version 450 core`). The default files are:

- `shaders/opengl/raster.vert`
- `shaders/opengl/raster.frag`

The viewer checks both files every 250 ms. A successful compile/link atomically replaces the
active program. A failure leaves the last valid program active and exposes the complete driver
log in the GLSL Shader panel. `F5` or the panel button forces a reload even when timestamps did
not change.

## Vertex inputs

| Location | GLSL type | Name | Meaning |
| --- | --- | --- | --- |
| 0 | `vec3` | `a_position` | World-space position |
| 1 | `vec3` | `a_normal` | World-space shading normal |
| 2 | `vec2` | `a_uv` | OBJ-style texture coordinate |
| 3 | `vec4` | `a_tangent` | World-space tangent in `xyz`; `w` is `-1`/`+1` handedness or `0` when the UV basis is invalid |

Scene geometry is already expressed in world space, so there is no model matrix. The required
camera uniform is:

```glsl
uniform mat4 u_view_projection;
```

Custom shaders may omit inputs or uniforms they do not use; OpenGL is allowed to optimize them
away.

## Fragment resources

The renderer sets these per-frame uniforms:

```glsl
uniform vec3 u_camera_position;
uniform vec3 u_environment;
uniform int u_directional_light_count;
uniform int u_point_light_count;
```

Directional and point lights use two `std430` shader-storage buffers:

```glsl
struct DirectionalLight { vec4 direction; vec4 radiance; };
struct PointLight { vec4 position; vec4 intensity; };

layout(std430, binding = 0) readonly buffer DirectionalLightBuffer {
    DirectionalLight u_directional_lights[];
};
layout(std430, binding = 1) readonly buffer PointLightBuffer {
    PointLight u_point_lights[];
};
```

The renderer draws one batch per material and sets:

```glsl
uniform int u_material_type;       // -1 fallback, 0 diffuse, 1 metal, 2 dielectric, 3 emissive
uniform vec3 u_base_color;
uniform vec3 u_emission;
uniform float u_opacity;
uniform float u_alpha_cutoff;
uniform float u_bump_scale;
uniform int u_two_sided;
uniform int u_has_diffuse_texture;
uniform int u_has_opacity_texture;
uniform int u_has_bump_texture;
```

Texture units are fixed:

```glsl
layout(binding = 0) uniform sampler2D u_diffuse_texture;
layout(binding = 1) uniform sampler2D u_opacity_texture;
layout(binding = 2) uniform sampler2D u_bump_texture;
```

Images are decoded to linear floating-point values on the CPU before upload. Texture addressing
uses repeat wrapping and bilinear filtering, and upload orientation preserves the existing OBJ/CPU
UV convention.

## Output and display transform

Fragment location 0 must contain a linear HDR color:

```glsl
layout(location = 0) out vec4 out_linear_color;
```

Do not apply exposure, tone mapping, gamma, or sRGB encoding in the scene shader. The viewer's
internal compositor applies the shared Display controls after OpenGL or Path has produced linear
HDR output. CPU Path uses a host framebuffer; CUDA Path uses a direct GL texture when interop is
active and a host fallback otherwise.

## Launch examples

Use the default shaders:

```powershell
.\build\default\bin\viewer.exe --scene builtin --mode opengl
```

Use another pair:

```powershell
.\build\default\bin\viewer.exe `
  --scene asset `
  --asset path\to\scene.obj `
  --mode opengl `
  --gl-vertex-shader path\to\custom.vert `
  --gl-fragment-shader path\to\custom.frag
```

The scene shaders affect only `opengl` mode. Raster, Ray, and Path retain their C++/CUDA shading.
All four modes share only the internal display compositor.
