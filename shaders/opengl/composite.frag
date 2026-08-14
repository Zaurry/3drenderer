#version 450 core
layout(binding = 0) uniform sampler2D u_opaque;
layout(binding = 1) uniform sampler2D u_accum;
layout(binding = 2) uniform sampler2D u_reveal;
layout(binding = 3) uniform sampler2D u_ao_bent_normal;
layout(binding = 4) uniform sampler2D u_view_normal;
layout(binding = 5) uniform sampler2D u_linear_depth;
uniform int u_ao_debug_view;
uniform float u_scene_radius;
in vec2 v_ndc;
layout(location = 0) out vec4 out_linear_color;
void main() {
    vec2 uv = v_ndc * 0.5 + 0.5;
    if (u_ao_debug_view != 0) {
        float depth = texture(u_linear_depth, uv).r;
        vec3 view_normal = texture(u_view_normal, uv).xyz;
        vec4 ao = texture(u_ao_bent_normal, uv);
        vec3 debug_color = vec3(0.0);
        if (u_ao_debug_view == 1) {
            debug_color = vec3(depth > 0.0 ? ao.a : 1.0);
        } else if (u_ao_debug_view == 2) {
            debug_color = depth > 0.0
                ? normalize(ao.xyz) * 0.5 + 0.5
                : vec3(0.0);
        } else if (u_ao_debug_view == 3) {
            debug_color = depth > 0.0
                ? normalize(view_normal) * 0.5 + 0.5
                : vec3(0.0);
        } else {
            debug_color = vec3(clamp(
                depth / max(u_scene_radius * 4.0, 1.0e-5),
                0.0,
                1.0));
        }
        out_linear_color = vec4(debug_color, 1.0);
        return;
    }
    vec3 opaque = texture(u_opaque, uv).rgb;
    vec4 accum = texture(u_accum, uv);
    float reveal = clamp(texture(u_reveal, uv).r, 0.0, 1.0);
    vec3 transparent = accum.rgb / max(accum.a, 1.0e-5);
    out_linear_color = vec4(transparent * (1.0 - reveal) + opaque * reveal, 1.0);
}
