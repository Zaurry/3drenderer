#version 450 core
layout(binding = 0) uniform sampler2D u_opaque;
layout(binding = 1) uniform sampler2D u_accum;
layout(binding = 2) uniform sampler2D u_reveal;
layout(binding = 3) uniform sampler2D u_ao_bent_normal;
layout(binding = 4) uniform sampler2D u_view_normal;
layout(binding = 5) uniform sampler2D u_linear_depth;
uniform int u_ao_debug_view;
uniform int u_skip_transparency;
uniform float u_scene_radius;
uniform int u_npr_style;
uniform float u_outline_width;
uniform float u_outline_strength;
in vec2 v_ndc;
layout(location = 0) out vec4 out_linear_color;

float npr_outline(vec2 uv) {
    if (u_outline_width <= 0.0 || u_outline_strength <= 0.0) return 0.0;
    float depth = texture(u_linear_depth, uv).r;
    vec3 normal = texture(u_view_normal, uv).xyz;
    vec2 texel = 1.0 / vec2(textureSize(u_linear_depth, 0));
    float edge = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            vec2 q = clamp(uv + vec2(x, y) * texel * u_outline_width,
                texel * 0.5, vec2(1.0) - texel * 0.5);
            float q_depth = texture(u_linear_depth, q).r;
            if ((depth > 0.0) != (q_depth > 0.0)) {
                edge = 1.0;
            } else if (depth > 0.0 && q_depth > 0.0) {
                vec3 q_normal = texture(u_view_normal, q).xyz;
                float normal_edge = 1.0 - clamp(dot(normal, q_normal), -1.0, 1.0);
                float depth_edge = abs(q_depth - depth) / max(min(depth, q_depth), 1.0e-5);
                edge = max(edge, max(smoothstep(0.015, 0.06, depth_edge),
                    smoothstep(0.12, 0.40, normal_edge)));
            }
        }
    }
    return edge * u_outline_strength;
}

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
    if (u_npr_style != 0) {
        if (u_npr_style == 2 && texture(u_linear_depth, uv).r <= 0.0)
            opaque = vec3(0.92, 0.89, 0.82);
        opaque = mix(opaque, vec3(0.025, 0.021, 0.018), npr_outline(uv));
    }
    if (u_skip_transparency != 0) {
        out_linear_color = vec4(opaque, 1.0);
        return;
    }
    vec4 accum = texture(u_accum, uv);
    float reveal = clamp(texture(u_reveal, uv).r, 0.0, 1.0);
    vec3 transparent = accum.rgb / max(accum.a, 1.0e-5);
    out_linear_color = vec4(transparent * (1.0 - reveal) + opaque * reveal, 1.0);
}
