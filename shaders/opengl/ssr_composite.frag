#version 450 core

layout(binding = 0) uniform sampler2D u_opaque;
layout(binding = 1) uniform sampler2D u_direct_lighting;
layout(binding = 2) uniform sampler2D u_raw_indirect;
layout(binding = 3) uniform sampler2D u_temporal_indirect;
layout(binding = 4) uniform sampler2D u_filtered_indirect;
layout(binding = 5) uniform sampler2D u_linear_depth;
layout(binding = 6) uniform sampler2D u_view_normal;
layout(binding = 7) uniform sampler2D u_ssr_material;
uniform ivec2 u_full_resolution;
uniform int u_max_history_frames;
uniform int u_debug_view;
in vec2 v_uv;
layout(location = 0) out vec4 out_linear_color;

void main() {
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), u_full_resolution - 1);
    vec3 indirect = max(texelFetch(u_filtered_indirect, pixel, 0).rgb, vec3(0.0));
    if (u_debug_view != 0) {
        vec3 color = indirect;
        if (u_debug_view == 1) color = texelFetch(u_raw_indirect, pixel, 0).rgb;
        if (u_debug_view == 2) color = vec3(texelFetch(u_raw_indirect, pixel, 0).a);
        if (u_debug_view == 3) color = texelFetch(u_temporal_indirect, pixel, 0).rgb;
        if (u_debug_view == 5) color = vec3(clamp(
            texelFetch(u_temporal_indirect, pixel, 0).a / float(max(u_max_history_frames, 1)), 0.0, 1.0));
        out_linear_color = vec4(max(color, vec3(0.0)), 1.0);
        return;
    }
    float depth = texelFetch(u_linear_depth, pixel, 0).r;
    vec3 normal = texelFetch(u_view_normal, pixel, 0).xyz;
    // Sky and pixels without a receiver retain the opaque background.
    vec3 color = depth > 0.0 && dot(normal, normal) > 1.0e-8
        ? texelFetch(u_direct_lighting, pixel, 0).rgb + indirect
        : texelFetch(u_opaque, pixel, 0).rgb;
    out_linear_color = vec4(max(color, vec3(0.0)), 1.0);
}
