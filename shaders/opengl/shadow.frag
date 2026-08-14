#version 450 core
layout(binding = 0) uniform sampler2D u_base_color_texture;
layout(binding = 1) uniform sampler2D u_opacity_texture;
uniform int u_alpha_mode;
uniform float u_alpha_cutoff;
uniform float u_opacity;
uniform int u_has_base_color_texture;
uniform int u_has_opacity_texture;
uniform vec4 u_texture_offset_scale[2];
uniform float u_texture_rotation[2];
uniform int u_texture_texcoord[2];
uniform int u_texture_top_left[2];
uniform int u_cube_pass;
uniform vec3 u_shadow_origin;
uniform vec3 u_shadow_direction;
uniform float u_shadow_near;
uniform float u_shadow_far;
in vec3 v_world_position;
in vec2 v_uv;
in vec2 v_uv1;
in float v_vertex_alpha;
layout(location = 0) out float out_linear_depth;
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}
vec2 material_uv(int slot) {
    vec2 uv = u_texture_texcoord[slot] == 1 ? v_uv1 : v_uv;
    uv *= u_texture_offset_scale[slot].zw;
    float cosine = cos(u_texture_rotation[slot]);
    float sine = sin(u_texture_rotation[slot]);
    uv = mat2(cosine, sine, -sine, cosine) * uv;
    uv += u_texture_offset_scale[slot].xy;
    if (u_texture_top_left[slot] != 0) {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}
void main() {
    if (u_alpha_mode == 1) {
        float opacity = u_opacity * v_vertex_alpha;
        if (u_has_base_color_texture != 0) {
            opacity *= texture(u_base_color_texture, material_uv(0)).a;
        }
        if (u_has_opacity_texture != 0) {
            opacity *= luminance(texture(u_opacity_texture, material_uv(1)).rgb);
        }
        if (opacity < u_alpha_cutoff) {
            discard;
        }
    }
    float linear_distance = u_cube_pass != 0
        ? length(v_world_position - u_shadow_origin)
        : dot(v_world_position - u_shadow_origin, u_shadow_direction);
    out_linear_depth = clamp(
        (linear_distance - u_shadow_near) /
        max(u_shadow_far - u_shadow_near, 1.0e-6),
        0.0,
        1.0);
}
