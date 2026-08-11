#version 450 core

layout(binding = 0) uniform sampler2D u_base_color_texture;
layout(binding = 1) uniform sampler2D u_opacity_texture;
layout(binding = 2) uniform sampler2D u_normal_or_bump_texture;

uniform vec3 u_camera_position;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_base_color;
uniform float u_opacity;
uniform float u_alpha_cutoff;
uniform float u_bump_scale;
uniform float u_normal_scale;
uniform int u_alpha_mode;
uniform int u_two_sided;
uniform int u_has_base_color_texture;
uniform int u_has_opacity_texture;
uniform int u_has_normal_texture;
uniform int u_has_bump_texture;
uniform vec4 u_texture_offset_scale[9];
uniform float u_texture_rotation[9];
uniform int u_texture_texcoord[9];
uniform int u_texture_top_left[9];

in VS_OUT {
    vec3 world_position;
    vec3 normal;
    vec2 uv;
    vec2 uv1;
    vec4 tangent;
    vec4 color;
} fragment_in;

layout(location = 0) out vec3 out_view_normal;
layout(location = 1) out float out_linear_depth;

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec2 material_uv(int slot) {
    vec2 uv = u_texture_texcoord[slot] == 1 ? fragment_in.uv1 : fragment_in.uv;
    uv *= u_texture_offset_scale[slot].zw;
    float c = cos(u_texture_rotation[slot]);
    float s = sin(u_texture_rotation[slot]);
    uv = mat2(c, s, -s, c) * uv;
    uv += u_texture_offset_scale[slot].xy;
    if (u_texture_top_left[slot] != 0) {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}

vec3 surface_normal() {
    vec3 normal = normalize(fragment_in.normal);
    if (!gl_FrontFacing && u_two_sided != 0) {
        normal = -normal;
    }
    if (abs(fragment_in.tangent.w) < 0.5) {
        return normal;
    }
    vec3 tangent = fragment_in.tangent.xyz -
        normal * dot(normal, fragment_in.tangent.xyz);
    if (dot(tangent, tangent) <= 1.0e-12) {
        return normal;
    }
    tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(normal, tangent)) * fragment_in.tangent.w;
    if (u_has_normal_texture != 0) {
        vec3 mapped = texture(
            u_normal_or_bump_texture,
            material_uv(2)).xyz * 2.0 - 1.0;
        mapped.xy *= u_normal_scale;
        return normalize(
            tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
    }
    if (u_has_bump_texture != 0) {
        vec2 texel = 1.0 / vec2(textureSize(u_normal_or_bump_texture, 0));
        vec2 uv = material_uv(2);
        float left = luminance(texture(
            u_normal_or_bump_texture, uv - vec2(texel.x, 0.0)).rgb);
        float right = luminance(texture(
            u_normal_or_bump_texture, uv + vec2(texel.x, 0.0)).rgb);
        float down = luminance(texture(
            u_normal_or_bump_texture, uv - vec2(0.0, texel.y)).rgb);
        float up = luminance(texture(
            u_normal_or_bump_texture, uv + vec2(0.0, texel.y)).rgb);
        vec3 candidate = normal -
            (tangent * (right - left) + bitangent * (up - down)) *
                (0.5 * u_bump_scale);
        return dot(candidate, candidate) > 1.0e-12
            ? normalize(candidate)
            : normal;
    }
    return normal;
}

void main() {
    vec4 base_sample = u_has_base_color_texture != 0
        ? texture(u_base_color_texture, material_uv(0))
        : vec4(1.0);
    float opacity = clamp(
        u_opacity * base_sample.a * fragment_in.color.a,
        0.0,
        1.0);
    if (u_has_opacity_texture != 0) {
        opacity *= luminance(texture(
            u_opacity_texture,
            material_uv(1)).rgb);
    }
    if (u_alpha_mode == 1 && opacity < u_alpha_cutoff) {
        discard;
    }

    vec3 world_normal = surface_normal();
    out_view_normal = normalize(vec3(
        dot(world_normal, u_camera_right),
        dot(world_normal, u_camera_up),
        -dot(world_normal, u_camera_forward)));
    out_linear_depth = max(
        dot(fragment_in.world_position - u_camera_position, u_camera_forward),
        0.0);
}
