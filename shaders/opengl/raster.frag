#version 450 core

struct DirectionalLight {
    vec4 direction;
    vec4 radiance;
};

struct PointLight {
    vec4 position;
    vec4 intensity;
};

layout(std430, binding = 0) readonly buffer DirectionalLightBuffer {
    DirectionalLight u_directional_lights[];
};

layout(std430, binding = 1) readonly buffer PointLightBuffer {
    PointLight u_point_lights[];
};

layout(binding = 0) uniform sampler2D u_diffuse_texture;
layout(binding = 1) uniform sampler2D u_opacity_texture;
layout(binding = 2) uniform sampler2D u_bump_texture;

uniform vec3 u_camera_position;
uniform vec3 u_environment;
uniform int u_material_type;
uniform vec3 u_base_color;
uniform vec3 u_emission;
uniform float u_opacity;
uniform float u_alpha_cutoff;
uniform float u_bump_scale;
uniform int u_two_sided;
uniform int u_has_diffuse_texture;
uniform int u_has_opacity_texture;
uniform int u_has_bump_texture;
uniform int u_directional_light_count;
uniform int u_point_light_count;

in VS_OUT {
    vec3 world_position;
    vec3 normal;
    vec2 uv;
    vec4 tangent;
} fragment_in;

layout(location = 0) out vec4 out_linear_color;

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 surface_normal() {
    vec3 normal = normalize(fragment_in.normal);
    if (!gl_FrontFacing && u_two_sided != 0) {
        normal = -normal;
    }
    if (u_has_bump_texture == 0 || abs(fragment_in.tangent.w) < 0.5) {
        return normal;
    }

    vec3 tangent = fragment_in.tangent.xyz;
    tangent = tangent - normal * dot(normal, tangent);
    if (dot(tangent, tangent) <= 1.0e-12) {
        return normal;
    }
    tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(normal, tangent)) * fragment_in.tangent.w;
    vec2 texel = 1.0 / vec2(textureSize(u_bump_texture, 0));
    float left = luminance(texture(u_bump_texture, fragment_in.uv - vec2(texel.x, 0.0)).rgb);
    float right = luminance(texture(u_bump_texture, fragment_in.uv + vec2(texel.x, 0.0)).rgb);
    float down = luminance(texture(u_bump_texture, fragment_in.uv - vec2(0.0, texel.y)).rgb);
    float up = luminance(texture(u_bump_texture, fragment_in.uv + vec2(0.0, texel.y)).rgb);
    vec3 gradient = tangent * ((right - left) * 0.5) +
                    bitangent * ((up - down) * 0.5);
    vec3 candidate = normal - gradient * u_bump_scale;
    return dot(candidate, candidate) > 1.0e-12 ? normalize(candidate) : normal;
}

vec3 shade_light(vec3 base_color, vec3 normal, vec3 view_direction, vec3 light_direction, vec3 radiance) {
    float n_dot_l = max(dot(normal, light_direction), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3(0.0);
    }
    vec3 half_direction = light_direction + view_direction;
    float specular = dot(half_direction, half_direction) > 1.0e-12
        ? pow(max(dot(normal, normalize(half_direction)), 0.0), 32.0) * 0.2
        : 0.0;
    return base_color * radiance * n_dot_l + radiance * specular;
}

void main() {
    vec3 base_color = u_base_color;
    if (u_has_diffuse_texture != 0) {
        base_color *= texture(u_diffuse_texture, fragment_in.uv).rgb;
    }

    float opacity = clamp(u_opacity, 0.0, 1.0);
    if (u_has_opacity_texture != 0) {
        opacity *= luminance(texture(u_opacity_texture, fragment_in.uv).rgb);
    }
    if (opacity < u_alpha_cutoff) {
        discard;
    }

    if (u_material_type == 3) {
        out_linear_color = vec4(u_emission, 1.0);
        return;
    }

    vec3 normal = surface_normal();
    vec3 view_direction = normalize(u_camera_position - fragment_in.world_position);
    vec3 color = base_color * u_environment * 0.15;
    for (int index = 0; index < u_directional_light_count; ++index) {
        vec3 direction = u_directional_lights[index].direction.xyz;
        if (dot(direction, direction) <= 1.0e-12) {
            continue;
        }
        color += shade_light(
            base_color,
            normal,
            view_direction,
            normalize(-direction),
            u_directional_lights[index].radiance.xyz);
    }
    for (int index = 0; index < u_point_light_count; ++index) {
        vec3 to_light = u_point_lights[index].position.xyz - fragment_in.world_position;
        float distance_squared = max(dot(to_light, to_light), 1.0e-12);
        color += shade_light(
            base_color,
            normal,
            view_direction,
            to_light / sqrt(distance_squared),
            u_point_lights[index].intensity.xyz / distance_squared);
    }
    out_linear_color = vec4(color, 1.0);
}
