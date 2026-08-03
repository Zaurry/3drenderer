#version 450 core

const float PI = 3.14159265358979323846;

struct DirectionalLight { vec4 direction; vec4 radiance; };
struct PointLight { vec4 position_range; vec4 intensity; };
struct SpotLight { vec4 position_range; vec4 direction_inner; vec4 intensity_outer; };

layout(std430, binding = 0) readonly buffer DirectionalLightBuffer {
    DirectionalLight u_directional_lights[];
};
layout(std430, binding = 1) readonly buffer PointLightBuffer {
    PointLight u_point_lights[];
};
layout(std430, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight u_spot_lights[];
};

layout(binding = 0) uniform sampler2D u_base_color_texture;
layout(binding = 1) uniform sampler2D u_opacity_texture;
layout(binding = 2) uniform sampler2D u_normal_or_bump_texture;
layout(binding = 3) uniform sampler2D u_metallic_roughness_texture;
layout(binding = 4) uniform sampler2D u_occlusion_texture;
layout(binding = 5) uniform sampler2D u_emissive_texture;
layout(binding = 6) uniform sampler2D u_specular_texture;
layout(binding = 7) uniform sampler2D u_specular_color_texture;
layout(binding = 8) uniform sampler2D u_specular_glossiness_texture;
layout(binding = 9) uniform samplerCube u_environment_prefilter;
layout(binding = 10) uniform sampler2D u_environment_brdf_lut;

uniform vec3 u_camera_position;
uniform vec3 u_environment_color;
uniform vec3 u_environment_sh[9];
uniform float u_environment_intensity;
uniform float u_environment_rotation_radians;
uniform float u_environment_mip_count;
uniform int u_has_environment_map;
uniform int u_material_type;
uniform int u_pbr_workflow;
uniform vec3 u_base_color;
uniform vec3 u_emission;
uniform float u_ior;
uniform vec3 u_specular_color;
uniform float u_specular_factor;
uniform float u_glossiness;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_opacity;
uniform float u_alpha_cutoff;
uniform float u_bump_scale;
uniform float u_normal_scale;
uniform float u_occlusion_strength;
uniform int u_alpha_mode;
uniform int u_two_sided;
uniform int u_has_base_color_texture;
uniform int u_has_opacity_texture;
uniform int u_has_normal_texture;
uniform int u_has_bump_texture;
uniform int u_has_metallic_roughness_texture;
uniform int u_has_occlusion_texture;
uniform int u_has_emissive_texture;
uniform int u_has_specular_texture;
uniform int u_has_specular_color_texture;
uniform int u_has_specular_glossiness_texture;
uniform vec4 u_texture_offset_scale[9];
uniform float u_texture_rotation[9];
uniform int u_texture_texcoord[9];
uniform int u_texture_top_left[9];
uniform int u_directional_light_count;
uniform int u_point_light_count;
uniform int u_spot_light_count;
uniform int u_transparent_pass;

in VS_OUT {
    vec3 world_position;
    vec3 normal;
    vec2 uv;
    vec2 uv1;
    vec4 tangent;
    vec4 color;
} fragment_in;

layout(location = 0) out vec4 out_linear_color;
layout(location = 1) out vec4 out_transparency_accum;
layout(location = 2) out float out_transparency_reveal;

void write_fragment(vec3 color, float opacity) {
    if (u_transparent_pass != 0) {
        float alpha = clamp(opacity, 0.0, 1.0);
        float depth_weight = pow(max(0.01, 1.0 - gl_FragCoord.z * 0.9), 3.0);
        float weight = clamp((pow(min(1.0, alpha * 10.0) + 0.01, 3.0) * 1.0e4) * depth_weight, 0.01, 3000.0);
        out_linear_color = vec4(0.0);
        out_transparency_accum = vec4(color * alpha, alpha) * weight;
        out_transparency_reveal = alpha;
    } else {
        out_linear_color = vec4(color, 1.0);
        out_transparency_accum = vec4(0.0);
        out_transparency_reveal = 0.0;
    }
}

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 rotate_y(vec3 direction, float radians) {
    float c = cos(radians);
    float s = sin(radians);
    return vec3(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z);
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
    vec3 tangent = fragment_in.tangent.xyz - normal * dot(normal, fragment_in.tangent.xyz);
    if (dot(tangent, tangent) <= 1.0e-12) {
        return normal;
    }
    tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(normal, tangent)) * fragment_in.tangent.w;
    if (u_has_normal_texture != 0) {
        vec3 mapped = texture(u_normal_or_bump_texture, material_uv(2)).xyz * 2.0 - 1.0;
        mapped.xy *= u_normal_scale;
        return normalize(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
    }
    if (u_has_bump_texture != 0) {
        vec2 texel = 1.0 / vec2(textureSize(u_normal_or_bump_texture, 0));
        vec2 uv = material_uv(2);
        float left = luminance(texture(u_normal_or_bump_texture, uv - vec2(texel.x, 0.0)).rgb);
        float right = luminance(texture(u_normal_or_bump_texture, uv + vec2(texel.x, 0.0)).rgb);
        float down = luminance(texture(u_normal_or_bump_texture, uv - vec2(0.0, texel.y)).rgb);
        float up = luminance(texture(u_normal_or_bump_texture, uv + vec2(0.0, texel.y)).rgb);
        vec3 candidate = normal - (tangent * (right - left) + bitangent * (up - down)) * (0.5 * u_bump_scale);
        return dot(candidate, candidate) > 1.0e-12 ? normalize(candidate) : normal;
    }
    return normal;
}

float distribution_ggx(vec3 normal, vec3 half_vector, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    float denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(PI * denominator * denominator, 1.0e-8);
}

float geometry_smith_g1(float n_dot_v, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float tangent2 = max(0.0, (1.0 - n_dot_v * n_dot_v) / max(n_dot_v * n_dot_v, 1.0e-8));
    return 2.0 / (1.0 + sqrt(1.0 + alpha2 * tangent2));
}

vec3 fresnel_schlick(float cosine, vec3 f0, vec3 f90) {
    return f0 + (f90 - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

float punctual_range_attenuation(float distance, float range) {
    if (range <= 0.0) {
        return 1.0;
    }
    float ratio = distance / range;
    float cutoff = max(0.0, 1.0 - ratio * ratio * ratio * ratio);
    return cutoff * cutoff;
}

vec3 evaluate_light(
    vec3 diffuse_color,
    vec3 specular_f0,
    vec3 specular_f90,
    vec3 diffuse_fresnel_f0,
    vec3 diffuse_fresnel_f90,
    int diffuse_fresnel_uses_max,
    float roughness,
    vec3 normal,
    vec3 view_direction,
    vec3 light_direction,
    vec3 radiance) {
    float n_dot_l = max(dot(normal, light_direction), 0.0);
    float n_dot_v = max(dot(normal, view_direction), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3(0.0);
    }
    vec3 half_vector = normalize(view_direction + light_direction);
    vec3 fresnel = fresnel_schlick(
        max(dot(view_direction, half_vector), 0.0),
        specular_f0,
        specular_f90);
    vec3 diffuse_fresnel = fresnel_schlick(
        max(dot(view_direction, half_vector), 0.0),
        diffuse_fresnel_f0,
        diffuse_fresnel_f90);
    float distribution = distribution_ggx(normal, half_vector, roughness);
    float geometry = geometry_smith_g1(n_dot_v, roughness) * geometry_smith_g1(n_dot_l, roughness);
    vec3 specular = fresnel * (distribution * geometry / max(4.0 * n_dot_v * n_dot_l, 1.0e-8));
    vec3 diffuse_weight = diffuse_fresnel_uses_max != 0
        ? vec3(1.0 - max(max(diffuse_fresnel.r, diffuse_fresnel.g), diffuse_fresnel.b))
        : vec3(1.0) - diffuse_fresnel;
    vec3 diffuse = diffuse_weight * diffuse_color / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

vec3 diffuse_environment_irradiance(vec3 local_normal) {
    float x = local_normal.x;
    float y = local_normal.y;
    float z = local_normal.z;
    float basis[9] = float[9](
        0.2820947918,
        0.4886025119 * y,
        0.4886025119 * z,
        0.4886025119 * x,
        1.0925484306 * x * y,
        1.0925484306 * y * z,
        0.3153915653 * (3.0 * z * z - 1.0),
        1.0925484306 * x * z,
        0.5462742153 * (x * x - y * y));
    vec3 result = vec3(0.0);
    for (int index = 0; index < 9; ++index) {
        float convolution = index == 0 ? PI : (index <= 3 ? 2.0 * PI / 3.0 : PI / 4.0);
        result += u_environment_sh[index] * basis[index] * convolution;
    }
    return max(result, vec3(0.0));
}

void main() {
    vec4 base_sample = u_has_base_color_texture != 0
        ? texture(u_base_color_texture, material_uv(0))
        : vec4(1.0);
    vec3 base_color = u_base_color * base_sample.rgb * fragment_in.color.rgb;
    float opacity = clamp(u_opacity * base_sample.a * fragment_in.color.a, 0.0, 1.0);
    if (u_has_opacity_texture != 0) {
        opacity *= luminance(texture(u_opacity_texture, material_uv(1)).rgb);
    }
    if (u_alpha_mode == 0) {
        opacity = 1.0;
    } else if (u_alpha_mode == 1 && opacity < u_alpha_cutoff) {
        discard;
    } else if (u_alpha_mode == 1) {
        opacity = 1.0;
    }

    vec3 emission = u_emission;
    if (u_has_emissive_texture != 0) {
        emission *= texture(u_emissive_texture, material_uv(5)).rgb;
    }
    if (u_material_type == 3) {
        write_fragment(emission, opacity);
        return;
    }

    float metallic = u_material_type == 1 ? 1.0 : clamp(u_metallic, 0.0, 1.0);
    float roughness = u_material_type == 0 ? 1.0 : clamp(u_roughness, 0.02, 1.0);
    vec3 diffuse_color;
    vec3 specular_f0;
    vec3 specular_f90;
    vec3 diffuse_fresnel_f0;
    vec3 diffuse_fresnel_f90;
    int diffuse_fresnel_uses_max;
    if (u_material_type == 4 && u_pbr_workflow == 1) {
        vec3 specular = max(u_specular_color, vec3(0.0));
        float glossiness = clamp(u_glossiness, 0.0, 1.0);
        if (u_has_specular_glossiness_texture != 0) {
            vec4 packed_specular_glossiness = texture(
                u_specular_glossiness_texture,
                material_uv(8));
            specular *= packed_specular_glossiness.rgb;
            glossiness *= packed_specular_glossiness.a;
        }
        specular_f0 = clamp(specular, vec3(0.0), vec3(1.0));
        specular_f90 = vec3(1.0);
        diffuse_fresnel_f0 = specular_f0;
        diffuse_fresnel_f90 = specular_f90;
        diffuse_fresnel_uses_max = 0;
        diffuse_color = base_color * (1.0 - max(max(specular_f0.r, specular_f0.g), specular_f0.b));
        metallic = 0.0;
        roughness = clamp(1.0 - glossiness, 0.02, 1.0);
    } else {
        if (u_has_metallic_roughness_texture != 0) {
            vec3 packed_value = texture(u_metallic_roughness_texture, material_uv(3)).rgb;
            roughness = clamp(roughness * packed_value.g, 0.02, 1.0);
            metallic = clamp(metallic * packed_value.b, 0.0, 1.0);
        }
        float specular_strength = clamp(u_specular_factor, 0.0, 1.0);
        if (u_has_specular_texture != 0) {
            specular_strength *= texture(u_specular_texture, material_uv(6)).a;
        }
        vec3 specular_color = max(u_specular_color, vec3(0.0));
        if (u_has_specular_color_texture != 0) {
            specular_color *= texture(u_specular_color_texture, material_uv(7)).rgb;
        }
        float ior_ratio = (max(u_ior, 1.0) - 1.0) / (max(u_ior, 1.0) + 1.0);
        vec3 dielectric_f0 = min(
            vec3(1.0),
            specular_color * (ior_ratio * ior_ratio)) * specular_strength;
        specular_f0 = mix(dielectric_f0, base_color, metallic);
        specular_f90 = mix(vec3(specular_strength), vec3(1.0), metallic);
        diffuse_color = base_color * (1.0 - metallic);
        diffuse_fresnel_f0 = dielectric_f0;
        diffuse_fresnel_f90 = vec3(specular_strength);
        diffuse_fresnel_uses_max = 1;
    }
    float occlusion = 1.0;
    if (u_has_occlusion_texture != 0) {
        float sampled = texture(u_occlusion_texture, material_uv(4)).r;
        occlusion = mix(1.0, sampled, clamp(u_occlusion_strength, 0.0, 1.0));
    }

    vec3 normal = surface_normal();
    vec3 view_direction = normalize(u_camera_position - fragment_in.world_position);
    vec3 color = emission;
    for (int index = 0; index < u_directional_light_count; ++index) {
        vec3 direction = u_directional_lights[index].direction.xyz;
        if (dot(direction, direction) > 1.0e-12) {
            color += evaluate_light(diffuse_color, specular_f0, specular_f90,
                diffuse_fresnel_f0, diffuse_fresnel_f90, diffuse_fresnel_uses_max,
                roughness, normal, view_direction,
                normalize(-direction), u_directional_lights[index].radiance.xyz);
        }
    }
    for (int index = 0; index < u_point_light_count; ++index) {
        vec3 to_light = u_point_lights[index].position_range.xyz - fragment_in.world_position;
        float distance2 = max(dot(to_light, to_light), 1.0e-12);
        float distance = sqrt(distance2);
        float range_attenuation = punctual_range_attenuation(
            distance,
            u_point_lights[index].position_range.w);
        color += evaluate_light(diffuse_color, specular_f0, specular_f90,
            diffuse_fresnel_f0, diffuse_fresnel_f90, diffuse_fresnel_uses_max,
            roughness, normal, view_direction,
            to_light / distance,
            u_point_lights[index].intensity.xyz * (range_attenuation / distance2));
    }
    for (int index = 0; index < u_spot_light_count; ++index) {
        vec3 to_light = u_spot_lights[index].position_range.xyz - fragment_in.world_position;
        float distance2 = max(dot(to_light, to_light), 1.0e-12);
        float distance = sqrt(distance2);
        float range = u_spot_lights[index].position_range.w;
        float range_attenuation = punctual_range_attenuation(distance, range);
        if (range_attenuation <= 0.0) {
            continue;
        }
        vec3 light_direction = to_light / distance;
        float cone_cosine = dot(-light_direction, normalize(u_spot_lights[index].direction_inner.xyz));
        float inner = u_spot_lights[index].direction_inner.w;
        float outer = u_spot_lights[index].intensity_outer.w;
        float cone = smoothstep(outer, inner, cone_cosine);
        color += evaluate_light(diffuse_color, specular_f0, specular_f90,
            diffuse_fresnel_f0, diffuse_fresnel_f90, diffuse_fresnel_uses_max,
            roughness, normal, view_direction,
            light_direction,
            u_spot_lights[index].intensity_outer.xyz *
                (cone * range_attenuation / distance2));
    }

    float n_dot_v = max(dot(normal, view_direction), 0.0);
    vec3 diffuse_fresnel = fresnel_schlick(
        n_dot_v,
        diffuse_fresnel_f0,
        diffuse_fresnel_f90);
    vec3 local_normal = rotate_y(normal, -u_environment_rotation_radians);
    vec3 diffuse_irradiance = u_has_environment_map != 0
        ? diffuse_environment_irradiance(local_normal) * u_environment_color * u_environment_intensity
        : u_environment_color * (PI * u_environment_intensity);
    vec3 diffuse_weight = diffuse_fresnel_uses_max != 0
        ? vec3(1.0 - max(max(diffuse_fresnel.r, diffuse_fresnel.g), diffuse_fresnel.b))
        : vec3(1.0) - diffuse_fresnel;
    vec3 diffuse_ibl = diffuse_weight * diffuse_color * diffuse_irradiance / PI;
    vec3 reflection = reflect(-view_direction, normal);
    vec3 local_reflection = rotate_y(reflection, -u_environment_rotation_radians);
    vec3 specular_radiance = u_has_environment_map != 0
        ? textureLod(u_environment_prefilter, local_reflection, roughness * max(u_environment_mip_count - 1.0, 0.0)).rgb *
            u_environment_color * u_environment_intensity
        : u_environment_color * u_environment_intensity;
    vec2 brdf = texture(u_environment_brdf_lut, vec2(n_dot_v, roughness)).rg;
    vec3 specular_ibl = specular_radiance * (specular_f0 * brdf.x + specular_f90 * brdf.y);
    color += (diffuse_ibl + specular_ibl) * occlusion;
    write_fragment(color, opacity);
}
