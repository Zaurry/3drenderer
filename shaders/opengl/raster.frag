#version 450 core

const float PI = 3.14159265358979323846;

struct DirectionalLight { vec4 direction_angular; vec4 radiance; vec4 shadow; };
struct PointLight { vec4 position_range; vec4 intensity; vec4 shadow; };
struct SpotLight { vec4 position_range; vec4 direction_inner; vec4 intensity_outer; vec4 shadow; };
struct RectAreaLight { vec4 position_two_sided; vec4 axis_u; vec4 axis_v; vec4 radiance; vec4 shadow; };

layout(std430, binding = 0) readonly buffer DirectionalLightBuffer {
    DirectionalLight u_directional_lights[];
};
layout(std430, binding = 1) readonly buffer PointLightBuffer {
    PointLight u_point_lights[];
};
layout(std430, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight u_spot_lights[];
};
layout(std430, binding = 3) readonly buffer RectAreaLightBuffer {
    RectAreaLight u_rect_area_lights[];
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
layout(binding = 11) uniform sampler2D u_ltc_matrix_lut;
layout(binding = 12) uniform sampler2D u_ltc_amplitude_lut;
layout(binding = 13) uniform sampler2DArray u_shadow_maps_2d;
layout(binding = 14) uniform samplerCubeArray u_shadow_maps_cube;
layout(binding = 15) uniform sampler2D u_ambient_occlusion_texture;

uniform vec3 u_camera_position;
uniform vec3 u_environment_color;
uniform vec3 u_environment_sh[9];
uniform float u_environment_intensity;
uniform float u_environment_rotation_radians;
uniform float u_environment_mip_count;
uniform int u_has_environment_map;
uniform int u_ibl_enabled;
uniform int u_ltc_area_lights_enabled;
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
uniform int u_rect_area_light_count;
uniform mat4 u_shadow_matrices[32];
uniform vec4 u_shadow_origin_far[32];
uniform vec4 u_shadow_direction_near[32];
uniform vec4 u_cube_shadow_position_far[32];
uniform float u_cube_shadow_near[32];
uniform float u_shadow_map_resolution;
uniform float u_shadow_constant_bias;
uniform float u_shadow_slope_bias;
uniform int u_pcss_enabled;
uniform int u_pcss_blocker_samples;
uniform int u_pcss_filter_samples;
uniform float u_pcss_max_penumbra_texels;
uniform float u_pcss_light_size_scale;
uniform int u_shadow_debug_view;
uniform int u_shadow_debug_slot;
uniform int u_transparent_pass;
uniform vec2 u_viewport_size;
uniform mat3 u_view_to_world;
uniform int u_ao_mode;
uniform int u_ao_bent_normals_enabled;

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

vec3 ltc_integrate_edge_vector(vec3 first, vec3 second) {
    float cosine = dot(first, second);
    float absolute_cosine = abs(cosine);
    float numerator = 0.8543985 +
        (0.4965155 + 0.0145206 * absolute_cosine) * absolute_cosine;
    float denominator = 3.4175940 +
        (4.1616724 + absolute_cosine) * absolute_cosine;
    float approximation = numerator / denominator;
    float theta_over_sine = cosine > 0.0
        ? approximation
        : 0.5 * inversesqrt(max(1.0 - cosine * cosine, 1.0e-7)) - approximation;
    return cross(first, second) * theta_over_sine;
}

float ltc_evaluate(
    vec3 normal,
    vec3 view_direction,
    vec3 position,
    mat3 inverse_ltc,
    vec3 points[4],
    bool two_sided) {
    vec3 tangent = view_direction - normal * dot(view_direction, normal);
    if (dot(tangent, tangent) <= 1.0e-10) {
        vec3 helper = abs(normal.z) < 0.999
            ? vec3(0.0, 0.0, 1.0)
            : vec3(1.0, 0.0, 0.0);
        tangent = normalize(cross(helper, normal));
    } else {
        tangent = normalize(tangent);
    }
    // LTC's polygon integration uses a left-handed view basis here. Using
    // +cross(N,T) reverses the polygon form-factor sign relative to the
    // receiver horizon and lets lights above a two-sided floor illuminate its
    // underside. This matches the reference LTC_Evaluate construction.
    vec3 bitangent = -cross(normal, tangent);
    inverse_ltc *= transpose(mat3(tangent, bitangent, normal));

    vec3 directions[4];
    for (int index = 0; index < 4; ++index) {
        directions[index] = normalize(inverse_ltc * (points[index] - position));
    }
    vec3 form_factor = vec3(0.0);
    form_factor += ltc_integrate_edge_vector(directions[0], directions[1]);
    form_factor += ltc_integrate_edge_vector(directions[1], directions[2]);
    form_factor += ltc_integrate_edge_vector(directions[2], directions[3]);
    form_factor += ltc_integrate_edge_vector(directions[3], directions[0]);
    float form_length = length(form_factor);
    if (form_length <= 1.0e-8) {
        return 0.0;
    }
    vec3 light_normal = normalize(cross(points[1] - points[0], points[3] - points[0]));
    // A receiver is behind the emitter when it lies opposite the polygon's
    // front normal. The previous test used the inverse vector and therefore
    // made one-sided LTC illumination disagree with the visible/CUDA quad.
    bool behind = dot(position - points[0], light_normal) < 0.0;
    float form_z = form_factor.z / form_length;
    if (behind) {
        form_z = -form_z;
    }
    const float lut_scale = 63.0 / 64.0;
    const float lut_bias = 0.5 / 64.0;
    vec2 horizon_uv = vec2(form_z * 0.5 + 0.5, form_length);
    horizon_uv = clamp(horizon_uv, vec2(0.0), vec2(1.0)) * lut_scale + lut_bias;
    float horizon_scale = texture(u_ltc_amplitude_lut, horizon_uv).w;
    float integral = form_length * horizon_scale;
    return behind && !two_sided ? 0.0 : integral;
}

bool rect_area_light_emits_toward_receiver(
    RectAreaLight light,
    vec3 receiver_position) {
    vec3 emission_normal = cross(light.axis_v.xyz, light.axis_u.xyz);
    if (dot(emission_normal, emission_normal) <= 1.0e-12) {
        return false;
    }
    if (light.position_two_sided.w > 0.5) {
        return true;
    }
    return dot(
        receiver_position - light.position_two_sided.xyz,
        emission_normal) > 0.0;
}

vec3 evaluate_rect_area_light(
    RectAreaLight light,
    vec3 diffuse_color,
    vec3 specular_f0,
    vec3 specular_f90,
    vec3 diffuse_fresnel_f0,
    vec3 diffuse_fresnel_f90,
    int diffuse_fresnel_uses_max,
    float roughness,
    vec3 normal,
    vec3 view_direction,
    vec3 position) {
    vec3 center = light.position_two_sided.xyz;
    vec3 axis_u = light.axis_u.xyz;
    vec3 axis_v = light.axis_v.xyz;
    if (dot(axis_u, axis_u) <= 1.0e-12 || dot(axis_v, axis_v) <= 1.0e-12) {
        return vec3(0.0);
    }
    if (!rect_area_light_emits_toward_receiver(light, position)) {
        return vec3(0.0);
    }
    vec3 points[4];
    // Winding makes axis_v x axis_u the emitting side, matching the visible quad.
    points[0] = center - axis_u - axis_v;
    points[1] = center - axis_u + axis_v;
    points[2] = center + axis_u + axis_v;
    points[3] = center + axis_u - axis_v;
    bool crosses_receiver_horizon = false;
    for (int index = 0; index < 4; ++index) {
        crosses_receiver_horizon = crosses_receiver_horizon ||
            dot(normal, points[index] - position) > 0.0;
    }
    if (!crosses_receiver_horizon) {
        return vec3(0.0);
    }
    bool two_sided = light.position_two_sided.w > 0.5;
    float n_dot_v = max(dot(normal, view_direction), 0.0);
    const float lut_scale = 63.0 / 64.0;
    const float lut_bias = 0.5 / 64.0;
    vec2 uv = vec2(roughness, sqrt(max(0.0, 1.0 - n_dot_v)));
    uv = uv * lut_scale + lut_bias;
    vec4 matrix_sample = texture(u_ltc_matrix_lut, uv);
    vec4 amplitude_sample = texture(u_ltc_amplitude_lut, uv);
    mat3 inverse_ltc = mat3(
        vec3(matrix_sample.x, 0.0, matrix_sample.y),
        vec3(0.0, 1.0, 0.0),
        vec3(matrix_sample.z, 0.0, matrix_sample.w));
    float specular_integral = ltc_evaluate(
        normal, view_direction, position, inverse_ltc, points, two_sided);
    float diffuse_integral = ltc_evaluate(
        normal, view_direction, position, mat3(1.0), points, two_sided);
    vec3 diffuse_fresnel = fresnel_schlick(
        n_dot_v, diffuse_fresnel_f0, diffuse_fresnel_f90);
    vec3 diffuse_weight = diffuse_fresnel_uses_max != 0
        ? vec3(1.0 - max(max(diffuse_fresnel.r, diffuse_fresnel.g), diffuse_fresnel.b))
        : vec3(1.0) - diffuse_fresnel;
    vec3 specular_amplitude =
        specular_f0 * amplitude_sample.x + specular_f90 * amplitude_sample.y;
    return light.radiance.xyz *
        (diffuse_weight * diffuse_color * diffuse_integral +
         specular_amplitude * specular_integral);
}

float stable_shadow_rotation(vec3 position, int slot) {
    vec3 cell = floor(position * 4096.0) / 4096.0;
    return 6.28318530718 * fract(sin(dot(
        cell + vec3(float(slot) * 0.173),
        vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

vec2 vogel_disk(int sample_index, int sample_count, float rotation) {
    float radius = sqrt((float(sample_index) + 0.5) / float(max(sample_count, 1)));
    float angle = float(sample_index) * 2.39996322973 + rotation;
    return radius * vec2(cos(angle), sin(angle));
}

float receiver_shadow_bias(vec3 normal, vec3 light_direction) {
    float slope = 1.0 - max(dot(normal, light_direction), 0.0);
    return u_shadow_constant_bias + u_shadow_slope_bias * slope;
}

float shadow_visibility_2d(
    int layer,
    int global_slot,
    vec3 position,
    vec3 normal,
    vec3 light_direction,
    float angular_radius,
    out float average_blocker,
    out float penumbra_texels) {
    average_blocker = 1.0;
    penumbra_texels = 0.0;
    if (layer < 0 || layer >= 32) {
        return 1.0;
    }
    vec4 projected = u_shadow_matrices[layer] * vec4(position, 1.0);
    if (abs(projected.w) <= 1.0e-8) {
        return 1.0;
    }
    vec3 normalized = projected.xyz / projected.w;
    vec2 uv = normalized.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }
    float near_plane = u_shadow_direction_near[layer].w;
    float far_plane = u_shadow_origin_far[layer].w;
    float linear_distance = dot(
        position - u_shadow_origin_far[layer].xyz,
        u_shadow_direction_near[layer].xyz);
    float receiver_depth = clamp(
        (linear_distance - near_plane) /
        max(far_plane - near_plane, 1.0e-6),
        0.0,
        1.0);
    float bias = receiver_shadow_bias(normal, light_direction);
    float center_depth = texture(u_shadow_maps_2d, vec3(uv, float(layer))).r;
    if (u_pcss_enabled == 0 || u_pcss_max_penumbra_texels <= 0.0 ||
        u_pcss_light_size_scale <= 0.0 || angular_radius <= 0.0) {
        average_blocker = center_depth;
        return center_depth + bias >= receiver_depth ? 1.0 : 0.0;
    }
    float search_radius_texels = clamp(
        angular_radius * u_pcss_light_size_scale * u_shadow_map_resolution,
        0.0,
        u_pcss_max_penumbra_texels);
    float rotation = stable_shadow_rotation(position, global_slot);
    float blocker_sum = 0.0;
    int blocker_count = 0;
    for (int sample_index = 0; sample_index < 64; ++sample_index) {
        if (sample_index >= u_pcss_blocker_samples) {
            break;
        }
        vec2 offset = vogel_disk(sample_index, u_pcss_blocker_samples, rotation) *
            (search_radius_texels / u_shadow_map_resolution);
        float sampled_depth = texture(
            u_shadow_maps_2d,
            vec3(uv + offset, float(layer))).r;
        if (sampled_depth + bias < receiver_depth) {
            blocker_sum += sampled_depth;
            ++blocker_count;
        }
    }
    if (blocker_count == 0) {
        return 1.0;
    }
    average_blocker = blocker_sum / float(blocker_count);
    float average_blocker_distance = near_plane + average_blocker *
        (far_plane - near_plane);
    penumbra_texels = clamp(
        (linear_distance - average_blocker_distance) /
            max(average_blocker_distance, 1.0e-4) * search_radius_texels,
        0.0,
        u_pcss_max_penumbra_texels);
    float visible_samples = 0.0;
    for (int sample_index = 0; sample_index < 64; ++sample_index) {
        if (sample_index >= u_pcss_filter_samples) {
            break;
        }
        vec2 offset = vogel_disk(sample_index, u_pcss_filter_samples, rotation) *
            (penumbra_texels / u_shadow_map_resolution);
        float sampled_depth = texture(
            u_shadow_maps_2d,
            vec3(uv + offset, float(layer))).r;
        visible_samples += sampled_depth + bias >= receiver_depth ? 1.0 : 0.0;
    }
    return visible_samples / float(max(u_pcss_filter_samples, 1));
}

float shadow_visibility_cube(
    int layer,
    int global_slot,
    vec3 position,
    vec3 normal,
    vec3 light_direction,
    float source_radius,
    out float average_blocker,
    out float penumbra_texels) {
    average_blocker = 1.0;
    penumbra_texels = 0.0;
    if (layer < 0 || layer >= 32) {
        return 1.0;
    }
    vec3 light_position = u_cube_shadow_position_far[layer].xyz;
    float far_plane = u_cube_shadow_position_far[layer].w;
    float near_plane = u_cube_shadow_near[layer];
    vec3 from_light = position - light_position;
    float receiver_distance = length(from_light);
    if (receiver_distance <= 1.0e-6) {
        return 1.0;
    }
    vec3 direction = from_light / receiver_distance;
    float receiver_depth = clamp(
        (receiver_distance - near_plane) /
        max(far_plane - near_plane, 1.0e-6),
        0.0,
        1.0);
    float bias = receiver_shadow_bias(normal, light_direction);
    float center_depth = texture(
        u_shadow_maps_cube,
        vec4(direction, float(layer))).r;
    if (u_pcss_enabled == 0 || u_pcss_max_penumbra_texels <= 0.0 ||
        u_pcss_light_size_scale <= 0.0 || source_radius <= 0.0) {
        average_blocker = center_depth;
        return center_depth + bias >= receiver_depth ? 1.0 : 0.0;
    }
    float source_angle = source_radius / max(receiver_distance, 1.0e-4);
    float search_radius_texels = clamp(
        source_angle * u_pcss_light_size_scale * u_shadow_map_resolution,
        0.0,
        u_pcss_max_penumbra_texels);
    vec3 helper = abs(direction.z) < 0.999
        ? vec3(0.0, 0.0, 1.0)
        : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, direction));
    vec3 bitangent = cross(direction, tangent);
    float rotation = stable_shadow_rotation(position, global_slot);
    float blocker_sum = 0.0;
    int blocker_count = 0;
    for (int sample_index = 0; sample_index < 64; ++sample_index) {
        if (sample_index >= u_pcss_blocker_samples) {
            break;
        }
        vec2 disk = vogel_disk(sample_index, u_pcss_blocker_samples, rotation);
        float angular_offset = search_radius_texels / u_shadow_map_resolution;
        vec3 sample_direction = normalize(
            direction + tangent * disk.x * angular_offset +
            bitangent * disk.y * angular_offset);
        float sampled_depth = texture(
            u_shadow_maps_cube,
            vec4(sample_direction, float(layer))).r;
        if (sampled_depth + bias < receiver_depth) {
            blocker_sum += sampled_depth;
            ++blocker_count;
        }
    }
    if (blocker_count == 0) {
        return 1.0;
    }
    average_blocker = blocker_sum / float(blocker_count);
    float average_blocker_distance = near_plane + average_blocker *
        (far_plane - near_plane);
    penumbra_texels = clamp(
        (receiver_distance - average_blocker_distance) /
            max(average_blocker_distance, 1.0e-4) * search_radius_texels,
        0.0,
        u_pcss_max_penumbra_texels);
    float visible_samples = 0.0;
    for (int sample_index = 0; sample_index < 64; ++sample_index) {
        if (sample_index >= u_pcss_filter_samples) {
            break;
        }
        vec2 disk = vogel_disk(sample_index, u_pcss_filter_samples, rotation);
        float angular_offset = penumbra_texels / u_shadow_map_resolution;
        vec3 sample_direction = normalize(
            direction + tangent * disk.x * angular_offset +
            bitangent * disk.y * angular_offset);
        float sampled_depth = texture(
            u_shadow_maps_cube,
            vec4(sample_direction, float(layer))).r;
        visible_samples += sampled_depth + bias >= receiver_depth ? 1.0 : 0.0;
    }
    return visible_samples / float(max(u_pcss_filter_samples, 1));
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

float gtso_visibility(
    vec3 bent_normal,
    vec3 reflection,
    float roughness,
    float ambient_visibility,
    float n_dot_v) {
    ambient_visibility = clamp(ambient_visibility, 0.0, 1.0);
    if (ambient_visibility >= 0.9999) {
        return 1.0;
    }
    float scalar_visibility = clamp(
        pow(
            max(n_dot_v + ambient_visibility, 0.0),
            exp2(-16.0 * roughness - 1.0)) -
            1.0 + ambient_visibility,
        0.0,
        1.0);
    float cone_cosine = sqrt(max(0.0, 1.0 - ambient_visibility));
    float lobe_width = mix(0.04, 1.0, roughness * roughness);
    float directional_visibility = smoothstep(
        cone_cosine - lobe_width,
        cone_cosine + lobe_width,
        dot(normalize(bent_normal), normalize(reflection)));
    return clamp(
        mix(
            directional_visibility * ambient_visibility,
            scalar_visibility,
            roughness),
        0.0,
        1.0);
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
    bool debug_shadow_matched = false;
    float debug_visibility = 1.0;
    float debug_blocker_depth = 1.0;
    float debug_penumbra_texels = 0.0;
    for (int index = 0; index < u_directional_light_count; ++index) {
        vec3 direction = u_directional_lights[index].direction_angular.xyz;
        if (dot(direction, direction) > 1.0e-12) {
            vec3 light_direction = normalize(-direction);
            int shadow_layer = int(round(u_directional_lights[index].shadow.y));
            int global_slot = int(round(u_directional_lights[index].shadow.z));
            float blocker_depth;
            float penumbra;
            float visibility = shadow_visibility_2d(
                shadow_layer,
                global_slot,
                fragment_in.world_position,
                normal,
                light_direction,
                max(u_directional_lights[index].shadow.x, 0.0),
                blocker_depth,
                penumbra);
            if (global_slot == u_shadow_debug_slot) {
                debug_shadow_matched = true;
                debug_visibility = visibility;
                debug_blocker_depth = blocker_depth;
                debug_penumbra_texels = penumbra;
            }
            color += visibility * evaluate_light(diffuse_color, specular_f0, specular_f90,
                diffuse_fresnel_f0, diffuse_fresnel_f90, diffuse_fresnel_uses_max,
                roughness, normal, view_direction,
                light_direction, u_directional_lights[index].radiance.xyz);
        }
    }
    for (int index = 0; index < u_point_light_count; ++index) {
        vec3 to_light = u_point_lights[index].position_range.xyz - fragment_in.world_position;
        float distance2 = max(dot(to_light, to_light), 1.0e-12);
        float distance = sqrt(distance2);
        float range_attenuation = punctual_range_attenuation(
            distance,
            u_point_lights[index].position_range.w);
        vec3 light_direction = to_light / distance;
        int shadow_layer = int(round(u_point_lights[index].shadow.y));
        int global_slot = int(round(u_point_lights[index].shadow.z));
        float blocker_depth;
        float penumbra;
        float visibility = shadow_visibility_cube(
            shadow_layer,
            global_slot,
            fragment_in.world_position,
            normal,
            light_direction,
            max(u_point_lights[index].shadow.x, 0.0),
            blocker_depth,
            penumbra);
        if (global_slot == u_shadow_debug_slot) {
            debug_shadow_matched = true;
            debug_visibility = visibility;
            debug_blocker_depth = blocker_depth;
            debug_penumbra_texels = penumbra;
        }
        color += visibility * evaluate_light(diffuse_color, specular_f0, specular_f90,
            diffuse_fresnel_f0, diffuse_fresnel_f90, diffuse_fresnel_uses_max,
            roughness, normal, view_direction,
            light_direction,
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
        int shadow_layer = int(round(u_spot_lights[index].shadow.y));
        int global_slot = int(round(u_spot_lights[index].shadow.z));
        float blocker_depth;
        float penumbra;
        float visibility = shadow_visibility_2d(
            shadow_layer,
            global_slot,
            fragment_in.world_position,
            normal,
            light_direction,
            max(u_spot_lights[index].shadow.x, 0.0) / max(distance, 1.0e-4),
            blocker_depth,
            penumbra);
        if (global_slot == u_shadow_debug_slot) {
            debug_shadow_matched = true;
            debug_visibility = visibility;
            debug_blocker_depth = blocker_depth;
            debug_penumbra_texels = penumbra;
        }
        color += visibility * evaluate_light(diffuse_color, specular_f0, specular_f90,
            diffuse_fresnel_f0, diffuse_fresnel_f90, diffuse_fresnel_uses_max,
            roughness, normal, view_direction,
            light_direction,
            u_spot_lights[index].intensity_outer.xyz *
                (cone * range_attenuation / distance2));
    }
    if (u_ltc_area_lights_enabled != 0) {
        for (int index = 0; index < u_rect_area_light_count; ++index) {
            if (!rect_area_light_emits_toward_receiver(
                    u_rect_area_lights[index],
                    fragment_in.world_position)) {
                continue;
            }
            vec3 to_light = u_rect_area_lights[index].position_two_sided.xyz -
                fragment_in.world_position;
            vec3 light_direction = dot(to_light, to_light) > 1.0e-12
                ? normalize(to_light)
                : normal;
            int shadow_layer = int(round(u_rect_area_lights[index].shadow.y));
            int global_slot = int(round(u_rect_area_lights[index].shadow.z));
            float blocker_depth = 1.0;
            float penumbra = 0.0;
            float visibility = 1.0;
            bool casts_shadows = u_rect_area_lights[index].shadow.w > 0.5;
            if (casts_shadows) {
                visibility = shadow_visibility_cube(
                    shadow_layer,
                    global_slot,
                    fragment_in.world_position,
                    normal,
                    light_direction,
                    max(u_rect_area_lights[index].shadow.x, 0.0),
                    blocker_depth,
                    penumbra);
            }
            if (casts_shadows && global_slot == u_shadow_debug_slot) {
                debug_shadow_matched = true;
                debug_visibility = visibility;
                debug_blocker_depth = blocker_depth;
                debug_penumbra_texels = penumbra;
            }
            color += visibility * evaluate_rect_area_light(
                u_rect_area_lights[index],
                diffuse_color,
                specular_f0,
                specular_f90,
                diffuse_fresnel_f0,
                diffuse_fresnel_f90,
                diffuse_fresnel_uses_max,
                roughness,
                normal,
                view_direction,
                fragment_in.world_position);
        }
    }

    if (u_ibl_enabled != 0) {
        float n_dot_v = max(dot(normal, view_direction), 0.0);
        float screen_ao = 1.0;
        vec3 bent_normal = normal;
        vec3 ambient_normal = normal;
        if (u_transparent_pass == 0 && u_ao_mode != 0) {
            vec2 ao_uv = gl_FragCoord.xy / max(u_viewport_size, vec2(1.0));
            vec4 ao_sample = texture(u_ambient_occlusion_texture, ao_uv);
            screen_ao = clamp(ao_sample.a, 0.0, 1.0);
            if (u_ao_mode == 2 && u_ao_bent_normals_enabled != 0 &&
                dot(ao_sample.xyz, ao_sample.xyz) > 1.0e-8) {
                bent_normal = normalize(u_view_to_world * normalize(ao_sample.xyz));
                float normal_alignment = dot(bent_normal, normal);
                if (normal_alignment < 0.01) {
                    bent_normal = normalize(
                        bent_normal + normal * (0.01 - normal_alignment));
                }
                ambient_normal = bent_normal;
            }
        }
        vec3 diffuse_fresnel = fresnel_schlick(
            n_dot_v,
            diffuse_fresnel_f0,
            diffuse_fresnel_f90);
        vec3 local_normal = rotate_y(
            ambient_normal,
            -u_environment_rotation_radians);
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
        float specular_visibility = screen_ao;
        if (u_ao_mode == 2 && u_ao_bent_normals_enabled != 0 &&
            u_transparent_pass == 0) {
            specular_visibility = gtso_visibility(
                bent_normal,
                reflection,
                roughness,
                screen_ao,
                n_dot_v);
        }
        color += diffuse_ibl * (occlusion * screen_ao) +
            specular_ibl * (occlusion * specular_visibility);
    }
    if (u_shadow_debug_view != 0) {
        float debug_value = 0.0;
        if (debug_shadow_matched) {
            if (u_shadow_debug_view == 1) {
                debug_value = debug_visibility;
            } else if (u_shadow_debug_view == 2) {
                debug_value = debug_blocker_depth;
            } else {
                debug_value = debug_penumbra_texels /
                    max(u_pcss_max_penumbra_texels, 1.0);
            }
        }
        write_fragment(vec3(clamp(debug_value, 0.0, 1.0)), opacity);
        return;
    }
    write_fragment(color, opacity);
}
