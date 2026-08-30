#version 450 core

layout(binding = 0) uniform sampler2D u_opaque;
layout(binding = 1) uniform sampler2D u_original_diffuse_ibl;
layout(binding = 2) uniform sampler2D u_raw_indirect;
layout(binding = 3) uniform sampler2D u_temporal_indirect;
layout(binding = 4) uniform sampler2D u_filtered_indirect;
layout(binding = 5) uniform sampler2D u_linear_depth;
layout(binding = 6) uniform sampler2D u_view_normal;
layout(binding = 7) uniform sampler2D u_ssgi_material;

uniform ivec2 u_full_resolution;
uniform ivec2 u_half_resolution;
uniform float u_strength;
uniform float u_depth_sigma;
uniform float u_normal_power;
uniform int u_max_history_frames;
uniform int u_debug_view;

in vec2 v_uv;
layout(location = 0) out vec4 out_linear_color;

struct SurfaceSample {
    float depth;
    vec3 normal;
    vec3 response;
};

SurfaceSample representative_surface(ivec2 half_pixel) {
    SurfaceSample result;
    result.depth = 0.0;
    result.normal = vec3(0.0);
    result.response = vec3(0.0);
    float nearest = 3.402823466e+38;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            ivec2 pixel = half_pixel * 2 + ivec2(x, y);
            if (any(greaterThanEqual(pixel, u_full_resolution))) {
                continue;
            }
            float depth = texelFetch(u_linear_depth, pixel, 0).r;
            vec3 normal = texelFetch(u_view_normal, pixel, 0).xyz;
            if (depth > 0.0 && dot(normal, normal) > 1.0e-8 &&
                depth < nearest) {
                nearest = depth;
                result.depth = depth;
                result.normal = normalize(normal);
                result.response = texelFetch(
                    u_ssgi_material, pixel, 0).rgb;
            }
        }
    }
    return result;
}

vec3 bilateral_upsample(ivec2 full_pixel, out bool reliable) {
    float center_depth = texelFetch(u_linear_depth, full_pixel, 0).r;
    vec3 center_normal = texelFetch(u_view_normal, full_pixel, 0).xyz;
    vec3 center_response = texelFetch(
        u_ssgi_material, full_pixel, 0).rgb;
    reliable = false;
    if (center_depth <= 0.0 || dot(center_normal, center_normal) <= 1.0e-8) {
        return vec3(0.0);
    }
    if (dot(center_response, center_response) <= 1.0e-12) {
        return vec3(0.0);
    }
    center_normal = normalize(center_normal);
    vec2 half_position = (vec2(full_pixel) + 0.5) * 0.5 - 0.5;
    ivec2 half_center = ivec2(round(half_position));
    vec3 sum = vec3(0.0);
    float weight_sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 half_pixel = half_center + ivec2(x, y);
            if (any(lessThan(half_pixel, ivec2(0))) ||
                any(greaterThanEqual(half_pixel, u_half_resolution))) {
                continue;
            }
            SurfaceSample sample_surface = representative_surface(half_pixel);
            if (sample_surface.depth <= 0.0 ||
                dot(sample_surface.normal, sample_surface.normal) <= 1.0e-8) {
                continue;
            }
            vec2 delta = vec2(half_pixel) - half_position;
            float spatial_weight = exp(-0.5 * dot(delta, delta));
            float depth_weight = exp(
                -abs(sample_surface.depth - center_depth) /
                max(u_depth_sigma, 1.0e-5));
            float normal_weight = pow(
                max(dot(sample_surface.normal, center_normal), 0.0),
                max(u_normal_power, 1.0));
            float material_weight = exp(
                -4.0 * length(sample_surface.response - center_response));
            float weight = spatial_weight * depth_weight * normal_weight *
                material_weight;
            sum += texelFetch(u_filtered_indirect, half_pixel, 0).rgb * weight;
            weight_sum += weight;
        }
    }
    reliable = weight_sum > 1.0e-5;
    return reliable ? sum / weight_sum : vec3(0.0);
}

void main() {
    ivec2 full_pixel = clamp(
        ivec2(gl_FragCoord.xy),
        ivec2(0),
        u_full_resolution - 1);
    vec3 opaque = texelFetch(u_opaque, full_pixel, 0).rgb;
    vec3 original_diffuse = texelFetch(
        u_original_diffuse_ibl, full_pixel, 0).rgb;
    bool reliable = false;
    vec3 filtered_residual = bilateral_upsample(full_pixel, reliable);
    if (!reliable) {
        filtered_residual = vec3(0.0);
    }

    ivec2 half_pixel = clamp(
        full_pixel / 2,
        ivec2(0),
        u_half_resolution - 1);
    if (u_debug_view != 0) {
        vec3 debug_color = vec3(0.0);
        if (u_debug_view == 1) {
            debug_color = original_diffuse + texelFetch(
                u_raw_indirect, half_pixel, 0).rgb;
        } else if (u_debug_view == 2) {
            debug_color = vec3(texelFetch(
                u_raw_indirect, half_pixel, 0).a);
        } else if (u_debug_view == 3) {
            debug_color = original_diffuse + texelFetch(
                u_temporal_indirect, half_pixel, 0).rgb;
        } else if (u_debug_view == 4) {
            debug_color = original_diffuse + filtered_residual;
        } else {
            float history_length = texelFetch(
                u_temporal_indirect, half_pixel, 0).a;
            debug_color = vec3(clamp(
                history_length /
                    float(max(u_max_history_frames, 1)),
                0.0,
                1.0));
        }
        out_linear_color = vec4(max(debug_color, vec3(0.0)), 1.0);
        return;
    }

    vec3 composed = opaque + clamp(u_strength, 0.0, 1.0) *
        filtered_residual;
    out_linear_color = vec4(max(composed, vec3(0.0)), 1.0);
}
