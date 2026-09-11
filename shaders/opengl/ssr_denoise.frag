#version 450 core

layout(binding = 0) uniform sampler2D u_source_indirect;
layout(binding = 1) uniform sampler2D u_moments;
layout(binding = 2) uniform sampler2D u_linear_depth;
layout(binding = 3) uniform sampler2D u_view_normal;

layout(binding = 4) uniform sampler2D u_ssr_pbr;
layout(binding = 5) uniform sampler2D u_ssr_material;

uniform ivec2 u_full_resolution;
uniform ivec2 u_trace_resolution;
uniform ivec2 u_filter_axis;
uniform int u_stride;
uniform float u_depth_sigma;
uniform float u_normal_power;

in vec2 v_uv;
layout(location = 0) out vec4 out_filtered_indirect;

struct SurfaceSample {
    float depth;
    vec3 normal;
};

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

SurfaceSample representative_surface(ivec2 pixel) {
    SurfaceSample result;
    result.depth = texelFetch(u_linear_depth, pixel, 0).r;
    result.normal = texelFetch(u_view_normal, pixel, 0).xyz;
    if (dot(result.normal, result.normal) > 1.0e-8) result.normal = normalize(result.normal);
    return result;
}

float kernel_weight(int offset) {
    int absolute_offset = abs(offset);
    return absolute_offset == 0
        ? 6.0 / 16.0
        : (absolute_offset == 1 ? 4.0 / 16.0 : 1.0 / 16.0);
}

void main() {
    ivec2 center_pixel = clamp(
        ivec2(gl_FragCoord.xy),
        ivec2(0),
        u_trace_resolution - 1);
    SurfaceSample center_surface = representative_surface(center_pixel);
    vec4 center_value = texelFetch(u_source_indirect, center_pixel, 0);
    if (center_surface.depth <= 0.0 ||
        dot(center_surface.normal, center_surface.normal) <= 1.0e-8) {
        out_filtered_indirect = center_value;
        return;
    }

    vec4 center_pbr = texelFetch(u_ssr_pbr, center_pixel, 0);
    vec3 center_material = texelFetch(u_ssr_material, center_pixel, 0).rgb;
    // Avoid spatially smearing a sharp, view-dependent reflection.
    float glossy = smoothstep(0.08, 0.35, center_pbr.a);
    if (glossy <= 0.0 && dot(center_pbr.rgb, center_pbr.rgb) > 1.0e-8) {
        out_filtered_indirect = center_value;
        return;
    }
    vec2 center_moments = texelFetch(u_moments, center_pixel, 0).rg;
    float variance = max(
        center_moments.y - center_moments.x * center_moments.x,
        0.0);
    float center_luminance = luminance(center_value.rgb);
    // Moments describe the noisy per-frame estimator, not the already
    // accumulated mean.  A wider variance kernel lets a-trous actually share
    // rare screen hits while depth and normal weights preserve geometry.
    float luminance_sigma = max(
        2.0 * sqrt(variance),
        max(
            0.05 * max(abs(center_luminance), abs(center_moments.x)),
            0.02));
    vec3 sum = vec3(0.0);
    float weight_sum = 0.0;
    for (int offset = -2; offset <= 2; ++offset) {
        ivec2 pixel = center_pixel +
            u_filter_axis * (offset * max(u_stride, 1));
        if (any(lessThan(pixel, ivec2(0))) ||
            any(greaterThanEqual(pixel, u_trace_resolution))) {
            continue;
        }
        SurfaceSample sample_surface = representative_surface(pixel);
        if (sample_surface.depth <= 0.0 ||
            dot(sample_surface.normal, sample_surface.normal) <= 1.0e-8) {
            continue;
        }
        vec3 sample_value = texelFetch(u_source_indirect, pixel, 0).rgb;
        float spatial_weight = kernel_weight(offset);
        float depth_weight = exp(
            -abs(sample_surface.depth - center_surface.depth) /
            max(u_depth_sigma, 1.0e-5));
        float normal_weight = pow(
            max(dot(sample_surface.normal, center_surface.normal), 0.0),
            max(u_normal_power, 1.0));
        float luminance_weight = exp(
            -abs(luminance(sample_value) - center_luminance) /
            luminance_sigma);
        vec4 sample_pbr = texelFetch(u_ssr_pbr, pixel, 0);
        vec3 sample_material = texelFetch(u_ssr_material, pixel, 0).rgb;
        float material_weight = exp(-32.0 * abs(sample_pbr.a - center_pbr.a) -
            8.0 * length(sample_pbr.rgb - center_pbr.rgb) -
            8.0 * length(sample_material - center_material));
        float weight = spatial_weight * depth_weight * normal_weight *
            luminance_weight * material_weight * (offset == 0 ? 1.0 : glossy);
        sum += sample_value * weight;
        weight_sum += weight;
    }
    out_filtered_indirect = weight_sum > 1.0e-8
        ? vec4(sum / weight_sum, center_value.a)
        : center_value;
}
