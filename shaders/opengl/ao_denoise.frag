#version 450 core

layout(binding = 0) uniform sampler2D u_source_ao;
layout(binding = 1) uniform sampler2D u_linear_depth;
layout(binding = 2) uniform sampler2D u_view_normal;

uniform vec2 u_resolution;
uniform vec2 u_filter_axis;
uniform int u_kernel_radius;
uniform float u_radius_world;
uniform float u_depth_sigma_fraction;
uniform float u_normal_power;

in vec2 v_uv;
layout(location = 0) out vec4 out_ao_bent_normal;

void main() {
    float center_depth = texture(u_linear_depth, v_uv).r;
    vec3 center_normal = texture(u_view_normal, v_uv).xyz;
    if (center_depth <= 0.0 || dot(center_normal, center_normal) <= 1.0e-8) {
        out_ao_bent_normal = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }
    center_normal = normalize(center_normal);
    vec2 texel = u_filter_axis / max(u_resolution, vec2(1.0));
    float depth_sigma = max(
        u_radius_world * u_depth_sigma_fraction,
        1.0e-5);
    float spatial_sigma = max(float(u_kernel_radius) * 0.75, 0.5);
    vec3 bent_sum = vec3(0.0);
    float visibility_sum = 0.0;
    float weight_sum = 0.0;
    for (int offset = -4; offset <= 4; ++offset) {
        if (abs(offset) > u_kernel_radius) {
            continue;
        }
        vec2 sample_uv = clamp(
            v_uv + texel * float(offset),
            vec2(0.0),
            vec2(1.0));
        float sample_depth = texture(u_linear_depth, sample_uv).r;
        vec3 sample_normal = texture(u_view_normal, sample_uv).xyz;
        if (sample_depth <= 0.0 || dot(sample_normal, sample_normal) <= 1.0e-8) {
            continue;
        }
        sample_normal = normalize(sample_normal);
        vec4 sample_ao = texture(u_source_ao, sample_uv);
        float spatial_weight = exp(
            -0.5 * float(offset * offset) /
            (spatial_sigma * spatial_sigma));
        float depth_weight = exp(
            -abs(sample_depth - center_depth) / depth_sigma);
        float normal_weight = pow(
            max(dot(sample_normal, center_normal), 0.0),
            u_normal_power);
        float weight = spatial_weight * depth_weight * normal_weight;
        bent_sum += sample_ao.xyz * weight;
        visibility_sum += sample_ao.w * weight;
        weight_sum += weight;
    }
    if (weight_sum <= 1.0e-8) {
        out_ao_bent_normal = texture(u_source_ao, v_uv);
        return;
    }
    vec3 bent_normal = bent_sum / weight_sum;
    bent_normal = dot(bent_normal, bent_normal) > 1.0e-8
        ? normalize(bent_normal)
        : center_normal;
    out_ao_bent_normal = vec4(
        bent_normal,
        clamp(visibility_sum / weight_sum, 0.0, 1.0));
}
