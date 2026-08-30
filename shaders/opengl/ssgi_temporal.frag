#version 450 core

layout(binding = 0) uniform sampler2D u_raw_indirect;
layout(binding = 1) uniform sampler2D u_linear_depth;
layout(binding = 2) uniform sampler2D u_view_normal;
layout(binding = 3) uniform sampler2D u_history_indirect;
layout(binding = 4) uniform sampler2D u_history_moments;
layout(binding = 5) uniform sampler2D u_history_depth;
layout(binding = 6) uniform sampler2D u_history_normal;

uniform ivec2 u_full_resolution;
uniform ivec2 u_half_resolution;
uniform vec2 u_camera_viewport;
uniform vec3 u_camera_position;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec2 u_previous_camera_viewport;
uniform vec3 u_previous_camera_position;
uniform vec3 u_previous_camera_forward;
uniform vec3 u_previous_camera_right;
uniform vec3 u_previous_camera_up;
uniform float u_thickness;
uniform int u_max_history_frames;
uniform int u_history_valid;

in vec2 v_uv;
layout(location = 0) out vec4 out_history_indirect;
layout(location = 1) out vec2 out_history_moments;
layout(location = 2) out float out_history_depth;
layout(location = 3) out vec3 out_history_normal;

struct SurfaceSample {
    ivec2 pixel;
    vec2 uv;
    float depth;
    vec3 normal;
};

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 rgb_to_ycocg(vec3 color) {
    return vec3(
        dot(color, vec3(0.25, 0.5, 0.25)),
        dot(color, vec3(0.5, 0.0, -0.5)),
        dot(color, vec3(-0.25, 0.5, -0.25)));
}

vec3 ycocg_to_rgb(vec3 value) {
    return vec3(
        value.x + value.y - value.z,
        value.x + value.z,
        value.x - value.y - value.z);
}

SurfaceSample representative_surface(ivec2 half_pixel) {
    SurfaceSample result;
    result.pixel = clamp(half_pixel * 2, ivec2(0), u_full_resolution - 1);
    result.depth = 0.0;
    result.normal = vec3(0.0);
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
                result.pixel = pixel;
                result.depth = depth;
                result.normal = normalize(normal);
            }
        }
    }
    result.uv = (vec2(result.pixel) + 0.5) / vec2(u_full_resolution);
    return result;
}

vec3 reconstruct_world_position(SurfaceSample surface) {
    vec2 ndc = surface.uv * 2.0 - 1.0;
    vec3 view_position = vec3(
        ndc.x * 0.5 * u_camera_viewport.x * surface.depth,
        ndc.y * 0.5 * u_camera_viewport.y * surface.depth,
        -surface.depth);
    return u_camera_position +
        u_camera_right * view_position.x +
        u_camera_up * view_position.y -
        u_camera_forward * view_position.z;
}

vec2 project_previous(vec3 world_position, out float predicted_depth) {
    vec3 relative = world_position - u_previous_camera_position;
    vec3 previous_view = vec3(
        dot(relative, u_previous_camera_right),
        dot(relative, u_previous_camera_up),
        -dot(relative, u_previous_camera_forward));
    predicted_depth = -previous_view.z;
    vec2 ndc = vec2(
        2.0 * previous_view.x /
            max(u_previous_camera_viewport.x * predicted_depth, 1.0e-6),
        2.0 * previous_view.y /
            max(u_previous_camera_viewport.y * predicted_depth, 1.0e-6));
    return ndc * 0.5 + 0.5;
}

bool valid_history_sample(
    ivec2 pixel,
    float predicted_depth,
    vec3 current_world_normal) {
    if (any(lessThan(pixel, ivec2(0))) ||
        any(greaterThanEqual(pixel, u_half_resolution))) {
        return false;
    }
    float depth = texelFetch(u_history_depth, pixel, 0).r;
    vec3 previous_normal = texelFetch(u_history_normal, pixel, 0).xyz;
    if (depth <= 0.0 || dot(previous_normal, previous_normal) <= 1.0e-8) {
        return false;
    }
    float depth_tolerance = max(
        2.0 * u_thickness,
        0.01 * predicted_depth);
    if (abs(depth - predicted_depth) > depth_tolerance) {
        return false;
    }
    vec3 previous_world_normal = normalize(
        u_previous_camera_right * previous_normal.x +
        u_previous_camera_up * previous_normal.y -
        u_previous_camera_forward * previous_normal.z);
    return dot(current_world_normal, previous_world_normal) >= 0.85;
}

void current_neighborhood_statistics(
    ivec2 center,
    out vec3 mean_value,
    out vec3 sigma_value) {
    vec3 value_sum = vec3(0.0);
    vec3 square_sum = vec3(0.0);
    float sample_count = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 pixel = clamp(
                center + ivec2(x, y),
                ivec2(0),
                u_half_resolution - 1);
            vec3 value = rgb_to_ycocg(
                texelFetch(u_raw_indirect, pixel, 0).rgb);
            value_sum += value;
            square_sum += value * value;
            sample_count += 1.0;
        }
    }
    mean_value = value_sum / max(sample_count, 1.0);
    sigma_value = sqrt(max(
        square_sum / max(sample_count, 1.0) - mean_value * mean_value,
        vec3(0.0)));
}

void main() {
    ivec2 half_pixel = clamp(
        ivec2(gl_FragCoord.xy),
        ivec2(0),
        u_half_resolution - 1);
    SurfaceSample surface = representative_surface(half_pixel);
    vec3 current = texelFetch(u_raw_indirect, half_pixel, 0).rgb;
    // RG16F moments must remain finite even for very bright HDR emitters.
    float current_luminance = clamp(luminance(current), -255.0, 255.0);
    if (surface.depth <= 0.0 || dot(surface.normal, surface.normal) <= 1.0e-8) {
        out_history_indirect = vec4(current, 0.0);
        out_history_moments = vec2(
            current_luminance,
            current_luminance * current_luminance);
        out_history_depth = 0.0;
        out_history_normal = vec3(0.0);
        return;
    }

    out_history_depth = surface.depth;
    out_history_normal = surface.normal;
    if (u_history_valid == 0) {
        out_history_indirect = vec4(current, 1.0);
        out_history_moments = vec2(
            current_luminance,
            current_luminance * current_luminance);
        return;
    }

    vec3 world_position = reconstruct_world_position(surface);
    float predicted_depth = 0.0;
    vec2 previous_uv = project_previous(world_position, predicted_depth);
    if (predicted_depth <= 0.0 ||
        any(lessThan(previous_uv, vec2(0.0))) ||
        any(greaterThanEqual(previous_uv, vec2(1.0)))) {
        out_history_indirect = vec4(current, 1.0);
        out_history_moments = vec2(
            current_luminance,
            current_luminance * current_luminance);
        return;
    }

    vec3 current_world_normal = normalize(
        u_camera_right * surface.normal.x +
        u_camera_up * surface.normal.y -
        u_camera_forward * surface.normal.z);
    vec2 history_position = previous_uv * vec2(u_half_resolution) - 0.5;
    ivec2 base = ivec2(floor(history_position));
    vec2 fraction = fract(history_position);
    vec3 history_sum = vec3(0.0);
    vec2 moment_sum = vec2(0.0);
    float length_sum = 0.0;
    float valid_weight_sum = 0.0;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            ivec2 pixel = base + ivec2(x, y);
            vec2 axis_weight = mix(
                vec2(1.0) - fraction,
                fraction,
                vec2(x, y));
            float weight = axis_weight.x * axis_weight.y;
            if (weight <= 0.0 || !valid_history_sample(
                    pixel,
                    predicted_depth,
                    current_world_normal)) {
                continue;
            }
            vec4 history = texelFetch(u_history_indirect, pixel, 0);
            history_sum += history.rgb * weight;
            length_sum += history.a * weight;
            moment_sum += texelFetch(u_history_moments, pixel, 0).rg * weight;
            valid_weight_sum += weight;
        }
    }
    if (valid_weight_sum <= 1.0e-6) {
        out_history_indirect = vec4(current, 1.0);
        out_history_moments = vec2(
            current_luminance,
            current_luminance * current_luminance);
        return;
    }

    vec3 history = history_sum / valid_weight_sum;
    vec2 moments = moment_sum / valid_weight_sum;
    float history_length = length_sum / valid_weight_sum;
    float variance = max(moments.y - moments.x * moments.x, 0.0);
    float history_sigma = sqrt(variance);
    vec3 neighborhood_mean;
    vec3 neighborhood_sigma;
    current_neighborhood_statistics(
        half_pixel, neighborhood_mean, neighborhood_sigma);
    vec3 sigma = max(
        neighborhood_sigma,
        vec3(history_sigma, 0.5 * history_sigma, 0.5 * history_sigma));
    vec3 clip_extension = 1.5 * sigma + vec3(1.0e-4);
    vec3 clipped_history_ycocg = clamp(
        rgb_to_ycocg(history),
        neighborhood_mean - clip_extension,
        neighborhood_mean + clip_extension);
    history = ycocg_to_rgb(clipped_history_ycocg);

    float maximum_weight = 1.0 -
        1.0 / float(max(u_max_history_frames, 1));
    float history_weight = min(
        history_length / (history_length + 1.0),
        maximum_weight);
    vec3 accumulated = mix(current, history, history_weight);
    vec2 current_moments = vec2(
        current_luminance,
        current_luminance * current_luminance);
    out_history_indirect = vec4(
        accumulated,
        min(history_length + 1.0, float(max(u_max_history_frames, 1))));
    out_history_moments = mix(
        current_moments, moments, history_weight);
}
