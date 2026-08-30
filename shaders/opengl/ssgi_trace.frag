#version 450 core

const float PI = 3.14159265358979323846;
// Additive R2 recurrence.  Unlike a per-frame hash this keeps consecutive
// samples well separated, which is important when only two rays are traced.
const vec2 R2_SEQUENCE = vec2(
    0.75487766624669276005,
    0.56984029099805326591);

layout(binding = 0) uniform sampler2D u_opaque;
layout(binding = 1) uniform sampler2D u_linear_depth;
layout(binding = 2) uniform sampler2D u_view_normal;
layout(binding = 3) uniform sampler2D u_ssgi_material;
layout(binding = 4) uniform sampler2D u_hiz;
layout(binding = 5) uniform sampler2D u_ambient_occlusion;
layout(binding = 6) uniform samplerCube u_environment_radiance;

uniform ivec2 u_full_resolution;
uniform ivec2 u_half_resolution;
uniform vec2 u_camera_viewport;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform int u_rays_per_pixel;
uniform int u_max_steps;
uniform int u_refinement_steps;
uniform float u_max_distance;
uniform float u_thickness;
uniform float u_edge_fade;
uniform int u_frame_index;
uniform int u_hiz_levels;
uniform int u_ao_enabled;
uniform vec3 u_environment_color;
uniform float u_environment_intensity;
uniform float u_environment_rotation_radians;
uniform int u_has_environment_map;
uniform int u_ibl_enabled;

in vec2 v_uv;
layout(location = 0) out vec4 out_raw_indirect;

struct SurfaceSample {
    ivec2 pixel;
    vec2 uv;
    float depth;
    vec3 normal;
    vec4 material;
};

float hash12(vec2 value) {
    vec3 p3 = fract(vec3(value.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 rotate_y(vec3 direction, float radians) {
    float c = cos(radians);
    float s = sin(radians);
    return vec3(
        c * direction.x + s * direction.z,
        direction.y,
        -s * direction.x + c * direction.z);
}

vec3 environment_radiance(vec3 view_direction) {
    if (u_ibl_enabled == 0) {
        return vec3(0.0);
    }
    mat3 view_to_world = mat3(
        u_camera_right,
        u_camera_up,
        -u_camera_forward);
    vec3 world_direction = normalize(view_to_world * view_direction);
    vec3 local_direction = rotate_y(
        world_direction,
        -u_environment_rotation_radians);
    vec3 radiance = u_has_environment_map != 0
        ? textureLod(u_environment_radiance, local_direction, 0.0).rgb
        : vec3(1.0);
    return max(
        radiance * u_environment_color * u_environment_intensity,
        vec3(0.0));
}

bool inside_screen(vec2 uv) {
    return all(greaterThanEqual(uv, vec2(0.0))) &&
        all(lessThan(uv, vec2(1.0)));
}

vec3 reconstruct_view_position(vec2 uv, float linear_depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(
        ndc.x * 0.5 * u_camera_viewport.x * linear_depth,
        ndc.y * 0.5 * u_camera_viewport.y * linear_depth,
        -linear_depth);
}

vec2 project_view_position(vec3 position) {
    float depth = max(-position.z, 1.0e-6);
    vec2 ndc = vec2(
        2.0 * position.x / (u_camera_viewport.x * depth),
        2.0 * position.y / (u_camera_viewport.y * depth));
    return ndc * 0.5 + 0.5;
}

SurfaceSample representative_surface(ivec2 half_pixel) {
    SurfaceSample result;
    result.pixel = clamp(half_pixel * 2, ivec2(0), u_full_resolution - 1);
    result.depth = 0.0;
    result.normal = vec3(0.0);
    result.material = vec4(0.0);
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
                result.material = texelFetch(u_ssgi_material, pixel, 0);
            }
        }
    }
    result.uv = (vec2(result.pixel) + 0.5) / vec2(u_full_resolution);
    return result;
}

vec3 cosine_hemisphere(vec2 xi) {
    float radius = sqrt(clamp(xi.x, 0.0, 1.0));
    float phi = 2.0 * PI * xi.y;
    return vec3(
        radius * cos(phi),
        radius * sin(phi),
        sqrt(max(0.0, 1.0 - xi.x)));
}

vec3 sample_direction(vec3 normal, ivec2 pixel, int ray_index) {
    vec2 pixel_rotation = vec2(
        hash12(vec2(pixel) + vec2(11.17, 47.31)),
        hash12(vec2(pixel.yx) + vec2(73.73, 29.41)));
    float sequence_index =
        float(u_frame_index * max(u_rays_per_pixel, 1) + ray_index) + 0.5;
    vec2 xi = fract(pixel_rotation + sequence_index * R2_SEQUENCE);
    vec3 local = cosine_hemisphere(xi);
    vec3 helper = abs(normal.z) < 0.999
        ? vec3(0.0, 0.0, 1.0)
        : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(helper, normal));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * local.x + bitangent * local.y +
        normal * local.z);
}

float screen_edge_confidence(vec2 uv) {
    float distance_to_edge = 1.0 - max(
        abs(uv.x * 2.0 - 1.0),
        abs(uv.y * 2.0 - 1.0));
    float width = clamp(u_edge_fade, 0.0, 0.5);
    return width <= 0.0
        ? 1.0
        : smoothstep(0.0, width, distance_to_edge);
}

float projected_derivative(vec3 position, vec3 direction, bool horizontal) {
    float coordinate = horizontal ? position.x : position.y;
    float velocity = horizontal ? direction.x : direction.y;
    return velocity * (-position.z) + coordinate * direction.z;
}

float boundary_ray_distance(
    vec3 origin,
    vec3 direction,
    float boundary_uv,
    bool horizontal) {
    float viewport = horizontal
        ? u_camera_viewport.x
        : u_camera_viewport.y;
    float coordinate = horizontal ? origin.x : origin.y;
    float velocity = horizontal ? direction.x : direction.y;
    float boundary_ndc = boundary_uv * 2.0 - 1.0;
    float denominator = velocity +
        0.5 * viewport * boundary_ndc * direction.z;
    if (abs(denominator) <= 1.0e-8) {
        return 3.402823466e+38;
    }
    return (-0.5 * viewport * boundary_ndc * origin.z - coordinate) /
        denominator;
}

float cell_exit_distance(
    vec3 origin,
    vec3 direction,
    float current_distance,
    vec3 position,
    vec2 uv,
    int level) {
    ivec2 level_size = textureSize(u_hiz, level);
    ivec2 cell = clamp(
        ivec2(floor(uv * vec2(level_size))),
        ivec2(0),
        level_size - 1);
    float x_sign = projected_derivative(position, direction, true);
    float y_sign = projected_derivative(position, direction, false);
    float boundary_x = float(cell.x + (x_sign >= 0.0 ? 1 : 0)) /
        float(level_size.x);
    float boundary_y = float(cell.y + (y_sign >= 0.0 ? 1 : 0)) /
        float(level_size.y);
    float tx = boundary_ray_distance(
        origin, direction, boundary_x, true);
    float ty = boundary_ray_distance(
        origin, direction, boundary_y, false);
    float exit_distance = 3.402823466e+38;
    if (tx > current_distance + 1.0e-6) {
        exit_distance = min(exit_distance, tx);
    }
    if (ty > current_distance + 1.0e-6) {
        exit_distance = min(exit_distance, ty);
    }
    return exit_distance;
}

bool refine_crossing(
    vec3 origin,
    vec3 direction,
    float near_distance,
    float far_distance,
    ivec2 origin_pixel,
    out vec2 hit_uv,
    out float confidence) {
    float refined_delta = 3.402823466e+38;
    hit_uv = vec2(0.0);
    for (int refine = 0; refine < 16; ++refine) {
        if (refine >= u_refinement_steps) {
            break;
        }
        float middle_distance = 0.5 * (near_distance + far_distance);
        vec3 middle_position = origin + direction * middle_distance;
        if (middle_position.z >= -1.0e-5) {
            far_distance = middle_distance;
            continue;
        }
        vec2 middle_uv = project_view_position(middle_position);
        if (!inside_screen(middle_uv)) {
            near_distance = middle_distance;
            continue;
        }
        float surface_depth = texture(u_linear_depth, middle_uv).r;
        float delta = -middle_position.z - surface_depth;
        if (surface_depth > 0.0 && delta >= 0.0) {
            far_distance = middle_distance;
            hit_uv = middle_uv;
            refined_delta = delta;
        } else {
            near_distance = middle_distance;
        }
    }
    if (u_refinement_steps == 0) {
        vec3 far_position = origin + direction * far_distance;
        hit_uv = project_view_position(far_position);
        float surface_depth = texture(u_linear_depth, hit_uv).r;
        refined_delta = -far_position.z - surface_depth;
    }
    if (!inside_screen(hit_uv) || refined_delta < 0.0 ||
        refined_delta > u_thickness || far_distance <= 2.0 * u_thickness) {
        return false;
    }
    ivec2 hit_pixel = clamp(
        ivec2(hit_uv * vec2(u_full_resolution)),
        ivec2(0),
        u_full_resolution - 1);
    if (all(equal(hit_pixel, origin_pixel))) {
        return false;
    }
    vec3 hit_normal = texelFetch(u_view_normal, hit_pixel, 0).xyz;
    if (dot(hit_normal, hit_normal) <= 1.0e-8 ||
        dot(normalize(hit_normal), -direction) <= 0.0) {
        return false;
    }
    confidence = screen_edge_confidence(hit_uv);
    return confidence > 0.0;
}

bool march_hiz(
    vec3 origin,
    vec3 direction,
    ivec2 origin_pixel,
    out vec2 hit_uv,
    out float confidence) {
    float distance_value = max(2.0 * u_thickness, 1.0e-4);
    int level = 0;
    hit_uv = vec2(0.0);
    confidence = 0.0;
    for (int visit = 0; visit < 256; ++visit) {
        if (visit >= u_max_steps || distance_value >= u_max_distance) {
            break;
        }
        vec3 position = origin + direction * distance_value;
        if (position.z >= -1.0e-5) {
            break;
        }
        vec2 uv = project_view_position(position);
        if (!inside_screen(uv)) {
            break;
        }
        ivec2 level_size = textureSize(u_hiz, level);
        ivec2 cell = clamp(
            ivec2(floor(uv * vec2(level_size))),
            ivec2(0),
            level_size - 1);
        vec2 depth_range = texelFetch(u_hiz, cell, level).rg;
        float exit_distance = cell_exit_distance(
            origin, direction, distance_value, position, uv, level);
        if (exit_distance >= 3.0e+38) {
            break;
        }
        exit_distance = min(
            max(exit_distance + max(u_thickness * 0.01, 1.0e-5),
                distance_value + 1.0e-5),
            u_max_distance);
        vec3 exit_position = origin + direction * exit_distance;
        float segment_near = min(-position.z, -exit_position.z);
        float segment_far = max(-position.z, -exit_position.z);
        bool valid_cell = depth_range.x <= depth_range.y &&
            depth_range.y > 0.0;
        bool in_front = valid_cell &&
            segment_far < depth_range.x - u_thickness;
        bool overlaps = valid_cell &&
            segment_far >= depth_range.x - u_thickness &&
            segment_near <= depth_range.y + u_thickness;
        if (!valid_cell || in_front) {
            distance_value = exit_distance;
            level = min(level + 1, max(u_hiz_levels - 1, 0));
            continue;
        }
        if (overlaps && level > 0) {
            --level;
            continue;
        }
        if (level == 0) {
            float start_surface = texture(u_linear_depth, uv).r;
            vec2 exit_uv = project_view_position(exit_position);
            if (!inside_screen(exit_uv)) {
                break;
            }
            float end_surface = texture(u_linear_depth, exit_uv).r;
            float start_delta = -position.z - start_surface;
            float end_delta = -exit_position.z - end_surface;
            if (start_surface > 0.0 && end_surface > 0.0 &&
                start_delta < 0.0 && end_delta >= 0.0 &&
                refine_crossing(
                    origin,
                    direction,
                    distance_value,
                    exit_distance,
                    origin_pixel,
                    hit_uv,
                    confidence)) {
                return true;
            }
        }
        distance_value = exit_distance;
        level = min(level + 1, max(u_hiz_levels - 1, 0));
    }
    return false;
}

vec3 stable_hit_radiance(vec2 hit_uv) {
    // A single sub-pixel HDR highlight otherwise becomes a long-lived
    // firefly after temporal accumulation.  Clamp only isolated highlights;
    // coherent bright regions retain their energy through their neighbours.
    ivec2 size = textureSize(u_opaque, 0);
    ivec2 center = clamp(
        ivec2(hit_uv * vec2(size)),
        ivec2(0),
        size - 1);
    vec3 center_color = max(textureLod(u_opaque, hit_uv, 0.0).rgb, vec3(0.0));
    float center_luminance = luminance(center_color);
    float luminance_sum = 0.0;
    float maximum_luminance = 0.0;
    int sample_count = 0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 pixel = clamp(center + ivec2(x, y), ivec2(0), size - 1);
            float sample_luminance = luminance(max(
                texelFetch(u_opaque, pixel, 0).rgb,
                vec3(0.0)));
            luminance_sum += sample_luminance;
            maximum_luminance = max(maximum_luminance, sample_luminance);
            ++sample_count;
        }
    }
    float trimmed_mean = (luminance_sum - maximum_luminance) /
        float(max(sample_count - 1, 1));
    float luminance_limit = max(1.0, 8.0 * trimmed_mean);
    float scale = center_luminance > luminance_limit
        ? luminance_limit / max(center_luminance, 1.0e-6)
        : 1.0;
    return center_color * scale;
}

void main() {
    ivec2 half_pixel = clamp(
        ivec2(gl_FragCoord.xy),
        ivec2(0),
        u_half_resolution - 1);
    SurfaceSample surface = representative_surface(half_pixel);
    if (surface.depth <= 0.0 || dot(surface.normal, surface.normal) <= 1.0e-8 ||
        dot(surface.material.rgb, surface.material.rgb) <= 1.0e-12) {
        out_raw_indirect = vec4(0.0);
        return;
    }

    vec3 origin = reconstruct_view_position(surface.uv, surface.depth);
    float screen_ao = u_ao_enabled != 0
        ? clamp(texture(u_ambient_occlusion, surface.uv).a, 0.0, 1.0)
        : 1.0;
    vec3 receiver_response = surface.material.rgb *
        (clamp(surface.material.a, 0.0, 1.0) * screen_ao);
    vec3 residual_sum = vec3(0.0);
    float confidence_sum = 0.0;
    for (int ray_index = 0; ray_index < 8; ++ray_index) {
        if (ray_index >= u_rays_per_pixel) {
            break;
        }
        vec3 direction = sample_direction(
            surface.normal, surface.pixel, ray_index);
        vec2 hit_uv;
        float confidence;
        vec3 radiance_residual = vec3(0.0);
        if (march_hiz(
                origin,
                direction,
                surface.pixel,
                hit_uv,
                confidence)) {
            vec3 hit_indirect = stable_hit_radiance(hit_uv) *
                receiver_response;
            // The sharp full-resolution raster IBL bypasses this texture.
            // Emit only a signed control-variate residual: subtract the
            // matching environment direction and add visible screen radiance.
            // Filtering this residual cannot blur the raster material detail.
            vec3 environment_indirect = environment_radiance(direction) *
                receiver_response;
            radiance_residual = confidence *
                (hit_indirect - environment_indirect);
            confidence_sum += confidence;
        }
        residual_sum += radiance_residual;
    }
    float ray_count = float(max(u_rays_per_pixel, 1));
    out_raw_indirect = vec4(
        residual_sum / ray_count,
        clamp(confidence_sum / ray_count, 0.0, 1.0));
}
