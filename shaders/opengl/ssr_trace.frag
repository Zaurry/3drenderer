#version 450 core

const float PI = 3.14159265358979323846;
// Additive R2 recurrence.  Unlike a per-frame hash this keeps consecutive
// samples well separated, which is important when only two rays are traced.
const vec2 R2_SEQUENCE = vec2(
    0.75487766624669276005,
    0.56984029099805326591);

layout(binding = 0) uniform sampler2D u_ray_radiance;
layout(binding = 1) uniform sampler2D u_linear_depth;
layout(binding = 2) uniform sampler2D u_view_normal;
layout(binding = 3) uniform sampler2D u_ssr_material;
layout(binding = 4) uniform sampler2D u_hiz;
layout(binding = 5) uniform sampler2D u_ambient_occlusion;
layout(binding = 6) uniform samplerCube u_environment_radiance;
layout(binding = 7) uniform sampler2D u_ssr_pbr;
layout(binding = 8) uniform sampler2D u_ssr_pbr_aux;
layout(binding = 9) uniform sampler2D u_diffuse_fresnel;
layout(binding = 10) uniform sampler2D u_environment_brdf_lut;

uniform ivec2 u_full_resolution;
uniform ivec2 u_trace_resolution;
uniform vec2 u_camera_viewport;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform int u_rays_per_pixel;
uniform int u_max_steps;
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

SurfaceSample representative_surface(ivec2 pixel) {
    SurfaceSample result;
    result.pixel = pixel;
    result.uv = (vec2(pixel) + 0.5) / vec2(u_full_resolution);
    result.depth = texelFetch(u_linear_depth, pixel, 0).r;
    result.normal = texelFetch(u_view_normal, pixel, 0).xyz;
    result.material = texelFetch(u_ssr_material, pixel, 0);
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

mat3 tangent_frame(vec3 normal) {
    vec3 helper = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0);
    vec3 tangent = normalize(cross(helper, normal));
    return mat3(tangent, cross(normal, tangent), normal);
}

// Heitz 2018: https://jcgt.org/published/0007/04/01/
vec3 sample_ggx_vndf(vec3 view, float alpha, vec2 xi) {
    vec3 stretched = normalize(vec3(alpha * view.xy, view.z));
    float lensq = dot(stretched.xy, stretched.xy);
    vec3 t1 = lensq > 0.0
        ? vec3(-stretched.y, stretched.x, 0.0) / sqrt(lensq)
        : vec3(1, 0, 0);
    vec3 t2 = cross(stretched, t1);
    vec2 disk = sqrt(xi.x) * vec2(cos(2.0 * PI * xi.y), sin(2.0 * PI * xi.y));
    float blend = 0.5 * (1.0 + stretched.z);
    disk.y = mix(sqrt(max(0.0, 1.0 - disk.x * disk.x)), disk.y, blend);
    vec3 nh = t1 * disk.x + t2 * disk.y + stretched *
        sqrt(max(0.0, 1.0 - dot(disk, disk)));
    return normalize(vec3(alpha * nh.xy, max(nh.z, 0.0)));
}

float smith_lambda(float cosine, float alpha) {
    float cosine2 = max(cosine * cosine, 1.0e-8);
    return 0.5 * (sqrt(1.0 + alpha * alpha * (1.0 - cosine2) / cosine2) - 1.0);
}

vec3 fresnel_schlick(float cosine, vec3 f0, vec3 f90) {
    return f0 + (f90 - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
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

bool march_hiz(
    vec3 origin,
    vec3 direction,
    ivec2 origin_pixel,
    out vec2 hit_uv,
    out float confidence) {
    float distance_value = 1.0e-5;
    float limit = u_max_distance;
    if (direction.z > 0.0) {
        limit = min(limit, (-1.0e-5 - origin.z) / direction.z);
    }
    int level = 0;
    hit_uv = vec2(0.0);
    confidence = 0.0;
    for (int visit = 0; visit < 256; ++visit) {
        if (visit >= u_max_steps || distance_value >= limit) break;
        vec3 position = origin + direction * distance_value;
        vec2 uv = project_view_position(position);
        if (!inside_screen(uv)) break;
        ivec2 level_size = textureSize(u_hiz, level);
        ivec2 cell = clamp(ivec2(uv * vec2(level_size)), ivec2(0), level_size - 1);
        vec2 depth_range = texelFetch(u_hiz, cell, level).rg;
        float exit_distance = min(limit, cell_exit_distance(
            origin, direction, distance_value, position, uv, level));
        float step_epsilon = max(1.0e-6, abs(exit_distance) * 1.0e-6);
        float exit_depth = -(origin + direction * exit_distance).z;
        bool overlaps = depth_range.x <= depth_range.y && depth_range.y > 0.0 &&
            max(-position.z, exit_depth) >= depth_range.x &&
            min(-position.z, exit_depth) <= depth_range.y + u_thickness;
        if (overlaps && level > 0) {
            --level;
            continue;
        }
        if (overlaps && level == 0 && any(notEqual(cell, origin_pixel))) {
            float entry = distance_value;
            if (abs(direction.z) > 1.0e-8) {
                float t0 = (-depth_range.x - origin.z) / direction.z;
                float t1 = (-depth_range.y - u_thickness - origin.z) / direction.z;
                entry = max(entry, min(t0, t1));
            }
            if (entry <= exit_distance) {
                hit_uv = project_view_position(origin + direction * entry);
                // Stay in the cell whose depth slab was intersected.
                hit_uv = clamp(hit_uv, (vec2(cell) + 0.001) / vec2(level_size),
                    (vec2(cell) + 0.999) / vec2(level_size));
                confidence = screen_edge_confidence(hit_uv);
                return true;
            }
        }
        distance_value = exit_distance + step_epsilon;
        level = min(level + 1, max(u_hiz_levels - 1, 0));
    }
    return false;
}

void main() {
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), u_trace_resolution - 1);
    SurfaceSample surface = representative_surface(pixel);
    vec4 pbr = texelFetch(u_ssr_pbr, pixel, 0);
    vec4 aux = texelFetch(u_ssr_pbr_aux, pixel, 0);
    vec4 diffuse_fresnel = texelFetch(u_diffuse_fresnel, pixel, 0);
    if (surface.depth <= 0.0 || dot(surface.normal, surface.normal) <= 1.0e-8 ||
        max(luminance(surface.material.rgb), luminance(aux.rgb)) <= 0.0) {
        out_raw_indirect = vec4(0.0);
        return;
    }
    surface.normal = normalize(surface.normal);
    vec3 position = reconstruct_view_position(surface.uv, surface.depth);
    mat3 frame = tangent_frame(surface.normal);
    vec3 view = transpose(frame) * normalize(-position);
    if (view.z <= 0.0) {
        out_raw_indirect = vec4(0.0);
        return;
    }
    float roughness = clamp(pbr.a, 0.02, 1.0);
    float alpha = roughness * roughness;
    float lambda_v = smith_lambda(view.z, alpha);
    vec4 energy = texture(u_environment_brdf_lut, vec2(view.z, roughness));
    vec3 average_fresnel = pbr.rgb + (aux.rgb - pbr.rgb) / 21.0;
    float missing_average = max(1.0 - energy.a, 0.0);
    vec3 multiscatter_scale = missing_average > 1.0e-6
        ? (1.0 - energy.b) * average_fresnel * average_fresnel * energy.a /
            (PI * missing_average * max(vec3(1.0) - average_fresnel *
                missing_average, vec3(1.0e-6)))
        : vec3(0.0);
    float diffuse_energy = luminance(surface.material.rgb +
        PI * missing_average * multiscatter_scale);
    float specular_energy = luminance(pbr.rgb * energy.r + aux.rgb * energy.g);
    float specular_probability = specular_energy <= 0.0 ? 0.0 :
        (diffuse_energy <= 0.0 ? 1.0 :
            specular_energy / (specular_energy + diffuse_energy));
    float fallback_visibility = clamp(aux.a, 0.0, 1.0) *
        (u_ao_enabled != 0 ? clamp(texelFetch(u_ambient_occlusion, pixel, 0).a, 0.0, 1.0) : 1.0);
    vec2 rotation = vec2(hash12(vec2(pixel) + vec2(11.17, 47.31)),
        hash12(vec2(pixel.yx) + vec2(73.73, 29.41)));
    float selection_rotation = hash12(vec2(pixel) + vec2(97.13, 17.71));
    float bias = max(1.0e-5, min(u_thickness * 0.05, surface.depth * 0.0001));
    vec3 origin = position + surface.normal * bias;
    vec3 radiance_sum = vec3(0.0);
    float hit_sum = 0.0;
    for (int ray_index = 0; ray_index < 8; ++ray_index) {
        if (ray_index >= u_rays_per_pixel) break;
        // Bound the recurrence index so long sessions retain fractional bits
        // in the float R2 samples instead of collapsing to a few directions.
        int sequence_index = (u_frame_index * max(u_rays_per_pixel, 1) + ray_index) % 4096;
        float sequence = float(sequence_index) + 0.5;
        vec2 xi = fract(rotation + sequence * R2_SEQUENCE);
        float selection = fract(selection_rotation + sequence * 0.414213562373095);
        vec3 light = selection < specular_probability
            ? reflect(-view, sample_ggx_vndf(view, alpha, xi))
            : cosine_hemisphere(xi);
        // A VNDF sample below the receiver hemisphere contributes zero;
        // resampling it would change the PDF and add energy.
        if (light.z <= 0.0) continue;
        vec3 half_vector = normalize(view + light);
        float alpha2 = alpha * alpha;
        // Stable at low roughness: avoid cancellation in the GGX denominator.
        float denominator = dot(half_vector.xy, half_vector.xy) +
            alpha2 * half_vector.z * half_vector.z;
        float distribution = alpha2 / max(PI * denominator * denominator, 1.0e-20);
        float geometry = 1.0 / (1.0 + lambda_v + smith_lambda(light.z, alpha));
        float specular_pdf = distribution / (4.0 * view.z * (1.0 + lambda_v));
        float pdf = mix(light.z / PI, specular_pdf, specular_probability);
        vec3 f = fresnel_schlick(dot(view, half_vector), pbr.rgb, aux.rgb);
        vec3 fd = fresnel_schlick(dot(view, half_vector), diffuse_fresnel.rgb,
            vec3(diffuse_fresnel.a));
        vec3 diffuse_weight = (int(surface.material.a + 0.5) & 1) != 0
            ? vec3(1.0 - max(max(fd.r, fd.g), fd.b)) : vec3(1.0) - fd;
        float energy_l = texture(u_environment_brdf_lut, vec2(light.z, roughness)).b;
        vec3 brdf = diffuse_weight * surface.material.rgb / PI +
            f * distribution * geometry / max(4.0 * view.z * light.z, 1.0e-8) +
            multiscatter_scale * (1.0 - energy_l);
        vec3 direction = normalize(frame * light);
        vec2 hit_uv;
        float confidence;
        vec3 incoming;
        if (march_hiz(origin, direction, pixel, hit_uv, confidence)) {
            // Visibility is binary even when radiance support fades at an
            // edge or the blocker faces away. Neither case exposes the sky.
            vec3 hit_normal = texture(u_view_normal, hit_uv).xyz;
            bool two_sided = (int(texture(u_ssr_material, hit_uv).a + 0.5) & 2) != 0;
            incoming = dot(hit_normal, -direction) > 0.0 || two_sided
                ? max(textureLod(u_ray_radiance, hit_uv, 0.0).rgb, vec3(0.0)) * confidence
                : vec3(0.0);
            hit_sum += 1.0;
        } else {
            // AO is a conservative approximation for unavailable geometry,
            // never an extra multiplier on a traced surface connection.
            incoming = environment_radiance(direction) * fallback_visibility;
        }
        radiance_sum += incoming * brdf * (light.z / max(pdf, 1.0e-20));
    }
    float count = float(max(u_rays_per_pixel, 1));
    out_raw_indirect = vec4(max(radiance_sum / count, vec3(0.0)), hit_sum / count);
}
