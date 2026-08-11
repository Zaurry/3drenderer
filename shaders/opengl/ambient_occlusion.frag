#version 450 core

const float PI = 3.14159265358979323846;
const float GOLDEN_ANGLE = 2.39996322972865332;

layout(binding = 0) uniform sampler2D u_linear_depth;
layout(binding = 1) uniform sampler2D u_view_normal;

uniform vec2 u_resolution;
uniform vec2 u_camera_viewport;
uniform int u_ao_mode;
uniform float u_radius_world;
uniform int u_ssao_sample_count;
uniform float u_ssao_bias_fraction;
uniform float u_ssao_intensity;
uniform int u_gtao_slice_count;
uniform int u_gtao_samples_per_side;
uniform float u_gtao_falloff_fraction;
uniform float u_gtao_thickness_fraction;
uniform float u_gtao_intensity;
uniform int u_gtao_bent_normals_enabled;

in vec2 v_uv;
layout(location = 0) out vec4 out_ao_bent_normal;

float stable_noise(vec2 pixel) {
    return fract(52.9829189 * fract(dot(
        pixel,
        vec2(0.06711056, 0.00583715))));
}

vec3 reconstruct_view_position(vec2 uv, float linear_depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(
        ndc.x * 0.5 * u_camera_viewport.x * linear_depth,
        ndc.y * 0.5 * u_camera_viewport.y * linear_depth,
        -linear_depth);
}

vec2 project_view_position(vec3 position) {
    float depth = max(-position.z, 1.0e-5);
    vec2 ndc = vec2(
        2.0 * position.x / (u_camera_viewport.x * depth),
        2.0 * position.y / (u_camera_viewport.y * depth));
    return ndc * 0.5 + 0.5;
}

bool inside_screen(vec2 uv) {
    return all(greaterThanEqual(uv, vec2(0.0))) &&
        all(lessThanEqual(uv, vec2(1.0)));
}

float apply_intensity(float visibility, float intensity) {
    if (intensity <= 0.0) {
        return 1.0;
    }
    return pow(clamp(visibility, 0.0, 1.0), intensity);
}

vec4 evaluate_ssao(vec3 center, vec3 normal, float center_depth) {
    float rotation = stable_noise(gl_FragCoord.xy) * 2.0 * PI;
    vec3 random_axis = vec3(cos(rotation), sin(rotation), 0.0);
    vec3 tangent = random_axis - normal * dot(random_axis, normal);
    if (dot(tangent, tangent) <= 1.0e-8) {
        tangent = cross(
            normal,
            abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                 : vec3(0.0, 1.0, 0.0));
    }
    tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(normal, tangent));
    float bias = u_radius_world * u_ssao_bias_fraction;
    float occlusion = 0.0;
    int used_samples = 0;
    for (int index = 0; index < 64; ++index) {
        if (index >= u_ssao_sample_count) {
            break;
        }
        float fraction = (float(index) + 0.5) /
            float(max(u_ssao_sample_count, 1));
        float phi = float(index) * GOLDEN_ANGLE + rotation;
        float radial = sqrt(fraction);
        float z = sqrt(max(0.0, 1.0 - fraction));
        vec3 local_direction = vec3(
            cos(phi) * radial,
            sin(phi) * radial,
            z);
        vec3 direction = normalize(
            tangent * local_direction.x +
            bitangent * local_direction.y +
            normal * local_direction.z);
        float distance_scale = mix(0.1, 1.0, fraction * fraction);
        vec3 test_position = center +
            direction * (u_radius_world * distance_scale);
        vec2 sample_uv = project_view_position(test_position);
        if (!inside_screen(sample_uv)) {
            continue;
        }
        float sample_depth = texture(u_linear_depth, sample_uv).r;
        if (sample_depth <= 0.0) {
            continue;
        }
        float test_depth = -test_position.z;
        float depth_delta = abs(center_depth - sample_depth);
        float range_weight = smoothstep(
            0.0,
            1.0,
            u_radius_world / max(depth_delta, 1.0e-5));
        occlusion += sample_depth < test_depth - bias
            ? range_weight
            : 0.0;
        used_samples += 1;
    }
    float visibility = used_samples > 0
        ? 1.0 - occlusion / float(used_samples)
        : 1.0;
    visibility = apply_intensity(visibility, u_ssao_intensity);
    return vec4(normal, visibility);
}

mat3 rotate_positive_z_to(vec3 target) {
    // OpenGL view-space positions are in front of the camera at negative Z,
    // so the surface-to-camera view direction points toward positive Z.  The
    // original GTAO reference uses a positive-Z-forward view space and rotates
    // from -Z.  Porting that basis verbatim creates an antipodal 180 degree
    // singularity at the center of an OpenGL viewport.
    vec3 from = vec3(0.0, 0.0, 1.0);
    target = normalize(target);
    float cosine = clamp(dot(from, target), -1.0, 1.0);
    if (cosine < -0.9999) {
        return mat3(
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, -1.0);
    }
    vec3 v = cross(from, target);
    mat3 skew = mat3(
        0.0, v.z, -v.y,
        -v.z, 0.0, v.x,
        v.y, -v.x, 0.0);
    return mat3(1.0) + skew +
        (skew * skew) / max(1.0 + cosine, 1.0e-5);
}

vec4 evaluate_gtao(vec3 center, vec3 normal, float center_depth) {
    vec3 view_direction = normalize(-center);
    float noise = stable_noise(gl_FragCoord.xy);
    float visibility_sum = 0.0;
    vec3 bent_normal_sum = vec3(0.0);
    mat3 local_to_view = rotate_positive_z_to(view_direction);
    vec2 projected_radius = vec2(
        u_radius_world /
            max(u_camera_viewport.x * center_depth, 1.0e-5),
        u_radius_world /
            max(u_camera_viewport.y * center_depth, 1.0e-5));

    for (int slice_index = 0; slice_index < 8; ++slice_index) {
        if (slice_index >= u_gtao_slice_count) {
            break;
        }
        float phi = PI *
            (float(slice_index) + noise) /
            float(max(u_gtao_slice_count, 1));
        vec2 omega = vec2(cos(phi), sin(phi));
        vec3 direction = vec3(omega, 0.0);
        vec3 ortho_direction = direction -
            dot(direction, view_direction) * view_direction;
        vec3 axis = cross(direction, view_direction);
        vec3 projected_normal = normal - axis * dot(normal, axis);
        float projected_length = length(projected_normal);
        if (projected_length <= 1.0e-6 ||
            dot(ortho_direction, ortho_direction) <= 1.0e-8) {
            continue;
        }
        float normal_sign = sign(dot(ortho_direction, projected_normal));
        if (normal_sign == 0.0) {
            normal_sign = 1.0;
        }
        float cos_normal = clamp(
            dot(projected_normal, view_direction) / projected_length,
            0.0,
            1.0);
        float normal_angle = normal_sign * acos(cos_normal);
        float horizon[2];

        for (int side = 0; side < 2; ++side) {
            float side_sign = side == 0 ? -1.0 : 1.0;
            float horizon_cosine = -1.0;
            for (int sample_index = 0; sample_index < 8; ++sample_index) {
                if (sample_index >= u_gtao_samples_per_side) {
                    break;
                }
                float sequence =
                    (float(sample_index) + 0.5 + noise * 0.5) /
                    float(max(u_gtao_samples_per_side, 1));
                float sample_fraction = min(sequence * sequence, 1.0);
                vec2 sample_uv = v_uv +
                    side_sign * omega * projected_radius * sample_fraction;
                if (!inside_screen(sample_uv)) {
                    continue;
                }
                float sample_depth = texture(u_linear_depth, sample_uv).r;
                if (sample_depth <= 0.0) {
                    continue;
                }
                vec3 sample_position = reconstruct_view_position(
                    sample_uv,
                    sample_depth);
                vec3 horizon_vector = sample_position - center;
                float distance_to_sample = length(horizon_vector);
                if (distance_to_sample <= 1.0e-5) {
                    continue;
                }
                float falloff_start = u_radius_world *
                    (1.0 - u_gtao_falloff_fraction);
                float distance_weight = 1.0 - smoothstep(
                    falloff_start,
                    u_radius_world,
                    distance_to_sample);
                float thickness_weight = 1.0;
                if (u_gtao_thickness_fraction > 0.0) {
                    float thickness = max(
                        u_radius_world * u_gtao_thickness_fraction,
                        1.0e-5);
                    thickness_weight = 1.0 - smoothstep(
                        thickness,
                        u_radius_world,
                        abs(sample_depth - center_depth));
                }
                float candidate = dot(
                    horizon_vector / distance_to_sample,
                    view_direction);
                candidate = mix(
                    -1.0,
                    candidate,
                    clamp(distance_weight * thickness_weight, 0.0, 1.0));
                horizon_cosine = max(horizon_cosine, candidate);
            }
            horizon[side] = normal_angle + clamp(
                side_sign * acos(clamp(horizon_cosine, -1.0, 1.0)) -
                    normal_angle,
                -0.5 * PI,
                0.5 * PI);
            visibility_sum += projected_length *
                (cos_normal +
                    2.0 * horizon[side] * sin(normal_angle) -
                    cos(2.0 * horizon[side] - normal_angle)) *
                0.25;
        }

        float t0 = (
            6.0 * sin(horizon[0] - normal_angle) -
            sin(3.0 * horizon[0] - normal_angle) +
            6.0 * sin(horizon[1] - normal_angle) -
            sin(3.0 * horizon[1] - normal_angle) +
            16.0 * sin(normal_angle) -
            3.0 * (sin(horizon[0] + normal_angle) +
                   sin(horizon[1] + normal_angle))) / 12.0;
        float t1 = (
            -cos(3.0 * horizon[0] - normal_angle) -
            cos(3.0 * horizon[1] - normal_angle) +
            8.0 * cos(normal_angle) -
            3.0 * (cos(horizon[0] + normal_angle) +
                   cos(horizon[1] + normal_angle))) / 12.0;
        vec3 local_bent_normal = vec3(
            omega.x * t0,
            omega.y * t0,
            t1);
        bent_normal_sum +=
            local_to_view * local_bent_normal * projected_length;
    }

    float visibility = clamp(
        visibility_sum / float(max(u_gtao_slice_count, 1)),
        0.0,
        1.0);
    visibility = apply_intensity(visibility, u_gtao_intensity);
    vec3 bent_normal = normal;
    if (u_gtao_bent_normals_enabled != 0 &&
        dot(bent_normal_sum, bent_normal_sum) > 1.0e-8) {
        bent_normal = normalize(bent_normal_sum);
        float hemisphere = dot(bent_normal, normal);
        if (hemisphere < 0.01) {
            bent_normal = normalize(
                bent_normal + normal * (0.01 - hemisphere));
        }
    }
    return vec4(bent_normal, visibility);
}

void main() {
    float center_depth = texture(u_linear_depth, v_uv).r;
    vec3 normal = texture(u_view_normal, v_uv).xyz;
    if (center_depth <= 0.0 || dot(normal, normal) <= 1.0e-8) {
        out_ao_bent_normal = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }
    normal = normalize(normal);
    vec3 center = reconstruct_view_position(v_uv, center_depth);
    out_ao_bent_normal = u_ao_mode == 1
        ? evaluate_ssao(center, normal, center_depth)
        : evaluate_gtao(center, normal, center_depth);
}
