#version 450 core

const float PI = 3.14159265358979323846;
const float SSR_MAX_ANISOTROPY = 8.0;
const int SSR_MAX_FILTER_TAPS = 9;

layout(binding = 0) uniform sampler2D u_opaque;
layout(binding = 1) uniform sampler2D u_linear_depth;
layout(binding = 2) uniform sampler2D u_view_normal;
layout(binding = 3) uniform sampler2D u_ssr_pbr;
layout(binding = 4) uniform sampler2D u_ssr_pbr_aux;
layout(binding = 5) uniform samplerCube u_environment_prefilter;
layout(binding = 6) uniform sampler2D u_environment_brdf_lut;
layout(binding = 7) uniform sampler2D u_ambient_occlusion_texture;

uniform vec2 u_camera_viewport;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform int u_max_steps;
uniform int u_refinement_steps;
uniform float u_max_distance;
uniform float u_thickness;
uniform float u_max_roughness;
uniform float u_intensity;
uniform float u_edge_fade;
uniform int u_jitter;
uniform vec3 u_environment_color;
uniform float u_environment_intensity;
uniform float u_environment_rotation_radians;
uniform float u_environment_mip_count;
uniform int u_has_environment_map;
uniform int u_ibl_enabled;
uniform int u_ao_mode;
uniform int u_ao_bent_normals_enabled;
uniform int u_debug_view;

in vec2 v_uv;
layout(location = 0) out vec4 out_linear_color;

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

vec3 rotate_y(vec3 direction, float radians) {
    float c = cos(radians);
    float s = sin(radians);
    return vec3(
        c * direction.x + s * direction.z,
        direction.y,
        -s * direction.x + c * direction.z);
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

float screen_edge_fade(vec2 uv) {
    vec2 ndc_distance = abs(uv * 2.0 - 1.0);
    float distance_to_edge = 1.0 - max(ndc_distance.x, ndc_distance.y);
    float fade_width = clamp(u_edge_fade, 0.0, 0.5);
    if (fade_width <= 0.0) {
        return 1.0;
    }
    return smoothstep(0.0, fade_width, distance_to_edge);
}

bool march_reflection(
    vec3 position,
    vec3 reflection,
    float thickness,
    float step,
    float jitter_value,
    out vec2 hit_uv,
    out float edge_fade,
    out float hit_distance) {
    float ray_distance = thickness * 2.0 + step * jitter_value;
    vec3 previous_position = position + reflection * ray_distance;
    vec2 previous_uv = project_view_position(previous_position);
    bool previous_valid = previous_position.z < -1.0e-5 &&
        inside_screen(previous_uv);
    float previous_delta = 0.0;
    if (previous_valid) {
        float previous_surface_depth = texture(u_linear_depth, previous_uv).r;
        previous_valid = previous_surface_depth > 0.0;
        previous_delta = -previous_position.z - previous_surface_depth;
    }
    hit_uv = vec2(0.0);
    edge_fade = 0.0;
    hit_distance = 0.0;
    for (int index = 0; index < 256; ++index) {
        if (index >= u_max_steps) {
            break;
        }
        ray_distance += step;
        vec3 ray_position = position + reflection * ray_distance;
        if (ray_position.z >= -1.0e-5) {
            return false;
        }
        vec2 uv = project_view_position(ray_position);
        if (!inside_screen(uv)) {
            return false;
        }
        float ray_depth = -ray_position.z;
        float surface_depth = texture(u_linear_depth, uv).r;
        bool current_valid = surface_depth > 0.0;
        float depth_delta = ray_depth - surface_depth;
        if (current_valid && previous_valid &&
            previous_delta < 0.0 && depth_delta >= 0.0) {
            vec3 near_position = previous_position;
            vec3 far_position = ray_position;
            vec2 refined_uv = uv;
            float refined_delta = depth_delta;
            for (int refine = 0; refine < 16; ++refine) {
                if (refine >= u_refinement_steps) {
                    break;
                }
                vec3 midpoint = (near_position + far_position) * 0.5;
                if (midpoint.z >= -1.0e-5) {
                    far_position = midpoint;
                    continue;
                }
                vec2 midpoint_uv = project_view_position(midpoint);
                if (!inside_screen(midpoint_uv)) {
                    near_position = midpoint;
                    continue;
                }
                float midpoint_ray_depth = -midpoint.z;
                float midpoint_surface = texture(
                    u_linear_depth,
                    midpoint_uv).r;
                if (midpoint_surface <= 0.0) {
                    near_position = midpoint;
                } else if (midpoint_ray_depth >= midpoint_surface) {
                    far_position = midpoint;
                    refined_uv = midpoint_uv;
                    refined_delta = midpoint_ray_depth - midpoint_surface;
                } else {
                    near_position = midpoint;
                }
            }
            // A bracket crossing alone is insufficient at silhouettes or with
            // coarse steps: require the refined sample to lie inside the
            // configured surface-thickness interval.
            if (refined_delta >= 0.0 && refined_delta <= thickness) {
                hit_uv = refined_uv;
                edge_fade = screen_edge_fade(refined_uv);
                hit_distance = max(
                    dot(far_position - position, reflection),
                    0.0);
                return edge_fade > 0.0;
            }
        }
        previous_position = ray_position;
        previous_valid = current_valid;
        previous_delta = depth_delta;
    }
    return false;
}

float ggx_reflection_cone_tangent(float roughness) {
    // The glTF perceptual roughness maps to the GGX slope parameter alpha=r^2.
    // A box-filtered mip should cover the lobe's central FWHM, not its wider
    // 50%-energy diameter: GGX reaches half maximum at
    // tan(theta_h) = alpha * sqrt(sqrt(2)-1). Reflection doubles theta_h.
    float alpha = pow(clamp(roughness, 0.0, 1.0), 2.0);
    float half_maximum_slope = alpha * 0.6435942529;
    return min(
        2.0 * half_maximum_slope /
            max(1.0 - half_maximum_slope * half_maximum_slope, 1.0e-4),
        8.0);
}

struct ReflectionFootprint {
    vec2 major_axis_uv;
    vec2 support_radius_uv;
    float major_radius_pixels;
    float minor_radius_pixels;
};

void reflection_lobe_basis(
    vec3 view_ray,
    vec3 view_normal,
    vec3 reflection,
    out vec3 major_direction,
    out vec3 minor_direction,
    out float minor_scale) {
    float n_dot_v = clamp(dot(view_normal, -view_ray), 0.0, 1.0);
    vec3 incident_tangent = view_ray -
        view_normal * dot(view_ray, view_normal);
    if (dot(incident_tangent, incident_tangent) > 1.0e-8) {
        incident_tangent = normalize(incident_tangent);
        // Differentiating reflect(I, N) with respect to the micro-normal
        // gives a wider lobe in the plane of incidence.  The orthogonal
        // derivative contracts by N.V, which is the source of specular
        // elongation at grazing angles.
        major_direction = normalize(-2.0 * (
            dot(view_ray, incident_tangent) * view_normal +
            dot(view_ray, view_normal) * incident_tangent));
        minor_direction = normalize(cross(reflection, major_direction));
        minor_scale = max(n_dot_v, 1.0 / SSR_MAX_ANISOTROPY);
        return;
    }

    vec3 helper = abs(reflection.z) < 0.999
        ? vec3(0.0, 0.0, 1.0)
        : vec3(0.0, 1.0, 0.0);
    major_direction = normalize(cross(helper, reflection));
    minor_direction = normalize(cross(reflection, major_direction));
    minor_scale = 1.0;
}

vec2 project_view_offset(vec3 position, vec3 offset) {
    float depth = max(-position.z, 1.0e-5);
    vec2 numerator = offset.xy * depth + position.xy * offset.z;
    return numerator / max(
        u_camera_viewport * depth * depth,
        vec2(1.0e-5));
}

ReflectionFootprint reflection_footprint(
    float roughness,
    float hit_distance,
    float hit_view_depth,
    vec2 hit_uv,
    vec3 view_ray,
    vec3 view_normal,
    vec3 reflection) {
    vec3 major_direction;
    vec3 minor_direction;
    float minor_scale;
    reflection_lobe_basis(
        view_ray,
        view_normal,
        reflection,
        major_direction,
        minor_direction,
        minor_scale);

    float major_radius_view = max(hit_distance, 0.0) *
        ggx_reflection_cone_tangent(roughness);
    float minor_radius_view = major_radius_view * minor_scale;
    vec3 hit_position = reconstruct_view_position(hit_uv, hit_view_depth);
    vec2 major_axis_uv = project_view_offset(
        hit_position,
        major_direction * major_radius_view);
    vec2 minor_axis_uv = project_view_offset(
        hit_position,
        minor_direction * minor_radius_view);

    vec2 source_size = vec2(textureSize(u_opaque, 0));
    vec2 major_axis_pixels = major_axis_uv * source_size;
    vec2 minor_axis_pixels = minor_axis_uv * source_size;
    float covariance_xx = major_axis_pixels.x * major_axis_pixels.x +
        minor_axis_pixels.x * minor_axis_pixels.x;
    float covariance_xy = major_axis_pixels.x * major_axis_pixels.y +
        minor_axis_pixels.x * minor_axis_pixels.y;
    float covariance_yy = major_axis_pixels.y * major_axis_pixels.y +
        minor_axis_pixels.y * minor_axis_pixels.y;
    float discriminant = sqrt(max(
        (covariance_xx - covariance_yy) *
            (covariance_xx - covariance_yy) +
            4.0 * covariance_xy * covariance_xy,
        0.0));
    float maximum_eigenvalue = max(
        0.5 * (covariance_xx + covariance_yy + discriminant),
        0.0);
    float minimum_eigenvalue = max(
        0.5 * (covariance_xx + covariance_yy - discriminant),
        0.0);
    vec2 major_direction_pixels;
    if (abs(covariance_xy) > 1.0e-8) {
        major_direction_pixels = normalize(vec2(
            covariance_xy,
            maximum_eigenvalue - covariance_xx));
    } else {
        major_direction_pixels = covariance_xx >= covariance_yy
            ? vec2(1.0, 0.0)
            : vec2(0.0, 1.0);
    }

    ReflectionFootprint footprint;
    footprint.major_radius_pixels = sqrt(maximum_eigenvalue);
    footprint.minor_radius_pixels = sqrt(minimum_eigenvalue);
    footprint.major_axis_uv = major_direction_pixels *
        footprint.major_radius_pixels / max(source_size, vec2(1.0));
    // Axis-aligned bounds of the projected ellipse keep every anisotropic
    // tap inside the valid screen support before edge fading reaches one.
    footprint.support_radius_uv = sqrt(max(vec2(
        major_axis_uv.x * major_axis_uv.x +
            minor_axis_uv.x * minor_axis_uv.x,
        major_axis_uv.y * major_axis_uv.y +
            minor_axis_uv.y * minor_axis_uv.y),
        vec2(0.0)));
    return footprint;
}

float reflection_lod(float radius_pixels) {
    float maximum_lod = float(max(textureQueryLevels(u_opaque) - 1, 0));
    // Bilinear sampling at a mip level already spans adjacent box-filtered
    // texels, so its effective support is about twice the nominal footprint.
    // Select from the minor half-width; explicit taps cover the major axis.
    float footprint = max(radius_pixels, 1.0);
    return clamp(log2(footprint), 0.0, maximum_lod);
}

int reflection_filter_tap_count(
    float major_radius_pixels,
    float minor_radius_pixels) {
    float effective_minor = max(minor_radius_pixels, 1.0);
    float anisotropy = clamp(
        major_radius_pixels / effective_minor,
        1.0,
        SSR_MAX_ANISOTROPY);
    int half_taps = int(ceil((anisotropy - 1.0) * 0.5));
    return min(1 + 2 * half_taps, SSR_MAX_FILTER_TAPS);
}

vec3 sample_elliptical_reflection(
    vec2 hit_uv,
    ReflectionFootprint footprint) {
    float lod = reflection_lod(footprint.minor_radius_pixels);
    float effective_minor = max(footprint.minor_radius_pixels, 1.0);
    float residual_scale = clamp(
        1.0 - effective_minor /
            max(footprint.major_radius_pixels, effective_minor),
        0.0,
        1.0);
    vec2 sample_axis_uv = footprint.major_axis_uv * residual_scale;
    int tap_count = reflection_filter_tap_count(
        footprint.major_radius_pixels,
        footprint.minor_radius_pixels);
    int half_taps = tap_count / 2;
    vec3 radiance = vec3(0.0);
    float weight_sum = 0.0;
    for (int tap_index = 0; tap_index < SSR_MAX_FILTER_TAPS; ++tap_index) {
        if (tap_index >= tap_count) {
            break;
        }
        float tap_position = half_taps > 0
            ? float(tap_index - half_taps) / float(half_taps)
            : 0.0;
        // The projected radius is the GGX FWHM radius, so the outer taps
        // retain half weight instead of using a box kernel.
        float weight = exp2(-tap_position * tap_position);
        radiance += textureLod(
            u_opaque,
            hit_uv + sample_axis_uv * tap_position,
            lod).rgb * weight;
        weight_sum += weight;
    }
    return radiance / max(weight_sum, 1.0e-5);
}

float reflection_footprint_edge_fade(
    vec2 uv,
    vec2 support_radius_uv) {
    if (max(support_radius_uv.x, support_radius_uv.y) <= 1.0e-8) {
        return 1.0;
    }
    vec2 edge_distance = min(uv, vec2(1.0) - uv);
    vec2 support = edge_distance / max(
        support_radius_uv,
        vec2(1.0e-8));
    float complete_radius = clamp(min(support.x, support.y), 0.0, 1.0);
    return smoothstep(0.0, 1.0, complete_radius);
}

void main() {
    float center_depth = texture(u_linear_depth, v_uv).r;
    vec3 view_normal = texture(u_view_normal, v_uv).xyz;
    vec3 opaque = texture(u_opaque, v_uv).rgb;
    vec4 pbr = texture(u_ssr_pbr, v_uv);
    vec4 pbr_aux = texture(u_ssr_pbr_aux, v_uv);
    float roughness = clamp(pbr.a, 0.02, 1.0);

    vec3 hit_radiance = vec3(0.0);
    float confidence = 0.0;
    bool has_surface = center_depth > 0.0 &&
        dot(view_normal, view_normal) > 1.0e-8;
    if (has_surface) {
        view_normal = normalize(view_normal);
        vec2 ndc = v_uv * 2.0 - 1.0;
        vec3 view_ray = normalize(vec3(
            ndc.x * 0.5 * u_camera_viewport.x,
            ndc.y * 0.5 * u_camera_viewport.y,
            -1.0));
        vec3 position = reconstruct_view_position(v_uv, center_depth);
        vec3 reflection_view = reflect(view_ray, view_normal);
        float step = u_max_distance / float(max(u_max_steps, 1));
        float jitter_value = u_jitter != 0
            ? stable_noise(gl_FragCoord.xy)
            : 0.5;
        vec2 hit_uv;
        float edge_fade = 0.0;
        float hit_distance = 0.0;
        if (march_reflection(
                position,
                reflection_view,
                u_thickness,
                step,
                jitter_value,
                hit_uv,
                edge_fade,
                hit_distance)) {
            float hit_view_depth = texture(u_linear_depth, hit_uv).r;
            ReflectionFootprint footprint = reflection_footprint(
                roughness,
                hit_distance,
                hit_view_depth,
                hit_uv,
                view_ray,
                view_normal,
                reflection_view);
            hit_radiance = sample_elliptical_reflection(hit_uv, footprint);
            confidence = edge_fade * reflection_footprint_edge_fade(
                hit_uv,
                footprint.support_radius_uv);
        }

        if (u_debug_view != 0) {
            if (u_debug_view == 1) {
                out_linear_color = vec4(hit_radiance, 1.0);
            } else {
                out_linear_color = vec4(vec3(confidence), 1.0);
            }
            return;
        }

        float n_dot_v = clamp(dot(view_normal, -view_ray), 0.0, 1.0);
        vec3 specular_f0 = pbr.rgb;
        vec3 specular_f90 = pbr_aux.rgb;
        float occlusion = clamp(pbr_aux.a, 0.0, 1.0);
        vec2 brdf = texture(
            u_environment_brdf_lut,
            vec2(n_dot_v, roughness)).rg;
        vec3 specular_response =
            specular_f0 * brdf.x + specular_f90 * brdf.y;

        mat3 view_to_world = mat3(
            u_camera_right,
            u_camera_up,
            -u_camera_forward);
        vec3 world_normal = view_to_world * view_normal;
        vec3 world_reflection = view_to_world * reflection_view;

        // Ambient-occlusion modulation mirrors the specular IBL path of
        // raster.frag so the correction can replace it without a residual.
        float screen_ao = 1.0;
        float specular_visibility = 1.0;
        if (u_ao_mode != 0) {
            vec4 ao_sample = texture(u_ambient_occlusion_texture, v_uv);
            screen_ao = clamp(ao_sample.a, 0.0, 1.0);
            specular_visibility = screen_ao;
            if (u_ao_mode == 2 && u_ao_bent_normals_enabled != 0 &&
                dot(ao_sample.xyz, ao_sample.xyz) > 1.0e-8) {
                vec3 bent_normal = normalize(
                    view_to_world * normalize(ao_sample.xyz));
                float normal_alignment = dot(bent_normal, world_normal);
                if (normal_alignment < 0.01) {
                    bent_normal = normalize(
                        bent_normal +
                        world_normal * (0.01 - normal_alignment));
                }
                specular_visibility = gtso_visibility(
                    bent_normal,
                    world_reflection,
                    roughness,
                    screen_ao,
                    n_dot_v);
            }
        }

        vec3 environment_term = vec3(0.0);
        if (u_ibl_enabled != 0) {
            vec3 local_reflection = rotate_y(
                world_reflection,
                -u_environment_rotation_radians);
            vec3 specular_radiance = u_has_environment_map != 0
                ? textureLod(
                    u_environment_prefilter,
                    local_reflection,
                    roughness * max(u_environment_mip_count - 1.0, 0.0)).rgb *
                    u_environment_color * u_environment_intensity
                : u_environment_color * u_environment_intensity;
            environment_term = specular_radiance *
                specular_response;
        }

        float roughness_weight = 1.0 - smoothstep(
            max(u_max_roughness - 0.05, 0.0),
            min(u_max_roughness + 0.05, 1.0),
            roughness);
        float weight = clamp(u_intensity, 0.0, 4.0) *
            confidence * roughness_weight;
        vec3 correction = (specular_response * hit_radiance - environment_term) *
            (occlusion * specular_visibility * weight);
        out_linear_color = vec4(opaque + correction, confidence);
        return;
    }

    if (u_debug_view != 0) {
        out_linear_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    out_linear_color = vec4(opaque, 0.0);
}
