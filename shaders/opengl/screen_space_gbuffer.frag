#version 450 core

layout(binding = 0) uniform sampler2D u_base_color_texture;
layout(binding = 1) uniform sampler2D u_opacity_texture;
layout(binding = 2) uniform sampler2D u_normal_or_bump_texture;
layout(binding = 3) uniform sampler2D u_metallic_roughness_texture;
layout(binding = 4) uniform sampler2D u_occlusion_texture;
layout(binding = 6) uniform sampler2D u_specular_texture;
layout(binding = 7) uniform sampler2D u_specular_color_texture;
layout(binding = 8) uniform sampler2D u_specular_glossiness_texture;

uniform vec3 u_camera_position;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_base_color;
uniform float u_opacity;
uniform float u_alpha_cutoff;
uniform float u_bump_scale;
uniform float u_normal_scale;
uniform int u_alpha_mode;
uniform int u_two_sided;
uniform int u_has_base_color_texture;
uniform int u_has_opacity_texture;
uniform int u_has_normal_texture;
uniform int u_has_bump_texture;
uniform int u_has_metallic_roughness_texture;
uniform int u_has_occlusion_texture;
uniform int u_has_specular_texture;
uniform int u_has_specular_color_texture;
uniform int u_has_specular_glossiness_texture;
uniform int u_material_type;
uniform int u_pbr_workflow;
uniform float u_ior;
uniform vec3 u_specular_color;
uniform float u_specular_factor;
uniform float u_glossiness;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_occlusion_strength;
uniform vec4 u_texture_offset_scale[9];
uniform float u_texture_rotation[9];
uniform int u_texture_texcoord[9];
uniform int u_texture_top_left[9];

in VS_OUT {
    vec3 world_position;
    vec3 normal;
    vec2 uv;
    vec2 uv1;
    vec4 tangent;
    vec4 color;
} fragment_in;

layout(location = 0) out vec3 out_view_normal;
layout(location = 1) out float out_linear_depth;
// Screen-space reflection material data: RGB = specular f0, A = roughness.
layout(location = 2) out vec4 out_ssr_pbr;
// Screen-space reflection material data: RGB = specular f90, A = occlusion.
layout(location = 3) out vec4 out_ssr_pbr_aux;
// RGB = diffuse color. A bits: 0 = Fresnel uses max, 1 = two-sided.
layout(location = 4) out vec4 out_ssr_material;
// RGB = diffuse Fresnel f0, A = achromatic diffuse Fresnel f90.
layout(location = 5) out vec4 out_ssr_diffuse_fresnel;

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 fresnel_schlick(float cosine, vec3 f0, vec3 f90) {
    return f0 + (f90 - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
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
    vec3 tangent = fragment_in.tangent.xyz -
        normal * dot(normal, fragment_in.tangent.xyz);
    if (dot(tangent, tangent) <= 1.0e-12) {
        return normal;
    }
    tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(normal, tangent)) * fragment_in.tangent.w;
    if (u_has_normal_texture != 0) {
        vec3 mapped = texture(
            u_normal_or_bump_texture,
            material_uv(2)).xyz * 2.0 - 1.0;
        mapped.xy *= u_normal_scale;
        return normalize(
            tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
    }
    if (u_has_bump_texture != 0) {
        vec2 texel = 1.0 / vec2(textureSize(u_normal_or_bump_texture, 0));
        vec2 uv = material_uv(2);
        float left = luminance(texture(
            u_normal_or_bump_texture, uv - vec2(texel.x, 0.0)).rgb);
        float right = luminance(texture(
            u_normal_or_bump_texture, uv + vec2(texel.x, 0.0)).rgb);
        float down = luminance(texture(
            u_normal_or_bump_texture, uv - vec2(0.0, texel.y)).rgb);
        float up = luminance(texture(
            u_normal_or_bump_texture, uv + vec2(0.0, texel.y)).rgb);
        vec3 candidate = normal -
            (tangent * (right - left) + bitangent * (up - down)) *
                (0.5 * u_bump_scale);
        return dot(candidate, candidate) > 1.0e-12
            ? normalize(candidate)
            : normal;
    }
    return normal;
}

void evaluate_ssr_material(
    vec3 base_color,
    out vec3 specular_f0,
    out vec3 specular_f90,
    out float roughness) {
    // Mirrors the material evaluation in raster.frag so the screen-space
    // reflection pass can reconstruct the exact specular response.
    if (u_material_type == 3) {
        specular_f0 = vec3(0.0);
        specular_f90 = vec3(0.0);
        roughness = 1.0;
        return;
    }
    float metallic = u_material_type == 1 ? 1.0 : clamp(u_metallic, 0.0, 1.0);
    roughness = u_material_type == 0 ? 1.0 : clamp(u_roughness, 0.02, 1.0);
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
        roughness = clamp(1.0 - glossiness, 0.02, 1.0);
        return;
    }
    if (u_has_metallic_roughness_texture != 0) {
        vec3 packed_value = texture(
            u_metallic_roughness_texture,
            material_uv(3)).rgb;
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
}

void main() {
    vec4 base_sample = u_has_base_color_texture != 0
        ? texture(u_base_color_texture, material_uv(0))
        : vec4(1.0);
    float opacity = clamp(
        u_opacity * base_sample.a * fragment_in.color.a,
        0.0,
        1.0);
    if (u_has_opacity_texture != 0) {
        opacity *= luminance(texture(
            u_opacity_texture,
            material_uv(1)).rgb);
    }
    if (u_alpha_mode == 1 && opacity < u_alpha_cutoff) {
        discard;
    }

    vec3 world_normal = surface_normal();
    out_view_normal = normalize(vec3(
        dot(world_normal, u_camera_right),
        dot(world_normal, u_camera_up),
        -dot(world_normal, u_camera_forward)));
    out_linear_depth = max(
        dot(fragment_in.world_position - u_camera_position, u_camera_forward),
        0.0);

    vec3 base_color = u_base_color * base_sample.rgb * fragment_in.color.rgb;
    vec3 specular_f0;
    vec3 specular_f90;
    float roughness;
    evaluate_ssr_material(base_color, specular_f0, specular_f90, roughness);
    float occlusion = 1.0;
    if (u_has_occlusion_texture != 0) {
        float sampled = texture(u_occlusion_texture, material_uv(4)).r;
        occlusion = mix(1.0, sampled, clamp(u_occlusion_strength, 0.0, 1.0));
    }
    out_ssr_pbr = vec4(specular_f0, roughness);
    out_ssr_pbr_aux = vec4(specular_f90, occlusion);

    vec3 diffuse_color = vec3(0.0);
    vec3 diffuse_f0 = vec3(0.0);
    float diffuse_f90 = 0.0;
    float diffuse_uses_max = 0.0;
    if (u_material_type != 3) {
        float metallic = u_material_type == 1
            ? 1.0
            : clamp(u_metallic, 0.0, 1.0);
        if (u_material_type == 4 && u_pbr_workflow == 1) {
            vec3 specular = max(u_specular_color, vec3(0.0));
            if (u_has_specular_glossiness_texture != 0) {
                specular *= texture(
                    u_specular_glossiness_texture,
                    material_uv(8)).rgb;
            }
            specular = clamp(specular, vec3(0.0), vec3(1.0));
            diffuse_color = base_color *
                (1.0 - max(max(specular.r, specular.g), specular.b));
            diffuse_f0 = specular;
            diffuse_f90 = 1.0;
        } else {
            if (u_has_metallic_roughness_texture != 0) {
                metallic = clamp(
                    metallic * texture(
                        u_metallic_roughness_texture,
                        material_uv(3)).b,
                    0.0,
                    1.0);
            }
            float specular_strength = clamp(u_specular_factor, 0.0, 1.0);
            if (u_has_specular_texture != 0) {
                specular_strength *= texture(
                    u_specular_texture, material_uv(6)).a;
            }
            vec3 material_specular_color = max(
                u_specular_color, vec3(0.0));
            if (u_has_specular_color_texture != 0) {
                material_specular_color *= texture(
                    u_specular_color_texture,
                    material_uv(7)).rgb;
            }
            float ior_ratio = (max(u_ior, 1.0) - 1.0) /
                (max(u_ior, 1.0) + 1.0);
            vec3 dielectric_f0 = min(
                vec3(1.0),
                material_specular_color * (ior_ratio * ior_ratio)) *
                specular_strength;
            diffuse_color = base_color * (1.0 - metallic);
            diffuse_f0 = dielectric_f0;
            diffuse_f90 = specular_strength;
            diffuse_uses_max = 1.0;
        }
    }
    out_ssr_material = vec4(max(diffuse_color, vec3(0.0)),
        diffuse_uses_max + (u_two_sided != 0 ? 2.0 : 0.0));
    out_ssr_diffuse_fresnel = vec4(diffuse_f0, diffuse_f90);
}
