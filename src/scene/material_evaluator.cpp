#include "scene/material_evaluator.h"

#include "scene/material.h"
#include "scene/scene.h"
#include "scene/texture.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace renderer {

namespace {

bool valid_texture_id(const Scene& scene, int texture_id) {
    return texture_id >= 0 && static_cast<std::size_t>(texture_id) < scene.textures.size();
}

Vec2 texture_uv(const HitRecord& hit, const TextureTransform& transform) {
    Vec2 uv = transform.texcoord == 1 ? hit.uv1 : hit.uv;
    uv = uv.cwiseProduct(transform.scale);
    const float cosine = std::cos(transform.rotation);
    const float sine = std::sin(transform.rotation);
    uv = Vec2(
        cosine * uv.x() - sine * uv.y(),
        sine * uv.x() + cosine * uv.y());
    return uv + transform.offset;
}

Vec3 bumped_normal(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit) {
    if (valid_texture_id(scene, material.normal_texture_id) &&
        hit.has_valid_uv_basis &&
        usable_direction(hit.shading_normal) &&
        usable_direction(hit.tangent) &&
        usable_direction(hit.bitangent)) {
        const Color encoded = scene.textures[
            static_cast<std::size_t>(material.normal_texture_id)].sample(
                texture_uv(hit, material.normal_texture_transform));
        Vec3 tangent_normal(
            encoded.x() * 2.0f - 1.0f,
            encoded.y() * 2.0f - 1.0f,
            encoded.z() * 2.0f - 1.0f);
        tangent_normal.x() *= material.normal_scale;
        tangent_normal.y() *= material.normal_scale;
        if (usable_direction(tangent_normal)) {
            tangent_normal.normalize();
            Vec3 perturbed =
                hit.tangent * tangent_normal.x() +
                hit.bitangent * tangent_normal.y() +
                hit.shading_normal * tangent_normal.z();
            if (usable_direction(perturbed)) {
                perturbed.normalize();
                return perturbed.dot(hit.geometric_normal) < 0.0f ? -perturbed : perturbed;
            }
        }
    }
    if (!valid_texture_id(scene, material.bump_texture_id) ||
        !hit.has_valid_uv_basis ||
        !usable_direction(hit.shading_normal) ||
        !usable_direction(hit.tangent) ||
        !usable_direction(hit.bitangent)) {
        return hit.shading_normal;
    }

    const ImageTexture& texture = scene.textures[static_cast<std::size_t>(material.bump_texture_id)];
    const Vec2 step = texture.texel_size();
    if (step.x() <= 0.0f || step.y() <= 0.0f) {
        return hit.shading_normal;
    }

    const float left = texture.sample_scalar(Vec2(hit.uv.x() - step.x(), hit.uv.y()));
    const float right = texture.sample_scalar(Vec2(hit.uv.x() + step.x(), hit.uv.y()));
    const float down = texture.sample_scalar(Vec2(hit.uv.x(), hit.uv.y() - step.y()));
    const float up = texture.sample_scalar(Vec2(hit.uv.x(), hit.uv.y() + step.y()));
    const float dh_du = (right - left) * 0.5f;
    const float dh_dv = (up - down) * 0.5f;
    const Vec3 gradient = hit.tangent * dh_du + hit.bitangent * dh_dv;
    const Vec3 candidate = hit.shading_normal - gradient * material.bump_scale;
    if (!usable_direction(candidate)) {
        return hit.shading_normal;
    }
    Vec3 perturbed = candidate.normalized();
    if (perturbed.dot(hit.geometric_normal) < 0.0f) {
        perturbed = -perturbed;
    }
    return perturbed;
}

}  // namespace

float sample_material_opacity(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit) {
    if (material.type == MaterialType::Pbr && material.alpha_mode == AlphaMode::Opaque) {
        return 1.0f;
    }
    float opacity = std::clamp(material.opacity * hit.vertex_alpha, 0.0f, 1.0f);
    if (valid_texture_id(scene, material.base_color_texture_id)) {
        opacity *= scene.textures[
            static_cast<std::size_t>(material.base_color_texture_id)].sample_alpha(
                texture_uv(
                    hit,
                    material.base_color_texture_transform));
    }
    if (valid_texture_id(scene, material.opacity_texture_id)) {
        opacity *= scene.textures[static_cast<std::size_t>(material.opacity_texture_id)].sample_scalar(hit.uv);
    }
    return std::clamp(opacity, 0.0f, 1.0f);
}

SurfaceMaterialSample evaluate_surface_material(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit) {
    SurfaceMaterialSample sample;
    sample.base_color = material.base_color;
    if (valid_texture_id(scene, material.base_color_texture_id)) {
        sample.base_color = sample.base_color.cwiseProduct(
            scene.textures[static_cast<std::size_t>(material.base_color_texture_id)].sample(
                texture_uv(hit, material.base_color_texture_transform)));
    } else if (valid_texture_id(scene, material.diffuse_texture_id)) {
        sample.base_color = sample.base_color.cwiseProduct(
            scene.textures[static_cast<std::size_t>(material.diffuse_texture_id)].sample(hit.uv));
    }
    sample.base_color = sample.base_color.cwiseProduct(hit.vertex_color);
    sample.opacity = std::clamp(material.opacity * hit.vertex_alpha, 0.0f, 1.0f);
    if (material.type == MaterialType::Pbr && material.alpha_mode == AlphaMode::Opaque) {
        sample.opacity = 1.0f;
    } else {
        if (valid_texture_id(scene, material.base_color_texture_id)) {
            sample.opacity *= scene.textures[
                static_cast<std::size_t>(material.base_color_texture_id)].sample_alpha(
                    texture_uv(hit, material.base_color_texture_transform));
        }
        if (valid_texture_id(scene, material.opacity_texture_id)) {
            sample.opacity *= scene.textures[
                static_cast<std::size_t>(material.opacity_texture_id)].sample_scalar(hit.uv);
        }
    }
    sample.opacity = std::clamp(sample.opacity, 0.0f, 1.0f);
    sample.metallic = std::clamp(
        material.type == MaterialType::Metal ? 1.0f : material.metallic,
        0.0f,
        1.0f);
    if (material.type == MaterialType::Pbr &&
        material.pbr_workflow == PbrWorkflow::SpecularGlossiness) {
        Color specular = material.specular_color.cwiseMax(Color::Zero());
        float glossiness = std::clamp(material.glossiness, 0.0f, 1.0f);
        if (valid_texture_id(scene, material.specular_glossiness_texture_id)) {
            const ImageTexture& texture = scene.textures[
                static_cast<std::size_t>(material.specular_glossiness_texture_id)];
            const Vec2 uv = texture_uv(hit, material.specular_glossiness_texture_transform);
            specular = specular.cwiseProduct(texture.sample(uv));
            glossiness *= texture.sample_alpha(uv);
        }
        sample.specular_f0 = specular.cwiseMax(Color::Zero()).cwiseMin(Color::Ones());
        sample.specular_f90 = Color::Ones();
        sample.diffuse_fresnel_f0 = sample.specular_f0;
        sample.diffuse_fresnel_f90 = sample.specular_f90;
        sample.diffuse_fresnel_uses_max = false;
        sample.diffuse_color = sample.base_color *
            (1.0f - std::clamp(sample.specular_f0.maxCoeff(), 0.0f, 1.0f));
        sample.metallic = 0.0f;
        sample.roughness = std::clamp(1.0f - glossiness, 0.02f, 1.0f);
    } else {
        sample.roughness = std::clamp(
            material.type == MaterialType::Diffuse ? 1.0f : material.roughness,
            0.02f,
            1.0f);
    }
    if (!(material.type == MaterialType::Pbr &&
          material.pbr_workflow == PbrWorkflow::SpecularGlossiness) &&
        valid_texture_id(scene, material.metallic_roughness_texture_id)) {
        const Color packed = scene.textures[
            static_cast<std::size_t>(material.metallic_roughness_texture_id)].sample(
                texture_uv(hit, material.metallic_roughness_texture_transform));
        sample.roughness = std::clamp(sample.roughness * packed.y(), 0.02f, 1.0f);
        sample.metallic = std::clamp(sample.metallic * packed.z(), 0.0f, 1.0f);
    }
    if (!(material.type == MaterialType::Pbr &&
          material.pbr_workflow == PbrWorkflow::SpecularGlossiness)) {
        float specular_strength = std::clamp(material.specular_factor, 0.0f, 1.0f);
        if (valid_texture_id(scene, material.specular_texture_id)) {
            specular_strength *= scene.textures[
                static_cast<std::size_t>(material.specular_texture_id)].sample_alpha(
                    texture_uv(hit, material.specular_texture_transform));
        }
        Color specular_color = material.specular_color.cwiseMax(Color::Zero());
        if (valid_texture_id(scene, material.specular_color_texture_id)) {
            specular_color = specular_color.cwiseProduct(
                scene.textures[
                    static_cast<std::size_t>(material.specular_color_texture_id)].sample(
                        texture_uv(hit, material.specular_color_texture_transform)));
        }
        const float ior = std::max(1.0f, material.ior);
        const float ior_ratio = (ior - 1.0f) / (ior + 1.0f);
        const float dielectric_base = ior_ratio * ior_ratio;
        const Color dielectric_f0 = (specular_color * dielectric_base)
            .cwiseMin(Color::Ones()) * specular_strength;
        const Color dielectric_f90 = Color::Constant(specular_strength);
        sample.specular_f0 = dielectric_f0 * (1.0f - sample.metallic) +
            sample.base_color * sample.metallic;
        sample.specular_f90 = dielectric_f90 * (1.0f - sample.metallic) +
            Color::Ones() * sample.metallic;
        sample.diffuse_color = sample.base_color * (1.0f - sample.metallic);
        sample.diffuse_fresnel_f0 = dielectric_f0;
        sample.diffuse_fresnel_f90 = dielectric_f90;
        sample.diffuse_fresnel_uses_max = true;
    }
    sample.occlusion = 1.0f;
    if (valid_texture_id(scene, material.occlusion_texture_id)) {
        const float value = scene.textures[
            static_cast<std::size_t>(material.occlusion_texture_id)].sample(
                texture_uv(hit, material.occlusion_texture_transform)).x();
        sample.occlusion = std::lerp(
            1.0f,
            std::clamp(value, 0.0f, 1.0f),
            std::clamp(material.occlusion_strength, 0.0f, 1.0f));
    }
    sample.emission = material.emission;
    if (valid_texture_id(scene, material.emissive_texture_id)) {
        sample.emission = sample.emission.cwiseProduct(
            scene.textures[static_cast<std::size_t>(material.emissive_texture_id)].sample(
                texture_uv(hit, material.emissive_texture_transform)));
    }
    sample.shading_normal = bumped_normal(scene, material, hit);
    return sample;
}

}  // namespace renderer
