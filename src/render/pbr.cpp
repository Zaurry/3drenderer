#include "render/pbr.h"

#include "render/mis_weight.h"
#include "render/shading_constants.h"

#include <algorithm>
#include <cmath>

namespace renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInversePi = 0.31830988618379067154f;

float saturate(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

Color fresnel_schlick(float cosine, const Color& f0, const Color& f90) {
    const float factor = std::pow(1.0f - saturate(cosine), 5.0f);
    return f0 + (f90 - f0) * factor;
}

float ggx_distribution(float n_dot_h, float alpha) {
    const float alpha_squared = alpha * alpha;
    const float denominator =
        n_dot_h * n_dot_h * (alpha_squared - 1.0f) + 1.0f;
    return alpha_squared /
        std::max(kPi * denominator * denominator, 1.0e-12f);
}

float smith_g1(float n_dot_v, float alpha) {
    if (n_dot_v <= 0.0f) {
        return 0.0f;
    }
    const float alpha_squared = alpha * alpha;
    const float tangent_squared =
        std::max(0.0f, (1.0f - n_dot_v * n_dot_v) / (n_dot_v * n_dot_v));
    return 2.0f /
        (1.0f + std::sqrt(1.0f + alpha_squared * tangent_squared));
}

float specular_probability(const PbrSurface& surface) {
    const Color luminance_weights(
        kLuminanceWeights[0],
        kLuminanceWeights[1],
        kLuminanceWeights[2]);
    const float diffuse_energy = std::max(0.0f, surface.diffuse_color.dot(luminance_weights));
    const float specular_energy = std::max(0.0f, surface.specular_f0.dot(luminance_weights));
    const float total = diffuse_energy + specular_energy;
    if (!(total > 0.0f)) {
        return 0.5f;
    }
    return std::clamp(specular_energy / total, 0.05f, 0.95f);
}

void basis(const Vec3& normal, Vec3& tangent, Vec3& bitangent) {
    const Vec3 helper = std::abs(normal.z()) < 0.999f
        ? Vec3(0.0f, 0.0f, 1.0f)
        : Vec3(1.0f, 0.0f, 0.0f);
    tangent = helper.cross(normal).normalized();
    bitangent = normal.cross(tangent);
}

Vec3 to_local(
    const Vec3& direction,
    const Vec3& tangent,
    const Vec3& bitangent,
    const Vec3& normal) {
    return Vec3(
        direction.dot(tangent),
        direction.dot(bitangent),
        direction.dot(normal));
}

Vec3 to_world(
    const Vec3& direction,
    const Vec3& tangent,
    const Vec3& bitangent,
    const Vec3& normal) {
    return (tangent * direction.x() + bitangent * direction.y() + normal * direction.z()).normalized();
}

Vec3 sample_visible_ggx(const Vec3& view, float alpha, float u1, float u2) {
    Vec3 stretched(alpha * view.x(), alpha * view.y(), view.z());
    stretched.normalize();
    const float lensq = stretched.x() * stretched.x() + stretched.y() * stretched.y();
    const Vec3 tangent1 = lensq > 1.0e-12f
        ? Vec3(-stretched.y(), stretched.x(), 0.0f) / std::sqrt(lensq)
        : Vec3(1.0f, 0.0f, 0.0f);
    const Vec3 tangent2 = stretched.cross(tangent1);
    const float radius = std::sqrt(saturate(u1));
    const float phi = 2.0f * kPi * saturate(u2);
    const float t1 = radius * std::cos(phi);
    float t2 = radius * std::sin(phi);
    const float blend = 0.5f * (1.0f + stretched.z());
    t2 = (1.0f - blend) * std::sqrt(std::max(0.0f, 1.0f - t1 * t1)) + blend * t2;
    const float t3 = std::sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2));
    const Vec3 normal = tangent1 * t1 + tangent2 * t2 + stretched * t3;
    return Vec3(alpha * normal.x(), alpha * normal.y(), std::max(0.0f, normal.z())).normalized();
}

}  // namespace

PbrEvaluation evaluate_pbr(
    const PbrSurface& input,
    const Vec3& normal,
    const Vec3& outgoing,
    const Vec3& incoming) {
    const PbrSurface surface{
        input.diffuse_color.cwiseMax(Color::Zero()),
        input.specular_f0.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        input.specular_f90.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        input.diffuse_fresnel_f0.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        input.diffuse_fresnel_f90.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        std::clamp(input.roughness, 0.02f, 1.0f),
        input.diffuse_fresnel_uses_max};
    const float n_dot_v = saturate(normal.dot(outgoing));
    const float n_dot_l = saturate(normal.dot(incoming));
    if (n_dot_v <= 0.0f || n_dot_l <= 0.0f) {
        return {};
    }
    const Vec3 half_vector_candidate = outgoing + incoming;
    if (half_vector_candidate.squaredNorm() <= 1.0e-20f) {
        return {};
    }
    const Vec3 half_vector = half_vector_candidate.normalized();
    const float n_dot_h = saturate(normal.dot(half_vector));
    const float v_dot_h = saturate(outgoing.dot(half_vector));
    const float alpha = surface.roughness * surface.roughness;
    const float distribution = ggx_distribution(n_dot_h, alpha);
    const float geometry = smith_g1(n_dot_v, alpha) * smith_g1(n_dot_l, alpha);
    const Color fresnel = fresnel_schlick(
        v_dot_h,
        surface.specular_f0,
        surface.specular_f90);
    const Color diffuse_fresnel = fresnel_schlick(
        v_dot_h,
        surface.diffuse_fresnel_f0,
        surface.diffuse_fresnel_f90);
    const Color specular = fresnel *
        (distribution * geometry / std::max(4.0f * n_dot_v * n_dot_l, 1.0e-12f));
    Color diffuse_weight;
    if (surface.diffuse_fresnel_uses_max) {
        diffuse_weight = Color::Constant(1.0f - diffuse_fresnel.maxCoeff());
    } else {
        diffuse_weight = Color::Ones() - diffuse_fresnel;
    }
    const Color diffuse = surface.diffuse_color.cwiseProduct(diffuse_weight) * kInversePi;
    const float diffuse_pdf = n_dot_l * kInversePi;
    const float specular_pdf = distribution * smith_g1(n_dot_v, alpha) /
        std::max(4.0f * n_dot_v, 1.0e-12f);
    const float probability = specular_probability(surface);
    return PbrEvaluation{
        diffuse + specular,
        (1.0f - probability) * diffuse_pdf + probability * specular_pdf};
}

PbrSample sample_pbr(
    const PbrSurface& input,
    const Vec3& normal,
    const Vec3& outgoing,
    float component_sample,
    float sample_x,
    float sample_y) {
    const PbrSurface surface{
        input.diffuse_color.cwiseMax(Color::Zero()),
        input.specular_f0.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        input.specular_f90.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        input.diffuse_fresnel_f0.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        input.diffuse_fresnel_f90.cwiseMax(Color::Zero()).cwiseMin(Color::Ones()),
        std::clamp(input.roughness, 0.02f, 1.0f),
        input.diffuse_fresnel_uses_max};
    if (normal.dot(outgoing) <= 0.0f) {
        return {};
    }
    Vec3 tangent;
    Vec3 bitangent;
    basis(normal, tangent, bitangent);
    Vec3 incoming;
    if (component_sample < specular_probability(surface)) {
        const Vec3 view_local = to_local(outgoing, tangent, bitangent, normal);
        const Vec3 half_local = sample_visible_ggx(
            view_local,
            surface.roughness * surface.roughness,
            sample_x,
            sample_y);
        const Vec3 half_world = to_world(half_local, tangent, bitangent, normal);
        incoming = (-outgoing + 2.0f * outgoing.dot(half_world) * half_world).normalized();
    } else {
        const float radius = std::sqrt(saturate(sample_x));
        const float phi = 2.0f * kPi * saturate(sample_y);
        const Vec3 local(
            radius * std::cos(phi),
            radius * std::sin(phi),
            std::sqrt(std::max(0.0f, 1.0f - sample_x)));
        incoming = to_world(local, tangent, bitangent, normal);
    }
    const PbrEvaluation evaluated = evaluate_pbr(surface, normal, outgoing, incoming);
    const float cosine = std::max(0.0f, normal.dot(incoming));
    if (!(evaluated.pdf > 0.0f) || cosine <= 0.0f || !evaluated.brdf.allFinite()) {
        return {};
    }
    return PbrSample{
        incoming,
        evaluated.brdf * (cosine / evaluated.pdf),
        evaluated.pdf,
        true};
}

}  // namespace renderer
