#pragma once

#include <cstdint>

namespace renderer {

enum class SceneRevisionDomain : std::uint32_t {
    None = 0,
    Topology = 1U << 0U,
    Geometry = 1U << 1U,
    Transforms = 1U << 2U,
    MaterialBindings = 1U << 3U,
    Materials = 1U << 4U,
    Textures = 1U << 5U,
    Lighting = 1U << 6U,
    Environment = 1U << 7U,
    All = (1U << 8U) - 1U,
};

constexpr SceneRevisionDomain operator|(
    SceneRevisionDomain left,
    SceneRevisionDomain right) {
    return static_cast<SceneRevisionDomain>(
        static_cast<std::uint32_t>(left) |
        static_cast<std::uint32_t>(right));
}

constexpr SceneRevisionDomain& operator|=(
    SceneRevisionDomain& left,
    SceneRevisionDomain right) {
    left = left | right;
    return left;
}

constexpr bool has_revision_domain(
    SceneRevisionDomain domains,
    SceneRevisionDomain domain) {
    return (static_cast<std::uint32_t>(domains) &
            static_cast<std::uint32_t>(domain)) != 0U;
}

struct SceneRevisions {
    std::uint64_t topology = 0;
    std::uint64_t geometry = 0;
    std::uint64_t transforms = 0;
    std::uint64_t material_bindings = 0;
    std::uint64_t materials = 0;
    std::uint64_t textures = 0;
    std::uint64_t lighting = 0;
    std::uint64_t environment = 0;

    bool empty() const {
        return topology == 0 && geometry == 0 && transforms == 0 &&
            material_bindings == 0 && materials == 0 && textures == 0 &&
            lighting == 0 && environment == 0;
    }

    bool operator==(const SceneRevisions&) const = default;
};

inline void advance_scene_revisions(
    SceneRevisions& revisions,
    SceneRevisionDomain domains) {
    if (has_revision_domain(domains, SceneRevisionDomain::Topology)) {
        ++revisions.topology;
    }
    if (has_revision_domain(domains, SceneRevisionDomain::Geometry)) {
        ++revisions.geometry;
    }
    if (has_revision_domain(domains, SceneRevisionDomain::Transforms)) {
        ++revisions.transforms;
    }
    if (has_revision_domain(domains, SceneRevisionDomain::MaterialBindings)) {
        ++revisions.material_bindings;
    }
    if (has_revision_domain(domains, SceneRevisionDomain::Materials)) {
        ++revisions.materials;
    }
    if (has_revision_domain(domains, SceneRevisionDomain::Textures)) {
        ++revisions.textures;
    }
    if (has_revision_domain(domains, SceneRevisionDomain::Lighting)) {
        ++revisions.lighting;
    }
    if (has_revision_domain(domains, SceneRevisionDomain::Environment)) {
        ++revisions.environment;
    }
}

}  // namespace renderer
