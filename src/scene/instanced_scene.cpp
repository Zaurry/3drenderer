#include "scene/instanced_scene.h"

#include "scene/primitive.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace renderer {

namespace {

MaterialSlot snapshot_material_slot(int material_id, std::size_t count) {
    if (material_id == -1) {
        return MaterialSlot::missing();
    }
    if (material_id < -1 ||
        static_cast<std::size_t>(material_id) >= count) {
        throw std::runtime_error(
            "scene geometry references an invalid material slot");
    }
    return MaterialSlot::bound(static_cast<std::uint32_t>(material_id));
}

Bounds3 transformed_bounds(const Bounds3& bounds, const Mat4& matrix) {
    Bounds3 result;
    for (int mask = 0; mask < 8; ++mask) {
        const Vec3 corner(
            (mask & 1) ? bounds.max.x() : bounds.min.x(),
            (mask & 2) ? bounds.max.y() : bounds.min.y(),
            (mask & 4) ? bounds.max.z() : bounds.min.z());
        const Vec4 transformed = matrix * Vec4(
            corner.x(),
            corner.y(),
            corner.z(),
            1.0f);
        result.expand(transformed.head<3>() / transformed.w());
    }
    return result;
}

}  // namespace

const std::shared_ptr<const Scene>& canonical_unit_sphere_geometry() {
    static const std::shared_ptr<const Scene> geometry = [] {
        Scene unit;
        unit.materials.push_back(diagnostic_material());
        unit.spheres.emplace_back(Vec3::Zero(), 1.0f, 0);
        tessellate_spheres(unit, 64, 32);
        return std::make_shared<const Scene>(std::move(unit));
    }();
    return geometry;
}

RenderSceneSnapshot make_render_scene_snapshot(Scene scene) {
    auto geometry = std::make_shared<Scene>(std::move(scene));
    std::vector<Sphere> spheres = std::move(geometry->spheres);
    geometry->spheres.clear();

    RenderSceneSnapshot snapshot;
    snapshot.revisions = geometry->revisions.empty()
        ? SceneRevisions{1, 1, 1, 1, 1, 1, 1, 1}
        : geometry->revisions;
    snapshot.textures = geometry->textures;
    snapshot.point_lights = geometry->point_lights;
    snapshot.directional_lights = geometry->directional_lights;
    snapshot.spot_lights = geometry->spot_lights;
    snapshot.environment = geometry->environment;
    snapshot.environment_map = geometry->environment_map;
    snapshot.environment_intensity = geometry->environment_intensity;
    snapshot.environment_rotation_degrees =
        geometry->environment_rotation_degrees;
    snapshot.environment_background_visible =
        geometry->environment_background_visible;

    if (!geometry->triangles.empty()) {
        RenderSceneAssetSnapshot asset;
        asset.asset_id = 1;
        asset.geometry_revision = snapshot.revisions.geometry;
        asset.local_scene = geometry;
        asset.triangle_material_slots.reserve(geometry->triangles.size());
        for (const Triangle& triangle : geometry->triangles) {
            asset.local_bounds.expand(triangle.bounds());
            asset.triangle_material_slots.push_back(snapshot_material_slot(
                triangle.material_id(),
                geometry->materials.size()));
        }
        snapshot.assets.push_back(std::move(asset));

        RenderSceneInstanceSnapshot instance;
        instance.object_id = 1;
        instance.asset_index = 0;
        instance.world_bounds = snapshot.assets.front().local_bounds;
        instance.materials = geometry->materials;
        snapshot.instances.push_back(std::move(instance));
    }

    if (!spheres.empty()) {
        const auto& unit_geometry = canonical_unit_sphere_geometry();
        RenderSceneAssetSnapshot sphere_asset;
        sphere_asset.asset_id = std::numeric_limits<std::uint64_t>::max();
        sphere_asset.geometry_revision = 1;
        sphere_asset.local_scene = unit_geometry;
        sphere_asset.local_bounds = Bounds3(
            -Vec3::Ones(),
            Vec3::Ones());
        sphere_asset.triangle_material_slots.assign(
            unit_geometry->triangles.size(),
            MaterialSlot::bound(0));
        const int sphere_asset_index =
            static_cast<int>(snapshot.assets.size());
        snapshot.assets.push_back(std::move(sphere_asset));

        for (const Sphere& sphere : spheres) {
            const MaterialSlot source_slot = snapshot_material_slot(
                sphere.material_id(),
                geometry->materials.size());
            RenderSceneInstanceSnapshot instance;
            instance.object_id =
                static_cast<std::uint64_t>(snapshot.instances.size() + 1);
            instance.asset_index = sphere_asset_index;
            instance.object_to_world = Mat4::Identity();
            instance.object_to_world.topLeftCorner<3, 3>() *= sphere.radius();
            instance.object_to_world.topRightCorner<3, 1>() = sphere.center();
            instance.world_to_object = instance.object_to_world.inverse();
            instance.normal_to_world = instance.object_to_world
                .topLeftCorner<3, 3>()
                .inverse()
                .transpose();
            instance.world_bounds = transformed_bounds(
                snapshot.assets[static_cast<std::size_t>(sphere_asset_index)]
                    .local_bounds,
                instance.object_to_world);
            instance.materials.push_back(
                source_slot.has_value()
                    ? geometry->materials[source_slot.value()]
                    : diagnostic_material());
            snapshot.instances.push_back(std::move(instance));
        }
    }
    return snapshot;
}

Scene flatten_render_scene_snapshot(
    const RenderSceneSnapshot& snapshot) {
    Scene result;
    result.revisions = snapshot.revisions;
    result.textures = snapshot.textures;
    result.point_lights = snapshot.point_lights;
    result.directional_lights = snapshot.directional_lights;
    result.spot_lights = snapshot.spot_lights;
    result.environment = snapshot.environment;
    result.environment_map = snapshot.environment_map;
    result.environment_intensity = snapshot.environment_intensity;
    result.environment_rotation_degrees =
        snapshot.environment_rotation_degrees;
    result.environment_background_visible =
        snapshot.environment_background_visible;

    for (const RenderSceneInstanceSnapshot& instance : snapshot.instances) {
        if (instance.asset_index < 0 ||
            static_cast<std::size_t>(instance.asset_index) >=
                snapshot.assets.size()) {
            throw std::runtime_error(
                "render snapshot instance references an invalid asset");
        }
        const RenderSceneAssetSnapshot& asset =
            snapshot.assets[static_cast<std::size_t>(instance.asset_index)];
        if (!asset.local_scene) {
            throw std::runtime_error(
                "render snapshot asset has no geometry");
        }
        if (!asset.local_scene->spheres.empty()) {
            throw std::runtime_error(
                "render snapshot assets must contain canonical mesh geometry");
        }
        if (asset.triangle_material_slots.size() !=
            asset.local_scene->triangles.size()) {
            throw std::runtime_error(
                "render snapshot material-slot table does not match geometry");
        }
        if (!instance.object_to_world.allFinite() ||
            !instance.normal_to_world.allFinite()) {
            throw std::runtime_error(
                "render snapshot instance transform is not finite");
        }

        const int material_base = static_cast<int>(result.materials.size());
        result.materials.insert(
            result.materials.end(),
            instance.materials.begin(),
            instance.materials.end());
        const int diagnostic_material_id =
            static_cast<int>(result.materials.size());
        result.materials.push_back(diagnostic_material());

        const Mat3 linear =
            instance.object_to_world.topLeftCorner<3, 3>();
        const float determinant = linear.determinant();
        if (!std::isfinite(determinant) ||
            std::abs(determinant) < 1.0e-12f) {
            throw std::runtime_error(
                "render snapshot instance transform is singular");
        }
        const float orientation_sign = determinant < 0.0f ? -1.0f : 1.0f;
        for (std::size_t triangle_index = 0;
             triangle_index < asset.local_scene->triangles.size();
             ++triangle_index) {
            const Triangle& triangle =
                asset.local_scene->triangles[triangle_index];
            TriangleVertex vertices[3];
            for (int vertex_index = 0; vertex_index < 3; ++vertex_index) {
                const TriangleVertex& source = triangle.vertex(vertex_index);
                vertices[vertex_index] = source;
                const Vec4 position = instance.object_to_world * Vec4(
                    source.position.x(),
                    source.position.y(),
                    source.position.z(),
                    1.0f);
                vertices[vertex_index].position = position.head<3>() / position.w();
                if (source.has_normal) {
                    const Vec3 normal = instance.normal_to_world * source.normal;
                    vertices[vertex_index].has_normal = usable_direction(normal);
                    vertices[vertex_index].normal =
                        vertices[vertex_index].has_normal
                        ? normal.normalized()
                        : Vec3::Zero();
                }
                if (source.has_tangent) {
                    const Vec3 tangent = linear * source.tangent.head<3>();
                    vertices[vertex_index].has_tangent = usable_direction(tangent);
                    vertices[vertex_index].tangent =
                        vertices[vertex_index].has_tangent
                        ? Vec4(
                              tangent.normalized().x(),
                              tangent.normalized().y(),
                              tangent.normalized().z(),
                              source.tangent.w() * orientation_sign)
                        : Vec4::Zero();
                }
            }
            const MaterialSlot& slot =
                asset.triangle_material_slots[triangle_index];
            int material_id = diagnostic_material_id;
            if (slot.has_value()) {
                if (slot.value() >= instance.materials.size()) {
                    throw std::runtime_error(
                        "render snapshot material slot is out of range");
                }
                material_id = material_base + static_cast<int>(slot.value());
            }
            result.triangles.emplace_back(
                vertices[0],
                vertices[1],
                vertices[2],
                material_id);
        }
    }
    return result;
}

}  // namespace renderer
