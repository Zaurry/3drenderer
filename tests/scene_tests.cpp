#include "test_framework.h"

#include "render/interactive/interactive_render_session.h"
#include "scene/scene_document.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>

int main() {
    renderer::Scene scene;
    scene.materials.push_back(renderer::Material{});
    scene.triangles.emplace_back(
        renderer::Vec3(-1.0f, -1.0f, 0.0f),
        renderer::Vec3(1.0f, -1.0f, 0.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        -1);
    renderer::SceneDocument document = renderer::SceneDocument::from_scene(
        std::move(scene),
        "Missing material");
    const renderer::ObjectId object_id = document.objects().front().id;
    RENDER_CHECK(document.duplicate_subtree(object_id) !=
        renderer::kInvalidObjectId);
    const renderer::RenderSceneSnapshot& snapshot =
        document.render_scene_snapshot();
    RENDER_CHECK(snapshot.instances.size() == 2);
    RENDER_CHECK(snapshot.assets.size() == 1);
    RENDER_CHECK(
        !snapshot.assets[0].triangle_material_slots[0].has_value());
    for (const renderer::Triangle& triangle : renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles) {
        RENDER_CHECK(triangle.material_id() >= 0);
        RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot())
            .materials[static_cast<std::size_t>(triangle.material_id())]
            .base_color.isApprox(renderer::Color(1.0f, 0.0f, 1.0f)));
    }

    const renderer::Mat4 original = document.world_matrix(object_id);
    renderer::Mat4 first = original;
    first(0, 3) = 1.0f;
    {
        auto edit = document.begin_edit("drag");
        RENDER_CHECK(document.set_world_matrix(object_id, first));
        edit.commit();
    }
    renderer::Mat4 second = original;
    second(0, 3) = 2.0f;
    {
        auto edit = document.begin_edit("drag");
        RENDER_CHECK(document.set_world_matrix(object_id, second));
        edit.commit();
    }
    RENDER_CHECK(document.undo());
    RENDER_CHECK(document.world_matrix(object_id).isApprox(original));
    RENDER_CHECK(document.redo());
    RENDER_CHECK(document.world_matrix(object_id).isApprox(second));

    renderer::SceneTransform millimeter_transform;
    millimeter_transform.local_matrix = renderer::Mat4::Identity();
    millimeter_transform.local_matrix.topLeftCorner<3, 3>() *= 0.001f;
    RENDER_CHECK(millimeter_transform.valid());
    renderer::SceneTransform singular_transform = millimeter_transform;
    singular_transform.local_matrix.row(0).head<3>().setZero();
    RENDER_CHECK(!singular_transform.valid());
    RENDER_CHECK(document.set_world_matrix(
        object_id,
        millimeter_transform.local_matrix));

    renderer::Scene sphere_scene;
    sphere_scene.materials.push_back(renderer::Material{});
    sphere_scene.spheres.emplace_back(
        renderer::Vec3(-2.0f, 0.0f, 0.0f),
        0.5f,
        0);
    sphere_scene.spheres.emplace_back(
        renderer::Vec3(3.0f, 1.0f, -1.0f),
        2.0f,
        0);
    const renderer::RenderSceneSnapshot sphere_snapshot =
        renderer::make_render_scene_snapshot(std::move(sphere_scene));
    RENDER_CHECK(sphere_snapshot.assets.size() == 1);
    RENDER_CHECK(sphere_snapshot.instances.size() == 2);
    RENDER_CHECK(
        sphere_snapshot.instances[0].asset_index ==
        sphere_snapshot.instances[1].asset_index);
    RENDER_CHECK(
        sphere_snapshot.assets[0].local_scene ==
        renderer::canonical_unit_sphere_geometry());
    RENDER_CHECK((sphere_snapshot.instances[0].object_to_world
        .topRightCorner<3, 1>()
        .isApprox(renderer::Vec3(-2.0f, 0.0f, 0.0f))));
    RENDER_CHECK(nearly_equal(
        sphere_snapshot.instances[1].object_to_world(0, 0),
        2.0f));

    renderer::SceneDocument rectangle_document;
    const renderer::ObjectId rectangle_id =
        rectangle_document.create_rect_area_light(
            "Key Rectangle",
            renderer::Vec3(1.0f, 2.0f, 3.0f),
            renderer::Vec3(0.0f, -1.0f, 0.0f),
            renderer::Color(8.0f, 4.0f, 2.0f),
            2.0f,
            4.0f,
            true);
    renderer::SceneLightProperties rectangle_properties;
    rectangle_properties.color = renderer::Color(8.0f, 4.0f, 2.0f);
    rectangle_properties.casts_shadows = false;
    rectangle_properties.shadow_priority = 7;
    rectangle_properties.area_width = 2.0f;
    rectangle_properties.area_height = 4.0f;
    rectangle_properties.two_sided = true;
    RENDER_CHECK(rectangle_document.set_light_properties(
        rectangle_id,
        rectangle_properties));
    rectangle_document.checkpoint();
    const renderer::SceneRevisions before_rectangle_color =
        rectangle_document.revisions();
    rectangle_properties.color = renderer::Color(9.0f, 4.0f, 2.0f);
    RENDER_CHECK(rectangle_document.set_light_properties(
        rectangle_id,
        rectangle_properties));
    RENDER_CHECK(
        rectangle_document.revisions().materials ==
        before_rectangle_color.materials + 1);
    RENDER_CHECK(
        rectangle_document.revisions().transforms ==
        before_rectangle_color.transforms);
    rectangle_document.checkpoint();
    const renderer::SceneRevisions before_rectangle_size =
        rectangle_document.revisions();
    rectangle_properties.area_width = 2.5f;
    RENDER_CHECK(rectangle_document.set_light_properties(
        rectangle_id,
        rectangle_properties));
    RENDER_CHECK(
        rectangle_document.revisions().transforms ==
        before_rectangle_size.transforms + 1);
    RENDER_CHECK(
        rectangle_document.revisions().materials ==
        before_rectangle_size.materials);
    rectangle_document.checkpoint();

    const renderer::RenderSceneSnapshot& rectangle_snapshot =
        rectangle_document.render_scene_snapshot();
    RENDER_CHECK(rectangle_snapshot.rect_area_lights.size() == 1);
    RENDER_CHECK(rectangle_snapshot.assets.size() == 1);
    RENDER_CHECK(rectangle_snapshot.instances.size() == 1);
    RENDER_CHECK(rectangle_snapshot.assets[0].local_scene ==
        renderer::canonical_unit_quad_geometry());
    RENDER_CHECK(rectangle_snapshot.assets[0].local_scene->triangles.size() == 2);
    RENDER_CHECK(nearly_equal(
        rectangle_snapshot.rect_area_lights[0].axis_u.norm(),
        1.25f));
    RENDER_CHECK(nearly_equal(
        rectangle_snapshot.rect_area_lights[0].axis_v.norm(),
        2.0f));
    RENDER_CHECK(!rectangle_snapshot.rect_area_lights[0].casts_shadows);
    RENDER_CHECK(rectangle_snapshot.rect_area_lights[0].shadow_priority == 7);
    RENDER_CHECK(rectangle_snapshot.rect_area_lights[0].two_sided);
    RENDER_CHECK(rectangle_snapshot.instances[0].materials.size() == 1);
    RENDER_CHECK(
        rectangle_snapshot.instances[0].materials[0].type ==
        renderer::MaterialType::Emissive);
    RENDER_CHECK(rectangle_snapshot.instances[0].materials[0].emission.isApprox(
        rectangle_properties.color));
    RENDER_CHECK(rectangle_snapshot.instances[0].materials[0].two_sided);
    const renderer::Scene flattened_rectangle =
        renderer::flatten_render_scene_snapshot(rectangle_snapshot);
    RENDER_CHECK(flattened_rectangle.rect_area_lights.size() == 1);
    RENDER_CHECK(flattened_rectangle.triangles.size() == 2);
    RENDER_CHECK(flattened_rectangle.materials.size() == 2);

    renderer::RectAreaLight one_sided_rectangle =
        rectangle_snapshot.rect_area_lights[0];
    one_sided_rectangle.two_sided = false;
    const renderer::Vec3 rectangle_emission_direction =
        renderer::rect_area_light_emission_direction(one_sided_rectangle);
    RENDER_CHECK(rectangle_emission_direction.isApprox(
        renderer::Vec3(0.0f, -1.0f, 0.0f),
        1.0e-5f));
    RENDER_CHECK(renderer::rect_area_light_emits_toward(
        one_sided_rectangle,
        one_sided_rectangle.position + rectangle_emission_direction));
    RENDER_CHECK(!renderer::rect_area_light_emits_toward(
        one_sided_rectangle,
        one_sided_rectangle.position - rectangle_emission_direction));
    one_sided_rectangle.two_sided = true;
    RENDER_CHECK(renderer::rect_area_light_emits_toward(
        one_sided_rectangle,
        one_sided_rectangle.position - rectangle_emission_direction));

    renderer::SceneDocument transformed_rectangle_document;
    const renderer::ObjectId transformed_rectangle_id =
        transformed_rectangle_document.create_rect_area_light(
            "Transform Rectangle",
            renderer::Vec3::Zero(),
            -renderer::Vec3::UnitZ(),
            renderer::Color::Ones(),
            2.0f,
            4.0f,
            false);
    renderer::SceneTrs transformed_rectangle_trs;
    transformed_rectangle_trs.translation = renderer::Vec3(3.0f, -2.0f, 5.0f);
    transformed_rectangle_trs.rotation_degrees = renderer::Vec3(25.0f, 40.0f, -15.0f);
    transformed_rectangle_trs.scale = renderer::Vec3(1.5f, 0.75f, 2.0f);
    RENDER_CHECK(transformed_rectangle_document.set_local_trs(
        transformed_rectangle_id,
        transformed_rectangle_trs));
    const renderer::Mat4 transformed_rectangle_world =
        transformed_rectangle_trs.matrix();
    const renderer::RectAreaLight& transformed_rectangle =
        transformed_rectangle_document.render_scene_snapshot()
            .rect_area_lights.at(0);
    RENDER_CHECK(transformed_rectangle.position.isApprox(
        transformed_rectangle_trs.translation));
    RENDER_CHECK(transformed_rectangle.axis_u.isApprox(
        transformed_rectangle_world.topLeftCorner<3, 3>() *
        renderer::Vec3(1.0f, 0.0f, 0.0f)));
    RENDER_CHECK(transformed_rectangle.axis_v.isApprox(
        transformed_rectangle_world.topLeftCorner<3, 3>() *
        renderer::Vec3(0.0f, 2.0f, 0.0f)));
    const renderer::Scene transformed_rectangle_flat =
        renderer::flatten_render_scene_snapshot(
            transformed_rectangle_document.render_scene_snapshot());
    RENDER_CHECK(
        transformed_rectangle_flat.triangles.at(0).geometric_normal().isApprox(
            renderer::rect_area_light_emission_direction(transformed_rectangle),
            1.0e-5f));
    renderer::SceneTrs gizmo_rectangle_trs = transformed_rectangle_trs;
    gizmo_rectangle_trs.translation += renderer::Vec3(-1.0f, 4.0f, 2.0f);
    gizmo_rectangle_trs.rotation_degrees += renderer::Vec3(10.0f, -20.0f, 35.0f);
    gizmo_rectangle_trs.scale = renderer::Vec3(0.5f, 2.0f, 1.25f);
    renderer::Mat4 gizmo_rectangle_world = gizmo_rectangle_trs.matrix();
    gizmo_rectangle_world(3, 0) = 2.0e-7f;
    gizmo_rectangle_world(3, 2) = -3.0e-7f;
    gizmo_rectangle_world(3, 3) = 0.9999991f;
    RENDER_CHECK(transformed_rectangle_document.set_world_matrix(
        transformed_rectangle_id,
        gizmo_rectangle_world));
    RENDER_CHECK(
        transformed_rectangle_document.find(transformed_rectangle_id)
            ->transform.local_matrix.row(3).transpose().isApprox(
                renderer::Vec4(0.0f, 0.0f, 0.0f, 1.0f),
                0.0f));
    const renderer::RectAreaLight& gizmo_rectangle =
        transformed_rectangle_document.render_scene_snapshot()
            .rect_area_lights.at(0);
    RENDER_CHECK(gizmo_rectangle.position.isApprox(
        gizmo_rectangle_trs.translation));
    RENDER_CHECK(renderer::rect_area_light_emission_direction(gizmo_rectangle)
        .isApprox(
            (gizmo_rectangle_trs.matrix().topLeftCorner<3, 3>() *
             -renderer::Vec3::UnitZ()).normalized(),
            1.0e-5f));

    const renderer::ObjectId rectangle_copy =
        rectangle_document.duplicate_subtree(rectangle_id);
    RENDER_CHECK(rectangle_copy != renderer::kInvalidObjectId);
    RENDER_CHECK(
        rectangle_document.render_scene_snapshot().rect_area_lights.size() == 2);
    RENDER_CHECK(rectangle_document.undo());
    RENDER_CHECK(
        rectangle_document.render_scene_snapshot().rect_area_lights.size() == 1);
    RENDER_CHECK(rectangle_document.redo());
    RENDER_CHECK(
        rectangle_document.render_scene_snapshot().rect_area_lights.size() == 2);
    RENDER_CHECK(rectangle_document.set_object_visible(rectangle_copy, false));
    rectangle_document.checkpoint();
    RENDER_CHECK(
        rectangle_document.render_scene_snapshot().rect_area_lights.size() == 1);

    const renderer::SceneDocument restored_rectangle =
        renderer::SceneDocument::from_session_snapshot(
            rectangle_document.session_snapshot(),
            64,
            64);
    const renderer::SceneObject* restored_rectangle_object =
        restored_rectangle.find(rectangle_id);
    RENDER_CHECK(restored_rectangle_object != nullptr);
    RENDER_CHECK(
        restored_rectangle_object->type == renderer::SceneObjectType::RectAreaLight);
    RENDER_CHECK(nearly_equal(restored_rectangle_object->area_width, 2.5f));
    RENDER_CHECK(nearly_equal(restored_rectangle_object->area_height, 4.0f));
    RENDER_CHECK(restored_rectangle_object->light_two_sided);
    RENDER_CHECK(!restored_rectangle_object->light_casts_shadows);
    RENDER_CHECK(restored_rectangle_object->light_shadow_priority == 7);

    const nlohmann::json rectangle_v5 = rectangle_document.session_snapshot();
    for (int legacy_version = 1; legacy_version <= 4; ++legacy_version) {
        nlohmann::json legacy = rectangle_v5;
        legacy["version"] = legacy_version;
        if (legacy_version < 3) {
            legacy["environment"] = rectangle_v5["environment"]["color"];
        }
        for (auto& object : legacy["objects"]) {
            object.erase("light_source_radius");
            object.erase("directional_angular_radius_radians");
            object.erase("light_casts_shadows");
            object.erase("light_shadow_priority");
            object.erase("area_width");
            object.erase("area_height");
            object.erase("light_two_sided");
            if (legacy_version < 4) {
                object.erase("local_matrix");
                object["translation"] = {0.0f, 0.0f, 0.0f};
                object["rotation_degrees"] = {0.0f, 0.0f, 0.0f};
                object["scale"] = {1.0f, 1.0f, 1.0f};
            }
        }
        const renderer::SceneDocument migrated =
            renderer::SceneDocument::from_session_snapshot(legacy, 64, 64);
        const renderer::SceneObject* migrated_rectangle =
            migrated.find(rectangle_id);
        RENDER_CHECK(migrated_rectangle != nullptr);
        RENDER_CHECK(nearly_equal(migrated_rectangle->light_source_radius, 0.05f));
        RENDER_CHECK(nearly_equal(
            migrated_rectangle->directional_angular_radius_radians,
            0.00464257581f));
        RENDER_CHECK(migrated_rectangle->light_casts_shadows);
        RENDER_CHECK(migrated_rectangle->light_shadow_priority == 0);
        RENDER_CHECK(nearly_equal(migrated_rectangle->area_width, 1.0f));
        RENDER_CHECK(nearly_equal(migrated_rectangle->area_height, 1.0f));
        RENDER_CHECK(!migrated_rectangle->light_two_sided);
    }

    renderer::Scene oriented_rectangle_scene;
    renderer::RectAreaLight oriented_rectangle;
    oriented_rectangle.position = renderer::Vec3(-1.0f, 3.0f, 2.0f);
    oriented_rectangle.axis_u = renderer::Vec3(0.0f, 0.0f, 1.5f);
    oriented_rectangle.axis_v = renderer::Vec3(0.0f, 2.0f, 0.0f);
    oriented_rectangle.radiance = renderer::Color(3.0f, 5.0f, 7.0f);
    oriented_rectangle_scene.rect_area_lights.push_back(oriented_rectangle);
    renderer::SceneDocument oriented_rectangle_document =
        renderer::SceneDocument::from_scene(
            std::move(oriented_rectangle_scene),
            "Oriented rectangle");
    const renderer::RectAreaLight& restored_orientation =
        oriented_rectangle_document.render_scene_snapshot().rect_area_lights.at(0);
    RENDER_CHECK(restored_orientation.position.isApprox(
        oriented_rectangle.position));
    RENDER_CHECK(restored_orientation.axis_u.isApprox(
        oriented_rectangle.axis_u));
    RENDER_CHECK(restored_orientation.axis_v.isApprox(
        oriented_rectangle.axis_v));

    renderer::Scene typed_light_scene;
    renderer::PointLight source_point;
    source_point.position = renderer::Vec3(1.0f, 2.0f, 3.0f);
    source_point.intensity = renderer::Color(4.0f, 5.0f, 6.0f);
    source_point.range = 9.0f;
    source_point.source_radius = 0.2f;
    source_point.casts_shadows = false;
    source_point.shadow_priority = 2;
    typed_light_scene.point_lights.push_back(source_point);
    renderer::DirectionalLight source_directional;
    source_directional.direction = renderer::Vec3(-1.0f, -2.0f, -3.0f).normalized();
    source_directional.radiance = renderer::Color(2.0f, 3.0f, 4.0f);
    source_directional.angular_radius_radians = 0.01f;
    source_directional.casts_shadows = false;
    source_directional.shadow_priority = 3;
    typed_light_scene.directional_lights.push_back(source_directional);
    renderer::SpotLight source_spot;
    source_spot.position = renderer::Vec3(-2.0f, 1.0f, 4.0f);
    source_spot.direction = renderer::Vec3(0.0f, -1.0f, -1.0f).normalized();
    source_spot.intensity = renderer::Color(7.0f, 6.0f, 5.0f);
    source_spot.range = 12.0f;
    source_spot.inner_cone_radians = 0.2f;
    source_spot.outer_cone_radians = 0.4f;
    source_spot.source_radius = 0.3f;
    source_spot.casts_shadows = false;
    source_spot.shadow_priority = 4;
    typed_light_scene.spot_lights.push_back(source_spot);
    const renderer::SceneDocument typed_light_document =
        renderer::SceneDocument::from_scene(
            std::move(typed_light_scene),
            "Typed lights");
    const renderer::RenderSceneSnapshot& typed_light_snapshot =
        typed_light_document.render_scene_snapshot();
    RENDER_CHECK(typed_light_snapshot.point_lights.size() == 1);
    RENDER_CHECK(nearly_equal(
        typed_light_snapshot.point_lights[0].source_radius,
        source_point.source_radius));
    RENDER_CHECK(!typed_light_snapshot.point_lights[0].casts_shadows);
    RENDER_CHECK(typed_light_snapshot.point_lights[0].shadow_priority == 2);
    RENDER_CHECK(typed_light_snapshot.directional_lights.size() == 1);
    RENDER_CHECK(nearly_equal(
        typed_light_snapshot.directional_lights[0].angular_radius_radians,
        source_directional.angular_radius_radians));
    RENDER_CHECK(!typed_light_snapshot.directional_lights[0].casts_shadows);
    RENDER_CHECK(typed_light_snapshot.directional_lights[0].shadow_priority == 3);
    RENDER_CHECK(typed_light_snapshot.spot_lights.size() == 1);
    RENDER_CHECK(nearly_equal(
        typed_light_snapshot.spot_lights[0].source_radius,
        source_spot.source_radius));
    RENDER_CHECK(!typed_light_snapshot.spot_lights[0].casts_shadows);
    RENDER_CHECK(typed_light_snapshot.spot_lights[0].shadow_priority == 4);

    renderer::SceneDocument another_document;
    const renderer::RenderSceneSnapshot& another_snapshot =
        another_document.render_scene_snapshot();
    RENDER_CHECK(snapshot.source_id != 0);
    RENDER_CHECK(another_snapshot.source_id != 0);
    RENDER_CHECK(snapshot.source_id != another_snapshot.source_id);
    RENDER_CHECK(
        renderer::scene_changes_for_snapshot(
            snapshot.source_id,
            another_snapshot.revisions,
            true,
            another_snapshot) == renderer::SceneChange::All);
    RENDER_CHECK(
        renderer::scene_changes_for_snapshot(
            another_snapshot.source_id,
            another_snapshot.revisions,
            true,
            another_snapshot) == renderer::SceneChange::None);
    RENDER_CHECK(
        renderer::scene_changes_for_snapshot(
            another_snapshot.source_id,
            another_snapshot.revisions,
            true,
            another_snapshot,
            renderer::SceneChange::Materials) ==
        renderer::SceneChange::None);
    renderer::RenderSceneSnapshot legacy_snapshot = another_snapshot;
    legacy_snapshot.source_id = 0;
    RENDER_CHECK(renderer::has_scene_change(
        renderer::scene_changes_for_snapshot(
            0,
            renderer::SceneRevisions{},
            true,
            legacy_snapshot,
            renderer::SceneChange::Materials),
        renderer::SceneChange::Materials));

    renderer::Scene invalid;
    invalid.materials.push_back(renderer::Material{});
    invalid.triangles.emplace_back(
        renderer::Vec3(-1.0f, -1.0f, 0.0f),
        renderer::Vec3(1.0f, -1.0f, 0.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        1);
    renderer::SceneDocument invalid_document =
        renderer::SceneDocument::from_scene(std::move(invalid), "Invalid");
    bool rejected = false;
    try {
        (void)invalid_document.render_scene_snapshot();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    RENDER_CHECK(rejected);
    std::cout << "scene_tests: all tests passed\n";
}
