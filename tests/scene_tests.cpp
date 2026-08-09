#include "test_framework.h"

#include "render/interactive/interactive_render_session.h"
#include "scene/scene_document.h"

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
