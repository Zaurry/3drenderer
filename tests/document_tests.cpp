#include "test_framework.h"

#include "core/io/atomic_file.h"
#include "scene/scene_document.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

RENDER_TEST(test_atomic_write_failure_preserves_original_bytes) {
    const std::filesystem::path directory = "document_contract_tests";
    const std::filesystem::path atomic_path = directory / "atomic.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const std::string original = "original bytes\n";
    renderer::write_file_atomically(atomic_path, original);
    renderer::set_atomic_write_failure_for_testing(
        renderer::AtomicWriteFailurePoint::BeforeReplace);
    bool failed = false;
    try {
        renderer::write_file_atomically(atomic_path, "replacement\n");
    } catch (const std::runtime_error&) {
        failed = true;
    }
    RENDER_CHECK(failed);
    std::ifstream atomic_input(atomic_path, std::ios::binary);
    const std::string actual{
        std::istreambuf_iterator<char>(atomic_input),
        std::istreambuf_iterator<char>()};
    atomic_input.close();
    RENDER_CHECK(actual == original);
    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_document_import_reparent_save_load_roundtrip) {
    const std::filesystem::path directory = "document_contract_tests";
    const std::filesystem::path obj_path = directory / "triangle.obj";
    const std::filesystem::path scene_path = directory / "matrix.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    {
        std::ofstream obj(obj_path);
        obj << "v -1 -1 0\n";
        obj << "v 1 -1 0\n";
        obj << "v 0 1 0\n";
        obj << "f 1 2 3\n";
    }
    renderer::SceneDocument document;
    const auto imported = document.import_path(obj_path, 64, 64);
    RENDER_CHECK(imported.size() == 1);
    const renderer::ObjectId mesh_id = imported.front();
    const renderer::ObjectId parent_id = document.create_group("Parent");
    renderer::Mat4 parent = renderer::Mat4::Identity();
    parent(0, 3) = -2.0f;
    RENDER_CHECK(document.set_world_matrix(parent_id, parent));
    document.checkpoint();
    renderer::Mat4 shear = renderer::Mat4::Identity();
    shear(0, 1) = 0.35f;
    shear(1, 2) = -0.2f;
    shear(2, 3) = -4.0f;
    RENDER_CHECK(document.set_world_matrix(mesh_id, shear));
    document.checkpoint();
    RENDER_CHECK(!document.local_trs(mesh_id).has_value());
    RENDER_CHECK(document.reparent(mesh_id, parent_id));
    RENDER_CHECK((document.world_matrix(mesh_id) - shear).cwiseAbs().maxCoeff() < 1.0e-5f);
    RENDER_CHECK(document.undo());
    RENDER_CHECK((document.world_matrix(mesh_id) - shear).cwiseAbs().maxCoeff() < 1.0e-5f);
    RENDER_CHECK(document.redo());
    document.save(scene_path);
    renderer::SceneDocument restored =
        renderer::SceneDocument::load(scene_path, 64, 64);
    RENDER_CHECK((restored.world_matrix(mesh_id) - shear).cwiseAbs().maxCoeff() < 1.0e-5f);
    RENDER_CHECK(!restored.local_trs(mesh_id).has_value());

    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_transaction_rejects_undo_and_redo) {
    renderer::SceneDocument document;
    auto edit = document.begin_edit();
    bool rejected = false;
    try {
        (void)document.undo();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    RENDER_CHECK(rejected);
    rejected = false;
    try {
        (void)document.redo();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    RENDER_CHECK(rejected);
    edit.cancel();
    RENDER_CHECK(!document.can_undo());
    RENDER_CHECK(!document.can_redo());
}

RENDER_TEST(test_undo_redo_import_manages_asset_lifetime) {
    const std::filesystem::path directory = "document_import_lifetime_tests";
    const std::filesystem::path obj_path = directory / "triangle.obj";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    {
        std::ofstream obj(obj_path);
        obj << "v -1 -1 0\n";
        obj << "v 1 -1 0\n";
        obj << "v 0 1 0\n";
        obj << "f 1 2 3\n";
    }

    renderer::SceneDocument document;
    document.import_path(obj_path, 64, 64);
    RENDER_CHECK(document.assets().size() == 1);
    RENDER_CHECK(document.render_scene_snapshot().instances.size() == 1);

    // Undoing the import detaches the orphaned asset from the live asset
    // list but records it with the history slot so redo can resurrect it.
    RENDER_CHECK(document.undo());
    RENDER_CHECK(document.assets().size() == 0);
    RENDER_CHECK(document.render_scene_snapshot().instances.empty());

    RENDER_CHECK(document.redo());
    RENDER_CHECK(document.assets().size() == 1);
    RENDER_CHECK(document.render_scene_snapshot().instances.size() == 1);
    RENDER_CHECK(
        document.render_scene_snapshot().instances[0].asset_index >= 0);

    // The snapshot geometry must be usable after the redo round-trip.
    const renderer::Scene flattened =
        renderer::flatten_render_scene_snapshot(
            document.render_scene_snapshot());
    RENDER_CHECK(flattened.triangles.size() == 1);

    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_document_hierarchy_depth_limit_is_enforced) {
    renderer::SceneDocument document;
    renderer::ObjectId parent = renderer::kInvalidObjectId;
    // One more level than the 1024 cap.
    for (int index = 0; index < 1025; ++index) {
        parent = document.create_group("chain", parent);
    }
    bool rejected = false;
    try {
        (void)document.render_scene_snapshot();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    RENDER_CHECK(rejected);
}

RENDER_TEST(test_document_reparent_rejects_over_deep_parent) {
    renderer::SceneDocument document;
    renderer::ObjectId deepest = renderer::kInvalidObjectId;
    // A legal chain that fills the depth cap exactly (depths 0..1023).
    for (int index = 0; index < 1024; ++index) {
        deepest = document.create_group("chain", deepest);
    }
    const renderer::ObjectId leaf = document.create_group("leaf");
    RENDER_CHECK(!document.reparent(leaf, deepest));

    // One level below the cap still accepts a child.
    renderer::SceneDocument shallow_document;
    renderer::ObjectId shallow_deepest = renderer::kInvalidObjectId;
    for (int index = 0; index < 1023; ++index) {
        shallow_deepest = shallow_document.create_group("chain", shallow_deepest);
    }
    const renderer::ObjectId shallow_leaf =
        shallow_document.create_group("leaf");
    RENDER_CHECK(shallow_document.reparent(shallow_leaf, shallow_deepest));
}
