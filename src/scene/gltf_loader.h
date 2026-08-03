#pragma once

#include "scene/scene_asset_loader.h"

#include <filesystem>
#include <string>
#include <vector>

namespace renderer {

struct GltfNodeAsset {
    enum class LightType { None, Directional, Point, Spot };

    std::string name;
    int mesh_index = -1;
    int parent_index = -1;
    Mat4 local_transform = Mat4::Identity();
    int camera_index = -1;
    LightType light_type = LightType::None;
    Color light_color = Color::Ones();
    float light_intensity = 1.0f;
    float light_range = 0.0f;
    float spot_inner_cone_radians = 0.0f;
    float spot_outer_cone_radians = 0.7853981634f;
};

struct GltfCameraAsset {
    std::string name;
    bool orthographic = false;
    float vertical_fov_radians = 0.7853981634f;
    float aspect_ratio = 0.0f;
    float x_magnification = 1.0f;
    float y_magnification = 1.0f;
    float near_plane = 0.01f;
    float far_plane = 1000.0f;
};

struct LoadedGltfScene {
    std::vector<LoadedScene> meshes;
    std::vector<GltfNodeAsset> nodes;
    std::vector<GltfCameraAsset> cameras;
    std::vector<std::string> warnings;
};

LoadedGltfScene load_gltf_scene(
    const std::filesystem::path& path,
    int width,
    int height);

}  // namespace renderer
