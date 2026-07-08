#pragma once

#include "core/math/bounds.h"
#include "scene/camera.h"
#include "scene/scene.h"

#include <string>

namespace renderer {

struct LoadedScene {
    Scene scene;
    Camera camera;
    Bounds3 bounds;
};

LoadedScene load_scene_asset(const std::string& path, int width, int height);

}  // namespace renderer
