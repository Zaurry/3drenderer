#pragma once

#include "scene/mesh.h"

#include <string>

namespace renderer {

Mesh load_obj_mesh(const std::string& path, int material_id);

}  // namespace renderer
