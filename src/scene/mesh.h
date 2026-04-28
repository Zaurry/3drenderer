#pragma once

#include "scene/primitive.h"

#include <vector>

namespace renderer {

struct Mesh {
    std::vector<Triangle> triangles;
};

}  // namespace renderer
