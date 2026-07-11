#pragma once

#include "core/math/bounds.h"
#include "core/math/ray.h"
#include "scene/primitive.h"

#include <vector>

namespace renderer {

struct BvhNode {
    Bounds3 bounds;
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;

    bool is_leaf() const;
};

class Bvh {
public:
    void build(const std::vector<Triangle>& triangles);
    bool intersect(const Ray& ray, float t_min, float t_max, HitRecord& hit) const;
    const std::vector<int>& primitive_indices() const;
    const std::vector<BvhNode>& nodes() const;

private:
    int build_recursive(int first, int count);

    std::vector<Triangle> triangles_;
    std::vector<int> primitive_indices_;
    std::vector<BvhNode> nodes_;
};

}  // namespace renderer
