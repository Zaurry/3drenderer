#include "acceleration/bvh.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace renderer {

namespace {

constexpr int max_leaf_triangles = 4;

}  // namespace

bool BvhNode::is_leaf() const {
    return count > 0;
}

void Bvh::build(const std::vector<Triangle>& triangles) {
    triangles_ = triangles;
    primitive_indices_.resize(triangles_.size());
    std::iota(primitive_indices_.begin(), primitive_indices_.end(), 0);
    nodes_.clear();

    if (triangles_.empty()) {
        return;
    }

    build_recursive(0, static_cast<int>(triangles_.size()));
}

bool Bvh::intersect(const Ray& ray, float t_min, float t_max, HitRecord& hit) const {
    if (nodes_.empty()) {
        return false;
    }

    bool hit_anything = false;
    float closest_t = t_max;
    std::vector<int> stack;
    stack.push_back(0);

    while (!stack.empty()) {
        const int node_index = stack.back();
        stack.pop_back();

        const BvhNode& node = nodes_[node_index];
        // BVH 先用 AABB 测试整组三角形：如果光线没有打到包围盒，就能一次性跳过盒内所有三角形。
        // 这样很多光线不再逐个检查场景里的每个三角形，而是快速拒绝大片空间中的几何体。
        if (!node.bounds.intersect(ray, t_min, closest_t)) {
            continue;
        }

        if (node.is_leaf()) {
            for (int i = 0; i < node.count; ++i) {
                const int primitive_index = primitive_indices_[node.first + i];
                HitRecord candidate;
                if (triangles_[primitive_index].intersect(ray, t_min, closest_t, candidate)) {
                    hit_anything = true;
                    closest_t = candidate.t;
                    hit = candidate;
                }
            }
            continue;
        }

        if (node.left >= 0) {
            stack.push_back(node.left);
        }
        if (node.right >= 0) {
            stack.push_back(node.right);
        }
    }

    return hit_anything;
}

const std::vector<int>& Bvh::primitive_indices() const {
    return primitive_indices_;
}

const std::vector<BvhNode>& Bvh::nodes() const {
    return nodes_;
}

int Bvh::build_recursive(int first, int count) {
    const int node_index = static_cast<int>(nodes_.size());
    nodes_.push_back(BvhNode());

    Bounds3 bounds;
    Bounds3 centroid_bounds;
    for (int i = 0; i < count; ++i) {
        const Triangle& triangle = triangles_[primitive_indices_[first + i]];
        bounds.expand(triangle.bounds());
        centroid_bounds.expand(triangle.centroid());
    }

    nodes_[node_index].bounds = bounds;

    if (count <= max_leaf_triangles) {
        BvhNode& node = nodes_[node_index];
        node.first = first;
        node.count = count;
        return node_index;
    }

    const int axis = centroid_bounds.longest_axis();
    const int mid = first + count / 2;
    auto begin = primitive_indices_.begin();
    std::nth_element(
        begin + first,
        begin + mid,
        begin + first + count,
        [this, axis](int lhs, int rhs) {
            return triangles_[lhs].centroid()[axis] < triangles_[rhs].centroid()[axis];
        });

    const int left = build_recursive(first, mid - first);
    const int right = build_recursive(mid, first + count - mid);
    nodes_[node_index].left = left;
    nodes_[node_index].right = right;
    return node_index;
}

}  // namespace renderer
