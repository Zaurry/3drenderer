#include "acceleration/bvh.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace renderer {

namespace {

constexpr int max_leaf_triangles = 4;
// Recursive construction halves an int count, leaving at most one sibling per bit of depth.
constexpr std::size_t max_traversal_stack = std::numeric_limits<unsigned int>::digits + 1;

}  // namespace

bool BvhNode::is_leaf() const {
    return count > 0;
}

void Bvh::build(const std::vector<Triangle>& triangles) {
    triangles_ = triangles;
    build_from(triangles_);
}

void Bvh::build_layout(const std::vector<Triangle>& triangles) {
    triangles_.clear();
    build_from(triangles);
}

void Bvh::build_from(const std::vector<Triangle>& triangles) {
    primitive_indices_.resize(triangles.size());
    std::iota(primitive_indices_.begin(), primitive_indices_.end(), 0);
    nodes_.clear();
    maximum_depth_ = 0;

    if (triangles.empty()) {
        return;
    }

    build_recursive(triangles, 0, static_cast<int>(triangles.size()), 1);
}

bool Bvh::intersect(const Ray& ray, float t_min, float t_max, HitRecord& hit) const {
    if (nodes_.empty()) {
        return false;
    }

    bool hit_anything = false;
    float closest_t = t_max;
    std::array<int, max_traversal_stack> stack{};
    std::size_t stack_size = 1;
    stack[0] = 0;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];

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
            stack[stack_size++] = node.left;
        }
        if (node.right >= 0) {
            stack[stack_size++] = node.right;
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

int Bvh::maximum_depth() const {
    return maximum_depth_;
}

int Bvh::build_recursive(
    const std::vector<Triangle>& triangles,
    int first,
    int count,
    int depth) {
    maximum_depth_ = std::max(maximum_depth_, depth);
    const int node_index = static_cast<int>(nodes_.size());
    nodes_.push_back(BvhNode());

    Bounds3 bounds;
    Bounds3 centroid_bounds;
    for (int i = 0; i < count; ++i) {
        const Triangle& triangle = triangles[primitive_indices_[first + i]];
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
        [&triangles, axis](int lhs, int rhs) {
            return triangles[lhs].centroid()[axis] <
                triangles[rhs].centroid()[axis];
        });

    const int left =
        build_recursive(triangles, first, mid - first, depth + 1);
    const int right =
        build_recursive(triangles, mid, first + count - mid, depth + 1);
    nodes_[node_index].left = left;
    nodes_[node_index].right = right;
    return node_index;
}

}  // namespace renderer
