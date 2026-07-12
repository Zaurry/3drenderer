#pragma once

#include "core/math/vec3.h"
#include "render/renderer.h"

namespace renderer {

class RasterizerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;

private:
    float edge_function(const Vec3& a, const Vec3& b, const Vec3& c) const;
};

}  // namespace renderer
