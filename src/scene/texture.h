#pragma once

#include "core/color.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"

#include <cmath>
#include <string>
#include <vector>

namespace renderer {

struct Material;
struct Scene;

enum class TextureEncoding { Srgb, Linear };

struct ConstantTexture {
    Color color = Color::Zero();

    Color sample(const Vec2& uv, const Vec3& p) const {
        (void)uv;
        (void)p;
        return color;
    }
};

struct CheckerTexture {
    Color even = Color::Zero();
    Color odd = Color::Zero();
    float scale = 8.0f;

    Color sample(const Vec2& uv, const Vec3& p) const {
        (void)uv;
        const float checker = std::floor(p.x() * scale) + std::floor(p.y() * scale) + std::floor(p.z() * scale);
        return static_cast<int>(checker) % 2 == 0 ? even : odd;
    }
};

class ImageTexture {
public:
    ImageTexture() = default;
    ImageTexture(int width, int height, std::vector<Color> pixels);

    static ImageTexture load(
        const std::string& path,
        TextureEncoding encoding = TextureEncoding::Srgb);

    int width() const;
    int height() const;
    Color sample(const Vec2& uv) const;
    float sample_scalar(const Vec2& uv) const;
    Vec2 texel_size() const;

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Color> pixels_;

    const Color& pixel(int x, int y) const;
};

Color sample_material_base_color(const Scene& scene, const Material& material, const Vec2& uv);

}  // namespace renderer
