#pragma once

#include "core/color.h"
#include "core/math/types.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace renderer {

struct Material;
struct Scene;

enum class TextureEncoding { Srgb, Linear };
enum class TextureUvOrigin { BottomLeft, TopLeft };
enum class TextureWrap { Repeat, ClampToEdge, MirroredRepeat };
enum class TextureFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
};

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
    ImageTexture(
        int width,
        int height,
        std::vector<Color> pixels,
        TextureUvOrigin uv_origin = TextureUvOrigin::BottomLeft);
    ImageTexture(
        int width,
        int height,
        std::vector<Color> pixels,
        std::vector<float> alpha,
        TextureUvOrigin uv_origin = TextureUvOrigin::BottomLeft);

    static ImageTexture load(
        const std::string& path,
        TextureEncoding encoding = TextureEncoding::Srgb,
        TextureUvOrigin uv_origin = TextureUvOrigin::BottomLeft);
    static ImageTexture load_from_memory(
        const unsigned char* bytes,
        std::size_t byte_count,
        TextureEncoding encoding,
        const std::string& label,
        TextureUvOrigin uv_origin = TextureUvOrigin::BottomLeft);

    int width() const;
    int height() const;
    Color sample(const Vec2& uv) const;
    float sample_alpha(const Vec2& uv) const;
    float sample_scalar(const Vec2& uv) const;
    Vec2 texel_size() const;
    const std::vector<Color>& pixels() const;
    const std::vector<float>& alphas() const;
    void set_sampler(
        TextureWrap wrap_s,
        TextureWrap wrap_t,
        TextureFilter min_filter,
        TextureFilter mag_filter);
    TextureWrap wrap_s() const;
    TextureWrap wrap_t() const;
    TextureFilter min_filter() const;
    TextureFilter mag_filter() const;
    TextureUvOrigin uv_origin() const;
    bool shares_pixel_storage_with(const ImageTexture& other) const;
    bool same_resource_view(const ImageTexture& other) const;

private:
    struct Storage {
        int width = 0;
        int height = 0;
        std::vector<Color> pixels;
        std::vector<float> alpha;
    };

    std::shared_ptr<const Storage> storage_;
    TextureUvOrigin uv_origin_ = TextureUvOrigin::BottomLeft;
    TextureWrap wrap_s_ = TextureWrap::Repeat;
    TextureWrap wrap_t_ = TextureWrap::Repeat;
    TextureFilter min_filter_ = TextureFilter::LinearMipmapLinear;
    TextureFilter mag_filter_ = TextureFilter::Linear;

    const Color& pixel(int x, int y) const;
    float alpha(int x, int y) const;
};

Color sample_material_base_color(const Scene& scene, const Material& material, const Vec2& uv);

}  // namespace renderer
