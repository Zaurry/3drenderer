#include "scene/texture.h"

#include "scene/material.h"
#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include "stb_image.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace renderer {

namespace {

Color multiply(const Color& a, const Color& b) {
    return Color(a.x * b.x, a.y * b.y, a.z * b.z);
}

double wrap01(double value) {
    const double wrapped = value - std::floor(value);
    return wrapped < 0.0 ? wrapped + 1.0 : wrapped;
}

int wrap_index(int value, int size) {
    const int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

Color lerp(const Color& a, const Color& b, double t) {
    return a * (1.0 - t) + b * t;
}

double decode_channel(unsigned char value, TextureEncoding encoding) {
    const double encoded = static_cast<double>(value) / 255.0;
    if (encoding == TextureEncoding::Linear) {
        return encoded;
    }
    if (encoded <= 0.04045) {
        return encoded / 12.92;
    }
    return std::pow((encoded + 0.055) / 1.055, 2.4);
}

bool valid_texture_id(const Scene& scene, int texture_id) {
    return texture_id >= 0 && static_cast<std::size_t>(texture_id) < scene.textures.size();
}

}  // namespace

ImageTexture::ImageTexture(int width, int height, std::vector<Color> pixels)
    : width_(width), height_(height), pixels_(std::move(pixels)) {
    if (width_ <= 0 || height_ <= 0) {
        throw std::invalid_argument("ImageTexture dimensions must be positive");
    }
    if (pixels_.size() != static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_)) {
        throw std::invalid_argument("ImageTexture pixel count does not match dimensions");
    }
}

ImageTexture ImageTexture::load(const std::string& path, TextureEncoding encoding) {
    int width = 0;
    int height = 0;
    int source_channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &source_channels, 3);
    if (!data) {
        std::string message = "Failed to load texture image: " + path;
        const char* reason = stbi_failure_reason();
        if (reason) {
            message += " (";
            message += reason;
            message += ")";
        }
        throw std::runtime_error(message);
    }

    std::vector<Color> pixels;
    pixels.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int i = 0; i < width * height; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * 3U;
        pixels.emplace_back(
            decode_channel(data[base], encoding),
            decode_channel(data[base + 1U], encoding),
            decode_channel(data[base + 2U], encoding));
    }
    stbi_image_free(data);
    return ImageTexture(width, height, std::move(pixels));
}

int ImageTexture::width() const {
    return width_;
}

int ImageTexture::height() const {
    return height_;
}

Color ImageTexture::sample(const Vec2& uv) const {
    if (width_ <= 0 || height_ <= 0 || pixels_.empty()) {
        return Color(1.0, 0.0, 1.0);
    }

    const double u = wrap01(uv.x);
    const double v = wrap01(uv.y);
    const double x = u * static_cast<double>(width_) - 0.5;
    const double y = (1.0 - v) * static_cast<double>(height_) - 0.5;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double tx = x - static_cast<double>(x0);
    const double ty = y - static_cast<double>(y0);

    const Color c00 = pixel(wrap_index(x0, width_), wrap_index(y0, height_));
    const Color c10 = pixel(wrap_index(x0 + 1, width_), wrap_index(y0, height_));
    const Color c01 = pixel(wrap_index(x0, width_), wrap_index(y0 + 1, height_));
    const Color c11 = pixel(wrap_index(x0 + 1, width_), wrap_index(y0 + 1, height_));
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

double ImageTexture::sample_scalar(const Vec2& uv) const {
    const Color color = sample(uv);
    return color.x * 0.2126 + color.y * 0.7152 + color.z * 0.0722;
}

Vec2 ImageTexture::texel_size() const {
    if (width_ <= 0 || height_ <= 0) {
        return Vec2();
    }
    return Vec2(1.0 / static_cast<double>(width_), 1.0 / static_cast<double>(height_));
}

const Color& ImageTexture::pixel(int x, int y) const {
    return pixels_[static_cast<std::size_t>(y * width_ + x)];
}

Color sample_material_base_color(const Scene& scene, const Material& material, const Vec2& uv) {
    if (!valid_texture_id(scene, material.diffuse_texture_id)) {
        return material.base_color;
    }
    return multiply(
        material.base_color,
        scene.textures[static_cast<std::size_t>(material.diffuse_texture_id)].sample(uv));
}

}  // namespace renderer
