#include "scene/texture.h"

#include "scene/material.h"
#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
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

float addressed_coordinate(float value, TextureWrap wrap) {
    if (wrap == TextureWrap::ClampToEdge) {
        return std::clamp(value, 0.0f, 1.0f);
    }
    if (wrap == TextureWrap::MirroredRepeat) {
        const float period = value - std::floor(value * 0.5f) * 2.0f;
        return period <= 1.0f ? period : 2.0f - period;
    }
    const float repeated = value - std::floor(value);
    return repeated < 0.0f ? repeated + 1.0f : repeated;
}

int addressed_index(int value, int size, TextureWrap wrap) {
    if (wrap != TextureWrap::Repeat) {
        return std::clamp(value, 0, size - 1);
    }
    const int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

Color lerp(const Color& a, const Color& b, float t) {
    return a * (1.0f - t) + b * t;
}

float decode_channel(unsigned char value, TextureEncoding encoding) {
    const float encoded = static_cast<float>(value) / 255.0f;
    if (encoding == TextureEncoding::Linear) {
        return encoded;
    }
    if (encoded <= 0.04045f) {
        return encoded / 12.92f;
    }
    return std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

bool valid_texture_id(const Scene& scene, int texture_id) {
    return texture_id >= 0 && static_cast<std::size_t>(texture_id) < scene.textures.size();
}

std::size_t checked_image_pixel_count(int width, int height, const std::string& label) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Texture image has invalid dimensions: " + label);
    }
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Texture image is too large: " + label);
    }
    return count;
}

}  // namespace

ImageTexture::ImageTexture(
    int width,
    int height,
    std::vector<Color> pixels,
    TextureUvOrigin uv_origin)
    : ImageTexture(
          width,
          height,
          std::move(pixels),
          std::vector<float>(
              static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
              1.0f),
          uv_origin) {}

ImageTexture::ImageTexture(
    int width,
    int height,
    std::vector<Color> pixels,
    std::vector<float> alpha,
    TextureUvOrigin uv_origin)
    : uv_origin_(uv_origin) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("ImageTexture dimensions must be positive");
    }
    if (pixels.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
        throw std::invalid_argument("ImageTexture pixel count does not match dimensions");
    }
    if (alpha.size() != pixels.size()) {
        throw std::invalid_argument("ImageTexture alpha count does not match dimensions");
    }
    auto storage = std::make_shared<Storage>();
    storage->width = width;
    storage->height = height;
    storage->pixels = std::move(pixels);
    storage->alpha = std::move(alpha);
    storage_ = std::move(storage);
}

ImageTexture ImageTexture::load(
    const std::string& path,
    TextureEncoding encoding,
    TextureUvOrigin uv_origin) {
    int width = 0;
    int height = 0;
    int source_channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &source_channels, 4);
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

    const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> owned_data(
        data,
        &stbi_image_free);
    const std::size_t pixel_count = checked_image_pixel_count(width, height, path);
    std::vector<Color> pixels;
    std::vector<float> alpha;
    pixels.reserve(pixel_count);
    alpha.reserve(pixel_count);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        const std::size_t base = i * 4U;
        pixels.emplace_back(
            decode_channel(data[base], encoding),
            decode_channel(data[base + 1U], encoding),
            decode_channel(data[base + 2U], encoding));
        alpha.push_back(static_cast<float>(data[base + 3U]) / 255.0f);
    }
    return ImageTexture(width, height, std::move(pixels), std::move(alpha), uv_origin);
}

ImageTexture ImageTexture::load_from_memory(
    const unsigned char* bytes,
    std::size_t byte_count,
    TextureEncoding encoding,
    const std::string& label,
    TextureUvOrigin uv_origin) {
    if (!bytes || byte_count == 0U ||
        byte_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Texture memory buffer is empty or too large: " + label);
    }
    int width = 0;
    int height = 0;
    int source_channels = 0;
    unsigned char* data = stbi_load_from_memory(
        bytes,
        static_cast<int>(byte_count),
        &width,
        &height,
        &source_channels,
        4);
    if (!data) {
        std::string message = "Failed to load texture image: " + label;
        if (const char* reason = stbi_failure_reason()) {
            message += " (" + std::string(reason) + ")";
        }
        throw std::runtime_error(message);
    }
    const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> owned_data(
        data,
        &stbi_image_free);
    const std::size_t pixel_count = checked_image_pixel_count(width, height, label);
    std::vector<Color> pixels;
    std::vector<float> alpha;
    pixels.reserve(pixel_count);
    alpha.reserve(pixel_count);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        const std::size_t base = i * 4U;
        pixels.emplace_back(
            decode_channel(data[base], encoding),
            decode_channel(data[base + 1U], encoding),
            decode_channel(data[base + 2U], encoding));
        alpha.push_back(static_cast<float>(data[base + 3U]) / 255.0f);
    }
    return ImageTexture(width, height, std::move(pixels), std::move(alpha), uv_origin);
}

int ImageTexture::width() const {
    return storage_ ? storage_->width : 0;
}

int ImageTexture::height() const {
    return storage_ ? storage_->height : 0;
}

Color ImageTexture::sample(const Vec2& uv) const {
    const int image_width = width();
    const int image_height = height();
    if (!storage_ || image_width <= 0 || image_height <= 0 || storage_->pixels.empty()) {
        return Color(1.0f, 0.0f, 1.0f);
    }

    const float u = addressed_coordinate(uv.x(), wrap_s_);
    const float source_v = uv_origin_ == TextureUvOrigin::TopLeft
        ? uv.y()
        : 1.0f - uv.y();
    const float v = addressed_coordinate(source_v, wrap_t_);
    if (mag_filter_ == TextureFilter::Nearest) {
        const int x = addressed_index(
            static_cast<int>(std::floor(u * static_cast<float>(image_width))),
            image_width,
            wrap_s_);
        const int y = addressed_index(
            static_cast<int>(std::floor(v * static_cast<float>(image_height))),
            image_height,
            wrap_t_);
        return pixel(x, y);
    }
    const float x = u * static_cast<float>(image_width) - 0.5f;
    const float y = v * static_cast<float>(image_height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const Color c00 = pixel(
        addressed_index(x0, image_width, wrap_s_),
        addressed_index(y0, image_height, wrap_t_));
    const Color c10 = pixel(
        addressed_index(x0 + 1, image_width, wrap_s_),
        addressed_index(y0, image_height, wrap_t_));
    const Color c01 = pixel(
        addressed_index(x0, image_width, wrap_s_),
        addressed_index(y0 + 1, image_height, wrap_t_));
    const Color c11 = pixel(
        addressed_index(x0 + 1, image_width, wrap_s_),
        addressed_index(y0 + 1, image_height, wrap_t_));
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

float ImageTexture::sample_alpha(const Vec2& uv) const {
    const int image_width = width();
    const int image_height = height();
    if (!storage_ || image_width <= 0 || image_height <= 0 || storage_->alpha.empty()) {
        return 1.0f;
    }
    const float u = addressed_coordinate(uv.x(), wrap_s_);
    const float source_v = uv_origin_ == TextureUvOrigin::TopLeft
        ? uv.y()
        : 1.0f - uv.y();
    const float v = addressed_coordinate(source_v, wrap_t_);
    if (mag_filter_ == TextureFilter::Nearest) {
        const int x = addressed_index(
            static_cast<int>(std::floor(u * static_cast<float>(image_width))),
            image_width,
            wrap_s_);
        const int y = addressed_index(
            static_cast<int>(std::floor(v * static_cast<float>(image_height))),
            image_height,
            wrap_t_);
        return alpha(x, y);
    }
    const float x = u * static_cast<float>(image_width) - 0.5f;
    const float y = v * static_cast<float>(image_height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float a00 = alpha(
        addressed_index(x0, image_width, wrap_s_),
        addressed_index(y0, image_height, wrap_t_));
    const float a10 = alpha(
        addressed_index(x0 + 1, image_width, wrap_s_),
        addressed_index(y0, image_height, wrap_t_));
    const float a01 = alpha(
        addressed_index(x0, image_width, wrap_s_),
        addressed_index(y0 + 1, image_height, wrap_t_));
    const float a11 = alpha(
        addressed_index(x0 + 1, image_width, wrap_s_),
        addressed_index(y0 + 1, image_height, wrap_t_));
    return std::lerp(std::lerp(a00, a10, tx), std::lerp(a01, a11, tx), ty);
}

float ImageTexture::sample_scalar(const Vec2& uv) const {
    const Color color = sample(uv);
    return color.dot(Color(0.2126f, 0.7152f, 0.0722f));
}

Vec2 ImageTexture::texel_size() const {
    const int image_width = width();
    const int image_height = height();
    if (image_width <= 0 || image_height <= 0) {
        return Vec2::Zero();
    }
    return Vec2(1.0f / static_cast<float>(image_width), 1.0f / static_cast<float>(image_height));
}

const std::vector<Color>& ImageTexture::pixels() const {
    static const std::vector<Color> empty;
    return storage_ ? storage_->pixels : empty;
}

const std::vector<float>& ImageTexture::alphas() const {
    static const std::vector<float> empty;
    return storage_ ? storage_->alpha : empty;
}

void ImageTexture::set_sampler(
    TextureWrap wrap_s,
    TextureWrap wrap_t,
    TextureFilter min_filter,
    TextureFilter mag_filter) {
    wrap_s_ = wrap_s;
    wrap_t_ = wrap_t;
    min_filter_ = min_filter;
    mag_filter_ = mag_filter;
}

TextureWrap ImageTexture::wrap_s() const { return wrap_s_; }
TextureWrap ImageTexture::wrap_t() const { return wrap_t_; }
TextureFilter ImageTexture::min_filter() const { return min_filter_; }
TextureFilter ImageTexture::mag_filter() const { return mag_filter_; }
TextureUvOrigin ImageTexture::uv_origin() const { return uv_origin_; }

bool ImageTexture::shares_pixel_storage_with(const ImageTexture& other) const {
    return storage_ && storage_ == other.storage_;
}

bool ImageTexture::same_resource_view(const ImageTexture& other) const {
    return shares_pixel_storage_with(other) &&
        uv_origin_ == other.uv_origin_ &&
        wrap_s_ == other.wrap_s_ &&
        wrap_t_ == other.wrap_t_ &&
        min_filter_ == other.min_filter_ &&
        mag_filter_ == other.mag_filter_;
}

const Color& ImageTexture::pixel(int x, int y) const {
    return storage_->pixels[static_cast<std::size_t>(y * width() + x)];
}

float ImageTexture::alpha(int x, int y) const {
    return storage_->alpha[static_cast<std::size_t>(y * width() + x)];
}

// Test-only helper: legacy diffuse-texture base color path. Product shading
// goes through material_evaluator.cpp; tests/renderer_tests.cpp exercises
// this function directly.
Color sample_material_base_color(const Scene& scene, const Material& material, const Vec2& uv) {
    if (!valid_texture_id(scene, material.diffuse_texture_id)) {
        return material.base_color;
    }
    return material.base_color.cwiseProduct(
        scene.textures[static_cast<std::size_t>(material.diffuse_texture_id)].sample(uv));
}

}  // namespace renderer
