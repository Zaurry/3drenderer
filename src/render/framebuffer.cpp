#include "render/framebuffer.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace renderer {

namespace {

std::size_t validated_pixel_count(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Framebuffer dimensions must be positive");
    }
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

}  // namespace

Framebuffer::Framebuffer(int width, int height) {
    resize(width, height);
}

int Framebuffer::width() const {
    return width_;
}

int Framebuffer::height() const {
    return height_;
}

void Framebuffer::resize(int width, int height) {
    pixels_.assign(validated_pixel_count(width, height), Color::Zero());
    width_ = width;
    height_ = height;
}

void Framebuffer::clear(const Color& color) {
    std::fill(pixels_.begin(), pixels_.end(), color);
}

void Framebuffer::set_pixel(int x, int y, const Color& color) {
    pixels_[static_cast<std::size_t>(index(x, y))] = color;
}

void Framebuffer::set_pixels(std::vector<Color> pixels) {
    if (pixels.size() != pixels_.size()) {
        throw std::invalid_argument("Framebuffer pixel count does not match dimensions");
    }
    pixels_ = std::move(pixels);
}

const Color& Framebuffer::pixel(int x, int y) const {
    return pixels_[static_cast<std::size_t>(index(x, y))];
}

std::vector<std::uint8_t> Framebuffer::to_rgba8() const {
    return to_rgba8(DisplaySettings{});
}

std::vector<std::uint8_t> Framebuffer::to_rgba8(const DisplaySettings& settings) const {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 4U);
    for (const Color& color : pixels_) {
        const Rgb8 converted = to_display_rgb8(color, settings);
        bytes.push_back(converted.r);
        bytes.push_back(converted.g);
        bytes.push_back(converted.b);
        bytes.push_back(255U);
    }
    return bytes;
}

int Framebuffer::index(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        throw std::out_of_range("Framebuffer coordinates are out of range");
    }
    return y * width_ + x;
}

}  // namespace renderer
