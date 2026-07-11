#include "render/depth_buffer.h"

#include <algorithm>
#include <stdexcept>

namespace renderer {

namespace {

std::size_t validated_value_count(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("DepthBuffer dimensions must be positive");
    }
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

}  // namespace

DepthBuffer::DepthBuffer(int width, int height) {
    resize(width, height);
}

int DepthBuffer::width() const {
    return width_;
}

int DepthBuffer::height() const {
    return height_;
}

void DepthBuffer::resize(int width, int height) {
    values_.assign(validated_value_count(width, height), 0.0f);
    width_ = width;
    height_ = height;
}

void DepthBuffer::clear(float value) {
    std::fill(values_.begin(), values_.end(), value);
}

float DepthBuffer::get(int x, int y) const {
    return values_[static_cast<std::size_t>(index(x, y))];
}

void DepthBuffer::set(int x, int y, float value) {
    values_[static_cast<std::size_t>(index(x, y))] = value;
}

int DepthBuffer::index(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        throw std::out_of_range("DepthBuffer coordinates are out of range");
    }
    return y * width_ + x;
}

}  // namespace renderer
