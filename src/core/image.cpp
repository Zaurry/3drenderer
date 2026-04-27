#include "core/image.h"

#include <filesystem>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include "stb_image_write.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace renderer {

Image::Image(int image_width, int image_height)
    : width_(image_width), height_(image_height), pixels_(static_cast<std::size_t>(image_width * image_height)) {
    if (image_width <= 0 || image_height <= 0) {
        throw std::invalid_argument("Image dimensions must be positive");
    }
}

int Image::width() const {
    return width_;
}

int Image::height() const {
    return height_;
}

void Image::set_pixel(int x, int y, const Color& color) {
    pixels_[static_cast<std::size_t>(index(x, y))] = color;
}

const Color& Image::pixel(int x, int y) const {
    return pixels_[static_cast<std::size_t>(index(x, y))];
}

Rgb8 Image::pixel_rgb8(int x, int y) const {
    return to_rgb8(pixel(x, y));
}

bool Image::write_png(const std::string& path) const {
    try {
        const std::filesystem::path output_path(path);
        const std::filesystem::path parent = output_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::vector<unsigned char> bytes;
        bytes.reserve(static_cast<std::size_t>(width_ * height_ * 3));
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                const Rgb8 converted = pixel_rgb8(x, y);
                bytes.push_back(converted.r);
                bytes.push_back(converted.g);
                bytes.push_back(converted.b);
            }
        }

        const int stride_bytes = width_ * 3;
        return stbi_write_png(path.c_str(), width_, height_, 3, bytes.data(), stride_bytes) != 0;
    } catch (const std::exception&) {
        return false;
    }
}

int Image::index(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        throw std::out_of_range("Image pixel coordinates are out of range");
    }
    return y * width_ + x;
}

}  // namespace renderer
