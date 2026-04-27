#pragma once

#include "core/color.h"

#include <string>
#include <vector>

namespace renderer {

class Image {
public:
    Image(int image_width, int image_height);

    int width() const;
    int height() const;

    void set_pixel(int x, int y, const Color& color);
    const Color& pixel(int x, int y) const;
    Rgb8 pixel_rgb8(int x, int y) const;
    bool write_png(const std::string& path) const;

private:
    int width_;
    int height_;
    std::vector<Color> pixels_;

    int index(int x, int y) const;
};

}  // namespace renderer
