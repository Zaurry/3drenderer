#pragma once

#include "core/color.h"

#include <cstdint>
#include <vector>

namespace renderer {

class Framebuffer {
public:
    Framebuffer(int width, int height);

    int width() const;
    int height() const;

    void resize(int width, int height);
    void clear(const Color& color);
    void set_pixel(int x, int y, const Color& color);
    const Color& pixel(int x, int y) const;
    std::vector<std::uint8_t> to_rgba8() const;

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Color> pixels_;

    int index(int x, int y) const;
};

}  // namespace renderer
