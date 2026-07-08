#pragma once

#include <vector>

namespace renderer {

class DepthBuffer {
public:
    DepthBuffer(int width, int height);

    int width() const;
    int height() const;

    void resize(int width, int height);
    void clear(double value);
    double get(int x, int y) const;
    void set(int x, int y, double value);

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<double> values_;

    int index(int x, int y) const;
};

}  // namespace renderer
