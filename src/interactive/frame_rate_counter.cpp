#include "interactive/frame_rate_counter.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace renderer {

namespace {

const char* mode_label(InteractiveRenderMode mode) {
    if (mode == InteractiveRenderMode::Raster) {
        return "raster";
    }
    if (mode == InteractiveRenderMode::Ray) {
        return "ray";
    }
    if (mode == InteractiveRenderMode::Path) {
        return "path";
    }
    return "opengl";
}

}  // namespace

FrameRateCounter::FrameRateCounter(float update_interval_seconds)
    : update_interval_seconds_(update_interval_seconds) {
    if (!std::isfinite(update_interval_seconds_) || update_interval_seconds_ <= 0.0f) {
        throw std::invalid_argument("FrameRateCounter update interval must be positive");
    }
}

bool FrameRateCounter::tick(float frame_seconds) {
    if (!std::isfinite(frame_seconds) || frame_seconds < 0.0f) {
        throw std::invalid_argument("FrameRateCounter frame seconds must be non-negative and finite");
    }

    accumulated_seconds_ += frame_seconds;
    ++accumulated_frames_;
    if (accumulated_seconds_ < update_interval_seconds_ || accumulated_seconds_ <= 0.0f) {
        return false;
    }

    snapshot_.valid = true;
    snapshot_.frames = accumulated_frames_;
    snapshot_.frames_per_second = static_cast<float>(accumulated_frames_) / accumulated_seconds_;
    snapshot_.milliseconds_per_frame = (accumulated_seconds_ * 1000.0f) / static_cast<float>(accumulated_frames_);
    accumulated_seconds_ = 0.0f;
    accumulated_frames_ = 0;
    return true;
}

void FrameRateCounter::reset() {
    accumulated_seconds_ = 0.0f;
    accumulated_frames_ = 0;
    snapshot_ = FrameRateSnapshot();
}

const FrameRateSnapshot& FrameRateCounter::snapshot() const {
    return snapshot_;
}

std::string format_viewer_title(
    InteractiveRenderMode mode,
    const FrameRateSnapshot& snapshot,
    int path_sample_count) {
    std::ostringstream title;
    title << "3D Renderer Viewer - " << mode_label(mode) << " - ";
    if (snapshot.valid) {
        title << std::fixed << std::setprecision(1)
              << snapshot.frames_per_second << " FPS - "
              << snapshot.milliseconds_per_frame << " ms";
    } else {
        title << "FPS --";
    }

    if (mode == InteractiveRenderMode::Path) {
        title << " - " << std::max(0, path_sample_count) << " spp";
    }
    return title.str();
}

}  // namespace renderer
