#pragma once

#include "render/interactive/interactive_render_session.h"

#include <string>

namespace renderer {

struct FrameRateSnapshot {
    bool valid = false;
    int frames = 0;
    double frames_per_second = 0.0;
    double milliseconds_per_frame = 0.0;
};

class FrameRateCounter {
public:
    explicit FrameRateCounter(double update_interval_seconds = 0.5);

    bool tick(double frame_seconds);
    void reset();
    const FrameRateSnapshot& snapshot() const;

private:
    double update_interval_seconds_ = 0.5;
    double accumulated_seconds_ = 0.0;
    int accumulated_frames_ = 0;
    FrameRateSnapshot snapshot_;
};

std::string format_viewer_title(
    InteractiveRenderMode mode,
    const FrameRateSnapshot& snapshot,
    int path_sample_count);

}  // namespace renderer
