#pragma once

#include "render/interactive/interactive_render_session.h"

#include <string>

namespace renderer {

struct FrameRateSnapshot {
    bool valid = false;
    int frames = 0;
    float frames_per_second = 0.0f;
    float milliseconds_per_frame = 0.0f;
};

class FrameRateCounter {
public:
    explicit FrameRateCounter(float update_interval_seconds = 0.5f);

    bool tick(float frame_seconds);
    void reset();
    const FrameRateSnapshot& snapshot() const;

private:
    float update_interval_seconds_ = 0.5f;
    float accumulated_seconds_ = 0.0f;
    int accumulated_frames_ = 0;
    FrameRateSnapshot snapshot_;
};

std::string format_viewer_title(
    InteractiveRenderMode mode,
    const FrameRateSnapshot& snapshot,
    int path_sample_count);

}  // namespace renderer
