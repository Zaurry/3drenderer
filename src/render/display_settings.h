#pragma once

#include "core/color.h"

namespace renderer {

enum class ToneMapper {
    None,
    Reinhard,
    Aces,
};

struct DisplaySettings {
    float exposure_ev = 0.0f;
    ToneMapper tone_mapper = ToneMapper::None;
};

Color apply_display_transform(const Color& linear_color, const DisplaySettings& settings);
Rgb8 to_display_rgb8(const Color& linear_color, const DisplaySettings& settings);

}  // namespace renderer
