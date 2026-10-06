#include "render/interactive/render_mode.h"

#include <stdexcept>

namespace renderer {

namespace {

constexpr std::array<RenderModeDescriptor, 3> kRenderModes{{
    {
        InteractiveRenderMode::OpenGl,
        "opengl",
        "OpenGL",
        1,
        RenderModeCapability::ShaderReload,
    },
    {
        InteractiveRenderMode::Rtrt,
        "rtrt",
        "RTRT",
        2,
        RenderModeCapability::Temporal,
    },
    {InteractiveRenderMode::Dxr, "dxr", "DXR", 3, RenderModeCapability::Temporal},
}};

}  // namespace

const std::array<RenderModeDescriptor, 3>& interactive_render_modes() {
    return kRenderModes;
}

const RenderModeDescriptor& render_mode_descriptor(InteractiveRenderMode mode) {
    for (const RenderModeDescriptor& descriptor : kRenderModes) {
        if (descriptor.mode == mode) {
            return descriptor;
        }
    }
    throw std::invalid_argument("unknown interactive render mode");
}

InteractiveRenderMode parse_interactive_render_mode(const std::string& value) {
    const std::string canonical = value == "gl" ? "opengl" : value == "path" ? "rtrt" : value;
    for (const RenderModeDescriptor& descriptor : kRenderModes) {
        if (canonical == descriptor.cli_name) {
            return descriptor.mode;
        }
    }
    throw std::invalid_argument("unknown mode: " + value);
}

InteractiveRenderMode interactive_render_mode_from_hotkey(int hotkey) {
    for (const RenderModeDescriptor& descriptor : kRenderModes) {
        if (descriptor.hotkey == hotkey) {
            return descriptor.mode;
        }
    }
    throw std::invalid_argument("unknown render mode hotkey");
}

}  // namespace renderer
