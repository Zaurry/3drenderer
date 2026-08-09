#pragma once

#include <array>
#include <string>

namespace renderer {

enum class InteractiveRenderMode {
    OpenGl,
    Path,
};

enum class RenderModeCapability : unsigned int {
    None = 0,
    Progressive = 1U << 0U,
    ShaderReload = 1U << 1U,
};

constexpr RenderModeCapability operator|(
    RenderModeCapability left,
    RenderModeCapability right) {
    return static_cast<RenderModeCapability>(
        static_cast<unsigned int>(left) |
        static_cast<unsigned int>(right));
}

constexpr bool has_capability(
    RenderModeCapability capabilities,
    RenderModeCapability capability) {
    return (
        static_cast<unsigned int>(capabilities) &
        static_cast<unsigned int>(capability)) != 0U;
}

struct RenderModeDescriptor {
    InteractiveRenderMode mode;
    const char* cli_name;
    const char* label;
    int hotkey;
    RenderModeCapability capabilities;
};

const std::array<RenderModeDescriptor, 2>& interactive_render_modes();
const RenderModeDescriptor& render_mode_descriptor(InteractiveRenderMode mode);
InteractiveRenderMode parse_interactive_render_mode(const std::string& value);
InteractiveRenderMode interactive_render_mode_from_hotkey(int hotkey);

}  // namespace renderer
