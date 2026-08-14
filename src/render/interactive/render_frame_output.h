#pragma once

#include "render/framebuffer.h"

#include <memory>
#include <variant>

namespace renderer {

// Typed lifetime owner for GPU-side frame outputs. Handles keep the owner
// alive until presentation completes, so the texture id stays valid.
class TextureLifetimeOwner {
public:
    virtual ~TextureLifetimeOwner() = default;
};

struct HostFrameHandle {
    std::shared_ptr<const Framebuffer> framebuffer;
};

struct OpenGlTextureHandle {
    unsigned int texture = 0;
    int width = 0;
    int height = 0;
    bool flip_y = false;
    // Keeps the backend resource owner alive until presentation completes.
    std::shared_ptr<const TextureLifetimeOwner> lifetime;
};

using RenderFrameOutput = std::variant<
    std::monostate,
    HostFrameHandle,
    OpenGlTextureHandle>;

}  // namespace renderer
