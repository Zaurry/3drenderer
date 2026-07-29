#pragma once

#include "render/framebuffer.h"

namespace renderer {

enum class RenderFrameOutputKind {
    None,
    HostFramebuffer,
    OpenGlTexture,
};

struct RenderTextureView {
    unsigned int texture = 0;
    int width = 0;
    int height = 0;
    bool flip_y = false;
};

class RenderFrameOutput {
public:
    static RenderFrameOutput host(const Framebuffer& framebuffer) {
        RenderFrameOutput output;
        output.kind_ = RenderFrameOutputKind::HostFramebuffer;
        output.framebuffer_ = &framebuffer;
        return output;
    }

    static RenderFrameOutput texture(RenderTextureView texture) {
        RenderFrameOutput output;
        output.kind_ = RenderFrameOutputKind::OpenGlTexture;
        output.texture_ = texture;
        return output;
    }

    RenderFrameOutputKind kind() const { return kind_; }
    const Framebuffer* framebuffer() const { return framebuffer_; }
    const RenderTextureView& texture() const { return texture_; }
    bool valid() const { return kind_ != RenderFrameOutputKind::None; }

private:
    RenderFrameOutputKind kind_ = RenderFrameOutputKind::None;
    const Framebuffer* framebuffer_ = nullptr;
    RenderTextureView texture_;
};

}  // namespace renderer
