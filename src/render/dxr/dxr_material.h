#pragma once
#include "scene/material.h"

namespace renderer {
// OBJ/legacy materials predate AlphaMode and use opacity/map_d as a cutout.
// glTF PBR honors its explicit mode, including OPAQUE with an alpha texture.
inline AlphaMode dxr_alpha_mode(const Material& material) {
    if(material.type!=MaterialType::Pbr && material.alpha_mode==AlphaMode::Opaque &&
        (material.opacity_texture_id>=0 || material.opacity<1))return AlphaMode::Mask;
    return material.alpha_mode;
}
}
