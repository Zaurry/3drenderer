#pragma once
#include "platform/d3d12/d3d12_context.h"
#include "scene/instanced_scene.h"
#include <string>

namespace renderer {
struct DxrOmmBake {
    static constexpr unsigned subdivision=4;
    std::vector<std::uint32_t> states,indices;
    std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_DESC> descriptions;
    std::array<std::uint64_t,4> counts{};
};
struct DxrOmmGpu {D3d12ResourcePtr array,indices;};
// Empty when there is no eligible static mask. Includes every alpha input;
// unrelated material edits deliberately preserve the cached acceleration data.
std::string dxr_opacity_signature(const RenderSceneSnapshot&,const RenderSceneInstanceSnapshot&);
DxrOmmBake bake_dxr_opacity(const RenderSceneAssetSnapshot&,const RenderSceneInstanceSnapshot&,const std::vector<ImageTexture>&);
DxrOmmGpu build_dxr_opacity(D3d12Context&,const DxrOmmBake&);
std::array<Vec2,3> dxr_microtriangle(unsigned index,unsigned level);
}
