#pragma once
#include <array>
#include <cstdint>
#include <type_traits>

namespace renderer {
struct DxrFloat4 {float x=0,y=0,z=0,w=0;};
struct DxrUint4 {std::uint32_t x=0,y=0,z=0,w=0;};
struct DxrGpuVertex {
    DxrFloat4 position,normal,uv,tangent,color;
};
struct DxrGpuInstance {
    DxrFloat4 world[3],inverse[3],previous[3];
    DxrUint4 geometry,identity;
};
struct DxrGpuTexture {
    std::uint32_t texture=0,sampler=0,texcoord=0;
    float rotation=0;
    DxrFloat4 transform{0,0,1,1};
};
// Texture order is shared verbatim with scene.hlsli.
enum DxrTextureSlot { DxrDiffuse,DxrOpacity,DxrBump,DxrBaseColor,DxrMetalRough,
    DxrNormal,DxrOcclusion,DxrEmissive,DxrSpecular,DxrSpecularColor,DxrSpecGloss,DxrTextureCount };
struct DxrGpuMaterial {
    DxrFloat4 base_opacity,emission_roughness,specular_metallic,optics,pbr;
    DxrUint4 flags;
    DxrGpuTexture textures[DxrTextureCount];
};
struct DxrGpuLight {
    DxrFloat4 position_type,direction_radius,radiance_range,axis_u_inner,axis_v_outer;
    DxrUint4 identity;
};
struct DxrGpuSurface {
    DxrFloat4 position_depth,normal_roughness,geometric_metallic,albedo_opacity;
    DxrFloat4 specular_distance,motion,emission;
    DxrUint4 identity;
    DxrFloat4 base,f90,diffuse_f0;
    DxrFloat4 view_path,virtual_normal,throughput,specular_motion;
};
struct DxrFrameConstants {
    DxrFloat4 eye,forward,right,up;
    DxrFloat4 previous_eye,previous_forward,previous_right,previous_up;
    DxrFloat4 environment,jitter;
    DxrFloat4 previous_jitter;
    DxrUint4 size; // render width,height,output width,height
    DxrUint4 frame; // frame index,SPP,bounces,reset
    DxrUint4 scene; // vertices,slots,instances,materials SRVs
    DxrUint4 lighting; // lights SRV,count,environment texture,environment alias table
    DxrUint4 light_extra; // emissive primitive map,selection CDF,remapping,environment light
    DxrUint4 guides; // current surface UAV,previous surface SRV,raw UAV,direct UAV
    DxrUint4 signals; // indirect UAV,diffuse UAV,specular UAV,output UAV
    DxrUint4 history; // DI buffer,PT buffer,previous color,current color
    DxrUint4 options; // DI,PT,debug,pass
    DxrUint4 sampling; // candidates,spatial count,history length,flags
    DxrUint4 path_sampling; // PT spatial samples,disocclusion samples,stationary frames,material detail
    DxrUint4 reconstruction; // normal/roughness,viewZ,motion UAVs,reserved
    DxrUint4 denoised; // denoised diffuse/specular SRVs,demodulated diffuse/specular UAVs
    DxrUint4 light_history; // previous light SRV,previous->current,current->previous,changed
    DxrUint4 reservoirs; // block row pitch,array pitch,neighbor offsets SRV,diagnostic UAV
    DxrUint4 dlss; // diffuse albedo,specular albedo,specular hit distance,SR input UAVs
    DxrUint4 dlss_guides; // two-channel motion,hardware depth UAVs
    DxrUint4 bsdf_tables; // GGX directional + average albedo SRV
    DxrFloat4 environment_info; // rotation radians,background visible,env width,height
};
static_assert(sizeof(DxrGpuVertex)==80);
static_assert(sizeof(DxrGpuInstance)==176);
static_assert(sizeof(DxrGpuMaterial)==448);
static_assert(sizeof(DxrGpuSurface)==240);
static_assert(std::is_trivially_copyable_v<DxrFrameConstants>);
} // namespace renderer
