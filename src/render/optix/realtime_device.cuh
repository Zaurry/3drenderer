#pragma once
#include "render/pathtracer/cuda_scene.cuh"
#include "render/optix/optix_realtime_renderer.h"
namespace renderer {
namespace {
// All images are top-left-origin, unexposed, linear HDR. Channels 0/1 are
// direct/indirect diffuse illumination, 2 specular, 3 transmission, 4 emission.
constexpr int kSignals = 4;
struct RtSignals { DVec3 c[5]{}; DVec3 direct{}; };
struct RtFiltered { DVec3 c[kSignals]{}; float variance[kSignals]{}; };
struct RtPrepared {
    DVec3 c[kSignals]{};
    float mean[kSignals]{}, variance[kSignals]{};
};
struct RtNeighborhoodGuide {
    DVec3 normal;
    float depth;
    unsigned long long object_id, asset_id;
    int material;
};
struct RtLuminance { float c[kSignals]; };
constexpr int kPrepareWidth=16, kPrepareHeight=8, kPrepareRadius=3;
constexpr int kPrepareStride=kPrepareWidth+2*kPrepareRadius;
constexpr int kPreparePixels=kPrepareStride*(kPrepareHeight+2*kPrepareRadius);
struct RtHistory {
    DVec3 color[kSignals]{};
    DVec2 moments[kSignals]{};
    float length[kSignals]{};
};
struct RtInstance {
    DMatrix3x4 current_to_previous;
    DMatrix3x3 normal_to_previous;
    unsigned long long object_id;
    unsigned long long asset_id;
};
struct RtGuide {
    DVec3 normal{}, albedo{}, previous_normal{};
    // Octahedral UNORM16 pairs keep both geometric normals without increasing
    // the bandwidth of the 96-byte guide used by every lighting/filter pass.
    unsigned geometric=0, previous_geometric=0;
    DVec2 motion{};
    float depth = 0, previous_depth = 0, roughness = 0, hit_distance = 0;
    unsigned long long object_id = 0, asset_id = 0;
    int material = -1, transparent = 0;
};
static_assert(sizeof(RtGuide)==96);
struct RtOutputGuide {
    DVec3 normal{};
    float depth = 0;
    unsigned long long object_id = 0;
    float history = 0;
    int material = -1;
};
struct RtFrame {
    DScene scene;
    DCamera camera, previous_camera;
    const RtInstance* instances;
    RealtimeRenderSettings settings;
    int width, height, output_width, output_height;
    unsigned long long frame_index, seed;
    DVec2 jitter, previous_jitter;
    int valid_history, shading_changed;
    unsigned long long hardware_scene=0;
    int deterministic_direct=0;
    int denoiser_reset=0;
    int stationary=0;
};
struct RtOptixParameters {
    RtFrame frame;DCompactHit* primary;RtGuide* guides;RtSignals* signals;DVec3* emission=nullptr;
    const int* optical_pixels=nullptr;const unsigned* optical_count=nullptr;
};
__device__ float rt_luma(DVec3 c) { return .2126f*c.x + .7152f*c.y + .0722f*c.z; }
__device__ DVec3 rt_safe(DVec3 c) {
    return finite(c) ? v3(fminf(fmaxf(c.x,0),1e15f),fminf(fmaxf(c.y,0),1e15f),fminf(fmaxf(c.z,0),1e15f)) : v3(0,0,0);
}
__device__ DVec3 rt_divide(DVec3 a, DVec3 b) {
    return v3(a.x/fmaxf(b.x,1e-12f),a.y/fmaxf(b.y,1e-12f),a.z/fmaxf(b.z,1e-12f));
}
__device__ DVec3 rt_albedo(const RtGuide& g) {
    return v3(fmaxf(.04f,g.albedo.x),fmaxf(.04f,g.albedo.y),fmaxf(.04f,g.albedo.z));
}
__device__ DVec3 rt_mix(DVec3 a, DVec3 b, float t) { return add(mul(a,1-t),mul(b,t)); }
__device__ unsigned rt_pack_normal(DVec3 n) {
    n=divv(n,fabsf(n.x)+fabsf(n.y)+fabsf(n.z));
    if(n.z<0) {
        const float x=n.x;
        n.x=(1-fabsf(n.y))*copysignf(1,n.x);n.y=(1-fabsf(x))*copysignf(1,n.y);
    }
    const unsigned x=unsigned(saturate(n.x*.5f+.5f)*65535+.5f);
    const unsigned y=unsigned(saturate(n.y*.5f+.5f)*65535+.5f);
    return x|(y<<16);
}
__device__ DVec3 rt_unpack_normal(unsigned packed) {
    DVec3 n=v3(float(packed&65535)*(2.0f/65535)-1,float(packed>>16)*(2.0f/65535)-1,0);
    n.z=1-fabsf(n.x)-fabsf(n.y);
    if(n.z<0) {
        const float x=n.x;
        n.x=(1-fabsf(n.y))*copysignf(1,n.x);n.y=(1-fabsf(x))*copysignf(1,n.y);
    }
    return normalize(n);
}
__device__ DVec2 rt_project(const DCamera& c, DVec3 p, float& depth) {
    const DVec3 d = sub(p,c.eye);
    depth = dot(d,c.forward);
    const float z = fmaxf(depth,1e-10f);
    return { .5f+dot(d,c.right)/(z*c.viewport_width),
             .5f-dot(d,c.up)/(z*c.viewport_height) };
}
__device__ DRay rt_primary(const RtFrame& f, int i) {
    const float u = (float(i%f.width)+.5f+f.jitter.x)/f.width;
    const float v = (float(i/f.width)+.5f+f.jitter.y)/f.height;
    return {f.camera.eye,normalize(add(add(f.camera.forward,
        mul(f.camera.right,(u-.5f)*f.camera.viewport_width)),
        mul(f.camera.up,(.5f-v)*f.camera.viewport_height)))};
}

__device__ bool rt_sharp_optics(const RtGuide& g) {
    return g.depth>0 && (g.transparent || (g.roughness<.06f && max_component(g.albedo)<.01f));
}

static_assert(std::is_trivially_copyable_v<RtOptixParameters>);
} // namespace
} // namespace renderer
