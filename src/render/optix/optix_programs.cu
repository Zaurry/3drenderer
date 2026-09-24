#define RTRT_OPTIX_DEVICE 1
#include "render/optix/realtime_device.cuh"
#include <optix.h>
#include <optix_device.h>

namespace renderer {
namespace {
extern "C" __constant__ __align__(16) unsigned char rt_optix_params[sizeof(RtOptixParameters)];
__device__ const RtOptixParameters& rt_optix() {
    return *reinterpret_cast<const RtOptixParameters*>(rt_optix_params);
}

// All visibility queries, including continuation and shadow rays, enter OptiX.
// Keep shading iterative in raygen so path length does not grow the trace stack.
__device__ bool rt_intersect(const RtFrame& f,const DRay& ray,float t_max,DCompactHit& hit,bool reorder=false) {
    hit.primitive_kind=-1;
    if(!f.hardware_scene)return false;
    const auto pointer=reinterpret_cast<unsigned long long>(&hit);
    unsigned low=unsigned(pointer),high=unsigned(pointer>>32);
    if(reorder && f.settings.shader_execution_reordering) {
        optixTraverse(f.hardware_scene,make_float3(ray.origin.x,ray.origin.y,ray.origin.z),
            make_float3(ray.direction.x,ray.direction.y,ray.direction.z),0,t_max,0,255,
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,0,2,0,low,high);
        unsigned material=0;
        if(optixHitObjectIsHit()) {
            const int instance=int(optixHitObjectGetInstanceId());
            const DAsset asset=f.scene.assets[f.scene.instances[instance].asset_index];
            const int primitive=int(optixHitObjectGetPrimitiveIndex());
            material=unsigned(optixGetPrimitiveType(optixHitObjectGetHitKind())==OPTIX_PRIMITIVE_TYPE_SPHERE
                ?sphere_material_id(f.scene,instance,asset.sphere_first+primitive)
                :triangle_material_id(f.scene,instance,asset.triangle_first+primitive));
        }
        optixReorder(material,8);
        optixInvoke(low,high);
    } else {
        optixTrace(f.hardware_scene,make_float3(ray.origin.x,ray.origin.y,ray.origin.z),
            make_float3(ray.direction.x,ray.direction.y,ray.direction.z),0,t_max,0,255,
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,0,2,0,low,high);
    }
    return hit.primitive_kind>=0;
}
__device__ bool rt_occluded(const RtFrame& f,const DRay& ray,float t_max) {
    if(!f.hardware_scene)return false;
    unsigned occluded=1;
    optixTrace(f.hardware_scene,make_float3(ray.origin.x,ray.origin.y,ray.origin.z),
        make_float3(ray.direction.x,ray.direction.y,ray.direction.z),0,t_max,0,255,
        OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES|OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT|
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,1,2,1,occluded);
    return occluded!=0;
}

#include "render/optix/integrator.cuh"

__device__ DCompactHit* rt_payload_hit() {
    return reinterpret_cast<DCompactHit*>((static_cast<unsigned long long>(optixGetPayload_1())<<32)|optixGetPayload_0());
}
extern "C" __global__ void __raygen__primary() {
    const auto& p=rt_optix();const auto i=optixGetLaunchIndex();
    rt_gbuffer_pixel(p.frame,p.guides,p.primary,int(i.y*p.frame.width+i.x),p.emission);
}
extern "C" __global__ void __raygen__lighting() {
    const auto& p=rt_optix();const auto i=optixGetLaunchIndex();
    rt_trace_pixel(p.frame,p.primary,p.guides,p.signals,int(i.y*p.frame.width+i.x));
}
extern "C" __global__ void __raygen__native_optics() {
    const auto& p=rt_optix();
    for(unsigned job=optixGetLaunchIndex().x;job<*p.optical_count;job+=optixGetLaunchDimensions().x)
        rt_native_optics_pixel(p.frame,p.primary,p.guides,p.emission,p.optical_pixels[job]);
}
extern "C" __global__ void __miss__radiance() {rt_payload_hit()->primitive_kind=-1;}
extern "C" __global__ void __miss__occlusion() {optixSetPayload_0(0);}

__device__ bool rt_accept(bool shadow) {
    DScene scene=rt_optix().frame.scene;
    // Blend coverage for shadow rays uses the same stable stochastic acceptance
    // as the reference tracer. Radiance paths handle blending in the integrator.
    scene.stochastic_alpha_test=shadow?1:0;
    const int instance=int(optixGetInstanceId());
    const DAsset asset=scene.assets[scene.instances[instance].asset_index];
    const auto origin=optixGetObjectRayOrigin(),direction=optixGetObjectRayDirection();
    const DRay local{v3(origin.x,origin.y,origin.z),v3(direction.x,direction.y,direction.z)};
    if(optixGetPrimitiveType()==OPTIX_PRIMITIVE_TYPE_SPHERE)
        return sphere_candidate_visible(scene,local,instance,asset.sphere_first+int(optixGetPrimitiveIndex()),optixGetRayTmax());
    const auto bary=optixGetTriangleBarycentrics();
    return triangle_candidate_visible(scene,local,instance,asset.triangle_first+int(optixGetPrimitiveIndex()),bary.x,bary.y);
}
extern "C" __global__ void __anyhit__radiance() {if(!rt_accept(false))optixIgnoreIntersection();}
extern "C" __global__ void __anyhit__occlusion() {if(!rt_accept(true))optixIgnoreIntersection();}
extern "C" __global__ void __closesthit__scene() {
    const auto& scene=rt_optix().frame.scene;
    const int instance=int(optixGetInstanceId());
    const DAsset asset=scene.assets[scene.instances[instance].asset_index];
    if(optixGetPrimitiveType()==OPTIX_PRIMITIVE_TYPE_SPHERE) {
        *rt_payload_hit()={optixGetRayTmax(),0,0,0,asset.sphere_first+int(optixGetPrimitiveIndex()),instance};
    } else {
        const auto bary=optixGetTriangleBarycentrics();
        *rt_payload_hit()={optixGetRayTmax(),bary.x,bary.y,1,asset.triangle_first+int(optixGetPrimitiveIndex()),instance};
    }
}
} // namespace
} // namespace renderer
