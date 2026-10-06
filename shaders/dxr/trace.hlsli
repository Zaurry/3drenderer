#ifndef RENDERER_DXR_TRACE
#define RENDERER_DXR_TRACE
#include "bsdf.hlsli"
#define DXR_TRACE_FLAGS RAY_FLAG_CULL_BACK_FACING_TRIANGLES
#if DXR_SER
struct [raypayload] Hit {
    uint instance : read(caller) : write(caller,closesthit,miss);
    uint primitive : read(caller) : write(caller,closesthit);
    float2 bary : read(caller) : write(caller,closesthit);
    float t : read(caller) : write(caller,closesthit);
    uint rng : read(caller,anyhit) : write(caller,anyhit);
};
#else
struct Hit {uint instance,primitive;float2 bary;float t;uint rng;};
#endif
bool AcceptHit(uint instance,uint primitive,float2 bary,float3 rayDir,inout uint rng) {
    Instance inst=LoadInstance(instance);Material m=LoadMaterial(MaterialIndex(inst,primitive));
    Vertex a,b,c;LoadVertices(inst,primitive,a,b,c);float3 w=float3(1-bary.x-bary.y,bary);
    if(!m.flags.y && dot(TransformNormal(inst,cross(b.position.xyz-a.position.xyz,c.position.xyz-a.position.xyz)),rayDir)>=0)return false;
    if(m.flags.x==0)return true;
    float alpha=SurfaceOpacity(m,a.uv*w.x+b.uv*w.y+c.uv*w.z,a.color.w*w.x+b.color.w*w.y+c.color.w*w.z);
    return m.flags.x==1?alpha>=m.optics.y:Random(rng)<alpha;
}
#if !DXR_INLINE
[shader("closesthit")]
void ClosestHit(inout Hit hit,BuiltInTriangleIntersectionAttributes attributes){hit.instance=InstanceID();hit.primitive=PrimitiveIndex();hit.bary=attributes.barycentrics;hit.t=RayTCurrent();}
[shader("anyhit")]
void AnyHit(inout Hit hit,BuiltInTriangleIntersectionAttributes attributes){if(!AcceptHit(InstanceID(),PrimitiveIndex(),attributes.barycentrics,WorldRayDirection(),hit.rng))IgnoreHit();}
[shader("miss")]
void Miss(inout Hit hit){hit.instance=0xffffffff;}
#endif
Hit Trace(float3 origin,float3 direction,inout uint rng,float limit=1e30,bool secondary=true) {
    RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=0;ray.TMax=limit;
    Hit hit=(Hit)0;hit.instance=0xffffffff;hit.rng=rng;
#if DXR_INLINE
    #if DXR_OMM
        RayQuery<RAY_FLAG_NONE,RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    #else
        RayQuery<RAY_FLAG_NONE> q;
    #endif
    q.TraceRayInline(sceneAS,DXR_TRACE_FLAGS,255,ray);
    while(q.Proceed())if(q.CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE)
        if(AcceptHit(q.CandidateInstanceID(),q.CandidatePrimitiveIndex(),q.CandidateTriangleBarycentrics(),direction,hit.rng))q.CommitNonOpaqueTriangleHit();
    if(q.CommittedStatus()==COMMITTED_TRIANGLE_HIT){hit.instance=q.CommittedInstanceID();hit.primitive=q.CommittedPrimitiveIndex();hit.bary=q.CommittedTriangleBarycentrics();hit.t=q.CommittedRayT();}
#else
    #if DXR_SER
        if(secondary){
            dx::HitObject object=dx::HitObject::TraceRay(sceneAS,DXR_TRACE_FLAGS,255,0,0,0,ray,hit);
            uint hint=255;
            if(object.IsHit())hint=MaterialIndex(LoadInstance(object.GetInstanceID()),object.GetPrimitiveIndex())&255;
            dx::MaybeReorderThread(object,hint,8);dx::HitObject::Invoke(object,hit);
        }
        else TraceRay(sceneAS,DXR_TRACE_FLAGS,255,0,0,0,ray,hit);
    #else
        TraceRay(sceneAS,DXR_TRACE_FLAGS,255,0,0,0,ray,hit);
    #endif
#endif
    rng=hit.rng;return hit;
}
bool Visible(Surface surface,float3 direction,float distance,inout uint rng) {
    float3 origin=Offset(surface.position,surface.geometric,direction);
    RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=0;ray.TMax=max(distance-length(origin-surface.position)-1e-4,1e-5);
    // Visibility only needs an accepted blocker. No closest-hit shading or
    // nearest-intersection search is required. AnyHit still handles alpha.
#if DXR_INLINE
    #if DXR_OMM
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    #else
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    #endif
    q.TraceRayInline(sceneAS,DXR_TRACE_FLAGS,255,ray);
    while(q.Proceed())if(q.CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE)
        if(AcceptHit(q.CandidateInstanceID(),q.CandidatePrimitiveIndex(),q.CandidateTriangleBarycentrics(),direction,rng))q.CommitNonOpaqueTriangleHit();
    return q.CommittedStatus()==COMMITTED_NOTHING;
#else
    Hit hit=(Hit)0;hit.rng=rng;
    TraceRay(sceneAS,DXR_TRACE_FLAGS|RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH|RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,255,0,0,0,ray,hit);
    rng=hit.rng;return hit.instance==0xffffffff;
#endif
}
float3 CameraDirection(float2 pixel){float2 uv=pixel/g.size.xy;return normalize(g.forward.xyz+g.right.xyz*((uv.x-.5)*g.right.w)+g.up.xyz*((.5-uv.y)*g.up.w));}
#endif
